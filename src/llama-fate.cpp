// llama-moe-cache -- FATE v2: Late Cache + Static Pinning
// Copyright (C) 2026 Ongun Manav
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License.
//
// For commercial licensing: ongunmnv@gmail.com

#include "llama-fate.h"
#include "llama-model.h"
#include "llama-hparams.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <unordered_set>

fate_system * g_fate = nullptr;

// ===========================================================================
// GPU VRAM pool
// ===========================================================================

bool fate_gpu_pool::init(ggml_backend_t backend, size_t slot_sz, size_t target_mb) {
    slot_bytes = slot_sz;
    size_t target_bytes = target_mb * 1024 * 1024;
    uint32_t want_slots = (uint32_t)(target_bytes / slot_bytes);

    while (want_slots > 4) {
        size_t total = (size_t)want_slots * slot_bytes + 256;
        struct ggml_init_params p = { 2 * ggml_tensor_overhead(), nullptr, true };
        ggml_context * tmp_ctx = ggml_init(p);
        if (!tmp_ctx) { want_slots -= 4; continue; }

        ggml_tensor * t = ggml_new_tensor_1d(tmp_ctx, GGML_TYPE_I8, (int64_t)total);
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
        ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, total + 256);

        if (!buf) {
            ggml_free(tmp_ctx);
            uint32_t step = std::max(want_slots / 10, (uint32_t)4);
            want_slots -= step;
            continue;
        }

        struct ggml_tallocr talloc = ggml_tallocr_new(buf);
        ggml_tallocr_alloc(&talloc, t);

        if (!t->data) {
            ggml_backend_buffer_free(buf);
            ggml_free(tmp_ctx);
            want_slots -= 4;
            continue;
        }

        buffer = buf; ctx = tmp_ctx; pool_tensor = t; n_slots = want_slots;
        break;
    }

    if (!pool_tensor || n_slots == 0) return false;
    slots.resize(n_slots);
    key_to_slot.reserve(n_slots * 2);

    fprintf(stderr, "FATE: GPU pool allocated: %u slots x %.1fMB = %zuMB\n",
            n_slots, (float)slot_bytes / (1024*1024),
            (size_t)n_slots * slot_bytes / (1024*1024));
    return true;
}

void fate_gpu_pool::free_pool() {
    if (buffer) ggml_backend_buffer_free(buffer);
    if (ctx)    ggml_free(ctx);
    buffer = nullptr; ctx = nullptr; pool_tensor = nullptr;
    n_slots = 0; slots.clear(); key_to_slot.clear();
}

int32_t fate_gpu_pool::lookup(uint64_t key) {
    auto it = key_to_slot.find(key);
    if (it == key_to_slot.end()) return -1;
    if (slots[it->second].state != fate_slot_state::READY) return -1;
    return (int32_t)it->second;
}

int32_t fate_gpu_pool::find_or_alloc(uint64_t key, uint64_t current_epoch) {
    // If key already has a slot, return it
    auto it = key_to_slot.find(key);
    if (it != key_to_slot.end()) {
        slots[it->second].last_used = current_epoch;
        return (int32_t)it->second;
    }

    // Find LRU evictable slot: not pinned, not LOADING, not current epoch
    int32_t best = -1;
    uint64_t oldest = UINT64_MAX;
    for (uint32_t i = 0; i < n_slots; i++) {
        if (slots[i].pinned) continue;
        if (slots[i].state == fate_slot_state::LOADING) continue;
        if (slots[i].key == UINT64_MAX) { best = (int32_t)i; break; }  // empty
        if (slots[i].last_used >= current_epoch) continue;  // used this epoch
        if (slots[i].last_used < oldest) { oldest = slots[i].last_used; best = (int32_t)i; }
    }
    if (best < 0) return -1;

    // Evict old slot
    if (slots[best].key != UINT64_MAX) {
        key_to_slot.erase(slots[best].key);
    }
    slots[best].key = key;
    slots[best].last_used = current_epoch;
    slots[best].state = fate_slot_state::EMPTY;
    slots[best].pinned = false;
    key_to_slot[key] = (uint32_t)best;
    return best;
}

void * fate_gpu_pool::slot_device_ptr(uint32_t idx) {
    if (!pool_tensor || !pool_tensor->data || idx >= n_slots) return nullptr;
    return (char *)pool_tensor->data + (size_t)idx * slot_bytes;
}

// ===========================================================================
// Helpers
// ===========================================================================

int fate_system::parse_layer(const char * name) {
    if (!name) return -1;
    const char * p = strstr(name, "blk.");
    if (!p) return -1;
    p += 4;
    char * end = nullptr;
    long v = strtol(p, &end, 10);
    if (end == p || *end != '.') return -1;
    return (int)v;
}

int fate_system::parse_tensor_kind(const char * name) {
    if (!name) return -1;
    if (strstr(name, "ffn_gate_up_exps")) return 3;
    if (strstr(name, "ffn_gate_exps"))    return 0;
    if (strstr(name, "ffn_up_exps"))      return 1;
    if (strstr(name, "ffn_down_exps"))    return 2;
    return -1;
}

// ===========================================================================
// Lifecycle
// ===========================================================================

bool fate_system::init(const llama_model & model, ggml_backend_t backend, int32_t cache_mb) {
    gpu_backend = backend;

    const auto & hp = model.hparams;
    n_layer       = hp.n_layer;
    n_expert      = hp.n_expert;
    n_expert_used = hp.n_expert_used;

    if (n_expert == 0) {
        fprintf(stderr, "FATE: not a MoE model\n");
        return false;
    }

    size_t gate_bytes_max = 0, up_bytes_max = 0, down_bytes_max = 0;
    for (uint32_t il = 0; il < n_layer && il < (uint32_t)model.layers.size(); il++) {
        const auto & lay = model.layers[il];
        if (lay.ffn_gate_exps && lay.ffn_gate_exps->nb[2] > 0)
            gate_bytes_max = std::max(gate_bytes_max, (size_t)lay.ffn_gate_exps->nb[2]);
        if (lay.ffn_up_exps && lay.ffn_up_exps->nb[2] > 0)
            up_bytes_max = std::max(up_bytes_max, (size_t)lay.ffn_up_exps->nb[2]);
        if (lay.ffn_down_exps && lay.ffn_down_exps->nb[2] > 0)
            down_bytes_max = std::max(down_bytes_max, (size_t)lay.ffn_down_exps->nb[2]);
    }

    expert_bytes_max = std::max({gate_bytes_max, up_bytes_max, down_bytes_max});
    if (expert_bytes_max == 0) return false;

    fprintf(stderr, "FATE v2: n_layer=%u n_expert=%u n_expert_used=%u\n", n_layer, n_expert, n_expert_used);
    fprintf(stderr, "FATE v2: max expert strides: gate=%zuB up=%zuB down=%zuB max=%.1fMB\n",
            gate_bytes_max, up_bytes_max, down_bytes_max, (float)expert_bytes_max / (1024*1024));

    size_t padded_slot = expert_bytes_max + 512;
    uint32_t min_slots = n_layer * n_expert_used * 3;
    size_t min_mb = (size_t)min_slots * padded_slot / (1024*1024) + 64;
    size_t target_mb = (cache_mb > 0) ? (size_t)cache_mb : std::max(min_mb, (size_t)5120);

    fprintf(stderr, "FATE v2: working set = %u slots (%.0fMB), target = %zuMB\n",
            min_slots, (float)min_slots * padded_slot / (1024*1024), target_mb);

    if (!pool.init(backend, padded_slot, target_mb)) {
        fprintf(stderr, "FATE v2: pool allocation failed\n");
        return false;
    }

    // Pin expert weight memory for truly async H2D
    uint32_t pinned = 0, total_tensors = 0;
    for (uint32_t il = 0; il < n_layer && il < (uint32_t)model.layers.size(); il++) {
        const auto & lay = model.layers[il];
        if (lay.ffn_gate_exps && lay.ffn_gate_exps->data) {
            pinned += fate_prefetch_pin_memory(lay.ffn_gate_exps->data, ggml_nbytes(lay.ffn_gate_exps));
            total_tensors++;
        }
        if (lay.ffn_up_exps && lay.ffn_up_exps->data) {
            pinned += fate_prefetch_pin_memory(lay.ffn_up_exps->data, ggml_nbytes(lay.ffn_up_exps));
            total_tensors++;
        }
        if (lay.ffn_down_exps && lay.ffn_down_exps->data) {
            pinned += fate_prefetch_pin_memory(lay.ffn_down_exps->data, ggml_nbytes(lay.ffn_down_exps));
            total_tensors++;
        }
        if (lay.ffn_gate_up_exps && lay.ffn_gate_up_exps->data) {
            pinned += fate_prefetch_pin_memory(lay.ffn_gate_up_exps->data, ggml_nbytes(lay.ffn_gate_up_exps));
            total_tensors++;
        }
    }
    all_pinned = (pinned == total_tensors);
    fprintf(stderr, "FATE v2: pinned %u/%u expert tensors\n", pinned, total_tensors);

    // Create prefetch stream for async H2D
    prefetch_stream = fate_prefetch_stream_create();
    if (!prefetch_stream) {
        fprintf(stderr, "FATE v2: WARNING -- prefetch stream creation failed\n");
    }

    // Create CUDA event for populate tracking
    populate_event = fate_event_create();

    // Register CPU source pointers
    sources.resize(n_layer * N_KINDS);
    for (uint32_t il = 0; il < n_layer && il < (uint32_t)model.layers.size(); il++) {
        const auto & lay = model.layers[il];
        if (lay.ffn_gate_exps && lay.ffn_gate_exps->data)
            sources[il * N_KINDS + 0] = { lay.ffn_gate_exps->data, (size_t)lay.ffn_gate_exps->nb[2] };
        if (lay.ffn_up_exps && lay.ffn_up_exps->data)
            sources[il * N_KINDS + 1] = { lay.ffn_up_exps->data, (size_t)lay.ffn_up_exps->nb[2] };
        if (lay.ffn_down_exps && lay.ffn_down_exps->data)
            sources[il * N_KINDS + 2] = { lay.ffn_down_exps->data, (size_t)lay.ffn_down_exps->nb[2] };
        if (lay.ffn_gate_up_exps && lay.ffn_gate_up_exps->data)
            sources[il * N_KINDS + 3] = { lay.ffn_gate_up_exps->data, (size_t)lay.ffn_gate_up_exps->nb[2] };
    }

    // Reserve space for miss tracking
    missed_list.reserve(n_layer * n_expert_used * N_KINDS);
    loading_slots.reserve(256);
    // Flat frequency array: O(1) indexed by (layer, kind, expert_id)
    access_freq.resize(n_layer * N_KINDS * n_expert, 0);

    fprintf(stderr, "FATE v2: system initialized (%u cache slots, late-cache + static pinning)\n", pool.n_slots);
    return true;
}

void fate_system::shutdown() {
    print_stats();

    // Complete any in-flight populates
    if (populate_in_flight && prefetch_stream) {
        fate_prefetch_sync(prefetch_stream);
        populate_in_flight = false;
    }

    if (populate_event)   fate_event_destroy(populate_event);
    if (prefetch_stream)  fate_prefetch_stream_destroy(prefetch_stream);
    populate_event = nullptr;
    prefetch_stream = nullptr;

    pool.free_pool();
}

// ===========================================================================
// Expert copy hook -- the hot path
//
// Called per-expert during MUL_MAT_ID weight copies.
// HIT (READY):  D2D from pool -> compute buffer, return true
// MISS:         record miss, return false (vanilla H2D handles it)
// ===========================================================================

bool fate_system::on_expert_copy(ggml_backend_t backend,
                                  struct ggml_tensor * dst,
                                  const void * /*src_data*/, size_t offset, size_t size,
                                  int32_t expert_id, int64_t /*n_expert_total*/,
                                  const char * tensor_name) {
    // Name cache: pointer-keyed O(1) lookup (tensor objects are persistent)
    int layer, kind;
    auto nc_it = name_cache.find(tensor_name);
    if (nc_it != name_cache.end()) {
        layer = nc_it->second.first;
        kind  = nc_it->second.second;
    } else {
        layer = parse_layer(tensor_name);
        kind  = parse_tensor_kind(tensor_name);
        if (layer >= 0 && kind >= 0) {
            name_cache[tensor_name] = {layer, kind};
        }
    }
    if (layer < 0 || kind < 0 || (uint32_t)layer >= n_layer) return false;

    // Detect token/ubatch boundary: layer goes backward -> new token
    if (layer < last_layer || (last_layer < 0 && layer == 0)) {
        on_token_start();
    }
    last_layer = layer;

    uint64_t key = fate_gpu_pool::make_key((uint32_t)layer, (uint32_t)kind, (uint32_t)expert_id);
    stats.accesses++;

    // Flat array frequency bump (O(1), cache-friendly)
    uint32_t fi = freq_index((uint32_t)layer, (uint32_t)kind, (uint32_t)expert_id);
    if (fi < access_freq.size() && access_freq[fi] < 65535) {
        access_freq[fi]++;
    }

    // Pool lookup -- only serve READY hits
    int32_t slot = pool.lookup(key);
    if (slot >= 0) {
        // HIT: explicit D2D from pool slot to compute buffer
        stats.hits++;
        void * slot_ptr = pool.slot_device_ptr((uint32_t)slot);
        void * dst_ptr = (char *)dst->data + offset;
        fate_d2d_copy(backend, dst_ptr, slot_ptr, size);
        pool.slots[slot].last_used = current_epoch;
        return true;
    }

    // MISS: record for later populate, return false for vanilla H2D
    stats.misses++;
    missed_list.push_back({key, (uint32_t)layer, (uint32_t)kind, (uint32_t)expert_id});
    return false;
}

// ===========================================================================
// Token boundary processing
// ===========================================================================

void fate_system::on_token_start() {
    current_epoch++;

    // Check if previous populate batch completed (LOADING -> READY)
    if (populate_in_flight && populate_event) {
        if (fate_event_query(populate_event)) {
            for (uint32_t idx : loading_slots) {
                if (idx < pool.n_slots) {
                    pool.slots[idx].state = fate_slot_state::READY;
                }
            }
            loading_slots.clear();
            populate_in_flight = false;
        }
        // If not done yet, leave as LOADING -- they'll complete later
    }

    // Process misses from previous token
    populate_misses();

    // Static pinning after warmup
    if (warmup_remaining > 0) {
        warmup_remaining--;
        if (warmup_remaining == 0) {
            pin_hot_experts();
        }
    }

    // Decay frequency every 500 tokens
    freq_decay_counter++;
    if (freq_decay_counter >= 500) {
        for (auto & v : access_freq) v >>= 1;
        freq_decay_counter = 0;
    }

    // Periodic stats logging
    if (current_epoch % 50 == 0 && current_epoch > 0) {
        float hr = stats.accesses > 0 ? 100.0f * stats.hits / stats.accesses : 0;
        fprintf(stderr, "FATE v2: epoch %llu, %.1f%% hit rate, %llu populates, %llu skipped\n",
                (unsigned long long)current_epoch, hr,
                (unsigned long long)stats.populates,
                (unsigned long long)stats.skipped);
    }
}

// ===========================================================================
// Async populate: queue H2D for qualified misses
// ===========================================================================

void fate_system::populate_misses() {
    if (missed_list.empty()) return;
    if (!prefetch_stream) { missed_list.clear(); return; }

    // DIAGNOSTIC: disable populate to isolate PCIe contention
    stats.skipped += missed_list.size();
    missed_list.clear();
    return;

    // If previous populate still in flight, drop misses (will re-record if needed)
    if (populate_in_flight) {
        stats.skipped += missed_list.size();
        missed_list.clear();
        return;
    }

    // Deduplicate and filter by admission
    std::unordered_set<uint64_t> seen;
    uint32_t queued = 0;

    for (auto & m : missed_list) {
        if (seen.count(m.key)) continue;
        seen.insert(m.key);

        // Already in pool?
        if (pool.key_to_slot.count(m.key)) continue;

        // Admission: need 2+ accesses (flat array lookup)
        uint32_t fi = freq_index(m.layer, m.kind, m.expert_id);
        if (fi >= access_freq.size() || access_freq[fi] < 2) {
            stats.skipped++;
            continue;
        }

        // Get source info
        uint32_t src_idx = m.layer * N_KINDS + m.kind;
        if (src_idx >= sources.size() || !sources[src_idx].base) continue;

        // Allocate pool slot
        int32_t slot = pool.find_or_alloc(m.key, current_epoch);
        if (slot < 0) continue;

        pool.slots[slot].state = fate_slot_state::LOADING;
        loading_slots.push_back((uint32_t)slot);

        // Compute source pointer and copy size
        void * dst = pool.slot_device_ptr((uint32_t)slot);
        size_t eb = sources[src_idx].expert_bytes;
        const void * src = (const char *)sources[src_idx].base + (size_t)m.expert_id * eb;
        size_t copy_n = (m.expert_id < n_expert - 1) ? eb + std::min(eb, (size_t)512) : eb;

        fate_prefetch_h2d(prefetch_stream, dst, src, copy_n);
        queued++;
    }
    missed_list.clear();

    if (queued > 0) {
        fate_event_record(populate_event, prefetch_stream);
        populate_in_flight = true;
        stats.populates += queued;
    }
}

// ===========================================================================
// Static expert pinning
// ===========================================================================

void fate_system::pin_hot_experts() {
    // Sync any in-flight populate (one-time cost during warmup)
    if (populate_in_flight && prefetch_stream) {
        fate_prefetch_sync(prefetch_stream);
        for (uint32_t idx : loading_slots) {
            if (idx < pool.n_slots) {
                pool.slots[idx].state = fate_slot_state::READY;
            }
        }
        loading_slots.clear();
        populate_in_flight = false;
    }

    // Collect candidates from flat frequency array
    struct pin_candidate { uint32_t layer; uint32_t kind; uint32_t eid; uint16_t freq; };
    std::vector<pin_candidate> candidates;
    candidates.reserve(1024);

    for (uint32_t l = 0; l < n_layer; l++) {
        for (uint32_t k = 0; k < N_KINDS; k++) {
            uint32_t src_idx = l * N_KINDS + k;
            if (src_idx >= sources.size() || !sources[src_idx].base) continue;
            for (uint32_t e = 0; e < n_expert; e++) {
                uint32_t fi = freq_index(l, k, e);
                if (fi < access_freq.size() && access_freq[fi] >= 3) {
                    candidates.push_back({l, k, e, access_freq[fi]});
                }
            }
        }
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const pin_candidate & a, const pin_candidate & b) { return a.freq > b.freq; });

    uint32_t pin_target = std::min((uint32_t)200, (uint32_t)candidates.size());
    uint32_t pinned = 0, loaded = 0;

    for (uint32_t i = 0; i < pin_target; i++) {
        auto & c = candidates[i];
        uint64_t key = fate_gpu_pool::make_key(c.layer, c.kind, c.eid);

        auto it = pool.key_to_slot.find(key);
        if (it != pool.key_to_slot.end()) {
            pool.slots[it->second].pinned = true;
            pool.slots[it->second].state = fate_slot_state::READY;
            pinned++;
            continue;
        }

        int32_t slot = pool.find_or_alloc(key, current_epoch);
        if (slot < 0) continue;

        pool.slots[slot].pinned = true;
        pool.slots[slot].state = fate_slot_state::LOADING;

        uint32_t src_idx = c.layer * N_KINDS + c.kind;
        void * dst = pool.slot_device_ptr((uint32_t)slot);
        size_t eb = sources[src_idx].expert_bytes;
        const void * src = (const char *)sources[src_idx].base + (size_t)c.eid * eb;
        size_t copy_n = (c.eid < n_expert - 1) ? eb + std::min(eb, (size_t)512) : eb;

        fate_prefetch_h2d(prefetch_stream, dst, src, copy_n);
        loaded++;
        pinned++;
    }

    // Sync to ensure all pins are loaded, then promote to READY
    if (loaded > 0 && prefetch_stream) {
        fate_prefetch_sync(prefetch_stream);
    }

    // Mark all just-loaded pinned slots as READY
    for (uint32_t i = 0; i < pool.n_slots; i++) {
        if (pool.slots[i].pinned && pool.slots[i].state == fate_slot_state::LOADING) {
            pool.slots[i].state = fate_slot_state::READY;
        }
    }

    fprintf(stderr, "FATE v2: pinned %u hot experts (%u loaded)\n", pinned, loaded);
}

// ===========================================================================
// Stats
// ===========================================================================

void fate_system::print_stats() const {
    float hr = (stats.accesses > 0) ? 100.0f * (float)stats.hits / (float)stats.accesses : 0.0f;
    uint32_t pinned_count = 0;
    for (uint32_t i = 0; i < pool.n_slots; i++) {
        if (pool.slots[i].pinned) pinned_count++;
    }
    fprintf(stderr, "\n======== FATE v2 CACHE STATS ========\n"
                    "  accesses   : %llu\n"
                    "  hits       : %llu (D2D from pool)\n"
                    "  misses     : %llu (vanilla H2D fallback)\n"
                    "  hit rate   : %.2f%%\n"
                    "  populates  : %llu (async H2D to pool)\n"
                    "  skipped    : %llu (in-flight or low freq)\n"
                    "  pool       : %u slots x %.1fMB (%u pinned)\n"
                    "  epochs     : %llu\n"
                    "=====================================\n\n",
            (unsigned long long)stats.accesses, (unsigned long long)stats.hits,
            (unsigned long long)stats.misses, hr,
            (unsigned long long)stats.populates,
            (unsigned long long)stats.skipped,
            pool.n_slots, (float)pool.slot_bytes / (1024*1024), pinned_count,
            (unsigned long long)current_epoch);
}
