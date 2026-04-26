// llama-moe-cache — FATE: predictive expert caching for MoE inference
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
    tick = 0;

    fprintf(stderr, "FATE: GPU pool allocated: %u slots × %.1fMB = %zuMB\n",
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

int32_t fate_gpu_pool::find_or_alloc(uint64_t key) {
    auto it = key_to_slot.find(key);
    if (it != key_to_slot.end()) {
        slots[it->second].last_used = ++tick;
        return (int32_t)it->second;
    }

    int32_t best = -1;
    uint64_t oldest = UINT64_MAX;
    for (uint32_t i = 0; i < n_slots; i++) {
        if (slots[i].key == UINT64_MAX) { best = (int32_t)i; break; }
        if (slots[i].last_used < oldest) { oldest = slots[i].last_used; best = (int32_t)i; }
    }
    if (best < 0) return -1;

    if (slots[best].key != UINT64_MAX) {
        key_to_slot.erase(slots[best].key);
    }
    slots[best].key = key;
    slots[best].last_used = ++tick;
    key_to_slot[key] = (uint32_t)best;
    return best;
}

void * fate_gpu_pool::slot_device_ptr(uint32_t idx) {
    if (!pool_tensor || !pool_tensor->data || idx >= n_slots) return nullptr;
    return (char *)pool_tensor->data + (size_t)idx * slot_bytes;
}

// ===========================================================================
// Prefetcher
// ===========================================================================

void fate_prefetcher::init(uint32_t nl, uint32_t ne, uint32_t neu, size_t max_expert_bytes) {
    n_layer = nl; n_expert = ne; n_expert_used = neu;
    cur.resize(nl);
    prev.resize(nl);
    sources.resize(nl * N_KINDS);
    last_layer = -1;
    prefetched = 0;

    stream = fate_prefetch_stream_create();
    if (!stream) {
        fprintf(stderr, "FATE: WARNING — prefetch stream creation failed\n");
        return;
    }

    // Allocate pinned staging pool for async H2D when sources aren't pinned.
    // Round-robin assignment eliminates the single-buffer race condition.
    staging_buf_size = max_expert_bytes + 512;
    uint32_t alloc_ok = 0;
    for (uint32_t i = 0; i < N_STAGING; i++) {
        staging_pool[i] = fate_prefetch_alloc_pinned(staging_buf_size);
        if (staging_pool[i]) alloc_ok++;
    }
    staging_idx_inline = 0;
    staging_idx_worker = N_STAGING / 2;  // worker uses upper half
    all_pinned = false;  // updated after pinning attempt in fate_system::init
    fprintf(stderr, "FATE: pinned staging pool: %u/%u × %.1fMB = %.1fMB\n",
            alloc_ok, N_STAGING,
            (float)staging_buf_size / (1024*1024),
            (float)alloc_ok * staging_buf_size / (1024*1024));

    quit = false;
    worker = std::thread([this]{ worker_fn(); });

    fprintf(stderr, "FATE: prefetch engine started (cross-layer + temporal prediction)\n");
}

void fate_prefetcher::worker_fn() {
    while (true) {
        std::vector<job> work;
        {
            std::unique_lock<std::mutex> lk(mtx);
            cv.wait(lk, [this]{ return has_jobs || quit; });
            if (quit && !has_jobs) return;
            work.swap(jobs);
            has_jobs = false;
            processing = true;
        }
        for (auto & j : work) {
            if (all_pinned) {
                // Source memory is pinned — direct async H2D (fastest path)
                fate_prefetch_h2d(stream, j.dst, j.src, j.n);
            } else {
                // Source not pinned — copy through staging pool for async DMA
                uint32_t si = staging_idx_worker % N_STAGING;
                if (si < N_STAGING / 2) si += N_STAGING / 2;  // stay in worker half [24..47]
                void * sbuf = staging_pool[si];
                if (sbuf && j.n <= staging_buf_size) {
                    memcpy(sbuf, j.src, j.n);
                    fate_prefetch_h2d(stream, j.dst, sbuf, j.n);
                    staging_idx_worker++;
                } else {
                    fate_prefetch_h2d(stream, j.dst, j.src, j.n);
                }
            }
        }
        {
            std::lock_guard<std::mutex> lk(mtx);
            processing = false;
        }
        done_cv.notify_all();
    }
}

void fate_prefetcher::submit(std::vector<job> && work) {
    if (work.empty() || !stream) return;
    prefetched += (uint64_t)work.size();
    {
        std::lock_guard<std::mutex> lk(mtx);
        jobs.insert(jobs.end(), work.begin(), work.end());
        has_jobs = true;
    }
    cv.notify_one();
}

void fate_prefetcher::register_src(uint32_t layer, uint32_t kind, const void * base, size_t eb) {
    uint32_t idx = layer * N_KINDS + kind;
    if (idx < sources.size() && !sources[idx].base) {
        size_t padded = eb + std::min(eb, (size_t)512);
        sources[idx] = {base, eb, padded};
    }
}

void fate_prefetcher::on_token_start(fate_gpu_pool & /*pool*/) {
    prev.swap(cur);
    for (auto & v : cur) v.clear();
    last_layer = -1;
    // No bulk prefetch — cross-layer prediction handles it per-layer
    // to avoid massive sync stalls.
}

void fate_prefetcher::on_expert(uint32_t layer, int32_t expert_id) {
    if (layer >= n_layer) return;
    auto & v = cur[layer];
    for (int32_t e : v) if (e == expert_id) return;
    v.push_back(expert_id);
}

void fate_prefetcher::on_layer_done(uint32_t layer, fate_gpu_pool & pool) {
    uint32_t next = layer + 1;
    if (next >= n_layer) return;

    std::unordered_set<int32_t> predicted;

    // Cross-layer prediction: next layer uses same experts as current layer
    for (int32_t e : cur[layer]) predicted.insert(e);

    // Temporal prediction: next layer uses same experts as previous token's next layer
    if (next < (uint32_t)prev.size()) {
        for (int32_t e : prev[next]) predicted.insert(e);
    }

    std::vector<job> work;
    for (int32_t eid : predicted) {
        for (uint32_t k = 0; k < N_KINDS; k++) {
            uint32_t idx = next * N_KINDS + k;
            if (idx >= sources.size() || !sources[idx].base) continue;

            uint64_t key = fate_gpu_pool::make_key(next, k, (uint32_t)eid);
            if (pool.key_to_slot.count(key)) continue;

            int32_t slot = pool.find_or_alloc(key);
            if (slot < 0) continue;

            void * dst = pool.slot_device_ptr((uint32_t)slot);
            const void * src = (const char *)sources[idx].base
                               + (size_t)eid * sources[idx].expert_bytes;
            // Use padded size for non-last experts (matches what the hook reads)
            size_t copy_n = ((uint32_t)eid < n_expert - 1)
                          ? sources[idx].padded_bytes
                          : sources[idx].expert_bytes;
            work.push_back({dst, src, copy_n});
        }
    }
    submit(std::move(work));
}

void fate_prefetcher::sync() {
    // Only wait for the worker thread to finish issuing cudaMemcpyAsync calls.
    // The actual GPU transfer is handled by the GPU barrier (fate_prefetch_insert_barrier)
    // so we never block the CPU waiting for PCIe DMA.
    std::unique_lock<std::mutex> lk(mtx);
    done_cv.wait(lk, [this]{ return !has_jobs && !processing; });
}

void fate_prefetcher::shutdown() {
    {
        std::lock_guard<std::mutex> lk(mtx);
        quit = true;
    }
    cv.notify_one();
    if (worker.joinable()) worker.join();
    fate_prefetch_sync(stream);
    fate_prefetch_stream_destroy(stream);
    stream = nullptr;
    for (uint32_t i = 0; i < N_STAGING; i++) {
        if (staging_pool[i]) { fate_prefetch_free_pinned(staging_pool[i]); staging_pool[i] = nullptr; }
    }
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

    // Use nb[2] (the actual per-expert stride the scheduler uses) instead of
    // ggml_nbytes/n_expert, because Q4_K_M uses different quant types per layer.
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

    fprintf(stderr, "FATE: n_layer=%u n_expert=%u n_expert_used=%u\n", n_layer, n_expert, n_expert_used);
    fprintf(stderr, "FATE: max expert strides: gate=%zuB up=%zuB down=%zuB max=%.1fMB\n",
            gate_bytes_max, up_bytes_max, down_bytes_max, (float)expert_bytes_max / (1024*1024));

    size_t padded_slot = expert_bytes_max + 512;
    uint32_t min_slots = n_layer * n_expert_used * 3;
    size_t min_mb = (size_t)min_slots * padded_slot / (1024*1024) + 64;
    // Use minimal pool for step-0 baseline (hook always returns false, pool unused)
    size_t target_mb = (cache_mb > 0) ? (size_t)cache_mb : 128;

    fprintf(stderr, "FATE: working set = %u slots (%.0fMB), target = %zuMB\n",
            min_slots, (float)min_slots * padded_slot / (1024*1024), target_mb);

    if (!pool.init(backend, padded_slot, target_mb)) {
        fprintf(stderr, "FATE: pool allocation failed\n");
        return false;
    }

    // Pin expert weight memory for truly async H2D prefetch
    uint32_t pinned = 0;
    for (uint32_t il = 0; il < n_layer && il < (uint32_t)model.layers.size(); il++) {
        const auto & lay = model.layers[il];
        if (lay.ffn_gate_exps && lay.ffn_gate_exps->data)
            pinned += fate_prefetch_pin_memory(lay.ffn_gate_exps->data, ggml_nbytes(lay.ffn_gate_exps));
        if (lay.ffn_up_exps && lay.ffn_up_exps->data)
            pinned += fate_prefetch_pin_memory(lay.ffn_up_exps->data, ggml_nbytes(lay.ffn_up_exps));
        if (lay.ffn_down_exps && lay.ffn_down_exps->data)
            pinned += fate_prefetch_pin_memory(lay.ffn_down_exps->data, ggml_nbytes(lay.ffn_down_exps));
    }
    fprintf(stderr, "FATE: pinned %u/%u expert tensors for async prefetch\n",
            pinned, n_layer * 3);

    // If all expert tensors are pinned, direct H2D is fastest — skip staging pool
    bool all_pinned = (pinned == n_layer * 3);

    // Init prefetcher with CPU source pointers for every expert tensor.
    // Use nb[2] as the per-expert stride (matches the scheduler's expert_size).
    prefetch.init(n_layer, n_expert, n_expert_used, expert_bytes_max);
    prefetch.all_pinned = all_pinned;
    if (all_pinned) {
        fprintf(stderr, "FATE: all tensors pinned — using direct async H2D (staging pool standby)\n");
    } else {
        fprintf(stderr, "FATE: not all tensors pinned — using staging pool for async H2D\n");
    }
    for (uint32_t il = 0; il < n_layer && il < (uint32_t)model.layers.size(); il++) {
        const auto & lay = model.layers[il];
        if (lay.ffn_gate_exps && lay.ffn_gate_exps->data)
            prefetch.register_src(il, 0, lay.ffn_gate_exps->data, (size_t)lay.ffn_gate_exps->nb[2]);
        if (lay.ffn_up_exps && lay.ffn_up_exps->data)
            prefetch.register_src(il, 1, lay.ffn_up_exps->data, (size_t)lay.ffn_up_exps->nb[2]);
        if (lay.ffn_down_exps && lay.ffn_down_exps->data)
            prefetch.register_src(il, 2, lay.ffn_down_exps->data, (size_t)lay.ffn_down_exps->nb[2]);
    }

    fprintf(stderr, "FATE: system initialized (%u cache slots + prefetch)\n", pool.n_slots);
    return true;
}

void fate_system::shutdown() {
    print_stats();
    prefetch.shutdown();
    pool.free_pool();
}

// ===========================================================================
// Expert copy hook — the hot path
//
// Called per-expert during MUL_MAT_ID weight copies.
// Detects layer transitions to drive prefetch.
// ===========================================================================

bool fate_system::on_expert_copy(ggml_backend_t backend,
                                  struct ggml_tensor * dst,
                                  const void * src_data, size_t offset, size_t size,
                                  int32_t expert_id, int64_t /*n_expert_total*/,
                                  const char * tensor_name) {
    // Step 0: Always-false baseline — measures per-expert iteration overhead
    // when hook is installed but does nothing. Vanilla H2D handles everything.
    (void)backend; (void)dst; (void)src_data; (void)offset; (void)size;
    (void)expert_id; (void)tensor_name;
    return false;
}

// ===========================================================================
// Stats
// ===========================================================================

void fate_system::print_stats() const {
    uint64_t a = stats.accesses.load();
    uint64_t h = stats.hits.load();
    uint64_t m = stats.misses.load();
    uint64_t p = prefetch.prefetched.load();
    float hr = (a > 0) ? 100.0f * (float)h / (float)a : 0.0f;
    fprintf(stderr, "\n======== FATE CACHE STATS ========\n"
                    "  accesses   : %llu\n"
                    "  hits       : %llu (D2D from pool)\n"
                    "  misses     : %llu (H2D fallback)\n"
                    "  hit rate   : %.2f%%\n"
                    "  prefetched : %llu (async H2D to pool)\n"
                    "  pool       : %u slots × %.1fMB\n"
                    "==================================\n\n",
            (unsigned long long)a, (unsigned long long)h,
            (unsigned long long)m, hr, (unsigned long long)p,
            pool.n_slots, (float)pool.slot_bytes / (1024*1024));
}
