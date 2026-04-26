// llama-moe-cache -- FATE v2: Late Cache + Static Pinning
// Copyright (C) 2026 Ongun Manav
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License.
//
// For commercial licensing: ongunmnv@gmail.com

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>

struct llama_model;

// CUDA functions (implemented in ggml-cuda.cu)
extern "C" {
    void * fate_prefetch_stream_create(void);
    void   fate_prefetch_h2d(void * stream, void * dst, const void * src, size_t n);
    void   fate_prefetch_sync(void * stream);
    void   fate_prefetch_stream_destroy(void * stream);
    void   fate_prefetch_insert_barrier(void * backend_ptr, void * prefetch_stream);
    bool   fate_prefetch_pin_memory(const void * ptr, size_t size);
    void * fate_prefetch_alloc_pinned(size_t size);
    void   fate_prefetch_free_pinned(void * p);
    void   fate_debug_d2h(void * dst, const void * src, size_t n);
    int    fate_debug_ptr_type(const void * ptr);
    // v2 event tracking
    void * fate_event_create(void);
    bool   fate_event_query(void * event);
    void   fate_event_record(void * event, void * stream);
    void   fate_event_destroy(void * event);
}

// ---------------------------------------------------------------------------
// Slot states for the GPU expert cache pool
// ---------------------------------------------------------------------------
enum class fate_slot_state : uint8_t {
    EMPTY,    // Slot unoccupied
    LOADING,  // Async H2D in progress -- NOT servable as hit
    READY     // Data valid -- servable as D2D cache hit
};

// ---------------------------------------------------------------------------
// GPU VRAM pool -- persistent buffer for caching expert weights
// ---------------------------------------------------------------------------
struct fate_gpu_pool {
    ggml_backend_buffer_t buffer      = nullptr;
    ggml_context *        ctx         = nullptr;
    ggml_tensor *         pool_tensor = nullptr;
    size_t                slot_bytes  = 0;
    uint32_t              n_slots     = 0;

    struct slot_info {
        uint64_t        key       = UINT64_MAX;
        uint64_t        last_used = 0;
        fate_slot_state state     = fate_slot_state::EMPTY;
        bool            pinned    = false;
    };
    std::vector<slot_info> slots;
    std::unordered_map<uint64_t, uint32_t> key_to_slot;

    bool     init(ggml_backend_t backend, size_t slot_bytes, size_t target_mb);
    void     free_pool();

    // Returns slot index if key exists AND state == READY, else -1
    int32_t  lookup(uint64_t key);

    // Allocate slot for key, evicting LRU non-pinned/non-LOADING/non-recent slot
    int32_t  find_or_alloc(uint64_t key, uint64_t current_epoch);

    void *   slot_device_ptr(uint32_t idx);

    static uint64_t make_key(uint32_t layer, uint32_t kind, uint32_t expert) {
        return ((uint64_t)layer << 16) | ((uint64_t)kind << 8) | expert;
    }
};

// ---------------------------------------------------------------------------
// Main FATE v2 system: Late Cache + Static Pinning
//
// On HIT  (READY slot): D2D from pool, return true  (~2us)
// On MISS:              record miss, return false    (vanilla H2D, zero penalty)
// Between tokens:       async H2D for qualified misses (admission >= 2)
// ---------------------------------------------------------------------------
struct fate_system {
    static const uint32_t N_KINDS = 4;

    uint32_t n_layer       = 0;
    uint32_t n_expert      = 0;
    uint32_t n_expert_used = 0;
    size_t   expert_bytes_max = 0;

    fate_gpu_pool    pool;
    ggml_backend_t   gpu_backend = nullptr;

    // CPU source pointers: [layer * N_KINDS + kind]
    struct tensor_src {
        const void * base         = nullptr;
        size_t       expert_bytes = 0;
    };
    std::vector<tensor_src> sources;

    // Prefetch stream + event for async H2D population
    void * prefetch_stream     = nullptr;
    void * populate_event      = nullptr;
    bool   populate_in_flight  = false;
    bool   all_pinned          = false;
    std::vector<uint32_t> loading_slots;

    // Missed expert records (per token, processed between tokens)
    struct miss_record {
        uint64_t key;
        uint32_t layer;
        uint32_t kind;
        uint32_t expert_id;
    };
    std::vector<miss_record> missed_list;

    // Access frequency for admission control (decayed, not cleared)
    std::unordered_map<uint64_t, uint32_t> access_freq;
    uint64_t freq_decay_counter = 0;

    // Epoch (token/ubatch counter)
    uint64_t current_epoch = 0;

    // Token boundary detection
    int32_t last_layer = -1;

    // Static pinning
    int32_t warmup_remaining = 50;

    // Stats
    struct {
        uint64_t accesses  = 0;
        uint64_t hits      = 0;
        uint64_t misses    = 0;
        uint64_t populates = 0;
        uint64_t skipped   = 0;
    } stats;

    bool init(const llama_model & model, ggml_backend_t backend, int32_t cache_mb = 0);
    void shutdown();

    bool on_expert_copy(ggml_backend_t backend,
                        struct ggml_tensor * dst,
                        const void * src_data, size_t offset, size_t size,
                        int32_t expert_id, int64_t n_expert_total,
                        const char * tensor_name);

    void on_token_start();
    void populate_misses();
    void pin_hot_experts();
    void print_stats() const;

    static int parse_layer(const char * name);
    static int parse_tensor_kind(const char * name);
};

extern fate_system * g_fate;
