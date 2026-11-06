// Owned fixture: object functions that wait at threadgroup barriers, for drawMeshThreads draws that
// leave a partial object threadgroup (the runtime lane, the conformance suite's
// `mesh-partial-barrier` cases, and the translator's interpreter test). Every thread of an object
// threadgroup adds 1 to out[0]; the thread at position (x, y) of the grid writes its two words at
// out[4 + 2 * (x + threads_per_grid.x * y)] once, on the path it takes: the word it read from
// threadgroup memory across the barriers, and its threadgroup's threads (x | y << 8) with its own
// index in it (<< 16). Its threadgroup's thread 0 launches no mesh threadgroup. ctl.x is a loop's
// trip count, ctl.y the threads that stay past the early return. Each function's threads read
// only words that threads which reached the same barriers wrote, so its answer is the reading where
// the threads that reach a barrier synchronize among themselves.
#include <metal_stdlib>
using namespace metal;

struct PartialPayload {
    uint unused;
};

struct PartialVertex {
    float4 position [[position]];
};

using PartialMesh = metal::mesh<PartialVertex, void, 3, 1, topology::triangle>;

// The word a thread writes beside the one it read: its threadgroup's threads and its index.
static uint shape(uint3 tptg, uint tid) {
    return tptg.x | (tptg.y << 8) | (tid << 16);
}

// The first of the thread's two words of out.
static uint slot(uint3 pos, uint3 tpg) {
    return 4u + 2u * (pos.x + tpg.x * pos.y);
}

// Barriers in uniform control flow, and in a loop of ctl.x iterations: each thread rotates its
// threadgroup's words by one each iteration.
[[object]] void mesh_partial_barrier_uniform(object_data PartialPayload &payload [[payload]],
                                             mesh_grid_properties grid,
                                             device uint *out [[buffer(0)]],
                                             constant uint4 &ctl [[buffer(1)]],
                                             uint tid [[thread_index_in_threadgroup]],
                                             uint3 tptg [[threads_per_threadgroup]],
                                             uint3 pos [[thread_position_in_grid]],
                                             uint3 tpg [[threads_per_grid]]) {
    threadgroup uint words[64];
    uint n = tptg.x * tptg.y * tptg.z;
    atomic_fetch_add_explicit((device atomic_uint *)out, 1u, memory_order_relaxed);
    words[tid] = tid * 3u + 1u;
    uint acc = 0u;
    for (uint i = 0u; i < ctl.x; i++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint next = words[(tid + 1u) % n];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        acc += next;
        words[tid] = next + 100u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint at = slot(pos, tpg);
    out[at] = acc + words[(tid + 2u) % n];
    out[at + 1u] = shape(tptg, tid);
    if (tid == 0u) {
        grid.set_threadgroups_per_grid(uint3(0, 0, 0));
    }
}

// A divergent early return: threads at or past ctl.y write a marker and return before the
// barrier; the rest exchange words across it.
[[object]] void mesh_partial_barrier_early_return(object_data PartialPayload &payload [[payload]],
                                                  mesh_grid_properties grid,
                                                  device uint *out [[buffer(0)]],
                                                  constant uint4 &ctl [[buffer(1)]],
                                                  uint tid [[thread_index_in_threadgroup]],
                                                  uint3 tptg [[threads_per_threadgroup]],
                                                  uint3 pos [[thread_position_in_grid]],
                                                  uint3 tpg [[threads_per_grid]]) {
    threadgroup uint words[64];
    uint n = tptg.x * tptg.y * tptg.z;
    uint stay = min(ctl.y, n);
    uint at = slot(pos, tpg);
    atomic_fetch_add_explicit((device atomic_uint *)out, 1u, memory_order_relaxed);
    if (tid == 0u) {
        grid.set_threadgroups_per_grid(uint3(0, 0, 0));
    }
    if (tid >= stay) {
        out[at] = 0xE0000000u | tid;
        out[at + 1u] = shape(tptg, tid);
        return;
    }
    words[tid] = tid * 3u + 1u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    out[at] = words[(tid + 1u) % stay];
    out[at + 1u] = shape(tptg, tid);
}

// A barrier under a thread-dependent condition: threads below ctl.y take it and exchange words
// across it, the rest write a marker and take no barrier.
[[object]] void mesh_partial_barrier_if(object_data PartialPayload &payload [[payload]],
                                        mesh_grid_properties grid,
                                        device uint *out [[buffer(0)]],
                                        constant uint4 &ctl [[buffer(1)]],
                                        uint tid [[thread_index_in_threadgroup]],
                                        uint3 tptg [[threads_per_threadgroup]],
                                        uint3 pos [[thread_position_in_grid]],
                                        uint3 tpg [[threads_per_grid]]) {
    threadgroup uint words[64];
    uint n = tptg.x * tptg.y * tptg.z;
    uint stay = min(ctl.y, n);
    uint at = slot(pos, tpg);
    atomic_fetch_add_explicit((device atomic_uint *)out, 1u, memory_order_relaxed);
    if (tid < stay) {
        words[tid] = tid * 5u + 2u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        out[at] = words[(tid + 1u) % stay];
    } else {
        out[at] = 0xE1000000u | tid;
    }
    out[at + 1u] = shape(tptg, tid);
    if (tid == 0u) {
        grid.set_threadgroups_per_grid(uint3(0, 0, 0));
    }
}

[[mesh]] void mesh_partial_barrier_mesh(PartialMesh output) {
}

fragment uint4 mesh_partial_barrier_fragment() {
    return uint4(1);
}
