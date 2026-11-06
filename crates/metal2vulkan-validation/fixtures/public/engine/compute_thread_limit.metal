// Owned fixture: kernels with and without `[[max_total_threads_per_threadgroup(n)]]`, for a
// compute pipeline's thread limits beside its descriptor's (the conformance suite's
// `compute-pipeline-limits` cases and the translator's tests). Each thread adds 1 to out[0] and
// thread 0 of threadgroup 0 stores its threadgroup's threads at out[1] to out[3].
#include <metal_stdlib>
using namespace metal;

static void count(device atomic_uint *out, uint3 tptg, uint3 group, uint tid) {
    atomic_fetch_add_explicit(&out[0], 1u, memory_order_relaxed);
    if (tid == 0u && group.x == 0u && group.y == 0u && group.z == 0u) {
        atomic_store_explicit(&out[1], tptg.x, memory_order_relaxed);
        atomic_store_explicit(&out[2], tptg.y, memory_order_relaxed);
        atomic_store_explicit(&out[3], tptg.z, memory_order_relaxed);
    }
}

kernel void compute_thread_limit_plain(device atomic_uint *out [[buffer(0)]],
                                       uint3 tptg [[threads_per_threadgroup]],
                                       uint3 group [[threadgroup_position_in_grid]],
                                       uint tid [[thread_index_in_threadgroup]]) {
    count(out, tptg, group, tid);
}

[[max_total_threads_per_threadgroup(64)]]
kernel void compute_thread_limit_64(device atomic_uint *out [[buffer(0)]],
                                    uint3 tptg [[threads_per_threadgroup]],
                                    uint3 group [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]]) {
    count(out, tptg, group, tid);
}
