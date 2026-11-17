// Owned fixture: a loop whose trip count is the thread's row, each iteration a simdgroup sum over
// the threads still in it. A kernel whose simdgroups are emulated runs the loop predicated, and
// only its threads' positions along y make the loop divergent: in a threadgroup of one thread
// along y every thread would run it alike, which the translator states as the unit axes that
// relieve the walk. Thread (x, y) runs (y & 3) + 1 iterations (y its row in the grid); iteration
// i adds i + 1 times simd_sum(1u), the threads of its simdgroup still in the loop, and out[t] is
// the total, t = thread_index_in_threadgroup. Written for this repository; no third-party source.
#include <metal_stdlib>
using namespace metal;

kernel void unit_axes_walk(device uint *out [[buffer(0)]],
                           uint2 gid [[thread_position_in_grid]],
                           uint t [[thread_index_in_threadgroup]]) {
    uint acc = 0u;
    uint n = (gid.y & 3u) + 1u;
    for (uint i = 0u; i < n; i++) {
        acc += simd_sum(1u) * (i + 1u);
    }
    out[t] = acc;
}
