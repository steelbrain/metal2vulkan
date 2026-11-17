// Owned fixture: a tree reduction over threadgroup memory indexed by the thread's position along
// x, a threadgroup barrier per step and none between a step's load and its store. In a
// threadgroup of two rows within one simdgroup (16 x 2, or 12 x 2), the rows' threads share x
// and so word x: only the simdgroup's lockstep orders one row's load of tg[x] before the other
// row's store to it, so both rows add the same words. A one-row threadgroup meets no such pair.
// The first 16 words start at 0 (a threadgroup has at least 16 threads), then word x at x; every
// thread writes tg[0], the sum of x below min(16, width): 120 at a width of 16 or more, 66 at 12. Written for this repository;
// no third-party source.
#include <metal_stdlib>
using namespace metal;

kernel void row_lockstep_tree(device uint *out [[buffer(0)]],
                              ushort2 position [[thread_position_in_threadgroup]],
                              ushort t [[thread_index_in_threadgroup]],
                              ushort2 p [[thread_position_in_grid]],
                              ushort2 grid [[threads_per_grid]]) {
    ushort x = position.x;
    threadgroup uint tg[64];
    if (t < 16u) {
        tg[t] = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tg[x] = x;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint h = 8u; h > 0u; h >>= 1) {
        if (x < h) {
            tg[x] += tg[x + h];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    out[uint(p.y) * uint(grid.x) + uint(p.x)] = tg[0];
}
