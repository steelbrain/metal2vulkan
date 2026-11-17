// Owned fixture: a tree reduction over threadgroup memory indexed by the thread's position along
// x, a threadgroup barrier per step and none between a step's load and its store. In a
// threadgroup of 16 x 2, one simdgroup spans its two rows, whose threads share x and so word x:
// only the simdgroup's lockstep orders one row's load of tg[x] before the other row's store to
// it, so both rows add the same words. A one-row threadgroup meets no such pair. Every thread
// writes tg[0], the sum 0 + 1 + ... + 15 = 120. Written for this repository; no third-party
// source.
#include <metal_stdlib>
using namespace metal;

kernel void row_lockstep_tree(device uint *out [[buffer(0)]],
                              ushort x [[thread_position_in_threadgroup]],
                              uint t [[thread_index_in_threadgroup]],
                              uint g [[threadgroup_position_in_grid]],
                              uint n [[threads_per_threadgroup]]) {
    threadgroup uint tg[64];
    tg[x] = x;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint h = 8u; h > 0u; h >>= 1) {
        if (x < h) {
            tg[x] += tg[x + h];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    out[g * n + t] = tg[0];
}
