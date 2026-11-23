// Owned fixture: a threadgroup load and store at a constant index outside the array, in a branch
// no thread of the case takes. One 64-word threadgroup array, the arena's only object. Thread t of
// each threadgroup stores 3 t + 1 at tg[t]; after a barrier it reads tg[(t + 1) & 63], and where
// t == 0 and flag != 0 also reads and writes tg[t - 1u], which the compiler, knowing t == 0,
// writes as a constant index 0xFFFFFFFF: a constant offset outside the arena, undefined in Metal.
// The case passes flag = 0, so no thread runs that branch, and after a second barrier
// out[gid] = tg[(t + 1) & 63] + tg[t]. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void threadgroup_constant_index_outside_the_array(device uint *out [[buffer(0)]],
                                                         constant uint &flag [[buffer(1)]],
                                                         uint t [[thread_position_in_threadgroup]],
                                                         uint gid [[thread_position_in_grid]]) {
    threadgroup uint tg[64];
    tg[t] = t * 3u + 1u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint v = tg[(t + 1u) & 63u];
    if (t == 0u && flag != 0u) {
        v += tg[t - 1u];
        tg[t - 1u] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    out[gid] = v + tg[t];
}
