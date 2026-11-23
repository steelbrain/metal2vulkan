// Owned fixture: a kernel's `[[stage_in]]` attribute beside threadgroup memory and a threadgroup
// barrier, so it observes its threadgroup partition and a body that holds a workgroup barrier.
//
// A `dispatchThreads` grid that is not a whole number of threadgroups runs its last threadgroup
// partial: each thread fetches its element, shares it through threadgroup memory, and reads the
// next thread's of its own threadgroup, so the threadgroup's size and every thread's index show.
#include <metal_stdlib>
using namespace metal;

struct Item {
    uint v [[attribute(0)]];
};

// Thread `gid`, `t` of its threadgroup of `n` threads, reports the word its threadgroup's next
// thread fetched (the last one's next is its first), with `n`, `gid` and `t`:
//     out[gid] = (word of thread (t + 1) % n, n, gid, t)
kernel void stage_in_neighbour(
    Item in [[stage_in]],
    device uint4 *out [[buffer(0)]],
    uint gid [[thread_position_in_grid]],
    uint t [[thread_index_in_threadgroup]],
    uint n [[threads_per_threadgroup]])
{
    threadgroup uint shared[1024];
    shared[t] = in.v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    out[gid] = uint4(shared[(t + 1) % n], n, gid, t);
}
