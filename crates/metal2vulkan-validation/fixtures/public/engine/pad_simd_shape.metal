// Owned fixture (written for this repository): how a threadgroup's threads fall into simdgroups.
// Thread t of the grid writes, at out[4 t .. 4 t + 3], its simdgroup index in the threadgroup, its
// lane in the simdgroup, the threads of its simdgroup (a simdgroup sum of 1), and its index in the
// threadgroup. A threadgroup whose threads are not whole simdgroups ends in a partial one.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void simd_shape(device uint *out [[buffer(0)]],
                       uint t [[thread_position_in_grid]],
                       uint i [[thread_index_in_threadgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]])
{
    device uint *o = out + 4u * t;
    o[0] = sg;
    o[1] = lane;
    o[2] = simd_sum(1u);
    o[3] = i;
}
