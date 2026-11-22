// Owned fixture (written for this repository): a kernel's simdgroup and quad lane moves in a
// branch, each from a lane that takes part. lane_moves_in_a_branch_of_a_kernel makes all eighteen
// moves in a branch the even lanes of a kernel take, with the value loaded before the branch, each
// from an even lane: a thread reads data only from an active thread (MSL 4.1 section 6.10.2).
// Loop-free.
#include <metal_stdlib>
using namespace metal;

// Every simdgroup and quad move in a branch the even lanes of a kernel take, each reading an even
// lane or none (K = 18).
kernel void lane_moves_in_a_branch_of_a_kernel(const device uint *in [[buffer(0)]],
                                               device uint *out [[buffer(1)]],
                                               uint t [[thread_position_in_grid]],
                                               uint lane [[thread_index_in_simdgroup]],
                                               uint q [[thread_index_in_quadgroup]])
{
    const uint x = in[t];
    device uint *o = out + 18u * t;
    if ((lane & 1u) == 0u) {
        o[0] = simd_shuffle(x, ushort(lane ^ 6u));
        o[1] = simd_shuffle_xor(x, ushort(2));
        o[2] = simd_shuffle_up(x, ushort(2));
        o[3] = simd_shuffle_down(x, ushort(4));
        o[4] = simd_shuffle_rotate_up(x, ushort(2));
        o[5] = simd_shuffle_rotate_down(x, ushort(2));
        o[6] = simd_broadcast(x, ushort(20));
        o[7] = simd_shuffle_and_fill_up(x, ~x, ushort(2));
        o[8] = simd_shuffle_and_fill_down(x, ~x, ushort(2));
        o[9] = quad_shuffle(x, ushort(q ^ 2u));
        o[10] = quad_shuffle_xor(x, ushort(2));
        o[11] = quad_broadcast(x, ushort(2));
        o[12] = quad_shuffle_up(x, ushort(2));
        o[13] = quad_shuffle_down(x, ushort(2));
        o[14] = quad_shuffle_rotate_up(x, ushort(2));
        o[15] = quad_shuffle_rotate_down(x, ushort(2));
        o[16] = quad_shuffle_and_fill_up(x, ~x, ushort(2));
        o[17] = quad_shuffle_and_fill_down(x, ~x, ushort(2));
    }
}
