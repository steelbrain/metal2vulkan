// Owned fixture (written for this repository): simdgroup and quad lane moves whose source lane
// does not take part, one family per vertex entry, and a kernel's moves in a branch from lanes
// that take part. The vertex functions run over
// rasterization-disabled points and read in[v] for vertex v; a draw of 38 points leaves lanes 6 to
// 31 of its second simdgroup past the draw, so its second quad has two live lanes. A vertex
// function's lane is the number of active lanes below it at entry, so each takes it from
// simd_prefix_exclusive_sum before any move. Each vertex function makes its first moves with every
// vertex of the draw, then its last ones in a branch only the even vertices take. Vertex v writes
// its K words to out[K v .. K v + K - 1]; an odd vertex leaves the branch's words unwritten.
// No move in a branch repeats one made before it: Metal's compiler merges a repeated move with
// the one before the branch, which every lane made. Metal reads all-zero bits from a lane that does
// not take part. lane_moves_in_a_branch_of_a_kernel makes all eighteen moves in a branch the even
// lanes of a kernel take, with the value loaded before the branch, each from an even lane: a
// thread reads data only from an active thread (MSL 4.1 section 6.10.2). The vertex functions'
// fills take an explicit modulo, the form Metal's compiler takes in a vertex function. Loop-free.
#include <metal_stdlib>
using namespace metal;

// simd_shuffle by a lane index and simd_shuffle_xor (K = 6).
vertex void simd_shuffles_by_index_and_xor(uint vid [[vertex_id]],
                                           const device uint *in [[buffer(0)]],
                                           device uint *out [[buffer(1)]])
{
    const uint x = in[vid];
    const uint lane = simd_prefix_exclusive_sum(1u);
    device uint *o = out + 6u * vid;
    o[0] = simd_shuffle(x, ushort((lane + 9u) & 31u));
    o[1] = simd_shuffle(x, ushort(31u - lane));
    o[2] = simd_shuffle_xor(x, ushort(8));
    if ((vid & 1u) == 0u) {
        o[3] = simd_shuffle(x, ushort((lane + 1u) & 31u));
        o[4] = simd_shuffle(x, ushort((lane + 2u) & 31u));
        o[5] = simd_shuffle_xor(x, ushort(1));
    }
}

// simd_shuffle_up and simd_shuffle_down (K = 5).
vertex void simd_shuffles_up_and_down(uint vid [[vertex_id]],
                                      const device uint *in [[buffer(0)]],
                                      device uint *out [[buffer(1)]])
{
    const uint x = in[vid];
    device uint *o = out + 5u * vid;
    o[0] = simd_shuffle_up(x, ushort(3));
    o[1] = simd_shuffle_down(x, ushort(5));
    if ((vid & 1u) == 0u) {
        o[2] = simd_shuffle_up(x, ushort(1));
        o[3] = simd_shuffle_down(x, ushort(1));
        o[4] = simd_shuffle_down(x, ushort(2));
    }
}

// simd_shuffle_rotate_up and _down, and the quad rotations (K = 6).
vertex void simd_and_quad_rotations(uint vid [[vertex_id]],
                                    const device uint *in [[buffer(0)]],
                                    device uint *out [[buffer(1)]])
{
    const uint x = in[vid];
    device uint *o = out + 6u * vid;
    o[0] = simd_shuffle_rotate_up(x, ushort(4));
    o[1] = simd_shuffle_rotate_down(x, ushort(6));
    o[2] = quad_shuffle_rotate_down(x, ushort(1));
    if ((vid & 1u) == 0u) {
        o[3] = simd_shuffle_rotate_up(x, ushort(1));
        o[4] = simd_shuffle_rotate_down(x, ushort(2));
        o[5] = quad_shuffle_rotate_up(x, ushort(1));
    }
}

// simd_broadcast by a constant lane (K = 4).
vertex void simd_broadcasts(uint vid [[vertex_id]],
                            const device uint *in [[buffer(0)]],
                            device uint *out [[buffer(1)]])
{
    const uint x = in[vid];
    device uint *o = out + 4u * vid;
    o[0] = simd_broadcast(x, ushort(21));
    o[1] = simd_broadcast(x, ushort(3));
    if ((vid & 1u) == 0u) {
        o[2] = simd_broadcast(x, ushort(5));
        o[3] = simd_broadcast(x, ushort(4));
    }
}

// simd_shuffle_and_fill_up and _down, the filling value ~x (K = 4).
vertex void simd_fills(uint vid [[vertex_id]],
                       const device uint *in [[buffer(0)]],
                       device uint *out [[buffer(1)]])
{
    const uint x = in[vid];
    device uint *o = out + 4u * vid;
    o[0] = simd_shuffle_and_fill_up(x, ~x, ushort(2), ushort(32));
    o[1] = simd_shuffle_and_fill_down(x, ~x, ushort(3), ushort(32));
    if ((vid & 1u) == 0u) {
        o[2] = simd_shuffle_and_fill_up(x, ~x, ushort(1), ushort(32));
        o[3] = simd_shuffle_and_fill_down(x, ~x, ushort(2), ushort(8));
    }
}

// quad_shuffle by a lane index, quad_shuffle_xor, quad_broadcast, the relative quad shuffles and
// the quad fills (K = 9).
vertex void quad_moves(uint vid [[vertex_id]],
                       const device uint *in [[buffer(0)]],
                       device uint *out [[buffer(1)]])
{
    const uint x = in[vid];
    const uint q = simd_prefix_exclusive_sum(1u) & 3u;
    device uint *o = out + 9u * vid;
    o[0] = quad_shuffle(x, ushort(q ^ 2u));
    o[1] = quad_shuffle_xor(x, ushort(1));
    o[2] = quad_broadcast(x, ushort(3));
    o[3] = quad_shuffle_up(x, ushort(1));
    o[4] = quad_shuffle_down(x, ushort(1));
    o[5] = quad_shuffle_and_fill_down(x, ~x, ushort(2), ushort(4));
    if ((vid & 1u) == 0u) {
        o[6] = quad_shuffle_xor(x, ushort(3));
        o[7] = quad_broadcast(x, ushort(2));
        o[8] = quad_shuffle_and_fill_up(x, ~x, ushort(1), ushort(4));
    }
}

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
