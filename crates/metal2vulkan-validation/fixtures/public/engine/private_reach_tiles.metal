// Owned MSL: kernels written for their largest tile, as a convolution's are, each thread keeping
// an accumulator `float acc[32][32][4][4][4]` (256 KiB) of which the pipeline's tile sizes reach
// a corner, and handing each sum to a visible function it declares and does not define
// (`private_reach_functions.metal` defines it; the pipeline links it).
//
// `private_reach_tiles` runs its tile loops `tile_p` and `tile_o` times (function constants 0 and
// 1) and its inner loop 4 times: with the tiles 2 and 4 it reaches acc[0..2][0..4][0..4][0][0],
// 32 floats. Each sum is 16 steps of `in[(tid + step) % 64] * (p * 16 + o * 4 + k + 1)`, and word
// `tid * 128 + w` of `out` is `postfix(sum, w)` for the sum's place `w = (p * tile_o + o) * 4 + k`.
// With small whole numbers in `in`, every value is a whole number a float holds exactly.
//
// `private_reach_rows` is the same but for its outer loop, which runs `in[64]` times, a count only
// the run knows: its first index has no bound, so it keeps the whole accumulator.
#include <metal_stdlib>
using namespace metal;

constant uint tile_p [[function_constant(0)]];
constant uint tile_o [[function_constant(1)]];

[[visible]] float postfix(float x, uint w);

kernel void private_reach_tiles(device float *out [[buffer(0)]],
                                device const float *in [[buffer(1)]],
                                uint tid [[thread_position_in_grid]]) {
    float acc[32][32][4][4][4];
    for (uint p = 0; p < tile_p; p++)
        for (uint o = 0; o < tile_o; o++)
            for (uint k = 0; k < 4; k++)
                acc[p][o][k][0][0] = 0.0f;
    for (uint step = 0; step < 16; step++) {
        float x = in[(tid + step) % 64];
        for (uint p = 0; p < tile_p; p++)
            for (uint o = 0; o < tile_o; o++)
                for (uint k = 0; k < 4; k++)
                    acc[p][o][k][0][0] += x * float(p * 16 + o * 4 + k + 1);
    }
    for (uint p = 0; p < tile_p; p++)
        for (uint o = 0; o < tile_o; o++)
            for (uint k = 0; k < 4; k++) {
                uint w = (p * tile_o + o) * 4 + k;
                out[tid * 128 + w] = postfix(acc[p][o][k][0][0], w);
            }
}

kernel void private_reach_rows(device float *out [[buffer(0)]],
                               device const float *in [[buffer(1)]],
                               uint tid [[thread_position_in_grid]]) {
    float acc[32][32][4][4][4];
    uint rows = uint(in[64]);
    for (uint p = 0; p < rows; p++)
        for (uint o = 0; o < tile_o; o++)
            for (uint k = 0; k < 4; k++)
                acc[p][o][k][0][0] = 0.0f;
    for (uint step = 0; step < 16; step++) {
        float x = in[(tid + step) % 64];
        for (uint p = 0; p < rows; p++)
            for (uint o = 0; o < tile_o; o++)
                for (uint k = 0; k < 4; k++)
                    acc[p][o][k][0][0] += x * float(p * 16 + o * 4 + k + 1);
    }
    for (uint p = 0; p < rows; p++)
        for (uint o = 0; o < tile_o; o++)
            for (uint k = 0; k < 4; k++) {
                uint w = (p * tile_o + o) * 4 + k;
                out[tid * 128 + w] = postfix(acc[p][o][k][0][0], w);
            }
}
