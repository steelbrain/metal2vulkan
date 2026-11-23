// metal-fast-math: off
// Owned fixture: the validation executor's fill of a 32-bit float depth texture through the
// runtime: one draw writes every texel of one level and slice from a buffer, as a guest fills a
// depth texture by rendering into it.
//
//  * depth_fill_vertex: one triangle over the whole target, vertex v at clip position
//    (2 (v << 1 & 2) - 1, 2 (v & 2) - 1, 0, 1).
//  * depth_fill_fragment: the pixel at (x, y) writes depth texels[y * width + x] under
//    `depth(any)`, `width` the level's width in texels.
//  * depth_fill_read: thread t stores texel (t % 4, t / 4) of a 2D depth texture four texels
//    wide, as the device tests read a filled texture back.
//
// Loop-free.
#include <metal_stdlib>
using namespace metal;

vertex float4 depth_fill_vertex(uint vertex_id [[vertex_id]]) {
    float2 corner = float2((vertex_id << 1) & 2, vertex_id & 2);
    return float4(corner * 2.0 - 1.0, 0.0, 1.0);
}

struct DepthFillOut {
    float depth [[depth(any)]];
};

fragment DepthFillOut depth_fill_fragment(float4 position [[position]],
                                          device const float *texels [[buffer(0)]],
                                          constant uint &width [[buffer(1)]]) {
    uint2 pixel = uint2(position.xy);
    return {texels[pixel.y * width + pixel.x]};
}

kernel void depth_fill_read(depth2d<float, access::read> depths [[texture(0)]],
                            device float *out [[buffer(0)]],
                            uint t [[thread_position_in_grid]]) {
    out[t] = depths.read(uint2(t % 4, t / 4));
}
