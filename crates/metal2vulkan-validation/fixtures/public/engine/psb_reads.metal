// Owned fixture: a rectangle and fragment functions that read the texture their own render pass
// renders, at the fragment's own pixel, as a pass that begins with a memory barrier over the render
// targets reads it: a solid color, a read and a pixel-coordinate sample of texture 0 with its
// channels reordered, a read of texture 0 beside a mask at texture 3, and a read of texture 3 over
// texture 0's alpha. Written for this repository; no third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct PsbVertex {
    float4 position [[position]];
};

// A rectangle, two triangles of six vertices, from corner r.xy to corner r.zw.
vertex PsbVertex psb_rect(uint vid [[vertex_id]], constant float4 &r [[buffer(0)]]) {
    bool right = vid == 1u || vid == 3u || vid == 4u;
    bool up = vid == 2u || vid == 4u || vid == 5u;
    PsbVertex out;
    out.position = float4(mix(r.xy, r.zw, float2(right ? 1.0 : 0.0, up ? 1.0 : 0.0)), 0.0, 1.0);
    return out;
}

fragment float4 psb_solid(PsbVertex in [[stage_in]], constant float4 &c [[buffer(0)]]) {
    return c;
}

// t's texel under the pixel, its channels as (b, r, g), opaque.
fragment float4 psb_read(PsbVertex in [[stage_in]], texture2d<float, access::read> t [[texture(0)]]) {
    float4 c = t.read(uint2(in.position.xy));
    return float4(c.b, c.r, c.g, 1.0);
}

// The same through a nearest sampler in pixel coordinates, at the pixel's center.
fragment float4 psb_sample(PsbVertex in [[stage_in]], texture2d<float> t [[texture(0)]]) {
    constexpr sampler s(coord::pixel, filter::nearest);
    float4 c = t.sample(s, in.position.xy);
    return float4(c.b, c.r, c.g, 1.0);
}

// t's texel under the pixel as psb_read reorders it, with the alpha of m's texel at the pixel
// wrapped to m's extent.
fragment float4 psb_masked(PsbVertex in [[stage_in]],
                           texture2d<float, access::read> t [[texture(0)]],
                           texture2d<float, access::read> m [[texture(3)]]) {
    uint2 p = uint2(in.position.xy);
    float4 c = t.read(p);
    float4 k = m.read(uint2(p.x % m.get_width(), p.y % m.get_height()));
    return float4(c.b, c.r, c.g, k.a);
}

// m's texel at the pixel wrapped to m's extent, its channels as (b, r, g), with the alpha of t's
// texel under the pixel.
fragment float4 psb_under(PsbVertex in [[stage_in]],
                          texture2d<float, access::read> t [[texture(0)]],
                          texture2d<float, access::read> m [[texture(3)]]) {
    uint2 p = uint2(in.position.xy);
    float4 c = t.read(p);
    float4 k = m.read(uint2(p.x % m.get_width(), p.y % m.get_height()));
    return float4(k.b, k.r, k.g, c.a);
}
