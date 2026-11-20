// Owned fixture: the shapes of a window server's composite on Metal's tile path, for a cost
// measurement. A pass holds a base color (color 0,
// an 8-bit or extended-range target the pass loads and stores) and a wide color (color 1, a
// memoryless `RGBA16Float` the pass clears). Each draw is a rectangle from a buffer's corners,
// its texture coordinates over the unit square:
//
// - tgp_layer: a layer's texture sampled into the base color, which the pipeline blends;
// - tgp_layer_wide: the same into the wide color;
// - tgp_blend_wide: the layer blended over the wide color in the function, which reads it;
// - tgp_resolve: the wide color read and written to the base color, clamped;
// - tgp_blend_base: the layer blended over the base color in the function, which reads it;
// - tgp_tile: a tile function over the implicit imageblock's base color, each cell scaled.
//
// Written for this repository; no third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct TgpVertex {
    float4 position [[position]];
    float2 uv;
};

// A rectangle, two triangles of six vertices, from corner r.xy to corner r.zw.
vertex TgpVertex tgp_rect(uint vid [[vertex_id]], constant float4 &r [[buffer(0)]]) {
    float2 corner = float2(vid == 1u || vid == 3u || vid == 4u ? 1.0 : 0.0,
                           vid == 2u || vid == 4u || vid == 5u ? 1.0 : 0.0);
    TgpVertex out;
    out.position = float4(mix(r.xy, r.zw, corner), 0.0, 1.0);
    out.uv = corner;
    return out;
}

struct TgpBase {
    half4 base [[color(0)]];
};

struct TgpWide {
    half4 wide [[color(1)]];
};

fragment TgpBase tgp_layer(TgpVertex in [[stage_in]], texture2d<half> t [[texture(0)]],
                           sampler s [[sampler(0)]]) {
    TgpBase out;
    out.base = t.sample(s, in.uv);
    return out;
}

fragment TgpWide tgp_layer_wide(TgpVertex in [[stage_in]], texture2d<half> t [[texture(0)]],
                                sampler s [[sampler(0)]]) {
    TgpWide out;
    out.wide = t.sample(s, in.uv);
    return out;
}

fragment TgpWide tgp_blend_wide(TgpVertex in [[stage_in]], half4 wide [[color(1)]],
                                texture2d<half> t [[texture(0)]], sampler s [[sampler(0)]]) {
    half4 src = t.sample(s, in.uv);
    TgpWide out;
    out.wide = src + wide * (1.0h - src.a);
    return out;
}

fragment TgpBase tgp_resolve(TgpVertex in [[stage_in]], half4 wide [[color(1)]]) {
    TgpBase out;
    out.base = clamp(wide, 0.0h, 1.0h);
    return out;
}

fragment TgpBase tgp_blend_base(TgpVertex in [[stage_in]], half4 base [[color(0)]],
                                texture2d<half> t [[texture(0)]], sampler s [[sampler(0)]]) {
    half4 src = t.sample(s, in.uv);
    TgpBase out;
    out.base = src + base * (1.0h - src.a);
    return out;
}

kernel void tgp_tile(imageblock<TgpBase, imageblock_layout_implicit> ib,
                     ushort2 lid [[thread_position_in_threadgroup]]) {
    TgpBase cell = ib.read(lid);
    cell.base = cell.base * 0.5h;
    ib.write(cell, lid);
}
