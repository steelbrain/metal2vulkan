#include <metal_stdlib>
using namespace metal;

// reconverge_sample's tail through a texture chosen by a ternary on two texture handles: per pixel
// or per 4 by 4 block (MSL's default is a non-uniform choice), or by a value the whole draw
// shares, drawn as small triangles. Each fragment reads 256 words of data (buffer
// 0) and two r16_float sources of 1024 by 2 texels and 11 levels: level n holds n + 1 in the
// first (texture 0) and 21 + n in the second (texture 1).

struct RsCover {
    float4 position [[position]];
};

// One triangle over the whole target.
vertex RsCover rs_cover(uint id [[vertex_id]]) {
    float2 at = float2(float((id << 1) & 2u), float(id & 2u));
    RsCover out;
    out.position = float4(at * 2.0f - 1.0f, 0.0f, 1.0f);
    return out;
}

// Small triangles, one in each 8 by 8 cell of a 64 by 64 target (vertex 3k + c is corner c of
// cell k), whose edges cross quads, so that many waves start at a helper lane.
vertex RsCover rs_cells(uint id [[vertex_id]]) {
    uint cell = id / 3u;
    uint corner = id % 3u;
    float2 corners[3] = {float2(1.3f, 0.7f), float2(6.6f, 1.9f), float2(2.1f, 6.8f)};
    float2 at = float2(float(cell % 8u), float(cell / 8u)) * 8.0f + corners[corner];
    RsCover out;
    out.position = float4(at.x / 32.0f - 1.0f, 1.0f - at.y / 32.0f, 0.0f, 1.0f);
    return out;
}

static uint rs_hash(uint2 q) {
    uint h = (q.x * 0x9E3779B1u) ^ (q.y * 0x85EBCA77u);
    h ^= h >> 15;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12;
    return h;
}

// The nearest sample at a level of detail of exactly q, the fine pair of v, and the trilinear
// sample at q + log2(1.5), as reconverge_sample's rc_tail.
static float4 rs_tail(texture2d<float> source, float2 xy, uint q, float v) {
    constexpr sampler nearest(filter::nearest, mip_filter::nearest, address::repeat);
    constexpr sampler trilinear(filter::linear, mip_filter::linear, address::repeat);
    float k = float(1u << q);
    float2 at = xy * float2(k / 1024.0f, k / 2.0f);
    float level = source.sample(nearest, at).r;
    float blend = source.sample(trilinear, at * 1.5f).r;
    return float4(level, dfdx(v), dfdy(v), blend);
}

// The texture chosen per pixel, sampled inside a selection uniform over 8 by 8 blocks; a block
// outside it writes (0, 0, 0, -5).
fragment float4 rc_select(float4 position [[position]], texture2d<float> first [[texture(0)]],
                          texture2d<float> second [[texture(1)]],
                          device const uint *data [[buffer(0)]]) {
    uint2 p = uint2(position.xy);
    uint h = rs_hash(p);
    uint b = rs_hash(p >> 3u);
    float4 out = float4(0.0f, 0.0f, 0.0f, -5.0f);
    if ((data[b & 255u] & 1u) != 0u) {
        texture2d<float> source = (data[(h >> 8) & 255u] & 1u) != 0u ? second : first;
        float v = float(data[h & 255u] & 255u);
        out = rs_tail(source, position.xy, rs_hash(p >> 1u) % 8u, v);
    }
    return out;
}

// The texture chosen per 4 by 4 block (the same for a quad, not for a wave), sampled inside the
// same selection.
fragment float4 rc_select_quad(float4 position [[position]], texture2d<float> first [[texture(0)]],
                               texture2d<float> second [[texture(1)]],
                               device const uint *data [[buffer(0)]]) {
    uint2 p = uint2(position.xy);
    uint h = rs_hash(p);
    uint b = rs_hash(p >> 3u);
    float4 out = float4(0.0f, 0.0f, 0.0f, -5.0f);
    if ((data[b & 255u] & 1u) != 0u) {
        uint c = rs_hash(p >> 2u);
        texture2d<float> source = (data[(c >> 8) & 255u] & 1u) != 0u ? second : first;
        float v = float(data[h & 255u] & 255u);
        out = rs_tail(source, position.xy, rs_hash(p >> 1u) % 8u, v);
    }
    return out;
}

// The texture chosen by a value the whole draw shares (`pick`, a small constant buffer), sampled
// inside a selection uniform over 8 by 8 blocks.
fragment float4 rc_pick(float4 position [[position]], texture2d<float> first [[texture(0)]],
                        texture2d<float> second [[texture(1)]],
                        device const uint *data [[buffer(0)]], constant uint &pick [[buffer(1)]]) {
    uint2 p = uint2(position.xy);
    uint h = rs_hash(p);
    uint b = rs_hash(p >> 3u);
    float4 out = float4(0.0f, 0.0f, 0.0f, -5.0f);
    if ((data[b & 255u] & 1u) != 0u) {
        texture2d<float> source = pick != 0u ? second : first;
        float v = float(data[h & 255u] & 255u);
        out = rs_tail(source, position.xy, rs_hash(p >> 1u) % 8u, v);
    }
    return out;
}
