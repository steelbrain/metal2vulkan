#include <metal_stdlib>
using namespace metal;

// An 11-level texture sampled at the implicit level of detail, and a fine dfdx/dfdy pair, after
// (or inside) each control-flow construct a fragment function reconverges from. Every fragment
// reads 256 words of data (buffer 0) and an r16_float source of 1024 by 2 texels whose level n
// holds n + 1 in every texel (texture 0).

struct RcCover {
    float4 position [[position]];
};

// One triangle over the whole target.
vertex RcCover rc_cover(uint id [[vertex_id]]) {
    float2 at = float2(float((id << 1) & 2u), float(id & 2u));
    RcCover out;
    out.position = float4(at * 2.0f - 1.0f, 0.0f, 1.0f);
    return out;
}

static uint rc_hash(uint2 q) {
    uint h = (q.x * 0x9E3779B1u) ^ (q.y * 0x85EBCA77u);
    h ^= h >> 15;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12;
    return h;
}

// What a case observes once its construct is behind it. The coordinate is linear in the pixel
// position with a quad-uniform scale of 2^q texels per pixel, so the level of detail is exactly q
// and the nearest sample reads level q's value, q + 1. The fine pair is of the lane's own v. The
// trilinear sample is at a level of detail of q + log2(1.5), between levels q and q + 1.
static float4 rc_tail(texture2d<float> source, float2 xy, uint q, float v) {
    constexpr sampler nearest(filter::nearest, mip_filter::nearest, address::repeat);
    constexpr sampler trilinear(filter::linear, mip_filter::linear, address::repeat);
    float k = float(1u << q);
    float2 at = xy * float2(k / 1024.0f, k / 2.0f);
    float level = source.sample(nearest, at).r;
    float blend = source.sample(trilinear, at * 1.5f).r;
    return float4(level, dfdx(v), dfdy(v), blend);
}

// After an if/else on data.
fragment float4 rc_if(float4 position [[position]], texture2d<float> source [[texture(0)]],
                      device const uint *data [[buffer(0)]]) {
    uint2 p = uint2(position.xy);
    uint h = rc_hash(p);
    uint q;
    float v;
    if ((data[h & 255u] & 1u) != 0u) {
        v = float(data[(h >> 8) & 255u] & 255u);
        q = rc_hash(p >> 1u) % 8u;
    } else {
        v = 256.0f + float(data[(h >> 16) & 255u] & 255u);
        q = (rc_hash(p >> 1u) + 8u) % 8u;
    }
    return rc_tail(source, position.xy, q, v);
}

// After a loop whose trip count is the pixel's own.
fragment float4 rc_loop(float4 position [[position]], texture2d<float> source [[texture(0)]],
                        device const uint *data [[buffer(0)]]) {
    uint2 p = uint2(position.xy);
    uint h = rc_hash(p);
    uint n = data[h & 255u] % 5u;
    float v = 0.0f;
    for (uint i = 0u; i < n; ++i) {
        v += float(data[(h + 37u * i) & 255u] & 63u);
    }
    return rc_tail(source, position.xy, rc_hash(p >> 1u) % 8u, v);
}

// After a loop a lane leaves by a break to the loop's own merge.
fragment float4 rc_break(float4 position [[position]], texture2d<float> source [[texture(0)]],
                         device const uint *data [[buffer(0)]]) {
    uint2 p = uint2(position.xy);
    uint h = rc_hash(p);
    uint n = 6u + data[255] % 3u;
    float v = 0.0f;
    for (uint i = 0u; i < n; ++i) {
        uint d = data[(h + 13u * i) & 255u];
        if ((d & 3u) == 0u) {
            break;
        }
        v += float(d & 31u);
    }
    return rc_tail(source, position.xy, rc_hash(p >> 1u) % 8u, v);
}

// After a switch with one label per case.
fragment float4 rc_switch(float4 position [[position]], texture2d<float> source [[texture(0)]],
                          device const uint *data [[buffer(0)]]) {
    uint2 p = uint2(position.xy);
    uint h = rc_hash(p);
    float v;
    switch (data[h & 255u] % 4u) {
    case 0u:
        v = float(data[(h >> 8) & 255u] & 127u);
        break;
    case 1u:
        v = 128.0f + float(data[(h >> 16) & 255u] & 127u);
        break;
    case 2u:
        v = 256.0f + float(data[(h >> 24) & 255u] & 127u);
        break;
    default:
        v = 384.0f + float(data[(h >> 4) & 255u] & 63u);
        break;
    }
    return rc_tail(source, position.xy, rc_hash(p >> 1u) % 8u, v);
}

// After a discard.
fragment float4 rc_discard(float4 position [[position]], texture2d<float> source [[texture(0)]],
                           device const uint *data [[buffer(0)]]) {
    uint2 p = uint2(position.xy);
    uint h = rc_hash(p);
    if (data[h & 255u] % 5u == 0u) {
        discard_fragment();
    }
    float v = float(data[(h >> 8) & 255u] & 255u);
    return rc_tail(source, position.xy, rc_hash(p >> 1u) % 8u, v);
}

// Inside selections and a loop whose conditions hold for whole 8 by 8 blocks, so every quad runs
// them together.
fragment float4 rc_grid(float4 position [[position]], texture2d<float> source [[texture(0)]],
                        device const uint *data [[buffer(0)]]) {
    uint2 p = uint2(position.xy);
    uint b = rc_hash(p >> 3u);
    uint q = rc_hash(p >> 1u) % 8u;
    float v = float(data[rc_hash(p) & 255u] & 255u);
    float4 out = float4(0.0f);
    if ((data[b & 255u] & 1u) != 0u) {
        if ((data[(b >> 8) & 255u] & 1u) != 0u) {
            uint n = 1u + data[(b >> 16) & 255u] % 3u;
            for (uint i = 0u; i < n; ++i) {
                if ((data[(b + i) & 255u] & 2u) != 0u) {
                    out += rc_tail(source, position.xy, (q + i) % 8u, v);
                }
            }
        } else {
            out = rc_tail(source, position.xy, q, v + 1.0f);
        }
    } else {
        out = float4(float(data[(b >> 24) & 255u] & 15u), 0.0f, 0.0f, -5.0f);
    }
    return out;
}

// A loop of the same trip count for every lane (data[250], 2) in which one pixel of each quad
// takes a continue in the iteration data[252] (0) names; in the iteration data[251] (1) names,
// all four pixels of the quad sample and take the fine pair.
fragment float4 rc_continue_quad(float4 position [[position]], texture2d<float> source [[texture(0)]],
                                 device const uint *data [[buffer(0)]]) {
    uint2 p = uint2(position.xy);
    uint h = rc_hash(p);
    uint q = rc_hash(p >> 1u) % 8u;
    float v = float(data[h & 255u] & 255u);
    float4 out = float4(0.0f);
    for (uint i = 0u; i < data[250]; ++i) {
        if (i == data[252] && ((p.x | p.y) & 1u) == 0u) {
            v += 1.0f;
            continue;
        }
        if (i == data[251]) {
            out = rc_tail(source, position.xy, q, v);
        }
        v += float(data[(h + i) & 255u] & 7u);
    }
    return out;
}

// The same loop with no continue in the source: an if/else at the end of the body, whose two
// arms each end the iteration.
fragment float4 rc_latches(float4 position [[position]], texture2d<float> source [[texture(0)]],
                           device const uint *data [[buffer(0)]]) {
    uint2 p = uint2(position.xy);
    uint h = rc_hash(p);
    uint q = rc_hash(p >> 1u) % 8u;
    float v = float(data[h & 255u] & 255u);
    float4 out = float4(0.0f);
    for (uint i = 0u; i < data[250]; ++i) {
        if (i == data[251]) {
            out = rc_tail(source, position.xy, q, v);
        }
        if (i == data[252] && ((p.x | p.y) & 1u) == 0u) {
            v += 1.0f;
        } else {
            v += float(data[(h + i) & 255u] & 7u);
        }
    }
    return out;
}
