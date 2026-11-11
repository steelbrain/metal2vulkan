// Owned fixture: a pass whose fetching draw sits between plain draws, which a device without
// rasterization-order color access runs as several render pass instances, over a stencil the
// draws before the fetch write and the draws from it test. Each fragment function writes color
// attachment 0 (an 8-bit normalized format) as one code `c` in its red, green and blue channels
// and 255 in alpha; `sfo_plane` writes a second color, `cover`, two floats, as well: its
// primitive's `k` and 1. `sfo_image` writes `sfo_code(k)` for its primitive `k`; `sfo_filter`
// reads color attachments 0 and 1 and writes back `3c + 1` or `5c + 3` by the parity of the bits
// set in `k`, plus the `k` the plane wrote into `cover` (wrapping at 256), so overlapping fetching
// primitives leave a code that holds only where each read what the one before it wrote and what
// the plane wrote; `sfo_plane` writes 7, `sfo_cover` 200.
// `sfo_fan` draws one fullscreen triangle a primitive; `sfo_steps` draws rectangles from a corner
// that steps with `k` (`sfo_corner`, in pixels from the target's top left) to the target's right
// and bottom edges, the target's sides in pixels at buffer 0. `k` is the primitive in its instance
// plus the instance's number, the draw's base instance included.
// Written for this repository; no third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct SfoVertex {
    float4 position [[position]];
    uint k [[flat]];
};

vertex SfoVertex sfo_fan(uint vid [[vertex_id]], uint iid [[instance_id]]) {
    uint j = vid % 3u;
    float2 p = float2(float((j << 1) & 2u), float(j & 2u));
    SfoVertex out;
    out.position = float4(p * 2.0 - 1.0, 0.0, 1.0);
    out.k = vid / 3u + iid;
    return out;
}

static uint2 sfo_corner(uint k) { return uint2((k * 37u + 1u) % 61u, (k * 11u + 1u) % 23u); }

vertex SfoVertex sfo_steps(uint vid [[vertex_id]], uint iid [[instance_id]],
                           constant float2 &size [[buffer(0)]]) {
    uint k = vid / 6u + iid;
    uint q = (0x312210u >> (4u * (vid % 6u))) & 0xfu;
    float2 corner = float2(sfo_corner(k));
    float2 at = float2((q & 1u) != 0u ? size.x : corner.x, (q & 2u) != 0u ? size.y : corner.y);
    SfoVertex out;
    out.position = float4(at.x / size.x * 2.0 - 1.0, 1.0 - at.y / size.y * 2.0, 0.0, 1.0);
    out.k = k;
    return out;
}

static float4 sfo_color(uint c) { return float4(float3(float(c & 255u) / 255.0), 1.0); }

static uint sfo_code(uint k) { return (k * 29u + 11u) & 255u; }

struct SfoPlane {
    float4 color [[color(0)]];
    float2 cover [[color(1)]];
};

fragment SfoPlane sfo_plane(SfoVertex in [[stage_in]]) {
    SfoPlane out;
    out.color = sfo_color(7u);
    out.cover = float2(float(in.k), 1.0);
    return out;
}

fragment float4 sfo_image(SfoVertex in [[stage_in]]) { return sfo_color(sfo_code(in.k)); }

fragment float4 sfo_filter(SfoVertex in [[stage_in]], float4 c [[color(0)]],
                           float2 cover [[color(1)]]) {
    uint code = uint(rint(c.r * 255.0));
    uint next = (popcount(in.k) & 1u) != 0u ? code * 3u + 1u : code * 5u + 3u;
    return sfo_color(next + uint(rint(cover.x)));
}

fragment float4 sfo_cover(SfoVertex in [[stage_in]]) { return sfo_color(200u); }
