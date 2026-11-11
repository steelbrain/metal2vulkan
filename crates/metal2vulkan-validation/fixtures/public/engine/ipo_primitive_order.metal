// Owned fixture: the order of ordered read-modify-writes across the primitives of one draw.
// `ipo_mix` reads color attachment 0 (an unsigned integer) and writes it back through one of two
// affine maps that do not commute, `3c + 1` or `5c + 3` (wrapping), picked by the parity of the bits
// set in its primitive's number `k` (`ipo_map`), so a draw of overlapping primitives leaves in each
// pixel a value that holds only where each fragment read what the fragment of the primitive before
// it wrote there. `ipo_rog_texture` and `ipo_rog_buffer` apply the same maps, writing no color, to
// the pixel's texel of a texture, or its word of a buffer (`width` words a row), in raster order
// group 0.
// `ipo_fan` draws fullscreen triangles; `ipo_steps` draws rectangles, each from a corner that steps
// with `k` (`ipo_corner`, in pixels from the target's top left) to the target's right and bottom
// edges, two triangles sharing a diagonal, the target's sides in pixels at buffer 0. `k` is the
// vertex's triangle (or rectangle) in its instance plus the instance's number, so a draw numbers
// its primitives in order either as one instance of many or as many instances of one.
// Written for this repository; no third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct IpoVertex {
    float4 position [[position]];
    uint k [[flat]];
};

vertex IpoVertex ipo_fan(uint vid [[vertex_id]], uint iid [[instance_id]]) {
    uint j = vid % 3u;
    float2 p = float2(float((j << 1) & 2u), float(j & 2u));
    IpoVertex out;
    out.position = float4(p * 2.0 - 1.0, 0.0, 1.0);
    out.k = vid / 3u + iid;
    return out;
}

static uint2 ipo_corner(uint k) {
    return uint2((k * 37u + 1u) % 61u, (k * 11u + 1u) % 23u);
}

vertex IpoVertex ipo_steps(uint vid [[vertex_id]], uint iid [[instance_id]],
                           constant float2 &size [[buffer(0)]]) {
    uint k = vid / 6u + iid;
    // The rectangle's corners 0 1 2, then 2 1 3: bit 0 picks the right edge, bit 1 the bottom.
    uint q = (0x312210u >> (4u * (vid % 6u))) & 0xfu;
    float2 corner = float2(ipo_corner(k));
    float2 px = float2((q & 1u) != 0u ? size.x : corner.x, (q & 2u) != 0u ? size.y : corner.y);
    IpoVertex out;
    out.position = float4(px.x / size.x * 2.0 - 1.0, 1.0 - px.y / size.y * 2.0, 0.0, 1.0);
    out.k = k;
    return out;
}

static uint ipo_map(uint k, uint c) {
    return (popcount(k) & 1u) != 0u ? c * 3u + 1u : c * 5u + 3u;
}

fragment uint ipo_mix(IpoVertex in [[stage_in]], uint c [[color(0)]]) {
    return ipo_map(in.k, c);
}

fragment void ipo_rog_texture(IpoVertex in [[stage_in]],
                              texture2d<uint, access::read_write> t
                              [[texture(0), raster_order_group(0)]]) {
    uint2 p = uint2(in.position.xy);
    t.write(uint4(ipo_map(in.k, t.read(p).x)), p);
}

fragment void ipo_rog_buffer(IpoVertex in [[stage_in]],
                             device uint *words [[buffer(0), raster_order_group(0)]],
                             constant uint &width [[buffer(1)]]) {
    uint i = uint(in.position.y) * width + uint(in.position.x);
    words[i] = ipo_map(in.k, words[i]);
}
