// Owned fixture: a pass that loads a second color (two halves) and a stencil `DontCare` and stores
// neither, whose first draw reaches some of the target's tiles and whose later draw reaches every
// tile and reads what the pass's loads left there. `dcs_rect` draws two triangles over the
// rectangle `rect` (left, top, right, bottom, in pixels, at buffer 0) of a target of `size` pixels
// (buffer 1). `dcs_mark` writes (0, 0.5, 0, 1) and the second color's (0.25, 0.5); `dcs_read`
// reads the second color (a framebuffer fetch) and writes (its x, its y, 1, 1); `dcs_flag` writes
// (1, 0, 1, 1).
// Written for this repository; no third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct DcsVertex {
    float4 position [[position]];
};

vertex DcsVertex dcs_rect(uint vid [[vertex_id]], constant float4 &rect [[buffer(0)]],
                          constant float2 &size [[buffer(1)]]) {
    uint q = (0x312210u >> (4u * (vid % 6u))) & 0xfu;
    float2 at = float2((q & 1u) != 0u ? rect.z : rect.x, (q & 2u) != 0u ? rect.w : rect.y);
    DcsVertex out;
    out.position = float4(at.x / size.x * 2.0 - 1.0, 1.0 - at.y / size.y * 2.0, 0.0, 1.0);
    return out;
}

struct DcsTwo {
    half4 color [[color(0)]];
    half2 second [[color(1)]];
};

fragment DcsTwo dcs_mark(DcsVertex in [[stage_in]]) {
    DcsTwo out;
    out.color = half4(0.0h, 0.5h, 0.0h, 1.0h);
    out.second = half2(0.25h, 0.5h);
    return out;
}

fragment half4 dcs_read(DcsVertex in [[stage_in]], half2 second [[color(1)]]) {
    return half4(second.x, second.y, 1.0h, 1.0h);
}

fragment half4 dcs_flag(DcsVertex in [[stage_in]]) { return half4(1.0h, 0.0h, 1.0h, 1.0h); }
