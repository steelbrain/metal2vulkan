// Owned fixture: fragment functions that read, at their own pixel, the texture their pass renders
// and select by the texel they read, as a compositor's blend does; the runtime's lane draws them
// with `tob_renders.metal`'s `tob_rect`. Written for this repository; no third-party source.
// Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

struct SsVertex {
    float4 position [[position]];
};

// The texel of t under the pixel, its red halved where it is above 0.4. The arm also adds the
// sine of the texel's green scaled by two to its blue, 0 for a texel of green 0, which keeps the
// arm a branch of its own.
fragment float4 ss_read_selecting(SsVertex in [[stage_in]],
                                  texture2d<float, access::read> t [[texture(0)]]) {
    float4 c = t.read(uint2(in.position.xy));
    float4 o = c;
    if (c.r > 0.4) {
        o.r = c.r * 0.5 + sin(c.g) * exp2(c.b);
    }
    return o;
}

// The same selection, whose arm also adds the texel's green times the derivative of the
// fragment's x across its quad: 0 for a texel of green 0.
fragment float4 ss_read_selecting_dfdx(SsVertex in [[stage_in]],
                                       texture2d<float, access::read> t [[texture(0)]]) {
    float4 c = t.read(uint2(in.position.xy));
    float4 o = c;
    if (c.r > 0.4) {
        o.r = c.r * 0.5 + sin(c.g) * exp2(c.b) + c.g * dfdx(in.position.x);
    }
    return o;
}
