// Owned fixture: a fragment function that writes **two colour attachments** from two buffers at
// the fragment's own texel, and one that writes the second attachment alone. The shape
// `xrb_dual.metal` already has for dual-source blending (two outputs at two `index`es of colour 0),
// moved across two attachment *indices*, so a case can state different per-attachment pass and
// pipeline state and read an exact texel out of each. The values come from a buffer rather than
// from an interpolated varying (`color_groups.metal`'s `cg_five`) because a varying's last bits are
// the rasterizer's answer and a cross whose point is an exact texel cannot afford a tolerance.
// Written for this repository; no third-party source.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Rt6Pair {
    float4 position [[position]];
};

// Metal's fullscreen triangle, (-1, -1), (3, -1), (-1, 3), at depth 0.5.
vertex Rt6Pair rt6_pair_vertex(uint vid [[vertex_id]]) {
    float2 a = float2(float(vid & 1u), float(vid >> 1));
    return Rt6Pair { float4(4.0 * a - 1.0, 0.5, 1.0) };
}

struct Rt6PairColors {
    float4 zero [[color(0)]];
    float4 one [[color(1)]];
};

// Only the second attachment: the pipeline that leaves colour 0 unwritten, so a case can ask what
// an attachment's own load and store actions leave where no fragment wrote it.
struct Rt6PairSecond {
    float4 one [[color(1)]];
};

// The fragment's texel in a row of `width` texels.
static uint rt6_pair_texel(Rt6Pair input, uint width) {
    return uint(input.position.y) * width + uint(input.position.x);
}

// Colour 0 from buffer 0 and colour 1 from buffer 2, each at the fragment's own texel, so the two
// attachments are written different values at every pixel.
fragment Rt6PairColors rt6_pair_two(Rt6Pair input [[stage_in]],
        const device float4 *zeros [[buffer(0)]], constant uint &width [[buffer(1)]],
        const device float4 *ones [[buffer(2)]]) {
    uint i = rt6_pair_texel(input, width);
    return Rt6PairColors { zeros[i], ones[i] };
}

// Colour 1 alone, from buffer 2 at the fragment's own texel.
fragment Rt6PairSecond rt6_pair_second(Rt6Pair input [[stage_in]],
        constant uint &width [[buffer(1)]], const device float4 *ones [[buffer(2)]]) {
    return Rt6PairSecond { ones[rt6_pair_texel(input, width)] };
}
