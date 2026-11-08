// Owned fixture: a fragment function that reads, at its own pixel, the texture its pass renders and
// discards the fragment by the texel it read, as a compositor's coverage test does; the runtime's
// lane draws it with `tob_renders.metal`'s `tob_rect`. Written for this repository; no third-party
// source. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct SdVertex {
    float4 position [[position]];
};

// The texel of t under the pixel; the fragment discarded where the texel's red is above 0.4.
fragment float4 sd_read_discarding(SdVertex in [[stage_in]],
                                   texture2d<float, access::read> t [[texture(0)]]) {
    float4 c = t.read(uint2(in.position.xy));
    if (c.r > 0.4) {
        discard_fragment();
    }
    return c;
}
