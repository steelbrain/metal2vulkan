// Owned fixture: a fragment function that reads, at its own pixel, a texture its pass renders as
// color 0 and writes what it read into color 1 alone, so the draw writes nothing of the color it
// reads: the in-place reads of the runtime's lane. Its vertex
// stage is `tob_rect` of `tob_renders.metal`. Written for this repository; no third-party source.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

struct IprVertex {
    float4 position [[position]];
};

struct IprSecond {
    float4 second [[color(1)]];
};

// The texel of t under the pixel, into color 1.
fragment IprSecond ipr_read_second(IprVertex in [[stage_in]],
                                   texture2d<float, access::read> t [[texture(0)]]) {
    IprSecond out;
    out.second = t.read(uint2(in.position.xy));
    return out;
}
