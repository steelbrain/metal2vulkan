// Owned fixture: a vertex function that writes six clip distances, so a rate-mapped pass's fan-out
// variant of a draw of it, which adds four of its own, needs ten, more than a device of eight
// carries (`pipeline::link::geometry`), and the runtime draws it region by region. Each vertex lies
// at the point of buffer 0 its vertex ID names, as `tob_triangle` of `tob_renders.metal` places it,
// and each clip distance is 4 more than its x, which no point of x at least -1 clips. Written for
// this repository; no third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct FanClipVertex {
    float4 position [[position]];
    float clip [[clip_distance]] [6];
};

vertex FanClipVertex fan_clip_triangle(uint vid [[vertex_id]], constant float2 *p [[buffer(0)]]) {
    float2 at = p[vid];
    FanClipVertex out;
    out.position = float4(at, 0.0, 1.0);
    out.clip[0] = 4.0 + at.x;
    out.clip[1] = 4.0 + at.x;
    out.clip[2] = 4.0 + at.x;
    out.clip[3] = 4.0 + at.x;
    out.clip[4] = 4.0 + at.x;
    out.clip[5] = 4.0 + at.x;
    return out;
}
