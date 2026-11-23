// Owned fixture: a vertex function that declares a render target slice and a viewport index and
// writes neither, so the compiler returns both `undef`, and a fragment function that reads both.
// Written for this repository; no third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Rt6Unwritten {
    float4 position [[position]];
    uint layer [[render_target_array_index]];
    uint viewport [[viewport_array_index]];
};

// Vertex `vid` of a triangle covering the whole target at depth 0.5; the slice and viewport are
// never written.
vertex Rt6Unwritten rt6_unwritten(uint vid [[vertex_id]]) {
    Rt6Unwritten out;
    float2 corner = float2((vid << 1) & 2u, vid & 2u);
    out.position = float4(corner * 2.0 - 1.0, 0.5, 1.0);
    return out;
}

// The viewport and slice the fragment reads: `0x10000 | viewport << 8 | slice`.
fragment uint rt6_unwritten_read(uint layer [[render_target_array_index]],
                                 uint viewport [[viewport_array_index]]) {
    return 0x10000u | (viewport << 8) | layer;
}
