// Owned fixture: a vertex function that reads its render target slice and viewport per triangle,
// from a device buffer at the triangle's index (`vertex_id / 3`), so every vertex of a triangle
// writes the same ones, as Metal requires, though nothing in its structure shows it; and a
// fragment function that reads both. Written for this repository; no third-party source.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Rt6PerPrimitive {
    float4 position [[position]];
    uint layer [[render_target_array_index]];
    uint viewport [[viewport_array_index]];
    uint index [[flat]];
};

// Vertex `vid` at `positions[vid]`, writing slice `indices[vid / 3] & 0xff`, viewport
// `indices[vid / 3] >> 8`, and the index itself.
vertex Rt6PerPrimitive rt6_per_primitive(uint vid [[vertex_id]],
                                         const device float4 *positions [[buffer(0)]],
                                         const device uint *indices [[buffer(1)]]) {
    uint index = indices[vid / 3];
    Rt6PerPrimitive out;
    out.position = positions[vid];
    out.layer = index & 0xffu;
    out.viewport = index >> 8;
    out.index = index;
    return out;
}

struct Rt6PerPrimitiveIn {
    uint index [[flat]];
};

// The index the vertex function read, and the viewport and slice the fragment reads:
// `index << 16 | viewport << 8 | slice`.
fragment uint rt6_per_primitive_read(Rt6PerPrimitiveIn in [[stage_in]],
                                     uint layer [[render_target_array_index]],
                                     uint viewport [[viewport_array_index]]) {
    return (in.index << 16) | (viewport << 8) | layer;
}
