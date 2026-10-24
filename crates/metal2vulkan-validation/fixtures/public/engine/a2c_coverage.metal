// Owned fixture: the fragment outputs an alpha-to-coverage case needs, which no committed fixture
// has. `rt6_render_state`'s `rt6_half` writes a fixed alpha of 0.5 and reports nothing, so the
// coverage Metal derived from an alpha can only be read there through a resolve, which averages the
// samples and so cannot say which of them was covered.
//
// Three things these functions add, each one an observable a case of
// `MTLRenderPipelineDescriptor.alphaToCoverageEnabled` cannot otherwise reach:
//
// - the colour comes from a buffer, so one pipeline asks every alpha rather than one;
// - `a2c_report` writes the coverage its invocation was given, as an integer, beside the colour, so
//   which samples the alpha covered is an exact answer on every device rather than a resolve's
//   average. It also counts its runs, which says whether Metal ran it per pixel or per sample;
// - `a2c_report_masked` writes a sample mask of its own beside that colour, which is how the
//   coverage the alpha produced and the mask the shader writes are compared in one draw; and
//   `a2c_rg` has no alpha component at all, which is the shape that leaves the coverage as
//   rasterized.
//
// Loop-free, no indexing outside a bound buffer.
#include <metal_stdlib>
using namespace metal;

struct A2cOut {
    float4 position [[position]];
};

// Metal's fullscreen triangle, (-1, -1), (3, -1), (-1, 3), at depth 0.5.
vertex A2cOut a2c_full(uint vid [[vertex_id]]) {
    float2 a = float2(float(vid & 1u), float(vid >> 1));
    return A2cOut { float4(4.0 * a - 1.0, 0.5, 1.0) };
}

// The buffer's colour, whose alpha is what alpha-to-coverage reads.
fragment float4 a2c_color(constant float4 &c [[buffer(0)]]) {
    return c;
}

// Two words per pixel of a target `width` wide: the runs of this pixel, and the coverage those
// runs were given, ORed. A case reads them as integers, so no resolve and no average is in the way.
struct A2cReport {
    float4 color [[color(0)]];
};

fragment A2cReport a2c_report(A2cOut in [[stage_in]], uint coverage [[sample_mask]],
                              constant float4 &c [[buffer(0)]],
                              constant uint &width [[buffer(1)]],
                              device atomic_uint *out [[buffer(2)]]) {
    uint i = (uint(in.position.y) * width + uint(in.position.x)) * 2u;
    atomic_fetch_add_explicit(&out[i], 1u, memory_order_relaxed);
    atomic_fetch_or_explicit(&out[i + 1u], coverage, memory_order_relaxed);
    return A2cReport { c };
}

// The same, and a sample mask of the shader's own beside the colour: what the draw stores is the
// coverage from the alpha and this mask together, and `out` says what the fragment was given
// before its own mask applied.
struct A2cReportMasked {
    float4 color [[color(0)]];
    uint mask [[sample_mask]];
};

fragment A2cReportMasked a2c_report_masked(A2cOut in [[stage_in]], uint coverage [[sample_mask]],
                                           constant float4 &c [[buffer(0)]],
                                           constant uint &width [[buffer(1)]],
                                           device atomic_uint *out [[buffer(2)]],
                                           constant uint &mask [[buffer(3)]]) {
    uint i = (uint(in.position.y) * width + uint(in.position.x)) * 2u;
    atomic_fetch_add_explicit(&out[i], 1u, memory_order_relaxed);
    atomic_fetch_or_explicit(&out[i + 1u], coverage, memory_order_relaxed);
    return A2cReportMasked { c, mask };
}

// A two-component output: there is no alpha channel for alpha-to-coverage to read.
fragment float2 a2c_rg(constant float4 &c [[buffer(0)]]) {
    return c.rg;
}
