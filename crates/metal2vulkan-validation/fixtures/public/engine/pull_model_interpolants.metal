// Owned fixture: Metal's pull-model interpolation, `interpolant<T, P>` fragment inputs read
// through `interpolate_at_center()`, `interpolate_at_centroid()`, `interpolate_at_sample(uint)`
// and `interpolate_at_offset(float2)`. `pm_place` places vertex v at the clip-space position
// buffer 0 holds for it and hands it buffer 1's value as every varying. Each `PmPair` function
// returns what one form interpolates of a perspective and a no-perspective `float4` input, into
// color attachments 0 and 1; `pm_narrow` and `pm_half` read every form of `float2`, `float`,
// `half` and `half4` inputs. `pm_offset_after_discard` reads at an offset after a discard, where
// some invocations of the pixel's quad may have left. `pm_selected` reads an interpolant chosen at
// run time, a form the translator refuses. `pm_read_samples` reads a multisampled attachment back,
// sample by sample.
// The conformance probe `pull_model_interpolation.m` measured Metal's answers for these
// functions.
// Written for this repository; no third-party source.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

struct PmVarying {
    float4 position [[position]];
    float4 p;
    float4 n;
    float2 p2;
    float s;
    half4 hp;
    half hn;
};

vertex PmVarying pm_place(uint vid [[vertex_id]], const device float4 *at [[buffer(0)]],
                          const device float4 *values [[buffer(1)]]) {
    PmVarying out;
    float4 v = values[vid];
    out.position = at[vid];
    out.p = v;
    out.n = v;
    out.p2 = v.xy;
    out.s = v.z;
    out.hp = half4(v);
    out.hn = half(v.w);
    return out;
}

struct PmIn {
    interpolant<float4, interpolation::perspective> p;
    interpolant<float4, interpolation::no_perspective> n;
};

struct PmPair {
    float4 p [[color(0)]];
    float4 n [[color(1)]];
};

fragment PmPair pm_center(PmIn in [[stage_in]]) {
    return {in.p.interpolate_at_center(), in.n.interpolate_at_center()};
}

fragment PmPair pm_centroid(PmIn in [[stage_in]]) {
    return {in.p.interpolate_at_centroid(), in.n.interpolate_at_centroid()};
}

fragment PmPair pm_sample(PmIn in [[stage_in]], constant uint &sample [[buffer(0)]]) {
    return {in.p.interpolate_at_sample(sample), in.n.interpolate_at_sample(sample)};
}

fragment PmPair pm_sample_id(PmIn in [[stage_in]], uint sample [[sample_id]]) {
    return {in.p.interpolate_at_sample(sample), in.n.interpolate_at_sample(sample)};
}

fragment PmPair pm_offset(PmIn in [[stage_in]], constant float2 &offset [[buffer(0)]]) {
    return {in.p.interpolate_at_offset(offset), in.n.interpolate_at_offset(offset)};
}

// Each pixel's own offset, `sweep.xy + floor(position.xy) * sweep.zw`: a sweep of offsets in one
// draw.
fragment PmPair pm_offset_sweep(PmIn in [[stage_in]], float4 position [[position]],
                                constant float4 &sweep [[buffer(0)]]) {
    float2 offset = sweep.xy + floor(position.xy) * sweep.zw;
    return {in.p.interpolate_at_offset(offset), in.n.interpolate_at_offset(offset)};
}

// `pm_offset`'s reads after a discard of the fragments whose offset's x is past 1.
fragment PmPair pm_offset_after_discard(PmIn in [[stage_in]], constant float2 &offset [[buffer(0)]]) {
    if (offset.x > 1.0f) {
        discard_fragment();
    }
    return {in.p.interpolate_at_offset(offset), in.n.interpolate_at_offset(offset)};
}

struct PmNarrow {
    interpolant<float2, interpolation::perspective> p2;
    interpolant<float, interpolation::no_perspective> s;
    interpolant<half4, interpolation::perspective> hp;
    interpolant<half, interpolation::no_perspective> hn;
};

struct PmFour {
    float4 center [[color(0)]];
    float4 centroid [[color(1)]];
    float4 sample [[color(2)]];
    float4 offset [[color(3)]];
};

// Every form of the `float2`, `float` and `half` inputs, each in one attachment.
fragment PmFour pm_narrow(PmNarrow in [[stage_in]], constant uint &sample [[buffer(0)]],
                          constant float2 &offset [[buffer(1)]]) {
    PmFour out;
    out.center = float4(in.p2.interpolate_at_center(), in.s.interpolate_at_center(),
                        in.hn.interpolate_at_center());
    out.centroid = float4(in.p2.interpolate_at_centroid(), in.s.interpolate_at_centroid(),
                          in.hn.interpolate_at_centroid());
    out.sample = float4(in.p2.interpolate_at_sample(sample), in.s.interpolate_at_sample(sample),
                        in.hn.interpolate_at_sample(sample));
    out.offset = float4(in.p2.interpolate_at_offset(offset), in.s.interpolate_at_offset(offset),
                        in.hn.interpolate_at_offset(offset));
    return out;
}

struct PmHalfFour {
    half4 center [[color(0)]];
    half4 centroid [[color(1)]];
    half4 sample [[color(2)]];
    half4 offset [[color(3)]];
};

// Every form of the `half4` input.
fragment PmHalfFour pm_half(PmNarrow in [[stage_in]], constant uint &sample [[buffer(0)]],
                            constant float2 &offset [[buffer(1)]]) {
    return {in.hp.interpolate_at_center(), in.hp.interpolate_at_centroid(),
            in.hp.interpolate_at_sample(sample), in.hp.interpolate_at_offset(offset)};
}

struct PmSame {
    interpolant<float4, interpolation::perspective> p;
    interpolant<float4, interpolation::perspective> n;
};

// An interpolant chosen at run time: a `select` of two interpolant handles.
fragment float4 pm_selected(PmSame in [[stage_in]], constant uint &which [[buffer(0)]]) {
    return (which != 0 ? in.p : in.n).interpolate_at_center();
}

// Copies sample `at.z` of each texel of a multisampled color attachment out, the samples of a
// texel consecutive, so a draw's every sample can be read back.
kernel void pm_read_samples(texture2d_ms<float, access::read> target [[texture(0)]],
                            device float4 *out [[buffer(0)]],
                            uint3 at [[thread_position_in_grid]]) {
    uint samples = target.get_num_samples();
    uint index = (at.y * target.get_width() + at.x) * samples + at.z;
    out[index] = target.read(at.xy, at.z);
}
