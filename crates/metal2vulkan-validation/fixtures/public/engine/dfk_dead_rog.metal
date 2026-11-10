// Owned fixture: inputs a function declares whose reads its function constant 0 (`dfr_reads`)
// decides, each read in the arm of a branch on it, so that with it false the function declares
// the input and reads nothing there.
//
// - dfr_color: color 0 as a `[[color(0)]]` input in raster order group 0; five colors.
// - dfr_buffer: a `const device` buffer in raster order group 0; five colors.
// - dfr_output: color 0 as a `[[color(0)]]` input, written to a color 0 output in raster order
//   group 0; one color.
// - dfr_lacks: color 1 as a `[[color(1)]]` input, beside a color 0 output alone; one color.
//
// Each writes the varying `tint` to color 0, or with the read live the read (halved for a color)
// plus `tint`; the five-color ones write `tint` swizzled to colors 1 to 3 and `tint` to color 4.
// The vertex function is `color_groups`' `cg_cover`, whose interface this one reads. Loop-free.
#include <metal_stdlib>
using namespace metal;

constant bool dfr_reads [[function_constant(0)]];

struct CgPoint {
    float4 position [[position]];
    float4 tint;
};

struct DfrFive {
    float4 c0 [[color(0)]];
    float4 c1 [[color(1)]];
    float4 c2 [[color(2)]];
    float4 c3 [[color(3)]];
    float4 c4 [[color(4)]];
};

struct DfrOrdered {
    float4 c0 [[color(0), raster_order_group(0)]];
};

fragment DfrFive dfr_color(CgPoint in [[stage_in]],
                           float4 first [[color(0), raster_order_group(0)]]) {
    float4 t = in.tint;
    float4 c0 = t;
    if (dfr_reads) {
        c0 = first * 0.5 + t;
    }
    return DfrFive { c0, t.yzwx, t.zwxy, t.wxyz, t };
}

fragment DfrFive dfr_buffer(CgPoint in [[stage_in]],
                            const device float4 *b [[buffer(0), raster_order_group(0)]]) {
    float4 t = in.tint;
    float4 c0 = t;
    if (dfr_reads) {
        c0 = b[0] + t;
    }
    return DfrFive { c0, t.yzwx, t.zwxy, t.wxyz, t };
}

fragment DfrOrdered dfr_output(CgPoint in [[stage_in]], float4 first [[color(0)]]) {
    float4 t = in.tint;
    float4 c0 = t;
    if (dfr_reads) {
        c0 = first * 0.5 + t;
    }
    return DfrOrdered { c0 };
}

fragment float4 dfr_lacks(CgPoint in [[stage_in]], float4 extra [[color(1)]]) {
    float4 t = in.tint;
    float4 c0 = t;
    if (dfr_reads) {
        c0 = extra * 0.5 + t;
    }
    return c0;
}
