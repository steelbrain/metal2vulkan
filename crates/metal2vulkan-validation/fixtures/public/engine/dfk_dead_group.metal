// Owned fixture: a fragment function of five colors that reads color 4 (`[[color(4)]]`) as its
// function constant 0 decides, which a device of four color attachments draws in two groups of
// colors (`pipeline::compile::color_group`): colors 0 to 3, then color 4.
//
// - dfg_branch: the read sits in the arm of a branch on `dfg_reads`; with it false the arm is dead
//   and nothing reads color 4, which the function still declares.
// - dfg_gated: the input itself is gated by `dfg_reads` and read in the arm of a branch on it; with
//   it false the function declares no color input.
//
// Each writes the varying `tint` swizzled to colors 0 to 3, and to color 4 `tint`, or with the read
// live half the read plus `tint`. The vertex function is `color_groups`' `cg_cover`, whose
// interface this one reads. Loop-free.
#include <metal_stdlib>
using namespace metal;

constant bool dfg_reads [[function_constant(0)]];

struct CgPoint {
    float4 position [[position]];
    float4 tint;
};

struct DfgFive {
    float4 c0 [[color(0)]];
    float4 c1 [[color(1)]];
    float4 c2 [[color(2)]];
    float4 c3 [[color(3)]];
    float4 c4 [[color(4)]];
};

fragment DfgFive dfg_branch(CgPoint in [[stage_in]], float4 last [[color(4)]]) {
    float4 t = in.tint;
    float4 c4 = t;
    if (dfg_reads) {
        c4 = last * 0.5 + t;
    }
    return DfgFive { t, t.yzwx, t.zwxy, t.wxyz, c4 };
}

fragment DfgFive dfg_gated(CgPoint in [[stage_in]],
                           float4 last [[color(4), function_constant(dfg_reads)]]) {
    float4 t = in.tint;
    float4 c4 = t;
    if (dfg_reads) {
        c4 = last * 0.5 + t;
    }
    return DfgFive { t, t.yzwx, t.zwxy, t.wxyz, c4 };
}
