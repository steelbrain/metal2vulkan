// Owned fixture: fragment functions that read or write a device buffer, for draws whose buffer
// may lie in the heap of their pass's attachments. `hpr_one` writes `b[0]` plus the varying `tint`
// to color 0; `hpr_five` writes it to color 0 and `tint` to colors 1 to 4, a pass of more colors
// than a device at Vulkan's floor draws at once (four), which the translation draws in groups;
// their vertex function is `color_groups`' `cg_cover`, whose interface they read.
// `hpr_level1_write` samples texture 0 through sampler 0 at the explicit level 1, as
// `alv_levels`' `alv_level1` does, returns it and writes it to `w[0]`; its vertex function is
// `alv_rect`. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct CgPoint {
    float4 position [[position]];
    float4 tint;
};

fragment float4 hpr_one(CgPoint in [[stage_in]], const device float4 *b [[buffer(0)]]) {
    return b[0] + in.tint;
}

struct HprFive {
    float4 c0 [[color(0)]];
    float4 c1 [[color(1)]];
    float4 c2 [[color(2)]];
    float4 c3 [[color(3)]];
    float4 c4 [[color(4)]];
};

fragment HprFive hpr_five(CgPoint in [[stage_in]], const device float4 *b [[buffer(0)]]) {
    float4 t = in.tint;
    return HprFive { b[0] + t, t, t, t, t };
}

struct AlvOut {
    float4 position [[position]];
    float2 uv;
};

fragment float4 hpr_level1_write(AlvOut in [[stage_in]], texture2d<float> t [[texture(0)]],
                                 sampler s [[sampler(0)]], device float4 *w [[buffer(0)]]) {
    float4 c = t.sample(s, in.uv, level(1.0f));
    w[0] = c;
    return c;
}
