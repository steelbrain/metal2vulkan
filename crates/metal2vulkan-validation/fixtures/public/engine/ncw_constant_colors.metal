// ncw_constant_colors.metal (metal2vulkan): fragment functions whose color outputs are constants,
// written from scratch for the color write's NaN note (`lower::stage_io::returns_no_nan`): a
// constant `half4` and `float4` none of whose lanes is a NaN, a constant `half4` whose red is one
// (bits 0x7e01), a struct of a constant color 0 and a computed color 1, and a computed `half4`.
#include <metal_stdlib>
using namespace metal;

struct NcwVertex {
    float4 position [[position]];
};

vertex NcwVertex ncw_vertex(uint vid [[vertex_id]]) {
    const float2 p[3] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
    return NcwVertex{float4(p[vid], 0, 1)};
}

fragment half4 ncw_constant_half() {
    return half4(0.5h, 0.25h, 0.75h, 1.0h);
}

fragment float4 ncw_constant_float() {
    return float4(0.5f, 0.25f, 0.75f, 1.0f);
}

fragment half4 ncw_constant_nan() {
    return half4(as_type<half>(ushort(0x7e01)), 0.25h, 0.75h, 1.0h);
}

struct NcwPair {
    half4 constant_color [[color(0)]];
    half4 computed_color [[color(1)]];
};

fragment NcwPair ncw_constant_pair(NcwVertex in [[stage_in]]) {
    return NcwPair{half4(0.5h, 0.25h, 0.75h, 1.0h), half4(half(in.position.x) * 0.001h)};
}

fragment half4 ncw_computed_half(NcwVertex in [[stage_in]]) {
    return half4(half(in.position.x) * 0.001h);
}
