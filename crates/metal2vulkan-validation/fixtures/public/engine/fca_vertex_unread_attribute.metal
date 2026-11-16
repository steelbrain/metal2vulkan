// Owned fixture: an ungated vertex attribute whose value reaches only a varying a function
// constant gates. With `has_color` false (or undefined) the `color` varying is not declared
// (MSL 5.8.1.6), so nothing the specialized function outputs reads attribute 1: Metal reports the
// attribute inactive and makes the pipeline from a vertex descriptor without it. With `has_color`
// true the varying is declared, attribute 1 is read, and a descriptor without it is refused.
// - `fca_vertex_unread_attribute`: the position from attribute 0, the `shade` varying a constant,
//   and `color` attribute 1 scaled and biased;
// - `fca_fragment_unread_attribute`: `color` where `has_color`, else `shade`. Loop-free.
#include <metal_stdlib>
using namespace metal;

constant bool has_color [[function_constant(0)]];

struct FcuAttributes {
    float4 position [[attribute(0)]];
    float4 color [[attribute(1)]];
};

struct FcuVaryings {
    float4 position [[position]];
    float4 shade;
    float4 color [[function_constant(has_color)]];
};

vertex FcuVaryings fca_vertex_unread_attribute(FcuAttributes in [[stage_in]]) {
    FcuVaryings out;
    out.position = in.position;
    out.shade = float4(0.25, 0.5, 0.75, 1.0);
    out.color = in.color * 0.5 + 0.25;
    return out;
}

fragment float4 fca_fragment_unread_attribute(FcuVaryings in [[stage_in]]) {
    return has_color ? in.color : in.shade;
}
