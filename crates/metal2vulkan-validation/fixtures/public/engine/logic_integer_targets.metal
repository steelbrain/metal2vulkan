// The integer functions of the logic-operation lane (validation/src/runtime_lane/logic_ops.rs),
// our own MSL: a fullscreen triangle and fragments writing the unsigned or signed integer color
// in buffer 0, which may lie past the attachment's channels' range.
#include <metal_stdlib>
using namespace metal;

struct Position {
    float4 position [[position]];
};

vertex Position lit_vertex(uint id [[vertex_id]]) {
    float2 p = float2((id << 1) & 2, id & 2);
    Position out;
    out.position = float4(p * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

fragment uint4 lit_fill_u(Position in [[stage_in]], constant uint4 *color [[buffer(0)]]) {
    return color[0];
}

fragment int4 lit_fill_i(Position in [[stage_in]], constant int4 *color [[buffer(0)]]) {
    return color[0];
}
