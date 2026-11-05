// The draw_costs example's functions (examples/draw_costs.rs), shaped as a guest's draws: a
// vertex function whose positions come from vertex buffer 0 by vertex and whose colour comes from
// buffer 1; a fragment function that returns fragment buffer 0's first value, and one that samples
// texture 0 with sampler 0.
#include <metal_stdlib>
using namespace metal;

struct DrawCostsOut {
    float4 position [[position]];
    float4 colour;
};

vertex DrawCostsOut draw_costs_vertex(uint vid [[vertex_id]],
                                      const device float4 *positions [[buffer(0)]],
                                      const device float4 *colours [[buffer(1)]]) {
    DrawCostsOut out;
    out.position = positions[vid];
    out.colour = colours[0];
    return out;
}

fragment float4 draw_costs_buffer(DrawCostsOut in [[stage_in]],
                                  const device float4 *colour [[buffer(0)]]) {
    return colour[0] * in.colour;
}

fragment float4 draw_costs_texture(DrawCostsOut in [[stage_in]],
                                   texture2d<float> source [[texture(0)]],
                                   sampler linear [[sampler(0)]]) {
    return source.sample(linear, in.position.xy / 64.0) * in.colour;
}
