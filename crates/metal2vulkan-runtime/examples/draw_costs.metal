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

// The compositor shape's functions: a quad's corner (a position in [0, 1] and its texture
// coordinate) from vertex buffer 0 by vertex, placed by buffer 1's scale and offset; fragments
// that sample texture 0 tinted by fragment buffer 0, masked by texture 1's alpha, or blended over
// texture 1 read at the fragment's own pixel (the pass's own color attachment, a backdrop).
struct DrawCostsCorner {
    float2 position;
    float2 uv;
};

struct DrawCostsPlace {
    float2 scale;
    float2 offset;
};

struct DrawCostsQuad {
    float4 position [[position]];
    float2 uv;
};

vertex DrawCostsQuad draw_costs_quad_vertex(uint vid [[vertex_id]],
                                            const device DrawCostsCorner *corners [[buffer(0)]],
                                            constant DrawCostsPlace &place [[buffer(1)]]) {
    DrawCostsCorner corner = corners[vid];
    DrawCostsQuad out;
    out.position = float4(corner.position * place.scale + place.offset, 0.0, 1.0);
    out.uv = corner.uv;
    return out;
}

fragment float4 draw_costs_quad(DrawCostsQuad in [[stage_in]],
                                texture2d<float> source [[texture(0)]],
                                sampler linear [[sampler(0)]],
                                constant float4 &tint [[buffer(0)]]) {
    return source.sample(linear, in.uv) * tint;
}

fragment float4 draw_costs_quad_masked(DrawCostsQuad in [[stage_in]],
                                       texture2d<float> source [[texture(0)]],
                                       texture2d<float> mask [[texture(1)]],
                                       sampler linear [[sampler(0)]],
                                       constant float4 &tint [[buffer(0)]]) {
    return source.sample(linear, in.uv) * tint * mask.sample(linear, in.uv).a;
}

fragment float4 draw_costs_quad_backdrop(DrawCostsQuad in [[stage_in]],
                                         texture2d<float> source [[texture(0)]],
                                         texture2d<float, access::read> backdrop [[texture(1)]],
                                         sampler linear [[sampler(0)]],
                                         constant float4 &tint [[buffer(0)]]) {
    float4 colour = source.sample(linear, in.uv) * tint;
    float4 under = backdrop.read(uint2(in.position.xy));
    return float4(mix(under.rgb, colour.rgb, colour.a), 1.0);
}
