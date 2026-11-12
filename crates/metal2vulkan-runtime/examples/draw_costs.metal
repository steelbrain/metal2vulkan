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

// The scene shape's functions, a 3D engine's draws into four colors: a vertex (a position and a
// texture coordinate, from vertex buffer 0 through the pipeline's vertex descriptor) placed by
// buffer 1's transform; fragments that sample eight textures, each with a sampler of its own,
// scaled by fragment buffer 0's material.
struct DrawCostsSceneVertex {
    float3 position [[attribute(0)]];
    float2 uv [[attribute(1)]];
};

struct DrawCostsSceneOut {
    float4 position [[position]];
    float2 uv;
};

vertex DrawCostsSceneOut draw_costs_scene_vertex(DrawCostsSceneVertex in [[stage_in]],
                                                 constant float4x4 &transform [[buffer(1)]]) {
    DrawCostsSceneOut out;
    out.position = transform * float4(in.position, 1.0);
    out.uv = in.uv;
    return out;
}

struct DrawCostsSceneColors {
    float4 albedo [[color(0)]];
    float4 normal [[color(1)]];
    float4 material [[color(2)]];
    float4 lit [[color(3)]];
};

fragment DrawCostsSceneColors draw_costs_scene(DrawCostsSceneOut in [[stage_in]],
                                               constant float4 &scale [[buffer(0)]],
                                               texture2d<float> t0 [[texture(0)]],
                                               texture2d<float> t1 [[texture(1)]],
                                               texture2d<float> t2 [[texture(2)]],
                                               texture2d<float> t3 [[texture(3)]],
                                               texture2d<float> t4 [[texture(4)]],
                                               texture2d<float> t5 [[texture(5)]],
                                               texture2d<float> t6 [[texture(6)]],
                                               texture2d<float> t7 [[texture(7)]],
                                               sampler s0 [[sampler(0)]],
                                               sampler s1 [[sampler(1)]],
                                               sampler s2 [[sampler(2)]],
                                               sampler s3 [[sampler(3)]],
                                               sampler s4 [[sampler(4)]],
                                               sampler s5 [[sampler(5)]],
                                               sampler s6 [[sampler(6)]],
                                               sampler s7 [[sampler(7)]]) {
    DrawCostsSceneColors out;
    out.albedo = t0.sample(s0, in.uv) * t1.sample(s1, in.uv) * scale;
    out.normal = t2.sample(s2, in.uv) + t3.sample(s3, in.uv);
    out.material = t4.sample(s4, in.uv) * t5.sample(s5, in.uv);
    out.lit = t6.sample(s6, in.uv) + t7.sample(s7, in.uv) * scale.w;
    return out;
}
