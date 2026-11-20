// The functions of the private depth compare clamp lane, our own MSL: a fullscreen triangle, and
// fragments that compare a `depth2d<float>` at the target's centre against buffer 0's references,
// `sample_compare` in red and `gather_compare`'s first texel in green: through `[[sampler(2)]]`
// with the texture at `[[texture(1)]]`, through a constexpr `less` sampler, and through an argument
// buffer's sampler.
#include <metal_stdlib>
using namespace metal;

struct Position {
    float4 position [[position]];
};

vertex Position pdc_vertex(uint id [[vertex_id]]) {
    float2 p = float2((id << 1) & 2, id & 2);
    Position out;
    out.position = float4(p * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

fragment float4 pdc_slot(Position in [[stage_in]], depth2d<float> d [[texture(1)]],
                         sampler s [[sampler(2)]], constant float4 *refs [[buffer(0)]]) {
    float r = d.sample_compare(s, float2(0.5), refs[0].x);
    float g = d.gather_compare(s, float2(0.5), refs[0].y).x;
    return float4(r, g, 0.0, 1.0);
}

constexpr sampler less_sampler(filter::nearest, address::clamp_to_edge, compare_func::less);

fragment float4 pdc_constexpr(Position in [[stage_in]], depth2d<float> d [[texture(0)]],
                              constant float4 *refs [[buffer(0)]]) {
    float r = d.sample_compare(less_sampler, float2(0.5), refs[0].x);
    float g = d.gather_compare(less_sampler, float2(0.5), refs[0].y).x;
    return float4(r, g, 0.0, 1.0);
}

struct Arguments {
    depth2d<float> d [[id(0)]];
    sampler s [[id(1)]];
};

fragment float4 pdc_argument(Position in [[stage_in]], constant Arguments &a [[buffer(1)]],
                             constant float4 *refs [[buffer(0)]]) {
    float r = a.d.sample_compare(a.s, float2(0.5), refs[0].x);
    return float4(r, 0.0, 0.0, 1.0);
}
