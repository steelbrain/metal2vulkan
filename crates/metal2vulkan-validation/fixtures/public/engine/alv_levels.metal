// Owned fixture: reads of a mipmapped texture at known levels, for draws that sample a texture
// their own render pass renders at another level (the runtime's `render/attached_levels.rs`).
// `alv_rect` covers the target with two triangles from its vertex id, passing the corner as the
// coordinate. `alv_level1` samples texture 0 through sampler 0 at the explicit level 1,
// `alv_level2` at level 2, and `alv_implicit` at the level of detail its coordinate's derivatives
// give. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct AlvOut {
    float4 position [[position]];
    float2 uv;
};

vertex AlvOut alv_rect(uint vid [[vertex_id]])
{
    const float2 corners[6] = {
        float2(0.0f, 0.0f), float2(1.0f, 0.0f), float2(0.0f, 1.0f),
        float2(1.0f, 0.0f), float2(1.0f, 1.0f), float2(0.0f, 1.0f),
    };
    float2 c = corners[vid];
    AlvOut out;
    out.position = float4(c.x * 2.0f - 1.0f, 1.0f - c.y * 2.0f, 0.0f, 1.0f);
    out.uv = c;
    return out;
}

fragment float4 alv_level1(AlvOut in [[stage_in]], texture2d<float> t [[texture(0)]],
                           sampler s [[sampler(0)]])
{
    return t.sample(s, in.uv, level(1.0f));
}

fragment float4 alv_level2(AlvOut in [[stage_in]], texture2d<float> t [[texture(0)]],
                           sampler s [[sampler(0)]])
{
    return t.sample(s, in.uv, level(2.0f));
}

fragment float4 alv_implicit(AlvOut in [[stage_in]], texture2d<float> t [[texture(0)]],
                             sampler s [[sampler(0)]])
{
    return t.sample(s, in.uv);
}
