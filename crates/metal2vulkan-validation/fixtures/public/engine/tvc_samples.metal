// Owned MSL for the texture view casts: each lane `i` samples the bound texture at one texel
// centre of one level and slice, `p[i]` = (level, slice, x, y) in the bound texture's own levels
// and slices, through a nearest sampler with a nearest mip filter at that explicit level, and
// writes the sample to `out[i]`. A slice of a cube is its face, and of a cube array its cube times
// six plus its face; a cube's texel centre is sampled along the direction through it, by the face
// table both Metal and Vulkan take (face 0 +X, 1 -X, 2 +Y, 3 -Y, 4 +Z, 5 -Z, with s and t in
// [-1, 1] across the face).
//   tvc_2d          a `texture2d<float>`;
//   tvc_2d_array    a `texture2d_array<float>`;
//   tvc_cube        a `texturecube<float>`;
//   tvc_cube_array  a `texturecube_array<float>`.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

constant sampler nn(filter::nearest, mip_filter::nearest, address::clamp_to_edge);

static float2 tvc_centre(uint2 texel, uint width, uint height)
{
    return (float2(texel) + 0.5f) / float2(float(width), float(height));
}

static float3 tvc_direction(uint face, float2 c)
{
    float s = 2.0f * c.x - 1.0f;
    float t = 2.0f * c.y - 1.0f;
    switch (face) {
    case 0:
        return float3(1.0f, -t, -s);
    case 1:
        return float3(-1.0f, -t, s);
    case 2:
        return float3(s, 1.0f, t);
    case 3:
        return float3(s, -1.0f, -t);
    case 4:
        return float3(s, -t, 1.0f);
    default:
        return float3(-s, -t, -1.0f);
    }
}

kernel void tvc_2d(texture2d<float> t [[texture(0)]], const device uint4 *p [[buffer(0)]],
                   device float4 *out [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    uint4 q = p[i];
    float2 c = tvc_centre(q.zw, t.get_width(q.x), t.get_height(q.x));
    out[i] = t.sample(nn, c, level(float(q.x)));
}

kernel void tvc_2d_array(texture2d_array<float> t [[texture(0)]], const device uint4 *p [[buffer(0)]],
                         device float4 *out [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    uint4 q = p[i];
    float2 c = tvc_centre(q.zw, t.get_width(q.x), t.get_height(q.x));
    out[i] = t.sample(nn, c, q.y, level(float(q.x)));
}

kernel void tvc_cube(texturecube<float> t [[texture(0)]], const device uint4 *p [[buffer(0)]],
                     device float4 *out [[buffer(1)]], uint i [[thread_position_in_grid]])
{
    uint4 q = p[i];
    float2 c = tvc_centre(q.zw, t.get_width(q.x), t.get_height(q.x));
    out[i] = t.sample(nn, tvc_direction(q.y, c), level(float(q.x)));
}

kernel void tvc_cube_array(texturecube_array<float> t [[texture(0)]],
                           const device uint4 *p [[buffer(0)]], device float4 *out [[buffer(1)]],
                           uint i [[thread_position_in_grid]])
{
    uint4 q = p[i];
    float2 c = tvc_centre(q.zw, t.get_width(q.x), t.get_height(q.x));
    out[i] = t.sample(nn, tvc_direction(q.y % 6u, c), q.y / 6u, level(float(q.x)));
}
