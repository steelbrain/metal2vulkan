// Owned fixture, without fast math (`-fno-fast-math`, so every float operation is
// `NoContraction`, which SPIRV-Cross writes through its `optnone` `spvFAdd` and kin): fragment
// functions that read a coverage texture, sample a texture at LOD 0, and pick, where the coverage
// is not 1 (where every covered pixel's coverage is 1, only the quad helpers past a primitive's
// edges do), a value an addition makes, with no branch: the compiler writes each pick as a select,
// which SPIRV-Cross writes as a `?:` whose arm holds the addition. Each writes a texture after
// it. `rch_select_arm` picks the sum as a component; `rch_select_and` tests the sum on the right
// of a `&&` whose left is the coverage test; `rch_select_sampled` picks a coordinate the sum makes
// and samples there at an implicit LOD, a sample that reads the sum after it; `rch_select_uniform`
// picks the sum where a constant block's flag says (a pick every invocation of a draw makes
// alike). `rch_select_vertex` draws the triangle whose three corners `buffer(0)` holds, in clip
// space.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

constexpr sampler select_sampler(filter::nearest, address::repeat);

vertex float4 rch_select_vertex(uint vid [[vertex_id]], constant float2 *corners [[buffer(0)]])
{
    return float4(corners[vid], 0.0f, 1.0f);
}

static float4 base(float4 position, texture2d<float> noise)
{
    float4 n = noise.sample(select_sampler, position.xy * 0.03125f, level(0.0f));
    return float4(1.0f, 0.5f, 0.25f, 1.0f) + n * 0.0078125f;
}

fragment void rch_select_arm(float4 position [[position]],
                             texture2d<float> noise [[texture(0)]],
                             texture2d<float> coverage [[texture(1)]],
                             texture2d<float, access::write> dst [[texture(2)]],
                             device const float4 *src [[buffer(0)]])
{
    uint2 p = uint2(position.xy);
    float c = coverage.read(p).x;
    float4 out = base(position, noise);
    float4 s = src[p.y * 64u + p.x];
    out.x = c != 1.0f ? s.x + out.x : out.x;
    dst.write(out, p);
}

fragment void rch_select_and(float4 position [[position]],
                             texture2d<float> noise [[texture(0)]],
                             texture2d<float> coverage [[texture(1)]],
                             texture2d<float, access::write> dst [[texture(2)]],
                             device const float4 *src [[buffer(0)]])
{
    uint2 p = uint2(position.xy);
    float c = coverage.read(p).x;
    float4 out = base(position, noise);
    float4 s = src[p.y * 64u + p.x];
    bool k = c != 1.0f && s.y + out.y > 0.75f;
    out.y = k ? 0.0f : out.y;
    dst.write(out, p);
}

fragment void rch_select_sampled(float4 position [[position]],
                                 texture2d<float> noise [[texture(0)]],
                                 texture2d<float> coverage [[texture(1)]],
                                 texture2d<float, access::write> dst [[texture(2)]],
                                 device const float4 *src [[buffer(0)]])
{
    uint2 p = uint2(position.xy);
    float c = coverage.read(p).x;
    float4 out = base(position, noise);
    float4 s = src[p.y * 64u + p.x];
    float2 uv = c != 1.0f ? float2(s.x + out.x, out.y) : position.xy * 0.03125f;
    out = noise.sample(select_sampler, uv);
    dst.write(out, p);
}

fragment void rch_select_uniform(float4 position [[position]],
                                 texture2d<float> noise [[texture(0)]],
                                 texture2d<float> coverage [[texture(1)]],
                                 texture2d<float, access::write> dst [[texture(2)]],
                                 device const float4 *src [[buffer(0)]],
                                 constant uint &flag [[buffer(1)]])
{
    uint2 p = uint2(position.xy);
    float c = coverage.read(p).x;
    float4 out = base(position, noise);
    float4 s = src[p.y * 64u + p.x];
    out.x = flag != 0u ? s.x + out.x : out.x;
    out.z = c * out.z;
    dst.write(out, p);
}
