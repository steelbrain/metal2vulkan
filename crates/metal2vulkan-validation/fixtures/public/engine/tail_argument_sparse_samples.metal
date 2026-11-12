// Owned MSL for the residency of a texture held in an argument buffer: a fragment's
// `sparse_sample` of the material's texture through a constexpr sampler that filters linearly
// between levels with anisotropy and repeats, its residency observed, and the fallback sample
// clamped to the least level the material names (`min_lod_clamp`), the shape a forward renderer
// that streams its material textures compiles. Metal reports a texture that is not sparse
// resident everywhere, so on such a texture the first function returns the sparse sample's value
// and the second states the residency it read beside the value's first two channels.
#include <metal_stdlib>
using namespace metal;

struct SparseMaterial {
    texture2d<half> t;
    uint min_mip;
};

struct VaryingIn {
    float4 position [[position]];
    float2 uv;
};

// A triangle covering the target, with `uv` its normalized position: (0, 0) at the bottom left
// and (1, 1) at the top right.
vertex VaryingIn argument_sparse_vertex(uint vid [[vertex_id]])
{
    float2 p = float2(float((vid << 1) & 2), float(vid & 2));
    VaryingIn out;
    out.position = float4(p * 2.0 - 1.0, 0.0, 1.0);
    out.uv = p;
    return out;
}

// The sample of `t` at `uv` where its texels are resident, else the sample from level `min_mip`
// on.
static half4 sample_material(texture2d<half> t, float2 uv, uint min_mip)
{
    constexpr sampler s(mip_filter::linear, mag_filter::linear, min_filter::linear,
                        address::repeat, max_anisotropy(10));
    sparse_color<half4> v = t.sparse_sample(s, uv);
    if (v.resident()) {
        return v.value();
    }
    return t.sample(s, uv, min_lod_clamp(min_mip));
}

fragment half4 argument_sparse_fragment(VaryingIn in [[stage_in]],
                                        constant SparseMaterial &m [[buffer(0)]])
{
    return sample_material(m.t, in.uv, m.min_mip);
}

// The sparse sample's first two channels, then 0.25 where its texels are resident and 0.75
// where they are not, then 1.
fragment half4 argument_sparse_residency_fragment(VaryingIn in [[stage_in]],
                                                  constant SparseMaterial &m [[buffer(0)]])
{
    constexpr sampler s(mip_filter::linear, mag_filter::linear, min_filter::linear,
                        address::repeat, max_anisotropy(10));
    sparse_color<half4> v = m.t.sparse_sample(s, in.uv);
    return half4(v.value().xy, v.resident() ? 0.25h : 0.75h, 1.0h);
}
