// Owned MSL: sampling, gathers, and LOD queries
// through runtime and constexpr samplers. Every function is loop-free.
#include <metal_stdlib>
using namespace metal;

// A runtime sampler at an explicit level: pixel or normalized per the bound sampler.
kernel void sample_level(texture2d<float> t [[texture(0)]],
                         sampler s [[sampler(0)]],
                         device float4 *out [[buffer(0)]],
                         uint gid [[thread_position_in_grid]]) {
    out[gid] = t.sample(s, float2(gid) * 0.25, level(1.0));
}

// Metal's implicit-LOD sample outside a fragment function reads the base level.
kernel void sample_base(texture2d<float> t [[texture(0)]],
                        sampler s [[sampler(0)]],
                        device float4 *out [[buffer(0)]],
                        uint gid [[thread_position_in_grid]]) {
    out[gid] = t.sample(s, float2(gid) * 0.25);
}

constexpr sampler repeat_linear(filter::linear, address::repeat, mip_filter::nearest);

// A constexpr sampler on an array texture with a constant offset.
kernel void sample_constexpr_array(texture2d_array<float> t [[texture(0)]],
                                   device float4 *out [[buffer(0)]],
                                   uint gid [[thread_position_in_grid]]) {
    out[gid] = t.sample(repeat_linear, float2(gid) * 0.125, gid & 3, level(0.0), int2(1, -1));
}

// Depth comparison at an explicit level; one runtime sampler with and without comparison
// needs two combined bindings.
kernel void sample_compare_depth(depth2d<float> d [[texture(0)]],
                                 texture2d<float> c [[texture(1)]],
                                 sampler s [[sampler(0)]],
                                 device float *out [[buffer(0)]],
                                 uint gid [[thread_position_in_grid]]) {
    float2 uv = float2(gid) * 0.5;
    out[gid] = d.sample_compare(s, uv, 0.5, level(0)) + c.sample(s, uv, level(0)).x;
}

constexpr sampler nearest_edge(filter::nearest, address::clamp_to_edge);

// Integer gathers of a component.
kernel void gather_int(texture2d<int> t [[texture(0)]],
                       device int4 *out [[buffer(0)]],
                       uint gid [[thread_position_in_grid]]) {
    out[gid] = t.gather(nearest_edge, float2(gid) * 0.25, int2(0), component::y);
}

// Half texels sampled with explicit gradients.
kernel void sample_grad(texture2d<half> t [[texture(0)]],
                        sampler s [[sampler(0)]],
                        device half4 *out [[buffer(0)]],
                        uint gid [[thread_position_in_grid]]) {
    float2 uv = float2(gid) * 0.25;
    out[gid] = t.sample(s, uv, gradient2d(float2(0.5, 0.0), float2(0.0, 0.5)));
}

constexpr sampler pixel_linear(coord::pixel, filter::linear, address::clamp_to_edge);

// A constexpr pixel-coordinate sampler: the device's unnormalized sampler reads the coordinate.
kernel void sample_pixel(texture2d<float> t [[texture(0)]],
                         device float4 *out [[buffer(0)]],
                         uint gid [[thread_position_in_grid]]) {
    out[gid] = t.sample(pixel_linear, float2(gid) + 0.5);
}

// Fragment sampling: implicit LOD with a bias, a cube, and the LOD queries (tangled).
fragment float4 sample_fragment(float4 pos [[position]],
                                texture2d<float> t [[texture(0)]],
                                texturecube<float> cube [[texture(1)]],
                                sampler s [[sampler(0)]]) {
    float2 uv = pos.xy * 0.0625;
    float4 a = t.sample(s, uv, bias(0.5));
    float4 b = cube.sample(repeat_linear, float3(uv, 1.0));
    float l = t.calculate_clamped_lod(repeat_linear, uv) + t.calculate_unclamped_lod(repeat_linear, uv);
    return a + b + l;
}

// A minimum-LOD clamp needs `shaderResourceMinLod`.
fragment float4 sample_min_lod(float4 pos [[position]],
                               texture2d<float> t [[texture(0)]],
                               sampler s [[sampler(0)]]) {
    return t.sample(s, pos.xy * 0.0625, min_lod_clamp(1.5));
}
