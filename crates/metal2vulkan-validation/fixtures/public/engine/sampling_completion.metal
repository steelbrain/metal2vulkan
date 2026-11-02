// Owned MSL: samplers selected among constexpr and
// runtime candidates, the exact nearest pixel fetch, sample offsets, and level of detail outside
// fragment functions. Every function is loop-free.
#include <metal_stdlib>
using namespace metal;

constexpr sampler repeat_linear(filter::linear, address::repeat, mip_filter::linear);
constexpr sampler pixel_nearest(coord::pixel, filter::nearest, address::clamp_to_edge);
constexpr sampler pixel_zero(coord::pixel, filter::nearest, address::clamp_to_zero);

// A sampler selected between a runtime and a constexpr sampler, at an explicit level (not
// tangled: a switch at the use, one arm per sampler).
kernel void sample_mixed_samplers(texture2d<float> t [[texture(0)]],
                                  sampler s [[sampler(0)]],
                                  device float4 *out [[buffer(0)]],
                                  uint gid [[thread_position_in_grid]]) {
    sampler u = (gid & 1) ? s : repeat_linear;
    out[gid] = t.sample(u, float2(gid) * 0.25, level(1.0));
}

// The same selection with an implicit-LOD, biased sample in a fragment function (tangled: the
// gradients are hoisted before the switch), and a gather through it.
fragment float4 fragment_mixed_samplers(float4 pos [[position]],
                                        texture2d<float> t [[texture(0)]],
                                        sampler s [[sampler(0)]]) {
    sampler u = pos.x > 8.0 ? s : repeat_linear;
    float2 uv = pos.xy * 0.0625;
    return t.sample(u, uv, bias(0.5)) + t.gather(u, uv);
}

// A nearest, clamp-to-edge pixel sampler: an exact texel fetch, on a 2D texture with a dynamic
// offset and on an array texture at a layer.
kernel void sample_pixel_nearest(texture2d<float> t [[texture(0)]],
                                 texture2d_array<float> a [[texture(1)]],
                                 device float4 *out [[buffer(0)]],
                                 uint gid [[thread_position_in_grid]]) {
    int2 o = int2(int(gid & 7) - 4, 1);
    out[gid] = t.sample(pixel_nearest, float2(gid) + 0.5, o)
        + a.sample(pixel_nearest, float2(gid, 3.0), gid & 3);
}

// An implicit-LOD sample through a selected texture and a nearest pixel sampler in a fragment
// function: every arm fetches, so nothing is tangled and no gradient is hoisted.
fragment float4 fragment_select_pixel(float4 pos [[position]],
                                      texture2d<float> a [[texture(0)]],
                                      texture2d<float> b [[texture(1)]]) {
    texture2d<float> t = pos.x > 8.0 ? a : b;
    return t.sample(pixel_nearest, pos.xy);
}

// A nearest pixel sampler that clamps to zero: the zero's alpha depends on the pixel format, so
// the device's unnormalized sampler reads the coordinate and applies its own border.
kernel void sample_pixel_zero(texture2d<float> t [[texture(0)]],
                              device float4 *out [[buffer(0)]],
                              uint gid [[thread_position_in_grid]]) {
    out[gid] = t.sample(pixel_zero, float2(gid) + 0.5);
}

// A bias and a minimum-LOD clamp outside a fragment function: Metal's default level there is 0,
// so the level is the bias, clamped from below (an explicit level).
kernel void sample_bias_kernel(texture2d<float> t [[texture(0)]],
                               sampler s [[sampler(0)]],
                               device float4 *out [[buffer(0)]],
                               uint gid [[thread_position_in_grid]]) {
    float2 uv = float2(gid) * 0.25;
    out[gid] = t.sample(s, uv, bias(float(gid & 3) * 0.5))
        + t.sample(s, uv, min_lod_clamp(1.5))
        + t.sample(s, uv, bias(0.5), min_lod_clamp(1.0));
}

struct VOut {
    float4 position [[position]];
};

// A biased sample in a vertex function: an explicit level, as in a kernel.
vertex VOut vertex_bias(uint vid [[vertex_id]],
                        texture2d<float> t [[texture(0)]],
                        sampler s [[sampler(0)]]) {
    VOut o;
    o.position = t.sample(s, float2(float(vid) * 0.5, 0.0), bias(1.0));
    return o;
}

// A dynamic offset on a sample through a runtime sampler (Vulkan admits `Offset` on gathers
// only without VK_KHR_maintenance8).
kernel void sample_dynamic_offset(texture2d<float> t [[texture(0)]],
                                  sampler s [[sampler(0)]],
                                  device float4 *out [[buffer(0)]],
                                  uint gid [[thread_position_in_grid]]) {
    int2 o = int2(int(gid & 7) - 4, 1);
    out[gid] = t.sample(s, float2(gid) * 0.25, level(0.0), o);
}

// An LOD query outside a fragment function: undefined in Metal.
kernel void lod_query_kernel(texture2d<float> t [[texture(0)]],
                             sampler s [[sampler(0)]],
                             device float *out [[buffer(0)]],
                             uint gid [[thread_position_in_grid]]) {
    out[gid] = t.calculate_clamped_lod(s, float2(gid) * 0.25);
}
