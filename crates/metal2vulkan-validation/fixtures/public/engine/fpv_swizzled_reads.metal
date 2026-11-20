// fpv_swizzled_reads.metal (metal2vulkan): kernels reading, sampling and gathering a
// float texture at positions a buffer gives, written from scratch for the swizzle lane. Thread
// `i` writes `o[i]` (buffer 0) from position `p[i]`
// (buffer 1): a texel coordinate for a read, a normalized coordinate for a sample or a gather
// through the runtime sampler at `[[sampler(0)]]`.
#include <metal_stdlib>
using namespace metal;

kernel void fpv_swz_read(texture2d<float, access::read> t [[texture(0)]],
                         device float4 *o [[buffer(0)]],
                         device const uint2 *p [[buffer(1)]],
                         uint i [[thread_position_in_grid]]) {
    o[i] = t.read(p[i]);
}

kernel void fpv_swz_sample(texture2d<float> t [[texture(0)]],
                           sampler s [[sampler(0)]],
                           device float4 *o [[buffer(0)]],
                           device const float2 *p [[buffer(1)]],
                           uint i [[thread_position_in_grid]]) {
    o[i] = t.sample(s, p[i]);
}

kernel void fpv_swz_gather_x(texture2d<float> t [[texture(0)]],
                             sampler s [[sampler(0)]],
                             device float4 *o [[buffer(0)]],
                             device const float2 *p [[buffer(1)]],
                             uint i [[thread_position_in_grid]]) {
    o[i] = t.gather(s, p[i], int2(0), component::x);
}

kernel void fpv_swz_gather_y(texture2d<float> t [[texture(0)]],
                             sampler s [[sampler(0)]],
                             device float4 *o [[buffer(0)]],
                             device const float2 *p [[buffer(1)]],
                             uint i [[thread_position_in_grid]]) {
    o[i] = t.gather(s, p[i], int2(0), component::y);
}

kernel void fpv_swz_gather_z(texture2d<float> t [[texture(0)]],
                             sampler s [[sampler(0)]],
                             device float4 *o [[buffer(0)]],
                             device const float2 *p [[buffer(1)]],
                             uint i [[thread_position_in_grid]]) {
    o[i] = t.gather(s, p[i], int2(0), component::z);
}

kernel void fpv_swz_gather_w(texture2d<float> t [[texture(0)]],
                             sampler s [[sampler(0)]],
                             device float4 *o [[buffer(0)]],
                             device const float2 *p [[buffer(1)]],
                             uint i [[thread_position_in_grid]]) {
    o[i] = t.gather(s, p[i], int2(0), component::w);
}
