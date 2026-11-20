// fpv_bfloat_textures.metal (metal2vulkan): kernels reading, sampling,
// gathering and writing a texture, written from scratch for the bfloat lane. Each thread takes its
// texel at `gid` of a row `width` texels
// wide, element `gid.y * width + gid.x` of buffer 0.
#include <metal_stdlib>
using namespace metal;

kernel void fpv_bfloat_fetch(texture2d<float, access::read> t [[texture(0)]],
                             device float4 *out [[buffer(0)]],
                             constant uint &width [[buffer(1)]],
                             uint2 gid [[thread_position_in_grid]]) {
    out[gid.y * width + gid.x] = t.read(gid);
}

kernel void fpv_bfloat_fetch_half(texture2d<half, access::read> t [[texture(0)]],
                                  device half4 *out [[buffer(0)]],
                                  constant uint &width [[buffer(1)]],
                                  uint2 gid [[thread_position_in_grid]]) {
    out[gid.y * width + gid.x] = t.read(gid);
}

kernel void fpv_bfloat_read_write(texture2d<float, access::read_write> t [[texture(0)]],
                                  device float4 *out [[buffer(0)]],
                                  constant uint &width [[buffer(1)]],
                                  uint2 gid [[thread_position_in_grid]]) {
    out[gid.y * width + gid.x] = t.read(gid);
}

// Off the texel's centre, where a linear filter would mix four texels.
kernel void fpv_bfloat_sample(texture2d<float> t [[texture(0)]],
                              sampler s [[sampler(0)]],
                              device float4 *out [[buffer(0)]],
                              constant uint &width [[buffer(1)]],
                              uint2 gid [[thread_position_in_grid]]) {
    float2 c = (float2(gid) + float2(0.75, 0.625)) / float2(t.get_width(), t.get_height());
    out[gid.y * width + gid.x] = t.sample(s, c);
}

// The same position, a `half` result: Metal rounds a bfloat sample's toward zero.
kernel void fpv_bfloat_sample_half(texture2d<half> t [[texture(0)]],
                                   sampler s [[sampler(0)]],
                                   device half4 *out [[buffer(0)]],
                                   constant uint &width [[buffer(1)]],
                                   uint2 gid [[thread_position_in_grid]]) {
    float2 c = (float2(gid) + float2(0.75, 0.625)) / float2(t.get_width(), t.get_height());
    out[gid.y * width + gid.x] = t.sample(s, c);
}

// The footprint's corner: texels gid and gid + 1 on each axis.
kernel void fpv_bfloat_gather(texture2d<float> t [[texture(0)]],
                              sampler s [[sampler(0)]],
                              device float4 *out [[buffer(0)]],
                              constant uint &width [[buffer(1)]],
                              uint2 gid [[thread_position_in_grid]]) {
    float2 c = (float2(gid) + 1.0) / float2(t.get_width(), t.get_height());
    out[gid.y * width + gid.x] = t.gather(s, c, int2(0), component::y);
}

kernel void fpv_bfloat_write(texture2d<float, access::write> t [[texture(0)]],
                             device const float4 *values [[buffer(0)]],
                             constant uint &width [[buffer(1)]],
                             uint2 gid [[thread_position_in_grid]]) {
    t.write(values[gid.y * width + gid.x], gid);
}

kernel void fpv_bfloat_write_half(texture2d<half, access::write> t [[texture(0)]],
                                  device const half4 *values [[buffer(0)]],
                                  constant uint &width [[buffer(1)]],
                                  uint2 gid [[thread_position_in_grid]]) {
    t.write(values[gid.y * width + gid.x], gid);
}
