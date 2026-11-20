// fpv_rgbx_textures.metal (metal2vulkan): kernels reading, sampling and
// writing a texture, written from scratch for the RGBX lane. Each
// thread takes its texel at `gid` of a row `width` texels wide, element `gid.y * width + gid.x`
// of buffer 0.
#include <metal_stdlib>
using namespace metal;

kernel void fpv_rgbx_read_write(texture2d<float, access::read_write> t [[texture(0)]],
                                device float4 *out [[buffer(0)]],
                                constant uint &width [[buffer(1)]],
                                uint2 gid [[thread_position_in_grid]]) {
    out[gid.y * width + gid.x] = t.read(gid);
}

kernel void fpv_rgbx_fetch(texture2d<float, access::read> t [[texture(0)]],
                           device float4 *out [[buffer(0)]],
                           constant uint &width [[buffer(1)]],
                           uint2 gid [[thread_position_in_grid]]) {
    out[gid.y * width + gid.x] = t.read(gid);
}

kernel void fpv_rgbx_sample(texture2d<float> t [[texture(0)]],
                            sampler s [[sampler(0)]],
                            device float4 *out [[buffer(0)]],
                            constant uint &width [[buffer(1)]],
                            uint2 gid [[thread_position_in_grid]]) {
    float2 c = (float2(gid) + 0.5) / float2(t.get_width(), t.get_height());
    out[gid.y * width + gid.x] = t.sample(s, c);
}

kernel void fpv_rgbx_write(texture2d<float, access::write> t [[texture(0)]],
                           device const float4 *values [[buffer(0)]],
                           constant uint &width [[buffer(1)]],
                           uint2 gid [[thread_position_in_grid]]) {
    t.write(values[gid.y * width + gid.x], gid);
}

kernel void fpv_rgbx_read_write_uint(texture2d<uint, access::read_write> t [[texture(0)]],
                                     device uint4 *out [[buffer(0)]],
                                     constant uint &width [[buffer(1)]],
                                     uint2 gid [[thread_position_in_grid]]) {
    out[gid.y * width + gid.x] = t.read(gid);
}

kernel void fpv_rgbx_write_uint(texture2d<uint, access::write> t [[texture(0)]],
                                device const uint4 *values [[buffer(0)]],
                                constant uint &width [[buffer(1)]],
                                uint2 gid [[thread_position_in_grid]]) {
    t.write(values[gid.y * width + gid.x], gid);
}

kernel void fpv_rgbx_read_write_int(texture2d<int, access::read_write> t [[texture(0)]],
                                    device int4 *out [[buffer(0)]],
                                    constant uint &width [[buffer(1)]],
                                    uint2 gid [[thread_position_in_grid]]) {
    out[gid.y * width + gid.x] = t.read(gid);
}

kernel void fpv_rgbx_write_int(texture2d<int, access::write> t [[texture(0)]],
                               device const int4 *values [[buffer(0)]],
                               constant uint &width [[buffer(1)]],
                               uint2 gid [[thread_position_in_grid]]) {
    t.write(values[gid.y * width + gid.x], gid);
}
