// fpv_srgb_storage.metal (metal2vulkan): kernels reading and writing an 8-bit
// sRGB texture at a storage slot, `float` and `half`, written from scratch for the sRGB storage
// lane. Each thread takes its texel at `gid` of a row `width` texels wide,
// element `gid.y * width + gid.x` of buffer 0.
#include <metal_stdlib>
using namespace metal;

kernel void fpv_srgb_read_write(texture2d<float, access::read_write> t [[texture(0)]],
                                device float4 *out [[buffer(0)]],
                                constant uint &width [[buffer(1)]],
                                uint2 gid [[thread_position_in_grid]]) {
    out[gid.y * width + gid.x] = t.read(gid);
}

kernel void fpv_srgb_read_write_half(texture2d<half, access::read_write> t [[texture(0)]],
                                     device half4 *out [[buffer(0)]],
                                     constant uint &width [[buffer(1)]],
                                     uint2 gid [[thread_position_in_grid]]) {
    out[gid.y * width + gid.x] = t.read(gid);
}

kernel void fpv_srgb_write(texture2d<float, access::write> t [[texture(0)]],
                           device const float4 *values [[buffer(0)]],
                           constant uint &width [[buffer(1)]],
                           uint2 gid [[thread_position_in_grid]]) {
    t.write(values[gid.y * width + gid.x], gid);
}

kernel void fpv_srgb_write_half(texture2d<half, access::write> t [[texture(0)]],
                                device const half4 *values [[buffer(0)]],
                                constant uint &width [[buffer(1)]],
                                uint2 gid [[thread_position_in_grid]]) {
    t.write(values[gid.y * width + gid.x], gid);
}

kernel void fpv_srgb_update(texture2d<float, access::read_write> t [[texture(0)]],
                            device const float4 *values [[buffer(0)]],
                            constant uint &width [[buffer(1)]],
                            uint2 gid [[thread_position_in_grid]]) {
    t.write(values[gid.y * width + gid.x], gid);
}

kernel void fpv_srgb_update_half(texture2d<half, access::read_write> t [[texture(0)]],
                                 device const half4 *values [[buffer(0)]],
                                 constant uint &width [[buffer(1)]],
                                 uint2 gid [[thread_position_in_grid]]) {
    t.write(values[gid.y * width + gid.x], gid);
}
