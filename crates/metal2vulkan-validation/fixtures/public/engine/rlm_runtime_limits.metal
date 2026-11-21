// metal-std: metal4.1
// metal-macos-min: 27.0
// Owned fixture: a shape of a validation requirement whose limit is the runtime's typed refusal
// (`Limit::Runtime`) beside the same shape without it, so a device test holds the refusal both
// ways (the runtime refuses the first of the pair at its pipeline's make and makes the second),
// and a pair of member reads the runtime makes both of.
//
//  * rlm_fence_acq_rel: a4_device_fence's two stores, out[t] = in[t] + 1 and out[t + 8] = 2 in[t]
//    for eight threads, around a device-scope fence of `acq_rel` order, which the translator
//    orders without the execution status word; a4_device_fence's `seq_cst` fence is the shape
//    `seq_cst_fence` names.
//  * rlm_member_texture_buffer: thread t reads texel t of a texture buffer an argument buffer
//    holds; rlm_member_texture: the same read of a 2D texture the argument buffer holds, texel
//    (t, 0).
//
// Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void rlm_fence_acq_rel(device const uint *in [[buffer(0)]],
                              device uint *out [[buffer(1)]],
                              uint t [[thread_position_in_grid]]) {
    out[t] = in[t] + 1u;
    atomic_thread_fence(mem_flags::mem_device, memory_order_acq_rel, thread_scope_device);
    out[t + 8u] = in[t] * 2u;
}

struct BufferMember {
    texture_buffer<float> texels [[id(0)]];
};

struct TextureMember {
    texture2d<float> texels [[id(0)]];
};

kernel void rlm_member_texture_buffer(device const BufferMember &members [[buffer(0)]],
                                      device float4 *out [[buffer(1)]],
                                      uint t [[thread_position_in_grid]]) {
    out[t] = members.texels.read(t);
}

kernel void rlm_member_texture(device const TextureMember &members [[buffer(0)]],
                               device float4 *out [[buffer(1)]],
                               uint t [[thread_position_in_grid]]) {
    out[t] = members.texels.read(uint2(t, 0u));
}
