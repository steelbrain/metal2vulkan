// metal-std: macos-metal1.1
// metal-macos-min: 10.11
// Owned fixture: a fragment function's sampling at AIR 1.8, the oldest AIR this toolchain emits
// (see `air_1_8_front_end`, whose surface has no sampling). Before AIR 2.0 the compiler declares
//
//  - `air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>,
//    i1, float)`: no minimum level-of-detail clamp and no trailing access operand;
//  - `air.sample_texture_2d_grad.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, <2 x
//    float>, <2 x float>, i1, <2 x i32>)`: the same for an explicit gradient;
//
// a buffer argument as `air.buffer` with `air.buffer_size` and no `air.address_space`, and the
// render target as `air.render_target` with the attachment alone. The sampler is an argument, not
// a `constexpr sampler` (whose AIR 1.8 form `air_1_8_front_end` explains). The coordinate is the
// pixel position over the 4x4 texture's width, so each pixel center is a texel center and its
// derivatives are exactly a quarter, which puts the gradient at level 0. Loop-free.
#include <metal_stdlib>
using namespace metal;

fragment float4 a18_sample_fs(float4 pos [[position]], constant float4 &tint [[buffer(0)]],
                              texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]]) {
    float2 uv = pos.xy * 0.25f;
    float4 a = t.sample(s, uv);
    float4 b = t.sample(s, uv, gradient2d(dfdx(uv), dfdy(uv)));
    return (a + b) * tint;
}
