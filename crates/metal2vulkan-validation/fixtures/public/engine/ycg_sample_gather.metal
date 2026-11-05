// ycg_sample_gather.metal (metal2vulkan): kernels that sample or gather a texture through the
// runtime sampler at `[[sampler(0)]]`, as a probe that asks each lane for one of the three, written
// from scratch for the conformance suite's two-plane YCbCr gather cases (`ycbcr_gathers`). Thread
// `i` writes `o[i]` (buffer 0) at the normalized position `p[i / 3]` (buffer 1): lane `i % 3` 0 a
// sample, 1 a gather of red, 2 a gather of blue. `ycg_float` reads a `float` texture into
// `float4`s, `ycg_half` a `half` one into `half4`s.
#include <metal_stdlib>
using namespace metal;

kernel void ycg_float(texture2d<float> t [[texture(0)]],
                      sampler s [[sampler(0)]],
                      device float4 *o [[buffer(0)]],
                      device const float2 *p [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) {
    float2 at = p[i / 3];
    float4 v;
    switch (i % 3) {
    case 1:
        v = t.gather(s, at, int2(0), component::x);
        break;
    case 2:
        v = t.gather(s, at, int2(0), component::z);
        break;
    default:
        v = t.sample(s, at);
        break;
    }
    o[i] = v;
}

kernel void ycg_half(texture2d<half> t [[texture(0)]],
                     sampler s [[sampler(0)]],
                     device half4 *o [[buffer(0)]],
                     device const float2 *p [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {
    float2 at = p[i / 3];
    half4 v;
    switch (i % 3) {
    case 1:
        v = t.gather(s, at, int2(0), component::x);
        break;
    case 2:
        v = t.gather(s, at, int2(0), component::z);
        break;
    default:
        v = t.sample(s, at);
        break;
    }
    o[i] = v;
}
