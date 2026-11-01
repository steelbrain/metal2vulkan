// Owned MSL: kernels calling functions a stitched library defines, as the probe
// `stitched_function_linked.m` builds them. `stitch_caller` stores `fused(in[0], in[1], in[2])`
// and `stitch_caller4` stores `fused4(in[0], in[1], in[2], in[3])` to word 0 of buffer 0, where
// `fused` and `fused4` are visible functions this library declares and does not define: the
// functions Metal's stitcher makes of the stitchable `sum2` and `product2`,
// `fused(a, b, c) = product2(sum2(a, b), c)` and `fused4(a, b, c, d) = sum2(fused(a, b, c), d)`
// (`metal2vulkan_devtools::stitched` writes their modules). Loop-free.
// metal-macos-min: 27.0
#include <metal_stdlib>
using namespace metal;

[[stitchable]] float sum2(float a, float b) { return a + b; }

[[stitchable]] float product2(float a, float b) { return a * b; }

[[visible]] float fused(float a, float b, float c);
[[visible]] float fused4(float a, float b, float c, float d);

kernel void stitch_caller(device float *out [[buffer(0)]], constant float *in [[buffer(1)]]) {
    out[0] = fused(in[0], in[1], in[2]);
}

kernel void stitch_caller4(device float *out [[buffer(0)]], constant float *in [[buffer(1)]]) {
    out[0] = fused4(in[0], in[1], in[2], in[3]);
}
