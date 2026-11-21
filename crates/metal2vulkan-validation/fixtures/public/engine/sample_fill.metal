// metal-fast-math: off
// Owned fixture: the validation executor's per-sample fill and read of a multisampled texture
// through the runtime (`executor_contract::multisample`), which uploads and copies no
// multisampled texture, as Metal neither: one draw writes every sample of one slice from a
// buffer, and one dispatch reads every sample back.
//
//  * sample_fill_vertex: one triangle over the whole target, vertex v at clip position
//    (2 (v << 1 & 2) - 1, 2 (v & 2) - 1, 0, 1).
//  * sample_fill_color: sample s of the pixel at (x, y) writes texels[(y * width + x) * samples
//    + s], through an unsigned-integer view of the attachment's texel size, so the bits cross
//    unconverted.
//  * sample_fill_depth: the same, writing depth texels[...] under `depth(any)`.
//  * sample_read_color, sample_read_depth, sample_read_stencil: the threadgroup at (x, y, s)
//    reads sample s of pixel (x, y), through an unsigned-integer view, as a bit-cast depth, or as
//    a stencil value (the first component, Metal leaving the others undefined), and stores its
//    `words` words at word ((y * width + x) * samples + s) * words.
//
// Loop-free.
#include <metal_stdlib>
using namespace metal;

struct SampleShape {
    uint width;
    uint samples;
    uint words;
};

vertex float4 sample_fill_vertex(uint vertex_id [[vertex_id]]) {
    float2 corner = float2((vertex_id << 1) & 2, vertex_id & 2);
    return float4(corner * 2.0 - 1.0, 0.0, 1.0);
}

fragment uint4 sample_fill_color(float4 position [[position]], uint sample [[sample_id]],
                                 device const uint4 *texels [[buffer(0)]],
                                 constant SampleShape &shape [[buffer(1)]]) {
    uint2 pixel = uint2(position.xy);
    return texels[(pixel.y * shape.width + pixel.x) * shape.samples + sample];
}

struct SampleFillDepth {
    float depth [[depth(any)]];
};

fragment SampleFillDepth sample_fill_depth(float4 position [[position]],
                                           uint sample [[sample_id]],
                                           device const float *texels [[buffer(0)]],
                                           constant SampleShape &shape [[buffer(1)]]) {
    uint2 pixel = uint2(position.xy);
    return {texels[(pixel.y * shape.width + pixel.x) * shape.samples + sample]};
}

static void store(uint4 texel, uint3 position, device uint *words, constant SampleShape &shape) {
    uint base = ((position.y * shape.width + position.x) * shape.samples + position.z) * shape.words;
    words[base] = texel.x;
    if (shape.words > 1) {
        words[base + 1] = texel.y;
    }
    if (shape.words > 2) {
        words[base + 2] = texel.z;
        words[base + 3] = texel.w;
    }
}

kernel void sample_read_color(texture2d_ms<uint, access::read> source [[texture(0)]],
                              device uint *words [[buffer(0)]],
                              constant SampleShape &shape [[buffer(1)]],
                              uint3 position [[threadgroup_position_in_grid]]) {
    store(source.read(position.xy, position.z), position, words, shape);
}

kernel void sample_read_depth(depth2d_ms<float, access::read> source [[texture(0)]],
                              device uint *words [[buffer(0)]],
                              constant SampleShape &shape [[buffer(1)]],
                              uint3 position [[threadgroup_position_in_grid]]) {
    store(uint4(as_type<uint>(source.read(position.xy, position.z))), position, words, shape);
}

kernel void sample_read_stencil(texture2d_ms<uint, access::read> source [[texture(0)]],
                                device uint *words [[buffer(0)]],
                                constant SampleShape &shape [[buffer(1)]],
                                uint3 position [[threadgroup_position_in_grid]]) {
    store(uint4(source.read(position.xy, position.z).x), position, words, shape);
}
