// The functions of the extended-range resolve lane, our own MSL: a fullscreen triangle, and a
// fragment function that runs once per sample and writes each sample its own color, read from
// buffer 0 at `(y * width + x) * samples + sample`, the width and the samples in buffer 1.
#include <metal_stdlib>
using namespace metal;

struct Position {
    float4 position [[position]];
};

struct Shape {
    uint width;
    uint samples;
};

vertex Position xrr_vertex(uint id [[vertex_id]]) {
    float2 p = float2((id << 1) & 2, id & 2);
    Position out;
    out.position = float4(p * 2.0 - 1.0, 0.5, 1.0);
    return out;
}

fragment float4 xrr_samples(Position in [[stage_in]], uint sample [[sample_id]],
                            constant float4 *colors [[buffer(0)]],
                            constant Shape &shape [[buffer(1)]]) {
    uint pixel = uint(in.position.y) * shape.width + uint(in.position.x);
    return colors[pixel * shape.samples + sample];
}
