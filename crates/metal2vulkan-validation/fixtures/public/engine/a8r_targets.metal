// The functions of the `A8Unorm` lane, our own MSL:
// a fullscreen triangle and one over the lower left half of the target, fragments writing the
// color in buffer 0 (and the second color as a second output), one also reading its attachment,
// and kernels that write, read, sample and read-and-write an 8 x 8 texture at every texel.
#include <metal_stdlib>
using namespace metal;

struct Position {
    float4 position [[position]];
};

vertex Position a8r_vertex(uint id [[vertex_id]]) {
    float2 p = float2((id << 1) & 2, id & 2);
    Position out;
    out.position = float4(p * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

// Its edge runs through the target's texel centers on the diagonal from the top left.
vertex Position a8r_half_vertex(uint id [[vertex_id]]) {
    float2 p = float2(id == 1 ? 1.0 : -1.0, id == 2 ? 1.0 : -1.0);
    Position out;
    out.position = float4(p, 0.0, 1.0);
    return out;
}

fragment float4 a8r_fill(Position in [[stage_in]], constant float4 *color [[buffer(0)]]) {
    return color[0];
}

struct Dual {
    float4 first [[color(0), index(0)]];
    float4 second [[color(0), index(1)]];
};

fragment Dual a8r_dual(Position in [[stage_in]], constant float4 *color [[buffer(0)]]) {
    Dual out;
    out.first = color[0];
    out.second = color[1];
    return out;
}

fragment float4 a8r_fetch(Position in [[stage_in]], float4 held [[color(0)]],
                          constant float4 *color [[buffer(0)]], device float4 *out [[buffer(1)]]) {
    uint2 p = uint2(in.position.xy);
    out[p.y * 8 + p.x] = held;
    return color[0];
}

kernel void a8r_write(texture2d<float, access::write> t [[texture(0)]],
                      constant float4 *color [[buffer(0)]], uint2 g [[thread_position_in_grid]]) {
    t.write(color[0], g);
}

kernel void a8r_write_half(texture2d<half, access::write> t [[texture(0)]],
                           constant float4 *color [[buffer(0)]],
                           uint2 g [[thread_position_in_grid]]) {
    t.write(half4(color[0]), g);
}

kernel void a8r_read(texture2d<float, access::read> t [[texture(0)]],
                     device float4 *out [[buffer(1)]], uint2 g [[thread_position_in_grid]]) {
    out[g.y * 8 + g.x] = t.read(g);
}

kernel void a8r_sample(texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]],
                       device float4 *out [[buffer(1)]], uint2 g [[thread_position_in_grid]]) {
    out[g.y * 8 + g.x] = t.sample(s, (float2(g) + 0.5) / 8.0);
}

kernel void a8r_read_write(texture2d<float, access::read_write> t [[texture(0)]],
                           constant float4 *color [[buffer(0)]], device float4 *out [[buffer(1)]],
                           uint2 g [[thread_position_in_grid]]) {
    out[g.y * 8 + g.x] = t.read(g);
    t.write(color[0], g);
}
