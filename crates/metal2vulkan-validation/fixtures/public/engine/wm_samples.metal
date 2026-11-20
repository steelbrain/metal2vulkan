// The functions of the word merge's multisampled lane, our own MSL: a fullscreen triangle and one
// over the lower left half of the target, a fragment writing the color in buffer 0, and kernels
// that read every sample of a 4-sample 8 x 8 texture and every texel of a single-sample one into
// buffer 1.
#include <metal_stdlib>
using namespace metal;

struct Position {
    float4 position [[position]];
};

vertex Position wm_vertex(uint id [[vertex_id]]) {
    float2 p = float2((id << 1) & 2, id & 2);
    Position out;
    out.position = float4(p * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

// Its edge runs through the target's texel centers on the diagonal from the top left.
vertex Position wm_half_vertex(uint id [[vertex_id]]) {
    float2 p = float2(id == 1 ? 1.0 : -1.0, id == 2 ? 1.0 : -1.0);
    Position out;
    out.position = float4(p, 0.0, 1.0);
    return out;
}

fragment float4 wm_fill(Position in [[stage_in]], constant float4 *color [[buffer(0)]]) {
    return color[0];
}

kernel void wm_read_samples(texture2d_ms<float, access::read> t [[texture(0)]],
                            device float4 *out [[buffer(1)]], uint2 g [[thread_position_in_grid]]) {
    for (uint s = 0; s < 4; s++) {
        out[(g.y * 8 + g.x) * 4 + s] = t.read(g, s);
    }
}

kernel void wm_read(texture2d<float, access::read> t [[texture(0)]],
                    device float4 *out [[buffer(1)]], uint2 g [[thread_position_in_grid]]) {
    out[g.y * 8 + g.x] = t.read(g);
}
