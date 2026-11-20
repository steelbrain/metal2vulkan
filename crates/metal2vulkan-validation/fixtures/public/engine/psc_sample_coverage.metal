// The functions of the private sample coverage lane, our own MSL: a fullscreen triangle at depth
// 0.5; fragment functions that write white, count their runs and the coverage they read (writing no
// color), or declare early fragment tests; and kernels that read every sample of a 16 x 16 color or
// depth texture into buffer 1, a mask of the samples written per pixel.
#include <metal_stdlib>
using namespace metal;

struct Position {
    float4 position [[position]];
};

vertex Position psc_vertex(uint id [[vertex_id]]) {
    float2 p = float2((id << 1) & 2, id & 2);
    Position out;
    out.position = float4(p * 2.0 - 1.0, 0.5, 1.0);
    return out;
}

fragment float4 psc_white() {
    return float4(1.0);
}

// Buffer 0 holds two words per pixel of a 16-pixel-wide target: the runs, and the coverage the
// runs read, ORed. It writes no color.
fragment void psc_counter(Position in [[stage_in]], uint mask [[sample_mask]],
                            device atomic_uint *runs [[buffer(0)]]) {
    uint i = (uint(in.position.y) * 16 + uint(in.position.x)) * 2;
    atomic_fetch_add_explicit(&runs[i], 1, memory_order_relaxed);
    atomic_fetch_or_explicit(&runs[i + 1], mask, memory_order_relaxed);
}

[[early_fragment_tests]] fragment float4 psc_early() {
    return float4(1.0);
}

kernel void psc_read_samples(texture2d_ms<float, access::read> t [[texture(0)]],
                             device uint *out [[buffer(1)]], uint2 g [[thread_position_in_grid]]) {
    uint m = 0;
    for (uint s = 0; s < t.get_num_samples(); s++) {
        if (t.read(g, s).r > 0.5) {
            m |= 1u << s;
        }
    }
    out[g.y * 16 + g.x] = m;
}

kernel void psc_read(texture2d<float, access::read> t [[texture(0)]],
                     device uint *out [[buffer(1)]], uint2 g [[thread_position_in_grid]]) {
    out[g.y * 16 + g.x] = t.read(g).r > 0.5 ? 1u : 0u;
}

// A sample's depth is below 0.75 where the draw wrote its 0.5 over the clear's 1.
kernel void psc_read_depth_samples(depth2d_ms<float, access::read> t [[texture(0)]],
                                   device uint *out [[buffer(1)]],
                                   uint2 g [[thread_position_in_grid]]) {
    uint m = 0;
    for (uint s = 0; s < t.get_num_samples(); s++) {
        if (t.read(g, s) < 0.75) {
            m |= 1u << s;
        }
    }
    out[g.y * 16 + g.x] = m;
}

kernel void psc_read_depth(depth2d<float, access::read> t [[texture(0)]],
                           device uint *out [[buffer(1)]], uint2 g [[thread_position_in_grid]]) {
    out[g.y * 16 + g.x] = t.read(g) < 0.75 ? 1u : 0u;
}
