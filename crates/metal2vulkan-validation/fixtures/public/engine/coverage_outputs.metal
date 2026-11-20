// The functions of the coverage outputs lane, our own MSL: fragment functions that read
// [[sample_mask]], write memory, so that Metal runs them once per covered sample, and return a
// color, one also a sample mask. Each counts its runs per pixel in buffer 0 and returns the count
// before its run and the coverage it read. The triangle covers the lower left half of the target
// inside the viewport, so no clipping splits it.
#include <metal_stdlib>
using namespace metal;

struct Position {
    float4 position [[position]];
};

vertex Position cvo_vertex(uint id [[vertex_id]]) {
    float2 p = float2(id & 1, id >> 1);
    Position out;
    out.position = float4(p * 2.0 - 1.0, 0.5, 1.0);
    return out;
}

fragment uint4 cvo_color(uint mask [[sample_mask]], float4 position [[position]],
                         device atomic_uint *runs [[buffer(0)]],
                         constant uint &width [[buffer(1)]]) {
    uint pixel = uint(position.y) * width + uint(position.x);
    uint run = atomic_fetch_add_explicit(&runs[pixel], 1u, memory_order_relaxed);
    return uint4(run, mask, 0u, 1u);
}

struct Masked {
    uint4 color [[color(0)]];
    uint mask [[sample_mask]];
};

// Each run keeps the sample whose index is its count.
fragment Masked cvo_masked(uint mask [[sample_mask]], float4 position [[position]],
                           device atomic_uint *runs [[buffer(0)]],
                           constant uint &width [[buffer(1)]]) {
    uint pixel = uint(position.y) * width + uint(position.x);
    uint run = atomic_fetch_add_explicit(&runs[pixel], 1u, memory_order_relaxed);
    Masked out;
    out.color = uint4(run, mask, 0u, 1u);
    out.mask = 1u << run;
    return out;
}

// Each sample's texel of a multisampled target, (red, green) packed as red | green << 8, or
// 0xffff where nothing wrote it (alpha 0).
kernel void cvo_read(texture2d_ms<uint> target [[texture(0)]], device uint *out [[buffer(0)]],
                     uint2 g [[thread_position_in_grid]]) {
    uint n = target.get_num_samples();
    for (uint s = 0; s < n; s++) {
        uint4 c = target.read(g, s);
        out[(g.y * target.get_width() + g.x) * n + s] = c.a == 0u ? 0xffffu : (c.r | c.g << 8);
    }
}
