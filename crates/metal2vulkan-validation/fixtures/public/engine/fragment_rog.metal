// Owned fixture: raster order groups. Each function names a group
// on a different kind of argument or output; the lowering runs the whole body inside a
// fragment-interlock critical section. `rog_loop` has a loop (run as compiled);
// nothing here runs on a GPU.
#include <metal_stdlib>
using namespace metal;

struct Surface {
    float4 position [[position]];
    float4 color;
};

// A per-pixel read-modify-write of a device buffer in a group.
fragment float4 rog_buffer(float4 position [[position]],
                           device float *acc [[buffer(0), raster_order_group(0)]],
                           constant uint &width [[buffer(1)]]) {
    uint i = uint(position.y) * width + uint(position.x);
    float v = acc[i] + 1.0;
    acc[i] = v;
    return float4(v, 0.0, 0.0, 1.0);
}

// A per-pixel read-modify-write of a read_write texture in a group.
fragment float4 rog_texture(float4 position [[position]],
                            texture2d<float, access::read_write> t [[texture(0), raster_order_group(0)]]) {
    uint2 p = uint2(position.xy);
    float4 v = t.read(p) + float4(0.25);
    t.write(v, p);
    return v;
}

struct Ordered {
    float4 color [[color(0), raster_order_group(0)]];
};

// A color output in a group.
fragment Ordered rog_output(float4 position [[position]]) {
    Ordered out;
    out.color = float4(position.x, position.y, 0.0, 1.0);
    return out;
}

// A discard in a group (demote semantics) before a read of the group's buffer: demote, or
// predication recorded in the body and killed by the entry after the critical section. (A group
// on a buffer the body never reads orders nothing, and would not interlock the function.)
fragment float4 rog_discard(Surface in [[stage_in]],
                            device float *acc [[buffer(0), raster_order_group(0)]]) {
    if (in.color.a < 0.5) {
        discard_fragment();
    }
    return in.color * acc[0];
}

// A loop in a group: the execution bound's forced exit returns from the body, and the entry ends
// the critical section.
fragment float4 rog_loop(float4 position [[position]],
                         device float *acc [[buffer(0), raster_order_group(0)]],
                         constant uint &n [[buffer(1)]]) {
    uint i = uint(position.x);
    float v = acc[i];
    for (uint k = 0; k < n; ++k) {
        v = v * 0.5 + 1.0;
    }
    acc[i] = v;
    return float4(v);
}

// A group in a function that runs at sample rate (it reads `sample_id`): where the device has
// only sample interlock, the sample mode orders exactly Metal's per-sample fragments.
fragment float4 rog_sample(float4 position [[position]],
                           uint sample [[sample_id]],
                           device float *acc [[buffer(0), raster_order_group(0)]]) {
    uint i = uint(position.x) * 4u + sample;
    float v = acc[i] + 1.0;
    acc[i] = v;
    return float4(v);
}

// A discard followed by a simdgroup reduction (demote semantics, not in a group): demote keeps
// the discarded thread a helper; predication would make it a live lane (refused).
fragment float4 discard_then_simd(Surface in [[stage_in]]) {
    float4 c = in.color;
    if (c.a < 0.5) {
        discard_fragment();
    }
    return float4(simd_sum(c.x), c.y, c.z, 1.0);
}
