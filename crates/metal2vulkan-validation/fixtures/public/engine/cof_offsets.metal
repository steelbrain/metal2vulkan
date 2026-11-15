// Owned fixture: the functions the runtime lane's `constant_offsets` cost run draws with, a small
// triangle placed by a float4 at buffer 0 that each draw binds at another offset of one buffer.
// The vertex function reads it through a constant reference (`cof_constant`) or through a device
// pointer at an index it cannot bound (`cof_device`), the two forms whose per-draw rebinds the run
// compares; the fragment function's colour is a constant reference too. Written for this
// repository; no third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct CofVertex {
    float4 position [[position]];
};

// The corner of vertex `vid` of a right triangle of unit legs: (0, 0), (1, 0) and (0, 1).
static float2 cof_corner(uint vid) {
    return float2(vid == 1u ? 1.0 : 0.0, vid == 2u ? 1.0 : 0.0);
}

// The triangle at q.xy, its legs q.zw long, its q a constant reference.
vertex CofVertex cof_constant(uint vid [[vertex_id]], constant float4 &q [[buffer(0)]]) {
    CofVertex out;
    out.position = float4(q.xy + cof_corner(vid) * q.zw, 0.0, 1.0);
    return out;
}

// The same triangle, its q read through a device pointer at the instance's element: an index the
// function cannot bound, so the slot's window runs to the end of the bound buffer. Drawn as one
// instance from instance 0, it reads q[0].
vertex CofVertex cof_device(uint vid [[vertex_id]], uint iid [[instance_id]],
                            const device float4 *q [[buffer(0)]]) {
    float4 r = q[iid];
    CofVertex out;
    out.position = float4(r.xy + cof_corner(vid) * r.zw, 0.0, 1.0);
    return out;
}

// The colour c.
fragment float4 cof_flat(CofVertex in [[stage_in]], constant float4 &c [[buffer(0)]]) {
    return c;
}
