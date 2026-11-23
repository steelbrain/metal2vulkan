// Owned fixture: writes to a storage cube and a storage cube array by face, at level 0, and their
// sizes. Thread (x, y, f) writes (x, y, f, 1) at texel (x, y) of face f of the cube `c` and of
// face f of the second cube of the cube array `a`; thread (0, 0, 0) stores the cube's width and
// height and the array's cube count to words 0, 1 and 2 of `out`. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void cube_face_writes(texturecube<float, access::write> c [[texture(0)]],
                             texturecube_array<float, access::write> a [[texture(1)]],
                             device uint *out [[buffer(0)]],
                             uint3 gid [[thread_position_in_grid]]) {
    float4 v = float4(float(gid.x), float(gid.y), float(gid.z), 1.0f);
    c.write(v, gid.xy, gid.z);
    a.write(v, gid.xy, gid.z, 1);
    if (gid.x == 0 && gid.y == 0 && gid.z == 0) {
        out[0] = c.get_width();
        out[1] = c.get_height();
        out[2] = a.get_array_size();
    }
}
