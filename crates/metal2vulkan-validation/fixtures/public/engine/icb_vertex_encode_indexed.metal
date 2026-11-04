// Owned fixture: a vertex function whose only effect is the indexed draw it encodes, as a pass
// with rasterization off encodes one a draw. The draw of instance `iid` (its base instance)
// writes command `iid`: a draw of 6 indices from the 16-bit `indices`, from index
// 6 * (iid % 64), once, of primitive type iid % 5 (point, line, line strip, triangle, triangle
// strip), at base vertex 0. The commands inherit the executing encoder's pipeline state and
// buffers. `ive_vertex` places vertex `vid` on a grid of 8 by 8 points, one pixel wide, and
// `ive_fragment` colours it green. Written for this repository; no third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Encoder {
    command_buffer commands;
};

vertex void ive_encode(device const Encoder &encoder [[buffer(0)]],
                       const device ushort *indices [[buffer(1)]],
                       uint iid [[instance_id]])
{
    render_command command(encoder.commands, iid);
    command.draw_indexed_primitives(static_cast<primitive_type>(iid % 5), 6,
                                    indices + 6 * (iid % 64), 1, 0, 0);
}

struct GridPoint {
    float4 position [[position]];
    float size [[point_size]];
};

vertex GridPoint ive_vertex(uint vid [[vertex_id]])
{
    GridPoint point;
    float x = float(vid % 8) / 4.0 - 0.875;
    float y = float(vid / 8 % 8) / 4.0 - 0.875;
    point.position = float4(x, y, 0.0, 1.0);
    point.size = 1.0;
    return point;
}

fragment half4 ive_fragment()
{
    return half4(0.0, 1.0, 0.0, 1.0);
}
