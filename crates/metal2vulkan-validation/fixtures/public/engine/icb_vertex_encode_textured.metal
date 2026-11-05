// Owned fixture: as icb_vertex_encode_indexed, a vertex function whose only effect is the indexed
// draw it encodes, as a pass with rasterization off encodes one a draw. The draw of instance
// `iid` (its base instance) writes command `iid`: a draw of 6 indices from the 16-bit `indices`,
// from index 6 * (iid % 64), once, of primitive type iid % 5 (point, line, line strip, triangle,
// triangle strip), at base vertex 0, whose fragment buffer 0 is the buffer slot `iid` of `slots`
// points at: an argument buffer whose entry names a texture, one buffer and one texture a command
// as a page's bind groups are. The commands inherit the executing encoder's pipeline state, and
// set their buffers themselves. `ivt_vertex` places vertex `vid` on a grid of 8 by 8 points, one
// pixel wide, and `ivt_fragment` colours it with texel (0, 0) of its command's entry's texture.
// Written for this repository; no third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Encoder {
    command_buffer commands;
};

struct Entry {
    texture2d<half> texture;
};

struct Slot {
    device Entry *entry;
};

vertex void ivt_encode(device const Encoder &encoder [[buffer(0)]],
                       const device ushort *indices [[buffer(1)]],
                       device const Slot *slots [[buffer(2)]],
                       uint iid [[instance_id]])
{
    render_command command(encoder.commands, iid);
    command.set_fragment_buffer(slots[iid].entry, 0);
    command.draw_indexed_primitives(static_cast<primitive_type>(iid % 5), 6,
                                    indices + 6 * (iid % 64), 1, 0, 0);
}

struct GridPoint {
    float4 position [[position]];
    float size [[point_size]];
};

vertex GridPoint ivt_vertex(uint vid [[vertex_id]])
{
    GridPoint point;
    float x = float(vid % 8) / 4.0 - 0.875;
    float y = float(vid / 8 % 8) / 4.0 - 0.875;
    point.position = float4(x, y, 0.0, 1.0);
    point.size = 1.0;
    return point;
}

fragment half4 ivt_fragment(device const Entry &entry [[buffer(0)]])
{
    return entry.texture.read(uint2(0, 0));
}
