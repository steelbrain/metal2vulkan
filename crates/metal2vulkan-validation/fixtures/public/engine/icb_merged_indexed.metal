// Owned fixture: indexed draws of triangles that a kernel encodes into an indirect command buffer
// and a render pass executes, each command with its own fragment buffer. Kernel
// `icb_indexed_<size>` encodes command `tid`: vertex buffer 0 its own word of `marks`, vertex
// buffer 1 the frame, vertex buffer 2 the vertices' pixel positions, fragment buffer 0 its own
// word of `tints`, and the indexed triangle draw `draws[tid]` (first index, index count, instance
// count, base vertex, base instance) of the 16-bit or 32-bit indices of `indices` from its first
// index. Vertex `vid` of instance `iid` lies at `xy[vid]` moved down `iid` rows of `frame.z`
// pixels (`frame` is the target's width, height and row stride in pixels) and carries its
// command's mark, `vid` and `iid` flat to the fragment function, which returns them with its
// command's tint. The vertex function writes no memory.
// Written for this repository; no third-party source.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

struct IndexedCommands {
    command_buffer commands;
};

struct IndexedDraw {
    uint first;
    uint count;
    uint instances;
    uint base_vertex;
    uint base_instance;
};

struct IndexedOut {
    float4 position [[position]];
    uint4 ids [[flat]];
};

static void set_buffers(thread render_command &command, uint tid, constant uint *marks,
                        constant float4 *frame, device const float2 *xy, constant uint *tints) {
    command.set_vertex_buffer(marks + tid, 0);
    command.set_vertex_buffer(frame, 1);
    command.set_vertex_buffer(xy, 2);
    command.set_fragment_buffer(tints + tid, 0);
}

kernel void icb_indexed_16(device const IndexedCommands &encoder [[buffer(0)]],
                           device const IndexedDraw *draws [[buffer(1)]],
                           constant uint *marks [[buffer(2)]],
                           constant float4 *frame [[buffer(3)]],
                           device const float2 *xy [[buffer(4)]],
                           constant uint *tints [[buffer(5)]],
                           device const ushort *indices [[buffer(6)]],
                           uint tid [[thread_position_in_grid]]) {
    render_command command(encoder.commands, tid);
    set_buffers(command, tid, marks, frame, xy, tints);
    IndexedDraw d = draws[tid];
    command.draw_indexed_primitives(primitive_type::triangle, d.count, indices + d.first,
                                    d.instances, d.base_vertex, d.base_instance);
}

kernel void icb_indexed_32(device const IndexedCommands &encoder [[buffer(0)]],
                           device const IndexedDraw *draws [[buffer(1)]],
                           constant uint *marks [[buffer(2)]],
                           constant float4 *frame [[buffer(3)]],
                           device const float2 *xy [[buffer(4)]],
                           constant uint *tints [[buffer(5)]],
                           device const uint *indices [[buffer(6)]],
                           uint tid [[thread_position_in_grid]]) {
    render_command command(encoder.commands, tid);
    set_buffers(command, tid, marks, frame, xy, tints);
    IndexedDraw d = draws[tid];
    command.draw_indexed_primitives(primitive_type::triangle, d.count, indices + d.first,
                                    d.instances, d.base_vertex, d.base_instance);
}

vertex IndexedOut icb_indexed_vertex(uint vid [[vertex_id]], uint iid [[instance_id]],
                                     constant uint &mark [[buffer(0)]],
                                     constant float4 &frame [[buffer(1)]],
                                     device const float2 *xy [[buffer(2)]]) {
    float2 at = xy[vid] + float2(0.0, float(iid) * frame.z);
    IndexedOut out;
    out.position = float4(2.0 * at.x / frame.x - 1.0, 1.0 - 2.0 * at.y / frame.y, 0.0, 1.0);
    out.ids = uint4(mark, vid, iid, 0);
    return out;
}

fragment uint4 icb_indexed_fragment(IndexedOut in [[stage_in]], constant uint &tint [[buffer(0)]]) {
    return uint4(in.ids.x, in.ids.y, in.ids.z, tint);
}
