// Owned fixture: draws of triangles whose vertex function fetches `[[stage_in]]` attributes, which
// a kernel encodes into an indirect command buffer and a render pass executes. Each kernel encodes
// command `tid` from `draws[tid]`: vertex buffer 0 the vertices' pixel positions from byte
// `xy_at` of `xy`, vertex buffer 1 the marks from byte `mark_at` of `marks`, vertex buffer 2 the
// frame, then a triangle draw: `ima_draws` of `count` vertices from vertex `first`,
// `ima_indexed` of `count` 32-bit indices from index `first` at base vertex `base_vertex`, and
// `ima_strided` as `ima_draws` with vertex buffer 0 bound at attribute stride `stride`; each of
// `instances` instances from base instance `base_instance`. The case's vertex descriptor places
// attribute 0 (a `float2` pixel position) in layout 0 and attribute 1 (a `uint` mark) in layout 1.
// Vertex `vid` of instance `iid` lies at its position moved down `iid` rows of `frame.z` pixels
// (`frame` is the target's width, height and row stride in pixels) and carries its mark, `vid`
// and `iid` flat to the fragment function, which returns them. The vertex function writes no
// memory, and the fragment function reads no buffer.
// Written for this repository; no third-party source.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

struct ImaCommands {
    command_buffer commands;
};

struct ImaDraw {
    uint first;
    uint count;
    uint instances;
    uint base_vertex;
    uint base_instance;
    uint xy_at;
    uint mark_at;
    uint stride;
};

struct ImaIn {
    float2 xy [[attribute(0)]];
    uint mark [[attribute(1)]];
};

struct ImaOut {
    float4 position [[position]];
    uint4 ids [[flat]];
};

kernel void ima_draws(device const ImaCommands &encoder [[buffer(0)]],
                      device const ImaDraw *draws [[buffer(1)]],
                      device const uchar *xy [[buffer(2)]],
                      device const uchar *marks [[buffer(3)]],
                      constant float4 *frame [[buffer(4)]],
                      uint tid [[thread_position_in_grid]]) {
    render_command command(encoder.commands, tid);
    ImaDraw d = draws[tid];
    command.set_vertex_buffer(xy + d.xy_at, 0);
    command.set_vertex_buffer(marks + d.mark_at, 1);
    command.set_vertex_buffer(frame, 2);
    command.draw_primitives(primitive_type::triangle, d.first, d.count, d.instances,
                            d.base_instance);
}

kernel void ima_indexed(device const ImaCommands &encoder [[buffer(0)]],
                        device const ImaDraw *draws [[buffer(1)]],
                        device const uchar *xy [[buffer(2)]],
                        device const uchar *marks [[buffer(3)]],
                        constant float4 *frame [[buffer(4)]],
                        device const uint *indices [[buffer(5)]],
                        uint tid [[thread_position_in_grid]]) {
    render_command command(encoder.commands, tid);
    ImaDraw d = draws[tid];
    command.set_vertex_buffer(xy + d.xy_at, 0);
    command.set_vertex_buffer(marks + d.mark_at, 1);
    command.set_vertex_buffer(frame, 2);
    command.draw_indexed_primitives(primitive_type::triangle, d.count, indices + d.first,
                                    d.instances, d.base_vertex, d.base_instance);
}

kernel void ima_strided(device const ImaCommands &encoder [[buffer(0)]],
                        device const ImaDraw *draws [[buffer(1)]],
                        device const uchar *xy [[buffer(2)]],
                        device const uchar *marks [[buffer(3)]],
                        constant float4 *frame [[buffer(4)]],
                        uint tid [[thread_position_in_grid]]) {
    render_command command(encoder.commands, tid);
    ImaDraw d = draws[tid];
    command.set_vertex_buffer(xy + d.xy_at, d.stride, 0);
    command.set_vertex_buffer(marks + d.mark_at, 1);
    command.set_vertex_buffer(frame, 2);
    command.draw_primitives(primitive_type::triangle, d.first, d.count, d.instances,
                            d.base_instance);
}

vertex ImaOut ima_vertex(ImaIn in [[stage_in]], uint vid [[vertex_id]], uint iid [[instance_id]],
                         constant float4 &frame [[buffer(2)]]) {
    float2 at = in.xy + float2(0.0, float(iid) * frame.z);
    ImaOut out;
    out.position = float4(2.0 * at.x / frame.x - 1.0, 1.0 - 2.0 * at.y / frame.y, 0.0, 1.0);
    out.ids = uint4(in.mark, vid, iid, 1);
    return out;
}

fragment uint4 ima_fragment(ImaOut in [[stage_in]]) {
    return in.ids;
}
