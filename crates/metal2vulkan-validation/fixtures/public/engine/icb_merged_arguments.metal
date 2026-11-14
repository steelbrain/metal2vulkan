// Owned fixture: indexed draws of triangles that a kernel encodes into an indirect command buffer
// and a render pass executes, under the encoder's pipeline state, whose vertex function fetches
// `[[stage_in]]` attributes and whose fragment function reads an argument buffer of two textures
// and a constant pointer. Each kernel encodes command `tid` from `draws[tid]` (first index, index
// count, instance count, base vertex, base instance), resetting the command where the index count
// is 0: `ica_own` binds the command's own buffers (vertex buffer 0 the vertices' pixel positions,
// vertex buffer 1 the marks, vertex buffer 2 the frame, fragment buffer 0 the argument buffer and
// fragment buffer 1 its own word of `tints`) and `ica_inherited` none, its commands drawing with
// the encoder's. The draw is of the 32-bit indices of `indices` from its first index. The case's
// vertex descriptor places attribute 0 (a `float2` pixel position) in layout 0 and attribute 1 (a
// `uint` mark) in layout 1. Vertex `vid` of instance `iid` lies at its position moved down `iid`
// rows of `frame.z` pixels (`frame` is the target's width, height and row stride in pixels) and
// carries its mark, `vid` and `iid` flat to the fragment function, which returns the mark, the
// texel of each texture at the mark's column (each `RGBA8Unorm` red byte), the word the pointer
// reaches at the mark, and its tint plus `iid`. The vertex function writes no memory.
// Written for this repository; no third-party source.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

struct IcaCommands {
    command_buffer commands;
};

struct IcaDraw {
    uint first;
    uint count;
    uint instances;
    uint base_vertex;
    uint base_instance;
};

struct IcaIn {
    float2 xy [[attribute(0)]];
    uint mark [[attribute(1)]];
};

struct IcaOut {
    float4 position [[position]];
    uint4 ids [[flat]];
};

struct IcaArguments {
    texture2d<float> first [[id(0)]];
    texture2d<float> second [[id(1)]];
    constant uint *words [[id(2)]];
};

kernel void ica_own(device const IcaCommands &encoder [[buffer(0)]],
                    device const IcaDraw *draws [[buffer(1)]],
                    device const float2 *xy [[buffer(2)]],
                    device const uint *marks [[buffer(3)]],
                    constant float4 *frame [[buffer(4)]],
                    constant IcaArguments *arguments [[buffer(5)]],
                    constant uint *tints [[buffer(6)]],
                    device const uint *indices [[buffer(7)]],
                    uint tid [[thread_position_in_grid]]) {
    render_command command(encoder.commands, tid);
    IcaDraw d = draws[tid];
    if (d.count == 0) {
        command.reset();
        return;
    }
    command.set_vertex_buffer(xy, 0);
    command.set_vertex_buffer(marks, 1);
    command.set_vertex_buffer(frame, 2);
    command.set_fragment_buffer(arguments, 0);
    command.set_fragment_buffer(tints + tid, 1);
    command.draw_indexed_primitives(primitive_type::triangle, d.count, indices + d.first,
                                    d.instances, d.base_vertex, d.base_instance);
}

kernel void ica_inherited(device const IcaCommands &encoder [[buffer(0)]],
                          device const IcaDraw *draws [[buffer(1)]],
                          device const uint *indices [[buffer(7)]],
                          uint tid [[thread_position_in_grid]]) {
    render_command command(encoder.commands, tid);
    IcaDraw d = draws[tid];
    if (d.count == 0) {
        command.reset();
        return;
    }
    command.draw_indexed_primitives(primitive_type::triangle, d.count, indices + d.first,
                                    d.instances, d.base_vertex, d.base_instance);
}

vertex IcaOut ica_vertex(IcaIn in [[stage_in]], uint vid [[vertex_id]], uint iid [[instance_id]],
                         constant float4 &frame [[buffer(2)]]) {
    float2 at = in.xy + float2(0.0, float(iid) * frame.z);
    IcaOut out;
    out.position = float4(2.0 * at.x / frame.x - 1.0, 1.0 - 2.0 * at.y / frame.y, 0.0, 1.0);
    out.ids = uint4(in.mark, vid, iid, 0);
    return out;
}

fragment uint4 ica_fragment(IcaOut in [[stage_in]],
                            constant IcaArguments &arguments [[buffer(0)]],
                            constant uint &tint [[buffer(1)]]) {
    uint mark = in.ids.x;
    uint2 texel = uint2(mark % 4, 0);
    uint first = uint(round(arguments.first.read(texel).x * 255.0));
    uint second = uint(round(arguments.second.read(texel).x * 255.0));
    return uint4(mark, first | (second << 8), arguments.words[mark % 4], tint + in.ids.z);
}
