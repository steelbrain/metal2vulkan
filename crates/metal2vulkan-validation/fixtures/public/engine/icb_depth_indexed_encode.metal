// Owned fixture: a GPU-driven depth-only pass, the shape of a culling renderer's depth prepass.
// `dio_reset` sets the execution range at buffer 0 to location 0, length 0. Thread `tid` of
// `dio_encode` keeps chunk `tid` where bit `tid` of `visible` is set: it takes the next command
// by an atomic increment of the range's length (buffer 4, the range's second word) and encodes
// there, onto the executing encoder's pipeline state, which the commands inherit, its own vertex
// buffers (the `positions` buffer at 0, the address of the constant `camera` argument at 1) and
// an indexed triangle-list draw of 6 16-bit indices from 6 * tid, once. `dio_vertex` reads its
// position as attribute 0 (a float3 from buffer 0) and places it at depth
// position.z * camera.depth_scale + camera.depth_offset. There is no fragment function. Written
// for this repository; no third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Encoder {
    command_buffer commands;
};

struct Camera {
    float depth_scale;
    float depth_offset;
};

kernel void dio_reset(device uint *range [[buffer(0)]])
{
    range[0] = 0;
    range[1] = 0;
}

kernel void dio_encode(device const Encoder &encoder [[buffer(0)]],
                       const device float *positions [[buffer(1)]],
                       const device ushort *indices [[buffer(2)]],
                       constant Camera &camera [[buffer(3)]],
                       device atomic_uint *length [[buffer(4)]],
                       constant uint &visible [[buffer(5)]],
                       uint tid [[thread_position_in_grid]])
{
    if (((visible >> tid) & 1) == 0) {
        return;
    }
    uint slot = atomic_fetch_add_explicit(length, 1, memory_order_relaxed);
    render_command command(encoder.commands, slot);
    command.set_vertex_buffer(positions, 0);
    command.set_vertex_buffer(&camera, 1);
    command.draw_indexed_primitives(primitive_type::triangle, 6, indices + 6 * tid, 1);
}

struct DioIn {
    float3 position [[attribute(0)]];
};

vertex float4 dio_vertex(DioIn in [[stage_in]], constant Camera &camera [[buffer(1)]])
{
    return float4(in.position.xy, in.position.z * camera.depth_scale + camera.depth_offset, 1.0);
}
