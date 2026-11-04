// Owned fixture: kernels that encode command 0 of an indirect command buffer named by their
// argument buffer, and the functions that command runs. `irw_encode_draw` sets a render command's
// state (taken from the argument buffer), its vertex buffer 0 (the corners at buffer 1) and a draw
// of one triangle; `irw_vertex` places the corners and `irw_fragment` colours green.
// `irw_encode_point` writes a render command of a buffer that inherits the encoder's state: its
// vertex buffer 0 (the point at buffer 1) and a draw of one point, which `irw_point_vertex` places,
// 4 pixels wide. `irw_encode_dispatch` sets a compute command's state (taken from the argument
// buffer), its kernel buffer 0 (the buffer at buffer 1) and a dispatch of one threadgroup of 4
// threads; `irw_mark` writes 0x100 plus its thread's index at that index. Written for this
// repository; no third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct RenderWrite {
    command_buffer commands [[id(0)]];
    render_pipeline_state state [[id(1)]];
};

kernel void irw_encode_draw(device const RenderWrite &table [[buffer(0)]],
                            constant float4 *corners [[buffer(1)]])
{
    render_command command(table.commands, 0);
    command.set_render_pipeline_state(table.state);
    command.set_vertex_buffer(corners, 0);
    command.draw_primitives(primitive_type::triangle, 0, 3, 1, 0);
}

struct Corner {
    float4 position [[position]];
};

vertex Corner irw_vertex(uint vid [[vertex_id]], constant float4 *corners [[buffer(0)]])
{
    Corner corner;
    corner.position = corners[vid];
    return corner;
}

struct InheritWrite {
    command_buffer commands [[id(0)]];
};

kernel void irw_encode_point(device const InheritWrite &table [[buffer(0)]],
                             constant float4 *point [[buffer(1)]])
{
    render_command command(table.commands, 0);
    command.set_vertex_buffer(point, 0);
    command.draw_primitives(primitive_type::point, 0, 1, 1, 0);
}

struct Point {
    float4 position [[position]];
    float size [[point_size]];
};

vertex Point irw_point_vertex(uint vid [[vertex_id]], constant float4 *corners [[buffer(0)]])
{
    Point point;
    point.position = corners[vid];
    point.size = 4.0;
    return point;
}

fragment half4 irw_fragment()
{
    return half4(0.0, 1.0, 0.0, 1.0);
}

struct ComputeWrite {
    command_buffer commands [[id(0)]];
    compute_pipeline_state state [[id(1)]];
};

kernel void irw_encode_dispatch(device const ComputeWrite &table [[buffer(0)]],
                                device uint *out [[buffer(1)]])
{
    compute_command command(table.commands, 0);
    command.set_compute_pipeline_state(table.state);
    command.set_kernel_buffer(out, 0);
    command.concurrent_dispatch_threadgroups(uint3(1, 1, 1), uint3(4, 1, 1));
}

kernel void irw_mark(device uint *out [[buffer(0)]], uint tid [[thread_position_in_grid]])
{
    out[tid] = 0x100u + tid;
}
