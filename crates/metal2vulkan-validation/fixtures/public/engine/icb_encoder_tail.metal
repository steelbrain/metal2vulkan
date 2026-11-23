// Owned fixture: an argument buffer whose first member is an indirect command buffer, in a buffer
// that goes on past the structure. Each function indexes the words after the structure by a
// signed run-time index, so the translation cannot prove an access stays off the command buffer's
// field, which a negative index reaches. Metal lays a Tier 2 argument buffer out as the equivalent
// C structure: the field is the eight bytes of the command buffer's `gpuResourceID`. Written for
// this repository; no third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Encoder {
    command_buffer commands;
    uint counts[2];
};

// A rasterization-disabled vertex function: command `iid` binds as its vertex buffer 0 and its
// fragment buffer 0 the buffers whose addresses the words past the encoder hold at entry
// iid * counts[0] and the entry after it.
vertex void tail_bind(device Encoder &encoder [[buffer(0)]],
                      uint iid [[instance_id]])
{
    device float *const device *tail =
        reinterpret_cast<device float *const device *>(&encoder + 1);
    int at = int(iid) * int(encoder.counts[0]);
    render_command command(encoder.commands, iid);
    command.set_vertex_buffer(tail[at], 0);
    command.set_fragment_buffer(tail[at + 1], 0);
}

// Thread `tid` writes, at entry tid * counts[0] of the words past the encoder, the entry after it
// plus counts[1].
kernel void tail_count(device Encoder &encoder [[buffer(0)]],
                       uint tid [[thread_position_in_grid]])
{
    device uint *tail = reinterpret_cast<device uint *>(&encoder + 1);
    int at = int(tid) * int(encoder.counts[0]);
    tail[at] = tail[at + 1] + encoder.counts[1];
}
