// Owned fixture: what an indirect command buffer of indexed triangle draws (`icb_merged_indexed`)
// meets beside its encoding.
// - Thread `tid` of `icb_extras_reset` resets render command `params[0] + tid` of
//   `encoder.commands`, as a guest's initialization kernels do; a reset command draws nothing.
// - `icb_extras_back_vertex` is `icb_merged_indexed`'s `icb_indexed_vertex` reading its position
//   four elements before the vertex's own, `xy[int(vid) - 4]`: a command whose buffer 2 starts
//   inside a buffer reads the bytes before its address. `icb_extras_back_fragment` returns what it
//   carries, as `icb_indexed_fragment` does.
// - `icb_extras_wide_vertex` reads fourteen vertex buffers, at the even indices 0 to 26: vertex
//   `vid` lies at `xy[vid]` of buffer 0 (clip space) moved by the sum of the thirteen others'
//   words, which a test binds as zeros. A command binds each of the fourteen slots apart from the
//   next, so each is a run of its own. `icb_extras_fragment` returns 1 in every channel.
// Written for this repository; no third-party source.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

struct ResetCommands {
    command_buffer commands;
};

struct ExtrasOut {
    float4 position [[position]];
    uint4 ids [[flat]];
};

kernel void icb_extras_reset(device const ResetCommands &encoder [[buffer(0)]],
                             constant uint *params [[buffer(1)]],
                             uint tid [[thread_position_in_grid]]) {
    render_command command(encoder.commands, params[0] + tid);
    command.reset();
}

vertex ExtrasOut icb_extras_back_vertex(uint vid [[vertex_id]], uint iid [[instance_id]],
                                        constant uint &mark [[buffer(0)]],
                                        constant float4 &frame [[buffer(1)]],
                                        device const float2 *xy [[buffer(2)]]) {
    float2 at = xy[int(vid) - 4] + float2(0.0, float(iid) * frame.z);
    ExtrasOut out;
    out.position = float4(2.0 * at.x / frame.x - 1.0, 1.0 - 2.0 * at.y / frame.y, 0.0, 1.0);
    out.ids = uint4(mark, vid, iid, 0);
    return out;
}

fragment uint4 icb_extras_back_fragment(ExtrasOut in [[stage_in]], constant uint &tint [[buffer(0)]]) {
    return uint4(in.ids.x, in.ids.y, in.ids.z, tint);
}

vertex float4 icb_extras_wide_vertex(uint vid [[vertex_id]],
                                     device const float2 *xy [[buffer(0)]],
                                     constant float &w2 [[buffer(2)]],
                                     constant float &w4 [[buffer(4)]],
                                     constant float &w6 [[buffer(6)]],
                                     constant float &w8 [[buffer(8)]],
                                     constant float &w10 [[buffer(10)]],
                                     constant float &w12 [[buffer(12)]],
                                     constant float &w14 [[buffer(14)]],
                                     constant float &w16 [[buffer(16)]],
                                     constant float &w18 [[buffer(18)]],
                                     constant float &w20 [[buffer(20)]],
                                     constant float &w22 [[buffer(22)]],
                                     constant float &w24 [[buffer(24)]],
                                     constant float &w26 [[buffer(26)]]) {
    float moved = w2 + w4 + w6 + w8 + w10 + w12 + w14 + w16 + w18 + w20 + w22 + w24 + w26;
    return float4(xy[vid] + float2(moved, moved), 0.0, 1.0);
}

fragment uint4 icb_extras_fragment() {
    return uint4(1);
}
