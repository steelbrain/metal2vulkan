// Owned fixture: the tile function our Metal probe wrote an indexed indirect draw's arguments or
// indices with, in the form of the vertex writer `icb_render_replay_template`: thread (0, 0) of
// each tile writes `params[0] *
// (k + 1) + params[1]` to word `k` of `out`, for `k` 0 and 1. A pass binds `params` and `out` as
// tile buffers 0 and 1 and dispatches one thread per tile. Written for this repository; no
// third-party source. Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

kernel void rrs_tile_writer(
    constant uint *params [[buffer(0)]],
    device uint *out [[buffer(1)]],
    ushort2 tid [[thread_position_in_threadgroup]])
{
    if (tid.x != 0 || tid.y != 0) {
        return;
    }
    out[0] = params[0] + params[1];
    out[1] = params[0] * 2u + params[1];
}
