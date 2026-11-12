// Owned MSL: the visible function `private_reach_tiles.metal`'s kernels call, `postfix(x, w) =
// x * 2 + w`, which their pipelines link. The kernel only gives the module an entry point.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

[[visible]] float postfix(float x, uint w) { return x * 2.0f + float(w); }

kernel void private_reach_functions(device float *out [[buffer(0)]],
                                    uint tid [[thread_position_in_grid]]) {
    out[tid] = float(tid);
}
