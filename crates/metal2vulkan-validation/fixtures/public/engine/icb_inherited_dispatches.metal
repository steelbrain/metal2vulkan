// Owned fixture: compute commands a kernel encodes into an indirect command buffer that inherits
// the executing encoder's pipeline state and buffers, and the kernel they run. `cid_encode`:
// thread `tid` encodes command `tid`: a reset where `tid % 5` is 4, else a dispatch of
// `1 + tid % 4` threadgroups of 64 threads; it binds nothing. `cid_count`: thread `gid` adds
// `input[gid] * multiplier + bias` to `sums[0]` and 1 to `sums[1 + gid]`, each by a relaxed
// atomic, reading only `thread_position_in_grid`, so it does not observe its threadgroup
// partition; the executing encoder binds `input` (buffer 0), `sums` (buffer 1), `multiplier`
// (buffer 2) and `bias` (buffer 3).
// Written for this repository; no third-party source.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

struct CidCommands {
    command_buffer commands;
};

kernel void cid_encode(device const CidCommands &encoder [[buffer(0)]],
                       uint tid [[thread_position_in_grid]]) {
    compute_command command(encoder.commands, tid);
    if (tid % 5 == 4) {
        command.reset();
        return;
    }
    command.concurrent_dispatch_threadgroups(uint3(1 + tid % 4, 1, 1), uint3(64, 1, 1));
}

kernel void cid_count(device const uint *input [[buffer(0)]],
                      device atomic_uint *sums [[buffer(1)]],
                      constant uint &multiplier [[buffer(2)]],
                      constant uint &bias [[buffer(3)]],
                      uint gid [[thread_position_in_grid]]) {
    atomic_fetch_add_explicit(&sums[0], input[gid] * multiplier + bias, memory_order_relaxed);
    atomic_fetch_add_explicit(&sums[1 + gid], 1u, memory_order_relaxed);
}
