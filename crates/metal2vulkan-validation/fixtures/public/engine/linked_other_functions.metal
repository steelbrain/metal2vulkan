// Owned MSL: visible functions of names that `linked_visible_functions.metal` and
// `visible_table_functions.metal` define, with other bodies, so that a pipeline can link two
// different functions of one name: one in `functions` and one in `binaryFunctions`. The kernel
// only gives the module an entry point. Loop-free.
#include <metal_stdlib>
using namespace metal;

[[visible]] uint linked_scale(uint x) { return x + 100u; }

[[visible]] uint table_add(uint a, uint b) { return a + b + 100u; }

kernel void linked_other_functions(device uint *out [[buffer(0)]],
                                   uint tid [[thread_position_in_grid]]) {
    out[tid] = tid;
}
