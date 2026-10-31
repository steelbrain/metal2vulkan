// Owned MSL: a kernel calling the visible function `custom_fn`, which it declares and does not
// define. The pipeline links a function of `linked_specialized_functions.metal` made with the
// specialized name `custom_fn` (`MTLFunctionDescriptor.specializedName`), a name no function
// there has. Loop-free.
#include <metal_stdlib>
using namespace metal;

[[visible]] float custom_fn(float x);

kernel void linked_specialized_caller(device float *out [[buffer(0)]],
                                      uint tid [[thread_position_in_grid]]) {
    out[tid] = custom_fn(float(tid) + 5.0f);
}
