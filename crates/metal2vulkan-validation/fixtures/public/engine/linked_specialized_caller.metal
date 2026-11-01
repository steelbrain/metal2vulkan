// Owned MSL: kernels calling visible functions they declare and do not define, as the probe
// `linked_function_specialized_name.m` calls them. `linked_specialized_caller` calls `custom_fn`,
// which the pipeline links as a function of `linked_specialized_functions.metal` made with the
// specialized name `custom_fn` (`MTLFunctionDescriptor.specializedName`), a name no function there
// has. `linked_specialized_base`, `linked_specialized_guarded` and `linked_specialized_free` call
// `base_fn`, `guarded_fn` and `plain_fn` by their own names. Loop-free.
#include <metal_stdlib>
using namespace metal;

[[visible]] float custom_fn(float x);
[[visible]] float base_fn(float x);
[[visible]] float guarded_fn(float x);
[[visible]] float plain_fn(float x);

kernel void linked_specialized_caller(device float *out [[buffer(0)]],
                                      uint tid [[thread_position_in_grid]]) {
    out[tid] = custom_fn(float(tid) + 5.0f);
}

kernel void linked_specialized_base(device float *out [[buffer(0)]],
                                    uint tid [[thread_position_in_grid]]) {
    out[tid] = base_fn(float(tid) + 5.0f);
}

kernel void linked_specialized_guarded(device float *out [[buffer(0)]],
                                       uint tid [[thread_position_in_grid]]) {
    out[tid] = guarded_fn(float(tid) + 5.0f);
}

kernel void linked_specialized_free(device float *out [[buffer(0)]],
                                    uint tid [[thread_position_in_grid]]) {
    out[tid] = plain_fn(float(tid) + 5.0f);
}
