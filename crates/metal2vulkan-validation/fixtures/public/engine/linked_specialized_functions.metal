// Owned MSL: visible functions of type `float(float)` in a module that declares a function
// constant, which a pipeline links under a specialized name, with the constant's value or without
// one. `base_fn` reads `scale` (the callee of the probe `linked_function_specialized_name.m`);
// `derived_fn` reads `twice`, a program-scope constant computed from `scale` in the module's
// static initializer; `guarded_fn` reads `scale` only where `is_function_constant_defined` says it
// has a value; `plain_fn` reads no constant. The kernel only gives the module an entry point.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

constant float scale [[function_constant(0)]];
constant float twice = scale * 2.0f;

[[visible]] float base_fn(float x) { return x * scale + 1.0f; }

[[visible]] float derived_fn(float x) { return x * twice; }

[[visible]] float guarded_fn(float x) {
    return is_function_constant_defined(scale) ? x * scale : x - 1.0f;
}

[[visible]] float plain_fn(float x) { return x + 2.0f; }

kernel void linked_specialized_functions(device float *out [[buffer(0)]],
                                         uint tid [[thread_position_in_grid]]) {
    out[tid] = float(tid);
}
