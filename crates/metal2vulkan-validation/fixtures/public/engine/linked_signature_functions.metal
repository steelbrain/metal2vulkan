// Owned MSL: the visible functions `linked_signature_caller.metal` calls, each defined at another
// type than the caller declares it with: fewer or more parameters, parameters and returns of
// another type of the same or another width, a value for a pointer and a pointer for a value, a
// pointer of another address space, and aggregates. The probe `linked_function_signatures.m` (the
// conformance crate's `probes/`) links each one into its kernel. The kernel only gives the module
// an entry point. Loop-free.
// metal-macos-min: 27.0
#include <metal_stdlib>
using namespace metal;

struct Pair { float a; float b; };
struct UPair { uint a; uint b; };
struct B8 { uchar c[8]; };
struct F6 { float a[6]; };

[[visible]] float extra(float a, float b) { return a + b; }
[[visible]] float extra_two(float a) { return a; }
[[visible]] float extra_pointer(float a, float b) { return a + b; }
[[visible]] float missing(float a, float b, float c) { return a + b + c; }
[[visible]] float missing_unused(float a, float b, float c) { return a + b; }
[[visible]] float missing_pointer_unused(float a, float b, device const float *p) { return a + b; }
[[visible]] float int_param(float a, int b, float c) { return a + float(b); }
[[visible]] float half2_param(float a, half2 b) { return a + float(b.x) + float(b.y); }
[[visible]] float uint2_param(uint2 v) { return float(v.x) + float(v.y); }
[[visible]] float half_param(float a, half b) { return a + float(b); }
[[visible]] uint uint_for_half(float a, uint b) { return b; }
[[visible]] float long_param(float a, long b) { return a + float(b); }
[[visible]] float float2_for_two(float a, float b) { return a + 10.0f * b; }
[[visible]] float bool_param(bool a, float b) { return a ? b : -b; }
[[visible]] float int_return(float a, float b) { return a + b; }
[[visible]] half float_for_half_return(float a, float b) { return half(a + b); }
[[visible]] float half_for_float_return(float a, float b) { return a + b; }
[[visible]] void value_for_void(float a, float b) { }
[[visible]] float void_for_value(float a, float b) { return a + b; }
[[visible]] float value_for_pointer(float x) { return x; }
[[visible]] float pointer_for_value(device const float *p) { return p == nullptr ? 1.0f : 2.0f; }
[[visible]] float constant_for_device(constant float *p) { return p[0] + p[1]; }
[[visible]] float device_for_constant(device const float *p) { return p[0] + p[1]; }
[[visible]] float pair_for_floats(Pair p) { return p.a + p.b; }
[[visible]] float uints_for_floats(UPair p) { return float(p.a) + float(p.b); }
[[visible]] float bytes_extra(B8 b, float x) { return x + float(b.c[0]); }
[[visible]] float uint_array_for_float_array(F6 v) { return v.a[0] + v.a[5]; }

kernel void linked_signature_functions(device uint *out [[buffer(0)]],
                                       uint tid [[thread_position_in_grid]]) {
    out[tid] = tid;
}
