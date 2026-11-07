// Owned MSL: kernels calling visible functions through declarations whose types differ from the
// definitions `linked_signature_functions.metal` gives them (`stitched_pair` is the function
// Metal's stitcher makes of `stitch_add(in0, in1)`, `float(float, float)`). The probe
// `linked_function_signatures.m` (the conformance crate's `probes/`) compiles this source and that
// one. Kernel `k_<case>` calls its function once on `in = {2, 3, 4, 5}` and stores the bits of
// the result to word 0 of `out`, then 0x5e47 to word 1 (`k_void_for_value` stores 7 to word 0
// after its call). Loop-free.
// metal-macos-min: 27.0
#include <metal_stdlib>
using namespace metal;

struct B8 { uchar c[8]; };
struct U6 { uint a[6]; };

[[stitchable]] float stitch_add(float a, float b) { return a + b; }

[[visible]] float stitched_pair(float a, float b, float c);
[[visible]] float extra(float a, float b, float c);
[[visible]] float extra_two(float a, float b, float c, float d);
[[visible]] float extra_pointer(float a, float b, device const float *p);
[[visible]] float missing(float a, float b);
[[visible]] float missing_unused(float a, float b);
[[visible]] float missing_pointer_unused(float a, float b);
[[visible]] float int_param(float a, float b, float c);
[[visible]] float half2_param(float a, float b);
[[visible]] float uint2_param(float2 v);
[[visible]] float half_param(float a, float b);
[[visible]] uint uint_for_half(float a, half b);
[[visible]] float long_param(float a, float b);
[[visible]] float float2_for_two(float2 v, float c);
[[visible]] float bool_param(float a, float b);
[[visible]] int int_return(float a, float b);
[[visible]] float float_for_half_return(float a, float b);
[[visible]] half half_for_float_return(float a, float b);
[[visible]] float value_for_void(float a, float b);
[[visible]] void void_for_value(float a, float b);
[[visible]] float value_for_pointer(device const float *p);
[[visible]] float pointer_for_value(ulong p);
[[visible]] float constant_for_device(device const float *p);
[[visible]] float device_for_constant(constant float *p);
[[visible]] float pair_for_floats(float a, float b);
[[visible]] float uints_for_floats(float a, float b);
[[visible]] float bytes_extra(B8 b, float x, float y);
[[visible]] float uint_array_for_float_array(U6 v);

#define K(name, expr)                                                                      \
    kernel void k_##name(device uint *out [[buffer(0)]],                                  \
                         device const float *in [[buffer(1)]]) {                          \
        out[0] = as_type<uint>(expr);                                                      \
        out[1] = 0x5e47;                                                                   \
    }

K(stitched_extra, stitched_pair(in[0], in[1], in[2]))
K(extra, extra(in[0], in[1], in[2]))
K(extra_two, extra_two(in[0], in[1], in[2], in[3]))
K(extra_pointer, extra_pointer(in[0], in[1], in))
K(missing, missing(in[0], in[1]))
K(missing_unused, missing_unused(in[0], in[1]))
K(missing_pointer_unused, missing_pointer_unused(in[0], in[1]))
K(int_param, int_param(in[0], in[1], in[2]))
K(half2_param, half2_param(in[0], in[1]))
K(uint2_param, uint2_param(float2(in[0], in[1])))
K(half_param, half_param(in[0], in[1]))
K(uint_for_half, uint_for_half(in[0], half(in[1])))
K(long_param, long_param(in[0], in[1]))
K(float2_for_two, float2_for_two(float2(in[0], in[1]), in[2]))
K(bool_param, bool_param(in[0], in[1]))
K(int_return, int_return(in[0], in[1]))
K(float_for_half_return, float_for_half_return(in[0], in[1]))
K(half_for_float_return, float(half_for_float_return(in[0], in[1])))
K(value_for_void, value_for_void(in[0], in[1]))
K(value_for_pointer, value_for_pointer(in))
K(pointer_for_value, pointer_for_value(0))
K(constant_for_device, constant_for_device(in))
K(pair_for_floats, pair_for_floats(in[0], in[1]))
K(uints_for_floats, uints_for_floats(in[0], in[1]))
K(bytes_extra, bytes_extra(B8{{1, 0, 0, 0, 0, 0, 0, 0}}, in[0], in[1]))
K(uint_array_for_float_array,
  uint_array_for_float_array(U6{{0x40000000u, 1, 2, 3, 4, 0x40400000u}}))

kernel void k_void_for_value(device uint *out [[buffer(0)]], device const float *in [[buffer(1)]]) {
    void_for_value(in[0], in[1]);
    out[0] = 7;
    out[1] = 0x5e47;
}

kernel void k_device_for_constant(device uint *out [[buffer(0)]], constant float *in [[buffer(1)]]) {
    out[0] = as_type<uint>(device_for_constant(in));
    out[1] = 0x5e47;
}
