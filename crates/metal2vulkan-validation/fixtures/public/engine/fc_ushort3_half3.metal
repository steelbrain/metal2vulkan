// Owned fixture: two `ushort3` function constants read as `half3` through `as_type`. The compiler
// folds each `as_type` into a `<3 x half>` load of the constant's `<3 x i16>` global, a read of
// the global through another type than its own, which the memory plan holds in the constant blob.
// Thread `t` stores, as floats, lane `t % 3` of each and their `half` sum. The conformance case
// `function-constants-read-as-half3` runs it with values every one of whose sums is exact.
#include <metal_stdlib>
using namespace metal;

constant ushort3 fch_a [[function_constant(0)]];
constant ushort3 fch_b [[function_constant(1)]];

kernel void fc_ushort3_half3(device float4 *out [[buffer(0)]],
                             uint t [[thread_position_in_grid]]) {
    half3 a = as_type<half3>(fch_a);
    half3 b = as_type<half3>(fch_b);
    uint lane = t % 3;
    out[t] = float4(float(a[lane]), float(b[lane]), float(a[lane] + b[lane]), float(t));
}
