// Owned fixture, written for this project: `sfold_fast` compiled without fast math, so no float
// operation carries a fast-math flag. A product by a zero keeps the sign IEEE arithmetic gives it,
// and a NaN operand's answer is what Metal makes of the same operations with no flag to license
// a rewrite.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

constant float k_zero [[function_constant(0)]];
constant float k_negzero [[function_constant(1)]];
constant float k_qnan [[function_constant(2)]];
constant float k_snan [[function_constant(3)]];
constant half h_zero [[function_constant(4)]];
constant half h_negzero [[function_constant(5)]];
constant half h_nan [[function_constant(6)]];

struct Rows {
	float4 f[15];
	half4 h[7];
};

kernel void sfold(device const float4 *in [[buffer(0)]],
                  device Rows *out [[buffer(1)]],
                  uint gid [[thread_position_in_grid]]) {
	float4 x = in[gid];
	device Rows &o = out[gid];
	o.f[0] = x * k_zero;
	o.f[1] = k_negzero * x;
	o.f[2] = x * k_negzero;
	o.f[3] = x + k_qnan;
	o.f[4] = k_snan + x;
	o.f[5] = x - k_snan;
	o.f[6] = k_qnan - x;
	o.f[7] = x * k_snan;
	o.f[8] = x / k_qnan;
	o.f[9] = k_snan / x;
	o.f[10] = fma(x, x, float4(k_qnan));
	o.f[11] = fma(float4(k_snan), x, x);
	o.f[12] = float4(k_qnan) + float4(k_snan);
	o.f[13] = float4(k_snan) * float4(k_qnan);
	o.f[14] = float4(k_qnan) * float4(k_zero);
	half4 y = half4(x);
	o.h[0] = y * h_zero;
	o.h[1] = h_negzero * y;
	o.h[2] = y + h_nan;
	o.h[3] = h_nan - y;
	o.h[4] = h_nan / y;
	o.h[5] = fma(y, y, half4(h_nan));
	o.h[6] = y * h_nan;
}
