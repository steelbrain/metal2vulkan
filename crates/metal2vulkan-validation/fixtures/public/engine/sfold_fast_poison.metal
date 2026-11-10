// Owned fixture, written for this project: an addition, subtraction, product and quotient of a NaN
// function constant and a value only the run knows, compiled with fast math, whose `nnan` makes
// each result poison once the constant is bound.
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
	float4 f[7];
	half4 h[4];
};

kernel void sfold(device const float4 *in [[buffer(0)]],
                  device Rows *out [[buffer(1)]],
                  uint gid [[thread_position_in_grid]]) {
	float4 x = in[gid];
	half4 y = half4(x);
	device Rows &o = out[gid];
	o.f[0] = x + k_qnan;
	o.f[1] = k_snan + x;
	o.f[2] = x - k_snan;
	o.f[3] = k_qnan - x;
	o.f[4] = x * k_snan;
	o.f[5] = x / k_qnan;
	o.f[6] = k_snan / x;
	o.h[0] = y + h_nan;
	o.h[1] = h_nan - y;
	o.h[2] = h_nan / y;
	o.h[3] = y * h_nan;
}
