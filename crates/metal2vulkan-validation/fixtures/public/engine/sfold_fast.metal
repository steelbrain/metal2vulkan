// Owned fixture, written for this project: float operations whose operand is a function constant,
// so what each folds to is known only once the constants are bound, compiled with fast math. The
// case supplies both zeros, a quiet NaN with its sign and a payload, a signalling NaN and a
// signalling `half` NaN: a product by a zero, a fused multiply-add with a NaN operand beside values
// only the run knows, and operations whose every operand is a constant, NaNs in each order.
// `sfold_precise` holds these rows and more without fast math; `sfold_fast_poison` holds the
// operations with a NaN constant beside a run value under fast math.
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
	float4 f[11];
	half4 h[3];
};

kernel void sfold(device const float4 *in [[buffer(0)]],
                  device Rows *out [[buffer(1)]],
                  uint gid [[thread_position_in_grid]]) {
	float4 x = in[gid];
	half4 y = half4(x);
	device Rows &o = out[gid];
	o.f[0] = x * k_zero;
	o.f[1] = k_negzero * x;
	o.f[2] = x * k_negzero;
	o.f[3] = fma(x, x, float4(k_qnan));
	o.f[4] = fma(float4(k_snan), x, x);
	o.f[5] = float4(k_qnan) + float4(k_snan);
	o.f[6] = float4(k_snan) * float4(k_qnan);
	o.f[7] = float4(k_qnan) * float4(k_zero);
	o.f[8] = float4(k_snan) - float4(k_qnan);
	o.f[9] = float4(k_qnan) / float4(k_snan);
	o.f[10] = fma(float4(k_snan), float4(k_qnan), float4(k_qnan));
	o.h[0] = y * h_zero;
	o.h[1] = h_negzero * y;
	o.h[2] = fma(y, y, half4(h_nan));
}
