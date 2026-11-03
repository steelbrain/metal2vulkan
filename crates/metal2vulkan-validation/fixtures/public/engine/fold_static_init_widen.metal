// Owned fixture, written for this project: a program-scope constant built from a `uchar4` function
// constant by a vector conversion, `uint4(kOps)`, which the AIR's static initializer computes with
// `air.convert.u.v4i32.u.v4i8`, and another from a `uchar` one. Specialization evaluates both
// conversions lane by lane. Thread i writes `kWide + kSelWide + i`.
#include <metal_stdlib>
using namespace metal;
constant uchar4 kOps [[function_constant(0)]];
constant uchar kSel [[function_constant(1)]];
constant uint4 kWide = uint4(kOps);
constant uint kSelWide = kSel;
kernel void k(device uint4 *out [[buffer(0)]], uint gid [[thread_position_in_grid]])
{
	out[gid] = kWide + kSelWide + gid;
}
