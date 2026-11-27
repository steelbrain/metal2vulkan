// Owned fixture: a function constant of type float that gates an argument and an output. Metal
// compiles `[[function_constant(s)]]` for a float `s` into a float predicate global that the
// static initializer writes from `air.normalize_function_constant_predicate.f32(s)`; the argument
// or output is present where that predicate is non-zero. The cases over these entries ask Metal
// which values of `s` keep it: 1.5, 0.0, -0.0, a quiet NaN, and none.
//
// - `fcf_gated_argument` reads the gated buffer exactly where `s` is defined and true as C reads
//   a float (`s != 0`, so a NaN is true and -0.0 false), and writes a sentinel elsewhere. A
//   presence that drops the argument where the body reads it gives a wrong value there.
// - `fcf_gated_sample_mask` always writes a colour and a sample mask of 0 to an output struct
//   whose mask member the same constant gates. Where the mask is an output the pixel is not
//   covered and keeps the target's initial bytes; where it is not, the pixel takes the colour.
//   The written values do not depend on `s`, so the target shows the presence itself, for every
//   value, including those where the kernel's body reads nothing.
//
// Precise maths, so a NaN compared with 0 is a defined question. Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

constant float s [[function_constant(0)]];

kernel void fcf_gated_argument(device float *out [[buffer(0)]],
                               device const float *gated [[buffer(1), function_constant(s)]],
                               uint i [[thread_position_in_grid]])
{
    out[i] = (is_function_constant_defined(s) && s) ? gated[i] : -7.0f;
}

struct FcfMasked {
    float4 color [[color(0)]];
    uint mask [[sample_mask, function_constant(s)]];
};

fragment FcfMasked fcf_gated_sample_mask()
{
    FcfMasked out;
    out.color = float4(0.25, 0.5, 0.75, 1.0);
    out.mask = 0u;
    return out;
}
