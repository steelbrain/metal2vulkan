// Owned fixture: a function constant of type half that gates an output, the half form of
// fc_float_gate. Metal compiles `[[function_constant(s)]]` for a half `s` into a half predicate
// global that the static initializer writes from
// `air.normalize_function_constant_predicate.f16(s)`; the output is present where that predicate
// is non-zero. The cases over this entry ask Metal which values of `s` keep it: 1.5, +0.0, -0.0,
// a quiet NaN, and none.
//
// `fch_gated_sample_mask` always writes a colour and a sample mask of 0 to an output struct whose
// mask member the constant gates. Where the mask is an output the pixel is not covered and keeps
// the target's initial bytes; where it is not, the pixel takes the colour. The written values do
// not depend on `s`, so the target shows the presence itself.
//
// Precise maths, as fc_float_gate. Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

constant half s [[function_constant(0)]];

struct FchMasked {
    float4 color [[color(0)]];
    uint mask [[sample_mask, function_constant(s)]];
};

fragment FchMasked fch_gated_sample_mask()
{
    FchMasked out;
    out.color = float4(0.25, 0.5, 0.75, 1.0);
    out.mask = 0u;
    return out;
}
