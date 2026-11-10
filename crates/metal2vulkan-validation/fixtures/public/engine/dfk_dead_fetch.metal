// Owned fixture: a framebuffer read (a `[[color(0)]]` input) that the function's constants
// decide, in three shapes, each under the compiler's default fast math, each returning one `half4`
// to color 0 (the shape of a composite whose pipeline blends color 0, `BGRA8Unorm`, with One over
// OneMinusSourceAlpha):
//
// - dfk_branch: the read sits in the arm of a branch on `dfk_reads` (function constant 0); with
//   it false the arm is dead and nothing reads the attachment.
// - dfk_scaled: the read's alpha is scaled by `dfk_weight` (function constant 1, a `half`) and
//   added to the loaded alpha; with it zero the fast-math product is `+0` whatever the read holds,
//   and nothing else reads the attachment.
// - dfk_gated: the input itself is gated by `dfk_reads` and read in the arm of a branch on it;
//   with it false the function declares no color input.
//
// With `dfk_reads` true (and `dfk_weight` not zero) each function reads the attachment.
//
// dfk_vertex draws one fullscreen triangle. Loop-free.
#include <metal_stdlib>
using namespace metal;

constant bool dfk_reads [[function_constant(0)]];
constant half dfk_weight [[function_constant(1)]];

struct DfkVertex {
    float4 position [[position]];
};

vertex DfkVertex dfk_vertex(uint vid [[vertex_id]])
{
    float2 p = float2(float((vid << 1) & 2u), float(vid & 2u));
    DfkVertex o;
    o.position = float4(p * 2.0f - 1.0f, 0.0f, 1.0f);
    return o;
}

fragment half4 dfk_branch(DfkVertex in [[stage_in]], half4 dst [[color(0)]],
                          const device half4 *src [[buffer(0)]])
{
    uint i = uint(in.position.y) * 64u + uint(in.position.x);
    half4 c = src[i];
    if (dfk_reads) {
        c = dst.a == 0.0h ? c : c * dst;
    }
    return c;
}

fragment half4 dfk_scaled(DfkVertex in [[stage_in]], half4 dst [[color(0)]],
                          const device half4 *src [[buffer(0)]])
{
    uint i = uint(in.position.y) * 64u + uint(in.position.x);
    half4 c = src[i];
    c.a = c.a + dst.a * dfk_weight;
    return c;
}

fragment half4 dfk_gated(DfkVertex in [[stage_in]],
                         half4 dst [[color(0), function_constant(dfk_reads)]],
                         const device half4 *src [[buffer(0)]])
{
    uint i = uint(in.position.y) * 64u + uint(in.position.x);
    half4 c = src[i];
    if (dfk_reads) {
        c = dst.a == 0.0h ? c : c * dst;
    }
    return c;
}
