// Owned fixture (NaN model): MSL's float-to-integer conversions, which section 8.6 of the MSL
// specification says take a NaN to 0, in a kernel and in a fragment function. Invocation `i`
// reads the float whose bits are `in[2i]` and the half in the low 16 bits of `in[2i + 1]` and
// answers four words: int(f), uint(f), int(h) and uint(h), the signed ones as their bits. The
// kernel writes them to `out[i]`; the fragment function returns them for the texel at x = i of a
// one-row RGBA32Uint target, which the vertex function's full-screen triangle (from
// [[vertex_id]], no vertex buffer) covers. Each operand is the loaded value plus 0.0, which a
// compile without fast math keeps (-0 + 0 is +0), so every entry holds a float operation without
// fast-math flags. `nan_to_int_fast` is the same program under the default fast math. Compiled
// without fast math. Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

static uint4 nan_to_int_words(device const uint *in, uint i)
{
    float f = as_type<float>(in[2 * i]) + 0.0f;
    half h = as_type<half>(ushort(in[2 * i + 1])) + 0.0h;
    return uint4(as_type<uint>(int(f)), uint(f), as_type<uint>(int(h)), uint(h));
}

kernel void nan_to_int_kernel(
    device const uint *in [[buffer(0)]],
    device uint4 *out [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    out[tid] = nan_to_int_words(in, tid);
}

struct NanToIntVertex {
    float4 position [[position]];
};

// The full-screen triangle (-1, -1), (3, -1), (-1, 3), computed from the vertex's index.
vertex NanToIntVertex nan_to_int_vertex(uint id [[vertex_id]])
{
    float2 a = float2(float(id & 1u), float(id >> 1));
    return NanToIntVertex { float4(4.0 * a - 1.0, 0.0, 1.0) };
}

fragment uint4 nan_to_int_fragment(
    NanToIntVertex input [[stage_in]],
    device const uint *in [[buffer(0)]])
{
    return nan_to_int_words(in, uint(input.position.x));
}
