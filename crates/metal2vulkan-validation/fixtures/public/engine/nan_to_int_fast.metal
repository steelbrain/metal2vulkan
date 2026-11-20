// Owned fixture (NaN model): `nan_to_int`'s program under the default fast math, which folds
// the `+ 0.0` away, so each conversion reads the loaded value and the entries hold no float
// operation without fast-math flags. Invocation `i` reads the float whose bits are `in[2i]` and
// the half in the low 16 bits of `in[2i + 1]` and answers four words: int(f), uint(f), int(h) and
// uint(h), the signed ones as their bits. The kernel writes them to `out[i]`; the fragment
// function returns them for the texel at x = i of a one-row RGBA32Uint target, which the vertex
// function's full-screen triangle (from [[vertex_id]], no vertex buffer) covers. Loop-free.
#include <metal_stdlib>
using namespace metal;

static uint4 nan_to_int_fast_words(device const uint *in, uint i)
{
    float f = as_type<float>(in[2 * i]) + 0.0f;
    half h = as_type<half>(ushort(in[2 * i + 1])) + 0.0h;
    return uint4(as_type<uint>(int(f)), uint(f), as_type<uint>(int(h)), uint(h));
}

kernel void nan_to_int_fast_kernel(
    device const uint *in [[buffer(0)]],
    device uint4 *out [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    out[tid] = nan_to_int_fast_words(in, tid);
}

struct NanToIntFastVertex {
    float4 position [[position]];
};

// The full-screen triangle (-1, -1), (3, -1), (-1, 3), computed from the vertex's index.
vertex NanToIntFastVertex nan_to_int_fast_vertex(uint id [[vertex_id]])
{
    float2 a = float2(float(id & 1u), float(id >> 1));
    return NanToIntFastVertex { float4(4.0 * a - 1.0, 0.0, 1.0) };
}

fragment uint4 nan_to_int_fast_fragment(
    NanToIntFastVertex input [[stage_in]],
    device const uint *in [[buffer(0)]])
{
    return nan_to_int_fast_words(in, uint(input.position.x));
}
