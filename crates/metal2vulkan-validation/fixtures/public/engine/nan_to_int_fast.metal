// Owned fixture (NaN model): `nan_to_int`'s program under the default fast math, which folds
// the `+ 0.0` away, so each conversion reads the loaded value and the entries hold no float
// operation without fast-math flags.
// Invocation `x` takes input `i = x / 5`: the float whose bits are `in[2i]` and the half in the
// low 16 bits of `in[2i + 1]`. It answers part `x % 5` of the input's twenty words, which convert
// each source to every integer width, signed and unsigned, in this order: char, uchar, short,
// ushort, int, uint (a word each, a signed one sign-extended), long and ulong (two words each, the
// low first), from the float, then the same from the half. The kernel writes `out[x]`; the
// fragment function returns it for the texel at x of a one-row RGBA32Uint target, which the
// vertex function's full-screen triangle (from [[vertex_id]], no vertex buffer) covers. Loop-free.
#include <metal_stdlib>
using namespace metal;

static uint4 nan_to_int_fast_part(device const uint *in, uint x)
{
    uint i = x / 5;
    float f = as_type<float>(in[2 * i]) + 0.0f;
    half h = as_type<half>(ushort(in[2 * i + 1])) + 0.0h;
    switch (x % 5) {
    case 0:
        return uint4(as_type<uint>(int(char(f))), uint(uchar(f)), as_type<uint>(int(short(f))),
                     uint(ushort(f)));
    case 1:
        return uint4(as_type<uint>(int(f)), uint(f), as_type<uint2>(long(f)));
    case 2:
        return uint4(as_type<uint2>(ulong(f)), as_type<uint>(int(char(h))), uint(uchar(h)));
    case 3:
        return uint4(as_type<uint>(int(short(h))), uint(ushort(h)), as_type<uint>(int(h)), uint(h));
    default:
        return uint4(as_type<uint2>(long(h)), as_type<uint2>(ulong(h)));
    }
}

kernel void nan_to_int_fast_kernel(
    device const uint *in [[buffer(0)]],
    device uint4 *out [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    out[tid] = nan_to_int_fast_part(in, tid);
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
    return nan_to_int_fast_part(in, uint(input.position.x));
}
