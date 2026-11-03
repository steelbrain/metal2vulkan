// Owned fixture: the LLVM bit-manipulation intrinsics Metal's compiler emits from MSL. MSL's own
// integer functions (reverse_bits, popcount, clz, ctz, rotate) compile to `air.*` calls; these
// `llvm.*` ones come from clang's builtins and from the shift-and-mask idioms LLVM recognizes:
// llvm.bitreverse (i8, i16, i32, i64, <4 x i32>), llvm.bswap (i16, i32, i64, <4 x i32>),
// llvm.fshl and llvm.fshr (rotations of i8, i16, i32, i64; fshl of <4 x i32>), llvm.ctpop (i32,
// i64), and llvm.ctlz and llvm.cttz (i16 with the zero-poison flag, i32 and i64 with it both
// ways). Each thread t reads one 64-bit word x[t] and one amount s[t], and writes words
// 48t .. 48t + 42; words 43 to 47 of its block keep their fill. A flag-true count reads only a
// non-zero operand. Written for this repository; no third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

// LLVM recognizes these as llvm.bitreverse, llvm.bswap and llvm.fshl of their type.
template <typename T> static T rev32(T b) {
    b = (b << 16u) | (b >> 16u);
    b = ((b & 0x55555555u) << 1u) | ((b & 0xAAAAAAAAu) >> 1u);
    b = ((b & 0x33333333u) << 2u) | ((b & 0xCCCCCCCCu) >> 2u);
    b = ((b & 0x0F0F0F0Fu) << 4u) | ((b & 0xF0F0F0F0u) >> 4u);
    b = ((b & 0x00FF00FFu) << 8u) | ((b & 0xFF00FF00u) >> 8u);
    return b;
}
template <typename T> static T swap32(T x) {
    return (x >> 24) | ((x >> 8) & 0xFF00u) | ((x << 8) & 0xFF0000u) | (x << 24);
}
template <typename T> static T rotl32(T x, T s) { return (x << (s & 31u)) | (x >> ((32u - s) & 31u)); }
static ulong rotl64(ulong x, ulong s) { return (x << (s & 63)) | (x >> ((64 - s) & 63)); }

kernel void bit_ops(device const ulong *x [[buffer(0)]],
                    device const uint *s [[buffer(1)]],
                    device uint *o [[buffer(2)]],
                    uint t [[thread_position_in_grid]]) {
    ulong w = x[t];
    uint a = uint(w), b = uint(w >> 32), n = s[t];
    ushort h = ushort(w);
    uchar c = uchar(w);
    device uint *r = o + t * 48;
    r[0] = __builtin_bitreverse8(c);
    r[1] = __builtin_bitreverse16(h);
    r[2] = __builtin_bitreverse32(a);
    ulong rw = __builtin_bitreverse64(w);
    r[3] = uint(rw);
    r[4] = uint(rw >> 32);
    r[5] = __builtin_bswap16(h);
    r[6] = __builtin_bswap32(a);
    ulong sw = __builtin_bswap64(w);
    r[7] = uint(sw);
    r[8] = uint(sw >> 32);
    r[9] = __builtin_rotateleft8(c, uchar(n));
    r[10] = __builtin_rotateleft16(h, ushort(n));
    r[11] = __builtin_rotateleft32(a, n);
    ulong lw = rotl64(w, ulong(n));
    r[12] = uint(lw);
    r[13] = uint(lw >> 32);
    r[14] = __builtin_rotateright8(c, uchar(n));
    r[15] = __builtin_rotateright16(h, ushort(n));
    r[16] = __builtin_rotateright32(a, n);
    ulong rr = __builtin_rotateright64(w, ulong(n));
    r[17] = uint(rr);
    r[18] = uint(rr >> 32);
    r[19] = __builtin_popcount(a);
    r[20] = __builtin_popcountl(w);
    // Zero answered by the width: ctlz and cttz with the flag false.
    r[21] = a ? __builtin_clz(a) : 32u;
    r[22] = a ? __builtin_ctz(a) : 32u;
    r[23] = w ? __builtin_clzl(w) : 64u;
    r[24] = w ? __builtin_ctzl(w) : 64u;
    // The flag true, on operands that are never zero.
    r[25] = __builtin_clzs(ushort(h | 1));
    r[26] = __builtin_ctzs(ushort(h | 0x8000));
    r[27] = __builtin_clz(a | 1u);
    r[28] = __builtin_ctz(a | 0x80000000u);
    r[29] = __builtin_clzl(w | 1ul);
    r[30] = __builtin_ctzl(w | (1ul << 63));
    uint4 v = uint4(a, b, a ^ b, ~a);
    uint4 vr = rev32(v), vs = swap32(v), vl = rotl32(v, v.yzwx);
    r[31] = vr.x;
    r[32] = vr.y;
    r[33] = vr.z;
    r[34] = vr.w;
    r[35] = vs.x;
    r[36] = vs.y;
    r[37] = vs.z;
    r[38] = vs.w;
    r[39] = vl.x;
    r[40] = vl.y;
    r[41] = vl.z;
    r[42] = vl.w;
}
