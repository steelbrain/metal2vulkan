// Owned fixture: `uint4` and `float4` accesses at each of the four word phases past a buffer's
// bound base, where the AIR states `align 16` and the address is only a multiple of 4, and the same
// accesses through a `float4` buffer, whose bound base is a multiple of 16. Loop-free; no float
// arithmetic but negation, so every answer is the words Metal moved. Where every input word is a
// normal `float` (its exponent field neither 0 nor 255), so is every `float` the kernels load or
// store: the keys of the `float4` updates have their exponent bits cleared.
//
// `vwp_words`: `src` and `dst` are `uint` buffers (argument alignment 4), which Metal binds at any
// multiple of 4. Thread t of 64 has phase p = t % 4 and vector i = t / 4. Through pointers cast to
// `device uint4 *` and `device float4 *` it reads a vector at word p + 4i of a region of `src`,
// writes one at word p + 4i of a region of `dst`, and reads, changes and writes back one there:
// the address is 4p bytes past a multiple of 16 from the bound base, whatever the AIR's `align 16`
// says. Each phase writes a region of its own, 65 words apart, so no two threads write a word and
// the word after each region is never written. Words of `dst`:
//
// - 0-255: the `uint4` read's lanes, words 4t to 4t + 3;
// - 256-511: the `float4` read's lanes, negated, as words;
// - 512-771: the `uint4` write (`src` words 512 + 4t to 515 + 4t), at 512 + 65p + 4i;
// - 772-1031: the `float4` write (`src` words 768 + 4t to 771 + 4t, negated), at 772 + 65p + 4i;
// - 1032-1291: the `uint4` update, `^=` `src` words 1024 + 4t to 1027 + 4t, at 1032 + 65p + 4i;
// - 1292-1551: the `float4` update, negated and then `^=` `src` words 1280 + 4t to 1283 + 4t
//   without their exponent bits, at 1292 + 65p + 4i.
//
// The two read regions of `src` are words 0-66 (the `uint4` read) and 256-322 (the `float4` read).
//
// `vwp_vec16`: `a` and `b` are `float4` buffers (argument alignment 16), which Metal binds at a
// multiple of 16, so every access is at a multiple of 16 from a 16-aligned base. Thread t of 64
// writes `-a[t]` to `b[t]`, updates `b[64 + t]` by `^=` the bits of `a[64 + t]` without their
// exponent bits, and copies
// `a[128 + t]` to `b[128 + t]` through `device uint4 *` pointers.
#include <metal_stdlib>
using namespace metal;

/// The words from one phase's region of `dst` to the next: 64 written and one never written.
constant uint PHASE = 65;

kernel void vwp_words(
    device const uint *src [[buffer(0)]],
    device uint *dst [[buffer(1)]],
    uint t [[thread_position_in_grid]])
{
    uint p = t % 4;
    uint i = t / 4;
    uint4 a = ((device const uint4 *)(src + p))[i];
    float4 b = -((device const float4 *)(src + 256 + p))[i];
    uint4 c = as_type<uint4>(b);
    dst[4 * t] = a.x;
    dst[4 * t + 1] = a.y;
    dst[4 * t + 2] = a.z;
    dst[4 * t + 3] = a.w;
    dst[256 + 4 * t] = c.x;
    dst[256 + 4 * t + 1] = c.y;
    dst[256 + 4 * t + 2] = c.z;
    dst[256 + 4 * t + 3] = c.w;
    uint4 w = uint4(src[512 + 4 * t], src[513 + 4 * t], src[514 + 4 * t], src[515 + 4 * t]);
    ((device uint4 *)(dst + 512 + PHASE * p))[i] = w;
    uint4 v = uint4(src[768 + 4 * t], src[769 + 4 * t], src[770 + 4 * t], src[771 + 4 * t]);
    ((device float4 *)(dst + 772 + PHASE * p))[i] = -as_type<float4>(v);
    uint4 k = uint4(src[1024 + 4 * t], src[1025 + 4 * t], src[1026 + 4 * t], src[1027 + 4 * t]);
    device uint4 *u = (device uint4 *)(dst + 1032 + PHASE * p) + i;
    *u = *u ^ k;
    uint4 m = uint4(src[1280 + 4 * t], src[1281 + 4 * t], src[1282 + 4 * t], src[1283 + 4 * t]);
    device float4 *f = (device float4 *)(dst + 1292 + PHASE * p) + i;
    *f = as_type<float4>(as_type<uint4>(-*f) ^ (m & 0x807fffffu));
}

kernel void vwp_vec16(
    device const float4 *a [[buffer(0)]],
    device float4 *b [[buffer(1)]],
    uint t [[thread_position_in_grid]])
{
    b[t] = -a[t];
    b[64 + t] = as_type<float4>(as_type<uint4>(b[64 + t]) ^ (as_type<uint4>(a[64 + t]) & 0x807fffffu));
    ((device uint4 *)b)[128 + t] = ((device const uint4 *)a)[128 + t];
}
