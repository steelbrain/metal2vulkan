// air-rename: _Z8agx_shfltt llvm.agx2.shfl
// Owned fixture: `llvm.agx2.shfl`, the interleave of the bits of two 16-bit values. MSL has no
// spelling for it, so the kernels call `agx_shfl`, which this source declares and never defines,
// and the directive above renames every call to the intrinsic in both companions
// (scripts/regen-air-fixtures). Measured on an M3 Max over all 2^32 pairs of operands: bit 2i of
// the result is bit i of the first operand, and bit 2i + 1 is bit i of the second.
#include <metal_stdlib>
using namespace metal;

uint agx_shfl(ushort even, ushort odd);

// Word i of out is the interleave of the low and the high half of word i of in.
kernel void agx_shfl_pairs(device const uint *in [[buffer(0)]],
                           device uint *out [[buffer(1)]],
                           uint i [[thread_position_in_grid]]) {
    uint p = in[i];
    out[i] = agx_shfl(ushort(p), ushort(p >> 16));
}

// The interleave, bit by bit.
static uint interleave_bits(uint even, uint odd) {
    uint r = 0;
    for (uint i = 0; i < 16; i++) {
        r |= ((even >> i) & 1u) << (2u * i);
        r |= ((odd >> i) & 1u) << (2u * i + 1u);
    }
    return r;
}

// Thread t interleaves four pairs: (t, 0), (0, t), (t, ~t), and t with the low half of
// t * 40503 + 12345: every 16-bit value is the first operand once and the second once beside a
// zero. Word 0 of out counts the pairs whose result is not the bit-by-bit interleave, word 1 is
// the wrapping sum of every result, and word 2 the wrapping sum of each result times its pair's
// index plus one (pair k of thread t is pair 4t + k).
kernel void agx_shfl_sweep(device atomic_uint *out [[buffer(0)]],
                           uint t [[thread_position_in_grid]]) {
    ushort x = ushort(t);
    ushort evens[4] = {x, 0, x, x};
    ushort odds[4] = {0, x, ushort(~x), ushort(t * 40503u + 12345u)};
    uint bad = 0;
    uint sum = 0;
    uint weighted = 0;
    for (uint k = 0; k < 4; k++) {
        uint r = agx_shfl(evens[k], odds[k]);
        bad += r != interleave_bits(evens[k], odds[k]) ? 1u : 0u;
        sum += r;
        weighted += r * (4u * t + k + 1u);
    }
    atomic_fetch_add_explicit(&out[0], bad, memory_order_relaxed);
    atomic_fetch_add_explicit(&out[1], sum, memory_order_relaxed);
    atomic_fetch_add_explicit(&out[2], weighted, memory_order_relaxed);
}
