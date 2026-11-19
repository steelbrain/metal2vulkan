// Owned fixture: threadgroup memory read before the program writes it, a declared threadgroup array
// and a threadgroup argument alike. Written for this repository; no third-party source.
//
// Thread t of threadgroup g, of n threads (one-dimensional), first reads every word i = t, t + n,
// ... of the declared array `decl` (WORDS words) and of the argument `arg`, whose length the
// encoder sets to WORDS words; a barrier; then stores a nonzero pattern into those words (bit 31
// set, g and i below it, bit 30 too in `arg`); a barrier; then reads the words of thread
// (t + 1) mod n. So every word a thread reads before the first barrier is one no thread of its
// threadgroup has written, every word of both is stored before the threadgroup ends, and each is
// read back once. Each simdgroup's first lane folds its lanes into four words of `out`:
//     out[0] |= 1 where a word of `decl` read nonzero before any write
//     out[1] |= 1 where a word of `arg` read nonzero before any write
//     out[2] += decl[i] + 2 arg[i] over the words read after the second barrier (wrapping)
//     out[3] += k * k, k the simdgroup's active threads
// so out[0] and out[1] are 0 on a GPU whose threadgroup memory reads zero until the program
// writes it, out[2] pins the patterns' stores and loads, and out[3] the simdgroups' shape.
#include <metal_stdlib>
using namespace metal;

constant uint WORDS = 2048;

kernel void tgz_reads(device atomic_uint *out [[buffer(0)]],
                      threadgroup uint *arg [[threadgroup(0)]],
                      uint t [[thread_index_in_threadgroup]],
                      uint g [[threadgroup_position_in_grid]],
                      uint n [[threads_per_threadgroup]]) {
    threadgroup uint decl[WORDS];
    uint seen_decl = 0u;
    uint seen_arg = 0u;
    for (uint i = t; i < WORDS; i += n) {
        seen_decl |= decl[i];
        seen_arg |= arg[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = t; i < WORDS; i += n) {
        uint pattern = 0x80000000u | ((g * WORDS + i) & 0x3fffffffu);
        decl[i] = pattern;
        arg[i] = pattern | 0x40000000u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint folded = 0u;
    for (uint i = (t + 1u) % n; i < WORDS; i += n) {
        folded += decl[i] + 2u * arg[i];
    }
    uint any_decl = simd_or(uint(seen_decl != 0u));
    uint any_arg = simd_or(uint(seen_arg != 0u));
    uint summed = simd_sum(folded);
    uint lanes = simd_sum(1u);
    if (simd_is_first()) {
        if (any_decl != 0u) {
            atomic_fetch_or_explicit(&out[0], 1u, memory_order_relaxed);
        }
        if (any_arg != 0u) {
            atomic_fetch_or_explicit(&out[1], 1u, memory_order_relaxed);
        }
        atomic_fetch_add_explicit(&out[2], summed, memory_order_relaxed);
        atomic_fetch_add_explicit(&out[3], lanes * lanes, memory_order_relaxed);
    }
}
