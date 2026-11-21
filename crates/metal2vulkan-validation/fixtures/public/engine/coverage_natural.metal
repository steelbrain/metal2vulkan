// Owned fixture: fragment functions that read [[sample_mask]] and write memory, and read no
// [[position]] beside a sample index or a sample-qualified varying. Documented Metal names no count
// of runs for a function that writes memory, so each runs as the device runs it, on every device,
// and its module carries PrecisionNote::DeviceFragmentRuns. Drawn by the generated full-screen
// triangle into a 1x1 pass without attachments; each function writes fixed words. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct NatPoint {
    float4 position [[position]];
};

struct NatUv {
    float2 uv [[sample_perspective]];
};

// Each run ORs the coverage it read into word 0, counts itself in word 1, and counts the coverage
// it read in word 2 + (coverage & 15): word 2 + c holds how many runs read coverage c.
fragment void nat_coverage(uint mask [[sample_mask]], device atomic_uint *words [[buffer(0)]]) {
    atomic_fetch_or_explicit(&words[0], mask, memory_order_relaxed);
    atomic_fetch_add_explicit(&words[1], 1u, memory_order_relaxed);
    atomic_fetch_add_explicit(&words[2u + (mask & 15u)], 1u, memory_order_relaxed);
}

// The same beside [[position]], which runs without a sample index read the pixel's center of:
// word 18 ORs the position's x and y, each times 4, rounded (x in bits 0-7, y in bits 8-15).
fragment void nat_coverage_at_pixel(NatPoint in [[stage_in]], uint mask [[sample_mask]],
                                    device atomic_uint *words [[buffer(0)]]) {
    atomic_fetch_or_explicit(&words[0], mask, memory_order_relaxed);
    atomic_fetch_add_explicit(&words[1], 1u, memory_order_relaxed);
    atomic_fetch_add_explicit(&words[2u + (mask & 15u)], 1u, memory_order_relaxed);
    uint2 at = uint2(in.position.xy * 4.0 + 0.5);
    atomic_fetch_or_explicit(&words[18], (at.x & 0xffu) | ((at.y & 0xffu) << 8),
                             memory_order_relaxed);
}

// Each run ORs its coverage, marked by 0x100, into its sample's word, and counts itself 8 words
// later: every covered sample's word holds the coverage its run read, and its count is 1.
fragment void nat_sample(uint mask [[sample_mask]], uint sample [[sample_id]],
                         device atomic_uint *words [[buffer(0)]]) {
    atomic_fetch_or_explicit(&words[sample & 7u], mask | 0x100u, memory_order_relaxed);
    atomic_fetch_add_explicit(&words[8u + (sample & 7u)], 1u, memory_order_relaxed);
}

// A sample-qualified varying beside the coverage, without the sample index: the generated
// companion's uv is 2a, so each run reads its sample's position over the pixel. Word
// round(uv.x * 8) ORs the coverage, round(uv.y * 8) in bits 8-15, and bit 16; word 16 + that slot
// counts the run.
fragment void nat_varying(NatUv in [[stage_in]], uint mask [[sample_mask]],
                          device atomic_uint *words [[buffer(0)]]) {
    uint2 at = uint2(in.uv * 8.0 + 0.5);
    uint slot = at.x & 15u;
    atomic_fetch_or_explicit(&words[slot], (mask & 0xffu) | ((at.y & 0xffu) << 8) | (1u << 16),
                             memory_order_relaxed);
    atomic_fetch_add_explicit(&words[16u + slot], 1u, memory_order_relaxed);
}
