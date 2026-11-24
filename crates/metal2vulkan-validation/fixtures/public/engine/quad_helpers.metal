// Owned fixture (written for this repository): a fragment's quad votes, ballot, `is_first` and
// active mask, and its simdgroup reductions and votes whose answers name no lane, over quads a
// triangle covers in part, so each such quad runs its uncovered pixels as helpers.
// - `qh_cells` draws a small triangle in each 8 by 8 cell of a 64 by 64 target, as
//   reconverge_select's `rs_cells` does, whose edges cross quads: a quad on an edge runs one to
//   three helpers. `qh_one` draws one triangle over three pixels of the quad at (2, 2) (vertices
//   3k to 3k + 2: the helper is quad lane k), the only fragments of its draw, so the simdgroup
//   that shades them holds that quad alone.
// - Each fragment function runs its group operations first, in control flow uniform over the
//   draw, and then takes a derivative of a value the answers feed. A helper still has that
//   derivative to compute when the operations run, so it is an active thread of each (MSL 4.1
//   sections 6.10.2 and 6.10.3).
// - v = a + b * (x & 1) + c * (y & 1), with a, b (2 to 9) and c (2 to 9) words of the pixel's
//   quad's hash in buffer 0, so the fine and the coarse derivative of v alike are b across and c
//   down. The quad functions write (answer, dfdx(v + answer), dfdy(v + answer), quad lane), the
//   simdgroup functions three answers and dfdx of v plus their sum. Every value is a small
//   integer, exact in a half.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

struct QhVertex {
    float4 position [[position]];
};

// A target pixel's place in clip space, for the 64 by 64 target.
static float4 qh_clip(float2 at) {
    return float4(at.x / 32.0f - 1.0f, 1.0f - at.y / 32.0f, 0.0f, 1.0f);
}

// Small triangles, one in each 8 by 8 cell of a 64 by 64 target (vertex 3k + c is corner c of
// cell k).
vertex QhVertex qh_cells(uint id [[vertex_id]]) {
    uint cell = id / 3u;
    uint corner = id % 3u;
    float2 corners[3] = {float2(1.3f, 0.7f), float2(6.6f, 1.9f), float2(2.1f, 6.8f)};
    QhVertex out;
    out.position = qh_clip(float2(float(cell % 8u), float(cell / 8u)) * 8.0f + corners[corner]);
    return out;
}

// One triangle over three pixels of the quad whose top-left pixel is (2, 2), leaving out quad
// lane k = id / 3 (column k & 1, row k >> 1). Its edges pass at least a third of a pixel from
// every pixel centre: x >= -0.15, y <= 2.15 and y >= x - 0.5 in the quad's own coordinates leave
// out (1, 0), mirrored across the quad's middle for the other lanes.
vertex QhVertex qh_one(uint id [[vertex_id]]) {
    uint lane = id / 3u;
    float2 corners[3] = {float2(-0.15f, -0.65f), float2(-0.15f, 2.15f), float2(2.65f, 2.15f)};
    float2 at = corners[id % 3u];
    if ((lane & 1u) == 0u) {
        at.x = 2.0f - at.x;
    }
    if ((lane >> 1) != 0u) {
        at.y = 2.0f - at.y;
    }
    QhVertex out;
    out.position = qh_clip(at + 2.0f);
    return out;
}

static uint qh_hash(uint2 q) {
    uint h = (q.x * 0x9E3779B1u) ^ (q.y * 0x85EBCA77u);
    h ^= h >> 15;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12;
    return h;
}

// v of pixel p: a + b * (x & 1) + c * (y & 1), from the words of its quad's hash.
static float qh_v(uint2 p, device const uint *data) {
    uint h = qh_hash(p >> 1u);
    float a = float(data[h & 255u] & 255u);
    float b = float(2u + (data[(h >> 8) & 255u] & 7u));
    float c = float(2u + (data[(h >> 16) & 255u] & 7u));
    return a + b * float(p.x & 1u) + c * float(p.y & 1u);
}

// A quad function's texel: its answer r, then the fine pair of v + r, which the helpers compute
// after r, and the pixel's quad lane.
static float4 qh_texel(float r, float v, ushort lane) {
    float u = v + r;
    return float4(r, dfdx(u), dfdy(u), float(lane));
}

static float qh_bits(quad_vote vote) {
    return float(uint(quad_vote::vote_t(vote)));
}

// Whether any lane of the quad is a helper.
fragment float4 qh_any(float4 position [[position]], ushort lane [[thread_index_in_quadgroup]],
                       device const uint *data [[buffer(0)]]) {
    uint2 p = uint2(position.xy);
    float r = float(quad_any(quad_is_helper_thread()));
    return qh_texel(r, qh_v(p, data), lane);
}

// Whether every lane of the quad is not a helper.
fragment float4 qh_all(float4 position [[position]], ushort lane [[thread_index_in_quadgroup]],
                       device const uint *data [[buffer(0)]]) {
    uint2 p = uint2(position.xy);
    float r = float(quad_all(!quad_is_helper_thread()));
    return qh_texel(r, qh_v(p, data), lane);
}

// The quad's helper lanes, as its ballot's four bits.
fragment float4 qh_ballot(float4 position [[position]], ushort lane [[thread_index_in_quadgroup]],
                          device const uint *data [[buffer(0)]]) {
    uint2 p = uint2(position.xy);
    float r = qh_bits(quad_ballot(quad_is_helper_thread()));
    return qh_texel(r, qh_v(p, data), lane);
}

// Whether the pixel is its quad's first active lane.
fragment float4 qh_is_first(float4 position [[position]],
                            ushort lane [[thread_index_in_quadgroup]],
                            device const uint *data [[buffer(0)]]) {
    uint2 p = uint2(position.xy);
    float r = float(quad_is_first());
    return qh_texel(r, qh_v(p, data), lane);
}

// The quad's active lanes.
fragment float4 qh_active(float4 position [[position]], ushort lane [[thread_index_in_quadgroup]],
                          device const uint *data [[buffer(0)]]) {
    uint2 p = uint2(position.xy);
    float r = qh_bits(quad_active_threads_mask());
    return qh_texel(r, qh_v(p, data), lane);
}

// A simdgroup function's texel: three answers, then dfdx of v plus their sum, which the helpers
// compute after them.
static float4 qs_texel(float3 r, float v) {
    return float4(r, dfdx(v + r.x + r.y + r.z));
}

// The low 32 bits of a simdgroup vote, the bits of a 32-wide simdgroup's lanes, counted.
static float qs_count(simd_vote vote) {
    return float(popcount(uint(static_cast<simd_vote::vote_t>(vote))));
}

// The simdgroup's sum, minimum and maximum of a byte of the pixel's hash.
fragment float4 qs_arith(float4 position [[position]], device const uint *data [[buffer(0)]]) {
    uint2 p = uint2(position.xy);
    uint w = data[qh_hash(p) & 255u] & 255u;
    float3 r = float3(float(simd_sum(w)), float(simd_min(w)), float(simd_max(w)));
    return qs_texel(r, qh_v(p, data));
}

// The simdgroup's and, or and xor of 11 bits of the pixel's hash.
fragment float4 qs_bits(float4 position [[position]], device const uint *data [[buffer(0)]]) {
    uint2 p = uint2(position.xy);
    uint w = data[qh_hash(p) & 255u] & 2047u;
    float3 r = float3(float(simd_and(w)), float(simd_or(w)), float(simd_xor(w)));
    return qs_texel(r, qh_v(p, data));
}

// Whether any lane of the simdgroup is a helper (1) and every lane is not (2), how many lanes its
// ballot of the helper query sets, and how many it holds active.
fragment float4 qs_votes(float4 position [[position]], device const uint *data [[buffer(0)]]) {
    uint2 p = uint2(position.xy);
    bool h = simd_is_helper_thread();
    float votes = float(uint(simd_any(h)) | uint(simd_all(!h)) << 1);
    float3 r = float3(votes, qs_count(simd_ballot(h)), qs_count(simd_active_threads_mask()));
    return qs_texel(r, qh_v(p, data));
}
