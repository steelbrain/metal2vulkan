// Owned fixture: fragment functions in the shapes in which a GPU was measured to lose a fragment's
// memory writes or to run them too often, which the runtime's device tests (`tests::lost_writes`)
// draw on every device and compare with Metal's answer. `lw_cover` draws one triangle over the
// whole target, its `uv` the target's position from 0 to 1 (top left 0, 1); `lw_half` draws the
// lower-left half of the target, so the pixels on its diagonal are partly covered.
//
// - `lw_early_write` declares early fragment tests and writes its pixel's coordinates into a
//   texture, `(x, y, 0.25, 1)`, at every sample count.
// - `lw_dropped` samples a texture, discards where `(x + 3 y) % 5` is 0, samples again, takes a
//   derivative, runs float arithmetic the module keeps apart
//   (`contract(off)`) in a branch only the discarded fragments enter, and writes a code into a
//   texture at its pixel: `(code, x, y, 1)`, the code the sampled red channel in 255ths plus 1.
// - `lw_count` reads `[[sample_mask]]` and writes memory: it adds 1 to its pixel's run count and
//   ORs the mask it read into its pixel's mask word. Metal runs such a function once per covered
//   sample, each run reading the pixel's whole coverage.
// Written for this repository; no third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct LwVertex {
    float4 position [[position]];
    float2 uv;
};

vertex LwVertex lw_cover(uint vid [[vertex_id]]) {
    float2 p = float2(float((vid << 1) & 2u), float(vid & 2u));
    LwVertex out;
    out.position = float4(p * 2.0 - 1.0, 0.0, 1.0);
    out.uv = p;
    return out;
}

vertex LwVertex lw_half(uint vid [[vertex_id]]) {
    float2 p = float2(float(vid == 1u), float(vid == 2u));
    LwVertex out;
    out.position = float4(p * 2.0 - 1.0, 0.0, 1.0);
    out.uv = p;
    return out;
}

[[early_fragment_tests]]
fragment float4 lw_early_write(LwVertex in [[stage_in]],
                               texture2d<float, access::write> out [[texture(0)]]) {
    uint2 at = uint2(in.position.xy);
    out.write(float4(float(at.x), float(at.y), 0.25, 1.0), at);
    return float4(0.0, 1.0, 0.0, 1.0);
}

struct LwGlobals {
    float scale;
    float bias;
};

static uint lw_code(half x) { return uint(rint(float(x) * 255.0)); }

fragment half4 lw_dropped(LwVertex in [[stage_in]], texture2d<half> noise [[texture(1)]],
                          constant LwGlobals &g [[buffer(1)]],
                          texture2d<uint, access::write> out [[texture(0)]]) {
#pragma clang fp contract(off)
    constexpr sampler smp(filter::nearest);
    uint2 p = uint2(in.position.xy);
    half4 m = noise.sample(smp, clamp(in.uv, 0.0, 1.0));
    float a = float(m.a) * g.scale + g.bias;
    bool hit = a > 0.5 && (p.x + 3u * p.y) % 5u == 0u;
    if (hit) {
        discard_fragment();
    }
    half4 s = noise.sample(smp, in.uv * 0.5);
    float w = fwidth(in.uv.x);
    half k = 0.0h;
    if (hit) {
        k = half(a) * half(g.scale) - half(w);
        k = k * k + half(g.bias) * k - 2.0h;
    }
    uint code = lw_code(s.r) + (w > 1.0e6 || k > 1000.0h ? 7u : 1u);
    out.write(uint4(code, p.x, p.y, 1u), p);
    return half4(s.rgb, 1.0h);
}

fragment float4 lw_count(LwVertex in [[stage_in]], uint mask [[sample_mask]],
                         device atomic_uint *runs [[buffer(0)]],
                         device atomic_uint *masks [[buffer(1)]],
                         constant uint &side [[buffer(2)]]) {
    uint2 at = uint2(in.position.xy);
    uint pixel = at.y * side + at.x;
    atomic_fetch_add_explicit(&runs[pixel], 1u, memory_order_relaxed);
    atomic_fetch_or_explicit(&masks[pixel], mask, memory_order_relaxed);
    return float4(0.0, 1.0, 0.0, 1.0);
}
