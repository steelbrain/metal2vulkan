// Owned fixture: a kernel that samples the two textures its argument buffer holds, a `half` one and
// a `float` one, through a constexpr nearest sampler and through a sampler slot, which the
// runtime's device test (`tests::argument_tier`) runs on every device and checks exactly. Thread
// `(x, y)` of a 2 by 2 grid samples texel `(x, y)` of each 2 by 2 texture at its center, and
// writes the four answers to `out[4 * (2 y + x)]` on: the `half` texture's through the constexpr
// sampler, the `float` texture's through it, then each through the slot's sampler.
// Written for this repository; no third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct HsTextures {
    texture2d<half> h [[id(0)]];
    texture2d<float> f [[id(1)]];
};

kernel void hs_sample(constant HsTextures &t [[buffer(0)]], device float4 *out [[buffer(1)]],
                      sampler slot [[sampler(0)]], uint2 g [[thread_position_in_grid]]) {
    constexpr sampler nearest(filter::nearest);
    float2 uv = (float2(g) + 0.5) / 2.0;
    uint at = 4u * (2u * g.y + g.x);
    out[at] = float4(t.h.sample(nearest, uv, level(0.0)));
    out[at + 1u] = t.f.sample(nearest, uv, level(0.0));
    out[at + 2u] = float4(t.h.sample(slot, uv, level(0.0)));
    out[at + 3u] = t.f.sample(slot, uv, level(0.0));
}
