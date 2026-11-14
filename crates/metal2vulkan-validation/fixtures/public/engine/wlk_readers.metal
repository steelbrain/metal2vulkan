// Owned fixture: fragment functions that read their own render target at the fragment's pixel and
// branch on the texel they read, as a window server's backdrop readers do: the read decides the
// trip count of a loop. `wlk_read_branch` reads the target as texture 0 (a slot no snapshot serves,
// its read deciding a branch); `wlk_fetch_branch` reads it as the color attachment it renders
// (`[[color(0)]]`), the same arithmetic. Both pair with `tob_rect` of `tob_renders.metal`. Written
// for this repository; no third-party source.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

struct WlkVertex {
    float4 position [[position]];
};

// The texel's alpha in quarters decides how many times it is scaled toward a constant.
static float4 wlk_shade(float4 c) {
    float4 acc = c;
    uint n = uint(c.a * 4.0f);
    for (uint i = 0u; i < n; i++) {
        acc = acc * 0.75f + 0.0625f;
    }
    return acc;
}

fragment float4 wlk_read_branch(WlkVertex in [[stage_in]],
                                texture2d<float, access::read> t [[texture(0)]]) {
    return wlk_shade(t.read(uint2(in.position.xy)));
}

fragment float4 wlk_fetch_branch(WlkVertex in [[stage_in]], float4 c [[color(0)]]) {
    return wlk_shade(c);
}
