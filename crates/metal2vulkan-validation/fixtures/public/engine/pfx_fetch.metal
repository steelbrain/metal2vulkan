// Owned fixture: what a fragment function reads of color 0 through programmable blending.
// `pfx_fetch_fill` writes color 0 alone, the `float4` of its pixel in a buffer of rows `width`
// long (a draw earlier in the pass sets the attachment's tile); `pfx_fetch_read` reads color 0 as
// a `float4` and `pfx_fetch_read_half` as a `half4`, each writing what it read to color 1, the
// attachment a test reads. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct PfxVertex {
    float4 position [[position]];
};

// The fullscreen triangle, (-1, -1), (3, -1), (-1, 3).
vertex PfxVertex pfx_fetch_vertex(uint id [[vertex_id]]) {
    float2 a = float2(float(id & 1u), float(id >> 1));
    return PfxVertex { float4(4.0 * a - 1.0, 0.0, 1.0) };
}

struct PfxFill {
    float4 color [[color(0)]];
};

fragment PfxFill pfx_fetch_fill(PfxVertex input [[stage_in]],
        const device float4 *values [[buffer(0)]], constant uint &width [[buffer(1)]]) {
    uint x = uint(input.position.x);
    uint y = uint(input.position.y);
    return PfxFill { values[y * width + x] };
}

struct PfxRead {
    float4 color [[color(1)]];
};

fragment PfxRead pfx_fetch_read(PfxVertex input [[stage_in]], float4 previous [[color(0)]]) {
    return PfxRead { previous };
}

fragment PfxRead pfx_fetch_read_half(PfxVertex input [[stage_in]], half4 previous [[color(0)]]) {
    return PfxRead { float4(previous) };
}
