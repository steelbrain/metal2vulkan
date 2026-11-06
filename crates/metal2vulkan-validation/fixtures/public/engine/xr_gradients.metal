// Owned fixture: gradient fills of the shape a 2D renderer draws into a wide-gamut target. A
// vertex function places a triangle from three corners in a buffer, and fragment functions
// evaluate a gradient at the pixel: its place in the unit square (`[[position]]` over the
// target's extent), the gradient's parameter there, and the color between the two stops around
// it, read from a buffer of stops (each a `float4` color and a `float` stop, as a renderer reads
// a storage buffer of them), then multiplied by the draw's alpha and premultiplied:
//
// - xrg_linear_half: along the line from `info.from` to `info.to`, a `half4` output;
// - xrg_radial_float: by the distance from `info.from` over the radius `info.to.x`, a `float4`
//   output.
//
// The stop loop runs at most 8 times whatever the count, and every fetch lies inside the stops
// buffer the case binds.
#include <metal_stdlib>
using namespace metal;

struct XrgVertex {
    float4 position [[position]];
};

// The triangle's corners in clip space.
vertex XrgVertex xrg_vertex(uint id [[vertex_id]], constant float2 *corners [[buffer(0)]]) {
    return XrgVertex { float4(corners[id % 3u], 0.0, 1.0) };
}

struct XrgStop {
    float4 color;
    float stop;
};

struct XrgInfo {
    float2 from;
    float2 to;
    float2 extent;
    float alpha;
    uint count;
};

// The color between the stops around `t`: the first stop's below it, the last one's above.
static float4 xrg_color(float t, const device XrgStop *stops, uint count) {
    uint n = clamp(count, 1u, 8u);
    float4 color = stops[0].color;
    for (uint i = 1; i < n; i++) {
        float low = stops[i - 1].stop;
        float high = stops[i].stop;
        if (t <= high) {
            float f = clamp((t - low) / max(high - low, 1.0e-6f), 0.0f, 1.0f);
            color = mix(stops[i - 1].color, stops[i].color, f);
            break;
        }
        color = stops[i].color;
    }
    return color;
}

fragment half4 xrg_linear_half(XrgVertex in [[stage_in]],
        const device XrgStop *stops [[buffer(0)]], constant XrgInfo &info [[buffer(1)]]) {
    float2 place = in.position.xy / info.extent;
    float2 d = info.to - info.from;
    float t = clamp(dot(place - info.from, d) / dot(d, d), 0.0f, 1.0f);
    float4 c = xrg_color(t, stops, info.count);
    float a = c.a * info.alpha;
    return half4(half3(c.rgb * a), half(a));
}

fragment float4 xrg_radial_float(XrgVertex in [[stage_in]],
        const device XrgStop *stops [[buffer(0)]], constant XrgInfo &info [[buffer(1)]]) {
    float2 place = in.position.xy / info.extent;
    float t = clamp(length(place - info.from) / info.to.x, 0.0f, 1.0f);
    float4 c = xrg_color(t, stops, info.count);
    float a = c.a * info.alpha;
    return float4(c.rgb * a, a);
}
