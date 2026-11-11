// Owned fixture: the shape of a fixed-function compositing fragment function that holds its
// colours in `half`. Two layers, each of three sources; a source is an edge-aware filter of 16
// unrolled taps of a `half` texture around the fragment's point: the taps' mean, then each tap
// weighed by a buffer's `half` weight and its distance from the mean, and the weighted sum
// normalized, so all 16 taps are live across the two passes. A colour-op
// interpreter (a 17-case switch over a buffer's op code, run twice a layer) combines a layer's
// sources, a `half` tint modulates the result, and a 40-case blend switch (Porter-Duff operators
// and separable blend modes, each composed with alpha) puts the layer over the colour so far.
// Every choice is read from the buffer, so the compiler keeps every case. The vertex function
// covers the target with two triangles and gives each layer its point. A measurement proxy for
// how `half` values are held: its arithmetic, selects and phis are all `half`.
#include <metal_stdlib>
using namespace metal;

struct hnq_in {
    float4 position [[position]];
    float2 uv0 [[user(locn0)]];
    float2 uv1 [[user(locn1)]];
    half4 tint [[user(locn2)]];
};

struct hnq_layer {
    float2 step[3];
    uint op[2];
    uint blend;
    uint pad;
    half4 k;
    half weights[3][16];
};

struct hnq_params {
    hnq_layer layer[2];
};

vertex hnq_in hnq_layer_vertex(uint vid [[vertex_id]])
{
    const float2 corners[6] = {float2(0, 0), float2(1, 0), float2(1, 1),
                               float2(0, 0), float2(1, 1), float2(0, 1)};
    float2 p = corners[vid % 6];
    hnq_in out;
    out.position = float4(p * 2.0 - 1.0, 0.0, 1.0);
    out.uv0 = p;
    out.uv1 = p * 0.75 + 0.125;
    out.tint = half4(1.0h, 0.875h, 0.75h, 1.0h);
    return out;
}

static half4 hnq_filter(texture2d<half> t, sampler s, float2 uv, float2 step,
                        constant half *w, half sharp)
{
    half4 tap[16];
    half4 mean = half4(0.0h);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        float2 o = float2(float(i & 3) - 1.5, float(i >> 2) - 1.5) * step;
        tap[i] = t.sample(s, uv + o);
        mean += tap[i];
    }
    mean *= 0.0625h;
    half4 sum = half4(0.0h);
    half total = 0.0h;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        half4 d = tap[i] - mean;
        half k = w[i] * max(1.0h - dot(d, d) * sharp, 0.0h);
        sum += tap[i] * k;
        total += k;
    }
    return sum / max(total, 0.001h);
}

static half4 hnq_colour_op(uint op, half4 a, half4 b, half4 c, half4 k)
{
    switch (op) {
    case 0: return a;
    case 1: return a * b;
    case 2: return a * b * 2.0h;
    case 3: return a * b * 4.0h;
    case 4: return a + b;
    case 5: return a + b - 0.5h;
    case 6: return (a + b - 0.5h) * 2.0h;
    case 7: return a - b;
    case 8: return a + b - a * b;
    case 9: return mix(b, a, c.a);
    case 10: return mix(b, a, k.a);
    case 11: return half4(dot(a.rgb - 0.5h, b.rgb - 0.5h) * 4.0h);
    case 12: return a * b + c;
    case 13: return mix(c, b, a);
    case 14: return a.a * b + c;
    case 15: return saturate(a) * k;
    default: return clamp(a * k + c, 0.0h, 1.0h);
    }
}

static half hnq_lum(half3 c)
{
    return dot(c, half3(0.3h, 0.59h, 0.11h));
}

static half4 hnq_blend(uint mode, half4 s, half4 d)
{
    half3 sc = s.rgb;
    half3 dc = d.rgb;
    half sa = s.a;
    half da = d.a;
    half3 r;
    switch (mode) {
    case 0: return s;
    case 1: return s + d * (1.0h - sa);
    case 2: return d + s * (1.0h - da);
    case 3: return s * da;
    case 4: return d * sa;
    case 5: return s * (1.0h - da);
    case 6: return d * (1.0h - sa);
    case 7: return half4(sc * da + dc * (1.0h - sa), da);
    case 8: return half4(dc * sa + sc * (1.0h - da), sa);
    case 9: return s * (1.0h - da) + d * (1.0h - sa);
    case 10: return min(s + d, half4(1.0h));
    case 11: r = sc * dc; break;
    case 12: r = sc + dc - sc * dc; break;
    case 13: r = select(1.0h - 2.0h * (1.0h - sc) * (1.0h - dc), 2.0h * sc * dc, dc <= 0.5h); break;
    case 14: r = min(sc, dc); break;
    case 15: r = max(sc, dc); break;
    case 16: r = min(dc / max(1.0h - sc, half3(0.001h)), half3(1.0h)); break;
    case 17: r = 1.0h - min((1.0h - dc) / max(sc, half3(0.001h)), half3(1.0h)); break;
    case 18: r = select(1.0h - 2.0h * (1.0h - sc) * (1.0h - dc), 2.0h * sc * dc, sc <= 0.5h); break;
    case 19: r = select(dc + (2.0h * sc - 1.0h) * (sqrt(dc) - dc),
                        dc - (1.0h - 2.0h * sc) * dc * (1.0h - dc), sc <= 0.5h); break;
    case 20: r = abs(sc - dc); break;
    case 21: r = sc + dc - 2.0h * sc * dc; break;
    case 22: r = max(sc + dc - 1.0h, half3(0.0h)); break;
    case 23: r = min(sc + dc, half3(1.0h)); break;
    case 24: r = select(dc / max(2.0h * (1.0h - sc), half3(0.001h)),
                        1.0h - (1.0h - dc) / max(2.0h * sc, half3(0.001h)), sc <= 0.5h); break;
    case 25: r = clamp(dc + 2.0h * sc - 1.0h, 0.0h, 1.0h); break;
    case 26: r = select(max(dc, 2.0h * sc - 1.0h), min(dc, 2.0h * sc), sc <= 0.5h); break;
    case 27: r = step(1.0h, sc + dc); break;
    case 28: r = max(dc - sc, half3(0.0h)); break;
    case 29: r = min(dc / max(sc, half3(0.001h)), half3(1.0h)); break;
    case 30: r = dc + (hnq_lum(sc) - hnq_lum(dc)); break;
    case 31: r = mix(half3(hnq_lum(dc)), dc, half3(max3(sc.r, sc.g, sc.b) - min3(sc.r, sc.g, sc.b))); break;
    case 32: r = sc + (hnq_lum(dc) - hnq_lum(sc)); break;
    case 33: r = mix(dc, sc, half3(0.5h)) * (1.0h + hnq_lum(sc) - hnq_lum(dc)); break;
    case 34: r = (sc + dc) * 0.5h; break;
    case 35: r = 1.0h - abs(1.0h - sc - dc); break;
    case 36: r = min(sc * sc / max(1.0h - dc, half3(0.001h)), half3(1.0h)); break;
    case 37: r = min(dc * dc / max(1.0h - sc, half3(0.001h)), half3(1.0h)); break;
    case 38: r = min(sc, dc) - max(sc, dc) + 1.0h; break;
    default: r = clamp(dc + sc - 0.5h, 0.0h, 1.0h); break;
    }
    half ra = sa + da * (1.0h - sa);
    half3 rc = (1.0h - da) * sc * sa + (1.0h - sa) * dc * da + sa * da * clamp(r, 0.0h, 1.0h);
    return half4(rc, ra);
}

fragment half4 hnq_half_layer_composite(hnq_in in [[stage_in]],
                                        constant hnq_params &p [[buffer(0)]],
                                        texture2d<half> t0 [[texture(0)]],
                                        texture2d<half> t1 [[texture(1)]],
                                        texture2d<half> t2 [[texture(2)]],
                                        texture2d<half> t3 [[texture(3)]],
                                        texture2d<half> t4 [[texture(4)]],
                                        texture2d<half> t5 [[texture(5)]],
                                        sampler s [[sampler(0)]])
{
    half4 dst = half4(0.0h);
    {
        constant hnq_layer &l = p.layer[0];
        half4 a = hnq_filter(t0, s, in.uv0, l.step[0], l.weights[0], l.k.z);
        half4 b = hnq_filter(t1, s, in.uv0, l.step[1], l.weights[1], l.k.z);
        half4 c = hnq_filter(t2, s, in.uv0, l.step[2], l.weights[2], l.k.z);
        half4 x = hnq_colour_op(l.op[0], a, b, c, l.k);
        x = hnq_colour_op(l.op[1], x, c, a, l.k) * in.tint;
        dst = hnq_blend(l.blend, x, dst);
    }
    {
        constant hnq_layer &l = p.layer[1];
        half4 a = hnq_filter(t3, s, in.uv1, l.step[0], l.weights[0], l.k.z);
        half4 b = hnq_filter(t4, s, in.uv1, l.step[1], l.weights[1], l.k.z);
        half4 c = hnq_filter(t5, s, in.uv1, l.step[2], l.weights[2], l.k.z);
        half4 x = hnq_colour_op(l.op[0], a, b, c, l.k);
        x = hnq_colour_op(l.op[1], x, c, a, l.k) * in.tint;
        dst = hnq_blend(l.blend, x, dst);
    }
    return dst;
}
