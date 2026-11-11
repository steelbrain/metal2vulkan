// Owned fixture: fragment functions that read their color attachment (framebuffer fetch), read
// and sample a texture, may discard on a pattern of pixel positions, sample again after the
// discard site, take a derivative, and write a code that depends on what they fetched: the shape
// of a compositing fragment whose storage writes a device without rasterization-order color
// access makes in the pixel interlock. Each fragment function writes color attachment 0 (an
// 8-bit normalized format) as one code `c` in its red, green and blue channels and 255 in alpha.
// `sfd_fill` writes 40. `sfd_alpha` writes the code it fetched plus the code its texture holds
// (every texel the same, so every sample and read returns it) plus 1, wrapping at 256, unless the
// pixel's position (x + 3 y) modulo `discard_mod` equals `discard_hit` (the globals at buffer 0;
// a hit past the modulus never discards), where it discards. `sfd_line`, `sfd_dropped` and
// `sfd_edge` write the same, with some of their float arithmetic under `contract(off)`, which the
// module keeps apart: `sfd_line` in straight-line code after the sample, `sfd_dropped` in a branch
// only the discarded fragments enter, and `sfd_edge` in a branch only the invocations past the
// edges of an `sfd_cells` rectangle enter (the globals' `cell`, the cell's sides in pixels), which
// are helper invocations. `sfd_fan` draws one fullscreen triangle; `sfd_cells` draws one
// rectangle per instance, in a grid of 8 by 4 cells over the target (its sides in pixels at
// buffer 0), each from 3 pixels right and 5 down of its cell's top left to 2 pixels left and 2 up
// of its bottom right, so its edges cut 2x2 quads.
// Written for this repository; no third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct SfdVertex {
    float4 position [[position]];
    float2 uv;
};

vertex SfdVertex sfd_fan(uint vid [[vertex_id]]) {
    uint j = vid % 3u;
    float2 p = float2(float((j << 1) & 2u), float(j & 2u));
    SfdVertex out;
    out.position = float4(p * 2.0 - 1.0, 0.0, 1.0);
    out.uv = p;
    return out;
}

vertex SfdVertex sfd_cells(uint vid [[vertex_id]], uint iid [[instance_id]],
                           constant float2 &size [[buffer(0)]]) {
    uint q = (0x312210u >> (4u * (vid % 6u))) & 0xfu;
    float2 cell = size / float2(8.0, 4.0);
    float2 origin = float2(float(iid % 8u), float(iid / 8u)) * cell;
    float2 from = origin + float2(3.0, 5.0);
    float2 to = origin + cell - float2(2.0, 2.0);
    float2 at = float2((q & 1u) != 0u ? to.x : from.x, (q & 2u) != 0u ? to.y : from.y);
    SfdVertex out;
    out.position = float4(at.x / size.x * 2.0 - 1.0, 1.0 - at.y / size.y * 2.0, 0.0, 1.0);
    out.uv = at / size;
    return out;
}

static half4 sfd_color(uint c) { return half4(half3(half(c & 255u) / 255.0h), 1.0h); }

static uint sfd_code(half x) { return uint(rint(float(x) * 255.0)); }

fragment half4 sfd_fill() { return sfd_color(40u); }

struct SfdGlobals {
    float scale;
    float bias;
    uint discard_mod;
    uint discard_hit;
    float2 cell;
};

fragment half4 sfd_alpha(SfdVertex in [[stage_in]], half4 dst [[color(0)]],
                         texture2d<half> tex [[texture(0)]], sampler smp [[sampler(0)]],
                         constant SfdGlobals &g [[buffer(0)]]) {
    uint2 p = uint2(in.position.xy);
    half4 base = tex.read(p % 4u);
    half4 m = tex.sample(smp, clamp(in.uv, 0.0, 1.0));
    float a = fma(float(m.a), g.scale, g.bias);
    if (a > 0.5 && (p.x + 3u * p.y) % g.discard_mod == g.discard_hit) {
        discard_fragment();
    }
    half4 s = tex.sample(smp, in.uv * 0.5);
    half3 mixed = fma(s.rgb, half3(1.0h), base.rgb * half(g.bias));
    float w = fwidth(in.uv.x);
    uint code = sfd_code(dst.r) + sfd_code(mixed.r) + (w > 1.0e6 ? 7u : 1u);
    return sfd_color(code);
}

fragment half4 sfd_line(SfdVertex in [[stage_in]], half4 dst [[color(0)]],
                        texture2d<half> tex [[texture(0)]], sampler smp [[sampler(0)]],
                        constant SfdGlobals &g [[buffer(0)]]) {
#pragma clang fp contract(off)
    uint2 p = uint2(in.position.xy);
    half4 base = tex.read(p % 4u);
    half4 m = tex.sample(smp, clamp(in.uv, 0.0, 1.0));
    float a = float(m.a) * g.scale + g.bias;
    if (a > 0.5 && (p.x + 3u * p.y) % g.discard_mod == g.discard_hit) {
        discard_fragment();
    }
    half4 s = tex.sample(smp, in.uv * 0.5);
    float w = fwidth(in.uv.x);
    half k = half(a) - 1.0h;
    half3 mixed = s.rgb + base.rgb * half(g.bias) + half3(k);
    uint code = sfd_code(dst.r) + sfd_code(mixed.r) + (w > 1.0e6 ? 7u : 1u);
    return sfd_color(code);
}

fragment half4 sfd_dropped(SfdVertex in [[stage_in]], half4 dst [[color(0)]],
                           texture2d<half> tex [[texture(0)]], sampler smp [[sampler(0)]],
                           constant SfdGlobals &g [[buffer(0)]]) {
#pragma clang fp contract(off)
    uint2 p = uint2(in.position.xy);
    half4 base = tex.read(p % 4u);
    half4 m = tex.sample(smp, clamp(in.uv, 0.0, 1.0));
    float a = float(m.a) * g.scale + g.bias;
    bool hit = a > 0.5 && (p.x + 3u * p.y) % g.discard_mod == g.discard_hit;
    if (hit) {
        discard_fragment();
    }
    half4 s = tex.sample(smp, in.uv * 0.5);
    float w = fwidth(in.uv.x);
    half k = 0.0h;
    if (hit) {
        k = half(a) * half(g.scale) - half(w);
        k = k * k + half(g.bias) * k - 2.0h;
    }
    half3 mixed = s.rgb + base.rgb * half(g.bias);
    uint code = sfd_code(dst.r) + sfd_code(mixed.r) + (w > 1.0e6 || k > 1000.0h ? 7u : 1u);
    return sfd_color(code);
}

fragment half4 sfd_edge(SfdVertex in [[stage_in]], half4 dst [[color(0)]],
                        texture2d<half> tex [[texture(0)]], sampler smp [[sampler(0)]],
                        constant SfdGlobals &g [[buffer(0)]]) {
#pragma clang fp contract(off)
    uint2 p = uint2(in.position.xy);
    half4 base = tex.read(p % 4u);
    half4 m = tex.sample(smp, clamp(in.uv, 0.0, 1.0));
    float a = float(m.a) * g.scale + g.bias;
    if (a > 0.5 && (p.x + 3u * p.y) % g.discard_mod == g.discard_hit) {
        discard_fragment();
    }
    half4 s = tex.sample(smp, in.uv * 0.5);
    float w = fwidth(in.uv.x);
    float2 local = fmod(in.position.xy, g.cell);
    half k = 0.0h;
    if (local.x < 3.0 || local.y < 5.0 || local.x >= g.cell.x - 2.0 || local.y >= g.cell.y - 2.0) {
        k = half(a) * half(g.scale) - half(w);
        k = k * k + half(g.bias) * k - 2.0h;
    }
    half3 mixed = s.rgb + base.rgb * half(g.bias);
    uint code = sfd_code(dst.r) + sfd_code(mixed.r) + (w > 1.0e6 || k > 1000.0h ? 7u : 1u);
    return sfd_color(code);
}
