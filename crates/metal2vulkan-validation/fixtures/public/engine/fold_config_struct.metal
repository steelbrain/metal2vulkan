// Owned fixture, written for this project: a fragment uber shader whose ten function constants
// are collected into a program-scope `constant` struct (two per-layer records, each a present
// flag, a texture-function selector and four colour opcodes; then a fetch flag, a blend selector,
// a destination selector and a layer count), which the shader reads its configuration from. The
// AIR's static initializer fills the struct, so a specialization reads constant fields of a
// constant global: once SCCP reads them, the paths the constants leave out are deleted, and the
// module comes to about the size of `fold_config_direct`, the same shader reading the constants
// directly. One read takes an index known only at run time, so the struct stays one object.
#include <metal_stdlib>
using namespace metal;

#ifdef BAKED
constant uint kTex0 = TEX0, kTex1 = TEX1, kBlend = BLEND, kDest = DEST;
constant bool kHas0 = HAS0, kHas1 = HAS1, kFetch = FETCH;
constant uchar4 kOps0 = uchar4(OPS0), kOps1 = uchar4(OPS1);
constant uchar kLayers = LAYERS;
#else
constant bool kHas0 [[function_constant(0)]];
constant bool kHas1 [[function_constant(1)]];
constant bool kFetch [[function_constant(2)]];
constant uchar kTex0 [[function_constant(3)]];
constant uchar kTex1 [[function_constant(4)]];
constant uchar kBlend [[function_constant(5)]];
constant uchar kDest [[function_constant(6)]];
constant uchar4 kOps0 [[function_constant(7)]];
constant uchar4 kOps1 [[function_constant(8)]];
constant uchar kLayers [[function_constant(9)]];
#endif

struct Layer {
	bool present;
	uchar tex;
	uchar4 ops;
};

struct Config {
	Layer layer[2];
	bool fetch;
	uchar blend;
	uchar dest;
	uchar layers;
};

#ifdef DIRECT
#define LAYER(i) ((i) == 0 ? Layer{kHas0, kTex0, kOps0} : Layer{kHas1, kTex1, kOps1})
#define CONFIG_FETCH kFetch
#define CONFIG_BLEND kBlend
#define CONFIG_DEST kDest
#define CONFIG_LAYERS kLayers
#else
constant Config kConfig = {
	{{kHas0, kTex0, kOps0}, {kHas1, kTex1, kOps1}},
	kFetch, kBlend, kDest, kLayers,
};
#define LAYER(i) kConfig.layer[(i)]
#define CONFIG_FETCH kConfig.fetch
#define CONFIG_BLEND kConfig.blend
#define CONFIG_DEST kConfig.dest
#define CONFIG_LAYERS kConfig.layers
#endif

struct In {
	float4 pos [[position]];
	float2 uv0;
	float2 uv1;
	float4 color;
};

struct Params {
	float4 k[16];
	float3x3 ycc;
	float gamma;
	float exposure;
};

static float4 sampleWith(uint fn, texture2d<float> t, sampler s, float2 uv, constant Params &p)
{
	switch (fn) {
	case 0: return t.sample(s, uv);
	case 1: return t.sample(s, uv, bias(p.k[0].x));
	case 2: {
		float4 a = 0;
		for (int y = -1; y <= 1; y++)
			for (int x = -1; x <= 1; x++)
				a += t.sample(s, uv, int2(x, y)) * (2.0f - abs(float(x))) * (2.0f - abs(float(y)));
		return a / 16.0f;
	}
	case 3: {
		const float2 size = float2(t.get_width(), t.get_height());
		const float2 c = uv * size - 0.5f, f = fract(c);
		const float2 w0 = f * (-0.5f + f * (1.0f - 0.5f * f));
		const float2 w1 = 1.0f + f * f * (-2.5f + 1.5f * f);
		const float2 w2 = f * (0.5f + f * (2.0f - 1.5f * f));
		const float2 w3 = f * f * (-0.5f + 0.5f * f);
		const float2 g0 = w0 + w1, g1 = w2 + w3;
		const float2 h0 = (w1 / g0 - 1.0f + floor(c) + 0.5f) / size;
		const float2 h1 = (w3 / g1 + 1.0f + floor(c) + 0.5f) / size;
		return g0.y * (g0.x * t.sample(s, float2(h0.x, h0.y)) + g1.x * t.sample(s, float2(h1.x, h0.y))) +
		       g1.y * (g0.x * t.sample(s, float2(h0.x, h1.y)) + g1.x * t.sample(s, float2(h1.x, h1.y)));
	}
	case 4: {
		const float4 v = t.sample(s, uv);
		return float4(p.ycc * (v.xyz - float3(0.0625f, 0.5f, 0.5f)), v.w);
	}
	case 5: {
		float4 a = 0;
		for (int y = -2; y <= 2; y++)
			for (int x = -2; x <= 2; x++)
				a += t.sample(s, uv, int2(x, y)) * exp(-float(x * x + y * y) * 0.3f);
		return a / 9.6f;
	}
	case 6: return t.read(uint2(uv * float2(t.get_width(), t.get_height())));
	default: {
		const float2 d = uv - 0.5f;
		const float r2 = dot(d, d);
		return t.sample(s, 0.5f + d * (1.0f + p.k[1].x * r2 + p.k[1].y * r2 * r2));
	}
	}
}

static float3 colourOp(uint op, float3 c, constant Params &p, uint i)
{
	const float4 k = p.k[2 + (i & 7)];
	switch (op) {
	case 0: return c;
	case 1: return c * k.xyz;
	case 2: return c + k.xyz;
	case 3: return pow(max(c, 0.0f), float3(p.gamma));
	case 4: return c / (c + k.x);
	case 5: return mix(float3(dot(c, float3(0.2126f, 0.7152f, 0.0722f))), c, k.y);
	case 6: return clamp(c, k.x, k.y);
	case 7: return c * exp2(p.exposure);
	case 8: {
		const float3 a = c * (c * 2.51f + 0.03f), b = c * (c * 2.43f + 0.59f) + 0.14f;
		return saturate(a / b);
	}
	case 9: {
		const float3 kk = float3(0.57735f);
		const float ca = cos(k.z), sa = sin(k.z);
		return c * ca + cross(kk, c) * sa + kk * dot(kk, c) * (1.0f - ca);
	}
	case 10: return select(c * 12.92f, 1.055f * pow(c, 1.0f / 2.4f) - 0.055f, c > 0.0031308f);
	case 11: return select(c / 12.92f, pow((c + 0.055f) / 1.055f, 2.4f), c > 0.04045f);
	case 12: return log2(max(c, 1e-5f)) * k.x + k.y;
	case 13: return exp2(c * k.x - k.y);
	case 14: return float3(c.z, c.y, c.x);
	default: return 1.0f - c;
	}
}

static float4 blendWith(uint fn, float4 s, float4 d)
{
	switch (fn) {
	case 0: return s;
	case 1: return s + d * (1.0f - s.a);
	case 2: return s * d;
	case 3: return s + d - s * d;
	case 4: return select(2.0f * s * d, 1.0f - 2.0f * (1.0f - s) * (1.0f - d), d > 0.5f);
	case 5: return min(s, d);
	case 6: return max(s, d);
	case 7: return select(d / max(1.0f - s, 1e-4f), 1.0f, s >= 1.0f);
	case 8: return select(1.0f - (1.0f - d) / max(s, 1e-4f), 0.0f, s <= 0.0f);
	case 9: return abs(d - s);
	case 10: return s + d - 2.0f * s * d;
	default: return float4(s.rgb * s.a + d.rgb * (1.0f - s.a), s.a + d.a * (1.0f - s.a));
	}
}

static float4 destination(uint fn, float4 c)
{
	switch (fn) {
	case 0: return c;
	case 1: return float4(c.rgb * c.a, c.a);
	case 2: return float4(c.rgb / max(c.a, 1e-4f), c.a);
	default: return float4(round(c * 255.0f) / 255.0f);
	}
}

// One layer: its texture, then its four colour opcodes (constant lanes, no loop).
static float4 layer(Layer l, float4 c, texture2d<float> t, sampler s, float2 uv, constant Params &p,
                    uint base)
{
	if (l.present)
		c *= sampleWith(l.tex, t, s, uv, p);
	float3 rgb = c.rgb;
	rgb = colourOp(l.ops.x, rgb, p, base + 0);
	rgb = colourOp(l.ops.y, rgb, p, base + 1);
	rgb = colourOp(l.ops.z, rgb, p, base + 2);
	rgb = colourOp(l.ops.w, rgb, p, base + 3);
	return float4(rgb, c.a);
}

struct Out {
	float4 color [[color(0)]];
};

fragment Out cfg_frag(In in [[stage_in]], texture2d<float> image0 [[texture(0)]],
                         texture2d<float> image1 [[texture(1)]], sampler s [[sampler(0)]],
                         constant Params &p [[buffer(0)]], float4 dst [[color(0)]])
{
	float4 c = in.color;
	if (CONFIG_LAYERS > 0)
		c = layer(LAYER(0), c, image0, s, in.uv0, p, 0);
	if (CONFIG_LAYERS > 1)
		c = layer(LAYER(1), c, image1, s, in.uv1, p, 4);
	// One read at an index known only at run time: the struct stays one object in memory.
	if (LAYER(uint(in.pos.x) & 1).present)
		c.a *= 0.5f;
	if (CONFIG_FETCH)
		c = blendWith(CONFIG_BLEND, c, dst);
	Out o;
	o.color = destination(CONFIG_DEST, c);
	return o;
}

vertex In cfg_vert(uint vid [[vertex_id]])
{
	In o;
	const float2 p = float2((vid << 1) & 2, vid & 2);
	o.pos = float4(p * 2.0f - 1.0f, 0.0f, 1.0f);
	o.uv0 = p;
	o.uv1 = p * 0.5f;
	o.color = 1.0f;
	return o;
}
