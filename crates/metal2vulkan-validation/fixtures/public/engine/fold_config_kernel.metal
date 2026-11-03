// Owned fixture, written for this project: a kernel uber shader in the shape of a compositor's
// fragment uber shader. Its function constants are collected into a program-scope `constant`
// struct (two per-layer records, each a present flag, a selector and four opcodes widened from a
// `uchar4` to a `uint4`, then a layer count), which the AIR's static initializer fills, and the
// kernel reads its configuration through a reference that `kGeneric` points at buffer 1 instead:
// the generic path, which runs the same program from a configuration known only at run time. Per
// thread it runs each present layer's selector and then its four opcodes, in loops whose trip
// counts are constant and whose counters index the configuration, and one read takes an index
// known only at run time. Thread i reads `in[i]` and writes `out[i]`; every operation is integer
// but one, a `float` multiply and add that is exact for every input.
//
// The loops: `fold_config` runs two layers and four opcodes each (constant trip counts).
#include <metal_stdlib>
using namespace metal;

constant bool kGeneric [[function_constant(0)]];
constant bool kHas0 [[function_constant(1)]];
constant bool kHas1 [[function_constant(2)]];
constant uchar kSel0 [[function_constant(3)]];
constant uchar kSel1 [[function_constant(4)]];
constant uchar4 kOps0 [[function_constant(5)]];
constant uchar4 kOps1 [[function_constant(6)]];
constant uchar kCount [[function_constant(7)]];

struct Layer {
	bool present;
	uchar sel;
	uint4 ops;
};

struct Config {
	Layer layer[2];
	uint count;
};

constant Config kConfig = {
	{{kHas0, kSel0, uint4(kOps0)}, {kHas1, kSel1, uint4(kOps1)}},
	kCount,
};

static uint op(uint code, uint v, uint k)
{
	switch (code) {
	case 0: return v;
	case 1: return v + k;
	case 2: return v * 3u + 1u;
	case 3: return v ^ (k << 3);
	case 4: return (v >> 1) | (v << 31);
	case 5: return v - k * 7u;
	case 6: return popcount(v) + v;
	case 7: return clz(v | 1u) * 5u + v;
	case 8: return as_type<uint>(float(v & 0xffffu) * 0.25f + 1.0f);
	case 9: return rotate(v, k + 1u);
	case 10: return mulhi(v, 0x9e3779b9u) + k;
	case 11: return (v & 0xff00ff00u) | ((v >> 8) & 0xffu) | ((v & 0xffu) << 16);
	case 12: return reverse_bits(v);
	case 13: return v * v + k;
	case 14: return max(v, k * 1000u) - min(v, k);
	default: return ~v;
	}
}

static uint select_fn(uint sel, uint v, uint gid)
{
	switch (sel) {
	case 0: return v;
	case 1: return v + gid;
	case 2: return v * (gid + 1u);
	case 3: return min(v, gid * 13u);
	default: return v ^ 0x5a5a5a5au;
	}
}

kernel void fold_config(device uint *out [[buffer(0)]], constant Config &dynamic [[buffer(1)]],
                        constant uint *in [[buffer(2)]], uint gid [[thread_position_in_grid]])
{
	constant Config &cfg = kGeneric ? dynamic : kConfig;
	uint v = in[gid];
	for (uint l = 0; l < 2; l++) {
		if (l < cfg.count && cfg.layer[l].present) {
			v = select_fn(cfg.layer[l].sel, v, gid);
			for (uint i = 0; i < 4; i++)
				v = op(cfg.layer[l].ops[i], v, i + l * 4u);
		}
	}
	// One read at an index known only at run time: the struct stays one object in memory.
	if (cfg.layer[gid & 1u].present)
		v += 0x100u;
	out[gid] = v;
}
