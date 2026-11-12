// Owned fixture: a program-scope table of constexpr samplers read at an index only the run knows,
// as a renderer's kernels pick the sampler a texture slot names (`samplers[id >> 32]`), sampling a
// texture slot and a texture held in an argument buffer.
//
// Thread i samples row 0 of a 4 by 1 texture at x = xs[i] through samplers[which[i]] and writes
// the texel; the four samplers differ in their address mode (repeat, clamp to edge) and filter
// (nearest, linear), so the texel says which one sampled.
#include <metal_stdlib>
using namespace metal;

constant constexpr array<sampler, 4> samplers = {
    sampler(address::repeat, filter::nearest),
    sampler(address::clamp_to_edge, filter::nearest),
    sampler(address::repeat, filter::linear),
    sampler(address::clamp_to_edge, filter::linear),
};

kernel void sampler_table_slot(texture2d<float> t [[texture(0)]],
                               device float4 *out [[buffer(0)]],
                               device const uint *which [[buffer(1)]],
                               device const float *xs [[buffer(2)]],
                               uint i [[thread_position_in_grid]])
{
    out[i] = t.sample(samplers[which[i]], float2(xs[i], 0.5f));
}

struct Held {
    texture2d<float> t;
};

kernel void sampler_table_held(device const Held *held [[buffer(3)]],
                               device float4 *out [[buffer(0)]],
                               device const uint *which [[buffer(1)]],
                               device const float *xs [[buffer(2)]],
                               uint i [[thread_position_in_grid]])
{
    out[i] = held[0].t.sample(samplers[which[i]], float2(xs[i], 0.5f));
}
