// Owned fixture (written for this repository): pixel-coordinate samples and gathers at texel and
// block edges, our own MSL. One thread per coordinate, loop-free.
//
// Metal reads a pixel coordinate `x` as given: its nearest texel is `floor(x)`, its gather's block
// and its linear filter's pair start at `floor(x - 0.5)`, with the filter's weight
// `x - 0.5 - floor(x - 0.5)`, each moved by a texel offset and then addressed. Every kernel here
// reads one texture at the coordinate its thread is given (`uv[g]`), through one of the ways a
// pixel sampler reaches a texture:
//
// - a constexpr `coord::pixel` sampler of each filter, of clamp to edge, to zero, and to each
//   border colour, without an offset, with a constant one and with a dynamic one (`k`);
// - the runtime sampler at `[[sampler(0)]]`, without an offset and with a constant one;
// - the sampler an argument buffer holds beside its texture, at `[[buffer(3)]]`, alike;
// - a 1D texture, and a 2D array's layer 1.
//
// `address::clamp_to_zero` and `address::clamp_to_border` are one addressing mode of Metal's under
// two border colours, so the kernels spell them apart by the colour.
#include <metal_stdlib>
using namespace metal;

constexpr sampler pe_nearest_edge_s(filter::nearest, address::clamp_to_edge, coord::pixel);
constexpr sampler pe_nearest_zero_s(filter::nearest, address::clamp_to_zero, coord::pixel);
constexpr sampler pe_nearest_black_s(filter::nearest, address::clamp_to_border,
                                   border_color::opaque_black, coord::pixel);
constexpr sampler pe_nearest_white_s(filter::nearest, address::clamp_to_border,
                                   border_color::opaque_white, coord::pixel);
constexpr sampler pe_linear_edge_s(filter::linear, address::clamp_to_edge, coord::pixel);
constexpr sampler pe_linear_zero_s(filter::linear, address::clamp_to_zero, coord::pixel);
constexpr sampler pe_linear_black_s(filter::linear, address::clamp_to_border,
                                  border_color::opaque_black, coord::pixel);
constexpr sampler pe_linear_white_s(filter::linear, address::clamp_to_border,
                                  border_color::opaque_white, coord::pixel);

// A kernel of texture slot 0, a 2D texture: `out[g]` is `expr` at `p = uv[g]`, `k` the dynamic
// offset.
#define PE_2D(name, expr)                                                                          \
    kernel void name(texture2d<float> t [[texture(0)]], device const float2 *uv [[buffer(0)]],    \
                     device float4 *out [[buffer(1)]], constant int2 &k [[buffer(2)]],             \
                     uint g [[thread_position_in_grid]])                                           \
    {                                                                                              \
        float2 p = uv[g];                                                                          \
        out[g] = expr;                                                                             \
    }

PE_2D(pe_nearest_zero, t.sample(pe_nearest_zero_s, p))
PE_2D(pe_nearest_black, t.sample(pe_nearest_black_s, p))
PE_2D(pe_nearest_white, t.sample(pe_nearest_white_s, p))
PE_2D(pe_nearest_zero_offset, t.sample(pe_nearest_zero_s, p, int2(1, -1)))
PE_2D(pe_nearest_white_dynamic, t.sample(pe_nearest_white_s, p, k))
PE_2D(pe_nearest_edge_offset, t.sample(pe_nearest_edge_s, p, int2(-2, 3)))
PE_2D(pe_linear_edge, t.sample(pe_linear_edge_s, p))
PE_2D(pe_linear_zero, t.sample(pe_linear_zero_s, p))
PE_2D(pe_linear_black, t.sample(pe_linear_black_s, p))
PE_2D(pe_linear_white, t.sample(pe_linear_white_s, p))
PE_2D(pe_linear_zero_offset, t.sample(pe_linear_zero_s, p, int2(2, -3)))
PE_2D(pe_linear_black_dynamic, t.sample(pe_linear_black_s, p, k))
PE_2D(pe_gather_zero, t.gather(pe_nearest_zero_s, p, int2(0), component::x))
PE_2D(pe_gather_black, t.gather(pe_nearest_black_s, p, int2(0), component::y))
PE_2D(pe_gather_white_offset, t.gather(pe_linear_white_s, p, int2(2, -3), component::x))
PE_2D(pe_gather_edge_dynamic, t.gather(pe_nearest_edge_s, p, k, component::y))

// Through the runtime sampler at `[[sampler(0)]]`.
kernel void pe_slot_sample(texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]],
                           device const float2 *uv [[buffer(0)]], device float4 *out [[buffer(1)]],
                           uint g [[thread_position_in_grid]])
{
    out[g] = t.sample(s, uv[g]);
}

kernel void pe_slot_sample_offset(texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]],
                                  device const float2 *uv [[buffer(0)]],
                                  device float4 *out [[buffer(1)]],
                                  uint g [[thread_position_in_grid]])
{
    out[g] = t.sample(s, uv[g], int2(-1, 2));
}

kernel void pe_slot_gather(texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]],
                           device const float2 *uv [[buffer(0)]], device float4 *out [[buffer(1)]],
                           uint g [[thread_position_in_grid]])
{
    out[g] = t.gather(s, uv[g], int2(0), component::x);
}

// The texture and the sampler an argument buffer holds, bound at `[[buffer(3)]]`.
struct PeArguments {
    texture2d<float> t [[id(0)]];
    sampler s [[id(1)]];
};

kernel void pe_argument_sample(constant PeArguments &a [[buffer(3)]],
                               device const float2 *uv [[buffer(0)]],
                               device float4 *out [[buffer(1)]], uint g [[thread_position_in_grid]])
{
    out[g] = a.t.sample(a.s, uv[g]);
}

kernel void pe_argument_sample_offset(constant PeArguments &a [[buffer(3)]],
                                      device const float2 *uv [[buffer(0)]],
                                      device float4 *out [[buffer(1)]],
                                      uint g [[thread_position_in_grid]])
{
    out[g] = a.t.sample(a.s, uv[g], int2(-1, 2));
}

kernel void pe_argument_gather(constant PeArguments &a [[buffer(3)]],
                               device const float2 *uv [[buffer(0)]],
                               device float4 *out [[buffer(1)]], uint g [[thread_position_in_grid]])
{
    out[g] = a.t.gather(a.s, uv[g], int2(0), component::x);
}

// A 1D texture at the coordinate's `x`, and a 2D array's layer 1.
kernel void pe_linear_zero_1d(texture1d<float> t [[texture(0)]],
                              device const float2 *uv [[buffer(0)]],
                              device float4 *out [[buffer(1)]], uint g [[thread_position_in_grid]])
{
    out[g] = t.sample(pe_linear_zero_s, uv[g].x);
}

kernel void pe_nearest_white_array(texture2d_array<float> t [[texture(0)]],
                                   device const float2 *uv [[buffer(0)]],
                                   device float4 *out [[buffer(1)]],
                                   uint g [[thread_position_in_grid]])
{
    out[g] = t.sample(pe_nearest_white_s, uv[g], 1);
}

kernel void pe_linear_zero_array(texture2d_array<float> t [[texture(0)]],
                                 device const float2 *uv [[buffer(0)]],
                                 device float4 *out [[buffer(1)]],
                                 uint g [[thread_position_in_grid]])
{
    out[g] = t.sample(pe_linear_zero_s, uv[g], 1);
}
