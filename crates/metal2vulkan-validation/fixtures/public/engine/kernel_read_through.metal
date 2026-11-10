// metal-std: metal4.1
// metal-macos-min: 27.0
// Owned fixture: Metal 4.1's `read(sampler s, coord[, offset][, lod])` (MSL 4.1 §6.13) through
// samplers of each state family. Thread i reads at p[i] (x, y, then the layer or z, then the level)
// and stores the texel:
//
//  * rt_repeat, rt_mirrored_offset (offset (-2, 1)), rt_zero, rt_border_ob, rt_border_ow:
//    a texture2d<float> through a constexpr sampler of that address mode on every axis;
//  * rt_mixed: clamp_to_border (opaque white) on s, repeat on t;
//  * rt_mip: a nearest mip filter; rt_mip_zero: clamp_to_zero with a linear mip filter;
//  * rt_filtered: a linear filter, pixel coordinates and a LOD clamp, which change nothing;
//  * rt_array: a texture2d_array<float> through clamp_to_zero with a nearest mip filter;
//  * rt_volume: a texture3d<float>, repeat on s, clamp_to_edge on t, clamp_to_border (opaque
//    white) on r;
//  * rt_line: a texture1d<float> through clamp_to_zero;
//  * rt_rw: a texture2d<float, access::read_write> through clamp_to_border (opaque white);
//  * rt_buffer: a texture_buffer<float> through mirrored_repeat; rt_buffer_ow through
//    clamp_to_border (opaque white);
//  * rt_int: a texture2d<int> through clamp_to_border (opaque white), stored as int4;
//  * rt_half: a texture2d<half> through clamp_to_border (opaque white), stored as half4;
//  * rt_alpha: a texture2d<float> through clamp_to_zero, its alpha alone stored;
//  * rt_arg, rt_arg_buffer, rt_arg_volume: a texture2d<float> (offset (1, -1)), a
//    texture_buffer<float>, a texture3d<float> through the runtime sampler [[sampler(0)]];
//  * rt_table: a texture2d<float> through an argument buffer's sampler;
//  * rt_heap_zero, rt_heap_ow, rt_heap_arg: an argument buffer's texture2d<float> through
//    clamp_to_zero, clamp_to_border (opaque white), and the runtime sampler [[sampler(0)]].

#include <metal_stdlib>
using namespace metal;

struct Samplers {
    sampler s;
};

struct Textures {
    texture2d<float> t;
};

#define PLANE(name, decl)                                                                      \
    kernel void name(texture2d<float> t [[texture(0)]], device const int4 *p [[buffer(0)]],    \
                     device float4 *out [[buffer(1)]], uint i [[thread_position_in_grid]]) {    \
        constexpr sampler s decl;                                                              \
        out[i] = t.read(s, p[i].xy, int2(0), uint(p[i].w));                                    \
    }

PLANE(rt_repeat, (address::repeat))
PLANE(rt_zero, (address::clamp_to_zero))
PLANE(rt_border_ob, (address::clamp_to_border, border_color::opaque_black))
PLANE(rt_border_ow, (address::clamp_to_border, border_color::opaque_white))
PLANE(rt_mixed, (s_address::clamp_to_border, t_address::repeat, border_color::opaque_white))
PLANE(rt_mip, (mip_filter::nearest))
PLANE(rt_mip_zero, (address::clamp_to_zero, mip_filter::linear))
PLANE(rt_filtered, (filter::linear, coord::pixel, lod_clamp(1.0f, 1.0f)))

kernel void rt_mirrored_offset(texture2d<float> t [[texture(0)]], device const int4 *p [[buffer(0)]],
                               device float4 *out [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    constexpr sampler s(address::mirrored_repeat);
    out[i] = t.read(s, p[i].xy, int2(-2, 1), uint(p[i].w));
}

kernel void rt_array(texture2d_array<float> t [[texture(0)]], device const int4 *p [[buffer(0)]],
                     device float4 *out [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    constexpr sampler s(address::clamp_to_zero, mip_filter::nearest);
    out[i] = t.read(s, p[i].xy, uint(p[i].z), int2(0), uint(p[i].w));
}

kernel void rt_volume(texture3d<float> t [[texture(0)]], device const int4 *p [[buffer(0)]],
                      device float4 *out [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    constexpr sampler s(s_address::repeat, t_address::clamp_to_edge, r_address::clamp_to_border,
                        border_color::opaque_white);
    out[i] = t.read(s, p[i].xyz, int3(0), uint(p[i].w));
}

kernel void rt_line(texture1d<float> t [[texture(0)]], device const int4 *p [[buffer(0)]],
                    device float4 *out [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    constexpr sampler s(address::clamp_to_zero);
    out[i] = t.read(s, p[i].x);
}

kernel void rt_rw(texture2d<float, access::read_write> t [[texture(0)]],
                  device const int4 *p [[buffer(0)]], device float4 *out [[buffer(1)]],
                  uint i [[thread_position_in_grid]]) {
    constexpr sampler s(address::clamp_to_border, border_color::opaque_white);
    out[i] = t.read(s, p[i].xy);
}

kernel void rt_buffer(texture_buffer<float> t [[texture(0)]], device const int4 *p [[buffer(0)]],
                      device float4 *out [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    constexpr sampler s(address::mirrored_repeat);
    out[i] = t.read(s, p[i].x);
}

kernel void rt_buffer_ow(texture_buffer<float> t [[texture(0)]], device const int4 *p [[buffer(0)]],
                         device float4 *out [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    constexpr sampler s(address::clamp_to_border, border_color::opaque_white);
    out[i] = t.read(s, p[i].x);
}

kernel void rt_int(texture2d<int> t [[texture(0)]], device const int4 *p [[buffer(0)]],
                   device int4 *out [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    constexpr sampler s(address::clamp_to_border, border_color::opaque_white);
    out[i] = t.read(s, p[i].xy, int2(0), uint(p[i].w));
}

kernel void rt_half(texture2d<half> t [[texture(0)]], device const int4 *p [[buffer(0)]],
                    device half4 *out [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    constexpr sampler s(address::clamp_to_border, border_color::opaque_white);
    out[i] = t.read(s, p[i].xy, int2(0), uint(p[i].w));
}

kernel void rt_alpha(texture2d<float> t [[texture(0)]], device const int4 *p [[buffer(0)]],
                     device float *out [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    constexpr sampler s(address::clamp_to_zero);
    out[i] = t.read(s, p[i].xy, int2(0), uint(p[i].w)).a;
}

kernel void rt_arg(texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]],
                   device const int4 *p [[buffer(0)]], device float4 *out [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
    out[i] = t.read(s, p[i].xy, int2(1, -1), uint(p[i].w));
}

kernel void rt_arg_buffer(texture_buffer<float> t [[texture(0)]], sampler s [[sampler(0)]],
                          device const int4 *p [[buffer(0)]], device float4 *out [[buffer(1)]],
                          uint i [[thread_position_in_grid]]) {
    out[i] = t.read(s, p[i].x);
}

kernel void rt_arg_volume(texture3d<float> t [[texture(0)]], sampler s [[sampler(0)]],
                          device const int4 *p [[buffer(0)]], device float4 *out [[buffer(1)]],
                          uint i [[thread_position_in_grid]]) {
    out[i] = t.read(s, p[i].xyz, int3(0), uint(p[i].w));
}

kernel void rt_table(texture2d<float> t [[texture(0)]], constant Samplers &a [[buffer(2)]],
                     device const int4 *p [[buffer(0)]], device float4 *out [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {
    out[i] = t.read(a.s, p[i].xy, int2(0), uint(p[i].w));
}

kernel void rt_heap_zero(constant Textures &a [[buffer(2)]], device const int4 *p [[buffer(0)]],
                         device float4 *out [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    constexpr sampler s(address::clamp_to_zero);
    out[i] = a.t.read(s, p[i].xy, int2(0), uint(p[i].w));
}

kernel void rt_heap_ow(constant Textures &a [[buffer(2)]], device const int4 *p [[buffer(0)]],
                       device float4 *out [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    constexpr sampler s(address::clamp_to_border, border_color::opaque_white);
    out[i] = a.t.read(s, p[i].xy, int2(0), uint(p[i].w));
}

kernel void rt_heap_arg(constant Textures &a [[buffer(2)]], sampler s [[sampler(0)]],
                        device const int4 *p [[buffer(0)]], device float4 *out [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
    out[i] = a.t.read(s, p[i].xy, int2(0), uint(p[i].w));
}
