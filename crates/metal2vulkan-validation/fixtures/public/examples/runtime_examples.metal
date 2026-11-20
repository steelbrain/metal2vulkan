// Owned fixture: the functions the harness's authored render workloads and metal2vulkan-runtime's draw
// tests run. Written from scratch for them; `scripts/regen-air-fixtures/
// regen-air-fixtures.sh --dir examples` compiles it. No translator sweep reads this directory.
#include <metal_stdlib>
using namespace metal;

// compute_dispatch, indirect_command_buffer and adopted_buffer:
// output[i] = input[i] * scale + i.
kernel void scale_and_offset(device const float *input [[buffer(0)]],
                             device float *output [[buffer(1)]],
                             constant float &scale [[buffer(2)]],
                             uint index [[thread_position_in_grid]]) {
    output[index] = input[index] * scale + float(index);
}

// render_to_texture: a triangle list whose vertices read their positions and colors from two
// buffers by vertex ID; each triangle takes one color.
struct Varyings {
    float4 position [[position]];
    float4 color [[flat]];
};

vertex Varyings colored_vertex(device const float2 *positions [[buffer(0)]],
                               device const float4 *colors [[buffer(1)]],
                               uint id [[vertex_id]]) {
    Varyings out;
    out.position = float4(positions[id], 0.0, 1.0);
    out.color = colors[id];
    return out;
}

fragment float4 colored_fragment(Varyings in [[stage_in]]) {
    return in.color;
}

// draw's point sizes: colored_vertex's points, each vertex's `[[point_size]]` read from a buffer.
struct SizedVaryings {
    float4 position [[position]];
    float4 color [[flat]];
    float size [[point_size]];
};

vertex SizedVaryings sized_vertex(device const float2 *positions [[buffer(0)]],
                                  device const float4 *colors [[buffer(1)]],
                                  device const float *sizes [[buffer(2)]],
                                  uint id [[vertex_id]]) {
    SizedVaryings out;
    out.position = float4(positions[id], 0.0, 1.0);
    out.color = colors[id];
    out.size = sizes[id];
    return out;
}

// The runtime's made_at_creation lane: each pixel of the triangles sampled from a half texture
// through a linear filter, at its position over an 8-texel side.
fragment half4 sampled_fragment(Varyings in [[stage_in]],
                                texture2d<half> source [[texture(0)]]) {
    constexpr sampler linear(filter::linear);
    return source.sample(linear, in.position.xy / 8.0);
}

// The new runtime's attachment reads: a draw that reads, through a texture, the attachment its
// own pass renders. Each pixel the attachment's texel at the fragment's own pixel, plus the
// draw's color.
fragment float4 own_pixel_fragment(Varyings in [[stage_in]],
                                   texture2d<float, access::read> target [[texture(0)]]) {
    return target.read(uint2(in.position.xy)) + in.color;
}

// Each pixel the attachment's first texel, plus the draw's color: a read of another pixel.
fragment float4 first_texel_fragment(Varyings in [[stage_in]],
                                     texture2d<float, access::read> target [[texture(0)]]) {
    return target.read(uint2(0, 0)) + in.color;
}

// Each pixel the texture's first level at the fragment's place in a 4-texel side, through a view
// of every level while the pass renders the second.
fragment float4 first_level_fragment(Varyings in [[stage_in]],
                                     texture2d<float> chain [[texture(0)]]) {
    constexpr sampler nearest(filter::nearest, mip_filter::nearest);
    return chain.sample(nearest, in.position.xy / 4.0, level(0.0));
}

// heap_placement: each texel of `source` with its color channels inverted, into `destination`.
kernel void invert_texels(texture2d<float, access::read> source [[texture(0)]],
                          texture2d<float, access::write> destination [[texture(1)]],
                          uint2 position [[thread_position_in_grid]]) {
    float4 texel = source.read(position);
    destination.write(float4(1.0 - texel.rgb, texel.a), position);
}
