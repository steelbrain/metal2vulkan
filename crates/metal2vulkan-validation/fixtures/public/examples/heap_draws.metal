// Owned fixture: the functions metal2vulkan-runtime's copy-over tests draw with
// (src/tests/copy_over.rs): one triangle over the whole target, and fragments that read a texture
// under them, directly or through an argument buffer, or write two colors, and a kernel that reads
// a texture through an argument buffer. Written from scratch for it;
// `scripts/regen-air-fixtures/regen-air-fixtures.sh --dir examples heap_draws` compiles it. No
// translator sweep reads this directory.
#include <metal_stdlib>
using namespace metal;

struct Covered {
    float4 position [[position]];
};

// Vertex `id` of one triangle that covers the whole target: (-1, -1), (3, -1) and (-1, 3).
vertex Covered cover_vertex(uint id [[vertex_id]]) {
    float2 corner = float2(float((id << 1) & 2), float(id & 2));
    Covered out;
    out.position = float4(corner * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

// The texel of `source` under each fragment.
fragment float4 read_fragment(Covered in [[stage_in]],
                              texture2d<float, access::read> source [[texture(0)]]) {
    return source.read(uint2(in.position.xy));
}

struct TwoColors {
    float4 first [[color(0)]];
    float4 second [[color(1)]];
};

// One color into each attachment: for a pipeline whose first attachment writes nothing.
fragment TwoColors two_colors_fragment(Covered in [[stage_in]]) {
    TwoColors out;
    out.first = float4(1.0, 0.0, 1.0, 1.0);
    out.second = float4(0.25, 0.5, 0.75, 1.0);
    return out;
}

// The texture the functions below read, at field 0 of their argument buffer: a heap's texture an
// encoder made resident with `useHeap:`, bound at no texture slot.
struct HeapTexture {
    texture2d<float, access::read> source [[id(0)]];
};

// The texel of the argument buffer's texture under each fragment.
fragment float4 read_argument_fragment(Covered in [[stage_in]],
                                       constant HeapTexture &arguments [[buffer(0)]]) {
    return arguments.source.read(uint2(in.position.xy));
}

// Each texel of the argument buffer's texture into `out`, packed `RGBA8Unorm`, rows packed.
kernel void read_argument_kernel(constant HeapTexture &arguments [[buffer(0)]],
                                 device uint *out [[buffer(1)]],
                                 uint2 at [[thread_position_in_grid]]) {
    uint width = arguments.source.get_width();
    out[at.y * width + at.x] = pack_float_to_unorm4x8(arguments.source.read(at));
}
