// metal-std: metal4.1
// metal-macos-min: 27.0
// Owned fixture: Metal 4.1's block reads, which Metal's compiler spells
// `air.block_read_{texture,depth}_2d[_array].<T>`. Each kernel's thread i reads the block at
// p[i].xy and level p[i].z and stores one of its two texels, the first where bit 0 of p[i].w is
// clear and the second where it is set (an array's layer is p[i].w >> 1):
//
//  * br_2x1: a texture2d<float> block of 2 x 1 through the read sampler (`block_read(coord,
//    offset, lod)`);
//  * br_1x2_offset: a texture2d<uint> block of 1 x 2 at the offset (-1, 2);
//  * br_edge: a texture2d<float> block of 2 x 1 through a constexpr clamp-to-edge sampler at the
//    offset (1, 0);
//  * br_edge_1x2: a texture2d<int> block of 1 x 2 through the clamp-to-edge sampler;
//  * br_half: a texture2d<half> block of 2 x 1, stored as half4;
//  * br_array: a texture2d_array<float> block of 1 x 2;
//  * br_depth: a depth2d<float> block of 2 x 1, stored as its float;
//  * br_sparse: a texture2d<float> sparse block of 2 x 1, stored with its alpha replaced by 2.0
//    where the texel is resident and 3.0 where it is not;
//  * br_rw: a texture2d<float, access::read_write> block of 2 x 1.
//
// Written for this repository; no third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

constexpr sampler edge(address::clamp_to_edge);

kernel void br_2x1(texture2d<float, access::read> t [[texture(0)]],
                   device const int4 *p [[buffer(0)]], device float4 *out [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
    int4 q = p[i];
    auto v = t.block_read<2, 1>(q.xy, int2(0), uint(q.z));
    out[i] = (q.w & 1) ? v.value<1, 0>() : v.value<0, 0>();
}

kernel void br_1x2_offset(texture2d<uint, access::read> t [[texture(0)]],
                          device const int4 *p [[buffer(0)]], device uint4 *out [[buffer(1)]],
                          uint i [[thread_position_in_grid]]) {
    int4 q = p[i];
    auto v = t.block_read<1, 2>(q.xy, int2(-1, 2), uint(q.z));
    out[i] = (q.w & 1) ? v.value<0, 1>() : v.value<0, 0>();
}

kernel void br_edge(texture2d<float, access::read> t [[texture(0)]],
                    device const int4 *p [[buffer(0)]], device float4 *out [[buffer(1)]],
                    uint i [[thread_position_in_grid]]) {
    int4 q = p[i];
    auto v = t.block_read<2, 1>(edge, q.xy, int2(1, 0), uint(q.z));
    out[i] = (q.w & 1) ? v.value<1, 0>() : v.value<0, 0>();
}

kernel void br_edge_1x2(texture2d<int, access::read> t [[texture(0)]],
                        device const int4 *p [[buffer(0)]], device int4 *out [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
    int4 q = p[i];
    auto v = t.block_read<1, 2>(edge, q.xy, int2(0), uint(q.z));
    out[i] = (q.w & 1) ? v.value<0, 1>() : v.value<0, 0>();
}

kernel void br_half(texture2d<half, access::read> t [[texture(0)]],
                    device const int4 *p [[buffer(0)]], device half4 *out [[buffer(1)]],
                    uint i [[thread_position_in_grid]]) {
    int4 q = p[i];
    auto v = t.block_read<2, 1>(q.xy, int2(0), uint(q.z));
    out[i] = (q.w & 1) ? v.value<1, 0>() : v.value<0, 0>();
}

kernel void br_array(texture2d_array<float, access::read> t [[texture(0)]],
                     device const int4 *p [[buffer(0)]], device float4 *out [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {
    int4 q = p[i];
    auto v = t.block_read<1, 2>(q.xy, uint(q.w >> 1), int2(0), uint(q.z));
    out[i] = (q.w & 1) ? v.value<0, 1>() : v.value<0, 0>();
}

kernel void br_depth(depth2d<float, access::read> t [[texture(0)]],
                     device const int4 *p [[buffer(0)]], device float *out [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {
    int4 q = p[i];
    auto v = t.block_read<2, 1>(q.xy, int2(0), uint(q.z));
    out[i] = (q.w & 1) ? v.value<1, 0>() : v.value<0, 0>();
}

kernel void br_sparse(texture2d<float, access::read> t [[texture(0)]],
                      device const int4 *p [[buffer(0)]], device float4 *out [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) {
    int4 q = p[i];
    auto v = t.sparse_block_read<2, 1>(q.xy, int2(0), uint(q.z));
    bool second = q.w & 1;
    float4 c = second ? v.value<1, 0>() : v.value<0, 0>();
    bool resident = second ? v.resident<1, 0>() : v.resident<0, 0>();
    out[i] = float4(c.xyz, resident ? 2.0 : 3.0);
}

kernel void br_rw(texture2d<float, access::read_write> t [[texture(0)]],
                  device const int4 *p [[buffer(0)]], device float4 *out [[buffer(1)]],
                  uint i [[thread_position_in_grid]]) {
    int4 q = p[i];
    auto v = t.block_read<2, 1>(q.xy, int2(0), uint(q.z));
    out[i] = (q.w & 1) ? v.value<1, 0>() : v.value<0, 0>();
}
