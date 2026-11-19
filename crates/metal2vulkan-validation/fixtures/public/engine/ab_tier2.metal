// Owned fixture: what Metal's argument buffers tier 2 adds over tier 1 beside writable textures
// (a8r_buffers' a8r_held holds those), one kernel a piece, for the runtime's device test
// tests/argument_tier.rs:
// - ab_copy: the GPU writes an argument buffer, copying a texture member from one argument buffer
//   to another, device to device, as tier 2's mutable argument buffers allow;
// - ab_read: reads texel (0, 0) through an argument buffer's texture member, whether the CPU wrote
//   the buffer, the GPU wrote it (ab_copy) or it is a Private buffer a blit filled;
// - ab_indexed: thread g reads texel (0, 0) of the texture member of element g of an array of
//   argument buffers that a pointer member of the bound one reaches (pointer indexing, and an
//   argument buffer that points at others).
#include <metal_stdlib>
using namespace metal;

struct Held {
    texture2d<float> t [[id(0)]];
};

kernel void ab_copy(device const Held *from [[buffer(0)]], device Held *to [[buffer(1)]]) {
    to->t = from->t;
}

kernel void ab_read(device const Held &held [[buffer(0)]], device float4 *out [[buffer(1)]]) {
    out[0] = held.t.read(uint2(0));
}

struct Table {
    device const Held *held [[id(0)]];
};

kernel void ab_indexed(device const Table &table [[buffer(0)]], device float4 *out [[buffer(1)]],
                       uint g [[thread_position_in_grid]]) {
    out[g] = table.held[g].t.read(uint2(0));
}
