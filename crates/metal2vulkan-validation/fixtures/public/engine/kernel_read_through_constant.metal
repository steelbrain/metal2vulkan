// A kernel whose parameter block is read through a `constant` pointer, its buffer node rewritten
// to the one Apple's compiler writes for a by-value argument it passes as a constant pointer: a
// read-only `device` buffer of one `char` (`air.address_space 1`, `air.arg_type_size 1`), as a
// Core Image kernel of macOS 13.7's frameworks declares its parameter block. MSL has no spelling
// for that node; the typed text is rewritten (`air-by-value-buffer`) and reassembled with
// `metal-as`, which Metal takes (`abi::check`'s `read_through_constant`).
//
// air-by-value-buffer: p
#include <metal_stdlib>
using namespace metal;

struct Params {
    float4 rect;
    float4x4 m;
    float3x3 n;
    float a, b, c, d, e;
};

kernel void read_through_constant(constant Params *p [[buffer(0)]],
                                  device float4 *out [[buffer(1)]],
                                  uint i [[thread_position_in_grid]]) {
    out[i] = p->rect + p->m[i & 3] * p->a + float4(p->n[i % 3], p->e);
}
