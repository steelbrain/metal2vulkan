// Owned fixture: a kernel that reads more buffers than a small profile's per-stage
// storage-buffer limit (vk12-floor allows four). Five slots are read only: two
// `device` arrays and a `device` array indexed by the thread (`Whole` windows), a `constant` array
// indexed by the thread (`Whole`), and a `device` struct reference (`Bounded`, 8 bytes); one slot
// is written. A profile with device addresses reaches the slots past the limit by their
// addresses; Profile F, which has none, refuses the kernel. Bind every buffer with at least as
// many words as threads: out[i] = a[i] + b[i] * p.scale + c[i] + (d[i] ^ p.bias). Loop-free.
//
// `descriptor_limit_atomic_load` has five storage slots too, one of them a `device` atomic that
// is only loaded (a 4-byte `Bounded` window, read only). out[i] = i < n ? a[i] + b[i] : d[i],
// where n is the atomic's value. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Params {
    uint scale;
    uint bias;
};

kernel void descriptor_limit_uniform_views(device const uint *a [[buffer(0)]],
                                           device const uint *b [[buffer(1)]],
                                           constant uint *c [[buffer(2)]],
                                           device const Params &p [[buffer(3)]],
                                           device const uint *d [[buffer(4)]],
                                           device uint *out [[buffer(5)]],
                                           uint i [[thread_position_in_grid]]) {
    out[i] = a[i] + b[i] * p.scale + c[i] + (d[i] ^ p.bias);
}

kernel void descriptor_limit_atomic_load(device const uint *a [[buffer(0)]],
                                         device const uint *b [[buffer(1)]],
                                         device atomic_uint &limit [[buffer(2)]],
                                         device const uint *d [[buffer(3)]],
                                         device uint *out [[buffer(4)]],
                                         uint i [[thread_position_in_grid]]) {
    uint n = atomic_load_explicit(&limit, memory_order_relaxed);
    out[i] = i < n ? a[i] + b[i] : d[i];
}
