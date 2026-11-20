// Owned fixture: buffers a kernel reaches through a device address, one kernel a piece, for the
// runtime's device test tests/argument_tier.rs and the translator's identity-domain tests:
// - ab_sum: out[i] = args.a[i] + args.b[i], a and b the two pointer members of an argument
//   buffer ([[id(0)]], [[id(1)]]), each the gpuAddress of a buffer an argument encoder wrote;
// - ab_address: out[i] = held.p[i], p a gpuAddress the CPU wrote into a constant buffer's bytes;
// - ab_slot_address: out[i] is the address of element i of buffer 0 as an integer, and that
//   element is incremented through the same pointer (a slot pointer converted to an integer).
#include <metal_stdlib>
using namespace metal;

struct Args {
    device const uint *a [[id(0)]];
    device const uint *b [[id(1)]];
};

kernel void ab_sum(constant Args &args [[buffer(0)]], device uint *out [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
    out[i] = args.a[i] + args.b[i];
}

struct Held {
    device const uint *p;
};

kernel void ab_address(constant Held &held [[buffer(0)]], device uint *out [[buffer(1)]],
                       uint i [[thread_position_in_grid]]) {
    out[i] = held.p[i];
}

kernel void ab_slot_address(device uint *buf [[buffer(0)]], device ulong *out [[buffer(1)]],
                            uint i [[thread_position_in_grid]]) {
    device uint *p = buf + i;
    out[i] = ulong(reinterpret_cast<uintptr_t>(p));
    *p += 1;
}
