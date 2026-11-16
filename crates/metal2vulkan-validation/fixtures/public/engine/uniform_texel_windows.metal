// Owned fixture: kernels whose `constant` windows end at the bound of a uniform view's texel form
// and just past it, for the runtime test of that form (`tests/uniform_texels.rs`): `utw_near` reads
// c[0] and c[1023], a window of 4096 bytes, which a device that takes
// `DriverWorkaround::ExternalUniformReadsSkipACache` reads through a uniform texel buffer;
// `utw_far` reads c[0] and c[1024], a window of 4100 bytes, which keeps its uniform buffer there.
// Written for this repository; no third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

// c[0] and c[1023] into out.
kernel void utw_near(constant uint *c [[buffer(0)]], device uint *out [[buffer(1)]]) {
    out[0] = c[0];
    out[1] = c[1023];
}

// c[0] and c[1024] into out.
kernel void utw_far(constant uint *c [[buffer(0)]], device uint *out [[buffer(1)]]) {
    out[0] = c[0];
    out[1] = c[1024];
}
