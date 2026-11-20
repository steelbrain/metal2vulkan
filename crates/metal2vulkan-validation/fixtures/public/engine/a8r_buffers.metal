// The texture-buffer and argument-buffer functions of the `A8Unorm` lane, our own MSL: kernels
// that read (as `float` and as
// `half`), write, and read and write a texture buffer at every texel, and one that reads and writes
// the 8 x 8 storage textures an argument buffer (buffer 2) holds.
#include <metal_stdlib>
using namespace metal;

kernel void a8r_buffer_read(texture_buffer<float, access::read> t [[texture(0)]],
                            device float4 *out [[buffer(1)]], uint g [[thread_position_in_grid]]) {
    out[g] = t.read(g);
}

kernel void a8r_buffer_read_half(texture_buffer<half, access::read> t [[texture(0)]],
                                 device float4 *out [[buffer(1)]],
                                 uint g [[thread_position_in_grid]]) {
    out[g] = float4(t.read(g));
}

kernel void a8r_buffer_write(texture_buffer<float, access::write> t [[texture(0)]],
                             constant float4 *color [[buffer(0)]],
                             uint g [[thread_position_in_grid]]) {
    t.write(color[0], g);
}

kernel void a8r_buffer_read_write(texture_buffer<float, access::read_write> t [[texture(0)]],
                                  constant float4 *color [[buffer(0)]],
                                  device float4 *out [[buffer(1)]],
                                  uint g [[thread_position_in_grid]]) {
    out[g] = t.read(g);
    t.write(color[0], g);
}

struct A8rHeld {
    texture2d<float, access::read_write> rw;
    texture2d<float, access::write> w;
};

kernel void a8r_held(constant A8rHeld &held [[buffer(2)]], constant float4 *color [[buffer(0)]],
                     device float4 *out [[buffer(1)]], uint2 g [[thread_position_in_grid]]) {
    out[g.y * 8 + g.x] = held.rw.read(g);
    held.rw.write(color[0], g);
    held.w.write(color[1], g);
}
