// The `uint2` overloads of `rasterization_rate_map_decoder`, both directions: a kernel over
// coordinates a buffer holds, and a fragment function over its own pixel.
//
// `rate_map_integer` maps `in[tid]` at the layer buffer 3 holds and writes the screen-to-physical
// answer and the physical-to-screen answer as one `uint4`. `rate_map_integer_pixel` maps its
// pixel (the integer part of its position) at the layer buffer 1 holds and returns both answers
// as its colour.
//
// Plain Metal's answers, measured on an Apple M3 Max under macOS 27.0.1 (the conformance probe
// `rate_map_integer_overloads`): the screen-to-physical answer is the `float2` decoder's at the
// pixel's centre, `c + 1/2`, converted to `uint`; the physical-to-screen answer is the bits of the
// `float` `(0 - offset) * reciprocal` of the entry of granule `c >> 5`, the coordinate's own value
// not read, so it is a `float`'s bits and not a screen coordinate.
#include <metal_stdlib>
using namespace metal;

kernel void rate_map_integer(constant rasterization_rate_map_data &map [[buffer(0)]],
                             device const uint2 *in [[buffer(1)]],
                             device uint4 *out [[buffer(2)]],
                             constant uint &layer [[buffer(3)]],
                             uint tid [[thread_position_in_grid]]) {
    rasterization_rate_map_decoder decoder(map);
    out[tid] = uint4(decoder.map_screen_to_physical_coordinates(in[tid], layer),
                     decoder.map_physical_to_screen_coordinates(in[tid], layer));
}

fragment uint4 rate_map_integer_pixel(float4 position [[position]],
                                      constant rasterization_rate_map_data &map [[buffer(0)]],
                                      constant uint &layer [[buffer(1)]]) {
    rasterization_rate_map_decoder decoder(map);
    uint2 pixel = uint2(position.xy);
    return uint4(decoder.map_screen_to_physical_coordinates(pixel, layer),
                 decoder.map_physical_to_screen_coordinates(pixel, layer));
}
