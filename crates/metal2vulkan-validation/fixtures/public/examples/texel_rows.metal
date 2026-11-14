// Owned fixture: the kernels the metal2vulkan-runtime crate's texel rows device tests run
// (src/tests/texel_rows_device.rs). Written from scratch for them; `scripts/regen-air-fixtures/
// regen-air-fixtures.sh --dir examples texel_rows` compiles it. No translator sweep reads this
// directory.
#include <metal_stdlib>
using namespace metal;

// Each texel of `texels` with its color channels inverted, in place; a thread past the
// texture's size stores nothing.
kernel void rows_invert(texture2d<float, access::read_write> texels [[texture(0)]],
                        uint2 position [[thread_position_in_grid]]) {
    if (position.x >= texels.get_width() || position.y >= texels.get_height()) {
        return;
    }
    float4 texel = texels.read(position);
    texels.write(float4(1.0 - texel.rgb, texel.a), position);
}

// Each texel of `source` (Metal's out-of-bounds texel past it) into `output`, `columns` elements
// a grid row.
kernel void rows_read(texture2d<float, access::read> source [[texture(0)]],
                      device float4 *output [[buffer(0)]],
                      constant uint &columns [[buffer(1)]],
                      uint2 position [[thread_position_in_grid]]) {
    output[position.y * columns + position.x] = source.read(position);
}

// Each element of `input`, `columns` elements a grid row, into texel `position` of
// `destination` (dropped past it).
kernel void rows_write(device const float4 *input [[buffer(0)]],
                       constant uint &columns [[buffer(1)]],
                       texture2d<float, access::write> destination [[texture(0)]],
                       uint2 position [[thread_position_in_grid]]) {
    destination.write(input[position.y * columns + position.x], position);
}

// The texel of `source` at each texel's center, sampled through a nearest filter, into
// `output`, `columns` elements a grid row.
kernel void rows_sample(texture2d<float> source [[texture(0)]],
                        device float4 *output [[buffer(0)]],
                        constant uint &columns [[buffer(1)]],
                        uint2 position [[thread_position_in_grid]]) {
    constexpr sampler nearest(filter::nearest, coord::pixel);
    output[position.y * columns + position.x] = source.sample(nearest, float2(position) + 0.5);
}
