// Owned fixture: the kernels the metal2vulkan-runtime crate's dispatch tiles device tests run
// (src/tests/dispatch_tiles.rs). Written from scratch for them; `scripts/regen-air-fixtures/
// regen-air-fixtures.sh --dir examples dispatch_tiles` compiles it. No translator sweep reads this
// directory. None of them observes its threadgroup partition: each reads only the grid's builtins.
#include <metal_stdlib>
using namespace metal;

// Each texel of `source` into `destination` at the same position; a thread past the
// destination's size stores nothing. Image-only.
kernel void tiles_copy(texture2d<float, access::read> source [[texture(0)]],
                       texture2d<float, access::write> destination [[texture(1)]],
                       uint2 position [[thread_position_in_grid]]) {
    if (position.x >= destination.get_width() || position.y >= destination.get_height()) {
        return;
    }
    destination.write(source.read(position), position);
}

// `tiles_copy` over 3D textures. Image-only.
kernel void tiles_copy_deep(texture3d<float, access::read> source [[texture(0)]],
                            texture3d<float, access::write> destination [[texture(1)]],
                            uint3 position [[thread_position_in_grid]]) {
    if (position.x >= destination.get_width() || position.y >= destination.get_height() ||
        position.z >= destination.get_depth()) {
        return;
    }
    destination.write(source.read(position), position);
}

// Each thread's position (each coordinate plus one) and the grid's size, at the thread's own
// element of `marks`, the grid's elements in row order. Writes device memory: not image-only.
kernel void tiles_mark(device uint4 *marks [[buffer(0)]],
                       uint2 position [[thread_position_in_grid]],
                       uint2 grid [[threads_per_grid]]) {
    marks[position.y * grid.x + position.x] = uint4(position.x + 1, position.y + 1, grid.x, grid.y);
}

// `tiles_mark` over a grid of three dimensions: the grid's depth in the last word's high half.
kernel void tiles_mark_deep(device uint4 *marks [[buffer(0)]],
                            uint3 position [[thread_position_in_grid]],
                            uint3 grid [[threads_per_grid]]) {
    marks[(position.z * grid.y + position.y) * grid.x + position.x] =
        uint4(position.x + 1, position.y + 1, position.z + 1, (grid.z << 16) | (grid.y << 8) | grid.x);
}
