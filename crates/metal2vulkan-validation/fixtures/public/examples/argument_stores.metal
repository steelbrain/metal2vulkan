// Owned fixture: the kernel the metal2vulkan-validation crate's heap textures lane runs
// (src/runtime_lane/heap_textures.rs) to write a buffer that none of its bindings names. Written
// from scratch for it; `scripts/regen-air-fixtures/regen-air-fixtures.sh --dir examples
// argument_stores` compiles it. No translator sweep reads this directory.
#include <metal_stdlib>
using namespace metal;

// An argument buffer that holds the address of the buffer the kernel writes.
struct Target {
    device float *words;
};

// `base + index` into word `index` of the buffer whose address `target` holds, which the
// encoder makes resident for writing (`useResource:usage:`).
kernel void store_through_argument(constant Target &target [[buffer(0)]],
                                   constant float &base [[buffer(1)]],
                                   uint index [[thread_position_in_grid]]) {
    target.words[index] = base + float(index);
}
