// Owned fixture: the functions the metal2vulkan-runtime crate's fixed_index_draws example runs
// (runtime/examples/fixed_index_draws.rs). Written from scratch for it; `scripts/regen-air-fixtures/
// regen-air-fixtures.sh --dir examples fixed_indices` compiles it. No translator sweep reads this
// directory.
#include <metal_stdlib>
using namespace metal;

// Where a draw's indices lie in the source, and what each is moved by.
struct FixArguments {
    uint first;
    uint count;
    uint base;
};

// A draw's 16-bit indices widened to 32 bits and moved by `base`, into a buffer of the draw's own,
// as a layer that fixes an index buffer up before each draw writes it; a restart index stays one.
kernel void fix_indices(device const ushort *source [[buffer(0)]],
                        device uint *fixed [[buffer(1)]],
                        constant FixArguments &arguments [[buffer(2)]],
                        uint index [[thread_position_in_grid]]) {
    if (index >= arguments.count) {
        return;
    }
    ushort value = source[arguments.first + index];
    fixed[index] = value == 0xffff ? 0xffffffffu : uint(value) + arguments.base;
}

// A vertex placed at its position in a buffer of positions, by its index.
vertex float4 placed_vertex(device const float2 *positions [[buffer(0)]],
                            uint id [[vertex_id]]) {
    return float4(positions[id], 0.0, 1.0);
}

// One color.
fragment float4 flat_fragment() {
    return float4(1.0, 0.5, 0.0, 1.0);
}
