// Owned fixture: MSL 3.2's program-scope buffer bindings. The kernel reads a constant struct and a
// constant word table bound at program scope, and writes through its own `[[buffer(0)]]` and
// through a program-scope `device uint *` bound at the same index, which Metal fills from the one
// buffer bound there (measured on an M3 Max: a global and a parameter of one type at one index
// read the one buffer).
#include <metal_stdlib>
using namespace metal;

struct Scale {
    float4 gain;
    uint bias;
};

constant Scale *constant psb_scale [[buffer(1)]];
constant uint *constant psb_words [[buffer(2)]];
device uint *constant psb_out [[buffer(0)]];

// Word i of out is word i of the table (word i + 32 on odd threads) times 3 plus the bias; word
// i + 64 is the bits of gain[i & 3] scaled by that word, written through the bound global.
kernel void program_scope_buffers(device uint *out [[buffer(0)]],
                                  uint i [[thread_position_in_grid]]) {
    constant uint *words = (i & 1u) != 0u ? psb_words + 32 : psb_words;
    uint w = words[i];
    out[i] = w * 3u + psb_scale->bias;
    psb_out[i + 64u] = as_type<uint>(psb_scale->gain[i & 3u] * float(w & 0xffu));
}
