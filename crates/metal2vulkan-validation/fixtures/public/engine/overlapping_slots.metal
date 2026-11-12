// Owned fixture: a sampled texture whose slot a function constant sets, beside a write texture at
// texture(1). Function constant 0 (`n`) is the sampled texture's index (`[[texture(n)]]`), so
// n = 1 puts both textures on texture slot 1, which Metal refuses when it makes the function
// ("texture argument src has invalid location (<1,1>)"), and n = 0 (what an undefined n reads
// too) makes a kernel whose thread gid writes the sample of `src` at the centre of texel gid of a
// 4 by 1 texture through sampler 0 (Metal's default sampler state: nearest, clamp to edge) into
// `dst` at gid.
#include <metal_stdlib>
using namespace metal;

constant uint n [[function_constant(0)]];

kernel void overlapping_slots(texture2d<float, access::write> dst [[texture(1)]],
                              texture2d<float> src [[texture(n)]],
                              sampler s [[sampler(0)]],
                              uint2 gid [[thread_position_in_grid]]) {
    dst.write(src.sample(s, (float2(gid) + 0.5) / 4.0), gid);
}
