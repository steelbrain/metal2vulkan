// Owned fixture: `read()` of a texture out of range. One thread reads ten texels, each at the
// coordinate and level of `at[i]` (x, y, level), so no coordinate is a constant the compiler sees:
// out[0..6] of `two_levels` (4 x 4 with 2 levels), out[7] of `one_level` (a view of its first
// level alone), out[8..10] of `red` (an R8Unorm texture of 1 level, a format without alpha).
// Metal documents an out-of-range read as zero in every component, alpha 1 for a format without
// alpha; which reads are out of range, and what each answers, is the case's. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void oor_texture_reads(texture2d<float, access::read> two_levels [[texture(0)]],
                              texture2d<float, access::read> one_level [[texture(1)]],
                              texture2d<float, access::read> red [[texture(2)]],
                              constant uint4 *at [[buffer(0)]],
                              device float4 *out [[buffer(1)]]) {
    out[0] = two_levels.read(at[0].xy, at[0].z);
    out[1] = two_levels.read(at[1].xy, at[1].z);
    out[2] = two_levels.read(at[2].xy, at[2].z);
    out[3] = two_levels.read(at[3].xy, at[3].z);
    out[4] = two_levels.read(at[4].xy, at[4].z);
    out[5] = two_levels.read(at[5].xy, at[5].z);
    out[6] = two_levels.read(at[6].xy, at[6].z);
    out[7] = one_level.read(at[7].xy, at[7].z);
    out[8] = red.read(at[8].xy, at[8].z);
    out[9] = red.read(at[9].xy, at[9].z);
    out[10] = red.read(at[10].xy, at[10].z);
}
