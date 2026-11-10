// Owned fixture (written for this repository): a `half` sample through the runtime sampler at
// slot 0, as a compute kernel of a 16 by 16 grid: lane `(x, y)` samples texture 0 at the
// coordinate `where[16 y + x]` and writes the `half4` it answers to `out[16 y + x]`. The runtime
// lane binds a 32-bit float texture and samplers of each filter. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void rgbx_sample_half(texture2d<half> t [[texture(0)]], sampler s [[sampler(0)]],
                             device half4 *out [[buffer(0)]], device const float2 *where [[buffer(1)]],
                             uint2 g [[thread_position_in_grid]])
{
    out[g.y * 16 + g.x] = half4(t.sample(s, where[g.y * 16 + g.x]));
}
