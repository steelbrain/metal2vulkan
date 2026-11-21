// Owned fixture: Metal's bicubic filter in a vertex function, the vertex half of a draw's
// bicubic sample (`bcu_fragment` is the fragment half). Vertex `vid` of one triangle that covers
// the target samples the texture at `p[vid]` through a clamp-to-edge bicubic sampler at
// normalized coordinates, in the implicit form (outside a fragment function, the base level), and
// passes the sample on; `bcu_vertex_color` writes the interpolated sample. Loop-free.
#include <metal_stdlib>
using namespace metal;

constant sampler bicubic_edge(filter::bicubic, address::clamp_to_edge, coord::normalized);

struct BcuVertexOut {
    float4 position [[position]];
    float4 color;
};

vertex BcuVertexOut bcu_vertex(uint vid [[vertex_id]],
                               texture2d<float, access::sample> t [[texture(0)]],
                               constant float2 *p [[buffer(0)]])
{
    BcuVertexOut o;
    o.position = float4(vid == 1 ? 3.0 : -1.0, vid == 2 ? 3.0 : -1.0, 0.0, 1.0);
    o.color = t.sample(bicubic_edge, p[vid]);
    return o;
}

fragment float4 bcu_vertex_color(BcuVertexOut in [[stage_in]])
{
    return in.color;
}
