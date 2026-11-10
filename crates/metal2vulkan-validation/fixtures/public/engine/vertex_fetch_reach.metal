// Owned fixture: a rasterization-disabled vertex function whose `[[stage_in]]` attribute (one
// uint, a case's vertex descriptor, at a layout's fixed stride) a draw fetches at elements whose
// displacement from the buffer's start lies past 2^32 bytes: a vertex start, a base vertex, an
// index, or a base instance past 2^30. Vertex `vid` of instance `iid` writes the attribute and
// the sum of its ids at the sum's low two bits:
//     out[(vid + iid) & 3] = (v, vid + iid)
// so a draw of four vertices of one instance, or of one vertex of four instances, writes four
// apart slots of `out` from any start. A command binds `out` as vertex buffer 0. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Vertex {
    uint v [[attribute(0)]];
};

vertex void vertex_fetch_reach(
    Vertex in [[stage_in]],
    device uint2 *out [[buffer(0)]],
    uint vid [[vertex_id]],
    uint iid [[instance_id]])
{
    uint at = vid + iid;
    out[at & 3] = uint2(in.v, at);
}
