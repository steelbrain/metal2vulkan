// Vertex amplification: [[amplification_id]], [[amplification_count]],
// [[shared]] outputs, and the per-view viewport and layer builtins beside them, in vertex,
// fragment, and post-tessellation vertex functions. Written for this repository; no third-party
// source.
#include <metal_stdlib>
using namespace metal;

struct AmpOut {
    float4 position [[position]];
    float4 color [[user(locn0)]];
    float2 uv [[shared]];
    uint viewport [[viewport_array_index]];
};

// Each amplification moves the position by its own offset. `uv` depends only on the vertex, so it
// is shared by every amplification; the viewport, a built-in output, is shared by default (MSL
// §5.2.3.3), so it must not depend on the amplification either. It depends on the instance alone,
// the same at every vertex of a primitive, as Metal asks of it.
vertex AmpOut amp_vertex(uint vid [[vertex_id]],
                         uint iid [[instance_id]],
                         ushort amp [[amplification_id]],
                         ushort count [[amplification_count]],
                         constant float4 *offsets [[buffer(0)]])
{
    float2 corner = float2(float((vid << 1) & 2u), float(vid & 2u));
    AmpOut out;
    out.position = float4(corner * 2.0f - 1.0f, 0.0f, 1.0f) + offsets[amp];
    out.color = float4(float(amp), float(count), 0.25f, 1.0f);
    out.uv = corner;
    out.viewport = iid & 1u;
    return out;
}

// Records, per vertex and amplification, the amplification id plus one and the count: an
// observation the CPU interpreter can make without rasterizing.
vertex float4 amp_vertex_store(uint vid [[vertex_id]],
                               uint amp [[amplification_id]],
                               uint count [[amplification_count]],
                               device uint *out [[buffer(0)]])
{
    out[vid * 8u + amp] = (count << 16) | (amp + 1u);
    return float4(0.0f, 0.0f, 0.0f, 1.0f);
}

struct AmpLayerOut {
    float4 position [[position]];
    uint layer [[render_target_array_index]];
};

// Chooses its render target slice itself, per instance: the layer output a multiview pipeline
// cannot declare.
vertex AmpLayerOut amp_vertex_layer(uint vid [[vertex_id]],
                                    uint iid [[instance_id]],
                                    ushort amp [[amplification_id]],
                                    constant uint *slices [[buffer(0)]])
{
    AmpLayerOut out;
    out.position = float4(float(vid), float(amp), 0.0f, 1.0f);
    out.layer = slices[iid & 1u];
    return out;
}

struct AmpFragIn {
    float4 color [[user(locn0)]];
    uint viewport [[viewport_array_index]];
};

fragment float4 amp_fragment(AmpFragIn in [[stage_in]],
                             ushort amp [[amplification_id]],
                             ushort count [[amplification_count]])
{
    return in.color + float4(float(amp), float(count), float(in.viewport), 0.0f);
}

// Reads the render target slice: a fragment input a multiview pipeline cannot declare.
fragment float4 amp_fragment_layer(uint layer [[render_target_array_index]],
                                   ushort amp [[amplification_id]])
{
    return float4(float(layer), float(amp), 0.0f, 1.0f);
}

// A post-tessellation vertex function reads the amplification builtins as a vertex function does.
[[patch(triangle, 3)]]
vertex float4 amp_post_tessellation(float3 bary [[position_in_patch]],
                                    ushort amp [[amplification_id]],
                                    ushort count [[amplification_count]])
{
    return float4(bary * float(amp + 1), float(count));
}
