// Owned fixture: triangle patches drawn into a colour target, as a patch draw renders them. Each
// patch's three control points come through the vertex descriptor (a clip-space `float2` each), and
// its colour is a per-patch attribute. The post-tessellation vertex function places each domain
// point at its barycentric blend of the control points, so the tessellated triangles cover the
// patch's own triangle exactly, and passes the colour on flat; the fragment function writes it.
// A pixel therefore says which patch covers it, if any: what the factors culled, which patch's
// factors and attributes a draw read, and, under a cull mode, which winding the tessellator gave
// the patch's triangles. Every operation is one rounding at most. Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

struct ControlPoint {
    float2 position [[attribute(0)]];
};

struct PatchIn {
    patch_control_point<ControlPoint> cps;
    float4 color [[attribute(1)]];
};

struct Out {
    float4 position [[position]];
    float4 color [[user(locn0), flat]];
};

[[patch(triangle, 3)]]
vertex Out tess_rendered_patches(PatchIn p [[stage_in]],
                                 float3 bc [[position_in_patch]])
{
    Out o;
    float2 at = p.cps[0].position * bc.x + p.cps[1].position * bc.y +
                p.cps[2].position * bc.z;
    o.position = float4(at, 0.0, 1.0);
    o.color = p.color;
    return o;
}

fragment float4 tess_rendered_patches_fragment(Out in [[stage_in]])
{
    return in.color;
}
