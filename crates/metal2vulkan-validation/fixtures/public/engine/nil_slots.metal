// Owned fixture: texture slots that hold no texture, read by every stage of a plain draw and of a
// patch draw. Written for this repository; no third-party source.
//
// Each function probes five texture sources: its texture argument `t` ([[texture(0)]]) and the
// four textures of the argument buffer `ab` ([[buffer(1)]], four 8-byte texture IDs). A probe of
// one source is 11 words: the texel a level-0 sample at (0.25, 0.75) gives through `s`
// ([[sampler(0)]]), the texel a read of (0, 0) gives, its width, its height and its mip levels.
// An argument buffer's texture is read and queried but not sampled (its sample words stay 0): the
// new runtime refuses a sample of an argument buffer's texture (its README's known gaps).
// One invocation of each function writes its record of RECORD words into `out` ([[buffer(2)]]):
// the vertex function's vertex 0 and the post-tessellation function's corner (1, 0, 0) of patch 0
// record 0, the fragment function's pixel (0, 0) record 1. A record is the five probes (words 0
// to 54), the post-tessellation function's three control points and its patch's tint (words 56
// to 71; zero in the others), and the stage that wrote it (word 72: 1 vertex, 2 post-tessellation,
// 3 fragment). Both draws cover the target with one triangle: the vertex function's from its
// vertex IDs, the patch's from its control points.
#include <metal_stdlib>
using namespace metal;

constant uint PROBE = 11;
constant uint RECORD = 80;

struct Textures {
    array<texture2d<float>, 4> t;
};

static void held(texture2d<float> t, device uint *out) {
    float4 b = t.read(uint2(0, 0));
    for (uint c = 0; c < 4; c++) {
        out[4 + c] = as_type<uint>(b[c]);
    }
    out[8] = t.get_width();
    out[9] = t.get_height();
    out[10] = t.get_num_mip_levels();
}

static void probe(texture2d<float> t, sampler s, device uint *out) {
    float4 a = t.sample(s, float2(0.25, 0.75), level(0.0));
    for (uint c = 0; c < 4; c++) {
        out[c] = as_type<uint>(a[c]);
    }
    held(t, out);
}

static void probes(texture2d<float> t, constant Textures &ab, sampler s, device uint *out) {
    probe(t, s, out);
    held(ab.t[0], out + PROBE);
    held(ab.t[1], out + 2 * PROBE);
    held(ab.t[2], out + 3 * PROBE);
    held(ab.t[3], out + 4 * PROBE);
}

struct Out {
    float4 position [[position]];
};

vertex Out nil_vertex(uint vid [[vertex_id]],
                      texture2d<float> t [[texture(0)]],
                      sampler s [[sampler(0)]],
                      constant Textures &ab [[buffer(1)]],
                      device uint *out [[buffer(2)]]) {
    if (vid == 0) {
        probes(t, ab, s, out);
        out[72] = 1;
    }
    Out o;
    o.position = float4(vid == 1 ? 3.0 : -1.0, vid == 2 ? 3.0 : -1.0, 0.0, 1.0);
    return o;
}

struct ControlPoint {
    float4 position [[attribute(0)]];
};

struct PatchIn {
    patch_control_point<ControlPoint> points;
    float4 tint [[attribute(5)]];
};

[[patch(triangle, 3)]]
vertex Out nil_post_tess(PatchIn p [[stage_in]],
                         float3 bc [[position_in_patch]],
                         uint patch [[patch_id]],
                         texture2d<float> t [[texture(0)]],
                         sampler s [[sampler(0)]],
                         constant Textures &ab [[buffer(1)]],
                         device uint *out [[buffer(2)]]) {
    if (patch == 0 && bc.x == 1.0) {
        probes(t, ab, s, out);
        for (uint i = 0; i < 3; i++) {
            for (uint c = 0; c < 4; c++) {
                out[56 + 4 * i + c] = as_type<uint>(p.points[i].position[c]);
            }
        }
        for (uint c = 0; c < 4; c++) {
            out[68 + c] = as_type<uint>(p.tint[c]);
        }
        out[72] = 2;
    }
    Out o;
    o.position = bc.x * p.points[0].position + bc.y * p.points[1].position +
                 bc.z * p.points[2].position;
    return o;
}

fragment float4 nil_fragment(Out in [[stage_in]],
                             texture2d<float> t [[texture(0)]],
                             sampler s [[sampler(0)]],
                             constant Textures &ab [[buffer(1)]],
                             device uint *out [[buffer(2)]]) {
    if (uint(in.position.x) == 0 && uint(in.position.y) == 0) {
        probes(t, ab, s, out + RECORD);
        out[RECORD + 72] = 3;
    }
    return float4(0.0, 1.0, 0.0, 1.0);
}
