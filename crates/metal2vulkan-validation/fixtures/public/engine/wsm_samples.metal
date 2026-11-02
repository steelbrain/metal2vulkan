// The functions of the multisampled word formats' lane
// (validation/src/runtime_lane/word_samples_msaa.rs), our own MSL, the shapes of our probe
// `msaa_word_formats.m`: a fullscreen triangle; a fragment function that runs once per sample and
// writes each sample its own color, read from buffer 0 at `(y * width + x) * samples + sample`
// (the width, the samples and whether to discard in buffer 1), discarding the last sample of every
// eighth pixel where asked; and kernels that read every sample of a multisampled texture into
// buffer 0, as `float`s and as the unsigned words of a view.
#include <metal_stdlib>
using namespace metal;

struct Position {
    float4 position [[position]];
};

vertex Position wsm_vertex(uint id [[vertex_id]]) {
    float2 p = float2((id << 1) & 2, id & 2);
    Position out;
    out.position = float4(p * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

fragment float4 wsm_fill(Position in [[stage_in]], uint s [[sample_id]],
                         constant float4 *values [[buffer(0)]],
                         constant uint *shape [[buffer(1)]]) {
    uint x = uint(floor(in.position.x)), y = uint(floor(in.position.y));
    uint p = y * shape[0] + x;
    if (shape[2] != 0 && p % 8 == 7 && s == shape[1] - 1) discard_fragment();
    return values[p * shape[1] + s];
}

kernel void wsm_read_f(texture2d_ms<float> t [[texture(0)]], device float4 *out [[buffer(0)]],
                       uint2 g [[thread_position_in_grid]]) {
    uint n = t.get_num_samples();
    if (g.x >= t.get_width() || g.y >= t.get_height()) return;
    for (uint s = 0; s < n; s++) out[(g.y * t.get_width() + g.x) * n + s] = t.read(g, s);
}

kernel void wsm_read_u(texture2d_ms<uint> t [[texture(0)]], device uint *out [[buffer(0)]],
                       uint2 g [[thread_position_in_grid]]) {
    uint n = t.get_num_samples();
    if (g.x >= t.get_width() || g.y >= t.get_height()) return;
    for (uint s = 0; s < n; s++) out[(g.y * t.get_width() + g.x) * n + s] = t.read(g, s).r;
}
