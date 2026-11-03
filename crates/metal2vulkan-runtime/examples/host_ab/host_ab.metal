// The host A/B benchmark's shaders (examples/host_ab): one source, compiled once into a metallib
// that both sides load, natively and through the runtime's translation.
//
//   xcrun -sdk macosx metal -DNONCE=<n> -o host_ab.metallib host_ab.metal
//
// NONCE (below 2^24, so each value is its own float) only changes a constant in each function,
// so each value gives functions of the same shape whose bytes differ: a cold pipeline creation
// finds them in no cache, Metal's or the runtime's.

#include <metal_stdlib>
using namespace metal;

#ifndef NONCE
#define NONCE 0
#endif

struct QuadOut {
    float4 position [[position]];
    float2 uv;
};

// A rectangle (x0, y0, x1, y1 in normalized device coordinates) as a strip of four vertices.
vertex QuadOut quad_vertex(uint vid [[vertex_id]], constant float4 &rect [[buffer(0)]]) {
    float2 corner = float2(float(vid & 1u), float(vid >> 1));
    QuadOut out;
    out.position = float4(mix(rect.xy, rect.zw, corner), float(NONCE) * 1.0e-12, 1.0);
    out.uv = float2(corner.x, 1.0 - corner.y);
    return out;
}

// An OpenGL guest's flat-shaded triangles (the provoking workload): each vertex at its position in
// buffer 0, with a colour of its own that the fragment reads flat, from the primitive's provoking
// vertex.
struct FlatOut {
    float4 position [[position]];
    uint colour [[flat]];
};

vertex FlatOut flat_vertex(uint vid [[vertex_id]], const device float4 *positions [[buffer(0)]]) {
    FlatOut out;
    out.position = positions[vid] + float4(0.0, 0.0, float(NONCE) * 1.0e-12, 0.0);
    out.colour = vid * 2654435761u;
    return out;
}

fragment float4 flat_fragment(FlatOut in [[stage_in]]) {
    return unpack_unorm4x8_to_float(in.colour) + float4(float(NONCE) * 1.0e-12);
}

// The small draws' fragment: one sample.
fragment float4 textured_fragment(QuadOut in [[stage_in]],
                                  texture2d<float> tex [[texture(0)]],
                                  sampler smp [[sampler(0)]]) {
    return tex.sample(smp, in.uv) + float4(float(NONCE) * 1.0e-12);
}

// The heavy work's parameters, read from a constant buffer.
struct HeavyParams {
    float4 coeffs[64];
    uint iterations;
    float scale;
};

// The heavy fragment: float ALU, half math, texture sampling and constant-buffer reads in a loop.
fragment half4 heavy_fragment(QuadOut in [[stage_in]],
                              constant HeavyParams &p [[buffer(0)]],
                              texture2d<half> tex [[texture(0)]],
                              sampler smp [[sampler(0)]]) {
    float2 uv = in.uv;
    float acc = float(NONCE) * 1.0e-12;
    half4 hacc = half4(0.0h);
    for (uint i = 0; i < p.iterations; i++) {
        float4 c = p.coeffs[i & 63u];
        acc = fma(acc, c.x, sin(uv.x * c.y + acc) * c.z);
        half4 t = tex.sample(smp, uv + float2(c.w, acc) * 0.01);
        hacc = fma(hacc, half4(half(c.x)), t * 0.5h);
        uv = fract(uv * 1.37 + c.zw);
    }
    return hacc + half4(half(acc * p.scale));
}

// The heavy kernel: float ALU, half math, texture reads, constant-buffer reads and threadgroup
// memory with barriers, in a loop. Threadgroups of 256 threads.
kernel void heavy_kernel(device float4 *out [[buffer(0)]],
                         constant HeavyParams &p [[buffer(1)]],
                         texture2d<half> tex [[texture(0)]],
                         uint tid [[thread_position_in_grid]],
                         uint lid [[thread_index_in_threadgroup]]) {
    threadgroup float shared_f[256];
    threadgroup half shared_h[256];
    float acc = float(tid) * 1.0e-6 + float(NONCE) * 1.0e-12;
    half hacc = 0.0h;
    for (uint i = 0; i < p.iterations; i++) {
        float4 c = p.coeffs[(i + lid) & 63u];
        acc = fma(acc, c.x, sin(acc * c.y) * c.z + c.w);
        half4 t = tex.read(uint2((tid + i * 7u) & 255u, ((tid >> 8) + i) & 255u));
        hacc = fma(hacc, half(c.x), dot(t, half4(0.25h)));
        shared_f[lid] = acc;
        shared_h[lid] = hacc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        acc += shared_f[(lid + 1u) & 255u] * 0.5;
        hacc += shared_h[(lid + 7u) & 255u] * 0.5h;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    out[tid] = float4(acc * p.scale, float(hacc), 0.0, 1.0);
}

// The feature kernels: each one Metal feature of the heavy kernel in the same loop, so their
// ratios say which feature its cost comes from. Each takes the heavy kernel's bindings
// (`out`, `p`, `tex`) and `src`, a device buffer as large as `out`.
kernel void k_fma(device float4 *out [[buffer(0)]], constant HeavyParams &p [[buffer(1)]],
                  uint tid [[thread_position_in_grid]]) {
    float a = float(tid) * 1.0e-6 + float(NONCE) * 1.0e-12;
    float b = 1.0;
    for (uint i = 0; i < p.iterations; i++) {
        a = fma(a, 0.999, b);
        b = fma(b, 1.0001, -a * 0.5);
    }
    out[tid] = float4(a, b, 0.0, 1.0);
}

kernel void k_sin(device float4 *out [[buffer(0)]], constant HeavyParams &p [[buffer(1)]],
                  uint tid [[thread_position_in_grid]]) {
    float a = float(tid) * 1.0e-6 + float(NONCE) * 1.0e-12;
    for (uint i = 0; i < p.iterations; i++) {
        a = sin(a * 1.3 + 0.7);
    }
    out[tid] = float4(a, 0.0, 0.0, 1.0);
}

kernel void k_half(device float4 *out [[buffer(0)]], constant HeavyParams &p [[buffer(1)]],
                   uint tid [[thread_position_in_grid]]) {
    half a = half(tid & 255u) * 0.01h + half(float(NONCE) * 1.0e-12);
    half b = 0.5h;
    for (uint i = 0; i < p.iterations; i++) {
        a = fma(a, 0.999h, b);
        b = b * 0.75h + a * 0.125h;
    }
    out[tid] = float4(float(a), float(b), 0.0, 1.0);
}

kernel void k_constant(device float4 *out [[buffer(0)]], constant HeavyParams &p [[buffer(1)]],
                       uint tid [[thread_position_in_grid]]) {
    float4 acc = float4(float(NONCE) * 1.0e-12);
    for (uint i = 0; i < p.iterations; i++) {
        acc = fma(acc, float4(0.5), p.coeffs[(i + tid) & 63u]);
    }
    out[tid] = acc;
}

kernel void k_device(device float4 *out [[buffer(0)]], constant HeavyParams &p [[buffer(1)]],
                     device const float4 *src [[buffer(2)]],
                     uint tid [[thread_position_in_grid]]) {
    float4 acc = float4(float(NONCE) * 1.0e-12);
    for (uint i = 0; i < p.iterations; i++) {
        acc = fma(acc, float4(0.5), src[(tid + i * 4099u) & 1048575u]);
    }
    out[tid] = acc;
}

kernel void k_texture(device float4 *out [[buffer(0)]], constant HeavyParams &p [[buffer(1)]],
                      texture2d<half> tex [[texture(0)]],
                      uint tid [[thread_position_in_grid]]) {
    half4 acc = half4(half(float(NONCE) * 1.0e-12));
    for (uint i = 0; i < p.iterations; i++) {
        acc = fma(acc, half4(0.5h), tex.read(uint2((tid + i * 7u) & 255u, ((tid >> 8) + i) & 255u)));
    }
    out[tid] = float4(acc);
}

kernel void k_texture_float(device float4 *out [[buffer(0)]], constant HeavyParams &p [[buffer(1)]],
                            texture2d<float> tex [[texture(0)]],
                            uint tid [[thread_position_in_grid]]) {
    float4 acc = float4(float(NONCE) * 1.0e-12);
    for (uint i = 0; i < p.iterations; i++) {
        acc = fma(acc, float4(0.5), tex.read(uint2((tid + i * 7u) & 255u, ((tid >> 8) + i) & 255u)));
    }
    out[tid] = acc;
}

kernel void k_threadgroup_float(device float4 *out [[buffer(0)]], constant HeavyParams &p [[buffer(1)]],
                                uint tid [[thread_position_in_grid]],
                                uint lid [[thread_index_in_threadgroup]]) {
    threadgroup float shared_f[256];
    float acc = float(tid) * 1.0e-6 + float(NONCE) * 1.0e-12;
    for (uint i = 0; i < p.iterations; i++) {
        shared_f[lid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        acc = fma(shared_f[(lid + 1u) & 255u], 0.5, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    out[tid] = float4(acc, 0.0, 0.0, 1.0);
}

kernel void k_threadgroup_half(device float4 *out [[buffer(0)]], constant HeavyParams &p [[buffer(1)]],
                               uint tid [[thread_position_in_grid]],
                               uint lid [[thread_index_in_threadgroup]]) {
    threadgroup half shared_h[256];
    half acc = half(tid & 255u) * 0.01h + half(float(NONCE) * 1.0e-12);
    for (uint i = 0; i < p.iterations; i++) {
        shared_h[lid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        acc = fma(shared_h[(lid + 7u) & 255u], 0.5h, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    out[tid] = float4(float(acc), 0.0, 0.0, 1.0);
}

// The small dispatches' kernels: two, so a dispatch can change the pipeline state.
kernel void tiny_kernel(device uint *out [[buffer(0)]],
                        constant uint &value [[buffer(1)]],
                        uint tid [[thread_position_in_grid]]) {
    out[tid] = value + tid + uint(NONCE);
}

kernel void tiny_kernel_b(device uint *out [[buffer(0)]],
                          constant uint &value [[buffer(1)]],
                          uint tid [[thread_position_in_grid]]) {
    out[tid] = value * 3u + tid + uint(NONCE);
}

// The handle dispatches' kernel: it reads its texture through a handle in an argument buffer, so
// each of its dispatches binds the device's texture table (`dispatch_handles`).
struct TinyHandles {
    texture2d<float> texture;
};

kernel void tiny_handle_kernel(device uint *out [[buffer(0)]],
                               constant TinyHandles &handles [[buffer(1)]],
                               uint tid [[thread_position_in_grid]]) {
    out[tid] = uint(handles.texture.read(uint2(0, 0)).x * 255.0) + tid + uint(NONCE);
}

// The many-draws workload's functions: a small rectangle per draw, of its own uniforms, and four
// fragment functions over two textures and one sampler.
struct DrawUniforms {
    float4 rect;   // x, y, width, height in NDC
    float4 color;
    float4 misc;   // x: depth, y: phase
};

struct DrawOut {
    float4 position [[position]];
    float2 uv;
    float4 color;
};

vertex DrawOut draws_vertex(uint vid [[vertex_id]],
                            const device float2 *quad [[buffer(0)]],
                            constant DrawUniforms &u [[buffer(1)]],
                            constant float4 &jitter [[buffer(2)]]) {
    float2 p = quad[vid];
    DrawOut out;
    out.position = float4(u.rect.xy + p * u.rect.zw + jitter.xy, u.misc.x, 1.0);
    out.uv = p;
    out.color = u.color + float4(float(NONCE) * 1.0e-12);
    return out;
}

fragment float4 draws_fragment0(DrawOut in [[stage_in]],
                                texture2d<float> t0 [[texture(0)]],
                                texture2d<float> t1 [[texture(1)]],
                                sampler s [[sampler(0)]],
                                constant DrawUniforms &u [[buffer(0)]]) {
    return t0.sample(s, in.uv) * in.color;
}

fragment float4 draws_fragment1(DrawOut in [[stage_in]],
                                texture2d<float> t0 [[texture(0)]],
                                texture2d<float> t1 [[texture(1)]],
                                sampler s [[sampler(0)]],
                                constant DrawUniforms &u [[buffer(0)]]) {
    return mix(t0.sample(s, in.uv), t1.sample(s, in.uv * 2.0), 0.5) * in.color;
}

fragment float4 draws_fragment2(DrawOut in [[stage_in]],
                                texture2d<float> t0 [[texture(0)]],
                                texture2d<float> t1 [[texture(1)]],
                                sampler s [[sampler(0)]],
                                constant DrawUniforms &u [[buffer(0)]]) {
    float r = length(in.uv - 0.5);
    return float4(in.color.rgb * (1.0 - r), 0.8) * t1.sample(s, in.uv);
}

fragment float4 draws_fragment3(DrawOut in [[stage_in]],
                                texture2d<float> t0 [[texture(0)]],
                                texture2d<float> t1 [[texture(1)]],
                                sampler s [[sampler(0)]],
                                constant DrawUniforms &u [[buffer(0)]]) {
    return float4(in.uv, fract(u.misc.y), 1.0) * t0.sample(s, in.uv.yx);
}

// The icb workload's functions (workloads::icb): a kernel that encodes one compute command a
// thread into an indirect command buffer, and the four states the commands run, each on its own
// 64-word window of `data`. Command `tid` runs state `tid % states` over one threadgroup of 64
// threads, with no barrier, as a culling kernel encodes the work it keeps. The states add by
// relaxed atomics: a kernel that reads and then writes a word by its grid position alone leaves
// the order of a simdgroup's accesses to Apple's lockstep, which the runtime makes explicit with a
// barrier that observes the threadgroup partition, so its commands could not run on the GPU
// alone.
struct IcbEncoder {
    command_buffer commands;
    compute_pipeline_state step0;
    compute_pipeline_state step1;
    compute_pipeline_state step2;
    compute_pipeline_state step3;
};

kernel void icb_encode(device const IcbEncoder &encoder [[buffer(0)]],
                       device uint *data [[buffer(1)]],
                       constant uint &states [[buffer(2)]],
                       uint tid [[thread_position_in_grid]]) {
    compute_command command(encoder.commands, tid);
    switch (tid % states) {
    case 0: command.set_compute_pipeline_state(encoder.step0); break;
    case 1: command.set_compute_pipeline_state(encoder.step1); break;
    case 2: command.set_compute_pipeline_state(encoder.step2); break;
    default: command.set_compute_pipeline_state(encoder.step3); break;
    }
    command.set_kernel_buffer(data + tid * 64, 0);
    command.concurrent_dispatch_threadgroups(uint3(1, 1, 1), uint3(64, 1, 1));
}

kernel void icb_step0(device atomic_uint *data [[buffer(0)]],
                      uint tid [[thread_position_in_grid]]) {
    atomic_fetch_add_explicit(&data[tid], 1u + uint(NONCE), memory_order_relaxed);
}

kernel void icb_step1(device atomic_uint *data [[buffer(0)]],
                      uint tid [[thread_position_in_grid]]) {
    atomic_fetch_add_explicit(&data[tid], 2u + uint(NONCE), memory_order_relaxed);
}

kernel void icb_step2(device atomic_uint *data [[buffer(0)]],
                      uint tid [[thread_position_in_grid]]) {
    atomic_fetch_add_explicit(&data[tid], 3u + uint(NONCE), memory_order_relaxed);
}

kernel void icb_step3(device atomic_uint *data [[buffer(0)]],
                      uint tid [[thread_position_in_grid]]) {
    atomic_fetch_add_explicit(&data[tid], 4u + uint(NONCE), memory_order_relaxed);
}

// The icb_draws workload's frame state and objects, as Apple's "Encoding Indirect Command Buffers
// on the GPU" lays them out: a translation and scale shared by every object, and each object's
// position, bounding radius and mesh vertex count.
struct IcbDrawFrame {
    float2 translation;
    float2 scale;
};

struct IcbDrawObject {
    float2 position;
    float radius;
    uint vertices;
};

// The argument buffer that holds the indirect command buffer.
struct IcbDrawContainer {
    command_buffer commands;
};

// Culls `object` against the view and, where it is visible, encodes its command: the frame state,
// the objects and the mesh as vertex buffers, and a draw of `type` of its mesh's vertices whose
// base instance is the object's index. An object out of view keeps the command the frame's reset
// left: nothing.
static void icb_draw_encode_object(uint object, constant IcbDrawFrame *frame,
                                   device const IcbDrawObject *objects, device const float2 *mesh,
                                   device const IcbDrawContainer *container, primitive_type type) {
    float2 center = frame->scale * (frame->translation + objects[object].position);
    float2 radius = frame->scale * objects[object].radius;
    bool visible = all(center + radius >= float2(-1.0)) && all(center - radius <= float2(1.0));
    render_command command(container->commands, object);
    if (visible) {
        command.set_vertex_buffer(frame, 0);
        command.set_vertex_buffer(objects, 1);
        command.set_vertex_buffer(mesh, 2);
        command.draw_primitives(type, 0, objects[object].vertices, 1, object);
    }
}

// Each thread encodes one object's command, a draw of triangles.
kernel void icb_draw_encode(uint object [[thread_position_in_grid]],
                            constant IcbDrawFrame *frame [[buffer(0)]],
                            device const IcbDrawObject *objects [[buffer(1)]],
                            device const float2 *mesh [[buffer(2)]],
                            device const IcbDrawContainer *container [[buffer(3)]]) {
    icb_draw_encode_object(object, frame, objects, mesh, container, primitive_type::triangle);
}

// icb_draw_encode, the draws' primitive type read from `type` (the CPU writes triangles), so the
// kernel's code does not tell which type its draws are of.
kernel void icb_draw_encode_any_type(uint object [[thread_position_in_grid]],
                                     constant IcbDrawFrame *frame [[buffer(0)]],
                                     device const IcbDrawObject *objects [[buffer(1)]],
                                     device const float2 *mesh [[buffer(2)]],
                                     device const IcbDrawContainer *container [[buffer(3)]],
                                     constant uint &type [[buffer(4)]]) {
    icb_draw_encode_object(object, frame, objects, mesh, container, primitive_type(type));
}

// An object's mesh vertex, moved to the object (its instance is the object's index).
vertex QuadOut icb_draw_vertex(uint vid [[vertex_id]], uint object [[instance_id]],
                               constant IcbDrawFrame *frame [[buffer(0)]],
                               device const IcbDrawObject *objects [[buffer(1)]],
                               device const float2 *mesh [[buffer(2)]]) {
    QuadOut out;
    float2 at = frame->scale * (frame->translation + objects[object].position + mesh[vid]);
    out.position = float4(at, float(NONCE) * 1.0e-12, 1.0);
    out.uv = mesh[vid] * 0.5 + 0.5;
    return out;
}

// The object's color from its mesh coordinates.
fragment float4 icb_draw_fragment(QuadOut in [[stage_in]]) {
    return float4(in.uv, float(NONCE) * 1.0e-12, 1.0);
}

// The layers workload's (m2vbench's composite mode): a layer's rectangle in pixels (x, y, w, h)
// and the viewport's size, its opacity and its corner radius.
struct LayerUniforms {
    float4 rect;
    float4 v;
};

struct LayerOut {
    float4 position [[position]];
    float2 uv;
    float4 shape;
};

// A layer's rectangle as a strip of four vertices.
vertex LayerOut layer_vertex(uint vid [[vertex_id]], constant LayerUniforms &u [[buffer(0)]]) {
    float2 p = float2(float(vid & 1u), float(vid >> 1));
    float2 ndc = (u.rect.xy + p * u.rect.zw) / u.v.xy * 2.0 - 1.0;
    LayerOut out;
    out.position = float4(ndc.x, -ndc.y, float(NONCE) * 1.0e-12, 1.0);
    out.uv = p;
    out.shape = float4(u.rect.zw, u.v.z, u.v.w);
    return out;
}

static float rounded_rect(float2 uv, float2 size, float r) {
    float2 q = abs(uv - 0.5) * size - (size * 0.5 - r);
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}

// A layer's texels inside its rounded corners, premultiplied by its opacity.
fragment float4 layer_fragment(LayerOut in [[stage_in]],
                               texture2d<float> t [[texture(0)]],
                               sampler s [[sampler(0)]]) {
    float a = saturate(0.5 - rounded_rect(in.uv, in.shape.xy, in.shape.w));
    return t.sample(s, in.uv) * (a * in.shape.z) + float4(float(NONCE) * 1.0e-12);
}

// A soft shadow under a layer grown by 32 pixels on each side.
fragment float4 shadow_fragment(LayerOut in [[stage_in]]) {
    float d = rounded_rect(in.uv, in.shape.xy, in.shape.w + 32.0) + 32.0;
    float a = 1.0 - smoothstep(-24.0, 24.0, d);
    return float4(0.0, 0.0, 0.0, 0.35 * a * in.shape.z + float(NONCE) * 1.0e-12);
}

// The calls workload's functions (workloads::calls), shaped as a guest's draws: positions from
// vertex buffer 0 by vertex, a colour from buffer 1 at an offset that moves; a fragment that
// samples texture 0 with sampler 0, or one that returns fragment buffer 0's first float4.
struct CallsOut {
    float4 position [[position]];
    float4 color;
};

vertex CallsOut calls_vertex(uint vid [[vertex_id]],
                             const device float4 *positions [[buffer(0)]],
                             const device float4 *colours [[buffer(1)]]) {
    CallsOut out;
    out.position = positions[vid] + float4(0.0, 0.0, float(NONCE) * 1.0e-12, 0.0);
    out.color = colours[0];
    return out;
}

fragment float4 calls_texture_fragment(CallsOut in [[stage_in]],
                                       texture2d<float> t [[texture(0)]],
                                       sampler s [[sampler(0)]]) {
    return t.sample(s, in.position.xy / 512.0) * in.color;
}

fragment float4 calls_buffer_fragment(CallsOut in [[stage_in]],
                                      const device float4 *c [[buffer(0)]]) {
    return c[0] * in.color;
}
