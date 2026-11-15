// Owned fixture: descriptor sets as Metal argument buffers, in the shape SPIRV-Cross writes for a
// Vulkan layer that encodes each descriptor set in one (tier 2): one argument buffer struct per
// set at [[buffer(set)]], every binding of the set a member at [[id(k)]] whether the stage uses it
// or not (an unused one as a padding member of its kind), a combined image sampler as a texture
// member and a sampler member, a texel buffer as a texture_buffer member, a descriptor array as an
// array member, and a uniform or storage buffer as a pointer member.
//
// Each fragment function reads one kind of member, so a refusal names its kind; abt_frag_sets
// reads them all, as one fragment function of a real layout does. The vertex functions draw one
// triangle over the target from the vertex ID (no stage input), and abt_vert_sets reads a texel
// buffer and a uniform buffer from set 0 as a skinning vertex function does.
//
// abt_vert_bones and abt_frag_bones are the checked pair: the vertex function reads texel
// `first + vid` of set 0's texel buffer (`first` the frame's offset.x) into a flat varying, with
// the index `first + 3 + vid`; the fragment function writes that varying plus the texel at the
// index times the frame's tint, plus set 1's albedo sampled at the centre. With the first vertex
// provoking, every fragment writes texel[first] + texel[first + 3] * tint + albedo.
#include <metal_stdlib>
using namespace metal;

struct Frame {
    float4 tint;
    float4 offset;
};

struct Material {
    float4 scale;
    uint layer;
    uint pad0;
    uint pad1;
    uint pad2;
};

// Set 0: a frame's bindings. 0: uniform buffer, 1: combined depth texture and comparison
// sampler, 3: uniform texel buffer, 4: storage texel buffer (unused by these stages: padding).
struct spvDescriptorSetBuffer0 {
    constant Frame* m_frame [[id(0)]];
    depth2d<float> shadowMap [[id(1)]];
    sampler shadowMapSmplr [[id(2)]];
    texture_buffer<float> bones [[id(3)]];
    texture_buffer<float, access::read_write> _m4_pad [[id(4)]];
};

// Set 1: a material's bindings. 0: sampled image, 1: sampler, 2: combined cube texture and
// sampler, 4: an array of 8 sampled images, 12: storage buffer, 13: storage image.
struct spvDescriptorSetBuffer1 {
    texture2d<float> albedo [[id(0)]];
    sampler linearSmplr [[id(1)]];
    texturecube<float> env [[id(2)]];
    sampler envSmplr [[id(3)]];
    array<texture2d<float>, 8> layers [[id(4)]];
    const device Material* materials [[id(12)]];
    texture2d<float, access::read_write> scratch [[id(13)]];
};

struct VertOut {
    float4 position [[position]];
    float2 uv [[user(locn0)]];
    float3 dir [[user(locn1)]];
    float4 shadow [[user(locn2)]];
    uint layer [[user(locn3)]] [[flat]];
};

struct FragIn {
    float2 uv [[user(locn0)]];
    float3 dir [[user(locn1)]];
    float4 shadow [[user(locn2)]];
    uint layer [[user(locn3)]] [[flat]];
};

struct FragOut {
    float4 color [[color(0)]];
};

static VertOut triangle(uint vid) {
    float2 uv = float2(float((vid << 1) & 2u), float(vid & 2u));
    VertOut out = {};
    out.position = float4(uv * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), 0.0f, 1.0f);
    out.uv = uv;
    out.dir = float3(uv * 2.0f - 1.0f, 1.0f);
    out.shadow = float4(uv, 0.5f, 1.0f);
    out.layer = vid % 8u;
    return out;
}

vertex VertOut abt_vert_plain(uint vid [[vertex_id]]) {
    return triangle(vid);
}

vertex VertOut abt_vert_sets(uint vid [[vertex_id]],
                             const device spvDescriptorSetBuffer0& spvDescriptorSet0 [[buffer(0)]]) {
    VertOut out = triangle(vid);
    float4 bone = spvDescriptorSet0.bones.read(vid);
    out.position.xy += bone.xy * (*spvDescriptorSet0.m_frame).offset.xy;
    return out;
}

fragment FragOut abt_frag_sample(FragIn in [[stage_in]],
                                 const device spvDescriptorSetBuffer1& spvDescriptorSet1 [[buffer(1)]]) {
    FragOut out = {};
    out.color = spvDescriptorSet1.albedo.sample(spvDescriptorSet1.linearSmplr, in.uv);
    return out;
}

fragment FragOut abt_frag_compare(FragIn in [[stage_in]],
                                  const device spvDescriptorSetBuffer0& spvDescriptorSet0 [[buffer(0)]]) {
    FragOut out = {};
    float lit = spvDescriptorSet0.shadowMap.sample_compare(spvDescriptorSet0.shadowMapSmplr,
                                                           in.shadow.xy, in.shadow.z);
    out.color = float4(lit);
    return out;
}

fragment FragOut abt_frag_cube(FragIn in [[stage_in]],
                               const device spvDescriptorSetBuffer1& spvDescriptorSet1 [[buffer(1)]]) {
    FragOut out = {};
    out.color = spvDescriptorSet1.env.sample(spvDescriptorSet1.envSmplr, in.dir);
    return out;
}

fragment FragOut abt_frag_array(FragIn in [[stage_in]],
                                const device spvDescriptorSetBuffer1& spvDescriptorSet1 [[buffer(1)]]) {
    FragOut out = {};
    out.color = spvDescriptorSet1.layers[in.layer].sample(spvDescriptorSet1.linearSmplr, in.uv);
    return out;
}

fragment FragOut abt_frag_texel_buffer(FragIn in [[stage_in]],
                                       const device spvDescriptorSetBuffer0& spvDescriptorSet0 [[buffer(0)]]) {
    FragOut out = {};
    out.color = spvDescriptorSet0.bones.read(in.layer);
    return out;
}

fragment FragOut abt_frag_storage(FragIn in [[stage_in]], float4 pos [[position]],
                                  const device spvDescriptorSetBuffer1& spvDescriptorSet1 [[buffer(1)]]) {
    FragOut out = {};
    out.color = spvDescriptorSet1.scratch.read(uint2(pos.xy));
    return out;
}

fragment FragOut abt_frag_sets(FragIn in [[stage_in]],
                               const device spvDescriptorSetBuffer0& spvDescriptorSet0 [[buffer(0)]],
                               const device spvDescriptorSetBuffer1& spvDescriptorSet1 [[buffer(1)]]) {
    FragOut out = {};
    Material m = spvDescriptorSet1.materials[in.layer];
    float4 base = spvDescriptorSet1.albedo.sample(spvDescriptorSet1.linearSmplr, in.uv);
    float4 layer = spvDescriptorSet1.layers[m.layer].sample(spvDescriptorSet1.linearSmplr, in.uv);
    float4 env = spvDescriptorSet1.env.sample(spvDescriptorSet1.envSmplr, in.dir);
    float lit = spvDescriptorSet0.shadowMap.sample_compare(spvDescriptorSet0.shadowMapSmplr,
                                                           in.shadow.xy, in.shadow.z);
    float4 bone = spvDescriptorSet0.bones.read(in.layer);
    out.color = (base * lit + env + layer + bone) * m.scale * (*spvDescriptorSet0.m_frame).tint;
    return out;
}

struct BonesOut {
    float4 position [[position]];
    float4 bone [[user(locn0)]] [[flat]];
    uint index [[user(locn1)]] [[flat]];
};

struct BonesIn {
    float4 bone [[user(locn0)]] [[flat]];
    uint index [[user(locn1)]] [[flat]];
};

vertex BonesOut abt_vert_bones(uint vid [[vertex_id]],
                               const device spvDescriptorSetBuffer0& spvDescriptorSet0 [[buffer(0)]]) {
    float2 uv = float2(float((vid << 1) & 2u), float(vid & 2u));
    uint first = uint((*spvDescriptorSet0.m_frame).offset.x);
    BonesOut out = {};
    out.position = float4(uv * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), 0.0f, 1.0f);
    out.bone = spvDescriptorSet0.bones.read(first + vid);
    out.index = first + 3u + vid;
    return out;
}

fragment FragOut abt_frag_bones(BonesIn in [[stage_in]],
                                const device spvDescriptorSetBuffer0& spvDescriptorSet0 [[buffer(0)]],
                                const device spvDescriptorSetBuffer1& spvDescriptorSet1 [[buffer(1)]]) {
    FragOut out = {};
    float4 texel = spvDescriptorSet0.bones.read(in.index);
    float4 albedo = spvDescriptorSet1.albedo.sample(spvDescriptorSet1.linearSmplr, float2(0.5f));
    out.color = in.bone + texel * (*spvDescriptorSet0.m_frame).tint + albedo;
    return out;
}
