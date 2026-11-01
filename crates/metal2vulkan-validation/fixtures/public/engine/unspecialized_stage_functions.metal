// Owned MSL: stage functions of every kind a pipeline state takes, each declaring a function
// constant or none, the shapes of the probe `undefined_function_constant_values.m`. Metal makes no
// state of a function made by `newFunctionWithName:` that declares a constant, whether it reads it
// only where `is_function_constant_defined` says (`unspecialized_guarded`) or the constant gates an
// argument (`unspecialized_gate`), and makes one of each made with constant values, none among
// them too. The `_free` functions declare none. Loop-free.
#include <metal_stdlib>
using namespace metal;

constant uint optional_count [[function_constant(0)]];
constant bool has_extra [[function_constant(1)]];
constant float vertex_scale [[function_constant(2)]];
constant float fragment_tint [[function_constant(3)]];
constant half tile_tint [[function_constant(4)]];
constant float mesh_scale [[function_constant(5)]];
constant uint object_count [[function_constant(6)]];

kernel void unspecialized_guarded(device uint *out [[buffer(0)]],
                                  uint tid [[thread_position_in_grid]]) {
    out[tid] = is_function_constant_defined(optional_count) ? optional_count : 7u;
}

kernel void unspecialized_free(device uint *out [[buffer(0)]],
                               uint tid [[thread_position_in_grid]]) {
    out[tid] = 9u;
}

kernel void unspecialized_gate(device uint *out [[buffer(0)]],
                               device const uint *extra [[buffer(1), function_constant(has_extra)]],
                               uint tid [[thread_position_in_grid]]) {
    out[tid] = has_extra ? extra[tid] : 2u;
}

vertex float4 unspecialized_vertex(uint vid [[vertex_id]]) {
    return float4(float(vid & 1u) * vertex_scale, float(vid >> 1), 0.0f, 1.0f);
}

vertex float4 unspecialized_vertex_free(uint vid [[vertex_id]]) {
    return float4(float(vid & 1u), float(vid >> 1), 0.0f, 1.0f);
}

fragment float4 unspecialized_fragment() { return float4(fragment_tint, 0.0f, 0.0f, 1.0f); }

fragment float4 unspecialized_fragment_free() { return float4(1.0f, 0.0f, 0.0f, 1.0f); }

struct TileColor {
    half4 color [[color(0)]];
};

kernel void unspecialized_tile(imageblock<TileColor, imageblock_layout_implicit> block,
                               ushort2 lid [[thread_position_in_threadgroup]]) {
    TileColor v = block.read(lid);
    v.color *= tile_tint;
    block.write(v, lid);
}

struct MeshVertex {
    float4 position [[position]];
};

using Triangle = metal::mesh<MeshVertex, void, 3, 1, topology::triangle>;

[[mesh]] void unspecialized_mesh(Triangle m, uint t [[thread_index_in_threadgroup]]) {
    if (t == 0u) {
        m.set_primitive_count(1u);
        m.set_index(0u, 0u);
        m.set_index(1u, 1u);
        m.set_index(2u, 2u);
    }
    if (t < 3u) {
        MeshVertex v;
        v.position = float4(float(t & 1u) * mesh_scale, float(t >> 1), 0.0f, 1.0f);
        m.set_vertex(t, v);
    }
}

[[mesh]] void unspecialized_mesh_free(Triangle m, uint t [[thread_index_in_threadgroup]]) {
    if (t == 0u) {
        m.set_primitive_count(1u);
        m.set_index(0u, 0u);
        m.set_index(1u, 1u);
        m.set_index(2u, 2u);
    }
    if (t < 3u) {
        MeshVertex v;
        v.position = float4(float(t & 1u), float(t >> 1), 0.0f, 1.0f);
        m.set_vertex(t, v);
    }
}

[[object]] void unspecialized_object(mesh_grid_properties grid) {
    grid.set_threadgroups_per_grid(uint3(object_count + 1u, 1u, 1u));
}
