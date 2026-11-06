// Owned fixture: the object and mesh functions of `crates/metal2vulkan-conformance/probes/
// mesh_pipeline_limits.m`, for the conformance suite's `mesh-limits` cases, which make mesh render
// pipelines of them under a descriptor's thread limits, mesh-grid limit, required threadgroup sizes,
// payload length and vertex amplification count, and read what the made state answers. Each object
// function launches one mesh threadgroup and passes a word, which its mesh function reads.
#include <metal_stdlib>
using namespace metal;

struct LimitsPayload {
    uint v[4];
};

struct LimitsVertex {
    float4 position [[position]];
};

using LimitsMesh = metal::mesh<LimitsVertex, void, 3, 1, topology::triangle>;

[[object]] void mesh_pipeline_limits_object(object_data LimitsPayload &p [[payload]],
                                            mesh_grid_properties g,
                                            uint t [[thread_index_in_threadgroup]]) {
    if (t == 0) {
        p.v[0] = 1;
        g.set_threadgroups_per_grid(uint3(1, 1, 1));
    }
}

[[object, max_total_threads_per_threadgroup(64)]]
void mesh_pipeline_limits_object_64(object_data LimitsPayload &p [[payload]],
                                    mesh_grid_properties g,
                                    uint t [[thread_index_in_threadgroup]]) {
    if (t == 0) {
        p.v[0] = 1;
        g.set_threadgroups_per_grid(uint3(1, 1, 1));
    }
}

[[object, max_total_threadgroups_per_mesh_grid(4)]]
void mesh_pipeline_limits_object_grid_4(object_data LimitsPayload &p [[payload]],
                                        mesh_grid_properties g,
                                        uint t [[thread_index_in_threadgroup]]) {
    if (t == 0) {
        p.v[0] = 1;
        g.set_threadgroups_per_grid(uint3(1, 1, 1));
    }
}

[[mesh]] void mesh_pipeline_limits_mesh(LimitsMesh out, const object_data LimitsPayload &p [[payload]],
                                        uint t [[thread_index_in_threadgroup]]) {
    if (t == 0) {
        out.set_primitive_count(p.v[0] - 1);
    }
}

[[mesh, max_total_threads_per_threadgroup(64)]]
void mesh_pipeline_limits_mesh_64(LimitsMesh out, const object_data LimitsPayload &p [[payload]],
                                  uint t [[thread_index_in_threadgroup]]) {
    if (t == 0) {
        out.set_primitive_count(p.v[0] - 1);
    }
}

[[mesh]] void mesh_pipeline_limits_mesh_alone(LimitsMesh out,
                                              uint t [[thread_index_in_threadgroup]]) {
    if (t == 0) {
        out.set_primitive_count(0);
    }
}
