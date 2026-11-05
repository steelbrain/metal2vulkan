// What a mesh render pipeline's creation answers for the descriptor's thread limits, its mesh-grid
// limit, its required threadgroup sizes, its payload length and its vertex amplification count,
// and what the made state's queries answer: `maxTotalThreadsPerObjectThreadgroup`,
// `maxTotalThreadsPerMeshThreadgroup`, `objectThreadExecutionWidth`, `meshThreadExecutionWidth`,
// `maxTotalThreadgroupsPerMeshGrid`, `requiredThreadsPerObjectThreadgroup`,
// `requiredThreadsPerMeshThreadgroup` and `maxTotalThreadsPerThreadgroup`.
//
// Build (no cargo target needed):
//   clang -fobjc-arc -O1 -fmodules -o mesh_pipeline_limits mesh_pipeline_limits.m \
//     -framework Foundation -framework Metal
//
// `mesh_pipeline_limits CASE` makes one pipeline, rasterization disabled, of the library below
// compiled from source, and prints `made` and the state's answers, or `refused` and the error's
// domain, code and description. `mesh_pipeline_limits --cases` lists the cases. Run one case a
// process: a stop of Metal's ends only its own.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>

static NSString *const SOURCE = @"#include <metal_stdlib>\n"
                                 "using namespace metal;\n"
                                 "struct P { uint v[4]; };\n"
                                 "struct V { float4 position [[position]]; };\n"
                                 "using M = metal::mesh<V, void, 3, 1, topology::triangle>;\n"
                                 "[[object]] void obj(object_data P& p [[payload]], mesh_grid_properties g,\n"
                                 "                    uint t [[thread_index_in_threadgroup]]) {\n"
                                 "  if (t == 0) { p.v[0] = 1; g.set_threadgroups_per_grid(uint3(1, 1, 1)); }\n"
                                 "}\n"
                                 "[[object, max_total_threads_per_threadgroup(64)]]\n"
                                 "void obj64(object_data P& p [[payload]], mesh_grid_properties g,\n"
                                 "           uint t [[thread_index_in_threadgroup]]) {\n"
                                 "  if (t == 0) { p.v[0] = 1; g.set_threadgroups_per_grid(uint3(1, 1, 1)); }\n"
                                 "}\n"
                                 "[[mesh]] void msh(M out, const object_data P& p [[payload]],\n"
                                 "                  uint t [[thread_index_in_threadgroup]]) {\n"
                                 "  if (t == 0) { out.set_primitive_count(p.v[0] - 1); }\n"
                                 "}\n"
                                 "[[mesh]] void alone(M out, uint t [[thread_index_in_threadgroup]]) {\n"
                                 "  if (t == 0) { out.set_primitive_count(0); }\n"
                                 "}\n";

// A case: the object function (NULL for none), the mesh function, and the descriptor's fields.
typedef struct {
    const char *name;
    const char *object;
    const char *mesh;
    NSUInteger max_object, max_mesh, max_grid, payload, amplification;
    MTLSize required_object, required_mesh;
} Case;

#define Z {0, 0, 0}
static const Case CASES[] = {
    {"plain", "obj", "msh", 0, 0, 0, 0, 1, Z, Z},
    {"no-object", NULL, "alone", 0, 0, 0, 0, 1, Z, Z},
    {"max-64-32", "obj", "msh", 64, 32, 0, 0, 1, Z, Z},
    {"max-object-2048", "obj", "msh", 2048, 0, 0, 0, 1, Z, Z},
    {"max-mesh-2048", "obj", "msh", 0, 2048, 0, 0, 1, Z, Z},
    {"max-mesh-2048-no-object", NULL, "alone", 0, 2048, 0, 0, 1, Z, Z},
    {"attribute-64", "obj64", "msh", 0, 0, 0, 0, 1, Z, Z},
    {"attribute-64-max-32", "obj64", "msh", 32, 0, 0, 0, 1, Z, Z},
    {"attribute-64-max-128", "obj64", "msh", 128, 0, 0, 0, 1, Z, Z},
    {"grid-8", "obj", "msh", 0, 0, 8, 0, 1, Z, Z},
    {"grid-2097152", "obj", "msh", 0, 0, 2097152, 0, 1, Z, Z},
    {"grid-8-no-object", NULL, "alone", 0, 0, 8, 0, 1, Z, Z},
    {"payload-64", "obj", "msh", 0, 0, 0, 64, 1, Z, Z},
    {"payload-16384", "obj", "msh", 0, 0, 0, 16384, 1, Z, Z},
    {"payload-16385", "obj", "msh", 0, 0, 0, 16385, 1, Z, Z},
    {"payload-16385-no-object", NULL, "alone", 0, 0, 0, 16385, 1, Z, Z},
    {"amplification-2", "obj", "msh", 0, 0, 0, 0, 2, Z, Z},
    {"amplification-0", "obj", "msh", 0, 0, 0, 0, 0, Z, Z},
    {"required-object-32x2", "obj", "msh", 0, 0, 0, 0, 1, {32, 2, 1}, Z},
    {"required-object-4x0", "obj", "msh", 0, 0, 0, 0, 1, {4, 0, 1}, Z},
    {"required-object-2048", "obj", "msh", 0, 0, 0, 0, 1, {2048, 1, 1}, Z},
    {"required-object-64-max-32", "obj", "msh", 32, 0, 0, 0, 1, {64, 1, 1}, Z},
    {"required-object-32-max-64", "obj", "msh", 64, 0, 0, 0, 1, {32, 1, 1}, Z},
    {"required-object-32-attribute-64", "obj64", "msh", 0, 0, 0, 0, 1, {32, 1, 1}, Z},
    {"required-object-no-object", NULL, "alone", 0, 0, 0, 0, 1, {32, 1, 1}, Z},
    {"required-mesh-16x2", "obj", "msh", 0, 0, 0, 0, 1, Z, {16, 2, 1}},
    {"required-mesh-4x0", "obj", "msh", 0, 0, 0, 0, 1, Z, {4, 0, 1}},
    {"required-mesh-2048", "obj", "msh", 0, 0, 0, 0, 1, Z, {2048, 1, 1}},
    {"required-mesh-64-max-32", "obj", "msh", 0, 32, 0, 0, 1, Z, {64, 1, 1}},
    {"required-mesh-32-max-64", "obj", "msh", 0, 64, 0, 0, 1, Z, {32, 1, 1}},
};

static void print_size(const char *what, MTLSize s) {
    printf("  %s %lu,%lu,%lu\n", what, (unsigned long)s.width, (unsigned long)s.height,
           (unsigned long)s.depth);
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        size_t n = sizeof CASES / sizeof CASES[0];
        if (argc == 2 && strcmp(argv[1], "--cases") == 0) {
            for (size_t i = 0; i < n; i++) {
                printf("%s\n", CASES[i].name);
            }
            return 0;
        }
        const Case *c = NULL;
        for (size_t i = 0; argc == 2 && i < n; i++) {
            if (strcmp(argv[1], CASES[i].name) == 0) {
                c = &CASES[i];
            }
        }
        if (c == NULL) {
            fprintf(stderr, "usage: mesh_pipeline_limits CASE | --cases\n");
            return 2;
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:SOURCE options:nil error:&error];
        if (library == nil) {
            printf("library %s\n", error.localizedDescription.UTF8String);
            return 1;
        }
        MTLMeshRenderPipelineDescriptor *d = [MTLMeshRenderPipelineDescriptor new];
        if (c->object != NULL) {
            d.objectFunction = [library newFunctionWithName:@(c->object)];
        }
        d.meshFunction = [library newFunctionWithName:@(c->mesh)];
        d.rasterizationEnabled = NO;
        d.maxTotalThreadsPerObjectThreadgroup = c->max_object;
        d.maxTotalThreadsPerMeshThreadgroup = c->max_mesh;
        d.maxTotalThreadgroupsPerMeshGrid = c->max_grid;
        d.payloadMemoryLength = c->payload;
        d.maxVertexAmplificationCount = c->amplification;
        d.requiredThreadsPerObjectThreadgroup = c->required_object;
        d.requiredThreadsPerMeshThreadgroup = c->required_mesh;
        printf("%s on %s\n", c->name, device.name.UTF8String);
        id<MTLRenderPipelineState> state =
            [device newRenderPipelineStateWithMeshDescriptor:d
                                                     options:MTLPipelineOptionNone
                                                  reflection:nil
                                                       error:&error];
        if (state == nil) {
            printf("refused %s %ld %s\n", error.domain.UTF8String, (long)error.code,
                   error.localizedDescription.UTF8String);
            return 0;
        }
        printf("made\n");
        printf("  maxTotalThreadsPerObjectThreadgroup %lu\n",
               (unsigned long)state.maxTotalThreadsPerObjectThreadgroup);
        printf("  maxTotalThreadsPerMeshThreadgroup %lu\n",
               (unsigned long)state.maxTotalThreadsPerMeshThreadgroup);
        printf("  objectThreadExecutionWidth %lu\n",
               (unsigned long)state.objectThreadExecutionWidth);
        printf("  meshThreadExecutionWidth %lu\n", (unsigned long)state.meshThreadExecutionWidth);
        printf("  maxTotalThreadgroupsPerMeshGrid %lu\n",
               (unsigned long)state.maxTotalThreadgroupsPerMeshGrid);
        printf("  maxTotalThreadsPerThreadgroup %lu\n",
               (unsigned long)state.maxTotalThreadsPerThreadgroup);
        print_size("requiredThreadsPerObjectThreadgroup", state.requiredThreadsPerObjectThreadgroup);
        print_size("requiredThreadsPerMeshThreadgroup", state.requiredThreadsPerMeshThreadgroup);
    }
    return 0;
}
