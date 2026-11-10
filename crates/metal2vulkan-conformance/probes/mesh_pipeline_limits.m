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
// `mesh_pipeline_limits CASE [SALT]` makes one pipeline, rasterization disabled, of the library
// below compiled from source, and prints `made` and the state's answers, or `refused` and the
// error's domain, code and description. A SALT's 32-bit FNV-1a hash is the source's `SALT`, which
// every function compares a thread's index with, so it is in each function's AIR: a salt never
// used before compiles functions the compiler has never seen (a first compile), and the same salt
// again compiles ones it has. (A comment would not do: Metal's compiler keys what it keeps, across
// processes, on the AIR, which a comment leaves as it is.) Without a salt `SALT` is 0 and each
// comparison folds away. `mesh_pipeline_limits --cases` lists the cases. Run one case a process: a
// stop of Metal's ends only its own.
//
// Measured on an M3 Max under macOS 27.0.1, plain and with `MTL_DEBUG_LAYER=1`, which agree:
// - Made, answering (object threads, mesh threads, object width, mesh width, mesh grid):
//   `plain` 1024, 1024, 32, 32, 1048576, and `maxTotalThreadsPerThreadgroup` 0 in every case;
//   `no-object` 3, 1024, 32, 32, 1048576; `max-64-32` 64 and 32; `attribute-64`,
//   `attribute-64-max-64` and `required-object-32-attribute-64` 64 for the object stage,
//   `mesh-attribute-64`, `mesh-attribute-64-max-64` and `required-mesh-32-attribute-64` 64 for
//   the mesh stage; `grid-8` 8, `grid-2097152` 2097152, `grid-8-no-object` 8; `grid-attribute-4`
//   1048576 (an object function's `max_total_threadgroups_per_mesh_grid` changes nothing), with
//   `grid-attribute-4-grid-8` 8 and `-grid-2` 2; `payload-64`, `payload-16384` and
//   `amplification-2` as `plain`. A required size is answered as given and changes no thread
//   limit: `required-object-32x2` (32,2,1), `required-mesh-16x2` (16,2,1),
//   `required-object-32-max-64` and `required-object-64-max-64` (64 for the object stage),
//   `required-mesh-64-max-64` (64 for the mesh stage).
// - Refused, domain `AGXMetalG15X_M1`, code 3: a maximum of 2048 for either stage, with or
//   without an object function (`max-object-2048`, `max-mesh-2048`, `max-mesh-2048-no-object`:
//   "Specified total max threads per threadgroup (2048) exceeds the maximum total threads per
//   threadgroup supported (1024)"); a function's maximum beside another of the descriptor's
//   (`attribute-64-max-32`, `-max-128`, `mesh-attribute-64-max-32`, `-max-128`: "Kernel specified
//   max total threads per threadgroup (64) must match compute pipeline specified max total threads
//   per threadgroup (32)"); a required mesh size of fewer threads than the mesh maximum
//   (`required-mesh-32-max-64`, the same words, 32 and 64); `payload-16385` ("Object shader payload
//   size (16385) exceeds the maximum payload size allowed (16384)"). Code 2:
//   `payload-16385-no-object` ("Payload size > 16384").
// - The process ends at the descriptor's validation (exit 134), with or without API validation:
//   `amplification-0` ("maxVertexAmplificationCount (0) cannot be set to zero."); a required size
//   with an axis of 0 (`required-object-4x0`, `required-mesh-4x0`: "... must not be 0."), of 2048
//   threads (`required-object-2048`, `required-mesh-2048`: "must be <= 1024 (device limit)"), of
//   more threads than its stage's maximum (`required-object-64-max-32`, `required-mesh-64-max-32`:
//   "must be <= maxTotalThreadsPerThreadgroup (32)"); a required object size without an object
//   function (`required-object-no-object`: "If objectFunction is nil,
//   requiredThreadsPerObjectThreadgroup must be [0, 0, 0] but is [32, 1, 1]").
// - The answers of a required size, and of a function's `[[max_total_threads_per_threadgroup]]`,
//   change with what the compiler compiled before, the answers above among them. Each creation of
//   a source compiled once and never before (a comment unique to it), on the same M3 Max: a
//   required mesh size of 32 under a mesh maximum of 64 is refused as above, plain and under API
//   validation, and made, answering 64, after the same source's compile under a required 64; a
//   required object size of 32 under an object maximum of 64 is refused with the same words; a
//   required mesh size of 16x2 with no maximum answered a mesh maximum of 1024 plain and 32 under
//   API validation, and a required 32 answered 1024 in one source and 32 in another; `obj64`
//   answered an object maximum of 32, and `msh64` a mesh maximum of 32, where this probe's
//   `attribute-64` and `mesh-attribute-64` answered 64. The conformance suite's `mesh-limits`
//   cases leave both out.
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
                                 "  if (t == 0 || t == SALT) { p.v[0] = 1; g.set_threadgroups_per_grid(uint3(1, 1, 1)); }\n"
                                 "}\n"
                                 "[[object, max_total_threads_per_threadgroup(64)]]\n"
                                 "void obj64(object_data P& p [[payload]], mesh_grid_properties g,\n"
                                 "           uint t [[thread_index_in_threadgroup]]) {\n"
                                 "  if (t == 0 || t == SALT) { p.v[0] = 1; g.set_threadgroups_per_grid(uint3(1, 1, 1)); }\n"
                                 "}\n"
                                 "[[mesh]] void msh(M out, const object_data P& p [[payload]],\n"
                                 "                  uint t [[thread_index_in_threadgroup]]) {\n"
                                 "  if (t == 0 || t == SALT) { out.set_primitive_count(p.v[0] - 1); }\n"
                                 "}\n"
                                 "[[mesh]] void alone(M out, uint t [[thread_index_in_threadgroup]]) {\n"
                                 "  if (t == 0 || t == SALT) { out.set_primitive_count(0); }\n"
                                 "}\n";

// The functions of further attributes, compiled with `SOURCE` for the cases that name them alone,
// so that a refusal of theirs leaves the other cases.
static NSString *const ATTRIBUTES =
    @"[[mesh, max_total_threads_per_threadgroup(64)]]\n"
     "void msh64(M out, const object_data P& p [[payload]],\n"
     "           uint t [[thread_index_in_threadgroup]]) {\n"
     "  if (t == 0 || t == SALT) { out.set_primitive_count(p.v[0] - 1); }\n"
     "}\n"
     "[[object, max_total_threadgroups_per_mesh_grid(4)]]\n"
     "void objgrid4(object_data P& p [[payload]], mesh_grid_properties g,\n"
     "              uint t [[thread_index_in_threadgroup]]) {\n"
     "  if (t == 0 || t == SALT) { p.v[0] = 1; g.set_threadgroups_per_grid(uint3(1, 1, 1)); }\n"
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
    {"required-mesh-64-max-64", "obj", "msh", 0, 64, 0, 0, 1, Z, {64, 1, 1}},
    {"required-object-64-max-64", "obj", "msh", 64, 0, 0, 0, 1, {64, 1, 1}, Z},
    {"attribute-64-max-64", "obj64", "msh", 64, 0, 0, 0, 1, Z, Z},
    {"mesh-attribute-64", "obj", "msh64", 0, 0, 0, 0, 1, Z, Z},
    {"mesh-attribute-64-max-32", "obj", "msh64", 0, 32, 0, 0, 1, Z, Z},
    {"mesh-attribute-64-max-64", "obj", "msh64", 0, 64, 0, 0, 1, Z, Z},
    {"mesh-attribute-64-max-128", "obj", "msh64", 0, 128, 0, 0, 1, Z, Z},
    {"required-mesh-32-attribute-64", "obj", "msh64", 0, 0, 0, 0, 1, Z, {32, 1, 1}},
    {"grid-attribute-4", "objgrid4", "msh", 0, 0, 0, 0, 1, Z, Z},
    {"grid-attribute-4-grid-8", "objgrid4", "msh", 0, 0, 8, 0, 1, Z, Z},
    {"grid-attribute-4-grid-2", "objgrid4", "msh", 0, 0, 2, 0, 1, Z, Z},
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
        for (size_t i = 0; (argc == 2 || argc == 3) && i < n; i++) {
            if (strcmp(argv[1], CASES[i].name) == 0) {
                c = &CASES[i];
            }
        }
        if (c == NULL) {
            fprintf(stderr, "usage: mesh_pipeline_limits CASE [SALT] | --cases\n");
            return 2;
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;
        BOOL attributes = (c->object != NULL && strcmp(c->object, "objgrid4") == 0) ||
                          strcmp(c->mesh, "msh64") == 0;
        NSString *source = attributes ? [SOURCE stringByAppendingString:ATTRIBUTES] : SOURCE;
        uint32_t salt = 0;
        if (argc == 3) {
            salt = 2166136261u;
            for (const char *at = argv[2]; *at != '\0'; at++) {
                salt = (salt ^ (uint8_t)*at) * 16777619u;
            }
        }
        source = [[NSString stringWithFormat:@"#define SALT %uu\n", salt]
            stringByAppendingString:source];
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
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
