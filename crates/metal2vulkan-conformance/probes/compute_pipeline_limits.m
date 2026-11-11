// What a compute pipeline's creation answers for the descriptor's `maxTotalThreadsPerThreadgroup`
// beside the kernel's own `[[max_total_threads_per_threadgroup(n)]]` and the descriptor's
// `requiredThreadsPerThreadgroup`, what the made state's `maxTotalThreadsPerThreadgroup` answers,
// and what a dispatch of another threadgroup size than the required one runs.
//
// Build (no cargo target needed):
//   clang -fobjc-arc -O1 -fmodules -o compute_pipeline_limits compute_pipeline_limits.m \
//     -framework Foundation -framework Metal
//
// `compute_pipeline_limits CASE [SALT]` makes one compute pipeline of the library below compiled
// from source, through a descriptor (`newComputePipelineStateWithDescriptor:options:reflection:
// error:`) or, for a `function-` case, from the function alone
// (`newComputePipelineStateWithFunction:error:`), and prints `made` and the state's
// `maxTotalThreadsPerThreadgroup`, or `refused` and the error's domain, code and description. A
// `dispatch-` case then encodes one dispatch of `count` and prints what ran: how many threads
// added to word 0, and the `threads_per_threadgroup` threadgroup 0's thread 0 stored in words 1 to
// 3. A SALT's 32-bit FNV-1a hash is the source's `SALT`, which each kernel compares a thread's
// index with, so it is in each kernel's AIR: a salt never used before compiles kernels the
// compiler has never seen (a first compile), and the same salt again compiles ones it has.
// Without a salt `SALT` is 0. `--cases` lists the cases. Run one case a process: a stop of
// Metal's ends only its own.
//
// Measured on an M3 Max under macOS 27.0.1, plain and with `MTL_DEBUG_LAYER=1`, which agree on
// every creation:
// - Refused, code 3: a maximum of 1025 or 2048 beside a kernel without one ("Specified total max
//   threads per threadgroup (1025) exceeds the maximum total threads per threadgroup supported
//   (1024)"); a maximum of 32, 128 or 2048 beside the kernel of 64 ("Kernel specified max total
//   threads per threadgroup (64) must match compute pipeline specified max total threads per
//   threadgroup (32)").
// - Made: 64 beside the kernel of 64; 96 and 1024 beside the kernel without one.
// - A dispatch of another size than the required 8x8 runs the size given on plain Metal:
//   `dispatchThreadgroups` at 4x4 ran 32 threads of `threads_per_threadgroup` (4, 4, 1), at 16x4
//   128, at 16x8 256, and `dispatchThreads` of 16x16 at 4x4 256 threads of (4, 4, 1). API
//   validation stops each (exit 134).
//
// Measured on an M3 Max under macOS 27.0.1, plain:
// - Descriptor validation ends the process for a required size with an axis of 0
//   (`required-4x0`, "must not be 0"), of more than 1024 threads (`required-2048`, "must be <=
//   1024 (device limit)") and of more threads than the descriptor's maximum
//   (`required-64-max-32`, "must be <= maxTotalThreadsPerThreadgroup (32)").
// - Each a first compile (a salt never used before), and the same again in its salt: refused,
//   code 3, `required-32-max-64` ("Kernel specified max total threads per threadgroup (32) must
//   match compute pipeline specified max total threads per threadgroup (64)") and
//   `required-128-attribute-64` ("requested required threads per threadgroup 128,1,1 but
//   'count64' expects max threads per threadgroup 64"); made, `required-32-attribute-64`
//   answering 32, `required-8x8` 64, `attribute-64` and `function-attribute-64` 64, `plain` 1024.
// - Later compiles of the same kernel answer from what the compiler compiled before, which it
//   keeps across processes. In one salt, after `required-32-attribute-64`: `attribute-64`,
//   `required-128-attribute-64` and `function-attribute-64` are made and answer 32. In another,
//   after `attribute-64`: `required-32-attribute-64` and `required-128-attribute-64` answer 64.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>

static NSString *const SOURCE =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void count(device atomic_uint *out [[buffer(0)]],\n"
     "                  uint3 tptg [[threads_per_threadgroup]],\n"
     "                  uint3 group [[threadgroup_position_in_grid]],\n"
     "                  uint tid [[thread_index_in_threadgroup]]) {\n"
     "  atomic_fetch_add_explicit(&out[0], 1, memory_order_relaxed);\n"
     "  if ((tid == 0 || tid == SALT) && group.x == 0 && group.y == 0 && group.z == 0) {\n"
     "    atomic_store_explicit(&out[1], tptg.x, memory_order_relaxed);\n"
     "    atomic_store_explicit(&out[2], tptg.y, memory_order_relaxed);\n"
     "    atomic_store_explicit(&out[3], tptg.z, memory_order_relaxed);\n"
     "  }\n"
     "}\n"
     "[[max_total_threads_per_threadgroup(64)]]\n"
     "kernel void count64(device atomic_uint *out [[buffer(0)]],\n"
     "                    uint tid [[thread_index_in_threadgroup]]) {\n"
     "  atomic_fetch_add_explicit(&out[tid == SALT && tid != 0 ? 1 : 0], 1, memory_order_relaxed);\n"
     "}\n";

// How a case makes its pipeline and what it dispatches.
typedef enum { DESCRIPTOR, FUNCTION } Making;
typedef enum { NONE, GROUPS, THREADS } Dispatch;

// A case: the kernel, how it is made, the descriptor's maximum and required size, and the
// dispatch (its grid, in threadgroups or threads, and its threads per threadgroup).
typedef struct {
    const char *name;
    const char *kernel;
    Making making;
    NSUInteger max;
    MTLSize required;
    Dispatch dispatch;
    MTLSize grid, threads_per_group;
} Case;

#define Z {0, 0, 0}
static const Case CASES[] = {
    {"plain", "count", DESCRIPTOR, 0, Z, NONE, Z, Z},
    {"max-96", "count", DESCRIPTOR, 96, Z, NONE, Z, Z},
    {"max-1024", "count", DESCRIPTOR, 1024, Z, NONE, Z, Z},
    {"max-1025", "count", DESCRIPTOR, 1025, Z, NONE, Z, Z},
    {"max-2048", "count", DESCRIPTOR, 2048, Z, NONE, Z, Z},
    {"attribute-64", "count64", DESCRIPTOR, 0, Z, NONE, Z, Z},
    {"attribute-64-max-32", "count64", DESCRIPTOR, 32, Z, NONE, Z, Z},
    {"attribute-64-max-64", "count64", DESCRIPTOR, 64, Z, NONE, Z, Z},
    {"attribute-64-max-128", "count64", DESCRIPTOR, 128, Z, NONE, Z, Z},
    {"attribute-64-max-2048", "count64", DESCRIPTOR, 2048, Z, NONE, Z, Z},
    {"function-plain", "count", FUNCTION, 0, Z, NONE, Z, Z},
    {"function-attribute-64", "count64", FUNCTION, 0, Z, NONE, Z, Z},
    {"required-8x8", "count", DESCRIPTOR, 0, {8, 8, 1}, NONE, Z, Z},
    {"required-4x0", "count", DESCRIPTOR, 0, {4, 0, 1}, NONE, Z, Z},
    {"required-2048", "count", DESCRIPTOR, 0, {2048, 1, 1}, NONE, Z, Z},
    {"required-64-max-32", "count", DESCRIPTOR, 32, {64, 1, 1}, NONE, Z, Z},
    {"required-32-max-64", "count", DESCRIPTOR, 64, {32, 1, 1}, NONE, Z, Z},
    {"required-32-attribute-64", "count64", DESCRIPTOR, 0, {32, 1, 1}, NONE, Z, Z},
    {"required-128-attribute-64", "count64", DESCRIPTOR, 0, {128, 1, 1}, NONE, Z, Z},
    {"dispatch-groups-plain-4x4", "count", DESCRIPTOR, 0, Z, GROUPS, {2, 1, 1}, {4, 4, 1}},
    {"dispatch-groups-required-8x8-at-8x8", "count", DESCRIPTOR, 0, {8, 8, 1}, GROUPS, {2, 1, 1},
     {8, 8, 1}},
    {"dispatch-groups-required-8x8-at-4x4", "count", DESCRIPTOR, 0, {8, 8, 1}, GROUPS, {2, 1, 1},
     {4, 4, 1}},
    {"dispatch-groups-required-8x8-at-16x4", "count", DESCRIPTOR, 0, {8, 8, 1}, GROUPS, {2, 1, 1},
     {16, 4, 1}},
    {"dispatch-groups-required-8x8-at-16x8", "count", DESCRIPTOR, 0, {8, 8, 1}, GROUPS,
     {2, 1, 1}, {16, 8, 1}},
    {"dispatch-threads-required-8x8-at-8x8", "count", DESCRIPTOR, 0, {8, 8, 1}, THREADS,
     {16, 16, 1}, {8, 8, 1}},
    {"dispatch-threads-required-8x8-at-4x4", "count", DESCRIPTOR, 0, {8, 8, 1}, THREADS,
     {16, 16, 1}, {4, 4, 1}},
};

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
            fprintf(stderr, "usage: compute_pipeline_limits CASE [SALT] | --cases\n");
            return 2;
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;
        uint32_t salt = 0;
        if (argc == 3) {
            salt = 2166136261u;
            for (const char *at = argv[2]; *at != '\0'; at++) {
                salt = (salt ^ (uint8_t)*at) * 16777619u;
            }
        }
        NSString *source = [[NSString stringWithFormat:@"#define SALT %uu\n", salt]
            stringByAppendingString:SOURCE];
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        if (library == nil) {
            printf("library %s\n", error.localizedDescription.UTF8String);
            return 1;
        }
        id<MTLFunction> function = [library newFunctionWithName:@(c->kernel)];
        printf("%s on %s\n", c->name, device.name.UTF8String);
        id<MTLComputePipelineState> state = nil;
        if (c->making == FUNCTION) {
            state = [device newComputePipelineStateWithFunction:function error:&error];
        } else {
            MTLComputePipelineDescriptor *d = [MTLComputePipelineDescriptor new];
            d.computeFunction = function;
            d.maxTotalThreadsPerThreadgroup = c->max;
            d.requiredThreadsPerThreadgroup = c->required;
            state = [device newComputePipelineStateWithDescriptor:d
                                                          options:MTLPipelineOptionNone
                                                       reflection:nil
                                                            error:&error];
        }
        if (state == nil) {
            printf("refused %s %ld %s\n", error.domain.UTF8String, (long)error.code,
                   error.localizedDescription.UTF8String);
            return 0;
        }
        printf("made\n");
        printf("  maxTotalThreadsPerThreadgroup %lu\n",
               (unsigned long)state.maxTotalThreadsPerThreadgroup);
        printf("  threadExecutionWidth %lu\n", (unsigned long)state.threadExecutionWidth);
        if (c->dispatch == NONE) {
            return 0;
        }
        id<MTLBuffer> out = [device newBufferWithLength:16 options:MTLResourceStorageModeShared];
        memset(out.contents, 0, 16);
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:state];
        [enc setBuffer:out offset:0 atIndex:0];
        if (c->dispatch == GROUPS) {
            [enc dispatchThreadgroups:c->grid threadsPerThreadgroup:c->threads_per_group];
        } else {
            [enc dispatchThreads:c->grid threadsPerThreadgroup:c->threads_per_group];
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        const uint32_t *w = out.contents;
        printf("  status %ld error %s\n", (long)cb.status,
               cb.error ? cb.error.localizedDescription.UTF8String : "none");
        printf("  ran %u threads_per_threadgroup %u,%u,%u\n", w[0], w[1], w[2], w[3]);
    }
    return 0;
}
