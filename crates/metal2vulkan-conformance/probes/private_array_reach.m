// What Metal answers for a kernel written for its largest tile: each thread keeps an accumulator
// `float acc[32][32][4][4][4]` (256 KiB) of which the pipeline's tile sizes (function constants)
// reach a corner, and hands each sum to a linked visible function. The owned fixtures
// `engine/private_reach_tiles.metal` and `engine/private_reach_functions.metal` hold the same
// source; the runtime lane's `runtime_lane::private_reach` holds the answers.
//
// Build (no cargo target needed):
//   clang -fobjc-arc -O1 -fmodules -o private_array_reach private_array_reach.m \
//     -framework Foundation -framework Metal
// Run: ./private_array_reach tiles   (or rows), one case a process, MTL_DEBUG_LAYER unset.
//
// `tiles` makes `private_reach_tiles` with tile_p 2 and tile_o 4, linking `postfix`, runs 4
// threads over `in[i] = i` and prints each thread's 32 words. `rows` makes `private_reach_rows`
// (its outer loop runs `in[64]` times, here 2, a count only the run knows) and prints the
// creation's answer and, where it made one, the same words.
//
// WHAT IT MEASURED (Apple M3 Max, macOS 27.0.1): `tiles` made its pipeline (status 4, completed),
// and word `w < 32` of thread `t` was `2 * (120 + 16 t) * (w + 1) + w` (thread 0: 240 481 722 ...
// 7711; thread 3: 336 673 1010 ... 10783). `rows` made no pipeline: "Compute function exceeds
// available stack space".
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>

static const char *CALLER =
    "#include <metal_stdlib>\nusing namespace metal;\n"
    "constant uint tile_p [[function_constant(0)]];\n"
    "constant uint tile_o [[function_constant(1)]];\n"
    "[[visible]] float postfix(float x, uint w);\n"
    "kernel void private_reach_tiles(device float *out [[buffer(0)]],\n"
    "                                device const float *in [[buffer(1)]],\n"
    "                                uint tid [[thread_position_in_grid]]) {\n"
    "    float acc[32][32][4][4][4];\n"
    "    for (uint p = 0; p < tile_p; p++)\n"
    "        for (uint o = 0; o < tile_o; o++)\n"
    "            for (uint k = 0; k < 4; k++)\n"
    "                acc[p][o][k][0][0] = 0.0f;\n"
    "    for (uint step = 0; step < 16; step++) {\n"
    "        float x = in[(tid + step) % 64];\n"
    "        for (uint p = 0; p < tile_p; p++)\n"
    "            for (uint o = 0; o < tile_o; o++)\n"
    "                for (uint k = 0; k < 4; k++)\n"
    "                    acc[p][o][k][0][0] += x * float(p * 16 + o * 4 + k + 1);\n"
    "    }\n"
    "    for (uint p = 0; p < tile_p; p++)\n"
    "        for (uint o = 0; o < tile_o; o++)\n"
    "            for (uint k = 0; k < 4; k++) {\n"
    "                uint w = (p * tile_o + o) * 4 + k;\n"
    "                out[tid * 128 + w] = postfix(acc[p][o][k][0][0], w);\n"
    "            }\n"
    "}\n"
    "kernel void private_reach_rows(device float *out [[buffer(0)]],\n"
    "                               device const float *in [[buffer(1)]],\n"
    "                               uint tid [[thread_position_in_grid]]) {\n"
    "    float acc[32][32][4][4][4];\n"
    "    uint rows = uint(in[64]);\n"
    "    for (uint p = 0; p < rows; p++)\n"
    "        for (uint o = 0; o < tile_o; o++)\n"
    "            for (uint k = 0; k < 4; k++)\n"
    "                acc[p][o][k][0][0] = 0.0f;\n"
    "    for (uint step = 0; step < 16; step++) {\n"
    "        float x = in[(tid + step) % 64];\n"
    "        for (uint p = 0; p < rows; p++)\n"
    "            for (uint o = 0; o < tile_o; o++)\n"
    "                for (uint k = 0; k < 4; k++)\n"
    "                    acc[p][o][k][0][0] += x * float(p * 16 + o * 4 + k + 1);\n"
    "    }\n"
    "    for (uint p = 0; p < rows; p++)\n"
    "        for (uint o = 0; o < tile_o; o++)\n"
    "            for (uint k = 0; k < 4; k++) {\n"
    "                uint w = (p * tile_o + o) * 4 + k;\n"
    "                out[tid * 128 + w] = postfix(acc[p][o][k][0][0], w);\n"
    "            }\n"
    "}\n";
static const char *CALLEE =
    "#include <metal_stdlib>\nusing namespace metal;\n"
    "[[visible]] float postfix(float x, uint w) { return x * 2.0f + float(w); }\n";

enum { THREADS = 4, WORDS = 128, SHOWN = 32 };

int main(int argc, char **argv) {
  if (argc < 2) return 2;
  const char *c = argv[1];
  @autoreleasepool {
    printf("RESULT case=%s debug_layer=%s\n", c,
           getenv("MTL_DEBUG_LAYER") ? getenv("MTL_DEBUG_LAYER") : "unset");
    id<MTLDevice> D = MTLCreateSystemDefaultDevice();
    printf("RESULT device=%s\n", D.name.UTF8String);
    id<MTLCommandQueue> Q = [D newCommandQueue];
    NSError *e = nil;
    id<MTLLibrary> caller = [D newLibraryWithSource:[NSString stringWithUTF8String:CALLER]
                                            options:nil
                                              error:&e];
    if (!caller) { printf("RESULT caller error %s\n", e.localizedDescription.UTF8String); return 1; }
    id<MTLLibrary> callee = [D newLibraryWithSource:[NSString stringWithUTF8String:CALLEE]
                                            options:nil
                                              error:&e];
    if (!callee) { printf("RESULT callee error %s\n", e.localizedDescription.UTF8String); return 1; }
    MTLFunctionConstantValues *v = [MTLFunctionConstantValues new];
    uint32_t tile_p = 2, tile_o = 4;
    [v setConstantValue:&tile_p type:MTLDataTypeUInt atIndex:0];
    [v setConstantValue:&tile_o type:MTLDataTypeUInt atIndex:1];
    NSString *name = strcmp(c, "rows") == 0 ? @"private_reach_rows" : @"private_reach_tiles";
    id<MTLFunction> kernel = [caller newFunctionWithName:name constantValues:v error:&e];
    if (!kernel) { printf("RESULT kernel error %s\n", e.localizedDescription.UTF8String); return 1; }
    MTLComputePipelineDescriptor *pd = [MTLComputePipelineDescriptor new];
    pd.computeFunction = kernel;
    MTLLinkedFunctions *lf = [MTLLinkedFunctions linkedFunctions];
    lf.functions = @[ [callee newFunctionWithName:@"postfix"] ];
    pd.linkedFunctions = lf;
    id<MTLComputePipelineState> ps = [D newComputePipelineStateWithDescriptor:pd
                                                                      options:0
                                                                   reflection:nil
                                                                        error:&e];
    if (!ps) { printf("RESULT pipeline error %s\n", e.localizedDescription.UTF8String); return 0; }
    printf("RESULT pipeline made\n");
    id<MTLBuffer> in = [D newBufferWithLength:65 * 4 options:MTLResourceStorageModeShared];
    float *ins = (float *)in.contents;
    for (int i = 0; i < 64; i++) ins[i] = (float)i;
    ins[64] = 2.0f;
    id<MTLBuffer> out = [D newBufferWithLength:THREADS * WORDS * 4
                                       options:MTLResourceStorageModeShared];
    memset(out.contents, 0, THREADS * WORDS * 4);
    id<MTLCommandBuffer> cb = [Q commandBuffer];
    id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
    [ce setComputePipelineState:ps];
    [ce setBuffer:out offset:0 atIndex:0];
    [ce setBuffer:in offset:0 atIndex:1];
    [ce dispatchThreads:MTLSizeMake(THREADS, 1, 1) threadsPerThreadgroup:MTLSizeMake(THREADS, 1, 1)];
    [ce endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    printf("RESULT status=%ld\n", (long)cb.status);
    const float *w = (const float *)out.contents;
    for (int t = 0; t < THREADS; t++) {
      printf("RESULT thread %d:", t);
      for (int i = 0; i < SHOWN; i++) printf(" %g", w[t * WORDS + i]);
      printf("\n");
    }
  }
  return 0;
}
