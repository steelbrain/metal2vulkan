// What Metal answers for a kernel that stores into its private arrays at an index only the run
// knows: the computation of `metal2vulkan-runtime`'s `private_sinks` test (whose kernels are
// owned LLVM IR of the same steps). Each thread `t` of 64 (two threadgroups of 32) fills two
// arrays of `N` words, `a[i] = 31 t + i` and `b[i] = 17 t + 3 i`, stores `t + 100000` into `a`'s
// word at byte offset `4 k` and `t + 200000` into `b[k]`, `k = in[t]`, then folds
// `s = 3 (3 s + a[j]) + b[j]` over `j` and reads byte `(5 t) mod 4N` of `a`.
//
// Build (no cargo target needed):
//   clang -fobjc-arc -O1 -fmodules -o private_sinks private_sinks.m \
//     -framework Foundation -framework Metal
// Run: ./private_sinks 4   (or 2100), one size a process, MTL_DEBUG_LAYER unset.
//
// `in[t]` is `(5 t + 1) mod N`, inside both arrays; an index past them is undefined in Metal and
// is not asked. Each thread's two output words, `s` and the byte, are printed in thread order, and
// compared with the same steps on the CPU (`reference`, the test's own).
//
// WHAT IT MEASURED (Apple M3 Max, macOS 27.0.1, job m3-max-65609): both sizes completed (status
// 4, no error), and all 64 threads equal the reference at N 4 and at N 2100 (thread 0: s
// 40500126, byte 0 at N 4; s 3075928718, byte 0 at N 2100).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *SOURCE =
    "#include <metal_stdlib>\nusing namespace metal;\n"
    "kernel void private_sinks(device uint *out [[buffer(0)]],\n"
    "                          device const uint *in [[buffer(1)]],\n"
    "                          uint t [[thread_position_in_grid]]) {\n"
    "    uint a[N];\n"
    "    uint b[N];\n"
    "    for (uint i = 0; i < N; i++) {\n"
    "        a[i] = 31 * t + i;\n"
    "        b[i] = 17 * t + 3 * i;\n"
    "    }\n"
    "    uint k = in[t];\n"
    "    *(thread uint *)((thread uchar *)a + 4 * k) = t + 100000;\n"
    "    b[k] = t + 200000;\n"
    "    uint s = 0;\n"
    "    for (uint j = 0; j < N; j++) s = 3 * (3 * s + a[j]) + b[j];\n"
    "    uint byte = ((thread uchar *)a)[(5 * t) % (4 * N)];\n"
    "    out[2 * t] = s;\n"
    "    out[2 * t + 1] = byte;\n"
    "}\n";

enum { GROUPS = 2, THREADS = 32 };

// Thread `t`'s two words with `n` words an array, its stores at `k` (which is inside both).
static void reference(uint32_t n, uint32_t t, uint32_t k, uint32_t *s, uint32_t *byte) {
  uint32_t *a = calloc(n, 4), *b = calloc(n, 4);
  for (uint32_t i = 0; i < n; i++) {
    a[i] = 31 * t + i;
    b[i] = 17 * t + 3 * i;
  }
  a[k] = t + 100000;
  b[k] = t + 200000;
  uint32_t f = 0;
  for (uint32_t j = 0; j < n; j++) f = 3 * (3 * f + a[j]) + b[j];
  uint32_t at = (5 * t) % (4 * n);
  *s = f;
  *byte = (a[at / 4] >> (8 * (at % 4))) & 0xff;
  free(a);
  free(b);
}

int main(int argc, char **argv) {
  if (argc < 2) return 2;
  int n = atoi(argv[1]);
  if (n <= 0) return 2;
  @autoreleasepool {
    printf("RESULT n=%d debug_layer=%s\n", n,
           getenv("MTL_DEBUG_LAYER") ? getenv("MTL_DEBUG_LAYER") : "unset");
    id<MTLDevice> D = MTLCreateSystemDefaultDevice();
    printf("RESULT device=%s os=%s\n", D.name.UTF8String,
           NSProcessInfo.processInfo.operatingSystemVersionString.UTF8String);
    id<MTLCommandQueue> Q = [D newCommandQueue];
    MTLCompileOptions *o = [MTLCompileOptions new];
    o.preprocessorMacros = @{@"N" : @(n)};
    NSError *e = nil;
    id<MTLLibrary> lib = [D newLibraryWithSource:[NSString stringWithUTF8String:SOURCE]
                                         options:o
                                           error:&e];
    if (!lib) { printf("RESULT library error %s\n", e.localizedDescription.UTF8String); return 1; }
    id<MTLFunction> f = [lib newFunctionWithName:@"private_sinks"];
    id<MTLComputePipelineState> ps = [D newComputePipelineStateWithFunction:f error:&e];
    if (!ps) { printf("RESULT pipeline error %s\n", e.localizedDescription.UTF8String); return 1; }
    int threads = GROUPS * THREADS;
    id<MTLBuffer> in = [D newBufferWithLength:threads * 4 options:MTLResourceStorageModeShared];
    uint32_t *ins = (uint32_t *)in.contents;
    for (int t = 0; t < threads; t++) ins[t] = (uint32_t)((5 * t + 1) % n);
    id<MTLBuffer> out = [D newBufferWithLength:threads * 8 options:MTLResourceStorageModeShared];
    memset(out.contents, 0xee, threads * 8);
    id<MTLCommandBuffer> cb = [Q commandBuffer];
    id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
    [ce setComputePipelineState:ps];
    [ce setBuffer:out offset:0 atIndex:0];
    [ce setBuffer:in offset:0 atIndex:1];
    [ce dispatchThreadgroups:MTLSizeMake(GROUPS, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(THREADS, 1, 1)];
    [ce endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    printf("RESULT status=%ld error=%s\n", (long)cb.status,
           cb.error ? cb.error.localizedDescription.UTF8String : "none");
    const uint32_t *w = (const uint32_t *)out.contents;
    int same = 0;
    for (int t = 0; t < threads; t++) {
      uint32_t s, byte;
      reference((uint32_t)n, (uint32_t)t, ins[t], &s, &byte);
      same += w[2 * t] == s && w[2 * t + 1] == byte;
      printf("RESULT t=%d s=%u byte=%u reference s=%u byte=%u\n", t, w[2 * t], w[2 * t + 1], s, byte);
    }
    printf("RESULT n=%d threads equal to the reference: %d of %d\n", n, same, threads);
  }
  return 0;
}
