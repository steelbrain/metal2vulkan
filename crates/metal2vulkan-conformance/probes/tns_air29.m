// What Metal answers for AIR 2.9's private tensor intrinsics, run without a matmul2d: the owned
// fixture `engine/tns_air29.metal`'s source (MSL 4.1), whose 32 threads initialize a tensor over
// device memory and one over threadgroup memory with extents read from a buffer (32, 16, 16, 32)
// and write the extents each answers. Prints the first two threads' words, any thread that differs
// from the buffer's extents, and the count of threads that answer them.
//
// Build (no cargo target needed):
//   clang -fobjc-arc -O1 -fmodules -o tns_air29 tns_air29.m \
//     -framework Foundation -framework Metal
// Run: ./tns_air29, MTL_DEBUG_LAYER unset.
//
// WHAT IT MEASURED (Apple M3 Max, macOS 27.0.1): the module docs of
// `runtime_lane::tensors_air29` in metal2vulkan-validation before 8871d40bdf.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>

static const char *SOURCE =
    "#include <metal_stdlib>\n"
    "#include <metal_tensor>\n"
    "using namespace metal;\n"
    "\n"
    "kernel void tns_air29(device const int8_t *a [[buffer(0)]],\n"
    "                      device const int *ext [[buffer(1)]],\n"
    "                      device int *out [[buffer(2)]],\n"
    "                      uint tid [[thread_index_in_threadgroup]]) {\n"
    "    threadgroup int8_t tb[16 * 32];\n"
    "    using DT = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>;\n"
    "    using TT = tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline>;\n"
    "    DT tA(const_cast<device int8_t *>(a), dextents<int32_t, 2>(ext[0], ext[1]));\n"
    "    TT tB(tb, dextents<int32_t, 2>(ext[2], ext[3]));\n"
    "    out[4 * tid + 0] = tA.get_extent(0);\n"
    "    out[4 * tid + 1] = tA.get_extent(1);\n"
    "    out[4 * tid + 2] = tB.get_extent(0);\n"
    "    out[4 * tid + 3] = tB.get_extent(1);\n"
    "}\n";

enum { THREADS = 32, WORDS = 4 };

int main(void) {
  @autoreleasepool {
    printf("RESULT debug_layer=%s\n", getenv("MTL_DEBUG_LAYER") ? getenv("MTL_DEBUG_LAYER") : "unset");
    id<MTLDevice> D = MTLCreateSystemDefaultDevice();
    printf("RESULT device=%s\n", D.name.UTF8String);
    id<MTLCommandQueue> Q = [D newCommandQueue];
    NSError *e = nil;
    MTLCompileOptions *o = [MTLCompileOptions new];
    o.languageVersion = MTLLanguageVersion4_1;
    id<MTLLibrary> lib = [D newLibraryWithSource:[NSString stringWithUTF8String:SOURCE]
                                         options:o
                                           error:&e];
    if (!lib) {
      printf("RESULT library error %s\n", e.localizedDescription.UTF8String);
      return 1;
    }
    id<MTLFunction> f = [lib newFunctionWithName:@"tns_air29"];
    id<MTLComputePipelineState> ps = [D newComputePipelineStateWithFunction:f error:&e];
    if (!ps) {
      printf("RESULT pipeline error %s\n", e.localizedDescription.UTF8String);
      return 1;
    }
    id<MTLBuffer> a = [D newBufferWithLength:512 options:MTLResourceStorageModeShared];
    memset(a.contents, 1, 512);
    const int extents[4] = {32, 16, 16, 32};
    id<MTLBuffer> ext = [D newBufferWithBytes:extents length:sizeof extents
                                      options:MTLResourceStorageModeShared];
    id<MTLBuffer> out = [D newBufferWithLength:THREADS * WORDS * 4
                                       options:MTLResourceStorageModeShared];
    memset(out.contents, 0, THREADS * WORDS * 4);
    id<MTLCommandBuffer> cb = [Q commandBuffer];
    id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
    [ce setComputePipelineState:ps];
    [ce setBuffer:a offset:0 atIndex:0];
    [ce setBuffer:ext offset:0 atIndex:1];
    [ce setBuffer:out offset:0 atIndex:2];
    [ce dispatchThreads:MTLSizeMake(THREADS, 1, 1) threadsPerThreadgroup:MTLSizeMake(THREADS, 1, 1)];
    [ce endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    printf("RESULT status=%ld\n", (long)cb.status);
    const int *w = (const int *)out.contents;
    int same = 0;
    for (int t = 0; t < THREADS; t++) {
      int ok = 1;
      for (int k = 0; k < WORDS; k++) ok &= w[t * WORDS + k] == extents[k];
      same += ok;
      if (t < 2 || !ok) printf("RESULT thread %d: %d %d %d %d\n", t, w[t * 4], w[t * 4 + 1], w[t * 4 + 2], w[t * 4 + 3]);
    }
    printf("RESULT threads answering the extents: %d of %d\n", same, THREADS);
  }
  return 0;
}
