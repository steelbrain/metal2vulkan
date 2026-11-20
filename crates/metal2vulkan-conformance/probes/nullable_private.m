// What Metal answers for a thread pointer that may be null: the owned fixture
// `engine/nullable_private.metal`'s source (its header gives each word), whose `closure_alloc`
// hands out the next element of a thread's array at an index only the run knows, or null, and
// whose `weigh` and `is_null` ask whether the pointer is null and write through it where it is
// not. Runs `nullable_private` over 8 threads in one threadgroup with `in[i] = i + 0.5` (16
// words), and prints each thread's 8 words.
//
// Build (no cargo target needed):
//   clang -fobjc-arc -O1 -fmodules -o nullable_private nullable_private.m \
//     -framework Foundation -framework Metal
// Run: ./nullable_private, MTL_DEBUG_LAYER unset.
//
// WHAT IT MEASURED (Apple M3 Max, macOS 27.0.1): the module docs of
// `runtime_lane::nullable_private` in metal2vulkan-validation before 8871d40bdf.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>

static const char *SOURCE =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "\n"
    "struct Closure {\n"
    "    float w;\n"
    "    int type;\n"
    "    float data[2];\n"
    "};\n"
    "\n"
    "struct Shader {\n"
    "    Closure closure[4];\n"
    "    int num;\n"
    "    int left;\n"
    "};\n"
    "\n"
    "static thread Closure *closure_alloc(thread Shader &sd, int type) {\n"
    "    if (sd.left == 0)\n"
    "        return nullptr;\n"
    "    thread Closure *sc = &sd.closure[sd.num];\n"
    "    sc->type = type;\n"
    "    sd.num++;\n"
    "    sd.left--;\n"
    "    return sc;\n"
    "}\n"
    "\n"
    "__attribute__((noinline)) float weigh(thread Closure *c, float x) {\n"
    "    if (c == nullptr)\n"
    "        return -1.0f;\n"
    "    c->data[0] = x;\n"
    "    c->w = 2.0f * x;\n"
    "    return c->w;\n"
    "}\n"
    "\n"
    "__attribute__((noinline)) float is_null(thread Closure *c) {\n"
    "    return c == nullptr ? 1.0f : 0.0f;\n"
    "}\n"
    "\n"
    "kernel void nullable_private(device float *out [[buffer(0)]],\n"
    "                             device const float *in [[buffer(1)]],\n"
    "                             uint t [[thread_position_in_grid]]) {\n"
    "    switch (t >> 31) {\n"
    "        Shader sd;\n"
    "    default:\n"
    "        sd.num = 0;\n"
    "        sd.left = int(t & 3);\n"
    "        device float *o = out + 8 * t;\n"
    "        for (int j = 0; j < 4; j++)\n"
    "            o[j] = weigh(closure_alloc(sd, j + 1), in[t] + float(j));\n"
    "        o[4] = float(sd.num);\n"
    "        int types = 0;\n"
    "        float data = 0.0f;\n"
    "        for (int i = 0; i < sd.num; i++) {\n"
    "            types += sd.closure[i].type;\n"
    "            data += sd.closure[i].data[0];\n"
    "        }\n"
    "        o[5] = float(types);\n"
    "        o[6] = data;\n"
    "        o[7] = is_null(closure_alloc(sd, 9));\n"
    "    }\n"
    "}\n";

enum { THREADS = 8, WORDS = 8 };

int main(void) {
  @autoreleasepool {
    printf("RESULT debug_layer=%s\n", getenv("MTL_DEBUG_LAYER") ? getenv("MTL_DEBUG_LAYER") : "unset");
    id<MTLDevice> D = MTLCreateSystemDefaultDevice();
    printf("RESULT device=%s\n", D.name.UTF8String);
    id<MTLCommandQueue> Q = [D newCommandQueue];
    NSError *e = nil;
    id<MTLLibrary> lib = [D newLibraryWithSource:[NSString stringWithUTF8String:SOURCE]
                                         options:[MTLCompileOptions new]
                                           error:&e];
    if (!lib) { printf("RESULT library error %s\n", e.localizedDescription.UTF8String); return 1; }
    id<MTLFunction> f = [lib newFunctionWithName:@"nullable_private"];
    id<MTLComputePipelineState> ps = [D newComputePipelineStateWithFunction:f error:&e];
    if (!ps) { printf("RESULT pipeline error %s\n", e.localizedDescription.UTF8String); return 1; }
    id<MTLBuffer> in = [D newBufferWithLength:16 * 4 options:MTLResourceStorageModeShared];
    float *ins = (float *)in.contents;
    for (int i = 0; i < 16; i++) ins[i] = (float)i + 0.5f;
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
      for (int i = 0; i < WORDS; i++) printf(" %g", w[t * WORDS + i]);
      printf("\n");
    }
  }
  return 0;
}
