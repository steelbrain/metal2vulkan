// What Metal answers for a program-scope constant that a function constant makes a division by
// zero, the shape of a convolution kernel whose tile sizes are function-constant bit fields:
// `threads = tile / 4` is 0 when the constant gives a tile of 2, and `stride = 512 / threads` then
// divides by zero in the static initializer (`air.static_init`) that computes the program-scope
// constants. The translator reads such a quotient as undefined, as an unsupplied constant's value
// (`canon::static_init`, `canon::tests::static_divisions`).
//
// Build: clang -fobjc-arc -O1 -fmodules -o static_init_division static_init_division.m \
//          -framework Foundation -framework Metal
// Run: ./static_init_division FC ITER, one case a process, MTL_DEBUG_LAYER unset. FC 1 makes the
// tile 2 and `threads` 0; FC 3 makes the tile 8 and `threads` 2. ITER is the program-scope
// constant `iter`'s initializer: `ushort(1)`, or `ushort(ushort(64) / stride)`.
//
// The kernel writes, for each of 4 threads, words 0 to 4: `stride`, `iter`, `threads`, the sum
// of `iter` reads of `in` (`in[i] = i + 1`) stepped by `stride`, and 11 where `stride` is 0, else
// 22. The output buffer starts filled with 0xee bytes, so an unwritten word reads 0xeeeeeeee.
//
// WHAT IT MEASURED (Apple M3 Max, macOS 27.0.1): every case made its pipeline and completed
// (status 4). FC 3: words 0x100 0x1 0x2 tid+1 0x16 with `iter` 1, and 0x100 0x0 0x2 0x0 0x16
// with `iter` 64 / `stride`. FC 1, `iter` 1: words 1 to 3 written (0x1, 0x0, tid+1), words 0 and 4
// (the quotient and the word chosen by it) left unwritten. FC 1, `iter` 64 / `stride`: no word
// written at all. An `iter` of `max(ushort(64 / stride), ushort(1))` is refused at
// `newFunctionWithName:constantValues:` for FC 1 and FC 3 alike ("cannot have global constructors
// (llvm.global_ctors)"), the division by zero aside.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *SOURCE =
    "#include <metal_stdlib>\nusing namespace metal;\n"
    "constant uint fc [[function_constant(0)]];\n"
    "constant ushort tile = ushort(1) << (fc & 7);\n"
    "constant ushort threads = tile / ushort(4);\n"
    "constant ushort stride = ushort(512) / threads;\n"
    "constant ushort iter = %s;\n"
    "kernel void static_init_division(device uint *out [[buffer(0)]],\n"
    "                                 device const uint *in [[buffer(1)]],\n"
    "                                 uint tid [[thread_position_in_grid]]) {\n"
    "    uint sum = 0;\n"
    "    for (ushort i = 0; i < iter; i++)\n"
    "        sum += in[(tid + uint(i) * uint(stride)) %% 64];\n"
    "    out[tid * 8 + 0] = stride;\n"
    "    out[tid * 8 + 1] = iter;\n"
    "    out[tid * 8 + 2] = threads;\n"
    "    out[tid * 8 + 3] = sum;\n"
    "    out[tid * 8 + 4] = stride == 0 ? 11u : 22u;\n"
    "}\n";

enum { THREADS = 4, WORDS = 8 };

int main(int argc, char **argv) {
  if (argc < 2) return 2;
  uint32_t fc = (uint32_t)strtoul(argv[1], NULL, 0);
  @autoreleasepool {
    printf("RESULT fc=%u debug_layer=%s\n", fc,
           getenv("MTL_DEBUG_LAYER") ? getenv("MTL_DEBUG_LAYER") : "unset");
    id<MTLDevice> D = MTLCreateSystemDefaultDevice();
    printf("RESULT device=%s\n", D.name.UTF8String);
    id<MTLCommandQueue> Q = [D newCommandQueue];
    NSError *e = nil;
    const char *iter = argc > 2 ? argv[2] : "ushort(1)";
    printf("RESULT iter=%s\n", iter);
    id<MTLLibrary> lib = [D newLibraryWithSource:[NSString stringWithFormat:[NSString stringWithUTF8String:SOURCE], iter]
                                         options:nil
                                           error:&e];
    if (!lib) { printf("RESULT library error %s\n", e.localizedDescription.UTF8String); return 1; }
    MTLFunctionConstantValues *v = [MTLFunctionConstantValues new];
    [v setConstantValue:&fc type:MTLDataTypeUInt atIndex:0];
    id<MTLFunction> k = [lib newFunctionWithName:@"static_init_division" constantValues:v error:&e];
    if (!k) { printf("RESULT function error %s\n", e.localizedDescription.UTF8String); return 0; }
    id<MTLComputePipelineState> ps = [D newComputePipelineStateWithFunction:k error:&e];
    if (!ps) { printf("RESULT pipeline error %s\n", e.localizedDescription.UTF8String); return 0; }
    printf("RESULT pipeline made\n");
    id<MTLBuffer> in = [D newBufferWithLength:64 * 4 options:MTLResourceStorageModeShared];
    uint32_t *ins = (uint32_t *)in.contents;
    for (int i = 0; i < 64; i++) ins[i] = (uint32_t)(i + 1);
    id<MTLBuffer> out = [D newBufferWithLength:THREADS * WORDS * 4
                                       options:MTLResourceStorageModeShared];
    memset(out.contents, 0xee, THREADS * WORDS * 4);
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
    const uint32_t *w = (const uint32_t *)out.contents;
    for (int t = 0; t < THREADS; t++) {
      printf("RESULT thread %d:", t);
      for (int i = 0; i < 5; i++) printf(" 0x%x", w[t * WORDS + i]);
      printf("\n");
    }
  }
  return 0;
}
