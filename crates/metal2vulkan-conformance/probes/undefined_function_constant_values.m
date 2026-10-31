// What plain Metal reads for a function constant the function's constant values leave undefined:
// read directly, of each scalar and vector type, and through program-scope constants derived from
// it (which Metal's compiler computes in the module's static initializer), for a constant Metal
// reports required and for one it reports optional (read guarded somewhere as well).
//
// Measured on an Apple M3 Max under macOS 27.0.1, `MTL_DEBUG_LAYER` out of the environment, one
// case a process; docs/API.md's accepted differences ("Function constants") quote the answers.
//
// Build (no cargo target needed):
//   clang -fobjc-arc -O1 -fmodules -o undefined_function_constant_values \
//     undefined_function_constant_values.m -framework Foundation -framework Metal
//
// Cases: `set0` and `set1` (`newFunctionWithName:constantValues:` with constant 17, `b_set`,
// false or true, every other constant undefined), `none` (`newFunctionWithName:`). Each prints
// every constant's `required`, the pipeline's creation, and the 20 words the kernel stores
// (the buffer is filled with 0xAB first). Case `gate` makes kernel `kg`, whose buffer 1 is gated
// on `!g_flag` and on `g_flag` (buffers 1 and 2) with `g_flag` undefined, and prints the buffers
// its pipeline's reflection binds and the words it stores.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>
static const char *SRC =
"#include <metal_stdlib>\nusing namespace metal;\n"
"constant uchar u_req [[function_constant(10)]];\n"
"constant uchar u_opt [[function_constant(11)]];\n"
"constant int i_req [[function_constant(12)]];\n"
"constant float f_req [[function_constant(13)]];\n"
"constant half h_req [[function_constant(14)]];\n"
"constant bool b_req [[function_constant(15)]];\n"
"constant ushort2 v_req [[function_constant(16)]];\n"
"constant bool b_set [[function_constant(17)]];\n"
"constant bool d_or = b_set || ((u_opt | u_req) != 0);\n"
"constant uchar d_add = u_opt + 3;\n"
"constant bool d_def = is_function_constant_defined(u_opt);\n"
"constant bool d_gt = b_set || (u_opt > 1);\n"
"kernel void k(device uint *out [[buffer(0)]], uint t [[thread_position_in_grid]]) {\n"
"  if (t != 0) return;\n"
"  out[0] = u_req; out[1] = u_opt; out[2] = as_type<uint>(i_req); out[3] = as_type<uint>(f_req);\n"
"  out[4] = as_type<ushort>(h_req); out[5] = b_req; out[6] = v_req.x; out[7] = v_req.y;\n"
"  out[8] = d_or; out[9] = d_add; out[10] = d_def; out[11] = d_gt; out[12] = b_set;\n"
"  out[13] = u_opt * 7 + 1; out[14] = u_opt == 0 ? 111 : 222; out[15] = i_req < 5 ? 1 : 2;\n"
"  out[16] = f_req == 0.0f ? 1 : 2; out[17] = isnan(f_req) ? 1 : 2;\n"
"  out[18] = is_function_constant_defined(u_opt) ? u_opt : 99; out[19] = b_req ? 5 : 6;\n"
"}\n"
"constant bool g_flag [[function_constant(20)]];\n"
"constant bool g_not = !g_flag;\n"
"kernel void kg(device uint *out [[buffer(0)]], device uint *on_not [[buffer(1), function_constant(g_not)]],\n"
"  device uint *on_flag [[buffer(2), function_constant(g_flag)]], uint t [[thread_position_in_grid]]) {\n"
"  if (t != 0) return;\n"
"  out[0] = g_not ? on_not[0] : 2; out[1] = g_flag ? on_flag[0] : 4; out[2] = g_not; out[3] = g_flag;\n"
"}\n";

int main(int argc, char **argv) {
  if (argc < 2) return 2;
  const char *c = argv[1];
  @autoreleasepool {
    printf("RESULT case=%s debug_layer=%s\n", c, getenv("MTL_DEBUG_LAYER") ? getenv("MTL_DEBUG_LAYER") : "unset");
    id<MTLDevice> D = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> Q = [D newCommandQueue];
    NSError *e = nil;
    id<MTLLibrary> L = [D newLibraryWithSource:[NSString stringWithUTF8String:SRC] options:nil error:&e];
    if (!L) { printf("RESULT library error %s\n", e.localizedDescription.UTF8String); return 1; }
    if (strcmp(c, "gate") == 0) {
      MTLFunctionConstantValues *v = [MTLFunctionConstantValues new];
      id<MTLFunction> g = [L newFunctionWithName:@"kg" constantValues:v error:&e];
      if (!g) { printf("RESULT function error %s\n", e.localizedDescription.UTF8String); return 0; }
      MTLComputePipelineReflection *r = nil;
      id<MTLComputePipelineState> ps = [D newComputePipelineStateWithFunction:g options:MTLPipelineOptionBindingInfo reflection:&r error:&e];
      if (!ps) { printf("RESULT pipeline error %s\n", e.localizedDescription.UTF8String); return 0; }
      for (id<MTLBinding> b in r.bindings) printf("RESULT binding %s index=%lu used=%d\n", b.name.UTF8String, (unsigned long)b.index, b.used);
      id<MTLBuffer> out = [D newBufferWithLength:64 options:MTLResourceStorageModeShared];
      id<MTLBuffer> one = [D newBufferWithLength:64 options:MTLResourceStorageModeShared];
      id<MTLBuffer> two = [D newBufferWithLength:64 options:MTLResourceStorageModeShared];
      memset(out.contents, 0xAB, 64); ((uint32_t *)one.contents)[0] = 111; ((uint32_t *)two.contents)[0] = 222;
      id<MTLCommandBuffer> cb = [Q commandBuffer];
      id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
      [ce setComputePipelineState:ps];
      [ce setBuffer:out offset:0 atIndex:0]; [ce setBuffer:one offset:0 atIndex:1]; [ce setBuffer:two offset:0 atIndex:2];
      [ce dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [ce endEncoding]; [cb commit]; [cb waitUntilCompleted];
      const uint32_t *w = out.contents;
      printf("RESULT status=%ld words %08x %08x %08x %08x\n", (long)cb.status, w[0], w[1], w[2], w[3]);
      return 0;
    }
    id<MTLFunction> plain = [L newFunctionWithName:@"k"];
    NSArray *keys = [plain.functionConstantsDictionary.allKeys sortedArrayUsingSelector:@selector(compare:)];
    for (NSString *k in keys) {
      MTLFunctionConstant *fc = plain.functionConstantsDictionary[k];
      printf("RESULT constant %lu %s type=%ld required=%d\n", (unsigned long)fc.index, fc.name.UTF8String, (long)fc.type, fc.required);
    }
    id<MTLFunction> f = plain;
    if (strcmp(c, "none") != 0) {
      MTLFunctionConstantValues *v = [MTLFunctionConstantValues new];
      BOOL b = strcmp(c, "set1") == 0;
      [v setConstantValue:&b type:MTLDataTypeBool atIndex:17];
      f = [L newFunctionWithName:@"k" constantValues:v error:&e];
      if (!f) { printf("RESULT function error %s\n", e.localizedDescription.UTF8String); return 0; }
    }
    id<MTLComputePipelineState> ps = [D newComputePipelineStateWithFunction:f error:&e];
    if (!ps) { printf("RESULT pipeline error %s\n", e.localizedDescription.UTF8String); return 0; }
    id<MTLBuffer> out = [D newBufferWithLength:256 options:MTLResourceStorageModeShared];
    memset(out.contents, 0xAB, 256);
    id<MTLCommandBuffer> cb = [Q commandBuffer];
    id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
    [ce setComputePipelineState:ps];
    [ce setBuffer:out offset:0 atIndex:0];
    [ce dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [ce endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    printf("RESULT status=%ld\n", (long)cb.status);
    const uint32_t *w = out.contents;
    printf("RESULT words");
    for (int i = 0; i < 20; i++) printf(" %08x", w[i]);
    printf("\n");
  }
  return 0;
}
