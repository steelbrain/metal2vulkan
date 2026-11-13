// Where plain Metal places a vector load or store whose address is aligned to its element but not
// to the vector: a `float4` at a byte offset that is a multiple of 4 and not of 16, a `half4` at
// one that is a multiple of 2 and not of 8, a `uchar4` at an odd one. The AIR of such an access
// carries the vector type's alignment (`align 16` on a `float4`), which the address does not have.
//
// Each case writes or reads rows of 21 elements, one thread a row: row `p` starts at element
// `21 * p`, so 3 rows in 4 start off the vector's alignment. A store writes, for `i` in 0..3, the
// vector `(p, 4i, 4i + 1, 4i + 2)` at the row's elements `4i .. 4i + 3`, then the scalars `c` at
// its elements 16..20. A load reads the same vectors from a buffer whose element `j` holds `j`,
// and writes each one's lanes plainly. The cases:
//
//   float-store   `device float *`, the store through `(device float4 *)(out + 21 * p)`
//   float-load    the load through `(device float4 *)(in + 21 * p)`, the lanes stored as floats
//   half-store    `device half *` rows of 21 halves (42 bytes), `half4` stores
//   uchar-store   `device uchar *` rows of 21 bytes, `uchar4` stores
//   tg-store      a `threadgroup float` array of 8 rows, written as the float case, copied out
//   const-load    `constant float *`, the load as `float-load`
//   offset-4      `device float4 *v` bound at offset 4: `v[p] = (p, p, p, p)` for 8 threads, read
//                 from the buffer's start, so the first word says where element 0 went
//
// Each case prints the device, then one line per row: `row p: <elements>`. A row placed at the real
// address reads `p 0 1 2 p 4 5 6 p 8 9 10 p 12 13 14 16 17 18 19 20` (the stores), or the element
// indices of its vectors (the loads).
//
// Build (no cargo target needed):
//   clang -fobjc-arc -O1 -fmodules -o vector_access_placement vector_access_placement.m \
//     -framework Foundation -framework Metal
//
// Measuring runs keep `MTL_DEBUG_LAYER` out of the environment; a layer message is taken in a
// separate run with `MTL_DEBUG_LAYER=1 MTL_DEBUG_LAYER_ERROR_MODE=nslog`.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static NSString *const kSource = @
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void float_store(device float *out [[buffer(0)]], uint p [[thread_position_in_grid]]) {\n"
    "  device float4 *v = (device float4 *)(out + 21 * p);\n"
    "  for (uint i = 0; i < 4; i++) v[i] = float4(p, 4 * i, 4 * i + 1, 4 * i + 2);\n"
    "  for (uint c = 16; c < 21; c++) out[21 * p + c] = c;\n"
    "}\n"
    "kernel void float_load(device const float *in [[buffer(0)]], device float *out [[buffer(1)]],\n"
    "                       uint p [[thread_position_in_grid]]) {\n"
    "  device const float4 *v = (device const float4 *)(in + 21 * p);\n"
    "  for (uint i = 0; i < 4; i++) { float4 x = v[i];\n"
    "    for (uint j = 0; j < 4; j++) out[21 * p + 4 * i + j] = x[j]; }\n"
    "  for (uint c = 16; c < 21; c++) out[21 * p + c] = in[21 * p + c];\n"
    "}\n"
    "kernel void half_store(device half *out [[buffer(0)]], uint p [[thread_position_in_grid]]) {\n"
    "  device half4 *v = (device half4 *)(out + 21 * p);\n"
    "  for (uint i = 0; i < 4; i++) v[i] = half4(p, 4 * i, 4 * i + 1, 4 * i + 2);\n"
    "  for (uint c = 16; c < 21; c++) out[21 * p + c] = c;\n"
    "}\n"
    "kernel void uchar_store(device uchar *out [[buffer(0)]], uint p [[thread_position_in_grid]]) {\n"
    "  device uchar4 *v = (device uchar4 *)(out + 21 * p);\n"
    "  for (uint i = 0; i < 4; i++) v[i] = uchar4(p, 4 * i, 4 * i + 1, 4 * i + 2);\n"
    "  for (uint c = 16; c < 21; c++) out[21 * p + c] = c;\n"
    "}\n"
    "kernel void tg_store(device float *out [[buffer(0)]], uint p [[thread_position_in_grid]]) {\n"
    "  threadgroup float t[21 * 8];\n"
    "  threadgroup float4 *v = (threadgroup float4 *)(t + 21 * p);\n"
    "  for (uint i = 0; i < 4; i++) v[i] = float4(p, 4 * i, 4 * i + 1, 4 * i + 2);\n"
    "  for (uint c = 16; c < 21; c++) t[21 * p + c] = c;\n"
    "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  for (uint c = 0; c < 21; c++) out[21 * p + c] = t[21 * p + c];\n"
    "}\n"
    "kernel void const_load(constant float *in [[buffer(0)]], device float *out [[buffer(1)]],\n"
    "                       uint p [[thread_position_in_grid]]) {\n"
    "  constant float4 *v = (constant float4 *)(in + 21 * p);\n"
    "  for (uint i = 0; i < 4; i++) { float4 x = v[i];\n"
    "    for (uint j = 0; j < 4; j++) out[21 * p + 4 * i + j] = x[j]; }\n"
    "  for (uint c = 16; c < 21; c++) out[21 * p + c] = in[21 * p + c];\n"
    "}\n"
    "kernel void offset_store(device float4 *v [[buffer(0)]], uint p [[thread_position_in_grid]]) {\n"
    "  v[p] = float4(p);\n"
    "}\n";

enum { ROWS = 8, ROW = 21 };

static void print_rows(const char *name, int kind, const void *bytes) {
  printf("%s\n", name);
  for (int p = 0; p < ROWS; p++) {
    printf("  row %d:", p);
    for (int j = 0; j < ROW; j++) {
      int at = p * ROW + j;
      if (kind == 0) printf(" %g", ((const float *)bytes)[at]);
      if (kind == 1) printf(" %g", (float)((const __fp16 *)bytes)[at]);
      if (kind == 2) printf(" %u", ((const unsigned char *)bytes)[at]);
    }
    printf("\n");
  }
}

int main(void) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSOperatingSystemVersion v = [[NSProcessInfo processInfo] operatingSystemVersion];
    printf("device %s, macOS %ld.%ld.%ld\n", device.name.UTF8String, (long)v.majorVersion,
           (long)v.minorVersion, (long)v.patchVersion);
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:kSource options:nil error:&error];
    if (!library) {
      printf("library: %s\n", error.description.UTF8String);
      return 1;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    const NSUInteger n = ROWS * ROW;
    float pattern[ROWS * ROW];
    for (NSUInteger j = 0; j < n; j++) pattern[j] = (float)j;
    struct {
      const char *name;
      const char *entry;
      int kind;
      BOOL load;
      NSUInteger element;
    } cases[] = {
        {"float-store", "float_store", 0, NO, 4}, {"float-load", "float_load", 0, YES, 4},
        {"half-store", "half_store", 1, NO, 2},   {"uchar-store", "uchar_store", 2, NO, 1},
        {"tg-store", "tg_store", 0, NO, 4},       {"const-load", "const_load", 0, YES, 4},
    };
    for (size_t k = 0; k < sizeof cases / sizeof cases[0]; k++) {
      id<MTLFunction> f = [library newFunctionWithName:@(cases[k].entry)];
      id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:f error:&error];
      if (!pso) {
        printf("%s: pipeline %s\n", cases[k].name, error.description.UTF8String);
        continue;
      }
      NSUInteger bytes = n * cases[k].element;
      id<MTLBuffer> out = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
      memset(out.contents, 0xEE, bytes);
      id<MTLBuffer> in = [device newBufferWithBytes:pattern
                                             length:sizeof pattern
                                            options:MTLResourceStorageModeShared];
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:pso];
      if (cases[k].load) {
        [enc setBuffer:in offset:0 atIndex:0];
        [enc setBuffer:out offset:0 atIndex:1];
      } else {
        [enc setBuffer:out offset:0 atIndex:0];
      }
      [enc dispatchThreads:MTLSizeMake(ROWS, 1, 1) threadsPerThreadgroup:MTLSizeMake(ROWS, 1, 1)];
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      if (cb.status != MTLCommandBufferStatusCompleted) {
        printf("%s: status %ld %s\n", cases[k].name, (long)cb.status,
               cb.error.description.UTF8String);
        continue;
      }
      print_rows(cases[k].name, cases[k].kind, out.contents);
    }
    // offset-4: a `float4` buffer bound 4 bytes past a 16-byte boundary.
    id<MTLFunction> f = [library newFunctionWithName:@"offset_store"];
    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:f error:&error];
    id<MTLBuffer> out = [device newBufferWithLength:256 options:MTLResourceStorageModeShared];
    memset(out.contents, 0xEE, 256);
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:out offset:4 atIndex:0];
    [enc dispatchThreads:MTLSizeMake(ROWS, 1, 1) threadsPerThreadgroup:MTLSizeMake(ROWS, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    printf("offset-4: status %ld; words 0..35:", (long)cb.status);
    for (int j = 0; j < 36; j++) {
      unsigned w = ((const unsigned *)out.contents)[j];
      if (w == 0xEEEEEEEEu) printf(" -");
      else printf(" %g", ((const float *)out.contents)[j]);
    }
    printf("\n");
  }
  return 0;
}
