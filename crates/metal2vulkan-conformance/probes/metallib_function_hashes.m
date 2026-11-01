// What plain Metal answers for a metallib whose function table's `HASH` tags are not the SHA-256
// of the bitcode the functions name: as written (a Core Image kernel library's, which Apple's
// `metallib -cikernel` writes so for some modules), and with one bit of every `HASH` flipped (an
// executable library's and a Core Image kernel library's).
//
// Measured on an Apple M3 Max under macOS 27.0.1, `MTL_DEBUG_LAYER` out of the environment;
// docs/API.md ("A library") states the answers.
//
// Build (no cargo target needed):
//   clang -fobjc-arc -O1 -fmodules -o metallib_function_hashes metallib_function_hashes.m \
//     -framework Foundation -framework Metal
//
// The libraries are written from scratch with Apple's toolchain, in a scratch directory:
//   kernel.metal:  #include <metal_stdlib>
//                  using namespace metal;
//                  kernel void k(device uint *o [[buffer(0)]], uint t [[thread_position_in_grid]])
//                  { o[t] = t * 3 + 1; }
//     xcrun metal -c kernel.metal -o kernel.air && xcrun metallib kernel.air -o kernel.metallib
//   ci.metal:      #include <CoreImage/CoreImage.h>
//                  using namespace metal;
//                  extern "C" { namespace coreimage {
//                  float4 k2(sample_t s, float a, float b) { float4 x = s * a + b;
//                    for (int j = 0; j < 3; j++) x = sin(x) * 2.0 + cos(x*b); return x; }
//                  }}
//     xcrun metal -fcikernel -c ci.metal -o ci.air && xcrun metallib -cikernel ci.air -o ci.metallib
//   four.metal:    ci.metal's prelude, then four kernels:
//                  float4 tint(sample_t s, float4 c) { return s * c; }
//                  float4 shift(sample_t s, float k) { return s + k; }
//                  float4 invert(sample_t s) { return float4(1.0 - s.rgb, s.a); }
//                  float4 scale(sample_t s, float k) { return s * k; }
//     (built as ci.metal is)
// (`ci.metallib`'s `HASH` is not its module's SHA-256 under Xcode 27.2's toolchain;
// `kernel.metallib`'s and `four.metallib`'s are.) Any other metallib may be named, the system's
// own included; the probe reads it at run time and keeps nothing of it.
//
// Usage: metallib_function_hashes <file.metallib> <as-is|flip-hash>. Prints whether
// `newLibraryWithData:error:` made a library, its type, `functionNames`, the private
// `externFunctionNames` (its order: the table's, its reverse, sorted, or other), how many table
// entries `newFunctionWithName:` makes and of which `functionType`s, and for the first kernel
// (`functionType` 3) a compute pipeline's creation and the first four words a four-thread dispatch
// stores.
//
// Measured: every case makes its library, as written and with every `HASH` flipped alike.
// `kernel.metallib`: type 0, one function name, no extern names, `k` of `functionType` 3, whose
// pipeline stores 1,4,7,10. `ci.metallib`: type 0, no function names, one extern name, `k2` of
// `functionType` 4. `four.metallib`: the four extern names, in an order that is neither the
// table's, its reverse, nor sorted; each of `functionType` 4.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static uint32_t u32(const uint8_t *p) { return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24; }
static uint16_t u16(const uint8_t *p) { return (uint16_t)(p[0] | p[1] << 8); }

int main(int argc, char **argv) {
  if (argc < 3) return 2;
  @autoreleasepool {
    NSMutableData *bytes = [NSMutableData dataWithContentsOfFile:@(argv[1])];
    if (!bytes || bytes.length < 0x58) { printf("RESULT unreadable\n"); return 1; }
    uint8_t *b = bytes.mutableBytes;
    size_t len = bytes.length;
    BOOL flip = strcmp(argv[2], "flip-hash") == 0;
    printf("RESULT file=%s mode=%s file_type=%u debug_layer=%s\n", argv[1], argv[2], b[10],
           getenv("MTL_DEBUG_LAYER") ? getenv("MTL_DEBUG_LAYER") : "unset");

    // The function table: a u32 count, then entries of a u32 size (counting itself) and a tag
    // list (`fourcc`, u16 length, bytes; `ENDT` ends it).
    uint64_t list = 0;
    memcpy(&list, b + 24, 8);
    NSMutableArray<NSString *> *names = [NSMutableArray array];
    if (list + 4 > len) { printf("RESULT bad-table\n"); return 1; }
    uint32_t count = u32(b + list);
    size_t at = list + 4;
    for (uint32_t i = 0; i < count && at + 4 <= len; i++) {
      size_t end = at + u32(b + at), t = at + 4;
      while (t + 4 <= end && memcmp(b + t, "ENDT", 4) != 0 && t + 6 <= end) {
        uint16_t n = u16(b + t + 4);
        if (memcmp(b + t, "NAME", 4) == 0 && n > 0)
          [names addObject:[NSString stringWithUTF8String:(const char *)b + t + 6]];
        if (memcmp(b + t, "HASH", 4) == 0 && n == 32 && flip) b[t + 6] ^= 1;
        t += 6 + n;
      }
      at = end;
    }

    id<MTLDevice> D = MTLCreateSystemDefaultDevice();
    dispatch_data_t data = dispatch_data_create(b, len, nil, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    NSError *err = nil;
    id<MTLLibrary> lib = [D newLibraryWithData:data error:&err];
    if (!lib) {
      printf("RESULT library=nil error=%s\n", err.localizedDescription.UTF8String);
      return 0;
    }
    printf("RESULT library=ok type=%ld functionNames=%lu", (long)lib.type,
           (unsigned long)lib.functionNames.count);
    SEL extern_names = NSSelectorFromString(@"externFunctionNames");
    if ([lib respondsToSelector:extern_names]) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Warc-performSelector-leaks"
      NSArray *e = [(id)lib performSelector:extern_names];
#pragma clang diagnostic pop
      BOOL same = e.count == names.count, reversed = same;
      for (NSUInteger i = 0; same && i < e.count; i++) same = [e[i] isEqual:names[i]];
      for (NSUInteger i = 0; reversed && i < e.count; i++)
        reversed = [e[i] isEqual:names[names.count - 1 - i]];
      BOOL sorted = YES;
      for (NSUInteger i = 1; sorted && i < e.count; i++) sorted = [e[i - 1] compare:e[i]] < 0;
      BOOL set = [[NSSet setWithArray:e] isEqual:[NSSet setWithArray:names]];
      printf(" externFunctionNames=%lu order=%s same_set=%s", (unsigned long)e.count,
             e.count < 2 ? "one" : same ? "table" : reversed ? "reverse" : sorted ? "sorted" : "other",
             set ? "yes" : "no");
    }
    printf(" table=%lu\n", (unsigned long)names.count);
    // Every table entry through `newFunctionWithName:`, tallied by `functionType`; the first
    // kernel's pipeline is made and dispatched.
    long made = 0, types[16] = {0};
    BOOL dispatched = NO;
    for (NSString *name in names) {
      id<MTLFunction> f = [lib newFunctionWithName:name];
      if (!f) continue;
      made++;
      if ((unsigned long)f.functionType < 16) types[f.functionType]++;
      if (dispatched || f.functionType != MTLFunctionTypeKernel) continue;
      dispatched = YES;
      id<MTLComputePipelineState> p = [D newComputePipelineStateWithFunction:f error:&err];
      if (!p) {
        printf("RESULT pipeline=nil error=%s\n", err.localizedDescription.UTF8String);
        continue;
      }
      id<MTLBuffer> out = [D newBufferWithLength:16 options:MTLResourceStorageModeShared];
      memset(out.contents, 0xAB, 16);
      id<MTLCommandQueue> Q = [D newCommandQueue];
      id<MTLCommandBuffer> cb = [Q commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:p];
      [enc setBuffer:out offset:0 atIndex:0];
      [enc dispatchThreads:MTLSizeMake(4, 1, 1) threadsPerThreadgroup:MTLSizeMake(4, 1, 1)];
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      const uint32_t *w = out.contents;
      printf("RESULT pipeline=ok words=%u,%u,%u,%u\n", w[0], w[1], w[2], w[3]);
    }
    printf("RESULT newFunctionWithName made=%ld of %lu functionType", made,
           (unsigned long)names.count);
    for (int t = 0; t < 16; t++)
      if (types[t]) printf(" %d:%ld", t, types[t]);
    printf("\n");
  }
  return 0;
}
