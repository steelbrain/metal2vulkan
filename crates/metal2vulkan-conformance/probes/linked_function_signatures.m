// What a compute pipeline makes and computes when its kernel calls a visible function through a
// declaration whose type differs from the linked definition's: more or fewer arguments than the
// definition has parameters, an argument or a return of another type of the same or another
// width, a pointer for a value and a value for a pointer, a pointer of another address space, and
// aggregates.
//
// Build (no cargo target needed):
//   clang -fobjc-arc -O1 -fmodules -o linked_function_signatures linked_function_signatures.m \
//     -framework Foundation -framework Metal
//
// Run one case a process: `linked_function_signatures list` prints the cases' names and
// `linked_function_signatures CASE DIR` runs one, DIR the validation crate's
// `fixtures/public/engine`. The caller library is the owned fixture
// `linked_signature_caller.metal` and the callee library `linked_signature_functions.metal`, both
// compiled from source (`newLibraryWithSource:`); the case `stitched_extra` links instead the
// function Metal's stitcher makes of the caller's `stitch_add(in0, in1)`, `stitched_pair`. Kernel
// `k_CASE` calls the function once on `in = {2, 3, 4, 5}` (a device buffer, a constant one for
// `device_for_constant`), with `out` filled with 0xffffffff beforehand: word 0 receives the bits
// of what the call returned, and word 1 the sentinel 0x5e47 after it. The definition is linked in
// `functions`, in `privateFunctions` for a case whose name ends in `_private` (its kernel is the
// one of the name without it). Nothing dereferences a pointer the reference did not pass.
//
// The double case compiles `float double_param(float a, double b)` alone.
//
// WHAT IT MEASURED on an Apple M3 Max under macOS 27.0.1, `MTL_DEBUG_LAYER` out of the
// environment (potluck jobs m3-max-3238, with the same sources inline, and m3-max-3300): every
// pipeline was made, every command buffer completed (status 4) with no error, and every kernel
// stored word 1. Word 0, by case (reference's type -> definition's):
//
//   stitched_extra, _private  (f, f, f) -> stitched (f, f)   0x40a00000  5: the third dropped
//   extra, extra_private      (f, f, f) -> (f, f)            0x40a00000  5
//   extra_two                 (f, f, f, f) -> (f)            0x40000000  2
//   extra_pointer             (f, f, device*) -> (f, f)      0x40a00000  5
//   missing                   (f, f) -> (f, f, f) a + b + c  0x40a00000  5: c read 0
//   missing_unused            (f, f) -> (f, f, f) a + b      0x40a00000  5
//   missing_pointer_unused    (f, f) -> (f, f, device*)      0x40a00000  5
//   int_param                 (f, f, f) -> (f, int, f)       0x4e808000  2 + 3.0's bits
//   half2_param               (f, f) -> (f, half2)           0x40840000  4.125: 3.0's bits
//   uint2_param               (float2) -> (uint2)            0x4f004000  the bits as uints
//   int_return                int (f, f) -> float            0x40a00000  5.0's bits
//   pair_for_floats           (f, f) -> (struct {f, f})      0x40a00000  5: its fields
//   uints_for_floats          (f, f) -> (struct {u, u})      0x4f004000  the bits as uints
//   bytes_extra               (B8, f, f) -> (B8, f)          0x40400000  3
//   void_for_value            void (f, f) -> float           0x00000007  the call ran
//   pointer_for_value         (ulong 0) -> (device*)         0x3f800000  p == nullptr
//   half_param                (f, f) -> (f, half)            nothing stored (0xffffffff)
//   uint_for_half             uint (f, half) -> (f, uint)    nothing stored
//   long_param                (f, f) -> (f, long)            nothing stored
//   float2_for_two            (float2, f) -> (f, f)          nothing stored
//   bool_param                (f, f) -> (bool, f)            nothing stored
//   float_for_half_return     float -> half                  nothing stored
//   half_for_float_return     half -> float                  nothing stored
//   value_for_void            float -> void                  nothing stored
//   value_for_pointer         (device*) -> (f)               nothing stored
//   constant_for_device       (device*) -> (constant*)       nothing stored
//   device_for_constant       (constant*) -> (device*)       nothing stored
//   uint_array_for_float_array (U6) -> (F6)                  nothing stored
//   double_param              "'double' is not supported in Metal"
//
// So Metal resolves a reference by name whatever the types, and passes the arguments by
// position: one past the definition's parameters is dropped, a parameter past the arguments reads
// 0, and a plain value of the same width passes its bits, as a result does. Where the widths or
// kinds differ the read is undefined: the compiler drops the store of a value computed from it.
// The consumer's probes on a Radeon Pro 5300M and a UHD Graphics 630 under macOS 26.7.1 agree on
// the defined cases; there the 5300M read a `float` passed to a `half` as its low 16 bits, a value
// for a pointer as the pointer's low 32 bits, and a `constant` pointer for a `device` one through
// it, where the 630 stored nothing. The translator's tests (`tests::engine_linked_signatures`,
// `runtime_lane::linked_signatures`) hold these answers, from
// `metal2vulkan_devtools::linked_signatures`.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>

static const char *CASES[] = {
    "stitched_extra", "stitched_extra_private", "extra", "extra_private", "extra_two",
    "extra_pointer", "missing", "missing_unused", "missing_pointer_unused", "int_param",
    "half2_param", "uint2_param", "half_param", "uint_for_half", "long_param", "float2_for_two",
    "bool_param", "int_return", "float_for_half_return", "half_for_float_return",
    "value_for_void", "void_for_value", "value_for_pointer", "pointer_for_value",
    "constant_for_device", "device_for_constant", "pair_for_floats", "uints_for_floats",
    "bytes_extra", "uint_array_for_float_array", "double_param",
};

static NSString *str(const char *s) { return [NSString stringWithUTF8String:s]; }

static id<MTLLibrary> compile(id<MTLDevice> D, NSString *dir, NSString *name) {
  NSError *e = nil;
  NSString *path = [dir stringByAppendingPathComponent:name];
  NSString *source = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&e];
  if (!source) {
    printf("RESULT read error %s\n", e.localizedDescription.UTF8String);
    return nil;
  }
  id<MTLLibrary> lib = [D newLibraryWithSource:source options:nil error:&e];
  if (!lib) printf("RESULT %s error %s\n", name.UTF8String, e.localizedDescription.UTF8String);
  return lib;
}

int main(int argc, char **argv) {
  if (argc < 2) return 2;
  const char *c = argv[1];
  if (strcmp(c, "list") == 0) {
    for (size_t i = 0; i < sizeof CASES / sizeof *CASES; i++) printf("%s\n", CASES[i]);
    return 0;
  }
  if (argc < 3) return 2;
  @autoreleasepool {
    id<MTLDevice> D = MTLCreateSystemDefaultDevice();
    printf("RESULT case=%s device=%s debug_layer=%s\n", c, D.name.UTF8String,
           getenv("MTL_DEBUG_LAYER") ? getenv("MTL_DEBUG_LAYER") : "unset");
    NSError *e = nil;
    if (strcmp(c, "double_param") == 0) {
      // Whether a definition can take a double at all.
      id<MTLLibrary> l = [D newLibraryWithSource:
          @"#include <metal_stdlib>\nusing namespace metal;\n"
           "[[visible]] float double_param(float a, double b) { return a + float(b); }\n"
                                         options:nil
                                           error:&e];
      printf("RESULT %s\n", l ? "double compiled"
                              : [NSString stringWithFormat:@"double source error %@",
                                    e.localizedDescription].UTF8String);
      return 0;
    }
    NSString *dir = str(argv[2]);
    id<MTLLibrary> lib = compile(D, dir, @"linked_signature_caller.metal");
    id<MTLLibrary> callee = compile(D, dir, @"linked_signature_functions.metal");
    if (!lib || !callee) return 1;
    NSString *name = str(c);
    BOOL private_ = [name hasSuffix:@"_private"];
    if (private_) name = [name substringToIndex:name.length - strlen("_private")];
    id<MTLFunction> linked = nil;
    if ([name isEqualToString:@"stitched_extra"]) {
      // stitched_pair(a, b) = stitch_add(a, b): a function of two inputs.
      MTLFunctionStitchingFunctionNode *sum = [[MTLFunctionStitchingFunctionNode alloc]
          initWithName:@"stitch_add"
             arguments:@[ [[MTLFunctionStitchingInputNode alloc] initWithArgumentIndex:0],
                          [[MTLFunctionStitchingInputNode alloc] initWithArgumentIndex:1] ]
          controlDependencies:@[]];
      MTLStitchedLibraryDescriptor *sd = [MTLStitchedLibraryDescriptor new];
      sd.functionGraphs = @[ [[MTLFunctionStitchingGraph alloc] initWithFunctionName:@"stitched_pair"
                                                                               nodes:@[]
                                                                          outputNode:sum
                                                                          attributes:@[]] ];
      sd.functions = @[ [lib newFunctionWithName:@"stitch_add"] ];
      id<MTLLibrary> stitched = [D newLibraryWithStitchedDescriptor:sd error:&e];
      if (!stitched) { printf("RESULT stitched error %s\n", e.localizedDescription.UTF8String); return 0; }
      linked = [stitched newFunctionWithName:@"stitched_pair"];
    } else {
      linked = [callee newFunctionWithName:name];
    }
    if (!linked) { printf("RESULT no function %s\n", name.UTF8String); return 1; }
    MTLComputePipelineDescriptor *pd = [MTLComputePipelineDescriptor new];
    pd.computeFunction = [lib newFunctionWithName:[@"k_" stringByAppendingString:name]];
    if (!pd.computeFunction) { printf("RESULT no kernel\n"); return 1; }
    MTLLinkedFunctions *lf = [MTLLinkedFunctions linkedFunctions];
    if (private_) lf.privateFunctions = @[ linked ];
    else lf.functions = @[ linked ];
    pd.linkedFunctions = lf;
    id<MTLComputePipelineState> ps = [D newComputePipelineStateWithDescriptor:pd
                                                                      options:0
                                                                   reflection:nil
                                                                        error:&e];
    if (!ps) {
      printf("RESULT pipeline error %s\n", e.localizedDescription.UTF8String);
      return 0;
    }
    const float in[4] = {2, 3, 4, 5};
    id<MTLBuffer> inb = [D newBufferWithBytes:in length:sizeof in options:MTLResourceStorageModeShared];
    id<MTLBuffer> out = [D newBufferWithLength:8 options:MTLResourceStorageModeShared];
    memset(out.contents, 0xff, 8);
    id<MTLCommandQueue> Q = [D newCommandQueue];
    id<MTLCommandBuffer> cb = [Q commandBuffer];
    id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
    [ce setComputePipelineState:ps];
    [ce setBuffer:out offset:0 atIndex:0];
    [ce setBuffer:inb offset:0 atIndex:1];
    [ce dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [ce endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    const uint32_t *w = out.contents;
    float f;
    memcpy(&f, &w[0], 4);
    printf("RESULT made status=%ld word0=0x%08x (%g) word1=0x%x%s\n", (long)cb.status, w[0], f,
           w[1], cb.error ? [NSString stringWithFormat:@" error %@", cb.error.localizedDescription].UTF8String : "");
  }
  return 0;
}
