// What a compute pipeline makes and computes when its kernel calls a visible function through a
// declaration whose type differs from the linked definition's: more or fewer arguments than the
// definition has parameters, an argument or a return of another type of the same or another size,
// a pointer for a value and a value for a pointer, a pointer of another address space, and
// aggregates.
//
// Build (no cargo target needed):
//   clang -fobjc-arc -O1 -fmodules -o linked_function_signatures linked_function_signatures.m \
//     -framework Foundation -framework Metal
//
// Run one case a process: `linked_function_signatures list` prints the cases' names and
// `linked_function_signatures CASE` runs one. The caller library declares each visible function at
// the reference's type and its kernel `k_CASE` calls it once on `in = {2, 3, 4, 5}` (a device
// buffer); the callee library defines it at the definition's type (the case `stitched_extra` links
// a function Metal's stitcher made instead). The definition is linked in `functions` (in
// `privateFunctions` for the cases whose name ends in `_private`). Word 0 of `out` (filled with
// 0xffffffff beforehand) receives the bits of what the call returned, and word 1 the sentinel
// 0x5e47 after it. Nothing dereferences a pointer the reference did not pass.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>

static const char *CALLER =
    "#include <metal_stdlib>\nusing namespace metal;\n"
    "struct B8 { uchar c[8]; };\n"
    "struct U6 { uint a[6]; };\n"
    "[[stitchable]] float stitch_add(float a, float b) { return a + b; }\n"
    "[[visible]] float stitched_pair(float a, float b, float c);\n"
    "[[visible]] float extra(float a, float b, float c);\n"
    "[[visible]] float extra_two(float a, float b, float c, float d);\n"
    "[[visible]] float extra_pointer(float a, float b, device const float *p);\n"
    "[[visible]] float missing(float a, float b);\n"
    "[[visible]] float missing_unused(float a, float b);\n"
    "[[visible]] float missing_pointer_unused(float a, float b);\n"
    "[[visible]] float int_param(float a, float b, float c);\n"
    "[[visible]] float half2_param(float a, float b);\n"
    "[[visible]] float uint2_param(float2 v);\n"
    "[[visible]] float half_param(float a, float b);\n"
    "[[visible]] uint uint_for_half(float a, half b);\n"
    "[[visible]] float long_param(float a, float b);\n"
    "[[visible]] float float2_for_two(float2 v, float c);\n"
    "[[visible]] float bool_param(float a, float b);\n"
    "[[visible]] int int_return(float a, float b);\n"
    "[[visible]] float float_for_half_return(float a, float b);\n"
    "[[visible]] half half_for_float_return(float a, float b);\n"
    "[[visible]] float value_for_void(float a, float b);\n"
    "[[visible]] void void_for_value(float a, float b);\n"
    "[[visible]] float value_for_pointer(device const float *p);\n"
    "[[visible]] float pointer_for_value(ulong p);\n"
    "[[visible]] float constant_for_device(device const float *p);\n"
    "[[visible]] float device_for_constant(constant float *p);\n"
    "[[visible]] float pair_for_floats(float a, float b);\n"
    "[[visible]] float uints_for_floats(float a, float b);\n"
    "[[visible]] float bytes_extra(B8 b, float x, float y);\n"
    "[[visible]] float uint_array_for_float_array(U6 v);\n"
    "#define K(name, expr) kernel void k_##name(device uint *out [[buffer(0)]],"
    " device const float *in [[buffer(1)]]) { out[0] = as_type<uint>(expr); out[1] = 0x5e47; }\n"
    "#define KV(name, stmt) kernel void k_##name(device uint *out [[buffer(0)]],"
    " device const float *in [[buffer(1)]]) { stmt; out[0] = 7; out[1] = 0x5e47; }\n"
    "K(stitched_extra, stitched_pair(in[0], in[1], in[2]))\n"
    "K(stitched_extra_private, stitched_pair(in[0], in[1], in[2]))\n"
    "K(extra, extra(in[0], in[1], in[2]))\n"
    "K(extra_private, extra(in[0], in[1], in[2]))\n"
    "K(extra_two, extra_two(in[0], in[1], in[2], in[3]))\n"
    "K(extra_pointer, extra_pointer(in[0], in[1], in))\n"
    "K(missing, missing(in[0], in[1]))\n"
    "K(missing_unused, missing_unused(in[0], in[1]))\n"
    "K(missing_pointer_unused, missing_pointer_unused(in[0], in[1]))\n"
    "K(int_param, int_param(in[0], in[1], in[2]))\n"
    "K(half2_param, half2_param(in[0], in[1]))\n"
    "K(uint2_param, uint2_param(float2(in[0], in[1])))\n"
    "K(half_param, half_param(in[0], in[1]))\n"
    "K(uint_for_half, uint_for_half(in[0], half(in[1])))\n"
    "K(long_param, long_param(in[0], in[1]))\n"
    "K(float2_for_two, float2_for_two(float2(in[0], in[1]), in[2]))\n"
    "K(bool_param, bool_param(in[0], in[1]))\n"
    "K(int_return, int_return(in[0], in[1]))\n"
    "K(float_for_half_return, float_for_half_return(in[0], in[1]))\n"
    "K(half_for_float_return, float(half_for_float_return(in[0], in[1])))\n"
    "K(value_for_void, value_for_void(in[0], in[1]))\n"
    "KV(void_for_value, void_for_value(in[0], in[1]))\n"
    "K(value_for_pointer, value_for_pointer(in))\n"
    "K(pointer_for_value, pointer_for_value(0))\n"
    "K(constant_for_device, constant_for_device(in))\n"
    "K(pair_for_floats, pair_for_floats(in[0], in[1]))\n"
    "K(uints_for_floats, uints_for_floats(in[0], in[1]))\n"
    "K(bytes_extra, bytes_extra(B8{{1, 0, 0, 0, 0, 0, 0, 0}}, in[0], in[1]))\n"
    "K(uint_array_for_float_array, uint_array_for_float_array(U6{{0x40000000u, 1, 2, 3, 4,"
    " 0x40400000u}}))\n"
    "kernel void k_device_for_constant(device uint *out [[buffer(0)]],"
    " constant float *in [[buffer(1)]]) {"
    " out[0] = as_type<uint>(device_for_constant(in)); out[1] = 0x5e47; }\n";

static const char *CALLEE =
    "#include <metal_stdlib>\nusing namespace metal;\n"
    "struct Pair { float a; float b; };\n"
    "struct UPair { uint a; uint b; };\n"
    "struct B8 { uchar c[8]; };\n"
    "struct F6 { float a[6]; };\n"
    "[[visible]] float extra(float a, float b) { return a + b; }\n"
    "[[visible]] float extra_two(float a) { return a; }\n"
    "[[visible]] float extra_pointer(float a, float b) { return a + b; }\n"
    "[[visible]] float missing(float a, float b, float c) { return a + b + c; }\n"
    "[[visible]] float missing_unused(float a, float b, float c) { return a + b; }\n"
    "[[visible]] float missing_pointer_unused(float a, float b, device const float *p)"
    " { return a + b; }\n"
    "[[visible]] float int_param(float a, int b, float c) { return a + float(b); }\n"
    "[[visible]] float half2_param(float a, half2 b) { return a + float(b.x) + float(b.y); }\n"
    "[[visible]] float uint2_param(uint2 v) { return float(v.x) + float(v.y); }\n"
    "[[visible]] float half_param(float a, half b) { return a + float(b); }\n"
    "[[visible]] uint uint_for_half(float a, uint b) { return b; }\n"
    "[[visible]] float long_param(float a, long b) { return a + float(b); }\n"
    "[[visible]] float float2_for_two(float a, float b) { return a + 10.0f * b; }\n"
    "[[visible]] float bool_param(bool a, float b) { return a ? b : -b; }\n"
    "[[visible]] float int_return(float a, float b) { return a + b; }\n"
    "[[visible]] half float_for_half_return(float a, float b) { return half(a + b); }\n"
    "[[visible]] float half_for_float_return(float a, float b) { return a + b; }\n"
    "[[visible]] void value_for_void(float a, float b) { }\n"
    "[[visible]] float void_for_value(float a, float b) { return a + b; }\n"
    "[[visible]] float value_for_pointer(float x) { return x; }\n"
    "[[visible]] float pointer_for_value(device const float *p)"
    " { return p == nullptr ? 1.0f : 2.0f; }\n"
    "[[visible]] float constant_for_device(constant float *p) { return p[0] + p[1]; }\n"
    "[[visible]] float device_for_constant(device const float *p) { return p[0] + p[1]; }\n"
    "[[visible]] float pair_for_floats(Pair p) { return p.a + p.b; }\n"
    "[[visible]] float uints_for_floats(UPair p) { return float(p.a) + float(p.b); }\n"
    "[[visible]] float bytes_extra(B8 b, float x) { return x + float(b.c[0]); }\n"
    "[[visible]] float uint_array_for_float_array(F6 v) { return v.a[0] + v.a[5]; }\n";

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

int main(int argc, char **argv) {
  if (argc < 2) return 2;
  const char *c = argv[1];
  if (strcmp(c, "list") == 0) {
    for (size_t i = 0; i < sizeof CASES / sizeof *CASES; i++) printf("%s\n", CASES[i]);
    return 0;
  }
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
    id<MTLLibrary> lib = [D newLibraryWithSource:str(CALLER) options:nil error:&e];
    if (!lib) { printf("RESULT caller error %s\n", e.localizedDescription.UTF8String); return 1; }
    id<MTLLibrary> callee = [D newLibraryWithSource:str(CALLEE) options:nil error:&e];
    if (!callee) { printf("RESULT callee error %s\n", e.localizedDescription.UTF8String); return 1; }
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
    pd.computeFunction = [lib newFunctionWithName:[@"k_" stringByAppendingString:str(c)]];
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
