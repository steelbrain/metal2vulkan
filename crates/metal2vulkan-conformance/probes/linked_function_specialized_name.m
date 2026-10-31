// Which name a visible-function reference resolves against when the pipeline links a function made
// by `newFunctionWithDescriptor:` with a `specializedName`: the function's own name, or the
// specialized one.
//
// Measured on an Apple M3 Max under macOS 27.0.1, `MTL_DEBUG_LAYER` out of the environment, one
// case a process.
//
// Build (no cargo target needed):
//   clang -fobjc-arc -O1 -fmodules -o linked_function_specialized_name \
//     linked_function_specialized_name.m -framework Foundation -framework Metal
//
// Kernel `k` calls the visible function `custom_fn`, which its library declares and does not
// define; the second library defines the visible function `base_fn`, which returns its argument
// times the function constant `scale` (index 0) plus 1. Cases: `specialized` (`base_fn` made with
// `specializedName` `custom_fn` and `scale` 3), `plain` (`base_fn` by name), `both` (the
// specialized function and the plain one, linked together), `other` (`specializedName`
// `another_fn`). Each prints the linked function's `name`, the pipeline's creation or its error,
// and the word the kernel stores (the call with 5).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>
static const char *CALLER =
    "#include <metal_stdlib>\nusing namespace metal;\n"
    "[[visible]] float custom_fn(float x);\n"
    "[[visible]] float base_fn(float x);\n"
    "[[visible]] float guarded_fn(float x);\n"
    "[[visible]] float plain_fn(float x);\n"
    "kernel void k(device float *out [[buffer(0)]], uint t [[thread_position_in_grid]]) {\n"
    "  if (t == 0) out[0] = custom_fn(5.0f);\n"
    "}\n"
    "kernel void kb(device float *out [[buffer(0)]], uint t [[thread_position_in_grid]]) {\n"
    "  if (t == 0) out[0] = base_fn(5.0f);\n"
    "}\n"
    "kernel void kg(device float *out [[buffer(0)]], uint t [[thread_position_in_grid]]) {\n"
    "  if (t == 0) out[0] = guarded_fn(5.0f);\n"
    "}\n"
    "kernel void kp(device float *out [[buffer(0)]], uint t [[thread_position_in_grid]]) {\n"
    "  if (t == 0) out[0] = plain_fn(5.0f);\n"
    "}\n";
static const char *CALLEE =
    "#include <metal_stdlib>\nusing namespace metal;\n"
    "constant float scale [[function_constant(0)]];\n"
    "constant float twice = scale * 2.0f;\n"
    "[[visible]] float base_fn(float x) { return x * scale + 1.0f; }\n"
    "[[visible]] float derived_fn(float x) { return x * twice; }\n"
    "[[visible]] float guarded_fn(float x) {\n"
    "  return is_function_constant_defined(scale) ? x * scale : x - 1.0f;\n"
    "}\n"
    "[[visible]] float plain_fn(float x) { return x + 2.0f; }\n";

// How a made function's constant values are given: `scale` 3, values that leave `scale` out, or
// none (nil, as `newFunctionWithName:` makes it).
enum values { SCALE3, EMPTY, NONE };

static id<MTLFunction> made_as(id<MTLLibrary> lib, NSString *name, NSString *specialized,
                               enum values values) {
  MTLFunctionDescriptor *d = [MTLFunctionDescriptor functionDescriptor];
  d.name = name;
  d.specializedName = specialized;
  if (values != NONE) {
    MTLFunctionConstantValues *v = [MTLFunctionConstantValues new];
    float scale = 3.0f;
    if (values == SCALE3) [v setConstantValue:&scale type:MTLDataTypeFloat atIndex:0];
    d.constantValues = v;
  }
  NSError *e = nil;
  id<MTLFunction> f = [lib newFunctionWithDescriptor:d error:&e];
  if (!f) printf("RESULT function error %s\n", e.localizedDescription.UTF8String);
  return f;
}

static id<MTLFunction> made(id<MTLLibrary> lib, NSString *specialized) {
  return made_as(lib, @"base_fn", specialized, SCALE3);
}

// The function `name` made by `newFunctionWithName:` (NONE) or
// `newFunctionWithName:constantValues:error:`.
static id<MTLFunction> by_name(id<MTLLibrary> lib, NSString *name, enum values values) {
  if (values == NONE) return [lib newFunctionWithName:name];
  MTLFunctionConstantValues *v = [MTLFunctionConstantValues new];
  float scale = 3.0f;
  if (values == SCALE3) [v setConstantValue:&scale type:MTLDataTypeFloat atIndex:0];
  NSError *e = nil;
  id<MTLFunction> f = [lib newFunctionWithName:name constantValues:v error:&e];
  if (!f) printf("RESULT function error %s\n", e.localizedDescription.UTF8String);
  return f;
}

int main(int argc, char **argv) {
  if (argc < 2) return 2;
  const char *c = argv[1];
  @autoreleasepool {
    printf("RESULT case=%s debug_layer=%s\n", c, getenv("MTL_DEBUG_LAYER") ? getenv("MTL_DEBUG_LAYER") : "unset");
    id<MTLDevice> D = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> Q = [D newCommandQueue];
    NSError *e = nil;
    id<MTLLibrary> caller = [D newLibraryWithSource:[NSString stringWithUTF8String:CALLER] options:nil error:&e];
    if (!caller) { printf("RESULT caller error %s\n", e.localizedDescription.UTF8String); return 1; }
    id<MTLLibrary> callee = [D newLibraryWithSource:[NSString stringWithUTF8String:CALLEE] options:nil error:&e];
    if (!callee) { printf("RESULT callee error %s\n", e.localizedDescription.UTF8String); return 1; }
    NSMutableArray<id<MTLFunction>> *linked = [NSMutableArray array];
    if (strcmp(c, "specialized") == 0 || strcmp(c, "both") == 0) [linked addObject:made(callee, @"custom_fn")];
    if (strcmp(c, "other") == 0) [linked addObject:made(callee, @"another_fn")];
    if (strcmp(c, "plain") == 0 || strcmp(c, "both") == 0) [linked addObject:made(callee, nil)];
    // The function constant: each function linked as `custom_fn`, its values as the case says.
    static const struct { const char *c; NSString *name; enum values values; } shapes[] = {
        {"unvalued", @"base_fn", NONE},       {"unvalued-empty", @"base_fn", EMPTY},
        {"guarded", @"guarded_fn", NONE},     {"guarded-empty", @"guarded_fn", EMPTY},
        {"guarded-valued", @"guarded_fn", SCALE3}, {"derived", @"derived_fn", SCALE3},
        {"derived-empty", @"derived_fn", EMPTY}, {"constant-free", @"plain_fn", NONE},
    };
    for (size_t i = 0; i < sizeof shapes / sizeof shapes[0]; i++)
      if (strcmp(c, shapes[i].c) == 0) {
        id<MTLFunction> f = made_as(callee, shapes[i].name, @"custom_fn", shapes[i].values);
        if (!f) return 0;
        [linked addObject:f];
      }
    // Linked under their own names, each called by its own kernel: `byname*` by name (with
    // values as the case says), `descriptor*` by `newFunctionWithDescriptor:` with no
    // specialized name.
    NSString *kernel = @"k";
    static const struct { const char *c; NSString *name; NSString *kernel; int descriptor;
                          enum values values; } own[] = {
        {"byname", @"base_fn", @"kb", 0, NONE},
        {"byname-empty", @"base_fn", @"kb", 0, EMPTY},
        {"byname-valued", @"base_fn", @"kb", 0, SCALE3},
        {"descriptor-nil", @"base_fn", @"kb", 1, NONE},
        {"descriptor-empty", @"base_fn", @"kb", 1, EMPTY},
        {"guarded-byname", @"guarded_fn", @"kg", 0, NONE},
        {"constant-free-byname", @"plain_fn", @"kp", 0, NONE},
    };
    for (size_t i = 0; i < sizeof own / sizeof own[0]; i++)
      if (strcmp(c, own[i].c) == 0) {
        id<MTLFunction> f = own[i].descriptor ? made_as(callee, own[i].name, nil, own[i].values)
                                              : by_name(callee, own[i].name, own[i].values);
        if (!f) return 0;
        [linked addObject:f];
        kernel = own[i].kernel;
      }
    for (id<MTLFunction> f in linked) printf("RESULT linked name=%s\n", f.name.UTF8String);
    MTLComputePipelineDescriptor *pd = [MTLComputePipelineDescriptor new];
    pd.computeFunction = [caller newFunctionWithName:kernel];
    MTLLinkedFunctions *lf = [MTLLinkedFunctions linkedFunctions];
    lf.functions = linked;
    pd.linkedFunctions = lf;
    id<MTLComputePipelineState> ps = [D newComputePipelineStateWithDescriptor:pd options:0 reflection:nil error:&e];
    if (!ps) { printf("RESULT pipeline error %s\n", e.localizedDescription.UTF8String); return 0; }
    id<MTLBuffer> out = [D newBufferWithLength:16 options:MTLResourceStorageModeShared];
    memset(out.contents, 0xAB, 16);
    id<MTLCommandBuffer> cb = [Q commandBuffer];
    id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
    [ce setComputePipelineState:ps];
    [ce setBuffer:out offset:0 atIndex:0];
    [ce dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [ce endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    printf("RESULT status=%ld word %g\n", (long)cb.status, ((float *)out.contents)[0]);
  }
  return 0;
}
