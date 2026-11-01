// What a compute pipeline computes when its kernel calls a function Metal's stitcher made
// (`newLibraryWithStitchedDescriptor:` over a `MTLFunctionStitchingGraph`), linked as a private
// linked function or a plain one, and the bytes of the stitched library.
//
// Measured on an Apple M3 Max under macOS 27.0.1 (26A434), `MTL_DEBUG_LAYER` out of the
// environment, one case a process; the compile layer's `tests::engine_stitched` and the runtime
// lane's `runtime_lane::stitched_functions` hold the answers.
//
// Build (no cargo target needed):
//   clang -fobjc-arc -O1 -fmodules -o stitched_function_linked stitched_function_linked.m \
//     -framework Foundation -framework Metal
//
// The source library defines the stitchable functions `sum2(a, b) = a + b` and
// `product2(a, b) = a * b`, and the kernels `stitch_caller` (stores `fused(in[0], in[1], in[2])`)
// and `stitch_caller4` (stores `fused4(in[0], in[1], in[2], in[3])`), which call the visible
// functions `fused` and `fused4` it declares and does not define. The stitched library holds
// `fused(a, b, c) = product2(sum2(a, b), c)` (two function nodes) and
// `fused4(a, b, c, d) = sum2(product2(sum2(a, b), c), d)` (three). Each kernel runs one thread on
// the inputs 2, 3, 4 and 5.
//
//   private    `fused` in `privateFunctions`, kernel `stitch_caller`
//   functions  `fused` in `functions`, kernel `stitch_caller`
//   deep       `fused4` in `privateFunctions`, kernel `stitch_caller4`
//   dump PATH  writes the stitched library's bytes (its `libraryDataContents`, read by key) to
//              PATH
//
// WHAT IT MEASURED. Every case made the stitched library: an `_MTLLibrary` of type 0
// (`MTLLibraryTypeExecutable`) whose functions `fused4` and `fused` are of `functionType` 5
// (`MTLFunctionTypeVisible`).
//
//   private    created; status 4 (completed), stored 20
//   functions  created; status 4, stored 20
//   deep       created; status 4, stored 25
//   dump       11484 bytes: a metallib of file type 0 holding `fused` and `fused4`, each a
//              visible function (`TYPE` 4) with a module of its own. Each module defines its
//              function with the graph's nodes inlined (`fused` is `fadd` of inputs 1 and 0, then
//              `fmul` by input 2), lists it in `air.visible` with an `air.stitching_info` whose
//              stitching types name internal helper functions that `llvm.compiler.used` keeps,
//              and has an `!air.signature` of kind `air.stitching_modules_signature`: the graph
//              (`air.stitching_input_node`s and `air.stitching_function_node`s `node_0`, ...,
//              `node_output`, the stitchable function each calls) and one
//              `air.module_signature` digest per stitchable function.
//
// So a stitched function is a closed module's visible function, linked as any visible function
// is, privately or not; its signature describes how it was made. `metal2vulkan_devtools::stitched`
// writes modules of that structure for the tests.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>

static const char *SOURCE =
    "#include <metal_stdlib>\nusing namespace metal;\n"
    "[[stitchable]] float sum2(float a, float b) { return a + b; }\n"
    "[[stitchable]] float product2(float a, float b) { return a * b; }\n"
    "[[visible]] float fused(float a, float b, float c);\n"
    "[[visible]] float fused4(float a, float b, float c, float d);\n"
    "kernel void stitch_caller(device float *out [[buffer(0)]],\n"
    "                          constant float *in [[buffer(1)]]) {\n"
    "  out[0] = fused(in[0], in[1], in[2]);\n"
    "}\n"
    "kernel void stitch_caller4(device float *out [[buffer(0)]],\n"
    "                           constant float *in [[buffer(1)]]) {\n"
    "  out[0] = fused4(in[0], in[1], in[2], in[3]);\n"
    "}\n";

static MTLFunctionStitchingInputNode *input(NSUInteger i) {
  return [[MTLFunctionStitchingInputNode alloc] initWithArgumentIndex:i];
}

static MTLFunctionStitchingFunctionNode *node(NSString *name, NSArray *arguments) {
  return [[MTLFunctionStitchingFunctionNode alloc] initWithName:name
                                                      arguments:arguments
                                            controlDependencies:@[]];
}

int main(int argc, char **argv) {
  if (argc < 2) return 2;
  const char *c = argv[1];
  @autoreleasepool {
    printf("RESULT case=%s debug_layer=%s\n", c,
           getenv("MTL_DEBUG_LAYER") ? getenv("MTL_DEBUG_LAYER") : "unset");
    id<MTLDevice> D = MTLCreateSystemDefaultDevice();
    NSError *e = nil;
    id<MTLLibrary> lib = [D newLibraryWithSource:[NSString stringWithUTF8String:SOURCE]
                                         options:nil
                                           error:&e];
    if (!lib) { printf("RESULT source error %s\n", e.localizedDescription.UTF8String); return 1; }
    MTLFunctionStitchingFunctionNode *sum = node(@"sum2", @[ input(0), input(1) ]);
    MTLFunctionStitchingFunctionNode *product = node(@"product2", @[ sum, input(2) ]);
    MTLFunctionStitchingGraph *fused =
        [[MTLFunctionStitchingGraph alloc] initWithFunctionName:@"fused"
                                                          nodes:@[ sum ]
                                                     outputNode:product
                                                     attributes:@[]];
    MTLFunctionStitchingFunctionNode *sum4 = node(@"sum2", @[ input(0), input(1) ]);
    MTLFunctionStitchingFunctionNode *product4 = node(@"product2", @[ sum4, input(2) ]);
    MTLFunctionStitchingFunctionNode *last4 = node(@"sum2", @[ product4, input(3) ]);
    MTLFunctionStitchingGraph *fused4 =
        [[MTLFunctionStitchingGraph alloc] initWithFunctionName:@"fused4"
                                                          nodes:@[ sum4, product4 ]
                                                     outputNode:last4
                                                     attributes:@[]];
    MTLStitchedLibraryDescriptor *sd = [MTLStitchedLibraryDescriptor new];
    sd.functionGraphs = @[ fused, fused4 ];
    sd.functions = @[ [lib newFunctionWithName:@"sum2"], [lib newFunctionWithName:@"product2"] ];
    id<MTLLibrary> stitched = [D newLibraryWithStitchedDescriptor:sd error:&e];
    if (!stitched) {
      printf("RESULT stitched error %s\n", e.localizedDescription.UTF8String);
      return 0;
    }
    printf("RESULT stitched class=%s type=%ld functions=%s\n",
           object_getClassName(stitched), (long)stitched.type,
           [[stitched.functionNames componentsJoinedByString:@","] UTF8String]);
    for (NSString *name in stitched.functionNames) {
      id<MTLFunction> f = [stitched newFunctionWithName:name];
      printf("RESULT function %s type=%ld\n", name.UTF8String, (long)f.functionType);
    }
    if (strcmp(c, "dump") == 0) {
      if (argc < 3) return 2;
      id data = nil;
      @try {
        data = [(NSObject *)stitched valueForKey:@"libraryDataContents"];
      } @catch (NSException *x) {
        printf("RESULT dump exception %s\n", x.reason.UTF8String);
        return 0;
      }
      if (![data isKindOfClass:[NSData class]]) {
        printf("RESULT dump not data: %s\n", data ? object_getClassName(data) : "nil");
        return 0;
      }
      NSData *bytes = data;
      [bytes writeToFile:[NSString stringWithUTF8String:argv[2]] atomically:NO];
      printf("RESULT dump bytes=%lu\n", (unsigned long)bytes.length);
      return 0;
    }
    NSString *callee = @"fused", *kernel = @"stitch_caller";
    BOOL private_ = YES;
    if (strcmp(c, "functions") == 0) private_ = NO;
    else if (strcmp(c, "deep") == 0) { callee = @"fused4"; kernel = @"stitch_caller4"; }
    else if (strcmp(c, "private") != 0) return 2;
    MTLComputePipelineDescriptor *pd = [MTLComputePipelineDescriptor new];
    pd.computeFunction = [lib newFunctionWithName:kernel];
    MTLLinkedFunctions *lf = [MTLLinkedFunctions linkedFunctions];
    id<MTLFunction> f = [stitched newFunctionWithName:callee];
    if (private_) lf.privateFunctions = @[ f ];
    else lf.functions = @[ f ];
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
    id<MTLBuffer> out = [D newBufferWithLength:16 options:MTLResourceStorageModeShared];
    memset(out.contents, 0xAB, 16);
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
    printf("RESULT status=%ld word %g\n", (long)cb.status, ((float *)out.contents)[0]);
  }
  return 0;
}
