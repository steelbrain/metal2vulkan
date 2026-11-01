// What plain Metal reads for a function constant the function's constant values leave undefined:
// read directly, of each scalar and vector type, and through program-scope constants derived from
// it (which Metal's compiler computes in the module's static initializer), for a constant Metal
// reports required and for one it reports optional (read guarded somewhere as well); and which
// stage functions made without constant values Metal makes a pipeline state of.
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
//
// Cases `<how>:<shape>` make a pipeline state of a stage function made `how`: `by-name`
// (`newFunctionWithName:`), `empty` (`newFunctionWithName:constantValues:` with no value) or
// `descriptor` (`newFunctionWithDescriptor:`, `constantValues` nil). Each prints every function's
// `functionConstantsDictionary` count, then the creation. Shapes: kernels `ko` (reads the optional
// `o_opt` only where `is_function_constant_defined`), `kn` (reads no constant), `kg` (the gate
// case's kernel: `g_flag` gates its arguments) and `k`, each a compute state, dispatched once
// where made (word 0 printed); `vertex` (`vc` reads `v_scale`, made `how`; `fp` by name),
// `fragment` (`vp` by name; `fc` reads `f_tint`, made `how`) and `plain` (`vp` and `fp`, which read
// none, made `how`), each a render state; `tile` (`tc` reads `t_tint`) and `tile-plain` (`tp` reads
// none), a tile state's function; `mesh` (`mc` reads `m_scale`, made `how`; `fp` by name),
// `object` (`oc` reads `o_count`, made `how`; `mp` and `fp` by name) and `mesh-plain` (`mp` made
// `how`, `fp` by name), each a mesh render state.
//
// WHAT IT MEASURED (an Apple M3 Max, macOS 27.0.1):
// - `by-name` of a function that declares a constant (its dictionary's count 1, `k`'s 8): the
//   creation ends the process (exit 134, `MTL_DEBUG_LAYER` unset) at Metal's descriptor
//   validation, for every stage, an optional guarded constant (`ko`) and a constant gating only
//   arguments (`kg`) alike:
//     `ko`, `kg`, `k`: "validateWithDevice:1623: failed assertion `Compute Pipeline Descriptor
//       Validation / function ko cannot be used to build a pipeline state. Use
//       newFunctionWithName:constantValues:... to get the specialized function";
//     `vertex`, `fragment`: "validateWithDevice:5101: failed assertion `Render Pipeline
//       Descriptor Validation / vertexFunction vc cannot be used ..." ("fragmentFunction fc");
//     `tile`: "validateWithDevice:1177: failed assertion `Tile Render Pipeline Descriptor
//       Validation / function tc cannot be used to build a pipeline state without setting constant
//       values first.";
//     `mesh`, `object`: "validateWithDevice:1956: failed assertion `Mesh Render Pipeline
//       Descriptor Validation / meshFunction mc cannot be used ..." ("objectFunction oc").
//   The `none` case is `by-name:k`'s and ends the same way.
// - `by-name` of a function that declares none (`kn`, `plain`, `tile-plain`, `mesh-plain`): made;
//   `kn` stores 9.
// - `empty` and `descriptor`, every shape: every function answers count 0, and the state is made;
//   `ko` stores 7 (`o_opt` undefined), `kn` 9, `k` 0, and `kg` leaves word 0 as filled
//   (0xabababab).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
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
"}\n"
"constant uchar o_opt [[function_constant(30)]];\n"
"kernel void ko(device uint *out [[buffer(0)]], uint t [[thread_position_in_grid]]) {\n"
"  if (t != 0) return;\n"
"  out[0] = is_function_constant_defined(o_opt) ? o_opt : 7;\n"
"}\n"
"kernel void kn(device uint *out [[buffer(0)]], uint t [[thread_position_in_grid]]) {\n"
"  if (t != 0) return;\n"
"  out[0] = 9;\n"
"}\n"
"constant float v_scale [[function_constant(31)]];\n"
"vertex float4 vc(uint v [[vertex_id]]) { return float4(float(v & 1) * v_scale, float(v >> 1), 0, 1); }\n"
"vertex float4 vp(uint v [[vertex_id]]) { return float4(float(v & 1), float(v >> 1), 0, 1); }\n"
"constant float f_tint [[function_constant(32)]];\n"
"fragment float4 fc() { return float4(f_tint, 0, 0, 1); }\n"
"fragment float4 fp() { return float4(1, 0, 0, 1); }\n"
"struct TB { half4 c [[color(0)]]; };\n"
"constant half t_tint [[function_constant(33)]];\n"
"kernel void tc(imageblock<TB, imageblock_layout_implicit> b, ushort2 l [[thread_position_in_threadgroup]]) {\n"
"  TB v = b.read(l); v.c *= t_tint; b.write(v, l);\n"
"}\n"
"kernel void tp(imageblock<TB, imageblock_layout_implicit> b, ushort2 l [[thread_position_in_threadgroup]]) {\n"
"  TB v = b.read(l); v.c *= 0.5h; b.write(v, l);\n"
"}\n"
"struct MV { float4 position [[position]]; };\n"
"using M = metal::mesh<MV, void, 3, 1, topology::triangle>;\n"
"constant float m_scale [[function_constant(34)]];\n"
"[[mesh]] void mc(M m, uint t [[thread_index_in_threadgroup]]) {\n"
"  if (t == 0) { m.set_primitive_count(1); m.set_index(0, 0); m.set_index(1, 1); m.set_index(2, 2); }\n"
"  if (t < 3) { MV v; v.position = float4(float(t & 1) * m_scale, float(t >> 1), 0, 1); m.set_vertex(t, v); }\n"
"}\n"
"[[mesh]] void mp(M m, uint t [[thread_index_in_threadgroup]]) {\n"
"  if (t == 0) { m.set_primitive_count(1); m.set_index(0, 0); m.set_index(1, 1); m.set_index(2, 2); }\n"
"  if (t < 3) { MV v; v.position = float4(float(t & 1), float(t >> 1), 0, 1); m.set_vertex(t, v); }\n"
"}\n"
"constant uint o_count [[function_constant(35)]];\n"
"[[object]] void oc(mesh_grid_properties g) { g.set_threadgroups_per_grid(uint3(o_count + 1, 1, 1)); }\n";

// `how` of `name`: `by-name` (`newFunctionWithName:`), `empty`
// (`newFunctionWithName:constantValues:` with no value), `descriptor` (`newFunctionWithDescriptor:`
// with nil `constantValues`). Prints the function's `functionConstantsDictionary` count.
static id<MTLFunction> made_as(id<MTLLibrary> L, const char *how, NSString *name) {
  NSError *e = nil;
  id<MTLFunction> f = nil;
  if (strcmp(how, "by-name") == 0) {
    f = [L newFunctionWithName:name];
  } else if (strcmp(how, "empty") == 0) {
    f = [L newFunctionWithName:name constantValues:[MTLFunctionConstantValues new] error:&e];
  } else {
    MTLFunctionDescriptor *d = [MTLFunctionDescriptor functionDescriptor];
    d.name = name;
    f = [L newFunctionWithDescriptor:d error:&e];
  }
  if (!f) { printf("RESULT function %s error %s\n", name.UTF8String, e.localizedDescription.UTF8String); exit(0); }
  printf("RESULT function %s %s constants=%lu\n", name.UTF8String, how, (unsigned long)f.functionConstantsDictionary.count);
  fflush(stdout);
  return f;
}

// Case `<how>:<shape>`: a kernel shape (`ko`, `kn`, `kg`, `k`) made `how` and its compute state,
// dispatched once where made (word 0 printed), or a render shape: `vertex` (`vc` made `how`, `fp`
// by name), `fragment` (`vp` by name, `fc` made `how`), `plain` (`vp` and `fp` made `how`) and its
// render state.
static int stage_case(id<MTLDevice> D, id<MTLCommandQueue> Q, id<MTLLibrary> L, const char *how, const char *shape) {
  NSError *e = nil;
  if (shape[0] == 'k') {
    id<MTLFunction> f = made_as(L, how, [NSString stringWithUTF8String:shape]);
    id<MTLComputePipelineState> ps = [D newComputePipelineStateWithFunction:f error:&e];
    if (!ps) { printf("RESULT pipeline error %s\n", e.localizedDescription.UTF8String); return 0; }
    id<MTLBuffer> out = [D newBufferWithLength:64 options:MTLResourceStorageModeShared];
    id<MTLBuffer> one = [D newBufferWithLength:64 options:MTLResourceStorageModeShared];
    memset(out.contents, 0xAB, 64);
    id<MTLCommandBuffer> cb = [Q commandBuffer];
    id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
    [ce setComputePipelineState:ps];
    [ce setBuffer:out offset:0 atIndex:0]; [ce setBuffer:one offset:0 atIndex:1]; [ce setBuffer:one offset:0 atIndex:2];
    [ce dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [ce endEncoding]; [cb commit]; [cb waitUntilCompleted];
    printf("RESULT pipeline made status=%ld word0 %08x\n", (long)cb.status, ((const uint32_t *)out.contents)[0]);
    return 0;
  }
  if (strcmp(shape, "tile") == 0 || strcmp(shape, "tile-plain") == 0) {
    MTLTileRenderPipelineDescriptor *t = [MTLTileRenderPipelineDescriptor new];
    t.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    t.threadgroupSizeMatchesTileSize = YES;
    t.tileFunction = made_as(L, how, strcmp(shape, "tile") == 0 ? @"tc" : @"tp");
    id<MTLRenderPipelineState> ps = [D newRenderPipelineStateWithTileDescriptor:t options:MTLPipelineOptionNone reflection:nil error:&e];
    if (!ps) { printf("RESULT pipeline error %s\n", e.localizedDescription.UTF8String); return 0; }
    printf("RESULT pipeline made\n");
    return 0;
  }
  if (strncmp(shape, "mesh", 4) == 0 || strcmp(shape, "object") == 0) {
    MTLMeshRenderPipelineDescriptor *m = [MTLMeshRenderPipelineDescriptor new];
    m.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    if (strcmp(shape, "object") == 0) m.objectFunction = made_as(L, how, @"oc");
    m.meshFunction = made_as(L, strcmp(shape, "object") == 0 ? "by-name" : how, strcmp(shape, "mesh") == 0 ? @"mc" : @"mp");
    m.fragmentFunction = made_as(L, "by-name", @"fp");
    id<MTLRenderPipelineState> ps = [D newRenderPipelineStateWithMeshDescriptor:m options:MTLPipelineOptionNone reflection:nil error:&e];
    if (!ps) { printf("RESULT pipeline error %s\n", e.localizedDescription.UTF8String); return 0; }
    printf("RESULT pipeline made\n");
    return 0;
  }
  MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
  d.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
  const char *v = strcmp(shape, "fragment") == 0 ? "by-name" : how;
  const char *fr = strcmp(shape, "vertex") == 0 ? "by-name" : how;
  d.vertexFunction = made_as(L, v, strcmp(shape, "vertex") == 0 ? @"vc" : @"vp");
  d.fragmentFunction = made_as(L, fr, strcmp(shape, "fragment") == 0 ? @"fc" : @"fp");
  id<MTLRenderPipelineState> ps = [D newRenderPipelineStateWithDescriptor:d error:&e];
  if (!ps) { printf("RESULT pipeline error %s\n", e.localizedDescription.UTF8String); return 0; }
  printf("RESULT pipeline made\n");
  return 0;
}

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
    const char *colon = strchr(c, ':');
    if (colon) {
      char how[32] = {0};
      strncpy(how, c, (size_t)(colon - c) < sizeof how - 1 ? (size_t)(colon - c) : sizeof how - 1);
      return stage_case(D, Q, L, how, colon + 1);
    }
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
