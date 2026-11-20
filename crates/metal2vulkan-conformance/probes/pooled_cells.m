// What Metal answers for thread pointers held in a thread's array of structs at indices only the
// run knows, among the structs' data: the owned fixture `engine/pooled_cells.metal`'s source (its
// header gives each word). Runs `pooled_closures`, `pooled_bytes` (buffer 2: `len[0]` = 8),
// `pooled_spill` (buffer 2: eight 8-byte words) and `pooled_context` (buffer 2: the address of a
// buffer holding the resource IDs of two 4 by 1 `R32Float` textures, 1 2 3 4 and 10 20 30 40) over
// 8 threads in one threadgroup with `in[i] = i + 0.5` (16 words), and prints each thread's 8 words.
//
// Build (no cargo target needed):
//   clang -fobjc-arc -O1 -fmodules -o pooled_cells pooled_cells.m \
//     -framework Foundation -framework Metal
// Run: ./pooled_cells, MTL_DEBUG_LAYER unset.
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
    "    thread float *extra;\n"
    "    float data[2];\n"
    "};\n"
    "\n"
    "struct Shader {\n"
    "    Closure closure[6];\n"
    "    int num;\n"
    "    int left;\n"
    "};\n"
    "\n"
    "static thread Closure *closure_alloc(thread Shader &sd, int type, float w) {\n"
    "    if (sd.left == 0)\n"
    "        return nullptr;\n"
    "    thread Closure *sc = &sd.closure[sd.num];\n"
    "    sc->type = type;\n"
    "    sc->w = w;\n"
    "    sc->extra = nullptr;\n"
    "    sd.num++;\n"
    "    sd.left--;\n"
    "    return sc;\n"
    "}\n"
    "\n"
    "static thread float *closure_alloc_extra(thread Shader &sd) {\n"
    "    if (sd.left == 0)\n"
    "        return nullptr;\n"
    "    sd.left--;\n"
    "    return &sd.closure[sd.num + sd.left].data[0];\n"
    "}\n"
    "\n"
    "// Three elements, each with extra room where some is left.\n"
    "static void fill(thread Shader &sd, float x) {\n"
    "    for (int j = 0; j < 3; j++) {\n"
    "        thread Closure *c = closure_alloc(sd, j + 1, x + float(j));\n"
    "        if (c == nullptr)\n"
    "            continue;\n"
    "        thread float *e = closure_alloc_extra(sd);\n"
    "        if (e != nullptr) {\n"
    "            e[0] = float(10 * (j + 1));\n"
    "            e[1] = x;\n"
    "            c->extra = e;\n"
    "        }\n"
    "    }\n"
    "}\n"
    "\n"
    "// Words [first..8) of `pooled_closures`.\n"
    "static void summarize(thread Shader &sd, device float *o, int first) {\n"
    "    int types = 0, extras = 0;\n"
    "    float seconds = 0.0f;\n"
    "    for (int i = 0; i < 3; i++) {\n"
    "        float v = -1.0f;\n"
    "        if (i < sd.num) {\n"
    "            thread float *e = sd.closure[i].extra;\n"
    "            v = sd.closure[i].w + (e != nullptr ? e[0] : 0.0f);\n"
    "            types += sd.closure[i].type;\n"
    "            if (e != nullptr) {\n"
    "                seconds += e[1];\n"
    "                extras++;\n"
    "            }\n"
    "        }\n"
    "        if (i >= first)\n"
    "            o[i] = v;\n"
    "    }\n"
    "    o[3] = float(sd.num);\n"
    "    o[4] = float(sd.left);\n"
    "    o[5] = float(types);\n"
    "    o[6] = seconds;\n"
    "    o[7] = float(extras);\n"
    "}\n"
    "\n"
    "kernel void pooled_closures(device float *out [[buffer(0)]],\n"
    "                            device const float *in [[buffer(1)]],\n"
    "                            uint t [[thread_position_in_grid]]) {\n"
    "    switch (t >> 31) {\n"
    "        Shader sd;\n"
    "    default:\n"
    "        sd.num = 0;\n"
    "        sd.left = 2 + int(t & 3);\n"
    "        fill(sd, in[t]);\n"
    "        summarize(sd, out + 8 * t, 0);\n"
    "    }\n"
    "}\n"
    "\n"
    "struct Holder {\n"
    "    thread float *p;\n"
    "    int n;\n"
    "};\n"
    "\n"
    "kernel void pooled_bytes(device float *out [[buffer(0)]],\n"
    "                         device const float *in [[buffer(1)]],\n"
    "                         device const int *len [[buffer(2)]],\n"
    "                         uint t [[thread_position_in_grid]]) {\n"
    "    switch (t >> 31) {\n"
    "        Shader sd;\n"
    "        float vals[4];\n"
    "        Holder h;\n"
    "    default:\n"
    "        for (int k = 0; k < 4; k++)\n"
    "            vals[k] = in[t] + float(k);\n"
    "        h.p = &vals[t & 3];\n"
    "        h.n = len[0];\n"
    "        sd.num = int(t & 1);\n"
    "        sd.left = 2;\n"
    "        thread Closure *c = closure_alloc(sd, 1, 1.0f);\n"
    "        thread uchar *d = (thread uchar *)&c->extra;\n"
    "        thread const uchar *s = (thread const uchar *)&h.p;\n"
    "        for (int k = 0; k < h.n; k++)\n"
    "            d[k] = s[k];\n"
    "        device float *o = out + 8 * t;\n"
    "        thread float *e = sd.closure[sd.num - 1].extra;\n"
    "        o[0] = *e;\n"
    "        *e += 100.0f;\n"
    "        o[1] = vals[t & 3];\n"
    "        o[2] = e == &vals[t & 3] ? 1.0f : 0.0f;\n"
    "        o[3] = float(sd.num);\n"
    "    }\n"
    "}\n"
    "\n"
    "kernel void pooled_spill(device float *out [[buffer(0)]],\n"
    "                         device const float *in [[buffer(1)]],\n"
    "                         volatile device ulong *spill [[buffer(2)]],\n"
    "                         uint t [[thread_position_in_grid]]) {\n"
    "    switch (t >> 31) {\n"
    "        Shader sd;\n"
    "        float vals[4];\n"
    "    default:\n"
    "        for (int k = 0; k < 4; k++)\n"
    "            vals[k] = in[t] + float(k);\n"
    "        thread float *p = &vals[t & 3];\n"
    "        ulong bits;\n"
    "        __builtin_memcpy(&bits, &p, sizeof(p));\n"
    "        spill[t] = bits;\n"
    "        sd.num = int(t & 1);\n"
    "        sd.left = 2;\n"
    "        thread Closure *c = closure_alloc(sd, 1, 1.0f);\n"
    "        ulong back = spill[t];\n"
    "        __builtin_memcpy(&c->extra, &back, sizeof(back));\n"
    "        device float *o = out + 8 * t;\n"
    "        thread float *e = sd.closure[sd.num - 1].extra;\n"
    "        o[0] = *e;\n"
    "        *e += 100.0f;\n"
    "        o[1] = vals[t & 3];\n"
    "        o[2] = e == &vals[t & 3] ? 1.0f : 0.0f;\n"
    "        o[3] = float(sd.num);\n"
    "    }\n"
    "}\n"
    "\n"
    "struct Held {\n"
    "    texture2d<float> t;\n"
    "};\n"
    "\n"
    "struct Anc {\n"
    "    device const Held *held;\n"
    "};\n"
    "\n"
    "struct Context {\n"
    "    constant Anc *anc;\n"
    "    int k;\n"
    "};\n"
    "\n"
    "__attribute__((noinline)) float shade(thread Context &ctx, thread Shader &sd, float x, uint t) {\n"
    "    fill(sd, x);\n"
    "    return ctx.anc->held[ctx.k].t.read(uint2(t & 3, 0)).x;\n"
    "}\n"
    "\n"
    "kernel void pooled_context(device float *out [[buffer(0)]],\n"
    "                           device const float *in [[buffer(1)]],\n"
    "                           constant Anc *anc [[buffer(2)]],\n"
    "                           uint t [[thread_position_in_grid]]) {\n"
    "    switch (t >> 31) {\n"
    "        Shader sd;\n"
    "        Context ctx;\n"
    "    default:\n"
    "        ctx.anc = anc;\n"
    "        ctx.k = int(t & 1);\n"
    "        sd.num = 0;\n"
    "        sd.left = 2 + int(t & 3);\n"
    "        device float *o = out + 8 * t;\n"
    "        o[0] = shade(ctx, sd, in[t], t);\n"
    "        summarize(sd, o, 1);\n"
    "    }\n"
    "}\n";

enum { THREADS = 8, WORDS = 8 };

static id<MTLBuffer> buffer(id<MTLDevice> D, const void *bytes, size_t n) {
  id<MTLBuffer> b = [D newBufferWithLength:n options:MTLResourceStorageModeShared];
  memcpy(b.contents, bytes, n);
  return b;
}

// One run of `name` over THREADS threads in one threadgroup: buffer 0 the words, buffer 1
// `in[i] = i + 0.5`, buffer 2 `extra` when given; `used` made resident.
static void run(id<MTLDevice> D, id<MTLCommandQueue> Q, id<MTLLibrary> lib, NSString *name,
                id<MTLBuffer> extra, NSArray<id<MTLResource>> *used) {
  NSError *e = nil;
  id<MTLFunction> f = [lib newFunctionWithName:name];
  id<MTLComputePipelineState> ps = [D newComputePipelineStateWithFunction:f error:&e];
  if (!ps) {
    printf("RESULT %s pipeline error %s\n", name.UTF8String, e.localizedDescription.UTF8String);
    return;
  }
  float ins[16];
  for (int i = 0; i < 16; i++) ins[i] = (float)i + 0.5f;
  id<MTLBuffer> in = buffer(D, ins, sizeof ins);
  id<MTLBuffer> out = [D newBufferWithLength:THREADS * WORDS * 4 options:MTLResourceStorageModeShared];
  memset(out.contents, 0, THREADS * WORDS * 4);
  id<MTLCommandBuffer> cb = [Q commandBuffer];
  id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
  [ce setComputePipelineState:ps];
  [ce setBuffer:out offset:0 atIndex:0];
  [ce setBuffer:in offset:0 atIndex:1];
  if (extra) [ce setBuffer:extra offset:0 atIndex:2];
  for (id<MTLResource> r in used) [ce useResource:r usage:MTLResourceUsageRead];
  [ce dispatchThreads:MTLSizeMake(THREADS, 1, 1) threadsPerThreadgroup:MTLSizeMake(THREADS, 1, 1)];
  [ce endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  printf("RESULT %s status=%ld\n", name.UTF8String, (long)cb.status);
  const float *w = (const float *)out.contents;
  for (int t = 0; t < THREADS; t++) {
    printf("RESULT %s thread %d:", name.UTF8String, t);
    for (int i = 0; i < WORDS; i++) printf(" %g", w[t * WORDS + i]);
    printf("\n");
  }
}

// A 4 by 1 `R32Float` texture holding `texels`.
static id<MTLTexture> texture(id<MTLDevice> D, const float *texels) {
  MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                                                                               width:4
                                                                              height:1
                                                                           mipmapped:NO];
  d.usage = MTLTextureUsageShaderRead;
  d.storageMode = MTLStorageModeShared;
  id<MTLTexture> t = [D newTextureWithDescriptor:d];
  [t replaceRegion:MTLRegionMake2D(0, 0, 4, 1) mipmapLevel:0 withBytes:texels bytesPerRow:16];
  return t;
}

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
    if (!lib) {
      printf("RESULT library error %s\n", e.localizedDescription.UTF8String);
      return 1;
    }
    run(D, Q, lib, @"pooled_closures", nil, @[]);
    int len = 8;
    run(D, Q, lib, @"pooled_bytes", buffer(D, &len, sizeof len), @[]);
    id<MTLBuffer> spill = [D newBufferWithLength:THREADS * 8 options:MTLResourceStorageModeShared];
    run(D, Q, lib, @"pooled_spill", spill, @[]);
    // held[0] and held[1]: two textures; anc: the address of held.
    const float a[4] = {1, 2, 3, 4}, b[4] = {10, 20, 30, 40};
    id<MTLTexture> t0 = texture(D, a), t1 = texture(D, b);
    MTLResourceID ids[2] = {t0.gpuResourceID, t1.gpuResourceID};
    id<MTLBuffer> held = buffer(D, ids, sizeof ids);
    uint64_t address = held.gpuAddress;
    id<MTLBuffer> anc = buffer(D, &address, sizeof address);
    run(D, Q, lib, @"pooled_context", anc, @[ t0, t1, held ]);
  }
  return 0;
}
