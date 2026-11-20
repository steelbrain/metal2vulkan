// What Metal answers for MSL 4.1's generic address space: pointers declared with no address
// space, which a device, thread or threadgroup pointer converts to, read and written through
// functions that take them, and asked which space they point into (`is_device_addrspace`,
// `is_thread_addrspace`, `is_threadgroup_addrspace`). The owned fixture
// `engine/generic_pointers.metal` holds the same source for `generic_spaces`, `generic_mixed`,
// `generic_null`, `generic_slots`, `generic_fields`, `generic_array`, `generic_captured` and
// `generic_functor` (its header gives each word); `generic_null` asks which space a null generic
// pointer answers, where the run decides whether a pointer is null, and the last five read
// generic pointers a struct holds, by reference, by value, in an array member passed as one
// aggregate, through a lambda that captures the struct by reference, and through a functor that
// holds its address and whose call operator takes the functor as a generic pointer.
//
// Build (no cargo target needed):
//   clang -fobjc-arc -O1 -fmodules -o generic_pointers generic_pointers.m \
//     -framework Foundation -framework Metal
// Run: ./generic_pointers spaces   (or mixed, null, slots, fields, array, captured, functor), one
// case a process,
// MTL_DEBUG_LAYER unset.
//
// Each case compiles the source as MSL 4.1, runs its kernel over 8 threads in one threadgroup
// with `in[i] = i` (16 words), and prints the words.
//
// WHAT IT MEASURED (Apple M3 Max, macOS 27.0.1): the module docs of
// `runtime_lane::generic_pointers` in metal2vulkan-validation before 8871d40bdf.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>

static const char *SOURCE =
    "#include <metal_stdlib>\nusing namespace metal;\n"
    "__attribute__((noinline)) float space(const float *p) {\n"
    "    return is_device_addrspace(p)        ? 1.0f\n"
    "           : is_thread_addrspace(p)      ? 2.0f\n"
    "           : is_threadgroup_addrspace(p) ? 3.0f\n"
    "                                         : 4.0f;\n"
    "}\n"
    "__attribute__((noinline)) float weigh(const float *p, uint k) {\n"
    "    return p[k] + 100.0f * space(p);\n"
    "}\n"
    "__attribute__((noinline)) void bump(float *p, uint k, float x) {\n"
    "    p[k] += x;\n"
    "}\n"
    "kernel void generic_spaces(device float *out [[buffer(0)]],\n"
    "                           device const float *in [[buffer(1)]],\n"
    "                           uint t [[thread_position_in_grid]],\n"
    "                           uint l [[thread_position_in_threadgroup]]) {\n"
    "    threadgroup float shared[8];\n"
    "    switch (t >> 31) {\n"
    "        float local[4];\n"
    "    default:\n"
    "        shared[l] = 2.0f * in[l];\n"
    "        local[0] = float(t);\n"
    "        local[1] = float(t) + 1.0f;\n"
    "        local[2] = float(t) + 2.0f;\n"
    "        local[3] = float(t) + 3.0f;\n"
    "        bump(local, t & 3, 0.5f);\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "        device const float *p = (t & 1) ? in + 8 : in;\n"
    "        device float *o = out + 8 * t;\n"
    "        o[0] = weigh(in, t);\n"
    "        o[1] = weigh(local, t & 3);\n"
    "        o[2] = weigh(shared, (t + 1) & 7);\n"
    "        o[3] = weigh(p, t);\n"
    "        o[4] = 2.0f;\n"
    "        bump(o, 4, 1.5f);\n"
    "        o[5] = space(shared);\n"
    "        o[6] = space(local);\n"
    "        o[7] = space(in);\n"
    "    }\n"
    "}\n"
    "kernel void generic_mixed(device float *out [[buffer(0)]],\n"
    "                          device const float *in [[buffer(1)]],\n"
    "                          uint t [[thread_position_in_grid]]) {\n"
    "    switch (t >> 31) {\n"
    "        float local[2];\n"
    "    default:\n"
    "        local[0] = 10.0f;\n"
    "        local[1] = 11.0f;\n"
    "        const float *p = in[t] < 4.0f ? (const float *)local : (const float *)in;\n"
    "        out[t] = weigh(p, t & 1);\n"
    "    }\n"
    "}\n"
    "kernel void generic_null(device float *out [[buffer(0)]],\n"
    "                         device const float *in [[buffer(1)]],\n"
    "                         uint t [[thread_position_in_grid]],\n"
    "                         uint l [[thread_position_in_threadgroup]]) {\n"
    "    threadgroup float shared[8];\n"
    "    switch (t >> 31) {\n"
    "        float local[2];\n"
    "    default:\n"
    "        local[0] = 1.0f;\n"
    "        shared[l] = 1.0f;\n"
    "        bool keep = in[t] < 4.0f;\n"
    "        out[4 * t + 0] = space(keep ? in : nullptr);\n"
    "        out[4 * t + 1] = space(keep ? (const float *)local : nullptr);\n"
    "        out[4 * t + 2] = space(keep ? (const float *)shared : nullptr);\n"
    "        out[4 * t + 3] = space(nullptr);\n"
    "    }\n"
    "}\n"
    "struct refs {\n"
    "    const float *a;\n"
    "    const float *b;\n"
    "    const float *c;\n"
    "    uint n;\n"
    "};\n"
    "__attribute__((noinline)) void read_refs(thread const refs &r, device float *o, uint t) {\n"
    "    o[0] = weigh(r.a, t);\n"
    "    o[1] = weigh(r.b, 1);\n"
    "    o[2] = space(r.c);\n"
    "    o[3] = float(r.n);\n"
    "}\n"
    "__attribute__((noinline)) void read_fields(refs r, device float *o, uint t) {\n"
    "    o[0] = weigh(r.a, t);\n"
    "    o[1] = weigh(r.b, 1);\n"
    "    o[2] = space(r.c);\n"
    "    o[3] = float(r.n);\n"
    "}\n"
    "kernel void generic_slots(device float *out [[buffer(0)]],\n"
    "                          device const float *in [[buffer(1)]],\n"
    "                          uint t [[thread_position_in_grid]]) {\n"
    "    switch (t >> 31) {\n"
    "        float local[2];\n"
    "        refs r;\n"
    "    default:\n"
    "        local[0] = float(t);\n"
    "        local[1] = float(t) + 1.0f;\n"
    "        r.a = in;\n"
    "        r.b = local;\n"
    "        r.c = in[t] < 4.0f ? in + 8 : nullptr;\n"
    "        r.n = t;\n"
    "        read_refs(r, out + 4 * t, t);\n"
    "    }\n"
    "}\n"
    "kernel void generic_fields(device float *out [[buffer(0)]],\n"
    "                           device const float *in [[buffer(1)]],\n"
    "                           uint t [[thread_position_in_grid]]) {\n"
    "    switch (t >> 31) {\n"
    "        float local[2];\n"
    "        refs r;\n"
    "    default:\n"
    "        local[0] = float(t);\n"
    "        local[1] = float(t) + 1.0f;\n"
    "        r.a = in;\n"
    "        r.b = local;\n"
    "        r.c = in[t] < 4.0f ? in + 8 : nullptr;\n"
    "        r.n = t;\n"
    "        read_fields(r, out + 4 * t, t);\n"
    "    }\n"
    "}\n"
    "struct ptrs {\n"
    "    const float *p[3];\n"
    "    const float *q;\n"
    "    uint n;\n"
    "};\n"
    "__attribute__((noinline)) void read_ptrs(ptrs r, device float *o, uint t) {\n"
    "    o[0] = weigh(r.p[0], t);\n"
    "    o[1] = weigh(r.q, 1);\n"
    "    o[2] = space(r.p[2]);\n"
    "    o[3] = float(r.n);\n"
    "}\n"
    "kernel void generic_array(device float *out [[buffer(0)]],\n"
    "                          device const float *in [[buffer(1)]],\n"
    "                          uint t [[thread_position_in_grid]]) {\n"
    "    switch (t >> 31) {\n"
    "        float local[2];\n"
    "        ptrs r;\n"
    "    default:\n"
    "        local[0] = float(t);\n"
    "        local[1] = float(t) + 1.0f;\n"
    "        r.p[0] = in;\n"
    "        r.p[1] = in + 4;\n"
    "        r.p[2] = in[t] < 4.0f ? in + 8 : nullptr;\n"
    "        r.q = local;\n"
    "        r.n = t;\n"
    "        read_ptrs(r, out + 4 * t, t);\n"
    "    }\n"
    "}\n"
    "kernel void generic_captured(device float *out [[buffer(0)]],\n"
    "                             device const float *in [[buffer(1)]],\n"
    "                             uint t [[thread_position_in_grid]]) {\n"
    "    switch (t >> 31) {\n"
    "        float local[2];\n"
    "        refs r;\n"
    "    default:\n"
    "        local[0] = float(t);\n"
    "        local[1] = float(t) + 1.0f;\n"
    "        r.a = in;\n"
    "        r.b = local;\n"
    "        r.c = in[t] < 4.0f ? in + 8 : nullptr;\n"
    "        r.n = t;\n"
    "        device float *o = out + 4 * t;\n"
    "        auto read = [&]() __attribute__((noinline)) {\n"
    "            o[0] = weigh(r.a, t);\n"
    "            o[1] = weigh(r.b, 1);\n"
    "            o[2] = space(r.c);\n"
    "            o[3] = float(r.n);\n"
    "        };\n"
    "        read();\n"
    "    }\n"
    "}\n"
    "struct reader {\n"
    "    thread const refs *r;\n"
    "    device float *o;\n"
    "    uint t;\n"
    "    __attribute__((noinline)) void operator()() const;\n"
    "};\n"
    "void reader::operator()() const {\n"
    "    o[0] = weigh(r->a, t);\n"
    "    o[1] = weigh(r->b, 1);\n"
    "    o[2] = space(r->c);\n"
    "    o[3] = float(r->n);\n"
    "}\n"
    "kernel void generic_functor(device float *out [[buffer(0)]],\n"
    "                            device const float *in [[buffer(1)]],\n"
    "                            uint t [[thread_position_in_grid]]) {\n"
    "    switch (t >> 31) {\n"
    "        float local[2];\n"
    "        refs r;\n"
    "        reader read;\n"
    "    default:\n"
    "        local[0] = float(t);\n"
    "        local[1] = float(t) + 1.0f;\n"
    "        r.a = in;\n"
    "        r.b = local;\n"
    "        r.c = in[t] < 4.0f ? in + 8 : nullptr;\n"
    "        r.n = t;\n"
    "        read.r = &r;\n"
    "        read.o = out + 4 * t;\n"
    "        read.t = t;\n"
    "        read();\n"
    "    }\n"
    "}\n";

enum { THREADS = 8, WORDS = 8 };

int main(int argc, char **argv) {
  if (argc < 2) return 2;
  const char *c = argv[1];
  @autoreleasepool {
    printf("RESULT case=%s debug_layer=%s\n", c,
           getenv("MTL_DEBUG_LAYER") ? getenv("MTL_DEBUG_LAYER") : "unset");
    id<MTLDevice> D = MTLCreateSystemDefaultDevice();
    printf("RESULT device=%s\n", D.name.UTF8String);
    id<MTLCommandQueue> Q = [D newCommandQueue];
    NSError *e = nil;
    MTLCompileOptions *o = [MTLCompileOptions new];
    o.languageVersion = MTLLanguageVersion4_1;
    id<MTLLibrary> lib = [D newLibraryWithSource:[NSString stringWithUTF8String:SOURCE]
                                         options:o
                                           error:&e];
    if (!lib) { printf("RESULT library error %s\n", e.localizedDescription.UTF8String); return 1; }
    NSString *name = strcmp(c, "mixed") == 0    ? @"generic_mixed"
                     : strcmp(c, "null") == 0   ? @"generic_null"
                     : strcmp(c, "slots") == 0  ? @"generic_slots"
                     : strcmp(c, "fields") == 0 ? @"generic_fields"
                     : strcmp(c, "array") == 0  ? @"generic_array"
                     : strcmp(c, "captured") == 0 ? @"generic_captured"
                     : strcmp(c, "functor") == 0  ? @"generic_functor"
                                                : @"generic_spaces";
    id<MTLFunction> f = [lib newFunctionWithName:name];
    id<MTLComputePipelineState> ps = [D newComputePipelineStateWithFunction:f error:&e];
    if (!ps) { printf("RESULT pipeline error %s\n", e.localizedDescription.UTF8String); return 0; }
    id<MTLBuffer> in = [D newBufferWithLength:16 * 4 options:MTLResourceStorageModeShared];
    float *ins = (float *)in.contents;
    for (int i = 0; i < 16; i++) ins[i] = (float)i;
    id<MTLBuffer> out = [D newBufferWithLength:THREADS * WORDS * 4
                                       options:MTLResourceStorageModeShared];
    memset(out.contents, 0, THREADS * WORDS * 4);
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
    const float *w = (const float *)out.contents;
    for (int t = 0; t < THREADS; t++) {
      printf("RESULT thread %d:", t);
      for (int i = 0; i < WORDS; i++) printf(" %g", w[t * WORDS + i]);
      printf("\n");
    }
  }
  return 0;
}
