// Where a kernel's loads and stores land when a buffer is bound at an offset that is not a
// multiple of its argument type's alignment, with Metal's API validation off (with it on, each
// such bind is reported and the work still runs).
//
//   clang -fobjc-arc -O1 -fmodules -o misaligned_buffer_binds misaligned_buffer_binds.m \
//       -framework Foundation -framework Metal
//   ./misaligned_buffer_binds
//
// For element types of alignment 2, 4, 8 and 16 (ushort, uint, uint2, uint4), each bound at
// offsets 0 (control), 1, 2, 3, 4 and 8 where the offset is not a multiple of the alignment: once
// with the source misaligned (a load) and once with the destination misaligned (a store), from
// random words. Two kernels per type: a thread-indexed copy (dst[t] = src[t], 64 threads) and a
// constant-indexed one (dst[0] = src[3], dst[2] = src[1]). For each case it prints the effective
// base offsets b (0 to 31) at which every copied element is found (a load) or the whole
// destination equals the expected bytes (a store); "none" with the changed byte span where no b
// matches. Then pointers stored in a buffer (as a program writes a table's addresses itself): a
// `device uint *` at a buffer's gpuAddress + 0, 1, 2 and 3, and a `device uint4 *` at + 0, 4 and
// 8, each read (dst[t] = p.a[t]) and written (p.b[t] = src[t]) through.
//
// On an Apple M3 Max under macOS 27.0.1 every case matched one b: the address rounded down to the
// smaller of the element's alignment and 4 (a ushort at +1 at +0 and at +3 at +2; a uint at +1, +2
// or +3 at +0; a uint2 or uint4 at +1, +2 or +3 at +0 and at +4 or +8 exactly there), loads,
// stores and stored pointers alike.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <string.h>

static NSString *const kSource = @
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "#define K(n, T) \\\n"
    "kernel void copy_##n(device const T *src [[buffer(0)]], device T *dst [[buffer(1)]], uint t [[thread_position_in_grid]]) { dst[t] = src[t]; } \\\n"
    "kernel void konst_##n(device const T *src [[buffer(0)]], device T *dst [[buffer(1)]], uint t [[thread_position_in_grid]]) { if (t == 0) { dst[0] = src[3]; dst[2] = src[1]; } }\n"
    "K(u16, ushort)\n"
    "K(u32, uint)\n"
    "K(u64, uint2)\n"
    "K(u128, uint4)\n"
    "#define P(n, T) \\\n"
    "struct ptrs_##n { device T *a; device T *b; }; \\\n"
    "kernel void ptr_##n(device const ptrs_##n &p [[buffer(0)]], device T *dst [[buffer(1)]], device const T *src [[buffer(2)]], uint t [[thread_position_in_grid]]) { dst[t] = p.a[t]; p.b[t] = src[t]; }\n"
    "P(u32, uint)\n"
    "P(u128, uint4)\n";

enum { N = 64, LEN = 8192, MAXB = 32 };

static uint64_t g_state = 0x9E3779B97F4A7C15ull;
static void fill(uint8_t *p, size_t n) {
    for (size_t i = 0; i < n; i++) {
        g_state ^= g_state << 13;
        g_state ^= g_state >> 7;
        g_state ^= g_state << 17;
        p[i] = (uint8_t)(g_state >> 24);
    }
}

typedef struct { unsigned d, s; } Pair;

int main(void) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { printf("no device\n"); return 1; }
        printf("device: %s\n", dev.name.UTF8String);
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:kSource options:nil error:&err];
        if (!lib) { printf("library: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        struct { const char *name; unsigned size; } types[] = {{"u16", 2}, {"u32", 4}, {"u64", 8}, {"u128", 16}};
        unsigned offsets[] = {0, 1, 2, 3, 4, 8};
        int failures = 0;
        for (unsigned ti = 0; ti < 4; ti++) {
            unsigned size = types[ti].size;
            for (int k = 0; k < 2; k++) {
                NSString *fn = [NSString stringWithFormat:@"%s_%s", k ? "konst" : "copy", types[ti].name];
                id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:fn] error:&err];
                if (!pso) { printf("%s: %s\n", fn.UTF8String, err.localizedDescription.UTF8String); return 1; }
                Pair pairs[N];
                unsigned np = 0;
                if (k == 0) { for (unsigned t = 0; t < N; t++) pairs[np++] = (Pair){t, t}; }
                else { pairs[np++] = (Pair){0, 3}; pairs[np++] = (Pair){2, 1}; }
                for (unsigned oi = 0; oi < 6; oi++) {
                    unsigned o = offsets[oi];
                    if (o != 0 && o % size == 0) continue;
                    for (int store = 0; store < 2; store++) {
                        id<MTLBuffer> sb = [dev newBufferWithLength:LEN options:MTLResourceStorageModeShared];
                        id<MTLBuffer> db = [dev newBufferWithLength:LEN options:MTLResourceStorageModeShared];
                        uint8_t s0[LEN], d0[LEN];
                        fill(s0, LEN);
                        fill(d0, LEN);
                        memcpy(sb.contents, s0, LEN);
                        memcpy(db.contents, d0, LEN);
                        id<MTLCommandBuffer> cb = [q commandBuffer];
                        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                        [e setComputePipelineState:pso];
                        [e setBuffer:sb offset:(store ? 0 : o) atIndex:0];
                        [e setBuffer:db offset:(store ? o : 0) atIndex:1];
                        [e dispatchThreads:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(N, 1, 1)];
                        [e endEncoding];
                        [cb commit];
                        [cb waitUntilCompleted];
                        const uint8_t *d1 = db.contents;
                        printf("%-10s %-5s offset %u: ", fn.UTF8String, store ? "store" : "load", o);
                        if (cb.status != MTLCommandBufferStatusCompleted) {
                            printf("command buffer status %ld: %s\n", (long)cb.status, cb.error.localizedDescription.UTF8String);
                            failures++;
                            continue;
                        }
                        int found = 0;
                        printf("b =");
                        for (unsigned b = 0; b < MAXB; b++) {
                            int ok = 1;
                            if (!store) {
                                for (unsigned i = 0; i < np && ok; i++)
                                    ok = memcmp(d1 + pairs[i].d * size, s0 + b + pairs[i].s * size, size) == 0;
                            } else {
                                uint8_t want[LEN];
                                memcpy(want, d0, LEN);
                                for (unsigned i = 0; i < np; i++)
                                    memcpy(want + b + pairs[i].d * size, s0 + pairs[i].s * size, size);
                                ok = memcmp(want, d1, LEN) == 0;
                            }
                            if (ok) { printf(" %u", b); found++; }
                        }
                        if (!found) {
                            long lo = -1, hi = -1, n = 0;
                            for (long i = 0; i < LEN; i++)
                                if (d1[i] != d0[i]) { if (lo < 0) lo = i; hi = i; n++; }
                            printf(" none; changed bytes %ld, from %ld to %ld", n, lo, hi);
                            failures++;
                        }
                        printf("%s\n", (found && (store ? 1 : 1)) ? (found == 1 ? "" : " (several)") : "");
                    }
                }
            }
        }
        struct { const char *name; unsigned size; unsigned offs[4]; unsigned n; } pt[] = {
            {"u32", 4, {0, 1, 2, 3}, 4}, {"u128", 16, {0, 4, 8, 0}, 3}};
        for (unsigned ti = 0; ti < 2; ti++) {
            unsigned size = pt[ti].size;
            NSString *fn = [NSString stringWithFormat:@"ptr_%s", pt[ti].name];
            id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:fn] error:&err];
            if (!pso) { printf("%s: %s\n", fn.UTF8String, err.localizedDescription.UTF8String); return 1; }
            for (unsigned oi = 0; oi < pt[ti].n; oi++) {
                unsigned o = pt[ti].offs[oi];
                id<MTLBuffer> ab = [dev newBufferWithLength:LEN options:MTLResourceStorageModeShared];
                id<MTLBuffer> bb = [dev newBufferWithLength:LEN options:MTLResourceStorageModeShared];
                id<MTLBuffer> db = [dev newBufferWithLength:LEN options:MTLResourceStorageModeShared];
                id<MTLBuffer> sb = [dev newBufferWithLength:LEN options:MTLResourceStorageModeShared];
                id<MTLBuffer> pb = [dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
                uint8_t a0[LEN], b0[LEN], d0[LEN], s0[LEN];
                fill(a0, LEN); fill(b0, LEN); fill(d0, LEN); fill(s0, LEN);
                memcpy(ab.contents, a0, LEN); memcpy(bb.contents, b0, LEN);
                memcpy(db.contents, d0, LEN); memcpy(sb.contents, s0, LEN);
                uint64_t addrs[2] = {ab.gpuAddress + o, bb.gpuAddress + o};
                memcpy(pb.contents, addrs, 16);
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                [e setComputePipelineState:pso];
                [e setBuffer:pb offset:0 atIndex:0];
                [e setBuffer:db offset:0 atIndex:1];
                [e setBuffer:sb offset:0 atIndex:2];
                [e useResource:ab usage:MTLResourceUsageRead];
                [e useResource:bb usage:MTLResourceUsageRead | MTLResourceUsageWrite];
                [e dispatchThreads:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(N, 1, 1)];
                [e endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                if (cb.status != MTLCommandBufferStatusCompleted) {
                    printf("%-10s gpuAddress + %u: command buffer status %ld: %s\n", fn.UTF8String, o, (long)cb.status, cb.error.localizedDescription.UTF8String);
                    failures++;
                    continue;
                }
                const uint8_t *d1 = db.contents, *b1 = bb.contents;
                for (int store = 0; store < 2; store++) {
                    printf("%-10s %-5s gpuAddress + %u: b =", fn.UTF8String, store ? "store" : "load", o);
                    int found = 0;
                    for (unsigned b = 0; b < MAXB; b++) {
                        int ok = 1;
                        if (!store) {
                            for (unsigned t = 0; t < N && ok; t++)
                                ok = memcmp(d1 + t * size, a0 + b + t * size, size) == 0;
                        } else {
                            uint8_t want[LEN];
                            memcpy(want, b0, LEN);
                            for (unsigned t = 0; t < N; t++) memcpy(want + b + t * size, s0 + t * size, size);
                            ok = memcmp(want, b1, LEN) == 0;
                        }
                        if (ok) { printf(" %u", b); found++; }
                    }
                    if (!found) {
                        const uint8_t *x1 = store ? b1 : d1, *x0 = store ? b0 : d0;
                        long lo = -1, hi = -1, n = 0;
                        for (long i = 0; i < LEN; i++)
                            if (x1[i] != x0[i]) { if (lo < 0) lo = i; hi = i; n++; }
                        printf(" none; changed bytes %ld, from %ld to %ld", n, lo, hi);
                        failures++;
                    }
                    printf("\n");
                }
            }
        }
        printf("cases without a matching b: %d\n", failures);
    }
    return 0;
}
