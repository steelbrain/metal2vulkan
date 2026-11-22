// What plain Metal answers for the 16x16x16 simdgroup matrix products,
// `air.simdgroup_matrix_16x16x16_[widening_]multiply_accumulate`: `<8 x R> (<8 x A>, i1,
// <8 x B>, i1, <8 x R>)`, each lane's eight cells of `A`, its transpose flag, of `B`, its flag,
// and of the accumulator `C`. MSL has no spelling for them; the owned fixture
// `engine/simd_matrix16.metal` declares one function per form, renamed to the intrinsic
// (scripts/regen-air-fixtures), in kernels `m16_<form>_<ta><tb>` that pass element `t` of each
// buffer as thread `t`'s cells: `float` factors (`f32_f32`), `half` factors (`f16_f16`) and signed
// 8-bit integers widened to `int` (`s_s`), each with the four flag pairs.
//
// Run with simd_matrix16.sh beside this file (it assembles the committed fixture's typed text
// into a metallib and builds this probe), or by hand:
//   xcrun metal-as simd_matrix16.air.ll -o m.air && xcrun metallib m.air -o m.metallib
//   clang -fobjc-arc -O1 -fmodules -o simd_matrix16 simd_matrix16.m \
//     -framework Foundation -framework Metal
//   ./simd_matrix16 m.metallib <form>      (one form a process; MTL_DEBUG_LAYER unset)
//
// For each flag pair it prints whether a compute pipeline was made (or Metal's error), then, for
// a pipeline Metal made, the measured relation of the product, as dispatches of 256 one-simdgroup
// threadgroups:
//
// - `pairs`: cell `p` (lane `p >> 3`, element `p & 7`) of one factor is 1 and every other cell of
//   it 0, the other factor's cell `q` holds a digit of `q` plus one (two passes for 8-bit
//   integers), and `C` is 0; result cell `d` is then 0 or the digit of the one cell `q` the product
//   pairs with `p` for `d`. A line `REL <form> <tt> <d> <hex>` lists, for result cell `d`, every
//   pair (`A` cell, `B` cell) as two bytes, sorted.
// - `accumulate`: `A` and `B` 0, `C` distinct integers: the result equals `C` cell for cell.
// - `check`: small integer cells everywhere, every sum exact in the result format: the result
//   equals `C + Σ A·B` over the measured pairs, cell for cell.
//
// Exit status: 0 when Metal answered (a refused pipeline is an answer), 2 when the probe could not
// ask (arguments, the metallib, a command buffer's error).
//
// WHAT IT MEASURED (Apple M3 Max, macOS 27.0.1, Apple metal 32023.922): Metal refuses every
// pipeline of the three forms and four flag pairs at its creation, in 1 to 42 ms, with
// `AGXMetalG15X_M1` code 3, "simdgroup_matrix<T,16,16x16> operations are supported by
// GPUFamily10 and later". No pipeline was made, so no relation was measured. The translator
// refuses these products `Unsupported{SimdgroupMatrix16x16x16, NotImplemented}` on every device.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum { F32, F16, S8, I32 } Fmt;

typedef struct {
    const char *name;
    Fmt a, b, r;
} Form;

static const Form FORMS[] = {
    {"f32_f32", F32, F32, F32},
    {"f16_f16", F16, F16, F32},
    {"s_s", S8, S8, I32},
};

enum { GROUPS = 256, CELLS = 256, NONE = 0xffff };

static unsigned fsize(Fmt f) {
    switch (f) {
    case F32:
    case I32:
        return 4;
    case F16:
        return 2;
    case S8:
        return 1;
    }
    return 4;
}

// The integers 1..base are exact in the format.
static unsigned base_of(Fmt f) { return f == S8 ? 16 : 256; }

static unsigned passes_of(Fmt f) {
    unsigned b = base_of(f), n = 0, reach = 1;
    while (reach < CELLS) {
        reach *= b;
        n++;
    }
    return n;
}

static void die(const char *what) {
    printf("PROBE-ERROR %s\n", what);
    exit(2);
}

static void put(Fmt f, uint8_t *base, size_t index, double v) {
    uint8_t *p = base + index * fsize(f);
    switch (f) {
    case F32: {
        float x = (float)v;
        memcpy(p, &x, 4);
        break;
    }
    case I32: {
        int32_t x = (int32_t)v;
        memcpy(p, &x, 4);
        break;
    }
    case F16: {
        _Float16 h = (_Float16)v;
        if ((double)h != v) {
            die("a half value is not exact");
        }
        memcpy(p, &h, 2);
        break;
    }
    case S8:
        if (v < -128 || v > 127) {
            die("an int8 value out of range");
        }
        *p = (uint8_t)(int8_t)v;
        break;
    }
}

static double get(Fmt r, const uint8_t *base, size_t index) {
    const uint8_t *p = base + index * fsize(r);
    if (r == I32) {
        int32_t x;
        memcpy(&x, p, 4);
        return x;
    }
    float x;
    memcpy(&x, p, 4);
    return x;
}

typedef struct {
    id<MTLCommandQueue> q;
    id<MTLComputePipelineState> pso;
    const Form *form;
    id<MTLBuffer> a, b, c, d;
} Run;

static void clear(Run *r) {
    memset(r->a.contents, 0, r->a.length);
    memset(r->b.contents, 0, r->b.length);
    memset(r->c.contents, 0, r->c.length);
    memset(r->d.contents, 0xa5, r->d.length);
}

static void dispatch(Run *r, NSUInteger groups) {
    id<MTLCommandBuffer> cb = [r->q commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:r->pso];
    [e setBuffer:r->a offset:0 atIndex:0];
    [e setBuffer:r->b offset:0 atIndex:1];
    [e setBuffer:r->c offset:0 atIndex:2];
    [e setBuffer:r->d offset:0 atIndex:3];
    [e dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
        printf("PROBE-ERROR command buffer status %ld: %s\n", (long)cb.status,
               cb.error.description.UTF8String);
        exit(2);
    }
}

// Cell `p` of simdgroup `s`: element `s * 256 + p` (lane `p >> 3`, element `p & 7`).
static size_t at(unsigned s, unsigned p) { return (size_t)s * CELLS + p; }

static uint64_t lcg(uint64_t *x) {
    *x = *x * 6364136223846793005ull + 1442695040888963407ull;
    return *x >> 33;
}

static void measure(Run *r, const char *tt) {
    const Form *f = r->form;
    // The one-hot factor is `A`, the counted one `B` (both factors of a form share a format).
    unsigned base = base_of(f->b), passes = passes_of(f->b);
    static uint16_t rel[CELLS][CELLS]; // [p][d]: the counted cell
    static int seen[CELLS][CELLS];     // passes with a nonzero result
    memset(rel, 0, sizeof rel);
    memset(seen, 0, sizeof seen);
    unsigned odd = 0;
    unsigned scale = 1;
    for (unsigned pass = 0; pass < passes; pass++, scale *= base) {
        clear(r);
        for (unsigned s = 0; s < GROUPS; s++) {
            put(f->a, r->a.contents, at(s, s), 1);
            for (unsigned q = 0; q < CELLS; q++) {
                put(f->b, r->b.contents, at(s, q), (q / scale) % base + 1);
            }
        }
        dispatch(r, GROUPS);
        for (unsigned s = 0; s < GROUPS; s++) {
            for (unsigned d = 0; d < CELLS; d++) {
                double v = get(f->r, r->d.contents, at(s, d));
                if (v == 0) {
                    continue;
                }
                if (v != floor(v) || v < 1 || v > base) {
                    if (odd++ < 8) {
                        printf("  odd value %s %s pass %u p %u d %u: %.9g\n", f->name, tt, pass,
                               s, d, v);
                    }
                    continue;
                }
                seen[s][d]++;
                rel[s][d] += (uint16_t)(((unsigned)v - 1) * scale);
            }
        }
    }
    unsigned mixed = 0, per_min = CELLS, per_max = 0;
    static unsigned char pairs[CELLS][CELLS][2];
    unsigned count[CELLS];
    uint64_t h = 1469598103934665603ull;
    for (unsigned d = 0; d < CELLS; d++) {
        count[d] = 0;
        // Ascending `A` cell, each with its one `B` cell: the pairs are sorted.
        for (unsigned p = 0; p < CELLS; p++) {
            if (seen[p][d] == 0) {
                continue;
            }
            if (seen[p][d] != (int)passes || rel[p][d] >= CELLS) {
                mixed++;
                continue;
            }
            pairs[d][count[d]][0] = (unsigned char)p;
            pairs[d][count[d]][1] = (unsigned char)rel[p][d];
            count[d]++;
        }
        if (count[d] < per_min) per_min = count[d];
        if (count[d] > per_max) per_max = count[d];
        printf("REL %s %s %u ", f->name, tt, d);
        for (unsigned i = 0; i < count[d]; i++) {
            printf("%02x%02x", pairs[d][i][0], pairs[d][i][1]);
            h = (h ^ pairs[d][i][0]) * 1099511628211ull;
            h = (h ^ pairs[d][i][1]) * 1099511628211ull;
        }
        h = (h ^ 0xff) * 1099511628211ull;
        printf("\n");
    }
    printf("%s %s pairs: per result cell %u..%u, odd values %u, partial cells %u, passes %u, "
           "relation %016llx\n",
           f->name, tt, per_min, per_max, odd, mixed, passes, (unsigned long long)h);

    // accumulate: A = B = 0, C distinct integers in two simdgroups.
    clear(r);
    for (unsigned s = 0; s < 2; s++) {
        for (unsigned d = 0; d < CELLS; d++) {
            put(f->r, r->c.contents, at(s, d), 1000.0 * s + d + 1);
        }
    }
    dispatch(r, 2);
    unsigned acc_bad = 0;
    for (unsigned s = 0; s < 2; s++) {
        for (unsigned d = 0; d < CELLS; d++) {
            if (get(f->r, r->d.contents, at(s, d)) != 1000.0 * s + d + 1) {
                acc_bad++;
            }
        }
    }
    printf("%s %s accumulate: %u of 512 cells differ from C\n", f->name, tt, acc_bad);

    // check: small integers everywhere against C + sum over the measured pairs.
    enum { CHECK = 8 };
    clear(r);
    uint64_t x = 0x5eed1234u;
    static double av[CHECK][CELLS], bv[CHECK][CELLS], cv[CHECK][CELLS];
    for (unsigned s = 0; s < CHECK; s++) {
        for (unsigned p = 0; p < CELLS; p++) {
            av[s][p] = (double)(int)(lcg(&x) % 7) - 3;
            bv[s][p] = (double)(int)(lcg(&x) % 7) - 3;
            cv[s][p] = (double)(int)(lcg(&x) % 17) - 8;
            put(f->a, r->a.contents, at(s, p), av[s][p]);
            put(f->b, r->b.contents, at(s, p), bv[s][p]);
            put(f->r, r->c.contents, at(s, p), cv[s][p]);
        }
    }
    dispatch(r, CHECK);
    unsigned bad = 0;
    for (unsigned s = 0; s < CHECK; s++) {
        for (unsigned d = 0; d < CELLS; d++) {
            double want = cv[s][d];
            for (unsigned i = 0; i < count[d]; i++) {
                want += av[s][pairs[d][i][0]] * bv[s][pairs[d][i][1]];
            }
            double got = get(f->r, r->d.contents, at(s, d));
            if (got != want && bad++ < 4) {
                printf("  check %s %s simdgroup %u cell %u: %.9g, measured pairs give %.9g\n",
                       f->name, tt, s, d, got, want);
            }
        }
    }
    printf("%s %s check: %u of %u cells differ from C + sum over the measured pairs\n", f->name,
           tt, bad, CHECK * CELLS);
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc != 3) {
            die("usage: simd_matrix16 <metallib> <form>");
        }
        const Form *form = NULL;
        for (size_t i = 0; i < sizeof FORMS / sizeof FORMS[0]; i++) {
            if (strcmp(FORMS[i].name, argv[2]) == 0) {
                form = &FORMS[i];
            }
        }
        if (!form) {
            die("unknown form");
        }
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            die("no Metal device");
        }
        printf("device %s, form %s\n", dev.name.UTF8String, form->name);
        NSError *err = nil;
        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
        id<MTLLibrary> lib = [dev newLibraryWithURL:url error:&err];
        if (!lib) {
            printf("PROBE-ERROR library: %s\n", err.description.UTF8String);
            return 2;
        }
        Run r = {.q = [dev newCommandQueue], .form = form};
        r.a = [dev newBufferWithLength:GROUPS * CELLS * fsize(form->a)
                               options:MTLResourceStorageModeShared];
        r.b = [dev newBufferWithLength:GROUPS * CELLS * fsize(form->b)
                               options:MTLResourceStorageModeShared];
        r.c = [dev newBufferWithLength:GROUPS * CELLS * fsize(form->r)
                               options:MTLResourceStorageModeShared];
        r.d = [dev newBufferWithLength:GROUPS * CELLS * fsize(form->r)
                               options:MTLResourceStorageModeShared];
        static const char *TT[] = {"00", "01", "10", "11"};
        for (int k = 0; k < 4; k++) {
            NSString *name = [NSString stringWithFormat:@"m16_%s_%s", form->name, TT[k]];
            id<MTLFunction> fn = [lib newFunctionWithName:name];
            if (!fn) {
                printf("%s %s function: missing from the library\n", form->name, TT[k]);
                return 2;
            }
            err = nil;
            NSDate *t0 = [NSDate date];
            id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn
                                                                                  error:&err];
            double ms = -[t0 timeIntervalSinceNow] * 1000.0;
            if (!pso) {
                printf("%s %s pipeline: REFUSED after %.0f ms: %s\n", form->name, TT[k], ms,
                       err.description.UTF8String);
                continue;
            }
            printf("%s %s pipeline: made in %.0f ms, threadExecutionWidth %lu\n", form->name,
                   TT[k], ms, (unsigned long)pso.threadExecutionWidth);
            if (pso.threadExecutionWidth != 32) {
                die("a simdgroup is not 32 threads");
            }
            r.pso = pso;
            measure(&r, TT[k]);
        }
    }
    return 0;
}
