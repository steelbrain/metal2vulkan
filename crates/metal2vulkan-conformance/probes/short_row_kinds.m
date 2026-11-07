// What plain Metal's blits do at a `bytesPerRow` below the region's row for the kinds of texture
// `short_row_blits.m` left out: whether each copy returns, where a copy into the texture reads
// each row, and which row's bytes stand where a copy out of it writes rows that overlap.
//
// The kinds, 2D, `Private` unless named:
//
//   ms4-rgba8   `RGBA8Unorm` multisampled, 4 samples, 8 x 8 (a pixel's samples one after
//               another in the buffer: 16 bytes a pixel, 128 a row). Out only.
//   ms2-r8      `R8Unorm` multisampled, 2 samples, 16 x 8 (32 bytes a row). Out only.
//   ms4-rgba8-w4, ms4-rgba8-w2, ms2-rgba8-w4, ms2-r32-w8: narrower multisampled ones, out only.
//   overbuf     `RGBA8Unorm` 32 x 8 over a `Shared` buffer at a `bytesPerRow` of 128 (the
//               texture's bytes read and written in that buffer directly).
//   depth16     `Depth16Unorm` 32 x 8.
//   stencil8    `Stencil8` 32 x 8.
//   d32s8-depth `Depth32Float_Stencil8` 32 x 8, its depth (`MTLBlitOptionDepthFromDepthStencil`).
//   d32s8-sten  The same, its stencil (`MTLBlitOptionStencilFromDepthStencil`).
//   xr10        `RGB10A8_2P_XR10` (550) 32 x 8 (4 bytes a texel in a blit's buffer).
//
// Each case runs in a process of its own (a driver assert ends that process alone; `DIED
// <status>` is then the reading), at `bytesPerRow`s below the row: half of it, half plus a copy
// unit where that is off 16, and the row less 16.
//
//   in   The texture filled from a buffer of bytes unique to their offset at `bytesPerRow`,
//        then copied out at its packed row. Each row is compared with the same bytes copied in
//        at the packed row and out again: read whole from `y x bytesPerRow` (`whole`), from
//        `y x (bytesPerRow rounded down to 16)` (`aligned`, Metal's aligned path), or neither.
//   out  The texture filled at its packed row, then copied out at `bytesPerRow` into a buffer of
//        0xCC. For each 16-byte chunk of the rows' span, the row whose bytes stand there, rows
//        `y x bytesPerRow` apart (`0`-`7`; `?` for none, `*` for more than one), the bytes past
//        the span written, and whether the buffer is the rows written in order (`later`), in
//        reverse order (`earlier`), the odd rows then the even ones (`even`), or neither.
//
// Build (no GPU needed to build):
//
//   xcrun clang -fobjc-arc -framework Metal -framework Foundation \
//       -o short_row_kinds short_row_kinds.m
//
// Run without MTL_DEBUG_LAYER (its validation stops a `bytesPerRow` below the row):
//
//   ./short_row_kinds

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static id<MTLDevice> gDevice = nil;
static id<MTLCommandQueue> gQueue = nil;

#define kHeight ((NSUInteger)8)

struct kind {
    const char *name;
    MTLPixelFormat pf;
    NSUInteger width;
    NSUInteger samples;
    NSUInteger unit; // bytes a pixel in a blit's buffer (a pixel's samples together)
    MTLBlitOption option;
    bool overBuffer;
    bool outOnly;
};

static const struct kind kKinds[] = {
    {"ms4-rgba8", MTLPixelFormatRGBA8Unorm, 8, 4, 16, MTLBlitOptionNone, false, true},
    {"ms2-r8", MTLPixelFormatR8Unorm, 16, 2, 2, MTLBlitOptionNone, false, true},
    {"ms4-rgba8-w4", MTLPixelFormatRGBA8Unorm, 4, 4, 16, MTLBlitOptionNone, false, true},
    {"ms4-rgba8-w2", MTLPixelFormatRGBA8Unorm, 2, 4, 16, MTLBlitOptionNone, false, true},
    {"ms2-rgba8-w4", MTLPixelFormatRGBA8Unorm, 4, 2, 8, MTLBlitOptionNone, false, true},
    {"ms2-r32-w8", MTLPixelFormatR32Uint, 8, 2, 8, MTLBlitOptionNone, false, true},
    {"overbuf", MTLPixelFormatRGBA8Unorm, 32, 1, 4, MTLBlitOptionNone, true, false},
    {"depth16", MTLPixelFormatDepth16Unorm, 32, 1, 2, MTLBlitOptionNone, false, false},
    {"stencil8", MTLPixelFormatStencil8, 32, 1, 1, MTLBlitOptionNone, false, false},
    {"d32s8-depth", MTLPixelFormatDepth32Float_Stencil8, 32, 1, 4,
     MTLBlitOptionDepthFromDepthStencil, false, false},
    {"d32s8-sten", MTLPixelFormatDepth32Float_Stencil8, 32, 1, 1,
     MTLBlitOptionStencilFromDepthStencil, false, false},
    {"xr10", (MTLPixelFormat)550, 32, 1, 4, MTLBlitOptionNone, false, false},
};

/// A byte unique to `(tag, index)` for any index below 2^24.
static uint8_t patternByte(uint32_t tag, uint32_t index) {
    uint32_t x = index * 2654435761u ^ tag * 40503u;
    x ^= x >> 15;
    x *= 2246822519u;
    x ^= x >> 13;
    return (uint8_t)(x ^ (x >> 24));
}

/// `length` bytes of the pattern `tag`, as a depth format holds them: a float depth's words in
/// [0, 1) (no NaN for the copy to change), a stencil's or unorm depth's bytes as they come.
static void fillPattern(const struct kind *k, uint8_t *p, NSUInteger length, uint32_t tag) {
    if (k->option == MTLBlitOptionDepthFromDepthStencil) {
        for (NSUInteger i = 0; i + 4 <= length; i += 4) {
            float v = (float)((patternByte(tag, (uint32_t)i) << 8 | patternByte(tag, (uint32_t)i + 1)) % 65521) / 65521.0f;
            memcpy(p + i, &v, 4);
        }
        return;
    }
    for (NSUInteger i = 0; i < length; i++) {
        p[i] = patternByte(tag, (uint32_t)i);
    }
}

static bool waitFor(id<MTLCommandBuffer> cb) {
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) {
        printf("    command buffer error %s\n", cb.error.localizedDescription.UTF8String);
        return false;
    }
    return true;
}

static id<MTLBuffer> sharedBuffer(NSUInteger length, uint8_t fill) {
    id<MTLBuffer> b = [gDevice newBufferWithLength:length options:MTLResourceStorageModeShared];
    memset(b.contents, fill, length);
    return b;
}

static NSUInteger rowOf(const struct kind *k) {
    return k->width * k->unit;
}

/// The kind's texture; for `overbuf`, `backing` the buffer it lies over.
static id<MTLTexture> makeTexture(const struct kind *k, id<MTLBuffer> *backing) {
    MTLTextureDescriptor *d = [MTLTextureDescriptor new];
    d.pixelFormat = k->pf;
    d.width = k->width;
    d.height = kHeight;
    d.usage = MTLTextureUsageShaderRead;
    if (k->samples > 1) {
        d.textureType = MTLTextureType2DMultisample;
        d.sampleCount = k->samples;
        d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    }
    if (k->overBuffer) {
        d.storageMode = MTLStorageModeShared;
        NSUInteger row = rowOf(k);
        *backing = sharedBuffer(row * kHeight, 0);
        return [*backing newTextureWithDescriptor:d offset:0 bytesPerRow:row];
    }
    d.storageMode = MTLStorageModePrivate;
    return [gDevice newTextureWithDescriptor:d];
}

static void copyIn(const struct kind *k, id<MTLTexture> t, id<MTLBuffer> src, NSUInteger bpr) {
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:src
               sourceOffset:0
          sourceBytesPerRow:bpr
        sourceBytesPerImage:bpr * kHeight
                 sourceSize:MTLSizeMake(k->width, kHeight, 1)
                  toTexture:t
           destinationSlice:0
           destinationLevel:0
          destinationOrigin:MTLOriginMake(0, 0, 0)
                    options:k->option];
    [blit endEncoding];
    waitFor(cb);
}

static void copyOut(const struct kind *k, id<MTLTexture> t, id<MTLBuffer> dst, NSUInteger bpr) {
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromTexture:t
                     sourceSlice:0
                     sourceLevel:0
                    sourceOrigin:MTLOriginMake(0, 0, 0)
                      sourceSize:MTLSizeMake(k->width, kHeight, 1)
                        toBuffer:dst
               destinationOffset:0
          destinationBytesPerRow:bpr
        destinationBytesPerImage:bpr * kHeight
                         options:k->option];
    [blit endEncoding];
    waitFor(cb);
}

/// The texture's rows as a copy out at the packed row reads them (for `overbuf`, its buffer's).
static void readPacked(const struct kind *k, id<MTLTexture> t, id<MTLBuffer> backing,
                       uint8_t *out) {
    NSUInteger row = rowOf(k);
    if (k->overBuffer) {
        memcpy(out, backing.contents, row * kHeight);
        return;
    }
    id<MTLBuffer> dst = sharedBuffer(row * kHeight, 0);
    copyOut(k, t, dst, row);
    memcpy(out, dst.contents, row * kHeight);
}

/// `bytes` (packed rows) into the texture: at the packed row, or for `overbuf` into its buffer.
static void writePacked(const struct kind *k, id<MTLTexture> t, id<MTLBuffer> backing,
                        const uint8_t *bytes) {
    NSUInteger row = rowOf(k);
    if (k->overBuffer) {
        memcpy(backing.contents, bytes, row * kHeight);
        return;
    }
    id<MTLBuffer> src = sharedBuffer(row * kHeight, 0);
    memcpy(src.contents, bytes, row * kHeight);
    copyIn(k, t, src, row);
}

static void inCase(const struct kind *k, NSUInteger bpr) {
    NSUInteger row = rowOf(k), length = bpr * (kHeight - 1) + row;
    id<MTLBuffer> backing = nil;
    id<MTLTexture> t = makeTexture(k, &backing);
    if (!t) {
        printf("in %s row=%lu bpr=%lu: no texture\n", k->name, (unsigned long)row,
               (unsigned long)bpr);
        return;
    }
    id<MTLBuffer> src = sharedBuffer(length, 0);
    uint8_t *p = src.contents;
    fillPattern(k, p, length, 1);
    copyIn(k, t, src, bpr);
    uint8_t *got = malloc(row * kHeight), *laid = malloc(row * kHeight);
    uint8_t *want[2] = {malloc(row * kHeight), malloc(row * kHeight)};
    readPacked(k, t, backing, got);
    NSUInteger aligned = bpr - bpr % 16, whole = 0, pathed = 0, neither = 0;
    // The source bytes at each pitch, copied into a texture of the kind at the packed row and
    // read back: what a row read whole from there holds.
    bool hasAligned = aligned != bpr && aligned != 0;
    for (int pass = 0; pass < (hasAligned ? 2 : 1); pass++) {
        NSUInteger pitch = pass == 0 ? bpr : aligned;
        for (NSUInteger r = 0; r < kHeight; r++) {
            memcpy(laid + r * row, p + r * pitch, row);
        }
        @autoreleasepool {
            id<MTLBuffer> rb = nil;
            id<MTLTexture> u = makeTexture(k, &rb);
            writePacked(k, u, rb, laid);
            readPacked(k, u, rb, want[pass]);
        }
    }
    for (NSUInteger y = 0; y < kHeight; y++) {
        bool w = memcmp(got + y * row, want[0] + y * row, row) == 0;
        bool a = hasAligned && memcmp(got + y * row, want[1] + y * row, row) == 0;
        whole += w;
        pathed += a && !w;
        neither += !w && !a;
    }
    printf("in %s row=%lu bpr=%lu: returned; whole %lu aligned %lu neither %lu of %lu\n", k->name,
           (unsigned long)row, (unsigned long)bpr, (unsigned long)whole, (unsigned long)pathed,
           (unsigned long)neither, (unsigned long)kHeight);
    free(got);
    free(want[0]);
    free(want[1]);
    free(laid);
}

/// `rows` packed rows of `row` bytes from `p` written `bpr` apart into `out`, in `order`.
static void writeRows(uint8_t *out, const uint8_t *p, NSUInteger row, NSUInteger bpr,
                      const NSUInteger *order) {
    for (NSUInteger k = 0; k < kHeight; k++) {
        memcpy(out + order[k] * bpr, p + order[k] * row, row);
    }
}

static void outCase(const struct kind *k, NSUInteger bpr) {
    NSUInteger row = rowOf(k);
    id<MTLBuffer> backing = nil;
    id<MTLTexture> t = makeTexture(k, &backing);
    if (!t) {
        printf("out %s row=%lu bpr=%lu: no texture\n", k->name, (unsigned long)row,
               (unsigned long)bpr);
        return;
    }
    uint8_t *p = malloc(row * kHeight), *ref = malloc(row * kHeight);
    fillPattern(k, p, row * kHeight, 2);
    // A multisampled texture too is filled by a copy in at its packed row, each pixel's samples
    // one after another, as a blit's buffer lays them. Each step is printed before it runs, so a
    // case Metal ends names the call it ended in.
    printf("  out %s bpr=%lu: copy in at the packed row\n", k->name, (unsigned long)bpr);
    fflush(stdout);
    writePacked(k, t, backing, p);
    printf("  out %s bpr=%lu: copy out at the packed row\n", k->name, (unsigned long)bpr);
    fflush(stdout);
    readPacked(k, t, backing, ref);
    NSUInteger span = bpr * (kHeight - 1) + row;
    id<MTLBuffer> dst = sharedBuffer(span + 64, 0xCC);
    printf("  out %s bpr=%lu: copy out at the pitch\n", k->name, (unsigned long)bpr);
    fflush(stdout);
    copyOut(k, t, dst, bpr);
    const uint8_t *q = dst.contents;
    printf("out %s row=%lu bpr=%lu: returned; packed %s; chunks ", k->name, (unsigned long)row,
           (unsigned long)bpr, memcmp(p, ref, row * kHeight) == 0 ? "as written" : "changed");
    for (NSUInteger c = 0; c < span; c += 16) {
        NSUInteger n = span - c < 16 ? span - c : 16;
        int winner = -1, matches = 0;
        for (NSUInteger y = 0; y < kHeight; y++) {
            if (c < y * bpr || c + n > y * bpr + row) {
                continue;
            }
            if (memcmp(q + c, ref + y * row + (c - y * bpr), n) == 0) {
                winner = (int)y;
                matches++;
            }
        }
        putchar(matches == 0 ? '?' : matches > 1 ? '*' : (char)('0' + winner));
    }
    NSUInteger past = 0;
    for (NSUInteger i = span; i < span + 64; i++) {
        past += q[i] != 0xCC;
    }
    NSUInteger order[kHeight], n = 0;
    uint8_t *later = malloc(span + 64), *earlier = malloc(span + 64), *even = malloc(span + 64);
    memset(later, 0xCC, span + 64);
    memset(earlier, 0xCC, span + 64);
    memset(even, 0xCC, span + 64);
    for (NSUInteger y = 0; y < kHeight; y++) {
        order[y] = y;
    }
    writeRows(later, ref, row, bpr, order);
    for (NSUInteger y = 0; y < kHeight; y++) {
        order[y] = kHeight - 1 - y;
    }
    writeRows(earlier, ref, row, bpr, order);
    for (NSUInteger y = 1; y < kHeight; y += 2) {
        order[n++] = y;
    }
    for (NSUInteger y = 0; y < kHeight; y += 2) {
        order[n++] = y;
    }
    writeRows(even, ref, row, bpr, order);
    const char *which = memcmp(q, later, span + 64) == 0     ? "later"
                        : memcmp(q, earlier, span + 64) == 0 ? "earlier"
                        : memcmp(q, even, span + 64) == 0    ? "even"
                                                             : "neither";
    printf(" past %lu; %s\n", (unsigned long)past, which);
    free(p);
    free(ref);
    free(later);
    free(earlier);
    free(even);
}

int main(int argc, const char **argv) {
    (void)argc;
    (void)argv;
    bool announced = false;
    for (size_t i = 0; i < sizeof kKinds / sizeof kKinds[0]; i++) {
        const struct kind *k = &kKinds[i];
        NSUInteger row = rowOf(k), bprs[3], n = 0;
        NSUInteger half = row / 2, halfPlus = row / 2 + k->unit, less = row - 16;
        if (half % k->unit == 0 && half > 0) {
            bprs[n++] = half;
        }
        if (halfPlus < row && halfPlus % 16 != 0) {
            bprs[n++] = halfPlus;
        }
        if (row > 16 && less > half && less % k->unit == 0) {
            bprs[n++] = less;
        }
        for (int dir = 0; dir < 2; dir++) {
            if (dir == 0 && k->outOnly) {
                continue;
            }
            for (NSUInteger b = 0; b < n; b++) {
                fflush(stdout);
                bool announce = !announced;
                announced = true;
                pid_t pid = fork();
                if (pid == 0) {
                    @autoreleasepool {
                        gDevice = MTLCreateSystemDefaultDevice();
                        gQueue = [gDevice newCommandQueue];
                        if (announce) {
                            printf("DEVICE %s / RELEASE %s\n", gDevice.name.UTF8String,
                                   [NSProcessInfo processInfo]
                                       .operatingSystemVersionString.UTF8String);
                        }
                        if (dir == 0) {
                            inCase(k, bprs[b]);
                        } else {
                            outCase(k, bprs[b]);
                        }
                    }
                    fflush(stdout);
                    _exit(0);
                }
                int status = 0;
                waitpid(pid, &status, 0);
                if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
                    printf("%s %s row=%lu bpr=%lu: DIED %d\n", dir == 0 ? "in" : "out", k->name,
                           (unsigned long)row, (unsigned long)bprs[b], status);
                }
            }
        }
    }
    return 0;
}
