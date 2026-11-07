// What decides whether plain Metal's blit between a buffer and a multisampled texture returns:
// the texture's usage, its storage mode, `allowGPUOptimizedContents`, the copy's call (with or
// without `options:`), a render pass that clears the texture first, or how the process that runs
// the copy started (forked without an exec, as `short_row_kinds.m` runs its cases, or spawned).
// And, on the paths that return, where Metal places the rows of the layouts the runtime's
// `multisample_buffer_layouts` test holds it to (`sample_interleave::placed_rows`).
//
// Every case runs in a process of its own (`DIED <status>` is then the reading), each step
// printed before it runs, so a case Metal ends names the call it ended in:
//
//   trip   A `Private`-or-named 2D multisample texture of `RGBA8Unorm` at 4 samples, 8 x 8, or of
//          `R8Unorm` at 2 samples, 16 x 8 (the first two kinds of `short_row_kinds.m`), filled
//          from a buffer at its packed row (`copyFromBuffer:...toTexture:`, offset 0) and copied
//          out at the packed row into a buffer of 0xCC (`copyFromTexture:...toBuffer:`).
//          Printed: whether each copy returned and whether the bytes came back as written.
//   place  One of the nine layouts `multisample_buffer_layouts` holds the runtime to, four rows
//          tall, `Private`, at usage `rt+sr` and `rt+sr+pfv`, spawned. `into`: the texture filled
//          from a buffer of 4096 bytes unique to their offset at the layout, then copied out at
//          the packed row; each row's offset in the source is where its bytes are found. `out`:
//          the texture filled at the packed row, then copied out at the layout into 4096 bytes of
//          0xCC; each row's offset is where its bytes are found there (`?`: nowhere). Printed
//          beside the rows the test expects.
//
// Usages: `rt+sr` (RenderTarget | ShaderRead, what `short_row_kinds.m` and the runtime test
// make), `rt+sr+pfv` (with PixelFormatView, what the conformance suite's multisample textures
// carry), `sr`, `sr+pfv`. `plain` is `allowGPUOptimizedContents` off. `noopt` is the call without
// `options:`. `clear` is a render pass that clears the texture before the copy in (render-target
// usages only). `fork` runs the case in a forked child that has not exec'd; every other case is
// spawned (`posix_spawn` of this binary).
//
// Build (no GPU needed to build):
//
//   xcrun clang -fobjc-arc -framework Metal -framework Foundation \
//       -o multisample_buffer_copies multisample_buffer_copies.m
//
// Run without MTL_DEBUG_LAYER (its validation stops the layouts off the pixel):
//
//   ./multisample_buffer_copies

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

struct format {
    const char *name;
    MTLPixelFormat pf;
    NSUInteger bpp;
};

static const struct format kFormats[] = {
    {"rgba8", MTLPixelFormatRGBA8Unorm, 4},  {"r8", MTLPixelFormatR8Unorm, 1},
    {"r16u", MTLPixelFormatR16Uint, 2},      {"r32u", MTLPixelFormatR32Uint, 4},
    {"rg32u", MTLPixelFormatRG32Uint, 8},    {"rgba32u", MTLPixelFormatRGBA32Uint, 16},
};

static const struct {
    const char *name;
    MTLTextureUsage usage;
} kUsages[] = {
    {"rt+sr", MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead},
    {"rt+sr+pfv",
     MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView},
    {"sr", MTLTextureUsageShaderRead},
    {"sr+pfv", MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView},
};

static const struct {
    const char *name;
    MTLStorageMode mode;
} kStorages[] = {{"private", MTLStorageModePrivate}, {"shared", MTLStorageModeShared}};

enum { kTrip = 0, kInto = 1, kOut = 2 };

/// One case, as its argument list spells it.
struct spec {
    int format, samples, width, height, usage, storage, noopt, clear, plain, dir, offset, bpr;
};

#define kFields 12
#define kLength ((NSUInteger)4096)

/// A byte unique to `(tag, index)` for any index below 2^24.
static uint8_t patternByte(uint32_t tag, uint32_t index) {
    uint32_t x = index * 2654435761u ^ tag * 40503u;
    x ^= x >> 15;
    x *= 2246822519u;
    x ^= x >> 13;
    return (uint8_t)(x ^ (x >> 24));
}

static id<MTLBuffer> buffer(id<MTLDevice> device, NSUInteger length, int tag) {
    id<MTLBuffer> b = [device newBufferWithLength:length options:MTLResourceStorageModeShared];
    uint8_t *p = b.contents;
    for (NSUInteger i = 0; i < length; i++) {
        p[i] = tag < 0 ? 0xCC : patternByte((uint32_t)tag, (uint32_t)i);
    }
    return b;
}

static bool finish(id<MTLCommandBuffer> cb) {
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) {
        printf(" command-buffer-error(%s)", cb.error.localizedDescription.UTF8String);
        return false;
    }
    return true;
}

static void step(const char *what) {
    printf(" %s", what);
}

static bool copyIn(const struct spec *c, id<MTLCommandQueue> q, id<MTLBuffer> src, id<MTLTexture> t,
                   NSUInteger offset, NSUInteger bpr) {
    step("copy-in");
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    MTLSize size = MTLSizeMake((NSUInteger)c->width, (NSUInteger)c->height, 1);
    if (c->noopt) {
        [blit copyFromBuffer:src
                   sourceOffset:offset
              sourceBytesPerRow:bpr
            sourceBytesPerImage:0
                     sourceSize:size
                      toTexture:t
               destinationSlice:0
               destinationLevel:0
              destinationOrigin:MTLOriginMake(0, 0, 0)];
    } else {
        [blit copyFromBuffer:src
                   sourceOffset:offset
              sourceBytesPerRow:bpr
            sourceBytesPerImage:0
                     sourceSize:size
                      toTexture:t
               destinationSlice:0
               destinationLevel:0
              destinationOrigin:MTLOriginMake(0, 0, 0)
                        options:MTLBlitOptionNone];
    }
    [blit endEncoding];
    return finish(cb);
}

static bool copyOut(const struct spec *c, id<MTLCommandQueue> q, id<MTLTexture> t,
                    id<MTLBuffer> dst, NSUInteger offset, NSUInteger bpr) {
    step("copy-out");
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    MTLSize size = MTLSizeMake((NSUInteger)c->width, (NSUInteger)c->height, 1);
    if (c->noopt) {
        [blit copyFromTexture:t
                         sourceSlice:0
                         sourceLevel:0
                        sourceOrigin:MTLOriginMake(0, 0, 0)
                          sourceSize:size
                            toBuffer:dst
                   destinationOffset:offset
              destinationBytesPerRow:bpr
            destinationBytesPerImage:0];
    } else {
        [blit copyFromTexture:t
                         sourceSlice:0
                         sourceLevel:0
                        sourceOrigin:MTLOriginMake(0, 0, 0)
                          sourceSize:size
                            toBuffer:dst
                   destinationOffset:offset
              destinationBytesPerRow:bpr
            destinationBytesPerImage:0
                             options:MTLBlitOptionNone];
    }
    [blit endEncoding];
    return finish(cb);
}

/// Where `needle`'s `n` bytes lie in `hay`'s `length`, or -1.
static long find(const uint8_t *hay, NSUInteger length, const uint8_t *needle, NSUInteger n) {
    for (NSUInteger i = 0; i + n <= length; i++) {
        if (memcmp(hay + i, needle, n) == 0) {
            return (long)i;
        }
    }
    return -1;
}

static void printRows(const long *rows, int n) {
    printf(" rows");
    for (int y = 0; y < n; y++) {
        if (rows[y] < 0) {
            printf(" ?");
        } else {
            printf(" %ld", rows[y]);
        }
    }
}

static int runCase(const struct spec *c) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q = [device newCommandQueue];
        const struct format *f = &kFormats[c->format];
        NSUInteger pixel = f->bpp * (NSUInteger)c->samples, row = pixel * (NSUInteger)c->width;
        NSUInteger packed = row * (NSUInteger)c->height;
        MTLTextureDescriptor *d = [MTLTextureDescriptor new];
        d.textureType = MTLTextureType2DMultisample;
        d.sampleCount = (NSUInteger)c->samples;
        d.pixelFormat = f->pf;
        d.width = (NSUInteger)c->width;
        d.height = (NSUInteger)c->height;
        d.usage = kUsages[c->usage].usage;
        d.storageMode = kStorages[c->storage].mode;
        d.allowGPUOptimizedContents = !c->plain;
        step("texture");
        id<MTLTexture> t = [device newTextureWithDescriptor:d];
        if (!t) {
            printf(" none\n");
            return 0;
        }
        if (c->clear) {
            step("clear");
            MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
            pass.colorAttachments[0].texture = t;
            pass.colorAttachments[0].loadAction = MTLLoadActionClear;
            pass.colorAttachments[0].storeAction = MTLStoreActionStore;
            pass.colorAttachments[0].clearColor = MTLClearColorMake(0.25, 0.5, 0.75, 1);
            id<MTLCommandBuffer> cb = [q commandBuffer];
            [[cb renderCommandEncoderWithDescriptor:pass] endEncoding];
            if (!finish(cb)) {
                printf("\n");
                return 0;
            }
        }
        if (c->dir == kTrip) {
            id<MTLBuffer> src = buffer(device, packed, 1), dst = buffer(device, packed, -1);
            if (!copyIn(c, q, src, t, 0, row) || !copyOut(c, q, t, dst, 0, row)) {
                printf("\n");
                return 0;
            }
            bool same = memcmp(src.contents, dst.contents, packed) == 0;
            printf(" returned; %s\n", same ? "as written" : "changed");
            return 0;
        }
        long rows[16];
        int n = c->height < 16 ? c->height : 16;
        if (c->dir == kInto) {
            id<MTLBuffer> src = buffer(device, kLength, 1), dst = buffer(device, packed, -1);
            if (!copyIn(c, q, src, t, (NSUInteger)c->offset, (NSUInteger)c->bpr) ||
                !copyOut(c, q, t, dst, 0, row)) {
                printf("\n");
                return 0;
            }
            for (int y = 0; y < n; y++) {
                rows[y] = find(src.contents, kLength, (const uint8_t *)dst.contents + y * row, row);
            }
        } else {
            id<MTLBuffer> src = buffer(device, packed, 1), dst = buffer(device, kLength, -1);
            if (!copyIn(c, q, src, t, 0, row) ||
                !copyOut(c, q, t, dst, (NSUInteger)c->offset, (NSUInteger)c->bpr)) {
                printf("\n");
                return 0;
            }
            for (int y = 0; y < n; y++) {
                rows[y] = find(dst.contents, kLength, (const uint8_t *)src.contents + y * row, row);
            }
        }
        printf(" returned;");
        printRows(rows, n);
        printf("\n");
        return 0;
    }
}

static const char *gSelf;

static void label(const struct spec *c, bool forked) {
    const char *dir = c->dir == kTrip ? "trip" : c->dir == kInto ? "into" : "out";
    printf("%s %s x%d w%d h%d %s %s%s%s%s%s", dir, kFormats[c->format].name, c->samples, c->width,
           c->height, kUsages[c->usage].name, kStorages[c->storage].name, c->plain ? " plain" : "",
           c->noopt ? " noopt" : "", c->clear ? " clear" : "", forked ? " fork" : "");
    if (c->dir != kTrip) {
        printf(" at (%d, %d)", c->offset, c->bpr);
    }
    printf(":");
}

static void report(int status) {
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        printf(" DIED %d\n", status);
    }
}

/// The case in a spawned process of this binary.
static void spawned(const struct spec *c) {
    label(c, false);
    const int *v = &c->format;
    char args[kFields][16];
    char *argv[kFields + 3];
    argv[0] = (char *)gSelf;
    argv[1] = "case";
    for (int i = 0; i < kFields; i++) {
        snprintf(args[i], sizeof args[i], "%d", v[i]);
        argv[i + 2] = args[i];
    }
    argv[kFields + 2] = NULL;
    pid_t pid;
    if (posix_spawn(&pid, gSelf, NULL, NULL, argv, environ) != 0) {
        printf(" no spawn\n");
        return;
    }
    int status = 0;
    waitpid(pid, &status, 0);
    report(status);
}

/// The case in a forked child that does not exec, as `short_row_kinds.m` runs its cases.
static void forked(const struct spec *c) {
    label(c, true);
    pid_t pid = fork();
    if (pid == 0) {
        runCase(c);
        _exit(0);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    report(status);
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    gSelf = argv[0];
    if (argc == kFields + 2 && strcmp(argv[1], "case") == 0) {
        struct spec c;
        int *v = &c.format;
        for (int i = 0; i < kFields; i++) {
            v[i] = atoi(argv[i + 2]);
        }
        return runCase(&c);
    }
    // The trips: the two kinds of `short_row_kinds.m`, every usage, storage, call and clear.
    const int kinds[2][3] = {{0, 4, 8}, {1, 2, 16}};
    for (int k = 0; k < 2; k++) {
        struct spec base = {kinds[k][0], kinds[k][1], kinds[k][2], 8, 0, 0, 0, 0, 0, kTrip, 0, 0};
        forked(&base);
        base.usage = 1;
        forked(&base);
        for (int u = 0; u < 4; u++) {
            for (int s = 0; s < 2; s++) {
                for (int o = 0; o < 2; o++) {
                    for (int cl = 0; cl < (u < 2 ? 2 : 1); cl++) {
                        struct spec c = base;
                        c.usage = u;
                        c.storage = s;
                        c.noopt = o;
                        c.clear = cl;
                        spawned(&c);
                    }
                }
            }
            if (u < 2) {
                struct spec c = base;
                c.usage = u;
                c.plain = 1;
                spawned(&c);
            }
        }
    }
    // The placements: the runtime test's layouts, its format, samples, width, offset and pitch,
    // and the rows it expects.
    static const struct {
        int dir, format, samples, width, offset, bpr;
        long rows[4];
    } kPlaces[] = {
        {kInto, 3, 4, 2, 5, 20, {4, 20, 36, 52}},     {kInto, 4, 2, 5, 28, 143, {28, 156, 284, 412}},
        {kInto, 5, 4, 1, 4, 64, {4, 68, 132, 196}},   {kInto, 5, 2, 1, 7, 32, {6, 38, 70, 102}},
        {kInto, 2, 4, 1, 4, 4, {4, 4, 4, 4}},         {kOut, 4, 2, 4, 7, 64, {4, 68, 132, 196}},
        {kOut, 5, 2, 1, 22, 32, {22, 54, 86, 118}},   {kOut, 5, 4, 1, 21, 64, {20, 84, 148, 212}},
        {kOut, 3, 4, 2, 0, 54, {0, 48, 96, 144}},
    };
    for (size_t i = 0; i < sizeof kPlaces / sizeof kPlaces[0]; i++) {
        for (int u = 0; u < 2; u++) {
            struct spec c = {kPlaces[i].format, kPlaces[i].samples, kPlaces[i].width, 4, u, 0, 0,
                             0, 0, kPlaces[i].dir, kPlaces[i].offset, kPlaces[i].bpr};
            printf("expects");
            printRows(kPlaces[i].rows, 4);
            printf(" | ");
            spawned(&c);
        }
    }
    return 0;
}
