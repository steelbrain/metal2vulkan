// Where plain Metal lays a one-plane YCbCr texture's texels in a placement heap's bytes, and where
// a blit at the row length `MTLPixelFormatGetInfoForDevice` describes moves them.
//
// Two questions, for the nine one-plane YCbCr formats the runtime holds for copies alone
// (`metal::format::copies_only`): 501 `YCBCR8_422_1P`, 504 `YCBCR10_444_1P`, 521 and 524 their
// sRGB forms, 560 `YCBCRA8_444_1P`, 562 `YCBCR8_422_1P_REV`, 563 `YCBCR10_444_1P_PQ`, 590
// `YCBCR10_444_1P_XR` and 607 `YCBCR10_444_1P_HLG`. Each texel is 2 bytes (the 4:2:2 formats, one
// luma byte and one chroma byte) or 4 (the 4:4:4 ones), as Metal's host copies move them.
//
//   heap  A `Private`, untracked placement heap with a `Private` buffer over all its bytes, a
//         texture at offset 0 (the shape of a consumer's heaps). The buffer is filled with 0xEE;
//         each level and slice of the texture is copied in from a `Shared` buffer of bytes unique
//         to that level and slice (`copyFromBuffer:...toTexture:` at the texture's packed row,
//         width x texel bytes); the heap's buffer is copied out. Printed per level and slice: where
//         its first row's bytes lie, how far its second row and the next slice lie past it, and
//         whether every texel lies at `slice x slicePitch + y x rowPitch + x x texel` (`rows`).
//         Then the heap's buffer is filled with a pattern of its byte offsets and each level and
//         slice copied out of the texture (`copyFromTexture:...toBuffer:`): whether each texel
//         reads the bytes at the same place (`back`). And `heapTextureSizeAndAlignWithDescriptor:`.
//   rows  A 16 x 16 2D texture, `Shared` and `Private`, no heap: a copy in from a buffer at
//         `bytesPerRow` of 16 x 1 and 16 x 2 bytes (the texel sizes `MTLPixelFormatGetInfoForDevice`
//         answers) and 16 x the texel's bytes, then the texture copied out at its packed row. And
//         `replaceRegion:` and `getBytes:` at those rows, for a `Shared` texture. Printed per
//         row length: whether Metal returned (each case in a process of its own: a driver assert
//         ends that process alone, and its exit status is the reading), and which bytes landed.
//
// Build (no GPU needed to build):
//
//   xcrun clang -fobjc-arc -framework Metal -framework Foundation \
//       -o one_plane_ycbcr_heap_rows one_plane_ycbcr_heap_rows.m
//
// Run without MTL_DEBUG_LAYER (the layer is not passive: it asserts on what plain Metal does):
//
//   ./one_plane_ycbcr_heap_rows heap
//   ./one_plane_ycbcr_heap_rows rows
//
// Every line is `<mode> <format> <shape> ...`; a forked case that Metal ends prints `DIED <status>`.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static id<MTLDevice> gDevice = nil;
static id<MTLCommandQueue> gQueue = nil;

static const NSUInteger kFormats[] = {501, 504, 521, 524, 560, 562, 563, 590, 607};

/// The bytes Metal's host copies move for one texel of `format`.
static NSUInteger texelBytes(NSUInteger format) {
    switch (format) {
    case 501:
    case 521:
    case 562:
        return 2;
    default:
        return 4;
    }
}

/// A byte unique to `(tag, index)` for any index below 2^24.
static uint8_t patternByte(uint32_t tag, uint32_t index) {
    uint32_t x = index * 2654435761u ^ tag * 40503u;
    x ^= x >> 15;
    x *= 2246822519u;
    x ^= x >> 13;
    return (uint8_t)(x ^ (x >> 24));
}

static void waitFor(id<MTLCommandBuffer> cb) {
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) {
        printf("    command buffer error %s\n", cb.error.localizedDescription.UTF8String);
    }
}

static id<MTLBuffer> sharedBuffer(NSUInteger length) {
    return [gDevice newBufferWithLength:length options:MTLResourceStorageModeShared];
}

/// The first offset in `hay` (of `hayLength` bytes) at which `needle` lies, or -1.
static long find(const uint8_t *hay, NSUInteger hayLength, const uint8_t *needle,
                 NSUInteger needleLength) {
    if (needleLength == 0 || needleLength > hayLength) {
        return -1;
    }
    void *at = memmem(hay, hayLength, needle, needleLength);
    return at ? (long)((const uint8_t *)at - hay) : -1;
}

struct shape {
    const char *name;
    MTLTextureType type;
    NSUInteger width, height, depth, slices, levels;
};

static const struct shape kShapes[] = {
    {"2d-8x8", MTLTextureType2D, 8, 8, 1, 1, 1},
    {"2d-16x16", MTLTextureType2D, 16, 16, 1, 1, 1},
    {"2d-64x64", MTLTextureType2D, 64, 64, 1, 1, 1},
    {"2d-6x3", MTLTextureType2D, 6, 3, 1, 1, 1},
    {"2d-13x7", MTLTextureType2D, 13, 7, 1, 1, 1},
    {"2d-100x30", MTLTextureType2D, 100, 30, 1, 1, 1},
    {"1d-37", MTLTextureType1D, 37, 1, 1, 1, 1},
    {"1darray-37x3", MTLTextureType1DArray, 37, 1, 1, 3, 1},
    {"2darray-16x16x2", MTLTextureType2DArray, 16, 16, 1, 2, 1},
    {"2darray-13x7x3", MTLTextureType2DArray, 13, 7, 1, 3, 1},
    {"2d-16x16-3levels", MTLTextureType2D, 16, 16, 1, 1, 3},
    {"3d-8x8x3", MTLTextureType3D, 8, 8, 3, 1, 1},
};

static MTLTextureDescriptor *descriptorOf(NSUInteger format, const struct shape *s,
                                          MTLTextureUsage usage, MTLStorageMode mode) {
    MTLTextureDescriptor *d = [MTLTextureDescriptor new];
    d.textureType = s->type;
    d.pixelFormat = (MTLPixelFormat)format;
    d.width = s->width;
    d.height = s->height;
    d.depth = s->depth;
    d.arrayLength = s->slices;
    d.mipmapLevelCount = s->levels;
    d.usage = usage;
    d.storageMode = mode;
    d.hazardTrackingMode = MTLHazardTrackingModeUntracked;
    return d;
}

static NSUInteger levelSide(NSUInteger side, NSUInteger level) {
    NSUInteger v = side >> level;
    return v ? v : 1;
}

/// The heap question for one format, shape and usage.
static void heapCase(NSUInteger format, const struct shape *s, MTLTextureUsage usage) {
    NSUInteger unit = texelBytes(format);
    MTLTextureDescriptor *d = descriptorOf(format, s, usage, MTLStorageModePrivate);
    MTLSizeAndAlign sa = [gDevice heapTextureSizeAndAlignWithDescriptor:d];
    NSUInteger heapBytes = sa.size * 4 + 65536;
    heapBytes = (heapBytes + 16383) & ~(NSUInteger)16383;
    MTLHeapDescriptor *hd = [MTLHeapDescriptor new];
    hd.type = MTLHeapTypePlacement;
    hd.storageMode = MTLStorageModePrivate;
    hd.hazardTrackingMode = MTLHazardTrackingModeUntracked;
    hd.size = heapBytes;
    id<MTLHeap> heap = [gDevice newHeapWithDescriptor:hd];
    const MTLResourceOptions options =
        MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeUntracked;
    id<MTLBuffer> over = [heap newBufferWithLength:heap.size options:options offset:0];
    id<MTLTexture> t = [heap newTextureWithDescriptor:d offset:0];
    printf("heap %lu %s usage=0x%lx answer=%lu align=%lu heap=%lu texture=%s\n",
           (unsigned long)format, s->name, (unsigned long)usage, (unsigned long)sa.size,
           (unsigned long)sa.align, (unsigned long)heap.size, t ? "made" : "nil");
    if (!t || !over) {
        return;
    }
    NSUInteger heapLength = over.length;
    id<MTLBuffer> back = sharedBuffer(heapLength);
    // Fill, then copy every level and slice in from bytes unique to it.
    NSUInteger slices = s->type == MTLTextureType3D ? 1 : s->slices;
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit fillBuffer:over range:NSMakeRange(0, heapLength) value:0xEE];
    [blit endEncoding];
    waitFor(cb);
    NSMutableArray *sources = [NSMutableArray array];
    for (NSUInteger level = 0; level < s->levels; level++) {
        NSUInteger w = levelSide(s->width, level), h = levelSide(s->height, level);
        NSUInteger dd = s->type == MTLTextureType3D ? levelSide(s->depth, level) : 1;
        if (s->type == MTLTextureType1D || s->type == MTLTextureType1DArray) {
            h = 1;
        }
        for (NSUInteger slice = 0; slice < slices; slice++) {
            NSUInteger row = w * unit, image = row * h;
            id<MTLBuffer> src = sharedBuffer(image * dd);
            uint8_t *p = src.contents;
            for (NSUInteger i = 0; i < image * dd; i++) {
                p[i] = patternByte((uint32_t)(level * 64 + slice + 1), (uint32_t)i);
            }
            [sources addObject:src];
            cb = [gQueue commandBuffer];
            blit = [cb blitCommandEncoder];
            [blit copyFromBuffer:src
                       sourceOffset:0
                  sourceBytesPerRow:row
                sourceBytesPerImage:image
                         sourceSize:MTLSizeMake(w, h, dd)
                          toTexture:t
                   destinationSlice:slice
                   destinationLevel:level
                  destinationOrigin:MTLOriginMake(0, 0, 0)];
            [blit endEncoding];
            waitFor(cb);
        }
    }
    cb = [gQueue commandBuffer];
    blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:over sourceOffset:0 toBuffer:back destinationOffset:0 size:heapLength];
    [blit endEncoding];
    waitFor(cb);
    const uint8_t *heapBytesRead = back.contents;
    NSUInteger untouched = 0, untouchedPast = 0;
    for (NSUInteger i = 0; i < heapLength; i++) {
        if (heapBytesRead[i] == 0xEE) {
            if (i < sa.size) {
                untouched++;
            } else {
                untouchedPast++;
            }
        }
    }
    printf("  untouched within the answer %lu of %lu, past it %lu of %lu\n",
           (unsigned long)untouched, (unsigned long)sa.size, (unsigned long)untouchedPast,
           (unsigned long)(heapLength - sa.size));
    NSUInteger k = 0;
    for (NSUInteger level = 0; level < s->levels; level++) {
        NSUInteger w = levelSide(s->width, level), h = levelSide(s->height, level);
        NSUInteger dd = s->type == MTLTextureType3D ? levelSide(s->depth, level) : 1;
        if (s->type == MTLTextureType1D || s->type == MTLTextureType1DArray) {
            h = 1;
        }
        long firstSlice = -1;
        for (NSUInteger slice = 0; slice < slices; slice++) {
            id<MTLBuffer> src = sources[k++];
            const uint8_t *p = src.contents;
            NSUInteger row = w * unit;
            long at = find(heapBytesRead, heapLength, p, row);
            long second = h * dd > 1 ? find(heapBytesRead, heapLength, p + row, row) : -1;
            long pitch = (at >= 0 && second >= 0) ? second - at : -1;
            if (slice == 0) {
                firstSlice = at;
            }
            // Every texel at at + image * z + y * pitch + x * unit, the image h rows of pitch.
            NSUInteger rowsHold = 0, texels = w * h * dd;
            long usedPitch = pitch >= 0 ? pitch : (long)row;
            for (NSUInteger z = 0; z < dd && at >= 0; z++) {
                for (NSUInteger y = 0; y < h; y++) {
                    for (NSUInteger x = 0; x < w; x++) {
                        long off = at + (long)((z * h + y) * (NSUInteger)usedPitch + x * unit);
                        if (off >= 0 && (NSUInteger)off + unit <= heapLength &&
                            memcmp(heapBytesRead + off, p + (z * h + y) * row + x * unit, unit) ==
                                0) {
                            rowsHold++;
                        }
                    }
                }
            }
            printf("  level %lu slice %lu (%lux%lux%lu): first row at %ld, row pitch %ld, "
                   "from slice 0 %ld; rows %lu of %lu texels\n",
                   (unsigned long)level, (unsigned long)slice, (unsigned long)w,
                   (unsigned long)h, (unsigned long)dd, at, pitch,
                   (at >= 0 && firstSlice >= 0) ? at - firstSlice : -1, (unsigned long)rowsHold,
                   (unsigned long)texels);
        }
    }
    // Back: the heap's bytes a pattern of their offsets, each level and slice copied out.
    id<MTLBuffer> offsets = sharedBuffer(heapLength);
    uint8_t *o = offsets.contents;
    for (NSUInteger i = 0; i < heapLength; i++) {
        o[i] = patternByte(0x7777, (uint32_t)i);
    }
    cb = [gQueue commandBuffer];
    blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:offsets sourceOffset:0 toBuffer:over destinationOffset:0 size:heapLength];
    [blit endEncoding];
    waitFor(cb);
    for (NSUInteger level = 0; level < s->levels; level++) {
        NSUInteger w = levelSide(s->width, level), h = levelSide(s->height, level);
        NSUInteger dd = s->type == MTLTextureType3D ? levelSide(s->depth, level) : 1;
        if (s->type == MTLTextureType1D || s->type == MTLTextureType1DArray) {
            h = 1;
        }
        for (NSUInteger slice = 0; slice < slices; slice++) {
            NSUInteger row = w * unit, image = row * h;
            id<MTLBuffer> out = sharedBuffer(image * dd);
            cb = [gQueue commandBuffer];
            blit = [cb blitCommandEncoder];
            [blit copyFromTexture:t
                             sourceSlice:slice
                             sourceLevel:level
                            sourceOrigin:MTLOriginMake(0, 0, 0)
                              sourceSize:MTLSizeMake(w, h, dd)
                                toBuffer:out
                       destinationOffset:0
                  destinationBytesPerRow:row
                destinationBytesPerImage:image];
            [blit endEncoding];
            waitFor(cb);
            const uint8_t *q = out.contents;
            // Where its first texel's bytes lie in the offsets, and whether every texel reads
            // the bytes at the packed rows from there.
            long at = find(o, heapLength, q, row);
            NSUInteger hold = 0;
            for (NSUInteger i = 0; i < image * dd && at >= 0; i++) {
                if ((NSUInteger)at + i < heapLength && q[i] == o[at + i]) {
                    hold++;
                }
            }
            printf("  back level %lu slice %lu: reads from %ld, packed rows %lu of %lu bytes\n",
                   (unsigned long)level, (unsigned long)slice, at, (unsigned long)hold,
                   (unsigned long)(image * dd));
        }
    }
}

static void heapMode(void) {
    for (NSUInteger f = 0; f < sizeof kFormats / sizeof kFormats[0]; f++) {
        for (NSUInteger i = 0; i < sizeof kShapes / sizeof kShapes[0]; i++) {
            @autoreleasepool {
                heapCase(kFormats[f], &kShapes[i], MTLTextureUsageShaderRead);
            }
        }
        @autoreleasepool {
            heapCase(kFormats[f], &kShapes[1],
                     MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView);
            heapCase(kFormats[f], &kShapes[2], MTLTextureUsageUnknown);
        }
        fflush(stdout);
    }
}

/// One row-length case of `rows`, in a process of its own: a copy in at `bytesPerRow`, by a blit
/// (`cpu` false) or `replaceRegion:` (`cpu` true), then the texture read back at its packed row.
static void rowsCase(NSUInteger format, MTLStorageMode mode, bool cpu, NSUInteger bytesPerRow) {
    const NSUInteger side = 16, unit = texelBytes(format), packed = side * unit;
    MTLTextureDescriptor *d = [MTLTextureDescriptor new];
    d.textureType = MTLTextureType2D;
    d.pixelFormat = (MTLPixelFormat)format;
    d.width = side;
    d.height = side;
    d.usage = MTLTextureUsageShaderRead;
    d.storageMode = mode;
    id<MTLTexture> t = [gDevice newTextureWithDescriptor:d];
    if (!t) {
        printf("rows %lu %s %s bpr=%lu: no texture\n", (unsigned long)format,
               mode == MTLStorageModeShared ? "shared" : "private", cpu ? "cpu" : "blit",
               (unsigned long)bytesPerRow);
        return;
    }
    // Zero it first, at the packed row.
    id<MTLBuffer> zeros = sharedBuffer(packed * side);
    memset(zeros.contents, 0, packed * side);
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:zeros
               sourceOffset:0
          sourceBytesPerRow:packed
        sourceBytesPerImage:packed * side
                 sourceSize:MTLSizeMake(side, side, 1)
                  toTexture:t
           destinationSlice:0
           destinationLevel:0
          destinationOrigin:MTLOriginMake(0, 0, 0)];
    [blit endEncoding];
    waitFor(cb);
    NSUInteger length = bytesPerRow * side;
    id<MTLBuffer> src = sharedBuffer(length);
    uint8_t *p = src.contents;
    for (NSUInteger i = 0; i < length; i++) {
        p[i] = (uint8_t)(i & 0xff) ^ (uint8_t)(i >> 8);
    }
    printf("rows %lu %s %s bpr=%lu: copying\n", (unsigned long)format,
           mode == MTLStorageModeShared ? "shared" : "private", cpu ? "cpu" : "blit",
           (unsigned long)bytesPerRow);
    fflush(stdout);
    if (cpu) {
        [t replaceRegion:MTLRegionMake2D(0, 0, side, side)
             mipmapLevel:0
               withBytes:p
             bytesPerRow:bytesPerRow];
    } else {
        cb = [gQueue commandBuffer];
        blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:src
                   sourceOffset:0
              sourceBytesPerRow:bytesPerRow
            sourceBytesPerImage:length
                     sourceSize:MTLSizeMake(side, side, 1)
                      toTexture:t
               destinationSlice:0
               destinationLevel:0
              destinationOrigin:MTLOriginMake(0, 0, 0)];
        [blit endEncoding];
        waitFor(cb);
    }
    id<MTLBuffer> out = sharedBuffer(packed * side);
    cb = [gQueue commandBuffer];
    blit = [cb blitCommandEncoder];
    [blit copyFromTexture:t
                     sourceSlice:0
                     sourceLevel:0
                    sourceOrigin:MTLOriginMake(0, 0, 0)
                      sourceSize:MTLSizeMake(side, side, 1)
                        toBuffer:out
               destinationOffset:0
          destinationBytesPerRow:packed
        destinationBytesPerImage:packed * side];
    [blit endEncoding];
    waitFor(cb);
    const uint8_t *q = out.contents;
    // Each packed row's bytes: equal to the source row at bytesPerRow (its first `packed` bytes
    // where the row holds that many), to the source's bytes packed, or neither.
    NSUInteger asRows = 0, asPacked = 0, zero = 0;
    for (NSUInteger y = 0; y < side; y++) {
        NSUInteger n = packed < bytesPerRow ? packed : bytesPerRow;
        if (memcmp(q + y * packed, p + y * bytesPerRow, n) == 0) {
            asRows++;
        }
        if ((y + 1) * packed <= length && memcmp(q + y * packed, p + y * packed, packed) == 0) {
            asPacked++;
        }
        bool allZero = true;
        for (NSUInteger x = 0; x < packed; x++) {
            allZero = allZero && q[y * packed + x] == 0;
        }
        zero += allZero;
    }
    printf("rows %lu %s %s bpr=%lu: returned; rows read at bytesPerRow %lu, packed %lu, zero %lu "
           "of %lu; row 0 ",
           (unsigned long)format, mode == MTLStorageModeShared ? "shared" : "private",
           cpu ? "cpu" : "blit", (unsigned long)bytesPerRow, (unsigned long)asRows,
           (unsigned long)asPacked, (unsigned long)zero, (unsigned long)side);
    for (NSUInteger x = 0; x < packed; x++) {
        printf("%02x", q[x]);
    }
    printf(" row 1 ");
    for (NSUInteger x = 0; x < packed; x++) {
        printf("%02x", q[packed + x]);
    }
    printf("\n");
    if (cpu) {
        // getBytes at the same row length, into a buffer of 0xCC.
        NSMutableData *got = [NSMutableData dataWithLength:length];
        memset(got.mutableBytes, 0xCC, length);
        [t getBytes:got.mutableBytes
            bytesPerRow:bytesPerRow
             fromRegion:MTLRegionMake2D(0, 0, side, side)
            mipmapLevel:0];
        const uint8_t *g = got.bytes;
        NSUInteger same = 0;
        for (NSUInteger i = 0; i < length; i++) {
            same += g[i] == p[i];
        }
        printf("rows %lu shared getBytes bpr=%lu: %lu of %lu bytes as written; row 0 ",
               (unsigned long)format, (unsigned long)bytesPerRow, (unsigned long)same,
               (unsigned long)length);
        for (NSUInteger x = 0; x < bytesPerRow; x++) {
            printf("%02x", g[x]);
        }
        printf("\n");
    }
    fflush(stdout);
}

static void rowsMode(void) {
    for (NSUInteger f = 0; f < sizeof kFormats / sizeof kFormats[0]; f++) {
        NSUInteger format = kFormats[f];
        NSUInteger lengths[] = {16, 32, 16 * texelBytes(format)};
        for (int m = 0; m < 2; m++) {
            MTLStorageMode mode = m ? MTLStorageModePrivate : MTLStorageModeShared;
            for (int cpu = 0; cpu < 2; cpu++) {
                if (cpu && mode == MTLStorageModePrivate) {
                    continue;
                }
                for (int l = 0; l < 3; l++) {
                    if (l == 2 && (lengths[2] == lengths[0] || lengths[2] == lengths[1])) {
                        continue;
                    }
                    fflush(stdout);
                    pid_t pid = fork();
                    if (pid == 0) {
                        @autoreleasepool {
                            gDevice = MTLCreateSystemDefaultDevice();
                            gQueue = [gDevice newCommandQueue];
                            rowsCase(format, mode, cpu, lengths[l]);
                        }
                        _exit(0);
                    }
                    int status = 0;
                    waitpid(pid, &status, 0);
                    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
                        printf("rows %lu %s %s bpr=%lu: DIED %d\n", (unsigned long)format,
                               mode == MTLStorageModeShared ? "shared" : "private",
                               cpu ? "cpu" : "blit", (unsigned long)lengths[l], status);
                    }
                }
            }
        }
    }
}

int main(int argc, const char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s heap|rows\n", argv[0]);
        return 2;
    }
    if (strcmp(argv[1], "heap") == 0) {
        gDevice = MTLCreateSystemDefaultDevice();
        gQueue = [gDevice newCommandQueue];
        printf("DEVICE %s / RELEASE %s\n", gDevice.name.UTF8String,
               [NSProcessInfo processInfo].operatingSystemVersionString.UTF8String);
        heapMode();
    } else if (strcmp(argv[1], "rows") == 0) {
        rowsMode();
    } else {
        return 2;
    }
    return 0;
}
