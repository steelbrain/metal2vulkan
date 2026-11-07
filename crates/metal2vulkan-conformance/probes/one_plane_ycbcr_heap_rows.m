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
//   map   As `heap`, each texel's index written into it and decoded back from the heap's bytes,
//         then the heap's bytes a pattern of their unit offsets read back out: per row (up to 8
//         rows of each slice), the offsets its texels landed at (`w`) and were read from (`r`),
//         as runs, `-1` for a texel found nowhere. `map <format>` runs one format.
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
//   ./one_plane_ycbcr_heap_rows map
//   ./one_plane_ycbcr_heap_rows rows
//
// WHAT IT MEASURED, Apple M3 Max, macOS 27.0.1 (26A434), plain Metal, each of the nine formats
// alike (4:2:2 at 2 bytes a texel, 4:4:4 at 4):
//
// - A 2D texture whose row is a multiple of 16 bytes (8 x 8, 16 x 16, 64 x 64; 100 x 30 at
//   4:4:4) lies one packed row after another from the texture's offset, every texel there both
//   ways; a 1D texture's one row of any length (37 texels) alike; a 1D or 2D array's slices each
//   at a multiple of 128 bytes (37 x 3: 128 or 256 apart; 16 x 16 x 2: 512 or 1024). Each answer
//   is the packed rows (an array's slices rounded up to 128), at 16, 128 for a 2D array.
// - A 2D texture of more than one row whose row is not a multiple of 16 bytes: the copy in lays
//   the rows the row rounded down to 16 bytes apart (6 x 3 at 4:2:2, 12-byte rows: 16 apart, past
//   the 36-byte answer; 13 x 7, 26-byte rows: 16 apart, every odd row's shared bytes the even
//   rows'), and the copy out reads them there, but a row under 16 bytes past the first reads
//   none of them back.
// - A mipmapped texture (16 x 16, 3 levels) and a 3D one (8 x 8 x 3) are laid out no one way:
//   their first level is found nowhere in the heap's bytes by the copy in, and read back from
//   past the answer (1536 bytes in for 501's 512-byte answer) by the copy out.
// - `rows`: a copy in at a `bytesPerRow` below the texel's bytes returns, each row read whole from
//   its own row's offset in the source, so the rows overlap there; a copy out at one writes each
//   row whole at its row's offset, the later row's bytes standing; `replaceRegion:` and
//   `getBytes:` at one move nothing ("AGX: Texture read/write assertion failed: bytes_per_row >=
//   used_bytes_per_row"). `Shared` and `Private` alike.
//
// Every line is `<mode> <format> <shape> ...`; a forked case that Metal ends prints `DIED <status>`.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
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

/// Print the offsets `at[0..n)` of a row's texels as runs of texels `unit` bytes apart.
static void printRuns(const long *at, NSUInteger n, NSUInteger unit) {
    NSUInteger i = 0;
    while (i < n) {
        NSUInteger j = i;
        while (j + 1 < n && at[j] >= 0 && at[j + 1] == at[j] + (long)unit) {
            j++;
        }
        printf(" %ld+%lu", at[i], (unsigned long)(j - i + 1));
        i = j + 1;
    }
    printf("\n");
}

/// The map question for one format and shape: each texel's index written into the texture (2 or
/// 4 bytes, little-endian), the heap's bytes read and each unit decoded back to the texel it
/// holds; then the heap's bytes a pattern of their unit offsets, the texture read out, and each
/// texel's value its offset. Printed per row (up to 8 rows, each slice): the offsets of its texels
/// as runs, written and read.
static void mapCase(NSUInteger format, const struct shape *s) {
    NSUInteger unit = texelBytes(format);
    MTLTextureDescriptor *d = descriptorOf(format, s, MTLTextureUsageShaderRead,
                                           MTLStorageModePrivate);
    MTLSizeAndAlign sa = [gDevice heapTextureSizeAndAlignWithDescriptor:d];
    NSUInteger heapBytes = (sa.size * 4 + 65536 + 16383) & ~(NSUInteger)16383;
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
    NSUInteger h = (s->type == MTLTextureType1D || s->type == MTLTextureType1DArray) ? 1 : s->height;
    NSUInteger w = s->width, slices = s->slices, row = w * unit, image = row * h;
    printf("map %lu %s answer=%lu align=%lu texture=%s\n", (unsigned long)format, s->name,
           (unsigned long)sa.size, (unsigned long)sa.align, t ? "made" : "nil");
    if (!t || !over) {
        return;
    }
    NSUInteger heapLength = over.length;
    // Write: texel k of slice z holds k + 1 + z * w * h (0 is the fill).
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit fillBuffer:over range:NSMakeRange(0, heapLength) value:0];
    [blit endEncoding];
    waitFor(cb);
    for (NSUInteger z = 0; z < slices; z++) {
        id<MTLBuffer> src = sharedBuffer(image);
        uint8_t *p = src.contents;
        for (NSUInteger k = 0; k < w * h; k++) {
            uint32_t v = (uint32_t)(k + 1 + z * w * h);
            memcpy(p + k * unit, &v, unit);
        }
        cb = [gQueue commandBuffer];
        blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:src
                   sourceOffset:0
              sourceBytesPerRow:row
            sourceBytesPerImage:image
                     sourceSize:MTLSizeMake(w, h, 1)
                      toTexture:t
               destinationSlice:z
               destinationLevel:0
              destinationOrigin:MTLOriginMake(0, 0, 0)];
        [blit endEncoding];
        waitFor(cb);
    }
    id<MTLBuffer> back = sharedBuffer(heapLength);
    cb = [gQueue commandBuffer];
    blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:over sourceOffset:0 toBuffer:back destinationOffset:0 size:heapLength];
    [blit endEncoding];
    waitFor(cb);
    NSUInteger count = w * h * slices;
    long *where = malloc(sizeof(long) * count);
    NSUInteger *seen = calloc(count, sizeof(NSUInteger));
    for (NSUInteger k = 0; k < count; k++) {
        where[k] = -1;
    }
    const uint8_t *b = back.contents;
    NSUInteger stray = 0, last = 0;
    for (NSUInteger off = 0; off + unit <= heapLength; off += unit) {
        uint32_t v = 0;
        memcpy(&v, b + off, unit);
        if (v == 0) {
            continue;
        }
        if (v > count) {
            stray++;
            continue;
        }
        if (where[v - 1] < 0) {
            where[v - 1] = (long)off;
        }
        seen[v - 1]++;
        last = off + unit;
    }
    NSUInteger missing = 0, twice = 0;
    for (NSUInteger k = 0; k < count; k++) {
        missing += seen[k] == 0;
        twice += seen[k] > 1;
    }
    printf("  written: last byte %lu, missing %lu, twice %lu, stray units %lu\n",
           (unsigned long)last, (unsigned long)missing, (unsigned long)twice,
           (unsigned long)stray);
    for (NSUInteger z = 0; z < slices; z++) {
        for (NSUInteger y = 0; y < h && y < 8; y++) {
            printf("  w z%lu y%lu:", (unsigned long)z, (unsigned long)y);
            printRuns(where + z * w * h + y * w, w, unit);
        }
    }
    // Read: the heap's units hold their unit offsets + 1.
    id<MTLBuffer> offsets = sharedBuffer(heapLength);
    uint8_t *o = offsets.contents;
    for (NSUInteger off = 0; off + unit <= heapLength; off += unit) {
        uint32_t v = (uint32_t)(off / unit + 1);
        memcpy(o + off, &v, unit);
    }
    cb = [gQueue commandBuffer];
    blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:offsets sourceOffset:0 toBuffer:over destinationOffset:0 size:heapLength];
    [blit endEncoding];
    waitFor(cb);
    for (NSUInteger z = 0; z < slices; z++) {
        id<MTLBuffer> out = sharedBuffer(image);
        cb = [gQueue commandBuffer];
        blit = [cb blitCommandEncoder];
        [blit copyFromTexture:t
                         sourceSlice:z
                         sourceLevel:0
                        sourceOrigin:MTLOriginMake(0, 0, 0)
                          sourceSize:MTLSizeMake(w, h, 1)
                            toBuffer:out
                   destinationOffset:0
              destinationBytesPerRow:row
            destinationBytesPerImage:image];
        [blit endEncoding];
        waitFor(cb);
        const uint8_t *q = out.contents;
        for (NSUInteger y = 0; y < h && y < 8; y++) {
            long at[4096];
            for (NSUInteger x = 0; x < w && x < 4096; x++) {
                uint32_t v = 0;
                memcpy(&v, q + y * row + x * unit, unit);
                at[x] = v ? (long)(v - 1) * (long)unit : -1;
            }
            printf("  r z%lu y%lu:", (unsigned long)z, (unsigned long)y);
            printRuns(at, w < 4096 ? w : 4096, unit);
        }
    }
    free(where);
    free(seen);
    fflush(stdout);
}

static const struct shape kMapShapes[] = {
    {"2d-1x3", MTLTextureType2D, 1, 3, 1, 1, 1},
    {"2d-2x3", MTLTextureType2D, 2, 3, 1, 1, 1},
    {"2d-3x3", MTLTextureType2D, 3, 3, 1, 1, 1},
    {"2d-4x3", MTLTextureType2D, 4, 3, 1, 1, 1},
    {"2d-5x3", MTLTextureType2D, 5, 3, 1, 1, 1},
    {"2d-6x3", MTLTextureType2D, 6, 3, 1, 1, 1},
    {"2d-7x3", MTLTextureType2D, 7, 3, 1, 1, 1},
    {"2d-10x3", MTLTextureType2D, 10, 3, 1, 1, 1},
    {"2d-12x3", MTLTextureType2D, 12, 3, 1, 1, 1},
    {"2d-13x7", MTLTextureType2D, 13, 7, 1, 1, 1},
    {"2d-20x3", MTLTextureType2D, 20, 3, 1, 1, 1},
    {"2d-24x3", MTLTextureType2D, 24, 3, 1, 1, 1},
    {"2d-33x3", MTLTextureType2D, 33, 3, 1, 1, 1},
    {"2d-100x30", MTLTextureType2D, 100, 30, 1, 1, 1},
    {"2darray-13x7x3", MTLTextureType2DArray, 13, 7, 1, 3, 1},
    {"2darray-6x3x2", MTLTextureType2DArray, 6, 3, 1, 2, 1},
};

static void mapMode(int argc, const char **argv) {
    for (NSUInteger f = 0; f < sizeof kFormats / sizeof kFormats[0]; f++) {
        NSUInteger format = kFormats[f];
        if (argc > 2 && (NSUInteger)atoi(argv[2]) != format) {
            continue;
        }
        for (NSUInteger i = 0; i < sizeof kMapShapes / sizeof kMapShapes[0]; i++) {
            @autoreleasepool {
                mapCase(format, &kMapShapes[i]);
            }
        }
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
    // The source holds every row a copy reads from y x bytesPerRow: the last row whole.
    NSUInteger length = bytesPerRow * (side - 1) + packed;
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
    if (!cpu) {
        // The texture's packed rows (`q`) copied out at the same row length into a buffer of
        // 0xCC: where each row's bytes land.
        id<MTLBuffer> outRows = sharedBuffer(bytesPerRow * side + packed);
        memset(outRows.contents, 0xCC, outRows.length);
        cb = [gQueue commandBuffer];
        blit = [cb blitCommandEncoder];
        [blit copyFromTexture:t
                         sourceSlice:0
                         sourceLevel:0
                        sourceOrigin:MTLOriginMake(0, 0, 0)
                          sourceSize:MTLSizeMake(side, side, 1)
                            toBuffer:outRows
                   destinationOffset:0
              destinationBytesPerRow:bytesPerRow
            destinationBytesPerImage:bytesPerRow * side];
        [blit endEncoding];
        waitFor(cb);
        const uint8_t *r = outRows.contents;
        // Row y's bytes whole at y x bytesPerRow, the last row's last, and the bytes past it.
        NSUInteger whole = 0, untouched = 0;
        for (NSUInteger y = 0; y < side; y++) {
            whole += memcmp(r + y * bytesPerRow, q + y * packed, packed) == 0;
        }
        for (NSUInteger i = 0; i < outRows.length; i++) {
            untouched += r[i] == 0xCC;
        }
        printf("rows %lu %s out bpr=%lu: rows whole at y x bytesPerRow %lu of %lu, last row "
               "there %s, bytes untouched %lu of %lu\n",
               (unsigned long)format, mode == MTLStorageModeShared ? "shared" : "private",
               (unsigned long)bytesPerRow, (unsigned long)whole, (unsigned long)side,
               memcmp(r + (side - 1) * bytesPerRow, q + (side - 1) * packed, packed) == 0 ? "yes"
                                                                                       : "no",
               (unsigned long)untouched, (unsigned long)outRows.length);
    }
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
        NSUInteger lengths[] = {16, 32, 16 * texelBytes(format), 8};
        for (int m = 0; m < 2; m++) {
            MTLStorageMode mode = m ? MTLStorageModePrivate : MTLStorageModeShared;
            for (int cpu = 0; cpu < 2; cpu++) {
                if (cpu && mode == MTLStorageModePrivate) {
                    continue;
                }
                for (int l = 0; l < 4; l++) {
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
    } else if (strcmp(argv[1], "map") == 0) {
        gDevice = MTLCreateSystemDefaultDevice();
        gQueue = [gDevice newCommandQueue];
        mapMode(argc, argv);
    } else if (strcmp(argv[1], "rows") == 0) {
        rowsMode();
    } else {
        return 2;
    }
    return 0;
}
