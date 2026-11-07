// HOW MUCH of a sparse tile a MOVE carries, by texture descriptor: is Metal's partial read-back of a
// moved tile a lossless-compression artefact?
//
// `sparse-mapping-move-inside-the-level` and its two siblings read a destination tile a move
// rebound as ZERO except for 128 of its 4,096 words (4 by 2 texel groups at a 16 by 16 stride),
// on an Apple M3 Max under macOS 27.0.1, over a `ShaderRead` `R32Float` sparse texture. That shape
// is the shape of a lossless-compression block's head, so the hypothesis is that the moved page's
// payload stays compressed while the destination texture's compression metadata does not follow
// it. Metal's documentation says only that "the tile mapping is moved from the source texture only
// if the destination texture tile is unmapped" (`MTLResourceStateCommandEncoder.h`), nothing of
// the moved tile's contents.
//
// This probe makes the same move under descriptors that do and do not let Metal compress the
// texture, and counts every word of each moved tile. If the partial read-back is compression's,
// the descriptors that turn compression off read whole tiles.
//
// THE VARIANTS (the texture descriptor of both textures; everything else the same):
//
//   read                     `ShaderRead`, the case's own descriptor.
//   read-unoptimized         `ShaderRead`, `allowGPUOptimizedContents = NO`.
//   read-write               `ShaderRead | ShaderWrite`.
//   read-write-unoptimized   `ShaderRead | ShaderWrite`, `allowGPUOptimizedContents = NO`.
//   read-view                `ShaderRead | PixelFormatView`.
//
// THE PROGRAM, per variant: a sparse heap of 16-KiB pages, two 256 by 256 nine-level `R32Float`
// sparse textures A and B of the variant's descriptor; A's level 0 (4 by 4 tiles) mapped whole and
// each tile filled by a blit with a word of its own (never 0 or 0xFFFFFFFF); B unmapped. Two moves
// in one resource state encoder: A(0,0) onto B(3,3) and A(1,1) onto B(0,0), both wholly inside the
// level. Then every texel of both levels 0 read back by a blit, and for each tile: how many of its
// 4,096 words are its expected word, how many are 0, how many are anything else.
//
// The control is in the same run: A's tiles the moves did not touch must read their own words in
// all 4,096 texels, or the fill or the read-back is what failed.
//
// RUN IT WITHOUT MTL_DEBUG_LAYER, one variant a process, each twice:
//
//   clang -fobjc-arc -fmodules -framework Metal -framework Foundation \
//       -o "$TMPDIR/sparse_move_compression" sparse_move_compression.m
//   for v in read read-unoptimized read-write read-write-unoptimized read-view; do
//     "$TMPDIR/sparse_move_compression" $v; "$TMPDIR/sparse_move_compression" $v; done
//
// Nothing here dispatches, draws or loops on a GPU; every region lies inside its level.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>

static const NSUInteger kSide = 256;
static const NSUInteger kLevels = 9;
static const NSUInteger kHeapBytes = 4u << 20;
static const NSUInteger kTile = 64;
static const NSUInteger kTiles = 4;
static const NSUInteger kTexelBytes = 4;

static id<MTLDevice> gDevice = nil;
static id<MTLCommandQueue> gQueue = nil;
static id<MTLHeap> gHeap = nil;

/// The word A's tile (x, y) is filled with: nonzero, never 0xFFFFFFFF, unique per tile.
static uint32_t WordOf(NSUInteger x, NSUInteger y) {
    return 0x3a100000u | (uint32_t)(y << 8) | (uint32_t)x;
}

static BOOL Run(id<MTLCommandBuffer> cb, const char *what) {
    [cb commit];
    [cb waitUntilCompleted];
    BOOL ok = cb.status == MTLCommandBufferStatusCompleted && cb.error == nil;
    printf("  %-8s status=%lu error=%s\n", what, (unsigned long)cb.status,
           cb.error == nil ? "(none)" : cb.error.localizedDescription.UTF8String);
    return ok;
}

static void FillTile(id<MTLTexture> texture, NSUInteger x, NSUInteger y, uint32_t word) {
    NSUInteger words = kTile * kTile;
    id<MTLBuffer> staging = [gDevice newBufferWithLength:words * kTexelBytes
                                                 options:MTLResourceStorageModeShared];
    uint32_t *at = (uint32_t *)staging.contents;
    for (NSUInteger i = 0; i < words; i += 1) {
        at[i] = word;
    }
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:staging
            sourceOffset:0
       sourceBytesPerRow:kTile * kTexelBytes
     sourceBytesPerImage:words * kTexelBytes
              sourceSize:MTLSizeMake(kTile, kTile, 1)
               toTexture:texture
        destinationSlice:0
        destinationLevel:0
       destinationOrigin:MTLOriginMake(x * kTile, y * kTile, 0)];
    [blit endEncoding];
    Run(cb, "fill");
}

/// Level 0's words, read back by a blit into a guard-filled buffer.
static NSData *ReadLevel(id<MTLTexture> texture) {
    NSUInteger bytes = kSide * kSide * kTexelBytes;
    id<MTLBuffer> into = [gDevice newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    memset(into.contents, 0x5a, bytes);
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromTexture:texture
              sourceSlice:0
              sourceLevel:0
             sourceOrigin:MTLOriginMake(0, 0, 0)
               sourceSize:MTLSizeMake(kSide, kSide, 1)
                 toBuffer:into
        destinationOffset:0
   destinationBytesPerRow:kSide * kTexelBytes
 destinationBytesPerImage:bytes];
    [blit endEncoding];
    Run(cb, "read");
    return [NSData dataWithBytes:into.contents length:bytes];
}

/// Print tile (tx, ty) of `level`'s words as counts against `expected`, 0 and the rest, and where
/// the expected words lie when the tile is partly them (the first eight such texels).
static void Count(NSData *level, const char *name, NSUInteger tx, NSUInteger ty,
                  uint32_t expected) {
    const uint32_t *at = (const uint32_t *)level.bytes;
    NSUInteger same = 0, zero = 0, other = 0;
    uint32_t firstOther = 0;
    NSUInteger shown = 0;
    char where[256] = {0};
    for (NSUInteger y = 0; y < kTile; y += 1) {
        for (NSUInteger x = 0; x < kTile; x += 1) {
            uint32_t w = at[(ty * kTile + y) * kSide + tx * kTile + x];
            if (w == expected) {
                same += 1;
                if (shown < 8) {
                    size_t len = strlen(where);
                    snprintf(where + len, sizeof where - len, " (%lu,%lu)", (unsigned long)x,
                             (unsigned long)y);
                    shown += 1;
                }
            } else if (w == 0) {
                zero += 1;
            } else {
                if (other == 0) {
                    firstOther = w;
                }
                other += 1;
            }
        }
    }
    printf("  %s(%lu,%lu) expect=0x%08x same=%lu zero=%lu other=%lu first_other=0x%08x "
           "same_at=%s\n",
           name, (unsigned long)tx, (unsigned long)ty, expected, (unsigned long)same,
           (unsigned long)zero, (unsigned long)other, firstOther, same == 4096 ? " all" : where);
}

static id<MTLTexture> MakeTexture(MTLTextureUsage usage, BOOL optimized) {
    MTLTextureDescriptor *td = [[MTLTextureDescriptor alloc] init];
    td.textureType = MTLTextureType2D;
    td.pixelFormat = MTLPixelFormatR32Float;
    td.width = kSide;
    td.height = kSide;
    td.mipmapLevelCount = kLevels;
    td.storageMode = MTLStorageModePrivate;
    td.usage = usage;
    td.allowGPUOptimizedContents = optimized;
    printf("  descriptor: format=%lu %lux%lu levels=%lu storage=%lu usage=0x%lx optimized=%d\n",
           (unsigned long)td.pixelFormat, (unsigned long)td.width, (unsigned long)td.height,
           (unsigned long)td.mipmapLevelCount, (unsigned long)td.storageMode,
           (unsigned long)td.usage, (int)td.allowGPUOptimizedContents);
    return [gHeap newTextureWithDescriptor:td];
}

int main(int argc, const char *argv[]) {
    if (argc != 2) {
        fprintf(stderr,
                "usage: %s read|read-unoptimized|read-write|read-write-unoptimized|read-view\n",
                argv[0]);
        return 2;
    }
    const char *v = argv[1];
    MTLTextureUsage usage = MTLTextureUsageShaderRead;
    BOOL optimized = YES;
    if (strcmp(v, "read") == 0) {
    } else if (strcmp(v, "read-unoptimized") == 0) {
        optimized = NO;
    } else if (strcmp(v, "read-write") == 0) {
        usage |= MTLTextureUsageShaderWrite;
    } else if (strcmp(v, "read-write-unoptimized") == 0) {
        usage |= MTLTextureUsageShaderWrite;
        optimized = NO;
    } else if (strcmp(v, "read-view") == 0) {
        usage |= MTLTextureUsagePixelFormatView;
    } else {
        fprintf(stderr, "unknown variant %s\n", v);
        return 2;
    }
    @autoreleasepool {
        gDevice = MTLCreateSystemDefaultDevice();
        if (gDevice == nil) {
            fprintf(stderr, "no Metal device\n");
            return 2;
        }
        printf("variant=%s device=%s class=%s\n", v, gDevice.name.UTF8String,
               object_getClassName(gDevice));
        MTLHeapDescriptor *hd = [[MTLHeapDescriptor alloc] init];
        hd.type = MTLHeapTypeSparse;
        hd.storageMode = MTLStorageModePrivate;
        hd.sparsePageSize = MTLSparsePageSize16;
        hd.size = kHeapBytes;
        gHeap = [gDevice newHeapWithDescriptor:hd];
        gQueue = [gDevice newCommandQueue];
        id<MTLTexture> a = MakeTexture(usage, optimized);
        id<MTLTexture> b = MakeTexture(usage, optimized);
        if (gHeap == nil || gQueue == nil || a == nil || b == nil) {
            fprintf(stderr, "no sparse heap, queue or texture\n");
            return 2;
        }
        MTLSize tile = [gDevice sparseTileSizeWithTextureType:MTLTextureType2D
                                                 pixelFormat:MTLPixelFormatR32Float
                                                 sampleCount:1
                                              sparsePageSize:MTLSparsePageSize16];
        if (tile.width != kTile || tile.height != kTile) {
            fprintf(stderr, "the device's tile is not this probe's grid\n");
            return 2;
        }
        {
            id<MTLCommandBuffer> cb = [gQueue commandBuffer];
            id<MTLResourceStateCommandEncoder> e = [cb resourceStateCommandEncoder];
            [e updateTextureMapping:a
                               mode:MTLSparseTextureMappingModeMap
                             region:MTLRegionMake2D(0, 0, kTiles, kTiles)
                           mipLevel:0
                              slice:0];
            [e endEncoding];
            Run(cb, "map");
        }
        for (NSUInteger y = 0; y < kTiles; y += 1) {
            for (NSUInteger x = 0; x < kTiles; x += 1) {
                FillTile(a, x, y, WordOf(x, y));
            }
        }
        printf("BEFORE usedSize=%lu\n", (unsigned long)gHeap.usedSize);
        NSData *before = ReadLevel(a);
        Count(before, "A", 0, 0, WordOf(0, 0));
        Count(before, "A", 1, 1, WordOf(1, 1));
        {
            id<MTLCommandBuffer> cb = [gQueue commandBuffer];
            id<MTLResourceStateCommandEncoder> e = [cb resourceStateCommandEncoder];
            [e moveTextureMappingsFromTexture:a
                                 sourceSlice:0
                                 sourceLevel:0
                                sourceOrigin:MTLOriginMake(0, 0, 0)
                                  sourceSize:MTLSizeMake(1, 1, 1)
                                   toTexture:b
                            destinationSlice:0
                            destinationLevel:0
                           destinationOrigin:MTLOriginMake(3, 3, 0)];
            [e moveTextureMappingsFromTexture:a
                                 sourceSlice:0
                                 sourceLevel:0
                                sourceOrigin:MTLOriginMake(1, 1, 0)
                                  sourceSize:MTLSizeMake(1, 1, 1)
                                   toTexture:b
                            destinationSlice:0
                            destinationLevel:0
                           destinationOrigin:MTLOriginMake(0, 0, 0)];
            [e endEncoding];
            Run(cb, "move");
        }
        printf("AFTER usedSize=%lu\n", (unsigned long)gHeap.usedSize);
        NSData *movedA = ReadLevel(a);
        NSData *movedB = ReadLevel(b);
        // The moved tiles, on their new side.
        Count(movedB, "B", 3, 3, WordOf(0, 0));
        Count(movedB, "B", 0, 0, WordOf(1, 1));
        // The control: A's tiles the moves did not touch keep their words.
        Count(movedA, "A", 2, 0, WordOf(2, 0));
        Count(movedA, "A", 3, 3, WordOf(3, 3));
        // The moved-from tiles, now unmapped.
        Count(movedA, "A", 0, 0, WordOf(0, 0));
        Count(movedA, "A", 1, 1, WordOf(1, 1));
    }
    return 0;
}
