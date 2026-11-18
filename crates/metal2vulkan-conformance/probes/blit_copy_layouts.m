// Plain Metal's bytes for the new runtime's blit copy tests (metal2vulkan-rt, tests::blit): the
// same textures, buffers, byte patterns and copies, each destination read back whole and printed
// as its length and FNV-1a 64 hash, which the tests hold their own results to.
//
// A pattern of `len` bytes from `seed` is byte i = (uint8_t)i * 31 + seed, as the tests make it.
// Every texture is Shared, written and read with replaceRegion and getBytes, rows tightly packed.
//
//   t2t-array   RGBA8Unorm 13x7 (3 levels), level 1 (pattern 1) region (1,0) 4x3 into slice 2
//               (pattern 4) of an RGBA8Unorm 8x8 2D array of 3 at (2,4)
//   t2t-volume  face 4 of an R8Unorm 4x4 cube (pattern 2) into an R8Unorm 4x4x3 3D texture
//               (pattern 5) at (0,0,1)
//   t2t-plane   that 3D texture's (1,1,1) 2x2x1 into an R8Unorm 4x4 2D texture (pattern 6) at (2,0)
//   t2t-slices  slice 0 (pattern 3) (0,0) 5x5 of the array into its slice 1 (pattern 7) at (3,3)
//   t2t-bc1     BC1RGBA 8x8 (pattern 8) (4,0) 4x4 into another (pattern 9) at (0,4)
//
// Copies between a Shared buffer and a texture, in one command buffer, each buffer `len` bytes of
// pattern `seed`:
//
//   b2t-level   4x2 texels from a buffer (256, 11) at 16, rows 32 apart, into level 1 (pattern 12)
//               of an RGBA8Unorm 13x7 texture of 3 levels at (1,1); the level read back
//   t2b-slice   slice 2 (pattern 13) (2,3) 5x4 of an RGBA8Unorm 8x8 2D array of 3 into a buffer
//               (1024, 14) at 64, rows 48 apart; the buffer read back
//   t2b-volume  (1,0,0) 3x4x3 of an R8Unorm 4x4x3 3D texture (pattern 15) into a buffer (128, 16)
//               at 8, rows 5 and images 24 apart; the buffer read back
//   b2t-line    8 texels from a buffer (256, 18) at 16, rows 32 and images 40 apart, into slice 1
//               (pattern 19) of an R32Float 1D array 16 wide of 2 at 3; the slice read back
//   b2t-bc1     4x4 texels from a buffer (64, 17) at 8, rows 16 apart, into a BC1RGBA 8x8
//               texture (pattern 20) at (4,4); the texture read back
//
// Build, and run plain and under Metal's validation layer, which stops the process at a copy it
// does not take (`blit_copy_format_pairs.m` asks which):
//
//   xcrun clang -fobjc-arc -framework Metal -framework Foundation \
//       -o blit_copy_layouts blit_copy_layouts.m
//   ./blit_copy_layouts
//   MTL_DEBUG_LAYER=1 ./blit_copy_layouts
//
// WHAT IT MEASURED (Apple M3 Max, macOS 27.0.1, job m3-max-43464; plain and under the
// validation layer alike):
//
//   t2t-array len 256 fnv fbcf35387a974105
//   t2t-volume len 48 fnv f8143509403cef55
//   t2t-plane len 16 fnv 52ff18149f17740d
//   t2t-slices len 256 fnv 3735fdb19af01a15
//   t2t-bc1 len 32 fnv c00d3bc22f3d14c5

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static NSData *pattern(NSUInteger len, uint8_t seed) {
    NSMutableData *data = [NSMutableData dataWithLength:len];
    uint8_t *bytes = data.mutableBytes;
    for (NSUInteger i = 0; i < len; i++) {
        bytes[i] = (uint8_t)((uint8_t)i * 31 + seed);
    }
    return data;
}

static uint64_t fnv(const uint8_t *bytes, NSUInteger len) {
    uint64_t hash = 0xcbf29ce484222325ull;
    for (NSUInteger i = 0; i < len; i++) {
        hash ^= bytes[i];
        hash *= 0x100000001b3ull;
    }
    return hash;
}

// One level and slice of a texture, whole, in elements of `unit` bytes covering `block` texels.
typedef struct {
    __unsafe_unretained id<MTLTexture> texture;
    NSUInteger slice, level, unit, block;
} Level;

static MTLSize extent(Level l) {
    id<MTLTexture> t = l.texture;
    return MTLSizeMake(MAX(t.width >> l.level, 1u), MAX(t.height >> l.level, 1u),
                       MAX(t.depth >> l.level, 1u));
}

static NSUInteger rowBytes(Level l) {
    return (extent(l).width + l.block - 1) / l.block * l.unit;
}

static NSUInteger rows(Level l) {
    return (extent(l).height + l.block - 1) / l.block;
}

static NSUInteger levelBytes(Level l) {
    return rowBytes(l) * rows(l) * extent(l).depth;
}

static void fillLevel(Level l, uint8_t seed) {
    MTLSize e = extent(l);
    NSData *bytes = pattern(levelBytes(l), seed);
    [l.texture replaceRegion:MTLRegionMake3D(0, 0, 0, e.width, e.height, e.depth)
                 mipmapLevel:l.level
                       slice:l.slice
                   withBytes:bytes.bytes
                 bytesPerRow:rowBytes(l)
               bytesPerImage:rowBytes(l) * rows(l)];
}

static void report(const char *name, Level l) {
    MTLSize e = extent(l);
    NSMutableData *bytes = [NSMutableData dataWithLength:levelBytes(l)];
    [l.texture getBytes:bytes.mutableBytes
            bytesPerRow:rowBytes(l)
          bytesPerImage:rowBytes(l) * rows(l)
             fromRegion:MTLRegionMake3D(0, 0, 0, e.width, e.height, e.depth)
            mipmapLevel:l.level
                  slice:l.slice];
    printf("%s len %lu fnv %016llx\n", name, (unsigned long)bytes.length,
           (unsigned long long)fnv(bytes.bytes, bytes.length));
}

static id<MTLBuffer> patterned(id<MTLDevice> device, NSUInteger len, uint8_t seed) {
    NSData *bytes = pattern(len, seed);
    return [device newBufferWithBytes:bytes.bytes length:len options:MTLResourceStorageModeShared];
}

static void reportBuffer(const char *name, id<MTLBuffer> buffer) {
    printf("%s len %lu fnv %016llx\n", name, (unsigned long)buffer.length,
           (unsigned long long)fnv(buffer.contents, buffer.length));
}

static id<MTLTexture> texture(id<MTLDevice> device, MTLTextureType type, MTLPixelFormat format,
                              NSUInteger w, NSUInteger h, NSUInteger d, NSUInteger levels,
                              NSUInteger array) {
    MTLTextureDescriptor *desc = [MTLTextureDescriptor new];
    desc.textureType = type;
    desc.pixelFormat = format;
    desc.width = w;
    desc.height = h;
    desc.depth = d;
    desc.mipmapLevelCount = levels;
    desc.arrayLength = array;
    desc.usage = MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModeShared;
    return [device newTextureWithDescriptor:desc];
}

static void copy(id<MTLBlitCommandEncoder> blit, Level from, MTLOrigin at, MTLSize size, Level to,
                 MTLOrigin toAt) {
    [blit copyFromTexture:from.texture
              sourceSlice:from.slice
              sourceLevel:from.level
             sourceOrigin:at
               sourceSize:size
                toTexture:to.texture
         destinationSlice:to.slice
         destinationLevel:to.level
        destinationOrigin:toAt];
}

static void run(id<MTLCommandQueue> queue, void (^encode)(id<MTLBlitCommandEncoder>)) {
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    encode(blit);
    [blit endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) {
        printf("command buffer error: %s\n", cb.error.localizedDescription.UTF8String);
    }
}

int main(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        printf("device %s, %s\n", device.name.UTF8String,
               NSProcessInfo.processInfo.operatingSystemVersionString.UTF8String);
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLTexture> rgba = texture(device, MTLTextureType2D, MTLPixelFormatRGBA8Unorm, 13, 7, 1, 3, 1);
        id<MTLTexture> words = texture(device, MTLTextureType2DArray, MTLPixelFormatRGBA8Unorm, 8, 8, 1, 1, 3);
        id<MTLTexture> cube = texture(device, MTLTextureTypeCube, MTLPixelFormatR8Unorm, 4, 4, 1, 1, 1);
        id<MTLTexture> volume = texture(device, MTLTextureType3D, MTLPixelFormatR8Unorm, 4, 4, 3, 1, 1);
        id<MTLTexture> plane = texture(device, MTLTextureType2D, MTLPixelFormatR8Unorm, 4, 4, 1, 1, 1);
        Level rgba1 = {rgba, 0, 1, 4, 1}, words2 = {words, 2, 0, 4, 1};
        Level words0 = {words, 0, 0, 4, 1}, words1 = {words, 1, 0, 4, 1};
        Level face = {cube, 4, 0, 1, 1}, image = {volume, 0, 0, 1, 1}, flat = {plane, 0, 0, 1, 1};
        fillLevel(rgba1, 1);
        fillLevel(face, 2);
        fillLevel(words0, 3);
        fillLevel(words2, 4);
        fillLevel(image, 5);
        fillLevel(flat, 6);
        fillLevel(words1, 7);

        run(queue, ^(id<MTLBlitCommandEncoder> blit) {
            copy(blit, rgba1, MTLOriginMake(1, 0, 0), MTLSizeMake(4, 3, 1), words2,
                 MTLOriginMake(2, 4, 0));
        });
        report("t2t-array", words2);

        run(queue, ^(id<MTLBlitCommandEncoder> blit) {
            copy(blit, face, MTLOriginMake(0, 0, 0), MTLSizeMake(4, 4, 1), image,
                 MTLOriginMake(0, 0, 1));
            copy(blit, image, MTLOriginMake(1, 1, 1), MTLSizeMake(2, 2, 1), flat,
                 MTLOriginMake(2, 0, 0));
            copy(blit, words0, MTLOriginMake(0, 0, 0), MTLSizeMake(5, 5, 1), words1,
                 MTLOriginMake(3, 3, 0));
        });
        report("t2t-volume", image);
        report("t2t-plane", flat);
        report("t2t-slices", words1);

        id<MTLTexture> bcA = texture(device, MTLTextureType2D, MTLPixelFormatBC1_RGBA, 8, 8, 1, 1, 1);
        id<MTLTexture> bcB = texture(device, MTLTextureType2D, MTLPixelFormatBC1_RGBA, 8, 8, 1, 1, 1);
        if (bcA && bcB) {
            Level a = {bcA, 0, 0, 8, 4}, b = {bcB, 0, 0, 8, 4};
            fillLevel(a, 8);
            fillLevel(b, 9);
            run(queue, ^(id<MTLBlitCommandEncoder> blit) {
                copy(blit, a, MTLOriginMake(4, 0, 0), MTLSizeMake(4, 4, 1), b,
                     MTLOriginMake(0, 4, 0));
            });
            report("t2t-bc1", b);
        } else {
            printf("t2t-bc1 no BC1 texture\n");
        }

        id<MTLTexture> levels = texture(device, MTLTextureType2D, MTLPixelFormatRGBA8Unorm, 13, 7, 1, 3, 1);
        id<MTLTexture> slices = texture(device, MTLTextureType2DArray, MTLPixelFormatRGBA8Unorm, 8, 8, 1, 1, 3);
        id<MTLTexture> bytes3d = texture(device, MTLTextureType3D, MTLPixelFormatR8Unorm, 4, 4, 3, 1, 1);
        id<MTLTexture> lines = texture(device, MTLTextureType1DArray, MTLPixelFormatR32Float, 16, 1, 1, 1, 2);
        id<MTLTexture> blocks = texture(device, MTLTextureType2D, MTLPixelFormatBC1_RGBA, 8, 8, 1, 1, 1);
        Level level1 = {levels, 0, 1, 4, 1}, slice2 = {slices, 2, 0, 4, 1};
        Level volume0 = {bytes3d, 0, 0, 1, 1}, line1 = {lines, 1, 0, 4, 1};
        Level blocks0 = {blocks, 0, 0, 8, 4};
        fillLevel(level1, 12);
        fillLevel(slice2, 13);
        fillLevel(volume0, 15);
        fillLevel(line1, 19);
        fillLevel(blocks0, 20);
        id<MTLBuffer> intoLevel = patterned(device, 256, 11);
        id<MTLBuffer> fromSlice = patterned(device, 1024, 14);
        id<MTLBuffer> fromVolume = patterned(device, 128, 16);
        id<MTLBuffer> intoLine = patterned(device, 256, 18);
        id<MTLBuffer> intoBlocks = patterned(device, 64, 17);
        run(queue, ^(id<MTLBlitCommandEncoder> blit) {
            [blit copyFromBuffer:intoLevel
                       sourceOffset:16
                  sourceBytesPerRow:32
                sourceBytesPerImage:0
                         sourceSize:MTLSizeMake(4, 2, 1)
                          toTexture:levels
                   destinationSlice:0
                   destinationLevel:1
                  destinationOrigin:MTLOriginMake(1, 1, 0)];
            [blit copyFromTexture:slices
                             sourceSlice:2
                             sourceLevel:0
                            sourceOrigin:MTLOriginMake(2, 3, 0)
                              sourceSize:MTLSizeMake(5, 4, 1)
                                toBuffer:fromSlice
                       destinationOffset:64
                  destinationBytesPerRow:48
                destinationBytesPerImage:0];
            [blit copyFromTexture:bytes3d
                             sourceSlice:0
                             sourceLevel:0
                            sourceOrigin:MTLOriginMake(1, 0, 0)
                              sourceSize:MTLSizeMake(3, 4, 3)
                                toBuffer:fromVolume
                       destinationOffset:8
                  destinationBytesPerRow:5
                destinationBytesPerImage:24];
            [blit copyFromBuffer:intoLine
                       sourceOffset:16
                  sourceBytesPerRow:32
                sourceBytesPerImage:40
                         sourceSize:MTLSizeMake(8, 1, 1)
                          toTexture:lines
                   destinationSlice:1
                   destinationLevel:0
                  destinationOrigin:MTLOriginMake(3, 0, 0)];
            if (blocks) {
                [blit copyFromBuffer:intoBlocks
                           sourceOffset:8
                      sourceBytesPerRow:16
                    sourceBytesPerImage:0
                             sourceSize:MTLSizeMake(4, 4, 1)
                              toTexture:blocks
                       destinationSlice:0
                       destinationLevel:0
                      destinationOrigin:MTLOriginMake(4, 4, 0)];
            }
        });
        report("b2t-level", level1);
        reportBuffer("t2b-slice", fromSlice);
        reportBuffer("t2b-volume", fromVolume);
        report("b2t-line", line1);
        if (blocks) {
            report("b2t-bc1", blocks0);
        } else {
            printf("b2t-bc1 no BC1 texture\n");
        }
    }
    return 0;
}
