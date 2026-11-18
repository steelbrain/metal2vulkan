// Plain Metal's bytes for the new runtime's blit copy tests (metal2vulkan-rt, tests::blit): the
// same textures, buffers, byte patterns and copies, each destination read back whole and printed
// as its length and FNV-1a 64 hash, which the tests hold their own results to.
//
// A pattern of `len` bytes from `seed` is byte i = (uint8_t)i * 31 + seed, as the tests make it.
// Every texture is Shared, written and read with replaceRegion and getBytes, rows tightly packed.
//
//   t2t-array   RGBA8Unorm 13x7 (3 levels), level 1 (pattern 1) region (1,0) 4x3 into slice 2
//               (pattern 4) of an R32Uint 8x8 2D array of 3 at (2,4)
//   t2t-volume  face 4 of an R8Unorm 4x4 cube (pattern 2) into an R8Unorm 4x4x3 3D texture
//               (pattern 5) at (0,0,1)
//   t2t-plane   that 3D texture's (1,1,1) 2x2x1 into an R8Unorm 4x4 2D texture (pattern 6) at (2,0)
//   t2t-slices  slice 0 (pattern 3) (0,0) 5x5 of the array into its slice 1 (pattern 7) at (3,3)
//   t2t-bc1     BC1RGBA 8x8 (pattern 8) (4,0) 4x4 into another (pattern 9) at (0,4)
//
// Build, and run plain and under Metal's validation layer, which stops the process at a copy it
// does not take (RGBA8Unorm into R32Uint above is one unit's bytes in two formats):
//
//   xcrun clang -fobjc-arc -framework Metal -framework Foundation \
//       -o blit_copy_layouts blit_copy_layouts.m
//   ./blit_copy_layouts
//   MTL_DEBUG_LAYER=1 ./blit_copy_layouts
//
// WHAT IT MEASURED: see the hashes in the tests (`tests::blit`), with the run they came from.

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

static void write(Level l, uint8_t seed) {
    MTLSize e = extent(l);
    NSData *bytes = pattern(levelBytes(l), seed);
    [l.texture replaceRegion:MTLRegionMake3D(0, 0, 0, e.width, e.height, e.depth)
                 mipmapLevel:l.level
                       slice:l.slice
                   withBytes:bytes.bytes
                 bytesPerRow:rowBytes(l)
               bytesPerImage:rowBytes(l) * rows(l)];
}

static void print(const char *name, Level l) {
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
        id<MTLTexture> words = texture(device, MTLTextureType2DArray, MTLPixelFormatR32Uint, 8, 8, 1, 1, 3);
        id<MTLTexture> cube = texture(device, MTLTextureTypeCube, MTLPixelFormatR8Unorm, 4, 4, 1, 1, 1);
        id<MTLTexture> volume = texture(device, MTLTextureType3D, MTLPixelFormatR8Unorm, 4, 4, 3, 1, 1);
        id<MTLTexture> plane = texture(device, MTLTextureType2D, MTLPixelFormatR8Unorm, 4, 4, 1, 1, 1);
        Level rgba1 = {rgba, 0, 1, 4, 1}, words2 = {words, 2, 0, 4, 1};
        Level words0 = {words, 0, 0, 4, 1}, words1 = {words, 1, 0, 4, 1};
        Level face = {cube, 4, 0, 1, 1}, image = {volume, 0, 0, 1, 1}, flat = {plane, 0, 0, 1, 1};
        write(rgba1, 1);
        write(face, 2);
        write(words0, 3);
        write(words2, 4);
        write(image, 5);
        write(flat, 6);
        write(words1, 7);

        run(queue, ^(id<MTLBlitCommandEncoder> blit) {
            copy(blit, rgba1, MTLOriginMake(1, 0, 0), MTLSizeMake(4, 3, 1), words2,
                 MTLOriginMake(2, 4, 0));
        });
        print("t2t-array", words2);

        run(queue, ^(id<MTLBlitCommandEncoder> blit) {
            copy(blit, face, MTLOriginMake(0, 0, 0), MTLSizeMake(4, 4, 1), image,
                 MTLOriginMake(0, 0, 1));
            copy(blit, image, MTLOriginMake(1, 1, 1), MTLSizeMake(2, 2, 1), flat,
                 MTLOriginMake(2, 0, 0));
            copy(blit, words0, MTLOriginMake(0, 0, 0), MTLSizeMake(5, 5, 1), words1,
                 MTLOriginMake(3, 3, 0));
        });
        print("t2t-volume", image);
        print("t2t-plane", flat);
        print("t2t-slices", words1);

        id<MTLTexture> bcA = texture(device, MTLTextureType2D, MTLPixelFormatBC1_RGBA, 8, 8, 1, 1, 1);
        id<MTLTexture> bcB = texture(device, MTLTextureType2D, MTLPixelFormatBC1_RGBA, 8, 8, 1, 1, 1);
        if (bcA && bcB) {
            Level a = {bcA, 0, 0, 8, 4}, b = {bcB, 0, 0, 8, 4};
            write(a, 8);
            write(b, 9);
            run(queue, ^(id<MTLBlitCommandEncoder> blit) {
                copy(blit, a, MTLOriginMake(4, 0, 0), MTLSizeMake(4, 4, 1), b,
                     MTLOriginMake(0, 4, 0));
            });
            print("t2t-bc1", b);
        } else {
            printf("t2t-bc1 no BC1 texture\n");
        }
    }
    return 0;
}
