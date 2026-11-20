// Which texture views Metal makes with `newTextureViewWithPixelFormat:textureType:levels:slices:`,
// what a view reports of itself, and the bytes a copy out of a view reads.
//
// ONE CASE PER PROCESS, chosen on the command line, run under MTL_DEBUG_LAYER=1: the validation
// layer stops the process at a call it does not take, so a case that prints "made" was made. The
// parent is RGBA8Unorm, ShaderRead usage unless the case says otherwise: an 8x8 2D texture of 3
// levels, or an 8x8 2D array of 3 slices and 2 levels. Each level and slice of the parent holds
// bytes `(uint8_t)(i * 31 + 21 + 7 * level + 50 * slice)`, written by replaceRegion. A view that is
// made prints its width, height, depth, mipmapLevelCount, arrayLength, textureType,
// parentRelativeLevel and parentRelativeSlice, and the FNV-1a 64 of its level 0, slice 0 bytes
// copied by a blit to a Shared buffer (rows of the format's bytes a texel, tight).
//
// Views of another pixel format of the same component layout (Apple's `pixelFormatView` page: a
// layout differs "if the number of components differs, or if their size or order is different"),
// each of a 2D parent of ShaderRead usage alone unless the name ends in `-usage`:
//
//   r8-a8              A8Unorm view of an R8Unorm parent
//   r8-a8-usage        the same, the parent of ShaderRead | PixelFormatView usage
//   r8-r8snorm         R8Snorm view of an R8Unorm parent
//   r8-r8uint          R8Uint view of an R8Unorm parent
//   rgba8-rgba8uint    RGBA8Uint view of the RGBA8Unorm parent
//   rgba8-bgra8        BGRA8Unorm view of the RGBA8Unorm parent
//   rgba8-bgra8srgb    BGRA8Unorm_sRGB view of the RGBA8Unorm parent
//   rg8-r16            R16Unorm view of an RG8Unorm parent
//
//   whole          2D, levels 0..3, slices 0..1
//   levels         2D, levels 1..3
//   level-one      2D, levels 2..3
//   array-2d       2D of the array, levels 0..2, slice 0
//   array-2d-last  2D of the array, levels 1..2, slice 2
//   array-slices   2DArray of the array, slices 1..3
//   levels-past    2D, levels 2..4
//   levels-empty   2D, levels 1..1
//   slices-past    2D of the array, slice 3
//   format         R32Uint view of the 2D parent, no PixelFormatView usage
//   format-usage   R32Uint view of a 2D parent of ShaderRead | PixelFormatView usage
//
//   xcrun clang -fobjc-arc -framework Metal -framework Foundation \
//       -o texture_view_rules texture_view_rules.m
//   for c in whole levels level-one array-2d array-2d-last array-slices levels-past \
//       levels-empty slices-past format format-usage r8-a8 r8-a8-usage r8-r8snorm r8-r8uint \
//       rgba8-rgba8uint rgba8-bgra8 rgba8-bgra8srgb rg8-r16; do
//     MTL_DEBUG_LAYER=1 ./texture_view_rules $c
//   done
//
// WHAT IT MEASURED: see the rt crate's `Texture::new_texture_view`, with the run it came from.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <string.h>

int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc != 2) {
            printf("usage: texture_view_rules CASE\n");
            return 2;
        }
        const char *name = argv[1];
        struct {
            const char *name;
            BOOL array;
            MTLTextureType type;
            NSUInteger level, levels, slice, slices;
            MTLPixelFormat format;
            BOOL view_usage;
            MTLPixelFormat parent;
            NSUInteger texel;
        } cases[] = {
            {"whole", NO, MTLTextureType2D, 0, 3, 0, 1, MTLPixelFormatRGBA8Unorm, NO},
            {"levels", NO, MTLTextureType2D, 1, 2, 0, 1, MTLPixelFormatRGBA8Unorm, NO},
            {"level-one", NO, MTLTextureType2D, 2, 1, 0, 1, MTLPixelFormatRGBA8Unorm, NO},
            {"array-2d", YES, MTLTextureType2D, 0, 2, 0, 1, MTLPixelFormatRGBA8Unorm, NO},
            {"array-2d-last", YES, MTLTextureType2D, 1, 1, 2, 1, MTLPixelFormatRGBA8Unorm, NO},
            {"array-slices", YES, MTLTextureType2DArray, 0, 2, 1, 2, MTLPixelFormatRGBA8Unorm, NO},
            {"levels-past", NO, MTLTextureType2D, 2, 2, 0, 1, MTLPixelFormatRGBA8Unorm, NO},
            {"levels-empty", NO, MTLTextureType2D, 1, 0, 0, 1, MTLPixelFormatRGBA8Unorm, NO},
            {"slices-past", YES, MTLTextureType2D, 0, 1, 3, 1, MTLPixelFormatRGBA8Unorm, NO},
            {"format", NO, MTLTextureType2D, 0, 3, 0, 1, MTLPixelFormatR32Uint, NO},
            {"format-usage", NO, MTLTextureType2D, 0, 3, 0, 1, MTLPixelFormatR32Uint, YES},
            {"r8-a8", NO, MTLTextureType2D, 0, 3, 0, 1, MTLPixelFormatA8Unorm, NO,
             MTLPixelFormatR8Unorm, 1},
            {"r8-a8-usage", NO, MTLTextureType2D, 0, 3, 0, 1, MTLPixelFormatA8Unorm, YES,
             MTLPixelFormatR8Unorm, 1},
            {"r8-r8snorm", NO, MTLTextureType2D, 0, 3, 0, 1, MTLPixelFormatR8Snorm, NO,
             MTLPixelFormatR8Unorm, 1},
            {"r8-r8uint", NO, MTLTextureType2D, 0, 3, 0, 1, MTLPixelFormatR8Uint, NO,
             MTLPixelFormatR8Unorm, 1},
            {"rgba8-rgba8uint", NO, MTLTextureType2D, 0, 3, 0, 1, MTLPixelFormatRGBA8Uint, NO},
            {"rgba8-bgra8", NO, MTLTextureType2D, 0, 3, 0, 1, MTLPixelFormatBGRA8Unorm, NO},
            {"rgba8-bgra8srgb", NO, MTLTextureType2D, 0, 3, 0, 1, MTLPixelFormatBGRA8Unorm_sRGB,
             NO},
            {"rg8-r16", NO, MTLTextureType2D, 0, 3, 0, 1, MTLPixelFormatR16Unorm, NO,
             MTLPixelFormatRG8Unorm, 2},
        };
        int found = -1;
        for (int i = 0; i < (int)(sizeof cases / sizeof cases[0]); i++) {
            if (strcmp(cases[i].name, name) == 0) found = i;
        }
        if (found < 0) {
            printf("no case %s\n", name);
            return 2;
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        BOOL array = cases[found].array;
        // A case that names no parent views the RGBA8Unorm one, 4 bytes a texel.
        MTLPixelFormat parent_format = cases[found].parent ? cases[found].parent
                                                           : MTLPixelFormatRGBA8Unorm;
        NSUInteger texel = cases[found].texel ? cases[found].texel : 4;
        MTLTextureDescriptor *d = [MTLTextureDescriptor new];
        d.textureType = array ? MTLTextureType2DArray : MTLTextureType2D;
        d.pixelFormat = parent_format;
        d.width = 8;
        d.height = 8;
        d.arrayLength = array ? 3 : 1;
        d.mipmapLevelCount = array ? 2 : 3;
        d.usage = MTLTextureUsageShaderRead |
                  (cases[found].view_usage ? MTLTextureUsagePixelFormatView : 0);
        d.storageMode = MTLStorageModeShared;
        id<MTLTexture> parent = [device newTextureWithDescriptor:d];
        uint8_t bytes[8 * 8 * 4];
        for (NSUInteger level = 0; level < d.mipmapLevelCount; level++) {
            for (NSUInteger slice = 0; slice < d.arrayLength; slice++) {
                NSUInteger side = 8 >> level;
                for (NSUInteger i = 0; i < side * side * texel; i++) {
                    bytes[i] = (uint8_t)(i * 31 + 21 + 7 * level + 50 * slice);
                }
                [parent replaceRegion:MTLRegionMake2D(0, 0, side, side)
                          mipmapLevel:level
                                slice:slice
                            withBytes:bytes
                          bytesPerRow:side * texel
                        bytesPerImage:side * side * texel];
            }
        }
        id<MTLTexture> view =
            [parent newTextureViewWithPixelFormat:cases[found].format
                                      textureType:cases[found].type
                                           levels:NSMakeRange(cases[found].level,
                                                              cases[found].levels)
                                           slices:NSMakeRange(cases[found].slice,
                                                              cases[found].slices)];
        if (view == nil) {
            printf("%s: nil\n", name);
            return 0;
        }
        printf("%s: made, width %lu height %lu depth %lu levels %lu arrayLength %lu type %lu "
               "parentRelativeLevel %lu parentRelativeSlice %lu\n",
               name, (unsigned long)view.width, (unsigned long)view.height,
               (unsigned long)view.depth, (unsigned long)view.mipmapLevelCount,
               (unsigned long)view.arrayLength, (unsigned long)view.textureType,
               (unsigned long)view.parentRelativeLevel, (unsigned long)view.parentRelativeSlice);
        NSUInteger side = view.width;
        NSUInteger len = side * view.height * texel;
        id<MTLBuffer> out = [device newBufferWithLength:len options:MTLResourceStorageModeShared];
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromTexture:view
                         sourceSlice:0
                         sourceLevel:0
                        sourceOrigin:MTLOriginMake(0, 0, 0)
                          sourceSize:MTLSizeMake(side, view.height, 1)
                            toBuffer:out
                   destinationOffset:0
              destinationBytesPerRow:side * texel
            destinationBytesPerImage:len];
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        const uint8_t *got = out.contents;
        uint64_t hash = 0xcbf29ce484222325ull;
        for (NSUInteger i = 0; i < len; i++) {
            hash ^= got[i];
            hash *= 0x100000001b3ull;
        }
        printf("%s len %lu fnv %016llx, status %ld\n", name, (unsigned long)len,
               (unsigned long long)hash, (long)cb.status);
    }
    return 0;
}
