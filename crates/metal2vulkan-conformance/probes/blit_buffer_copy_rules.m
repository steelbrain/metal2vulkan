// Which buffer layouts Metal takes in `copyFromBuffer:sourceOffset:sourceBytesPerRow:
// sourceBytesPerImage:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:`
// and `copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:
// destinationOffset:destinationBytesPerRow:destinationBytesPerImage:`.
//
// ONE CASE PER PROCESS, chosen on the command line, run under MTL_DEBUG_LAYER=1: the validation
// layer stops the process at a copy it does not take, so a case that prints "taken" was taken.
// Every texture is Private with one level; every buffer Shared, 4096 bytes. b2t is buffer to
// texture, t2b texture to buffer.
//
//   fine           b2t RGBA8Unorm 8x8, 4x2 texels, offset 4, bytesPerRow 16 (the control)
//   offset-pixel   b2t RGBA8Unorm 8x8, 4x2 texels, offset 2: not a whole pixel
//   offset-four    b2t R8Unorm 8x8, 4x2 texels, offset 2: whole pixels, not a multiple of 4
//   row-pixel      b2t RGBA8Unorm 8x8, 4x2 texels, bytesPerRow 18: not whole pixels
//   row-short      b2t RGBA8Unorm 8x8, 4x2 texels, bytesPerRow 12: shorter than a row
//   image-2d       b2t RGBA8Unorm 8x8, 4x2 texels, bytesPerRow 16, bytesPerImage 40
//   image-odd      b2t R8Unorm 4x4x2 3D, 4x4x2 texels, bytesPerRow 4, bytesPerImage 18: not
//                  whole rows
//   image-short    b2t R8Unorm 4x4x2 3D, 4x4x2 texels, bytesPerRow 4, bytesPerImage 12
//   array-1d       b2t R32Float 16 wide 1D array of 2, slice 1, 8 texels at 3, bytesPerRow 32,
//                  bytesPerImage 40
//   block-row      b2t BC1_RGBA 8x8, 4x4 texels, bytesPerRow 20: not whole blocks
//   depth-offset   t2b Depth16Unorm 8x8, 4x2 texels, offset 2: a whole pixel, not a multiple of 4
//   t2b-row-pixel  t2b RGBA8Unorm 8x8, 4x2 texels, bytesPerRow 18
//   one-row-zero   b2t RGBA8Unorm 8x8, 4x1 texels, bytesPerRow 0
//   past-end       b2t RGBA8Unorm 8x8, 4x2 texels, offset 4080, bytesPerRow 16: past the
//                  buffer's 4096 bytes
//
//   xcrun clang -fobjc-arc -framework Metal -framework Foundation \
//       -o blit_buffer_copy_rules blit_buffer_copy_rules.m
//   for c in fine offset-pixel offset-four row-pixel row-short image-2d image-odd image-short \
//       array-1d block-row depth-offset t2b-row-pixel one-row-zero past-end; do
//       MTL_DEBUG_LAYER=1 ./blit_buffer_copy_rules $c; done
//
// WHAT IT MEASURED: see the rt crate's buffer copy refusals, with the run they came from.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <string.h>

typedef struct {
    const char *name;
    BOOL toBuffer;
    MTLTextureType type;
    MTLPixelFormat format;
    NSUInteger width, height, depth, arrayLength, slice;
    NSUInteger x, w, h, d;
    NSUInteger offset, bytesPerRow, bytesPerImage;
} Case;

static const Case cases[] = {
    {"fine", NO, MTLTextureType2D, MTLPixelFormatRGBA8Unorm, 8, 8, 1, 1, 0, 0, 4, 2, 1, 4, 16, 0},
    {"offset-pixel", NO, MTLTextureType2D, MTLPixelFormatRGBA8Unorm, 8, 8, 1, 1, 0, 0, 4, 2, 1, 2, 16, 0},
    {"offset-four", NO, MTLTextureType2D, MTLPixelFormatR8Unorm, 8, 8, 1, 1, 0, 0, 4, 2, 1, 2, 8, 0},
    {"row-pixel", NO, MTLTextureType2D, MTLPixelFormatRGBA8Unorm, 8, 8, 1, 1, 0, 0, 4, 2, 1, 0, 18, 0},
    {"row-short", NO, MTLTextureType2D, MTLPixelFormatRGBA8Unorm, 8, 8, 1, 1, 0, 0, 4, 2, 1, 0, 12, 0},
    {"image-2d", NO, MTLTextureType2D, MTLPixelFormatRGBA8Unorm, 8, 8, 1, 1, 0, 0, 4, 2, 1, 0, 16, 40},
    {"image-odd", NO, MTLTextureType3D, MTLPixelFormatR8Unorm, 4, 4, 2, 1, 0, 0, 4, 4, 2, 0, 4, 18},
    {"image-short", NO, MTLTextureType3D, MTLPixelFormatR8Unorm, 4, 4, 2, 1, 0, 0, 4, 4, 2, 0, 4, 12},
    {"array-1d", NO, MTLTextureType1DArray, MTLPixelFormatR32Float, 16, 1, 1, 2, 1, 3, 8, 1, 1, 0, 32, 40},
    {"block-row", NO, MTLTextureType2D, MTLPixelFormatBC1_RGBA, 8, 8, 1, 1, 0, 0, 4, 4, 1, 0, 20, 0},
    {"depth-offset", YES, MTLTextureType2D, MTLPixelFormatDepth16Unorm, 8, 8, 1, 1, 0, 0, 4, 2, 1, 2, 8, 0},
    {"t2b-row-pixel", YES, MTLTextureType2D, MTLPixelFormatRGBA8Unorm, 8, 8, 1, 1, 0, 0, 4, 2, 1, 0, 18, 0},
    {"one-row-zero", NO, MTLTextureType2D, MTLPixelFormatRGBA8Unorm, 8, 8, 1, 1, 0, 0, 4, 1, 1, 0, 0, 0},
    {"past-end", NO, MTLTextureType2D, MTLPixelFormatRGBA8Unorm, 8, 8, 1, 1, 0, 0, 4, 2, 1, 4080, 16, 0},
};

int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc != 2) {
            printf("usage: blit_buffer_copy_rules CASE\n");
            return 2;
        }
        const Case *c = NULL;
        for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
            if (strcmp(cases[i].name, argv[1]) == 0) c = &cases[i];
        }
        if (!c) {
            printf("no case %s\n", argv[1]);
            return 2;
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        MTLTextureDescriptor *desc = [MTLTextureDescriptor new];
        desc.textureType = c->type;
        desc.pixelFormat = c->format;
        desc.width = c->width;
        desc.height = c->height;
        desc.depth = c->depth;
        desc.arrayLength = c->arrayLength;
        desc.usage = MTLTextureUsageShaderRead;
        desc.storageMode = MTLStorageModePrivate;
        id<MTLTexture> texture = [device newTextureWithDescriptor:desc];
        id<MTLBuffer> buffer = [device newBufferWithLength:4096 options:MTLResourceStorageModeShared];
        if (!texture || !buffer) {
            printf("%s: no texture or buffer\n", c->name);
            return 1;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        MTLOrigin origin = MTLOriginMake(c->x, 0, 0);
        MTLSize size = MTLSizeMake(c->w, c->h, c->d);
        if (c->toBuffer) {
            [blit copyFromTexture:texture
                             sourceSlice:c->slice
                             sourceLevel:0
                            sourceOrigin:origin
                              sourceSize:size
                                toBuffer:buffer
                       destinationOffset:c->offset
                  destinationBytesPerRow:c->bytesPerRow
                destinationBytesPerImage:c->bytesPerImage];
        } else {
            [blit copyFromBuffer:buffer
                       sourceOffset:c->offset
                  sourceBytesPerRow:c->bytesPerRow
                sourceBytesPerImage:c->bytesPerImage
                         sourceSize:size
                          toTexture:texture
                   destinationSlice:c->slice
                   destinationLevel:0
                  destinationOrigin:origin];
        }
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        printf("%s: taken, status %ld%s%s\n", c->name, (long)cb.status, cb.error ? ", error " : "",
               cb.error ? cb.error.localizedDescription.UTF8String : "");
    }
    return 0;
}
