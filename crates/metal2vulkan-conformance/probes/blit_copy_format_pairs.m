// Which pairs of pixel formats Metal takes in `copyFromTexture:sourceSlice:sourceLevel:
// sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:`.
//
// ONE CASE PER PROCESS, chosen on the command line, run under MTL_DEBUG_LAYER=1: the validation
// layer stops the process at a copy it does not take, so a case that prints "taken" was taken.
// Both textures are Private 2D 8x8 with one level (the source of `samples` 2D multisample, four
// samples); the copy is the first 4x4 texels.
//
//   same           RGBA8Unorm into RGBA8Unorm (the control: taken)
//   srgb           RGBA8Unorm into RGBA8Unorm_sRGB
//   uint           RGBA8Unorm into RGBA8Uint
//   word           RGBA8Unorm into R32Uint
//   float          R32Uint into R32Float
//   order          RGBA8Unorm into BGRA8Unorm
//   depth          Depth32Float into R32Float
//   block          BC1_RGBA into RG32Uint
//   samples        RGBA8Unorm of four samples into RGBA8Unorm of one
//
//   xcrun clang -fobjc-arc -framework Metal -framework Foundation \
//       -o blit_copy_format_pairs blit_copy_format_pairs.m
//   for c in same srgb uint word float order depth block samples; do
//       MTL_DEBUG_LAYER=1 ./blit_copy_format_pairs $c; done
//
// WHAT IT MEASURED (Apple M3 Max, macOS 27.0.1, job m3-max-43433, before `samples` was added):
// `same` taken; every pair of two formats stopped by the validation layer with "[sourceTexture
// pixelFormat] must equal [destinationTexture pixelFormat]".

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <string.h>

static id<MTLTexture> make(id<MTLDevice> device, MTLPixelFormat format, NSUInteger samples) {
    MTLTextureDescriptor *desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:8 height:8 mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModePrivate;
    if (samples > 1) {
        desc.textureType = MTLTextureType2DMultisample;
        desc.sampleCount = samples;
        desc.usage = MTLTextureUsageRenderTarget;
    }
    return [device newTextureWithDescriptor:desc];
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc != 2) {
            printf("usage: blit_copy_format_pairs CASE\n");
            return 2;
        }
        const char *name = argv[1];
        struct {
            const char *name;
            MTLPixelFormat from, to;
            NSUInteger samples;
        } cases[] = {
            {"same", MTLPixelFormatRGBA8Unorm, MTLPixelFormatRGBA8Unorm, 1},
            {"srgb", MTLPixelFormatRGBA8Unorm, MTLPixelFormatRGBA8Unorm_sRGB, 1},
            {"uint", MTLPixelFormatRGBA8Unorm, MTLPixelFormatRGBA8Uint, 1},
            {"word", MTLPixelFormatRGBA8Unorm, MTLPixelFormatR32Uint, 1},
            {"float", MTLPixelFormatR32Uint, MTLPixelFormatR32Float, 1},
            {"order", MTLPixelFormatRGBA8Unorm, MTLPixelFormatBGRA8Unorm, 1},
            {"depth", MTLPixelFormatDepth32Float, MTLPixelFormatR32Float, 1},
            {"block", MTLPixelFormatBC1_RGBA, MTLPixelFormatRG32Uint, 1},
            {"samples", MTLPixelFormatRGBA8Unorm, MTLPixelFormatRGBA8Unorm, 4},
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
        id<MTLTexture> from = make(device, cases[found].from, cases[found].samples);
        id<MTLTexture> to = make(device, cases[found].to, 1);
        if (!from || !to) {
            printf("%s: no texture\n", name);
            return 1;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        NSUInteger n = 4;
        [blit copyFromTexture:from
                  sourceSlice:0
                  sourceLevel:0
                 sourceOrigin:MTLOriginMake(0, 0, 0)
                   sourceSize:MTLSizeMake(n, n, 1)
                    toTexture:to
             destinationSlice:0
             destinationLevel:0
            destinationOrigin:MTLOriginMake(0, 0, 0)];
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        printf("%s: taken, status %ld%s%s\n", name, (long)cb.status, cb.error ? ", error " : "",
               cb.error ? cb.error.localizedDescription.UTF8String : "");
    }
    return 0;
}
