// Whether Metal reads a render pass attachment's `resolveDepthPlane` of a resolve texture that is
// not 3D, for a colour, a depth and a stencil attachment, and its `depthPlane` of a 2D texture.
//
// ONE CASE PER PROCESS, chosen on the command line, run once under MTL_DEBUG_LAYER=1 and once
// without it: the validation layer stops the process at a call it does not take, so a case that
// prints "made yes" got its encoder, and one that prints "texels" ran and read back. Every texture
// is 8x8 and Private. The pass has a 4-sample RGBA8Unorm colour attachment (RenderTarget), cleared
// to (0.25, 0.5, 0.75, 1.0), and draws nothing; the case names what resolves and where:
//
//   color0    the colour, MultisampleResolve, into a 1-sample RGBA8Unorm 2D texture
//             (RenderTarget), resolveDepthPlane 0
//   color3    the same at resolveDepthPlane 3
//   slice3    the same at resolveSlice 3, a slice the 2D texture lacks: the control that shows
//             what the layer stops
//   volume3   the same into an 8x8x4 3D texture at resolveDepthPlane 3, read at plane 3
//   depth0    beside the colour (DontCare), a 4-sample Depth32Float depth attachment cleared to
//             0.25, MultisampleResolve by Sample0 into a 1-sample Depth32Float 2D texture,
//             resolveDepthPlane 0
//   depth3    the same at resolveDepthPlane 3
//   stencil0  beside the colour (DontCare), a 4-sample Stencil8 stencil attachment cleared to
//             0x5a, MultisampleResolve by Sample0 into a 1-sample Stencil8 2D texture,
//             resolveDepthPlane 0
//   stencil3  the same at resolveDepthPlane 3
//   plane3    no resolve: a 1-sample RGBA8Unorm 2D colour attachment at depthPlane 3, cleared and
//             stored (Apple's page for depthPlane: "If the texture isn't a 3D texture, then Metal
//             ignores this property.")
//
// A case prints whether the encoder was made, the command buffer's status and error, and, read
// back by a blit into a Shared buffer (rows tight, 0xee before), the FNV-1a 64 of the texels it
// wrote and the first texel's bytes.
//
//   xcrun clang -fobjc-arc -framework Metal -framework Foundation \
//       -o resolve_depth_planes resolve_depth_planes.m
//   for c in color0 color3 slice3 volume3 depth0 depth3 stencil0 stencil3 plane3; do
//     MTL_DEBUG_LAYER=1 ./resolve_depth_planes $c
//     env -u MTL_DEBUG_LAYER ./resolve_depth_planes $c
//   done
//
// WHAT IT MEASURED, Apple M3 Max, macOS 27.0.1 (Build 26A434), job m3-max-73495:
//
//   - with the layer off, every case makes its encoder and completes (status 4, no error), and
//     color3, depth3 and stencil3 read exactly what color0, depth0 and stencil0 read (texels
//     FNV-1a 607aafd9f8dd4ba5, first 40 80 bf ff; e02b8804d001b725, 0.25; 9d1d83ba5bf55125, 5a):
//     a resolve texture's plane but 0 is not read of a 2D texture. slice3 and plane3 read
//     607aafd9f8dd4ba5 too;
//   - with the layer on, color0, depth0, stencil0 and volume3 run as with it off, and color3,
//     depth3, stencil3 and plane3 end the process at the encoder (SIGABRT, exit 134): "depth plane
//     is 3, but the texture has a depth of 1 at mip level 0" (plane3: "depth plane index is 3"),
//     slice3 "Slice is 3, but the texture has only 1 slices";
//   - volume3 resolves into the 3D texture's plane 3, read there (607aafd9f8dd4ba5).

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const NSUInteger SIDE = 8;

static id<MTLTexture> texture(id<MTLDevice> device, MTLTextureType type, MTLPixelFormat format,
                              NSUInteger samples, NSUInteger depth) {
    MTLTextureDescriptor *d = [MTLTextureDescriptor new];
    d.textureType = type;
    d.pixelFormat = format;
    d.width = SIDE;
    d.height = SIDE;
    d.depth = depth;
    d.sampleCount = samples;
    d.usage = MTLTextureUsageRenderTarget;
    d.storageMode = MTLStorageModePrivate;
    return [device newTextureWithDescriptor:d];
}

static id<MTLTexture> four(id<MTLDevice> device, MTLPixelFormat format) {
    return texture(device, MTLTextureType2DMultisample, format, 4, 1);
}

static id<MTLTexture> one(id<MTLDevice> device, MTLPixelFormat format) {
    return texture(device, MTLTextureType2D, format, 1, 1);
}

static uint64_t fnv(const uint8_t *bytes, NSUInteger length) {
    uint64_t hash = 0xcbf29ce484222325ull;
    for (NSUInteger i = 0; i < length; i++) {
        hash ^= bytes[i];
        hash *= 0x100000001b3ull;
    }
    return hash;
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        setvbuf(stdout, NULL, _IONBF, 0);
        if (argc != 2) {
            printf("usage: resolve_depth_planes CASE\n");
            return 2;
        }
        const char *name = argv[1];
        const char *layer = getenv("MTL_DEBUG_LAYER");
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        printf("%s: %s, MTL_DEBUG_LAYER %s\n", name, device.name.UTF8String,
               layer ? layer : "unset");
        size_t length = strlen(name);
        NSUInteger plane = name[length - 1] == '3' ? 3 : 0;
        MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
        MTLRenderPassColorAttachmentDescriptor *c = pass.colorAttachments[0];
        c.texture = four(device, MTLPixelFormatRGBA8Unorm);
        c.loadAction = MTLLoadActionClear;
        c.clearColor = MTLClearColorMake(0.25, 0.5, 0.75, 1.0);
        c.storeAction = MTLStoreActionDontCare;
        id<MTLTexture> read = nil;
        NSUInteger unit = 4;
        NSUInteger at = 0;
        if (strncmp(name, "color", 5) == 0 || strcmp(name, "slice3") == 0) {
            read = one(device, MTLPixelFormatRGBA8Unorm);
            c.storeAction = MTLStoreActionMultisampleResolve;
            c.resolveTexture = read;
            if (strcmp(name, "slice3") == 0) {
                c.resolveSlice = 3;
            } else {
                c.resolveDepthPlane = plane;
            }
        } else if (strcmp(name, "volume3") == 0) {
            read = texture(device, MTLTextureType3D, MTLPixelFormatRGBA8Unorm, 1, 4);
            c.storeAction = MTLStoreActionMultisampleResolve;
            c.resolveTexture = read;
            c.resolveDepthPlane = plane;
            at = plane;
        } else if (strncmp(name, "depth", 5) == 0) {
            MTLRenderPassDepthAttachmentDescriptor *z = pass.depthAttachment;
            read = one(device, MTLPixelFormatDepth32Float);
            z.texture = four(device, MTLPixelFormatDepth32Float);
            z.loadAction = MTLLoadActionClear;
            z.clearDepth = 0.25;
            z.storeAction = MTLStoreActionMultisampleResolve;
            z.depthResolveFilter = MTLMultisampleDepthResolveFilterSample0;
            z.resolveTexture = read;
            z.resolveDepthPlane = plane;
        } else if (strncmp(name, "stencil", 7) == 0) {
            MTLRenderPassStencilAttachmentDescriptor *s = pass.stencilAttachment;
            read = one(device, MTLPixelFormatStencil8);
            unit = 1;
            s.texture = four(device, MTLPixelFormatStencil8);
            s.loadAction = MTLLoadActionClear;
            s.clearStencil = 0x5a;
            s.storeAction = MTLStoreActionMultisampleResolve;
            s.stencilResolveFilter = MTLMultisampleStencilResolveFilterSample0;
            s.resolveTexture = read;
            s.resolveDepthPlane = plane;
        } else if (strcmp(name, "plane3") == 0) {
            read = one(device, MTLPixelFormatRGBA8Unorm);
            c.texture = read;
            c.storeAction = MTLStoreActionStore;
            c.depthPlane = plane;
        } else {
            printf("no case %s\n", name);
            return 2;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        printf("%s: making the encoder\n", name);
        id<MTLRenderCommandEncoder> encoder = [cb renderCommandEncoderWithDescriptor:pass];
        printf("%s: made %s\n", name, encoder ? "yes" : "no");
        if (encoder == nil) {
            return 0;
        }
        [encoder endEncoding];
        NSUInteger row = SIDE * unit;
        id<MTLBuffer> out = [device newBufferWithLength:row * SIDE
                                                options:MTLResourceStorageModeShared];
        memset(out.contents, 0xee, out.length);
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromTexture:read
                     sourceSlice:0
                     sourceLevel:0
                    sourceOrigin:MTLOriginMake(0, 0, at)
                      sourceSize:MTLSizeMake(SIDE, SIDE, 1)
                        toBuffer:out
               destinationOffset:0
          destinationBytesPerRow:row
        destinationBytesPerImage:row * SIDE];
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        printf("%s: status %ld, error %s\n", name, (long)cb.status,
               cb.error ? cb.error.description.UTF8String : "none");
        const uint8_t *bytes = out.contents;
        printf("%s: texels fnv %016llx, first", name, (unsigned long long)fnv(bytes, out.length));
        for (NSUInteger i = 0; i < unit; i++) {
            printf(" %02x", bytes[i]);
        }
        if (unit == 4 && strncmp(name, "depth", 5) == 0) {
            float depth;
            memcpy(&depth, bytes, 4);
            printf(" (%g)", depth);
        }
        printf("\n");
    }
    return 0;
}
