// Whether Metal resolves a multisample colour attachment into a texture made without
// `MTLTextureUsageRenderTarget`, and what the resolved texels read.
//
// ONE CASE PER PROCESS, chosen on the command line, run once under MTL_DEBUG_LAYER=1 and once
// without it: the validation layer stops the process at a call it does not take, so a case that
// prints "made" got its encoder, and one that prints "texels" ran and read back. The pass is an
// 8x8 4-sample BGRA8Unorm_sRGB colour attachment (RenderTarget, Private), cleared to
// (0.25, 0.5, 0.75, 1.0) and stored MultisampleResolve, with an 8x8 4-sample Depth32Float_Stencil8
// texture (RenderTarget, Private) as its depth and its stencil attachment, cleared and not stored.
// One draw of a triangle in (1, 0, 0, 1) covers the lower-left half of the target through its
// diagonal, so the pixels on the diagonal resolve the average of their covered samples. The
// resolve texture is an 8x8 1-sample BGRA8Unorm_sRGB 2D texture of the case's usage (Private), or
// a drawable's texture:
//
//   rt          RenderTarget
//   read        ShaderRead
//   readwrite   ShaderRead | ShaderWrite
//   drawable    a headless CAMetalLayer's nextDrawable, framebufferOnly YES (no read-back: Metal
//               refuses a blit out of a framebuffer-only texture)
//   drawable-readable  the same, framebufferOnly NO, read back
//   control     RenderTarget, but RGBA8Unorm_sRGB: a resolve Metal's validation stops (the
//               formats differ), which shows the layer is on
//
// A case prints the resolve texture's usage, whether the encoder was made, the command buffer's
// status and error, and, read back by a blit into a Shared buffer (BGRA bytes, rows tight), the
// FNV-1a 64 of the 256 bytes and the texels at (0, 0), (7, 7), (3, 4) and (4, 3).
//
//   xcrun clang -fobjc-arc -framework Metal -framework QuartzCore -framework Foundation \
//       -o resolve_texture_usage resolve_texture_usage.m
//   for c in rt read readwrite drawable drawable-readable control; do
//     MTL_DEBUG_LAYER=1 ./resolve_texture_usage $c
//     env -u MTL_DEBUG_LAYER ./resolve_texture_usage $c
//   done
//
// WHAT IT MEASURED: see the runtime's `RenderPassKind::ResolveUsage`, with the run it came from.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#include <string.h>

static const char *SOURCE =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "vertex float4 corner(uint id [[vertex_id]]) {\n"
    "    float2 p[3] = { float2(-1, -1), float2(1, -1), float2(-1, 1) };\n"
    "    return float4(p[id], 0.5, 1);\n"
    "}\n"
    "fragment half4 red() { return half4(1, 0, 0, 1); }\n";

static id<MTLTexture> texture(id<MTLDevice> device, MTLPixelFormat format, NSUInteger samples,
                              MTLTextureUsage usage) {
    MTLTextureDescriptor *d = [MTLTextureDescriptor new];
    d.textureType = samples > 1 ? MTLTextureType2DMultisample : MTLTextureType2D;
    d.pixelFormat = format;
    d.width = 8;
    d.height = 8;
    d.sampleCount = samples;
    d.usage = usage;
    d.storageMode = MTLStorageModePrivate;
    return [device newTextureWithDescriptor:d];
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc != 2) {
            printf("usage: resolve_texture_usage CASE\n");
            return 2;
        }
        const char *name = argv[1];
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        const MTLPixelFormat color = MTLPixelFormatBGRA8Unorm_sRGB;
        const MTLPixelFormat depth = MTLPixelFormatDepth32Float_Stencil8;
        id<MTLTexture> resolve = nil;
        id<CAMetalDrawable> drawable = nil;
        BOOL readable = YES;
        if (strcmp(name, "rt") == 0) {
            resolve = texture(device, color, 1, MTLTextureUsageRenderTarget);
        } else if (strcmp(name, "read") == 0) {
            resolve = texture(device, color, 1, MTLTextureUsageShaderRead);
        } else if (strcmp(name, "readwrite") == 0) {
            resolve = texture(device, color, 1,
                              MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
        } else if (strcmp(name, "control") == 0) {
            resolve = texture(device, MTLPixelFormatRGBA8Unorm_sRGB, 1,
                              MTLTextureUsageRenderTarget);
        } else if (strcmp(name, "drawable") == 0 || strcmp(name, "drawable-readable") == 0) {
            CAMetalLayer *layer = [CAMetalLayer layer];
            layer.device = device;
            layer.pixelFormat = color;
            layer.drawableSize = CGSizeMake(8, 8);
            readable = strcmp(name, "drawable-readable") == 0;
            layer.framebufferOnly = !readable;
            drawable = [layer nextDrawable];
            if (drawable == nil) {
                printf("%s: no drawable\n", name);
                return 0;
            }
            resolve = drawable.texture;
        } else {
            printf("no case %s\n", name);
            return 2;
        }
        printf("%s: resolve usage 0x%lx, %lux%lu, samples %lu\n", name,
               (unsigned long)resolve.usage, (unsigned long)resolve.width,
               (unsigned long)resolve.height, (unsigned long)resolve.sampleCount);
        id<MTLTexture> msaa = texture(device, color, 4, MTLTextureUsageRenderTarget);
        id<MTLTexture> ds = texture(device, depth, 4, MTLTextureUsageRenderTarget);
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:@(SOURCE) options:nil error:&error];
        if (library == nil) {
            printf("%s: no library: %s\n", name, error.description.UTF8String);
            return 3;
        }
        MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
        pd.vertexFunction = [library newFunctionWithName:@"corner"];
        pd.fragmentFunction = [library newFunctionWithName:@"red"];
        pd.colorAttachments[0].pixelFormat = color;
        pd.depthAttachmentPixelFormat = depth;
        pd.stencilAttachmentPixelFormat = depth;
        pd.rasterSampleCount = 4;
        id<MTLRenderPipelineState> state = [device newRenderPipelineStateWithDescriptor:pd
                                                                                  error:&error];
        if (state == nil) {
            printf("%s: no pipeline: %s\n", name, error.description.UTF8String);
            return 3;
        }
        MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
        pass.colorAttachments[0].texture = msaa;
        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass.colorAttachments[0].clearColor = MTLClearColorMake(0.25, 0.5, 0.75, 1.0);
        pass.colorAttachments[0].storeAction = MTLStoreActionMultisampleResolve;
        pass.colorAttachments[0].resolveTexture = resolve;
        pass.depthAttachment.texture = ds;
        pass.depthAttachment.loadAction = MTLLoadActionClear;
        pass.depthAttachment.storeAction = MTLStoreActionDontCare;
        pass.stencilAttachment.texture = ds;
        pass.stencilAttachment.loadAction = MTLLoadActionClear;
        pass.stencilAttachment.storeAction = MTLStoreActionDontCare;
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLRenderCommandEncoder> encoder = [cb renderCommandEncoderWithDescriptor:pass];
        printf("%s: encoder %s\n", name, encoder ? "made" : "nil");
        if (encoder == nil) {
            return 0;
        }
        [encoder setRenderPipelineState:state];
        [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [encoder endEncoding];
        id<MTLBuffer> out = [device newBufferWithLength:256 options:MTLResourceStorageModeShared];
        memset(out.contents, 0xee, 256);
        if (readable) {
            id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
            [blit copyFromTexture:resolve
                             sourceSlice:0
                             sourceLevel:0
                            sourceOrigin:MTLOriginMake(0, 0, 0)
                              sourceSize:MTLSizeMake(8, 8, 1)
                                toBuffer:out
                       destinationOffset:0
                  destinationBytesPerRow:32
                destinationBytesPerImage:256];
            [blit endEncoding];
        }
        [cb commit];
        [cb waitUntilCompleted];
        printf("%s: status %ld error %s\n", name, (long)cb.status,
               cb.error ? cb.error.description.UTF8String : "none");
        if (!readable) {
            return 0;
        }
        const uint8_t *got = out.contents;
        uint64_t hash = 0xcbf29ce484222325ull;
        for (NSUInteger i = 0; i < 256; i++) {
            hash ^= got[i];
            hash *= 0x100000001b3ull;
        }
        printf("%s: texels fnv %016llx", name, (unsigned long long)hash);
        const int at[4][2] = {{0, 0}, {7, 7}, {3, 4}, {4, 3}};
        for (int i = 0; i < 4; i++) {
            const uint8_t *t = got + (at[i][1] * 8 + at[i][0]) * 4;
            printf(" (%d,%d)=%02x%02x%02x%02x", at[i][0], at[i][1], t[0], t[1], t[2], t[3]);
        }
        printf("\n");
    }
    return 0;
}
