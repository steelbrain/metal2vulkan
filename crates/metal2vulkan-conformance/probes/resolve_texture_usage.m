// Whether Metal resolves a multisample colour attachment into a texture made without
// `MTLTextureUsageRenderTarget`, into such a texture over a Shared buffer among them, and what the
// resolved texels read.
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
// Three cases take a guest's pass instead, at 600x600: a first pass clears the colour attachment
// to the clear above and its depth and stencil, and stores all three; a second pass loads all
// three, stores the depth and the stencil, draws nothing, and resolves the colour
// (MultisampleResolve) into a 1-sample BGRA8Unorm_sRGB 2D texture of ShaderRead usage alone:
//
//   over        Shared, made over a Shared buffer at offset 0 (newTextureWithDescriptor:offset:
//               bytesPerRow:), its rows the row (2,400 bytes) rounded up to the device's
//               minimumLinearTextureAlignmentForPixelFormat:, the buffer exactly its rows
//   over-padded the same at 2,560 bytes a row (the row rounded up to 256), so each row has 160
//               bytes past its texels
//   read600     Private, of its own memory: the control the two read as
//
// Before the pass the buffer's bytes are 0xee. A case prints the resolve texture's usage, whether
// the encoder was made, the command buffer's status and error, and, read back by a blit into a
// Shared buffer (BGRA bytes, rows tight), the FNV-1a 64 of its bytes and the texels at (0, 0),
// (7, 7), (3, 4) and (4, 3) (at 600x600: (0, 0), (599, 599), (299, 300) and (300, 299)). An over
// case prints, too, the alignment, the pitch and the buffer's length, what the texture answers
// of its buffer, the FNV-1a 64 of the texels read from the buffer's own bytes at its pitch,
// whether those equal the blit's, and how many bytes past the texels changed from 0xee.
//
//   xcrun clang -fobjc-arc -framework Metal -framework QuartzCore -framework Foundation \
//       -o resolve_texture_usage resolve_texture_usage.m
//   for c in rt read readwrite drawable drawable-readable control over over-padded read600; do
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

// The side of every texture of the case: 8, or 600 for a guest's pass.
static NSUInteger side = 8;

static id<MTLTexture> texture(id<MTLDevice> device, MTLPixelFormat format, NSUInteger samples,
                              MTLTextureUsage usage) {
    MTLTextureDescriptor *d = [MTLTextureDescriptor new];
    d.textureType = samples > 1 ? MTLTextureType2DMultisample : MTLTextureType2D;
    d.pixelFormat = format;
    d.width = side;
    d.height = side;
    d.sampleCount = samples;
    d.usage = usage;
    d.storageMode = MTLStorageModePrivate;
    return [device newTextureWithDescriptor:d];
}

static uint64_t fnv(uint64_t hash, const uint8_t *bytes, NSUInteger length) {
    for (NSUInteger i = 0; i < length; i++) {
        hash ^= bytes[i];
        hash *= 0x100000001b3ull;
    }
    return hash;
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
        id<MTLBuffer> backing = nil;
        NSUInteger pitch = 0;
        BOOL readable = YES;
        BOOL guest = NO;
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
        } else if (strcmp(name, "read600") == 0) {
            side = 600;
            guest = YES;
            resolve = texture(device, color, 1, MTLTextureUsageShaderRead);
        } else if (strcmp(name, "over") == 0 || strcmp(name, "over-padded") == 0) {
            side = 600;
            guest = YES;
            NSUInteger align = [device minimumLinearTextureAlignmentForPixelFormat:color];
            NSUInteger row = side * 4;
            pitch = strcmp(name, "over") == 0 ? (row + align - 1) / align * align
                                               : (row + 255) / 256 * 256;
            backing = [device newBufferWithLength:pitch * side
                                          options:MTLResourceStorageModeShared];
            memset(backing.contents, 0xee, backing.length);
            MTLTextureDescriptor *d = [MTLTextureDescriptor new];
            d.textureType = MTLTextureType2D;
            d.pixelFormat = color;
            d.width = side;
            d.height = side;
            d.usage = MTLTextureUsageShaderRead;
            d.storageMode = MTLStorageModeShared;
            resolve = [backing newTextureWithDescriptor:d offset:0 bytesPerRow:pitch];
            printf("%s: minimum linear alignment %lu, bytes per row %lu, buffer %lu bytes\n", name,
                   (unsigned long)align, (unsigned long)pitch, (unsigned long)backing.length);
            if (resolve == nil) {
                printf("%s: no texture over the buffer\n", name);
                return 0;
            }
            printf("%s: texture's buffer %s, offset %lu, bytes per row %lu, storage %lu\n", name,
                   resolve.buffer == backing ? "the buffer" : "another",
                   (unsigned long)resolve.bufferOffset, (unsigned long)resolve.bufferBytesPerRow,
                   (unsigned long)resolve.storageMode);
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
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
        pass.colorAttachments[0].texture = msaa;
        pass.colorAttachments[0].clearColor = MTLClearColorMake(0.25, 0.5, 0.75, 1.0);
        pass.depthAttachment.texture = ds;
        pass.stencilAttachment.texture = ds;
        if (guest) {
            // The first pass: every attachment cleared and stored.
            pass.colorAttachments[0].loadAction = MTLLoadActionClear;
            pass.colorAttachments[0].storeAction = MTLStoreActionStore;
            pass.depthAttachment.loadAction = MTLLoadActionClear;
            pass.depthAttachment.storeAction = MTLStoreActionStore;
            pass.stencilAttachment.loadAction = MTLLoadActionClear;
            pass.stencilAttachment.storeAction = MTLStoreActionStore;
            id<MTLRenderCommandEncoder> first = [cb renderCommandEncoderWithDescriptor:pass];
            if (first == nil) {
                printf("%s: first encoder nil\n", name);
                return 0;
            }
            [first endEncoding];
            // The guest's pass: every attachment loaded, the colour resolved.
            pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
            pass.depthAttachment.loadAction = MTLLoadActionLoad;
            pass.stencilAttachment.loadAction = MTLLoadActionLoad;
        } else {
            pass.colorAttachments[0].loadAction = MTLLoadActionClear;
            pass.depthAttachment.loadAction = MTLLoadActionClear;
            pass.depthAttachment.storeAction = MTLStoreActionDontCare;
            pass.stencilAttachment.loadAction = MTLLoadActionClear;
            pass.stencilAttachment.storeAction = MTLStoreActionDontCare;
        }
        pass.colorAttachments[0].storeAction = MTLStoreActionMultisampleResolve;
        pass.colorAttachments[0].resolveTexture = resolve;
        id<MTLRenderCommandEncoder> encoder = [cb renderCommandEncoderWithDescriptor:pass];
        printf("%s: encoder %s\n", name, encoder ? "made" : "nil");
        if (encoder == nil) {
            return 0;
        }
        if (!guest) {
            [encoder setRenderPipelineState:state];
            [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        }
        [encoder endEncoding];
        const NSUInteger row = side * 4;
        const NSUInteger bytes = row * side;
        id<MTLBuffer> out = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        memset(out.contents, 0xee, bytes);
        if (readable) {
            id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
            [blit copyFromTexture:resolve
                             sourceSlice:0
                             sourceLevel:0
                            sourceOrigin:MTLOriginMake(0, 0, 0)
                              sourceSize:MTLSizeMake(side, side, 1)
                                toBuffer:out
                       destinationOffset:0
                  destinationBytesPerRow:row
                destinationBytesPerImage:bytes];
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
        printf("%s: texels fnv %016llx", name,
               (unsigned long long)fnv(0xcbf29ce484222325ull, got, bytes));
        const int last = (int)side - 1;
        const int half = (int)side / 2;
        const int at[4][2] = {{0, 0}, {last, last}, {half - 1, half}, {half, half - 1}};
        for (int i = 0; i < 4; i++) {
            const uint8_t *t = got + (at[i][1] * side + at[i][0]) * 4;
            printf(" (%d,%d)=%02x%02x%02x%02x", at[i][0], at[i][1], t[0], t[1], t[2], t[3]);
        }
        printf("\n");
        if (backing != nil) {
            const uint8_t *held = backing.contents;
            uint64_t hash = 0xcbf29ce484222325ull;
            BOOL same = YES;
            NSUInteger changed = 0;
            for (NSUInteger y = 0; y < side; y++) {
                hash = fnv(hash, held + y * pitch, row);
                same = same && memcmp(held + y * pitch, got + y * row, row) == 0;
                for (NSUInteger x = row; x < pitch; x++) {
                    changed += held[y * pitch + x] != 0xee;
                }
            }
            printf("%s: buffer's texels fnv %016llx, %s the blit's; %lu of %lu bytes past the "
                   "texels changed\n",
                   name, (unsigned long long)hash, same ? "equal to" : "NOT equal to",
                   (unsigned long)changed, (unsigned long)((pitch - row) * side));
        }
    }
    return 0;
}
