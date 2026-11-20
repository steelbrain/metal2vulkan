// Two measurements the conformance port of `rotated_textures` and `rotated_targets` turns on.
//
//   1. **The tile rule at rotation 4.** A pass that loads a rotated frame and draws only its
//      top-left corner is said to leave the loaded texels mirrored left to right in each 32x32 tile
//      the draw reaches, and in place in the tiles it does not reach. A 64x16 frame is two tiles
//      wide, so it separates the two; the 8x4 and 32x32 frames the other probe and the lane use are
//      one tile and cannot. A runtime that mirrors every tile differs from Metal only here, so
//      the comparison needs Metal's own answer.
//
//   2. **ARGB8Unorm (format 84) as a render target.** The lane `rotated_targets` checks our
//      ARGB8Unorm target against our *own* RGBA8Unorm target with the bytes rotated, never against
//      Metal. This asks Metal directly: the same destination texels and the same drawn values into
//      a target of each format, byte for byte, unblended and under one blend.
//
// Nothing here writes outside a texture's or a surface's own bytes, so no step can fault the GPU.
//
// clang -fobjc-arc -framework Metal -framework Foundation -framework IOSurface \
//     -o rotated_tiles_and_argb rotated_tiles_and_argb.m

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>

@interface MTLTextureDescriptor (M2VPrivateRotation)
- (void)setRotation:(NSUInteger)rotation;
- (void)setFramebufferOnly:(BOOL)framebufferOnly;
@end

static const uint8_t kPad = 0xAB;

static NSString *const kSource =
    @"#include <metal_stdlib>\n"
    @"using namespace metal;\n"
    @"struct Out { float4 position [[position]]; };\n"
    @"vertex Out tob_rect(uint vid [[vertex_id]], constant float4 &corners [[buffer(0)]]) {\n"
    @"  float2 p[6] = { float2(corners.x, corners.y), float2(corners.z, corners.y),\n"
    @"                  float2(corners.x, corners.w), float2(corners.z, corners.y),\n"
    @"                  float2(corners.z, corners.w), float2(corners.x, corners.w) };\n"
    @"  Out o; o.position = float4(p[vid], 0.0, 1.0); return o;\n"
    @"}\n"
    @"fragment float4 tob_name(Out in [[stage_in]]) {\n"
    @"  float x = floor(in.position.x), y = floor(in.position.y);\n"
    @"  return float4(0x55 / 255.0, y / 255.0, x / 255.0, 1.0);\n"
    @"}\n"
    // The target fill: one float4 per texel out of a buffer, indexed by the pass position, so the
    // drawn value of a texel is exactly the code the host put there.
    @"fragment float4 rt_fill(Out in [[stage_in]], constant float4 *values [[buffer(0)]],\n"
    @"                        constant uint &width [[buffer(1)]]) {\n"
    @"  uint x = uint(floor(in.position.x)), y = uint(floor(in.position.y));\n"
    @"  return values[y * width + x];\n"
    @"}\n";

static IOSurfaceRef makeSurface(size_t w, size_t h, size_t bpr) {
    NSDictionary *properties = @{
        (__bridge NSString *)kIOSurfaceWidth : @(w),
        (__bridge NSString *)kIOSurfaceHeight : @(h),
        (__bridge NSString *)kIOSurfaceBytesPerElement : @(4),
        (__bridge NSString *)kIOSurfaceBytesPerRow : @(bpr),
        (__bridge NSString *)kIOSurfacePixelFormat : @((uint32_t)'BGRA'),
    };
    IOSurfaceRef surface = IOSurfaceCreate((__bridge CFDictionaryRef)properties);
    if (surface == NULL) { return NULL; }
    IOSurfaceLock(surface, 0, NULL);
    memset(IOSurfaceGetBaseAddress(surface), kPad, IOSurfaceGetAllocSize(surface));
    IOSurfaceUnlock(surface, 0, NULL);
    return surface;
}

int main(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) { fprintf(stderr, "no Metal device\n"); return 2; }
        printf("DEVICE %s\n", device.name.UTF8String);
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:kSource options:nil error:&error];
        if (library == nil) { fprintf(stderr, "library: %s\n", error.description.UTF8String); return 2; }
        id<MTLCommandQueue> queue = [device newCommandQueue];

        // ---- 1. the tile rule at rotation 4, on a frame two 32x32 tiles wide ----
        {
            MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
            pd.vertexFunction = [library newFunctionWithName:@"tob_rect"];
            pd.fragmentFunction = [library newFunctionWithName:@"tob_name"];
            pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
            id<MTLRenderPipelineState> pipeline =
                [device newRenderPipelineStateWithDescriptor:pd error:&error];
            if (pipeline == nil) { fprintf(stderr, "pipeline: %s\n", error.description.UTF8String); return 2; }

            // 64x16 is two tiles wide and half a tile tall; the turned plane is 16x64, which is one
            // tile wide and two tall.
            const size_t w = 64, h = 16;
            for (NSUInteger rotation = 1; rotation <= 4; rotation += 3) {   // 1, then 4
                const size_t sw = h, sh = w;                 // the quarter turn's plane
                size_t bpr = sw * 4;
                if (bpr < w * 4) { bpr = w * 4; }            // the stride rule: at least w * 4
                IOSurfaceRef surface = makeSurface(sw, sh, bpr);
                if (surface == NULL) { printf("TILE rotation %lu: no surface\n", (unsigned long)rotation); continue; }
                MTLTextureDescriptor *d =
                    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                      width:w height:h mipmapped:NO];
                d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
                d.storageMode = MTLStorageModeShared;
                [d setFramebufferOnly:YES];
                [d setRotation:rotation];
                id<MTLTexture> texture = [device newTextureWithDescriptor:d iosurface:surface plane:0];
                if (texture == nil) { printf("TILE rotation %lu: nil texture\n", (unsigned long)rotation);
                                      CFRelease(surface); continue; }

                // Every plane texel names its own stored position.
                IOSurfaceLock(surface, 0, NULL);
                uint8_t *bytes = (uint8_t *)IOSurfaceGetBaseAddress(surface);
                for (size_t sy = 0; sy < sh; sy += 1) {
                    for (size_t sx = 0; sx < sw; sx += 1) {
                        uint8_t *at = bytes + sy * bpr + sx * 4;
                        at[0] = (uint8_t)sx; at[1] = (uint8_t)sy; at[2] = 0x33; at[3] = 0x77;
                    }
                }
                IOSurfaceUnlock(surface, 0, NULL);

                // Load, and draw only the top-left 8x4 pass texels.
                const size_t px = 8, py = 4;
                float part[4] = { -1.0f, 1.0f - 2.0f * (float)py / (float)h,
                                  -1.0f + 2.0f * (float)px / (float)w, 1.0f };
                id<MTLBuffer> corners = [device newBufferWithBytes:part length:sizeof(part)
                                                          options:MTLResourceStorageModeShared];
                MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
                rp.colorAttachments[0].texture = texture;
                rp.colorAttachments[0].loadAction = MTLLoadActionLoad;
                rp.colorAttachments[0].storeAction = MTLStoreActionStore;
                id<MTLCommandBuffer> commands = [queue commandBuffer];
                id<MTLRenderCommandEncoder> e = [commands renderCommandEncoderWithDescriptor:rp];
                [e setRenderPipelineState:pipeline];
                [e setVertexBuffer:corners offset:0 atIndex:0];
                [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
                [e endEncoding];
                [commands commit];
                [commands waitUntilCompleted];
                if (commands.status != MTLCommandBufferStatusCompleted) {
                    printf("TILE rotation %lu: status %lu\n", (unsigned long)rotation,
                           (unsigned long)commands.status);
                    CFRelease(surface); continue;
                }

                // Classify every plane texel: drawn, kept in place, mirrored left to right across
                // the whole plane width, or something else. Counted per 32x32 tile of the plane.
                IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL);
                bytes = (uint8_t *)IOSurfaceGetBaseAddress(surface);
                printf("TILE rotation %lu plane %zux%zu stride %zu, drawn %zux%zu pass texels\n",
                       (unsigned long)rotation, sw, sh, bpr, px, py);
                for (size_t ty = 0; ty * 32 < sh; ty += 1) {
                    for (size_t tx = 0; tx * 32 < sw; tx += 1) {
                        size_t drawn = 0, kept = 0, mirrored = 0, other = 0;
                        for (size_t sy = ty * 32; sy < sh && sy < (ty + 1) * 32; sy += 1) {
                            for (size_t sx = tx * 32; sx < sw && sx < (tx + 1) * 32; sx += 1) {
                                const uint8_t *at = bytes + sy * bpr + sx * 4;
                                if (at[2] == 0x55 && at[3] == 0xff) { drawn += 1; }
                                else if (at[0] == (uint8_t)sx && at[1] == (uint8_t)sy) { kept += 1; }
                                else if (at[0] == (uint8_t)(sw - 1 - sx) && at[1] == (uint8_t)sy) { mirrored += 1; }
                                else { other += 1; }
                            }
                        }
                        printf("  plane tile (%zu,%zu): drawn %zu kept %zu mirrored %zu other %zu\n",
                               tx, ty, drawn, kept, mirrored, other);
                    }
                }
                IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
                CFRelease(surface);
            }
        }

        // ---- 2. ARGB8Unorm (84) against RGBA8Unorm (70), byte for byte ----
        {
            const size_t w = 16, h = 4, count = w * h;
            // Destination codes and drawn values: fixed, so the two formats are asked the same
            // question. The drawn values include ties at half a code and values outside [0, 1],
            // which is where a rounding rule shows.
            uint8_t destination[64 * 4];
            float values[64 * 4];
            for (size_t i = 0; i < count; i += 1) {
                for (size_t c = 0; c < 4; c += 1) {
                    destination[i * 4 + c] = (uint8_t)((i * 7 + c * 61) % 256);
                    const size_t pick = (i + c) % 5;
                    float v;
                    if (pick == 0) { v = (float)((i * 11 + c) % 256) / 255.0f; }
                    else if (pick == 1) { v = ((float)((i * 3 + c) % 255) + 0.5f) / 255.0f; }
                    else if (pick == 2) { v = 1.25f; }
                    else if (pick == 3) { v = -0.25f; }
                    else { v = (float)((i * 29 + c * 7) % 1000) / 1000.0f; }
                    values[i * 4 + c] = v;
                }
            }
            const MTLPixelFormat formats[2] = { (MTLPixelFormat)70, (MTLPixelFormat)84 };
            const char *names[2] = { "RGBA8Unorm", "ARGB8Unorm" };
            // Unblended, then one straight-alpha blend, which is where a rotated image's alpha
            // channel would show up in a colour factor.
            for (int blended = 0; blended <= 1; blended += 1) {
                for (int at = 0; at < 2; at += 1) {
                    MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
                    pd.vertexFunction = [library newFunctionWithName:@"tob_rect"];
                    pd.fragmentFunction = [library newFunctionWithName:@"rt_fill"];
                    pd.colorAttachments[0].pixelFormat = formats[at];
                    if (blended) {
                        pd.colorAttachments[0].blendingEnabled = YES;
                        pd.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
                        pd.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
                        pd.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorSourceAlpha;
                        pd.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
                    }
                    id<MTLRenderPipelineState> pipeline =
                        [device newRenderPipelineStateWithDescriptor:pd error:&error];
                    if (pipeline == nil) {
                        printf("ARGB %s blended %d: pipeline refused: %s\n", names[at], blended,
                               error.localizedDescription.UTF8String);
                        continue;
                    }
                    MTLTextureDescriptor *d =
                        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:formats[at]
                                                                          width:w height:h mipmapped:NO];
                    d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
                    d.storageMode = MTLStorageModeShared;
                    id<MTLTexture> target = [device newTextureWithDescriptor:d];
                    if (target == nil) { printf("ARGB %s: nil texture\n", names[at]); continue; }
                    // ARGB8Unorm holds an RGBA8Unorm texel's bytes rotated, alpha first, so the
                    // same destination goes in rotated for it.
                    uint8_t rotated[64 * 4];
                    for (size_t i = 0; i < count; i += 1) {
                        if (at == 0) { memcpy(rotated + i * 4, destination + i * 4, 4); }
                        else {
                            rotated[i * 4 + 0] = destination[i * 4 + 3];
                            rotated[i * 4 + 1] = destination[i * 4 + 0];
                            rotated[i * 4 + 2] = destination[i * 4 + 1];
                            rotated[i * 4 + 3] = destination[i * 4 + 2];
                        }
                    }
                    [target replaceRegion:MTLRegionMake2D(0, 0, w, h)
                              mipmapLevel:0
                                withBytes:rotated
                              bytesPerRow:w * 4];
                    float whole[4] = { -1.0f, -1.0f, 1.0f, 1.0f };
                    id<MTLBuffer> corners = [device newBufferWithBytes:whole length:sizeof(whole)
                                                              options:MTLResourceStorageModeShared];
                    id<MTLBuffer> table = [device newBufferWithBytes:values
                                                             length:count * 16
                                                            options:MTLResourceStorageModeShared];
                    uint32_t width32 = (uint32_t)w;
                    id<MTLBuffer> widthBuffer = [device newBufferWithBytes:&width32 length:4
                                                                  options:MTLResourceStorageModeShared];
                    MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
                    rp.colorAttachments[0].texture = target;
                    rp.colorAttachments[0].loadAction = MTLLoadActionLoad;
                    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
                    id<MTLCommandBuffer> commands = [queue commandBuffer];
                    id<MTLRenderCommandEncoder> e = [commands renderCommandEncoderWithDescriptor:rp];
                    [e setRenderPipelineState:pipeline];
                    [e setVertexBuffer:corners offset:0 atIndex:0];
                    [e setFragmentBuffer:table offset:0 atIndex:0];
                    [e setFragmentBuffer:widthBuffer offset:0 atIndex:1];
                    [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
                    [e endEncoding];
                    [commands commit];
                    [commands waitUntilCompleted];
                    if (commands.status != MTLCommandBufferStatusCompleted) {
                        printf("ARGB %s blended %d: status %lu\n", names[at], blended,
                               (unsigned long)commands.status);
                        continue;
                    }
                    uint8_t got[64 * 4];
                    [target getBytes:got bytesPerRow:w * 4
                          fromRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0];
                    printf("ARGB %s blended %d:", names[at], blended);
                    for (size_t i = 0; i < count; i += 1) {
                        printf(" %02x%02x%02x%02x", got[i * 4], got[i * 4 + 1],
                               got[i * 4 + 2], got[i * 4 + 3]);
                    }
                    printf("\n");
                }
            }
        }
        printf("DONE\n");
    }
    return 0;
}
