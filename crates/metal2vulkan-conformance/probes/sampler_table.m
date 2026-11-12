// What Metal answers for a program-scope table of constexpr samplers read at an index only the
// run knows: the owned fixture `engine/sampler_table.metal`'s `sampler_table_slot` (a texture
// slot) and `sampler_table_held` (a texture held in an argument buffer). The runtime lane's
// `runtime_lane::sampler_tables` holds the answers.
//
// Build (no cargo target needed):
//   clang -fobjc-arc -O1 -fmodules -o sampler_table sampler_table.m \
//     -framework Foundation -framework Metal
// Run: ./sampler_table <the fixture's .metal>, MTL_DEBUG_LAYER unset.
//
// The texture is 4 by 1 RGBA8Unorm: red, green, blue, white. Thread i = 4c + s samples at
// x = X[c] (texel units -0.5, 0.75, 3.75, 4.5, 2.5: outside the texture on both sides, and a
// quarter of the way between texel centers) and y = 0.5 through samplers[s] (repeat nearest,
// clamp nearest, repeat linear, clamp linear). Each kernel prints each thread's four floats as
// words.
//
// WHAT IT MEASURED (Apple M3 Max, macOS 27.0.1): see the runtime lane's `METAL`.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>

enum { THREADS = 20 };
static const float X[5] = {-0.125f, 0.1875f, 0.9375f, 1.125f, 0.625f};

static int run(id<MTLDevice> d, id<MTLLibrary> lib, NSString *name, id<MTLTexture> tex) {
    NSError *e = nil;
    id<MTLFunction> f = [lib newFunctionWithName:name];
    id<MTLComputePipelineState> p = [d newComputePipelineStateWithFunction:f error:&e];
    if (!p) {
        printf("%s: no pipeline: %s\n", name.UTF8String, e.localizedDescription.UTF8String);
        return 1;
    }
    uint32_t which[THREADS];
    float xs[THREADS];
    for (int i = 0; i < THREADS; i++) {
        which[i] = (uint32_t)(i % 4);
        xs[i] = X[i / 4];
    }
    id<MTLBuffer> out = [d newBufferWithLength:16 * THREADS options:0];
    memset(out.contents, 0x5a, out.length);
    id<MTLBuffer> w = [d newBufferWithBytes:which length:sizeof which options:0];
    id<MTLBuffer> x = [d newBufferWithBytes:xs length:sizeof xs options:0];
    MTLResourceID rid = tex.gpuResourceID;
    id<MTLBuffer> held = [d newBufferWithBytes:&rid length:sizeof rid options:0];
    id<MTLCommandQueue> q = [d newCommandQueue];
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:p];
    [enc setBuffer:out offset:0 atIndex:0];
    [enc setBuffer:w offset:0 atIndex:1];
    [enc setBuffer:x offset:0 atIndex:2];
    [enc setBuffer:held offset:0 atIndex:3];
    [enc setTexture:tex atIndex:0];
    [enc useResource:tex usage:MTLResourceUsageRead];
    [enc dispatchThreads:MTLSizeMake(THREADS, 1, 1) threadsPerThreadgroup:MTLSizeMake(THREADS, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    printf("%s: status %ld\n", name.UTF8String, (long)cb.status);
    const uint32_t *words = out.contents;
    for (int i = 0; i < THREADS; i++) {
        printf("  thread %2d:", i);
        for (int k = 0; k < 4; k++) printf(" %08x", words[4 * i + k]);
        printf("\n");
    }
    return 0;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc != 2) {
            fprintf(stderr, "usage: %s SAMPLER_TABLE.metal\n", argv[0]);
            return 2;
        }
        id<MTLDevice> d = MTLCreateSystemDefaultDevice();
        printf("device: %s\n", d.name.UTF8String);
        NSError *e = nil;
        NSString *src = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]
                                                  encoding:NSUTF8StringEncoding
                                                     error:&e];
        id<MTLLibrary> lib = src ? [d newLibraryWithSource:src options:nil error:&e] : nil;
        if (!lib) {
            printf("no library: %s\n", e.localizedDescription.UTF8String);
            return 1;
        }
        MTLTextureDescriptor *td =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                               width:4
                                                              height:1
                                                           mipmapped:NO];
        td.usage = MTLTextureUsageShaderRead;
        id<MTLTexture> tex = [d newTextureWithDescriptor:td];
        const uint8_t texels[16] = {255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255};
        [tex replaceRegion:MTLRegionMake2D(0, 0, 4, 1) mipmapLevel:0 withBytes:texels bytesPerRow:16];
        int bad = run(d, lib, @"sampler_table_slot", tex);
        bad |= run(d, lib, @"sampler_table_held", tex);
        return bad;
    }
}
