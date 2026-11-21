// Plain Metal's answer, with the API validation layer OFF, for a render pass with no attachment
// at each `renderTargetWidth`, `renderTargetHeight` and `defaultRasterSampleCount` the runtime's
// `Refusal::NoAttachmentPass` names (a side of 0, a count of 0, Metal's default), beside the legal
// neighbour of each. One shape per process, so a process that ends is attributable to the shape
// that ended it; every process prints the device and the release first.
//
// Build (no GPU needed):
//
//   clang -fobjc-arc -fmodules -framework Metal -framework Foundation \
//       -o no_attachment_passes no_attachment_passes.m
//
// Run one shape per process, WITHOUT MTL_DEBUG_LAYER:
//
//   ./no_attachment_passes <shape>
//   ./no_attachment_passes --list
//
// A shape `pass-<width>-<height>-<count>` makes the pass's encoder and ends it. A shape
// `draw-<width>-<height>-<count>-<samples>` also draws, in that pass, the fullscreen triangle
// through a pipeline state of `<samples>` raster samples and no attachment, whose fragment function
// counts its runs: the count says how many pixels and samples the pass drew at. Every shape prints
// `DEVICE <name>`, `RELEASE <os>`, then `READ <key> <value>` lines, then `DONE <shape>`. A shape
// whose process Metal ends prints no `DONE`, and its exit status is the reading.
//
// WHAT IT MEASURED, Apple M3 Max, macOS 27.0.1 (Build 26A434), job m3-max-62174:
//
//   - a new descriptor's `defaultRasterSampleCount` is 0;
//   - `pass-64-64-1` makes its encoder; `pass-0-64-1`, `pass-64-0-1` and `pass-0-0-1` make none
//     (nil), and their command buffers complete;
//   - `pass-64-64-0` and `draw-4-4-0-1` end the process at the encoder (SIGABRT, exit 134);
//   - `draw-4-4-1-1` runs its fragment function 16 times, and `draw-4-4-4-4` 64 times, once per
//     sample of each pixel, as a memory-writing function runs at 4 samples on this device.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

static id<MTLDevice> gDevice = nil;
static id<MTLCommandQueue> gQueue = nil;

static void readv(const char *key, const char *fmt, ...) {
    printf("READ %s ", key);
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    printf("\n");
    fflush(stdout);
}

static NSString *kSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "struct P { float4 position [[position]]; };\n"
     "vertex P na_cover(uint id [[vertex_id]]) {\n"
     "    float2 a = float2(float(id & 1u), float(id >> 1));\n"
     "    return P { float4(4.0 * a - 1.0, 0.0, 1.0) };\n"
     "}\n"
     "fragment void na_count(P in [[stage_in]], device atomic_uint *runs [[buffer(0)]]) {\n"
     "    atomic_fetch_add_explicit(runs, 1u, memory_order_relaxed);\n"
     "}\n";

static const char *kShapes[] = {
    "pass-64-64-1", "pass-0-64-1", "pass-64-0-1", "pass-0-0-1", "pass-64-64-0",
    "draw-4-4-1-1", "draw-4-4-4-4", "draw-4-4-0-1",
};

static int run(const char *shape) {
    unsigned long width = 0, height = 0, count = 0, samples = 0;
    BOOL draws = strncmp(shape, "draw-", 5) == 0;
    int parsed = draws ? sscanf(shape, "draw-%lu-%lu-%lu-%lu", &width, &height, &count, &samples)
                       : sscanf(shape, "pass-%lu-%lu-%lu", &width, &height, &count);
    if (parsed != (draws ? 4 : 3)) {
        fprintf(stderr, "unknown shape %s\n", shape);
        return 2;
    }
    id<MTLRenderPipelineState> state = nil;
    id<MTLBuffer> runs = nil;
    if (draws) {
        NSError *error = nil;
        id<MTLLibrary> library = [gDevice newLibraryWithSource:kSource options:nil error:&error];
        if (library == nil) {
            readv("library", "nil %s", error.localizedDescription.UTF8String);
            return 3;
        }
        MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
        d.vertexFunction = [library newFunctionWithName:@"na_cover"];
        d.fragmentFunction = [library newFunctionWithName:@"na_count"];
        d.rasterSampleCount = samples;
        state = [gDevice newRenderPipelineStateWithDescriptor:d error:&error];
        readv("state", "%s", state ? "made" : error.localizedDescription.UTF8String);
        if (state == nil) {
            return 3;
        }
        runs = [gDevice newBufferWithLength:4 options:MTLResourceStorageModeShared];
        memset(runs.contents, 0, 4);
    }
    MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
    readv("default", "defaultRasterSampleCount %lu", (unsigned long)rp.defaultRasterSampleCount);
    rp.renderTargetWidth = width;
    rp.renderTargetHeight = height;
    rp.defaultRasterSampleCount = count;
    readv("descriptor", "%lu by %lu at %lu", width, height, count);
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
    readv("encoder", "made %s", enc ? "yes" : "nil");
    if (enc != nil && draws) {
        [enc setRenderPipelineState:state];
        [enc setFragmentBuffer:runs offset:0 atIndex:0];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    readv("command-buffer", "status %ld error %s", (long)cb.status,
          cb.error ? cb.error.localizedDescription.UTF8String : "(none)");
    if (draws) {
        readv("runs", "%u", *(const unsigned *)runs.contents);
    }
    printf("DONE %s\n", shape);
    fflush(stdout);
    return 0;
}

int main(int argc, const char *argv[]) {
    if (argc == 2 && strcmp(argv[1], "--list") == 0) {
        for (size_t i = 0; i < sizeof(kShapes) / sizeof(kShapes[0]); i++) {
            printf("%s\n", kShapes[i]);
        }
        return 0;
    }
    if (argc != 2) {
        fprintf(stderr, "usage: %s <shape>|--list\n", argv[0]);
        return 2;
    }
    @autoreleasepool {
        gDevice = MTLCreateSystemDefaultDevice();
        if (gDevice == nil) {
            fprintf(stderr, "no Metal device\n");
            return 2;
        }
        gQueue = [gDevice newCommandQueue];
        printf("DEVICE %s\n", gDevice.name.UTF8String);
        printf("RELEASE %s\n", NSProcessInfo.processInfo.operatingSystemVersionString.UTF8String);
        printf("SHAPE %s\n", argv[1]);
        fflush(stdout);
        return run(argv[1]);
    }
}
