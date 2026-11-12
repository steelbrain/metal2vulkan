// What Metal answers for a kernel's `[[stage_in]]` attribute whose location carries the
// attribute's name as its user key, as Apple's OpenGL front end writes a kernel's stage-in node
// (`!"air.stage_in", !"air.location_index", i32 0, i32 1, !"attrib_0", ...`). MSL has no spelling
// for the key; the owned fixture `engine/kernel_stage_in.air.ll` is rewritten so and reassembled.
// The runtime lane's `runtime_lane::stage_in_strides` holds the answers.
//
// Build and run (no cargo target needed), with S a scratch directory and F the fixture:
//   sed -E 's/(!"air.stage_in", !"air.location_index", i32 ([0-9]+), i32 1, )/\1!"attrib_\2", /g' \
//     F.air.ll > S/keyed.air.ll
//   for f in F S/keyed; do xcrun metal-as $f.air.ll -o S/$(basename $f).air; done
//   for f in kernel_stage_in keyed; do xcrun metallib S/$f.air -o S/$f.metallib; done
//   clang -fobjc-arc -O1 -fmodules -o S/stage_in_user_key stage_in_user_key.m \
//     -framework Foundation -framework Metal
//   S/stage_in_user_key S/kernel_stage_in.metallib S/keyed.metallib
//
// Each library's `stage_in_fetch` is made with the lane's fixed-stride descriptor: a `float3` at 4
// and a `half2` at 16 of layout 1 (stride 24, by the thread's x), a `ushort4` at 0 of layout 2
// (stride 8, by its y), a `char` at 5 of layout 3 (stride 16, constant), over a 5 by 3 grid in 2 by
// 2 threadgroups, the stage-in region the grid. It prints each library's 12 words per thread and
// whether the libraries' words are equal.
//
// WHAT IT MEASURED (Apple M3 Max, macOS 27.0.1): both libraries made their pipelines and wrote the
// same words, thread t = 5y + x reading layout 1 at 24x and layout 2 at 8y (thread 1: 41300000
// 41400000 41500000 3f800000 00000000 00000064 000000c8 0000ffff fffffff9 00003c01 00004001
// 00000001), which are the words `runtime_lane::stage_in_strides` requires of the keyed form.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>

enum { W = 5, H = 3, ELEMENTS = 16, WORDS = 12 };

static void layouts(uint8_t *l1, uint8_t *l2, uint8_t *l3) {
    memset(l1, 0xaa, 24 * ELEMENTS);
    for (uint32_t e = 0; e < ELEMENTS; e++) {
        for (uint32_t k = 0; k < 3; k++) {
            float v = (float)(10 * e + k + 1);
            memcpy(l1 + 24 * e + 4 + 4 * k, &v, 4);
        }
        uint16_t h0 = 0x3c00 + e, h1 = 0x4000 + e;
        memcpy(l1 + 24 * e + 16, &h0, 2);
        memcpy(l1 + 24 * e + 18, &h1, 2);
        uint16_t u[4] = {e, e + 100, e + 200, 0xffff - e};
        memcpy(l2 + 8 * e, u, 8);
    }
    memset(l3, 0x11, 16);
    l3[5] = (uint8_t)(int8_t)-7;
}

// The words `path`'s `stage_in_fetch` writes, or 0 where Metal made no pipeline.
static int run(id<MTLDevice> d, const char *path, uint32_t *out) {
    NSError *e = nil;
    NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
    id<MTLLibrary> lib = [d newLibraryWithURL:url error:&e];
    if (!lib) {
        printf("%s: no library: %s\n", path, e.localizedDescription.UTF8String);
        return 0;
    }
    id<MTLFunction> f = [lib newFunctionWithName:@"stage_in_fetch"];
    MTLStageInputOutputDescriptor *in = [MTLStageInputOutputDescriptor stageInputOutputDescriptor];
    struct { MTLAttributeFormat format; NSUInteger offset, buffer; } attrs[4] = {
        {MTLAttributeFormatFloat3, 4, 1},
        {MTLAttributeFormatUShort4, 0, 2},
        {MTLAttributeFormatChar, 5, 3},
        {MTLAttributeFormatHalf2, 16, 1},
    };
    for (int i = 0; i < 4; i++) {
        in.attributes[i].format = attrs[i].format;
        in.attributes[i].offset = attrs[i].offset;
        in.attributes[i].bufferIndex = attrs[i].buffer;
    }
    struct { NSUInteger stride; MTLStepFunction step; NSUInteger rate; } ls[3] = {
        {24, MTLStepFunctionThreadPositionInGridX, 1},
        {8, MTLStepFunctionThreadPositionInGridY, 1},
        {16, MTLStepFunctionConstant, 0},
    };
    for (int i = 0; i < 3; i++) {
        in.layouts[i + 1].stride = ls[i].stride;
        in.layouts[i + 1].stepFunction = ls[i].step;
        in.layouts[i + 1].stepRate = ls[i].rate;
    }
    MTLComputePipelineDescriptor *pd = [MTLComputePipelineDescriptor new];
    pd.computeFunction = f;
    pd.stageInputDescriptor = in;
    id<MTLComputePipelineState> p =
        [d newComputePipelineStateWithDescriptor:pd options:0 reflection:nil error:&e];
    if (!p) {
        printf("%s: no pipeline: %s\n", path, e.localizedDescription.UTF8String);
        return 0;
    }
    uint8_t l1[24 * ELEMENTS], l2[8 * ELEMENTS], l3[16];
    layouts(l1, l2, l3);
    id<MTLBuffer> b1 = [d newBufferWithBytes:l1 length:sizeof l1 options:0];
    id<MTLBuffer> b2 = [d newBufferWithBytes:l2 length:sizeof l2 options:0];
    id<MTLBuffer> b3 = [d newBufferWithBytes:l3 length:sizeof l3 options:0];
    id<MTLBuffer> o = [d newBufferWithLength:4 * WORDS * W * H options:0];
    memset(o.contents, 0, o.length);
    id<MTLCommandQueue> q = [d newCommandQueue];
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:p];
    [enc setBuffer:o offset:0 atIndex:0];
    [enc setBuffer:b1 offset:0 atIndex:1];
    [enc setBuffer:b2 offset:0 atIndex:2];
    [enc setBuffer:b3 offset:0 atIndex:3];
    [enc setStageInRegion:MTLRegionMake3D(0, 0, 0, W, H, 1)];
    [enc dispatchThreads:MTLSizeMake(W, H, 1) threadsPerThreadgroup:MTLSizeMake(2, 2, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
        printf("%s: status %ld\n", path, (long)cb.status);
        return 0;
    }
    memcpy(out, o.contents, o.length);
    printf("%s: pipeline made\n", path);
    for (int t = 0; t < W * H; t++) {
        printf("  thread %2d:", t);
        for (int w = 0; w < WORDS; w++) printf(" %08x", out[WORDS * t + w]);
        printf("\n");
    }
    return 1;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc != 3) {
            fprintf(stderr, "usage: %s PLAIN.metallib KEYED.metallib\n", argv[0]);
            return 2;
        }
        id<MTLDevice> d = MTLCreateSystemDefaultDevice();
        printf("device: %s\n", d.name.UTF8String);
        static uint32_t plain[WORDS * W * H], keyed[WORDS * W * H];
        int a = run(d, argv[1], plain), b = run(d, argv[2], keyed);
        if (a && b) printf("words equal: %s\n", memcmp(plain, keyed, sizeof plain) ? "no" : "yes");
    }
    return 0;
}
