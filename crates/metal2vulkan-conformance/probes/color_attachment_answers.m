// What plain Metal answers for each pipeline the compile layer refuses as
// `DescriptorError::ColorAttachment`: the state (CREATED), nil with an NSError (NIL, its domain,
// code and text), or the process ends (Metal's validation asserts; the caller sees the exit status
// and the assertion's text on stderr).
//
//   color_attachment_answers <shape>    one shape a process: an assertion ends it.
//   color_attachment_answers list       the shapes.
//
// Each shape makes one pipeline and draws nothing: a render pipeline of a full-screen-triangle
// vertex function and the fragment function `f` its shape names, or a tile pipeline of its tile
// function `t`, over the color attachments it names (RGBA8Unorm unless it says otherwise). `-fetch`
// shapes' functions also read the attachment they write (`[[color(0)]]` input).
//
//   bfloat-blend[-bare|-fetch]  RGBA16Bfloat, blending on, a `float4` output (none: no fragment
//                               function).
//   dual-no-second[-fetch]      a blend of the second source (Source1Color over
//                               OneMinusSource1Color, Source1Alpha over OneMinusSource1Alpha), a
//                               function writing `[[color(0)]]` alone.
//   dual-second-only            the same blend, a function writing `[[color(0), index(1)]]` alone.
//   dual-both                   the same blend, a function writing both (the control).
//   half-into-uint              a `half4` output into RGBA8Uint.
//   float-into-uint[-fetch]     a `float4` output into RGBA8Uint (fetch: reading it `uint4`).
//   uint-into-float             a `uint4` output into RGBA8Unorm.
//   int-into-uint               an `int4` output into RGBA8Uint.
//   uint-into-sint              a `uint4` output into RGBA8Sint.
//   fewer-components[-fetch]    a `float2` output into RGBA8Unorm.
//   float-into-float            a `float4` output (the control).
//   format-no-output            attachments 0 and 1, a function writing 0 alone.
//   a2c-uint[-no-raster]        alpha to coverage over RGBA8Uint written by a `uint4` output
//                               (rasterization off).
//   input-unattached            a function reading `[[color(1)]]`, no attachment 1.
//   input-uint-of-float         a function reading `uint4 [[color(0)]]` of RGBA8Unorm.
//   input-uint-of-xr            the same read of BGR10_XR.
//   input-uint-of-a8            the same read of A8Unorm.
//   tile-unformatted            a tile function whose implicit imageblock's member is
//                               `half4 [[color(1)]]`, no format at attachment 1.
//   tile-uint-of-float          a tile function whose member `uint4 [[color(0)]]` is over
//                               RGBA8Unorm.
//   tile-control                a tile function of a `half4 [[color(0)]]` member.
//
// Run it WITHOUT MTL_DEBUG_LAYER for plain Metal's answers.
//
//   clang -O2 -fobjc-arc -framework Metal -framework Foundation -o color_attachment_answers \
//       color_attachment_answers.m
//
// Nothing here can fault a GPU: no command buffer is committed.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>

static NSString *const kHeader = @"#include <metal_stdlib>\n"
                                  "using namespace metal;\n"
                                  "struct V { float4 p [[position]]; };\n"
                                  "vertex V vtx(uint id [[vertex_id]]) {\n"
                                  "    float2 a = float2(float(id & 1u), float(id >> 1));\n"
                                  "    return V{ float4(4.0 * a - 1.0, 0.0, 1.0) };\n"
                                  "}\n";

typedef struct {
    const char *name;
    const char *function;  // the fragment or tile function's source, named `f` or `t`
    BOOL tile;
    BOOL bare;  // no fragment function
    MTLPixelFormat formats[2];
    BOOL blended;
    BOOL dual;
    BOOL alphaToCoverage;
    BOOL noRasterization;
} Shape;

// RGBA16Bfloat's raw value (macOS 27's `MTLPixelFormatRGBA16Bfloat`), so the probe builds against
// an SDK without the name.
#define RGBA16BF ((MTLPixelFormat)116)
#define NONE MTLPixelFormatInvalid
#define UNORM MTLPixelFormatRGBA8Unorm
#define UINT MTLPixelFormatRGBA8Uint
#define FLOAT4 "fragment float4 f(V in [[stage_in]]) { return float4(0.25, 0.5, 0.75, 1.0); }"
#define UINT4 "fragment uint4 f(V in [[stage_in]]) { return uint4(1, 2, 3, 4); }"
#define TILE(member, body)                                                                      \
    "struct B { " member " };\n"                                                               \
    "kernel void t(imageblock<B, imageblock_layout_implicit> ib,\n"                            \
    "              ushort2 tid [[thread_position_in_threadgroup]]) {\n"                         \
    "    B b = ib.read(tid); " body " ib.write(b, tid);\n"                                    \
    "}"

static const Shape kShapes[] = {
    {.name = "bfloat-blend", .function = FLOAT4, .formats = {RGBA16BF, NONE}, .blended = YES},
    {.name = "bfloat-blend-bare", .function = FLOAT4, .bare = YES, .formats = {RGBA16BF, NONE},
     .blended = YES},
    {.name = "bfloat-blend-fetch",
     .function = "fragment float4 f(V in [[stage_in]], float4 c0 [[color(0)]]) { return c0 * 0.5; }",
     .formats = {RGBA16BF, NONE}, .blended = YES},
    {.name = "dual-no-second", .function = FLOAT4, .formats = {UNORM, NONE}, .blended = YES,
     .dual = YES},
    {.name = "dual-no-second-fetch",
     .function = "fragment float4 f(V in [[stage_in]], float4 c0 [[color(0)]]) { return c0 * 0.5; }",
     .formats = {UNORM, NONE}, .blended = YES, .dual = YES},
    {.name = "dual-second-only",
     .function = "struct O { float4 b [[color(0), index(1)]]; };\n"
                 "fragment O f(V in [[stage_in]]) { return O{ float4(0.5) }; }",
     .formats = {UNORM, NONE}, .blended = YES, .dual = YES},
    {.name = "dual-both",
     .function = "struct O { float4 a [[color(0), index(0)]]; float4 b [[color(0), index(1)]]; };\n"
                 "fragment O f(V in [[stage_in]]) { return O{ float4(0.25), float4(0.5) }; }",
     .formats = {UNORM, NONE}, .blended = YES, .dual = YES},
    {.name = "half-into-uint",
     .function = "fragment half4 f(V in [[stage_in]]) { return half4(0.25h, 0.5h, 0.75h, 1.0h); }",
     .formats = {UINT, NONE}},
    {.name = "float-into-uint", .function = FLOAT4, .formats = {UINT, NONE}},
    {.name = "uint-into-float", .function = UINT4, .formats = {UNORM, NONE}},
    {.name = "int-into-uint",
     .function = "fragment int4 f(V in [[stage_in]]) { return int4(1, 2, 3, 4); }",
     .formats = {UINT, NONE}},
    {.name = "uint-into-sint", .function = UINT4, .formats = {MTLPixelFormatRGBA8Sint, NONE}},
    {.name = "float-into-uint-fetch",
     .function = "fragment float4 f(V in [[stage_in]], uint4 c0 [[color(0)]]) { return float4(c0); }",
     .formats = {UINT, NONE}},
    {.name = "fewer-components",
     .function = "fragment float2 f(V in [[stage_in]]) { return float2(0.25, 0.5); }",
     .formats = {UNORM, NONE}},
    {.name = "fewer-components-fetch",
     .function = "fragment float2 f(V in [[stage_in]], float4 c0 [[color(0)]]) { return c0.xy; }",
     .formats = {UNORM, NONE}},
    {.name = "float-into-float", .function = FLOAT4, .formats = {UNORM, NONE}},
    {.name = "format-no-output", .function = FLOAT4, .formats = {UNORM, UNORM}},
    {.name = "a2c-uint", .function = UINT4, .formats = {UINT, NONE}, .alphaToCoverage = YES},
    {.name = "a2c-uint-no-raster", .function = UINT4, .formats = {UINT, NONE},
     .alphaToCoverage = YES, .noRasterization = YES},
    {.name = "input-unattached",
     .function = "fragment float4 f(V in [[stage_in]], float4 c1 [[color(1)]]) { return c1; }",
     .formats = {UNORM, NONE}},
    {.name = "input-uint-of-float",
     .function = "fragment float4 f(V in [[stage_in]], uint4 c0 [[color(0)]]) { return float4(c0) / 255.0; }",
     .formats = {UNORM, NONE}},
    {.name = "input-uint-of-xr",
     .function = "fragment float4 f(V in [[stage_in]], uint4 c0 [[color(0)]]) { return float4(c0) / 255.0; }",
     .formats = {MTLPixelFormatBGR10_XR, NONE}},
    {.name = "input-uint-of-a8",
     .function = "fragment float4 f(V in [[stage_in]], uint4 c0 [[color(0)]]) { return float4(c0) / 255.0; }",
     .formats = {MTLPixelFormatA8Unorm, NONE}},
    {.name = "tile-unformatted", .function = TILE("half4 c1 [[color(1)]];", "b.c1 += 1.0h;"),
     .tile = YES, .formats = {UNORM, NONE}},
    {.name = "tile-uint-of-float", .function = TILE("uint4 c0 [[color(0)]];", "b.c0 += 1u;"),
     .tile = YES, .formats = {UNORM, NONE}},
    {.name = "tile-control", .function = TILE("half4 c0 [[color(0)]];", "b.c0 += 1.0h;"),
     .tile = YES, .formats = {UNORM, NONE}},
};

static void answer(const char *shape, id state, NSError *error) {
    if (state) {
        printf("SHAPE %s: CREATED\n", shape);
    } else {
        printf("SHAPE %s: NIL domain=%s code=%ld text=%s\n", shape, error.domain.UTF8String,
               (long)error.code, (error.localizedDescription ?: @"(none)").UTF8String);
    }
    fflush(stdout);
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc != 2) {
            fprintf(stderr, "usage: %s <shape>|list\n", argv[0]);
            return 2;
        }
        size_t count = sizeof kShapes / sizeof kShapes[0];
        if (strcmp(argv[1], "list") == 0) {
            for (size_t i = 0; i < count; i++) printf("%s\n", kShapes[i].name);
            return 0;
        }
        const Shape *s = NULL;
        for (size_t i = 0; i < count; i++) {
            if (strcmp(argv[1], kShapes[i].name) == 0) s = &kShapes[i];
        }
        if (!s) {
            fprintf(stderr, "no shape %s\n", argv[1]);
            return 2;
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSString *source = [kHeader stringByAppendingFormat:@"%s\n", s->function];
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source
                                                      options:[MTLCompileOptions new]
                                                        error:&error];
        if (!library) {
            printf("SHAPE %s: COMPILE-ERROR %s\n", s->name,
                   error.localizedDescription.UTF8String);
            return 0;
        }
        const MTLPixelFormat *formats = s->formats;
        printf("SHAPE %s: making (device %s)\n", s->name, device.name.UTF8String);
        fflush(stdout);
        if (s->tile) {
            MTLTileRenderPipelineDescriptor *td = [MTLTileRenderPipelineDescriptor new];
            td.tileFunction = [library newFunctionWithName:@"t"];
            for (int i = 0; i < 2; i++) td.colorAttachments[i].pixelFormat = formats[i];
            td.threadgroupSizeMatchesTileSize = YES;
            id state = [device newRenderPipelineStateWithTileDescriptor:td
                                                                options:MTLPipelineOptionNone
                                                             reflection:nil
                                                                  error:&error];
            answer(s->name, state, error);
            return 0;
        }
        MTLRenderPipelineDescriptor *rp = [MTLRenderPipelineDescriptor new];
        rp.vertexFunction = [library newFunctionWithName:@"vtx"];
        if (!s->bare) rp.fragmentFunction = [library newFunctionWithName:@"f"];
        for (int i = 0; i < 2; i++) rp.colorAttachments[i].pixelFormat = formats[i];
        rp.alphaToCoverageEnabled = s->alphaToCoverage;
        rp.rasterizationEnabled = !s->noRasterization;
        MTLRenderPipelineColorAttachmentDescriptor *c = rp.colorAttachments[0];
        c.blendingEnabled = s->blended;
        if (s->dual) {
            c.sourceRGBBlendFactor = MTLBlendFactorSource1Color;
            c.destinationRGBBlendFactor = MTLBlendFactorOneMinusSource1Color;
            c.sourceAlphaBlendFactor = MTLBlendFactorSource1Alpha;
            c.destinationAlphaBlendFactor = MTLBlendFactorOneMinusSource1Alpha;
        }
        id state = [device newRenderPipelineStateWithDescriptor:rp error:&error];
        answer(s->name, state, error);
    }
    return 0;
}
