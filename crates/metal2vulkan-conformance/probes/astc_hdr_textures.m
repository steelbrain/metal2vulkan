// What plain Metal does with ASTC HDR textures (MTLPixelFormat 221-236, the `_HDR` forms of every
// footprint), the reference the runtime's ASTC HDR textures and the translator's
// `bind::AstcDecode` HDR decode are read against.
//
// One case a process, named on the command line:
//
//   create    which descriptors make a texture, each HDR format by texture type, storage mode and
//             usage (`ShaderRead`, `ShaderWrite`, `RenderTarget`, all three);
//   reads F BLOCKS ACROSS DOWN OUT [VIEW]
//             the texels of a texture of raw format F, ACROSS x DOWN blocks of the file BLOCKS
//             (16 bytes each, row-major) written by `replaceRegion`, read through a view in raw
//             format VIEW where given: each texel's `read` as `float4`, its `read` as `half4`
//             and its `sample` at its centre (nearest), written to OUT as three arrays of `float4`
//             (the `half`s widened), each row-major;
//   zero F    what a new texture's texels read (its zero blocks);
//   bytes F   `getBytes` of blocks `replaceRegion` wrote, and of blocks a blit from a buffer wrote,
//             blitted back into a buffer;
//   render F  a render pipeline whose colour attachment is F, and a pass storing a clear and a
//             draw into a texture of F: what each answers and what its blocks hold after;
//   write F   a kernel writing every texel of a texture of F (`access::write`): what its blocks
//             hold after.
//
// WHAT IT MEASURED, Apple M3 Max, macOS 27.0.1: see the module docs of the translator's
// `bind::astc_decode` and the runtime's `astc.rs`, which state Metal's answers.
//
// Build outside the Mac's GPU lock, then run each case WITHOUT MTL_DEBUG_LAYER on it:
//
//   clang -fobjc-arc -framework Metal -framework Foundation -o astc_hdr_textures \
//     astc_hdr_textures.m
//   maclock /usr/bin/env -u MTL_DEBUG_LAYER ./astc_hdr_textures create

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static id<MTLDevice> gDevice;
static id<MTLCommandQueue> gQueue;

typedef struct {
    MTLPixelFormat raw;
    const char *name;
    unsigned bw, bh;
} Format;

static const Format kFormats[] = {
    {221, "ASTC_4x2_HDR", 4, 2},    {222, "ASTC_4x4_HDR", 4, 4},
    {223, "ASTC_5x4_HDR", 5, 4},    {224, "ASTC_5x5_HDR", 5, 5},
    {225, "ASTC_6x5_HDR", 6, 5},    {226, "ASTC_6x6_HDR", 6, 6},
    {227, "ASTC_8x4_HDR", 8, 4},    {228, "ASTC_8x5_HDR", 8, 5},
    {229, "ASTC_8x6_HDR", 8, 6},    {230, "ASTC_8x8_HDR", 8, 8},
    {231, "ASTC_10x5_HDR", 10, 5},  {232, "ASTC_10x6_HDR", 10, 6},
    {233, "ASTC_10x8_HDR", 10, 8},  {234, "ASTC_10x10_HDR", 10, 10},
    {235, "ASTC_12x10_HDR", 12, 10}, {236, "ASTC_12x12_HDR", 12, 12},
    // The LDR and sRGB forms, for views and for LDR blocks read as LDR.
    {204, "ASTC_4x4_LDR", 4, 4},    {210, "ASTC_8x5_LDR", 8, 5},
    {212, "ASTC_8x8_LDR", 8, 8},    {218, "ASTC_12x12_LDR", 12, 12},
    {186, "ASTC_4x4_sRGB", 4, 4},   {194, "ASTC_8x8_sRGB", 8, 8},
};

static const Format *format_of(long raw) {
    for (size_t i = 0; i < sizeof(kFormats) / sizeof(kFormats[0]); i++) {
        if ((long)kFormats[i].raw == raw) {
            return &kFormats[i];
        }
    }
    fprintf(stderr, "unknown format %ld\n", raw);
    exit(2);
}

static const char *const kShaders =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void reads(texture2d<float> t [[texture(0)]],\n"
    "                  texture2d<half> h [[texture(1)]],\n"
    "                  device float4 *out [[buffer(0)]],\n"
    "                  constant uint2 &size [[buffer(1)]],\n"
    "                  uint2 g [[thread_position_in_grid]]) {\n"
    "    if (g.x >= size.x || g.y >= size.y) return;\n"
    "    constexpr sampler s(filter::nearest, coord::normalized, address::clamp_to_edge);\n"
    "    uint at = g.y * size.x + g.x;\n"
    "    uint n = size.x * size.y;\n"
    "    out[at] = t.read(g);\n"
    "    out[n + at] = float4(h.read(g));\n"
    "    out[2 * n + at] = t.sample(s, (float2(g) + 0.5) / float2(size));\n"
    "}\n"
    "kernel void writes(texture2d<float, access::write> t [[texture(0)]],\n"
    "                   uint2 g [[thread_position_in_grid]]) {\n"
    "    t.write(float4(0.25, 2.0, 16.0, 1.0), g);\n"
    "}\n"
    "struct VOut { float4 position [[position]]; };\n"
    "vertex VOut vmain(uint vid [[vertex_id]]) {\n"
    "    float2 p = float2((vid << 1) & 2, vid & 2);\n"
    "    VOut o; o.position = float4(p * 2.0 - 1.0, 0.0, 1.0); return o;\n"
    "}\n"
    "fragment float4 fmain(VOut in [[stage_in]]) { return float4(0.25, 2.0, 16.0, 1.0); }\n";

static id<MTLLibrary> library(void) {
    NSError *err = nil;
    id<MTLLibrary> lib =
        [gDevice newLibraryWithSource:[NSString stringWithUTF8String:kShaders]
                              options:nil
                                error:&err];
    if (lib == nil) {
        fprintf(stderr, "library: %s\n", err.localizedDescription.UTF8String);
        exit(1);
    }
    return lib;
}

static id<MTLComputePipelineState> kernel_named(const char *name) {
    NSError *err = nil;
    id<MTLFunction> f = [library() newFunctionWithName:[NSString stringWithUTF8String:name]];
    id<MTLComputePipelineState> p = [gDevice newComputePipelineStateWithFunction:f error:&err];
    if (p == nil) {
        fprintf(stderr, "pipeline %s: %s\n", name, err.localizedDescription.UTF8String);
        exit(1);
    }
    return p;
}

static MTLTextureDescriptor *desc2d(const Format *f, NSUInteger across, NSUInteger down,
                                    MTLStorageMode mode, MTLTextureUsage usage) {
    MTLTextureDescriptor *d = [[MTLTextureDescriptor alloc] init];
    d.textureType = MTLTextureType2D;
    d.pixelFormat = f->raw;
    d.width = across * f->bw;
    d.height = down * f->bh;
    d.storageMode = mode;
    d.usage = usage;
    return d;
}

static void print_bytes(const char *what, const unsigned char *p, size_t n) {
    printf("%s", what);
    for (size_t i = 0; i < n; i++) {
        printf("%s%02x", i % 16 == 0 ? " " : "", p[i]);
    }
    printf("\n");
}

// ---------------------------------------------------------------------------------------------
// create

static const struct {
    MTLTextureType type;
    const char *name;
} kTypes[] = {
    {MTLTextureType1D, "1D"},
    {MTLTextureType1DArray, "1DArray"},
    {MTLTextureType2D, "2D"},
    {MTLTextureType2DArray, "2DArray"},
    {MTLTextureType2DMultisample, "2DMS"},
    {MTLTextureType2DMultisampleArray, "2DMSArray"},
    {MTLTextureTypeCube, "Cube"},
    {MTLTextureTypeCubeArray, "CubeArray"},
    {MTLTextureType3D, "3D"},
};
static const struct {
    MTLStorageMode mode;
    const char *name;
} kModes[] = {
    {MTLStorageModeShared, "Shared"},
    {MTLStorageModePrivate, "Private"},
    {MTLStorageModeMemoryless, "Memoryless"},
};
static const struct {
    MTLTextureUsage usage;
    const char *name;
} kUsages[] = {
    {MTLTextureUsageShaderRead, "Read"},
    {MTLTextureUsageShaderWrite, "Write"},
    {MTLTextureUsageRenderTarget, "RenderTarget"},
    {MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsageRenderTarget,
     "Read|Write|RenderTarget"},
    {MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView, "Read|PixelFormatView"},
};

#define COUNT(a) (sizeof(a) / sizeof((a)[0]))

// One descriptor, in a process of its own: Metal ends the process at some.
static void case_create1(size_t fi, size_t ti, size_t mi, size_t ui) {
    const Format *f = &kFormats[fi];
    MTLTextureType t = kTypes[ti].type;
    BOOL one_d = t == MTLTextureType1D || t == MTLTextureType1DArray;
    BOOL cube = t == MTLTextureTypeCube || t == MTLTextureTypeCubeArray;
    MTLTextureDescriptor *d = [[MTLTextureDescriptor alloc] init];
    d.textureType = t;
    d.pixelFormat = f->raw;
    d.width = 2 * f->bw * (cube ? 6 : 1);
    d.height = one_d ? 1 : cube ? d.width : 2 * f->bh;
    d.depth = t == MTLTextureType3D ? 2 : 1;
    d.arrayLength = (t == MTLTextureType1DArray || t == MTLTextureType2DArray ||
                     t == MTLTextureTypeCubeArray || t == MTLTextureType2DMultisampleArray)
                        ? 2
                        : 1;
    d.sampleCount = (t == MTLTextureType2DMultisample || t == MTLTextureType2DMultisampleArray)
                        ? 4
                        : 1;
    d.storageMode = kModes[mi].mode;
    d.usage = kUsages[ui].usage;
    id<MTLTexture> tex = [gDevice newTextureWithDescriptor:d];
    printf("%s\n", tex ? "made" : "nil");
}

static void case_create(const char *self) {
    for (size_t fi = 0; fi < 16; fi++) {
        for (size_t ti = 0; ti < COUNT(kTypes); ti++) {
            for (size_t mi = 0; mi < COUNT(kModes); mi++) {
                for (size_t ui = 0; ui < COUNT(kUsages); ui++) {
                    char cmd[512];
                    snprintf(cmd, sizeof(cmd), "'%s' create1 %zu %zu %zu %zu 2>&1", self, fi, ti,
                             mi, ui);
                    FILE *p = popen(cmd, "r");
                    char line[4096] = "";
                    char last[4096] = "(nothing)";
                    while (fgets(line, sizeof(line), p) != NULL) {
                        line[strcspn(line, "\n")] = 0;
                        if (line[0] != 0 && strncmp(line, "device ", 7) != 0) {
                            snprintf(last, sizeof(last), "%s", line);
                        }
                    }
                    int status = pclose(p);
                    printf("create %s %s %s %s: %s%s\n", kFormats[fi].name, kTypes[ti].name,
                           kModes[mi].name, kUsages[ui].name, status == 0 ? "" : "ENDED: ",
                           last);
                    fflush(stdout);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// reads

static id<MTLBuffer> read_texels(id<MTLTexture> t, NSUInteger w, NSUInteger h) {
    id<MTLComputePipelineState> p = kernel_named("reads");
    NSUInteger n = w * h;
    id<MTLBuffer> out = [gDevice newBufferWithLength:3 * n * 16
                                             options:MTLResourceStorageModeShared];
    memset(out.contents, 0xcd, 3 * n * 16);
    uint32_t size[2] = {(uint32_t)w, (uint32_t)h};
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:p];
    [e setTexture:t atIndex:0];
    [e setTexture:t atIndex:1];
    [e setBuffer:out offset:0 atIndex:0];
    [e setBytes:size length:sizeof(size) atIndex:1];
    [e dispatchThreads:MTLSizeMake(w, h, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    printf("reads status %ld error %s\n", (long)cb.status,
           cb.error ? cb.error.localizedDescription.UTF8String : "(none)");
    return out;
}

static void case_reads(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr, "reads F BLOCKS ACROSS DOWN OUT [VIEW]\n");
        exit(2);
    }
    const Format *f = format_of(atol(argv[2]));
    NSUInteger across = (NSUInteger)atol(argv[4]);
    NSUInteger down = (NSUInteger)atol(argv[5]);
    FILE *in = fopen(argv[3], "rb");
    size_t bytes = across * down * 16;
    unsigned char *blocks = malloc(bytes);
    if (in == NULL || fread(blocks, 1, bytes, in) != bytes) {
        fprintf(stderr, "cannot read %zu bytes of %s\n", bytes, argv[3]);
        exit(1);
    }
    fclose(in);
    MTLTextureUsage usage = MTLTextureUsageShaderRead;
    if (argc > 7) {
        usage |= MTLTextureUsagePixelFormatView;
    }
    id<MTLTexture> t = [gDevice newTextureWithDescriptor:desc2d(f, across, down,
                                                                MTLStorageModeShared, usage)];
    if (t == nil) {
        printf("reads %s: no texture\n", f->name);
        return;
    }
    [t replaceRegion:MTLRegionMake2D(0, 0, t.width, t.height)
         mipmapLevel:0
           withBytes:blocks
         bytesPerRow:across * 16];
    if (argc > 7) {
        const Format *v = format_of(atol(argv[7]));
        id<MTLTexture> view = [t newTextureViewWithPixelFormat:v->raw];
        printf("view %s as %s: %s\n", f->name, v->name, view ? "made" : "nil");
        if (view == nil) {
            return;
        }
        t = view;
    }
    id<MTLBuffer> out = read_texels(t, t.width, t.height);
    size_t n = 3 * t.width * t.height * 16;
    FILE *o = fopen(argv[6], "wb");
    if (o == NULL || fwrite(out.contents, 1, n, o) != n) {
        fprintf(stderr, "cannot write %s\n", argv[6]);
        exit(1);
    }
    fclose(o);
    printf("reads %s %lux%lu blocks: done\n", f->name, (unsigned long)across,
           (unsigned long)down);
}

// ---------------------------------------------------------------------------------------------
// zero, bytes

static void case_zero(const Format *f) {
    id<MTLTexture> t = [gDevice newTextureWithDescriptor:desc2d(f, 1, 1, MTLStorageModePrivate,
                                                                MTLTextureUsageShaderRead)];
    id<MTLBuffer> out = read_texels(t, t.width, t.height);
    const float *all = (const float *)out.contents;
    size_t n = t.width * t.height;
    float v[12];
    for (int k = 0; k < 3; k++) {
        memcpy(v + 4 * k, all + 4 * n * (size_t)k, 16);
    }
    for (int k = 0; k < 3; k++) {
        uint32_t bits[4];
        memcpy(bits, v + 4 * k, 16);
        printf("zero %s %s: %g %g %g %g (%08x %08x %08x %08x)\n", f->name,
               k == 0 ? "read" : k == 1 ? "half" : "sample", v[4 * k], v[4 * k + 1],
               v[4 * k + 2], v[4 * k + 3], bits[0], bits[1], bits[2], bits[3]);
    }
}

static void case_bytes(const Format *f) {
    unsigned char blocks[4 * 16];
    for (int i = 0; i < 64; i++) {
        blocks[i] = (unsigned char)(i * 37 + 11);
    }
    id<MTLTexture> t = [gDevice newTextureWithDescriptor:desc2d(f, 2, 2, MTLStorageModeShared,
                                                                MTLTextureUsageShaderRead)];
    [t replaceRegion:MTLRegionMake2D(0, 0, t.width, t.height)
         mipmapLevel:0
           withBytes:blocks
         bytesPerRow:32];
    unsigned char back[64];
    memset(back, 0, sizeof(back));
    [t getBytes:back bytesPerRow:32 fromRegion:MTLRegionMake2D(0, 0, t.width, t.height)
        mipmapLevel:0];
    printf("bytes %s replaceRegion/getBytes: %s\n", f->name,
           memcmp(blocks, back, 64) ? "differ" : "same");
    if (memcmp(blocks, back, 64)) {
        print_bytes("  wrote", blocks, 64);
        print_bytes("  read ", back, 64);
    }
    id<MTLTexture> p = [gDevice newTextureWithDescriptor:desc2d(f, 2, 2, MTLStorageModePrivate,
                                                                MTLTextureUsageShaderRead)];
    id<MTLBuffer> src = [gDevice newBufferWithBytes:blocks
                                             length:64
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> dst = [gDevice newBufferWithLength:64 options:MTLResourceStorageModeShared];
    memset(dst.contents, 0, 64);
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
    [b copyFromBuffer:src
               sourceOffset:0
          sourceBytesPerRow:32
        sourceBytesPerImage:64
                 sourceSize:MTLSizeMake(p.width, p.height, 1)
                  toTexture:p
           destinationSlice:0
           destinationLevel:0
          destinationOrigin:MTLOriginMake(0, 0, 0)];
    [b copyFromTexture:p
                     sourceSlice:0
                     sourceLevel:0
                    sourceOrigin:MTLOriginMake(0, 0, 0)
                      sourceSize:MTLSizeMake(p.width, p.height, 1)
                        toBuffer:dst
               destinationOffset:0
          destinationBytesPerRow:32
        destinationBytesPerImage:64];
    [b endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    printf("bytes %s blit in and out: status %ld %s\n", f->name, (long)cb.status,
           memcmp(blocks, dst.contents, 64) ? "differ" : "same");
}

// ---------------------------------------------------------------------------------------------
// render, write

static const unsigned char kMarker = 0x5a;

static void report_blocks(const char *what, const Format *f, id<MTLTexture> t, size_t blocks) {
    unsigned char back[4 * 16];
    memset(back, 0, sizeof(back));
    [t getBytes:back bytesPerRow:32 fromRegion:MTLRegionMake2D(0, 0, t.width, t.height)
        mipmapLevel:0];
    size_t kept = 0;
    for (size_t i = 0; i < blocks * 16; i++) {
        kept += back[i] == kMarker;
    }
    printf("%s %s: %zu of %zu bytes still the marker\n", what, f->name, kept, blocks * 16);
    if (kept != blocks * 16) {
        print_bytes("  blocks", back, blocks * 16);
    }
}

static id<MTLTexture> marked(const Format *f, MTLTextureUsage usage) {
    id<MTLTexture> t =
        [gDevice newTextureWithDescriptor:desc2d(f, 2, 2, MTLStorageModeShared, usage)];
    unsigned char blocks[64];
    memset(blocks, kMarker, sizeof(blocks));
    [t replaceRegion:MTLRegionMake2D(0, 0, t.width, t.height)
         mipmapLevel:0
           withBytes:blocks
         bytesPerRow:32];
    return t;
}

static void case_render(const Format *f) {
    // The pass alone first: Metal ends the process at a pipeline of the format.
    id<MTLTexture> t = marked(f, MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget);
    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = t;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(0.5, 4.0, 8.0, 1.0);
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:pass];
    printf("render %s clear-pass encoder: %s\n", f->name, e ? "made" : "nil");
    fflush(stdout);
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    printf("render %s clear-pass: status %ld error %s\n", f->name, (long)cb.status,
           cb.error ? cb.error.localizedDescription.UTF8String : "(none)");
    report_blocks("render clear", f, t, 4);
    fflush(stdout);
    id<MTLLibrary> lib = library();
    MTLRenderPipelineDescriptor *rp = [[MTLRenderPipelineDescriptor alloc] init];
    rp.vertexFunction = [lib newFunctionWithName:@"vmain"];
    rp.fragmentFunction = [lib newFunctionWithName:@"fmain"];
    rp.colorAttachments[0].pixelFormat = f->raw;
    NSError *err = nil;
    id<MTLRenderPipelineState> ps = [gDevice newRenderPipelineStateWithDescriptor:rp error:&err];
    printf("render %s pipeline: %s %s\n", f->name, ps ? "made" : "nil",
           err ? err.localizedDescription.UTF8String : "");
}

static void case_write(const Format *f) {
    id<MTLComputePipelineState> p = kernel_named("writes");
    id<MTLTexture> t = marked(f, MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:p];
    [e setTexture:t atIndex:0];
    [e dispatchThreads:MTLSizeMake(t.width, t.height, 1)
        threadsPerThreadgroup:MTLSizeMake(4, 4, 1)];
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    printf("write %s: status %ld error %s\n", f->name, (long)cb.status,
           cb.error ? cb.error.localizedDescription.UTF8String : "(none)");
    report_blocks("write", f, t, 4);
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 2) {
            fprintf(stderr, "usage: %s create | reads ... | zero F | bytes F | render F | "
                            "write F\n",
                    argv[0]);
            return 2;
        }
        gDevice = MTLCreateSystemDefaultDevice();
        gQueue = [gDevice newCommandQueue];
        printf("device %s\n", gDevice.name.UTF8String);
        const char *c = argv[1];
        if (strcmp(c, "create") == 0) {
            case_create(argv[0]);
        } else if (argc > 5 && strcmp(c, "create1") == 0) {
            case_create1((size_t)atol(argv[2]), (size_t)atol(argv[3]), (size_t)atol(argv[4]),
                         (size_t)atol(argv[5]));
        } else if (strcmp(c, "reads") == 0) {
            case_reads(argc, argv);
        } else if (argc > 2 && strcmp(c, "zero") == 0) {
            case_zero(format_of(atol(argv[2])));
        } else if (argc > 2 && strcmp(c, "bytes") == 0) {
            case_bytes(format_of(atol(argv[2])));
        } else if (argc > 2 && strcmp(c, "render") == 0) {
            case_render(format_of(atol(argv[2])));
        } else if (argc > 2 && strcmp(c, "write") == 0) {
            case_write(format_of(atol(argv[2])));
        } else {
            fprintf(stderr, "unknown case %s\n", c);
            return 2;
        }
    }
    return 0;
}
