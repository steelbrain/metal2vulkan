// What plain Metal does with an ASTC texture (LDR, sRGB and HDR, ASTC's 4x2 and 8x4 footprints
// among them) as a render pass's attachment and as a kernel's write, and how its copies treat a
// level whose sides are not whole blocks: the reference the runtime's ASTC textures are read
// against where `astc_hdr_textures.m` did not reach.
//
// One case a process, named on the command line (F a raw MTLPixelFormat):
//
//   pipelines  a render pipeline whose colour attachment 0 is each ASTC format, each in a child
//              process (Metal ends the process at some): made, nil, or ENDED;
//   pass F LOAD STORE DRAW
//              a pass whose colour attachment 0 is a texture of F holding marker blocks and whose
//              attachment 1 is an RGBA8Unorm texture of the same size: attachment 0's load action
//              LOAD (clear, load, dontcare) and store action STORE (store, dontcare), attachment
//              1 cleared and stored; DRAW 1 draws a full-screen triangle through a pipeline whose
//              attachment 0 is `Invalid` and whose attachment 1 is RGBA8Unorm (its fragment
//              writes `color(1)` alone), DRAW 0 draws nothing. What the command buffer answers,
//              the blocks after, and the RGBA8 texels after;
//   write F N  a kernel writing N texels (1: the first; 0: every one) of a texture of F holding
//              marker blocks (`access::write`): what the command buffer answers and the blocks
//              after;
//   readwrite F  a kernel reading every texel of a texture of F holding marker blocks through an
//              `access::read_write` binding, then writing it, and querying its size: what it
//              reads, the size it answers, and the blocks after;
//   digest F SEED ldr|hdr ACROSS DOWN
//              the FNV-1a digest of the `float` texels a kernel reads of ACROSS x DOWN seeded
//              blocks of F (the generator of the translator's ASTC decode tests, LDR or HDR);
//   partial F  a texture of F 10 x 5 texels with 2 levels: `replaceRegion` of the whole level 0
//              (rows of whole blocks), `getBytes` of it, then `replaceRegion` and `getBytes` of a
//              region ending at the level's edge, of one inside one block, and level 1's whole
//              region; a blit from a buffer of the whole level 0 (`sourceSize` 10 x 5) and one
//              into a buffer: the bytes each moves;
//   create1 F TYPE  one 2D, 2DArray, Cube or 3D texture of F, Shared, read, write and
//              render-target usage: made or nil.
//
// WHAT IT MEASURED, Apple M3 Max, macOS 27.0.1: see the module docs of the runtime's `astc.rs`,
// which state Metal's answers.
//
// Build it, then run each case WITHOUT MTL_DEBUG_LAYER, with the Mac's GPU to itself:
//
//   clang -fobjc-arc -framework Metal -framework Foundation -o astc_passes_writes_footprints \
//     astc_passes_writes_footprints.m
//   /usr/bin/env -u MTL_DEBUG_LAYER ./astc_passes_writes_footprints pass 204 clear store 1

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
    {204, "ASTC_4x4_LDR", 4, 4},   {186, "ASTC_4x4_sRGB", 4, 4},  {222, "ASTC_4x4_HDR", 4, 4},
    {212, "ASTC_8x8_LDR", 8, 8},   {194, "ASTC_8x8_sRGB", 8, 8},  {230, "ASTC_8x8_HDR", 8, 8},
    {203, "ASTC_4x2_LDR", 4, 2},   {185, "ASTC_4x2_sRGB", 4, 2},  {221, "ASTC_4x2_HDR", 4, 2},
    {209, "ASTC_8x4_LDR", 8, 4},   {191, "ASTC_8x4_sRGB", 8, 4},  {227, "ASTC_8x4_HDR", 8, 4},
    {218, "ASTC_12x12_LDR", 12, 12},
};

#define COUNT(a) (sizeof(a) / sizeof((a)[0]))

static const Format *format_of(long raw) {
    for (size_t i = 0; i < COUNT(kFormats); i++) {
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
    "kernel void writes(texture2d<float, access::write> t [[texture(0)]],\n"
    "                   constant uint &n [[buffer(0)]],\n"
    "                   uint2 g [[thread_position_in_grid]]) {\n"
    "    if (n == 1 && (g.x != 0 || g.y != 0)) return;\n"
    "    t.write(float4(0.25, 0.5, 0.75, 1.0), g);\n"
    "}\n"
    "struct VOut { float4 position [[position]]; };\n"
    "vertex VOut vmain(uint vid [[vertex_id]]) {\n"
    "    float2 p = float2((vid << 1) & 2, vid & 2);\n"
    "    VOut o; o.position = float4(p * 2.0 - 1.0, 0.0, 1.0); return o;\n"
    "}\n"
    "struct FOut { float4 one [[color(1)]]; };\n"
    "fragment FOut fone(VOut in [[stage_in]]) { FOut o; o.one = float4(1.0, 0.0, 1.0, 1.0);\n"
    "    return o; }\n"
    "fragment float4 fzero(VOut in [[stage_in]]) { return float4(0.25, 0.5, 0.75, 1.0); }\n"
    "kernel void reads(texture2d<float> t [[texture(0)]],\n"
    "                  device float4 *out [[buffer(0)]],\n"
    "                  uint2 g [[thread_position_in_grid]]) {\n"
    "    if (g.x >= t.get_width() || g.y >= t.get_height()) return;\n"
    "    out[g.y * t.get_width() + g.x] = t.read(g);\n"
    "}\n"
    "kernel void readwrites(texture2d<float, access::read_write> t [[texture(0)]],\n"
    "                       device float4 *out [[buffer(0)]],\n"
    "                       uint2 g [[thread_position_in_grid]]) {\n"
    "    uint at = g.y * t.get_width() + g.x;\n"
    "    out[at] = t.read(g);\n"
    "    t.write(float4(0.25, 0.5, 0.75, 1.0), g);\n"
    "    if (at == 0) out[64] = float4(t.get_width(), t.get_height(), 0, 0);\n"
    "}\n";

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

static void print_bytes(const char *what, const unsigned char *p, size_t n) {
    printf("%s", what);
    for (size_t i = 0; i < n; i++) {
        printf("%s%02x", i % 16 == 0 ? " " : "", p[i]);
    }
    printf("\n");
}

static const unsigned char kMarker = 0x5a;

// A 2 x 2-block texture of `f` holding marker blocks.
static id<MTLTexture> marked(const Format *f, MTLTextureUsage usage) {
    MTLTextureDescriptor *d = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:f->raw
                                     width:2 * f->bw
                                    height:2 * f->bh
                                 mipmapped:NO];
    d.storageMode = MTLStorageModeShared;
    d.usage = usage;
    id<MTLTexture> t = [gDevice newTextureWithDescriptor:d];
    unsigned char blocks[64];
    memset(blocks, kMarker, sizeof(blocks));
    [t replaceRegion:MTLRegionMake2D(0, 0, t.width, t.height)
         mipmapLevel:0
           withBytes:blocks
         bytesPerRow:32];
    return t;
}

static void report_blocks(const char *what, const Format *f, id<MTLTexture> t) {
    unsigned char back[64];
    memset(back, 0, sizeof(back));
    [t getBytes:back bytesPerRow:32 fromRegion:MTLRegionMake2D(0, 0, t.width, t.height)
        mipmapLevel:0];
    size_t kept = 0;
    for (size_t i = 0; i < sizeof(back); i++) {
        kept += back[i] == kMarker;
    }
    printf("%s %s: %zu of 64 bytes still the marker\n", what, f->name, kept);
    if (kept != sizeof(back)) {
        print_bytes("  blocks", back, sizeof(back));
    }
}

static void finish(const char *what, id<MTLCommandBuffer> cb) {
    [cb commit];
    [cb waitUntilCompleted];
    printf("%s: status %ld error %s\n", what, (long)cb.status,
           cb.error ? cb.error.localizedDescription.UTF8String : "(none)");
}

// ---------------------------------------------------------------------------------------------

static void case_pipeline1(const Format *f) {
    id<MTLLibrary> lib = library();
    MTLRenderPipelineDescriptor *rp = [[MTLRenderPipelineDescriptor alloc] init];
    rp.vertexFunction = [lib newFunctionWithName:@"vmain"];
    rp.fragmentFunction = [lib newFunctionWithName:@"fzero"];
    rp.colorAttachments[0].pixelFormat = f->raw;
    NSError *err = nil;
    id<MTLRenderPipelineState> ps = [gDevice newRenderPipelineStateWithDescriptor:rp error:&err];
    printf("%s %s\n", ps ? "made" : "nil", err ? err.localizedDescription.UTF8String : "");
}

static void run_child(const char *self, const char *args, const char *label) {
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "'%s' %s 2>&1", self, args);
    FILE *p = popen(cmd, "r");
    char line[1024];
    char last[1024] = "";
    while (fgets(line, sizeof(line), p)) {
        if (strncmp(line, "device ", 7) != 0) {
            strncpy(last, line, sizeof(last) - 1);
        }
    }
    int status = pclose(p);
    last[strcspn(last, "\n")] = 0;
    printf("%s: %s%s\n", label, status == 0 ? "" : "ENDED: ", last);
}

static void case_pipelines(const char *self) {
    for (size_t i = 0; i < COUNT(kFormats); i++) {
        char args[64], label[128];
        snprintf(args, sizeof(args), "pipeline1 %lu", (unsigned long)kFormats[i].raw);
        snprintf(label, sizeof(label), "pipeline %s", kFormats[i].name);
        run_child(self, args, label);
    }
    static const char *const kTypeNames[] = {"2D", "2DArray", "Cube", "3D"};
    for (size_t i = 0; i < COUNT(kFormats); i++) {
        for (size_t t = 0; t < COUNT(kTypeNames); t++) {
            char args[64], label[128];
            snprintf(args, sizeof(args), "create1 %lu %zu", (unsigned long)kFormats[i].raw, t);
            snprintf(label, sizeof(label), "create %s %s", kFormats[i].name, kTypeNames[t]);
            run_child(self, args, label);
        }
    }
}

static void case_create1(const Format *f, long type) {
    MTLTextureDescriptor *d = [[MTLTextureDescriptor alloc] init];
    static const MTLTextureType kTypes[] = {MTLTextureType2D, MTLTextureType2DArray,
                                            MTLTextureTypeCube, MTLTextureType3D};
    d.textureType = kTypes[type];
    d.pixelFormat = f->raw;
    d.width = 2 * f->bw;
    d.height = d.textureType == MTLTextureTypeCube ? d.width : 2 * f->bh;
    d.depth = d.textureType == MTLTextureType3D ? 2 : 1;
    d.arrayLength = d.textureType == MTLTextureType2DArray ? 2 : 1;
    d.storageMode = MTLStorageModeShared;
    d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite |
              MTLTextureUsageRenderTarget;
    printf("%s\n", [gDevice newTextureWithDescriptor:d] ? "made" : "nil");
}

static MTLLoadAction load_of(const char *s) {
    if (strcmp(s, "clear") == 0) return MTLLoadActionClear;
    if (strcmp(s, "load") == 0) return MTLLoadActionLoad;
    return MTLLoadActionDontCare;
}

static void case_pass(const Format *f, const char *load, const char *store, int draw) {
    id<MTLTexture> t = marked(f, MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget);
    MTLTextureDescriptor *od = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                     width:t.width
                                    height:t.height
                                 mipmapped:NO];
    od.storageMode = MTLStorageModeShared;
    od.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
    id<MTLTexture> other = [gDevice newTextureWithDescriptor:od];
    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = t;
    pass.colorAttachments[0].loadAction = load_of(load);
    pass.colorAttachments[0].clearColor = MTLClearColorMake(0.5, 4.0, 8.0, 1.0);
    pass.colorAttachments[0].storeAction =
        strcmp(store, "store") == 0 ? MTLStoreActionStore : MTLStoreActionDontCare;
    pass.colorAttachments[1].texture = other;
    pass.colorAttachments[1].loadAction = MTLLoadActionClear;
    pass.colorAttachments[1].clearColor = MTLClearColorMake(0.0, 1.0, 0.0, 1.0);
    pass.colorAttachments[1].storeAction = MTLStoreActionStore;
    id<MTLRenderPipelineState> ps = nil;
    if (draw) {
        id<MTLLibrary> lib = library();
        MTLRenderPipelineDescriptor *rp = [[MTLRenderPipelineDescriptor alloc] init];
        rp.vertexFunction = [lib newFunctionWithName:@"vmain"];
        rp.fragmentFunction = [lib newFunctionWithName:@"fone"];
        rp.colorAttachments[1].pixelFormat = MTLPixelFormatRGBA8Unorm;
        NSError *err = nil;
        ps = [gDevice newRenderPipelineStateWithDescriptor:rp error:&err];
        printf("pass pipeline: %s %s\n", ps ? "made" : "nil",
               err ? err.localizedDescription.UTF8String : "");
    }
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:pass];
    printf("pass encoder: %s\n", e ? "made" : "nil");
    fflush(stdout);
    if (ps) {
        [e setRenderPipelineState:ps];
        [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }
    [e endEncoding];
    char what[128];
    snprintf(what, sizeof(what), "pass %s %s %s draw %d", f->name, load, store, draw);
    finish(what, cb);
    report_blocks("pass", f, t);
    unsigned char texels[4 * 24 * 24];
    size_t n = t.width * t.height * 4;
    [other getBytes:texels bytesPerRow:t.width * 4
         fromRegion:MTLRegionMake2D(0, 0, t.width, t.height)
        mipmapLevel:0];
    size_t green = 0, magenta = 0;
    for (size_t i = 0; i < n; i += 4) {
        green += texels[i] == 0 && texels[i + 1] == 255 && texels[i + 2] == 0;
        magenta += texels[i] == 255 && texels[i + 1] == 0 && texels[i + 2] == 255;
    }
    printf("pass other attachment: %zu green, %zu magenta of %zu texels\n", green, magenta,
           n / 4);
}

static void case_write(const Format *f, int n) {
    NSError *err = nil;
    id<MTLFunction> fn = [library() newFunctionWithName:@"writes"];
    id<MTLComputePipelineState> p = [gDevice newComputePipelineStateWithFunction:fn error:&err];
    id<MTLTexture> t = marked(f, MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:p];
    [e setTexture:t atIndex:0];
    uint32_t count = (uint32_t)n;
    [e setBytes:&count length:4 atIndex:0];
    [e dispatchThreads:MTLSizeMake(t.width, t.height, 1)
        threadsPerThreadgroup:MTLSizeMake(4, 2, 1)];
    [e endEncoding];
    char what[128];
    snprintf(what, sizeof(what), "write %s %d", f->name, n);
    finish(what, cb);
    report_blocks("write", f, t);
}

static void case_readwrite(const Format *f) {
    NSError *err = nil;
    id<MTLFunction> fn = [library() newFunctionWithName:@"readwrites"];
    id<MTLComputePipelineState> p = [gDevice newComputePipelineStateWithFunction:fn error:&err];
    printf("readwrite pipeline: %s\n", p ? "made" : err.localizedDescription.UTF8String);
    fflush(stdout);
    id<MTLTexture> t = marked(f, MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
    id<MTLBuffer> out = [gDevice newBufferWithLength:65 * 16 options:MTLResourceStorageModeShared];
    memset(out.contents, 0xee, out.length);
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:p];
    [e setTexture:t atIndex:0];
    [e setBuffer:out offset:0 atIndex:0];
    [e dispatchThreads:MTLSizeMake(t.width, t.height, 1)
        threadsPerThreadgroup:MTLSizeMake(4, 2, 1)];
    [e endEncoding];
    char what[128];
    snprintf(what, sizeof(what), "readwrite %s", f->name);
    finish(what, cb);
    const float *o = out.contents;
    printf("readwrite %s read texel 0: %g %g %g %g, texel 1: %g %g %g %g, size %g x %g\n",
           f->name, o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7], o[256], o[257]);
    report_blocks("readwrite", f, t);
}

// xorshift64* from `seed | 1`, as the translator's ASTC decode tests draw their seeded blocks.
static uint64_t gState;
static uint64_t next_draw(void) {
    gState ^= gState << 13;
    gState ^= gState >> 7;
    gState ^= gState << 17;
    return gState * 0x2545F4914F6CDD1DULL;
}

// `count` seeded blocks: the LDR tests' (1 in 8 an LDR void extent) or, `hdr`, the HDR tests' (1
// in 16 an LDR void extent, 1 in 16 an HDR one, a quarter of those with an infinite or NaN red).
static void seeded(unsigned char *out, size_t count, uint64_t seed, int hdr) {
    gState = seed | 1;
    const uint64_t extents = ((1ULL << 52) - 1) << 12;
    for (size_t b = 0; b < count; b++) {
        uint64_t low = next_draw(), high = next_draw(), k = next_draw();
        if (!hdr && k % 8 == 0) {
            low = (low & ~0xFFFULL) | 0xDFC;
            if ((k >> 8) & 1) low |= extents;
        } else if (hdr && k % 16 == 0) {
            low = (low & ~0xFFFULL) | 0xDFC | extents;
        } else if (hdr && k % 16 == 1) {
            low = (low & ~0xFFFULL) | 0xFFC | extents;
            if (((k >> 8) & 3) == 0) {
                static const uint64_t kReds[] = {0x7C00, 0xFC00, 0x7E00, 0xFFFF};
                high = (high & ~0xFFFFULL) | kReds[(k >> 10) & 3];
            }
        }
        memcpy(out + b * 16, &low, 8);
        memcpy(out + b * 16 + 8, &high, 8);
    }
}

// The FNV-1a digest of the `float` texels Metal reads of ACROSS x DOWN seeded blocks of `f`
// (`hdr` picks the generator), row-major.
static void case_digest(const Format *f, uint64_t seed, int hdr, size_t across, size_t down) {
    size_t count = across * down;
    unsigned char *blocks = malloc(count * 16);
    seeded(blocks, count, seed, hdr);
    MTLTextureDescriptor *d = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:f->raw
                                     width:across * f->bw
                                    height:down * f->bh
                                 mipmapped:NO];
    d.storageMode = MTLStorageModeShared;
    d.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> t = [gDevice newTextureWithDescriptor:d];
    [t replaceRegion:MTLRegionMake2D(0, 0, t.width, t.height) mipmapLevel:0 withBytes:blocks
         bytesPerRow:across * 16];
    size_t n = t.width * t.height;
    id<MTLBuffer> out = [gDevice newBufferWithLength:n * 16 options:MTLResourceStorageModeShared];
    NSError *err = nil;
    id<MTLComputePipelineState> p = [gDevice
        newComputePipelineStateWithFunction:[library() newFunctionWithName:@"reads"]
                                      error:&err];
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:p];
    [e setTexture:t atIndex:0];
    [e setBuffer:out offset:0 atIndex:0];
    [e dispatchThreads:MTLSizeMake(t.width, t.height, 1)
        threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
    [e endEncoding];
    finish("digest", cb);
    uint64_t h = 0xcbf29ce484222325ULL;
    const unsigned char *bytes = out.contents;
    for (size_t i = 0; i < n * 16; i++) {
        h = (h ^ bytes[i]) * 0x100000001b3ULL;
    }
    printf("digest %s seed %llu %s %zux%zu blocks: 0x%016llx\n", f->name,
           (unsigned long long)seed, hdr ? "hdr" : "ldr", across, down, (unsigned long long)h);
    free(blocks);
}

// A recognizable block: its index in every byte's high nibble, the byte's place in the low.
static void fill_blocks(unsigned char *p, size_t blocks, unsigned char base) {
    for (size_t b = 0; b < blocks; b++) {
        for (size_t i = 0; i < 16; i++) {
            p[b * 16 + i] = (unsigned char)(base + ((b & 0xf) << 4) + i);
        }
    }
}

static void get_level(id<MTLTexture> t, const Format *f, NSUInteger level, const char *what) {
    NSUInteger w = MAX(1u, t.width >> level), h = MAX(1u, t.height >> level);
    NSUInteger across = (w + f->bw - 1) / f->bw, down = (h + f->bh - 1) / f->bh;
    unsigned char back[16 * 64];
    memset(back, 0xee, sizeof(back));
    [t getBytes:back bytesPerRow:across * 16 fromRegion:MTLRegionMake2D(0, 0, w, h)
        mipmapLevel:level];
    printf("%s (level %lu, %lux%lu texels, %lux%lu blocks):\n", what, (unsigned long)level,
           (unsigned long)w, (unsigned long)h, (unsigned long)across, (unsigned long)down);
    print_bytes("  ", back, across * down * 16);
}

static void case_partial(const Format *f) {
    MTLTextureDescriptor *d = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:f->raw
                                     width:10
                                    height:5
                                 mipmapped:NO];
    d.mipmapLevelCount = 2;
    d.storageMode = MTLStorageModeShared;
    d.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> t = [gDevice newTextureWithDescriptor:d];
    NSUInteger across = (10 + f->bw - 1) / f->bw, down = (5 + f->bh - 1) / f->bh;
    printf("partial %s: 10x5, %lux%lu blocks at level 0\n", f->name, (unsigned long)across,
           (unsigned long)down);
    unsigned char blocks[16 * 64];
    fill_blocks(blocks, across * down, 0);
    [t replaceRegion:MTLRegionMake2D(0, 0, 10, 5) mipmapLevel:0 withBytes:blocks
         bytesPerRow:across * 16];
    get_level(t, f, 0, "whole level written");
    // A region ending at the level's edge: the last block across and down.
    unsigned char one[64];
    memset(one, 0xa0, 16);
    NSUInteger x = (across - 1) * f->bw, y = (down - 1) * f->bh;
    [t replaceRegion:MTLRegionMake2D(x, y, 10 - x, 5 - y) mipmapLevel:0 withBytes:one
         bytesPerRow:16];
    get_level(t, f, 0, "edge region written (the last block, 0xa0)");
    // A region inside one block, not at an edge: its first texel alone.
    memset(one, 0xb0, 16);
    [t replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:one bytesPerRow:16];
    get_level(t, f, 0, "one texel's region written (0xb0)");
    // A region of two blocks' texels less one texel across.
    memset(one, 0xc0, 32);
    [t replaceRegion:MTLRegionMake2D(0, 0, 2 * f->bw - 1, f->bh) mipmapLevel:0 withBytes:one
         bytesPerRow:32];
    get_level(t, f, 0, "two blocks less a texel written (0xc0)");
    unsigned char sub[16];
    memset(sub, 0xee, 16);
    [t getBytes:sub bytesPerRow:16 fromRegion:MTLRegionMake2D(x, y, 10 - x, 5 - y)
        mipmapLevel:0];
    print_bytes("edge region read", sub, 16);
    NSUInteger l1a = (5 + f->bw - 1) / f->bw, l1d = (2 + f->bh - 1) / f->bh;
    fill_blocks(blocks, l1a * l1d, 0x08);
    [t replaceRegion:MTLRegionMake2D(0, 0, 5, 2) mipmapLevel:1 withBytes:blocks
         bytesPerRow:l1a * 16];
    get_level(t, f, 1, "level 1 written");
    // Blits: a buffer of the whole level 0 into the texture, the texture into a buffer.
    fill_blocks(blocks, across * down, 0x01);
    id<MTLBuffer> src = [gDevice newBufferWithBytes:blocks length:across * down * 16
                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> dst = [gDevice newBufferWithLength:across * down * 16 + 64
                                             options:MTLResourceStorageModeShared];
    memset(dst.contents, 0xee, dst.length);
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
    [b copyFromBuffer:src sourceOffset:0 sourceBytesPerRow:across * 16
        sourceBytesPerImage:across * down * 16
                 sourceSize:MTLSizeMake(10, 5, 1)
                  toTexture:t
           destinationSlice:0
           destinationLevel:0
          destinationOrigin:MTLOriginMake(0, 0, 0)];
    [b copyFromTexture:t sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(x, 0, 0)
                     sourceSize:MTLSizeMake(10 - x, 5, 1)
                       toBuffer:dst
              destinationOffset:0
         destinationBytesPerRow:16
       destinationBytesPerImage:16 * down];
    [b endEncoding];
    finish("partial blits", cb);
    get_level(t, f, 0, "level 0 after the blit from a buffer (base 0x01)");
    print_bytes("last column blitted into a buffer", dst.contents, 16 * down + 16);
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 2) {
            fprintf(stderr, "usage: %s pipelines | pass F LOAD STORE DRAW | write F N | "
                            "partial F\n",
                    argv[0]);
            return 2;
        }
        gDevice = MTLCreateSystemDefaultDevice();
        gQueue = [gDevice newCommandQueue];
        printf("device %s\n", gDevice.name.UTF8String);
        const char *c = argv[1];
        if (strcmp(c, "pipelines") == 0) {
            case_pipelines(argv[0]);
        } else if (argc > 2 && strcmp(c, "pipeline1") == 0) {
            case_pipeline1(format_of(atol(argv[2])));
        } else if (argc > 3 && strcmp(c, "create1") == 0) {
            case_create1(format_of(atol(argv[2])), atol(argv[3]));
        } else if (argc > 5 && strcmp(c, "pass") == 0) {
            case_pass(format_of(atol(argv[2])), argv[3], argv[4], atoi(argv[5]));
        } else if (argc > 3 && strcmp(c, "write") == 0) {
            case_write(format_of(atol(argv[2])), atoi(argv[3]));
        } else if (argc > 6 && strcmp(c, "digest") == 0) {
            case_digest(format_of(atol(argv[2])), strtoull(argv[3], NULL, 10),
                        strcmp(argv[4], "hdr") == 0, (size_t)atol(argv[5]), (size_t)atol(argv[6]));
        } else if (argc > 2 && strcmp(c, "readwrite") == 0) {
            case_readwrite(format_of(atol(argv[2])));
        } else if (argc > 2 && strcmp(c, "partial") == 0) {
            case_partial(format_of(atol(argv[2])));
        } else {
            fprintf(stderr, "unknown case %s\n", c);
            return 2;
        }
    }
    return 0;
}
