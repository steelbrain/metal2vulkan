// Which of two rows' bytes stand where an IOSurface plane's GPU rows share them.
//
// Metal's GPU reads and writes a texture over an IOSurface plane from the plane's offset rounded
// down to 16, its rows `bytesPerRow` rounded down to 16 apart. Where that pitch is below the
// texture's own row, consecutive rows share 1 to 15 bytes, and a write of every row writes each
// shared byte twice. Two recorded claims disagree about which write stands: `blit_plan::RowsCopy`
// says the later row's and `linear_plan.rs` says the earlier row's.
//
// The probe answers it per command kind, per degree of overlap, in one process a case so a race
// shows as disagreement between processes rather than as one averaged answer:
//
//   in.<n>    copyFromBuffer:toTexture: over the whole plane
//   t2t.<n>   copyFromTexture:toTexture: from a plain texture over the whole plane
//   kern.<n>  a kernel writing every texel
//   out.<n>   copyFromTexture:toBuffer: out of the plane (which bytes a GPU read takes)
//   kern2.<n> the same kernel at a 32 x 1 threadgroup instead of 8 x 4
//   rot.<n>   copyFromBuffer:toTexture: into a framebuffer-only rotated texture whose plane
//             holds the quarter turn, the shape `blit_plan::RowsCopy` names
//   cpu.<n>   replaceRegion: of the whole texture, the CPU's write
//   none.<n>  no GPU work at all: the readback's sensitivity control
//   facts     every configuration's surface and texture facts, no GPU work
//
// Every source byte is distinct (`1 + y * width + x`, at most 188 for the widest case), so a
// plane byte names the row and column it came from instead of being inferred; the pad is 0xFF,
// which no source byte takes. Nothing writes outside the plane's own bytes.
//
// MEASURED, on an Apple M3 Max under macOS 27.0.1, no validation layer, one case a process and
// 1 to 23 fresh processes a case (the sweep's own records). Which row's bytes stand is not one
// rule: it is the order Metal writes the rows in, and that order follows the command, the GPU
// pitch and the row count, and for one shape it is not the same from run to run.
//
//   copyFromBuffer:toTexture: at a GPU pitch of 16 -- rows of 17, 20, 24 and 31 bytes sharing 1,
//   4, 8 and 15, over 2 to 12 rows, R8Unorm and RGBA8Unorm alike -- keeps the earlier row's bytes
//   at the boundary below an even row and the later row's below an odd one, strictly alternating
//   (e l e l ...), the same in every process (5 to 12 a case). At a pitch of 32 it keeps the
//   earlier row's at every boundary over 4 rows (33x4, 41x4, 47x4) and the later row's at every
//   boundary over 3, 5 and 6 rows (33x3, 33x5, 33x6, 47x5).
//
//   copyFromTexture:toTexture: answers the same in every one of those shapes but 33 bytes a row
//   at a pitch of 32 over 4 rows, where 23 fresh processes gave three whole-plane answers: every
//   boundary the earlier row's 12 times, every boundary the later row's 10 times, and once
//   later-earlier-earlier. The two stable answers are the two orders the other shapes show, so
//   there the engine picks between them run by run.
//
//   A kernel writing every texel keeps the earlier row's within the first threadgroup's rows and
//   the later row's past them at an 8 x 4 threadgroup, and a different set at 32 x 1 (20x4 and
//   31x4 e e e -> e l l, 20x8 e e e l l l l -> l e l e l e l): the winner moves with the dispatch,
//   not with Metal.
//
//   A framebuffer-only rotated texture (rotation 1) over a plane holding the quarter turn at a
//   pitch off 16 -- the one shape whose GPU rows can share bytes -- answers exactly as the plain
//   plane does: e l e at a pitch of 16 over 4 rows, e e e at 32. It does not keep the later row's
//   bytes.
//
//   A texture wider than its plane, over a plane whose pitch is a multiple of 16 (configurations
//   20 to 28, three processes a case and kind, beside their twins of the same GPU pitch and row
//   at a pitch off 16, 1, 2, 3, 12 and 6): Metal makes it, and copyFromTexture:toBuffer: reads
//   row y whole from plane byte pitch * y. Its GPU writes give the shared bytes the answers the
//   twins give: copyFromBuffer:toTexture: and copyFromTexture:toTexture: e l e ... at a pitch of
//   16 over 4 and 8 rows, R8Unorm and RGBA8Unorm alike; the kernel at 8 x 4 the earlier row's
//   within the first threadgroup's rows and the later row's past them, and at 32 x 1 another
//   set, alike in every case but one of three 32 x 1 runs over 20x8, which kept l l l e l e l
//   where its twin and the other two runs kept l e l e l e l. Past the twins' reach, at a pitch
//   of 32 copyFromBuffer:toTexture: keeps the earlier row's over 33x4, whose rows share 1 byte,
//   and the later row's over 48x4 and 64x3, whose rows share 16 and 32, so how far wider rows
//   overlap does change the order; copyFromTexture:toTexture: answers as it does there but over
//   33x4, where three processes gave three answers (l l l, l e e, e e e). Where
//   three rows share a byte (40x4 at a pitch of 16) the kernel at 8 x 4 keeps the earliest row's
//   everywhere, at 32 x 1 the latest row's everywhere, and the blits mix all three rows.
//   replaceRegion writes the rows in order: the latest row's bytes stand wherever rows meet, in
//   all nine configurations and every process (the twins' CPU rows lie a row apart and share
//   nothing).
//
// Controls: with the blit left out every shared byte reads the pad (6 processes, and one for each
// wider configuration); at a pitch of 32 over a 20-byte row nothing is shared and the rows land
// at the pitch; copyFromTexture:toBuffer: reads row y from plane byte pitch * y, so the sharing is
// real; and no run dirtied a byte past the written span.
//
// clang -fobjc-arc -framework Metal -framework Foundation -framework IOSurface \
//     -o overlapping_plane_row_writes overlapping_plane_row_writes.m

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>

// The private descriptor setters a rotated texture needs, both checked with
// `respondsToSelector:` before any use.
@interface MTLTextureDescriptor (M2VPrivateRotation)
- (void)setRotation:(NSUInteger)rotation;
- (void)setFramebufferOnly:(BOOL)framebufferOnly;
@end

// The byte every plane starts as, which no source byte takes.
static const uint8_t kPad = 0xFF;

// A configuration: the texture's format and extent, the plane's pitch, and the plane's width in
// elements where it is narrower than the texture (0: the texture's own width).
typedef struct {
    const char *name;
    MTLPixelFormat format;
    size_t texelBytes;
    size_t width;
    size_t height;
    size_t bytesPerRow;
    size_t planeWidth;
} Config;

// Rows of 17 to 47 bytes sharing 1, 4, 8 and 15 bytes at each of the two pitches 16 and 32, one
// configuration whose shared bytes are a whole 4-byte texel, one of two rows, and the
// non-overlapping neighbour (pitch 32 over a 20-byte row), whose rows must land at the pitch,
// then the same overlaps over 2, 3, 5, 6 and 8 rows, which is what tells a rule from an order.
static const Config kConfigs[] = {
    { "r8.17x4.bpr17", MTLPixelFormatR8Unorm, 1, 17, 4, 17 },
    { "r8.20x4.bpr20", MTLPixelFormatR8Unorm, 1, 20, 4, 20 },
    { "r8.24x4.bpr24", MTLPixelFormatR8Unorm, 1, 24, 4, 24 },
    { "r8.31x4.bpr31", MTLPixelFormatR8Unorm, 1, 31, 4, 31 },
    { "r8.33x4.bpr33", MTLPixelFormatR8Unorm, 1, 33, 4, 33 },
    { "r8.47x4.bpr47", MTLPixelFormatR8Unorm, 1, 47, 4, 47 },
    { "rgba8.5x4.bpr20", MTLPixelFormatRGBA8Unorm, 4, 5, 4, 20 },
    { "r8.31x2.bpr31", MTLPixelFormatR8Unorm, 1, 31, 2, 31 },
    { "r8.20x4.bpr32", MTLPixelFormatR8Unorm, 1, 20, 4, 32 },
    { "r8.20x2.bpr20", MTLPixelFormatR8Unorm, 1, 20, 2, 20 },
    { "r8.20x3.bpr20", MTLPixelFormatR8Unorm, 1, 20, 3, 20 },
    { "r8.20x5.bpr20", MTLPixelFormatR8Unorm, 1, 20, 5, 20 },
    { "r8.20x8.bpr20", MTLPixelFormatR8Unorm, 1, 20, 8, 20 },
    { "r8.17x8.bpr17", MTLPixelFormatR8Unorm, 1, 17, 8, 17 },
    { "r8.33x6.bpr33", MTLPixelFormatR8Unorm, 1, 33, 6, 33 },
    { "r8.33x3.bpr33", MTLPixelFormatR8Unorm, 1, 33, 3, 33 },
    { "r8.33x5.bpr33", MTLPixelFormatR8Unorm, 1, 33, 5, 33 },
    { "r8.41x4.bpr41", MTLPixelFormatR8Unorm, 1, 41, 4, 41 },
    { "r8.47x5.bpr47", MTLPixelFormatR8Unorm, 1, 47, 5, 47 },
    { "r8.20x12.bpr20", MTLPixelFormatR8Unorm, 1, 20, 12, 20 },
    // A texture wider than its plane, over a plane whose pitch is a multiple of 16: its rows
    // `bytesPerRow` apart on the GPU and the CPU alike, sharing 1 to 24 bytes, a byte of
    // `r8.40x4.w16` shared by three rows and every byte of `r8.64x3.w32` past the first pitch by
    // two.
    { "r8.20x4.w16", MTLPixelFormatR8Unorm, 1, 20, 4, 16, 16 },
    { "r8.24x4.w16", MTLPixelFormatR8Unorm, 1, 24, 4, 16, 16 },
    { "r8.31x4.w16", MTLPixelFormatR8Unorm, 1, 31, 4, 16, 16 },
    { "r8.20x8.w16", MTLPixelFormatR8Unorm, 1, 20, 8, 16, 16 },
    { "r8.40x4.w16", MTLPixelFormatR8Unorm, 1, 40, 4, 16, 16 },
    { "r8.33x4.w32", MTLPixelFormatR8Unorm, 1, 33, 4, 32, 32 },
    { "r8.48x4.w32", MTLPixelFormatR8Unorm, 1, 48, 4, 32, 32 },
    { "r8.64x3.w32", MTLPixelFormatR8Unorm, 1, 64, 3, 32, 32 },
    { "rgba8.5x4.w4", MTLPixelFormatRGBA8Unorm, 4, 5, 4, 16, 4 },
};
static const size_t kConfigCount = sizeof(kConfigs) / sizeof(kConfigs[0]);

static NSString *const kKernelSource =
    @"#include <metal_stdlib>\n"
    @"using namespace metal;\n"
    @"kernel void name_r8(texture2d<float, access::write> t [[texture(0)]],\n"
    @"                    constant uint &w [[buffer(0)]],\n"
    @"                    uint2 gid [[thread_position_in_grid]]) {\n"
    @"  float v = float(1 + gid.y * w + gid.x) / 255.0;\n"
    @"  t.write(float4(v, v, v, v), gid);\n"
    @"}\n";

// A one-plane IOSurface of `w x h` elements of `texelBytes`, its rows `bpr` bytes apart, every
// byte kPad. The plane is given explicitly so IOSurface keeps a pitch off 16.
static IOSurfaceRef makeSurface(size_t w, size_t h, size_t texelBytes, size_t bpr) {
    NSDictionary *plane = @{
        (__bridge NSString *)kIOSurfacePlaneWidth : @(w),
        (__bridge NSString *)kIOSurfacePlaneHeight : @(h),
        (__bridge NSString *)kIOSurfacePlaneBytesPerElement : @(texelBytes),
        (__bridge NSString *)kIOSurfacePlaneBytesPerRow : @(bpr),
        (__bridge NSString *)kIOSurfacePlaneOffset : @(0),
        (__bridge NSString *)kIOSurfacePlaneSize : @(bpr * h),
    };
    NSDictionary *properties = @{
        (__bridge NSString *)kIOSurfaceWidth : @(w),
        (__bridge NSString *)kIOSurfaceHeight : @(h),
        (__bridge NSString *)kIOSurfaceBytesPerElement : @(texelBytes),
        (__bridge NSString *)kIOSurfacePlaneInfo : @[ plane ],
    };
    IOSurfaceRef surface = IOSurfaceCreate((__bridge CFDictionaryRef)properties);
    if (surface == NULL) { return NULL; }
    IOSurfaceLock(surface, 0, NULL);
    memset(IOSurfaceGetBaseAddress(surface), kPad, IOSurfaceGetAllocSize(surface));
    IOSurfaceUnlock(surface, 0, NULL);
    return surface;
}

// The same surface, its plane holding the quarter turn (`h x w` elements) at the same pitch, so
// a rotated texture of `w x h` lies over it.
static IOSurfaceRef makeTurnedSurface(size_t w, size_t h, size_t texelBytes, size_t bpr) {
    return makeSurface(h, w, texelBytes, bpr);
}

// A framebuffer-only rotated texture of the configuration's extent over `surface`, or nil.
static id<MTLTexture> rotatedTexture(id<MTLDevice> device, IOSurfaceRef surface, const Config *c) {
    MTLTextureDescriptor *d =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:c->format
                                                          width:c->width
                                                         height:c->height
                                                      mipmapped:NO];
    d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    d.storageMode = MTLStorageModeShared;
    if (![d respondsToSelector:@selector(setRotation:)]
        || ![d respondsToSelector:@selector(setFramebufferOnly:)]) {
        fprintf(stderr, "the descriptor answers neither private setter\n");
        return nil;
    }
    [d setFramebufferOnly:YES];
    [d setRotation:1];
    return [device newTextureWithDescriptor:d iosurface:surface plane:0];
}

// The plane's width in elements: the configuration's own, or the texture's.
static size_t planeWidth(const Config *c) { return c->planeWidth ? c->planeWidth : c->width; }

static id<MTLTexture> planeTexture(id<MTLDevice> device, IOSurfaceRef surface, const Config *c) {
    MTLTextureDescriptor *d =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:c->format
                                                          width:c->width
                                                         height:c->height
                                                      mipmapped:NO];
    d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    d.storageMode = MTLStorageModeShared;
    return [device newTextureWithDescriptor:d iosurface:surface plane:0];
}

// Every source byte, tightly packed: byte (y, x) is `1 + y * w + x`, distinct over the whole
// source and never kPad.
static NSMutableData *sourceBytes(const Config *c) {
    const size_t row = c->width * c->texelBytes;
    NSMutableData *data = [NSMutableData dataWithLength:row * c->height];
    uint8_t *bytes = (uint8_t *)data.mutableBytes;
    for (size_t y = 0; y < c->height; y += 1) {
        for (size_t k = 0; k < row; k += 1) { bytes[y * row + k] = (uint8_t)(1 + y * row + k); }
    }
    return data;
}

// Print the plane's bytes, each decoded to the source byte it holds.
static void report(const char *tag, const Config *c, IOSurfaceRef surface, size_t pitch) {
    const size_t row = c->width * c->texelBytes;
    const size_t span = pitch * (c->height - 1) + row;
    IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL);
    const uint8_t *bytes = (const uint8_t *)IOSurfaceGetBaseAddress(surface);
    printf("%s %s pitch %zu row %zu overlap %zu alloc %zu\n", tag, c->name, pitch, row,
           row > pitch ? row - pitch : 0, (size_t)IOSurfaceGetAllocSize(surface));
    printf("%s BYTES", tag);
    for (size_t p = 0; p < span; p += 1) { printf(" %02x", bytes[p]); }
    printf("\n");
    // Each shared byte: which row and column of the source stands there.
    if (row > pitch) {
        const size_t overlap = row - pitch;
        for (size_t y = 1; y < c->height; y += 1) {
            printf("%s SHARED row %zu|%zu:", tag, y - 1, y);
            for (size_t k = 0; k < overlap; k += 1) {
                const size_t p = pitch * y + k;
                const uint8_t v = bytes[p];
                if (v == kPad) { printf(" at%zu=pad", p); continue; }
                const size_t at = (size_t)v - 1;
                const size_t sy = at / row, sx = at % row;
                const char *who = "other";
                if (sy == y - 1 && sx == pitch + k) { who = "earlier"; }
                if (sy == y && sx == k) { who = "later"; }
                printf(" at%zu=%02x(row%zu,byte%zu,%s)", p, v, sy, sx, who);
            }
            printf("\n");
        }
    }
    // Every byte past the written span must still be pad.
    size_t dirty = 0;
    for (size_t p = span; p < (size_t)IOSurfaceGetAllocSize(surface); p += 1) {
        if (bytes[p] != kPad) { dirty += 1; }
    }
    printf("%s PASTSPAN %zu dirty of %zu\n", tag, dirty,
           (size_t)IOSurfaceGetAllocSize(surface) - span);
    IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
}

// Commit and wait, with a completion handler so a hang reads as a timeout rather than an error.
static int finish(id<MTLCommandBuffer> commands, const char *tag) {
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    [commands addCompletedHandler:^(id<MTLCommandBuffer> _) { dispatch_semaphore_signal(done); }];
    [commands commit];
    if (dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 20ull * NSEC_PER_SEC))) {
        printf("%s TIMEOUT 20s\n", tag);
        return 3;
    }
    if (commands.status != MTLCommandBufferStatusCompleted) {
        printf("%s STATUS %lu error %s\n", tag, (unsigned long)commands.status,
               commands.error.description.UTF8String);
        return 4;
    }
    return 0;
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        const char *wanted = argc > 1 ? argv[1] : "facts";
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) { fprintf(stderr, "no Metal device\n"); return 2; }
        printf("DEVICE %s\n", device.name.UTF8String);
        printf("CASE %s\n", wanted);

        if (strcmp(wanted, "facts") == 0) {
            for (size_t i = 0; i < kConfigCount; i += 1) {
                const Config *c = &kConfigs[i];
                IOSurfaceRef surface = makeSurface(planeWidth(c), c->height, c->texelBytes,
                                                   c->bytesPerRow);
                if (surface == NULL) { printf("FACTS %zu %s no surface\n", i, c->name); continue; }
                id<MTLTexture> texture = planeTexture(device, surface, c);
                printf("FACTS %zu %s asked bpr %zu, surface bpr %zu, offset %zu, alloc %zu, "
                       "planes %zu, texture %s",
                       i, c->name, c->bytesPerRow,
                       (size_t)IOSurfaceGetBytesPerRowOfPlane(surface, 0),
                       (size_t)((const uint8_t *)IOSurfaceGetBaseAddressOfPlane(surface, 0) - (const uint8_t *)IOSurfaceGetBaseAddress(surface)),
                       (size_t)IOSurfaceGetAllocSize(surface),
                       (size_t)IOSurfaceGetPlaneCount(surface),
                       texture == nil ? "nil" : "made");
                if (texture != nil) {
                    printf(" (%lux%lu)", (unsigned long)texture.width,
                           (unsigned long)texture.height);
                }
                printf("\n");
                CFRelease(surface);
            }
            printf("DONE\n");
            return 0;
        }

        // Every other case is <kind>.<index>.
        char kind[16] = { 0 };
        size_t index = 0;
        if (sscanf(wanted, "%15[^.].%zu", kind, &index) != 2 || index >= kConfigCount) {
            fprintf(stderr, "usage: %s facts | (in|t2t|kern|kern2|out|rot|cpu|none).<0..%zu>\n", argv[0],
                    kConfigCount - 1);
            return 2;
        }
        const Config *c = &kConfigs[index];
        const int turned = strcmp(kind, "rot") == 0;
        IOSurfaceRef surface = turned
            ? makeTurnedSurface(c->width, c->height, c->texelBytes, c->bytesPerRow)
            : makeSurface(planeWidth(c), c->height, c->texelBytes, c->bytesPerRow);
        if (surface == NULL) { fprintf(stderr, "no surface\n"); return 2; }
        const size_t bpr = (size_t)IOSurfaceGetBytesPerRowOfPlane(surface, 0);
        const size_t pitch = (bpr / 16) * 16;
        printf("SURFACE bpr %zu offset %zu alloc %zu planes %zu\n", bpr,
               (size_t)((const uint8_t *)IOSurfaceGetBaseAddressOfPlane(surface, 0) - (const uint8_t *)IOSurfaceGetBaseAddress(surface)),
               (size_t)IOSurfaceGetAllocSize(surface), (size_t)IOSurfaceGetPlaneCount(surface));
        id<MTLTexture> texture = turned ? rotatedTexture(device, surface, c)
                                        : planeTexture(device, surface, c);
        if (texture == nil) { fprintf(stderr, "nil texture\n"); return 2; }
        printf("TEXTURE %lux%lu\n", (unsigned long)texture.width, (unsigned long)texture.height);
        id<MTLCommandQueue> queue = [device newCommandQueue];
        const size_t row = c->width * c->texelBytes;
        MTLRegion whole = MTLRegionMake2D(0, 0, c->width, c->height);

        if (strcmp(kind, "none") == 0) {
            report("NONE", c, surface, pitch);
            printf("DONE\n");
            CFRelease(surface);
            return 0;
        }

        if (strcmp(kind, "in") == 0 || turned) {
            NSMutableData *src = sourceBytes(c);
            id<MTLBuffer> buffer = [device newBufferWithBytes:src.bytes
                                                      length:src.length
                                                     options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> commands = [queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [commands blitCommandEncoder];
            [blit copyFromBuffer:buffer
                    sourceOffset:0
               sourceBytesPerRow:row
             sourceBytesPerImage:row * c->height
                      sourceSize:MTLSizeMake(c->width, c->height, 1)
                       toTexture:texture
                destinationSlice:0
                destinationLevel:0
               destinationOrigin:MTLOriginMake(0, 0, 0)];
            [blit endEncoding];
            int bad = finish(commands, turned ? "ROT" : "IN");
            if (bad) { CFRelease(surface); return bad; }
            report(turned ? "ROT" : "IN", c, surface, pitch);
            printf("DONE\n");
            CFRelease(surface);
            return 0;
        }

        if (strcmp(kind, "cpu") == 0) {
            NSMutableData *src = sourceBytes(c);
            [texture replaceRegion:whole mipmapLevel:0 withBytes:src.bytes bytesPerRow:row];
            report("CPU", c, surface, pitch);
            printf("DONE\n");
            CFRelease(surface);
            return 0;
        }

        if (strcmp(kind, "t2t") == 0) {
            MTLTextureDescriptor *d =
                [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:c->format
                                                                  width:c->width
                                                                 height:c->height
                                                              mipmapped:NO];
            d.usage = MTLTextureUsageShaderRead;
            d.storageMode = MTLStorageModeShared;
            id<MTLTexture> plain = [device newTextureWithDescriptor:d];
            if (plain == nil) { fprintf(stderr, "nil plain texture\n"); CFRelease(surface); return 2; }
            NSMutableData *src = sourceBytes(c);
            [plain replaceRegion:whole mipmapLevel:0 withBytes:src.bytes bytesPerRow:row];
            id<MTLCommandBuffer> commands = [queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [commands blitCommandEncoder];
            [blit copyFromTexture:plain
                      sourceSlice:0
                      sourceLevel:0
                     sourceOrigin:MTLOriginMake(0, 0, 0)
                       sourceSize:MTLSizeMake(c->width, c->height, 1)
                        toTexture:texture
                 destinationSlice:0
                 destinationLevel:0
                destinationOrigin:MTLOriginMake(0, 0, 0)];
            [blit endEncoding];
            int bad = finish(commands, "T2T");
            if (bad) { CFRelease(surface); return bad; }
            report("T2T", c, surface, pitch);
            printf("DONE\n");
            CFRelease(surface);
            return 0;
        }

        if (strcmp(kind, "kern") == 0 || strcmp(kind, "kern2") == 0) {
            if (c->texelBytes != 1) { fprintf(stderr, "kern is R8 only\n"); CFRelease(surface); return 2; }
            NSError *error = nil;
            id<MTLLibrary> library = [device newLibraryWithSource:kKernelSource
                                                         options:nil
                                                           error:&error];
            if (library == nil) { fprintf(stderr, "library: %s\n", error.description.UTF8String); CFRelease(surface); return 2; }
            id<MTLComputePipelineState> pipeline =
                [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"name_r8"]
                                                     error:&error];
            if (pipeline == nil) { fprintf(stderr, "pipeline: %s\n", error.description.UTF8String); CFRelease(surface); return 2; }
            uint32_t w = (uint32_t)c->width;
            id<MTLBuffer> widths = [device newBufferWithBytes:&w
                                                      length:sizeof(w)
                                                     options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> commands = [queue commandBuffer];
            id<MTLComputeCommandEncoder> e = [commands computeCommandEncoder];
            [e setComputePipelineState:pipeline];
            [e setTexture:texture atIndex:0];
            [e setBuffer:widths offset:0 atIndex:0];
            const int wide = strcmp(kind, "kern2") == 0;
            [e dispatchThreads:MTLSizeMake(c->width, c->height, 1)
        threadsPerThreadgroup:wide ? MTLSizeMake(32, 1, 1) : MTLSizeMake(8, 4, 1)];
            [e endEncoding];
            int bad = finish(commands, "KERN");
            if (bad) { CFRelease(surface); return bad; }
            report("KERN", c, surface, pitch);
            printf("DONE\n");
            CFRelease(surface);
            return 0;
        }

        if (strcmp(kind, "out") == 0) {
            // Every plane byte its own linear position plus one, so the buffer's rows name the
            // plane bytes a GPU read of each row took.
            const size_t alloc = (size_t)IOSurfaceGetAllocSize(surface);
            IOSurfaceLock(surface, 0, NULL);
            uint8_t *bytes = (uint8_t *)IOSurfaceGetBaseAddress(surface);
            for (size_t p = 0; p < alloc; p += 1) { bytes[p] = (uint8_t)(1 + p % 250); }
            IOSurfaceUnlock(surface, 0, NULL);
            id<MTLBuffer> buffer = [device newBufferWithLength:row * c->height
                                                      options:MTLResourceStorageModeShared];
            memset(buffer.contents, kPad, buffer.length);
            id<MTLCommandBuffer> commands = [queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [commands blitCommandEncoder];
            [blit copyFromTexture:texture
                      sourceSlice:0
                      sourceLevel:0
                     sourceOrigin:MTLOriginMake(0, 0, 0)
                       sourceSize:MTLSizeMake(c->width, c->height, 1)
                         toBuffer:buffer
                destinationOffset:0
           destinationBytesPerRow:row
         destinationBytesPerImage:row * c->height];
            [blit endEncoding];
            int bad = finish(commands, "OUT");
            if (bad) { CFRelease(surface); return bad; }
            const uint8_t *got = (const uint8_t *)buffer.contents;
            for (size_t y = 0; y < c->height; y += 1) {
                printf("OUT row %zu (plane byte %zu expected first):", y, pitch * y);
                for (size_t k = 0; k < row; k += 1) {
                    const uint8_t v = got[y * row + k];
                    printf(" %02x@%d", v, v == kPad ? -1 : (int)((size_t)v - 1));
                }
                printf("\n");
            }
            printf("DONE\n");
            CFRelease(surface);
            return 0;
        }

        fprintf(stderr, "unknown kind %s\n", kind);
        CFRelease(surface);
        return 2;
    }
}
