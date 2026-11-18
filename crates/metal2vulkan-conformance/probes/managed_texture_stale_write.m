// What a texture over a Managed buffer shows after the host writes the buffer with no
// `didModifyRange:`: the stale case Metal's Managed contract leaves undefined.
//
// ONE CASE PER PROCESS, chosen on the command line, run under MTL_DEBUG_LAYER=1. A 16x4
// RGBA8Unorm texture lies over an 8192-byte Managed buffer at offset 0, its rows one linear
// alignment past the device's own pitch for them (`minimumLinearTextureAlignmentForPixelFormat:`).
// The buffer starts as byte i = (uint8_t)(i % 251), told with `didModifyRange:` over all of it. A
// blit copies the texture into a Shared texture, the command buffer is waited on, and the copy's
// bytes are compared with the buffer's rows. The host then writes byte i = (uint8_t)(i % 251) ^
// 0x5a over the buffer, and a second command buffer copies the texture again.
//
//   stale      the host's second write is not told
//   modified   the host's second write is told with didModifyRange: over the whole buffer
//
//   xcrun clang -fobjc-arc -framework Metal -framework Foundation \
//       -o managed_texture_stale_write managed_texture_stale_write.m
//   for c in stale modified; do MTL_DEBUG_LAYER=1 ./managed_texture_stale_write $c; done
//
// Each case prints what each copy shows: "first" (the first bytes' rows), "second" (the second
// write's rows) or "other", with the copy's FNV-1a 64.
//
// The macOS 27 SDK marks Managed storage and didModifyRange: deprecated ("Managed storage has no
// effect on Apple Silicon"), so the build warns four times.
//
// WHAT IT MEASURED: an M3 Max on macOS 27.0.1 shows "second" after both writes, the untold one
// included: its texture over a Managed buffer aliases the buffer's bytes. The documented contract,
// not this answer, is the runtime's rule (the rt crate's README, on textures over a Managed
// buffer's device copy).

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <string.h>

enum { WIDTH = 16, HEIGHT = 4, ROW = WIDTH * 4, LEN = 8192 };

static uint64_t fnv(const uint8_t *bytes, size_t len) {
    uint64_t hash = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < len; i++) {
        hash ^= bytes[i];
        hash *= 0x100000001b3ull;
    }
    return hash;
}

// The texture's rows as they lie in `bytes`, `pitch` apart, packed into `out`.
static void rows(const uint8_t *bytes, NSUInteger pitch, uint8_t *out) {
    for (int row = 0; row < HEIGHT; row++) memcpy(out + row * ROW, bytes + row * pitch, ROW);
}

// Copies `texture` into `scratch` in a command buffer of its own, waits, and prints what the copy
// shows against `first` and `second`.
static void use(const char *what, id<MTLCommandQueue> queue, id<MTLTexture> texture,
                id<MTLTexture> scratch, const uint8_t *first, const uint8_t *second) {
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromTexture:texture
              sourceSlice:0
              sourceLevel:0
             sourceOrigin:MTLOriginMake(0, 0, 0)
               sourceSize:MTLSizeMake(WIDTH, HEIGHT, 1)
                toTexture:scratch
         destinationSlice:0
         destinationLevel:0
        destinationOrigin:MTLOriginMake(0, 0, 0)];
    [blit endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    uint8_t read[ROW * HEIGHT];
    [scratch getBytes:read bytesPerRow:ROW fromRegion:MTLRegionMake2D(0, 0, WIDTH, HEIGHT)
          mipmapLevel:0];
    const char *shows = memcmp(read, first, sizeof read) == 0    ? "first"
                        : memcmp(read, second, sizeof read) == 0 ? "second"
                                                                 : "other";
    printf("%s: status %ld, shows %s, fnv %016llx\n", what, (long)cb.status, shows,
           (unsigned long long)fnv(read, sizeof read));
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc != 2 || (strcmp(argv[1], "stale") != 0 && strcmp(argv[1], "modified") != 0)) {
            printf("usage: managed_texture_stale_write stale|modified\n");
            return 2;
        }
        BOOL told = strcmp(argv[1], "modified") == 0;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        NSUInteger align =
            [device minimumLinearTextureAlignmentForPixelFormat:MTLPixelFormatRGBA8Unorm];
        NSUInteger pitch = (ROW + align - 1) / align * align + align;
        printf("%s: %s, alignment %lu, pitch %lu\n", argv[1], device.name.UTF8String,
               (unsigned long)align, (unsigned long)pitch);
        id<MTLBuffer> buffer = [device newBufferWithLength:LEN
                                                   options:MTLResourceStorageModeManaged];
        uint8_t *contents = buffer.contents;
        for (int i = 0; i < LEN; i++) contents[i] = (uint8_t)(i % 251);
        [buffer didModifyRange:NSMakeRange(0, LEN)];
        uint8_t first[ROW * HEIGHT], second[ROW * HEIGHT];
        rows(contents, pitch, first);
        MTLTextureDescriptor *d =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                               width:WIDTH
                                                              height:HEIGHT
                                                           mipmapped:NO];
        d.storageMode = MTLStorageModeManaged;
        d.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
        id<MTLTexture> texture = [buffer newTextureWithDescriptor:d offset:0 bytesPerRow:pitch];
        d.storageMode = MTLStorageModeShared;
        id<MTLTexture> scratch = [device newTextureWithDescriptor:d];
        if (texture == nil || scratch == nil) {
            printf("%s: no texture\n", argv[1]);
            return 1;
        }
        uint8_t written[LEN];
        for (int i = 0; i < LEN; i++) written[i] = (uint8_t)(i % 251) ^ 0x5a;
        rows(written, pitch, second);
        use("first use", queue, texture, scratch, first, second);
        memcpy(contents, written, LEN);
        if (told) [buffer didModifyRange:NSMakeRange(0, LEN)];
        use(told ? "after a told write" : "after an untold write", queue, texture, scratch, first,
            second);
    }
    return 0;
}
