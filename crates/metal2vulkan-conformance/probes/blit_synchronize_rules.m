// Which resources Metal takes in `synchronizeResource:` and `synchronizeTexture:slice:level:`,
// and the bytes the host reads after a GPU write and a synchronize.
//
// ONE CASE PER PROCESS, chosen on the command line, run under MTL_DEBUG_LAYER=1: the validation
// layer stops the process at a call it does not take, so a case that prints "taken" was taken.
// A buffer case fills bytes 4..12 of a 64-byte buffer, byte i = (uint8_t)i * 31 + 21 before the
// fill, with 0xab, synchronizes the buffer in the same blit encoder, and prints the bytes' FNV-1a
// 64 where the host reaches them. A texture case synchronizes a 4x4 RGBA8Unorm texture; a
// Managed one is first written by copyFromBuffer: from a Shared buffer of the same 64 bytes before
// the fill, and its getBytes: (rows 16 bytes apart) after the command buffer completes is hashed.
//
//   managed-buffer   a Managed buffer
//   shared-buffer    a Shared buffer
//   private-buffer   a Private buffer (nothing printed but the verdict)
//   managed-texture  a Managed texture, synchronizeResource:
//   managed-slice    a Managed texture, synchronizeTexture:slice:0 level:0
//   shared-texture   a Shared texture, synchronizeResource:
//   private-texture  a Private texture, synchronizeResource:
//
//   xcrun clang -fobjc-arc -framework Metal -framework Foundation \
//       -o blit_synchronize_rules blit_synchronize_rules.m
//   for c in managed-buffer shared-buffer private-buffer managed-texture managed-slice \
//       shared-texture private-texture; do MTL_DEBUG_LAYER=1 ./blit_synchronize_rules $c; done
//
// WHAT IT MEASURED: see the rt crate's `BlitCommandEncoder::synchronize_resource`, with the run it
// came from.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <string.h>

int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc != 2) {
            printf("usage: blit_synchronize_rules CASE\n");
            return 2;
        }
        const char *name = argv[1];
        struct {
            const char *name;
            BOOL texture;
            MTLStorageMode mode;
            BOOL slice;
        } cases[] = {
            {"managed-buffer", NO, MTLStorageModeManaged, NO},
            {"shared-buffer", NO, MTLStorageModeShared, NO},
            {"private-buffer", NO, MTLStorageModePrivate, NO},
            {"managed-texture", YES, MTLStorageModeManaged, NO},
            {"managed-slice", YES, MTLStorageModeManaged, YES},
            {"shared-texture", YES, MTLStorageModeShared, NO},
            {"private-texture", YES, MTLStorageModePrivate, NO},
        };
        int found = -1;
        for (int i = 0; i < (int)(sizeof cases / sizeof cases[0]); i++) {
            if (strcmp(cases[i].name, name) == 0) found = i;
        }
        if (found < 0) {
            printf("no case %s\n", name);
            return 2;
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        MTLStorageMode mode = cases[found].mode;
        id<MTLBuffer> buffer = nil;
        id<MTLTexture> texture = nil;
        uint8_t start[64];
        for (int i = 0; i < 64; i++) start[i] = (uint8_t)((uint8_t)i * 31 + 21);
        if (!cases[found].texture) {
            MTLResourceOptions options = (MTLResourceOptions)mode << MTLResourceStorageModeShift;
            buffer = mode == MTLStorageModePrivate
                         ? [device newBufferWithLength:64 options:options]
                         : [device newBufferWithBytes:start length:64 options:options];
            [blit fillBuffer:buffer range:NSMakeRange(4, 8) value:0xab];
            [blit synchronizeResource:buffer];
        } else {
            MTLTextureDescriptor *d =
                [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                   width:4
                                                                  height:4
                                                               mipmapped:NO];
            d.storageMode = mode;
            texture = [device newTextureWithDescriptor:d];
            if (mode == MTLStorageModeManaged) {
                id<MTLBuffer> source = [device newBufferWithBytes:start
                                                           length:64
                                                          options:MTLResourceStorageModeShared];
                [blit copyFromBuffer:source
                           sourceOffset:0
                      sourceBytesPerRow:16
                    sourceBytesPerImage:64
                             sourceSize:MTLSizeMake(4, 4, 1)
                              toTexture:texture
                       destinationSlice:0
                       destinationLevel:0
                      destinationOrigin:MTLOriginMake(0, 0, 0)];
            }
            if (cases[found].slice) {
                [blit synchronizeTexture:texture slice:0 level:0];
            } else {
                [blit synchronizeResource:texture];
            }
        }
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        printf("%s: taken, status %ld%s%s\n", name, (long)cb.status, cb.error ? ", error " : "",
               cb.error ? cb.error.localizedDescription.UTF8String : "");
        uint8_t read[64];
        const uint8_t *bytes = NULL;
        if (buffer != nil && mode != MTLStorageModePrivate) {
            bytes = buffer.contents;
        } else if (texture != nil && mode == MTLStorageModeManaged) {
            [texture getBytes:read
                  bytesPerRow:16
                   fromRegion:MTLRegionMake2D(0, 0, 4, 4)
                  mipmapLevel:0];
            bytes = read;
        }
        if (bytes != NULL) {
            uint64_t hash = 0xcbf29ce484222325ull;
            for (int i = 0; i < 64; i++) {
                hash ^= bytes[i];
                hash *= 0x100000001b3ull;
            }
            printf("%s len 64 fnv %016llx\n", name, (unsigned long long)hash);
        }
    }
    return 0;
}
