// Which ranges Metal takes in `fillBuffer:range:value:`, and the bytes a fill leaves.
//
// ONE CASE PER PROCESS, chosen on the command line, run under MTL_DEBUG_LAYER=1: the validation
// layer stops the process at a fill it does not take, so a case that prints "taken" was taken.
// Each fills a Shared buffer of 64 bytes, byte i = (uint8_t)i * 31 + 21 before the fill, with
// 0xab, and prints the buffer's FNV-1a 64 after it.
//
//   fine           range 4..12: whole words
//   offset-byte    range 1..9: an offset of no whole word
//   length-byte    range 4..10: a length of no whole word
//   bytes          range 3..13: neither
//   private        range 4..12 of a Private buffer (nothing printed but the verdict)
//
//   xcrun clang -fobjc-arc -framework Metal -framework Foundation \
//       -o blit_fill_rules blit_fill_rules.m
//   for c in fine offset-byte length-byte bytes private; do
//       MTL_DEBUG_LAYER=1 ./blit_fill_rules $c; done
//
// WHAT IT MEASURED: see the rt crate's `BlitCommandEncoder::fill_buffer`, with the run it came
// from.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <string.h>

int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc != 2) {
            printf("usage: blit_fill_rules CASE\n");
            return 2;
        }
        const char *name = argv[1];
        struct {
            const char *name;
            NSUInteger location, length;
            BOOL private_;
        } cases[] = {
            {"fine", 4, 8, NO},  {"offset-byte", 1, 8, NO}, {"length-byte", 4, 6, NO},
            {"bytes", 3, 10, NO}, {"private", 4, 8, YES},
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
        uint8_t start[64];
        for (int i = 0; i < 64; i++) start[i] = (uint8_t)((uint8_t)i * 31 + 21);
        MTLResourceOptions options =
            cases[found].private_ ? MTLResourceStorageModePrivate : MTLResourceStorageModeShared;
        id<MTLBuffer> buffer = cases[found].private_
                                   ? [device newBufferWithLength:64 options:options]
                                   : [device newBufferWithBytes:start length:64 options:options];
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit fillBuffer:buffer
                   range:NSMakeRange(cases[found].location, cases[found].length)
                   value:0xab];
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        printf("%s: taken, status %ld%s%s\n", name, (long)cb.status, cb.error ? ", error " : "",
               cb.error ? cb.error.localizedDescription.UTF8String : "");
        if (!cases[found].private_) {
            const uint8_t *bytes = buffer.contents;
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
