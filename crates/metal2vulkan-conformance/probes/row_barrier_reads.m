// What Metal answers for a kernel that reads, after a threadgroup barrier, a word another row's
// thread of its threadgroup stored: the harness fixture's `row_barrier`
// (`metal2vulkan-validation/fixtures/public/harness/workloads.metal`, whose comment gives it). Each
// thread stores `((y << 16) | x) + 1` to its cell, waits at
// `threadgroup_barrier(mem_flags::mem_device)`, and copies the word of its column in row `y ^ 1`
// into `seen`, where that row is in the grid; the rest of `seen` keeps the 0xEEEEEEEE it was made
// with.
//
// Build (no cargo target needed):
//   clang -fobjc-arc -O1 -fmodules -o row_barrier_reads row_barrier_reads.m \
//     -framework Foundation -framework Metal
// Run: ./row_barrier_reads <path of workloads.metal>, MTL_DEBUG_LAYER unset.
//
// It compiles the fixture's source, then dispatches `row_barrier` with `dispatchThreads` over the
// grids the new runtime's device test runs (512 x 3 and 512 x 4 in threadgroups of 32 x 2, 96 x 5
// in threadgroups of 32 x 4), and prints for each the words that differ from that answer and the
// SHA-256 of `seen`.
//
// WHAT IT MEASURED (Apple M3 Max, macOS 27.0.1): see the new runtime's device test
// `every_device_keeps_a_kernel_that_reads_across_rows_after_a_barrier_at_its_own_size`.
#import <CommonCrypto/CommonDigest.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>

static const uint32_t FILL = 0xEEEEEEEEu;

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc != 2) {
            fprintf(stderr, "usage: %s <workloads.metal>\n", argv[0]);
            return 64;
        }
        NSError *error = nil;
        NSString *source = [NSString stringWithContentsOfFile:@(argv[1])
                                                     encoding:NSUTF8StringEncoding
                                                        error:&error];
        if (!source) {
            fprintf(stderr, "read: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        if (!library) {
            fprintf(stderr, "compile: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }
        id<MTLFunction> function = [library newFunctionWithName:@"row_barrier"];
        id<MTLComputePipelineState> state = [device newComputePipelineStateWithFunction:function
                                                                                  error:&error];
        if (!state) {
            fprintf(stderr, "state: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }
        printf("device %s, macOS %s\n", device.name.UTF8String,
               NSProcessInfo.processInfo.operatingSystemVersionString.UTF8String);
        const uint32_t shapes[3][4] = {{512, 3, 32, 2}, {512, 4, 32, 2}, {96, 5, 32, 4}};
        id<MTLCommandQueue> queue = [device newCommandQueue];
        for (int s = 0; s < 3; s++) {
            uint32_t width = shapes[s][0], height = shapes[s][1];
            size_t words = (size_t)width * height;
            id<MTLBuffer> cells = [device newBufferWithLength:4 * words
                                                      options:MTLResourceStorageModeShared];
            id<MTLBuffer> seen = [device newBufferWithLength:4 * words
                                                     options:MTLResourceStorageModeShared];
            memset(cells.contents, 0, 4 * words);
            uint32_t *out = seen.contents;
            for (size_t i = 0; i < words; i++) {
                out[i] = FILL;
            }
            id<MTLCommandBuffer> commands = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
            [encoder setComputePipelineState:state];
            [encoder setBuffer:cells offset:0 atIndex:0];
            [encoder setBuffer:seen offset:0 atIndex:1];
            [encoder dispatchThreads:MTLSizeMake(width, height, 1)
                threadsPerThreadgroup:MTLSizeMake(shapes[s][2], shapes[s][3], 1)];
            [encoder endEncoding];
            [commands commit];
            [commands waitUntilCompleted];
            if (commands.error) {
                fprintf(stderr, "run: %s\n", commands.error.localizedDescription.UTF8String);
                return 1;
            }
            size_t differ = 0, first = words;
            for (uint32_t y = 0; y < height; y++) {
                for (uint32_t x = 0; x < width; x++) {
                    uint32_t other = y ^ 1;
                    uint32_t want = other < height ? ((other << 16) | x) + 1 : FILL;
                    size_t i = (size_t)y * width + x;
                    if (out[i] != want) {
                        if (differ == 0) {
                            first = i;
                        }
                        differ++;
                    }
                }
            }
            unsigned char digest[CC_SHA256_DIGEST_LENGTH];
            CC_SHA256(out, (CC_LONG)(4 * words), digest);
            printf("row_barrier %u x %u in %u x %u: %zu words, %zu differ", width, height,
                   shapes[s][2], shapes[s][3], words, differ);
            if (differ) {
                printf(" (first at word %zu: 0x%08x)", first, out[first]);
            }
            printf(", sha256 ");
            for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++) {
                printf("%02x", digest[i]);
            }
            printf("\n");
        }
    }
    return 0;
}
