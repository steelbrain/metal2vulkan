// Which sizes Metal takes in a compute encoder's `setImageblockWidth:height:`, and the extent a
// kernel reads with none set.
//
// ONE CASE PER PROCESS, chosen on the command line, run under MTL_DEBUG_LAYER=1: the validation
// layer stops the process at a size it does not take, so a case that prints "taken" was taken.
// Each dispatches four 8 by 8 threadgroups of a kernel whose explicit imageblock has 8-byte cells
// and prints the extent its first thread read (`get_width`, `get_height`).
//
//   fine           16 by 16
//   width-24       24 by 16: a width of no power of two
//   height-12      16 by 12: a height of no power of two
//   zero           0 by 0
//   cells-2048     64 by 32: more than 1024 cells
//   wide-1024      1024 by 1
//   tall-1024      1 by 1024
//   bounded-256    16 by 16 for a state whose maxTotalThreadsPerThreadgroup is 64: the cells'
//                  bound is not the state's threads
//   before-state   32 by 32, set before setComputePipelineState
//   never-set      no call
//
//   xcrun clang -fobjc-arc -framework Metal -framework Foundation \
//       -o imageblock_size_rules imageblock_size_rules.m
//   for c in fine width-24 height-12 zero cells-2048 wide-1024 tall-1024 bounded-256 \
//       before-state never-set; do MTL_DEBUG_LAYER=1 ./imageblock_size_rules $c; done
//
// WHAT IT MEASURED: see the rt crate's `ComputeCommandEncoder::set_imageblock_size`, with the run
// it came from.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <string.h>

static NSString *const SOURCE =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "struct Cell { float a; uint b; };\n"
     "kernel void extent(imageblock<Cell, imageblock_layout_explicit> img,\n"
     "                   device uint *out [[buffer(0)]],\n"
     "                   ushort2 lid [[thread_position_in_threadgroup]]) {\n"
     "  if (lid.x == 0 && lid.y == 0) { out[0] = img.get_width(); out[1] = img.get_height(); }\n"
     "}\n"
     "[[max_total_threads_per_threadgroup(64)]]\n"
     "kernel void bounded(imageblock<Cell, imageblock_layout_explicit> img,\n"
     "                    device uint *out [[buffer(0)]],\n"
     "                    ushort2 lid [[thread_position_in_threadgroup]]) {\n"
     "  if (lid.x == 0 && lid.y == 0) { out[0] = img.get_width(); out[1] = img.get_height(); }\n"
     "}\n";

int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc != 2) {
            printf("usage: imageblock_size_rules CASE\n");
            return 2;
        }
        const char *name = argv[1];
        struct {
            const char *name;
            NSUInteger width, height;
            BOOL set, before, bounded;
        } cases[] = {
            {"fine", 16, 16, YES, NO, NO},        {"width-24", 24, 16, YES, NO, NO},
            {"height-12", 16, 12, YES, NO, NO},   {"zero", 0, 0, YES, NO, NO},
            {"cells-2048", 64, 32, YES, NO, NO},  {"wide-1024", 1024, 1, YES, NO, NO},
            {"tall-1024", 1, 1024, YES, NO, NO},  {"bounded-256", 16, 16, YES, NO, YES},
            {"before-state", 32, 32, YES, YES, NO}, {"never-set", 0, 0, NO, NO, NO},
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
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:SOURCE
                                                      options:[MTLCompileOptions new]
                                                        error:&error];
        if (!library) {
            printf("library: %s\n", error.localizedDescription.UTF8String);
            return 2;
        }
        NSString *entry = cases[found].bounded ? @"bounded" : @"extent";
        id<MTLComputePipelineState> state =
            [device newComputePipelineStateWithFunction:[library newFunctionWithName:entry]
                                                  error:&error];
        if (!state) {
            printf("state: %s\n", error.localizedDescription.UTF8String);
            return 2;
        }
        id<MTLBuffer> out = [device newBufferWithLength:16 options:MTLResourceStorageModeShared];
        memset(out.contents, 0xff, 16);
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
        if (cases[found].set && cases[found].before) {
            [encoder setImageblockWidth:cases[found].width height:cases[found].height];
        }
        [encoder setComputePipelineState:state];
        [encoder setBuffer:out offset:0 atIndex:0];
        if (cases[found].set && !cases[found].before) {
            [encoder setImageblockWidth:cases[found].width height:cases[found].height];
        }
        [encoder dispatchThreadgroups:MTLSizeMake(2, 2, 1)
                threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
        [encoder endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        const uint32_t *words = out.contents;
        printf("%s: taken, status %ld, maxTotalThreadsPerThreadgroup %lu, read %u by %u%s%s\n",
               name, (long)cb.status, (unsigned long)state.maxTotalThreadsPerThreadgroup, words[0],
               words[1], cb.error ? ", error " : "",
               cb.error ? cb.error.localizedDescription.UTF8String : "");
    }
    return 0;
}
