// What Metal reports for a command buffer that does no work: its GPUStartTime and GPUEndTime, when
// its completed handler runs against the command buffer committed before it on the same queue, and
// its status and error at completion.
//
// Three kinds of command buffer, each committed in two places, five runs of each:
//
//   empty         nothing encoded.
//   fence         one compute encoder that waits on an MTLFence and updates it, with no dispatch.
//   empty-kernel  one compute encoder dispatching `kernel void empty() {}` over 1x1x1.
//
//   alone         the queue is idle: the run before has completed.
//   after-busy    a busy command buffer (one dispatch of a kernel that spins a few milliseconds) is
//                 committed on the same queue just before it.
//
// Each run prints, as milliseconds after the run's first commit on the host clock (the clock
// GPUStartTime and GPUEndTime are on), the busy command buffer's GPU times and completed-handler
// time and order, then the probed command buffer's: GPU times (a raw 0 prints as 0), whether start
// and end are zero or equal, status, error, and its handler's time and order (0 runs first).
//
// Build and run (no cargo target needed):
//   xcrun clang -fobjc-arc -framework Metal -framework Foundation \
//       -o empty_command_buffer_times empty_command_buffer_times.m
//   ./empty_command_buffer_times
//   MTL_DEBUG_LAYER=1 ./empty_command_buffer_times
//
// WHAT IT MEASURED: see the commit that records the run.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>
#include <stdatomic.h>

static double now_s(void) {
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0) {
        mach_timebase_info(&tb);
    }
    return (double)mach_absolute_time() * tb.numer / tb.denom / 1e9;
}

static atomic_int order;

typedef struct {
    int order;
    double at;
    double start, end;
    long status;
    long code;
    BOOL error;
} Done;

static void record(Done *d, id<MTLCommandBuffer> cb) {
    d->order = atomic_fetch_add(&order, 1);
    d->at = now_s();
    d->start = cb.GPUStartTime;
    d->end = cb.GPUEndTime;
    d->status = (long)cb.status;
    d->error = cb.error != nil;
    d->code = cb.error ? (long)cb.error.code : 0;
}

static void time_of(double t, double t0) {
    if (t == 0) {
        printf("0");
    } else {
        printf("%.3f", (t - t0) * 1e3);
    }
}

int main(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        NSString *source = @"#include <metal_stdlib>\n"
                            "using namespace metal;\n"
                            "kernel void spin(device uint *out [[buffer(0)]],\n"
                            "                 uint id [[thread_position_in_grid]]) {\n"
                            "    uint x = id;\n"
                            "    for (uint i = 0; i < 2000000u; i++) { x = x * 1664525u + 1013904223u; }\n"
                            "    out[id] = x;\n"
                            "}\n"
                            "kernel void empty() {}\n";
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        if (!library) {
            printf("library: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }
        id<MTLComputePipelineState> spin =
            [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"spin"] error:&error];
        id<MTLComputePipelineState> empty =
            [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"empty"] error:&error];
        if (!spin || !empty) {
            printf("pipeline: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }
        id<MTLBuffer> out = [device newBufferWithLength:4096 * 4 options:MTLResourceStorageModeShared];
        id<MTLFence> fence = [device newFence];
        printf("device %s, %s\n", device.name.UTF8String,
               NSProcessInfo.processInfo.operatingSystemVersionString.UTF8String);
        const char *kinds[] = {"empty", "fence", "empty-kernel"};
        const char *places[] = {"alone", "after-busy"};
        for (int kind = 0; kind < 3; kind++) {
            for (int place = 0; place < 2; place++) {
                for (int run = 0; run < 5; run++) {
                    atomic_store(&order, 0);
                    __block Done busy_done = {-1, 0, 0, 0, 0, 0, NO};
                    __block Done done = {-1, 0, 0, 0, 0, 0, NO};
                    dispatch_semaphore_t finished = dispatch_semaphore_create(0);
                    double t0 = now_s();
                    if (place == 1) {
                        id<MTLCommandBuffer> busy = [queue commandBuffer];
                        id<MTLComputeCommandEncoder> e = [busy computeCommandEncoder];
                        [e setComputePipelineState:spin];
                        [e setBuffer:out offset:0 atIndex:0];
                        [e dispatchThreads:MTLSizeMake(4096, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                        [e endEncoding];
                        [busy addCompletedHandler:^(id<MTLCommandBuffer> cb) {
                          record(&busy_done, cb);
                          dispatch_semaphore_signal(finished);
                        }];
                        [busy commit];
                    }
                    id<MTLCommandBuffer> cb = [queue commandBuffer];
                    if (kind == 1) {
                        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                        [e waitForFence:fence];
                        [e updateFence:fence];
                        [e endEncoding];
                    } else if (kind == 2) {
                        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                        [e setComputePipelineState:empty];
                        [e dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                        [e endEncoding];
                    }
                    [cb addCompletedHandler:^(id<MTLCommandBuffer> c) {
                      record(&done, c);
                      dispatch_semaphore_signal(finished);
                    }];
                    [cb commit];
                    for (int i = 0; i < 1 + place; i++) {
                        dispatch_semaphore_wait(finished, DISPATCH_TIME_FOREVER);
                    }
                    printf("CASE %s %s run %d:", kinds[kind], places[place], run);
                    if (place == 1) {
                        printf(" busy gpu=[");
                        time_of(busy_done.start, t0);
                        printf(", ");
                        time_of(busy_done.end, t0);
                        printf("] handler order %d at %.3f;", busy_done.order, (busy_done.at - t0) * 1e3);
                    }
                    printf(" this gpu=[");
                    time_of(done.start, t0);
                    printf(", ");
                    time_of(done.end, t0);
                    printf("] start-zero %d end-zero %d equal %d status %ld error %s",
                           done.start == 0, done.end == 0, done.start == done.end, done.status,
                           done.error ? "yes" : "nil");
                    if (done.error) {
                        printf(" (code %ld)", done.code);
                    }
                    printf(" handler order %d at %.3f\n", done.order, (done.at - t0) * 1e3);
                }
            }
        }
        return 0;
    }
}
