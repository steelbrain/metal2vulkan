// Whether a fence wait on one MTLCommandQueue waits for an update encoded, and committed earlier,
// in a command buffer of another queue that Metal holds before it runs: at an event wait, behind
// such a command buffer of its queue, or behind a command buffer enqueued and not committed.
// Metal's headers say an update and a wait are each "evaluated at kernel submission to maintain
// global order and prevent deadlock"; this measures what that means for a command buffer held.
//
// ONE CASE PER PROCESS, chosen on the command line, three runs each. In every case `u`, on queue
// A, runs `produce` (a kernel that spins a few milliseconds, then writes 0x5a5a5a5a to each word of
// X) and updates fence F; `w`, on queue B, committed after `u`, waits on F and runs `consume` (Y =
// X). X and Y are 4096 words, zeroed before each run.
//
//   control              `u` and `w` committed, nothing held.
//   control-nofence      as control with no fence call: whether the queues overlap at all, so a
//                        control that reads the writes shows the fence doing it.
//   shared-event         `u` waits for an MTLSharedEvent's value 1 before `produce`; the host sets
//                        it 2 s after `w`'s commit.
//   event                `u` waits for an MTLEvent's value 1 before `produce`; a command buffer on
//                        queue C signals it, committed 2 s after `w`'s commit.
//   behind-shared-event  on queue A, `h` (no work) waits for an MTLSharedEvent's value 1, then
//                        `u`, with no wait of its own; the host sets the value 2 s after `w`'s
//                        commit.
//   enqueue              on queue A, `x` is enqueued and not committed, then `u`; `x` (no work) is
//                        committed 2 s after `w`'s commit.
//   empty-waiter         as control, but `w`'s encoder only waits on F: no dispatch, nothing for
//                        the GPU. Whether its completion waits for `u`.
//
// A second argument, `untracked`, makes X and Y with MTLResourceHazardTrackingModeUntracked, so
// only the fence orders `w` after `u` (Metal tracks a tracked buffer's writes across command
// buffers).
//
// Each run reports whether `w` completed within the 2 s before the release, after how long, the
// statuses of `u` and `w` then, and how many words of Y held 0x5a5a5a5a when `w` completed; then,
// after the release, both statuses and errors, whether `w`'s GPU start came before `u`'s GPU
// end, and when `w`'s completed handler ran against `u`'s GPU end.
//
// Build and run (no cargo target needed):
//   xcrun clang -fobjc-arc -framework Metal -framework Foundation \
//       -o cross_queue_fence_held_update cross_queue_fence_held_update.m
//   ./cross_queue_fence_held_update shared-event
//   ./cross_queue_fence_held_update control untracked
//
// Measuring runs keep `MTL_DEBUG_LAYER` OUT of the environment.
//
// WHAT IT MEASURED (Apple M3 Max, macOS 27.0.1, three runs of each case, plain and under
// MTL_DEBUG_LAYER=1 alike, which reported nothing; `produce` ran about 29 ms; every command buffer
// ended Completed with no error):
//   control, untracked (jobs m3-max-52188, m3-max-52194): `w`'s GPU start 0.3 to 1 us after
//     `u`'s GPU end; Y held all 4096 words.
//   control-nofence, untracked (job m3-max-52188): `w` ran beside `u`, starting 4 us after it;
//     Y held none. With tracked buffers (job m3-max-52186) Metal ran `w` after `u` even with no
//     fence, as it tracks their writes.
//   shared-event, event, behind-shared-event, enqueue, tracked and untracked (jobs m3-max-52186,
//     m3-max-52188): in every run `w` completed 0.2 to 3.5 ms after its commit, before the
//     release, Y holding none of the words, `u` still Committed (status 2); `u` ran once
//     released.
//   empty-waiter, untracked (job m3-max-52194): `w`'s GPU span covered `u`'s, starting within
//     0.05 ms of `u`'s start and ending 0.1 ms after its end; its completed handler ran 0.11 to
//     0.16 ms after `u`'s GPU end.
// So a fence wait waits for an update of another queue submitted before it, a wait with no work
// after it included, and not for an update in a command buffer Metal holds before it runs, at an
// event wait, behind one held so, or behind one enqueued and not committed.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>
#include <stdio.h>
#include <string.h>

enum { WORDS = 4096 };

static double now_ms(void) {
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0) {
        mach_timebase_info(&tb);
    }
    return (double)mach_absolute_time() * tb.numer / tb.denom / 1e6;
}

static int written(id<MTLBuffer> buffer) {
    const uint32_t *words = buffer.contents;
    int count = 0;
    for (int i = 0; i < WORDS; i++) {
        count += words[i] == 0x5a5a5a5au;
    }
    return count;
}

static void dispatch(id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> pipeline,
                     id<MTLBuffer> a, id<MTLBuffer> b) {
    [e setComputePipelineState:pipeline];
    [e setBuffer:a offset:0 atIndex:0];
    if (b) {
        [e setBuffer:b offset:0 atIndex:1];
    }
    [e dispatchThreads:MTLSizeMake(WORDS, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc != 2 && !(argc == 3 && strcmp(argv[2], "untracked") == 0)) {
            fprintf(stderr, "usage: %s CASE [untracked]\n", argv[0]);
            return 2;
        }
        MTLResourceOptions options = MTLResourceStorageModeShared;
        if (argc == 3) {
            options |= MTLResourceHazardTrackingModeUntracked;
        }
        const char *kase = argv[1];
        const char *cases[] = {"control", "control-nofence", "shared-event", "event",
                               "behind-shared-event", "enqueue", "empty-waiter"};
        int which = -1;
        for (int i = 0; i < 7; i++) {
            if (strcmp(kase, cases[i]) == 0) {
                which = i;
            }
        }
        if (which < 0) {
            fprintf(stderr, "unknown case %s\n", kase);
            return 2;
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSString *source =
            @"#include <metal_stdlib>\n"
             "using namespace metal;\n"
             "kernel void produce(device uint *x [[buffer(0)]], uint id [[thread_position_in_grid]]) {\n"
             "    uint s = id;\n"
             "    for (uint i = 0; i < 2000000u; i++) { s = s * 1664525u + 1013904223u; }\n"
             "    x[id] = s == 0x12345678u ? 1u : 0x5a5a5a5au;\n"
             "}\n"
             "kernel void consume(device const uint *x [[buffer(0)]], device uint *y [[buffer(1)]],\n"
             "                    uint id [[thread_position_in_grid]]) {\n"
             "    y[id] = x[id];\n"
             "}\n";
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        if (!library) {
            printf("library: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }
        id<MTLComputePipelineState> produce =
            [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"produce"]
                                                  error:&error];
        id<MTLComputePipelineState> consume =
            [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"consume"]
                                                  error:&error];
        if (!produce || !consume) {
            printf("pipeline: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }
        printf("device %s, %s, case %s, buffers %s, MTL_DEBUG_LAYER %s\n", device.name.UTF8String,
               NSProcessInfo.processInfo.operatingSystemVersionString.UTF8String, kase,
               argc == 3 ? "untracked" : "tracked",
               getenv("MTL_DEBUG_LAYER") ? getenv("MTL_DEBUG_LAYER") : "unset");
        BOOL fenced = which != 1;
        BOOL held = which >= 2 && which <= 5;
        for (int run = 0; run < 3; run++) {
            id<MTLCommandQueue> qa = [device newCommandQueue];
            id<MTLCommandQueue> qb = [device newCommandQueue];
            id<MTLCommandQueue> qc = [device newCommandQueue];
            id<MTLBuffer> x = [device newBufferWithLength:WORDS * 4 options:options];
            id<MTLBuffer> y = [device newBufferWithLength:WORDS * 4 options:options];
            memset(x.contents, 0, WORDS * 4);
            memset(y.contents, 0, WORDS * 4);
            id<MTLFence> fence = [device newFence];
            id<MTLSharedEvent> shared = [device newSharedEvent];
            id<MTLEvent> event = [device newEvent];
            id<MTLCommandBuffer> h = nil;
            id<MTLCommandBuffer> enqueued = nil;
            if (which == 4) {
                h = [qa commandBuffer];
                [h encodeWaitForEvent:shared value:1];
                [h commit];
            }
            if (which == 5) {
                enqueued = [qa commandBuffer];
                [enqueued enqueue];
            }
            id<MTLCommandBuffer> u = [qa commandBuffer];
            if (which == 2) {
                [u encodeWaitForEvent:shared value:1];
            }
            if (which == 3) {
                [u encodeWaitForEvent:event value:1];
            }
            id<MTLComputeCommandEncoder> e = [u computeCommandEncoder];
            dispatch(e, produce, x, nil);
            if (fenced) {
                [e updateFence:fence];
            }
            [e endEncoding];
            [u commit];
            id<MTLCommandBuffer> w = [qb commandBuffer];
            e = [w computeCommandEncoder];
            if (fenced) {
                [e waitForFence:fence];
            }
            if (which != 6) {
                dispatch(e, consume, x, y);
            }
            [e endEncoding];
            __block double w_done = 0;
            __block int w_read = -1;
            dispatch_semaphore_t finished = dispatch_semaphore_create(0);
            [w addCompletedHandler:^(id<MTLCommandBuffer> c) {
              w_done = now_ms();
              w_read = written(y);
              dispatch_semaphore_signal(finished);
            }];
            double t0 = now_ms();
            [w commit];
            long late = 0;
            if (held) {
                late = dispatch_semaphore_wait(finished,
                                               dispatch_time(DISPATCH_TIME_NOW, 2 * NSEC_PER_SEC));
                printf("RUN %d: before the release w completed %s", run, late ? "NO" : "yes");
                if (!late) {
                    printf(" after %.3f ms, Y words written %d", w_done - t0, w_read);
                }
                printf("; status u %ld w %ld\n", (long)u.status, (long)w.status);
                double released = now_ms();
                if (which == 2 || which == 4) {
                    shared.signaledValue = 1;
                } else if (which == 3) {
                    id<MTLCommandBuffer> s = [qc commandBuffer];
                    [s encodeSignalEvent:event value:1];
                    [s commit];
                } else if (which == 5) {
                    [enqueued commit];
                }
                if (late) {
                    late = dispatch_semaphore_wait(
                        finished, dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC));
                    printf("RUN %d: after the release w completed %s", run, late ? "NO (5 s)" : "yes");
                    if (!late) {
                        printf(" %.3f ms after it, Y words written %d", w_done - released, w_read);
                    }
                    printf("\n");
                }
            } else {
                late = dispatch_semaphore_wait(finished,
                                               dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC));
                printf("RUN %d: w completed %s", run, late ? "NO (5 s)" : "yes");
                if (!late) {
                    printf(" after %.3f ms, Y words written %d", w_done - t0, w_read);
                }
                printf("\n");
            }
            [u waitUntilCompleted];
            [w waitUntilCompleted];
            printf("RUN %d: final status u %ld (error %s) w %ld (error %s); u gpu %.3f..%.3f ms, "
                   "w gpu %.3f..%.3f ms (from w's commit); w started before u ended: %s; "
                   "w's handler %.3f ms after u's GPU end\n",
                   run, (long)u.status, u.error ? u.error.localizedDescription.UTF8String : "nil",
                   (long)w.status, w.error ? w.error.localizedDescription.UTF8String : "nil",
                   u.GPUStartTime * 1e3 - t0, u.GPUEndTime * 1e3 - t0, w.GPUStartTime * 1e3 - t0,
                   w.GPUEndTime * 1e3 - t0, w.GPUStartTime < u.GPUEndTime ? "yes" : "no",
                   w_done - u.GPUEndTime * 1e3);
        }
        return 0;
    }
}
