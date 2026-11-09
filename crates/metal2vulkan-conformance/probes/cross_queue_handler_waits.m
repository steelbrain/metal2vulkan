// Whether a handler of one command queue's command buffer that blocks, waiting for another
// command buffer's completed handler, holds that completion back: on two `MTLCommandQueue`s, and
// on one as the control.
//
// The runtime's submission handlers (a runtime extension, `CommandBuffer::add_submission_handler`)
// run on its scheduler thread, and a batch held for one kernel submission with the batches after
// it runs them as it is held (`command_buffer/group.rs`). This probe measures what Metal does when
// a scheduled handler, the nearest Metal point, waits for the completed handler of a command
// buffer of another queue, so the runtime is held to that answer.
//
// ONE CASE PER PROCESS, chosen on the command line. In every case `a` blits 64 MiB of fills and has
// a completed handler that signals a semaphore; `b` is empty and has a handler that waits at most
// 5 s for that semaphore.
//
//   scheduled-two-queues        `a` on queue A, then `b` on queue B; `b`'s SCHEDULED handler
//                               waits.
//   scheduled-two-queues-b-first `b` on queue B first, then `a` on queue A, committed from the main
//                               thread once `b`'s scheduled handler began its wait.
//   completed-two-queues        as scheduled-two-queues, `b`'s COMPLETED handler waits.
//   scheduled-one-queue         `a`, then `b`, both on queue A; `b`'s scheduled handler waits.
//
// Each case reports the device and release, whether the semaphore came within the 5 s and after
// how long, both command buffers' final status and error, and the threads the handlers ran on.
//
// WHAT IT MEASURED: see the commit that adds it.
//
// Build (no cargo target needed):
//   clang -fobjc-arc -O1 -fmodules -o cross_queue_handler_waits cross_queue_handler_waits.m \
//     -framework Foundation -framework Metal
//
// Measuring runs keep `MTL_DEBUG_LAYER` OUT of the environment.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

static void P(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fputs("RESULT ", stdout);
  vfprintf(stdout, fmt, ap);
  fputc('\n', stdout);
  fflush(stdout);
  va_end(ap);
}

static uint64_t thread_id(void) {
  uint64_t id = 0;
  pthread_threadid_np(NULL, &id);
  return id;
}

static const char *status_name(MTLCommandBufferStatus status) {
  switch (status) {
  case MTLCommandBufferStatusNotEnqueued: return "NotEnqueued";
  case MTLCommandBufferStatusEnqueued: return "Enqueued";
  case MTLCommandBufferStatusCommitted: return "Committed";
  case MTLCommandBufferStatusScheduled: return "Scheduled";
  case MTLCommandBufferStatusCompleted: return "Completed";
  case MTLCommandBufferStatusError: return "Error";
  }
  return "?";
}

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc != 2) {
      fprintf(stderr, "usage: %s scheduled-two-queues|scheduled-two-queues-b-first|"
                      "completed-two-queues|scheduled-one-queue\n", argv[0]);
      return 2;
    }
    const char *which = argv[1];
    BOOL first_b = strcmp(which, "scheduled-two-queues-b-first") == 0;
    BOOL completed = strcmp(which, "completed-two-queues") == 0;
    BOOL one_queue = strcmp(which, "scheduled-one-queue") == 0;
    if (!first_b && !completed && !one_queue && strcmp(which, "scheduled-two-queues") != 0) {
      fprintf(stderr, "unknown case %s\n", which);
      return 2;
    }
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSOperatingSystemVersion v = [[NSProcessInfo processInfo] operatingSystemVersion];
    P("device %s macOS %ld.%ld.%ld case %s", device.name.UTF8String, (long)v.majorVersion,
      (long)v.minorVersion, (long)v.patchVersion, which);
    id<MTLCommandQueue> queue_a = [device newCommandQueue];
    id<MTLCommandQueue> queue_b = one_queue ? queue_a : [device newCommandQueue];
    const NSUInteger bytes = 64u << 20;
    id<MTLBuffer> buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModePrivate];

    dispatch_semaphore_t a_completed = dispatch_semaphore_create(0);
    dispatch_semaphore_t b_waiting = dispatch_semaphore_create(0);
    __block uint64_t a_thread = 0, b_thread = 0, main_thread = thread_id();
    __block long arrived = -1;
    __block double waited_ms = 0;

    id<MTLCommandBuffer> a = [queue_a commandBuffer];
    id<MTLBlitCommandEncoder> blit = [a blitCommandEncoder];
    for (int fill = 0; fill < 16; fill++) {
      [blit fillBuffer:buffer range:NSMakeRange(0, bytes) value:(uint8_t)fill];
    }
    [blit endEncoding];
    [a addCompletedHandler:^(id<MTLCommandBuffer> cb) {
      (void)cb;
      a_thread = thread_id();
      dispatch_semaphore_signal(a_completed);
    }];

    id<MTLCommandBuffer> b = [queue_b commandBuffer];
    void (^wait_for_a)(id<MTLCommandBuffer>) = ^(id<MTLCommandBuffer> cb) {
      (void)cb;
      b_thread = thread_id();
      dispatch_semaphore_signal(b_waiting);
      NSDate *start = [NSDate date];
      arrived = dispatch_semaphore_wait(a_completed,
                                        dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC));
      waited_ms = -[start timeIntervalSinceNow] * 1000.0;
    };
    if (completed) {
      [b addCompletedHandler:wait_for_a];
    } else {
      [b addScheduledHandler:wait_for_a];
    }

    if (first_b) {
      [b commit];
      long began = dispatch_semaphore_wait(b_waiting,
                                           dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC));
      P("b's handler began its wait before a's commit: %s", began == 0 ? "yes" : "no");
      [a commit];
    } else {
      [a commit];
      [b commit];
    }
    [a waitUntilCompleted];
    [b waitUntilCompleted];
    P("a's completed handler reached b's waiting handler: %s, after %.1f ms",
      arrived == 0 ? "yes" : "NO (5 s timeout)", waited_ms);
    P("a %s error %s, b %s error %s", status_name(a.status),
      a.error ? a.error.localizedDescription.UTF8String : "none", status_name(b.status),
      b.error ? b.error.localizedDescription.UTF8String : "none");
    P("threads: main %llu, a's completed handler %llu, b's waiting handler %llu",
      (unsigned long long)main_thread, (unsigned long long)a_thread,
      (unsigned long long)b_thread);
  }
  return 0;
}
