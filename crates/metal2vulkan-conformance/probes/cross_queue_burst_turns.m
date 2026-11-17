// Whether a small command buffer committed on one queue after a burst of command buffers on
// another waits for the whole burst, on Metal.
//
// The runtime's scheduler makes command queues that share a `VkQueue` take turns there
// (`QueueTurnPolicy::Fair`, `queue_turns.rs`): a queue keeps about 2 ms of its own expected work
// submitted ahead while another submits there, 16 ms while none does, so another queue's command
// buffer committed after a burst waits behind that much of it and one command buffer, not the
// whole burst. This probe measures what Metal does in that shape.
//
// ONE CASE PER PROCESS, chosen on the command line; five rounds a case. In every round queue A
// runs one command buffer alone, then commits BURST (64) command buffers back to back, each one
// dispatch of a kernel that spins about 4 ms (calibrated first, alone) and writes A's buffer; 5 ms
// after A's last commit, B commits one small command buffer that writes B's own buffer:
//
//   compute      B: one dispatch of one thread on queue B (no copy engine).
//   blit         B: a 256-byte fill on queue B.
//   render       B: a render pass on queue B that clears a 4x4 texture and draws one triangle.
//   same-queue   control: B's compute on queue A, behind the whole burst by Metal's contract.
//   one-long     control: A commits the burst's 64 dispatches as ONE command buffer, then B's
//                compute on queue B.
//
// Each round reports, in ms from B's commit: B's GPU start and end and its completed handler; how
// many of A's command buffers ended after B's commit and before B's GPU start (A's work B waited
// behind), and how many began after B's GPU end (A's work B went ahead of); and A's burst's last
// GPU end. Every resource is tracked; A and B share none. Each round's GPU work is about 0.26 s,
// far below Metal's 5 s command buffer timeout; no fault, no untracked write, no wait that can
// hang.
//
// WHAT IT MEASURED (MTL_DEBUG_LAYER unset; five rounds a case, one process a case):
//
//   M3 Max, macOS 27.0.1: B's GPU start 0.1-0.2 ms after its commit for compute and blit, 4.2-4.4
//     ms for render, ahead of 56-62 of A's 64 command buffers (they began after B's end): the M3
//     runs the second queue's command buffer among the first queue's burst.
//   UHD 630, macOS 26.7.1: B's GPU start only after 60-63 of A's 64 had ended, 77-240 ms after
//     its commit, compute, blit and render alike: the 630 runs the burst first.
//   Not measured yet: same-queue and one-long on either, and every case on the Radeon Pro 5300M.
//
// A's queue allows BURST + 2 command buffers (`newCommandQueueWithMaxCommandBufferCount:`): at
// the default 64, same-queue's B, made on A's queue while A's 64 were not yet committed, would
// wait for a free one forever.
//
// Run with the case alone for the system default device, or with a part of a device's name
// first to pick it among `MTLCopyAllDevices()` (an x86 Mac's two GPUs: `5300M`, `630`):
//   cross_queue_burst_turns [DEVICE] compute|blit|render|same-queue|one-long
//
// Build (no cargo target needed; add `-arch x86_64` for an x86 Mac):
//   clang -fobjc-arc -O1 -fmodules -o cross_queue_burst_turns cross_queue_burst_turns.m \
//     -framework Foundation -framework Metal
//
// Measuring runs keep `MTL_DEBUG_LAYER` OUT of the environment.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void P(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fputs("RESULT ", stdout);
  vfprintf(stdout, fmt, ap);
  fputc('\n', stdout);
  fflush(stdout);
  va_end(ap);
}

/// The host's clock in seconds, the one `GPUStartTime` and `GPUEndTime` answer on.
static double host_seconds(void) {
  static mach_timebase_info_data_t base;
  if (base.denom == 0) {
    mach_timebase_info(&base);
  }
  return (double)mach_absolute_time() * base.numer / base.denom / 1e9;
}

static const int BURST = 64;
static const int ROUNDS = 5;
static const NSUInteger THREADS = 256;

static NSString *const kSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void spin(device uint *b [[buffer(0)]],\n"
     "                 constant uint &iters [[buffer(1)]],\n"
     "                 uint gid [[thread_position_in_grid]]) {\n"
     "  uint x = gid * 2654435761u;\n"
     "  for (uint i = 0; i < iters; i++) { x = x * 1664525u + 1013904223u; }\n"
     "  b[gid] = x == 0x12345678u ? 1u : 0xFFFFFFFFu;\n"
     "}\n"
     "kernel void one(device uint *b [[buffer(0)]]) { b[0] = 7u; }\n"
     "vertex float4 corner(uint vid [[vertex_id]]) {\n"
     "  float2 p[3] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};\n"
     "  return float4(p[vid], 0, 1);\n"
     "}\n"
     "fragment float4 shade() { return float4(0.25, 0.5, 0.75, 1); }\n";

typedef struct {
  id<MTLDevice> device;
  id<MTLComputePipelineState> spin;
  id<MTLComputePipelineState> one;
  id<MTLRenderPipelineState> draw;
  id<MTLBuffer> a_words;
  id<MTLBuffer> b_words;
  id<MTLTexture> target;
} Probe;

static void encode_spin(Probe *p, id<MTLComputeCommandEncoder> e, uint32_t iters) {
  [e setComputePipelineState:p->spin];
  [e setBuffer:p->a_words offset:0 atIndex:0];
  [e setBytes:&iters length:sizeof iters atIndex:1];
  [e dispatchThreadgroups:MTLSizeMake(THREADS / 64, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}

/// A command buffer of `dispatches` spins on `queue`.
static id<MTLCommandBuffer> spins(Probe *p, id<MTLCommandQueue> queue, uint32_t iters,
                                  int dispatches) {
  id<MTLCommandBuffer> cb = [queue commandBuffer];
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
  for (int d = 0; d < dispatches; d++) {
    encode_spin(p, e, iters);
  }
  [e endEncoding];
  return cb;
}

static double time_alone(Probe *p, id<MTLCommandQueue> queue, uint32_t iters) {
  id<MTLCommandBuffer> cb = spins(p, queue, iters, 1);
  [cb commit];
  [cb waitUntilCompleted];
  return (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
}

/// B's small command buffer of `kind` on `queue`.
static id<MTLCommandBuffer> small(Probe *p, id<MTLCommandQueue> queue, const char *kind) {
  id<MTLCommandBuffer> cb = [queue commandBuffer];
  if (strcmp(kind, "blit") == 0) {
    id<MTLBlitCommandEncoder> e = [cb blitCommandEncoder];
    [e fillBuffer:p->b_words range:NSMakeRange(0, 256) value:1];
    [e endEncoding];
  } else if (strcmp(kind, "render") == 0) {
    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = p->target;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
    id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:pass];
    [e setRenderPipelineState:p->draw];
    [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [e endEncoding];
  } else {
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:p->one];
    [e setBuffer:p->b_words offset:0 atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [e endEncoding];
  }
  return cb;
}

static void round_of(Probe *p, const char *which, uint32_t iters, int round) {
  BOOL same_queue = strcmp(which, "same-queue") == 0;
  BOOL one_long = strcmp(which, "one-long") == 0;
  const char *kind = same_queue || one_long ? "compute" : which;
  id<MTLCommandQueue> queue_a = [p->device newCommandQueueWithMaxCommandBufferCount:BURST + 2];
  id<MTLCommandQueue> queue_b = same_queue ? queue_a : [p->device newCommandQueue];
  // A's queue warm, alone.
  id<MTLCommandBuffer> warm = spins(p, queue_a, iters, 1);
  [warm commit];
  [warm waitUntilCompleted];
  int count = one_long ? 1 : BURST;
  NSMutableArray<id<MTLCommandBuffer>> *burst = [NSMutableArray arrayWithCapacity:count];
  for (int c = 0; c < count; c++) {
    [burst addObject:spins(p, queue_a, iters, one_long ? BURST : 1)];
  }
  id<MTLCommandBuffer> b = small(p, queue_b, kind);
  __block double handled = 0;
  [b addCompletedHandler:^(id<MTLCommandBuffer> cb) {
    (void)cb;
    handled = host_seconds();
  }];
  for (id<MTLCommandBuffer> cb in burst) {
    [cb commit];
  }
  usleep(5000);
  double committed = host_seconds();
  [b commit];
  [b waitUntilCompleted];
  for (id<MTLCommandBuffer> cb in burst) {
    [cb waitUntilCompleted];
  }
  int behind = 0, after = 0;
  double last_end = 0;
  for (id<MTLCommandBuffer> cb in burst) {
    if (cb.GPUEndTime > committed && cb.GPUEndTime <= b.GPUStartTime) {
      behind++;
    }
    if (cb.GPUStartTime >= b.GPUEndTime) {
      after++;
    }
    last_end = MAX(last_end, cb.GPUEndTime);
  }
  double ms = 1000.0;
  P("%s round %d: B's GPU start %.2f end %.2f completed handler %.2f ms; A's command buffers "
    "ending between B's commit and its start %d, starting after B's end %d of %d; A's last end "
    "%.2f ms; B %s",
    which, round, (b.GPUStartTime - committed) * ms, (b.GPUEndTime - committed) * ms,
    (handled - committed) * ms, behind, after, count, (last_end - committed) * ms,
    b.error ? b.error.localizedDescription.UTF8String : "no error");
}

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc < 2 || argc > 3) {
      fprintf(stderr, "usage: %s [DEVICE] compute|blit|render|same-queue|one-long\n", argv[0]);
      return 2;
    }
    const char *which = argv[argc - 1];
    const char *cases[] = {"compute", "blit", "render", "same-queue", "one-long"};
    BOOL known = NO;
    for (int c = 0; c < 5; c++) {
      known |= strcmp(which, cases[c]) == 0;
    }
    if (!known) {
      fprintf(stderr, "unknown case %s\n", which);
      return 2;
    }
    Probe p;
    p.device = MTLCreateSystemDefaultDevice();
    if (argc == 3) {
      p.device = nil;
      for (id<MTLDevice> device in MTLCopyAllDevices()) {
        if (strstr(device.name.UTF8String, argv[1])) {
          p.device = device;
        }
      }
      if (!p.device) {
        P("no device named like %s", argv[1]);
        return 1;
      }
    }
    NSOperatingSystemVersion v = [[NSProcessInfo processInfo] operatingSystemVersion];
    P("device %s macOS %ld.%ld.%ld case %s MTL_DEBUG_LAYER=%s", p.device.name.UTF8String,
      (long)v.majorVersion, (long)v.minorVersion, (long)v.patchVersion, which,
      getenv("MTL_DEBUG_LAYER") ? getenv("MTL_DEBUG_LAYER") : "unset");
    NSError *error = nil;
    id<MTLLibrary> library = [p.device newLibraryWithSource:kSource options:nil error:&error];
    if (!library) {
      P("library: %s", error.localizedDescription.UTF8String);
      return 1;
    }
    p.spin = [p.device newComputePipelineStateWithFunction:[library newFunctionWithName:@"spin"]
                                                     error:&error];
    p.one = [p.device newComputePipelineStateWithFunction:[library newFunctionWithName:@"one"]
                                                    error:&error];
    MTLRenderPipelineDescriptor *desc = [MTLRenderPipelineDescriptor new];
    desc.vertexFunction = [library newFunctionWithName:@"corner"];
    desc.fragmentFunction = [library newFunctionWithName:@"shade"];
    desc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    p.draw = [p.device newRenderPipelineStateWithDescriptor:desc error:&error];
    if (!p.spin || !p.one || !p.draw) {
      P("pipeline: %s", error.localizedDescription.UTF8String);
      return 1;
    }
    p.a_words = [p.device newBufferWithLength:THREADS * 4 options:MTLResourceStorageModePrivate];
    p.b_words = [p.device newBufferWithLength:4096 options:MTLResourceStorageModePrivate];
    MTLTextureDescriptor *t =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                           width:4
                                                          height:4
                                                       mipmapped:NO];
    t.usage = MTLTextureUsageRenderTarget;
    t.storageMode = MTLStorageModePrivate;
    p.target = [p.device newTextureWithDescriptor:t];
    id<MTLCommandQueue> calibration = [p.device newCommandQueue];
    uint32_t iters = 1u << 10;
    double ms = time_alone(&p, calibration, iters);
    for (int step = 0; step < 12 && (ms < 3.5 || ms > 4.5); step++) {
      double scale = ms > 0.1 ? 4.0 / ms : 8.0;
      iters = (uint32_t)MIN((double)(1u << 28), MAX(1.0, iters * scale));
      ms = time_alone(&p, calibration, iters);
    }
    P("one dispatch alone: %u iterations, %.2f ms", iters, ms);
    for (int round = 0; round < ROUNDS; round++) {
      round_of(&p, which, iters, round);
    }
  }
  return 0;
}
