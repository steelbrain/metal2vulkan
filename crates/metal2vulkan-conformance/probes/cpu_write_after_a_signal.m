// What a command buffer's encoders read of a linear texture over a `Shared` buffer when the CPU
// writes the buffer WHILE THE COMMAND BUFFER RUNS: after an encoder's `encodeSignalEvent:value:`
// the CPU sees, and with no signal at all.
//
// The runtime carries a shadowed texture's image across a command buffer's encoders, so a later
// encoder reads the bytes as of the commit, not a CPU write made while the command buffer ran
// (`shadow.rs`'s **Copy in** rule, measured with no signal). It drops that carry where the next
// encoder starts another batch: after a signal, another queue's work may run first. This probe
// measures what Metal's encoder after a signal reads of a CPU write made once the signal was seen,
// so the copy in that the dropped carry makes is held to Metal's answer.
//
// ONE CASE PER PROCESS, chosen on the command line. Every case: an 8x8 `RGBA8Uint` texture over a
// `Shared` buffer at 48 bytes a row (`newTextureWithDescriptor:offset:bytesPerRow:`), holding OLD
// words; kernel `read_texels` stores the 64 texels it reads into a buffer of its own; kernel `spin`
// loads a word nothing raises until its cap of loads runs out. The GPU does not see a CPU store to
// the word while the kernel runs (a first version raised it: the spin ran on until Metal ended the
// command buffer, "Caused GPU Hang Error", in every case), and Metal ended one dispatch of about
// 1.5 s alike, and some of about 0.5 s. So a spin is SLICES dispatches of a cap timed first, in
// command buffers of their own, for about SLICE_MS of GPU time each, and the CPU writes while the
// spin runs.
//
//   signal-then-encoder  enc1 reads (out1), signals E=1; enc2 spins; enc3 reads (out2). The CPU
//                        waits until E reads 1, then writes NEW words over the buffer.
//   signal-same-encoder  enc1 reads (out1), signals E=1; enc2 spins, then reads (out2) in the
//                        same encoder. The CPU as above.
//   no-signal            enc1 reads (out1), then spins in the same encoder; enc2 reads (out2).
//                        The CPU sleeps 250 ms after the commit, then writes: the measured rule's
//                        case, which must read OLD in out2.
//   before-commit        as no-signal, but the CPU writes before the commit: both read NEW, so the
//                        probe can tell OLD from NEW.
//
// Each case reports the device and release, the timed cap, whether the command buffer was still
// running when the CPU wrote, its status and error, its GPU time, and for out1 and out2 how many
// of the 64 texels read OLD and how many NEW.
//
// WHAT IT MEASURED, three runs of each case (the M3 Max, macOS 27.0.1), every run alike:
//
//   before-commit        out1 NEW, out2 NEW.
//   no-signal            out1 OLD, out2 OLD: the write, made while the spin ran, reached neither.
//   signal-then-encoder  out1 OLD, out2 NEW: the write, made once the signal was seen and while
//                        the spin ran, reached the read after the signal.
//   signal-same-encoder  out1 OLD, out2 NEW: the same, the read a later dispatch of the encoder
//                        after the signal.
//
// Build (no cargo target needed):
//   clang -fobjc-arc -O1 -fmodules -o cpu_write_after_a_signal cpu_write_after_a_signal.m \
//     -framework Foundation -framework Metal
//
// Measuring runs keep `MTL_DEBUG_LAYER` OUT of the environment.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

static void P(const char *fmt, ...) { va_list ap; va_start(ap,fmt);
  fputs("RESULT ",stdout); vfprintf(stdout,fmt,ap); fputc('\n',stdout); fflush(stdout); va_end(ap); }

static const char *SRC =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"kernel void read_texels(texture2d<uint, access::read> t [[texture(0)]],\n"
"                        device uint* out [[buffer(0)]], uint2 g [[thread_position_in_grid]]) {\n"
"  uint4 c = t.read(g);\n"
"  out[g.y * 8 + g.x] = c.x | (c.y << 8) | (c.z << 16) | (c.w << 24); }\n"
"kernel void spin(device atomic_uint* word [[buffer(0)]], device uint* loads [[buffer(1)]],\n"
"                 constant uint& cap [[buffer(2)]]) {\n"
"  uint n = 0;\n"
"  while (n < cap && atomic_load_explicit(word, memory_order_relaxed) == 0) n++;\n"
"  *loads = n; }\n";

static const char *STATUS[] = {"NotEnqueued","Enqueued","Committed","Scheduled","Completed","Error"};

enum { SIDE = 8, ROW = 48, SLICE_MS = 10, SLICES = 150, MEASURED_MS = 50 };

static unsigned OLD(unsigned i) { return 0x01000000u * (i & 0x7f) + 0x00101010u; }
static unsigned NEW(unsigned i) { return 0x81000000u + 0x00010101u * (i & 0x3f) + 0x00202020u; }

static id<MTLComputePipelineState> state(id<MTLDevice> d, id<MTLLibrary> l, NSString *name) {
  NSError *e = nil;
  id<MTLComputePipelineState> s = [d newComputePipelineStateWithFunction:[l newFunctionWithName:name] error:&e];
  if (!s) { P("fatal=pipeline %s", [[e description] UTF8String]); exit(3); }
  return s;
}

static void write_words(id<MTLBuffer> b, unsigned (*word)(unsigned)) {
  unsigned char *bytes = (unsigned char *)b.contents;
  for (unsigned y = 0; y < SIDE; y++)
    for (unsigned x = 0; x < SIDE; x++) {
      unsigned w = word(y * SIDE + x);
      memcpy(bytes + y * ROW + x * 4, &w, 4);
    }
}

static void tally(const char *what, id<MTLBuffer> out) {
  const unsigned *w = (const unsigned *)out.contents;
  unsigned old = 0, fresh = 0, other = 0;
  for (unsigned i = 0; i < SIDE * SIDE; i++) {
    if (w[i] == OLD(i)) old++; else if (w[i] == NEW(i)) fresh++; else other++;
  }
  P("%s old=%u new=%u other=%u", what, old, fresh, other);
}

int main(int argc, const char **argv) {
  if (argc != 2) { P("usage: %s <case>", argv[0]); return 2; }
  const char *shape = argv[1];
  int signalEncoder = !strcmp(shape, "signal-then-encoder");
  int signalSame = !strcmp(shape, "signal-same-encoder");
  int noSignal = !strcmp(shape, "no-signal");
  int beforeCommit = !strcmp(shape, "before-commit");
  if (!(signalEncoder || signalSame || noSignal || beforeCommit)) {
    P("fatal=unknown-case %s", shape); return 2;
  }
  @autoreleasepool {
    id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    if (!d) { P("fatal=no-device"); return 3; }
    P("DEVICE %s / RELEASE %s / CASE %s", [[d name] UTF8String],
      [[[NSProcessInfo processInfo] operatingSystemVersionString] UTF8String], shape);
    id<MTLCommandQueue> q = [d newCommandQueue];
    NSError *e = nil;
    id<MTLLibrary> l = [d newLibraryWithSource:[NSString stringWithUTF8String:SRC] options:nil error:&e];
    if (!l) { P("fatal=library %s", [[e description] UTF8String]); return 3; }
    id<MTLComputePipelineState> reads = state(d, l, @"read_texels");
    id<MTLComputePipelineState> spins = state(d, l, @"spin");

    id<MTLBuffer> truth = [d newBufferWithLength:4096 options:MTLResourceStorageModeShared];
    memset(truth.contents, 0, 4096);
    write_words(truth, OLD);
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Uint
                                                                                  width:SIDE height:SIDE mipmapped:NO];
    td.usage = MTLTextureUsageShaderRead; td.storageMode = MTLStorageModeShared;
    id<MTLTexture> t = [truth newTextureWithDescriptor:td offset:0 bytesPerRow:ROW];
    if (!t) { P("fatal=no-texture"); return 3; }
    id<MTLBuffer> word = [d newBufferWithLength:4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> loads = [d newBufferWithLength:4 options:MTLResourceStorageModeShared];
    memset(word.contents, 0, 4); memset(loads.contents, 0, 4);
    id<MTLBuffer> out1 = [d newBufferWithLength:4 * SIDE * SIDE options:MTLResourceStorageModeShared];
    id<MTLBuffer> out2 = [d newBufferWithLength:4 * SIDE * SIDE options:MTLResourceStorageModeShared];
    memset(out1.contents, 0xee, 4 * SIDE * SIDE); memset(out2.contents, 0xee, 4 * SIDE * SIDE);
    id<MTLSharedEvent> event = [d newSharedEvent];
    __block unsigned cap = 1u << 16;

    void (^read)(id<MTLComputeCommandEncoder>, id<MTLBuffer>) = ^(id<MTLComputeCommandEncoder> c, id<MTLBuffer> out) {
      [c setComputePipelineState:reads]; [c setTexture:t atIndex:0]; [c setBuffer:out offset:0 atIndex:0];
      [c dispatchThreads:MTLSizeMake(SIDE, SIDE, 1) threadsPerThreadgroup:MTLSizeMake(SIDE, 1, 1)];
    };
    __block unsigned slices = 1;
    void (^spin)(id<MTLComputeCommandEncoder>) = ^(id<MTLComputeCommandEncoder> c) {
      [c setComputePipelineState:spins]; [c setBuffer:word offset:0 atIndex:0];
      [c setBuffer:loads offset:0 atIndex:1]; [c setBytes:&cap length:4 atIndex:2];
      for (unsigned k = 0; k < slices; k++)
        [c dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    };

    // Time the cap: single dispatches of growing caps alone, by their GPU time, until one takes
    // MEASURED_MS, scaled to SLICE_MS; the first, unmeasured, makes the pipeline's first use.
    for (int warm = 1;; warm = 0) {
      id<MTLCommandBuffer> t = [q commandBuffer];
      id<MTLComputeCommandEncoder> tc = [t computeCommandEncoder];
      spin(tc); [tc endEncoding]; [t commit]; [t waitUntilCompleted];
      double took = t.GPUEndTime - t.GPUStartTime;
      if (t.status != MTLCommandBufferStatusCompleted) { P("fatal=timing %s", STATUS[t.status]); return 3; }
      if (!warm && took * 1000.0 >= MEASURED_MS) {
        double scaled = (double)cap * (SLICE_MS / 1000.0) / took;
        cap = scaled < 1 ? 1u : (unsigned)scaled;
        break;
      }
      if (!warm) cap <<= 2;
    }
    slices = SLICES;
    P("cap=%u loads for about %d ms, %d dispatches", cap, SLICE_MS, SLICES);

    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> c = [cb computeCommandEncoder];
    read(c, out1);
    if (noSignal || beforeCommit) {
      spin(c); [c endEncoding];
      c = [cb computeCommandEncoder]; read(c, out2); [c endEncoding];
    } else {
      [c endEncoding];
      [cb encodeSignalEvent:event value:1];
      c = [cb computeCommandEncoder]; spin(c);
      if (signalEncoder) { [c endEncoding]; c = [cb computeCommandEncoder]; }
      read(c, out2); [c endEncoding];
    }
    if (beforeCommit) write_words(truth, NEW);
    [cb commit];
    if (beforeCommit) {
      // nothing: the write was before the commit
    } else if (noSignal) {
      usleep(250000);
      write_words(truth, NEW);
    } else {
      unsigned waited = 0;
      while (event.signaledValue < 1 && waited < 5000) { usleep(1000); waited++; }
      P("signal seen=%d after ~%u ms", event.signaledValue >= 1, waited);
      write_words(truth, NEW);
    }
    MTLCommandBufferStatus at_write = cb.status;
    [cb waitUntilCompleted];
    P("running at the write=%d (%s)", at_write < MTLCommandBufferStatusCompleted, STATUS[at_write]);
    P("status=%s error=%s gpu=%.0f ms", STATUS[cb.status],
      cb.error ? [[cb.error description] UTF8String] : "none", (cb.GPUEndTime - cb.GPUStartTime) * 1000.0);
    P("spin loads=%u of %u", *(const unsigned *)loads.contents, cap);
    tally("out1 (before)", out1);
    tally("out2 (after)", out2);
  }
  return 0;
}
