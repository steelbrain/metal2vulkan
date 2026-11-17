// Whether Metal orders a fill and a copy inside one blit command encoder where the two meet only
// through memory two no-copy buffers share.
//
// The runtime's host aliases (`host_aliases.rs`) give each use of a no-copy buffer made over
// exactly the bytes of a live one a use of the memory's own key, so its tracker orders a write
// through one and a read through the other inside an encoder as it orders one resource's. This
// probe measures what Metal does in that shape: in ONE blit command encoder of one command buffer,
// `fillBuffer:` writes a value through no-copy buffer 1, then `copyFromBuffer:` reads no-copy
// buffer 2, over the same page-aligned pages, into a third, Shared buffer; the command buffer is
// committed and waited for, and the copy's bytes are compared with the fill's value.
//
// Cases, ROUNDS (20) rounds each, every round with its own fill value:
//
//   nocopy-two-one-encoder  the copy reads buffer 2.
//   nocopy-one-one-encoder  control: the copy reads buffer 1, the fill's own (it must see the
//                           whole fill).
//
// Before each round, with the GPU idle (the round before was waited for), the host writes the
// pages with another value (`before`) and the third buffer with a third. Each round reports its
// fill value, how many of the copy's 32-bit words held the fill's value, `before` or neither, how
// many of the pages' words held the fill's value after it, and the command buffer's GPU time. Each
// case ends with the rounds whose copy saw the whole fill, none of it, or part. Every buffer is
// Shared and tracked (`MTLHazardTrackingModeTracked`), 32 MiB each; no fault, no untracked access,
// no wait that can hang.
//
// WHAT IT MEASURED (MTL_DEBUG_LAYER unset; 20 rounds a case, one process a device):
//
//   case                     M3 Max, 27.0.1   Radeon Pro 5300M, 26.7.1   UHD 630, 26.7.1
//   nocopy-two-one-encoder   part in 20       part in 20                 whole in 20
//   nocopy-one-one-encoder   whole in 20      whole in 20                whole in 20
//
// Where the copy saw part of the fill, the rest of its 8.39M words held the bytes from before:
// about 8.07M-8.11M of them the fill on the M3, 0.18M-0.30M on the 5300M. So Metal orders the
// fill and the copy through one buffer on all three, and through two no-copy buffers over the
// same pages on the 630 alone: the M3 and the 5300M ran the copy alongside the fill.
//
// Run with the case alone for the system default device, or with a part of a device's name first
// to pick it among `MTLCopyAllDevices()` (an x86 Mac's two GPUs: `5300M`, `630`); `all` runs both
// cases, the control second:
//   nocopy_alias_one_encoder [DEVICE] nocopy-two-one-encoder|nocopy-one-one-encoder|all
//
// Build (no cargo target needed; for an x86 Mac on an older macOS add `-arch x86_64` and a
// `-mmacosx-version-min=` no newer than its release):
//   clang -fobjc-arc -O1 -fmodules -o nocopy_alias_one_encoder nocopy_alias_one_encoder.m \
//     -framework Foundation -framework Metal
//
// Measuring runs keep `MTL_DEBUG_LAYER` OUT of the environment.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdarg.h>
#include <stdint.h>
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

static const NSUInteger kBytes = 32u << 20;
static const int kRounds = 20;

// How many of 32 MiB's words hold the fill's value, `before`, or neither.
typedef struct {
  unsigned long fill, before, other;
} Words;

static Words count(const void *bytes, uint8_t fill, uint8_t before) {
  Words w = {0, 0, 0};
  const uint32_t f = fill * 0x01010101u, b = before * 0x01010101u;
  const uint32_t *words = (const uint32_t *)bytes;
  for (NSUInteger i = 0; i < kBytes / 4; i++) {
    if (words[i] == f) {
      w.fill++;
    } else if (words[i] == b) {
      w.before++;
    } else {
      w.other++;
    }
  }
  return w;
}

static id<MTLBuffer> over_pages(id<MTLDevice> device, void *pages) {
  return [device newBufferWithBytesNoCopy:pages
                                   length:kBytes
                                  options:MTLResourceStorageModeShared |
                                          MTLResourceHazardTrackingModeTracked
                              deallocator:nil];
}

// One case's rounds; 1 where a buffer could not be made or a command buffer failed.
static int run_case(id<MTLDevice> device, id<MTLCommandQueue> queue, const char *name) {
  const BOOL two = strcmp(name, "nocopy-two-one-encoder") == 0;
  void *pages = NULL;
  if (posix_memalign(&pages, (size_t)getpagesize(), kBytes) != 0) {
    P("%s: no pages", name);
    return 1;
  }
  int whole = 0, none = 0, part = 0, failed = 0;
  @autoreleasepool {
    id<MTLBuffer> one = over_pages(device, pages);
    id<MTLBuffer> source = two ? over_pages(device, pages) : one;
    id<MTLBuffer> out =
        [device newBufferWithLength:kBytes
                            options:MTLResourceStorageModeShared |
                                    MTLResourceHazardTrackingModeTracked];
    if (!one || !source || !out) {
      P("%s: a buffer was refused", name);
      failed = 1;
    }
    for (int round = 0; round < kRounds && !failed; round++) {
      @autoreleasepool {
        const uint8_t fill = (uint8_t)(17 + round * 13);
        const uint8_t before = fill ^ 0xff, mark = fill ^ 0x0f;
        memset(pages, before, kBytes);
        memset(out.contents, mark, kBytes);
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit fillBuffer:one range:NSMakeRange(0, kBytes) value:fill];
        [blit copyFromBuffer:source
                 sourceOffset:0
                     toBuffer:out
            destinationOffset:0
                         size:kBytes];
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
          P("%s round %d: status %ld, %s", name, round, (long)cb.status,
            cb.error.localizedDescription.UTF8String);
          failed++;
          continue;
        }
        const Words copy = count(out.contents, fill, before);
        const Words landed = count(pages, fill, before);
        P("%s round %d: fill 0x%02x; the copy's words: %lu fill, %lu before, %lu other; the "
          "pages' words: %lu fill of %lu; GPU %.3f ms",
          name, round, fill, copy.fill, copy.before, copy.other, landed.fill,
          (unsigned long)(kBytes / 4), (cb.GPUEndTime - cb.GPUStartTime) * 1e3);
        if (copy.fill == kBytes / 4) {
          whole++;
        } else if (copy.fill == 0) {
          none++;
        } else {
          part++;
        }
      }
    }
  }
  free(pages);
  P("%s: %d rounds; the copy saw the whole fill in %d, none of it in %d, part of it in %d; %d "
    "failed",
    name, kRounds, whole, none, part, failed);
  return failed ? 1 : 0;
}

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc < 2 || argc > 3) {
      fprintf(stderr, "usage: %s [DEVICE] nocopy-two-one-encoder|nocopy-one-one-encoder|all\n",
              argv[0]);
      return 2;
    }
    const char *which = argv[argc - 1];
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (argc == 3) {
      device = nil;
      for (id<MTLDevice> candidate in MTLCopyAllDevices()) {
        if (strstr(candidate.name.UTF8String, argv[1])) {
          device = candidate;
        }
      }
    }
    if (!device) {
      P("no device named like %s", argc == 3 ? argv[1] : "the default");
      return 1;
    }
    NSOperatingSystemVersion v = [[NSProcessInfo processInfo] operatingSystemVersion];
    P("device %s macOS %ld.%ld.%ld", device.name.UTF8String, (long)v.majorVersion,
      (long)v.minorVersion, (long)v.patchVersion);
    id<MTLCommandQueue> queue = [device newCommandQueue];
    const char *cases[] = {"nocopy-two-one-encoder", "nocopy-one-one-encoder"};
    int ran = 0, failed = 0;
    for (int c = 0; c < 2; c++) {
      if (strcmp(which, "all") == 0 || strcmp(which, cases[c]) == 0) {
        ran++;
        failed += run_case(device, queue, cases[c]);
      }
    }
    if (!ran) {
      fprintf(stderr, "unknown case %s\n", which);
      return 2;
    }
    return failed ? 1 : 0;
  }
}
