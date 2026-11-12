// Whether Metal orders a later command buffer of another queue behind an earlier one where the
// two meet only through memory two resources share (two textures over one IOSurface, two no-copy
// buffers over one allocation), with one resource on both sides as the control.
//
// The runtime's scheduler holds a later commit of another queue behind an earlier batch it holds
// for a reason of its own (its pipelines' making, its host step, its checks) where their tracked
// uses conflict (`scheduler/order.rs`). Its host aliases add a use of one memory key for each
// resource over the same host bytes (`host_aliases.rs`), and that order leaves those added uses
// out. This probe measures the case that rests on: queue A commits a long compute write, then
// queue B commits a short blit read of the same bytes, with no event between them.
//
// Cases, each run three times in this process with fresh queues and resources:
//
//   surface-two   A writes texture 1 of an IOSurface, B copies texture 2 of the same surface.
//   surface-one   control: B copies texture 1.
//   nocopy-two    A writes no-copy buffer 1, B copies no-copy buffer 2 over the same pages.
//   nocopy-one    control: B copies buffer 1.
//
// Every resource is tracked (`MTLHazardTrackingModeTracked`), its bytes 0 before A. A's kernel
// spins about 200 ms per thread (calibrated first, alone), then writes all ones. Each run reports
// A's and B's GPU start and end from A's start, whether B ended before A did, and the first word B
// read (0 = before A's write).
//
// WHAT IT MEASURED: not yet run.
//
// Build (no cargo target needed):
//   clang -fobjc-arc -O1 -fmodules -o cross_queue_surface_order cross_queue_surface_order.m \
//     -framework Foundation -framework Metal -framework IOSurface
//
// Measuring runs keep `MTL_DEBUG_LAYER` OUT of the environment.
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
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

static const NSUInteger kSize = 256;
static const NSUInteger kRow = kSize * 4;
static const NSUInteger kBytes = kRow * kSize;

static NSString *const kSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "static uint spin(uint seed, uint iters) {\n"
     "  uint x = seed;\n"
     "  for (uint i = 0; i < iters; i++) { x = x * 1664525u + 1013904223u; }\n"
     "  return x;\n"
     "}\n"
     "kernel void write_texture(texture2d<float, access::write> t [[texture(0)]],\n"
     "                          constant uint &iters [[buffer(0)]],\n"
     "                          uint2 gid [[thread_position_in_grid]]) {\n"
     "  uint x = spin(gid.x * 2654435761u + gid.y, iters);\n"
     "  t.write(float4(x == 0x12345678u ? 0.5 : 1.0), gid);\n"
     "}\n"
     "kernel void write_buffer(device uint *b [[buffer(0)]],\n"
     "                         constant uint &iters [[buffer(1)]],\n"
     "                         uint gid [[thread_position_in_grid]]) {\n"
     "  uint x = spin(gid * 2654435761u, iters);\n"
     "  b[gid] = x == 0x12345678u ? 1u : 0xFFFFFFFFu;\n"
     "}\n";

typedef struct {
  id<MTLDevice> device;
  id<MTLComputePipelineState> texture_writer;
  id<MTLComputePipelineState> buffer_writer;
} Probe;

static IOSurfaceRef new_surface(void) {
  NSDictionary *props = @{
    (id)kIOSurfaceWidth : @(kSize),
    (id)kIOSurfaceHeight : @(kSize),
    (id)kIOSurfaceBytesPerElement : @4,
    (id)kIOSurfaceBytesPerRow : @(kRow),
    (id)kIOSurfacePixelFormat : @((uint32_t)'BGRA'),
  };
  IOSurfaceRef surface = IOSurfaceCreate((__bridge CFDictionaryRef)props);
  IOSurfaceLock(surface, 0, NULL);
  memset(IOSurfaceGetBaseAddress(surface), 0, IOSurfaceGetAllocSize(surface));
  IOSurfaceUnlock(surface, 0, NULL);
  return surface;
}

static id<MTLTexture> over_surface(Probe *p, IOSurfaceRef surface) {
  MTLTextureDescriptor *desc =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                         width:kSize
                                                        height:kSize
                                                     mipmapped:NO];
  desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  desc.storageMode = MTLStorageModeShared;
  desc.hazardTrackingMode = MTLHazardTrackingModeTracked;
  return [p->device newTextureWithDescriptor:desc iosurface:surface plane:0];
}

static id<MTLBuffer> over_pages(Probe *p, void *pages) {
  return [p->device newBufferWithBytesNoCopy:pages
                                      length:kBytes
                                     options:MTLResourceStorageModeShared |
                                             MTLResourceHazardTrackingModeTracked
                                 deallocator:nil];
}

// A's command buffer: the long write through `texture` or `buffer`.
static id<MTLCommandBuffer> long_write(Probe *p, id<MTLCommandQueue> queue, id<MTLTexture> texture,
                                       id<MTLBuffer> buffer, uint32_t iters) {
  id<MTLCommandBuffer> cb = [queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  if (texture) {
    [enc setComputePipelineState:p->texture_writer];
    [enc setTexture:texture atIndex:0];
    [enc setBytes:&iters length:4 atIndex:0];
    [enc dispatchThreads:MTLSizeMake(kSize, kSize, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
  } else {
    [enc setComputePipelineState:p->buffer_writer];
    [enc setBuffer:buffer offset:0 atIndex:0];
    [enc setBytes:&iters length:4 atIndex:1];
    [enc dispatchThreads:MTLSizeMake(kSize * kSize, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  }
  [enc endEncoding];
  return cb;
}

// A alone, through a fresh texture: its GPU time in ms for `iters`.
static double time_alone(Probe *p, id<MTLCommandQueue> queue, uint32_t iters) {
  IOSurfaceRef surface = new_surface();
  id<MTLTexture> texture = over_surface(p, surface);
  id<MTLCommandBuffer> cb = long_write(p, queue, texture, nil, iters);
  [cb commit];
  [cb waitUntilCompleted];
  CFRelease(surface);
  return (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
}

static void run_case(Probe *p, const char *name, uint32_t iters, int run) {
  BOOL surface_case = strncmp(name, "surface", 7) == 0;
  BOOL two = strstr(name, "-two") != NULL;
  id<MTLCommandQueue> queue_a = [p->device newCommandQueue];
  id<MTLCommandQueue> queue_b = [p->device newCommandQueue];
  id<MTLBuffer> out = [p->device newBufferWithLength:kBytes options:MTLResourceStorageModeShared];
  memset(out.contents, 0xEE, kBytes);
  IOSurfaceRef surface = NULL;
  void *pages = NULL;
  id<MTLTexture> written_t = nil, read_t = nil;
  id<MTLBuffer> written_b = nil, read_b = nil;
  if (surface_case) {
    surface = new_surface();
    written_t = over_surface(p, surface);
    read_t = two ? over_surface(p, surface) : written_t;
  } else {
    if (posix_memalign(&pages, 16384, kBytes) != 0) {
      P("%s: posix_memalign failed", name);
      exit(1);
    }
    memset(pages, 0, kBytes);
    written_b = over_pages(p, pages);
    read_b = two ? over_pages(p, pages) : written_b;
  }

  id<MTLCommandBuffer> a = long_write(p, queue_a, written_t, written_b, iters);
  id<MTLCommandBuffer> b = [queue_b commandBuffer];
  id<MTLBlitCommandEncoder> blit = [b blitCommandEncoder];
  if (surface_case) {
    [blit copyFromTexture:read_t
              sourceSlice:0
              sourceLevel:0
             sourceOrigin:MTLOriginMake(0, 0, 0)
               sourceSize:MTLSizeMake(kSize, kSize, 1)
                 toBuffer:out
        destinationOffset:0
   destinationBytesPerRow:kRow
 destinationBytesPerImage:kBytes];
  } else {
    [blit copyFromBuffer:read_b sourceOffset:0 toBuffer:out destinationOffset:0 size:kBytes];
  }
  [blit endEncoding];

  [a commit];
  [b commit];
  [b waitUntilCompleted];
  [a waitUntilCompleted];

  double base = a.GPUStartTime;
  uint32_t first = ((uint32_t *)out.contents)[0];
  uint32_t last = ((uint32_t *)out.contents)[kBytes / 4 - 1];
  P("%s run %d: a %.2f..%.2f ms, b %.2f..%.2f ms from a's start; b ended %s a ended; b read "
    "first %08x last %08x; a %s, b %s",
    name, run, 0.0, (a.GPUEndTime - base) * 1000.0, (b.GPUStartTime - base) * 1000.0,
    (b.GPUEndTime - base) * 1000.0, b.GPUEndTime < a.GPUEndTime ? "BEFORE" : "after", first, last,
    a.error ? a.error.localizedDescription.UTF8String : "ok",
    b.error ? b.error.localizedDescription.UTF8String : "ok");
  if (surface) {
    CFRelease(surface);
  }
  written_b = nil;
  read_b = nil;
  free(pages);
}

int main(void) {
  @autoreleasepool {
    Probe p;
    p.device = MTLCreateSystemDefaultDevice();
    NSOperatingSystemVersion v = [[NSProcessInfo processInfo] operatingSystemVersion];
    P("device %s macOS %ld.%ld.%ld", p.device.name.UTF8String, (long)v.majorVersion,
      (long)v.minorVersion, (long)v.patchVersion);
    NSError *error = nil;
    id<MTLLibrary> library = [p.device newLibraryWithSource:kSource options:nil error:&error];
    if (!library) {
      P("library: %s", error.localizedDescription.UTF8String);
      return 1;
    }
    p.texture_writer = [p.device
        newComputePipelineStateWithFunction:[library newFunctionWithName:@"write_texture"]
                                      error:&error];
    p.buffer_writer = [p.device
        newComputePipelineStateWithFunction:[library newFunctionWithName:@"write_buffer"]
                                      error:&error];
    if (!p.texture_writer || !p.buffer_writer) {
      P("pipeline: %s", error.localizedDescription.UTF8String);
      return 1;
    }
    id<MTLCommandQueue> calibration = [p.device newCommandQueue];
    uint32_t iters = 1u << 14;
    double ms = time_alone(&p, calibration, iters);
    while (ms < 100.0 && iters < (1u << 28)) {
      double scale = ms > 1.0 ? 200.0 / ms : 8.0;
      iters = (uint32_t)MIN((double)(1u << 28), iters * scale);
      ms = time_alone(&p, calibration, iters);
    }
    P("A alone: %u iterations, %.1f ms", iters, ms);
    const char *cases[] = {"surface-two", "surface-one", "nocopy-two", "nocopy-one"};
    for (int c = 0; c < 4; c++) {
      for (int run = 0; run < 3; run++) {
        run_case(&p, cases[c], iters, run);
      }
    }
  }
  return 0;
}
