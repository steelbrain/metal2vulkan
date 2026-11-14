// What Metal answers for the writes into memory of its private option bit 24 that are not a
// shader's or the blit engine's buffer stores: a render pass storing an attachment there and the
// command buffers after it, a resolve into it, a `CustomSampleDepthStore` attachment there, a
// kernel's texture write, a blit into a texture over it, and the CPU's `replaceRegion:` of such a
// texture. `readonly_gpu_writes.m` measured the buffer stores (dropped) and the store-out
// (`PageFault`); this probe asks WHERE each other shape ends: at a call as it is encoded (the
// process ends there), at the GPU (the command buffer's error), or nowhere.
//
// ONE SHAPE PER PROCESS, chosen on the command line, because a shape that ends the process takes
// the others with it, and a shape that faults poisons what follows. Each stage prints a line and
// flushes before the call it names, so the last `stage=` line of a process that ended is the call
// that ended it. An alarm ends a process stuck past 20 s (exit by SIGALRM).
//
// Every shape runs over two backings: `bit24` (`newBufferWithBytesNoCopy:` of the bit, or a heap
// of the bit for the depth shape) and `plain` (the same without it), the writable control that
// makes a bit-24 answer a measurement.
//
// After the shape's command buffer, three more are committed, each a fill of a buffer of the
// runtime's own followed by a shared event signal: one on the same queue at once, one on a new
// queue, and one on the same queue 100 ms on. Their status, error, the event's value, whether the
// fill landed, and the time from commit to completion answer what a fault leaves the process.
// `store-then-read` asks the same of command buffers that name the faulted memory: after the
// store-out, copies out of the buffer and out of the texture over it, on the same queue and a new
// one.
//
// clang -fobjc-arc -framework Metal -framework Foundation -o readonly_encode_answers \
//     readonly_encode_answers.m

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <sys/mman.h>
#import <unistd.h>

// Metal's private `MTLResourceOptions` bit 24 (`ResourceOptions::PRIVATE_BIT`).
static const NSUInteger kPrivateBit = 1ull << 24;
static const size_t kRegion = 64 * 1024;
static const uint8_t kPad = 0xAB;
static const NSUInteger kTargetOffset = 0x1000;
static const NSUInteger kSide = 4;
static const NSUInteger kPitch = 32;

static void say(const char *key, NSString *value) {
  printf("%s=%s\n", key, value.UTF8String);
  fflush(stdout);
}

static void stage(const char *name) {
  printf("stage=%s\n", name);
  fflush(stdout);
}

static NSString *statusWord(MTLCommandBufferStatus status) {
  switch (status) {
    case MTLCommandBufferStatusNotEnqueued: return @"NotEnqueued";
    case MTLCommandBufferStatusEnqueued: return @"Enqueued";
    case MTLCommandBufferStatusCommitted: return @"Committed";
    case MTLCommandBufferStatusScheduled: return @"Scheduled";
    case MTLCommandBufferStatusCompleted: return @"Completed";
    case MTLCommandBufferStatusError: return @"Error";
  }
  return @"?";
}

static void sayCommandBuffer(const char *prefix, id<MTLCommandBuffer> cb, double millis) {
  char key[96];
  snprintf(key, sizeof(key), "%s.status", prefix);
  say(key, statusWord(cb.status));
  snprintf(key, sizeof(key), "%s.error", prefix);
  say(key, cb.error == nil ? @"none"
                           : [NSString stringWithFormat:@"code %ld: %@", (long)cb.error.code,
                                                        cb.error.localizedDescription]);
  if (millis >= 0) {
    snprintf(key, sizeof(key), "%s.commit_to_completed_ms", prefix);
    say(key, [NSString stringWithFormat:@"%.3f", millis]);
  }
}

// How many bytes of `length` at `bytes` are no longer `pad`.
static NSUInteger moved(const uint8_t *bytes, size_t length, uint8_t pad) {
  NSUInteger count = 0;
  for (size_t i = 0; i < length; i++) {
    count += bytes[i] != pad;
  }
  return count;
}

static NSString *const kShaders =
    @"#include <metal_stdlib>\n"
    @"using namespace metal;\n"
    @"vertex float4 ro_vert(uint vid [[vertex_id]]) {\n"
    @"  const float2 p[3] = { float2(-1, -3), float2(-1, 1), float2(3, 1) };\n"
    @"  return float4(p[vid], 0.5, 1);\n"
    @"}\n"
    @"fragment float4 ro_frag() { return float4(0.25, 0.5, 0.75, 1); }\n"
    @"kernel void ro_texture_write(texture2d<float, access::write> out [[texture(0)]],\n"
    @"                             uint2 at [[thread_position_in_grid]]) {\n"
    @"  out.write(float4(0.25, 0.5, 0.75, 1), at);\n"
    @"}\n";

static id<MTLLibrary> gLibrary;

static id<MTLRenderPipelineState> pipeline(id<MTLDevice> device, NSUInteger samples,
                                           MTLPixelFormat depth) {
  MTLRenderPipelineDescriptor *descriptor = [MTLRenderPipelineDescriptor new];
  descriptor.vertexFunction = [gLibrary newFunctionWithName:@"ro_vert"];
  descriptor.fragmentFunction = [gLibrary newFunctionWithName:@"ro_frag"];
  descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
  descriptor.depthAttachmentPixelFormat = depth;
  descriptor.rasterSampleCount = samples;
  NSError *error = nil;
  id<MTLRenderPipelineState> state =
      [device newRenderPipelineStateWithDescriptor:descriptor error:&error];
  if (state == nil) {
    say("pipeline", [NSString stringWithFormat:@"failed: %@", error]);
  }
  return state;
}

static id<MTLTexture> privateTexture(id<MTLDevice> device, MTLPixelFormat format,
                                     NSUInteger samples, MTLTextureUsage usage) {
  MTLTextureDescriptor *descriptor = [MTLTextureDescriptor new];
  descriptor.textureType = samples > 1 ? MTLTextureType2DMultisample : MTLTextureType2D;
  descriptor.pixelFormat = format;
  descriptor.width = kSide;
  descriptor.height = kSide;
  descriptor.sampleCount = samples;
  descriptor.usage = usage;
  descriptor.storageMode = MTLStorageModePrivate;
  return [device newTextureWithDescriptor:descriptor];
}

// The 4 x 4 RGBA8Unorm texture of `usage` over `buffer` at kTargetOffset.
static id<MTLTexture> overBuffer(id<MTLBuffer> buffer, MTLTextureUsage usage) {
  MTLTextureDescriptor *descriptor =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                         width:kSide
                                                        height:kSide
                                                     mipmapped:NO];
  descriptor.usage = usage;
  descriptor.storageMode = MTLStorageModeShared;
  descriptor.allowGPUOptimizedContents = NO;
  stage("new-texture-over-buffer");
  id<MTLTexture> texture = [buffer newTextureWithDescriptor:descriptor
                                                     offset:kTargetOffset
                                                bytesPerRow:kPitch];
  say("texture", texture == nil ? @"nil" : @"made");
  return texture;
}

// Encode `shape` into `cb` over `buffer` (or a heap of the backing for the depth shape): NO where
// something it needs was not made.
static BOOL encodeShape(id<MTLDevice> device, id<MTLCommandBuffer> cb, NSString *shape,
                        id<MTLBuffer> buffer, BOOL bit24) {
  if ([shape isEqualToString:@"store"]) {
    id<MTLTexture> target = overBuffer(buffer, MTLTextureUsageRenderTarget);
    id<MTLRenderPipelineState> state = pipeline(device, 1, MTLPixelFormatInvalid);
    if (target == nil || state == nil) return NO;
    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = target;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    stage("render-command-encoder");
    id<MTLRenderCommandEncoder> render = [cb renderCommandEncoderWithDescriptor:pass];
    [render setRenderPipelineState:state];
    stage("draw");
    [render drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    stage("end-encoding");
    [render endEncoding];
    return YES;
  }
  if ([shape isEqualToString:@"resolve-into"] || [shape isEqualToString:@"resolve-late"]) {
    BOOL late = [shape isEqualToString:@"resolve-late"];
    id<MTLTexture> resolveInto = overBuffer(buffer, MTLTextureUsageRenderTarget);
    id<MTLTexture> samples =
        privateTexture(device, MTLPixelFormatRGBA8Unorm, 4, MTLTextureUsageRenderTarget);
    id<MTLRenderPipelineState> state = pipeline(device, 4, MTLPixelFormatInvalid);
    if (resolveInto == nil || samples == nil || state == nil) return NO;
    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = samples;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction =
        late ? MTLStoreActionUnknown : MTLStoreActionMultisampleResolve;
    pass.colorAttachments[0].resolveTexture = resolveInto;
    stage("render-command-encoder");
    id<MTLRenderCommandEncoder> render = [cb renderCommandEncoderWithDescriptor:pass];
    [render setRenderPipelineState:state];
    stage("draw");
    [render drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    if (late) {
      stage("set-color-store-action");
      [render setColorStoreAction:MTLStoreActionMultisampleResolve atIndex:0];
    }
    stage("end-encoding");
    [render endEncoding];
    return YES;
  }
  if ([shape isEqualToString:@"custom-depth"] || [shape isEqualToString:@"custom-depth-late"]) {
    BOOL late = [shape isEqualToString:@"custom-depth-late"];
    MTLTextureDescriptor *descriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                                           width:kSide
                                                          height:kSide
                                                       mipmapped:NO];
    descriptor.usage = MTLTextureUsageRenderTarget;
    descriptor.storageMode = MTLStorageModePrivate;
    MTLSizeAndAlign sized = [device heapTextureSizeAndAlignWithDescriptor:descriptor];
    MTLHeapDescriptor *heapDescriptor = [MTLHeapDescriptor new];
    heapDescriptor.type = MTLHeapTypePlacement;
    heapDescriptor.storageMode = MTLStorageModePrivate;
    heapDescriptor.size = sized.size + sized.align + 4096;
    heapDescriptor.resourceOptions = MTLResourceStorageModePrivate | (bit24 ? kPrivateBit : 0);
    stage("new-heap");
    id<MTLHeap> heap = [device newHeapWithDescriptor:heapDescriptor];
    if (heap == nil) {
      say("heap", @"nil");
      return NO;
    }
    stage("new-texture-in-heap");
    id<MTLTexture> depth = [heap newTextureWithDescriptor:descriptor offset:0];
    id<MTLTexture> color =
        privateTexture(device, MTLPixelFormatRGBA8Unorm, 1, MTLTextureUsageRenderTarget);
    id<MTLRenderPipelineState> state = pipeline(device, 1, MTLPixelFormatDepth32Float);
    if (depth == nil || color == nil || state == nil) return NO;
    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = color;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionDontCare;
    pass.depthAttachment.texture = depth;
    pass.depthAttachment.loadAction = MTLLoadActionClear;
    pass.depthAttachment.storeAction =
        late ? MTLStoreActionUnknown : MTLStoreActionCustomSampleDepthStore;
    stage("render-command-encoder");
    id<MTLRenderCommandEncoder> render = [cb renderCommandEncoderWithDescriptor:pass];
    [render setRenderPipelineState:state];
    stage("draw");
    [render drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    if (late) {
      stage("set-depth-store-action");
      [render setDepthStoreAction:MTLStoreActionCustomSampleDepthStore];
    }
    stage("end-encoding");
    [render endEncoding];
    return YES;
  }
  if ([shape isEqualToString:@"kernel-write"]) {
    id<MTLTexture> target =
        overBuffer(buffer, MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
    NSError *error = nil;
    id<MTLComputePipelineState> state = [device
        newComputePipelineStateWithFunction:[gLibrary newFunctionWithName:@"ro_texture_write"]
                                      error:&error];
    if (target == nil || state == nil) return NO;
    stage("compute-command-encoder");
    id<MTLComputeCommandEncoder> compute = [cb computeCommandEncoder];
    [compute setComputePipelineState:state];
    stage("set-texture");
    [compute setTexture:target atIndex:0];
    stage("dispatch");
    [compute dispatchThreads:MTLSizeMake(kSide, kSide, 1)
        threadsPerThreadgroup:MTLSizeMake(kSide, kSide, 1)];
    stage("end-encoding");
    [compute endEncoding];
    return YES;
  }
  if ([shape isEqualToString:@"blit-buffer-into"] || [shape isEqualToString:@"blit-texture-into"]) {
    id<MTLTexture> target = overBuffer(buffer, MTLTextureUsageShaderRead);
    if (target == nil) return NO;
    stage("blit-command-encoder");
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    if ([shape isEqualToString:@"blit-buffer-into"]) {
      id<MTLBuffer> source = [device newBufferWithLength:kPitch * kSide
                                                 options:MTLResourceStorageModeShared];
      memset(source.contents, 0x3C, source.length);
      stage("copy-from-buffer-to-texture");
      [blit copyFromBuffer:source
                 sourceOffset:0
            sourceBytesPerRow:kPitch
          sourceBytesPerImage:kPitch * kSide
                   sourceSize:MTLSizeMake(kSide, kSide, 1)
                    toTexture:target
             destinationSlice:0
             destinationLevel:0
            destinationOrigin:MTLOriginMake(0, 0, 0)];
    } else {
      id<MTLTexture> source =
          privateTexture(device, MTLPixelFormatRGBA8Unorm, 1, MTLTextureUsageShaderRead);
      stage("copy-from-texture-to-texture");
      [blit copyFromTexture:source toTexture:target];
    }
    stage("end-encoding");
    [blit endEncoding];
    return YES;
  }
  say("shape", [NSString stringWithFormat:@"unknown-%@", shape]);
  return NO;
}

// A command buffer on `queue` of a fill of `out` and a signal of `event` to `value`, committed and
// waited for, reported under `prefix`.
static void later(const char *prefix, id<MTLCommandQueue> queue, id<MTLSharedEvent> event,
                  uint64_t value) {
  id<MTLBuffer> out = [queue.device newBufferWithLength:256 options:MTLResourceStorageModeShared];
  memset(out.contents, kPad, out.length);
  id<MTLCommandBuffer> cb = [queue commandBuffer];
  id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
  [blit fillBuffer:out range:NSMakeRange(0, out.length) value:0x11];
  [blit endEncoding];
  [cb encodeSignalEvent:event value:value];
  NSDate *start = [NSDate date];
  [cb commit];
  [cb waitUntilCompleted];
  double millis = -[start timeIntervalSinceNow] * 1000.0;
  sayCommandBuffer(prefix, cb, millis);
  char key[96];
  snprintf(key, sizeof(key), "%s.fill_landed", prefix);
  say(key, moved(out.contents, out.length, kPad) == out.length ? @"yes" : @"no");
  snprintf(key, sizeof(key), "%s.event_value", prefix);
  say(key, [NSString stringWithFormat:@"%llu", event.signaledValue]);
}

// A command buffer on `queue` that copies the region at kTargetOffset of `buffer`, and the texels
// of `texture` where given, into buffers of the runtime's own, committed and waited for: whether a
// command buffer naming the memory a fault was over runs, reported under `prefix`.
static void laterReads(const char *prefix, id<MTLCommandQueue> queue, id<MTLBuffer> buffer,
                       id<MTLTexture> texture) {
  id<MTLBuffer> out = [queue.device newBufferWithLength:kPitch * kSide
                                                options:MTLResourceStorageModeShared];
  memset(out.contents, 0x5C, out.length);
  id<MTLCommandBuffer> cb = [queue commandBuffer];
  id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
  if (texture == nil) {
    [blit copyFromBuffer:buffer
             sourceOffset:kTargetOffset
                 toBuffer:out
        destinationOffset:0
                     size:out.length];
  } else {
    [blit copyFromTexture:texture
                     sourceSlice:0
                     sourceLevel:0
                    sourceOrigin:MTLOriginMake(0, 0, 0)
                      sourceSize:MTLSizeMake(kSide, kSide, 1)
                        toBuffer:out
               destinationOffset:0
          destinationBytesPerRow:kPitch
        destinationBytesPerImage:kPitch * kSide];
  }
  [blit endEncoding];
  NSDate *start = [NSDate date];
  [cb commit];
  [cb waitUntilCompleted];
  sayCommandBuffer(prefix, cb, -[start timeIntervalSinceNow] * 1000.0);
  char key[96];
  snprintf(key, sizeof(key), "%s.read_pad_bytes", prefix);
  say(key, [NSString stringWithFormat:@"%lu", (unsigned long)(out.length - moved(out.contents,
                                                                                 out.length, kPad))]);
}

static int usage(void) {
  fprintf(stderr,
          "usage: readonly_encode_answers <bit24|plain> <shape>\n"
          "  shape  store | resolve-into | resolve-late | custom-depth | custom-depth-late |\n"
          "         kernel-write | blit-buffer-into | blit-texture-into | replace |\n"
          "         store-then-read\n");
  return 2;
}

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc != 3) return usage();
    alarm(20);
    BOOL bit24 = strcmp(argv[1], "bit24") == 0;
    if (!bit24 && strcmp(argv[1], "plain") != 0) return usage();
    NSString *shape = [NSString stringWithUTF8String:argv[2]];
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    say("device", device.name);
    say("os", [[NSProcessInfo processInfo] operatingSystemVersionString]);
    NSError *error = nil;
    gLibrary = [device newLibraryWithSource:kShaders options:nil error:&error];
    if (gLibrary == nil) {
      say("library", [NSString stringWithFormat:@"failed: %@", error]);
      return 2;
    }
    void *pages = mmap(NULL, kRegion, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (pages == MAP_FAILED) return 2;
    memset(pages, kPad, kRegion);
    NSUInteger options = MTLResourceStorageModeShared | (bit24 ? kPrivateBit : 0);
    stage("new-buffer-no-copy");
    id<MTLBuffer> buffer = [device newBufferWithBytesNoCopy:pages
                                                     length:kRegion
                                                    options:options
                                                deallocator:nil];
    if (buffer == nil) {
      say("buffer", @"nil");
      return 2;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLSharedEvent> event = [device newSharedEvent];

    if ([shape isEqualToString:@"replace"]) {
      id<MTLTexture> target = overBuffer(buffer, MTLTextureUsageShaderRead);
      if (target == nil) return 2;
      uint8_t texels[kPitch * kSide];
      memset(texels, 0x3C, sizeof(texels));
      stage("replace-region");
      [target replaceRegion:MTLRegionMake2D(0, 0, kSide, kSide)
                mipmapLevel:0
                  withBytes:texels
                bytesPerRow:kPitch];
      stage("replaced");
      say("pages_moved",
          [NSString stringWithFormat:@"%lu", (unsigned long)moved(pages, kRegion, kPad)]);
    } else if (![shape isEqualToString:@"store-then-read"]) {
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      if (!encodeShape(device, cb, shape, buffer, bit24)) return 2;
      [cb encodeSignalEvent:event value:1];
      stage("commit");
      NSDate *start = [NSDate date];
      [cb commit];
      [cb waitUntilCompleted];
      double millis = -[start timeIntervalSinceNow] * 1000.0;
      stage("completed");
      sayCommandBuffer("cb1", cb, millis);
      say("cb1.event_value", [NSString stringWithFormat:@"%llu", event.signaledValue]);
      say("pages_moved",
          [NSString stringWithFormat:@"%lu", (unsigned long)moved(pages, kRegion, kPad)]);
    }
    if ([shape isEqualToString:@"store-then-read"]) {
      // The store-out of `store`, then command buffers that name the memory it faulted on: a
      // copy out of the buffer and out of the texture over it, on the same queue and a new one.
      id<MTLTexture> target = overBuffer(buffer, MTLTextureUsageRenderTarget);
      id<MTLRenderPipelineState> state = pipeline(device, 1, MTLPixelFormatInvalid);
      if (target == nil || state == nil) return 2;
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
      pass.colorAttachments[0].texture = target;
      pass.colorAttachments[0].loadAction = MTLLoadActionClear;
      pass.colorAttachments[0].storeAction = MTLStoreActionStore;
      id<MTLRenderCommandEncoder> render = [cb renderCommandEncoderWithDescriptor:pass];
      [render setRenderPipelineState:state];
      [render drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
      [render endEncoding];
      stage("commit");
      [cb commit];
      [cb waitUntilCompleted];
      sayCommandBuffer("cb1", cb, -1);
      laterReads("same_queue_reads_buffer", queue, buffer, nil);
      laterReads("same_queue_reads_texture", queue, buffer, target);
      id<MTLCommandQueue> other = [device newCommandQueue];
      laterReads("new_queue_reads_buffer", other, buffer, nil);
      laterReads("new_queue_reads_texture", other, buffer, target);
    }
    later("same_queue", queue, event, 2);
    later("new_queue", [device newCommandQueue], event, 3);
    usleep(100 * 1000);
    later("same_queue_later", queue, event, 4);
    stage("done");
  }
  return 0;
}
