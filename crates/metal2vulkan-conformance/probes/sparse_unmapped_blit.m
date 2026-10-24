// What Metal does with a BLIT over an UNMAPPED sparse tile, on this chip.
//
// The runtime's own docs say a sparse texture's unmapped tiles read zero and that a blit's writes
// into them are dropped, and they attribute that to measurements on an M3 Max and an M4 -- but
// those measurements were a SHADER's read(), never a blit's. This probe measures the blit, both
// ways round, and the one question neither read can answer on its own: whether a write made while
// a tile was unmapped is still there when the tile is mapped afterwards.
//
// Each reading is printed as a per-tile verdict, in this order; each has a blit `-out` half and a
// kernel `-shader` half, so the two read paths are always side by side:
//
//   after-blit          copyFromBuffer: a nonzero pattern over the WHOLE level, then the level read
//                       back. The blit read is the half nothing had measured; the kernel read is
//                       the half already measured, here as the CONTROL that this texture's mapping
//                       is what the probe thinks it is.
//   after-map           one tile that was unmapped during the blit is MAPPED, and the level read
//                       again: zero would say Metal dropped the write and zeroed the tile, the
//                       pattern would say it kept the write somewhere, and anything else is a third
//                       answer the next reading is there to name.
//   fresh-plain         an ORDINARY Private texture of the same format and extent, off any heap,
//                       read before anything writes it. This is what says whether the third answer
//                       is Metal's unwritten pattern or undefined memory, and it is the one leg
//                       without which no claim about the newly mapped tile could be made.
//   after-remap         one tile that was MAPPED during the blit is unmapped and mapped again, then
//                       read: whether a round trip through unmapped loses its texels.
//   after-second-blit   a second copyFromBuffer: over the whole level, which is the control that a
//                       mapped tile takes a blit at all -- without it "the write was dropped" and
//                       "no copy ever ran" read the same.
//
// WHAT IT MEASURED, Apple M3 Max, macOS 27.0.1 (26A434), three runs each in its own process, all
// three identical, over a 256 by 256 R32Uint texture of 64-texel (16 KiB) tiles with eight of its
// sixteen mapped in a checkerboard (usedSize 131,072):
//
//   after-blit        every one of the 4,096 texels of each MAPPED tile holds the copy's word;
//                     every texel of each UNMAPPED tile is ZERO. The blit read and the kernel read
//                     agree tile for tile, so the blit half now has the answer the shader half had.
//   after-map         the newly mapped tile (usedSize 147,456, exactly one tile more) reads
//                     0xFFFFFFFF at all 4,096 texels: not zero, and not the dropped write.
//   fresh-plain       a fresh ordinary Private texture of the same format reads 0xFFFFFFFF at all
//                     65,536 of its texels, so that value is Metal's UNWRITTEN PATTERN and the
//                     newly mapped tile is simply still unwritten as far as Metal is concerned.
//   after-remap       the round-tripped tile reads 0xFFFFFFFF too: its texels are gone.
//   after-second-blit the two now-mapped tiles hold the copy's words and the seven still-unmapped
//                     ones are still zero.
//
// THE REGION OF updateTextureMapping: IS IN TILES, NOT TEXELS. A texel-unit region reads as a tile
// region far past the grid, which Metal CLIPS to the whole grid -- so the first "checkerboard" this
// probe asked for mapped all sixteen tiles and every reading said `pattern`, which is what a
// vacuous measurement looks like from the inside. usedSize is the tell: 262,144 bytes is sixteen
// tiles, not eight.
//
// A tile's verdict is read off EVERY texel of it, not a corner: `pattern` means every texel is the
// one the pattern put there, `zero` every texel is 0, and anything else is printed as a tally so a
// partly-written tile cannot read as either.
//
// The tiles are a CHECKERBOARD, so every reading has mapped and unmapped tiles side by side and
// "the blit did nothing at all" is visible as the mapped tiles reading zero too.
//
// ONE PROCESS, because each reading depends on the one before it and a sparse fault would end the
// process whatever the shape. Nothing here loops and nothing indexes outside a bound buffer.
//
//   clang -fobjc-arc -framework Metal -framework Foundation -o sparse_unmapped_blit sparse_unmapped_blit.m

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// The level's extent in tiles. 4 by 4 keeps the texture small and still gives every tile a
// neighbour of the other state on both axes.
static const NSUInteger kTilesX = 4;
static const NSUInteger kTilesY = 4;

// The word the pattern writes at texel (x, y) of the level: never zero, and its own number, so a
// texel that moved between tiles is visible as well as one that was dropped.
static uint32_t patternAt(NSUInteger x, NSUInteger y, NSUInteger width) {
  return (uint32_t)(1 + y * width + x);
}

static NSString *kSource =
  @"#include <metal_stdlib>\n"
   "using namespace metal;\n"
   "kernel void readall(texture2d<uint, access::read> src [[texture(0)]],\n"
   "                    device uint *out [[buffer(0)]],\n"
   "                    constant uint &width [[buffer(1)]],\n"
   "                    uint2 gid [[thread_position_in_grid]]) {\n"
   "  out[gid.y * width + gid.x] = src.read(gid).x;\n"
   "}\n";

static void say(const char *key, NSString *value) {
  printf("%s=%s\n", key, value.UTF8String);
  fflush(stdout);
}

// One tile's verdict over `words`, a whole level of `width` by `height` uints.
static NSString *tileVerdict(const uint32_t *words, NSUInteger width,
                             NSUInteger tileW, NSUInteger tileH,
                             NSUInteger tx, NSUInteger ty) {
  NSUInteger zero = 0, pattern = 0, other = 0;
  uint32_t firstOther = 0;
  for (NSUInteger y = ty * tileH; y < (ty + 1) * tileH; y++) {
    for (NSUInteger x = tx * tileW; x < (tx + 1) * tileW; x++) {
      uint32_t held = words[y * width + x];
      if (held == 0) zero++;
      else if (held == patternAt(x, y, width)) pattern++;
      else { if (other == 0) firstOther = held; other++; }
    }
  }
  NSUInteger total = tileW * tileH;
  if (zero == total) return @"zero";
  if (pattern == total) return @"pattern";
  return [NSString stringWithFormat:@"mixed(zero %lu, pattern %lu, other %lu, first-other %u)",
                   (unsigned long)zero, (unsigned long)pattern, (unsigned long)other, firstOther];
}

static void report(const char *reading, const uint32_t *words, NSUInteger width,
                   NSUInteger tileW, NSUInteger tileH) {
  for (NSUInteger ty = 0; ty < kTilesY; ty++) {
    NSMutableString *row = [NSMutableString string];
    for (NSUInteger tx = 0; tx < kTilesX; tx++) {
      if (tx) [row appendString:@" | "];
      [row appendString:tileVerdict(words, width, tileW, tileH, tx, ty)];
    }
    char key[128];
    snprintf(key, sizeof(key), "%s-row%lu", reading, (unsigned long)ty);
    say(key, row);
  }
}

// Whether tile (tx, ty) is mapped at the start: a checkerboard.
static BOOL mappedAtStart(NSUInteger tx, NSUInteger ty) { return ((tx + ty) % 2) == 0; }

int main(void) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) { say("device", @"none"); return 1; }
    say("device", device.name);
    say("os", [[NSProcessInfo processInfo] operatingSystemVersionString]);

    MTLPixelFormat format = MTLPixelFormatR32Uint;
    MTLSize tile = [device sparseTileSizeWithTextureType:MTLTextureType2D
                                            pixelFormat:format
                                            sampleCount:1];
    say("tile-texels", [NSString stringWithFormat:@"%lux%lu", (unsigned long)tile.width, (unsigned long)tile.height]);
    say("tile-bytes", [NSString stringWithFormat:@"%lu", (unsigned long)device.sparseTileSizeInBytes]);

    NSUInteger tileW = tile.width, tileH = tile.height;
    NSUInteger width = tileW * kTilesX, height = tileH * kTilesY;
    say("level", [NSString stringWithFormat:@"%lux%lu", (unsigned long)width, (unsigned long)height]);

    MTLTextureDescriptor *td =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                         width:width height:height mipmapped:NO];
    td.usage = MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModePrivate;

    MTLHeapDescriptor *hd = [MTLHeapDescriptor new];
    hd.type = MTLHeapTypeSparse;
    hd.storageMode = MTLStorageModePrivate;
    hd.size = 64 * 1024 * 1024;
    id<MTLHeap> heap = [device newHeapWithDescriptor:hd];
    if (!heap) { say("heap", @"nil"); return 1; }
    say("heap-size", [NSString stringWithFormat:@"%lu", (unsigned long)heap.size]);
    say("heap-used-before", [NSString stringWithFormat:@"%lu", (unsigned long)heap.usedSize]);

    id<MTLTexture> texture = [heap newTextureWithDescriptor:td];
    if (!texture) { say("texture", @"nil"); return 1; }
    say("texture-is-sparse", texture.isSparse ? @"YES" : @"NO");

    NSUInteger texels = width * height;
    NSUInteger bytes = texels * 4;
    id<MTLBuffer> source = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    uint32_t *src = source.contents;
    for (NSUInteger y = 0; y < height; y++)
      for (NSUInteger x = 0; x < width; x++) src[y * width + x] = patternAt(x, y, width);
    id<MTLBuffer> out = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> shaderOut = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    uint32_t w32 = (uint32_t)width;
    id<MTLBuffer> widthBuf = [device newBufferWithBytes:&w32 length:4 options:MTLResourceStorageModeShared];

    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:kSource options:nil error:&error];
    if (!library) { say("library-error", error.description); return 1; }
    id<MTLComputePipelineState> readall =
      [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"readall"]
                                           error:&error];
    if (!readall) { say("pipeline-error", error.description); return 1; }

    id<MTLCommandQueue> queue = [device newCommandQueue];

    // ---- map the checkerboard -----------------------------------------------------------------
    {
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      id<MTLResourceStateCommandEncoder> enc = [cb resourceStateCommandEncoder];
      for (NSUInteger ty = 0; ty < kTilesY; ty++)
        for (NSUInteger tx = 0; tx < kTilesX; tx++)
          if (mappedAtStart(tx, ty))
            [enc updateTextureMapping:texture
                                 mode:MTLSparseTextureMappingModeMap
                               region:MTLRegionMake2D(tx, ty, 1, 1)
                            mipLevel:0 slice:0];
      [enc endEncoding];
      say("about-to-commit", @"mapping");
      [cb commit];
      [cb waitUntilCompleted];
      say("mapping-status", [NSString stringWithFormat:@"%ld", (long)cb.status]);
      say("mapping-error", cb.error ? cb.error.description : @"none");
      say("heap-used-after-map", [NSString stringWithFormat:@"%lu", (unsigned long)heap.usedSize]);
    }

    // ---- the blit in, and both reads out ------------------------------------------------------
    void (^readOut)(const char *) = ^(const char *name) {
      memset(out.contents, 0xCC, bytes);
      memset(shaderOut.contents, 0xCC, bytes);
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
      [blit copyFromTexture:texture sourceSlice:0 sourceLevel:0
               sourceOrigin:MTLOriginMake(0, 0, 0)
                 sourceSize:MTLSizeMake(width, height, 1)
                   toBuffer:out destinationOffset:0
          destinationBytesPerRow:width * 4 destinationBytesPerImage:bytes];
      [blit endEncoding];
      id<MTLComputeCommandEncoder> compute = [cb computeCommandEncoder];
      [compute setComputePipelineState:readall];
      [compute setTexture:texture atIndex:0];
      [compute setBuffer:shaderOut offset:0 atIndex:0];
      [compute setBuffer:widthBuf offset:0 atIndex:1];
      [compute dispatchThreads:MTLSizeMake(width, height, 1)
        threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
      [compute endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      char key[128];
      snprintf(key, sizeof(key), "%s-status", name);
      say(key, [NSString stringWithFormat:@"%ld", (long)cb.status]);
      snprintf(key, sizeof(key), "%s-error", name);
      say(key, cb.error ? cb.error.description : @"none");
      char outName[128], shName[128];
      snprintf(outName, sizeof(outName), "%s-out", name);
      snprintf(shName, sizeof(shName), "%s-shader", name);
      report(outName, out.contents, width, tileW, tileH);
      report(shName, shaderOut.contents, width, tileW, tileH);
    };

    {
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
      [blit copyFromBuffer:source sourceOffset:0 sourceBytesPerRow:width * 4
          sourceBytesPerImage:bytes sourceSize:MTLSizeMake(width, height, 1)
                    toTexture:texture destinationSlice:0 destinationLevel:0
            destinationOrigin:MTLOriginMake(0, 0, 0)];
      [blit endEncoding];
      say("about-to-commit", @"blit-in");
      [cb commit];
      [cb waitUntilCompleted];
      say("blit-in-status", [NSString stringWithFormat:@"%ld", (long)cb.status]);
      say("blit-in-error", cb.error ? cb.error.description : @"none");
    }
    say("reading", @"after-blit");
    readOut("after-blit");

    // ---- map one tile that was unmapped during the blit ---------------------------------------
    say("newly-mapped-tile", @"1,0");
    {
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      id<MTLResourceStateCommandEncoder> enc = [cb resourceStateCommandEncoder];
      [enc updateTextureMapping:texture mode:MTLSparseTextureMappingModeMap
                         region:MTLRegionMake2D(1, 0, 1, 1)
                      mipLevel:0 slice:0];
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      say("map-one-status", [NSString stringWithFormat:@"%ld", (long)cb.status]);
      say("heap-used-after-map-one", [NSString stringWithFormat:@"%lu", (unsigned long)heap.usedSize]);
    }
    say("reading", @"after-map");
    readOut("after-map");

    // ---- a round trip through unmapped, on a tile that held the pattern -----------------------
    say("round-trip-tile", @"0,0");
    {
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      id<MTLResourceStateCommandEncoder> enc = [cb resourceStateCommandEncoder];
      [enc updateTextureMapping:texture mode:MTLSparseTextureMappingModeUnmap
                         region:MTLRegionMake2D(0, 0, 1, 1) mipLevel:0 slice:0];
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      say("unmap-status", [NSString stringWithFormat:@"%ld", (long)cb.status]);
      say("heap-used-after-unmap", [NSString stringWithFormat:@"%lu", (unsigned long)heap.usedSize]);
      cb = [queue commandBuffer];
      enc = [cb resourceStateCommandEncoder];
      [enc updateTextureMapping:texture mode:MTLSparseTextureMappingModeMap
                         region:MTLRegionMake2D(0, 0, 1, 1) mipLevel:0 slice:0];
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      say("remap-status", [NSString stringWithFormat:@"%ld", (long)cb.status]);
      say("heap-used-after-remap", [NSString stringWithFormat:@"%lu", (unsigned long)heap.usedSize]);
    }
    say("reading", @"after-remap");
    readOut("after-remap");

    // ---- the control for the 0xffffffff: an ORDINARY Private texture of the same format and
    // extent, read before anything writes it. If a newly mapped sparse tile reads what a fresh
    // ordinary texture reads, then what it holds is Metal's unwritten pattern and not the dropped
    // write and not undefined memory -- and that is a claim only this leg can make.
    {
      id<MTLTexture> fresh = [device newTextureWithDescriptor:td];
      memset(out.contents, 0xCC, bytes);
      memset(shaderOut.contents, 0xCC, bytes);
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
      [blit copyFromTexture:fresh sourceSlice:0 sourceLevel:0
               sourceOrigin:MTLOriginMake(0, 0, 0)
                 sourceSize:MTLSizeMake(width, height, 1)
                   toBuffer:out destinationOffset:0
          destinationBytesPerRow:width * 4 destinationBytesPerImage:bytes];
      [blit endEncoding];
      id<MTLComputeCommandEncoder> compute = [cb computeCommandEncoder];
      [compute setComputePipelineState:readall];
      [compute setTexture:fresh atIndex:0];
      [compute setBuffer:shaderOut offset:0 atIndex:0];
      [compute setBuffer:widthBuf offset:0 atIndex:1];
      [compute dispatchThreads:MTLSizeMake(width, height, 1)
        threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
      [compute endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      say("fresh-plain-status", [NSString stringWithFormat:@"%ld", (long)cb.status]);
      report("fresh-plain-out", out.contents, width, tileW, tileH);
      report("fresh-plain-shader", shaderOut.contents, width, tileW, tileH);
    }

    // ---- and the control that a newly mapped tile takes a write at all --------------------------
    {
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
      [blit copyFromBuffer:source sourceOffset:0 sourceBytesPerRow:width * 4
          sourceBytesPerImage:bytes sourceSize:MTLSizeMake(width, height, 1)
                    toTexture:texture destinationSlice:0 destinationLevel:0
            destinationOrigin:MTLOriginMake(0, 0, 0)];
      [blit endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      say("second-blit-status", [NSString stringWithFormat:@"%ld", (long)cb.status]);
    }
    say("reading", @"after-second-blit");
    readOut("after-second-blit");

    NSMutableString *layout = [NSMutableString string];
    for (NSUInteger ty = 0; ty < kTilesY; ty++) {
      for (NSUInteger tx = 0; tx < kTilesX; tx++)
        [layout appendString:mappedAtStart(tx, ty) ? @"M" : @"."];
      [layout appendString:@"/"];
    }
    say("mapping-at-blit", layout);
  }
  return 0;
}
