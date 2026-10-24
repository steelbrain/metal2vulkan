// What plain Metal does with a sparse mapping update whose level, slice or region lies past the
// texture's own, through the DIRECT selector
// `updateTextureMapping:mode:region:mipLevel:slice:`.
//
// Why a probe and not a conformance case: Metal's API validation layer refuses all four shapes,
// and the suite's own sparse family bounds every region it states by the texture's tile grid on
// purpose, so a case cannot ask these questions without giving up that bound. The runtime's
// indirect path already follows plain Metal and clips (`UnitGrid::clipped_region`, measured for
// the indirect arguments), while its direct path refuses typed (`UnitGrid::checked_region`), and
// what settles which of the two is right is what plain Metal does with the direct call.
//
// Build and run (one shape per process, so a fault is attributable to the shape that caused it):
//
//   clang -fobjc-arc -fmodules -framework Metal -framework Foundation \
//       -o /tmp/sparse_bounds sparse_mapping_past_the_bounds.m
//   for shape in control level slice region origin; do /tmp/sparse_bounds $shape; done
//
// Run it WITHOUT MTL_DEBUG_LAYER: the layer's assertions end the process, and plain Metal is the
// behavioural oracle. Each shape prints the heap's `usedSize` before and after the commit, the
// command buffer's status and its error, so "Metal took the call and mapped nothing" is
// distinguishable from "Metal mapped something" and from "the command buffer failed".
//
// The shape names, and what each asks:
//
//   control  one tile of level 0, which must raise usedSize by one page: without it a run where
//            every reading stayed at 0 would read the same as a run where nothing works at all.
//   level    mipLevel 9 of a 9-level texture.
//   slice    slice 2 of a 2-slice array texture.
//   region   level 0, origin 0, 8 by 8 tiles where the level holds 4 by 4.
//   origin   level 0, origin (4, 4) where the level's last tile is (3, 3), one tile.
//
// The same three bounds through the MOVE selector
// `moveTextureMappingsFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:…`,
// which the runtime's direct update shares a bounds check with. A move changes no `usedSize`, so
// what these shapes can say is whether Metal **takes the call** — a command buffer that completes
// with no error — and not which tiles it moved, which needs a texel read this probe does not do.
// Each maps one tile of level 0 slice 0 first, so the source has something to move.
//
//   move-level   source level 9 of a 9-level texture, onto level 0.
//   move-slice   source slice 2 of a 2-slice array texture, onto slice 0.
//   move-region  source size 8 by 8 tiles where level 0 holds 4 by 4.
//
// The texture is the one the suite's sparse family uses: 256 by 256 R32Float, nine levels, which
// at a 16-KiB page is a 64 by 64 texel tile, so level 0 holds 4 by 4 tiles, level 1 holds 2 by 2,
// level 2 holds one and `firstMipmapInTail` is 3. The heap is 4 MiB, far more than one tile.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static const NSUInteger kSide = 256;
static const NSUInteger kLevels = 9;
static const NSUInteger kHeapBytes = 4u << 20;

int main(int argc, const char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s control|level|slice|region|origin\n", argv[0]);
        return 2;
    }
    const char *shape = argv[1];
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            fprintf(stderr, "no Metal device\n");
            return 2;
        }
        BOOL moving = strncmp(shape, "move-", 5) == 0;
        BOOL array = strcmp(shape, "slice") == 0 || strcmp(shape, "move-slice") == 0;

        MTLTextureDescriptor *td = [[MTLTextureDescriptor alloc] init];
        td.textureType = array ? MTLTextureType2DArray : MTLTextureType2D;
        td.pixelFormat = MTLPixelFormatR32Float;
        td.width = kSide;
        td.height = kSide;
        td.mipmapLevelCount = kLevels;
        td.arrayLength = array ? 2 : 1;
        td.storageMode = MTLStorageModePrivate;
        td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;

        MTLHeapDescriptor *hd = [[MTLHeapDescriptor alloc] init];
        hd.type = MTLHeapTypeSparse;
        hd.storageMode = MTLStorageModePrivate;
        hd.sparsePageSize = MTLSparsePageSize16;
        hd.size = kHeapBytes;
        id<MTLHeap> heap = [device newHeapWithDescriptor:hd];
        if (heap == nil) {
            fprintf(stderr, "no sparse heap\n");
            return 2;
        }
        id<MTLTexture> texture = [heap newTextureWithDescriptor:td];
        if (texture == nil) {
            fprintf(stderr, "no sparse texture\n");
            return 2;
        }

        MTLSize tile = [device sparseTileSizeWithTextureType:td.textureType
                                                pixelFormat:td.pixelFormat
                                                sampleCount:1
                                             sparsePageSize:MTLSparsePageSize16];
        printf("shape=%s tile=%lux%lux%lu firstMipmapInTail=%lu tileBytes=%lu\n", shape,
               (unsigned long)tile.width, (unsigned long)tile.height, (unsigned long)tile.depth,
               (unsigned long)texture.firstMipmapInTail, (unsigned long)device.sparseTileSizeInBytes);

        // The region is stated in TILES, which is the unit these selectors take.
        MTLRegion region = MTLRegionMake3D(0, 0, 0, 1, 1, 1);
        NSUInteger level = 0;
        NSUInteger slice = 0;
        if (strcmp(shape, "level") == 0 || strcmp(shape, "move-level") == 0) {
            level = kLevels;
        } else if (strcmp(shape, "slice") == 0 || strcmp(shape, "move-slice") == 0) {
            slice = 2;
        } else if (strcmp(shape, "region") == 0 || strcmp(shape, "move-region") == 0) {
            region = MTLRegionMake3D(0, 0, 0, 8, 8, 1);
        } else if (strcmp(shape, "origin") == 0) {
            region = MTLRegionMake3D(4, 4, 0, 1, 1, 1);
        } else if (strcmp(shape, "control") != 0) {
            fprintf(stderr, "unknown shape %s\n", shape);
            return 2;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];

        // A move needs a mapped source tile to have anything to move, in a command buffer of its
        // own so its own completion is not what the shape's reading reports.
        if (moving) {
            id<MTLCommandBuffer> first = [queue commandBuffer];
            id<MTLResourceStateCommandEncoder> e = [first resourceStateCommandEncoder];
            [e updateTextureMapping:texture
                               mode:MTLSparseTextureMappingModeMap
                             region:MTLRegionMake3D(0, 0, 0, 1, 1, 1)
                           mipLevel:0
                              slice:0];
            [e endEncoding];
            [first commit];
            [first waitUntilCompleted];
            printf("mapped one tile: status = %lu usedSize = %lu\n", (unsigned long)first.status,
                   (unsigned long)heap.usedSize);
        }

        printf("usedSize before = %lu\n", (unsigned long)heap.usedSize);
        fflush(stdout);

        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLResourceStateCommandEncoder> encoder = [cb resourceStateCommandEncoder];
        if (moving) {
            [encoder moveTextureMappingsFromTexture:texture
                                       sourceSlice:slice
                                       sourceLevel:level
                                      sourceOrigin:region.origin
                                        sourceSize:region.size
                                         toTexture:texture
                                  destinationSlice:0
                                  destinationLevel:0
                                 destinationOrigin:MTLOriginMake(0, 0, 0)];
        } else {
            [encoder updateTextureMapping:texture
                                    mode:MTLSparseTextureMappingModeMap
                                  region:region
                                mipLevel:level
                                   slice:slice];
        }
        [encoder endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        printf("usedSize after  = %lu\n", (unsigned long)heap.usedSize);
        printf("status = %lu error = %s\n", (unsigned long)cb.status,
               cb.error == nil ? "(none)" : cb.error.localizedDescription.UTF8String);
    }
    return 0;
}
