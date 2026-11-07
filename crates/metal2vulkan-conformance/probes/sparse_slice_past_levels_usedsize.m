// What plain Metal's heap `usedSize` charges for a sparse mapping update whose slice lies past the
// texture's own slices, on levels OTHER THAN 0 and on the mip tail, through each of the three
// forms a mapping update takes: the direct selector
// `updateTextureMapping:mode:region:mipLevel:slice:`, the indirect selector
// `updateTextureMapping:mode:indirectBuffer:indirectBufferOffset:`, and the direct selector through
// a texture VIEW of the sparse texture.
//
// Why a probe and not a conformance case: Metal's API validation layer refuses an out-of-bounds
// slice through the direct selector (and through a view), and the suite's sparse family bounds
// every region it states by the texture's tile grid on purpose. `sparse_indirect_usedsize.m` and
// `sparse_mapping_past_the_bounds.m` measured the charge on level 0 alone; the runtime applies the
// same rule (`UnitGrid::clipped_region`: the region clipped to the level's tiles, one page a
// tile, the slice's whole tail one unit at `firstMipmapInTail`, nothing at a level inside the
// tail past it) to every other level, unmeasured. This probe measures it.
//
// Build and run (one shape per process, so a configuration is never perturbed by another's
// mapping, and a fault is attributable to the shape that caused it):
//
//   clang -fobjc-arc -framework Metal -framework Foundation \
//       -o sparse_slice_past_levels sparse_slice_past_levels_usedsize.m
//   for form in direct indirect view; do
//     for level in l1 l1-big l2 tail tail-inner l1-unmap; do
//       ./sparse_slice_past_levels $form-$level
//     done
//   done
//
// Run it WITHOUT MTL_DEBUG_LAYER: the layer's assertions end the process on the direct and view
// forms, and plain Metal is the behavioural oracle.
//
// Every shape reads the device's name back off the object, prints `usedSize` twice at each step,
// measures its out-of-bounds region first, and ends with the SAME region on an in-bounds slice
// through the same form -- the control, read after the measurement so the measurement ran alone.
// A shape whose control does not move the count (where the rule says the in-bounds region maps
// something) has measured nothing, whatever its own reading said.
//
// The forms (the slice past the texture's own is slice 2 of a 2-slice texture):
//
//   direct-*    the direct selector on the texture, slice 2; the control slice 0.
//   indirect-*  one `MTLMapIndirectArguments` naming slice 2; the control slice 0.
//   view-*      the direct selector on a view of the texture's slice 1 alone, all nine levels,
//               naming the view's slice 1, which is the texture's slice 2; the control the view's
//               slice 0, the texture's slice 1. (`sparse/encoder.rs`: an update through a view
//               names the texture's level whatever the view's levels, and the texture's slice the
//               view's first plus the view's slice.)
//
// The levels (the texture's level 1 holds 2 by 2 tiles, level 2 one, `firstMipmapInTail` 3):
//
//   l1          level 1, one tile at (0, 0). The rule: one page.
//   l1-big      level 1, 8 by 8 tiles at (0, 0), where the level holds 2 by 2. The rule: four.
//   l2          level 2, one tile, the level's only one. The rule: one page.
//   tail        level 3, `firstMipmapInTail`, one tile at (0, 0). The rule: the slice's tail, as
//               many pages as the control's in-bounds tail takes.
//   tail-inner  level 5, inside the tail past its first level. The rule: nothing, as the control's
//               in-bounds level 5 maps nothing.
//   l1-unmap    level 1, one tile: mapped, then unmapped. The rule: the page given back.
//
// The texture is the one the suite's sparse family uses, as a 2-slice array: 256 by 256 R32Float,
// nine levels, which at a 16-KiB page is a 64 by 64 texel tile. The heap is 4 MiB, far more than
// any shape maps.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static const NSUInteger kSide = 256;
static const NSUInteger kLevels = 9;
static const NSUInteger kSlices = 2;
static const NSUInteger kHeapBytes = 4u << 20;
static const NSUInteger kArgumentWords = 8;

// `usedSize` twice, so that a step whose two readings disagree is visible as such.
static void report(const char *step, id<MTLHeap> heap) {
    unsigned long long first = (unsigned long long)heap.usedSize;
    unsigned long long again = (unsigned long long)heap.usedSize;
    printf("%-40s usedSize = %llu (again %llu)%s\n", step, first, again,
           first == again ? "" : "  <-- THE TWO READINGS DISAGREE");
    fflush(stdout);
}

static void finish(const char *step, id<MTLCommandBuffer> cb, id<MTLHeap> heap) {
    [cb commit];
    [cb waitUntilCompleted];
    printf("%-40s status = %lu error = %s\n", step, (unsigned long)cb.status,
           cb.error == nil ? "(none)" : cb.error.localizedDescription.UTF8String);
    report(step, heap);
}

// One update of `region` of `level` and `slice` through `form`, in a command buffer of its own:
// the direct selector on `texture`, the indirect one over `arguments`, or the direct selector on
// `view` (whose slice 0 is the texture's slice 1).
static void update(const char *step, const char *form, id<MTLCommandQueue> queue,
                   id<MTLTexture> texture, id<MTLTexture> view, id<MTLBuffer> arguments,
                   id<MTLHeap> heap, MTLSparseTextureMappingMode mode, MTLRegion region,
                   NSUInteger level, NSUInteger slice) {
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLResourceStateCommandEncoder> encoder = [cb resourceStateCommandEncoder];
    if (strcmp(form, "indirect") == 0) {
        uint32_t *words = (uint32_t *)arguments.contents;
        words[0] = 1;
        uint32_t *argument = words + 1;
        argument[0] = (uint32_t)region.origin.x;
        argument[1] = (uint32_t)region.origin.y;
        argument[2] = (uint32_t)region.origin.z;
        argument[3] = (uint32_t)region.size.width;
        argument[4] = (uint32_t)region.size.height;
        argument[5] = (uint32_t)region.size.depth;
        argument[6] = (uint32_t)level;
        argument[7] = (uint32_t)slice;
        [encoder updateTextureMapping:texture
                                 mode:mode
                       indirectBuffer:arguments
                 indirectBufferOffset:0];
    } else {
        id<MTLTexture> target = strcmp(form, "view") == 0 ? view : texture;
        [encoder updateTextureMapping:target mode:mode region:region mipLevel:level slice:slice];
    }
    [encoder endEncoding];
    finish(step, cb, heap);
}

int main(int argc, const char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <form>-<level>  (the names are in this file's header)\n",
                argv[0]);
        return 2;
    }
    const char *shape = argv[1];
    const char *dash = strchr(shape, '-');
    if (dash == NULL) {
        fprintf(stderr, "unknown shape %s\n", shape);
        return 2;
    }
    char form[16] = {0};
    memcpy(form, shape, MIN((size_t)(dash - shape), sizeof form - 1));
    const char *level_name = dash + 1;
    if (strcmp(form, "direct") != 0 && strcmp(form, "indirect") != 0 &&
        strcmp(form, "view") != 0) {
        fprintf(stderr, "unknown form %s\n", form);
        return 2;
    }
    NSUInteger level;
    MTLRegion region = MTLRegionMake3D(0, 0, 0, 1, 1, 1);
    BOOL unmap = NO;
    if (strcmp(level_name, "l1") == 0) {
        level = 1;
    } else if (strcmp(level_name, "l1-big") == 0) {
        level = 1;
        region = MTLRegionMake3D(0, 0, 0, 8, 8, 1);
    } else if (strcmp(level_name, "l2") == 0) {
        level = 2;
    } else if (strcmp(level_name, "tail") == 0) {
        level = 3;
    } else if (strcmp(level_name, "tail-inner") == 0) {
        level = 5;
    } else if (strcmp(level_name, "l1-unmap") == 0) {
        level = 1;
        unmap = YES;
    } else {
        fprintf(stderr, "unknown level %s\n", level_name);
        return 2;
    }
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            fprintf(stderr, "no Metal device\n");
            return 2;
        }

        MTLTextureDescriptor *td = [[MTLTextureDescriptor alloc] init];
        td.textureType = MTLTextureType2DArray;
        td.pixelFormat = MTLPixelFormatR32Float;
        td.width = kSide;
        td.height = kSide;
        td.mipmapLevelCount = kLevels;
        td.arrayLength = kSlices;
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
        // The texture's slice 1 alone, every level.
        id<MTLTexture> view = [texture newTextureViewWithPixelFormat:MTLPixelFormatR32Float
                                                         textureType:MTLTextureType2DArray
                                                              levels:NSMakeRange(0, kLevels)
                                                              slices:NSMakeRange(1, 1)];
        if (view == nil) {
            fprintf(stderr, "no view\n");
            return 2;
        }

        MTLSize tile = [device sparseTileSizeWithTextureType:td.textureType
                                                pixelFormat:td.pixelFormat
                                                sampleCount:1
                                             sparsePageSize:MTLSparsePageSize16];
        printf("shape=%s device=\"%s\" layer=%s\n", shape, device.name.UTF8String,
               getenv("MTL_DEBUG_LAYER") == NULL ? "off" : getenv("MTL_DEBUG_LAYER"));
        printf("slices=%lu levels=%lu tile=%lux%lux%lu tileBytes=%lu firstMipmapInTail=%lu "
               "tailBytes=%lu view slices=%lu\n",
               (unsigned long)texture.arrayLength, (unsigned long)texture.mipmapLevelCount,
               (unsigned long)tile.width, (unsigned long)tile.height, (unsigned long)tile.depth,
               (unsigned long)device.sparseTileSizeInBytes,
               (unsigned long)texture.firstMipmapInTail,
               (unsigned long)texture.tailSizeInBytes, (unsigned long)view.arrayLength);
        fflush(stdout);

        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLBuffer> arguments = [device newBufferWithLength:4 * (1 + kArgumentWords)
                                                      options:MTLResourceStorageModeShared];
        memset(arguments.contents, 0, arguments.length);

        // The view's slice 1 is the texture's slice 2; its slice 0 the texture's slice 1.
        BOOL viewing = strcmp(form, "view") == 0;
        NSUInteger past = viewing ? 1 : kSlices;
        NSUInteger inside = 0;

        report("start", heap);
        char step[64];
        snprintf(step, sizeof step, "%s map level %lu, slice past", form, (unsigned long)level);
        update(step, form, queue, texture, view, arguments, heap, MTLSparseTextureMappingModeMap,
               region, level, past);
        if (unmap) {
            snprintf(step, sizeof step, "%s UNMAP level %lu, slice past", form,
                     (unsigned long)level);
            update(step, form, queue, texture, view, arguments, heap,
                   MTLSparseTextureMappingModeUnmap, region, level, past);
        }
        snprintf(step, sizeof step, "control: %s map level %lu, in bounds", form,
                 (unsigned long)level);
        update(step, form, queue, texture, view, arguments, heap, MTLSparseTextureMappingModeMap,
               region, level, inside);
        if (unmap) {
            snprintf(step, sizeof step, "control: %s UNMAP level %lu, in bounds", form,
                     (unsigned long)level);
            update(step, form, queue, texture, view, arguments, heap,
                   MTLSparseTextureMappingModeUnmap, region, level, inside);
        }
    }
    return 0;
}
