// What plain Metal answers for the `uint2` overloads of `rasterization_rate_map_decoder`'s
// `map_screen_to_physical_coordinates` and `map_physical_to_screen_coordinates`, beside the `float2`
// ones at the same coordinates and the map's own CPU answers, along each axis of four maps: 64 by 64
// of quality 1; 256 by 128 falling to 0.25; 200 by 96 of two layers; 3648 by 1734 falling to 0.3.
// Each axis is walked at every integer coordinate from 0 to 48 past the larger of its screen and
// physical extents (the other axis held at a third of its screen), at each of the map's layers and
// two past them, which read their horizontal tables unclamped.
//
// Build (no cargo target needed):
//   clang -fobjc-arc -O1 -fmodules -o rate_map_integer_overloads rate_map_integer_overloads.m \
//     -framework Foundation -framework Metal
//
// Run without `MTL_DEBUG_LAYER`. It prints the device and release, then per map a line
// `MAP <name> <W>x<H> layers <L> layer <l> physical <PW>x<PH> size <bytes>` per layer, the parameter
// data's six tables of 1026 words (`WORDS <name> <first word>: <words>`, hex), and per layer, axis
// and coordinate `c`:
//
//   ROW <name> <layer> <axis> <c> | <s2p uint2> | <p2s uint2> | <s2p float2 at c + 1/2>
//       | <p2s float2 at c + 1/2> | <p2s float2 at c> | <s2p float2 at c> | cpu <s2p> | <p2s>
//
// `crates/metal2vulkan/src/bind/rate_map/apple_integer_vectors.txt` is a sample of its output on an
// Apple M3 Max under macOS 27.0.1. The reading it measured (`rate_map_data`, **Decoding**): the
// `uint2` screen-to-physical answer is the `float2` decoder's at `c + 1/2`, converted to `uint`;
// the `uint2` physical-to-screen answer is the bits of the `float` `(0 - offset) * reciprocal` of
// the entry of granule `c >> 5`, the coordinate's value unread.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>

static NSString *const kSource = @
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void probe(constant rasterization_rate_map_data &map [[buffer(0)]],\n"
    "                  device uint *out [[buffer(1)]],\n"
    "                  constant uint4 &args [[buffer(2)]],\n"
    "                  uint tid [[thread_position_in_grid]]) {\n"
    "  rasterization_rate_map_decoder d(map);\n"
    "  uint2 c = args.z == 0 ? uint2(tid, args.y) : uint2(args.x, tid);\n"
    "  uint2 a = d.map_screen_to_physical_coordinates(c, args.w);\n"
    "  uint2 b = d.map_physical_to_screen_coordinates(c, args.w);\n"
    "  float2 fc = float2(c);\n"
    "  float2 f0 = d.map_screen_to_physical_coordinates(fc + 0.5f, args.w);\n"
    "  float2 f1 = d.map_physical_to_screen_coordinates(fc + 0.5f, args.w);\n"
    "  float2 f2 = d.map_physical_to_screen_coordinates(fc, args.w);\n"
    "  float2 f3 = d.map_screen_to_physical_coordinates(fc, args.w);\n"
    "  device uint *o = out + tid * 12;\n"
    "  o[0] = a.x; o[1] = a.y; o[2] = b.x; o[3] = b.y;\n"
    "  o[4] = as_type<uint>(f0.x); o[5] = as_type<uint>(f0.y);\n"
    "  o[6] = as_type<uint>(f1.x); o[7] = as_type<uint>(f1.y);\n"
    "  o[8] = as_type<uint>(f2.x); o[9] = as_type<uint>(f2.y);\n"
    "  o[10] = as_type<uint>(f3.x); o[11] = as_type<uint>(f3.y);\n"
    "}\n";

static float bits(uint32_t w) {
    float f;
    memcpy(&f, &w, 4);
    return f;
}

typedef struct {
    const char *name;
    NSUInteger w, h;
    int layers;
    int hn[2], vn[2];
    float hq[2][8], vq[2][8];
} MapSpec;

static void run(id<MTLDevice> dev, id<MTLComputePipelineState> pso, id<MTLCommandQueue> q,
                const MapSpec *spec) {
    NSMutableArray<MTLRasterizationRateLayerDescriptor *> *layers = [NSMutableArray array];
    for (int l = 0; l < spec->layers; l++) {
        MTLRasterizationRateLayerDescriptor *ld = [[MTLRasterizationRateLayerDescriptor alloc]
            initWithSampleCount:MTLSizeMake(spec->hn[l], spec->vn[l], 0)];
        for (int i = 0; i < spec->hn[l]; i++) ld.horizontalSampleStorage[i] = spec->hq[l][i];
        for (int i = 0; i < spec->vn[l]; i++) ld.verticalSampleStorage[i] = spec->vq[l][i];
        [layers addObject:ld];
    }
    // The array form wants a C array of pointers.
    MTLRasterizationRateLayerDescriptor *arr[2] = {nil, nil};
    for (int l = 0; l < spec->layers; l++) arr[l] = layers[l];
    MTLRasterizationRateMapDescriptor *d = [MTLRasterizationRateMapDescriptor rasterizationRateMapDescriptorWithScreenSize:
                                               MTLSizeMake(spec->w, spec->h, 1)
                                                                             layerCount:spec->layers
                                                                                 layers:arr];
    id<MTLRasterizationRateMap> map = [dev newRasterizationRateMapWithDescriptor:d];
    if (!map) {
        printf("MAP %s nil\n", spec->name);
        return;
    }
    MTLSizeAndAlign sa = map.parameterBufferSizeAndAlign;
    id<MTLBuffer> params = [dev newBufferWithLength:sa.size options:MTLResourceStorageModeShared];
    [map copyParameterDataToBuffer:params offset:0];
    for (int l = 0; l < spec->layers; l++) {
        MTLSize p = [map physicalSizeForLayer:l];
        printf("MAP %s %lux%lu layers %d layer %d physical %lux%lu size %lu\n", spec->name,
               (unsigned long)spec->w, (unsigned long)spec->h, spec->layers, l,
               (unsigned long)p.width, (unsigned long)p.height, (unsigned long)sa.size);
    }
    const uint32_t *words = params.contents;
    NSUInteger nwords = sa.size / 4;
    // Each table, of 1026 words.
    for (NSUInteger t = 0; t < 6; t++) {
        NSUInteger at = t * 1026;
        printf("WORDS %s %lu:", spec->name, (unsigned long)at);
        for (NSUInteger i = 0; i < 1026 && at + i < nwords; i++) printf(" %08x", words[at + i]);
        printf("\n");
    }
    // Two layers past the map's too: the decoders read their horizontal tables unclamped.
    for (int l = 0; l < spec->layers + 2; l++) {
        MTLSize phys = [map physicalSizeForLayer:(l < spec->layers ? l : 0)];
        for (uint32_t axis = 0; axis < 2; axis++) {
            NSUInteger extent = axis == 0 ? spec->w : spec->h;
            NSUInteger pext = axis == 0 ? phys.width : phys.height;
            NSUInteger n = (extent > pext ? extent : pext) + 48;
            uint32_t other = axis == 0 ? (uint32_t)(spec->h / 3) : (uint32_t)(spec->w / 3);
            uint32_t args[4] = {other, other, axis, (uint32_t)l};
            id<MTLBuffer> out = [dev newBufferWithLength:n * 12 * 4 options:MTLResourceStorageModeShared];
            memset(out.contents, 0xab, n * 12 * 4);
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:pso];
            [e setBuffer:params offset:0 atIndex:0];
            [e setBuffer:out offset:0 atIndex:1];
            [e setBytes:args length:sizeof args atIndex:2];
            [e dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [e endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.error) {
                printf("ERROR %s\n", cb.error.localizedDescription.UTF8String);
                return;
            }
            const uint32_t *o = out.contents;
            for (NSUInteger c = 0; c < n; c++) {
                const uint32_t *r = o + c * 12;
                MTLCoordinate2D sc = MTLCoordinate2DMake(axis == 0 ? c + 0.5f : other + 0.5f,
                                                         axis == 0 ? other + 0.5f : c + 0.5f);
                int cl = l < spec->layers ? l : 0;
                MTLCoordinate2D cs = [map mapScreenToPhysicalCoordinates:sc forLayer:cl];
                MTLCoordinate2D cp = [map mapPhysicalToScreenCoordinates:sc forLayer:cl];
                printf("ROW %s %d %u %lu | %u %u | %u %u | %.9g %.9g | %.9g %.9g | %.9g %.9g | %.9g "
                       "%.9g | cpu %.9g %.9g | %.9g %.9g\n",
                       spec->name, l, axis, (unsigned long)c, r[0], r[1], r[2], r[3], bits(r[4]),
                       bits(r[5]), bits(r[6]), bits(r[7]), bits(r[8]), bits(r[9]), bits(r[10]),
                       bits(r[11]), cs.x, cs.y, cp.x, cp.y);
            }
        }
    }
}

int main(void) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        printf("DEVICE %s / RELEASE %s\n", dev.name.UTF8String,
               NSProcessInfo.processInfo.operatingSystemVersionString.UTF8String);
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:kSource options:nil error:&err];
        if (!lib) {
            printf("COMPILE %s\n", err.localizedDescription.UTF8String);
            return 1;
        }
        id<MTLComputePipelineState> pso =
            [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"probe"] error:&err];
        if (!pso) {
            printf("PIPELINE %s\n", err.localizedDescription.UTF8String);
            return 1;
        }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        MapSpec specs[] = {
            {"uniform", 64, 64, 1, {1, 0}, {1, 0}, {{1.0f}}, {{1.0f}}},
            {"falling", 256, 128, 1, {4, 0}, {2, 0}, {{1.0f, 0.5f, 0.25f, 0.5f}}, {{0.5f, 1.0f}}},
            {"twolayer", 200, 96, 2, {2, 3}, {2, 2},
             {{1.0f, 0.3f}, {0.3f, 1.0f, 0.6f}}, {{1.0f, 0.4f}, {0.7f, 0.2f}}},
            {"large", 3648, 1734, 1, {5, 0}, {5, 0}, {{0.3f, 0.6f, 1.0f, 0.6f, 0.3f}},
             {{0.3f, 0.6f, 1.0f, 0.6f, 0.3f}}},
        };
        for (size_t i = 0; i < sizeof specs / sizeof specs[0]; i++) run(dev, pso, q, &specs[i]);
        printf("DONE\n");
    }
    return 0;
}
