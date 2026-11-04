// Plain Metal's answer, with the API validation layer OFF, for a kernel that encodes command 0 of
// an indirect command buffer (its pipeline state taken from the argument buffer that names the
// command buffer) and a later encoder of the same command buffer that executes it, by where the
// command buffer makes the indirect command buffer resident for writing:
//
//   same     in the kernel's encoder (Metal's documented contract);
//   after    only in a compute encoder after the kernel's, which runs nothing;
//   before   only in a compute encoder before the kernel's, which runs nothing;
//   none     nowhere.
//
// Each shape runs for a render command (`irw_encode_draw`: a green triangle over a 16x16 target
// cleared to red), a render command of a buffer that inherits the encoder's state
// (`irw_encode_point`: a green point 4 pixels wide at the target's centre, beside command 1, which
// the CPU wrote, a green triangle in the target's bottom-left corner, run over commands 0 and 1),
// and a compute command (`irw_encode_dispatch`: `irw_mark` writes 0x100 plus its thread's index
// into words 0 to 3 of a zeroed buffer). It prints, per kind, whether the executed commands ran:
// the target's centre texel (and the corner's), and the buffer's four words. The functions are the
// owned fixture `crates/metal2vulkan-validation/fixtures/public/engine/icb_routed_writes.metal`,
// whose path is the second argument; the runtime's lane
// `runtime_lane::write_routes` runs the same shapes.
//
// Build (no GPU needed to build):
//
//   clang -fobjc-arc -fmodules -framework Metal -framework Foundation \
//       -o icb_routed_writes icb_routed_writes.m
//
// Run ONE SHAPE PER PROCESS, WITHOUT MTL_DEBUG_LAYER:
//
//   ./icb_routed_writes <same|after|before|none> <path of icb_routed_writes.metal>
//
// A shape whose process Metal ends prints no `DONE`, and its exit status is the reading.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

typedef enum { SAME, AFTER, BEFORE, NONE } Shape;

static void fail(NSString *what, NSError *error) {
    printf("FAIL %s: %s\n", what.UTF8String, error.localizedDescription.UTF8String);
    exit(2);
}

// A compute encoder of `buffer` that makes `icb` resident for writing and runs nothing.
static void residentAlone(id<MTLCommandBuffer> buffer, id<MTLIndirectCommandBuffer> icb) {
    id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];
    [encoder useResource:icb usage:MTLResourceUsageWrite];
    [encoder endEncoding];
}

// The kernel `writer` over `table` and `argument`, with `icb` made resident for writing in its
// encoder, or only in an encoder before or after it, as `shape` says.
static void encodeWriter(id<MTLCommandBuffer> buffer, Shape shape,
                         id<MTLComputePipelineState> writer, id<MTLBuffer> table,
                         id<MTLBuffer> argument, id<MTLIndirectCommandBuffer> icb) {
    if (shape == BEFORE) {
        residentAlone(buffer, icb);
    }
    id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];
    if (shape == SAME) {
        [encoder useResource:icb usage:MTLResourceUsageWrite];
    }
    [encoder setComputePipelineState:writer];
    [encoder setBuffer:table offset:0 atIndex:0];
    [encoder setBuffer:argument offset:0 atIndex:1];
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];
    if (shape == AFTER) {
        residentAlone(buffer, icb);
    }
}

// A table of `icb`'s ID and `state`'s, as the kernels' argument structs lay them out.
static id<MTLBuffer> tableOf(id<MTLDevice> device, MTLResourceID icb, MTLResourceID state) {
    MTLResourceID ids[2] = {icb, state};
    return [device newBufferWithBytes:ids length:sizeof ids options:MTLResourceStorageModeShared];
}

static void render(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLLibrary> library,
                   Shape shape) {
    NSError *error = nil;
    MTLRenderPipelineDescriptor *descriptor = [MTLRenderPipelineDescriptor new];
    descriptor.vertexFunction = [library newFunctionWithName:@"irw_vertex"];
    descriptor.fragmentFunction = [library newFunctionWithName:@"irw_fragment"];
    descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    descriptor.supportIndirectCommandBuffers = YES;
    id<MTLRenderPipelineState> state = [device newRenderPipelineStateWithDescriptor:descriptor
                                                                              error:&error];
    if (!state) fail(@"render state", error);
    id<MTLComputePipelineState> writer = [device
        newComputePipelineStateWithFunction:[library newFunctionWithName:@"irw_encode_draw"]
                                      error:&error];
    if (!writer) fail(@"draw writer", error);
    MTLIndirectCommandBufferDescriptor *commands = [MTLIndirectCommandBufferDescriptor new];
    commands.commandTypes = MTLIndirectCommandTypeDraw;
    commands.inheritPipelineState = NO;
    commands.inheritBuffers = NO;
    commands.maxVertexBufferBindCount = 1;
    commands.maxFragmentBufferBindCount = 0;
    id<MTLIndirectCommandBuffer> icb = [device newIndirectCommandBufferWithDescriptor:commands
                                                                      maxCommandCount:1
                                                                              options:0];
    float corners[12] = {-1, -1, 0, 1, 3, -1, 0, 1, -1, 3, 0, 1};
    id<MTLBuffer> cornerBuffer = [device newBufferWithBytes:corners
                                                     length:sizeof corners
                                                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> table = tableOf(device, icb.gpuResourceID, state.gpuResourceID);
    MTLTextureDescriptor *targetDescriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                           width:16
                                                          height:16
                                                       mipmapped:NO];
    targetDescriptor.usage = MTLTextureUsageRenderTarget;
    targetDescriptor.storageMode = MTLStorageModeShared;
    id<MTLTexture> target = [device newTextureWithDescriptor:targetDescriptor];
    id<MTLCommandBuffer> buffer = [queue commandBuffer];
    encodeWriter(buffer, shape, writer, table, cornerBuffer, icb);
    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = target;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(1, 0, 0, 1);
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> encoder = [buffer renderCommandEncoderWithDescriptor:pass];
    [encoder useResource:cornerBuffer usage:MTLResourceUsageRead stages:MTLRenderStageVertex];
    [encoder executeCommandsInBuffer:icb withRange:NSMakeRange(0, 1)];
    [encoder endEncoding];
    [buffer commit];
    [buffer waitUntilCompleted];
    uint8_t texel[4] = {0};
    [target getBytes:texel bytesPerRow:64 fromRegion:MTLRegionMake2D(8, 8, 1, 1) mipmapLevel:0];
    printf("render: status %ld error %s; centre texel %u %u %u %u\n", (long)buffer.status,
           buffer.error ? buffer.error.localizedDescription.UTF8String : "none", texel[0],
           texel[1], texel[2], texel[3]);
}

static void points(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLLibrary> library,
                   Shape shape) {
    NSError *error = nil;
    MTLRenderPipelineDescriptor *descriptor = [MTLRenderPipelineDescriptor new];
    descriptor.vertexFunction = [library newFunctionWithName:@"irw_point_vertex"];
    descriptor.fragmentFunction = [library newFunctionWithName:@"irw_fragment"];
    descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    descriptor.supportIndirectCommandBuffers = YES;
    id<MTLRenderPipelineState> state = [device newRenderPipelineStateWithDescriptor:descriptor
                                                                              error:&error];
    if (!state) fail(@"point state", error);
    id<MTLComputePipelineState> writer = [device
        newComputePipelineStateWithFunction:[library newFunctionWithName:@"irw_encode_point"]
                                      error:&error];
    if (!writer) fail(@"point writer", error);
    MTLIndirectCommandBufferDescriptor *commands = [MTLIndirectCommandBufferDescriptor new];
    commands.commandTypes = MTLIndirectCommandTypeDraw;
    commands.inheritPipelineState = YES;
    commands.inheritBuffers = NO;
    commands.maxVertexBufferBindCount = 1;
    commands.maxFragmentBufferBindCount = 0;
    id<MTLIndirectCommandBuffer> icb = [device newIndirectCommandBufferWithDescriptor:commands
                                                                      maxCommandCount:2
                                                                              options:0];
    float corner[12] = {-1, -1, 0, 1, -0.75, -1, 0, 1, -1, -0.75, 0, 1};
    id<MTLBuffer> cornerBuffer = [device newBufferWithBytes:corner
                                                     length:sizeof corner
                                                    options:MTLResourceStorageModeShared];
    float centre[4] = {0, 0, 0, 1};
    id<MTLBuffer> centreBuffer = [device newBufferWithBytes:centre
                                                     length:sizeof centre
                                                    options:MTLResourceStorageModeShared];
    id<MTLIndirectRenderCommand> cpu = [icb indirectRenderCommandAtIndex:1];
    [cpu setVertexBuffer:cornerBuffer offset:0 atIndex:0];
    [cpu drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3 instanceCount:1
           baseInstance:0];
    id<MTLBuffer> table = tableOf(device, icb.gpuResourceID, state.gpuResourceID);
    MTLTextureDescriptor *targetDescriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                           width:16
                                                          height:16
                                                       mipmapped:NO];
    targetDescriptor.usage = MTLTextureUsageRenderTarget;
    targetDescriptor.storageMode = MTLStorageModeShared;
    id<MTLTexture> target = [device newTextureWithDescriptor:targetDescriptor];
    id<MTLCommandBuffer> buffer = [queue commandBuffer];
    encodeWriter(buffer, shape, writer, table, centreBuffer, icb);
    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = target;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(1, 0, 0, 1);
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> encoder = [buffer renderCommandEncoderWithDescriptor:pass];
    [encoder setRenderPipelineState:state];
    [encoder useResource:cornerBuffer usage:MTLResourceUsageRead stages:MTLRenderStageVertex];
    [encoder useResource:centreBuffer usage:MTLResourceUsageRead stages:MTLRenderStageVertex];
    [encoder executeCommandsInBuffer:icb withRange:NSMakeRange(0, 2)];
    [encoder endEncoding];
    [buffer commit];
    [buffer waitUntilCompleted];
    uint8_t texel[4] = {0}, corner_texel[4] = {0};
    [target getBytes:texel bytesPerRow:64 fromRegion:MTLRegionMake2D(8, 8, 1, 1) mipmapLevel:0];
    [target getBytes:corner_texel
         bytesPerRow:64
          fromRegion:MTLRegionMake2D(0, 15, 1, 1)
         mipmapLevel:0];
    printf("points: status %ld error %s; centre texel %u %u %u %u; corner texel %u %u %u %u\n",
           (long)buffer.status,
           buffer.error ? buffer.error.localizedDescription.UTF8String : "none", texel[0],
           texel[1], texel[2], texel[3], corner_texel[0], corner_texel[1], corner_texel[2],
           corner_texel[3]);
}

static void compute(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLLibrary> library,
                    Shape shape) {
    NSError *error = nil;
    MTLComputePipelineDescriptor *descriptor = [MTLComputePipelineDescriptor new];
    descriptor.computeFunction = [library newFunctionWithName:@"irw_mark"];
    descriptor.supportIndirectCommandBuffers = YES;
    id<MTLComputePipelineState> state =
        [device newComputePipelineStateWithDescriptor:descriptor
                                              options:MTLPipelineOptionNone
                                           reflection:nil
                                                error:&error];
    if (!state) fail(@"mark state", error);
    id<MTLComputePipelineState> writer = [device
        newComputePipelineStateWithFunction:[library newFunctionWithName:@"irw_encode_dispatch"]
                                      error:&error];
    if (!writer) fail(@"dispatch writer", error);
    MTLIndirectCommandBufferDescriptor *commands = [MTLIndirectCommandBufferDescriptor new];
    commands.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
    commands.inheritPipelineState = NO;
    commands.inheritBuffers = NO;
    commands.maxKernelBufferBindCount = 1;
    id<MTLIndirectCommandBuffer> icb = [device newIndirectCommandBufferWithDescriptor:commands
                                                                      maxCommandCount:1
                                                                              options:0];
    uint32_t zero[4] = {0, 0, 0, 0};
    id<MTLBuffer> out = [device newBufferWithBytes:zero
                                            length:sizeof zero
                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> table = tableOf(device, icb.gpuResourceID, state.gpuResourceID);
    id<MTLCommandBuffer> buffer = [queue commandBuffer];
    encodeWriter(buffer, shape, writer, table, out, icb);
    id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];
    [encoder useResource:out usage:MTLResourceUsageWrite];
    [encoder executeCommandsInBuffer:icb withRange:NSMakeRange(0, 1)];
    [encoder endEncoding];
    [buffer commit];
    [buffer waitUntilCompleted];
    const uint32_t *words = out.contents;
    printf("compute: status %ld error %s; words %#x %#x %#x %#x\n", (long)buffer.status,
           buffer.error ? buffer.error.localizedDescription.UTF8String : "none", words[0],
           words[1], words[2], words[3]);
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc != 3) {
            printf("usage: %s <same|after|before|none> <icb_routed_writes.metal>\n", argv[0]);
            return 2;
        }
        NSString *name = @(argv[1]);
        Shape shape = [name isEqual:@"same"]     ? SAME
                      : [name isEqual:@"after"]  ? AFTER
                      : [name isEqual:@"before"] ? BEFORE
                                                 : NONE;
        NSError *error = nil;
        NSString *source = [NSString stringWithContentsOfFile:@(argv[2])
                                                     encoding:NSUTF8StringEncoding
                                                        error:&error];
        if (!source) fail(@"source", error);
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        if (!library) fail(@"library", error);
        id<MTLCommandQueue> queue = [device newCommandQueue];
        printf("%s on %s, %s\n", argv[1], device.name.UTF8String,
               [NSProcessInfo processInfo].operatingSystemVersionString.UTF8String);
        render(device, queue, library, shape);
        points(device, queue, library, shape);
        compute(device, queue, library, shape);
        printf("DONE\n");
    }
    return 0;
}
