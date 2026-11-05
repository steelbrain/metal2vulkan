// Plain Metal's answer, with the API validation layer OFF, for an execution of an indirect command
// buffer whose CPU-written command 0 names a render pipeline state of another color format than
// the executing pass's target, in a command buffer X that waits for a shared event and is
// committed first, on one queue; and a command buffer Y on another queue, committed after X, whose
// kernel writes command 0 again (a valid state, taken from its argument buffer, a vertex buffer and
// a draw of one triangle) and which then signals the event:
//
//   written      Y's kernel writes command 0, then Y signals;
//   unwritten    Y signals alone: X runs the CPU's command.
//
// X's render encoder makes the indirect command buffer resident for reading and writing, as a
// guest's does; Y's compute encoder makes it resident for writing. It prints X's and Y's status and
// error and X's target's centre texel (a 16x16 RGBA8Unorm target cleared to red: green where the
// kernel's command ran). The functions are the owned fixture
// `crates/metal2vulkan-validation/fixtures/public/engine/icb_routed_writes.metal`, whose path is
// the second argument; the runtime's lane `runtime_lane::write_routes` runs the same shape.
//
// Build (no GPU needed to build):
//
//   clang -fobjc-arc -fmodules -framework Metal -framework Foundation \
//       -o icb_cross_queue_writes icb_cross_queue_writes.m
//
// Run ONE SHAPE PER PROCESS, WITHOUT MTL_DEBUG_LAYER:
//
//   ./icb_cross_queue_writes <written|unwritten> <path of icb_routed_writes.metal>
//
// A shape whose process Metal ends prints no `DONE`, and its exit status is the reading.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static void fail(NSString *what, NSError *error) {
    printf("FAIL %s: %s\n", what.UTF8String, error.localizedDescription.UTF8String);
    exit(2);
}

// A render state of the fixture's triangle drawing into `format`.
static id<MTLRenderPipelineState> stateOf(id<MTLDevice> device, id<MTLLibrary> library,
                                          MTLPixelFormat format) {
    NSError *error = nil;
    MTLRenderPipelineDescriptor *descriptor = [MTLRenderPipelineDescriptor new];
    descriptor.vertexFunction = [library newFunctionWithName:@"irw_vertex"];
    descriptor.fragmentFunction = [library newFunctionWithName:@"irw_fragment"];
    descriptor.colorAttachments[0].pixelFormat = format;
    descriptor.supportIndirectCommandBuffers = YES;
    id<MTLRenderPipelineState> state = [device newRenderPipelineStateWithDescriptor:descriptor
                                                                              error:&error];
    if (!state) fail(@"render state", error);
    return state;
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc != 3) {
            printf("usage: %s <written|unwritten> <icb_routed_writes.metal>\n", argv[0]);
            return 2;
        }
        BOOL written = [@(argv[1]) isEqual:@"written"];
        NSError *error = nil;
        NSString *source = [NSString stringWithContentsOfFile:@(argv[2])
                                                     encoding:NSUTF8StringEncoding
                                                        error:&error];
        if (!source) fail(@"source", error);
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        if (!library) fail(@"library", error);
        printf("%s on %s, %s\n", argv[1], device.name.UTF8String,
               [NSProcessInfo processInfo].operatingSystemVersionString.UTF8String);
        id<MTLCommandQueue> executing = [device newCommandQueue];
        id<MTLCommandQueue> writing = [device newCommandQueue];
        id<MTLSharedEvent> event = [device newSharedEvent];
        id<MTLRenderPipelineState> good = stateOf(device, library, MTLPixelFormatRGBA8Unorm);
        id<MTLRenderPipelineState> other = stateOf(device, library, MTLPixelFormatRGBA16Float);
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
        id<MTLIndirectCommandBuffer> icb =
            [device newIndirectCommandBufferWithDescriptor:commands maxCommandCount:1 options:0];
        float corners[12] = {-1, -1, 0, 1, 3, -1, 0, 1, -1, 3, 0, 1};
        id<MTLBuffer> cornerBuffer = [device newBufferWithBytes:corners
                                                         length:sizeof corners
                                                        options:MTLResourceStorageModeShared];
        id<MTLIndirectRenderCommand> cpu = [icb indirectRenderCommandAtIndex:0];
        [cpu setRenderPipelineState:other];
        [cpu setVertexBuffer:cornerBuffer offset:0 atIndex:0];
        [cpu drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3 instanceCount:1
               baseInstance:0];
        MTLResourceID ids[2] = {icb.gpuResourceID, good.gpuResourceID};
        id<MTLBuffer> table = [device newBufferWithBytes:ids
                                                  length:sizeof ids
                                                 options:MTLResourceStorageModeShared];
        MTLTextureDescriptor *targetDescriptor =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                               width:16
                                                              height:16
                                                           mipmapped:NO];
        targetDescriptor.usage = MTLTextureUsageRenderTarget;
        targetDescriptor.storageMode = MTLStorageModeShared;
        id<MTLTexture> target = [device newTextureWithDescriptor:targetDescriptor];

        // X: waits for the event, then executes command 0; committed first.
        id<MTLCommandBuffer> x = [executing commandBuffer];
        [x encodeWaitForEvent:event value:1];
        MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
        pass.colorAttachments[0].texture = target;
        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass.colorAttachments[0].clearColor = MTLClearColorMake(1, 0, 0, 1);
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        id<MTLRenderCommandEncoder> render = [x renderCommandEncoderWithDescriptor:pass];
        [render useResource:icb
                      usage:MTLResourceUsageRead | MTLResourceUsageWrite
                     stages:MTLRenderStageVertex | MTLRenderStageFragment];
        [render useResource:cornerBuffer usage:MTLResourceUsageRead stages:MTLRenderStageVertex];
        [render executeCommandsInBuffer:icb withRange:NSMakeRange(0, 1)];
        [render endEncoding];
        [x commit];

        // Y: on the other queue, writes command 0 again (or not), then signals.
        id<MTLCommandBuffer> y = [writing commandBuffer];
        if (written) {
            id<MTLComputeCommandEncoder> compute = [y computeCommandEncoder];
            [compute useResource:icb usage:MTLResourceUsageWrite];
            [compute setComputePipelineState:writer];
            [compute setBuffer:table offset:0 atIndex:0];
            [compute setBuffer:cornerBuffer offset:0 atIndex:1];
            [compute dispatchThreads:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [compute endEncoding];
        }
        [y encodeSignalEvent:event value:1];
        [y commit];
        [y waitUntilCompleted];
        [x waitUntilCompleted];
        uint8_t texel[4] = {0};
        [target getBytes:texel bytesPerRow:64 fromRegion:MTLRegionMake2D(8, 8, 1, 1) mipmapLevel:0];
        printf("X: status %ld error %s; centre texel %u %u %u %u\n", (long)x.status,
               x.error ? x.error.localizedDescription.UTF8String : "none", texel[0], texel[1],
               texel[2], texel[3]);
        printf("Y: status %ld error %s\n", (long)y.status,
               y.error ? y.error.localizedDescription.UTF8String : "none");
        printf("DONE\n");
    }
    return 0;
}
