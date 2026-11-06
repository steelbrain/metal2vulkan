// How a mesh render pipeline of `maxVertexAmplificationCount` 2 runs an amplified draw: how many
// times its object and mesh functions run, which `[[amplification_id]]` and
// `[[amplification_count]]` its mesh function reads, and what each amplification renders.
//
// Build (no cargo target needed):
//   clang -fobjc-arc -O1 -fmodules -o mesh_amplification mesh_amplification.m \
//     -framework Foundation -framework Metal
//
// `mesh_amplification CASE` draws one `drawMeshThreadgroups` of 2 object threadgroups of 4 threads,
// each launching 1 mesh threadgroup of 4 threads, into a 4x4 two-slice `RGBA32Uint` array target,
// under `setVertexAmplificationCount:2 viewMappings:` (slice offsets 0 and 1), or, for `plain`,
// without amplification on a pipeline of maximum 1. Every object thread adds 1 to word 0 of a
// shared buffer, every mesh thread 1 to word 1 and its `amplification_id` to word 2; each mesh
// thread 0 adds 1 to word 4 + amplification_id and stores `amplification_count` at word 3. Each
// mesh threadgroup draws one triangle over every pixel, its fragment writing
// (amplification id + 1, object threadgroup + 1, primitive data, 7). The probe prints the words,
// and each slice's texel (0, 0) and how many texels of it were drawn. `--cases` lists the cases.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>

static NSString *const SOURCE =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "struct P { uint group; };\n"
     "struct V { float4 position [[position]]; };\n"
     "struct Prim { uint4 data [[flat]]; };\n"
     "using M = metal::mesh<V, Prim, 3, 1, topology::triangle>;\n"
     "[[object]] void obj(object_data P& p [[payload]], mesh_grid_properties g,\n"
     "                    device atomic_uint *w [[buffer(0)]],\n"
     "                    uint t [[thread_index_in_threadgroup]],\n"
     "                    uint group [[threadgroup_position_in_grid]]) {\n"
     "  atomic_fetch_add_explicit(&w[0], 1u, memory_order_relaxed);\n"
     "  if (t == 0) { p.group = group; g.set_threadgroups_per_grid(uint3(1, 1, 1)); }\n"
     "}\n"
     "[[mesh]] void msh(M out, const object_data P& p [[payload]],\n"
     "                  device atomic_uint *w [[buffer(0)]],\n"
     "                  uint t [[thread_index_in_threadgroup]],\n"
     "                  uint amp [[amplification_id]],\n"
     "                  uint count [[amplification_count]]) {\n"
     "  atomic_fetch_add_explicit(&w[1], 1u, memory_order_relaxed);\n"
     "  atomic_fetch_add_explicit(&w[2], amp, memory_order_relaxed);\n"
     "  if (t == 0) {\n"
     "    atomic_fetch_add_explicit(&w[4 + min(amp, 3u)], 1u, memory_order_relaxed);\n"
     "    atomic_store_explicit(&w[3], count, memory_order_relaxed);\n"
     "    V v;\n"
     "    v.position = float4(-1.0, -1.0, 0.5, 1.0); out.set_vertex(0, v);\n"
     "    v.position = float4(3.0, -1.0, 0.5, 1.0); out.set_vertex(1, v);\n"
     "    v.position = float4(-1.0, 3.0, 0.5, 1.0); out.set_vertex(2, v);\n"
     "    out.set_index(0, 0); out.set_index(1, 1); out.set_index(2, 2);\n"
     "    Prim q; q.data = uint4(amp + 1, p.group + 1, count, 7);\n"
     "    out.set_primitive(0, q);\n"
     "    out.set_primitive_count(1);\n"
     "  }\n"
     "}\n"
     "struct In { V v; Prim p; };\n"
     "fragment uint4 frag(In in [[stage_in]]) { return in.p.data; }\n";

int main(int argc, const char **argv) {
    @autoreleasepool {
        const char *cases[] = {"plain", "amplified-2"};
        if (argc == 2 && strcmp(argv[1], "--cases") == 0) {
            for (int i = 0; i < 2; i++) {
                printf("%s\n", cases[i]);
            }
            return 0;
        }
        if (argc != 2 || (strcmp(argv[1], cases[0]) != 0 && strcmp(argv[1], cases[1]) != 0)) {
            fprintf(stderr, "usage: mesh_amplification CASE | --cases\n");
            return 2;
        }
        BOOL amplified = strcmp(argv[1], "amplified-2") == 0;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        printf("%s on %s supportsVertexAmplificationCount:2 %d\n", argv[1],
               device.name.UTF8String, [device supportsVertexAmplificationCount:2]);
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:SOURCE options:nil error:&error];
        if (library == nil) {
            printf("library %s\n", error.localizedDescription.UTF8String);
            return 1;
        }
        MTLMeshRenderPipelineDescriptor *d = [MTLMeshRenderPipelineDescriptor new];
        d.objectFunction = [library newFunctionWithName:@"obj"];
        d.meshFunction = [library newFunctionWithName:@"msh"];
        d.fragmentFunction = [library newFunctionWithName:@"frag"];
        d.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Uint;
        d.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
        d.maxVertexAmplificationCount = amplified ? 2 : 1;
        id<MTLRenderPipelineState> state =
            [device newRenderPipelineStateWithMeshDescriptor:d
                                                     options:MTLPipelineOptionNone
                                                  reflection:nil
                                                       error:&error];
        if (state == nil) {
            printf("refused %s %ld %s\n", error.domain.UTF8String, (long)error.code,
                   error.localizedDescription.UTF8String);
            return 0;
        }
        MTLTextureDescriptor *td =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Uint
                                                               width:4
                                                              height:4
                                                           mipmapped:NO];
        td.textureType = MTLTextureType2DArray;
        td.arrayLength = 2;
        td.usage = MTLTextureUsageRenderTarget;
        td.storageMode = MTLStorageModeShared;
        id<MTLTexture> target = [device newTextureWithDescriptor:td];
        id<MTLBuffer> words = [device newBufferWithLength:64 options:MTLResourceStorageModeShared];
        memset(words.contents, 0, 64);
        MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
        pass.colorAttachments[0].texture = target;
        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        pass.renderTargetArrayLength = 2;
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:pass];
        [enc setRenderPipelineState:state];
        [enc setObjectBuffer:words offset:0 atIndex:0];
        [enc setMeshBuffer:words offset:0 atIndex:0];
        if (amplified) {
            MTLVertexAmplificationViewMapping maps[2] = {{0, 0}, {0, 1}};
            [enc setVertexAmplificationCount:2 viewMappings:maps];
        }
        [enc drawMeshThreadgroups:MTLSizeMake(2, 1, 1)
            threadsPerObjectThreadgroup:MTLSizeMake(4, 1, 1)
              threadsPerMeshThreadgroup:MTLSizeMake(4, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        printf("status %ld error %s\n", (long)cb.status,
               cb.error ? cb.error.localizedDescription.UTF8String : "none");
        const uint32_t *w = words.contents;
        printf("object-threads %u mesh-threads %u amplification-ids %u count %u per-amp %u %u %u %u\n",
               w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]);
        for (NSUInteger slice = 0; slice < 2; slice++) {
            uint32_t texels[4 * 4 * 4];
            [target getBytes:texels
                 bytesPerRow:4 * 16
               bytesPerImage:4 * 4 * 16
                  fromRegion:MTLRegionMake2D(0, 0, 4, 4)
                 mipmapLevel:0
                       slice:slice];
            int drawn = 0;
            for (int i = 0; i < 16; i++) {
                drawn += texels[4 * i] != 0 || texels[4 * i + 3] != 0;
            }
            printf("slice %lu texel0 %u %u %u %u drawn %d\n", (unsigned long)slice, texels[0],
                   texels[1], texels[2], texels[3], drawn);
        }
    }
    return 0;
}
