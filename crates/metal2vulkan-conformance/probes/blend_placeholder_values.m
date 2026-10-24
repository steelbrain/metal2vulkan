// What plain Metal does with MTL4's two blend specialization placeholders on a CLASSIC
// MTLRenderPipelineDescriptor: MTLBlendFactorUnspecialized (19) and
// MTLBlendOperationUnspecialized (5).
//
// ONE SHAPE PER PROCESS, chosen on the command line, because Metal's pipeline validation may end
// the process and a process that died says nothing about the shapes after the one that killed it.
//
//   readback <field>=<raw> ...   set each field, read every field back, make NO pipeline.
//   create   <field>=<raw> ...   set them, then newRenderPipelineStateWithDescriptor:error:.
//   draw     <field>=<raw> ...   that, then one draw, and the texel it stored.
//
// field: srgb drgb rgbop salpha dalpha alphaop blending
//
// WHY A PROBE AND NOT A CONFORMANCE CASE, for one half of it: the value one past each placeholder
// ends plain Metal's process, and a case cannot ask a question whose answer kills the process it
// runs in. The accepting half IS a case now -- `blend-sweep-placeholders` of `cases/blend_sweeps.rs`
// holds it against Metal from here on -- and this probe stays because the aborting half has nowhere
// else to live and because the accepting half was measured here first.
//
// WHAT IT MEASURED, Apple M3 Max, macOS 27.0.1 (26A434), one run a shape, every other blend field
// One/One/Add so a slot falling back to its default reads differently from one keeping what was set:
//
//   readback      the setter KEEPS both placeholders: the descriptor reads 19 and 5 back.
//   create        newRenderPipelineStateWithDescriptor:error: answers a state with NO error for
//                 the placeholder in any of the four factor slots and either operation slot.
//   draw          the draw stores what each slot's own DEFAULT computes: a source slot blends the
//                 placeholder as One (the texel is the control's), a destination slot as Zero (the
//                 colour reads back as the source alone), an operation slot as Add (told from
//                 Subtract, whose own row differs, under a factor pair of two different numbers).
//                 All six fields at once store the source untouched.
//   one past it   `./blend_placeholders create srgb=20` dies on `blendFactorSource:4823: failed
//                 assertion 'Invalid blend factor'`, and `rgbop=6` on `validateWithDevice:5101:
//                 failed assertion 'Render Pipeline Descriptor Validation / func is not a valid
//                 MTLBlendOperation.'`, both with NO API validation layer enabled. So 19 and 5 are
//                 accepted specifically and not by a range check Metal skips -- which is the whole
//                 reason the accepting half is a fact about the placeholders rather than about
//                 out-of-range enums.
//
// Run it WITHOUT MTL_DEBUG_LAYER: plain Metal is the behavioural oracle, and the layer's
// assertions would end the process before the shape under test.
//
//   clang -fobjc-arc -framework Metal -framework Foundation -o blend_placeholders blend_placeholder_values.m

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static const NSUInteger kSide = 4;

// The source the fragment returns and the destination the host prefills, chosen so every factor's
// product is its own number: no two of Zero/One/src/dst/1-src/1-dst agree on any channel.
static const uint8_t kDest[4] = {0x20, 0x40, 0x60, 0x80};

static NSString *kSource =
  @"#include <metal_stdlib>\n"
   "using namespace metal;\n"
   "vertex float4 vmain(uint vid [[vertex_id]]) {\n"
   "  float2 p[3] = {float2(-1,-3), float2(-1,1), float2(3,1)};\n"
   "  return float4(p[vid], 0.5, 1);\n"
   "}\n"
   "fragment float4 fmain() { return float4(0.25, 0.5, 0.75, 0.375); }\n";

static void say(const char *key, NSString *value) {
  printf("%s=%s\n", key, value.UTF8String);
  fflush(stdout);
}

int main(int argc, const char **argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s readback|create|draw <field>=<raw> ...\n", argv[0]); return 2; }
  NSString *shape = @(argv[1]);

  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) { say("device", @"none"); return 1; }
    say("device", device.name);
    say("shape", shape);

    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:kSource options:nil error:&error];
    if (!library) { say("library-error", error.description); return 1; }
    id<MTLFunction> vertexFn = [library newFunctionWithName:@"vmain"];
    id<MTLFunction> fragmentFn = [library newFunctionWithName:@"fmain"];

    MTLRenderPipelineDescriptor *pipe = [MTLRenderPipelineDescriptor new];
    pipe.vertexFunction = vertexFn;
    pipe.fragmentFunction = fragmentFn;
    MTLRenderPipelineColorAttachmentDescriptor *color = pipe.colorAttachments[0];
    color.pixelFormat = MTLPixelFormatRGBA8Unorm;
    color.blendingEnabled = YES;
    // Known-good starting state in every slot, each stated so that "the field was ignored" and
    // "the field took the value I set" are told apart: One/One/Add everywhere, which is NOT the
    // destination slots' own default (Zero), so a destination slot that falls back to its default
    // reads differently from one that kept what this probe set.
    color.sourceRGBBlendFactor = MTLBlendFactorOne;
    color.destinationRGBBlendFactor = MTLBlendFactorOne;
    color.rgbBlendOperation = MTLBlendOperationAdd;
    color.sourceAlphaBlendFactor = MTLBlendFactorOne;
    color.destinationAlphaBlendFactor = MTLBlendFactorOne;
    color.alphaBlendOperation = MTLBlendOperationAdd;

    for (int i = 2; i < argc; i++) {
      NSString *pair = @(argv[i]);
      NSArray<NSString *> *parts = [pair componentsSeparatedByString:@"="];
      if (parts.count != 2) { fprintf(stderr, "bad pair %s\n", argv[i]); return 2; }
      NSString *field = parts[0];
      NSUInteger raw = (NSUInteger)strtoul(parts[1].UTF8String, NULL, 0);
      say("about-to-set", pair);
      if ([field isEqual:@"srgb"]) color.sourceRGBBlendFactor = (MTLBlendFactor)raw;
      else if ([field isEqual:@"drgb"]) color.destinationRGBBlendFactor = (MTLBlendFactor)raw;
      else if ([field isEqual:@"rgbop"]) color.rgbBlendOperation = (MTLBlendOperation)raw;
      else if ([field isEqual:@"salpha"]) color.sourceAlphaBlendFactor = (MTLBlendFactor)raw;
      else if ([field isEqual:@"dalpha"]) color.destinationAlphaBlendFactor = (MTLBlendFactor)raw;
      else if ([field isEqual:@"alphaop"]) color.alphaBlendOperation = (MTLBlendOperation)raw;
      else if ([field isEqual:@"blending"]) color.blendingEnabled = raw != 0;
      else { fprintf(stderr, "unknown field %s\n", field.UTF8String); return 2; }
      say("set", @"returned");
    }

    say("read-srgb", [NSString stringWithFormat:@"%lu", (unsigned long)color.sourceRGBBlendFactor]);
    say("read-drgb", [NSString stringWithFormat:@"%lu", (unsigned long)color.destinationRGBBlendFactor]);
    say("read-rgbop", [NSString stringWithFormat:@"%lu", (unsigned long)color.rgbBlendOperation]);
    say("read-salpha", [NSString stringWithFormat:@"%lu", (unsigned long)color.sourceAlphaBlendFactor]);
    say("read-dalpha", [NSString stringWithFormat:@"%lu", (unsigned long)color.destinationAlphaBlendFactor]);
    say("read-alphaop", [NSString stringWithFormat:@"%lu", (unsigned long)color.alphaBlendOperation]);
    say("read-blending", color.blendingEnabled ? @"YES" : @"NO");
    if ([shape isEqual:@"readback"]) return 0;

    say("about-to-create", @"newRenderPipelineStateWithDescriptor");
    error = nil;
    id<MTLRenderPipelineState> state = [device newRenderPipelineStateWithDescriptor:pipe error:&error];
    say("create-returned", state ? @"state" : @"nil");
    say("create-error", error ? error.description : @"none");
    if (!state || [shape isEqual:@"create"]) return state ? 0 : 1;

    MTLTextureDescriptor *td =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                         width:kSide height:kSide mipmapped:NO];
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModeShared;
    id<MTLTexture> target = [device newTextureWithDescriptor:td];
    uint8_t prefill[4 * 4 * 4];
    for (NSUInteger i = 0; i < kSide * kSide; i++) memcpy(prefill + 4 * i, kDest, 4);
    [target replaceRegion:MTLRegionMake2D(0, 0, kSide, kSide) mipmapLevel:0
                withBytes:prefill bytesPerRow:4 * kSide];

    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor new];
    pass.colorAttachments[0].texture = target;
    pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;

    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> buffer = [queue commandBuffer];
    id<MTLRenderCommandEncoder> encoder = [buffer renderCommandEncoderWithDescriptor:pass];
    [encoder setRenderPipelineState:state];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [encoder endEncoding];
    say("about-to-commit", @"commit");
    [buffer commit];
    [buffer waitUntilCompleted];
    say("buffer-status", [NSString stringWithFormat:@"%ld", (long)buffer.status]);
    say("buffer-error", buffer.error ? buffer.error.description : @"none");

    uint8_t out[4 * 4 * 4];
    memset(out, 0x5C, sizeof(out));
    [target getBytes:out bytesPerRow:4 * kSide
          fromRegion:MTLRegionMake2D(0, 0, kSide, kSide) mipmapLevel:0];
    say("texel", [NSString stringWithFormat:@"%02x %02x %02x %02x", out[0], out[1], out[2], out[3]]);
    say("dest-was", [NSString stringWithFormat:@"%02x %02x %02x %02x", kDest[0], kDest[1], kDest[2], kDest[3]]);
  }
  return 0;
}
