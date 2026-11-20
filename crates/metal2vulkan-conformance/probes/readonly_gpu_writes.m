// What Metal does with a GPU write into memory whose backing is read-only.
//
// The owner's rule is that where Metal drops such a write we drop it too, on every driver, with no
// refusal. Four points are already measured: a blit
// fill completes with no error and is visible to a later command of the SAME command buffer but
// not to a later command buffer; a kernel's stores are dropped; the pages' protection is
// unchanged; and a CPU write through a read-only mapping ends the process. This probe re-measures
// those and then measures the shapes nobody has: a render attachment (colour, depth, stencil), a
// resolve, a visibility result, a draw-stage store, and a store through an address the shader
// loaded out of an argument buffer.
//
// ONE SHAPE PER PROCESS, chosen on the command line, because the owner's own report says Metal
// either drops such a write OR faults, and a fault inside a command buffer holding five shapes
// says nothing about which one caused it. A fault is a legitimate measurement here; it is why
// this is a probe and not a conformance case.
//
// EVERY SHAPE RUNS AGAINST A WRITABLE TWIN OF ITS OWN CREATION PATH, and that control is what
// makes the reading mean anything: "the write was dropped" and "my shader wrote nothing" read
// identically against a read-only buffer alone, and a control made a different way would be
// comparing two creation paths as well as two backing states.
//
// There are TWO read-only states, not one, and the point of the backing list is to compare them:
//
//   plain            newBufferWithBytesNoCopy over the host pages, MTLResourceStorageModeShared.
//                    The writable control for the bit-24 column.
//   bit24            the same with Metal's private MTLResourceOptions bit 24.
//   viewed           a read-only alias of the same pages (mach_vm_remap sharing them, then
//                    mprotect PROT_READ), imported by newBufferWithBytesNoCopy with NO bit.
//                    Whether it agrees with bit24 says whether such an alias emulates the bit
//                    faithfully.
//   descriptor       a no-copy buffer made through the private MTLBufferDescriptor
//                    (setContents:, setLength:, setNoCopy:YES, newBufferWithDescriptor:), which
//                    is how the consuming project makes its own. The writable control for the SPI
//                    column.
//   replaced         that descriptor buffer, detachBacking'd and then
//                    replaceBackingWithRanges:readOnly:YES over the probe's pages. THIS IS THE
//                    CONSUMING PROJECT'S PRODUCTION PATH, and bit 24 is only a validated stand-in
//                    for it where the two agree.
//   nocopy-replaced  the same SPI pair on a newBufferWithBytesNoCopy: buffer, which answers the
//                    open question of whether the creation path matters to the SPI at all.
//   detached         the descriptor buffer detachBacking'd and NOT replaced: the reference leg.
//                    The detached state is already well characterised elsewhere, and without this
//                    leg a dropped write under `replaced` could be the detach's doing rather than
//                    the replace's.
//
// WHAT THE TWO READ-ONLY STATES MEASURED, eight runs a shape (the M3 Max, macOS 27.0.1):
// bit 24 and the SPI pair AGREE on all ten shapes measurable under both -- byte for byte, error
// for error, and flap for flap. So bit 24 is a validated stand-in for the consuming project's
// production path rather than an assumed one. `nocopy-replaced` agreed too, so the creation path
// is irrelevant to the SPI. The detach-only leg did NOT agree: it errors BOTH command buffers on
// every shape, blit fill included, which is how we know the drop under `replaced` is the replace's
// doing and not the detach's.
//
// Depth and stencil are measurable in the bit-24 heap form ONLY, and the reason is measured rather
// than assumed: `detachBacking` responds on MTLHeap and answers NO for a heap of
// `newHeapWithDescriptor:`, Private and Shared alike, because there is no Metal route to a heap
// over the caller's bytes for it to detach -- `MTLHeapDescriptor` answers none of `setContents:`,
// `setNoCopy:` or `setDeallocator:`, and `MTLDevice` does not respond to
// `newHeapWithBytesNoCopy:length:options:deallocator:`.
//
// Two things about the SPI that cost the first shape of this probe its answer, both now handled:
// the selector is TWO-part, `replaceBackingWithRanges:readOnly:`, taking the ranges as one
// `MTLResourceAddressRangeArray` object -- a three-part `...:count:readOnly:` does not exist, and
// its `respondsToSelector:` NO was misread as the OS not offering the SPI; and the replace answers
// NO on an object still ATTACHED, so detachBacking goes first and its own answer is reported
// separately, because otherwise a NO says nothing about which of the two calls refused. The
// device's `supportsResourceDetachBacking` is reported on every run before any of this, so a
// negative result is never confused with an unsupported device.
//
// Each run prints one flat key=value report on stdout: the backing's protections, the buffer's
// own facts, each command buffer's status and error, and WHICH BYTES MOVED in three read-backs --
// a copy taken in the SAME command buffer as the write, a copy taken in a LATER command buffer,
// and the CPU's own view of the pages afterwards. The whole 64 KiB region is compared every time,
// not only the bytes the write aimed at, because for read-only memory the bytes that did not move
// are the result.
//
// Nothing here writes outside the region the probe allocated, and no shape loops. The risk it
// carries is Metal's own: a shape Metal faults on will fault, which is the thing being measured.
//
// clang -fobjc-arc -framework Metal -framework Foundation -framework QuartzCore \
//     -o readonly_gpu_writes readonly_gpu_writes.m

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <mach/mach.h>
#import <mach/mach_vm.h>
#import <objc/runtime.h>
#import <sys/mman.h>

// Metal's private `MTLResourceOptions` bit 24: the resource's memory takes the GPU's writes
// nowhere (`ResourceOptions::PRIVATE_BIT`).
static const NSUInteger kPrivateBit = 1ull << 24;

// The region the probe allocates and hands Metal: 16 pages of 4 KiB.
static const size_t kRegion = 64 * 1024;

// The byte every one of those bytes starts as, so a byte the GPU wrote is a byte that is no
// longer it, and a byte nothing wrote is visibly this one.
static const uint8_t kPad = 0xAB;

// The byte each read-back buffer starts as, distinct from kPad, so a copy that wrote nothing at
// all is visible as itself rather than as "the pages were untouched".
static const uint8_t kUnread = 0x5C;

// Where in the region each shape aims its write, and how far it reaches. One page in, so a write
// that ran off either end shows in bytes the shape never named.
static const NSUInteger kTargetOffset = 0x1000;
static const NSUInteger kTargetLength = 0x400;

// The 4 x 4 RGBA8Unorm target a render shape draws into, over the region at kTargetOffset: a row
// of 16 bytes at a pitch of 32, so half the target's span is padding no texel covers and a store
// that wrote whole pitches shows.
static const NSUInteger kTargetSide = 4;
static const NSUInteger kTargetPitch = 32;

// The report line, declared here because the SPI helpers below report as they go and the
// reporting section itself is further down (it needs none of this).
static void say(const char *key, NSString *value);

// Apple's backing SPI, in the form the consuming project's own code uses: TWO arguments, the
// ranges as one `MTLResourceAddressRangeArray` object. An earlier shape of this probe tried a
// three-part `replaceBackingWithRanges:count:readOnly:`, which does not exist, and read the
// resulting `respondsToSelector:` NO as the OS not offering the SPI at all. It does offer it.
//
// Declared as protocols rather than imported, because no public header has them; every call is
// behind `respondsToSelector:`, and the device's own `supportsResourceDetachBacking` is checked
// first so a negative result is never confused with an unsupported device.
@protocol M2VDetachableBacking <NSObject>
- (BOOL)replaceBackingWithRanges:(id)ranges readOnly:(BOOL)readOnly;
- (BOOL)detachBacking;
@end

// The private device calls: whether the SPI is offered at all, and the descriptor creation path
// the consuming project makes its no-copy buffers through.
@protocol M2VBackingDevice <MTLDevice>
- (BOOL)supportsResourceDetachBacking;
- (id<MTLBuffer>)newBufferWithDescriptor:(id)descriptor;
@end

// The private `MTLBufferDescriptor`, which is the OTHER half of why the first attempt found
// nothing: the consuming project does not make these buffers with
// `newBufferWithBytesNoCopy:length:options:deallocator:` at all. Reached through
// `NSClassFromString`, so a release without the class is a reported absence rather than a link
// error.
@protocol M2VBufferDescriptor <NSObject>
- (void)setContents:(void *)contents;
- (void)setLength:(NSUInteger)length;
- (void)setNoCopy:(BOOL)noCopy;
- (void)setResourceOptions:(NSUInteger)options;
@end

// One range of a backing replacement, as the SPI's own 16-byte shape: a host virtual address and a
// byte length. `MTLResourceAddressRangeArray` copies them, so this need not outlive the call.
typedef struct {
  uint64_t address;
  uint64_t length;
} M2VAddressRange;

@protocol M2VAddressRangeArray <NSObject>
- (instancetype)initWithRanges:(const void *)ranges count:(NSUInteger)count;
@end

// An `MTLResourceAddressRangeArray` over `count` ranges, or nil where the class is not there.
static id rangeArray(const M2VAddressRange *ranges, NSUInteger count) {
  Class class = NSClassFromString(@"MTLResourceAddressRangeArray");
  if (class == Nil) {
    say("spi.MTLResourceAddressRangeArray", @"absent");
    return nil;
  }
  id array = [class alloc];
  if (![array respondsToSelector:@selector(initWithRanges:count:)]) {
    say("spi.initWithRanges", @"absent");
    return nil;
  }
  return [(id<M2VAddressRangeArray>)array initWithRanges:ranges count:count];
}

// A no-copy buffer over `pages` made through the private `MTLBufferDescriptor`, which is the path
// the consuming project uses: nil where the class or any of its setters is not there.
static id<MTLBuffer> descriptorBuffer(id<MTLDevice> device, void *pages, NSUInteger length,
                                      NSUInteger options) {
  Class class = NSClassFromString(@"MTLBufferDescriptor");
  if (class == Nil) {
    say("spi.MTLBufferDescriptor", @"absent");
    return nil;
  }
  id descriptor = [class new];
  for (NSString *name in @[ @"setContents:", @"setLength:", @"setNoCopy:" ]) {
    if (![descriptor respondsToSelector:NSSelectorFromString(name)]) {
      say("spi.MTLBufferDescriptor.setter",
          ([NSString stringWithFormat:@"%@ absent", name]));
      return nil;
    }
  }
  id<M2VBufferDescriptor> typed = (id<M2VBufferDescriptor>)descriptor;
  [typed setContents:pages];
  [typed setLength:length];
  [typed setNoCopy:YES];
  if ([descriptor respondsToSelector:@selector(setResourceOptions:)]) {
    [typed setResourceOptions:options];
  }
  id<M2VBackingDevice> backing = (id<M2VBackingDevice>)device;
  if (![device respondsToSelector:@selector(newBufferWithDescriptor:)]) {
    say("spi.newBufferWithDescriptor", @"absent");
    return nil;
  }
  return [backing newBufferWithDescriptor:descriptor];
}

// Say what `object` answers of the two backing selectors, under `prefix`.
static void sayBackingSelectors(const char *prefix, id object) {
  char key[160];
  snprintf(key, sizeof(key), "%s.detachBacking", prefix);
  say(key, [object respondsToSelector:@selector(detachBacking)] ? @"responds" : @"absent");
  snprintf(key, sizeof(key), "%s.replaceBackingWithRanges_readOnly", prefix);
  say(key, [object respondsToSelector:@selector(replaceBackingWithRanges:readOnly:)]
               ? @"responds"
               : @"absent");
}

// Detach `object`'s backing and then replace it, read-only, with the `length` bytes at `pages`.
//
// THE ORDER MATTERS and a NO is ambiguous without it: the replace answers NO on an object still
// attached, so the detach's own answer is reported separately. `detach_only` stops after the
// detach, which is the reference leg -- the detached state is already well characterised, and
// without it a dropped write could be the detach's doing rather than the replace's.
static BOOL detachAndReplace(const char *prefix, id object, void *pages, NSUInteger length,
                             BOOL detach_only) {
  char key[160];
  id<M2VDetachableBacking> typed = (id<M2VDetachableBacking>)object;
  if (![object respondsToSelector:@selector(detachBacking)]) {
    snprintf(key, sizeof(key), "%s.detach", prefix);
    say(key, @"selector-absent");
    return NO;
  }
  BOOL detached = [typed detachBacking];
  snprintf(key, sizeof(key), "%s.detach", prefix);
  say(key, detached ? @"yes" : @"no");
  if (!detached || detach_only) {
    return detached;
  }
  if (![object respondsToSelector:@selector(replaceBackingWithRanges:readOnly:)]) {
    snprintf(key, sizeof(key), "%s.replace", prefix);
    say(key, @"selector-absent");
    return NO;
  }
  M2VAddressRange range = {(uint64_t)(uintptr_t)pages, length};
  id array = rangeArray(&range, 1);
  if (array == nil) {
    return NO;
  }
  BOOL replaced = [typed replaceBackingWithRanges:array readOnly:YES];
  snprintf(key, sizeof(key), "%s.replace", prefix);
  say(key, replaced ? @"yes" : @"no");
  return replaced;
}

// The shaders every storing shape runs. Each store writes 0xC0DE0000 plus the index: a word whose
// every byte differs from kPad and kUnread, so a stored word is unmistakable in a read-back.
static NSString *const kShaders =
    @"#include <metal_stdlib>\n"
    @"using namespace metal;\n"
    // A store into a bound device buffer, one word a thread.
    @"kernel void ro_store(device uint *out [[buffer(0)]],\n"
    @"                     uint i [[thread_position_in_grid]]) {\n"
    @"  out[i] = 0xC0DE0000u + i;\n"
    @"}\n"
    // The same store through an address the shader loads out of an argument buffer, which is the
    // shape the owner is least confident can be matched exactly.
    @"struct RoArgs { device uint *target; };\n"
    @"kernel void ro_store_loaded(constant RoArgs &args [[buffer(0)]],\n"
    @"                            uint i [[thread_position_in_grid]]) {\n"
    @"  args.target[i] = 0xC0DE0000u + i;\n"
    @"}\n"
    // A triangle covering the whole 4 x 4 target.
    @"vertex float4 ro_vert(uint vid [[vertex_id]]) {\n"
    @"  const float2 p[3] = { float2(-1, -3), float2(-1, 1), float2(3, 1) };\n"
    @"  return float4(p[vid], 0.5, 1);\n"
    @"}\n"
    // The same triangle, storing a word per vertex into a bound buffer on the way.
    @"vertex float4 ro_vert_store(uint vid [[vertex_id]],\n"
    @"                            device uint *out [[buffer(1)]]) {\n"
    @"  out[vid] = 0xC0DE0000u + vid;\n"
    @"  const float2 p[3] = { float2(-1, -3), float2(-1, 1), float2(3, 1) };\n"
    @"  return float4(p[vid], 0.5, 1);\n"
    @"}\n"
    // An opaque colour, so a stored texel is distinguishable from the clear and from kPad.
    @"fragment float4 ro_frag() {\n"
    @"  return float4(0.25, 0.5, 0.75, 1);\n"
    @"}\n"
    // The same, storing a word per covered pixel into a bound buffer.
    @"fragment float4 ro_frag_store(float4 pos [[position]],\n"
    @"                              device uint *out [[buffer(0)]]) {\n"
    @"  uint i = uint(pos.y) * 4u + uint(pos.x);\n"
    @"  out[i] = 0xC0DE0000u + i;\n"
    @"  return float4(0.25, 0.5, 0.75, 1);\n"
    @"}\n"
    // The same, through an address loaded out of an argument buffer.
    @"fragment float4 ro_frag_loaded(float4 pos [[position]],\n"
    @"                               constant RoArgs &args [[buffer(0)]]) {\n"
    @"  uint i = uint(pos.y) * 4u + uint(pos.x);\n"
    @"  args.target[i] = 0xC0DE0000u + i;\n"
    @"  return float4(0.25, 0.5, 0.75, 1);\n"
    @"}\n";

// ---------------------------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------------------------

// The run a report line belongs to, so repeated runs of one shape are told apart. -1 is a line
// about the process rather than a run.
static int gRun = -1;

static void say(const char *key, NSString *value) {
  if (gRun >= 0) {
    printf("run%d.%s=%s\n", gRun, key, value.UTF8String);
  } else {
    printf("%s=%s\n", key, value.UTF8String);
  }
  fflush(stdout);
}

static void sayu(const char *key, unsigned long long value) {
  say(key, ([NSString stringWithFormat:@"%llu", value]));
}

static void sayx(const char *key, unsigned long long value) {
  say(key, ([NSString stringWithFormat:@"%#llx", value]));
}

// The runs of `bytes` that are not `expected`, as `[start,end) n-bytes first-word` triples, with
// no more than eight runs named so one wild write cannot bury the report.
//
// This is the whole measurement: a read-back whose every byte is still the pad means the write
// reached nothing, and a read-back whose changed runs lie outside the shape's own target range
// means the write reached memory the shape never named.
static NSString *changedRuns(const uint8_t *bytes, size_t length, uint8_t expected) {
  NSMutableString *out = [NSMutableString string];
  size_t runs = 0, changed = 0, at = 0;
  while (at < length) {
    if (bytes[at] == expected) {
      at++;
      continue;
    }
    size_t start = at;
    while (at < length && bytes[at] != expected) {
      at++;
    }
    changed += at - start;
    runs++;
    if (runs <= 8) {
      uint32_t word = 0;
      memcpy(&word, bytes + start, MIN(sizeof(word), at - start));
      [out appendFormat:@"%s[%#zx,%#zx) %zu %#010x", runs == 1 ? "" : " ", start, at,
                        at - start, word];
    }
  }
  if (runs == 0) {
    return @"none";
  }
  return [NSString stringWithFormat:@"%zu-runs %zu-bytes %@%s", runs, changed, out,
                                    runs > 8 ? " ..." : ""];
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
  return [NSString stringWithFormat:@"unknown-%ld", (long)status];
}

static void sayCommandBuffer(const char *prefix, id<MTLCommandBuffer> cb) {
  char key[128];
  snprintf(key, sizeof(key), "%s.status", prefix);
  say(key, statusWord(cb.status));
  snprintf(key, sizeof(key), "%s.error", prefix);
  say(key, cb.error == nil ? @"none"
                           : [NSString stringWithFormat:@"%@ / %@", cb.error.domain,
                                                        cb.error.localizedDescription]);
}

// ---------------------------------------------------------------------------------------------
// The backing
// ---------------------------------------------------------------------------------------------

// What a backing came out as: the buffer Metal made, the pages the CPU reads, and the pointer the
// buffer was made over (the pages themselves, or a read-only alias of them).
typedef struct {
  id<MTLBuffer> buffer;
  uint8_t *pages;
  void *imported;
  NSString *note;
} Backing;

// The current and maximum protection of the page at `address`, as the kernel reports them. A
// read-only backing is an alias whose MAXIMUM protection is read, which is the fact the owner's
// report names, so both are printed.
static void sayProtection(const char *prefix, void *address) {
  mach_vm_address_t base = (mach_vm_address_t)address;
  mach_vm_size_t size = 0;
  vm_region_basic_info_data_64_t info;
  mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
  mach_port_t object = MACH_PORT_NULL;
  kern_return_t got = mach_vm_region(mach_task_self(), &base, &size, VM_REGION_BASIC_INFO_64,
                                     (vm_region_info_t)&info, &count, &object);
  char key[128];
  if (got != KERN_SUCCESS) {
    snprintf(key, sizeof(key), "%s.protection", prefix);
    say(key, ([NSString stringWithFormat:@"unreadable-%d", got]));
    return;
  }
  snprintf(key, sizeof(key), "%s.cur_protection", prefix);
  sayu(key, (unsigned long long)info.protection);
  snprintf(key, sizeof(key), "%s.max_protection", prefix);
  sayu(key, (unsigned long long)info.max_protection);
  snprintf(key, sizeof(key), "%s.shared", prefix);
  sayu(key, (unsigned long long)info.shared);
}

// A read-only alias of `length` bytes of the pages at `pointer`, sharing them.
static void *readOnlyView(void *pointer, size_t length) {
  mach_vm_address_t view = 0;
  vm_prot_t current = 0, maximum = 0;
  kern_return_t remapped =
      mach_vm_remap(mach_task_self(), &view, length, 0, VM_FLAGS_ANYWHERE, mach_task_self(),
                    (mach_vm_address_t)pointer, false, &current, &maximum, VM_INHERIT_SHARE);
  if (remapped != KERN_SUCCESS) {
    say("view.remap", ([NSString stringWithFormat:@"failed-%d", remapped]));
    return NULL;
  }
  if (mprotect((void *)view, length, PROT_READ) != 0) {
    say("view.mprotect", @"failed");
    return NULL;
  }
  return (void *)view;
}

static Backing makeBacking(id<MTLDevice> device, NSString *which) {
  Backing out = {nil, NULL, NULL, @"none"};
  // The pages the probe owns and the CPU reads. mmap rather than malloc so the region is
  // page-aligned and whole pages, which is what every one of these calls needs.
  void *pages = mmap(NULL, kRegion, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
  if (pages == MAP_FAILED) {
    say("backing", @"mmap-failed");
    return out;
  }
  memset(pages, kPad, kRegion);
  out.pages = (uint8_t *)pages;
  out.imported = pages;

  // Which creation path, and whether the backing SPI is then used on the buffer.
  //
  // `descriptor` is the writable CONTROL for the SPI column and is not optional: the control has
  // to be made the same way as the measurement, or the column is comparing two creation paths as
  // well as two backing states.
  BOOL byDescriptor = [which hasPrefix:@"descriptor"] || [which isEqualToString:@"replaced"] ||
                      [which isEqualToString:@"detached"];
  BOOL replaces = [which isEqualToString:@"replaced"] ||
                  [which isEqualToString:@"nocopy-replaced"];
  BOOL detachesOnly = [which isEqualToString:@"detached"];

  NSUInteger options = MTLResourceStorageModeShared;
  if ([which isEqualToString:@"bit24"]) {
    options |= kPrivateBit;
  } else if ([which isEqualToString:@"viewed"]) {
    void *view = readOnlyView(pages, kRegion);
    if (view == NULL) {
      say("backing", @"view-failed");
      return out;
    }
    out.imported = view;
    out.note = @"imported through a read-only alias of the pages";
  }

  if (byDescriptor) {
    out.buffer = descriptorBuffer(device, out.imported, kRegion, options);
    out.note = @"made through the private MTLBufferDescriptor, setNoCopy:YES";
  } else {
    out.buffer = [device newBufferWithBytesNoCopy:out.imported
                                           length:kRegion
                                          options:options
                                      deallocator:nil];
  }
  if (out.buffer == nil) {
    say("backing", @"nil-buffer");
    return out;
  }
  // Which selectors THIS creation path's buffer answers. Reported for every backing, because
  // whether a `newBufferWithBytesNoCopy:` buffer answers them at all is an open question the
  // consuming project has not checked.
  sayBackingSelectors("buffer", out.buffer);

  if (replaces || detachesOnly) {
    // The replace answers NO on an object still attached, so the detach goes first and its own
    // answer is reported: without it a NO here says nothing about which call refused.
    if (!detachAndReplace("buffer", out.buffer, pages, kRegion, detachesOnly)) {
      out.buffer = nil;
      return out;
    }
    out.note = detachesOnly
                   ? @"backing detached and NOT replaced: the reference leg"
                   : @"backing detached, then replaced read-only over the pages through the SPI";
  }
  return out;
}

// ---------------------------------------------------------------------------------------------
// The shapes
// ---------------------------------------------------------------------------------------------

typedef struct {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLLibrary> library;
  Backing backing;
  // Where a copy taken in the same command buffer as the write lands, and where one taken in a
  // later command buffer lands. Both start as kUnread.
  id<MTLBuffer> sameCb;
  id<MTLBuffer> laterCb;
} Probe;

static id<MTLFunction> functionNamed(Probe *probe, NSString *name) {
  id<MTLFunction> function = [probe->library newFunctionWithName:name];
  if (function == nil) {
    say("shape", ([NSString stringWithFormat:@"no-function-%@", name]));
  }
  return function;
}

// The 4 x 4 RGBA8Unorm render target over the backing at kTargetOffset.
static id<MTLTexture> targetOverBacking(Probe *probe, MTLTextureUsage usage) {
  MTLTextureDescriptor *descriptor = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                   width:kTargetSide
                                  height:kTargetSide
                               mipmapped:NO];
  descriptor.usage = usage;
  descriptor.storageMode = MTLStorageModeShared;
  descriptor.allowGPUOptimizedContents = NO;
  id<MTLTexture> texture = [probe->backing.buffer newTextureWithDescriptor:descriptor
                                                                   offset:kTargetOffset
                                                              bytesPerRow:kTargetPitch];
  say("target.over_buffer", texture == nil ? @"nil" : @"made");
  return texture;
}

static id<MTLRenderPipelineState> renderPipeline(Probe *probe, NSString *vertex,
                                                 NSString *fragment, NSUInteger samples) {
  MTLRenderPipelineDescriptor *descriptor = [MTLRenderPipelineDescriptor new];
  descriptor.vertexFunction = functionNamed(probe, vertex);
  descriptor.fragmentFunction = functionNamed(probe, fragment);
  descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
  descriptor.rasterSampleCount = samples;
  NSError *error = nil;
  id<MTLRenderPipelineState> state =
      [probe->device newRenderPipelineStateWithDescriptor:descriptor error:&error];
  if (state == nil) {
    say("render_pipeline", ([NSString stringWithFormat:@"failed: %@", error]));
  }
  return state;
}

// Encode the copy that answers "what does a read of the backing see from here?" into `cb`, into
// `into`.
static void encodeReadBack(id<MTLCommandBuffer> cb, Probe *probe, id<MTLBuffer> into) {
  id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
  [blit copyFromBuffer:probe->backing.buffer
          sourceOffset:0
              toBuffer:into
     destinationOffset:0
                  size:kRegion];
  [blit endEncoding];
}

// Encode one shape's write into `cb`, answering NO where the shape could not be set up at all (a
// nil texture, a pipeline that would not compile), so a run says "unavailable" rather than
// reporting an untouched region as a dropped write.
static BOOL encodeShape(id<MTLCommandBuffer> cb, Probe *probe, NSString *shape) {
  id<MTLBuffer> backing = probe->backing.buffer;

  if ([shape isEqualToString:@"blit-fill"]) {
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit fillBuffer:backing range:NSMakeRange(kTargetOffset, kTargetLength) value:0x11];
    [blit endEncoding];
    return YES;
  }

  if ([shape isEqualToString:@"kernel-store"]) {
    id<MTLFunction> function = functionNamed(probe, @"ro_store");
    if (function == nil) return NO;
    NSError *error = nil;
    id<MTLComputePipelineState> state =
        [probe->device newComputePipelineStateWithFunction:function error:&error];
    if (state == nil) {
      say("compute_pipeline", ([NSString stringWithFormat:@"failed: %@", error]));
      return NO;
    }
    id<MTLComputeCommandEncoder> compute = [cb computeCommandEncoder];
    [compute setComputePipelineState:state];
    [compute setBuffer:backing offset:kTargetOffset atIndex:0];
    [compute dispatchThreads:MTLSizeMake(kTargetLength / 4, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [compute endEncoding];
    return YES;
  }

  if ([shape isEqualToString:@"kernel-store-loaded"]) {
    // The store goes through an address the shader loads out of an argument buffer, which an
    // MTLArgumentEncoder wrote. The backing is made resident for writing by hand, because
    // nothing in the command buffer names it.
    id<MTLFunction> function = functionNamed(probe, @"ro_store_loaded");
    if (function == nil) return NO;
    NSError *error = nil;
    id<MTLComputePipelineState> state =
        [probe->device newComputePipelineStateWithFunction:function error:&error];
    if (state == nil) {
      say("compute_pipeline", ([NSString stringWithFormat:@"failed: %@", error]));
      return NO;
    }
    id<MTLArgumentEncoder> arguments = [function newArgumentEncoderWithBufferIndex:0];
    if (arguments == nil) {
      say("argument_encoder", @"nil");
      return NO;
    }
    id<MTLBuffer> argumentBuffer =
        [probe->device newBufferWithLength:MAX(arguments.encodedLength, (NSUInteger)16)
                                   options:MTLResourceStorageModeShared];
    [arguments setArgumentBuffer:argumentBuffer offset:0];
    [arguments setBuffer:backing offset:kTargetOffset atIndex:0];
    sayu("argument_encoder.encoded_length", arguments.encodedLength);
    id<MTLComputeCommandEncoder> compute = [cb computeCommandEncoder];
    [compute setComputePipelineState:state];
    [compute setBuffer:argumentBuffer offset:0 atIndex:0];
    [compute useResource:backing usage:MTLResourceUsageWrite];
    [compute dispatchThreads:MTLSizeMake(kTargetLength / 4, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [compute endEncoding];
    return YES;
  }

  if ([shape isEqualToString:@"color-attachment"] ||
      [shape isEqualToString:@"color-attachment-draw"]) {
    BOOL draws = [shape isEqualToString:@"color-attachment-draw"];
    id<MTLTexture> target = targetOverBacking(probe, MTLTextureUsageRenderTarget);
    if (target == nil) return NO;
    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = target;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(1.0, 0.0, 0.5, 1.0);
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderPipelineState> state = nil;
    if (draws) {
      state = renderPipeline(probe, @"ro_vert", @"ro_frag", 1);
      if (state == nil) return NO;
    }
    id<MTLRenderCommandEncoder> render = [cb renderCommandEncoderWithDescriptor:pass];
    if (draws) {
      [render setRenderPipelineState:state];
      [render drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }
    [render endEncoding];
    return YES;
  }

  if ([shape isEqualToString:@"resolve"]) {
    // A 4x multisample colour attachment of Metal's own memory, resolved into the target over the
    // backing: the resolve is the only write the pass makes into read-only memory.
    id<MTLTexture> resolveInto = targetOverBacking(probe, MTLTextureUsageRenderTarget);
    if (resolveInto == nil) return NO;
    MTLTextureDescriptor *descriptor = [MTLTextureDescriptor new];
    descriptor.textureType = MTLTextureType2DMultisample;
    descriptor.pixelFormat = MTLPixelFormatRGBA8Unorm;
    descriptor.width = kTargetSide;
    descriptor.height = kTargetSide;
    descriptor.sampleCount = 4;
    descriptor.usage = MTLTextureUsageRenderTarget;
    descriptor.storageMode = MTLStorageModePrivate;
    id<MTLTexture> samples = [probe->device newTextureWithDescriptor:descriptor];
    if (samples == nil) {
      say("multisample_target", @"nil");
      return NO;
    }
    id<MTLRenderPipelineState> state = renderPipeline(probe, @"ro_vert", @"ro_frag", 4);
    if (state == nil) return NO;
    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = samples;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(1.0, 0.0, 0.5, 1.0);
    pass.colorAttachments[0].storeAction = MTLStoreActionMultisampleResolve;
    pass.colorAttachments[0].resolveTexture = resolveInto;
    id<MTLRenderCommandEncoder> render = [cb renderCommandEncoderWithDescriptor:pass];
    [render setRenderPipelineState:state];
    [render drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [render endEncoding];
    return YES;
  }

  if ([shape isEqualToString:@"visibility"]) {
    // The pass's visibility-result buffer IS the read-only backing. The colour attachment is
    // Metal's own memory, so the only write into read-only memory is the result word.
    MTLTextureDescriptor *descriptor = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                     width:kTargetSide
                                    height:kTargetSide
                                 mipmapped:NO];
    descriptor.usage = MTLTextureUsageRenderTarget;
    descriptor.storageMode = MTLStorageModePrivate;
    id<MTLTexture> target = [probe->device newTextureWithDescriptor:descriptor];
    if (target == nil) return NO;
    id<MTLRenderPipelineState> state = renderPipeline(probe, @"ro_vert", @"ro_frag", 1);
    if (state == nil) return NO;
    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = target;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    pass.visibilityResultBuffer = probe->backing.buffer;
    id<MTLRenderCommandEncoder> render = [cb renderCommandEncoderWithDescriptor:pass];
    [render setRenderPipelineState:state];
    [render setVisibilityResultMode:MTLVisibilityResultModeCounting offset:kTargetOffset];
    [render drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [render setVisibilityResultMode:MTLVisibilityResultModeDisabled offset:kTargetOffset];
    [render endEncoding];
    return YES;
  }

  if ([shape isEqualToString:@"vertex-store"] || [shape isEqualToString:@"fragment-store"] ||
      [shape isEqualToString:@"fragment-store-loaded"]) {
    // A draw whose stage stores into the read-only backing through a bound writable buffer, or
    // through an address loaded out of an argument buffer. The colour attachment is Metal's own
    // memory, so the stage's store is the only write into read-only memory.
    BOOL vertex = [shape isEqualToString:@"vertex-store"];
    BOOL loaded = [shape isEqualToString:@"fragment-store-loaded"];
    MTLTextureDescriptor *descriptor = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                     width:kTargetSide
                                    height:kTargetSide
                                 mipmapped:NO];
    descriptor.usage = MTLTextureUsageRenderTarget;
    descriptor.storageMode = MTLStorageModePrivate;
    id<MTLTexture> target = [probe->device newTextureWithDescriptor:descriptor];
    if (target == nil) return NO;
    id<MTLRenderPipelineState> state =
        renderPipeline(probe, vertex ? @"ro_vert_store" : @"ro_vert",
                       vertex ? @"ro_frag" : (loaded ? @"ro_frag_loaded" : @"ro_frag_store"), 1);
    if (state == nil) return NO;
    id<MTLBuffer> bound = probe->backing.buffer;
    NSUInteger boundOffset = kTargetOffset;
    id<MTLFunction> function = nil;
    if (loaded) {
      function = functionNamed(probe, @"ro_frag_loaded");
      if (function == nil) return NO;
      id<MTLArgumentEncoder> arguments = [function newArgumentEncoderWithBufferIndex:0];
      if (arguments == nil) {
        say("argument_encoder", @"nil");
        return NO;
      }
      bound = [probe->device newBufferWithLength:MAX(arguments.encodedLength, (NSUInteger)16)
                                        options:MTLResourceStorageModeShared];
      [arguments setArgumentBuffer:bound offset:0];
      [arguments setBuffer:probe->backing.buffer offset:kTargetOffset atIndex:0];
      boundOffset = 0;
      sayu("argument_encoder.encoded_length", arguments.encodedLength);
    }
    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = target;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> render = [cb renderCommandEncoderWithDescriptor:pass];
    [render setRenderPipelineState:state];
    if (vertex) {
      [render setVertexBuffer:bound offset:boundOffset atIndex:1];
    } else {
      [render setFragmentBuffer:bound offset:boundOffset atIndex:0];
      if (loaded) {
        [render useResource:probe->backing.buffer
                      usage:MTLResourceUsageWrite
                     stages:MTLRenderStageFragment];
      }
    }
    [render drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [render endEncoding];
    return YES;
  }

  say("shape", ([NSString stringWithFormat:@"unknown-%@", shape]));
  return NO;
}

// ---------------------------------------------------------------------------------------------
// The depth and stencil shapes, which need a heap rather than a buffer
// ---------------------------------------------------------------------------------------------

// A depth or a stencil texture cannot lie over an MTLBuffer at all -- Metal makes a linear texture
// of colour formats only -- so a depth attachment's read-only backing has to be a HEAP's. Two
// forms of that, and this is where the SPI buys something bit 24 cannot:
//
//   bit24     a heap of Metal's own memory made with bit 24. No host pointer at all, so there is
//             no CPU leg and no page protection to report.
//   replaced  a heap whose backing is DETACHED and then replaced, read-only, over the probe's own
//             pages -- the SPI is on MTLHeap as well as on MTLBuffer. So this form puts a depth
//             texture over host pages, which the bit-24 form could not, and it has a CPU leg.
//
// `plain` and `descriptor` are the writable controls for the two columns.
static int runDepthOrStencil(id<MTLDevice> device, id<MTLCommandQueue> queue, NSString *shape,
                             NSString *which, NSMutableArray *answers) {
  BOOL privateBit = [which isEqualToString:@"bit24"];
  BOOL replaces = [which isEqualToString:@"replaced"];
  BOOL detachesOnly = [which isEqualToString:@"detached"];
  BOOL stencil = [shape isEqualToString:@"stencil-attachment"];
  MTLPixelFormat format = stencil ? MTLPixelFormatStencil8 : MTLPixelFormatDepth32Float;
  MTLTextureDescriptor *descriptor = [MTLTextureDescriptor
      texture2DDescriptorWithPixelFormat:format
                                   width:kTargetSide
                                  height:kTargetSide
                               mipmapped:NO];
  descriptor.usage = MTLTextureUsageRenderTarget;
  // The SPI forms need a SHARED heap: `detachBacking` answered NO on a placement heap of PRIVATE
  // storage, both selectors responding, which is why this follows the backing rather than being
  // Private throughout. A depth or stencil texture of shared storage is one this chip makes.
  MTLStorageMode storage =
      (replaces || detachesOnly) ? MTLStorageModeShared : MTLStorageModePrivate;
  descriptor.storageMode = storage;
  descriptor.allowGPUOptimizedContents = NO;
  sayu("heap.storage_mode", (unsigned long long)storage);
  MTLSizeAndAlign sized = [device heapTextureSizeAndAlignWithDescriptor:descriptor];
  sayu("heap.texture_size", sized.size);
  sayu("heap.texture_align", sized.align);

  MTLHeapDescriptor *heapDescriptor = [MTLHeapDescriptor new];
  heapDescriptor.type = MTLHeapTypePlacement;
  heapDescriptor.storageMode = storage;
  heapDescriptor.size = sized.size + sized.align + 4096;
  heapDescriptor.resourceOptions =
      (storage == MTLStorageModeShared ? MTLResourceStorageModeShared
                                       : MTLResourceStorageModePrivate) |
      (privateBit ? kPrivateBit : 0);
  // Whether a heap can be made over the CALLER's bytes at all, which is the only shape whose
  // backing there would be anything to detach: every successful detach measured here was on a
  // no-copy buffer. Reported as the selectors the descriptor answers, so "no route" is evidence
  // rather than an assumption.
  {
    NSMutableArray *answered = [NSMutableArray array];
    for (NSString *name in @[ @"setContents:", @"setNoCopy:", @"setDeallocator:" ]) {
      if ([heapDescriptor respondsToSelector:NSSelectorFromString(name)]) {
        [answered addObject:name];
      }
    }
    say("heap_descriptor.no_copy_setters",
        answered.count == 0 ? @"none" : [answered componentsJoinedByString:@" "]);
    say("device.newHeapWithBytesNoCopy",
        [device respondsToSelector:NSSelectorFromString(
                                       @"newHeapWithBytesNoCopy:length:options:deallocator:")]
            ? @"responds"
            : @"absent");
  }
  id<MTLHeap> heap = [device newHeapWithDescriptor:heapDescriptor];
  if (heap == nil) {
    say("heap", @"nil");
    return 2;
  }
  sayx("heap.resource_options", (unsigned long long)heap.resourceOptions);
  sayu("heap.size", heap.size);
  sayBackingSelectors("heap", heap);

  // The pages the replaced form puts under the heap, and the CPU's own view of them. Mapped
  // whole-page and filled with the pad, as a buffer backing is, so an untouched byte is visible.
  uint8_t *pages = NULL;
  if (replaces || detachesOnly) {
    void *mapped =
        mmap(NULL, kRegion, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (mapped == MAP_FAILED) {
      say("heap.pages", @"mmap-failed");
      return 2;
    }
    memset(mapped, kPad, kRegion);
    pages = (uint8_t *)mapped;
    sayProtection("heap_pages", pages);
    // The detach goes first and answers for itself: the replace refuses an attached object, so a
    // NO would otherwise not say which call refused. The texture is placed AFTER this, because
    // the heap's bytes are what the replace swaps.
    if (!detachAndReplace("heap", heap, pages, kRegion, detachesOnly)) {
      return 2;
    }
  }

  id<MTLTexture> target = [heap newTextureWithDescriptor:descriptor offset:0];
  if (target == nil) {
    say("heap.texture", @"nil");
    return 2;
  }
  say("heap.texture", @"made");

  // Where the read-back lands: an ordinary shared buffer, pre-filled, so a copy that wrote
  // nothing reads as itself.
  NSUInteger readBytes = kTargetSide * kTargetSide * (stencil ? 1 : 4);
  NSUInteger pitch = stencil ? 256 : 256;
  id<MTLBuffer> out = [device newBufferWithLength:pitch * kTargetSide
                                         options:MTLResourceStorageModeShared];
  memset(out.contents, kUnread, out.length);
  id<MTLBuffer> out2 = [device newBufferWithLength:pitch * kTargetSide
                                          options:MTLResourceStorageModeShared];
  memset(out2.contents, kUnread, out2.length);

  MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
  if (stencil) {
    pass.stencilAttachment.texture = target;
    pass.stencilAttachment.loadAction = MTLLoadActionClear;
    pass.stencilAttachment.clearStencil = 0x37;
    pass.stencilAttachment.storeAction = MTLStoreActionStore;
  } else {
    pass.depthAttachment.texture = target;
    pass.depthAttachment.loadAction = MTLLoadActionClear;
    pass.depthAttachment.clearDepth = 0.75;
    pass.depthAttachment.storeAction = MTLStoreActionStore;
  }
  pass.renderTargetWidth = kTargetSide;
  pass.renderTargetHeight = kTargetSide;

  id<MTLCommandBuffer> cb = [queue commandBuffer];
  id<MTLRenderCommandEncoder> render = [cb renderCommandEncoderWithDescriptor:pass];
  [render endEncoding];
  id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
  [blit copyFromTexture:target
             sourceSlice:0
             sourceLevel:0
            sourceOrigin:MTLOriginMake(0, 0, 0)
              sourceSize:MTLSizeMake(kTargetSide, kTargetSide, 1)
                toBuffer:out
       destinationOffset:0
  destinationBytesPerRow:pitch
destinationBytesPerImage:pitch * kTargetSide];
  [blit endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  sayCommandBuffer("cb1", cb);

  id<MTLCommandBuffer> cb2 = [queue commandBuffer];
  id<MTLBlitCommandEncoder> blit2 = [cb2 blitCommandEncoder];
  [blit2 copyFromTexture:target
              sourceSlice:0
              sourceLevel:0
             sourceOrigin:MTLOriginMake(0, 0, 0)
               sourceSize:MTLSizeMake(kTargetSide, kTargetSide, 1)
                 toBuffer:out2
        destinationOffset:0
   destinationBytesPerRow:pitch
 destinationBytesPerImage:pitch * kTargetSide];
  [blit2 endEncoding];
  [cb2 commit];
  [cb2 waitUntilCompleted];
  sayCommandBuffer("cb2", cb2);

  NSString *same = changedRuns((const uint8_t *)out.contents, out.length, kUnread);
  NSString *later = changedRuns((const uint8_t *)out2.contents, out2.length, kUnread);
  say("same_cb", same);
  say("later_cb", later);
  sayu("read.bytes_per_row", pitch);
  sayu("read.row_bytes", readBytes / kTargetSide);
  NSString *cpu = pages == NULL ? @"not-applicable-heap-is-device-memory"
                                : changedRuns(pages, kRegion, kPad);
  say("cpu", cpu);
  [answers addObject:[NSString stringWithFormat:@"status=%@/%@ same=%@ later=%@ cpu=%@",
                                                statusWord(cb.status), statusWord(cb2.status),
                                                same, later, cpu]];
  if (pages != NULL) {
    munmap(pages, kRegion);
  }
  return 0;
}

// ---------------------------------------------------------------------------------------------

static int usage(void) {
  fprintf(stderr,
          "usage: readonly_gpu_writes <backing> <shape> [runs]\n"
          "  backing  plain | bit24 | viewed        (newBufferWithBytesNoCopy:)\n"
          "           descriptor | replaced | detached   (private MTLBufferDescriptor)\n"
          "           nocopy-replaced                    (the SPI on a no-copy buffer)\n"
          "  shape    facts | blit-fill | kernel-store | kernel-store-loaded |\n"
          "           color-attachment | color-attachment-draw | resolve | visibility |\n"
          "           vertex-store | fragment-store | fragment-store-loaded |\n"
          "           depth-attachment | stencil-attachment\n"
          "  runs     how many times to repeat the shape in this process (default 1)\n"
          "One shape per process: a shape Metal faults on must not take the others with it.\n"
          "Several runs per process because Metal's answer to a question of this kind has been\n"
          "measured flapping between runs, so a single reading establishes nothing. Each run\n"
          "makes its own pages, its own buffer and its own command buffers; the last line tallies\n"
          "the distinct answers seen, and more than one of them IS the measurement.\n");
  return 2;
}

// One run of a buffer-backed shape: its own pages, its own buffer, its own two command buffers.
//
// Appends the run's three answers to `answers` as one string, so the tally at the end can say
// whether Metal gave the same answer every time. `library` is made once for the process, because
// compiling the shaders each run would be work the measurement does not need and Metal's shader
// compiler is the thing that wedges this device.
static int runBufferShape(id<MTLDevice> device, id<MTLCommandQueue> queue, NSString *which,
                          NSString *shape, id<MTLLibrary> library, NSMutableArray *answers) {
  Probe probe = {device, queue, library, {nil, NULL, NULL, @"none"}, nil, nil};
  probe.backing = makeBacking(device, which);
  if (probe.backing.buffer == nil) {
    return 2;
  }
  say("backing.note", probe.backing.note);
  sayx("buffer.resource_options", (unsigned long long)probe.backing.buffer.resourceOptions);
  sayu("buffer.storage_mode", (unsigned long long)probe.backing.buffer.storageMode);
  sayu("buffer.length", probe.backing.buffer.length);
  say("buffer.contents_is_imported",
      probe.backing.buffer.contents == probe.backing.imported ? @"yes" : @"no");
  say("buffer.contents_is_pages",
      probe.backing.buffer.contents == probe.backing.pages ? @"yes" : @"no");
  sayProtection("pages", probe.backing.pages);
  sayProtection("imported", probe.backing.imported);

  if ([shape isEqualToString:@"facts"]) {
    // The facts alone: no GPU work at all. Which selectors this creation path's buffer answers is
    // already said by `sayBackingSelectors` above, for every backing, which is what answers the
    // open question of whether a `newBufferWithBytesNoCopy:` buffer takes the SPI at all.
    NSString *cpu = changedRuns(probe.backing.pages, kRegion, kPad);
    say("cpu", cpu);
    [answers addObject:[NSString stringWithFormat:@"cpu=%@", cpu]];
    return 0;
  }

  probe.sameCb = [device newBufferWithLength:kRegion options:MTLResourceStorageModeShared];
  probe.laterCb = [device newBufferWithLength:kRegion options:MTLResourceStorageModeShared];
  memset(probe.sameCb.contents, kUnread, kRegion);
  memset(probe.laterCb.contents, kUnread, kRegion);

  // One command buffer: the write, then the read-back. Metal orders a command buffer's encoders,
  // so the copy sees whatever the write left for a later command of the same command buffer --
  // which is the surprising half of the fact already written down, and the reason both halves of
  // the round trip are in ONE command buffer here.
  id<MTLCommandBuffer> cb = [queue commandBuffer];
  if (!encodeShape(cb, &probe, shape)) {
    return 2;
  }
  encodeReadBack(cb, &probe, probe.sameCb);
  [cb commit];
  [cb waitUntilCompleted];
  sayCommandBuffer("cb1", cb);

  // A second command buffer, which is where the fact says the GPU's write is gone.
  id<MTLCommandBuffer> cb2 = [queue commandBuffer];
  encodeReadBack(cb2, &probe, probe.laterCb);
  [cb2 commit];
  [cb2 waitUntilCompleted];
  sayCommandBuffer("cb2", cb2);

  NSString *same =
      changedRuns((const uint8_t *)probe.sameCb.contents, kRegion, kPad);
  NSString *later =
      changedRuns((const uint8_t *)probe.laterCb.contents, kRegion, kPad);
  // The CPU's own view of the pages it owns. Read only: a CPU WRITE through a read-only mapping
  // ends the process, which is Metal's own behaviour and is not in scope here.
  NSString *cpu = changedRuns(probe.backing.pages, kRegion, kPad);
  say("same_cb", same);
  say("later_cb", later);
  say("cpu", cpu);
  sayProtection("pages_after", probe.backing.pages);
  [answers addObject:[NSString stringWithFormat:@"status=%@/%@ same=%@ later=%@ cpu=%@",
                                                statusWord(cb.status), statusWord(cb2.status),
                                                same, later, cpu]];
  return 0;
}

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc != 3 && argc != 4) {
      return usage();
    }
    NSString *which = [NSString stringWithUTF8String:argv[1]];
    NSString *shape = [NSString stringWithUTF8String:argv[2]];
    NSArray *backings = @[
      @"plain", @"bit24", @"viewed", @"descriptor", @"replaced", @"nocopy-replaced", @"detached"
    ];
    if (![backings containsObject:which]) {
      return usage();
    }
    int runs = argc == 4 ? atoi(argv[3]) : 1;
    if (runs < 1 || runs > 64) {
      return usage();
    }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      say("device", @"none");
      return 2;
    }
    say("device.name", device.name);
    // Checked FIRST and always, so a negative result from the SPI is never read as the OS not
    // offering it: that confusion is what made the first shape of this probe report the SPI
    // absent on this release.
    say("device.supportsResourceDetachBacking",
        ![device respondsToSelector:@selector(supportsResourceDetachBacking)]
            ? @"selector-absent"
            : ([(id<M2VBackingDevice>)device supportsResourceDetachBacking] ? @"yes" : @"no"));
    say("backing.kind", which);
    say("shape.name", shape);
    sayu("runs.asked", (unsigned long long)runs);
    sayu("region.bytes", kRegion);
    sayx("target.offset", kTargetOffset);
    sayx("target.length", kTargetLength);
    sayu("pad.byte", kPad);
    sayu("unread.byte", kUnread);

    // Every run of the process shares this one queue. A queue whose second command buffer fails
    // with a page fault is revoked, so from
    // a faulting shape's second run on, both command buffers fail with AccessRevoked and the
    // later read-back never runs: read a faulting shape's later command buffer from run 0 (M3 Max,
    // macOS 27.0.1, m3-max-29110: one fault leaves the next command buffer Completed; runs 1 to 3
    // of four on one queue answer Error/Error).
    id<MTLCommandQueue> queue = [device newCommandQueue];
    NSMutableArray *answers = [NSMutableArray array];

    // The depth and stencil shapes are a heap's, not a buffer's. The bit-24 form of such a heap
    // has no host pointer, so no CPU leg; the replaced form has one, because the SPI is on
    // MTLHeap too. `viewed` and `nocopy-replaced` are buffer creation paths and mean nothing here.
    BOOL heapShape = [shape isEqualToString:@"depth-attachment"] ||
                     [shape isEqualToString:@"stencil-attachment"];
    if (heapShape && ([which isEqualToString:@"viewed"] ||
                      [which isEqualToString:@"nocopy-replaced"])) {
      say("shape", @"that backing is a buffer creation path and a heap has none");
      return 2;
    }

    // The shaders, compiled once for the process. The two shapes that need no function skip it,
    // so a clear-only measurement never waits on Metal's shader compiler.
    id<MTLLibrary> library = nil;
    if (!heapShape && ![shape isEqualToString:@"facts"] &&
        ![shape isEqualToString:@"blit-fill"] && ![shape isEqualToString:@"color-attachment"]) {
      NSError *error = nil;
      library = [device newLibraryWithSource:kShaders options:nil error:&error];
      if (library == nil) {
        say("library", ([NSString stringWithFormat:@"failed: %@", error]));
        return 2;
      }
    }

    int code = 0;
    for (int run = 0; run < runs && code == 0; run++) {
      gRun = run;
      code = heapShape ? runDepthOrStencil(device, queue, shape, which, answers)
                       : runBufferShape(device, queue, which, shape, library, answers);
    }
    gRun = -1;

    // The tally, which is the measurement when Metal's answer is not stable. The consuming
    // project's own control case for a nearby question came out negative in 248 of 800 runs
    // against 39 of 40 on an earlier date, so a shape read once has established nothing. More
    // than one distinct answer here is a FLAPPING result and is reported as one, never resolved
    // into a majority.
    NSCountedSet *distinct = [NSCountedSet setWithArray:answers];
    sayu("runs.completed", (unsigned long long)answers.count);
    sayu("answers.distinct", (unsigned long long)distinct.count);
    say("answers.stable", distinct.count <= 1 ? @"yes" : @"NO-FLAPPING");
    for (NSString *answer in distinct) {
      printf("answer[%lu/%lu]=%s\n", (unsigned long)[distinct countForObject:answer],
             (unsigned long)answers.count, answer.UTF8String);
    }
    fflush(stdout);
    return code;
  }
}
