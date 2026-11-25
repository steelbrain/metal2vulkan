// What plain Metal answers for a metallib whose function entries lack one of the tags the AIR
// front end's reader requires of an entry (`NAME`, `TYPE`, `HASH`, `MDSZ`, `OFFT`, `VERS`), with
// the header's format version as Apple's toolchain writes it today and as 2.2 (an older
// toolchain's, whose entries carry no `MDSZ`).
//
// metallib_without_entry_tags.sh compiles its MSL with Apple's toolchain into the metallib this
// probe reads, once per run, each run with its own SEED so that no run is answered from another's
// compiled functions, and runs each case once with MTL_DEBUG_LAYER=1 and once without, ONE CASE
// PER PROCESS.
//
// Usage: metallib_without_entry_tags <file.metallib> <seed> <case>
//   as-is          the file as written (the control), and its structure printed: per function its
//                  tags, MDSZ, its module's start (OFFT's bitcode offset), the bitcode wrapper's
//                  offset and size there, the next module's start, and which of three ranges'
//                  SHA-256 is its HASH: MDSZ bytes, the wrapper's (offset + size), and up to the
//                  next module's start (or the section's end)
//   no-TAG         TAG (NAME, TYPE, HASH, MDSZ, OFFT or VERS) left out of every entry; the
//                  entries shrink and the extended header moves up after them, leaving zeros up to
//                  where it ended, so no other byte or offset of the file moves
//   no-TAG-shift   the same, with every byte after the entries moved up and the header's file size
//                  and section offsets (and an extended-header region's offset) moved with it
//   v2.2           as-is with the header's version 2.2, its os byte and os version 0, as the older
//                  toolchain writes them
//   no-TAG-v2.2    no-TAG at that version
//
// A case prints whether `newLibraryWithData:error:` made a library (or its error), its
// `functionNames` count, and per function of the file's table, by its original NAME:
// `newFunctionWithName:` made or not and its `functionType`; each kernel's compute pipeline and the
// four words a four-thread dispatch stores (fill_a stores t * 3 + SEED, fill_b t * 5 + SEED); and
// the render pipeline of vert and frag and the texel its full-screen triangle draws into a 4x4
// RGBA8Unorm texture (expected: R = (SEED % 251) / 250, G = (SEED % 13) / 12, B = 0.25, A = 1).
//
// Build: clang -fobjc-arc -O1 -fmodules -o metallib_without_entry_tags \
//          metallib_without_entry_tags.m -framework Foundation -framework Metal
//
// WHAT IT MEASURED, Apple M3 Max, macOS 27.0.1 (Build 26A434), Apple metal 32023.922, job
// m3-max-73748, every case below alike with MTL_DEBUG_LAYER=1 and without:
//
//   - as-is: header 2.9, every entry NAME, TYPE, HASH, MDSZ, OFFT, VERS. Each of the four modules
//     is a bitcode wrapper padded to a multiple of 16 bytes, the next written after it; MDSZ is
//     the distance to the next module's start (or the section's end), 8 or 12 bytes past the
//     wrapper's offset plus size, and HASH is the SHA-256 of those MDSZ bytes, not of the
//     wrapper's alone.
//     The extended header is ENDT alone. Library, functions, both kernels' words and the texel as
//     expected;
//   - v2.2, no-MDSZ, no-MDSZ-shift, no-MDSZ-v2.2: the same answers as as-is: Metal makes the
//     library and its four functions, the kernels store the expected words and the render
//     pipeline draws the expected texel;
//   - no-NAME (and -shift): the library is made with no function names, and no function is made;
//   - no-TYPE (and -shift): the library is made with four function names, and the first
//     newFunctionWithName: ends the process with the layer on and off (SIGABRT, exit 134):
//     "validateMTLFunctionType: failed assertion `type is not a valid MTLFunctionType.'";
//   - no-HASH (and -shift): the library and functions are made and run, with the layer on as
//     expected, but every later run (seeds of its own) stores the first no-HASH run's words and
//     draws its texel: a function without HASH is answered with what Metal compiled for another;
//   - no-OFFT (and -shift): every function is made; fill_a runs as expected, fill_b's pipeline
//     fails ("Function fill_b was not found in the module."), and the render pipeline's
//     ("Compilation failed due to an interrupted connection: XPC_ERROR_CONNECTION_INTERRUPTED"):
//     every function reads the first module;
//   - no-VERS (and -shift): every function is made and every pipeline fails ("Function fill_a is
//     using language version 0.0 which is incompatible with this OS.").
#import <CommonCrypto/CommonDigest.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t u32(const uint8_t *p) { return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24; }
static uint16_t u16(const uint8_t *p) { return (uint16_t)(p[0] | p[1] << 8); }
static uint64_t u64(const uint8_t *p) { return (uint64_t)u32(p) | (uint64_t)u32(p + 4) << 32; }
static void put64(uint8_t *p, uint64_t v) { for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i)); }
static void put32(uint8_t *p, uint32_t v) { for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i)); }

typedef struct {
  size_t at, size;      // the entry: its size word's offset and its size
  char name[256];
  int type;
  uint64_t mdsz, offt_bitcode;
  int has_mdsz, has_offt;
  uint8_t hash[32];
  int has_hash;
  char tags[256];       // its tags in order, with their payload sizes
} Entry;

// The function table's entries (at most 64).
static int entries(const uint8_t *b, size_t len, Entry *out, int max) {
  uint64_t list = u64(b + 24);
  if (list + 4 > len) return -1;
  uint32_t count = u32(b + list);
  size_t at = list + 4;
  int n = 0;
  for (uint32_t i = 0; i < count && n < max && at + 4 <= len; i++, n++) {
    Entry *e = &out[n];
    memset(e, 0, sizeof *e);
    e->at = at;
    e->size = u32(b + at);
    e->type = -1;
    size_t end = at + e->size, t = at + 4;
    while (t + 4 <= end && memcmp(b + t, "ENDT", 4) != 0 && t + 6 <= end) {
      uint16_t m = u16(b + t + 4);
      const uint8_t *v = b + t + 6;
      char one[32];
      snprintf(one, sizeof one, "%s%.4s:%u", e->tags[0] ? "," : "", (const char *)b + t, m);
      strncat(e->tags, one, sizeof e->tags - strlen(e->tags) - 1);
      if (memcmp(b + t, "NAME", 4) == 0 && m > 0 && m < sizeof e->name) memcpy(e->name, v, m);
      if (memcmp(b + t, "TYPE", 4) == 0 && m == 1) e->type = v[0];
      if (memcmp(b + t, "MDSZ", 4) == 0 && m == 8) { e->mdsz = u64(v); e->has_mdsz = 1; }
      if (memcmp(b + t, "OFFT", 4) == 0 && m == 24) { e->offt_bitcode = u64(v + 16); e->has_offt = 1; }
      if (memcmp(b + t, "HASH", 4) == 0 && m == 32) { memcpy(e->hash, v, 32); e->has_hash = 1; }
      t += 6 + m;
    }
    at = end;
  }
  return n;
}

static int hash_is(const uint8_t *b, size_t len, uint64_t start, uint64_t size, const uint8_t *want) {
  if (start + size > len) return 0;
  uint8_t d[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256(b + start, (CC_LONG)size, d);
  return memcmp(d, want, 32) == 0;
}

// The file's structure (the as-is case).
static void structure(const uint8_t *b, size_t len, const Entry *e, int n) {
  uint64_t list = u64(b + 24), list_size = u64(b + 32), bc = u64(b + 72), bc_size = u64(b + 80);
  size_t entries_end = list + 4 + list_size;
  printf("HEADER version=%u.%u file_type=%u os=%u os_version=%u.%u file_size=%llu len=%zu list=%llu+%llu"
         " public=%llu+%llu private=%llu+%llu bitcode=%llu+%llu after_bitcode=%llu\n",
         u16(b + 6), u16(b + 8), b[10], b[11], u16(b + 12), u16(b + 14),
         (unsigned long long)u64(b + 16), len, (unsigned long long)list,
         (unsigned long long)list_size, (unsigned long long)u64(b + 40),
         (unsigned long long)u64(b + 48), (unsigned long long)u64(b + 56),
         (unsigned long long)u64(b + 64), (unsigned long long)bc, (unsigned long long)bc_size,
         (unsigned long long)(len - (bc + bc_size)));
  printf("EXTENDED at=%zu", entries_end);
  for (size_t t = entries_end; t + 4 <= len;) {
    if (memcmp(b + t, "ENDT", 4) == 0) { printf(" ENDT end=%zu\n", t + 4); break; }
    uint16_t m = u16(b + t + 4);
    printf(" %.4s:%u", (const char *)b + t, m);
    t += 6 + m;
  }
  for (int i = 0; i < n; i++) {
    uint64_t start = bc + e[i].offt_bitcode, next = bc + bc_size;
    for (int j = 0; j < n; j++) {
      uint64_t s = bc + e[j].offt_bitcode;
      if (s > start && s < next) next = s;
    }
    const uint8_t *w = b + start;
    int wrapped = start + 20 <= len && u32(w) == 0x0B17C0DE;
    uint64_t wend = wrapped ? (uint64_t)u32(w + 8) + u32(w + 12) : 0;
    printf("ENTRY %d name=%s type=%d tags=%s mdsz=%llu start=%llu wrapper=%s", i, e[i].name,
           e[i].type, e[i].tags, (unsigned long long)e[i].mdsz,
           (unsigned long long)e[i].offt_bitcode, wrapped ? "yes" : "no");
    if (wrapped)
      printf(" wrapper_offset=%u wrapper_size=%u wrapper_end=%llu", u32(w + 8), u32(w + 12),
             (unsigned long long)wend);
    printf(" to_next=%llu", (unsigned long long)(next - start));
    if (e[i].has_hash)
      printf(" hash_of_mdsz=%s hash_of_wrapper_end=%s hash_of_to_next=%s",
             e[i].has_mdsz && hash_is(b, len, start, e[i].mdsz, e[i].hash) ? "yes" : "no",
             wrapped && hash_is(b, len, start, wend, e[i].hash) ? "yes" : "no",
             hash_is(b, len, start, next - start, e[i].hash) ? "yes" : "no");
    printf("\n");
  }
}

// The file with `drop` left out of every entry (NULL: none), its bytes after the entries moved
// up (`shift`) or the extended header moved up and zeros left behind it, and at version 2.2
// (`v22`).
static NSMutableData *edited(const uint8_t *b, size_t len, const char *drop, BOOL shift, BOOL v22) {
  uint64_t list = u64(b + 24), list_size = u64(b + 32);
  size_t old_end = list + 4 + list_size;
  uint32_t count = u32(b + list);
  NSMutableData *table = [NSMutableData data];
  size_t at = list + 4;
  for (uint32_t i = 0; i < count; i++) {
    size_t size = u32(b + at), end = at + size, t = at + 4;
    NSMutableData *tags = [NSMutableData data];
    while (t + 4 <= end) {
      if (memcmp(b + t, "ENDT", 4) == 0) { [tags appendBytes:"ENDT" length:4]; break; }
      size_t one = 6 + u16(b + t + 4);
      if (!drop || memcmp(b + t, drop, 4) != 0) [tags appendBytes:b + t length:one];
      t += one;
    }
    uint8_t word[4];
    put32(word, (uint32_t)(tags.length + 4));
    [table appendBytes:word length:4];
    [table appendData:tags];
    at = end;
  }
  size_t ext_end = old_end;
  while (ext_end + 4 <= len && memcmp(b + ext_end, "ENDT", 4) != 0) ext_end += 6 + u16(b + ext_end + 4);
  ext_end += 4;
  size_t delta = list_size - table.length;
  NSMutableData *out;
  if (shift) {
    out = [NSMutableData dataWithBytes:b length:list + 4];
    [out appendData:table];
    [out appendBytes:b + old_end length:len - old_end];
    uint8_t *o = out.mutableBytes;
    put64(o + 16, u64(o + 16) - delta);
    for (int f = 40; f <= 72; f += 16)
      if (u64(o + f) >= old_end) put64(o + f, u64(o + f) - delta);
    // An extended-header region (`HDYN`, `ILST`): its offset, past the entries, moves too.
    for (size_t t = list + 4 + table.length; t + 4 <= out.length && memcmp(o + t, "ENDT", 4) != 0;
         t += 6 + u16(o + t + 4))
      if ((memcmp(o + t, "HDYN", 4) == 0 || memcmp(o + t, "ILST", 4) == 0) && u16(o + t + 4) == 16 &&
          u64(o + t + 6) >= old_end)
        put64(o + t + 6, u64(o + t + 6) - delta);
  } else {
    out = [NSMutableData dataWithBytes:b length:len];
    uint8_t *o = out.mutableBytes;
    memcpy(o + list + 4, table.bytes, table.length);
    memmove(o + list + 4 + table.length, b + old_end, ext_end - old_end);
    memset(o + list + 4 + table.length + (ext_end - old_end), 0, delta);
  }
  uint8_t *o = out.mutableBytes;
  put64(o + 32, table.length);
  if (v22) {
    o[6] = 2, o[7] = 0, o[8] = 2, o[9] = 0;
    o[11] = 0;
    o[12] = o[13] = o[14] = o[15] = 0;
  }
  printf("EDIT drop=%s shift=%s v2.2=%s removed=%zu file=%zu\n", drop ? drop : "none",
         shift ? "yes" : "no", v22 ? "yes" : "no", delta, out.length);
  return out;
}

static void run_kernel(id<MTLDevice> D, id<MTLCommandQueue> Q, id<MTLFunction> f, uint32_t step,
                       uint32_t seed) {
  NSError *err = nil;
  id<MTLComputePipelineState> p = [D newComputePipelineStateWithFunction:f error:&err];
  if (!p) {
    printf("  pipeline=nil error=%s\n", err.localizedDescription.UTF8String);
    return;
  }
  id<MTLBuffer> out = [D newBufferWithLength:16 options:MTLResourceStorageModeShared];
  memset(out.contents, 0xAB, 16);
  id<MTLCommandBuffer> cb = [Q commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:p];
  [enc setBuffer:out offset:0 atIndex:0];
  [enc dispatchThreads:MTLSizeMake(4, 1, 1) threadsPerThreadgroup:MTLSizeMake(4, 1, 1)];
  [enc endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  const uint32_t *w = out.contents;
  int right = 1;
  for (uint32_t t = 0; t < 4; t++) right &= w[t] == t * step + seed;
  printf("  pipeline=ok status=%ld words=%u,%u,%u,%u expected=%s\n", (long)cb.status, w[0], w[1], w[2],
         w[3], right ? "yes" : "no");
}

static void run_render(id<MTLDevice> D, id<MTLCommandQueue> Q, id<MTLFunction> v, id<MTLFunction> f,
                       uint32_t seed) {
  MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
  d.vertexFunction = v;
  d.fragmentFunction = f;
  d.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
  NSError *err = nil;
  id<MTLRenderPipelineState> p = [D newRenderPipelineStateWithDescriptor:d error:&err];
  if (!p) {
    printf("RENDER pipeline=nil error=%s\n", err.localizedDescription.UTF8String);
    return;
  }
  MTLTextureDescriptor *td =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:4 height:4 mipmapped:NO];
  td.usage = MTLTextureUsageRenderTarget;
  td.storageMode = MTLStorageModeShared;
  id<MTLTexture> t = [D newTextureWithDescriptor:td];
  MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
  rp.colorAttachments[0].texture = t;
  rp.colorAttachments[0].loadAction = MTLLoadActionClear;
  rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
  rp.colorAttachments[0].storeAction = MTLStoreActionStore;
  id<MTLCommandBuffer> cb = [Q commandBuffer];
  id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
  [enc setRenderPipelineState:p];
  [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
  [enc endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  uint8_t px[4 * 4 * 4];
  [t getBytes:px bytesPerRow:16 fromRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0];
  int want[4] = {(int)((seed % 251) * 255.0 / 250.0 + 0.5), (int)((seed % 13) * 255.0 / 12.0 + 0.5), 64, 255};
  int right = 1;
  for (int i = 0; i < 16; i++)
    for (int c = 0; c < 4; c++) right &= abs(px[4 * i + c] - want[c]) <= 1;
  printf("RENDER pipeline=ok status=%ld texel=%u,%u,%u,%u expected=%s\n", (long)cb.status, px[0], px[1],
         px[2], px[3], right ? "yes" : "no");
}

int main(int argc, char **argv) {
  if (argc < 4) return 2;
  @autoreleasepool {
    NSData *file = [NSData dataWithContentsOfFile:@(argv[1])];
    if (!file || file.length < 0x58) { printf("RESULT unreadable\n"); return 1; }
    const uint8_t *b = file.bytes;
    size_t len = file.length;
    uint32_t seed = (uint32_t)strtoul(argv[2], NULL, 10);
    const char *c = argv[3];
    printf("RESULT case=%s seed=%u debug_layer=%s\n", c, seed,
           getenv("MTL_DEBUG_LAYER") ? getenv("MTL_DEBUG_LAYER") : "unset");
    Entry e[64];
    int n = entries(b, len, e, 64);
    if (n < 0) { printf("RESULT bad-table\n"); return 1; }

    static const char *tags[] = {"NAME", "TYPE", "HASH", "MDSZ", "OFFT", "VERS"};
    const char *drop = NULL;
    BOOL shift = NO, v22 = NO, known = strcmp(c, "as-is") == 0 || strcmp(c, "v2.2") == 0;
    v22 = strcmp(c, "v2.2") == 0;
    for (int i = 0; i < 6; i++) {
      char a[32], s[32], v[32];
      snprintf(a, sizeof a, "no-%s", tags[i]);
      snprintf(s, sizeof s, "no-%s-shift", tags[i]);
      snprintf(v, sizeof v, "no-%s-v2.2", tags[i]);
      if (!strcmp(c, a) || !strcmp(c, s) || !strcmp(c, v)) {
        drop = tags[i], shift = !strcmp(c, s), v22 = !strcmp(c, v), known = YES;
      }
    }
    if (!known) { printf("RESULT unknown-case\n"); return 2; }
    if (!strcmp(c, "as-is")) structure(b, len, e, n);
    NSData *bytes = (drop || v22) ? edited(b, len, drop, shift, v22) : file;

    id<MTLDevice> D = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> Q = [D newCommandQueue];
    dispatch_data_t data = dispatch_data_create(bytes.bytes, bytes.length, nil, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    NSError *err = nil;
    id<MTLLibrary> lib = [D newLibraryWithData:data error:&err];
    if (!lib) {
      printf("RESULT library=nil error=%s\n", err.localizedDescription.UTF8String);
      return 0;
    }
    printf("RESULT library=ok type=%ld functionNames=%lu table=%d\n", (long)lib.type,
           (unsigned long)lib.functionNames.count, n);
    id<MTLFunction> vert = nil, frag = nil;
    int made = 0;
    for (int i = 0; i < n; i++) {
      NSString *name = @(e[i].name);
      id<MTLFunction> f = [lib newFunctionWithName:name];
      printf("FUNCTION %d name=%s made=%s", i, e[i].name, f ? "yes" : "no");
      if (!f) { printf("\n"); continue; }
      made++;
      printf(" functionType=%ld\n", (long)f.functionType);
      if ([name isEqual:@"fill_a"]) run_kernel(D, Q, f, 3, seed);
      if ([name isEqual:@"fill_b"]) run_kernel(D, Q, f, 5, seed);
      if ([name isEqual:@"vert"]) vert = f;
      if ([name isEqual:@"frag"]) frag = f;
    }
    if (vert && frag) run_render(D, Q, vert, frag, seed);
    printf("RESULT made=%d of %d\n", made, n);
  }
  return 0;
}
