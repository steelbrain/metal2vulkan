// What a pipeline's creation answers for the functions of its linked `binaryFunctions`: one held
// there alone that the stage calls (by a direct call to its `[[visible]]` declaration, or through
// a visible-function table), one name held twice (by `functions` and `binaryFunctions`, or twice
// by one list; the same library function twice, or two functions of one name), and a function not
// compiled to a binary there or among the functions added to a made state; and which function a
// stage that is made calls.
//
// Measured on an Apple M3 Max under macOS 27.0.1 (26A434), one case a process, `MTL_DEBUG_LAYER`
// out of the environment and again set to 1, of libraries compiled from source and of metallibs;
// the runtime's `pipeline/additions.rs` holds the answers, and the runtime lane's
// `binary_functions_link_as_plain_metal_links_them` and
// `a_kernel_runs_the_functions_its_descriptor_links`, and the translator's
// `tests::engine_linking` and `tests::engine_tables`, test them.
//
// Build (no cargo target needed):
//   clang -fobjc-arc -O1 -fmodules -o binary_function_linking binary_function_linking.m \
//     -framework Foundation -framework Metal
//
// `binary_function_linking CASE` compiles each library from its source (`newLibraryWithSource:`);
// `binary_function_linking CASE DIR` reads each from `DIR/<name>.metallib` (`newLibraryWithURL:`),
// a library compiled ahead of time, as the runtime's libraries are. Make them with:
//   ./binary_function_linking --write-sources DIR
//   for n in caller a b c d; do
//     xcrun -sdk macosx metal -c DIR/$n.metal -o DIR/$n.air
//     xcrun -sdk macosx metallib DIR/$n.air -o DIR/$n.metallib
//   done
//
// The caller library declares the visible function `add_one(uint)` and defines none. Its kernel
// `direct` stores `add_one(t)`, `table` stores `ops[t % ops.size()](t)` through the table at
// buffer 1, and `none` stores `t`, over four threads; its vertex functions `vdirect`, `vtable`
// and `vnone` store the same by vertex ID (rasterization disabled, four points); its fragment
// functions `fdirect`, `ftable` and `fnone` write `add_one(7)`, `ops[0](7)` and 7 into a 1x1
// `R32Uint` target (vertex function `vcover`); its tile functions `tdirect` and `tnone` and its
// mesh functions `mdirect` and `mnone` (fragment function `fnone`) do as the kernels do. Library
// A defines `add_one(x) = x + 1` and `times_two(x) = x * 2`; library B defines
// `add_one(x) = x + 100`; library C defines `add_one(x) = helper(x) + 1` and declares the visible
// `helper`, which library D defines as `x * 10`. "bin" is a function made by
// `newFunctionWithDescriptor:` with `MTLFunctionOptionCompileToBinary`, "plain" one made by
// `newFunctionWithName:`. A table's entries are the handles of the shape's functions in the
// order listed below. Each case prints the creation (or its error's domain, code, description and
// `userInfo` keys) and the words stored (the buffer is filled with 0xAB first, read as `fill`);
// a tile case makes its state and runs nothing.
//
// A case is `<stage>-<shape>-<caller>`: the stage `c` a compute pipeline, `v` a render pipeline's
// vertex stage, `f` its fragment stage, `t` a tile pipeline, `m` a mesh pipeline's mesh stage;
// (`o` that mesh pipeline's object stage, object function `onone`, and `n` its fragment stage,
// fragment function `fnone`, each with mesh function `mnone`, the caller naming the mesh
// function); the caller `direct`, `table` or `none`; the shape what the stage links:
//
//   functions          functions [A.add_one plain]
//   binary-only        binaryFunctions [A.add_one bin]
//   not-precompiled    binaryFunctions [A.add_one plain]
//   np-both            functions and binaryFunctions [the one A.add_one plain]
//   np-other           functions [A.add_one plain], binaryFunctions [A.times_two plain]
//   both-same          functions [A.add_one plain], binaryFunctions [A.add_one bin]
//   both-other         functions [A.add_one plain], binaryFunctions [B.add_one bin]
//   both-swap          functions [B.add_one plain], binaryFunctions [A.add_one bin]
//   twice-bin          binaryFunctions [A.add_one bin, A.add_one bin] (made twice)
//   twice-other-bin    binaryFunctions [A.add_one bin, B.add_one bin]
//   twice-fn           functions [A.add_one plain, A.add_one plain] (made twice)
//   twice-other-fn     functions [A.add_one plain, B.add_one plain]
//   nested-functions   functions [C.add_one plain, D.helper plain]
//   nested-private     functions [C.add_one plain], privateFunctions [D.helper plain]
//   nested-binary      functions [C.add_one plain], binaryFunctions [D.helper bin]
//
// And the additions, each to a state of the `functions` shape (its stage's support flag on but
// where the case says): `c-add-bin-table` adds [A.times_two bin], and its table's entries are
// [add_one, times_two]; `c-add-other-name-table` adds [B.add_one bin], entries [A.add_one,
// B.add_one]; `c-add-not-precompiled-table`, `v-add-not-precompiled-direct`,
// `f-add-not-precompiled-direct` and `t-add-not-precompiled-direct` add [A.times_two plain] to
// the stage; `c-add-unsupported-not-precompiled-table` does so without the support flag; and
// `v-add-tile-not-precompiled-direct`, `v-add-fragment-not-precompiled-direct` and
// `t-add-vertex-not-precompiled-direct` add it to a stage the state lacks (a drawing state's tile
// stage and the fragment stage it has no function of, a tile state's vertex stage).
//
// `np-beside-bin` and `np-after-bin` link in `binaryFunctions` the function
// `newFunctionWithName:` makes of A's `add_one` while a function `newFunctionWithDescriptor:`
// made of it compiled to a binary lives, made after it or before it, and print whether the two
// are one object and each one's `options`.
//
// WHAT IT MEASURED. Source-compiled libraries and metallibs answer alike, case by case. The
// error's domain is `AGXMetalG15X_M1` and its `userInfo` holds `NSLocalizedDescription` alone.
//
//   *-binary-only-direct (c v f t)   nil, code 3 "error: unresolved visible function reference:
//                                    add_one" / "Reason: visible function not loaded", plain and
//                                    under validation; c-nested-binary-* alike, naming `helper`
//   *-binary-only-table, -none       made; the table calls it (1 2 3 4, word 8)
//   *-both-same, -both-other, -both-swap, *-twice-bin, *-twice-other-bin (c v f t)
//                                    made, plain; validation stops each ("All linked functions
//                                    must have unique names. Function 'add_one' is a duplicate
//                                    symbol in binaryFunctions, symbol first found in functions.")
//                                    A direct call calls the one of `functions` (both-other 1 2 3 4,
//                                    both-swap 100 101 102 103, f-both-swap 107); a table holds
//                                    each, each handle its own function (both-other-table
//                                    1 101 3 103, both-swap-table 100 2 102 4,
//                                    twice-other-bin-table 1 101 3 103); twice-other-bin-direct is
//                                    unresolved, code 3
//   c-twice-fn, c-twice-other-fn     nil, code 3 "error: duplicated symbol: add_one" /
//                                    "Defined by: linked-air-blob-2"; validation stops it
//   *-not-precompiled-table, -none, *-np-both, *-np-other
//                                    nil, code 2: c "Internal error during compute pipeline
//                                    creation", t "Internal error during tile pipeline creation",
//                                    v f n an empty description; validation stops each ("Function
//                                    'add_one' was found in binaryFunctions array but is not
//                                    precompiled.")
//   *-not-precompiled-direct         nil, code 3, unresolved, as binary-only-direct, plain;
//                                    validation stops it as not precompiled
//   c-add-bin-table                  made, 1 2 3 6; c-add-other-name-table made (validation too),
//                                    1 101 3 103
//   c-add-not-precompiled-table, c-add-unsupported-not-precompiled-table, v-, f-, t-add-not-
//   precompiled-direct               nil, code 1 "Failed to allocate a new pipeline"; validation
//                                    stops each (not precompiled, or the support flag)
//   v-add-tile-, v-add-fragment-, t-add-vertex-not-precompiled-direct
//                                    made, plain; validation stops the support flag
//   m-*, o-binary-only-none          plain Metal's process ends at the creation: "Mesh Render
//                                    Pipeline Descriptor Validation / Binary linked functions are
//                                    not supported for object and mesh shaders"; n-binary-only-none
//                                    made
//   np-beside-bin, np-after-bin      "plain is the binary function; options plain 1 binary 1":
//                                    while either lives, `newFunctionWithName:` and
//                                    `newFunctionWithDescriptor:` with
//                                    `MTLFunctionOptionCompileToBinary` of one library function
//                                    answer one object, compiled to a binary, so both-same,
//                                    twice-bin and twice-fn list one object twice; the table case
//                                    is made, the direct one unresolved, as binary-only's.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>

static const char *CALLER =
    "#include <metal_stdlib>\nusing namespace metal;\n"
    "[[visible]] uint add_one(uint x);\n"
    "kernel void direct(device uint *out [[buffer(0)]], uint t [[thread_position_in_grid]]) {\n"
    "  out[t] = add_one(t);\n"
    "}\n"
    "kernel void table(device uint *out [[buffer(0)]],\n"
    "                  visible_function_table<uint(uint)> ops [[buffer(1)]],\n"
    "                  uint t [[thread_position_in_grid]]) {\n"
    "  out[t] = ops[t % ops.size()](t);\n"
    "}\n"
    "kernel void none(device uint *out [[buffer(0)]], uint t [[thread_position_in_grid]]) {\n"
    "  out[t] = t;\n"
    "}\n"
    "vertex void vdirect(device uint *out [[buffer(0)]], uint v [[vertex_id]]) {\n"
    "  out[v] = add_one(v);\n"
    "}\n"
    "vertex void vtable(device uint *out [[buffer(0)]],\n"
    "                   visible_function_table<uint(uint)> ops [[buffer(1)]],\n"
    "                   uint v [[vertex_id]]) {\n"
    "  out[v] = ops[v % ops.size()](v);\n"
    "}\n"
    "vertex void vnone(device uint *out [[buffer(0)]], uint v [[vertex_id]]) { out[v] = v; }\n"
    "struct Cover { float4 p [[position]]; };\n"
    "vertex Cover vcover(uint v [[vertex_id]]) {\n"
    "  float2 xy = float2(v == 1 ? 3.0f : -1.0f, v == 2 ? 3.0f : -1.0f);\n"
    "  Cover c; c.p = float4(xy, 0.0f, 1.0f); return c;\n"
    "}\n"
    "fragment uint4 fdirect() { return uint4(add_one(7u), 0, 0, 0); }\n"
    "fragment uint4 ftable(visible_function_table<uint(uint)> ops [[buffer(1)]]) {\n"
    "  return uint4(ops[0](7u), 0, 0, 0);\n"
    "}\n"
    "fragment uint4 fnone() { return uint4(7u, 0, 0, 0); }\n"
    "kernel void tdirect(device uint *out [[buffer(0)]], ushort2 t [[thread_position_in_threadgroup]]) {\n"
    "  if (t.x == 0 && t.y == 0) out[0] = add_one(0u);\n"
    "}\n"
    "kernel void tnone(ushort2 t [[thread_position_in_threadgroup]]) {}\n"
    "struct MeshVertex { float4 p [[position]]; };\n"
    "using Mesh1 = metal::mesh<MeshVertex, void, 3, 1, metal::topology::triangle>;\n"
    "[[mesh]] void mdirect(Mesh1 m, device uint *out [[buffer(0)]],\n"
    "                      uint t [[thread_index_in_threadgroup]]) {\n"
    "  out[t] = add_one(t);\n"
    "  if (t == 0) m.set_primitive_count(0);\n"
    "}\n"
    "[[mesh]] void mnone(Mesh1 m, device uint *out [[buffer(0)]],\n"
    "                    uint t [[thread_index_in_threadgroup]]) {\n"
    "  out[t] = t;\n"
    "  if (t == 0) m.set_primitive_count(0);\n"
    "}\n"
    "[[object]] void onone(mesh_grid_properties grid) {\n"
    "  grid.set_threadgroups_per_grid(uint3(1, 1, 1));\n"
    "}\n";
static const char *LIB_A =
    "#include <metal_stdlib>\nusing namespace metal;\n"
    "[[visible]] uint add_one(uint x) { return x + 1; }\n"
    "[[visible]] uint times_two(uint x) { return x * 2; }\n";
static const char *LIB_B =
    "#include <metal_stdlib>\nusing namespace metal;\n"
    "[[visible]] uint add_one(uint x) { return x + 100; }\n";
static const char *LIB_C =
    "#include <metal_stdlib>\nusing namespace metal;\n"
    "[[visible]] uint helper(uint x);\n"
    "[[visible]] uint add_one(uint x) { return helper(x) + 1; }\n";
static const char *LIB_D =
    "#include <metal_stdlib>\nusing namespace metal;\n"
    "[[visible]] uint helper(uint x) { return x * 10; }\n";

static id<MTLDevice> D;
static id<MTLLibrary> A, B, C, LD, CALLER_LIB;

static id<MTLFunction> plain(id<MTLLibrary> lib, NSString *name) {
  return [lib newFunctionWithName:name];
}

static id<MTLFunction> bin(id<MTLLibrary> lib, NSString *name) {
  MTLFunctionDescriptor *d = [MTLFunctionDescriptor functionDescriptor];
  d.name = name;
  d.options = MTLFunctionOptionCompileToBinary;
  NSError *e = nil;
  id<MTLFunction> f = [lib newFunctionWithDescriptor:d error:&e];
  if (!f) printf("RESULT function error %s\n", e.localizedDescription.UTF8String);
  return f;
}

static void print_error(const char *what, NSError *e) {
  printf("RESULT %s error domain=%s code=%ld description=\"%s\" keys=", what,
         e.domain.UTF8String, (long)e.code, e.localizedDescription.UTF8String);
  for (NSString *k in e.userInfo) printf("%s,", k.UTF8String);
  printf("\n");
  fflush(stdout);
}

static void print_words(id<MTLCommandBuffer> cb, id<MTLBuffer> out, int n) {
  printf("RESULT ran status=%ld words", (long)cb.status);
  for (int i = 0; i < n; i++) {
    uint32_t w = ((uint32_t *)out.contents)[i];
    if (w == 0xABABABABu) printf(" fill"); else printf(" %u", w);
  }
  printf("\n");
}

// The case's parts: its stage letter, its shape and its caller.
static char stage_letter;
static char shape[64];
static const char *caller;

static int parse(const char *c) {
  const char *first = strchr(c, '-'), *last = strrchr(c, '-');
  if (!first || first == last || (size_t)(last - first - 1) >= sizeof shape) return 0;
  stage_letter = c[0];
  memcpy(shape, first + 1, last - first - 1);
  shape[last - first - 1] = 0;
  caller = last + 1;
  return 1;
}

static int is(const char *s) { return strcmp(shape, s) == 0; }

// The linked functions the shape says, and the table entries' functions, in order. An addition's
// shape links the `functions` shape.
static MTLLinkedFunctions *linked_for(NSMutableArray<id<MTLFunction>> *entries) {
  MTLLinkedFunctions *lf = [MTLLinkedFunctions linkedFunctions];
  NSArray<id<MTLFunction>> *functions = @[], *binaries = @[], *privates = @[];
  if (is("functions") || strncmp(shape, "add-", 4) == 0) {
    functions = @[ plain(A, @"add_one") ];
  } else if (is("binary-only")) {
    binaries = @[ bin(A, @"add_one") ];
  } else if (is("not-precompiled")) {
    binaries = @[ plain(A, @"add_one") ];
  } else if (is("np-beside-bin") || is("np-after-bin")) {
    // The plain function made beside a function compiled to a binary of the same library
    // function, which the case keeps alive: made before it, or after it.
    static id<MTLFunction> made_plain, made_bin;
    if (is("np-beside-bin")) {
      made_plain = plain(A, @"add_one");
      made_bin = bin(A, @"add_one");
    } else {
      made_bin = bin(A, @"add_one");
      made_plain = plain(A, @"add_one");
    }
    printf("RESULT plain %s the binary function; options plain %lu binary %lu\n",
           made_plain == made_bin ? "is" : "is not", (unsigned long)made_plain.options,
           (unsigned long)made_bin.options);
    binaries = @[ made_plain ];
  } else if (is("np-both")) {
    id<MTLFunction> a = plain(A, @"add_one");
    functions = @[ a ];
    binaries = @[ a ];
  } else if (is("np-other")) {
    functions = @[ plain(A, @"add_one") ];
    binaries = @[ plain(A, @"times_two") ];
  } else if (is("both-same")) {
    functions = @[ plain(A, @"add_one") ];
    binaries = @[ bin(A, @"add_one") ];
  } else if (is("both-other")) {
    functions = @[ plain(A, @"add_one") ];
    binaries = @[ bin(B, @"add_one") ];
  } else if (is("both-swap")) {
    functions = @[ plain(B, @"add_one") ];
    binaries = @[ bin(A, @"add_one") ];
  } else if (is("twice-bin")) {
    binaries = @[ bin(A, @"add_one"), bin(A, @"add_one") ];
  } else if (is("twice-other-bin")) {
    binaries = @[ bin(A, @"add_one"), bin(B, @"add_one") ];
  } else if (is("twice-fn")) {
    functions = @[ plain(A, @"add_one"), plain(A, @"add_one") ];
  } else if (is("twice-other-fn")) {
    functions = @[ plain(A, @"add_one"), plain(B, @"add_one") ];
  } else if (is("nested-functions")) {
    functions = @[ plain(C, @"add_one"), plain(LD, @"helper") ];
  } else if (is("nested-private")) {
    functions = @[ plain(C, @"add_one") ];
    privates = @[ plain(LD, @"helper") ];
  } else if (is("nested-binary")) {
    functions = @[ plain(C, @"add_one") ];
    binaries = @[ bin(LD, @"helper") ];
  } else {
    printf("RESULT unknown shape %s\n", shape);
    exit(2);
  }
  // The entries: the callable functions, `functions` then `binaryFunctions`, the one object
  // listed in both once.
  for (id<MTLFunction> f in functions) [entries addObject:f];
  for (id<MTLFunction> f in binaries)
    if (![entries containsObject:f]) [entries addObject:f];
  lf.functions = functions;
  lf.binaryFunctions = binaries;
  lf.privateFunctions = privates;
  return lf;
}

// The functions an addition adds.
static NSArray<id<MTLFunction>> *added_functions(void) {
  if (is("add-bin")) return @[ bin(A, @"times_two") ];
  if (is("add-other-name")) return @[ bin(B, @"add_one") ];
  return @[ plain(A, @"times_two") ];
}

static NSString *named(NSString *direct, NSString *table, NSString *none) {
  if (strcmp(caller, "table") == 0) return table;
  if (strcmp(caller, "none") == 0) return none;
  return direct;
}

static void fill_table(id<MTLVisibleFunctionTable> vft, NSArray<id<MTLFunctionHandle>> *handles) {
  for (NSUInteger i = 0; i < handles.count; i++) [vft setFunction:handles[i] atIndex:i];
}

static int compute(void) {
  NSMutableArray<id<MTLFunction>> *entries = [NSMutableArray array];
  int adds = strncmp(shape, "add-", 4) == 0;
  MTLComputePipelineDescriptor *pd = [MTLComputePipelineDescriptor new];
  pd.computeFunction = [CALLER_LIB newFunctionWithName:named(@"direct", @"table", @"none")];
  pd.linkedFunctions = linked_for(entries);
  pd.supportAddingBinaryFunctions = adds && !strstr(shape, "unsupported");
  NSError *e = nil;
  printf("RESULT creating\n");
  fflush(stdout);
  id<MTLComputePipelineState> ps = [D newComputePipelineStateWithDescriptor:pd options:0
                                                                 reflection:nil error:&e];
  if (!ps) { print_error("pipeline", e); return 0; }
  printf("RESULT pipeline made\n");
  if (adds) {
    NSArray *added = added_functions();
    printf("RESULT adding\n");
    fflush(stdout);
    id<MTLComputePipelineState> more =
        [ps newComputePipelineStateWithAdditionalBinaryFunctions:added error:&e];
    if (!more) { print_error("added", e); return 0; }
    printf("RESULT added made\n");
    ps = more;
    [entries addObjectsFromArray:added];
  }
  NSMutableArray<id<MTLFunctionHandle>> *handles = [NSMutableArray array];
  for (id<MTLFunction> f in entries) {
    id<MTLFunctionHandle> h = [ps functionHandleWithFunction:f];
    printf("RESULT handle %s %s\n", f.name.UTF8String, h ? "made" : "nil");
    if (h) [handles addObject:h];
  }
  id<MTLVisibleFunctionTable> vft = nil;
  if (strcmp(caller, "table") == 0) {
    MTLVisibleFunctionTableDescriptor *td = [MTLVisibleFunctionTableDescriptor new];
    td.functionCount = handles.count;
    vft = [ps newVisibleFunctionTableWithDescriptor:td];
    fill_table(vft, handles);
  }
  id<MTLBuffer> out = [D newBufferWithLength:16 options:MTLResourceStorageModeShared];
  memset(out.contents, 0xAB, 16);
  id<MTLCommandQueue> q = [D newCommandQueue];
  id<MTLCommandBuffer> cb = [q commandBuffer];
  id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
  [ce setComputePipelineState:ps];
  [ce setBuffer:out offset:0 atIndex:0];
  if (vft) [ce setVisibleFunctionTable:vft atBufferIndex:1];
  [ce dispatchThreads:MTLSizeMake(4, 1, 1) threadsPerThreadgroup:MTLSizeMake(4, 1, 1)];
  [ce endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  print_words(cb, out, 4);
  return 0;
}

static id<MTLTexture> target(void) {
  MTLTextureDescriptor *t = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint
                                                                               width:1 height:1
                                                                           mipmapped:NO];
  t.usage = MTLTextureUsageRenderTarget;
  t.storageMode = MTLStorageModeShared;
  id<MTLTexture> made = [D newTextureWithDescriptor:t];
  uint32_t fill = 0xABABABABu;
  [made replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:&fill bytesPerRow:4];
  return made;
}

static MTLRenderPassDescriptor *pass(id<MTLTexture> texture) {
  MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
  rp.colorAttachments[0].texture = texture;
  rp.colorAttachments[0].loadAction = MTLLoadActionLoad;
  rp.colorAttachments[0].storeAction = MTLStoreActionStore;
  return rp;
}

static int render(void) {
  NSMutableArray<id<MTLFunction>> *entries = [NSMutableArray array];
  int vertex = stage_letter == 'v', adds = strncmp(shape, "add-", 4) == 0;
  MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
  pd.colorAttachments[0].pixelFormat = MTLPixelFormatR32Uint;
  MTLLinkedFunctions *lf = linked_for(entries);
  if (vertex) {
    pd.vertexFunction = [CALLER_LIB newFunctionWithName:named(@"vdirect", @"vtable", @"vnone")];
    pd.vertexLinkedFunctions = lf;
    pd.rasterizationEnabled = NO;
    pd.supportAddingVertexBinaryFunctions = adds;
  } else {
    pd.vertexFunction = [CALLER_LIB newFunctionWithName:@"vcover"];
    pd.fragmentFunction = [CALLER_LIB newFunctionWithName:named(@"fdirect", @"ftable", @"fnone")];
    pd.fragmentLinkedFunctions = lf;
    pd.supportAddingFragmentBinaryFunctions = adds;
  }
  NSError *e = nil;
  printf("RESULT creating\n");
  fflush(stdout);
  id<MTLRenderPipelineState> ps = [D newRenderPipelineStateWithDescriptor:pd error:&e];
  if (!ps) { print_error("pipeline", e); return 0; }
  printf("RESULT pipeline made\n");
  MTLRenderStages stage = vertex ? MTLRenderStageVertex : MTLRenderStageFragment;
  if (adds) {
    MTLRenderPipelineFunctionsDescriptor *fd = [MTLRenderPipelineFunctionsDescriptor new];
    if (is("add-tile-not-precompiled")) fd.tileAdditionalBinaryFunctions = added_functions();
    else if (is("add-fragment-not-precompiled")) fd.fragmentAdditionalBinaryFunctions = added_functions();
    else if (vertex) fd.vertexAdditionalBinaryFunctions = added_functions();
    else fd.fragmentAdditionalBinaryFunctions = added_functions();
    printf("RESULT adding\n");
    fflush(stdout);
    id<MTLRenderPipelineState> more =
        [ps newRenderPipelineStateWithAdditionalBinaryFunctions:fd error:&e];
    if (!more) { print_error("added", e); return 0; }
    printf("RESULT added made\n");
    ps = more;
  }
  NSMutableArray<id<MTLFunctionHandle>> *handles = [NSMutableArray array];
  for (id<MTLFunction> f in entries) {
    id<MTLFunctionHandle> h = [ps functionHandleWithFunction:f stage:stage];
    printf("RESULT handle %s %s\n", f.name.UTF8String, h ? "made" : "nil");
    if (h) [handles addObject:h];
  }
  id<MTLVisibleFunctionTable> vft = nil;
  if (strcmp(caller, "table") == 0) {
    MTLVisibleFunctionTableDescriptor *td = [MTLVisibleFunctionTableDescriptor new];
    td.functionCount = handles.count;
    vft = [ps newVisibleFunctionTableWithDescriptor:td stage:stage];
    fill_table(vft, handles);
  }
  id<MTLTexture> texture = target();
  id<MTLBuffer> out = [D newBufferWithLength:16 options:MTLResourceStorageModeShared];
  memset(out.contents, 0xAB, 16);
  id<MTLCommandQueue> q = [D newCommandQueue];
  id<MTLCommandBuffer> cb = [q commandBuffer];
  id<MTLRenderCommandEncoder> re = [cb renderCommandEncoderWithDescriptor:pass(texture)];
  [re setRenderPipelineState:ps];
  if (vertex) {
    [re setVertexBuffer:out offset:0 atIndex:0];
    if (vft) [re setVertexVisibleFunctionTable:vft atBufferIndex:1];
    [re drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:4];
  } else {
    if (vft) [re setFragmentVisibleFunctionTable:vft atBufferIndex:1];
    [re drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
  }
  [re endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  if (vertex) {
    print_words(cb, out, 4);
  } else {
    uint32_t word = 0;
    [texture getBytes:&word bytesPerRow:4 fromRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0];
    printf("RESULT ran status=%ld word %s%u\n", (long)cb.status,
           word == 0xABABABABu ? "fill " : "", word);
  }
  return 0;
}

static int tile(void) {
  NSMutableArray<id<MTLFunction>> *entries = [NSMutableArray array];
  int adds = strncmp(shape, "add-", 4) == 0;
  MTLTileRenderPipelineDescriptor *pd = [MTLTileRenderPipelineDescriptor new];
  pd.tileFunction = [CALLER_LIB newFunctionWithName:named(@"tdirect", @"tdirect", @"tnone")];
  pd.colorAttachments[0].pixelFormat = MTLPixelFormatR32Uint;
  pd.threadgroupSizeMatchesTileSize = YES;
  pd.linkedFunctions = linked_for(entries);
  pd.supportAddingBinaryFunctions = adds;
  NSError *e = nil;
  printf("RESULT creating\n");
  fflush(stdout);
  id<MTLRenderPipelineState> ps = [D newRenderPipelineStateWithTileDescriptor:pd options:0
                                                                   reflection:nil error:&e];
  if (!ps) { print_error("pipeline", e); return 0; }
  printf("RESULT pipeline made\n");
  if (adds) {
    MTLRenderPipelineFunctionsDescriptor *fd = [MTLRenderPipelineFunctionsDescriptor new];
    if (is("add-vertex-not-precompiled")) fd.vertexAdditionalBinaryFunctions = added_functions();
    else fd.tileAdditionalBinaryFunctions = added_functions();
    printf("RESULT adding\n");
    fflush(stdout);
    id<MTLRenderPipelineState> more =
        [ps newRenderPipelineStateWithAdditionalBinaryFunctions:fd error:&e];
    if (!more) { print_error("added", e); return 0; }
    printf("RESULT added made\n");
  }
  return 0;
}

static int mesh(void) {
  NSMutableArray<id<MTLFunction>> *entries = [NSMutableArray array];
  MTLMeshRenderPipelineDescriptor *pd = [MTLMeshRenderPipelineDescriptor new];
  pd.meshFunction = [CALLER_LIB newFunctionWithName:named(@"mdirect", @"mdirect", @"mnone")];
  if (stage_letter != 'm') pd.meshFunction = [CALLER_LIB newFunctionWithName:@"mnone"];
  pd.fragmentFunction = [CALLER_LIB newFunctionWithName:@"fnone"];
  pd.colorAttachments[0].pixelFormat = MTLPixelFormatR32Uint;
  MTLLinkedFunctions *lf = linked_for(entries);
  if (stage_letter == 'o') {
    pd.objectFunction = [CALLER_LIB newFunctionWithName:@"onone"];
    pd.objectLinkedFunctions = lf;
  } else if (stage_letter == 'n') {
    pd.fragmentLinkedFunctions = lf;
  } else {
    pd.meshLinkedFunctions = lf;
  }
  NSError *e = nil;
  printf("RESULT creating\n");
  fflush(stdout);
  id<MTLRenderPipelineState> ps = [D newRenderPipelineStateWithMeshDescriptor:pd options:0
                                                                   reflection:nil error:&e];
  if (!ps) { print_error("pipeline", e); return 0; }
  printf("RESULT pipeline made\n");
  id<MTLTexture> texture = target();
  id<MTLBuffer> out = [D newBufferWithLength:16 options:MTLResourceStorageModeShared];
  memset(out.contents, 0xAB, 16);
  id<MTLCommandQueue> q = [D newCommandQueue];
  id<MTLCommandBuffer> cb = [q commandBuffer];
  id<MTLRenderCommandEncoder> re = [cb renderCommandEncoderWithDescriptor:pass(texture)];
  [re setRenderPipelineState:ps];
  [re setMeshBuffer:out offset:0 atIndex:0];
  [re drawMeshThreadgroups:MTLSizeMake(1, 1, 1) threadsPerObjectThreadgroup:MTLSizeMake(1, 1, 1)
      threadsPerMeshThreadgroup:MTLSizeMake(4, 1, 1)];
  [re endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  print_words(cb, out, 4);
  return 0;
}

// The libraries' sources, by the file name each is written to and its metallib read from.
static const struct { const char *name; const char **source; } SOURCES[] = {
    {"caller", &CALLER}, {"a", &LIB_A}, {"b", &LIB_B}, {"c", &LIB_C}, {"d", &LIB_D},
};

// The metallibs' directory, or NULL to compile each library from its source.
static const char *metallibs;

// The library `what`: compiled from `source` by `newLibraryWithSource:`, or, given a metallib
// directory, read from `<dir>/<what>.metallib` by `newLibraryWithURL:`.
static id<MTLLibrary> library(const char *source, const char *what) {
  NSError *e = nil;
  id<MTLLibrary> lib;
  if (metallibs) {
    NSString *path = [NSString stringWithFormat:@"%s/%s.metallib", metallibs, what];
    lib = [D newLibraryWithURL:[NSURL fileURLWithPath:path] error:&e];
  } else {
    lib = [D newLibraryWithSource:[NSString stringWithUTF8String:source] options:nil error:&e];
  }
  if (!lib) {
    printf("RESULT library %s error %s\n", what, e.localizedDescription.UTF8String);
    exit(1);
  }
  return lib;
}

int main(int argc, char **argv) {
  if (argc == 3 && strcmp(argv[1], "--write-sources") == 0) {
    for (size_t i = 0; i < sizeof SOURCES / sizeof SOURCES[0]; i++) {
      char path[4096];
      snprintf(path, sizeof path, "%s/%s.metal", argv[2], SOURCES[i].name);
      FILE *f = fopen(path, "w");
      if (!f) return 1;
      fputs(*SOURCES[i].source, f);
      fclose(f);
    }
    return 0;
  }
  if (argc < 2 || !parse(argv[1])) return 2;
  metallibs = argc > 2 ? argv[2] : NULL;
  @autoreleasepool {
    printf("RESULT case=%s debug_layer=%s libraries=%s\n", argv[1],
           getenv("MTL_DEBUG_LAYER") ? getenv("MTL_DEBUG_LAYER") : "unset",
           metallibs ? "metallib" : "source");
    D = MTLCreateSystemDefaultDevice();
    CALLER_LIB = library(CALLER, "caller");
    A = library(LIB_A, "a");
    B = library(LIB_B, "b");
    C = library(LIB_C, "c");
    LD = library(LIB_D, "d");
    switch (stage_letter) {
      case 'c': return compute();
      case 'v':
      case 'f': return render();
      case 't': return tile();
      case 'm':
      case 'o':
      case 'n': return mesh();
      default: return 2;
    }
  }
}
