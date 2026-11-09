# metal2vulkan-validation

The repository's validation tooling (not published): authored cases run on Metal and on Vulkan and
compared, and bounded audits over a private AIR corpus. [docs/VALIDATION.md](../../docs/VALIDATION.md)
says which tool proves which claim; this README describes the crate and its case model.

## The case model

A case is a checked manifest. Every GPU resource, initial byte, function constant, dispatch or draw
parameter, output region and comparison rule comes from it. Runners do not infer, seed, resize,
repair or reinterpret cases. Kernel, fragment and vertex cases share one literal-resource
preparation contract across the checker, the Metal oracle and the Vulkan candidate: buffers,
textures, explicit and reflected static samplers, function-constant bit patterns,
acceleration-structure shadows and render targets. Texture and render-target outputs are compared
as tightly packed selected regions, not backend-specific row-pitched storage.

## Commands

| Command | What it does |
|---|---|
| `corpus-harvest` | Extract and sanitize AIR from macOS metallibs into deterministic private shards |
| `corpus-index` | Sync (incrementally) or check the disposable SQLite index; `--rebuild` for recovery |
| `corpus-case-check` | Validate or install a manifest; `--recheck-all` re-checks every installed case |
| `corpus-status` | Report the missing-evidence queue; `--c3` gives each case's verdict on every venue |
| `corpus-triage` | Run the structural, capability and bounded translation audits over indexed rows |
| `corpus-pipeline` | Ask a Vulkan driver to create each translated pipeline, without dispatching |
| `corpus-metal` | Qualify cases on Metal (`--device-type gpu`) |
| `corpus-vulkan` | Run cases on a Vulkan device against their Metal observations (`--device-type cpu\|gpu`) |
| `corpus-moltenvk` | The same through MoltenVK (macOS) |
| `corpus-draw-ab` | Draw logged fragment functions through the runtime and compare two builds |
| `command-replay` | Run a recorded stretch of Metal commands through the runtime |
| `format-scout` | Drive every Metal pixel format through the runtime, one use at a time |

Each command's `--help` lists its options.

## Running the tests

```sh
cargo nextest run -p metal2vulkan-validation --all-features
```

That run opens no device. Tests that execute on a device compile only under `--cfg m2v_cpu_icd` (a
CPU Vulkan ICD such as lavapipe) or `--cfg m2v_gpu` (a real GPU); see
[docs/VALIDATION.md](../../docs/VALIDATION.md#device-tests-and-gpu-safety). MoltenVK execution is
macOS-only; native Vulkan execution is Linux-only.

## The corpus

[corpus/README.md](corpus/README.md) describes the formats. In short:

- Harvest merges new rows only into their own shards and publishes their locations to the index;
  a following index sync reads no source bytes. Library modules are kept apart, with every parent
  library they came from, so linked functions resolve exactly.
- Candidate observations record the product's source fingerprint; `corpus-status` compares it
  instead of translating again, so any translator change makes older candidate evidence stale
  until it is rerun.
- Editing a manifest replaces its `(air_sha256, name)` slot; only the old identity's observations
  are removed.

```sh
cargo run -p metal2vulkan-validation --bin corpus-index
# Read an indexed AIR function and write one manifest for it.
cargo run -p metal2vulkan-validation --bin corpus-case-check -- --manifest /path/to/case.json --install
cargo run -p metal2vulkan-validation --bin corpus-index
cargo run -p metal2vulkan-validation --bin corpus-status
```

## Case features

Acceleration structures are literal resources too. Instance entries carry host-defined child
references for introspection. Primitive entries carry tightly packed little-endian triangle
vertices; those exact bytes build Metal's native BLAS and the Vulkan geometry shadow.

Kernel, fragment, and vertex cases share one literal-resource preparation contract across the checker,
Metal oracle, and Vulkan candidate. It covers buffers, textures, explicit and reflected static
samplers, function-constant bit patterns, acceleration-structure shadows, and render targets;
texture and render-target outputs are compared as tightly packed selected regions rather than
backend-specific row-pitched storage.

Fragment cases use their authored draw and render-target records directly. The Metal oracle and
Vulkan candidate each synthesize the same fullscreen vertex interface from the fragment AIR
varyings, then execute a real render pipeline. Vulkan hashes both SPIR-V stages into the candidate
dependency, so changes to the generated companion invalidate evidence instead of being hidden.

Vertex cases author ordinary per-vertex attributes plus a `vertex_observation`. The generated
fragment observer routes either rasterized position or one typed user varying into render target
zero, padding scalars and short vectors to a four-component float, signed-integer, or
unsigned-integer attachment. Metal and Vulkan consume the same draw and attribute bytes; Vulkan
uses a negative-height viewport for the same screen-space orientation as Metal. The Metal pipeline
also derives its required input topology class from the authored draw primitive, including for
layered rendering. When vertex AIR writes or fragment AIR consumes a render-target array index, the
Metal oracle uses one-layer array attachments and explicitly activates that layer in the render
pass.

A vertex entry with no raster position may instead omit `vertex_observation` and all attachments,
then select a writable shader resource as its output. Both executors run that draw with
rasterization disabled; Vulkan emits no fragment companion, while Metal uses a private 1x1 encoder
sink that cannot participate in the authored observation. This covers void vertex functions whose
contract is a buffer or storage-texture side effect rather than raster output.

Post-tessellation vertex cases replace `draw` with a `tessellation` record: exact binary16 edge
and inner factors per patch, instance/amplification counts, and typed control-point/per-patch
records. Metal consumes those records through native per-patch vertex stepping and `drawPatches`;
Vulkan generates the matching vertex and tessellation-control stages, including reflected 16-bit
system-value types. All generated SPIR-V modules participate in the candidate dependency hash.

Object and mesh cases replace `dispatch` with a `mesh_draw` record: the grid of
`drawMeshThreadgroups:` (`{"threadgroups": [x, y, z]}`) or `drawMeshThreads:` (`{"threads": [...]}`),
the threads per object and per mesh threadgroup (at most 1024 each, Metal's limit), and
`rasterization` (absent is on). The case's entry is its own stage's function, and every resource
it authors is that function's. The pipeline's other stages are partners named from the same AIR
module (`object`, `mesh`, `fragment`). Each partner binds no resource and is loop-free, and the
stages link as Metal links them: payload sizes, and fragment inputs by user key. A rasterizing
draw must have a fragment function, and the checker refuses one that does not: Metal's
descriptor validation aborts on such a pipeline. The draw names its fragment partner, or a mesh
case states an `observation` in its place (`{"kind": "position"}`,
`{"kind": "vertex_data", "index": n}` or `{"kind": "primitive_data", "index": n}`, counting the
struct's user data fields in declaration order). Both executors then draw it with a generated
fragment function that writes that output to render target 0, as a vertex case's observer does;
a primitive data field is read flat. An object case may state `{"kind": "launch"}` in place of
both its mesh and fragment partners. A generated mesh function then draws, for each mesh
threadgroup the object function launches, one row of an `RGBA32Uint` render target 0. The row is
the threadgroup's linear index in the launched grid. Pixel 0 holds the grid and the row
(`(X, Y, Z, row)`), and each later pixel 16 bytes of the payload, read flat at its declared size;
a word past that size reads 0. Every other pixel keeps the target's initial bytes, so a write
past the payload or the grid shows. The draw must launch one object threadgroup, because a mesh
threadgroup cannot tell which object threadgroup launched it, and the payload must be whole words
of at most 1344 bytes. A mesh case may state `object_launch` (`{"payload_b64": ...,
"mesh_threadgroups": [x, y, z]}`) in place of its object partner. A generated object function
then writes those payload bytes from thread 0 of each object threadgroup and launches that grid
of mesh threadgroups (at most 1024), so every object threadgroup launches the same work. The
payload is whole words, at least as many bytes as the mesh function reads, and at most Metal's
16 KB. Without rasterization the case selects a buffer its entry writes. The payload length is
the pipeline's default. A stage the case leaves absent and nothing generates is a requirement
(`mesh_companion`). An object row whose payload the launch observer reads, and a mesh row whose
payload a generated object function writes, need none. Nor does a mesh row whose function sets no
vertex or primitive and writes no device memory: it has no effect a stage observes, so it draws
without rasterization and its case observes the empty output (`"output": {"kind": "none"}`). The Metal oracle draws a case through
`MTLMeshRenderPipelineDescriptor` and the case's own stage's setters. The engine's candidate
draws it through the runtime's mesh render pipeline state and `draw_mesh_threadgroups` or
`draw_mesh_threads`. On a device without `VK_EXT_mesh_shader` the runtime emulates the draw and
announces `emulated-mesh-draws`. An object case's `threadgroup_memory` sets its arguments'
lengths: the oracle's through `setObjectThreadgroupMemoryLength:atIndex:`, the candidate's
through the runtime's `set_object_threadgroup_memory_length`.

Dynamic Metal `threadgroup` arguments are declared through `threadgroup_memory` binding/length
pairs. Metal binds that exact allocation; Vulkan uses the translator's descriptor-free `Workgroup`
storage. The schema intentionally has no initial bytes because both APIs leave this storage
uninitialized—authored shaders must initialize every location they read.

Compute imageblocks use explicit `imageblock.dimensions`; explicit layouts have no host bytes. The
checker requires those dimensions to match the dispatch threadgroup x/y size. Implicit layouts
reuse authored `render_targets` as their initial attachment values and additionally require
`imageblock.implicit_coverage: full_single_sample`. Both runners therefore begin with the same
fully covered, single-sample pixels: Metal encodes a color-write-disabled coverage draw before its
kernel tile dispatch, while Vulkan initializes the reflected attachment image directly. This is an
authored execution fact, not an executor-supplied default.

Custom fragment imageblocks use the distinct `fragment_imageblock` resource. Each accessed master
member is identified by its AIR `user(...)` semantic and carries an exact tightly packed `half`,
`half4`, `uchar4`, or `ushort` plane; scalar `half` is the backward-compatible default format.
`output.kind: fragment_imageblock` reads a selected plane region. The Metal runner uses same-pass
initializer/resolver fragments, while Vulkan binds the matching reflected storage format under
ordered fragment interlock.

Textures nested in an `air.indirect_buffer` are authored through
`argument_buffer_textures`, keyed by owning buffer binding plus field byte offset. Product
reflection supplies the Metal argument-encoder `[[id(n)]]` and the synthetic Vulkan descriptor,
so manifests do not encode backend-specific binding numbers. A fixed embedded array authors one
texture per handle at `field_offset + 8 * element`; Metal binds consecutive argument IDs and Vulkan
binds those same elements into the single reflected descriptor array.

A top-level `array_ref<void>` uses `device_buffer_arrays`: its literal base binding and extent own a
sorted sparse list of separately allocated device-buffer elements. Unauthored slots are null, and
`output.kind: device_buffer_array_element` selects one writable element for comparison. Metal
flattens elements to consecutive native buffer indices; Vulkan fills an address-table allocation
with the corresponding device addresses.

Top-level `array<texture..., N>` and runtime `array_ref<texture...>` arguments use
`texture_arrays`. Element order is the AIR handle index. Fixed arrays must author exactly `N`
elements; runtime arrays author their valid logical prefix. Vulkan binds that prefix into the
reflected 128-descriptor array and fills the unreachable remainder with the final valid descriptor;
Metal binds the same prefix to consecutive texture slots.

When function-constant alternatives declare a layered texture and a fixed texture array at the
same Metal base slot, the selected fixed-array case sets `overrides_texture_at_base: true` on the
array. The checker accepts that alias only for matching input-only array/non-array shape families,
formats, and sample counts with an authored function constant; all other slot overlaps remain
invalid. Metal then binds the declared fixed array last, while Vulkan keeps the alternatives in
their reflected descriptor locations.

Texture `dimensions` describe spatial extent and array layers. Multisample textures additionally
carry an explicit `sample_count`; literal bytes are tightly packed by layer, row, texel, then sample.
This keeps 2D multisample arrays unambiguous instead of overloading the third dimension.

Kernel `[[stage_in]]` attributes use `kernel_stage_inputs`, keyed by AIR attribute location. Each
literal supplies a typed format, the product's reflected runtime-array stride, and record bytes.
Product reflection owns the collision-free synthetic buffer slot: Metal uses it in an
`MTLStageInputOutputDescriptor` with x-grid stepping, and Vulkan binds the same bytes as the
read-only StorageBuffer array indexed by `GlobalInvocationId.x`.
