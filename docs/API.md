# The API

`metal2vulkan` compiles Metal AIR into Vulkan pipelines through an API shaped like Metal's object
lifecycle. You hand it Metal-shaped inputs (a device description, a library, pipeline
descriptors, what a draw or dispatch binds) and get back plain data to create and record Vulkan
objects with: SPIR-V per stage, one pipeline layout, fixed-function state, and per-draw descriptor
writes and push-constant bytes. You never parse AIR, never patch SPIR-V, never derive binding
numbers from Metal indices, and never classify errors by message text.

This guide walks the flow on the repository's own fixtures. Every `rust` block is quoted from
[`examples/api_preview.rs`](../crates/metal2vulkan/examples/api_preview.rs), which the test suite
runs (`tests/api_preview_example.rs` fails when a block here stops matching it). Sections 9 and 10
also cover the runtime crate (`metal2vulkan-runtime`): the precision notes it carries and its deliberate
differences from Metal. The runtime's own guide is [its README](../crates/metal2vulkan-runtime/README.md).

## Stability

- **No feature needed.** Feature `ash` adds conversions to `ash::vk` and
  `DeviceDescription::query`; `serde` serializes the plain-data values; `cli` builds the command.
- **Frozen names, additive changes.** The surface is the items at the roots of `error`, `device`,
  `library`, `reflection`, `pipeline`, `bind`, `cache`, `metal` (with `metal::{format, sampler,
  state, vertex}`), `vk` and `ash_interop`, plus `Budget` and `CancelToken` at the crate root.
  `tests/api_surface.rs` pins every name. Structs and enums are `#[non_exhaustive]` or opaque, so
  build values with `new`, `Default` or constants, assign fields, and match with a wildcard arm.
  The exceptions are exhaustive on purpose: keys and Vulkan structs (`SlotRef`, `ResourceSlot`,
  `ApiVersion`, ...), and the enums a consumer must interpret completely to bind and draw
  correctly (`StageSource`, `SlotClass`, `VertexStride`, `PrecisionNote`, and Metal's `PrimitiveType`, `IndexType`, `CullMode`, `Winding`
  and `TextureType`), where a new variant breaks your build on purpose.
- **Semver** covers this surface from 0.2.0 (still `0.x`: a breaking change bumps the minor
  version).
- **Not API**: anything the reference does not show. Hidden items, every submodule of `library`,
  `pipeline`, `bind` and `device` (write `pipeline::SlotRef`, never `pipeline::sym::SlotRef`), and
  the translator core are private.
- **The AIR front end** (container, bitcode reader, typed IR, metadata ABI, intrinsics,
  canonicalization) is the `metal2vulkan-air` crate, pinned exactly. The input and budget errors
  this crate re-exports are that crate's types.

`scripts/api-docs` renders the reference of exactly this surface (add `--open` to open it); it
fails when a page links to a hidden or internal item.

## Run the tour

```sh
PATH=/opt/homebrew/opt/llvm/bin:$PATH cargo run -p metal2vulkan --example api_preview
```

No GPU is involved. The fixtures are committed as LLVM text, so the example assembles them with
`llvm-as` first; the API itself reads bitcode only.

## 1. A device

A `DeviceDescription` states the Vulkan physical device as you create it: API version, enabled
features and extensions, limits, subgroup and float-controls properties, format support and the
driver's identity. Three ways to get one:

- `DeviceDescription::from_khronos_json(json, profile_name)` reads a Khronos profiles JSON. The
  tour uses the committed profiles under `crates/metal2vulkan-validation/profiles/`, from a bare
  Vulkan 1.2 device to desktop GPUs and MoltenVK.
- `unsafe DeviceDescription::query(&instance, physical_device, instance_api_version)` (feature
  `ash`) reads a physical device. That is what it supports: enable at least
  `description.recommended_enables()` at `vkCreateDevice`, then pass
  `with_enabled(&enabled)` so the description states what is enabled. `with_enabled` refuses a set
  that breaks one of Vulkan's co-enable rules, naming its VUID. The recommendation keeps those
  rules itself. It names neither `robustBufferAccess` nor `robustBufferAccess2`: the translator
  checks every buffer access in the shader and reads neither. If you enable `robustBufferAccess`
  yourself on a device that does not report `robustBufferAccessUpdateAfterBind`, leave out the
  buffer update-after-bind features the recommendation holds
  (VUID-VkDeviceCreateInfo-robustBufferAccess-10247); a device made without the texel-buffer ones
  refuses every texture buffer, naming the feature it lacks. `VkFeature::exposure()` and
  `VkExtension::requires(version)` say where each feature lives at the device's version, to chain
  the create info.
- Build one field by field.

```rust
let description = DeviceDescription::from_khronos_json(&json, None)?;
let device = Device::from_queries(description, DevicePolicy::default())?;
```

`Device::from_queries` needs no Vulkan device: it derives the profile from the description's
queries alone, reading no vendor, device or driver identity. It refuses a device below Vulkan 1.2,
a description whose `maxPushConstantsSize` is below 128 bytes, Vulkan's least and the least push
window a stage is lowered for (`Unsupported{PushWindow}`, a `Limit` of `MaxPushConstantsSize`),
and an inconsistent description or policy. The push window is the device's `maxPushConstantsSize`
up to 256 bytes (`DeviceProfile::push_window`, section 5). A
`Device` is cheap to clone; every compile takes it.

The guide's examples are quoted from `examples/api_preview.rs`. Its tour device takes
`DevicePolicy::default()`, the runtime's policy, which compiles under the identity domains; the
consumer pushes their fields itself (a texture's or sampler's resource ID in
`PushField::SlotGuestBase`, `PushField::ConstexprSampler`, `PushField::FetchedColor`,
`PushField::TessellationFactorAddress`, the indirect commands' `PushField::IndirectRange`,
`IndirectSequence` and `IndirectBase`, and `ZeroAddress`, `BicubicTexels` and
`PrivateArenaAddress`).

### The policy

`DevicePolicy` holds your product choices; together with the description they decide everything a
compile does. `DevicePolicy::default()` is the one constructor: the runtime compiles under it,
and so do the command line, the compile worker and every validation tool. It holds:

- **Bounds on a buffer-slot access at a run-time offset.** Such an access stays inside the range
  the consumer supplied: the shader checks it on every device, whatever buffer robustness the
  device enables.
- **The device's arithmetic.** The device's own maths library, divisions, fused multiply-adds,
  `float` denormals, rounding order of `dot`/`mix`/`fast::sqrt`, the grouping and fusion of
  fast-math sums and products (each operation in the AIR's order, its own fast-math flags its
  decoration), NaN bits, and the operand of `fmin`, `fmax`, `min3`, `max3`, `clamp` and
  `saturate`, on every device and with no exact form.
  Each records a precision note with its bound, on every device.
- **The device's native forms.** The device's own sampler (its filter, footprint, level,
  anisotropy, texel conversion and comparison, a two-plane YCbCr texture's filter among them),
  with no switch: the device's answer is the contract, recorded as `PrecisionNote::DeviceSampling`
  with its measured bound. The device's own `half` rounding and its blend of a `half` color have
  no switch either: each records its note.
- **The identity domains** (`PointerDomain::Identity` with `HandleDomain::Identity`, taken
  together; the default, the runtime's form): pointers are device addresses used directly and a
  texture's or sampler's handle is its `gpuResourceID` ([RESOURCE-ID.md](RESOURCE-ID.md)). They
  need `bufferDeviceAddress` and `shaderInt64`; `Device::from_queries` refuses a device without
  them (`Unsupported{IdentityDomains}`). Under
  them a buffer slot is reached at its own device address (`gpuAddress` plus the bind offset),
  pushed less 2^31 as `PushField::SlotGuestBase` (section 5), with no root table and no
  descriptor. The constant blob (the program-scope constants whose address merges, escapes or is
  compared, or that are read through another type than their own) is the module's own there: a
  `Private` array of its words, zero-padded to 16 bytes, which every load is checked against, and
  no stage carries a `constant_blob`. A blob past 64 KiB is refused
  (`Unsupported{IdentityDomains}`, naming the constant data's size). An access of the slot whose byte offset is not a
  translation-time constant is checked against the slot's buffer (`PushField::SlotBounds`), from
  the buffer's start to its end, as Metal's domain is the buffer: an offset before the bind
  offset reads and writes the buffer's bytes there. Outside the buffer, a load reads zero and a
  store or atomic does nothing (section 5). An access at a constant offset is not checked. A
  pointer loaded from memory or made from an integer, null included, is used as given: no binding
  bounds its accesses, so the bounds the resident-table domains place on an access do not hold
  for it. A slot
  whose argument states an alignment of at least 4 (`air.arg_type_align_size`) is reached from
  that address as it is: each access is at the pushed word plus its 32-bit byte offset plus 2^31,
  modulo 2^32, zero-extended, which is the address plus the offset read as signed, exact as the
  runtime makes no buffer longer than 2 GiB (`MAX_BUFFER_LENGTH`); a constant part of the
  offset joins the 2^31 in one constant. An access of 8 or 16 bytes at an offset proven a multiple
  of its size is one `uvec2` or `uvec4`. The function states `RuntimeFact::AddressAlignment` with
  that alignment (at most 16), and the runtime refuses a dispatch whose bound address is not a
  multiple of it, as Metal's API validation refuses the bind offset (an accepted difference,
  section 10). A slot whose argument states an alignment below 4, or none, is reached the same way where each of
  its accesses is one 1- or 2-byte unit no wider than that alignment (an `i8` access of a `uchar`
  buffer, an `i16` one of a `ushort` buffer) on a device with that width's storage access. Any
  other slot is reached from the word below each access, its offset sign-extended, so an access
  before the bind offset reaches below the bind address as Metal's does. Fields past a
  function's push window go to a parameter block in memory, read at the device address
  `PushField::ArgumentTableAddress`, which the window always holds. The function reads the block
  16 bytes at a time at multiples of 16 (its size, `argument_table_size`, is a multiple of 16) and
  never writes it, so the consumer places it at a multiple of 16 and leaves it unchanged while
  the command reads it. Every stage's window is `W` bytes, the device's `maxPushConstantsSize` up
  to 256 (`DeviceProfile::push_window`; at least 128, Vulkan's least), the one value a consumer's
  pipeline layout holds as its push range; the profile's fingerprint holds the limit, so devices
  of two windows share no artifact. A kernel's window is `[0, W)`: a dispatch region and 24 buffer
  addresses below its tail at 256 bytes, 8 at 128. A render
  pipeline's two functions share one range `[0, W)`: the vertex function's window is
  `[0, W - 8)`, `[0, 248)` at 256 bytes and `[0, 120)` at 128, and the fragment function's runs
  from the vertex function's end to `W`. A post-tessellation
  vertex function's window is `[24, W - 8)`, and its fragment function's starts at 24 or past: the
  pipeline's control stage pushes its fields (8 to 24 bytes) at 0, so the pair's fields, which
  spill where they pass their windows, never take its bytes. A
  vertex function is lowered without knowing its fragment function, so it leaves the last 8
  bytes, where a fragment function whose fields spill always pushes its block's address; that
  costs only a vertex function whose own fields pass `W - 8` bytes, which spills where it would
  have filled them. The fields only a checked access reads
  (`PushField::ZeroAddress` and each `PushField::SlotBounds`) come last in the order, so they
  are the first to move to the block. Each function's push ends with its tail, 24 bytes or more above its other pushed fields: its highest
  `PushField::SlotBounds` or `PushField::ZeroAddress`, moved there, or where it pushes neither, the
  zero word `PushField::TailWord` at `PushPlacement::tail_word`, with which its `push_range` ends;
  a function that pushes one field of 8 bytes or fewer has none. A driver that uploads a stage's
  push bytes up to the highest one the stage reads then holds every load it widens from the other
  fields. The function reads its tail at its start, and each of its other pushed fields reads as
  null where the tail holds what the consumer never pushes there (a tail word other than zero,
  bounds whose two words are both all ones, a zero-range address of all ones), so the consumer
  pushes the tail word zero with the other fields; the bytes of the hole are not read. A kernel that reads a grid builtin reads `PushField::DispatchRegion` on
  every dispatch: a dispatch the host knows states Metal's grid and its piece's first threadgroup
  there, and an indirect dispatch a zero grid, whose kernel takes `NumWorkgroups` times its
  workgroup size. It reads its first threadgroup from every region (0 where the
  dispatch is not split, and for an indirect one) and declares `MultiRegion`, default 0: at 0
  Metal's threadgroup size is the pipeline's workgroup size; at 1 it is the region's, for a region
  whose workgroup size is not Metal's (an edge region). A kernel whose workgroup is its
  threadgroup (one outside the flattened SIMD-group tiers, without compute derivatives) also
  declares `SpecKey::DeepThreadgroup`, default false: at true its workgroup is one row of the
  threadgroup's threads, `(sx·sy·sz, 1, 1)`, each thread's position its `LocalInvocationIndex`
  unflattened in Metal's order, so a threadgroup deeper than a device's
  `maxComputeWorkGroupSize[2]` runs; a runtime sets it for a region deeper than
  `pipeline::DEEP_THREADGROUP` (64) alone, so every other pipeline specializes it away. Such a
  kernel's `WorkgroupSize` parts are the plain constants `SpecKey::WorkgroupSize`, one an axis:
  the threadgroup, or one row of it where it is deep (section 3 names the call that gives them);
  no module's `WorkgroupSize` part is an `OpSpecConstantOp`. `MetalDeviceProperties::max_threads_per_threadgroup` is
  the capped `maxComputeWorkGroupInvocations` on every axis. A texture slot's `gpuResourceID` is pushed as
  its `PushField::SlotGuestBase`, and its texture is the element at the ID's index of set 1's two
  fixed arrays, the same layout for every pipeline ([RESOURCE-ID.md](RESOURCE-ID.md)); it binds no
  descriptor of its own and declares no specialization constant. A storage cube or cube array slot
  is declared as the 2D array of its faces and written and read by face at layer
  `6 · cube + face`, so the storage element a consumer holds for a cube texture is that 2D-array
  view of its layers. An argument buffer's texture is
  the element at the index of the `gpuResourceID` its member holds, with no handle table, so a
  slot and a member of one class index one array. No variant is chosen there: a
  texture reads through the plain form whatever texture its ID names, its `float` or `half` reads
  take the device's sRGB decode with `PrecisionNote::SrgbTexelDecode`, and no texture is sparse
  (the consumer refuses a sparse texture). A texture slot sampled through a runtime sampler slot
  samples set 1's element of the sampler arrays at the index of the sampler's pushed
  `gpuResourceID`, joined to the texture's element by `OpSampledImage`, and reads the sampler's
  word from that ID. One sampled through a constexpr sampler samples the element at the index the
  consumer pushes for it (`PushField::ConstexprSampler`), where the consumer places the device
  sampler `CombinedSampler::constexpr_sampler` describes, normalized: the shader scales a pixel
  sampler's coordinate. An integer write to a storage slot without a format saturates to the channel
  widths the consumer pushes (`PushField::TextureIntegerChannels`). A sample of a 2D texture
  slot through a bicubic constexpr sampler, at normalized or pixel coordinates, reads how Metal's
  bicubic filter treats the bound texture's texels (its inner taps, its clamp and its border
  texels) from the word the consumer pushes for that slot (`PushField::BicubicTexels`:
  `metal::format::bicubic_texels_word` of the bound texture's format and swizzle,
  `metal::format::BICUBIC_TEXELS_UNBOUND` for a slot bound to nothing), the bits the resident-table
  domains specialize `SpecKey::BicubicTexels` to. `bicubic_texels_word` has no word for a
  compressed, YUV, stencil, integer, extended-range or 8-bit sRGB format, so such a texture has
  no bicubic sample there (the runtime refuses the dispatch, `Gap::BicubicTexels`). Its
  arithmetic is noted `PrecisionNote::BicubicFilterArithmetic`, as on the resident-table domains. A
  fragment function under the pipeline's private alpha test
  (`PrivateRenderState::alpha_test_enabled`, a compare function other
  than `Never` or `Always`) compares its color 0's alpha with the code the consumer pushes
  (`PushField::AlphaTestReference`): `bind::alpha_test_code` of the encoder's
  `setAlphaTestReferenceValue:`, 1 until the encoder sets it. It is Metal's 8-bit unorm code of the
  reference (a NaN taken as 0, clamped to `[0, 1]`, the exact product with 255 rounded to
  nearest), the form the function converts its alpha to, so a consumer that pushes it compares as
  Metal does. A fragment function's read of a color attachment (framebuffer fetch, section 4) on a
  device with neither ordered form takes no input attachment: it reads, at the integer part of
  its position (its own sample where the pass is multisampled), level 0, the sampled-texture
  array's element at the index the consumer pushes for that attachment
  (`PushField::FetchedColor { color, texels }`, `texels` the read's numeric class), and a
  read-only signed-integer attachment joins `signed_color_attachments`. A vertex function that
  writes `[[render_target_array_index]]` compiles in the variant that writes `Layer`, which stores
  0 for an index at or past the render pass's length the consumer pushes
  (`PushField::RenderTargetArrayLength`, `renderTargetArrayLength` at least 1), and one that writes
  `[[viewport_array_index]]` stores `ViewportIndex` 0 for an index at or past the draw's viewport
  count the consumer pushes (`PushField::ViewportCount`). A kernel without a
  call stack whose private bytes pass the 16 KiB floor every device holds keeps its private memory
  in the consumer's device memory: every alloca it indexes at run time is in its private arena,
  whose words for an invocation are the slot of `RuntimeFact::PrivateArena { slot_bytes }` (the
  arena's bytes rounded up to 16) at the address the consumer pushes
  (`PushField::PrivateArenaAddress`, a `u64`, a multiple of 16) plus `i * slot_bytes`, `i` the
  invocation's index in its Vulkan dispatch, `((wz * ny + wy) * nx + wx) * n` plus its
  `LocalInvocationIndex` (`n` the workgroup's invocations), counted in 32 bits. The consumer
  supplies `count * slot_bytes` bytes for each dispatch of `count` invocations, and keeps a
  dispatch's invocations under `2^32`. The memory needs no zeros: Metal's private memory is
  undefined until the thread writes it. Every precision note is
  static there: no constant or variant decides one. So far they lower a kernel that reaches memory
  through its buffer slots and the device addresses it loads (an argument buffer's pointer
  member, a stored `gpuAddress`) or makes from integers, and reads, writes, sizes or tests for nil
  its texture slots and its argument buffers' textures, those of an argument buffer a pointer
  member reaches included, reads and writes its `ulong` texture slots (an `RG32` unsigned image,
  declared `Rg32ui` where it is a storage image, the element at its ID's index), updates its
  texture slots atomically (a slot declared `R32ui` or `R32i`, the storage-image array's element
  at its ID's index), writes and reads its storage texture slots at a level other than the
  constant 0 (element `first + min(l, max(n, 1) - 1)` of set 1's storage-image array where the
  level `l` is below the bound texture's level count `n`, `PushField::TextureLevels`, and
  `first` its run's first element, `PushField::TextureLevelBase`:
  [RESOURCE-ID.md](RESOURCE-ID.md)), reads, writes and updates atomically its texture buffers, slots and
  argument buffers' members alike (a uniform texel buffer where it reads one, a storage texel
  buffer where it writes one, the element of set 1's texel-buffer array at its ID's index), and
  samples its texture slots
  through its runtime sampler slots and its constexpr samplers and reads them through those (MSL
  4.1's `read(sampler, coord)`, its border texel from bits 0-11 of the ID's word). Any other
  function is refused as `Unsupported{IdentityDomains}`, whose symbol names the first such form
  the function holds: one that samples through an argument buffer's sampler or samples an
  argument buffer's texture, emulates a sampler's anisotropy, samples a 1D
  texture through a bicubic filter, uses a texture buffer's parts or rows
  form, a slot's swizzle, a two-plane, extended-range or sRGB-twin form, reads an
  argument buffer's texture
  through another numeric type or as raw texels, or uses any other state of a texture or sampler
  beyond its ID.
  An argument buffer's texture reads the element at its ID's index masked to its low 16 bits,
  so a bad ID reads an element of the arrays ([RESOURCE-ID.md](RESOURCE-ID.md)); the change that
  first lowers an argument buffer's sampler masks its index the same way.
  A function writes an indirect command buffer's render commands, under the identity domains
  alone, through the `gpuResourceID` an argument buffer's `command_buffer` member holds: the
  device address of the buffer's records, laid out as `pipeline::IndirectDrawLayout` states. Any
  other access that may reach the member's 8 bytes, such as an index past the structure that
  runs back into it, is a plain access of the argument buffer's memory there, reading or writing
  that word, as Metal's Tier 2 argument buffer is laid out as the equivalent C structure; under
  the resident-table domains it is `Unsupported{ArgumentBufferObjects}`. A
  command's `set_vertex_buffer` and `set_fragment_buffer`, its `draw_indexed_primitives` of
  triangles and its `reset` write them, each only where the command's index is below the
  buffer's capacity; any other indirect-command operation is
  `Unsupported{GpuEncodedIndirectCommands}`. A render pipeline state made with
  `support_indirect_command_buffers` compiles its vertex function under
  `LoweringInputs::indirect_range`: it reads the execution's range at `PushField::IndirectRange`
  and its command, `PushField::IndirectSequence` (a part's sequence index) plus
  `PushField::IndirectBase` (the part's first command), a command outside the range draws nothing,
  and the function pulls every vertex attribute it reads.
  Where the GPU writes the range, the parts' counts come from a kernel built from no Metal
  function, `pipeline::own_kernel(OwnKernel::ExecutionCounts, profile)`: one workgroup a part
  reads the range and writes the part's count, the commands it runs through the range's last,
  into the parameter block `pipeline::ExecutionCounts` lays out, where each part's
  `sequenceCountAddress` reads it. Its interface is a compiled function's: the one push range,
  holding the block's device address at offset 0, and no descriptor.
- **The resident-table domains** (`DevicePolicy::RESIDENT_TABLES`: `PointerDomain::ResidentTable`
  with `HandleDomain::Table`) exist for the validation executor alone, which builds the resident
  root and the handle table itself. Besides it, only the translator's own tests of that lowering
  compile under them.

Loops run as written: no execution bound is translated in.

### What the device answers

```rust
let device = runtime_device("moltenvk-m3")?;
let enables = device.required_enables();
// ...
hex(&device.profile().fingerprint()),
// ...
println!("  Metal properties: {:?}", device.metal_properties());
```

- `required_enables()`: everything an artifact compiled for this device can need enabled.
- `metal_properties()`: the answers to the guest's device queries (thread execution width,
  threadgroup limits, argument-buffer and read-write texture tiers, ...).
- `profile().fingerprint()`: a hash of every profile field, the policy included. Every cache key
  includes it, so an artifact never serves another device. Facts that follow the machine's memory
  (`DeviceDescription::allocation`: `maxMemoryAllocationSize`, `maxBufferSize`) are not profile
  fields, so two machines that differ only there share artifacts.
- `profile().spirv_validation()`: the `spirv-val` environments and relaxations every module for
  the device is valid under. The command line runs exactly these.
- The profile's facts: `enabled()`, `limits()`, `images()`, `fragment()`, `descriptors()`,
  its `subgroups()` model and `simd_tier(...)`, and the Function
  storage a thread holds (`function_storage()`: `FunctionStorage::Measured` bytes for a measured
  device, or `FunctionStorage::Assumed` with the 16 KiB floor, past which a vertex or fragment
  function runs with `FunctionStorageAssumed` up to 192 KiB and any other stage is refused).

### Driver behavior

Where Vulkan leaves an answer to the driver, the profile takes the device's own answer, recorded
as a precision note, or, where the answer decides a capability, its emulation or a refusal.
`Device::from_queries` reads no driver identity, so no answer is taken as Metal's because a driver
was measured to give it.

Metal semantics the translation keeps on every device, beyond Vulkan's: Metal's 32-lane
simdgroups (and, under the resident-table domains alone, the order Apple's lockstep gives a
simdgroup's threadgroup and device accesses within a barrier interval). `fneg` and
`fabs` are the device's `OpFNegate` and `FAbs`, so a NaN's payload and a denormal are the
device's (`native-nan-encoding`, `device-denormals`). A NaN's bits, the
operand `fmin`, `fmax`, `min3`, `max3`, `clamp` and `saturate` pick beside a NaN or between two
zeros, a `half` read from its bits, which is the device's widening on every device (a NaN's
bits, and a denormal on a device that may flush 16-bit ones), and a `half` rounded from a value
below the smallest normal `half` on a device without `shaderFloat16` (its `OpQuantizeToF16`'s
zero) are the device's, each recorded as a precision note.

## 2. A library

`Library::parse` takes a metallib, a bitcode wrapper or a raw AIR module; `Library::from_air`
takes the last two. LLVM text is not an input: assemble `.ll` with `llvm-as` first. Listing
functions decodes nothing.

```rust
let sampling = Library::from_air(&fixture_bitcode("sampling")?)?;
for info in sampling.functions() {
    println!("library: {} ({:?})", info.name, info.stage);
}
```

- `library.function(name)` finds a function like `newFunctionWithName:`; a missing name is
  `Error::InvalidDescriptor(EntryNotFound)`.
- A metallib function's `info.air_hash` is the SHA-256 of its bitcode, its content key. In every
  library but a Core Image kernel library it is the function's `HASH` tag, checked against the
  bitcode: a mismatch is `InvalidInput(Container(HashMismatch))`, though Metal checks no `HASH`
  and loads such a library. A Core Image kernel library (file type 1, `metallib -cikernel`)
  lists its kernels as extern functions (`FunctionStage::Extern`, no stage) over one module, and
  Apple's toolchain writes them a `HASH` that, for some modules, is not that module's SHA-256
  (what it digests is not known); Core Image's own libraries are such. Its `HASH` is read as 32
  bytes and kept as read, and `air_hash` is the module's SHA-256, computed. The conformance
  crate's `probes/metallib_function_hashes.m` measures what Metal answers for both kinds of
  library.
- `function.reflection(&budget)` answers as `reflectionForFunctionWithName:` does: the function's
  arguments in declaration order with their types, access, indices and use, in Metal's shape
  (module `reflection`). It is `None` for a metallib function whose library carries no reflection,
  where Metal answers nil. An argument whose Metal answer was not measured is
  `Unsupported(Feature::Reflection)`.
- A pipeline descriptor's `reflection(&budget)` answers what the `reflection:` out-parameter of
  Metal's creation call does.
- A pipeline descriptor's `slot_writes(&budget)` answers, per stage, what the function may store
  through the object bound at each slot its arguments bind (`SlotWrites`, one `SlotWrite` per
  binding). `SlotWrite::may_write(slots, binding_type, index)` answers for one slot:
  `SlotWritability::Never` where the declaration is read-only (a `constant` or `const device`
  buffer, a texture declared `access::read` or `access::sample`, a sampler, a function table, an
  acceleration structure, an imageblock, a mesh function's payload) and for a slot no argument
  binds, `Maybe` where it is write-only or read-write, and `Unknown(SlotWriteGap)` for a slot that
  one argument alone leaves undetermined.

  The values are the ones `reflection` gives as `Binding::access`, worked out without the type
  description a reflection must also give. That is why a state that compiles always answers here,
  while its `reflection` is refused where Metal's answer for one argument's *type* was not
  measured, and why uncertainty stays on the slot it belongs to instead of refusing the state:
  a caller asking whether a read-only buffer may sit at a slot gets an answer per slot, and an
  undetermined slot reads as neither writable nor read-only.
- `function.argument_buffer_layout(index, &constants, &budget)` gives the layout of the argument
  buffer the function takes at a buffer index, as `newArgumentEncoderWithBufferIndex:` encodes it:
  its length, alignment, and each argument's offset and type by `[[id(n)]]`.
  `ArgumentBufferLayout::from_descriptors` gives the layout `newArgumentEncoderWithArguments:`
  makes of a list of `ArgumentDescriptor`s. Each argument is 8 bytes: a buffer's address plus
  offset, or a resource ID.
- `function.properties(&budget)` answers what `MTLFunction` states about the function.

### One entry of a bare module

A tool that holds bitcode and an entry name, not a library, compiles through the profile:

```rust
let bitcode = fixture_bitcode("sampling")?;
let entry = EntryDescriptor::new(&bitcode, EntryRef::Name("sample_level"));
let profile = device.profile();
let (compiled, report) = profile.compile_entry(&entry, &LoweringInputs::new(), &budget);
let alone = profile.link_standalone(&compiled?)?;
```

`link_standalone` links the function with no partner stage. The report says what the compile cost
against its budget, whether it succeeded or not.

## 3. A compute pipeline

Descriptors mirror Metal's. Each stage is a `StageSource`: a `FunctionDescriptor` to compile (a
function, its constant values, the functions it links) or a `CompiledFunction` to link as it is.
Every compile takes a `Budget`:

```rust
let cancel = CancelToken::new();
let mut budget = Budget::standard();
budget.cancel = Some(cancel.clone());
let kernel = FunctionDescriptor::new(sampling.function("sample_level")?);
let descriptor = ComputePipelineDescriptor::new(StageSource::Function(kernel));
let pipeline = device.compile_compute_pipeline(&descriptor, &budget)?;
```

`Budget::standard()` is a 20-second deadline. `cancel.cancel()` from any thread ends the attempt
at its next poll point with `Error::Cancelled`; a passed deadline is `Error::Budget`. Both are
transient. Deterministic caps (work units, a memory ledger, structural limits) apply whatever the
budget, and exceeding one is `Error::LimitExceeded`, permanent for that input.

A `ComputePipeline` is immutable plain data:

- `stage`: a `StageModule` (SPIR-V 1.5 words valid for the device, the entry point, specialization
  constants, a required subgroup size and stage flags, precision notes, `private_bytes`, the
  private memory a thread keeps, as `StageInterface::private_bytes` states it too, and `stats`,
  what the module contains);
- `layout`: the pipeline layout, dense and complete; create it as it is;
- `interface`, `dependencies`: what it binds and what bind-time state it depends on (section 5);
- `metal_info`: answers to Metal's per-pipeline queries (`maxTotalThreadsPerThreadgroup`, ...);
- `requirements`: what the device must have enabled.

Do not create the compute `VkPipeline` from `stage` alone: its specialization defaults are
placeholders (a kernel's local size is 1). Create each pipeline with the values its dispatch
gives the specialization constants' keys, and with the stage's `stage_flags` and
`required_subgroup_size`. Every key a kernel's threadgroup size decides comes from one call,
`pipeline::threadgroup_constants(size)`: `SpecKey::LocalSize` on each axis, `FlatLocalSize`,
`DeepThreadgroup` and the `WorkgroupSize` parts on each axis. Write each of them the stage
declares from it, never one alone: a kernel that declares `DeepThreadgroup` takes its Vulkan
workgroup from the `WorkgroupSize` parts only, so one given its `LocalSize` without them runs
workgroups of the parts' defaults. A required subgroup size needs `subgroupSizeControl` and the stage in
`requiredSubgroupSizeStages`; full subgroups need `computeFullSubgroups`; a size that varies
(`ALLOW_VARYING_SUBGROUP_SIZE`) needs `subgroupSizeControl`. A pipeline compiled for
one device pins only what that device can take; `DeviceProfile::check` tells whether another
device can.

A kernel whose simdgroups run on 32-lane subgroups pins a required size of 32 with full
subgroups. On a device whose compiler chooses a kernel's SIMD width between a `minSubgroupSize`
below 32 and a `maxSubgroupSize` of 32 (ANV), that pin compiles the kernel in the device's widest
mode, where each lane's value takes four times the registers of the narrowest one. A kernel whose
entry, inlined, holds more than 256 AIR instructions and whose values are estimated to hold more
than 36 32-bit words at once (live intervals over the body in layout order, a value read in a loop
held to the loop's end) takes the emulated simdgroup tier there instead: each simdgroup is 32
consecutive threads, its operations and lockstep barriers go through workgroup memory and
workgroup barriers, and its module pins no subgroup size, so the driver chooses the width. Both
forms are exact. The runtime counts such kernels (`WorkCounts::widest_pin_kernels_emulated`). A
kernel the emulated tier refuses (a 64-bit exchange, or a scratch that does not fit beside its
threadgroup memory) keeps the pin.

The bounds come from an authored kernel measured on an Iris Xe (RPL-P, ANV): `S` steps, each a
branch on one of `K` live `float` accumulators whose arm stores a device word and loads the next
(a lockstep pair in divergent flow, so a subgroup barrier there), with `P` 1 the arm's accesses
going through a pointer it loads from memory (a resident-set search a step), then every
accumulator updated. Pressure, not size, separates the cases. The driver's compile time
(`vkCreateComputePipelines`) and the GPU time of a dispatch of 12,288 threads, pinned and
emulated, one run each (compile times varied up to twice between runs):

| `P K S`    | AIR instructions | estimated words | compile ms, pinned | emulated | GPU ms, pinned | emulated |
|------------|-----------------:|----------------:|-------------------:|---------:|---------------:|---------:|
| `0 16 128` |            7,850 |              24 |                589 |      559 |          0.129 |    0.159 |
| `1 4 32`   |              874 |              34 |                783 |      766 |          0.010 |    0.018 |
| `0 32 64`  |            7,050 |              40 |              6,652 |      282 |          1.061 |    0.107 |
| `1 16 16`  |            1,042 |              46 |                971 |      300 |          0.070 |    0.015 |
| `1 16 64`  |            4,066 |              46 |             11,987 |    2,585 |          0.319 |    0.074 |
| `0 48 8`   |            1,362 |              56 |                458 |       38 |          0.168 |    0.021 |
| `1 48 32`  |            5,186 |              78 |             25,799 |    1,277 |          1.985 |    0.081 |

At 34 estimated words or fewer the two compile alike and the pinned kernel runs as fast or
faster; from 40 the pinned compile took 1.6 to 24 times the emulated one and its GPU time 1.5 to
25 times. At or below 256 instructions every pinned compile measured took at most 102 ms. Answers
were the same bits on both tiers. The estimate costs time linear in the body: 5 ms of a 3.4 s
release translation of a 162,914-instruction kernel.

The 32 lanes are needed only by operations that read outside a quad. A kernel whose every
cross-lane operation is a quad operation (no simdgroup operation, multiply-add, or simdgroup barrier
or fence) pins nothing where the kernel stage runs full subgroups, can take a size, and has a
narrowest size below 32 lanes and at least 4 (ANV's 8 to 32, lavapipe's 8): it takes its quads
from the subgroup at the size the device gives, with full subgroups,
`ALLOW_VARYING_SUBGROUP_SIZE` and a local size that is a multiple of `maxSubgroupSize`. A quad is
four lanes of one subgroup at every such size (Vulkan 9.27.9), so its answers do not change, and
the widest mode is not weighed for it. The pin to 32 stays where 32 is the device's narrowest size
(NVIDIA's 32, RDNA's 32 to 64); a device whose compute stage takes no size runs the API's
`subgroupSize`. A kernel with compute derivatives keeps the tier its class takes.


On the emulated tier, a simdgroup operation or barrier in a block whose control flow differs
between threads is walked predicated: each thread carries a live flag, the threads that do not run
the block take part in its exchanges as inactive lanes, and a loop that holds one votes on its
exit. A kernel whose threads' positions alone make such a block differ, along some axes
(`thread_position_in_threadgroup` or `thread_position_in_grid`, read by lane), states
`RuntimeFact::UnitAxes { axes: 0, relief }`: `relief` holds, at bit `m - 1`, each mask `m` of axes
(bit 0 `x`, bit 1 `y`, bit 2 `z`) under which fewer of those blocks differ once the positions along
them are taken as uniform. In a threadgroup of one thread along an axis, Metal's
`threads_per_threadgroup` is 1 there, so every thread's `thread_position_in_threadgroup` on that
axis is 0 and its `thread_position_in_grid` is its threadgroup's position, partial threadgroups
included. A dispatch of such threadgroups, where its axes are among the masks of `relief`, may
run on the kernel's unit-axes variant (`VariantKey::unit_axes`, which `LoweringInputs::unit_axes`
compiles): the same kernel with those positions uniform, walked without the predication they
caused, giving the same answers. Its compile states `UnitAxes { axes, relief: 0 }`, and a
dispatch with more than one thread along one of its axes takes the variant without them. A kernel has at most seven
such variants per variant it runs on, one per mask. On the Iris Xe, an authored kernel of four shuffles each way and a
simdgroup sum in a loop of 8 to 11 trips, in threadgroups of 8 x 1 x 16, ran 1.14 to 2.16 times
the GPU time predicated (its trip count read from `thread_position_in_grid.y`) as with the same
trip counts read from `threadgroup_position_in_grid.y`, over four sizes from 46 to 126 estimated
words (one run each), with the same bits.

### What the module contains

Every stage states the size and shape of its module, counted by the verifier that runs on every
translation, in every build: `stage.stats`, a `ModuleStats`. A function compiled on its own states
the same counts as `module_stats()`, before it is linked. Reading them is how a caller watches the
size of what it hands a driver, and what in it grew.

```rust
let stats = pipeline.stage.stats;
println!(
    "{} blocks, {} instructions",
    stats.blocks, stats.instructions
);
assert!(stats.reachable_blocks <= stats.blocks);
```

- `blocks` and `instructions` are the module's size: a larger module costs more to compile on the
  drivers measured. Canonicalization inlines every direct call before the module is emitted, so
  they count the whole program, not one function.
- `functions` -- the entry point and the functions left beside it -- says how far `blocks` stands
  for that program. A driver inlines before it charges, so a module of few functions is one whose
  blocks are the program's, while a module that calls a many-block function from many sites holds
  a program far larger than its own blocks; the two read alike in `blocks` alone. Lowering emits
  such functions itself: the handle-table search (unrolled branch-free steps) and the root-table
  search (a loop), each emitted once and called at every access.
- `entry_weight` is what the stage bound weighs: what a driver compiles once it has inlined every
  call, the entry's instructions with each call counted as its callee's body, after repeated
  regions are rolled into loops, with each subgroup instruction counted as 64 instructions, as a
  driver's compile grows with one many times faster than with any other instruction. A kernel
  past `pipeline::COMPILED_STAGE_WEIGHT` (39,014, the heaviest kernel
  measured to compile within 512 MiB of its creating process's resident growth on every driver
  measured) is one whose creation may take the host's memory: the translator makes it all the
  same, and `metal2vulkan-runtime` compiles such a kernel only in a helper process the consumer
  caps (its README, "Kernels past the stage bound", which gives the figures and the render
  stages' own bound). A kernel past the bound only because it emulates its SIMD groups is planned
  again keeping the device's widest subgroup mode pinned: the re-plan takes the pinned module
  where it weighs less than the emulated one by the stage weight; otherwise the emulated one goes
  on.
- Compare one module against the next of the same shader, not one shader against another. Measured
  cost per block differed several-fold between a synthetic module and a real workload, so a
  per-block figure taken from one shader predicts nothing for the next; `instructions` is the
  second number for that reason.
- `constructs` (selection, switch and loop headers) and `loops` say whose blocks those are.
  Translation adds blocks of its own -- inlining, and the structurizer's layout of control flow as
  SPIR-V requires -- and the input has blocks of its own. The structurizer makes no loop the input
  does not have, so a `blocks` that has risen while `loops` holds still is translation's work or a
  changed shape of the same control flow, not new control flow in the shader.
- `reachable_blocks` is a bound, not a lever. It is never above `blocks`, and below it by the merge
  blocks SPIR-V requires a construct to declare that no path reaches -- both arms of a conditional
  inside a loop leaving the construct, say -- which the module terminates with `OpUnreachable`. The
  difference is blocks the driver still reads and no invocation runs.
- The AIR side of the comparison is yours, through `metal2vulkan-air`. For the counts the input
  itself holds, before inlining, decode it and count the blocks of each function's body
  (`metal2vulkan_air::decode`, then each `ir::Function`'s `body`); for the ones canonicalization
  produced, walk the entry `DeviceProfile::specialized_entry` gives, which is that AIR specialized
  and canonicalized. A decoded function's `ir::InstKind::Call` sites give each callee's call
  count, which is what turns its own blocks into the program inlining will make of it.
- `inert` says the module holds no loop and that no instruction of it writes memory other than an
  invocation's own and its workgroup's (`Function`, `Private` and `Workgroup` storage), writes an
  image, or emits or drops an output of its stage. Every invocation of such a module ends and
  leaves nothing behind, so a dispatch of an inert kernel computes nothing anyone can read. An
  instruction the
  check does not classify as reading or computing alone (an interlock, a ray query) makes a
  module not inert.

The counts are of the module `spirv` holds: the verifier counts the module it checked, and
serialization and relocation assign link numbers without adding or removing an instruction. They
are part of the encoded artifact, so a cache hit states them too, and they are left out of
`content_digest`, which already covers the words they describe.

### A kernel with `[[stage_in]]`

A kernel's `[[stage_in]]` attributes are fetched in the shader, as Metal does. The compile asks for
the stage-input descriptor it needs, and you supply it:

```rust
match device.compile_compute_pipeline(&descriptor, budget) {
    Err(Error::NeedsInput(RuntimeInput::StageInputDescriptor { attribute })) => {
        let mut stage_input = StageInputDescriptor::new();
        let fetch = StageInputAttributeDescriptor::new(AttributeFormat::Float, 0, 1);
        stage_input.attributes.insert(attribute, fetch);
        let mut layout = StageInputLayoutDescriptor::new(Some(4));
        layout.step_function = StepFunction::ThreadPositionInGridX;
        stage_input.layouts.insert(1, layout);
        descriptor.stage_input = Some(stage_input);
    }
// ...
let mut inputs = LoweringInputs::new();
inputs.stage_input = descriptor.stage_input.clone();
let compiled = device.compile_function(&kernel, &inputs, budget)?;
```

A fresh `StageInputLayoutDescriptor` steps per vertex at rate 1, as a fresh Metal layout does. A
layout stride of `None` is `MTLBufferLayoutStrideDynamic`: the stride comes from the encoder.

A kernel that takes a patch control-point input (`air.patch_control_point_input`, the input of a
post-tessellation vertex function) is refused, `InvalidInput(Abi(UnknownKey))` at that argument and
key. Metal's compiler writes that input for no kernel, and Metal makes no compute pipeline of a
kernel that takes one: on an M3 Max under macOS 27.0.1 it makes the library and the function, then
fails the pipeline with "Internal compiler error." under every compute step function. A kernel reads
vertices by index through `[[stage_in]]` under a stage-input descriptor (one element per thread), or
through buffer arguments it reads at any index.

### Other compute features

- **Visible functions and recursion.** Linked visible functions, function tables and
  intersection function tables compile; recursion through visible functions runs on a stack of
  frames sized by `max_call_stack_depth`. A call that finds the stack full returns zero.
- **Linked functions: names and constants.** A visible-function reference resolves to the linked
  function whose `LinkedFunction::name` it names: its specialized name where it was made with one
  (`with_specialized_name`, Metal's `MTLFunctionDescriptor.specializedName`), else its library
  function's own; its body is still that library function's. A linked function whose module
  declares function constants is specialized with its own values before it is linked, its
  module's static initializer run over them. A function made specialized
  (`LinkedFunction::specialized`: with constant values, or by `newFunctionWithDescriptor:` with or
  without them) reads a constant it was given no value for as an entry's undefined constants read
  (section 10, "Function constants"). One made unspecialized (`newFunctionWithName:`) that reaches
  one of its module's constants refuses the pipeline at a call that survives specialization,
  `DescriptorError::LinkedFunction` with `LinkProblem::UnresolvedConstant { index }`, guarded by
  `is_function_constant_defined` or not, as Metal fails the creation ("cannot lower module with
  unresolved function constants"); one that reaches none links. The specialized name and the
  specialization are part of the compile key. A reference resolves against `functions` and
  `private_functions` (`LinkedList::resolves_references`), never against `binary_functions`,
  whose functions only a visible-function table calls: a reference only a binary function names
  is `NeedsInput(LinkedFunction)`, as Metal fails the creation ("unresolved visible function
  reference"). Two different functions of one name, one in `functions` and one in
  `binary_functions`, are two functions a table holds, each with its handle, and the reference
  calls the one of `functions`, as Metal links them.
- **Linked functions: types.** A reference resolves to the linked function of its name whatever
  the definition's type is, and Metal makes every such pipeline (the probe
  `linked_function_signatures.m`, an M3 Max under macOS 27.0.1). The call passes its arguments by
  position: one past the definition's last parameter is dropped, and a parameter past the
  reference's last argument reads zero. An argument of another plain-value type (an integer or
  floating-point scalar or vector, a `bool` as one bit) passes its bits: all of them at the same
  width, its low bits to a narrower parameter, its bits with zeros above to a wider one. The result
  passes back the same way; a definition returning nothing returns zero, and a reference returning
  nothing drops the result. A struct a `[[visible]]` function takes by value reaches it as its
  fields, so a reference passing the fields one by one is passing it. A pointer for a value or a
  value for a pointer, a pointer of another address space, and an aggregate of another type refuse
  the pipeline at a call that survives specialization, `Unsupported` with `Feature::Linking`
  (section 10, "Linked functions").
- **Stitched functions.** A function Metal's stitcher made (`newLibraryWithStitchedDescriptor:`)
  links like any other visible function, in `functions` or in `privateFunctions`: hand the
  stitched library's bytes to `Library::from_metallib`, and link its function. Each stitched
  function is a module of its own whose `!air.signature` describes the graph it was made from;
  the module's own code, the graph's nodes inlined, is what is linked. The translator makes no
  stitched library itself. A module whose linkage signature has no reading (a linked-constants
  signature on a module that still declares function constants) refuses the pipeline at a call
  that survives specialization, `Unsupported` with `Feature::Linking`.
- **Threadgroup memory.** No threadgroup memory access is
  bounded: a load or store, atomic, memory intrinsic, pointer cell or imageblock access of a
  threadgroup arena, and a typed threadgroup object's index, run as the program states them, and
  one past the memory is undefined, as in Metal. A load, store or atomic of the arena at a constant
  offset outside it, undefined in Metal too, is a zero load, no store, or an atomic that writes
  nothing and returns zero: no access chain may index the arena's words with a constant past them.
  A kernel whose every threadgroup access is a
  whole, aligned `float` (a scalar or a vector of two to four), with no atomic, memory intrinsic,
  imageblock or pointer kept in it, and whose SIMD groups are not emulated, holds its threadgroup
  memory as a `Workgroup` array of `float`: each access is one typed load or store a lane, with no
  branch and no bitcast, at the word `offset / 4` and the words after it. The array is the
  memory's own words (the static part and the specialized `[[threadgroup(n)]]` lengths), at least
  four: the memory Metal allocates, so its threadgroup memory is Metal's static length and lengths,
  and a length filling the limit runs. The strategy census names its accesses
  `access.threadgroup.load.floats` and `access.threadgroup.store.floats`. Every other kernel holds
  `u32` words.
- **Threadgroup memory before the program writes it.** A function's threadgroup memory reads zero until the program writes it: the `Workgroup` variables
  that hold it, its threadgroup arena (declared threadgroup variables, `[[threadgroup(n)]]`
  arguments and an explicit imageblock) and its typed threadgroup objects, have a null
  initializer, in every threadgroup of every dispatch. That needs
  `shaderZeroInitializeWorkgroupMemory`, which Vulkan 1.3 requires; a function that uses
  threadgroup memory on a profile without it is `Unsupported` with
  `Feature::ZeroedThreadgroupMemory`. The translator's own `Workgroup` words have none: an
  emulated SIMD group's exchange scratch and vote word, an object function's mesh grid, a mesh
  function's primitive count and staged indices. Each is written
  before any read of it, and an object function's payload "is assumed uninitialized at the entry
  of the object function" (the Metal Shading Language Specification, the `payload` attribute). Metal's documents give
  threadgroup memory no first contents: `MTLComputeCommandEncoder.h`'s
  `setThreadgroupMemoryLength:atIndex:` reads "Set the threadgroup memory byte length at the
  binding point specified by the index. This applies to all compute kernels.", Apple's page for it
  "Configures the size of a block of threadgroup memory." and points to the specification's
  section 4.4, which says "Threadgroup variables in a kernel, mesh, or object function only exist
  for the lifetime of the threadgroup that executes the kernel." and nothing of what they hold
  first. An Apple M3 Max under macOS 27.0.1 reads zero there, a declared array and a
  `[[threadgroup(n)]]` argument alike, in the later threadgroups of a dispatch of 2048
  threadgroups of 16 KiB each, many more than its cores hold at once, and in a second dispatch
  (the conformance case `compute-threadgroup-memory-reads-zero`); recorded programs read entries of
  it they have not written.
- **Binary functions, required threadgroup sizes, buffer mutability.** Whether a binary function
  was compiled to a binary (`LinkedFunction::precompiled`), required threadgroup sizes and buffer
  mutability change no translation and are in no compile key; the runtime checks them as Metal
  does.
- **Dynamic libraries.** `DynamicLibrary::new(library, &budget)` checks a dynamic metallib as
  Metal's `newDynamicLibrary:` does; pipelines link its exported functions.
- **OpenCL C kernels.** Metal compiles OpenCL C (`newLibraryWithSource` with the compile options'
  `sourceLanguage` 1, which the GL renderer's compute path uses), naming each kernel
  `__entry_<name>`, and its AIR has shapes MSL's does not:
  - A helper left out of line that passes its parameter on as an intrinsic's immediate operand:
    OpenCL C's `barrier(flags)` passes its flags to `air.wg.barrier`, and `mem_fence(flags)`
    computes `air.atomic.fence`'s flags from them with `llvm.bitreverse.i2`. Each inlined copy
    reads its own call site's constants, folded through the integer arithmetic between them. An
    immediate that is still not a constant after that is
    `InvalidInput(Intrinsic(ImmediateNotConstant))` at the call. `llvm.ctpop`, `llvm.bitreverse`,
    `llvm.bswap` and the scalar `llvm.umin`, `llvm.umax`, `llvm.smin` and `llvm.smax` are folded
    over constants, on a width no SPIR-V type has (`i2`) too; an integer `llvm.*` operation on such
    a width that is not folded is refused (`Lowering(Arith)`).
  - An argument passed by value (a kernel's scalar, vector or struct parameter, `air.constant`):
    the entry reads it from a constant-space pointer at its location, so it is a read-only
    constant buffer holding its own bytes, bound with `set_bytes` or a buffer at an offset. The
    reflection lists it as such a buffer of its own data type, size and alignment, a struct as
    `Struct` with its members, as the M3 Max answers: `float k` is `Float`, 4 bytes aligned to 4,
    and `struct { float a; int b; }` is `Struct`, 8 bytes aligned to 4.
  - A required work-group size (`reqd_work_group_size(x, y, z)`, `air.work_group_size`) caps
    the pipeline's `max_total_threads_per_threadgroup` at `x * y * z` threads, as the M3 Max
    answers (64 for `(64, 1, 1)`). A dispatch in smaller threadgroups runs, as it does there.
  - OpenCL mode. Metal makes a pipeline of such a kernel only with
    `ComputePipelineDescriptor::open_cl_mode_enabled` (`openCLModeEnabled`, Apple SPI), and the
    crate compiles the kernel as Metal links it then (`LoweringInputs::open_cl_mode`, part of the
    key). Three hidden buffers come before the kernel's own: `__global_offset_and_num_dims` at
    index 0, four `uint`s the dispatcher fills with the dispatch's global offset in each of its
    dimensions and its dimension count, `__printf_buffer` at 1 and
    `__image_order_data_type_table` at 2. Each argument the kernel binds at a buffer index sits 3
    past its own; threadgroup memory (`__local`), textures and samplers keep theirs. The
    reflection lists the kernel's arguments at those indices, then the three buffers (read-only
    `uint` 4 bytes, read-write `char` 1 byte, read-only `uint` 4 bytes), each used, as the M3 Max
    answers. OpenCL C's work-item functions read the dispatch: `get_global_id(d)` is the thread's
    position in the grid plus the offset in `d`, `get_global_offset(d)` and `get_work_dim()` read
    buffer 0, `get_local_id`, `get_group_id`, `get_local_size`, `get_global_size` and
    `get_num_groups` are the built-ins of those names, and a dimension the kernel reads from its
    data selects among the three. A dimension past 2 answers OpenCL's 0 for an id or an offset
    and 1 for a size or a count. A compiled kernel serves only a pipeline of the mode it was
    compiled in (`RuntimeFact::OpenClMode`, `InvalidDescriptor(OpenClMode)` otherwise). Without
    the mode Metal refuses the pipeline ("Internal compiler error"); the crate compiles the AIR as
    Metal AIR, its buffers at their own indices, and refuses a read of `get_global_offset` or
    `get_work_dim` (`Unsupported` with `Lowering(Dispatch)`).
  - Images in OpenCL mode. `get_image_channel_order(image)` and
    `get_image_channel_data_type(image)` (`air.get_channel_order_texture_<kind>` and
    `air.get_channel_data_type_texture_<kind>`, `Query::ChannelOrder` and
    `Query::ChannelDataType`) read the third hidden buffer as it holds them: word `2t + 1` for the
    order and word `2t` for the data type of the image at texture index `t`, as an M3 Max (macOS
    27.0.1) and a Radeon Pro 5300M (macOS 26.7.1) answer. Metal refuses the queries of a buffer or
    a depth image, and the crate decodes none (`UnknownIntrinsic`); without the mode it refuses
    each query (`Unsupported` with `Lowering(Texture)`). Apple's OpenCL binds a 2D texture of
    height 1 for an `image1d_t`, which the AIR declares `texture1d`, and a 2D array texture of
    height 1 for an `image1d_array_t` (`texture1d_array`). In the mode the kernel's slot is that
    texture's type (`TextureType::Type2D`, `Type2DArray`), and each read, sample and write of the
    image, `get_image_width` and `get_image_array_size` is the 2D texture's at row 0 (a sampled
    coordinate at the row's centre), as both Macs answer. A real 1D texture bound for an
    `image1d_t`, or a 1D array texture for an `image1d_array_t`, which Apple's OpenCL never binds,
    takes the kernel's variant that declares that slot as the 1D texture
    (`RuntimeFact::OpenClLineTexture`, `VariantKey::open_cl_lines`): each read, sample and size
    query answers the 1D texture's values, of the layer an array access names, as a Radeon Pro
    5300M (macOS 26.7.1) answers (an Intel UHD Graphics 630 reads layer 0 of the array, an M3 Max
    texel 0 at every coordinate). Apple's OpenCL binds an image of most
    channel orders as a swizzled view (on an M3 Max, macOS 27.0.1, with `CL_UNORM_INT8`: `CL_R`,
    `CL_RG`, `CL_A`, `CL_RA` and `CL_INTENSITY` as `R8Unorm` or `RG8Unorm` views, `CL_BGRA`,
    `CL_ARGB` and `CL_RGB` as `RGBA8Unorm` views). A read (with a sampler or without) answers the
    swizzled texel, a write stores it with the swizzle undone (each channel from the first lane
    whose source it is), and a query answers the view's size and the hidden buffer's words, which
    no swizzle reaches, as plain Metal answers for the same view; the order is never applied twice.
    A storage slot the kernel only queries binds the texture's view with no swizzle. The reflection
    states each image argument as the texture bound for it, an `image1d_t` `TextureType::Type2D`
    and an `image1d_array_t` `Type2DArray`, of data type `DataType::None` (the AIR names its component
    `unknown`) whatever the kernel reads it as, as an M3 Max (macOS 27.0.1) answers for 1D, 1D
    array, 2D and 3D images read as float, int, uint and half, written, only queried and unused.
  - OpenCL C's `native_<function>` and `half_<function>` of `float` types
    (`air.native_<stem>.<type>` and `air.half_<stem>.<type>`, for `cos`, `exp`, `exp10`, `exp2`,
    `log`, `log10`, `log2`, `powr`, `rsqrt`, `sin`, `sqrt` and `tan`, scalar and of every vector
    width) are the function's fast form, as `fast::<function>` is (`Precision::Fast`). OpenCL C
    leaves their precision to the implementation, and Metal's devices differ:
    `native_powr(100, 0.5)` is `0x1.400002p+3` on the M3 Max and `0x1.4p+3` on the Radeon Pro
    5300M.
  - OpenCL C's vector data functions (`vload<n>`, `vstore<n>`, `vload_half[<n>]`,
    `vloada_half<n>`, `vstore_half[<n>][_<rounding>]` and `vstorea_half<n>[_<rounding>]`, which
    the compiler spells `air.vload[a][_half].<space>.<type>` and
    `air.vstore[a][_half].<space>[.<rounding>].<type>`, `VectorData`) read and write `n` elements
    at `p + offset * n` (`offset * 4` for `vloada_half3` and `vstorea_half3`), each at its own
    element's alignment, in every address space and of every element type the compiler spells.
    `vload3` of a `uint` pointer one element past a buffer's start reads the three words that
    follow, as the M3 Max, a Radeon Pro 5300M and a UHD Graphics 630 (macOS 26.7.1) do.
    `vstorea_half3` writes three halves of each four, as the M3 Max and the UHD Graphics 630 do
    (the Radeon Pro 5300M's writes the fourth too). The `half` forms widen each `half` read to
    `float` and narrow each lane stored to `half`; a store's rounding is an accepted difference
    (section 10, Conversions).

## 4. A render pipeline

```rust
let mut vertex_descriptor = VertexDescriptor::new();
let layout = VertexBufferLayoutDescriptor::new(Some(24));
vertex_descriptor.layouts.insert(0, layout);
// ...
let mut descriptor = RenderPipelineDescriptor::new(StageSource::Function(vertex));
descriptor.fragment = Some(StageSource::Function(fragment));
descriptor.vertex_descriptor = Some(vertex_descriptor);
let color = ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm);
descriptor.color_attachments[0] = Some(color);
// ...
let pipeline = device.compile_render_pipeline(&descriptor, &Budget::standard())?;
```

A `RenderPipeline` holds:

- `stages`: vertex, then fragment when rasterization is enabled;
- `layout`: one pipeline layout for every stage;
- `vertex_input`: Vulkan bindings and attributes. Outside the identity domains every attribute is
  pulled in the shader through a checked descriptor view: nothing bounds fixed-function input
  without buffer robustness (VUID-vkCmdDraw-None-02721), which the translator never leaves an
  access to. Under the identity domains nothing is pulled, as the pulled fetch reads through a
  descriptor: every attribute is fetched in fixed function
  (a fetch outside its buffer reads what the device gives), an unnormalized signed-integer format
  into a variable declared `int`, the constant step at vertex rate on a binding of stride 0 (no
  divisor); an attribute fixed-function input cannot take is refused,
  `Unsupported{VertexStepRate}` (a per-instance rate above 1, naming the missing
  `vertexAttributeInstanceRateDivisor` or the `maxVertexAttribDivisor` it passes; a device that
  lacks only `supportsNonZeroFirstInstance` takes the divisor, and the consumer refuses each draw
  through it at a base instance other than 0),
  `Unsupported{Format}` (a format the device cannot fetch as `VERTEX_BUFFER`,
  and `FloatRGB9E5`) or `Unsupported{VertexInputLayout}` (an offset or stride off the format's
  alignment or past the device's limits, more attributes or layouts than its limits, or the
  constant step's stride of 0 under the portability subset);
- `graphics`: the fixed-function state (topology, attachment formats and blend states, sample
  count, dynamic states, input attachments, ...). Build
  `VkPipelineColorBlendStateCreateInfo::pAttachments` from `graphics.color_blend_attachments()`.

Where Metal's rules differ from Vulkan's, the pipeline gives Metal's answer:

- A color attachment no fragment output names keeps its texels (an empty write mask).
- What Metal refuses, the link refuses as `InvalidDescriptor(ColorAttachment)`: an output with
  fewer components than its attachment's channels, a blend that reads the second output without
  the first, alpha to coverage over an integer attachment 0. Function-constant values that put two
  outputs at one attachment are `InvalidDescriptor(DuplicateColorOutput)`, and values that put two
  arguments of one argument table (buffer, threadgroup memory, texture or sampler) on overlapping
  slots are `InvalidDescriptor(OverlappingArgument)`, naming the later argument, of any stage:
  Metal refuses both when it makes the function.
- `ColorAttachment`'s `metal` is how plain Metal answers the pipeline (measured on an Apple M3 Max
  under macOS 27.0.1): `MetalAnswer::NilWithError`, nil and an `NSError` (code 3), for an output or
  color input of another numeric class than its attachment, an output with fewer components, a
  blend of the second source without both outputs, and a color input or tile imageblock member of
  an attachment without a format; `MetalAnswer::EndsProcess`, Metal's descriptor validation
  asserting, for a blend into a bfloat attachment and for alpha to coverage over an integer
  attachment 0. A layer that answers as Metal does ends the process only for the second.
- On a device without `independentBlend`, attachments the function does not write copy the blend
  state of those it does; where those differ the pipeline is refused, naming the feature.
- A packed 16-bit attachment (`A1BGR5Unorm`, `ABGR4Unorm`, `BGRA4Unorm`) of a format the device
  samples but does not render (the RTX 5080) renders through an `R16_UINT` view of its texture
  (`metal::format::WordAttachment`, `LoweringInputs::word_colors`): the fragment function writes
  the word Metal stores and `graphics` gives the attachment that format; a pass clears it to the
  word of Metal's clear. Exact, for a write unblended (or blended one and zero). Another blend
  and a mask of some of its channels are refused as the device lacking the format; a color input
  of one is `NotImplemented(Format)`.
- An 8-bit sRGB attachment of one or two channels (`R8Unorm_sRGB`, `RG8Unorm_sRGB`) the device
  does not render (the RTX 5080: it renders neither `R8_SRGB` nor `R8G8_SRGB`, and holds the
  latter in its UNORM twin) renders through an `R8_UINT` or `R16_UINT` view of its texture the
  same way: the fragment function writes the bytes of the device's sRGB encode
  (`DevicePackConversion`), and a pass clears it to the bytes of Metal's clear. A partial mask, and
  a blend of one other than one and zero, are refused as the device lacking the format; a color
  input of one is `NotImplemented(Format)`.
- An `RGB9E5Float` attachment the device does not render (the RTX 5080) renders through an
  `R32_UINT` view of its texture: the fragment function writes Metal's word of a `float` output
  (each channel clamped to `[0, 65408]`, a NaN 0, the exponent of the largest, every mantissa
  truncated), and a pass clears it to the word of Metal's clear. Exact. A `half` output to one
  is `NotImplemented(Format)` (Metal rounds the `float` the `half` was converted from, half up);
  a blend, or a mask of some of its channels, is refused as the device lacking the format; a
  color input of one is `NotImplemented(Format)`.
- A multisampled texture of one of these formats, on a device that samples the format but
  renders it at no count (the RTX 5080), is held in its word's format (`R8_UINT`, `R16_UINT` or
  `R32_UINT`) wherever the device renders that format at the texture's count:
  its samples are the words Metal stores, read through the format's own view as Metal reads them,
  and copied bitwise. Its resolve is the crate-owned one, a pass after the pass that reads the
  stored words: Metal's for `RGB9E5Float` (each channel's mean of the samples'
  values, rounded to the nearest `half`, then encoded with its mantissas rounded half up) and an
  8-bit sRGB format, every pixel; for a packed 16-bit format each field's mean of the samples'
  codes, a tie rounded down (a one-bit field's up), within one code of Metal's, which averages its
  tile's finer values. A device whose word's format renders one sample at that
  count refuses the texture (the runtime's `Limit::TextureSamples`).
- A fragment function that reads a color attachment (framebuffer fetch) reads it as an input
  attachment without rasterization-order attachment access, on every device, and the consumer
  records a barrier of the subpass's dependency on itself, by region, before each draw that reads
  input attachments after an earlier draw of the subpass, or its load operations, wrote a color,
  and before each draw that writes a color after an earlier one read them: a fetch reads what every
  earlier draw stored, Metal's answer, but the overlapping fragments of one draw are not ordered, which the fragment stage
  records (`PrecisionNote::UnorderedFetchWithinDraw`; the runtime does all of this itself). Under
  the identity domains the read is the pushed texture of section 1 instead, at the fragment's own
  pixel, with the same note: the consumer reads the attachment's texture there and orders each
  fetching draw after the earlier draws of its pass. Linking refuses an attachment the read
  cannot take: none attached, or one of another numeric class than the read's
  (`InvalidDescriptor` at the color attachment), or one of a rotated, Y'CbCr or extended-range
  format (`Unsupported{Format}`). A read of one of Apple's extended-range formats (`BGR10_XR`,
  `BGRA10_XR`, `RGB10A8_2P_XR10` and their `_sRGB` twins), and a blend into one, are refused on
  every device (`Unsupported{Format}` at the color attachment): Metal's tile holds
  such an attachment as `half` through the render pass and rounds it to the codes only at the
  store, so a read or a blend after an earlier draw's write sees that draw's `half`, which the
  stored codes do not keep. An unblended write, under any write mask, is built. A write the
  function would compute over its own read of the attachment, where no Vulkan blend or write mask
  gives Metal's, is not built (`Unsupported{FramebufferFetch}`, not implemented, on every device):
  a padded extended-range attachment's blend, a write of some of the codes one word of an
  extended-range attachment holds, and a blend no Vulkan state gives on an attachment held with
  its channels rotated. A raster order group on a color input, or on a color output of an
  attachment the function reads, orders that read alone and takes no interlock; a group on
  anything else needs the pixel interlock (`Unsupported{RasterOrderGroups}`). A `[[stencil]]` output uses
  `VK_EXT_shader_stencil_export`; a device without it refuses the function
  (`Unsupported{StencilExport}` naming the extension).
- `[[barycentric_coord]]` reads `BaryCoordKHR` or `BaryCoordNoPerspKHR`, whose vertex order is
  Vulkan's first-vertex order and Metal's for every primitive type. A `float3` read takes the
  first two weights and computes the third as `1 - (x + y)` in `float`, as Metal does, not the
  device's own third lane. On a device without `fragmentShaderBarycentric` the function is
  refused, `Unsupported{Barycentrics}` needing the feature.
- A vertex function stores `Layer` or `ViewportIndex` as it writes the index, on a device with
  `shaderOutputLayer` or `shaderOutputViewportIndex`; on a device without the output it is
  refused needing it, as `Unsupported(LayeredRendering)`, or as
  `Unsupported(VertexAmplification)` for an index the amplification's offset alone sets. Metal
  requires the same value at every vertex of a primitive ("If values differ, the behavior and
  value passed to the fragment function are undefined.", the Metal Shading Language
  specification), and Vulkan the same of `Layer` and `ViewportIndex`
  (VUID-VkPipelineShaderStageCreateInfo-stage-02596 and -02597); where a program breaks that, the
  layer or viewport is the device's choice (section 10). An index the function declares and never
  writes (`undef` in every value it returns), which Metal gives no value, is stored as 0, the same
  at every vertex. The stored `Layer` is 0 at or past the pass's
  length, which each draw pushes, so in a pass of one layer every index renders slice 0, as
  Metal's does. A device without `multiViewport` draws through one viewport, so there the vertex
  function stores no viewport index at all; a draw that selects among more than one viewport
  there is refused at the draw, `Unsupported(LayeredRendering)` needing `multiViewport`. A
  fragment function that reads an index no stage writes, on a device that can declare neither
  builtin, reads 0, Metal's value with no vertex write.
- `[[primitive_id]]` counts the draw's primitives in each instance from 0; vertex start, base
  vertex and base instance do not move it. A fragment stage declares `PrimitiveId` through
  `geometryShader` or `tessellationShader`. Without both it reads a flat varying the vertex
  function writes (`IoRole::PrimitiveId`, `PushField::VerticesPerPrimitive`); an indirect draw's
  arguments are then read on the CPU only without `shaderDrawParameters`. An indexed draw runs
  de-indexed (`VariantKey::primitive_id_indices`: needs device addresses, shades every index).
  `NotImplemented(PrimitiveId)`: an indexed strip, an indirect indexed draw, indices past their
  buffer, a vertex function that writes memory in that form, and a mesh pipeline.
  A draw of Metal's private quads or quad strip, which the consumer lists as two triangles a quad
  runs the fragment variant that reads `PrimitiveId` shifted
  right by one (`VariantKey::quad_primitive_ids`), as Metal counts one primitive a quad. A
  private type's draw whose vertex function writes the varying, or writes memory (Metal runs it
  once for each vertex of its list, where an indexed draw may run a shared vertex once), runs the
  private list form (`VariantKey::private_list`; needs device
  addresses): the consumer draws its list without indices from its first index as first vertex,
  and the vertex function reads each position's index, and Metal's base vertex in the word before
  the list, at the list's address (`PushField::CaptureIndexAddress`), adding them as Metal's vertex
  ID wraps, so each position runs once and the positions count Metal's primitives.
- A device draws to at most the lesser of its `maxColorAttachments` and
  `maxFragmentOutputAttachments` color attachments at once (eight at the runtime's floor), where
  Metal takes eight. A pipeline with an attachment at an index at or past the first is refused
  `InterfaceComponents` at `MaxColorAttachments`, and a fragment function with a color output at
  a location at or past the second at `MaxFragmentOutputAttachments`.
- A pipeline's band variant (`VariantKey::line_band`, of a `BandPrimitive`) draws each line, or
  each edge of a triangle in fill mode `Lines`, as the parallelogram Metal draws for a line under
  Apple's `setLineWidth:` at a width the device's own lines do not draw (the runtime chooses
  where: its README's line widths). Link adds two stages after the vertex stage
  (`pipeline::link::band`): a tessellation control stage that passes the primitive's vertices
  through, and an evaluation stage that places each vertex of a quad patch at an endpoint, moved
  across the segment along its minor axis by half the width in framebuffer pixels. A line's patch
  is two triangles over its parallelogram, front-facing under `LineBandState::FRONT_FACE`; a
  triangle's is the fans of the quad domain's degenerate inner ring over its three edges'
  parallelograms, which face as the triangle does under the draw's front face (a zero-area
  triangle faces back). `graphics.line_band` (`LineBandState`) states the draw's contract: each
  primitive one patch of its vertices (`PATCH_LIST`, `BandPrimitive::patch_control_points`), a
  line's without culling under front face `LineBandState::FRONT_FACE` and without depth bias, a
  triangle's under the draw's cull mode, front face and depth bias, and the band's `[sx, sy, h, f]`
  pushed to the evaluation stage at `push_offset` (Vulkan viewport 0's half
  extent, the half width, and for triangles 1 under a clockwise front face, -1 under a
  counter-clockwise one). A smooth varying and the clip distances are each endpoint's, a flat or
  integer varying the first vertex's. It needs `tessellationShader` (`Unsupported{WideLines}`
  needing it); a post-tessellation vertex function, a vertex function that writes a layer or
  viewport index, a fragment function that reads barycentric weights and multiview are not
  implemented.
- `graphics` also states what the runtime needs to record draws correctly: whether the fragment
  function reads flat inputs, barycentrics or its position's depth, whether the vertex function
  writes memory or a layer or viewport index, and which colors the function may write.
- On a device with `VK_KHR_portability_subset` (MoltenVK), each of the subset's missing features
  is checked against what the pipeline uses.

`metal::state` maps the encoder-side Metal state a pipeline does not own (cull mode, winding,
depth/stencil state, ...) to Vulkan values.

### Tessellation

A post-tessellation vertex function makes a tessellation pipeline:

```rust
let vertex = FunctionDescriptor::new(library.function("tess_quad_buffer")?);
let mut descriptor = RenderPipelineDescriptor::new(StageSource::Function(vertex));
descriptor.tessellation = Some(TessellationDescriptor::default());
let pipeline = device.compile_render_pipeline(&descriptor, &Budget::standard())?;
```

Its `stages` are a synthesized vertex and tessellation control stage, the function as the
evaluation stage, and the fragment stage. It draws patches only (`drawPatches:` and its
indirect and indexed forms), with the factors from a tessellation factor buffer. The device's tessellator
orders the triangles (`PrecisionNote::TessellationTriangleOrder` where it is not Apple's).

A device without `tessellationShader` refuses the function (`Unsupported{Tessellation}`, naming
the feature), and a fragment `[[primitive_id]]` behind it is `NotImplemented(Tessellation)`.

### Mesh pipelines

`Device::compile_mesh_pipeline` compiles a `MeshPipelineDescriptor` (an optional object function,
a mesh function and a fragment function). It draws mesh draws only (`drawMeshThreadgroups:`,
`drawMeshThreads:` and their indirect forms). The
stages are task and mesh shaders, which need `VK_EXT_mesh_shader` with `meshShader` (and
`taskShader` for an object function). A device without them (MoltenVK, older ANV) refuses the
pipeline and each of its object and mesh functions (`Unsupported{MeshShading}`, naming what it
lacks).

An object or mesh function that calls a SIMD-group, quad-group or SIMD-group matrix function is
refused on every device (`Unsupported{SubgroupOperations}`, `QuadOperations` or `SimdgroupMatrix`,
naming the stage's `VkProperty::SubgroupStage`), a device that lists the task and mesh stages in
`subgroupSupportedStages` included: the translator models no subgroup facts for those stages. Metal
gives both stages `[[thread_index_in_simdgroup]]`, which compiles, the lane derived from the linear
thread index as a kernel's is.

The mesh lowering is only compiled and validated: no mesh draw it makes has been compared with
Metal on a GPU, and the runtime makes no mesh pipeline (its README's known gaps of render pipeline
states). What follows is what the lowering emits.

A grid Metal draws nothing for (an axis past 65,535 threadgroups, or more than 1,048,575 of them,
measured on the M3 Max) draws nothing: an object function's wrapper launches nothing for such a
grid.

An object grid runs at any count Metal runs, past the task stage's limits too, on a device with
device addresses, `shaderInt64`, `multiDrawIndirect` and `drawIndirectCount` (every committed
profile with the extension): the object function is compiled folded, its object stage reading the
threadgroups Metal runs and its grid from a 32-byte region record at its
`PushField::ReplayRegionAddress`, so a draw runs its grid in pieces within the task limits. Metal's
threads per grid wrap to 32 bits along each axis, as its builtins report them. The lowering takes
a grid of up to 2^29 threadgroups on every device, the most Metal's GPU ran within its watchdog's
5 seconds; Metal's GPU discards a larger one and fails its command buffer with Metal's error.

### Tile pipelines

`Device::compile_tile_pipeline` compiles a `TileRenderPipelineDescriptor` (the tile function and the
render pass's color attachment formats) into a `ComputePipeline` that stands in for
`dispatchThreadsPerTile:`. Dispatch it as threadgroups over the render area, one threadgroup per
tile, and bind each attachment it reaches as a storage image at its `ResourceKind::TileAttachment`
binding.
Its threads are numbered as Metal's tile dispatch numbers them, for every threadgroup of even sides
up to 32x32 and the single thread; a threadgroup with an odd side, whose quads Metal leaves partly
empty, is refused. A tile function runs on a single-sample pass: a multisample
imageblock's operations (its sample count, the colors a pixel holds and their coverage masks, a
store to the samples a coverage mask selects) are refused as not implemented
(`Feature::Imageblocks`, naming the intrinsic).

## 5. The interface

The pipeline states what it binds and how:

```rust
for (slot, interface) in &pipeline.interface.slots {
    println!("  slot {slot:?}: {interface:?}");
}
for set in &pipeline.layout.sets {
    for binding in &set.bindings {
// ...
for dependency in &pipeline.dependencies {
    let (fact, mechanism) = (dependency.fact, dependency.mechanism);
    println!("  depends on {fact:?} via {mechanism:?}");
}
```

- `interface.slots` holds every statically used Metal slot (`SlotRef`: stage, table, index): how
  the shader accesses it, and for a buffer the byte window it can touch. A buffer slot's `kind`
  says whether Metal binds a buffer, a visible-function table or an intersection function table
  there.
- Under the identity domains (section 1) a buffer slot's `PushField::SlotGuestBase` is its
  address, the bound buffer's `gpuAddress` plus the bind offset (or the address of its
  `setBytes` copy, or of the consumer's nil buffer where nothing is bound), less 2^31, modulo
  2^64. An access through the slot adds its 32-bit byte offset plus 2^31, modulo 2^32,
  zero-extended, to that word; a pointer compare, a pulled vertex fetch and an access from the
  word below it add the 2^31 back to read the address itself, and so does a slot pointer
  converted to an integer or stored, which is the slot's address plus its offset. A runtime
  writes the word so. A pointer loaded from memory (an argument buffer's pointer member, a `gpuAddress` the
  program stored) and a pointer made from an integer are the device address the word holds,
  unbiased and used directly, unchecked: there is no root table, no `PushField::RootTableAddress`
  and no search. A texture's or sampler's `SlotGuestBase` is its `gpuResourceID`, unbiased. A
  device-replay variant (`VariantKey::replay_bindings`, section 6) reads each buffer slot's
  address from its replayed command's root entry, at `PushField::RootTableAddress`, which the
  identity domains do not read: it is refused, `Unsupported{IdentityDomains}` naming that field.
- Under the identity domains, a buffer slot that an access may reach at a
  byte offset that is not a translation-time constant (an index the memory plan does not prove
  one value, a looping `memcpy` or `memset`, a pulled fetch) also reads
  `PushField::SlotBounds`: a `uvec2`, the bound buffer's bytes before the bind offset (the bind
  offset), then the buffer's length; 0 and the length of its `setBytes` copy; 0 and 0 where
  nothing is bound. Each such access whose offset the module does not hold as a constant is
  checked against the buffer: for an `n`-byte access at its 32-bit byte offset `m` from the bind
  offset, read as signed, the normative check `(len >= n) && (m + before <= len - n)` in `u32`.
  The domain is the bound buffer, as Metal's is and as a storage view of a whole window's is
  (from the buffer's start): an offset before the bind offset is inside where the buffer holds
  it, and one before the buffer's start wraps to at least 2^31, outside. The module computes it
  as `m + before < limit`, the slot's limit for `n`-byte accesses `len - min(len, n - 1)` (0 for
  a slot shorter than one access), made once per slot and size in the function's first block:
  one add and one compare per access. A load outside reads each of its units at
  `PushField::ZeroAddress`, a `u64` the consumer pushes: the address of 16 bytes of zeros, a
  multiple of 16, that nothing writes. That is one select of each unit's address, with no select
  of the value and no branch, and the load touches nothing of the slot. A store or an atomic
  outside sits in the check's selection and runs nothing; an atomic's answer there is 0. Nil and a
  slot shorter than one access push a length that every such access past it fails. An access at
  a constant offset, and an access through a pointer loaded from memory or made from an integer,
  is not checked. A runtime writes `SlotBounds` from the bound buffer; `ZeroAddress` is the
  runtime's own memory.
  What the check costs, measured on the runtime against the same build with the check
  removed (release, no validation layer): in the compute harness's GPU time, `wide_bindings`
  34.7% slower on a Radeon 780M (job m3-max-56485), 38.5% on an RTX 3090 (m3-max-56486) and 68%
  on an Iris Xe (m3-max-56546); `bind_2000` 17.9%, 32.1% and 11.3%; `constant_managed` and
  `constant_private` 8.0% and 8.3% on the 780M, 22.2% and 25.8% on the 3090, level on the Iris
  Xe; `row_cells` level on the 780M and the 3090, 9.5% on the Iris Xe. A recorded replay
  stream's GPU time a frame, its players built from 653014c278 (the same jobs): 3.3% more on the
  780M over 438 frames (one round), level on the 3090 over 464 frames, and 5.5% more on the Iris
  Xe over 200 frames (m3-max-56641). Making two recorded sessions' render pipelines cold takes
  3.5% and 3.8% more driver time on two RTX 3090s with the check than without it (players built
  from 84b4f52e4b, jobs m3-max-58076 and m3-max-58098).
- The layout numbers set 0 densely, in `ResourceSlot` order. Set 1 holds the heap arrays (the
  resources reached through argument buffers and resource IDs); `pipeline::heap_sampler_capacity`
  is the most elements a set-1 sampler array has on a profile. A texture buffer reached so is an
  element of a texel-buffer class: a uniform texel buffer array where the function only reads it,
  a storage texel buffer array declared without a format where it writes it, each needing the
  texel buffer kind's non-uniform indexing and update-after-bind features (a storage one also
  `shaderStorageImageWriteWithoutFormat`, and where it is read `shaderStorageImageReadWithoutFormat`
  or, without it, the formats' own read bits, section 10). A heap set the consumer keeps for
  every command holds each live texture at its resource ID, texture buffers among them. Push
  constants carry the values the runtime writes per command. A stage that writes the command
  buffer's status (a
  `seq_cst` fence, an argument buffer's cubic sampler) reaches it by device address, a push hole, where the device addresses buffers
  (`bufferDeviceAddress` and `shaderInt64`), and through a set-0 storage buffer of its own
  (`SyntheticBinding::ExecutionStatus`) where it does not. That binding counts against the
  stage's storage-buffer limits: past one, an absent read or a fence is refused
  (`DescriptorCount`), and a stage that would bind it only to test its argument samplers for a
  cubic filter binds nothing.
- A heap texture sampled through one of several constexpr samplers, the one a sampler handle names
  at run time (a table of constexpr samplers read at a computed index), samples once through a
  set-0 array of those samplers (`ResourceKind::SamplerArray`, `BindingSource::SamplerArray`),
  at the element the handle names. The binding's `immutable_samplers` holds each element's sampler,
  `count` of them, each made as a heap texture sampled through that sampler alone bakes it; it takes
  no descriptor write, and a pushed set 0 pushes them. `StageInterface::sampler_arrays` lists each
  array's constexpr samplers by its slot. The module takes this form where the candidates' handles
  are consecutive, each candidate's sampling is the same but for its sampler, and the device indexes
  a sampler array by a value that may differ between invocations
  (`shaderSampledImageArrayNonUniformIndexing`, which heap textures already need); otherwise it
  switches over the candidates, one sampling per candidate. Under the identity domains, whose
  stages bind no set-0 sampler, the same form samples once through the set-1 sampler array of the
  use, at the index the consumer pushes for the candidate the handle names (one
  `PushField::ConstexprSampler` per candidate), where the sampling also reads a float class
  without comparison and in normalized coordinates; otherwise it switches over the candidates,
  each through the constexpr sampler the consumer pushes.
- Each binding's `variables` is the most variables one stage's entry point uses at it (those its
  `OpEntryPoint` lists): a function that reaches a buffer through several types uses a variable of
  each type at the buffer's binding. A set-0 uniform or storage buffer binding may be created as its dynamic type
  (`VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC`, `..._STORAGE_BUFFER_DYNAMIC`), its descriptor
  written at offset 0 and its range's offset bound as the dynamic offset: the module reads the same
  bytes, and commands that differ only in offsets can bind one set. The runtime does so for each
  such binding of one descriptor and one variable that no stage the pipeline adds reads, in binding
  order within `maxDescriptorSetUniformBuffersDynamic` and `maxDescriptorSetStorageBuffersDynamic`,
  where it does not push set 0; the rest stay of their own types. Keep a binding of several
  variables static on MoltenVK 1.4.1: its SPIRV-Cross writes Metal that does not compile for a
  dynamic buffer an entry point uses through more than one variable.
- On MoltenVK, push set 0 (a push descriptor set layout) for a pipeline whose fragment stage writes
  a storage buffer of set 0 inside its interlock (an interlock execution mode, and a `StorageBuffer`
  variable of set 0 not decorated `NonWritable`): binding through Metal argument buffers, MoltenVK
  packs a bound set 0 into one argument buffer, and Metal leaves such a buffer out of its raster
  order group where the function also reads another buffer of that argument buffer, so overlapping
  fragments of a draw read it unordered (measured on an M3 Max with 1.4.2). A pushed set's buffers
  are discrete arguments, which Metal orders. MoltenVK's pushed sets do not hold their buffers'
  sizes, so bind set 0 there for a pipeline a stage of which reads a buffer's length
  (`OpArrayLength`).
- `dependencies` lists the bind-time state the pipeline depends on (`RuntimeFact`: threadgroup
  sizes, sampler coordinate modes, buffer ranges, ...) and how this profile honors each
  (`Mechanism`: a specialization constant, a push field, a plan-time check, a variant), which a
  runtime honors at each draw or dispatch.
- The format class of a bound texture is read by the shader as push data, not specialized, so binds
  that differ only in their textures' formats share one `VkPipeline`. On MoltenVK a slot read with a
  `half` result also specializes one bit of it: whether the class converts its texels to `half` by
  rounding to nearest even alone (every class but those with 32-bit float channels, 16-bit
  normalized ones and YCbCr fields on the grid `8192`), so a slot bound to both kinds of format
  takes two pipelines. Other bound values (a slot's presence, a sampler's word) are
  specialization constants. A buffer slot's storage view reads at its residual, its range from
  the buffer's start, so the bytes before a bind offset stay reachable.
- A texture-buffer slot compiled under `LoweringInputs::texel_buffer_parts` (its function only
  reads the texture buffer, not through a sampler, writes it, updates it by an atomic or asks its
  width, on a device that indexes an array of the slot's kind of texel buffer by a `NonUniform`
  value: `RuntimeFact::TexelBufferParts`) is an array of that many texel buffers, each part but
  the last of `P` texels, the device's `maxTexelBufferElements` rounded down to a multiple of
  `minTexelBufferOffsetAlignment`. The function reads texel `i` as texel `i - k · P` of part
  `k = min(⌊i / P⌋, parts - 1)`; a texel past the last lies past the last part's end, so its
  read is Metal's out-of-bounds texel and its write is dropped, as through one view, and the
  width is `(parts - 1) · P` plus the last part's. Each part is the device's own view in the
  format one view takes, so a read converts its texel as that view does, and no precision note
  is recorded. The runtime binds a texture buffer as one view and refuses a width past the limit
  (`Limit::TextureSize`).
- `interface.reaches_command_buffers` and `interface.texture_writing_stages` say, provably, which
  pipelines may touch an indirect command buffer and which stages may write a texture.
  `interface.command_writes` (`CommandWrites`) says what its functions may write into the
  records they reach: the indirect-command operations they may call, and the primitive types
  their draws may pass (a draw's constant type, or all five where it is not a constant). A clear
  bit is proven.
- `interface.writes_memory` says, provably, whether a pipeline's functions write memory another
  invocation or a later command can read: a store to a buffer or through a device address, a
  texture write, or an atomic update. A compute pipeline without it has no effect a program can
  observe, so a runtime may run nothing for its dispatches.
- `interface.reads_unbound_resources` and `interface.writes_unbound_resources` say, provably,
  whether a pipeline's functions may read or write a resource through anything but their bound
  slots: through a pointer or texture they load from memory or make from an integer (an argument
  buffer's resources, an address stored in a buffer, a resource ID), an indirect command buffer,
  ray tracing, or a call the function cannot see into. Metal requires such resources to be
  declared with `useResource` or `useHeap`. With each slot's `access`, they are what a compute
  encoder needs to place a barrier only on a hazard.
- `interface.unbound_storage_textures` says, provably, whether a pipeline's functions may reach
  a texture through its storage image through anything but their bound slots: a texture write or
  atomic, or a read the operation does not declare `access::sample` or `access::read`, through a
  texture they load from memory or make from an integer (an argument buffer's texture,
  a resource ID), or an intersection or a call the function cannot see into. A runtime that
  gives some textures no storage image needs it to find such an access to one.

### Binding

The translator plans no binding and calls no Vulkan to bind. A runtime binds each pipeline from
what it states: its layout's bindings by Metal slot (`ResourceSlot`, set 0 densely in that order
and set 1's heap arrays), the push fields its stages read (`PushField`), its specialization
constants (`SpecKey`), and its `dependencies`. The runtime crate, `metal2vulkan-runtime`, is that
runtime on every device; its README states what it binds, pushes and refuses.

Textures follow Metal's texel rules: sparse textures report residency where the device can,
`Depth32Float` texels are the device's depth image's (a value it clamps or a denormal it flushes is
the device's, `PrecisionNote::DepthDenormals`), a `ulong` texture is an `RG32` unsigned image
whose two words the function joins and splits (its atomics are refused, `ImageAtomics`), and
`half` and sRGB reads give Metal's values or record
their bound (an sRGB texel is the device's own decode wherever the device samples its format,
`srgb-texel-decode`). A read's, sample's or
gather's `{T, i8}` result (the texel and its residency word) may pass through phis, selects,
freezes and member insertions and extractions, Apple's GL code generator's swizzled read among
them. A function whose effects depend on a bit of the word other than bit 0 (`resident()`'s), or
that stores, passes or returns a value holding the word, is refused (`Lowering(Texture)`): Metal
establishes no other bit.

## 6. At bind time

A kernel's dispatch runs as one or more regions, each a group count with its specialization and
push constants, and each distinct specialization is one `VkPipeline`. A kernel's
partial-threadgroups variant (`LoweringInputs::partial_threadgroups`) runs a grid that is not
whole threadgroups as the threadgroups its threads fill at the dispatch's threadgroup size, the
last along an axis Metal's partial threadgroup, so every remainder takes the same `VkPipeline`; a
kernel the variant cannot run exactly (a tile function, compute derivatives, emulated
simdgroups) is refused it, `Unsupported{PartialThreadgroups,
NotImplemented}`. On the identity domains the variant of a kernel that does not observe its
threadgroup partition (`KernelFacts::partition_oblivious`) runs its grid in tiles instead
(`RuntimeFact::GridTiles`): a direct dispatch's threads run in `pipeline::grid_tile`'s tiles, one
workgroup of `pipeline::GRID_TILE_THREADS` each with the tile as the region's nominal size
(`MultiRegion` 1), the region flagged `NONUNIFORM` where the tile does not divide the grid, and a
range of threadgroups at its own size (`MultiRegion` 0); the kernel runs the same positions with
the same answers: where the region is flagged `NONUNIFORM`, every invocation compares its
position with the region's grid and one past it returns before the body, a test inside one
branch on the flag; an unflagged region runs no such test, so a runtime flags every region of a
grid its workgroups do not divide. At `MultiRegion` 0 a runtime may run a direct dispatch of the
variant in workgroups of a size it picks, flagged where they do not divide the grid, whose
invocations past it return before the body as a tile's do; an indirect dispatch's region states
no grid and is unflagged, and all of its invocations run.
`Device::kernel_observes_partition(&kernel, &inputs, &budget, &entries)` answers, before any
lowering, what `CompiledFunction::observes_threadgroup_partition` answers for the compile under the
same inputs (from the specialized entry `entries` holds, ordered for the device where its
profile takes the lockstep order), so a runtime lowers only the variant a creation needs.
`CompiledFunction::is_image_only` (`KernelFacts::image_only`) states, from the AIR alone, a kernel
that samples, gathers or reads a texture and loads, stores and updates atomically nothing through
a `device` pointer (`constant` loads and texture writes allowed), which a runtime may run in
square workgroups. On that variant a grid of whole threadgroups, of `dispatchThreads` or
`dispatchThreadgroups`, takes the same regions unflagged (each `MultiRegion` 1, every thread
live), so one `VkPipeline` of a threadgroup size runs all of a kernel's direct dispatches. The
runtime lowers a kernel that does not observe its partition once, as that variant, and runs every
direct dispatch of it in its tiles; a kernel that observes its partition runs its whole grids,
threadgroup dispatches and indirect dispatches on its plain module at their own threadgroup size,
and a `dispatchThreads` grid that is not a whole number of threadgroups on its
partial-threadgroups variant.

### Levels a slot reads

Each texture slot of a stage states the levels of its bound view its reads reach, counted from the
view's first level (`StageModule::level_reads`, one `LevelReads` a slot): whether some read may
reach any level (`any`: a level the function computes, a level-count query used beyond the
containment of a constant level, a level-of-detail query, a sample through no sampler or with a
minimum level of detail, a storage access, or a use the analysis does not follow), the highest
constant level a fetch, size query or gather names (`named`) and the lowest whose texels a fetch
or gather reads (`lowest_read`), and its samples by the sampler each reads through (`sampled`, with the highest
and the lowest constant level of detail). A sample's level is its sampler's to bound.

### Indirect command buffers

An indirect command buffer's commands are records in ordinary buffer storage
(`bind::indirect_commands`): one format whatever the descriptor, written by the CPU. Under the
resident-table domains a function's encoder operations are refused,
`Unsupported{GpuEncodedIndirectCommands, NotImplemented}`; under the identity domains a function
writes the render commands section 1 names, into the records `pipeline::IndirectDrawLayout`
states, and is refused the same way for any other operation.
The translator replays no records. For a consumer that replays them, a render pipeline compiles
as its replay variant (`VariantKey::replay_bindings`), whose vertex and fragment functions read
each command's buffers from slot tables in memory, and as its merged variant
(`VariantKey::merged_draws`), whose vertex function finds its command in one merged draw by a
prefix of the commands' first vertices and runs the command's vertex,
instance and buffers, in the commands' order. A buffer's records start at
`bind::indirect_commands::records_address` of its ID, outside every Metal address; records may
run past later IDs' starts, and a resident set keeps each records range whole. Storage may hold a written flag per command,
a `u32` a command past the records (`bind::indirect_commands::written_flag_offset`): a consumer
whose own writes of the records set the flags may reset only the flagged records of a run of
commands, clearing each flag with its record.

## 7. Caching

The crate keeps no cache; keys and encodings let you keep one across processes. Keys are SHA-256
hashes of the inputs, `cache::SEMANTICS_VERSION` and the profile fingerprint, known before
compiling (`device.compute_pipeline_key(&descriptor)`, `render_pipeline_key(&descriptor,
&budget)`, `function_compile_key`):

```rust
let bytes = pipeline.to_bytes();
let cached = ComputePipeline::from_bytes_for_key(&bytes, &pipeline.key, &device)?;
assert_eq!(cached, pipeline);
```

A decode checks the format and semantics versions, the key, the profile fingerprint and a
checksum. Every `ArtifactError` is a cache miss: compile again and replace the entry.

Functions cache on their own, so pipelines that share a function share its compile, and a
pipeline links a cached function without decoding it again:

```rust
let kernel = FunctionDescriptor::new(sampling.function("sample_level")?);
let inputs = LoweringInputs::new();
let key = device.function_compile_key(&kernel, &inputs);
let compiled = device.compile_function(&kernel, &inputs, &budget)?;
let bytes = compiled.to_bytes();
let function = CompiledFunction::from_bytes_for_key(&bytes, &key, &device)?;
let descriptor = ComputePipelineDescriptor::new(StageSource::Compiled(function));
let linked = device.compile_compute_pipeline(&descriptor, &budget)?;
assert_eq!(linked, pipeline);
```

- A compiled function states before linking what its stage will: `stage()`, `requirements()`
  (check it against another profile with `DeviceProfile::check`), `precision_notes()` (the
  pipeline's stage adds those of its own state: a fragment stage's attachment writes, a vertex
  stage's formats its fixed-function vertex input converts),
  `reconvergence()`, `required_subgroup_size()`, `requires_full_subgroups()`, `module_stats()`
  (section 3) and `writes_memory()` (stores or atomics, which a render pipeline drawn in more than
  one color group refuses).
  `DeviceProfile::compile_entry` (a bare module's entry) also returns a `CompileReport` of what
  the compile cost.
- A render pipeline's stages compile under inputs that a decode of its functions learns.
  `render_pipeline_key` returns `RenderPipelineKeys`: each stage's function key, and the inputs to
  compile a missing stage under. So does a compute pipeline's kernel:
  `kernel_inputs_read(&kernel, &inputs, &budget, &entries)` narrows the stage-input descriptor to
  the attributes its entry reads, and the kernel keys and compiles under those, so states that
  differ only in an attribute it never reads share one compiled function. Those are the plain
  module's inputs; `inputs.kernel_variant(variant)` gives each `KernelVariant`'s (`Plain`,
  `PartialThreadgroups`, `EmulatedSimdgroups`), so consumers that lower a kernel's variants on one
  device lower one module per variant. What a creation reads of
  the descriptor past the fetch -- its layout agreement, its reflection, and `compute_pipeline_key`
  -- reads it whole.
- A key names an artifact by its inputs and versions; a content digest names it by what it says.
  `pipeline.content_digest()` hashes everything but the key and profile. A cache of driver objects keyed on the digest survives a
  release that bumped `SEMANTICS_VERSION` without changing the output. A digest identifies
  content; it authenticates nothing.
- A compiled function keeps its last relocation beside it, so a sweep of pipeline states over
  one function pair takes the same `Arc<[u32]>` on every creation.
- A memo that holds pipelines in memory and bounds itself by bytes can weigh one it never encoded.
  `pipeline.footprint()` is the bytes a `RenderPipeline` or a `ComputePipeline` keeps live --
  every word of every stage, each stage's constant blob and notes, and its own storage -- so a
  pipeline kept without
  `to_bytes()` is weighed by what it costs rather than by the zero length of an encoding that was
  not made.
- Versions move independently: the crate's (the API), `SEMANTICS_VERSION` (the output),
  `ARTIFACT_FORMAT_VERSION` (the encoding) and `RUNTIME_VERSION` (what the runtime does with an
  unchanged artifact).

## 8. Refusals, requested inputs, and precision

Every failure is typed; match variants, never messages. A refusal says what would change the
answer:

```rust
let floor = runtime_device("runtime-floor")?;
let min_lod = FunctionDescriptor::new(sampling.function("sample_min_lod")?);
match floor.compile_function(&min_lod, &LoweringInputs::new(), &Budget::standard()) {
    Err(Error::Unsupported(refusal)) => match (&refusal.feature, &refusal.needs) {
        (feature, Needs::AnyOf(alternatives)) => {
            println!("refused {feature:?}: needs any of {alternatives:?}");
        }
        (feature, needs) => println!("refused {feature:?}: {needs:?}"),
    },
```

`Error` has these variants:

| Variant | Means | Retry |
|---|---|---|
| `Unsupported` | The device cannot run this exactly. `feature` names the Metal feature, `needs` what would change it | No |
| `NeedsInput` | The compile needs a pipeline input you did not supply | After supplying it |
| `InvalidInput` | The AIR or container is malformed | No |
| `InvalidDescriptor` | The descriptor is invalid as Metal would refuse it, or names a missing entry | No |
| `LimitExceeded` | A deterministic cap (work units, ledger bytes, output size, structural limits) | No |
| `Budget` | The deadline passed | Later |
| `Cancelled` | The `CancelToken` was triggered | Later |
| `Internal` | A translator bug (a panic, a verifier failure) | No; report it |

`Needs` says what a refusal needs:

- `AnyOf`: alternative sets of device capabilities; a device that enables every capability of one
  set compiles it.
- `NotImplemented`: a gap of this version. `NotRepresentable`: no Vulkan mechanism exists.

`location` and `symbol` are diagnostics only. A render pipeline's refusal of one attachment's
state names it (`location.attachment()`). [REFUSALS.md](REFUSALS.md) lists every refusal in the
code, with its device condition.

A compile that needs an input asks for it (`Error::NeedsInput`) instead of guessing: a storage
texture's format, a linked function, a kernel's stage-input descriptor. Supply it and compile
again. A function-constant value is no such input: a constant the request leaves undefined
compiles, as plain Metal's does (section 10, "Function constants").

A storage texture's format (`RuntimeInput::StorageTextureFormat { slot }`) comes from what is
bound: `metal::format::storage_image_choice(pixel_format, device.profile())`, placed in a
`VariantKey`'s `storage_images` and compiled as a variant.

**What to cache.** `error.retry()` says whether the same request can succeed later.

**Where a refusal comes from.** `error.basis()` (a `RefusalBasis`) says where the belief that the
input must be refused comes from: a refusal's follows from its `needs` (`Device` for what a
device lacks, `Ours` for `NotImplemented`); a cap, the deadline and a translator bug are `Ours`;
malformed input, an invalid descriptor, a requested input and a cancellation are `None`. The
runtime's `Error::basis` answers a translation error so.

**Precision.** Where an output gives a near-equivalent of Metal's answer, its stage lists it in

`precision_notes` (a `PrecisionNote`, each documenting its bound), and `reconvergence` says
whether subgroup reconvergence after divergence matches Metal's
(`Reconvergence::ImplementationDefined` where the profile lacks maximal reconvergence). A note
that holds only under some bound values is in `specialized_notes`;
`stage.precision_notes_bound(&specialization, &|key| bound(key))` gives the notes a pipeline
created under `specialization` reaches where a draw or dispatch binds the values `bound` gives.
Section 9 lists every note.

**The runtime's errors** are its own type, `metal2vulkan_runtime::Error`;

[REFUSALS.md](REFUSALS.md) lists its refusals with their basis, and its
[README](../crates/metal2vulkan-runtime/README.md) states each known gap.

## 9. Degrade notices

Where an answer may differ from Metal's, the operation carries a precision note
(`metal2vulkan::pipeline::PrecisionNote`), fixed per operation and never decided at run time: by
the translated module, the pixel format, the descriptor or the state, on every device.
`metal2vulkan-runtime` gives them through these items:

- `ComputePipelineState::precision_notes` and `RenderPipelineState::precision_notes`: the
  translator's notes of the state's modules, with `SwizzledOpaqueBlackBorder` where a function
  samples through a constexpr sampler whose address mode reads an opaque-black border;
- `Texture::precision_note`: `CompressedTexelDecode` where the texture's own pixel format is
  block-compressed;
- `Texture::FLOAT_DEPTH_UPLOAD_NOTE` (`RestrictedDepthUpload`, of an upload into a 32-bit float
  depth by `replace_region`, `upload_region` or a blit's `copy_from_buffer`);
- `SamplerState::precision_note`: `SwizzledOpaqueBlackBorder` where an address mode reads an
  opaque-black border;
- `DepthStencilState::precision_note`: `DepthDenormals` where the state tests or writes depth;
- `RenderPassDescriptor::precision_note`: `DeviceRoundedResolve` where a color attachment
  resolves;
- `BlitCommandEncoder::GENERATE_MIPMAPS_NOTE` (`DeviceFilteredMipmaps`, of `generate_mipmaps`),
  `RenderCommandEncoder::DEPTH_BIAS_NOTE` (`DeviceDepthBias`, of a draw of triangles under a depth
  bias), `RenderCommandEncoder::LINE_NOTE` (`DeviceLineRasterization`, of a draw of lines or a line
  strip, and of a draw of triangles under the `Lines` fill mode) and
  `RenderCommandEncoder::VISIBILITY_NOTE` (`DeviceOcclusionCount`, of a `Boolean` or `Counting`
  visibility result).

A note is never fatal: the work runs on every device. The runtime prints and sends no note; a
consumer reads them through these items. The translator's own API gives a stage's notes
(section 8).

Each entry below is named by a code, `metal2vulkan-degrade-` and the note's name in kebab case
(`PrecisionNote::NativeFma` is `metal2vulkan-degrade-native-fma`), with its category, and says what
makes an operation carry the note and what may differ. Each `PrecisionNote`'s rustdoc states its
bound.

- `metal2vulkan-degrade-device-rounded-resolve` (precision). `DeviceRoundedResolve`: A multisample
  color resolve averages through Vulkan's `AVERAGE` mode, which leaves the rounding of the average
  (and the sRGB encoding) to the device: any
  color attachment but an extended-range one and one resolving into its format's sRGB twin (each
  resolved after the pass). Also, on every device, a resolve into an 8-bit UNORM attachment's
  sRGB twin where the device has
  the sRGB format: the crate-owned resolve decodes the samples and encodes their average by the
  device's own sRGB conversions, where Metal's resolve takes its own 12-bit tables (with correctly
  rounded conversions, 1,758 of the 65,536 pairs of codes at 2 samples land one code from
  Metal's; MoltenVK 1.4.2 on the M3 Max: 696 of 262,144 channels at 2 samples, one code below;
  lavapipe, Mesa 26.0.8: 57,865 of 524,288 at 4, within one code). Measured against Metal's rules
  at 2 and 4 samples on RADV (Mesa 26.0.8, the Radeon 780M and the RX
  7900 XTX), ANV (Mesa 26.2.4, the Iris Xe), NVIDIA (615.71, the RTX 5080 Laptop GPU) and lavapipe
  (Mesa 26.2.4, at 4 samples): every unorm, snorm and packed 16-bit unorm
  channel within one code of Metal's, an sRGB channel within two (NVIDIA's at 4 samples; one
  elsewhere). Each rounds an 8-bit unorm tie down (RADV and ANV: 32512 of `R8Unorm`'s 65536 pairs
  of samples one code below at 2 samples); an sRGB or packed channel lands above Metal's code in
  some texels and below in others. A float channel's average differs in its last bits, counted and
  not bounded (RADV: 8565 of `R32Float`'s 65536 channels at 2 samples). Cost: a resolved channel can
  differ from Metal's by one code (an sRGB channel by two) where the average lies near a rounding
  boundary, a float channel in its last bits, and wholly where a sample is a
  NaN or an infinity: Metal's resolve counts a half or small-float NaN as 0 and a half infinity as
  65536 of its sign, and makes any 32-bit float NaN 0x7fc00000, where the device's average may
  keep them.
  On `metal2vulkan-runtime` it is the static note `PrecisionNote::DeviceRoundedResolve` of a render
  pass that resolves a color attachment (`RenderPassDescriptor::precision_note`), by Vulkan's
  `AVERAGE` in the pass's own rendering, on every device: an extended-range, bfloat or YCbCr
  attachment's resolve is `Gap::RenderPass` of `RenderPassKind::Resolve`, and an integer one is
  `Refusal::ResolveFormat`.
- `metal2vulkan-degrade-device-filtered-mipmaps` (precision). `DeviceFilteredMipmaps`:
  `generate_mipmaps` of a filtered format (an integer format's texels are picked as plain Metal
  picks them), the static note `BlitCommandEncoder::GENERATE_MIPMAPS_NOTE` on every device: each
  level is the device's linear filter of the one above at the scaled texel centres, its blit chain
  (`vkCmdBlitImage` with `VK_FILTER_LINEAR`); a format the device's blits cannot filter is
  `Gap::MipmapFilter`. The device's filter is not Apple's datapath, whose answers also differ
  between Apple's GPUs (an M3 Max's and an M4's by up to a unit in the last place). No bound
  against Metal's answer is measured. Cost: a channel can differ from Metal's where the filtered
  value lies near a rounding boundary, and wholly where a box averages infinities, NaNs or a float
  format's extremes.
- `metal2vulkan-degrade-capped-lod-bias` (precision). `CappedLodBias`: A sampler state whose
  `lodBias` the consumer has the device add to every sample's level (`SamplerUse::lod_bias`, for
  a GPU that adds it, as a Mac with an Intel or AMD GPU does; the runtime sets it nowhere and
  samples as an Apple GPU does, with no bias), where the device's samplers cannot add it as it is
  (`metal::sampler::DeviceLodBias`): past the device's `maxSamplerLodBias` (an infinity included),
  a NaN, or a bias other than 0 where the device's portability subset withholds
  `samplerMipLodBias` (MoltenVK). Cost: its samples add the device's
  limit of the bias's sign where the bias is past it, and no bias where the device withholds it or
  the bias is NaN: a sample whose level with what the device adds has not reached the last level
  the sampler reads (a positive bias) or level 0 (a negative one) reads another level than Metal's,
  which adds the whole bias and then clamps the level into the texture's levels; what Metal samples
  for a NaN bias is not measured. The translator states the same difference as the sampler state's
  static note, `PrecisionNote::CappedLodBias` (`DeviceLodBias::note`), which no pipeline stage
  records.
- `metal2vulkan-degrade-device-reconvergence` (precision). `DeviceReconvergence`: A stage holds a
  tangled instruction (a simdgroup or quad operation, or a barrier whose lanes depend on
  reconvergence) on a device without maximal reconvergence (`shaderMaximalReconvergence`): after
  divergent control flow the device decides which lanes run it together, where Metal's lanes
  reconverge maximally, so such an operation may see other lanes active than Metal's. Under
  subgroup-uniform control flow (`shaderSubgroupUniformControlFlow`) the lanes reconverge where
  control flow is uniform. Static per stage: recorded for any tangled instruction in the entry.
- `metal2vulkan-degrade-simdgroup-access-order` (precision). `SimdgroupAccessOrder`: A kernel
  holds two accesses of threadgroup or device memory, at least one a write, that different threads
  of one SIMD-group may make to the same bytes between two barriers, and no barrier or atomic
  orders them (a lane reads the slot another lane of its SIMD-group wrote one instruction earlier).
  An Apple GPU runs a SIMD-group's 32 threads in lockstep, so such a pair runs in program order
  there; the module keeps the pair as the AIR states it, and the device runs the threads in its own
  order between barriers, so the values the pair reads and leaves may be other than Metal's. The
  Metal Shading Language calls such a pair a data race, and the difference is an accepted one
  (section 10, "The threads of a SIMD-group run in no order between barriers"). It is the static
  note `PrecisionNote::SimdgroupAccessOrder` of every kernel in which the translator finds such a
  pair, from the AIR's structure alone, on every device under the identity domains
  (`metal2vulkan-runtime`). The resident-table domains order each pair with barriers of their own
  and record none.
- `metal2vulkan-degrade-snorm-minimum-kept` (precision). `SnormMinimumKept`: A draw in a render pass
  that loads a single-sampled color attachment of a signed normalized format (`R8Snorm`, `RG8Snorm`,
  `RGBA8Snorm`, `R16Snorm`, `RG16Snorm`, `RGBA16Snorm`) and does not discard it at the store. Cost:
  the texels the pass's draws do not write keep their bytes; Metal's pass rewrites a channel's
  minimum code (-128, or -32768 for 16 bits) as the code above it (-127, -32767) in each tile a draw
  reaches: a texel whose bytes no Metal write left (uploaded or copied bytes) keeps a code Metal's
  pass would change. It is the static note `PrecisionNote::SnormMinimumKept` of every
  single-sampled render pipeline that draws into such an attachment, whatever its passes load, on
  its fragment stage (its vertex stage where it has none), on every device.
- `metal2vulkan-degrade-shared-exponent-kept` (precision). `SharedExponentKept`: A draw in a render
  pass that loads a single-sampled `RGB9E5Float` color attachment and does not discard it at the
  store. Cost: the texels the pass's draws do not write keep their words; Metal's pass normalizes a
  word whose largest mantissa is below 256 (its exponent above 0) in each tile a draw reaches: a
  texel whose bytes no Metal write left (uploaded or copied words) keeps a word Metal would change,
  of the same value. It is the static note `PrecisionNote::SharedExponentKept` of every
  single-sampled render pipeline that draws into such an attachment, whatever its passes load, on
  its fragment stage (its vertex stage where it has none), on every device.
- `metal2vulkan-degrade-compressed-texel-decode` (precision). `CompressedTexelDecode`: The static
  note of every texture or view whose own pixel format is block-compressed (BC1 to BC7, and the
  ASTC, ETC2 and EAC formats where the device makes them; `Texture::precision_note`), on every
  device. The texture's image is its
  own image of that format, which the device samples through its own decode: no BC texture keeps a
  decoded twin. The bounds are the
  largest differences from Metal's texels measured on random blocks of every format; a driver not
  named is unmeasured. Validation derives no tolerance from them; the conformance suite's BC1 to
  BC5 decode cases hold each texel to its family's largest bound on any driver (an sRGB form's to
  the sRGB decode's largest step over 4/255), and its ASTC cases each linear (`_LDR`) texel to
  1/2048, plus both values' rounding to `float`, justified by the note alone.
  Cost: a read, sample or gather of its texels answers the device's decode of its blocks (an sRGB
  form's code through the device's sRGB decode), where Metal interpolates with 6-bit weights: a BC1
  to BC3 channel up to 4/255 off Metal's on NVIDIA (615.71.09, the RTX 3090 and RTX 5080 Laptop
  GPU; BC3's alpha 6/255), 3/255 on lavapipe, 1/255 on ANV; a BC4 or BC5 channel up to 7.4/255
  (signed 0.065) on NVIDIA, 1.3/255 (signed 0.013) on ANV, 2.4/255 (signed 0.022) on lavapipe, and on RADV
  (Mesa 26.2.4, the Radeon 780M) a signed texel with an endpoint of -128 up to 1.7/255; a BC7
  channel a `float` ulp on lavapipe; its blocks' bytes, copies and sizes are Metal's. An ASTC LDR
  channel is up to 1/2048 off Metal's on ANV (Mesa 26.2.6, the Iris Xe of Raptor Lake-P), over the
  fourteen footprints. Measured
  on more drivers: NVIDIA's 620.70.06 (the RTX 2080) a BC1 to BC3 channel 3/255 (BC3's alpha
  6/255) and a BC4 or BC5 channel 7.33/255 (signed 0.064); ANV (the Iris Xe of Raptor Lake-P) and
  Intel's Windows driver (the UHD 630) BC3's alpha 2/255 and a BC4 or BC5 channel 1.4/255 (signed
  0.0134), a BC4 or BC5 texel within 0.0001 of the specification's exact decode there, so what
  moves is Metal's 6-bit weights.
- `metal2vulkan-degrade-depth-denormals` (precision). `DepthDenormals`: A write that may store a
  denormal in the 32-bit float depth of a depth image (a pass's clear to one, a blit's copy into a
  `Depth32Float` or `Depth32Float_Stencil8` texture's depth, a `replace_region` whose bytes hold
  one): a driver may store one a write on the GPU gives it as a zero (RADV and ANV do; a host copy
  keeps it), and a depth test may compare one otherwise than Metal's (every driver measured but
  MoltenVK on an Apple M4). Cost: a denormal depth may read, sample and copy out as a
  zero, or the depth test may compare it as zero (a positive one) or below zero (a negative one),
  where Metal keeps it and compares a positive one above zero and a negative one not below it.
  On `metal2vulkan-runtime` it is the static note `PrecisionNote::DepthDenormals` of every
  depth-stencil state that tests or writes depth (`DepthStencilState::precision_note`), on every
  device.
- `metal2vulkan-degrade-f16-fma-double-rounding` (precision). `PrecisionNote::F16FmaDoubleRounding`:
  Widened half fma rounds twice.
- `metal2vulkan-degrade-f16-int-conv-double-rounding` (precision).
  `PrecisionNote::F16IntConvDoubleRounding`: Widened half-integer conversions round twice.
- `metal2vulkan-degrade-grad-from-bias` (precision). `PrecisionNote::GradFromBias`: Bias sampling
  emulated with gradients may pick another level and another anisotropic footprint than Metal's:
  derivatives hoisted out of a per-fragment texture choice, which scale by `2^B` through every
  sampler, and, on every device, a bias whose sum with
  the sampler's own may lie beyond the device's `maxSamplerLodBias`, which Vulkan clamps, or that an
  anisotropic sampler applies to its footprint's minor axis alone. On a device without
  `maintenance8`, a fragment's implicit-level sample with a dynamic texel offset samples with its
  derivatives too, taken before the switch over the offset's values.
- `metal2vulkan-degrade-device-transcendentals` (precision). `PrecisionNote::DeviceTranscendentals`:
  Transcendentals answer within Apple's stated bound, not Metal's bits: the device's precision for a
  call no other note bounds. Apple states no answer at a NaN operand under fast math, and where the
  device's compiler takes one math mode for the whole entry (MoltenVK) a precise function of a
  fast-math entry answers by its fast form, so a NaN operand may answer a number (`precise::sin` and
  `precise::cos` of a NaN answer `0` on the Apple M3 Max through MoltenVK 1.4.2, where Metal answers
  the NaN).
- `metal2vulkan-degrade-reduction-order` (precision). `PrecisionNote::ReductionOrder`: Float
  reductions use the device's association order.
- `metal2vulkan-degrade-device-denormals` (precision). `PrecisionNote::DeviceDenormals`: `float`
  denormals follow the device: Metal reads an `f32` denormal operand of its float arithmetic,
  comparisons, and functions as the zero of its sign and flushes a denormal result to one, where a
  Vulkan device keeps or flushes them by its own float controls (this translation declares no
  `DenormFlushToZero`). Recorded where the module leaves an operation's denormals to the device,
  on every device: `+ - *`, the comparisons, `fma`, `dot`, `sqrt`, `rsqrt`, `trunc`, `rint` and
  `ldexp` on `float` (by the device's products); `fneg`, `fabs`, `floor` and `ceil` on `float`;
  the divisions and the remainder, the device's maths functions (`fast::fmod`'s fused
  multiply-add, `sign` and `fract` among them), `mix` by `FMix`, `fast::sqrt` by `Sqrt`, `dot` of
  vectors of other than two to four lanes, whose chunks' products the device sums where Metal
  flushes each one below `2^-126`, and a conversion to or from `double`, where Metal's reading is
  not measured. The operations that read a denormal on its bits answer Metal on every device:
  `copysign` and `atan2`. Where the minimum and maximum
  functions and `saturate` are the device's `NMin`, `NMax` and `NClamp`, their pick of a denormal
  is `NativeMinMax`'s. Off MoltenVK the device's multiply and fused multiply-add, and `ldexp` on
  `float` by that multiply, round a result whose exact value is below `2^-126` up to `2^-126`, which
  Metal flushes. A nearest pixel-coordinate sample at its texel's
  centre `floor(x) + 0.5`, and a pixel fetch at `floor(x) + k` under an offset, records it for that
  `floor`: Metal's sampler reads a negative subnormal `x` as texel -1, where MoltenVK's `floor`
  reads it as the zero of its sign, texel 0 (an Apple M3 Max). An address mode that reads texel -1
  as texel 0 (clamp to edge, mirror clamp to edge, mirrored repeat) answers alike, so the sample
  records it only where its `s` or `t` mode is repeat, clamp to zero or clamp to the border: through
  a constexpr sampler of such a mode; through a runtime sampler slot only under a pipeline made for
  a nearest pixel state (`SpecKey::NearestPixelSampler`), never under a normalized one, and there
  where the bound state is of such a mode (`SpecKey::BoundSamplerWord`); through an argument
  buffer's sampler
  `argument-pixel-sampler-scaling` records it instead. A pixel fetch under an offset records it
  outright.
- `metal2vulkan-degrade-pixel-sampler-scaling` (precision). `PrecisionNote::PixelSamplerScaling`:
  A pixel-coordinate sampler's coordinate reaches the device's normalized sampler divided by the
  texture's level-0 size in the shader, and the device reads its texels and weights from that
  coordinate in its own arithmetic: within a few units in the last place of the edge of a linear
  filter's pair of texels, a gather's block or one of the filter's sub-texel steps it may read the
  neighbouring pair, block or step, where Metal's sampler reads the coordinate as given. Recorded
  for a linear sample, a gather and a level query through a constexpr pixel sampler of an array, a
  3D texture or a texture whose residency the function observes, and through a runtime sampler
  slot (where the bound state reads pixel coordinates, and for a sample or comparison where it is
  not a nearest one of one isotropic level). Measured: the device's normalized
  sampler's answer within one `f32` ulp of the quotient on MoltenVK and at the quotient on
  lavapipe; a linear downscale by 8 within 1/255 of Metal's on MoltenVK and 2/255 on lavapipe.
  Under the identity domains it is also recorded for a linear sample and a gather through any
  constexpr pixel sampler, whose device sampler the consumer makes normalized. Nothing is
  recorded where Metal's texel or footprint follows from the coordinate itself: elsewhere a
  constexpr pixel sampler's sample or gather of a 1D or 2D texture that is no array hands the
  coordinate, moved by its offset, to the device's unnormalized sampler; a nearest sample elsewhere
  hands the device its texel's centre, where the device reads exactly Metal's texel; and a nearest
  constexpr sampler clamped to the edge fetches its texel.
- `metal2vulkan-degrade-device-arithmetic` (precision). `PrecisionNote::DeviceArithmetic`: Float
  division, remainder, square root, fused multiply-add, and dot products use Vulkan's precision
  bounds (e.g. 2.5 ULP division, a possibly unfused `Fma`) rather than correct rounding.
- `metal2vulkan-degrade-device-signed-zero-inf-nan` (precision).
  `PrecisionNote::DeviceSignedZeroInfNan`: Float operations LLVM requires to keep signed zeros,
  infinities, and NaNs run without `SignedZeroInfNanPreserve`, which the device does not support at
  their width. Without the mode the device may answer such a zero with either sign.
- `metal2vulkan-degrade-device-ray-intersection` (precision).
  `PrecisionNote::DeviceRayIntersection`: A ray intersection traversed by the device's ray query.
- `metal2vulkan-degrade-fast-math-rewrites` (precision). `PrecisionNote::FastMathRewrites`: Float
  arithmetic whose AIR lets a compiler rewrite it with a neighbouring operation: an addition or
  subtraction and the multiply feeding it both carry LLVM's `contract`, two chained additions,
  subtractions, or multiplies both carry `reassoc`, or an `llvm.fmuladd` (fused or not, as the
  implementation chooses). The module states each operation in the AIR's order with its own
  flags: one carrying both `contract` and `reassoc` is the device's compiler's to group and fuse,
  any other is kept apart (`NoContraction`); Metal's compiler groups and fuses by its own choice.
  On a device without `shaderFloat16`, a chain of such `half` additions, subtractions and products
  is held at `float` precision between the roundings where its values leave it. On
  MoltenVK a `half` addition, subtraction or product carrying both flags is the 16-bit instruction
  left undecorated, whose conversions Metal's fast compile folds away: the `float` result,
  unrounded. Cost: a sum of mixed signs grouped otherwise, or a multiply-add fused on one side
  alone, differs from Metal's by any amount relative to the result; a held `half` chain, or a
  MoltenVK `half` operation left to the compiler, by the roundings it skips, a value inside it past
  `half`'s range staying finite.
- `metal2vulkan-degrade-tessellation-winding` (precision). `PrecisionNote::TessellationWinding`: A
  tessellation pipeline's vertex order: Metal's clockwise output winding is declared
  `VertexOrderCcw` and its counter-clockwise one `VertexOrderCw`, as MoltenVK maps them under
  Vulkan's default upper-left domain origin. Measured for both windings against Metal's bytes
  (recorded on an Apple M3 Max) by a consumer's device test,
  `tessellation_states_tessellate_as_metals`, on RADV (Radeon 780M, RX 7900 XTX), NVIDIA Turing
  and MoltenVK; other drivers are not measured.
- `metal2vulkan-degrade-tessellation-factor-edge-cases` (precision).
  `PrecisionNote::TessellationFactorEdgeCases`: A tessellation pipeline's factors at the edges of
  Metal's stated rules: a subnormal edge factor is positive and does not cull (its level is the
  range's lower end), where a flush to zero would cull the patch; and a NaN inside factor's level is
  the range's lower end.
- `metal2vulkan-degrade-tessellation-factor-scale` (precision).
  `PrecisionNote::TessellationFactorScale`: A tessellation pipeline with
  `tessellationFactorScaleEnabled` multiplies each factor by the encoder's scale rounded to `half`,
  exactly in `float`, after the cull.
- `metal2vulkan-degrade-tessellation-factor-instance` (precision).
  `PrecisionNote::TessellationFactorInstance`: A tessellation pipeline whose factor step function is
  per instance reads instance `i`'s factors at `i · instanceStride`, `i` counting the draw's base
  instance as `[[instance_id]]` does.
- `metal2vulkan-degrade-tessellation-fractional-spacing` (precision).
  `PrecisionNote::TessellationFractionalSpacing`: A tessellation pipeline of a fractional partition
  (`fractionalOdd`, `fractionalEven`) drawn through the device's tessellation stages: Vulkan leaves
  where a fractional spacing places its vertices to the implementation (its two shorter segments'
  length need only fall monotonically with the rounded level less the level, and their place on the
  edge is the implementation's), where Metal's tessellator places them by its own rule. Measured
  equal to Metal's, at factors within the device's range, by a consumer's device tests on RADV
  (Radeon 780M, RX 7900 XTX), NVIDIA Turing, ANV (Intel Iris Xe) and MoltenVK; other drivers,
  lavapipe among them, and factors past the range are not measured. A device without
  `tessellationShader` draws Metal's tessellator through the emulation and records none.
- `metal2vulkan-degrade-tessellation-triangle-order` (precision).
  `PrecisionNote::TessellationTriangleOrder`: A tessellation pipeline drawn through the device's
  tessellation stages (a device with `tessellationShader`), on every device (no query tells
  Apple's tessellator apart): the device's
  tessellator orders each patch's triangles, picks each one's first corner, and joins the patch's
  rings where Vulkan leaves that to the implementation. The domain points are Metal's; a flat
  varying, a value interpolated across another triangulation, and the order a patch's triangles
  blend in may differ.
- `metal2vulkan-degrade-unmeasured-group-speculation` (precision).
  `PrecisionNote::UnmeasuredGroupSpeculation`: A function holds a branch whose arms hold a quad or
  simdgroup operation that answers over every lane of the branch's block when Metal's backend
  speculates the arms (a quad shuffle, broadcast or vote, a simdgroup shuffle, `simd_is_first`),
  where no measurement says whether Metal speculates them: an arm of an instruction, size or shape
  no probe covers. The branch keeps the AIR's lanes, so that operation may answer over the lanes
  that take the arm where Metal's answers over the block's. A branch Metal was measured to
  speculate is speculated, and one it was measured to keep is kept, with no note.
- `metal2vulkan-degrade-device-sample-level` (precision). `PrecisionNote::DeviceSampleLevel`: An
  extended-range texture's (`BGR10_XR`, `BGRA10_XR` and their sRGB twins) fragment sample at an
  implicit level, or with explicit gradients, on every device: its texels are read in the shader, at
  Vulkan's level of detail computed there and Metal's level rule, where a constexpr sampler states a
  mip filter or a runtime sampler's word does. The device takes the level of detail by Vulkan's
  rules, whose scale factor may be approximated; Metal's level of detail from derivatives is not
  measured. The sample may read another level or filter than Metal's near a boundary.
- `metal2vulkan-degrade-unordered-fetch-within-draw` (precision).
  `PrecisionNote::UnorderedFetchWithinDraw`: A fragment function reads a color attachment
  (framebuffer fetch), on every device. It reads the attachment as an input
  attachment, and the runtime records a barrier of the draws' subpass's dependency on itself, by
  region, before each draw that reads input
  attachments after an earlier draw of its render pass instance, or its load operations, wrote a
  color, and before each draw that writes a color after an earlier one read them: a fetch reads what
  every earlier draw stored at its pixel, Metal's answer. Under the identity domains it reads the
  attachment's texture at its own pixel instead (`PushField::FetchedColor`), and the consumer orders
  each fetching draw after the earlier draws of its pass by a barrier between them (the runtime's
  `Metrics::attachment_loop_reads`, or its copies into the attachment's shadow on a device without
  the attachment feedback loop). Vulkan orders nothing between the overlapping fragments of one draw
  there, where Metal orders them in primitive order: a fragment that covers a pixel an earlier
  fragment of its own draw covered may read the texel from before that fragment's write. Exact for a
  draw whose primitives do not overlap at a pixel; otherwise each such pixel holds the result of
  some order of the draw's fragments there. A raster order group on the color input, or on a color
  output of an attachment the function reads, takes the same form.
- `metal2vulkan-degrade-fp8-value` (precision). `PrecisionNote::Fp8Value`: An 8-bit float converts
  to an integer from the format's documented value with Metal's saturating float-to-integer rule,
  where Metal's device here answers otherwise: Metal on Apple M3 compiles these reads but extends
  the carrier byte as an integer, the same for every format, so the result is unverified on Metal.
- `metal2vulkan-degrade-multisample-position` (precision). `PrecisionNote::MultisamplePosition`: A
  fragment function of a render pipeline that rasterizes more than one sample reads a value Metal
  reads at the pixel center in every invocation and Vulkan evaluates at an implementation-defined
  position within the fragment: `[[position]]`'s z or w (the interpolated depth and `1/w`, whose
  position Vulkan does not state), a center-sampled varying on a device without its pixel form
  (interpolation functions whose `InterpolateAtOffset` reads from the pixel center: "an
  implementation-defined position within the fragment area being shaded"), a centroid varying or an
  `interpolant` read through `interpolate_at_centroid` (anywhere in the fragment's covered area,
  where Metal, measured on Apple M3 Max, moves it to the lowest covered sample's location only under
  partial coverage), an `interpolant` read through `interpolate_at_center` as a center varying,
  `[[point_coord]]` ("the fragment center, or the location of a sample"), or `[[barycentric_coord]]`
  (its builtin). A function that runs per sample under sample shading reads all but the center
  varying where Vulkan evaluates them for its sample, where Metal reads every one as the pixel's in
  each sample's run (Apple M3 Max at 4x, measured). Each is evaluated at a point of the pixel, so it
  differs from Metal's by at most its change over half the pixel's diagonal, and a centroid varying
  by at most its change across the pixel's covered area; exact where the device evaluates a read
  where Metal does.
- `metal2vulkan-degrade-bicubic-filter-arithmetic` (precision).
  `PrecisionNote::BicubicFilterArithmetic`: a filter emulated by fetches (Metal's bicubic filter on
  every format, and an extended-range texture's linear, 3D and linear-mip-filter samples) sums its
  taps in the device's arithmetic: the Catmull-Rom weights computed in `f32` from the fraction, the
  taps summed by the device's multiply and add, a bilinear, trilinear or level blend by its `FMix`
  at the `f32` fractions, and a `half` result the device's narrowing. The footprint, addressing,
  filter choice, level and clamps are Metal's. Metal's fixed-point weights, exact grid sums, float
  accumulators and sixty-fourth level fractions are not reproduced: a lane may differ by a few grid
  steps (`1/4080` at 8 bits), by up to `2.1e-3` on an extended-range sRGB texture's bicubic sample,
  and by up to `1/64` of the two levels' difference under a linear mip filter. Measured for the
  bicubic filter by the conformance case `bicubic-texel-classes` (Metal on an Apple M3 Max, macOS
  27.0.1): Metal's results lie on the format's grid, a sixteenth of a code for `RGBA8Unorm`,
  `R8Unorm` and `RGBA8Snorm`, a whole code for `RGBA16Unorm` and an exact `half` for
  `RGBA16Float`, and at the six `RGBA16Float` samples whose sum lies exactly halfway between two
  halves Metal rounds away from zero. Ours is the unrounded `f32` sum; `R32Float`'s samples
  compare exactly. The largest difference per format, `float` and `half` results together, over
  lavapipe, the M3 on KosmicKrisp and on MoltenVK, a Radeon 780M, an RTX 3090 and an Iris Xe:
  `RGBA8Unorm` `2.44e-4`, `R8Unorm` `1.53e-4`, `RGBA8Snorm` `4.88e-4`, `RGBA16Float` `3.74e-4`
  (`4.73e-4` on lavapipe in the authored bicubic edge cases, at values in `[1, 2)` where a `half`
  step is `9.77e-4`, at normalized and at pixel coordinates alike), `RGBA16Unorm` `2.16e-4` on lavapipe, KosmicKrisp and the 3090 and `7.8e-6` elsewhere (lavapipe's
  `half` results keep their `f32` values). Through a runtime sampler slot it holds only where the
  bound sampler filters (a linear minification or magnification filter, anisotropy, a cubic filter
  or a linear mip filter): a nearest sampler's sample is the decoded texel at Metal's position.
  Through an argument buffer it holds only where the command can name a texture held by its raw
  texels in a form the function filters (`BGR10_XR`, `BGRA10_XR`, the padded forms; not a bfloat
  one).
- `metal2vulkan-degrade-sparse-residency` (precision). `PrecisionNote::SparseResidency`: A stage
  samples or gathers a texture whose residency it observes (`sparse_color::resident`) through the
  slot's sparse form, which the module has where the profile enables `shaderResourceResidency`:
  the word is the device's `OpImageSparse*` residency, whose footprint may count a texel of zero
  weight Metal does not or call a footprint of the border alone not resident where Metal answers
  resident, and an unmapped texel's lanes the device's non-resident value. Where the
  device has no residency of its own, the slot's tile map answers Metal's footprint rule exactly,
  unrecorded.
- `metal2vulkan-degrade-device-nan-encoding` (precision). `PrecisionNote::DeviceNanEncoding`: A
  `float` NaN result's bit pattern follows the device: Metal answers the canonical `0x7fc00000` for
  the NaN results of its float arithmetic and functions, and Vulkan states no encoding, so a device
  may answer another sign or payload (llvmpipe: `0xffc00000`). A texture write records it where the stored NaN is the
  device's: a constant texel none of whose lanes is a NaN records nothing, and a `half` or
  `float` write through an argument buffer holds only where the command can name a texture of a
  format family whose stored NaN is the device's (for a `half` write, a texture of any format that
  holds a NaN; for a `float` write, a 16-bit float or `RG11B10Float` one). So does a fragment
  function's color output where the device's conversion stores its NaN, but one every return of
  which holds such a constant records nothing.
- `metal2vulkan-degrade-restricted-depth-range` (precision). `PrecisionNote::RestrictedDepthRange`:
  The device compares a depth texture's texels, on every device: without
  `VK_EXT_depth_range_unrestricted`, which no runtime enables, Vulkan gives a depth image no
  value outside `[0, 1]`, whatever its driver is measured to keep: an upload of one is the
  device's (`metal2vulkan-degrade-restricted-depth-upload`), and a render write is clamped to the
  viewport's depth range. A comparison of a texel Metal holds outside `[0, 1]` can differ; one
  within it is exact.
- `metal2vulkan-degrade-restricted-depth-upload` (precision).
  `PrecisionNote::RestrictedDepthUpload`: An upload into a 32-bit float depth (`Depth32Float`, or
  the depth of `Depth32Float_Stencil8`), the device's copy into the depth aspect
  (`vkCmdCopyBufferToImage`), on every device. On `metal2vulkan-runtime` it is the static note
  `Texture::FLOAT_DEPTH_UPLOAD_NOTE` of `replace_region`, `upload_region` and a blit's
  `copy_from_buffer` into one, decided by the format, never by the bytes. Metal keeps every
  `float` there bit for bit, and its comparisons compare it unclamped (measured on an Apple M3 Max
  under macOS 27.0.1, and an AMD Radeon Pro 5300M and an Intel UHD Graphics 630 under macOS
  26.7.1: negative values, values above 1, denormals, infinities and NaNs). Vulkan takes no value
  outside `[0, 1]` into a depth aspect without `VK_EXT_depth_range_unrestricted`
  (`VUID-vkCmdCopyBufferToImage-pRegions-07931`), which no runtime enables. Measured on a
  Private 4 x 4 texture by a blit's copy (16 values: `-0.0`, three denormals of either sign, -1,
  -0.25, 2, the next float above 1, 1e30, both infinities, three NaNs, 0.25 and 0.5): NVIDIA's
  driver (620.70.06, an RTX 3090) keeps every value bit for bit and compares it unclamped, as
  Metal does; RADV (Mesa 26.2.6, the Radeon 780M) and ANV (Mesa 26.2.6, the Iris Xe) store a value
  below 0, `-inf` and a NaN of either sign as `+0.0`, one above 1 and `+inf` as 1.0, and a
  denormal and `-0.0` as `+0.0`, and compare what they stored; the validation layer reports none
  of these copies. Cost: a depth uploaded outside `[0, 1]` is the device's, and so is every read,
  sample, comparison and copy of it; one within it is copied exactly, except a denormal or
  `-0.0`, which a device may store as `+0.0` (`metal2vulkan-degrade-depth-denormals`).
- `metal2vulkan-degrade-cube-gradient-footprint` (precision).
  `PrecisionNote::CubeGradientFootprint`: An explicit-gradient sample of a cube texture takes
  Metal's footprint in the form Vulkan's rule reads, in every stage on every device. Cost: the
  sample takes the device's logarithm of Metal's footprint and, if anisotropic, the device's taps,
  so a level or blend near a boundary can differ from Metal's. Where the shader takes Metal's level
  and taps itself (`PrecisionNote::AnisotropicFilter`) this note is not recorded.
- `metal2vulkan-degrade-half-texel-conversion` (precision). `PrecisionNote::HalfTexelConversion`:
  The module reads, gathers, or samples a texture with a `half` result, the device's `float` texel
  rounded to nearest even on every device: within one `half` ulp of Metal's, but a 32-bit float
  texel past `±65504`, which Metal saturates in a sample and takes to the infinity in a read: the
  infinity of its sign in both, and `±65504` in both on a driver that converts a 2D image's fetched
  texel against the 16-bit float controls (AMD's open Linux driver, AMDVLK 2025.Q2.1, on a Radeon
  780M). Recorded outright for every such access, a heap element's and a depth texture's included:
  the device's `float` texel is its own.
- `metal2vulkan-degrade-srgb-texel-decode` (precision). `PrecisionNote::SrgbTexelDecode`: The module
  reads, samples, or gathers a texture with a `half` or `float` result through the device's sRGB
  decode, or decodes the device's filter of its codes, where the texture may be bound in an 8-bit
  sRGB format. The device's decode is the only form wherever it samples the format (an 8-bit sRGB
  format, or the sRGB form of a BC, PVRTC, ETC2, EAC or ASTC format, through the texture's or its
  decoded twin's sRGB view), so it holds where a command binds such a texture, at an argument
  buffer's texture access, and for a render pipeline that reads an sRGB color attachment as an
  input attachment, on every device. Where the
  device holds the format as its UNORM twin, or a storage texture holds it through its UNORM
  alias (at a slot, or held by an argument buffer), the module decodes the codes (or the
  device's filter of them) by the MSL transfer through the device's `Pow`, at every such access on
  every device.
- `metal2vulkan-degrade-function-storage-assumed` (precision).
  `PrecisionNote::FunctionStorageAssumed`: A vertex or fragment function keeps more private memory
  (its `alloca`s, a call stack among them, or constants held as `Private` variables) than the floor,
  16 KiB a thread, on a device whose Function storage was not measured, up to the 192 KiB a thread
  Metal was measured to hold (`device::function_storage`). The risk is the driver's: past what it
  holds it can answer wrongly or hang the GPU with no error.
- `metal2vulkan-degrade-quads-counted-before-discard` (precision).
  `PrecisionNote::QuadsCountedBeforeDiscard`: A fragment function runs a non-quad simdgroup
  operation (a reduction, scan, ballot, broadcast, shuffle, vote, or simdgroup matrix) before a
  discard on a path to it, in a shape where Metal's placement of it is not known.
- `metal2vulkan-degrade-view-base-texel-offset` (precision). `PrecisionNote::ViewBaseTexelOffset`:
  The module places a texel offset on an axis that may repeat or mirror as Metal does in an access
  of an argument buffer's texture: in an emulated raw sample (`heap_raw`) or a bicubic filter's
  footprint.
- `metal2vulkan-degrade-device-color-conversion` (precision).
  `PrecisionNote::DeviceColorConversion`: A fragment function writes a float to a color channel
  whose conversion the device makes, on every device: a normalized, sRGB or small float channel,
  an attachment that blends with factors other than one and zero
  (`ChannelWrite::Device`), the alpha of attachment 0 under alpha-to-coverage, a format whose
  conversion Metal was not measured with (the packed 16-bit unorm formats, `A8Unorm`, the packed
  floats). Near a tie between two codes the device may store the other one: the RTX 5080 Laptop
  GPU (NVIDIA 615.71.09) stores 0.5 in an 8-bit unorm channel as `0x7f`, and the lower code of
  every input up to two `f32` steps above a tie, where Metal stores `0x80` and rounds the exact
  product to nearest, a tie up; ANV, RADV and lavapipe round the `f32` product with ties to even.
  Also a `float` texel a function writes to a storage texture that may hold a normalized or
  half-float format, or `ARGB8Unorm`; an integer texel written into a storage texture of
  the float class that may hold a half-float or small-float format, which Metal rounds to nearest
  even and the device's store as it rounds a `float` (toward zero on every measured driver, one
  unit in the last place, a finite value past 65504 stored as 65504); and a fragment function's
  blend into an `ARGB8Unorm` attachment over its input attachment where no Vulkan blend state on its rotated
  image gives Metal's (a destination alpha in a color factor, say), whose result the attachment's
  write converts; and a render pipeline's private logic operation that the fragment function
  computes on a device without `logicOp` into an 8- or 16-bit unorm or snorm attachment, which
  converts the output to the codes it operates on by the device's `float` product rounded to
  nearest even where Metal rounds the exact product (the two differ only near a half-code tie).
  The recorded bound past one code, measured on
  lavapipe against plain Metal on an Apple M3 Max: a blend into an 8-bit unorm attachment
  (`RGBA8Unorm`, `ARGB8Unorm`) within 2 codes of Metal's; a fragment function that fetches an 8-bit
  sRGB attachment and writes it again, within 2 codes once a later draw reads an earlier one's code
  (with `srgb-texel-decode`); a blend into an 8-bit sRGB attachment, within 4 codes; a blend into a
  4-sample `A1BGR5Unorm` attachment the device renders itself (Mesa 26.2.4), within 2 codes, over a
  sample whose fill it stored a code from Metal's. RADV (Mesa 26.2.6 on the Radeon 780M) blends
  into an `RGB9E5Float` attachment it renders itself a source whose channels and alpha are halves
  rounded toward zero, a value past 65504 as 65504, where Metal blends the `float`s (Vulkan asks a
  blend for no more precision and range than the attachment's): at 2 and 4 samples, 30 of 96
  blended samples are not Metal's, up to 33 units of the larger value's last mantissa bit apart
  where the source passes 65504, and 7 where the half's alpha weighs a destination far larger than
  the source. Another device
  is not bounded past one code (an NVIDIA RTX 5080 Laptop GPU's blend into an 8-bit sRGB attachment
  stored up to 11 codes from Metal's).
- `metal2vulkan-degrade-device-interpolation` (precision). `PrecisionNote::DeviceInterpolation`: A
  fragment function reads a varying the device interpolates (`center_perspective`, `centroid`,
  `sample` or `no_perspective`, not `flat`), on every device. Vulkan states no precision for
  interpolation, and RADV, ANV, lavapipe and NVIDIA's driver each round
  most interpolated lanes otherwise than Metal, up to about `1.5e-4` apart (`3e-4` on lavapipe) on
  the probes. RADV's read of a perspective input at a sample or an offset is further off, and
  records `extrapolated-interpolation`.
- `metal2vulkan-degrade-device-interpolation-offset` (precision).
  `PrecisionNote::DeviceInterpolationOffset`: A fragment function reads an `interpolant` through
  `interpolate_at_offset`, which Metal evaluates at the offset as given from the pixel's upper-left
  corner, at any distance and on no grid (an Apple M3 Max, macOS 27.0.1: 0.999 read at 0.999, `1e9`
  extrapolated, a NaN or infinite offset NaN). The device reads at the offset moved to its origin
  (the pixel center; 7/16 on MoltenVK) and clamped to its `[minInterpolationOffset,
  maxInterpolationOffset]`, which Vulkan requires, a NaN to the minimum, and rounds it to its
  `subPixelInterpolationOffsetBits` (at least sixteenths): within the range the read lies within
  half a step of Metal's, past it at the range's nearer end. On MoltenVK Metal reads the offset
  itself, so within the range (`[-1/16, 15/16]` from the corner on the M3 Max) the read is Metal's.
- `metal2vulkan-degrade-extrapolated-interpolation` (precision).
  `PrecisionNote::ExtrapolatedInterpolation`: A fragment function reads a perspective input at a
  sample or an offset (`interpolate_at_sample`, `interpolate_at_offset`, or a sample-qualified
  varying a function looping over its covered samples reads at each pass's sample), recorded on
  every device: a device may answer such a read with the value at the pixel center extrapolated
  along the quad's coarse derivatives, as RADV does, where Metal
  interpolates in perspective at the location itself. The answer is first-order, off by the
  value's curvature across the distance, which grows with the triangle's spread of clip `w`: on a
  triangle of `w` 1, 2 and 4 over a 6x6 target, up to about 1.0 in a lane whose vertices hand 3,
  -2 and 0.5 (the Radeon 780M, Mesa 26.2.6). A no-perspective input is exact there and records
  nothing. It is the device's own interpolation, which Vulkan states no precision for.
- `metal2vulkan-degrade-quads-split-by-clipping` (precision). `PrecisionNote::QuadsSplitByClipping`:
  A fragment function runs a quad or simdgroup operation, recorded on every device: a device may
  shade a triangle it clips as several triangles, and then a quad a seam between the pieces crosses
  runs once per piece, each run with the other piece's pixels as helper invocations, so the
  operation counts or combines fewer active lanes than Metal's, whose quad holds all four.
- `metal2vulkan-degrade-fragment-simdgroup-grouping` (precision).
  `PrecisionNote::FragmentSimdgroupGrouping`: A fragment function runs a simdgroup operation (a
  reduction, scan, vote, ballot, shuffle, broadcast or AGX multiply-add) or reads its
  `[[thread_index_in_simdgroup]]`, or discards, asks `quad_is_helper_thread` or
  `simd_is_helper_thread` and runs a quad operation, on every device. A
  simdgroup operation runs over the pixels the device's rasterizer packs into a subgroup, so a sum,
  a ballot or a shuffle combines other pixels than on an Apple GPU, whose packing is measured, and
  the lane is the pixel's place in the device's subgroup, or in its aligned 32 lanes where the
  subgroup holds 64 or 128 (each such run of 32 lanes is one simdgroup, as
  `[[threads_per_simdgroup]]`'s 32 says). An Apple GPU also shades some quads of a
  function that discards as several partial quads from the function's start, each run holding the
  quad's other covered pixels as helpers, where the device shades them as one quad: a ballot of the
  helper query names covered pixels as helpers on the M3 Max (3 of the 16 quads of an 8x8 target, 2
  to 4 of the 12 of a 7x5 one, under the full-screen triangle with every third pixel discarded) and
  none on the device. A device whose fragment subgroup holds fewer than two quads combines the
  one quad's pixels alone, so a reduction ahead of a `discard_fragment` counts no quad the discard
  drops, where an Apple GPU's simdgroup counts the dropped quads of its 8x4 pixel block; the
  device states one fragment subgroup size, so no required size widens it, and no exact form
  exists there.
- `metal2vulkan-degrade-vertex-simdgroup-grouping` (precision).
  `PrecisionNote::VertexSimdgroupGrouping`: A vertex function runs a simdgroup or quad operation
  (a reduction, scan, vote, ballot, shuffle, broadcast or AGX multiply-add), on every device.
  Metal's shading language defines its SIMD-group and quad-group functions for kernel and
  fragment functions (MSL 4.1 sections 6.10.2 and 6.10.3); in a vertex function the groups are the
  device's own. The operation runs over the vertices the device packs into a subgroup, in its
  lane order, so a sum, a vote or a shuffle can combine other vertices than on the M3 Max, whose
  answers group 32 consecutive vertices. Vulkan neither fills a vertex subgroup nor fixes which
  vertices share one: an Iris Xe (Mesa 26.2.6) whose vertex subgroup size is 32 runs 8 of 64
  points in each, the even or the odd vertices of a run of 16.
- `metal2vulkan-degrade-blended-write-rounding` (precision). `PrecisionNote::BlendedWriteRounding`:
  A fragment function writes a float to a half-float color channel of a render pipeline under
  `TextureWriteRoundingMode::NearestEven` that blends with factors other than one and zero
  (`ChannelWrite::BlendedNearestEven`): Metal rounds the blend's result to nearest even, and the
  device's blend converts it by its own rounding (MoltenVK's toward zero), so a blended value off
  the half grid may store the neighbouring half; under a constant factor Metal's result is neither
  rule's on 2,376 of 580,508 `RGBA16Float` channels near the half denormals.
- `metal2vulkan-degrade-coverage-reads-rounded-alpha` (precision).
  `PrecisionNote::CoverageReadsRoundedAlpha`: A render pipeline covers by alpha
  (`alphaToCoverageEnabled`) under `TextureWriteRoundingMode::NearestEven`, and its fragment
  function writes a `float` alpha to a half-float attachment 0 that stores it as written: Metal
  stores the alpha rounded to the nearest-even half and reads the coverage from the alpha before
  rounding, which no device separates, so the function rounds the alpha and the coverage reads the
  rounded one. The stored alpha is Metal's; a fragment whose alpha lies within half a `half` step
  of a coverage threshold may cover one sample more or fewer (2 of 16,384 alphas on an Apple M3
  Max). Which samples an alpha covers is the device's own (`device-alpha-coverage`).
- `metal2vulkan-degrade-device-alpha-coverage` (precision). `PrecisionNote::DeviceAlphaCoverage`: A
  render pipeline covers by alpha (`alphaToCoverageEnabled`, over a four-component float first
  output at attachment 0), on every device. The device converts the alpha to its coverage mask by
  its own algorithm, which Vulkan does not specify ("It is intended that the number of 1's in this
  value be proportional to the alpha value", and "The algorithm may be different at different
  framebuffer coordinates"). Metal, measured over 639 alphas a sample count on an Apple M3 Max under
  macOS 27.0.1, covers a pixel at one sample above an alpha of 1/2, and at two and four samples
  `ceil(alpha * n - t)` samples, `t` a 2x2 ordered dither of the pixel's window coordinates (1/8,
  5/8; 7/8, 3/8), either count within an `f32` ulp of a threshold. Recorded bound, pixel by pixel
  against Metal's reading: on lavapipe (Mesa 26.0.8, arm64) every pixel covers `floor(alpha * n)` or
  `ceil(alpha * n)` samples, so within one sample of Metal's and Metal's where `alpha * n` is an
  integer; at one sample, where lavapipe dithers too, a whole fragment is stored or dropped where
  its alpha lies between its own threshold and 1/2 (2,560 of the sweep's 10,224 pixels); at four
  samples a pixel is one sample apart only within an `f32` ulp of a threshold (96 of 10,224). An
  NVIDIA RTX 5080 Laptop GPU stores a fragment of alpha 1/2 at one sample, where Metal drops it;
  every other device and count is unmeasured.
- `metal2vulkan-degrade-rotated-blend-rounding` (precision). `PrecisionNote::RotatedBlendRounding`:
  A fragment function writes an attachment held with its channels rotated (`ARGB8Unorm`) whose blend
  is not exact by construction (`LoweringInputs::rotated_blends`): every blend but a minimum, a
  maximum, and one whose Vulkan factors are all `ZERO` or `ONE`. A blend may compute a factor pair
  apart in its color and alpha arithmetic, as Apple's does, and the rotation moves Metal's blue into
  the image's alpha and Metal's alpha into its red, so a channel may store one code apart from
  Metal's.
- `metal2vulkan-degrade-linked-sampler-numbering` (precision).
  `PrecisionNote::LinkedSamplerNumbering`: The sampler resource IDs a pipeline state takes for the
  constexpr samplers of the visible functions its stage links may be other than Metal's: two or more
  linked functions bring constexpr samplers, which Metal numbers in an order the pipeline's inputs
  do not decide; or a function a table can hold that linking did not import could not be planned, so
  its samplers are not known; or a dynamic library's symbol, a binary function or a function-pointer
  group's function brings them, whose numbering is not measured.
- `metal2vulkan-degrade-argument-pixel-sampler-scaling` (precision).
  `PrecisionNote::ArgumentPixelSamplerScaling`: A sample, gather or comparison through an argument
  buffer's sampler whose word the specialized module cannot read may hand the device a pixel
  coordinate scaled by the texture's size, where the sampler states pixel coordinates, with
  `pixel-sampler-scaling`'s bound. A sample through a state of nearest filters, one level and no
  anisotropy hands the device its texel's centre instead, where the device reads exactly Metal's
  texel but at a negative subnormal coordinate, whose `floor` is the device's
  (`device-denormals`' reading, which this note records for such a sampler).
- `metal2vulkan-degrade-half-output-precision` (precision). `PrecisionNote::HalfOutputPrecision`: A
  fragment function returns a `half` color output (a scalar or vector of AIR's `half`) to an
  attachment the pipeline has, and either its module is compiled without fast math
  (`air.compile.fast_math_disable`) and that attachment blends, factors one and zero included, with
  some channel written (`StageIo::half_blend_colors`, which the pipeline's stage reads), or its
  module names no math mode (under fast math the pipeline's stage records
  `PrecisionNote::HalfOutputFastMathBlend` instead).
- `metal2vulkan-degrade-anisotropic-filter` (precision). `PrecisionNote::AnisotropicFilter`: The
  module samples a texture through an anisotropic sampler with Metal's anisotropic filter emulated
  by the device's samples at explicit levels (`lower::sampler::anisotropic`), on a device without
  `samplerAnisotropy`, every Vulkan sampler being isotropic there, each tap the device's isotropic
  filter: a sample without comparison of a 2D texture or 2D array at the fragment's implicit
  level or with explicit gradients, or of a cube or cube array with explicit gradients, through a
  constexpr sampler of `maxAnisotropy` above 1, or through a runtime sampler slot whose word states
  anisotropy (under that word alone). A device with the feature samples with its own anisotropic
  filter. The footprint's logarithms and powers are the device's `Log2` and `Exp2`, its level and
  spacing unrounded, and a linear mip filter's two levels the device's own blend at the fractional
  level, where Metal takes a piecewise-linear `log2`, 256ths and 4096ths. Cost: a
  cube's taps step across its face by Metal's face gradients, and a tap past the face's edge reads
  the next face as the device projects it, which Metal's own is not measured against.
- `metal2vulkan-degrade-device-anisotropic-filter` (precision).
  `PrecisionNote::DeviceAnisotropicFilter`: The module samples a texture through a sampler whose
  anisotropy is known where Metal's anisotropic filter is not emulated
  (`PrecisionNote::AnisotropicFilter`'s reach), on a device without `samplerAnisotropy`, which
  filters it isotropically: a cube, a 3D texture, a comparison, a sample whose residency the
  function observes, or a texture of another class than its declared one, at an implicit level or
  with explicit gradients, through a constexpr sampler of `maxAnisotropy` above 1 or a runtime
  sampler slot whose word states anisotropy (under that word alone). An argument buffer's
  anisotropic sampler, which the function cannot record, is refused there (`SamplerAnisotropy`).
- `metal2vulkan-degrade-device-point-size` (precision). `PrecisionNote::DevicePointSize`: A vertex
  function writes `[[point_size]]`, or a mesh function that outputs points `set_point_size`, on a
  device whose `pointSizeRange` ends below Metal's largest point, 511 pixels
  (`Limits::max_point_size`: 255 on ANV, 256 on lavapipe). A line or triangle mesh's size draws
  nothing, and its function records no note. The render pipeline state carries the note whatever
  primitive a draw of it takes; a draw of lines or triangles of it is Metal's exactly.
- `metal2vulkan-degrade-clipped-coverage-split` (precision). `PrecisionNote::ClippedCoverageSplit`:
  A fragment function compiled for a multisampled pipeline reads `[[sample_mask]]`, or reads a
  centroid varying and runs once per pixel, on every device (no query tells Apple's rasterizer
  apart): Apple's GPU draws a triangle
  that extends past the viewport as pieces whose shared edges cross pixels, and such a pixel's
  fragments can run once per piece, each reading that piece's coverage, where the device hands the
  pixel's fragment its coverage by the whole primitive (lavapipe by its own pieces). A centroid
  varying is read at the centroid of the whole coverage there.
- `metal2vulkan-degrade-looped-sample-outputs` (precision). `PrecisionNote::LoopedSampleOutputs`: A
  fragment function that reads `[[sample_mask]]`, runs once per covered sample on Metal and writes
  an output runs once per pixel in a multisampled pipeline, its body once per covered sample, on
  every device. The samples Metal writes are written, but each with the last
  pass's outputs, where Metal writes each sample's from that sample's run: equal where every pass
  computes the same outputs. Vulkan hands an invocation under sample shading only its own sample's
  bit of the coverage.
- `metal2vulkan-degrade-half-output-fast-math-blend` (precision).
  `PrecisionNote::HalfOutputFastMathBlend`: A fragment function of a module compiled with fast math
  (`air.compile.fast_math_enable`) returns a `half` color output to an attachment the pipeline has
  that stores some channel (its write mask names one of its format's channels;
  `StageIo::half_fast_colors`, which the pipeline's stage reads), where
  `PrecisionNote::HalfOutputPrecision`'s conditions hold. Every device receives the output widened
  to `float`.
- `metal2vulkan-degrade-device-sampling` (precision). `PrecisionNote::DeviceSampling`: Native device
  sampler, on every device (the module requires `VkProperty::DeviceSampling`), its level clamp the
  descriptor's as given (a `lodMinClamp` past level 15, past every texture's last level, handed as
  15, which reads the same level; Metal holds it on a `1/64` grid, and its nearest mip filter reads
  the lower level at a clamp halfway between two, where the device may read the upper) and a
  subnormal coordinate the device's to read (Metal reads a normalized one as a zero before
  addressing, and a negative pixel one as texel -1). That includes a linear sample of a two-plane
  YCbCr texture, whose texels hold Metal's reads exactly: the device's filter of them lands within
  1.183 codes (1/255) of Metal's sum on the format's grid, so a draw binding such a texture makes no
  pipeline of its own.
- `metal2vulkan-degrade-native-transcendentals` (precision). `PrecisionNote::NativeTranscendentals`:
  Native device maths: the module computes a `float` or `half` math-library function by the device's
  own `GLSL.std.450` instruction (`exp`, `log`, `pow`, `sinh`, `cosh` and `tanh` among them; a
  composition of the device's `Exp2` and `Log2` only for `exp10` and `log10`, which no instruction
  is); `fast::sqrt` the device's `Sqrt`, and a `half` `sqrt` its 16-bit `Sqrt` where the module
  computes `half` in 16 bits under the 16-bit float controls; the fast `sin`, `cos`, `tan`, `sinh`,
  `cosh`, `tanh` and `atanh` the device's instructions, not MSL's fast-math formulas (`fast::sin` of
  `1e7` is the device's sine, where Metal answers `+0`; `fast::tanh(45)` is `1`, where Metal answers
  NaN); `fabs`, `sign` and `fract` the device's `FAbs`, `FSign` and `Fract`, their special operands
  the instruction's (Metal's `sign` of a NaN, a zero and an `f32` denormal is `+0`, its `fract` of
  an infinity `+0`); a `half` function's signed zeros and NaN signs its `float` instruction's
  (`half` `sin(-0)` is `-0`, where Metal answers `+0`), and on every device a `half` `fmod`'s zero
  remainder its `float` remainder's (`half` `fmod(-2, 0.5)` is `-0`, where Metal answers `+0`);
  `pow`, `powr` and `atan2` C99's special values by the device's comparisons around its instruction
  (`pow(x, 1)` the device's `Pow`, where Metal answers `x`; an `f32` denormal operand as the device
  reads it); and `round` C99's from the device's `Trunc` on every device (`round(-0.3)` is `-0`,
  where Metal answers `+0`).
- `metal2vulkan-degrade-native-division` (precision). `PrecisionNote::NativeDivision`: Native device
  maths: a precise `float` or `half` division, or Metal's fast division or reciprocal, is the
  device's `OpFDiv`, on every device: within Vulkan's 2.5 ulp, where
  Metal's precise division is correctly rounded. A `half` one is the `float` quotient rounded to
  `half`, or, where the module computes `half` in 16 bits under the 16-bit float controls, the
  device's 16-bit `OpFDiv`: within 2.5 `half` ulp, its divisor scaled by a power of two into
  `[2^-14, 2^14]`.
- `metal2vulkan-degrade-native-fma` (precision). `PrecisionNote::NativeFma`: Native device maths: a
  `float` fused multiply-add, or a `half` one outside the 16-bit float controls, is the device's
  `GLSL.std.450 Fma` (on the
  `half` operands held in `float`, its result rounded to `half`), where Metal's rounds once. On a
  device that computes `half` in 16 bits under the 16-bit float controls, a `half` one is the
  device's 16-bit `Fma` on the `half` operands. Recorded bound: a `half` result
  held in `float` within one `half` ulp of Metal's; a `float` one, and a 16-bit one, within half an
  ulp of the product plus half an ulp of the result, in its own format.
- `metal2vulkan-degrade-device-texel-conversion` (precision).
  `PrecisionNote::DeviceTexelConversion`: Native device sampling (a texel conversion): the module
  hands a read, sample or gather with a `float` result the device's own conversion of the texel's
  codes, on every device. Recorded bound, of a texel read from one texel of one level: an 8-, 10- or
  16-bit normalized code within one `f32` ulp of Metal's `c / (2^b - 1)` where measured (lavapipe
  multiplies by the reciprocal: some 8-bit codes one ulp high; some 10-bit `BGR10A2Unorm`, a 16-bit
  and 512 `Depth16Unorm` codes one ulp low), and a 10-bit code within 0.015 of a code where the
  device widens it by replicating its bits to 16, `(c << 6 | c >> 4) / 65535` (NVIDIA's RTX 5080
  Laptop GPU); a 4-, 5- or 6-bit channel of a packed format within one `f32` ulp where the device
  converts it by the reciprocal (lavapipe: some `ABGR4Unorm` codes one ulp high), and within 0.18 of
  a code where it widens a 5- or 6-bit channel by replicating its bits (ANV, NVIDIA; a 4-bit
  channel's replication is exact); a half-float or small-float texel's NaN in the device's encoding,
  where Metal answers `0x7fc00000`; a 32-bit float texel's NaN in a gather in the device's encoding,
  and its denormal in a gather a zero of its sign, where Metal hands back the texel's bits (NVIDIA's
  615.71.09 on the RTX 5080 Laptop GPU gathers every NaN as `0x7fffffff` and every denormal as a
  zero); a 32-bit float texel's NaN in a nearest sample in the device's encoding, and its denormal
  there a zero of its sign on a device that flushes it there (ANV's Mesa 26.2.6 on the Iris Xe
  samples every NaN as `0xffc00000` and every denormal as a zero of its sign, and reads and gathers
  both as stored; RADV's Mesa 26.2.6 on the RX 7900 XTX and the Radeon 780M samples a denormal as a
  zero and gathers it as stored); every other class Metal's value.
- `metal2vulkan-degrade-device-vertex-conversion` (precision).
  `PrecisionNote::DeviceVertexConversion`: Native vertex input (a conversion): a render pipeline's
  vertex descriptor gives a normalized format (`UCharNormalized` through `Short4Normalized` of
  either sign, `UChar4Normalized_BGRA`, `Int1010102Normalized`, `UInt1010102Normalized`) to an
  attribute that fixed-function vertex input fetches, so the attribute is the device's `UNORM` or
  `SNORM` conversion of its codes, where Metal's fetch answers the code over its maximum correctly
  rounded (an `Int1010102Normalized` 10-bit field times `float(1/511)`): within one `f32` ulp where
  measured (lavapipe reads some 8-bit unsigned codes one ulp high), but lavapipe's per-instance
  fetch, which reads the most negative code of an 8- or 16-bit signed format unclamped (-128/127,
  -32768/32767) where Metal reads -1. Recorded on the pipeline's vertex stage on every device; one
  module serves every format the device fetches. A normalized attribute the shader pulls from its
  buffer instead (an offset, stride or step fixed-function input cannot fetch, a kernel's
  `[[stage_in]]` attribute) is the module's product of the code and its maximum's reciprocal,
  rounded once, the signed one clamped to -1: within one `f32` ulp of Metal's quotient (some 8-bit
  unsigned codes one ulp high, some signed, 10-bit and 16-bit codes one ulp toward zero; 0, the
  maximum and -1 exact). The function records it on every device, read as `float` for every
  normalized format but `Int1010102Normalized` (Metal's own product), and read as `half` for a
  16-bit format (one `half` ulp at four codes).
- `metal2vulkan-degrade-native-half-conversion` (precision). `PrecisionNote::NativeHalfConversion`:
  Native half rounding (a conversion): the module rounds a `half` result it computes in `float` (a
  `half` operation's, a function's, a conversion from `float`'s, a texel's) by the device's
  conversion of the `float` to 16 bits and its widening, on every device with `shaderFloat16`.
  Without the float controls that make that conversion Metal's `half(float)` (`RoundingModeRTE`,
  `DenormPreserve` and `SignedZeroInfNanPreserve` for 16 bits) the conversion is the device's own.
  Under them it is Metal's by the Vulkan spec, but a driver's compiler may fold the conversion's
  round trip into the value it started from (NVIDIA's does) or fuse the `float` addition,
  subtraction, multiply or fused multiply-add computing the value into the narrowing and round its
  exact result once (RADV's does, one `half` ulp from Metal's where the `float` result lies on a
  `half` midpoint), so a pipeline that rounds through the pair records the note there too, a
  static note of the rounding. Under those controls a `half` `+`, `-`, `*`, `/` and `sqrt` are the
  device's 16-bit instructions, whose results take no such rounding. Also recorded where a `half`
  read from its bits is the device's widening, or a `half` encoded to its bits (where they are not
  the bits it was read from) is the device's narrowing, and that conversion is not under those
  controls: the device's conversion without them, and `GLSL.std.450 UnpackHalf2x16` and
  `PackHalf2x16` on every device without `shaderFloat16`. A half denormal may read or encode as the
  zero of its sign.
- `metal2vulkan-degrade-native-half-quantize` (precision). `PrecisionNote::NativeHalfQuantize`:
  Native half rounding (a quantize): the module rounds a `half` result it computes in `float` (a
  `half` operation's, a function's, a conversion from `float`'s, a texel's) by the device's
  `OpQuantizeToF16` alone, on every device without `shaderFloat16`, its quantize measured or not: a
  value below the smallest normal `half` may round to a zero where Metal answers the half denormal
  (or `2^-14`, from just below it). Where the rounding's input is a zero or too small for a normal
  `half`, the result may be either zero, so a negative zero may come out positive, as the SPIR-V
  specification lets the quantize answer: a program sees it in a reciprocal's or `atan2`'s sign
  and in the bits written to a `half` target or buffer. The form not taken, copying the input's
  sign bit into the result by integer operations, would keep Metal's sign at about 4 instructions
  per `half` rounding on every device without `shaderFloat16`. An integer's `half` rounds exactly
  instead.
- `metal2vulkan-degrade-native-nan-encoding` (precision). `PrecisionNote::NativeNanEncoding`: Native
  device maths: a `float` or `half` operation's NaN result is the device's, its sign and payload
  unselected, where Metal answers a NaN, on every device: a `float`
  `fneg` and `fabs`, which keep a NaN operand's payload on Metal, a `half` `fract`, which returns
  it, and a precise `float` `pow` of a NaN base to the exponent one, which returns the base, among
  them. A `float` `ldexp`, which the module computes on the bits, returns a NaN operand itself as
  Metal does, on every device, and records nothing. Also
  recorded wherever a `half` is read from its bits, on every device, since the read is the
  device's widening (its conversion with `shaderFloat16`, `UnpackHalf2x16` without): a NaN's bits
  the device's, where Metal widens every `half` NaN to `0x7fc00000` and moves one it does not
  widen with its bits. And wherever a `half` is encoded to its bits (a store, a reinterpretation, a
  texel's bits) where they are not the bits it was read from (a `half` moved unchanged from its
  read keeps them, through a struct's member, a vector the translation joins from read lanes (a
  stage-in or pulled vertex attribute's fetch, a packed float's unpack, a bitcast's lanes), a
  containment check and the choice among a pointer's candidate buffers too), on every device,
  since the encode is
  the device's narrowing (its conversion with `shaderFloat16`, `PackHalf2x16` without): a NaN's
  bits the device's, where Metal keeps a `half` NaN's sign and payload. And wherever the entry
  reaches a `float` constant whose bits are a signalling NaN (every exponent bit set, the quiet bit
  clear, a nonzero payload), whether the program wrote it or Metal's compiler left it deciding an
  `fadd`, `fsub`, `fmul`, `fdiv` or fused multiply-add whose other operands only the run knows:
  Metal keeps its bits, and the device's move of it as a `float` may quiet it (an RTX 3090 stores
  `0x7fc00321` where Metal stores `0x7f800321`; the other cards in the runs so far kept the bits).
  A `half` constant moves as its bits and records nothing; `double` is not measured.
- `metal2vulkan-degrade-native-min-max` (precision). `PrecisionNote::NativeMinMax`: Native device
  maths: Metal's `fmin`, `fmax`, `min3`, `max3`, `clamp` or `saturate` (its `clamp(x, +0, 1)`) on
  `float` or `half` is the device's `GLSL.std.450` `NMin`, `NMax` or `NClamp`, and `median3`
  `min(max(c, min(a, b)), max(a, b))` of the device's `NMin` and `NMax`, on every device, whose pick
  of two NaNs and of an `f32` denormal is the device's. The result is one of the operands, Metal's
  but at three ties (of two NaNs a NaN of the device's bits, where Metal answers the second
  operand's; of `+0` and `-0` either zero; an `f32` denormal beside a zero or a denormal compared by
  its value, where Metal compares it as a zero and answers the second) and one undefined answer:
  `fast::median3` of two NaN first operands and a number answers the number, as precise `median3`
  does on Metal, where Metal's fast form answers the second NaN. MSL leaves a NaN operand under fast
  math undefined (section 8.1).
- `metal2vulkan-degrade-native-dot` (precision). `PrecisionNote::NativeDot`: Native device maths:
  Metal's `dot` on `float` and `half` is the device's `OpDot` (a `half` one summed in `float` and
  rounded once to `half`, or the device's 16-bit `OpDot` where the module computes `half` in 16
  bits under the 16-bit float controls), not Metal's fused chain of its products, on every device.
- `metal2vulkan-degrade-native-mix` (precision). `PrecisionNote::NativeMix`: Native device maths:
  Metal's `mix(x, y, t)` on `float` and `half` is the device's `FMix` (a `half` one computed in
  `float` and rounded once to `half`, or the device's 16-bit `FMix` where the module computes
  `half` in 16 bits under the 16-bit float controls), not Metal's `fma(t, y, fma(-t, x, x))`, on
  every device.
- `metal2vulkan-degrade-native-logic-operation` (precision).
  `PrecisionNote::NativeLogicOperation`: A render pipeline's private logic operation (other than
  copy and no-op) is drawn by the device's `logicOp` into an `RGB10A2Unorm`, `BGR10A2Unorm` or
  signed normalized attachment the fragment function writes, with a channel its write mask keeps.
  Vulkan's operation takes the fragment's value converted to the channel's own code and the stored
  code as it is, and stores its result as it is; Metal widens a 10-bit (or 2-bit) code and the
  fragment's to 16-bit codes, operates, and narrows the result to the channel's code, rounded to
  nearest, and reads a signed channel's lowest code (`0x80`, or `0x8000`) as the code above it and
  stores a result of the lowest code as the code above (probe `logic_operation` on an Apple M3 Max,
  macOS 27.0.1). So a channel may store another code than Metal's (lavapipe's xor into
  `RGB10A2Unorm` stored green 938 where Metal's rule gives 937, and its xor of `0xe0` over an
  `RGBA8Snorm` channel's `0x80` stored `0x60` where Metal's gives `0x61`), by a distance the note
  does not bound: the result follows the codes' bits. The device's `logicOp` is its native form
  there, and no other form is built for it; a device without `logicOp` refuses the operation
  (`Unsupported{PrivateRenderState(LogicOperation)}` naming `logicOp`). A signed channel the draw does not write (outside the
  write mask, or every channel under no-op) keeps its lowest code where any draw's unwritten
  texel does (`snorm-minimum-kept`).
- `metal2vulkan-degrade-device-depth-bias` (precision). `PrecisionNote::DeviceDepthBias`: A draw
  of triangles under a depth bias (`setDepthBias:slopeScale:clamp:` with a bias or a slope scale
  other than 0) on `metal2vulkan-runtime`, whose render encoder carries it as a static note
  (`RenderCommandEncoder::DEPTH_BIAS_NOTE`): the biased depth is the device's. Vulkan biases by
  `m × slopeScale + r × depthBias`, where the maximum depth slope `m` is the device's (exact, or
  the larger of the two derivatives' magnitudes) and the unit `r` is `2^(e - n)` of the primitive's
  largest depth exponent for a float depth attachment and the device's for a fixed-point one (at
  most `2 × 2^-n`). Apple's documentation states neither the slope measure nor the unit. A draw of
  lines is not biased, as Metal biases triangles alone.
- `metal2vulkan-degrade-swizzled-opaque-black-border` (precision).
  `PrecisionNote::SwizzledOpaqueBlackBorder`: A sampler that reads its
  border (`ClampToZero` or `ClampToBorderColor` on an axis, both of which read the sampler's
  colour) with an opaque-black border colour. A sampler state carries it as a static note
  (`SamplerState::precision_note`), decided by its descriptor alone on every device; a compute or
  render pipeline whose function samples through such a constexpr sampler carries it among its
  notes (`precision_notes`), decided by the sampler's own constants. Sampled
  through a view whose Vulkan component mapping is not the identity (the texture's swizzle, or the
  mapping its pixel format takes), its border texel is the device's answer on a device without
  `borderColorSwizzleFromImage` (`VK_EXT_border_color_swizzle`): Vulkan leaves it undefined unless
  the sampler names the view's mapping, which a sampler that meets every view cannot, and the
  NVIDIA driver reads `(0, 0, 0, 1)`, where Metal reads the border colour in the format's channels
  through the swizzle. A device with the feature reads Metal's texel.
- `metal2vulkan-degrade-device-line-rasterization` (precision).
  `PrecisionNote::DeviceLineRasterization`: A draw of lines (a line or a line strip, or a
  triangle's edges under the `Lines` fill mode), on every device, which carries it
  (`RenderCommandEncoder::LINE_NOTE`). Metal covers the pixels whose diamond a line exits. Lines
  draw by Vulkan's Bresenham mode (`bresenhamLines`, the runtime's floor), which lets an
  implementation's fragment lie up to one pixel in x or y from the diamond-exit rule's, and its
  fragment count differ by one (the Vulkan specification, "Bresenham Line Segment
  Rasterization"), and whose diamond is open, so no Vulkan control decides the pixel of a line that
  only touches a diamond's vertex: where a line of the conformance case
  `primitive-type-restart-indices` runs through the boundary point (3.5, 1.0), Metal on an Apple M3
  Max covers the lower pixel and NVIDIA's driver the upper (the accepted difference "A line's
  tie-break is the device's own", section 10). A state that enables alpha to coverage or alpha to one draws its lines by the device's default
  rule instead, as Vulkan's Bresenham mode takes neither
  (`VUID-VkGraphicsPipelineCreateInfo-lineRasterizationMode-02766`): a device whose `strictLines`
  is set (lavapipe, NVIDIA's driver) covers the rectangle of the line's width, so the pixels at a
  line's two ends, and on either side of a pixel boundary the line runs along, may differ from
  Metal's.
- `metal2vulkan-degrade-device-pack-conversion` (precision). `PrecisionNote::DevicePackConversion`:
  Native device maths (a conversion): `air.pack` or `air.unpack` of a normalized format is the
  device's: `GLSL.std.450`'s `Pack`/`Unpack` `Unorm4x8`, `Snorm4x8`, `Unorm2x16` and `Snorm2x16`,
  their definition by the device's `NClamp`, multiply, `RoundEven` and reciprocal for `rgb10a2` and
  `rgb565`, and the MSL transfer functions by the device's `Pow` for `unorm4x8_srgb`'s colour. Metal
  rounds the exact product and divides correctly rounded (`unorm565` in `float`: the product
  rounded to `float` first, and the code times the `float` nearest the reciprocal, as the
  definition does), answers code 0 for a NaN channel, and
  takes an sRGB colour through its twelve-bit table; a packed code is within one of Metal's, an
  unpacked value within 2.5 ulp, a decoded sRGB colour within half a twelve-bit step (`1 / 8190`)
  of Metal's step beyond its `Pow`'s bound. Also a write of an 8-bit sRGB storage texture held
  through its UNORM alias (at a slot, or held by an argument buffer), which encodes by
  `unorm4x8_srgb`'s pack. Recorded on every device. The same conversions
  serve an implicit imageblock access of a normalized attachment (a `half` load rounding the
  unpacked `float` once more), a fragment function's read and write of an 8-bit sRGB attachment
  through its word view. So does every conversion of
  an extended-range, padded 10-bit or bfloat texel (`BGR10_XR`, `BGRA10_XR`, `RGB10A8_2P_XR10`,
  `R10Unorm_X6`, `RG10Unorm_X12`, their sRGB twins, `RGB10A2Unorm_sRGB`, `R16Bfloat`, `RG16Bfloat`,
  `RGBA16Bfloat`), which no Vulkan format holds: a read is `float(c - 384)` (a padded code `c`)
  times the `f32` nearest `1 / 510` (`1 / 1023`), a write `RoundEven` of the clamped value's `f32`
  product plus 384, an sRGB twin's colour the MSL transfer by the device's `Pow` on the magnitude,
  a `half` result the device's narrowing, a bfloat code's NaN as it is (a write's kept a NaN by
  its quiet bit). Metal divides correctly rounded, rounds the exact product, rounds a `half` with
  ties away from zero, converts the sRGB twins through its own tables, makes a bfloat NaN
  `0x7fc00000` or `0x7fc0`, and rounds a bfloat sample's `half` toward zero: a linear value within
  one `f32` ulp, a code within one (`R10Unorm_X6_sRGB`'s and `RGB10A2Unorm_sRGB`'s colour within
  two below `0.0031308`, where Metal rounds the value to `2^-13` before its table, a step there
  spanning 1.6 codes), an sRGB twin's decoded value within half a `2^-13` step of Metal's table
  (`R10Unorm_X6_sRGB`'s and `RGB10A2Unorm_sRGB`'s colour 1.64 steps) beyond the `Pow`'s bound, a
  `half` result within one `half` ulp. A texture a heap holds by its raw texels alone
  (`BGR10_XR`, `BGRA10_XR`, `R10Unorm_X6`, `RG10Unorm_X12` and the bfloat formats), whose
  `float` read, sample or gather through an argument buffer the
  function decodes the same way: such a stage records the note as a specialized note, holding
  where a texture its handles can name is one (`SpecKey::HeapRawTextures`, which a runtime gives
  from the textures the command makes resident or binds), so a reading of the module alone
  holds it, and so does a command that can name one.
- `metal2vulkan-degrade-device-fragment-runs` (precision). `PrecisionNote::DeviceFragmentRuns`: A
  fragment function that writes memory (`StageInterface::writes_memory`) in a render pipeline of
  more than one sample, which the pipeline's compile records on its fragment stage, decided by the
  sample count and the function's writes alone. It runs as many times per pixel as the device runs
  it, each run reading the pixel's whole coverage as `[[sample_mask]]`, so its writes repeat as
  often as the device's runs, and whatever it derives from its own writes across runs follows the
  device's count too, its stored colours and `[[sample_mask]]` output included, not only memory.
  Apple's GPUs were measured to run it once per covered sample;
  documented Metal names no count (MSL section 5.5 makes a function run per sample only for a
  sample-qualified input, `[[sample_id]]` or `[[color(m)]]`). Vulkan runs a fragment shader
  without sample shading at least once per pixel and more where the device chooses (vkspec 35.5).
- `metal2vulkan-degrade-device-occlusion-count` (precision). `PrecisionNote::DeviceOcclusionCount`:
  A `Boolean` or `Counting` visibility result (`set_visibility_result_mode`), the static note
  `RenderCommandEncoder::VISIBILITY_NOTE` on every device. The count is the device's precise
  occlusion query's, where Metal documents "the number of samples that passed depth and stencil
  tests". Vulkan does not hold devices to one count ("Occlusion Queries"): a device may kill
  fragments before rasterization, which then count nothing, and may count the samples that pass
  once for each colour output; and where `earlyFragmentMultisampleCoverageAfterSampleCounting` is
  false, a fragment function with early fragment tests may shade and drop coverage before the
  samples are counted. No bound against Metal's count is measured. Cost: a count can differ from
  Metal's, and a `Boolean` result with it, where a sample is killed, dropped or counted once per
  colour output.

## 10. Accepted differences

Where the runtime answers otherwise than Metal on purpose, the difference is listed here. Most
carry no degrade notice, since nothing is slower or approximate for them; an entry whose answer is
the device's own names the static note its stage records. Most are answers Metal leaves undefined
or ends the process for, and answers the runtime adds beyond Metal's API. The rustdoc of each item
named states the rule in full.

**Draws**

- **A vertex function's `[[render_target_array_index]]` and `[[viewport_array_index]]` are stored
  as the function writes them.** Metal requires the same value at every vertex of a primitive;
  where a program breaks that, the layer or viewport is the device's choice. Metal leaves such a
  primitive undefined (the Metal Shading Language specification), and Vulkan asks the same value
  of `Layer` and `ViewportIndex` at every vertex of a primitive
  (VUID-VkPipelineShaderStageCreateInfo-stage-02596 and -02597).
- **A point near the viewport's edge or the clip volume is clipped or culled by the device's own
  rule**: its `pointClippingBehavior` and the driver's handling of the viewport. Metal states no
  clipping rule for a point (`MTLPrimitiveType.point`; the MSL specification's `[[point_size]]`).
  Measured, Metal on an Apple M3 Max keeps a point whose centre lies within 1.5 w of the
  viewport's centre and covers its square whole, past the viewport's edge, where lavapipe covers
  no pixel whose centre lies on or past that edge and culls no point at the clip volume (the
  conformance case `point-viewport-coverage`).
- **A filled primitive reaching past a viewport whose edge crosses a pixel is covered at that edge by
  the device's own rule.** Metal states only that "fragments that lie outside of the viewport are
  clipped" (`MTLRenderCommandEncoder.h`, `setViewport:`), not which a pixel the viewport's edge
  crosses is. Measured (the conformance case `fractional-viewport-coverage`), Metal on an Apple M3
  Max covers every pixel the viewport overlaps, a pixel it holds only part of included, and so do
  RADV on an RX 7900 XTX, two Radeon 780Ms (1002:15bf, 1002:1900) and an RX 550, and NVIDIA's driver
  on an RTX 3090, an RTX 2080 and an RTX 5080 (m3-max-64279, 64288, 64280, 64323, 64286, 64287,
  64282). Two devices differ. lavapipe covers such a pixel only where its sample lies in the
  primitive clipped to the viewport (344 elements). The Iris Xe (ANV) covers every pixel the
  viewport's near edges cross, as Metal does, and no pixel its far edges cross (m3-max-64281): where
  that pixel's sample lies outside the viewport or on its edge, this is the difference (240
  elements); where the sample lies inside, it is a driver bug, a known gap of the runtime (its
  README, Known gaps of render encoders).
- **A line's tie-break is the device's own.** Lines draw by Bresenham's rule, which is Metal's
  (`bresenhamLines`, the runtime's floor): the pixels whose diamond the line exits, each within
  the one pixel Vulkan allows an implementation (`PrecisionNote::DeviceLineRasterization`). Vulkan's
  diamond is open (`|p - x| + |q - y| < 0.5`), so a line that only touches a diamond's vertex
  exits neither diamond the vertex joins, and no Vulkan control decides which pixel a conforming
  driver covers there. Measured, where a line of the conformance case
  `primitive-type-restart-indices` runs through the boundary point (3.5, 1.0), Metal on an Apple M3
  Max covers the lower pixel and NVIDIA's driver on an RTX 5080 the upper.
- **A memory-writing fragment function runs as many times as the device runs it.** Metal's GPUs
  were measured to run a memory-writing fragment once per covered sample; documented Metal names
  no count; here the count is the device's, for its memory writes and for anything it derives from
  them across runs, stored colours and sample mask included. The pipeline's fragment stage records
  `PrecisionNote::DeviceFragmentRuns` (section 9). The conformance case `a2c-off-4-alpha-50` records
  four runs of a fully covered pixel at four samples on an Apple M3 Max, and compares the runs word
  as nonzero alone.
- **A `Boolean` visibility result is the count of samples that passed.** Metal documents a
  nonzero result where a sample passed the depth and stencil tests and 0 where none did
  (`MTLVisibilityResultModeBoolean`); an Apple M3 Max writes 1. Here a `Boolean` mode writes the
  same precise count as `Counting`, nonzero exactly where a sample passed
  (`RenderCommandEncoder::VISIBILITY_NOTE`, section 9). The conformance cases `visibility-boolean-*`
  compare its word as nonzero alone.

**Resource IDs and identifiers**

- **A nil, stale or forged texture or sampler ID reads a defined element.** A texture ID no live
  texture holds reads a null descriptor: zeros, 0 to its size and level queries, and its writes
  dropped. A sampler ID no live sampler state holds samples through its array's default sampler,
  nearest and clamped to the edge, and a comparison through it answers 0. The Metal Shading
  Language leaves every member function of a null texture undefined (section 6.13.18), and
  Metal's read through a stale or forged ID is undefined.
- **Two live textures with one ID: a function reads the holder**, the texture placed there last.

**Render passes**

- **The runtime's `DontCare` store writes what the pass rendered** (`metal2vulkan-runtime`'s
  render encoder stores a `DontCare` attachment as it stores a `Store` one, its README's render
  passes). Apple's page for
  `MTLStoreAction.dontCare`: "The GPU has permission to discard the rendered contents of the
  attachment at the end of the render pass, replacing them with arbitrary data." and "Some GPUs
  may still store the contents back to the texture, but you can't rely on that behavior. You need
  to assume that GPU discarded the texture's contents." `MTLRenderPass.h` states only that it is
  the default store action. An Apple M3 Max on macOS 27.0.1 leaves the texels of a texture over a
  buffer as they were before a clearing pass; the runtime stores the clear there. The bytes
  of the buffer outside the texels keep what they held on both.

- **A barrier over the render targets orders what it names, as a Mac2 GPU's does.** The Metal
  Feature Set Tables give `MTLBarrierScopeRenderTargets` to Mac2 GPUs, not to Apple GPUs, and an
  M3 Max (macOS 27.0.1) orders nothing with it: a draw after one that samples its own pass's color
  attachment's texture reads the texels from before the pass, whatever the pass's load action or
  draws wrote before the barrier. The runtime orders the draws before such a barrier ahead of the
  next draw that reads an attachment of the pass, which reads what they rendered (the runtime's
  README, Render encoders). The conformance cases `pass-start-barrier-*` keep the M3 Max's texels
  and take the Mac2 answer as the one other.
- **A draw that reads its own pass's color attachment with no barrier reads, at the pass's first
  such draw, what the load and the draws before it rendered**: the runtime orders them ahead of
  that draw (the runtime's README, Render encoders). Metal defines no answer here, and the Macs
  disagree: an M3 Max (macOS 27.0.1) read the texels from before the pass at every pixel, in a
  pass that clears too; an AMD Radeon Pro 5300M and an Intel UHD Graphics 630 (macOS 26.7.1) read
  the earlier draw's texels at 98.6% and 99.99% of the pixels, and at all of them after a barrier
  over the render targets; the rest read the load's or the clear's texels.

**Heaps and memory**

- **The runtime answers as a Mac that keeps no texture compressed**, as a Mac with an Intel or
  AMD GPU keeps none, where an Apple silicon Mac's GPU keeps textures losslessly compressed: a new
  texture reads zero until written (the runtime's README, Textures), where an Apple silicon Mac's
  new `Private` texture reads zero or the compressed texture's fixed pattern, and a GPU blit
  through any texture keeps its bytes, where Apple's blit through a compressed texture writes some
  formats' canonical encoding. Metal leaves a new texture's contents undefined. The conformance
  suite's venue on the runtime presents no compression, so the answers such a Mac gives hold
  there.
- **The runtime's heap texture size and alignment are the device's own** (`metal2vulkan-runtime`'s
  `Device::heap_texture_size_and_align`, its README's textures): the device's own image
  requirements, not Metal's, the size rounded up to and the alignment raised to the device's
  `bufferImageGranularity`, so a buffer placed by the same answers never shares a granularity page
  with the texture. The answers differ by device, so the consumer places by this device's. The
  conformance suite does not compare the two numbers with Metal's; it holds each answer to three
  steps: (1) the alignment is a power of two; (2) a Private descriptor's texture is made at offset
  `align` in a heap of `align + size` bytes; (3) there, level 0's first slice written from a buffer
  reads back unchanged. A Shared, Managed or Memoryless descriptor is held to step 1 alone, as the
  runtime places heap textures in Private blocks only. A descriptor whose bytes the runtime does not
  move (depth and stencil both, multisample, `Depth32Float`) is held to steps 1 and 2.
- **Apple's private texture usage bits are kept** in `TextureUsage` (`NO_CONCURRENT_ACCESS`,
  `VIDEO`, `BLOCK_WRITES_ONLY`).
- **A heap's buffer and texture over the same bytes alias them, in the device's layout**, not
  Apple's tiled or compressed one.
- **The runtime's Managed copies read the consumer's bytes when the GPU reaches them**
  (`metal2vulkan-runtime`'s `Buffer::did_modify_range` and `Texture::upload_region`). A Managed
  buffer with a device copy (on a device that is neither integrated nor a CPU) takes its named
  ranges into that copy at the next commit of a command buffer, with commands or none, on any
  queue, by device copies ahead of that command buffer's commands; a Managed texture's uploaded region is one
  device copy from its named bytes into its image, ahead of every command buffer committed after the
  call on every queue. The copy reads the bytes as they are when the GPU runs it, so a write the
  consumer makes after the commit and before then may be seen. Metal leaves it undefined: an Apple
  M3 and an Intel UHD 630 take the bytes when the GPU reads them, an AMD Radeon Pro 5300M at the
  commit. That 5300M takes a named range at the first commit whose encoder binds its buffer (a
  dispatch of a zero grid included) and none at a commit that binds nothing; the runtime takes it
  at the next commit of any command buffer: the same commit where that one binds the buffer, an
  earlier one where it binds nothing, and never a later one. Only a write made after that earlier
  commit without naming it again can tell them apart, and Metal leaves its bytes undefined for
  the GPU.

**Function constants**

- **A function constant the request leaves undefined reads as zero.** A value computed from it
  alone is its type's zero (an `and`, `or` or `mul` with a constant that decides it is that
  constant), and a branch or `select` on one takes its false way; a value it meets with one the
  run computes is computed with that zero. As Metal's, `is_function_constant_defined` is false for
  it and an argument gated on it, negated or not, is absent. Metal makes the same pipeline of a
  specialized function (made with constant values, none among them too, or by
  `newFunctionWithDescriptor:`), and its reads of the constant vary from read to read: on an M3
  Max under macOS 27.0.1 it read `u + 3` as 0, `u * 7 + 1` as 1, `u == 0 ? 111 : 222` as 222 and
  `b ? 5 : 6` as 5. The compile is the same for a function made by `newFunctionWithName:`, which
  Metal makes no pipeline state of where its function declares a constant: its creation ends the
  process at every stage ("function ko cannot be used to build a pipeline state. Use
  newFunctionWithName:constantValues:... to get the specialized function"), whether the constant
  is required, read only where `is_function_constant_defined` says, or only gates an argument.
  `FunctionDescriptor::needs_specialization` answers it (`FunctionDescriptor::specialized` and the
  function's `function_constants_dictionary`); the runtime does not ask it and makes the state,
  each constant without a value reading as above. A linked function keeps its own rule (section
  3, "Linked functions: names and constants").
- **A fast-math float operation of a NaN function constant and a value the run computes stores
  that NaN.** An `fadd`, `fsub`, `fmul` or `fdiv` that carries `nnan` (Metal's default fast math)
  and whose operand is a function constant bound to a NaN is `poison` in the AIR once the
  constants are bound, so any value is a correct answer for it. The translation answers the NaN
  operand, its bits unchanged, as it does for the same operation without fast math, where that NaN
  is Metal's answer. Metal answers a NaN for a fragment function's output of it on an Apple M3 Max,
  an AMD Radeon Pro 5300M and an Intel UHD Graphics 630 under macOS 27.0.1. On the M3 Max a
  kernel's store of it writes nothing, `float` and `half` alike, and the memory keeps what it
  held, while the same kernel's store of the run value is written (the fixture
  `sfold_fast_poison`, case `2d9a7e3091e1f9d9`, job m3-max-12877; the case stays out of the
  corpus, whose comparisons are exact).
- **A linked function's constant without a value reads as zero too**, where the function was made
  specialized (section 3, "Linked functions: names and constants"). Metal makes that pipeline,
  and its read is undefined: a kernel storing `x * scale + 1` of such a function stored nothing on
  an M3 Max under macOS 27.0.1.
- **A program-scope constant that the function constants make a division by zero reads as an
  undefined constant reads.** A static initializer's scalar integer division or remainder by
  zero (`udiv`, `sdiv`, `urem`, `srem`) has a result the AIR leaves undefined, and so does
  everything computed from it: the translation reads it as zero, and a branch or `select` on it
  takes its false way, as for a constant the request leaves undefined. Metal makes that pipeline
  on an M3 Max under macOS 27.0.1 and stores nothing of the quotient, nor of a value chosen by it;
  where a loop's trip count is computed from it, the kernel stores nothing at all (the probe
  `static_init_division.m`). A vector division with a zero lane is still
  `Unsupported{StaticInitializer}`.

**Linked functions**

- **A reference of another type reads zero or its arguments' bits where Metal's read is
  undefined** (section 3, "Linked functions: types"). A parameter the reference passes nothing
  for reads zero, as Metal's read zero on an M3 Max under macOS 27.0.1 and on a Radeon Pro 5300M
  and a UHD Graphics 630 under macOS 26.7.1. An argument of another width passes its low bits to a
  narrower parameter, as the 5300M did for a `float` passed to a `half`, where the M3 Max and the
  UHD 630 left the parameter undefined (a kernel storing a value computed from it stored nothing);
  it passes its bits with zeros above to a wider parameter, and a result of another width or one a
  definition returning nothing gives passes the same way or reads zero, where the M3 Max stored
  nothing.
- **A reference passing a pointer for a value or a value for a pointer, a pointer of another
  address space, or an aggregate of another type is refused** (`Unsupported`,
  `Feature::Linking`), where Metal makes the pipeline: on the M3 Max each such read was undefined
  (the kernel stored nothing); on the 5300M a value for a pointer read the pointer's low 32 bits
  and a `constant` pointer for a `device` one read through it.

**Conversions**

- **`air.convert` with a rounding (`.rte`, `.rtz`, `.rtp`, `.rtn`) and `.sat` after its types is
  honored in a compute pipeline of OpenCL mode and is the conversion its types name elsewhere.**
  Apple's OpenCL compiler spells these on every `convert_<type>[_sat][_<mode>]`.

  In OpenCL mode (`ComputePipelineDescriptor::open_cl_mode_enabled`) Metal honors them. Measured
  over OpenCL C of our own, one work-item per operand (probe jobs m3-max-15651 on a Radeon Pro
  5300M and m3-max-15652 on a UHD Graphics 630, both macOS 26.7.1):

  | Form, OpenCL mode | Radeon Pro 5300M | UHD Graphics 630 | Runtime |
  |---|---|---|---|
  | integer of 8 to 64 bits to `float`, `.rtz`, `.rtp`, `.rtn` | rounded as named | rounded as named | rounded as named |
  | `float` to an integer of 8 to 64 bits, `.rte`, `.rtp`, `.rtn` | rounded as named, then clamped where `.sat` | as the 5300M, but a `uint` of 2^31 or more is `0x7fffffff` and a `uchar` with `.sat` above 127 is 127 | rounded as named, then saturated (the 5300M's answer) |
  | integer to integer, `.sat` | clamped to the destination's range | clamped to the destination's range | clamped to the destination's range |
  | a `half` side, any suffix | no pipeline | no pipeline | the conversion its types name |

  An integer to `float` by `.rte`, a `float` to an integer by `.rtz`, and an integer to an integer
  with a rounding and no `.sat` are the conversion their types name on both, as on the runtime.
  Neither creates a pipeline of OpenCL mode for a conversion with a `half` side (its library
  compiles), so no Metal answer exists for one, and the runtime makes the pipeline. The device has
  no directed conversion (Vulkan's integer-to-float conversion rounds in a direction of its own,
  its float-to-integer one truncates), so the runtime builds each exactly: a `float` is rounded to
  an integral value (`RoundEven`, `Ceil`, `Floor`) before the conversion, and the device's
  conversion of an integer to `float` moves one unit in the last place where it lies on the wrong
  side of the integer.

  Outside OpenCL mode Metal's compilers do not answer them alike. Measured over the owned fixture
  `kernel_convert_rounding_modes`'s edge values, one kernel per form (probe jobs m3-max-14380 on a
  Radeon Pro 5300M and m3-max-14381 on a UHD Graphics 630, both macOS 26.7.1; m3-max-14382 on the
  M3 Max, macOS 27.0.1), the runtime differs from a Mac in three places:

  | Form | M3 Max | Radeon Pro 5300M | UHD Graphics 630 | Runtime |
  |---|---|---|---|---|
  | float to integer, `.rte` | rounds to nearest even | truncates | no pipeline | truncates |
  | float to `half`, `.rtz`, `.rtp`, `.rtn` | `.rtz` as named; no pipeline for the others | nearest even | as named | nearest even |
  | float to integer, `.sat` | an unsigned destination keeps a negative's wrapped bits; a `long` of an infinity is 0, past its range wraps | an 8-bit destination is not clamped; a `long` is clamped to `int`'s range | no pipeline | clamps, as its conversion without the suffix |

  Everywhere else every Mac that creates the pipeline answers the conversion without the suffix,
  which is no difference: an integer to `float` or `half` by any rounding is rounded to nearest
  even (the M3 Max creates no pipeline for a directed one), an integer to an integer with `.sat`
  wraps (the M3 Max creates none), and a float to an integer by `.rtp` or `.rtn` truncates on the
  5300M (the others create none). The M3 Max's compiler service exits on the forms it does not
  compile, where the runtime compiles them all.
- **`vstore_half` and `vstorea_half` with a rounding (`_rtz`, `_rtp`, `_rtn`) store each lane as
  the device converts a `float` to `half`, to nearest even.** Measured over 32 values among them
  ties, overflows past the largest `half`, subnormals and zeros, with the `vstore_half4` forms alike
  (probe jobs m3-max-15563 on the M3 Max, macOS 27.0.1; m3-max-15564 on a Radeon Pro 5300M and
  m3-max-15584 on a UHD Graphics 630, macOS 26.7.1):

  | Form | M3 Max | Radeon Pro 5300M | UHD Graphics 630 | Runtime |
  |---|---|---|---|---|
  | `vstore_half_rtz`, `_rtp`, `_rtn` (`70000.0` by `_rtz`) | as named (`0x7bff`) | nearest even (`0x7c00`) | nearest even (`0x7c00`) | nearest even (`0x7c00`) |
- **A float past an integer's range converts to the integer's nearest bound, a `long` of an
  infinity too, where Metal's is 0.** MSL states two answers of a conversion from a floating-point
  type to an integer (MSL 8.6): it rounds toward zero, and a NaN converts to 0. Metal's compiler
  spells every such conversion of MSL as an `air.convert` call (measured over `int`, `uint`,
  `static_cast` and two-lane vector forms of `float` and `half`, in a kernel and in fragment
  functions, with fast math on and off: probe job m3-max-55309, Metal compiler 32023.922), which
  the runtime converts so, and which takes a value past the destination's range to its nearest
  bound. Metal on the M3 Max (macOS 27.0.1) answers the same at every width from `char` to
  `ulong`, from `float` and from `half`, but converts either infinity to a `long` as 0, where
  `ulong` of +inf is its maximum (conformance case `nan-to-integer-conversions`). A plain LLVM
  `fptosi` or `fptoui`, which Metal's compiler does not emit, is Vulkan's `OpConvertFToS` or
  `OpConvertFToU`, whose result for a NaN or a value its integer cannot hold SPIR-V leaves
  undefined: the device's own.

**Inline assembly**

- **An empty `asm volatile("")` is no compiler barrier.** It is the one inline assembly MSL
  accepts (a template, an operand or a clobber is an "illegal asm statement"), and it assembles to
  no instruction: the translator drops its call, `asm sideeffect "", ""` of type `void ()`, with
  no notice. In Metal it acts on the compiler alone: no memory access moves across it, and a loop
  that holds it is kept. Only a racy spin loop over plain (non-atomic) memory can tell the
  difference, and MSL's memory model leaves that loop undefined; a loop with a visible effect runs
  every iteration either way. Any other inline assembly, which only hand-written AIR can hold, is
  refused where an entry reaches it (`Feature::InlineAsm`).

**Kernels**

- **A buffer argument a kernel only reads is read-only memory to the device** where the AIR
  declares the argument unaliased (the `air-buffer-no-alias` attribute, which the AIR of Metal's
  current compiler carries on every buffer argument) and no store, atomic, volatile or coherent
  access of the kernel reaches it, through the argument or through a pointer its address may come
  back as: its views are decorated `NonWritable`, and under the identity domains (the runtime)
  its device-address loads go through `NonWritable` places. Under the identity domains a `constant`
  buffer argument is read-only memory too, whether or not the AIR declares it unaliased: Metal's
  `constant` address space is read-only. A dispatch that binds memory to such an argument and to
  an argument the kernel writes may read those bytes as they were before its own writes, where the
  device reads read-only memory through another path (RADV reads a uniform address of it with a
  scalar load). The attribute states that no other argument reaches that memory.
- **A dispatch that reaches a buffer bound off its argument's alignment is refused** (the
  alignment the kernel's AIR states for the argument, `air.arg_type_align_size`, at most 16;
  `RuntimeFact::AddressAlignment`): the runtime refuses the dispatch
  (`Refusal::BufferOffsetAlignment`) where the bound address, the buffer's `gpuAddress` plus the
  offset, is not a multiple of it, before anything is recorded, as Metal's API validation refuses
  such a bind offset ("must be a multiple of 4 but was set to 2" for a `uint *` argument).
  Without validation an Apple M3 Max (macOS 27.0.1) runs it, reaching each element at its address
  rounded down to the smaller of the element's alignment and 4, for loads, stores and pointers
  stored in a buffer alike (`crates/metal2vulkan-conformance/probes/misaligned_buffer_binds.m`).
- **A recursive call past the call stack yields zero**: the stack holds what Metal's byte model
  holds for the pipeline's `max_call_stack_depth`; a call that finds it full returns zero, where
  Metal's answer is corrupt.
- **The threads of a SIMD-group run in no order between barriers** under the identity domains (the
  runtime), where an Apple GPU runs them in lockstep: the Metal Shading Language orders their
  memory accesses only through barriers and atomics, and calls two of them that conflict between
  barriers a data race. A kernel holding such a pair carries the static note
  `PrecisionNote::SimdgroupAccessOrder` (`metal2vulkan-degrade-simdgroup-access-order`).
- **A simdgroup barrier in divergent code waits for its tangle**, as SPIR-V defines it. The Metal
  Shading Language requires every thread of the SIMD-group to execute a `simdgroup_barrier` that
  any of them executes in a conditional or a loop iteration, so Metal defines no answer there.
- **A state whose descriptor requires a threadgroup size answers that size's threads** as its
  `maxTotalThreadsPerThreadgroup`, lowered only to the device's own limit
  (`ComputePipelineState::max_total_threads_per_threadgroup`). A compute state compiled by Metal
  for the first time answers the same (M3 Max, macOS 27.0.1, each state of a source compiled
  once: 1 for 1x1x1, 21 for 7x3x1, 128 for 8x8x2, 1024 for 32x32x1). The runtime differs in
  that each state answers its own size: Metal hands a state of a function it has already
  compiled under another size the earlier state's limit, in the same process or a later one. A
  compute state of the same `MTLFunction` answered 16 after a 4x4x1 state, with an 8x8x1 size
  and with none, where the runtime's answers 64 and the compiled limit
  (`compute_pipeline_limits.m`).
- **A reset compute pipeline descriptor is a fresh one**, every field at its default:
  `ComputePipelineDescriptor::new` stands for `reset`, as `MTLComputePipeline.h` documents it
  ("Restore all compute pipeline descriptor properties to their default values", and of
  `maxTotalThreadsPerThreadgroup`, "If it is not set, returns zero"). Metal's `reset` on the M3
  Max (macOS 27.0.1) restores every other field and keeps three as the program set them:
  `threadGroupSizeIsMultipleOfThreadExecutionWidth`, `maxTotalThreadsPerThreadgroup` and
  `requiredThreadsPerThreadgroup` (true, 96 and 8 by 6 by 2 in `compute-descriptor-reset`, where
  the runtime answers false, 0 and 0 by 0 by 0). The tile and mesh render pipeline descriptors'
  resets restore every field on the same machine.
- **An OpenCL work-item read of a dimension past 2 answers OpenCL's value** in a pipeline of
  OpenCL mode: 0 for `get_global_id`, `get_local_id`, `get_group_id` and `get_global_offset`, 1
  for `get_local_size`, `get_global_size` and `get_num_groups`, as OpenCL C defines them. The M3
  Max (macOS 27.0.1) leaves unwritten the store of `get_global_id(3) + get_local_size(4) +
  get_global_size(5)`, where the runtime stores 2.

**Queues and command buffers**

- **The runtime schedules a command buffer once its submit thread is free**
  (`metal2vulkan-runtime`'s `CommandBuffer::status`, `add_scheduled_handler` and
  `wait_until_scheduled`). Apple defines `MTLCommandBufferStatusScheduled` as "All dependencies have
  been resolved and the command buffer has been scheduled for execution", and promises no time
  between the two. Every queue's command buffers go to the device's one `VkQueue`, which takes one
  `vkQueueSubmit2` call at a time, all made by the device's one submit thread, and the runtime
  schedules each command buffer there once that thread's earlier calls have returned, whichever
  queue they were for (the runtime's README, Queues, command buffers and events). So a command
  buffer whose wait an earlier submission's signal meets stays `Committed`, its scheduled handlers
  not run, for as long as that submission's driver call lasts, where an Apple M3 Max had scheduled
  it while the signalling command buffer still ran.
- **A command buffer that waits for a shared event's value is `Scheduled` as it is submitted**
  (`metal2vulkan-runtime`'s `CommandBufferStatus::Scheduled`, `Device::new_shared_event`), its
  scheduled handlers run, while the GPU still waits for the value, as for an `MTLEvent`'s value.
  Apple's MTLCommandBuffer.h describes `MTLCommandBufferStatusScheduled` as "All dependencies
  have been resolved and the command buffer has been scheduled for execution" (lines 47-48), and
  `encodeWaitForEvent:value:` as a command that "pauses execution of this command buffer until
  the specified event reaches a given value" (lines 405-406): execution pauses after scheduling.
  An Apple M3 Max with macOS 27.0.1 holds a command buffer that waits for an `MTLSharedEvent`'s
  value not yet reached at `Committed`, with no scheduled handler run, and its queue's next
  command buffer with it, until the value lands, and holds none for an `MTLEvent` (the
  conformance case `event-wait-behind-a-running-signal`). No documented Metal states that hold,
  so the runtime keeps none but while the signal's own queue holds it for room: its waiter reads
  `Committed` until the signal is staged, which the M3 Max's hold covers. The GPU's order and
  when each command buffer completes are the same either way; what moves is when the program's
  threads and handlers see `Scheduled`.

**Command queues**

- **A write whose queue holds nothing back is seen by another queue's command buffer committed
  after it** (`metal2vulkan-runtime`). A writer whose queue has room on the device's `VkQueue`
  (fewer than two submissions in flight always, the runtime's README) is submitted at once, ahead of every command buffer committed after it on any queue, so a later
  reader on another queue reads the write, on every device and every run: all 1024 words in the
  conformance case `fence-none-tracked-behind-one`, whose writer's queue has one busy command
  buffer in flight. That is the order Apple describes for a tracked resource: the framework
  prevents subsequent commands from running until write operations finish. Metal's 0 words there
  is measured, not documented: on an M3 Max with macOS 27.0.1, the copy on the other queue ran
  before the fill and read none of it in 5 runs of 5 (job m3-max-52432), and behind three busy
  command buffers likewise (job m3-max-52415); Metal's tracking ordered the copy only against work
  it had already submitted. The runtime's README states what was measured
  (`crates/metal2vulkan-runtime/README.md`, "Queues, command buffers and events").
