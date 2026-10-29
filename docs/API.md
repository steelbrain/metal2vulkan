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
also cover the runtime crate (`metal2vulkan-runtime`): its degrade notices and its deliberate
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
  correctly (`StageSource`, `PushHole`, `HeapSource`, `ImageSource`, `SamplerSource`,
  `UseTarget`, `SlotClass`, `StorageAttachmentView`, `VertexStride`, `TileMapping`, `CopySource`,
  `DrawCommand`, `PrecisionNote`, and Metal's `PrimitiveType`, `IndexType`, `CullMode`, `Winding`
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
  `description.recommended_enables(&policy)` at `vkCreateDevice`, then pass
  `with_enabled(&enabled)` so the description states what is enabled. `with_enabled` refuses a set
  that breaks one of Vulkan's co-enable rules, naming its VUID. `VkFeature::exposure()` and
  `VkExtension::requires(version)` say where each feature lives at the device's version, to chain
  the create info.
- Build one field by field.

```rust
let description = DeviceDescription::from_khronos_json(&json, None)?;
let device = Device::new(description, DevicePolicy::default())?;
```

`Device::new` refuses a device below Vulkan 1.2 and an inconsistent description or policy. A
`Device` is cheap to clone; every compile and plan takes it.

### The policy

`DevicePolicy` holds your product choices; together with the description they decide everything a
compile does. `DevicePolicy::default()` is `DevicePolicy::UNTRUSTED`:

- **Containment and isolation on.** Every access stays inside the resources the binder supplied,
  and guest shaders cannot reach each other's resources. Where the device enables
  `robustBufferAccess`, buffer accesses and vertex input are left to it. (MoltenVK advertises the
  feature without implementing it; there such accesses are not contained, as on native Metal.)
- **The device's arithmetic.** The device's own maths library, divisions, fused multiply-adds,
  `float` denormals, rounding order of `dot`/`mix`/`fast::sqrt`, the grouping and fusion of
  fast-math sums and products (each operation in the AIR's order, its own fast-math flags its
  decoration), NaN bits, and the operand of `fmin`, `fmax`, `min3`, `max3`, `clamp` and
  `saturate`, on every device and with no exact form.
  Each records a precision note with its bound; on MoltenVK on an Apple GPU, whose `float`
  functions are Metal's own, they are Metal's answers, but for a NaN's bits where the device is
  not measured to answer Metal's.
- **Fast native forms off Apple's GPUs.** The device's own sampler (its filter, footprint, level,
  anisotropy, texel conversion and comparison, a two-plane YCbCr texture's filter among them),
  with no switch: the device's answer is the contract, recorded as `PrecisionNote::DeviceSampling`
  with its measured bound. The device's own `half` rounding and its blend of a `half` color have
  no switch either: each records its note. [FAST-PATHS.md](FAST-PATHS.md) lists these device gaps
  with their bounds. On MoltenVK on an Apple GPU, whose sampler and `float` functions are Metal's
  own, most of them change nothing.
- `DevicePolicy::exact_answers()` selects Metal's answer for each of the runtime's choices that
  offers one (the batch order and render lowering below, the heap texture compression, command
  buffer completion and textures over host pages) and the AIR's rounding of each `half` operation
  on a device without `shaderFloat16` (below), each listed in [FAST-PATHS.md](FAST-PATHS.md)
  with its bound and the measured cost of its exact form.
- **`half` chains held at `float` precision without `shaderFloat16`** (`HalfChainPolicy::AtExits`,
  the default; changes emitted code). A device without 16-bit floats has no `half` arithmetic, so
  a module computes each `half` `+ - *` in `float` and rounds it by the device's quantize. An
  operation whose flags hold `contract` and `reassoc` and whose result only such operations read
  is left unrounded: the chain rounds where a value leaves it, one quantize per exit instead of
  one per operation, recorded `FastMathRewrites`. `half_chains = HalfChainPolicy::EachOperation`,
  which `exact_answers()` selects, rounds every operation. A device with `shaderFloat16` computes
  `half` alike under either value.
- **Batches of a queue ordered by their ends** (`BatchOrderPolicy::Ends`, the runtime's choice; no
  pipeline differs). On ANV, RADV and NVIDIA's driver, each measured to start no kernel submission
  of a `VkQueue` before the one before it completed, each batch of command buffers ends with a
  barrier ordering its work before every later command of its `VkQueue`: a batch that depends on
  an earlier one there begins with no barrier of its own, and a command buffer's GPU start is
  taken once the work before it completed, which is when such a driver starts it. Nothing is
  announced: on those drivers the work's order, its results and its GPU times are the same under
  either value. `batch_order = BatchOrderPolicy::Exact`, which `exact_answers()` selects, orders
  each batch after the earlier ones of its queue only by its work's dependencies on theirs, as
  Metal does (an M3 Max starts an independent command buffer committed behind running work while
  that work runs), and takes a command buffer's GPU start as its work begins. Every other driver
  takes that form under either value.
- **A render encoder's calls lowered inside each call** (`RenderLoweringPolicy::AtCall`, the
  default, which `exact_answers()` selects too; the runtime's choice, no pipeline differs): each
  call works out and records its commands before it returns, and returns every refusal.
  `render_lowering = RenderLoweringPolicy::Worker` logs each call and lowers it later, in order,
  before the command buffer's submission: in batches of 32 calls on the device's lowering thread
  while the program goes on. A program that runs 16 batches ahead of that thread waits for it in
  its next call, and one 4 encoders' ends ahead in its next `end_encoding`. `end_encoding` hands the encoder's end to that thread too and returns; the next
  call that follows the end in its command buffer (another encoder, an event or semaphore
  command, `commit_failed`, `refusal`, `used_resources`, `ended_encoders`, `failed_pipeline`,
  `dropped_draw`)
  lowers on the program's thread what the thread has not, and a rebind or detach of host pages
  and `Device::work_counts` lower every encoder's first. `commit` makes its checks at the call
  (committed already, an encoder open, one abandoned) and returns, its status `Committed`, its own work left to run once the end is lowered, on the
  thread that lowers it; while such a commit waits, every later commit of the device waits
  behind it, so the device's scheduler still takes commits in their order across its queues. A
  wait for the command buffer lowers on the waiting thread what that thread has not. A refusal
  only the commit's own work finds (an executed indirect command's recording, the commit's
  uploads) fails the command buffer, nothing of it run, with `CallRefused` of `commit` at index
  0 where no call of it was refused before: it commits again with nothing made for its commands,
  so it completes, in its queue's order, its completed handlers run and its waits return, as
  every committed command buffer of Metal's does. An end that checks the depth/stencil
  attachment's store actions or store action options (one `Unknown`, options set, or any store
  action or resolve target set by a call) is lowered inside `end_encoding`, which returns its
  refusal. The call still makes Metal's checks of its arguments and of the encoder's state, and returns
  their refusals: the argument tables' and state setters', the objects' devices, the pipeline
  state's against the pass, and a draw's kind, arguments and settings. It copies the bytes it
  was lent and holds the objects it names. A call those checks do not cover is lowered inside
  the call, as under `AtCall`: store actions, resolve targets, visibility result modes,
  threadgroup memory lengths, fences, patch and mesh draws, a draw while a visible-function
  table or a vertex buffer of more than 4 GiB past its offset is bound, and any call while a
  buffer of the device is detached. From a pipeline state that fetches the pass's colors, reads
  its tile memory, uses an imageblock, draws patches or meshes, or renders a color in another
  format than its attachment's; a tile dispatch; an execution of indirect render commands; a
  memory barrier; and a draw that gives the pass its hidden depth and stencil, every later call
  of the encoder is lowered inside the call, and so is every call of a pass the device draws in
  color groups. A refusal only the lowering finds (a runtime gap of the draw's variant or
  instance, out of memory, a Vulkan error) fails the command buffer instead:
  `CommandBufferError::CallRefused` names the call (`EncoderCall`: its Metal selector and its
  index among the encoder's calls; the end is `endEncoding`, the index after the encoder's last
  call), `CommandBuffer::refusal` returns the error the call would
  have returned, and that encoder and every one after it in the command buffer run nothing.
  Notices a logged call's lowering announces name the call (`DegradeNotice::call`) and may reach
  the message sink on the lowering thread, and `DeviceOptions::lowering_refuses` turns the kinds
  it lists into such refusals (section 9).
- **A pipeline still being made waited for a bound** (`PipelineWaitPolicy::Bounded`, the
  default; the runtime's choice, no pipeline differs). A render or compute state's creation
  hands the driver's making of its predicted pipeline to the runtime's pool and waits for it
  `DevicePolicy::pipeline_creation_wait` at most; a committed command buffer waits until its
  commit plus `DevicePolicy::pipeline_commit_wait` at most, one deadline its commands share, for
  its draws' and dispatches' (a compute encoder's, a render pass's tile dispatches) pipelines
  still being made, an execution's replayed draws among them (whose pipelines the commit hands to
  the pool), then records each such command as nothing (`draw-dropped-still-making`,
  `dispatch-dropped-still-making`, the first named by `CommandBuffer::dropped_draw` with
  `VK_TIMEOUT`, counted `WorkCounts::draws_dropped_still_making` and
  `dispatches_dropped_still_making`), and later command buffers draw and dispatch with the
  pipeline once made. Both bounds are 4.5 s by default. What one call waits for the driver's
  makings: a state's creation, its creation wait at most; a `commit`, until its own deadline at
  most (it records there what it can, an execution of render commands the CPU writes among
  them; every other recording runs on the runtime's scheduler thread, which no call waits for). A
  creation waits on no command buffer's deadline and a commit on no creation's, so a host step
  that creates a state and commits a command buffer waits their sum at most, 9 s by default,
  under a guest's 10 s FIFO check, where a driver's cold compile took 11.6 s (MoltenVK with a cold
  Metal compiler service). Made until made under this value too, in the call that makes them: a
  draw that records nothing, a draw in a pass of custom sample positions, and one recorded again
  into another instance of its pass (each in the draw's call or an execution's replay), a device
  replay's draws (in its `commit`), and a making that failed, made again by its command's
  recording. `pipeline_waits = PipelineWaitPolicy::Exact`, which `exact_answers()` selects, makes
  the predicted pipeline before the creation returns, as Metal compiles there, and waits for
  every draw's and dispatch's until made.

Loops run as written: no execution bound is translated in.

### What the device answers

```rust
let device = committed_device("moltenvk-m3")?;
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
- The profile's facts: `enabled()`, `limits()`, `images()`, `fragment()`, `descriptors()`, the
  driver workarounds and measured deviations it takes (`workarounds()`, `driver_deviations()`),
  its `subgroups()` model and `simd_tier(...)`, how it contains an out-of-bounds access
  (`containment()`, `checks_buffers_in_shader()`, `checks_atomics_in_shader()`), and the Function
  storage a thread holds (`function_storage()`: `FunctionStorage::Measured` bytes for a measured
  device, or `FunctionStorage::Assumed` with the 16 KiB floor, past which a vertex or fragment
  function runs with `FunctionStorageAssumed` up to 192 KiB and any other stage is refused).
- `neutral_kinds()`: the neutral resources plans bind where nothing compatible is bound. Create
  each once, up front.

### Measured driver behavior

Where Vulkan leaves an answer to the driver and a driver's answer is Metal's, the profile relies on
it only where that driver was measured to give it, keyed on the description's driver identity
(`driver_id`, `vendor_id`, and `device_id` where one GPU differs from another; MoltenVK's
`device_id` is the macOS release and the Apple GPU family, so the M3 Max and the M4 share one under
macOS 27), never on its `driver_version` and never on a claim of the description itself. A row holds
at every release of its driver. A GPU that reports more than one `device_id` is read by the one it
was measured under (`DeviceDescription::canonical_device_id`): the Radeon 780M of Hawk Point
(`0x1900`) as Phoenix's (`0x15bf`). These are `DriverDeviation`s (simdgroups that run in lockstep,
Metal's own `float` functions, sampler and canonical NaNs, color writes, ...), measured on MoltenVK
on Apple's M3 Max and M4, with a few per-GPU facts measured elsewhere. A driver or GPU that was not
measured takes the route each deviation's docs name for it: the device's own answer, recorded as a
precision note, or, where the answer decides a capability, its emulation or a refusal.

Metal semantics the translation keeps on every device, beyond Vulkan's: Metal's 32-lane
simdgroups (including lockstep races on threadgroup memory within a barrier interval). `fneg` and
`fabs` are the device's `OpFNegate` and `FAbs`, so a NaN's payload and a denormal are the
device's (`native-nan-encoding`, `device-denormals`). A NaN's bits, the
operand `fmin`, `fmax`, `min3`, `max3`, `clamp` and `saturate` pick beside a NaN or between two
zeros, a `half` read from its bits, which is the device's widening on every device (a NaN's
bits, and a denormal on a device that may flush 16-bit ones), and a `half` rounded from a value
below the smallest normal `half` on a device without `shaderFloat16` (its `OpQuantizeToF16`'s
zero) are the device's, each recorded as a precision note.

What the runtime enables for itself outside the vocabulary enters no enable set, compile profile
or fingerprint: among it the extensions of its dma-buf fallback for no-copy memory whose pointer
the device refuses (`HostMemory::with_dma_buf`), `VK_EXT_external_memory_dma_buf` with
`VK_KHR_external_memory_fd`, and `VK_EXT_image_drm_format_modifier` with
`VK_EXT_queue_family_foreign` beside them, wherever a device lists them.

A texture over host pages is one over a no-copy buffer's (imported by their pointer or as
dma-bufs), or over a host-visible buffer of the runtime's own longer than 32 MiB where the device
imports host pointers and the memory class's best type for the buffer takes such an import: that
buffer lies in fresh anonymous pages imported by their pointer, zero as mapped, so that nothing
writes its zeros (`WorkCounts::fresh_page_buffers_made`). A class whose best memory type takes no
such import -- a discrete card's write-combined class, whose best type for the buffer is a
device-local host-visible one -- is placed as ever and its bytes set to zero on the host; the
buffer reads zero either way, which is all a program sees.

An image bound to imported memory starts in `VK_IMAGE_LAYOUT_UNDEFINED`
(`VUID-VkImageCreateInfo-pNext-01443`), and its first move, like any layout transition, may read
and write every byte bound to it, where Metal keeps every CPU write to bytes a command buffer does
not write. Under the default `HostPageTexturePolicy::InPlace` the runtime relies on the driver
there, checked: a texture over host pages whose declared usage only samples or reads it (none of
`RenderTarget`, `ShaderWrite` and `ShaderAtomic`) is a linear image aliasing the pages (one over
dma-buf pages a `DRM_FORMAT_MOD_LINEAR` image over a dma-buf of its own pages, acquired from
`VK_QUEUE_FAMILY_FOREIGN_EXT`) wherever the device's own check passed for that kind of pages
(`WorkCounts::host_textures_in_place`, `dma_buf_textures_in_place`,
`fresh_page_textures_in_place`). The check makes such images of every texel size, small and
3840x2160, over each kind of pages while a CPU thread writes beside their texels, and passes a kind
only where every first move kept every byte. It runs on a thread of its own, which the first such
texture starts, once per driver, driver version and runtime release: its verdict is kept in the
shader cache's directory (a device without one checks once per process) and recalled as the next
device is made (`WorkCounts::in_place_checks_run`, `_passed`, `_failed`, `_unchecked`,
`_recalled`). Such a texture's commands read the texels in the memory the buffer's commands read,
so a CPU write reaches them as it reaches the buffer's.

Every other texture over host pages is a shadow (`shadowed-texture`, the notice naming why): under
`HostPageTexturePolicy::Shadow` (`DevicePolicy::exact_answers`); one whose usage renders into it or
writes it; one made before the check answered, or where it failed or made no image; and one no
image in place serves (the bytes' pitch, placement or memory, or over dma-buf pages no dma-buf of
the texture's own pages). `WorkCounts::host_texture_shadows_*` counts each reason apart (one
whose usage renders into it, `_render_target`, apart from one that writes it through a shader
alone, `_shader_write`). A shadow
is an optimal image of the runtime's own memory, whose texels the encoders copy in from the pages
before a use and back (merged back: only the bytes the commands changed) after a write; no image
is bound to the pages, and `Texture::vk_image` gives `None` for it. A texture in place renders and
is written beyond its declared usage, as plain Metal allows, through its usage shadow and an
attachment stand-in, whose copies touch its texels alone.
On a discrete GPU, a texture whose policy and usage take the image in place, over a no-copy
buffer whose memory carries a log of its writes (`HostMemory::with_dirty_pages`), tracked, one
slice laid out as one copy, is instead a device copy (`WorkCounts::device_copy_textures`, whatever
the check answered): a shadow whose copies in take only the bands of rows the log reports changed
(`dirty_copies_in_gated` and kin). Its notice is not announced: it holds the bytes wherever a
command reads it.
A texture placed in a heap over dma-buf pages is an image over a dma-buf of its own pages, which its
first submission acquires from `VK_QUEUE_FAMILY_FOREIGN_EXT`; a linear image over a buffer of the
runtime's own memory starts `VK_IMAGE_LAYOUT_PREINITIALIZED`, and its move from there keeps its
bytes. FAST-PATHS.md's costs and known gaps state what this costs and differs from Metal in.

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
  `ArgumentBufferLayout::from_descriptors` and `from_buffer_binding` give the layouts of Metal's
  other two encoder constructors. Each argument is 8 bytes: a buffer's address plus offset, or a
  resource ID.
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
placeholders (a kernel's local size is 1). Create each pipeline from a planned dispatch's
`DispatchRegion` (section 6), with the region's `specialization`, `stage_flags` and
`required_subgroup_size`. A required subgroup size needs `subgroupSizeControl` and the stage in
`requiredSubgroupSizeStages`; full subgroups need `computeFullSubgroups`. A pipeline compiled for
one device pins only what that device can take; `DeviceProfile::check` tells whether another
device can.

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
  such functions itself: the handle-table and root-table searches are unrolled branch-free steps,
  emitted once and called at every access.
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

### Other compute features

- **Visible functions and recursion.** Linked visible functions, function tables and
  intersection function tables compile; recursion through visible functions runs on a stack of
  frames sized by `max_call_stack_depth`. A call that finds the stack full returns zero.
- **Binary functions, required threadgroup sizes, buffer mutability.** These descriptor fields
  change no translation and are in no compile key; the runtime checks them as Metal does.
- **Dynamic libraries.** `DynamicLibrary::new(library, &budget)` checks a dynamic metallib as
  Metal's `newDynamicLibrary:` does; pipelines link its exported functions.

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

- `stages`: vertex, then fragment when rasterization is enabled, with three geometry stages
  before the fragment stage when the fragment function reads `[[barycentric_coord]]` on a device
  without `fragmentShaderBarycentric`, or the vertex function selects its slice or viewport on a
  device without the output (below). A `VkPipeline` takes the stages
  `RenderPipeline::runs_stage(at, topology)` says its topology runs;
- `layout`: one pipeline layout for every stage;
- `vertex_input`: Vulkan bindings and attributes. An attribute Vulkan cannot fetch as described is
  pulled in the shader instead, and so is every attribute on a device without
  `robustBufferAccess` (unless containment is off);
- `graphics`: the fixed-function state (topology, attachment formats and blend states, sample
  count, dynamic states, input attachments, ...). Build
  `VkPipelineColorBlendStateCreateInfo::pAttachments` from `graphics.color_blend_attachments()`.

Where Metal's rules differ from Vulkan's, the pipeline gives Metal's answer:

- A color attachment no fragment output names keeps its texels (an empty write mask).
- What Metal refuses, the link refuses as `InvalidDescriptor(ColorAttachment)`: an output with
  fewer components than its attachment's channels, a blend that reads the second output without
  the first, alpha to coverage over an integer attachment 0. Function-constant values that put two
  outputs at one attachment are `InvalidDescriptor(DuplicateColorOutput)`.
- On a device without `independentBlend`, attachments the function does not write copy the blend
  state of those it does; where those differ the pipeline is refused, naming the feature.
- A packed 16-bit attachment (`A1BGR5Unorm`, `ABGR4Unorm`, `BGRA4Unorm`) of a format the device
  samples but does not render (the RTX 5080) renders through an `R16_UINT` view of its texture
  (`metal::format::WordAttachment`, `LoweringInputs::word_colors`): the fragment function writes
  the word Metal stores and `graphics` gives the attachment that format; a pass clears it to the
  word of Metal's clear. Exact, for a write unblended (or blended one and zero); under a mask of
  some of its channels the function merges the word into the texel in the interlocked storage
  form (a `GraphicsState::storage_attachments` entry of `R16_UINT`) where the device orders fetch
  through the pixel interlock. Blended otherwise, there too the function computes Metal's blend of
  a `half4` or `float4` output over the word it reads, within one code of Metal's
  (`PrecisionNote::DeviceColorConversion`: the attachment holds codes where Metal's tile holds a
  clear's or an earlier blend's result in the pass as a `half`, and under fast math Metal sums a
  few states otherwise; `docs/FAST-PATHS.md`). A blend of the second source, and a blend or
  partial mask on a device without the interlock, is refused as the device lacking the format; a
  resolve of one, and a blend of an output of fewer lanes, is `NotImplemented`; a color input of
  one is read in the interlocked storage form, the word decoded to each channel's code over its
  maximum code rounded to `half`, nearest even, which is Metal's value (measured on an Apple M3
  Max under macOS 27.0.1 for a `float4` and a `half4` input, every code of each 4-bit channel).
- An 8-bit sRGB attachment of one or two channels (`R8Unorm_sRGB`, `RG8Unorm_sRGB`) the device
  does not render (the RTX 5080: it renders neither `R8_SRGB` nor `R8G8_SRGB`, and holds the
  latter in its UNORM twin) renders through an `R8_UINT` or `R16_UINT` view of its texture the
  same way: the fragment function writes the bytes of the device's sRGB encode
  (`DevicePackConversion`), a pass clears it to the bytes of Metal's clear, and a partial mask
  merges the word in the interlock, or over the attachment's input attachment on a device that
  orders fetch through rasterization-order input attachments (lavapipe;
  `RuntimeFact::WordInputMerge`). A blend of one
  other than one and zero is refused as the device lacking the format; a color input or resolve
  of one is `NotImplemented(Format)`.
- An `RGB9E5Float` attachment the device does not render (the RTX 5080) renders through an
  `R32_UINT` view of its texture: the fragment function writes Metal's word of a `float` output
  (each channel clamped to `[0, 65408]`, a NaN 0, the exponent of the largest, every mantissa
  truncated), and a pass clears it to the word of Metal's clear. Exact. A `half` output to one
  is `NotImplemented(Format)` (Metal rounds the `float` the `half` was converted from, half up);
  a blend, or a mask of some of its channels, is refused as the device lacking the format; a
  color input or resolve of one is `NotImplemented(Format)`.
- A fragment function that reads a color attachment (framebuffer fetch) reads it in an ordered
  form wherever the device has one, and is refused (`Unsupported{FramebufferFetch}`) where it has
  none: an unordered read could see a pixel before an earlier fragment's write, which Metal never
  shows. The ordered forms are rasterization-order attachment access, or the interlocked
  storage form, which binds the attachment as a storage image viewed in
  `StorageAttachmentUse::view_format()`: its own format, an 8-bit sRGB format's unsigned-integer
  view, or, where the device declares a BGRA-order format neither typed nor without a format, its
  RGBA-order twin (the shader exchanges red and blue). A `[[stencil]]` output uses
  `VK_EXT_shader_stencil_export`.
- `[[barycentric_coord]]` reads `BaryCoordKHR` or `BaryCoordNoPerspKHR`, whose vertex order is
  Vulkan's first-vertex order and Metal's for every primitive type. A `float3` read takes the
  first two weights and computes the third as `1 - (x + y)` in `float`, as Metal does, not the
  device's own third lane.
  On a device without `fragmentShaderBarycentric` the weights are a varying instead: the pipeline
  carries a generated geometry stage for each topology class (points, lines, triangles, in that
  order), which writes `(1, 0)`, `(0, 1)` and `(0, 0)` at the primitive's vertices in their order
  and passes the vertex function's outputs, clip distances, point size and the primitive ID
  through, and a draw runs its class's. A `centroid` or `sample` qualifier reads the pixel
  center's weights, as Metal does. That needs `geometryShader`; without it, or for what a geometry
  stage cannot carry (tessellation, mesh pipelines, a vertex function's own layer or viewport
  index, amplification without `multiviewGeometryShader`, a written point size in a point draw
  without `shaderTessellationAndGeometryPointSize`, more varyings than the geometry limits), the
  pipeline is refused, naming what it needs. Measured on the Iris Xe (ANV), such a pipeline takes
  about 2.6 ms longer to create, and a geometry stage's presence costs about 15% of fill rate.
- A primitive renders into the slice and through the viewport its first vertex selects, the range
  check applying to that vertex's index, and its fragments read the first vertex's indices as
  written, where Vulkan asks every vertex of a primitive to write the same `Layer` and
  `ViewportIndex`. A vertex function stores an index to its builtin only where it writes
  the same at every vertex of a primitive, decided from its structure (computed from constants,
  `constant` memory, the instance, the amplification and per-instance attributes, as
  `StageIo::uniform_outputs` lists), or on MoltenVK, which hands the builtins to Metal
  (`DriverDeviation::MetalFirstVertexIndices`). Otherwise, and on a device without
  `shaderOutputLayer` or `shaderOutputViewportIndex`, the vertex function stores the index it
  selects to a flat varying and the same generated geometry stages (one set serving the weights, the
  primitive ID and the indices) store `Layer` or `ViewportIndex` from the primitive's first vertex
  at every vertex they emit; where it stores one index so, it stores the other so too. The variant
  for a pass of one layer selects no slice and carries no stage. A device without `multiViewport`
  draws through one viewport, so there the vertex function stores no viewport index at all; a draw
  that selects among more than one viewport there is refused at the draw,
  `Unsupported(LayeredRendering)` needing `multiViewport`. Without `geometryShader` either, a vertex
  function that selects its slice is refused needing either feature (its pass of one layer still
  runs), and so is one that selects its viewport on a device with `multiViewport`; one whose index
  differs among a primitive's vertices is refused needing `geometryShader`. The stages cannot serve
  a post-tessellation vertex function or an emulated mesh function, nor sit beside a vertex function
  that writes the builtins itself (a device with the outputs but without
  `fragmentShaderBarycentric`): those are refused as not implemented. A fragment function that reads
  an index no stage writes, on a device that can declare neither builtin, reads 0, Metal's value
  with no vertex write.
- `[[primitive_id]]` counts the draw's primitives in each instance from 0; vertex start, base
  vertex and base instance do not move it. A fragment stage declares `PrimitiveId` through
  `geometryShader` or `tessellationShader`. Without both it reads a flat varying the vertex
  function writes (`IoRole::PrimitiveId`, `PushField::VerticesPerPrimitive`); an indirect draw's
  arguments are then read on the CPU only without `shaderDrawParameters`. An indexed draw runs
  de-indexed (`VariantKey::primitive_id_indices`: needs device addresses, shades every index).
  `NotImplemented(PrimitiveId)`: an indexed strip, an indirect indexed draw, indices past their
  buffer, a vertex function that writes memory in that form, and a mesh pipeline.
  A draw of Metal's private quads or quad strip, which the consumer lists as two triangles a quad
  (`BindState::quad_primitive_ids`), runs the fragment variant that reads `PrimitiveId` shifted
  right by one (`VariantKey::quad_primitive_ids`), as Metal counts one primitive a quad. A
  private type's draw whose vertex function writes the varying, or writes memory (Metal runs it
  once for each vertex of its list, where an indexed draw may run a shared vertex once), runs the
  private list form (`VariantKey::private_list`, `BindShape::private_list`; needs device
  addresses): the consumer draws its list without indices from its first index as first vertex,
  and the vertex function reads each position's index, and Metal's base vertex in the word before
  the list, at the list's address (`PushHole::CaptureIndexAddress`), adding them as Metal's vertex
  ID wraps, so each position runs once and the positions count Metal's primitives.
  Behind an emulated post-tessellation vertex function the varying carries the patch's position
  in its draw (tessellation, below).
- A device draws to at most the lesser of its `maxColorAttachments` and
  `maxFragmentOutputAttachments` color attachments at once (four at Vulkan's floor), where Metal
  takes eight. A pipeline whose attachments with a pixel format number more draws them in groups
  of that many, in the order of their indices: variant `VariantKey::color_group` of the pipeline
  draws one group, its fragment function writing that group's attachments alone at locations `0..`
  and its `graphics` holding them alone (`graphics.color_group`, a `ColorGroup`;
  `ColorGroup::of(present, width, group)` names a group, and `ColorGroup::width_on(profile)` is a
  device's group width). A pipeline whose
  attachments number no more but reach an index at or past the count (`color(0)` and `color(5)` on
  a device of four) takes one group, renumbered. Every group's variant has the same vertex module;
  where there is more than one group, their fragment stages' `FragDepth`, `SampleMask` and
  `FragStencilRefEXT` are decorated `Invariant` and record
  `PrecisionNote::ColorAttachmentGroups`. What each group's draw would run again or read from
  another group is the pipeline's limit refusal (`InterfaceComponents` at `MaxColorAttachments`):
  a vertex or fragment function that writes memory, alpha to coverage, a tessellation pipeline, a
  `[[color(n)]]` input of another group, imageblocks and raster order groups. Mesh and tile
  pipelines take no groups. The runtime draws such a pass group by group, each group's colors in
  a pass of its own over a twin of the depth and stencil (the runtime's `render/color_groups.rs`;
  what it refuses there is `RuntimeGap::ColorAttachmentGroups`).
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
evaluation stage, and the fragment stage. It draws patches only (`DrawCall::Patches` or
`PatchesIndirect`), with the factors from a tessellation factor buffer. The device's tessellator
orders the triangles (`PrecisionNote::TessellationTriangleOrder` where it is not Apple's).

On a device without `tessellationShader` the pipeline is an emulation with no tessellation stage:
the function is its vertex stage (`TRIANGLE_LIST`), and `graphics.emulated_tessellation` states
the tessellation state it bakes in. Each patch draw's `DrawCommands::tessellation_dispatch` is a
compute scan (`bind::tessellation_scan_module`, one workgroup) to record before the render pass,
and chunk draws of up to 2^30 triangles that replace `draw`; `TessellationDispatch` states the
contract (buffers, order, barriers). The scan binds four storage buffers; the vertex function
binds two to four beside its own, counted against `maxPerStageDescriptorStorageBuffers`
(`DescriptorCount` past it). A fragment `[[primitive_id]]` reads the patch's position in the draw
from the primitive ID's varying (`NotImplemented(Tessellation)` behind tessellation stages), and
an indirect patch draw's `[[base_instance]]` is read on the GPU.

### Mesh pipelines

`Device::compile_mesh_pipeline` compiles a `MeshPipelineDescriptor` (an optional object function,
a mesh function and a fragment function), and `compile_mesh_variant` its variants. Its plan draws
mesh draws only (`DrawCall::MeshThreadgroups`, `MeshThreads` and their indirect forms). With
`VK_EXT_mesh_shader` the stages are task and mesh shaders. Without it (MoltenVK, older ANV) the
pipeline is an exact emulation: the object and mesh functions run as compute stages that write the
mesh's vertices and primitives as records into buffers, and a synthesized vertex stage draws those
records; the plan says which dispatches and draw to record.

A grid Metal draws nothing for (an axis past `bind::METAL_MESH_GRID_AXIS`, or more than
`bind::METAL_MESH_GRID_TOTAL` threadgroups, measured on the M3 Max) draws nothing in every form:
a direct draw's command is zeroed, an object function's wrapper launches nothing for such a grid,
and an indirect draw reads its counts on the GPU. With the extension, an indirect draw's
`DrawCommands::mesh_arguments` is a pass to record before the render pass
(`bind::mesh_arguments_module`, one invocation): it writes the draw's counts into the consumer's
scratch (`MeshArguments::scratch_bytes`), which the draw reads in place of Metal's arguments.

An object grid runs at any count Metal runs, past the task stage's limits too, on a device with
device addresses, `shaderInt64`, `multiDrawIndirect` and `drawIndirectCount` (every committed
profile with the extension): the object function is compiled folded, and each draw runs its grid
in pieces within the task limits (`bind::fold_object_grid`), its object stage reading the
threadgroups Metal runs and its grid from a 32-byte region record at its
`PushHole::ReplayRegionAddress`. A direct draw's fold is `DrawCommands::object_fold`, whose bytes
the consumer uploads (the record, and its pieces' commands where `draw` is indirect); an indirect
draw's arguments pass (`MeshArgumentsForm::FoldedObject`) writes the record, the pieces and their
commands, which `draw` (`DrawMeshTasksIndirectCount`) reads. Metal's threads per grid wrap to 32
bits along each axis, as its builtins report them; a direct draw runs only the threadgroups those
threads fill, rounded up, an indirect one every threadgroup (`bind::metal_object_grid`). Every
device runs a grid of up to `bind::METAL_OBJECT_GRID_TOTAL` (2^29) threadgroups, the most Metal's GPU
ran within its watchdog's 5 seconds. A larger one, which Metal's GPU discards, draws nothing and
fails its command buffer with Metal's error (the runtime's `CommandBufferError::GpuRecoveryVictim`):
a direct draw says so in `DrawCommands::discards_command_buffer`, and the consumer fails the command
buffer from the draw's encoder on, as Metal does; an indirect draw's arguments pass sets
`StatusBits::MESH_GRID_DISCARDED` in the command buffer's `ExecutionStatus`. Where the device cannot
launch a grid Metal draws, the draw draws nothing and the command buffer fails with the runtime's
own error (`CommandBufferError::MeshGridPastLimits`, not Metal's): an indirect draw's object grid past
the task limits of a device that does not fold object grids, or a mesh grid past the mesh limits,
through `StatusBits::MESH_GRID_PAST_LIMITS`, or a fold of more pieces than
`MeshArguments::OBJECT_PIECES` (only under task limits below Vulkan's required least), its `past`.
A direct object grid past the task limits of a device that does not fold is refused naming the
limit.

Without the extension, an indirect draw with an object function first dispatches an object
arguments pass (`MeshDispatch::object_arguments`, `bind::mesh_object_arguments_module`, one
invocation over the scan's bindings): it reads Metal's arguments, reserves the object threadgroups'
slots in the command buffer's records pool, and writes the object function's
`VkDispatchIndirectCommand` into the work buffer (`MeshDispatch::OBJECT_DISPATCH_OFFSET`). A grid
Metal's GPU discards sets `StatusBits::MESH_GRID_DISCARDED`; one past the device's compute limits
sets `MESH_GRID_PAST_LIMITS`; slots the rest of the pool cannot hold set `MESH_RECORDS_OVERFLOW`.
Each draws nothing. A grid whose threads per grid wrap along an axis runs every threadgroup, and
the object function reports the wrapped grid as Metal's: an unfolded object function's
`threadgroups_per_grid` is the threadgroups its wrapped threads fill.

### Tile pipelines

`Device::compile_tile_pipeline` compiles a `TileRenderPipelineDescriptor` (the tile function and the
render pass's color attachment formats) into a `ComputePipeline` that stands in for
`dispatchThreadsPerTile:`. Dispatch it as `Dispatch::Threadgroups` over the render area, one
threadgroup per tile, with `BindState::imageblock_size` and `BindState::render_area` set, and bind
each attachment it reaches as a storage image where the plan writes `WriteSource::TileAttachment`.
Its threads are numbered as Metal's tile dispatch numbers them, for every threadgroup of even sides
up to 32x32 and the single thread; a threadgroup with an odd side, whose quads Metal leaves partly
empty, is refused. A tile function runs on a single-sample pass: a multisample
imageblock's operations (its sample count, the colors a pixel holds and their coverage masks, a
store to the samples a coverage mask selects) are refused as not implemented
(`Feature::Imageblocks`, naming the intrinsic).

## 5. The interface and the plan

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
- The layout numbers set 0 densely, in `ResourceSlot` order. Set 1 holds the heap arrays (the
  resources reached through argument buffers and resource IDs). Push constants carry the values the
  binder writes per command. A stage that writes the command buffer's `ExecutionStatus` (an absent
  argument's read, a `seq_cst` fence, an argument buffer's cubic sampler) reaches it by device
  address, a push hole, where the device addresses buffers (`bufferDeviceAddress` and
  `shaderInt64`), and through a set-0 storage buffer of its own
  (`SyntheticBinding::ExecutionStatus`) where it does not. That binding counts against the
  stage's storage-buffer limits: past one, an absent read or a fence is refused
  (`DescriptorCount`), and a stage that would bind it only to test its argument samplers for a
  cubic filter binds nothing, so the plan refuses a cubic sampler listed beside it
  (`SamplerBicubicFilter`).
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
- `dependencies` lists the bind-time state the pipeline depends on (`RuntimeFact`: threadgroup
  sizes, sampler coordinate modes, buffer ranges, ...) and how this profile honors each
  (`Mechanism`: a specialization constant, a push field, a plan-time check, a variant). The plan
  acts on them; you do not have to.
- The format class of a bound texture is read by the shader as push data, not specialized, so binds
  that differ only in their textures' formats share one `VkPipeline`. On MoltenVK a slot read with a
  `half` result also specializes one bit of it: whether the class converts its texels to `half` by
  rounding to nearest even alone (every class but those with 32-bit float channels, 16-bit
  normalized ones and YCbCr fields on the grid `8192`), so a slot bound to both kinds of format
  takes two pipelines. Other bound values (a slot's
  presence, a sampler's word, whether a uniform buffer's descriptor range starts at its bind offset)
  are specialization constants. The last lets a driver load the constants ahead of the shader:
  inline bytes take that form, and so does a buffer bound at a multiple of
  `max(minUniformBufferOffsetAlignment, 16)` at a slot whose table tells that alignment apart
  (`state.view_base_alignment = BindState::view_base_alignment_of(limits)`, and the slot's bit in
  its table's `at_bind_views`; `interface.uniform_view_buffers(stage)` lists the slots that can
  take it). A buffer bound at any other offset, or at a slot left out, takes the other form, a
  pipeline of its own: leave out a slot bound at offsets that move, so that its draws keep one
  shape and one pipeline (the runtime leaves out each slot of a state once it finds one bound at
  an offset the alignment does not divide).
- A two-plane YCbCr texture whose planes a function may read in place states them in its facts
  (`TextureFacts::ycbcr_in_place`, a `YCbCrInPlace` of its planes; the runtime states one on a
  device whose memory is the host's, `DeviceProfile::reads_ycbcr_planes_in_place`: an integrated
  GPU or a CPU device). A slot whose function only samples it (nearest or linear, no comparison,
  anisotropy or `min_lod_clamp`) or reads its size (`RuntimeFact::YCbCrInPlace`) takes the
  pipeline's in-place variant (`VariantKey::ycbcr_in_place`): the slot binds a uniform texel buffer
  over the planes' bytes (`WriteSource::YCbCrInPlace`, in `YCbCrInPlace::format`), its push field
  (`PushField::YCbCrInPlace`) holds `YCbCrInPlace::words`, and the function converts and filters
  the codes as Metal does, exactly. Any other slot, and any texture without the fact, binds the
  texture's image of converted texels. Where the consumer also holds the planes as a linear image
  of their multi-planar 4:2:0 format over their bytes, which the device's sampler Y′CbCr
  conversion reads with Metal's reconstruction, it states so (`YCbCrInPlace::with_planes_image`,
  its conversion `YCbCrInPlace::sampler`, a `YCbCrPlanesSampler`). A slot whose every sample the
  device's sampler takes (the fact's `device`: no border address mode, no texel offset placed by a
  parent's size; a runtime sampler's word without a border mode) then takes the planes variant
  (`VariantKey::ycbcr_planes_sampled`): the slot is a combined image sampler of the image
  (`ResourceKind::YCbCrPlanes`, `BindingSource::YCbCrPlanes`, written from
  `WriteSource::YCbCrPlanesImage`) whose immutable sampler is `SamplerDesc::ycbcr` of the
  conversion. The consumer chains its `VkSamplerYcbcrConversionInfo` into that sampler and makes
  the image's view with an identically defined conversion, and sizes a pool for the format's
  `combinedImageSamplerDescriptorCount` descriptors at the binding. Each sample is the device's,
  at level 0, its conversion and filter of the codes rather than Metal's.
- `interface.reaches_command_buffers` and `interface.texture_writing_stages` say, provably, which
  pipelines may touch an indirect command buffer and which stages may write a texture.
  `interface.command_writes` (`CommandWrites`) says what its functions may write into the
  records they reach: the indirect-command operations they may call, and the primitive types
  their draws may pass (a draw's constant type, or all five where it is not a constant). A clear
  bit is proven.

### Plans

Per draw or dispatch, describe what is bound, keyed by Metal slot, as a `BindState`. The binder
never sees your resources, only facts about them:

```rust
let mut table = StageTable::default();
table.buffers.insert(0, buffer(4096, 0));
let facts = TextureFacts::new(TextureType::Type2D, PixelFormat::RGBA8Unorm);
let texture = BoundTexture::Texture {
    facts,
    resource_id: None,
};
table.textures.insert(0, texture);
let sampler = BoundSampler::new(SamplerDescriptor::new());
table.samplers.insert(0, sampler);
let mut state = BindState::default();
state.stages.insert(Stage::Kernel, table);

let plan = pipeline.plan(&device, &state.shape())?;
assert!(plan.variant.is_empty(), "this shape needs no other variant");
```

`state.shape()` keeps only what can change a plan (which slots are bound with what kind of
resource, formats, level counts, sampler modes, alignment classes). A plan is a pure function of the
pipeline, the device and the shape: memoize it on `(PipelineKey, BindShape)`. A consumer that keeps
the shapes it planned can find a state's among them without making it: `state.has_shape(&shape)`
answers `state.shape() == shape`, compared in place, and `state.position_of_shape(&shapes)` finds
it among many, making the shape of the resources the state lists at most once.

The plan holds one `DescriptorWrite` per layout binding element (the resource at a slot, a buffer
the binder synthesizes, or a neutral resource where nothing compatible is bound), the vertex
buffers, and the `ResourceUse`s that drive staging, writeback and barriers. List in the state
every live sampler state and visible-function table the command can reach through argument buffers
(`argument_samplers`, function tables), and the resources it uses (`used_resources`).

A plan whose `variant` is not empty asks for another variant of the pipeline (a storage texture's
format, a buffer range past the descriptor limit, a vertex buffer that fixed-function input cannot
fetch, ...). Compile it and plan it for the same shape: `device.compile_render_variant(&descriptor,
&plan.variant, &budget)` for a render pipeline, or compile the kernel with
`plan.variant.lowering_inputs(Stage::Kernel)` and link it as `StageSource::Compiled`. A dispatch can
ask too (`plan.dispatch_variant(&dispatch)`, or `BindError::NeedsVariant` from instantiation).

Textures follow Metal's texel rules: sparse textures report residency where the device can,
`Depth32Float` texels are the device's depth image's (a value it clamps or a denormal it flushes is
announced, `restricted-depth-upload`, `depth-denormals`), `ulong` textures and their atomics use
64-bit images where the device has them, and `half` and sRGB reads give Metal's values or record
their bound (an sRGB texel is the device's own decode wherever the device samples its format,
`srgb-texel-decode`). The
rustdoc of `BindState`, `TextureFacts` and `BindingPlan` states each rule.

## 6. At bind time

Instantiate the plan per dispatch or draw:

```rust
let dispatch = Dispatch::Threads {
    threads: [100, 1, 1],
    threads_per_group: [32, 1, 1],
};
let commands = plan.instantiate_dispatch(&state, &dispatch)?;
for region in &commands.regions {
// ...
println!("  descriptor ranges: {:?}", commands.descriptor_ranges);
println!("  push holes: {:?}", commands.push_holes);
```

```rust
let plan = pipeline.plan(device, &state.shape())?;
let draw = DrawCall::Primitives {
    vertex_start: 0,
    vertex_count: 6,
    instance_count: 1,
    base_instance: 0,
};
let commands = plan.instantiate_draw(&state, &draw)?;
```

`instantiate_draw_into` writes the same commands over a `DrawCommands` you keep, in the storage
its vectors already hold: keep one from draw to draw (`DrawCommands::default()` to start) and the
draws allocate none of it again once it holds as much.

A dispatch is one or more regions (a grid that is not whole threadgroups adds edge regions), each
with its group count, specialization and push constants: create one `VkPipeline` per distinct
specialization. A kernel that steps a stage-in attribute by thread position, or reads
`[[grid_origin]]` or `[[grid_size]]`, needs the encoder's stage-in region:

```rust
match plan.instantiate_dispatch(&state, &dispatch) {
    Err(BindError::MissingStageInRegion) => {}
// ...
state.stage_in_region = Some(StageInRegion::new([0, 0, 0], [64, 1, 1]));
let commands = plan.instantiate_dispatch(&state, &dispatch)?;
```

A tessellation pipeline draws patches, with the factors' buffer in the state:

```rust
let factors = TessellationFactorBuffer::new(4096, 0, 0);
state.tessellation_factors = TessellationFactors::new(Some(factors), 1.0);
let plan = pipeline.plan(device, &state.shape())?;
let draw = DrawCall::Patches {
    control_points: 4,
    patch_start: 0,
    patch_count: 16,
    patch_index: None,
    instance_count: 1,
    base_instance: 0,
    control_point_index: None,
};
let commands = plan.instantiate_draw(&state, &draw)?;
```

What you record from `DispatchCommands` or `DrawCommands`:

- **Descriptors.** Write each of the plan's `descriptors`, binding each buffer descriptor over
  exactly its `descriptor_ranges` entry. Upload `descriptor_bytes` (argument and handle tables) and
  the resident root tables where present. A handle table maps each resident resource's
  `gpuResourceID` to its heap element; a consumer that keeps one heap for every command buffer can
  bind one table of every texture ID instead (`bind::handle_entry`, `bind::heap_element`,
  `bind::handle_entry_flags`), as the runtime does.
- **Push constants.** Write `push_constants`, and fill each of the `push_holes` with the device
  address it names: the uploaded root table, the command buffer's `ExecutionStatus` buffer, an
  indirect command buffer's records. A snapshot variant's `PushHole::SnapshotPhase` is a word, not
  an address (below).
- **The dispatch or draw.** Record `regions` or `draw` as given: the binder has already applied
  Metal's offsets, bases and clamps.
- **Status.** Per command buffer, keep one `ExecutionStatus` buffer for what translated shaders
  report (a resident-set miss, for one), and read it when the command buffer completes. A plan
  binds it as `WriteSource::ExecutionStatus` on a device that does not address buffers: bind the
  same buffer there, its `ExecutionStatus::BYTES` bytes.

### Snapshot reads

A draw that reads, at its fragments' own pixels, a texture its own subpass renders can read it
through a snapshot instead of a copy taken before the render pass instance. The interface lists
the fragment slots that allow it (`StageInterface::snapshot_textures`, `StageModule::
snapshot_textures`): own-pixel slots whose texels reach no derivative, sample, quad or subgroup
operation, branch, integer divisor, dynamic vector index or store outside the invocation, directly
or through a call, in a function that writes no memory.
Name the slots in `BindState::snapshot_textures`; the plan keeps those its pipeline can read so
and states them in `variant.snapshot_textures`, each with the form its snapshot holds texels in
(`SnapshotForm::of` the slot's bound view and declared type: the texel's codes for an 8- or 16-bit
normalized, half-float or integer format of the slot's class, which a read decodes as Metal reads
the format, and the fetched lanes' bits for any other), compiled like any other variant. It keeps
none on a
device without `fragmentShaderPixelInterlock`, `fragmentStoresAndAtomics`,
`attachmentFeedbackLoopLayout` or `extendedDynamicState3ColorWriteMask`, for a pipeline of more
than one sample, a depth or stencil attachment, attachments in the storage form, input
attachments, an imageblock, tile memory, a shader-side stencil or tests, tessellation or mesh
stages, or a function that writes memory, and for an amplified or layered draw. Serve every other
read as before.

A pipeline of that variant states `GraphicsState::snapshot_reads`: create it with
`VK_PIPELINE_CREATE_COLOR_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT`; its `dynamic_states` hold
`VK_DYNAMIC_STATE_COLOR_WRITE_MASK_EXT`, and its requirements `attachmentFeedbackLoopLayout` and
`extendedDynamicState3ColorWriteMask`. Bind the slot's own texture in the attachment's feedback-loop
layout, and at `WriteSource::SnapshotTexels` a storage image of the slot's form's
`SnapshotForm::format` (`R8G8B8A8_UINT`, `R16G16B16A16_UINT` or `R32G32B32A32_UINT`) covering the
render area, in `GENERAL` layout. Record each draw twice in its subpass: first with every color
write mask 0 and `PushHole::SnapshotPhase` set to 1, each fragment copying its texel into the
snapshot and running nothing else; then with the pipeline's own masks and the phase 0 (the plan's
bytes), each fragment reading the snapshot. A by-region pipeline barrier before the first draw
makes the attachment's earlier writes visible to the copy, and one between the two makes the copy
visible to the reads and orders it before the second draw's color writes. Each step rests on a
sentence of the Vulkan specification: the feedback loop's per-pixel barrier, the color write mask
leaving memory unmodified, the invariance of fragment generation under write masks (Rule 2,
Corollary 1), and the ordered pixel interlock. One draw alone cannot be exact: nothing orders a
fragment shader's read of a texel before the same draw's color write of it.

### Indirect command buffers

An indirect command buffer's commands are records in ordinary buffer storage
(`bind::indirect_commands`): one format whatever the descriptor, written by the CPU or by a kernel.
The binder replays records into dispatches and draws: on the host, or on the device through a
preprocessing kernel (`replay_preprocess`), and through `VK_EXT_device_generated_commands` where
the device has it (`replay_preprocess_generated`). Draws of primitives under the encoder's
pipeline replay on the device through `draw_replay_preprocess`: one indirect draw per command
and primitive type the execution names, the pipeline compiled as its replay variant
(`VariantKey::replay_bindings`), whose vertex and fragment functions read each command's buffers
from the slot tables the preprocess writes. Draws of one list type or points whose fragment function
reads no buffer and no `[[primitive_id]]` may instead merge into one draw (`DrawReplay::merged`):
two merge passes (`draw_merge_module`) write each command's first vertex in the merged draw, and
the vertex function, compiled as its merged variant (`VariantKey::merged_draws`), finds its
command by that prefix and runs the command's vertex, instance and buffers, in the commands'
order. A kernel that encodes commands writes the records that start at
`bind::indirect_commands::records_address` of the buffer's ID, outside every Metal address, and
finds them only at their start: records may run past later IDs' starts, and a resident set
(`bind::ResidentSet`) keeps each records range whole.

## 7. Caching

The crate keeps no cache; keys and encodings let you keep one across processes. Keys are SHA-256
hashes of the inputs, `cache::SEMANTICS_VERSION` and the profile fingerprint, known before
compiling (`device.compute_pipeline_key(&descriptor)`, `render_pipeline_key(&descriptor,
&budget)`, `tile_pipeline_key`, `function_compile_key`):

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
  `compile_function_with_report` also returns a `CompileReport` of what the compile cost.
- A render pipeline's stages compile under inputs that a decode of its functions learns.
  `render_pipeline_key` returns `RenderPipelineKeys`: each stage's function key, and the inputs to
  compile a missing stage under. So does a compute pipeline's kernel:
  `kernel_inputs_read(&kernel, &inputs, &budget, &entries)` narrows the stage-input descriptor to
  the attributes its entry reads, and the kernel keys and compiles under those, so states that
  differ only in an attribute it never reads share one compiled function. What a creation reads of
  the descriptor past the fetch -- its layout agreement, its reflection, and `compute_pipeline_key`
  -- reads it whole.
- A key names an artifact by its inputs and versions; a content digest names it by what it says.
  `pipeline.content_digest()` hashes everything but the key and profile, and `plan.content_digest()`
  (with `serde`) everything a plan states. A cache of driver objects keyed on the digest survives a
  release that bumped `SEMANTICS_VERSION` without changing the output. A digest identifies
  content; it authenticates nothing.
- A caller that names a stage's module by its words need not hash them twice. A compiled
  function keeps its last relocation beside it, so a sweep of pipeline states over one function
  pair takes the same `Arc<[u32]>` on every creation;
  `compiled.relocated_words_digest(&stage.spirv, derive)` gives the digest `derive` produced over
  those very words, or `None` where they are not the function's last relocation -- a miss costs no
  call of `derive` at all, so it is safe to ask first in a creation's hot path. The digest a caller
  gets is always one its own `derive` made, so nothing it names by the digest moves.
- A memo that holds pipelines in memory and bounds itself by bytes can weigh one it never encoded.
  `pipeline.footprint()` is the bytes a `RenderPipeline` keeps live -- every word of every stage,
  each stage's constant blob and notes, and its own storage -- so a pipeline kept without
  `to_bytes()` is weighed by what it costs rather than by the zero length of an encoding that was
  not made.
- Versions move independently: the crate's (the API), `SEMANTICS_VERSION` (the output),
  `ARTIFACT_FORMAT_VERSION` (the encoding) and `RUNTIME_VERSION` (what the runtime does with an
  unchanged artifact). `SOURCE_DIGEST` names the sources a build was compiled from, for caches kept
  across builds.

## 8. Refusals, requested inputs, and precision

Every failure is typed; match variants, never messages. A refusal says what would change the
answer:

```rust
let floor = committed_device("vk12-floor")?;
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
- `DriverDefect`: a defect the device's driver is measured to have. `RuntimeDriverDefect`: one in a
  command the runtime records.
- `MetalGpuFamily`: a Metal GPU family the device does not claim, where Metal itself refuses the
  input without it.

`location` and `symbol` are diagnostics only. A render pipeline's refusal of one attachment's
state names it (`location.attachment()`). [REFUSALS.md](REFUSALS.md) lists every refusal in the
code, with its device condition.

A compile that needs an input asks for it (`Error::NeedsInput`) instead of guessing: a
function-constant value, a storage texture's format, a linked function, a kernel's stage-input
descriptor. Supply it and compile again:

```rust
match device.compile_compute_pipeline(&descriptor, &budget) {
    Err(Error::NeedsInput(RuntimeInput::FunctionConstant { index })) => {
        let words = [1u32, 2, 3, 4].map(u32::to_le_bytes).concat();
        let value = FunctionConstantValue::new(DataType::UInt4, words);
        kernel.constants = FunctionConstantValues::defaults().with_index(index, value);
    }
```

A storage texture's format (`RuntimeInput::StorageTextureFormat { slot }`) comes from what is
bound: `metal::format::storage_image_choice(pixel_format, device.profile())`, placed in a
`VariantKey`'s `storage_images` and compiled as a variant. The plan asks for it itself where the
bound format needs a declared format.

**What to cache.** `error.retry()` says whether the same request can succeed later, and
`error.is_permanent_for()` which key may memoize the failure (the input, the input on this device,
or never). A `BindError` from `plan` depends only on the pipeline, the device and the shape, so it
memoizes beside the plan; one from instantiation does not.

**Precision.** Where an output gives a near-equivalent of Metal's answer, its stage lists it in
`precision_notes` (a `PrecisionNote`, each documenting its bound), and `reconvergence` says
whether subgroup reconvergence after divergence matches Metal's
(`Reconvergence::ImplementationDefined` where the profile lacks maximal reconvergence). A note
that holds only under some bound values is in `specialized_notes`;
`stage.precision_notes_bound(&plan.specialization, &|key| plan.specialized_value(key))` gives the
notes a plan reaches. Section 9 lists every note.

**The runtime's errors** are its own type, `metal2vulkan_runtime::Error`, which tells apart what
plain Metal itself refuses (`Invalid`), what plain Metal answers nil for (`Nil`, `MetalError`), what
the device lacks (`Unsupported`), what the runtime has not built (`NotImplemented`) and the
translator's errors (`Compile`). Its [README](../crates/metal2vulkan-runtime/README.md#errors)
lists them.

## 9. Degrade notices

Where the runtime gives Metal's answer only at a cost Metal does not pay, or gives a recorded
near-equivalent, it still answers, and says so. Each kind of degradation is a
`metal2vulkan_runtime::Degradation`:

- **performance**: Metal's answer, at an extra cost (a copy, a staging or scratch resource, an
  extra pass, a slower variant);
- **precision**: a recorded near-equivalent of Metal's answer, including every `PrecisionNote` a
  pipeline stage records;
- **validity**: Metal's answer exactly, through a use of the device its driver does not advertise
  but was measured to run as Metal does, so Vulkan's validation layer reports the use.

How a notice reaches you:

- Each kind is announced **once per device**, the first time the device takes it. A kind listed in
  `DeviceOptions::announce_each` (`RawDevice::announce_each` for an adopted device) is announced at
  every occurrence instead. The notes a pipeline's making records are announced at the first
  dispatch or draw that binds it.
- Through the device's message sink (`DeviceOptions::messages`) as a `DebugMessage` of `Warning`
  severity whose `id_name` is the kind's code and whose `degradation` holds a `DegradeNotice`: the
  `Degradation`, the object degraded, and the reason. Without a sink, one line on stderr:
  `metal2vulkan: <object>: <reason>; <cost> [<code>]`.
- A notice is never fatal: the work runs. The one exception is opt-in.
  `DeviceOptions::lowering_refuses` (`RawDevice::lowering_refuses`; empty by default) lists kinds
  that, when a render encoder lowers a call it logged (`RenderLoweringPolicy::Worker`, section 1),
  are not announced. Instead they refuse that call: `CommandBufferError::CallRefused` names it,
  and `CommandBuffer::refusal` returns `Error::Degraded` with the notice. A call lowered inside
  the call announces them as any kind.
- A notice announced while a render encoder lowers a call it logged names that call
  (`DegradeNotice::call`: its Metal selector and its index among the encoder's calls); `None`
  elsewhere.
- Decode a notice by its `Degradation` or its code, never its text. The codes are a contract: none
  is renamed.
- `Degradation::exactness()` says whether the program still gets Metal's answer: `Exact` (most
  performance and validity kinds), `ExactWithinPool` (Metal's answer while a command buffer's pool
  holds it, else the command buffer fails), `OneOfMetals` (Metal's own answer is not one value, and
  the runtime gives one of them), `FailsInstead(error)` (the command buffer fails rather than
  differ), or `NearEquivalent` (the precision notes and other near-equivalents).
- `Device::degraded_objects(kind)` counts the textures a device met as a kind that is announced
  once per object.

With `METAL2VULKAN_LOG_PIPELINES` set, a device also logs one line per render `VkPipeline` it
makes (id name `metal2vulkan-pipeline-creation`, `Info` severity), with where its time went: the
wait for its turn, the digests that name its parts, each library (made by this making, from its
driver cache record or not, or another caller's) and the link; and, once per reason, why a draw
that reads its own pass's color attachment after a barrier took the split rather than a snapshot
(id name `metal2vulkan-snapshot-fallback`, `Info` severity; `WorkCounts::reading_draws_snapshotted`
counts those that took a snapshot), and now and then a count of the draws the snapshot's area rule
weighed, by their scissors' share of the target, texel bytes and side of the rule (id name
`metal2vulkan-snapshot-readers`, `Info` severity). With `METAL2VULKAN_LOG_MEMORY` set, a device
logs one line per `VkDeviceMemory` it allocates, imports or frees (id name `metal2vulkan-memory`,
`Info` severity): what it holds (a block of buffers or images, a host-memory import with its host
address, a dma-buf import), its handle, memory type, bytes and device address range, where the
block holds buffers that take device addresses (its own buffer of addresses gives the range; the
log makes no Vulkan object to ask one). On NVIDIA that range is the GPU virtual address a fault
report names. It
also logs each placement of 4 MiB or more in a pooled block as it is placed and returned: an image
or a buffer, its memory class where placed, its bytes, its offset and its block's handle and type.
Whatever the variables, a placement heap's `newTextureWithDescriptor:offset:` that makes no texture
(`Nil` or `OutOfMemory`) at an offset inside the heap says so (id name
`metal2vulkan-heap-texture-refused`, `Info` severity): the descriptor, the offset, the heap's size,
options and memory, the size and alignment the size query answers the descriptor, what the making
planned (the size and alignment it places the texture by, its image's route, tiling and memory
requirements) and the refusal. A device's self-check that the driver does not make (the one
pipeline it creates to ask whether a set-0 buffer binding of several variables may be a dynamic
descriptor, `FAST-PATHS.md`) hands the sink one line saying so, with the driver's result and the
messages it reported inside that creation (id name `metal2vulkan-self-check-refused`, `Info`
severity), in place of those messages: they answer the runtime's own question and name none of the
program's work (MoltenVK 1.4.2 reports two errors there). That text is for people, not a contract.

Each entry below gives the code, its category, the `Degradation` kind (or the `PrecisionNote`) and
what triggers it; a runtime kind's cost follows. The kind's rustdoc states the trigger, cost and
how to avoid it in full. [FAST-PATHS.md](FAST-PATHS.md) lists which notes the device's native
forms and the default policy's fast forms record, and each `PrecisionNote`'s rustdoc states its
bound.

- `metal2vulkan-degrade-shadowed-texture` (performance). `ShadowedTexture`: A 2D texture over a
  buffer is a shadow image: the buffer is over host pages (a no-copy buffer's, imported by their
  pointer or as dma-bufs, or the fresh pages a host-visible buffer of the runtime's own longer than
  32 MiB lies in), where an image bound to them would start `UNDEFINED` and Vulkan lets its layout
  transitions (its first move among them) read and write any byte bound to it, where Metal keeps
  every CPU write, and the device takes no image in place there: the policy keeps the exact form
  (`HostPageTexturePolicy::Shadow`), the texture's usage renders into it or writes it
  (`RenderTarget`, `ShaderWrite`, `ShaderAtomic`), the device's check that the move keeps every
  byte over that kind of pages has not answered yet, found a byte changed or made no image, or no
  image in place serves the texels (its rows share bytes, the device lays no linear image's rows at
  the buffer's pitch, the image's placement or memory misses the pages, the bytes were detached, or
  no dma-buf of the texture's own pages serves); the notice names which, and
  `WorkCounts::host_texture_shadows_*` counts each; or its rows (or an array's slices) lie with the buffer's bytes between them,
  which an image over them would hold in its subresource, where Vulkan lets the image's layout
  transitions read and write any byte of its subresources, and a render pass's load and store
  operations read and write back any byte of an attachment's image subresource, and Metal leaves
  them as the program writes them; or no linear image of the device lays its texels out as Metal
  does in the buffer's memory, holding no byte past them; or the buffer is over the caller's
  `VkBuffer`, whose memory backs no image of the runtime's. Also a heap's texture over host pages
  imported as dma-bufs where the device makes no image over a dma-buf of the texture's own pages
  (the notice names why; `WorkCounts::dma_buf_image_fallbacks` counts it). A heap's texture is a
  shadow of the heap's bytes in its range in three more shapes, each announced with its own reason.
  A single-sample depth or stencil texture whose texels fit its image's bytes, on a device whose
  driver lays such images apart from the memory they are bound to
  (`DriverFact::DepthStencilImagesLieApart`): one no such shadow serves, a multisampled one among
  them, is `heap-depth-stencil-apart` instead. A texture the device binds no image
  of to the heap's memory where the heap places it: its optimal image takes no memory type of the
  heap's and no linear image of the device serves (a `Shared` heap's depth texture on MoltenVK), or
  the heap lies over host pages that no linear image over them serves, or its images ask more than
  Metal's answer for the texture (every 4:2:2 texture among them) and a shadow of its rows serves.
  And a heap's `RGB10A8_2P_XR10` texture, or its sRGB twin, of a 1D, 2D, 1D array, 2D array or 3D
  type, of any level count, of one sample, whose moves fit the device: its image holds each texel's
  colour word and alpha byte side by side where Metal keeps two planes in the heap's bytes, so the
  image is a shadow of those planes (a mipmapped one's levels past the first, and a 1D array's
  texels, lie in the image alone: `heap-two-plane-layout`). Where Metal's layout of the texture in
  the heap's bytes is measured and the moves fit the device, the first two shadow Metal's tiles
  there (the texture's elements, a multisample texture's samples) rather than its rows, and so do
  two textures that would otherwise need no shadow: one whose image the driver keeps GPU-optimized
  where Metal's texture is not (`DriverFact::DepthStencilImagesStayGpuOptimized`), and a
  multisample texture whose image lies in memory of its own. Cost: its texels are copied into an image before each use and back after each
  write (over a buffer's bytes but a tracked private buffer's, kept before a write and merged back
  after it). A render pass that loads it as an attachment, takes it in no other way and draws under
  scissors copies in only the texels inside them (`WorkCounts::shadow_texels_copied_in`). Over a
  buffer's bytes that another writer may
  write while a command buffer runs, the image is kept in a scratch buffer before the first command
  that writes through it (a one-sample colour texture in Metal's tiles, right after its copy in, by
  a copy of the bytes that copy took: `WorkCounts::shadow_keeps_from_staging`), and each copy back
  is a dispatch that merges only the words the commands
  changed, so that writer's writes to the texels they left stay, as on Metal: the CPU's, over a
  `Shared` or `Managed` buffer's bytes, host pages or the caller's `VkBuffer`, and another queue's
  commands', over an untracked buffer's. After a render pass that loads and stores it as a color
  attachment inside its draws' scissors, or a command that replaces every texel over host pages
  that log their writes, the dispatch stores each 256-byte block of the rows it merges that holds
  a changed word whole (`WorkCounts::shadow_merge_dispatches_in_blocks`): Metal's own store of
  such a pass writes back every texel of each tile its draws reach, so a write to a texel the
  pass left inside its render area while it runs is not kept on Metal either. A tracked `Private` buffer's bytes have no such writer (the
  runtime orders another queue's commands that touch them before or after the command buffer, as
  Metal does), and are copied back (`shadow_texels_copied_back`), as a command that replaces every
  texel is. The scratch keeps the rectangle the commands write in, where they bound it and the
  copies are cut so (`WorkCounts::shadow_texels_kept`). Where the device is measured to run it
  faster so (`DriverFact::ShadowCopyBackDispatchFaster`), a copy back of a one-slice texture of 4-,
  8- or 16-byte texels is a compute dispatch that stores the same bytes
  (`WorkCounts::shadow_copies_back_dispatched`), and so is a merge's copy of such a texture's image
  into its scratch buffer before the commands (`WorkCounts::shadow_merge_copies_dispatched`), which
  on a device measured faster so (`DriverFact::ShadowCopyInDispatchFaster`) also takes the copy in
  of the same texels before it, storing them in the image through that view
  (`WorkCounts::shadow_copies_in_dispatched`); after them, a cut merge reads the texels the commands
  left from the image itself, through an unsigned-integer view of it, with no copy of them
  (`WorkCounts::shadow_merges_fetched`). On a device with the first fact alone, such a merge keeps
  the texels in an image of the texture's size in place of its scratch buffer, copied from the
  texture's image at the same coordinates, and reads them from there
  (`WorkCounts::shadow_merges_kept_in_images`). Its command queue keeps that image for its later
  merges of the same size and format, whatever their texture, each taking it once the encoder that
  took it last ended (`WorkCounts::shadow_merge_kept_images_made`,
  `shadow_merge_kept_images_followed`, `shadow_merge_kept_images_live`). Where the image has no
  unsigned-integer view (any device without that fact), a cut merge of a one-slice `BGRA8Unorm` or
  `RGBA8Unorm` texture reads the texels the commands left from the image through a view of its own
  format, rebuilding each byte from its normalized value (Vulkan defines that value as the byte over
  255), with no copy of them (`WorkCounts::shadow_merges_fetched`); every other format, the sRGB and
  16-bit float ones among them, copies them.
- `metal2vulkan-degrade-staged-texture-buffer` (performance). `StagedTextureBuffer`: A texture
  buffer is viewed through a staging buffer: its texels start where the device views none
  (`minTexelBufferOffsetAlignment`). Cost: its texels are copied into a staging buffer before each
  use and back after each write.
- `metal2vulkan-degrade-shadow-merge` (performance). `ShadowMerge`: One command writes a buffer's
  bytes through a shadowed texture or a staged texture buffer and also another way. Cost: the
  shadow's texels are kept in a scratch buffer of one to three times the texture's bytes (or an
  image of the texture's size) before the command, and after it a dispatch merges the bytes it changed through the texture into the
  buffer's, in place where the texture's bytes start and end in whole words of the buffer: one to
  three copies and a dispatch in place of one copy back. The merge's dispatches are counted as the
  runtime plans them, in windows of at most the device's storage-buffer range and workgroup count
  (`WorkCounts::shadow_merge_dispatches_planned`), together with the words they examine
  (`shadow_merge_words_examined`): a dispatch reads each word of the region before the commands and
  after them and stores only where the two differ, so the words it stores are at most the words it
  examined and only the GPU knows them.
- `metal2vulkan-degrade-copied-draw-bytes` (performance). `CopiedDrawBytes`: A draw's index buffer
  or indirect arguments start where Vulkan reads neither. Cost: they are copied into a buffer of the
  command buffer before the pass, and the draw reads the copy.
- `metal2vulkan-degrade-isolated-index-bytes` (performance). `IsolatedIndexBytes`: An indexed draw
  reads index positions past where Metal's fetch stops reading its index buffer (its length rounded
  up to 4 bytes), for which Vulkan states no index: a direct draw whose positions run past there, on
  every device; or an indirect draw that stays one draw (a driver measured to read zero past the
  bound range, `DriverFact::RobustIndexFetchReadsZero`) on a device that enables
  `robustBufferAccess2` but not `maintenance5`, whose bound range runs to the end of the index
  buffer's `VkBuffer`, on past the bytes into its padding or into a heap's bytes after a placed
  buffer. Cost: its whole indices inside the buffer are copied into a buffer of the command buffer
  before the pass, between two barriers, and the draw reads the copy: a direct draw's at its start,
  every position past them zero.
- `metal2vulkan-degrade-unordered-draw-bytes` (performance). `UnorderedDrawBytes`: An indexed
  indirect draw reads arguments or indices a draw of its encoder wrote, with no memory barrier in
  the encoder before it, a race Metal does not order; or a draw's vertex stage reads a buffer of its
  vertex table a fragment stage of its render pass wrote. Cost: they are copied into a buffer of the
  command buffer before the pass, between two barriers, and the draw reads the bytes from before the
  write, which Metal's draw read in every run measured.
- `metal2vulkan-degrade-unorm-depth-bias` (precision). `UnormDepthBias`: A draw biases a
  `Depth16Unorm` depth by a constant, or by a slope alone, which brings Metal's constant unit, on a
  device whose driver is not measured to store Metal's codes so
  (`DriverDeviation::MetalDepth16BiasedStores`): the device biases by Metal's constant in its own
  unit and converts the depth. Cost: a biased depth can store another code than Metal's: the
  device biases by the constant c in its own unit, up to twice 2^-16, where Metal biases by c + 1
  units of 1/65535 (c - 1 below 0, one unit for a slope alone), and rounds the biased sum its own
  way.
- `metal2vulkan-degrade-unorm-depth-rounding` (precision). `UnormDepthRounding`: A draw tests or
  writes a `Depth16Unorm` depth on a device whose driver is not measured to convert a depth to
  Metal's code (`DriverDeviation::MetalDepth16Stores`), or a pass clears one to a depth that a
  device rounding to nearest takes above Metal's truncation: the device converts the depth. Cost:
  a depth near a code's midpoint can store the code next to Metal's: the device converts the depth
  to the 16-bit code its own way, where Metal's draw rounds to nearest, a tie up, and its clear
  truncates.
- `metal2vulkan-degrade-point-guard-band` (precision). `PointGuardBand`: A draw of points keeps
  Vulkan's clipping of points, where Metal's guard band cannot be emulated: in a rate-mapped pass,
  on a device without `shaderClipDistance` or with too few clip distances beside the vertex
  function's own, or through a viewport whose widened Vulkan viewport would leave the device's
  limits or not be a `float` exactly. Cost: a point whose center lies past its viewport, within
  Metal's guard band of 1.5 times the viewport's half-size, is culled where Metal draws it, and a
  wide point can lose pixels at a fractional viewport edge by the device's own clipping.
- `metal2vulkan-degrade-nan-depth` (precision). `NanDepth`: A draw whose fragment function writes
  its depth, in depth clamp mode through a range above 0, keeps the device's clamp of that depth,
  where the function cannot clamp it itself under a Vulkan clamp that holds 0: on a device without
  `VK_EXT_depth_clamp_control` whose vertex stage cannot compensate the depth exactly, a mesh
  pipeline there, through viewports of different clamps or depth ranges, under a depth bias that
  saturates or offsets the range, or past the device's clip distances. Cost: a NaN depth the
  fragment function writes: a negative one in depth clamp mode stores the depth clamp range's low
  end, where Metal stores 0; a positive one runs no stencil operation, where Metal runs the
  depth-failure one.
- `metal2vulkan-degrade-biased-depth-past-unit-range` (precision). `BiasedDepthPastUnitRange`: A
  draw of triangles in depth clip mode under a depth bias of a constant or slope factor other than
  0, on a device that enabled `VK_EXT_depth_range_unrestricted` but lacks `depthClamp`,
  `VK_EXT_depth_clip_enable` or `VK_EXT_depth_clamp_control`: such a device stores a depth the bias
  takes past `[0, 1]` unclamped, where Metal clamps it. Cost: a depth a bias takes past [0, 1] in
  depth clip mode is stored unclamped, where Metal stores 0 or 1.
- `metal2vulkan-degrade-clipped-depth-pieces` (precision). `ClippedDepthPieces`: A draw of lines or
  triangles whose vertex stage clamps each vertex's depth as Metal does, through a viewport depth
  range with a negative depth in depth clip mode or past `[0, 1]` in depth clamp mode: Metal clips a
  primitive at its near and far planes and at its guard band first, and clamps the depths of the
  pieces' vertices. Cost: a primitive that crosses Metal's near or far plane, or its guard band of
  1.5 times the viewport's half-size, stores the interpolation of its own vertices' clamped depths,
  where Metal clamps the depths of the pieces it clips the primitive into.
- `metal2vulkan-degrade-sample-depth-clamp` (precision). `SampleDepthClamp`: A draw of lines or
  triangles through such a range whose vertex stage cannot clamp each vertex's depth: its fragment
  function reads its position's depth, a mesh pipeline, on a device without `shaderClipDistance` or
  past its clip distances, or through viewports whose depths need different compensations on a
  device that takes the ranges natively; and a draw of triangles in depth clamp mode under a depth
  bias that moves the depth through a range inside `[0, 1]`, where Metal adds the bias to each
  vertex's depth clamped to `[0, 1]`, which a primitive past its near or far plane shows. Cost: a
  sloped primitive's depth is clamped at each sample, where Metal clamps each vertex's depth before
  the interpolation, at 0 in depth clip mode and to [0, 1] in depth clamp mode.
- `metal2vulkan-degrade-depth-bias-past-one` (precision). `DepthBiasPastOne`: A draw of triangles in
  depth clamp mode through a viewport depth range whose larger depth lies past 1, under a depth bias
  of factors, whose Vulkan clamp holds at most 1. Cost: a depth the bias moves past 1 stores 1,
  where Metal clamps it to the range's larger depth.
- `metal2vulkan-degrade-in-place-vertex-reads` (precision). `InPlaceVertexReads`: A draw's vertex
  stage reads in place bytes a fragment stage of its render pass wrote, through an argument buffer
  or a resident resource, of a buffer another stage of the pass or the vertex function writes, or in
  an execution of render commands. Cost: the device may order the fragments' write before the read,
  where Metal's vertex stages read the bytes from before the pass's fragments in every run measured.
- `metal2vulkan-degrade-fragments-before-geometry` (precision). `FragmentsBeforeGeometry`: A draw's
  object or mesh stage reads in place bytes a fragment stage of its render pass wrote (through an
  argument buffer or a resident resource, or of a buffer another stage of the pass or the stage's
  function writes; an emulated stage's where its work records after the pass's earlier render pass
  instances), or a draw's vertex, object or mesh stage writes bytes a fragment stage of its render
  pass read or wrote: Metal's GPUs run every draw's vertex, object and mesh functions before any
  draw's fragments of the pass. Cost: the device may run the pass's earlier fragments before the
  draw's stage, and does across a split of the pass (a command's barrier, a memory barrier): the
  read sees their write, their read misses the write, or the write outlasts theirs, where Metal's
  fragments run after every draw's vertex, object and mesh functions in every run measured.
- `metal2vulkan-degrade-indirect-index-bounds-pass` (performance). `IndirectIndexBoundsPass`: An
  indexed indirect draw whose index fetch nothing bounds (the device lacks `robustBufferAccess2`,
  its driver is not measured to read zero past the bound range,
  `DriverFact::RobustIndexFetchReadsZero`, or the index offset lies at or past the buffer's length)
  is bounded on the GPU before the pass. Cost: two dispatches before the pass write two draws'
  arguments and the indices past the buffer into the command buffer's index pool (a sixty-fourth of
  the device's largest device-local heap), between barriers, two for a run of such draws over one
  arguments and one index buffer, and the draw is two indirect draws; a draw the pool cannot hold
  fails the command buffer.
- `metal2vulkan-degrade-emulated-mesh-draws` (performance). `EmulatedMeshDraws`: A mesh render
  pipeline's draws on a device without `VK_EXT_mesh_shader`. Cost: the object and mesh functions run
  as compute dispatches before the pass, between barriers, writing records a synthesized vertex
  stage draws; a draw whose mesh threadgroups the GPU decides is scanned first and reserves its
  records in the command buffer's records pool (a sixty-fourth of the device's largest device-local
  heap), and a draw the pool cannot hold fails the command buffer.
- `metal2vulkan-degrade-emulated-tessellation-draws` (performance). `EmulatedTessellationDraws`: A
  tessellation pipeline's patch draws on a device without `tessellationShader`. Cost: each patch
  draw runs a scan dispatch before the pass, between barriers: one workgroup that processes the
  draw's patches, instance by instance, into records in order, so its GPU time grows with the draw's
  patches times its instances, apart from its draw; the draw then draws the tessellator's triangles
  with the post-tessellation vertex function as a plain vertex function, up to 2^30 triangles per
  indirect draw; an indirect draw reserves its records in the command buffer's records pool (a
  sixty-fourth of the device's largest device-local heap), and a draw the pool cannot hold fails the
  command buffer.
- `metal2vulkan-degrade-synthesized-geometry-stage` (performance). `SynthesizedGeometryStage`: A
  render pipeline state whose draws run a synthesized geometry stage
  (`RuntimeFact::SynthesizedGeometry`): its fragment function reads `[[barycentric_coord]]` on a
  device without `fragmentShaderBarycentric`, or its vertex function selects its primitive's render
  target slice or viewport on a device that cannot store `Layer` or `ViewportIndex` from a vertex
  function. Announced at the state's making, naming which. Cost: each draw runs a geometry stage of
  its topology class between the vertex and fragment stages, which emits each primitive's vertices
  with the barycentric weights or the first vertex's slice and viewport.
- `metal2vulkan-degrade-landed-visibility-results` (performance). `LandedVisibilityResults`:
  Visibility results start where Vulkan writes no query result. Cost: they land in an upload region
  and are copied on into the buffer after the pass.
- `metal2vulkan-degrade-overlapping-buffer-copy` (performance). `OverlappingBufferCopy`: A blit
  copies between overlapping ranges of one buffer, which Vulkan's copy forbids. Cost: the range goes
  through a scratch buffer of its size, which its creation clears by a fill it waits for: two copies
  instead of one.
- `metal2vulkan-degrade-split-buffer-fill` (performance). `SplitBufferFill`: A blit fills bytes that
  are not whole 4-byte words at multiples of 4. Cost: the words are filled and the up to 3 bytes at
  each end copied from a 1-KiB pattern buffer the encoder makes once.
- `metal2vulkan-degrade-staged-texture-copy` (performance). `StagedTextureCopy`: A blit copies
  between a buffer and a texture in a layout Vulkan cannot state. Cost: the rows go through a
  tightly packed scratch buffer: two copies instead of one, and for a single-sample texture the
  scratch's creation clears it by a fill it waits for.
- `metal2vulkan-degrade-overlapping-texel-copy` (performance). `OverlappingTexelCopy`: A blit copies
  texel rows of a texture over a buffer to or from bytes of the same buffer that they share, which
  Vulkan's copy forbids. Cost: the rows go through a tightly packed scratch buffer, which its
  creation clears by a fill it waits for: two copies instead of one.
- `metal2vulkan-degrade-split-dispatch` (performance). `SplitDispatch`: A dispatch is more than one
  Vulkan dispatch: a group count is past the device's `maxComputeWorkGroupCount`, or its grid is not
  a whole number of threadgroups and the kernel observes its threadgroup partition (a kernel that
  does not runs any such grid as one dispatch of its own threadgroup size, every invocation taking
  the grid thread it counts to). Cost: one Vulkan dispatch per region (the interior and each partial
  edge, per chunk of group counts), each on its own specialized pipeline, a barrier between each
  two.
- `metal2vulkan-degrade-emulated-simdgroups` (performance). `EmulatedSimdgroups`: A dispatch whose
  threadgroups are not whole subgroups of the kernel's pinned subgroup size runs on the kernel's
  emulated-simdgroups variant. Cost: a second compile of the kernel whose simdgroup operations go
  through threadgroup memory between barriers.
- `metal2vulkan-degrade-separate-depth-stencil` (performance). `SeparateDepthStencil`: A pass of
  separate depth and stencil textures, or of a stencil-only texture alone at custom sample
  positions, renders into a scratch image of a combined format. Cost: each loaded aspect is copied
  into the scratch image before the pass and each stored one back after it, through a staging
  buffer, or a multisampled colour image for multisampled textures, or sample-rate draws where the
  device's copies of those are missing or wrong.
- `metal2vulkan-degrade-one-aspect-resolve` (performance). `OneAspectResolve`: A pass resolves a
  combined depth/stencil attachment one aspect at a time, on a device that resolves the two aspects
  only together or only in one mode, or where its aspects resolve into different textures. Cost: a
  scratch image per aspect, the second aspect resolved after the pass in a render pass of its own
  from the stored samples, and each resolved aspect copied into the resolve texture after the pass
  through a staging buffer.
- `metal2vulkan-degrade-resolve-scratch` (performance). `ResolveScratch`: A pass resolves into a
  texture that cannot be an attachment (its usage lacks `RenderTarget`), or a layered pass into a
  texture whose slices from the resolve slice hold fewer layers than the pass's. Cost: the pass
  resolves into a scratch image, whose render area is copied into the texture after the pass.
- `metal2vulkan-degrade-shader-resolve` (performance). `ShaderResolve`: A multisample color resolve
  drawn after the pass by the crate-owned resolve: of an extended-range attachment, whose codes no
  device's resolve averages, or into the attachment's format's sRGB twin, whose encoding no
  device's resolve writes; or a stencil resolved by `DepthResolvedSample` after `Min` or `Max`, which no
  Vulkan resolve mode states, or a `Min` or `Max` depth on a device without the mode, drawn after
  the pass. Cost: one more render pass instance after the pass reads the stored samples (or the
  sample planes) and writes the resolve texture, where the device would resolve in the pass; where
  the attachment's image takes no input attachment usage, a copy of its samples over the render
  area into a scratch image first.
- `metal2vulkan-degrade-shader-color-conversion` (performance). `ShaderColorConversion`: A fragment
  function writes a float to a half-float color attachment of a pipeline under
  `TextureWriteRoundingMode::NearestEven`, whose rounding no device's conversion gives
  (`StageModule::rounded_colors`). Cost: the fragment function rounds each such channel to nearest
  even before it writes it: about 35 more instructions per channel.
- `metal2vulkan-degrade-render-pass-split` (performance). `RenderPassSplit`: A Metal render pass
  runs as several Vulkan render pass instances: a framebuffer fetch through the pixel interlock
  keeps its attachment out of its draws' subpass, and the pass's draws leave different attachments
  out, or its last draws leave out an attachment that resolves, which resolves in one more instance;
  or a fetch through input attachments reads an attachment whose image takes no input attachment
  usage (a caller's) after other draws recorded, and the stand-in that has it makes it an input
  attachment of the later draws' subpass alone; or a tile dispatch runs between two instances; or
  some of its draws render some of its layers, each run of them in an instance over those layers; or
  a memory barrier over buffers or textures orders its draws after it behind those before it, which
  a barrier between instances does. Not where the pixel interlock alone splits the pass on a device
  whose driver ran such a pass no slower split than whole
  (`DriverFact::InterlockedPassSplitNoSlower`): the split is then its faster form. Cost: each
  instance after the first loads the attachments and each before the last stores them, with a
  barrier from everything before to everything after between them: on a tiler, a tile flush per
  split.
- `metal2vulkan-degrade-amplification-draws` (performance). `AmplificationDraws`: A draw under
  vertex amplification that Vulkan multiview cannot express (view mappings that offset the viewport
  or name layers out of order, more amplifications than the device's views, or a pipeline that
  chooses or reads its render target slice) draws one amplification at a time
  (`bind::DrawCommands::amplifications`). Cost: one Vulkan draw per amplification, each after the
  amplification's push data, on the pipeline state's variant that reads it, where multiview would
  draw them all at once.
- `metal2vulkan-degrade-memoryless-backed` (performance). `MemorylessBacked`: A memoryless
  attachment renders into a backing image of the device's own memory, because a framebuffer fetch
  through the pixel interlock reads it or the pass splits; not for a split `render-pass-split`
  leaves unannounced. A color's backing is one its command queue keeps for its passes of the
  attachment's format, samples and size. Cost: a scratch image, a color's kept by its command queue
  for its passes of that shape and a depth or stencil one's per pass: memory, and the bandwidth to
  write and read it, that Metal does not spend on a memoryless texture.
- `metal2vulkan-degrade-memoryless-texture-buffer` (performance). `MemorylessTextureBuffer`: A
  memoryless texture buffer (`newTextureWithDescriptor:` of a `TextureBuffer` descriptor of
  `MTLStorageModeMemoryless`, which plain Metal makes), whose texels are a buffer's bytes: the
  buffer is a `Private` one of the device's, since no Vulkan buffer takes memory lazily. The
  texture answers the descriptor's memoryless mode, as Metal's does, and a command that uses it
  fails its command buffer, as plain Metal's. Cost: a Private buffer of the texture's row of
  texels: memory Metal spends none of on a memoryless texture. Nothing reaches those bytes.
- `metal2vulkan-degrade-fetch-through-scratch` (performance). `FetchThroughScratch`: A color render
  target that a framebuffer fetch reads lacks the usage the fetch's form needs, so its pass renders
  into a scratch image that has it: `STORAGE` in the pixel interlock, or for its sample planes
  sampled usage with a view of its texel size's unsigned-integer format, either of which the runtime
  gives a texture only where it leaves the image's layout (its compression) alone; or input
  attachment usage, which a caller's image may lack. Cost: a scratch image per pass, the texture
  copied into it before the pass on a `Load` and back after it on a `Store`.
- `metal2vulkan-degrade-fetched-target-storage` (performance). `FetchedTargetStorage`: A runtime
  texture that a pass rendered into a scratch image for want of `STORAGE` (`fetch-through-scratch`
  through the pixel interlock) took its storage form: its image made again with `STORAGE` and its
  texels copied in once, at the commit of the device's only uncommitted command buffer; with a
  submission check, once that command buffer's checks ran at its submission and every encoder of it
  ran. Announced once per device, whatever `DeviceOptions::announce_each` says, before that command
  buffer is scheduled: on the scheduler's thread once it submitted it, outside any program call or
  host step, or in the call that makes a command buffer meanwhile. Cost: the texture's image laid
  out as one with `STORAGE` usage from then on (on MoltenVK `ShaderWrite`, without lossless
  compression), and one copy of its texels, where Metal's texture keeps its usage.
- `metal2vulkan-degrade-positions-through-scratch` (performance). `PositionsThroughScratch`: A
  depth/stencil render target whose image was not made for the pass's custom sample positions (a
  caller's image without `VK_IMAGE_CREATE_SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_BIT_EXT`, which Vulkan
  needs to rasterize a depth/stencil attachment at them), or was on a driver that keeps a depth
  rendered at them per sample only through a copy
  (`metal2vulkan::device::DriverFact::PositionedDepthReadAtStandardPositions`: NVIDIA's, measured),
  renders into a scratch image made for them. Cost: a scratch image per pass, each aspect copied
  into it before the pass on a `Load` and back after it on a `Store`.
- `metal2vulkan-degrade-sample-planes` (performance). `SamplePlanes`: A framebuffer fetch through
  the pixel interlock in a multisampled pass, on a device without multisampled storage images, reads
  and writes the attachment as its sample planes: a single-sampled array image of one layer per
  sample. Cost: a planes image per fetched attachment and pass; a compute dispatch copies the
  samples into it before the draws that fetch (on a `Load`, or after other draws rendered it) and a
  sample-rate pass copies them back after those draws (before other draws render it, and where the
  pass keeps or resolves it).
- `metal2vulkan-degrade-shader-side-stencil` (performance). `ShaderSideStencil`: A render pass's
  draws test and write the stencil in the fragment shader, on a device without
  `VK_EXT_shader_stencil_export`: the function's `[[stencil]]` output is Metal's stencil reference,
  which only that extension hands the fixed-function test. Cost: an 8-bit shadow image of the
  stencil per pass, which the stencil is copied into before each run of such draws and back from
  after it (through a staging buffer at one sample), each run in a render pass instance of its own,
  its fragment shader testing and writing the shadow in the pixel interlock.
- `metal2vulkan-degrade-tile-dispatch-scratch` (performance). `TileDispatchScratch`: A tile dispatch
  stages its pass's attachments through storage views in their staging formats, which it takes of a
  scratch image of each attachment's format: each is copied into its scratch before the dispatch and
  back after it. Cost: a scratch image per staged attachment and pass, over the pixels the dispatch
  covers, which each dispatch copies the attachment into before it and back from after it, between
  barriers from everything before to everything after: memory and bandwidth that Metal's tile memory
  does not spend.
- `metal2vulkan-degrade-fragment-imageblock-storage` (performance). `FragmentImageblockStorage`: A
  render pass's draws' fragment functions declare an explicit imageblock, which they read and write
  in a storage image of the render area. Cost: an R32Uint array image per pass over the render area,
  a layer per 4 bytes of a pixel's imageblock and per sample, cleared before the pass, which the
  draws read and write in the pixel interlock: memory and bandwidth that Metal's tile memory does
  not spend.
- `metal2vulkan-degrade-wrapped-image-layout-transitions` (performance).
  `WrappedImageLayoutTransitions`: A command buffer uses a texture over the caller's image whose
  home layout is not `GENERAL`: every image the runtime uses rests in `GENERAL`. Cost: two
  whole-image barriers per batch that uses it: into GENERAL before its first encoder, and back to
  its home layout at its end.
- `metal2vulkan-degrade-host-replayed-indirect-commands` (performance).
  `HostReplayedIndirectCommands`: An execution of an indirect command buffer
  (`executeCommandsInBuffer:withRange:`) whose commands are known only once the work before it
  completed: an encoder made the buffer resident for writing, so the GPU may write its records; it
  reads its range from a buffer; or it lies inside a block of command predication. The runtime
  records such an execution on the CPU then where the GPU cannot run it alone. A render execution:
  where its records may hold, at the commit or as its batch is ready, more than draws of one
  primitive type under the encoder's state, or what its check looks for; its notice names the first
  rule that kept it from the GPU alone. A compute execution: with a caller's check
  (`execute_commands_in_buffer_checked`) that may see what it looks for
  (`IndirectCommandCheckNeeds`: a state that may encode indirect commands, the encoder's where the
  buffer inherits it, else one its records may name), at the commit or as its batch is ready; with
  its range read from a buffer; of a buffer that inherits the encoder's buffers; where its records
  may name a pipeline state whose kernel observes its threadgroup partition or fetches
  `[[stage_in]]` attributes, or a state the CPU set that the program released since; and, on a
  device without `VK_EXT_device_generated_commands`, where the host step is faster: where its
  commands run with no barrier between them and its commands times the pipeline states its records
  may name past four pass 256 (the states: every live one made with `supportIndirectCommandBuffers`,
  or the encoder's where the buffer inherits it), where they run with one and its commands times the
  states past twelve pass 512, and where its commands times its states pass 32,768. Every other
  execution runs on the GPU alone and announces nothing. Cost: one GPU-to-CPU round trip per
  execution: its command buffer's work before it, and every batch submitted before that on any
  queue, completes before the CPU records its commands from their records and submits the rest; the
  device's scheduler submits nothing else while it records them, and the first command of a
  threadgroup size or threadgroup lengths its pipeline state has not dispatched makes that
  VkPipeline then. The commands run in the large-buffer form (their range-limited buffer slots at
  device addresses, with in-shader bounds checks), whose variants the commit compiles for every
  pipeline state a record may name (or, for an execution committed before the buffer was made
  resident for writing, the use_resource call that made it so); without device addresses, a
  command's first buffer shape compiles its variant then.
- `metal2vulkan-degrade-uncalibrated-gpu-times` (precision). `UncalibratedGpuTimes`: A command
  buffer's GPU times on a device that calibrates its timestamps against no host clock the runtime
  reads (no `VK_KHR_calibrated_timestamps` or `VK_EXT_calibrated_timestamps` with the device's time
  domain and a monotonic host one), whose calibration failed as it completed, or whose calibration
  placed them outside the command buffer's life (before its first submission, after its completion)
  by more than its stated deviation and the clocks' drift allow. Cost: the end is the host time the
  runtime saw the command buffer complete, late by that latency, and the start the GPU's exact
  duration before it.
- `metal2vulkan-degrade-untimed-command-buffer` (precision). `UntimedCommandBuffer`: A command
  buffer's GPU times on a device whose timestamp pools had no slot free at the command buffer's
  commit, and which could make no pool more: every slot was held by a command buffer in flight.
  Cost: its GPU start and end times are refused (`Unsupported`, `GpuTimes`), where Metal gives them;
  its kernel times stay.
- `metal2vulkan-degrade-gpu-times-from-host` (precision). `GpuTimesFromHost`: A command buffer's GPU
  times on a device told to answer them from the host (`DeviceOptions::gpu_times_from_host`,
  `RawDevice::gpu_times_from_host`), or whose driver's timestamps wait for every queue's
  (`DriverFact::TimestampsWaitForEveryQueue`: MoltenVK 1.4.1 on Apple's GPUs, 1.4.2 on the M3
  Max), which writes no GPU timestamps. Cost: its GPU start is its kernel end time (or its queue's
  latest GPU end before it) and its GPU end the host time the runtime saw it complete: a window its
  work ran in, which holds the submission's and the completion's latencies, not the GPU's own start
  and end.
- `metal2vulkan-degrade-device-rounded-resolve` (precision). `DeviceRoundedResolve`: A multisample
  color resolve averages through Vulkan's `AVERAGE` mode, which leaves the rounding of the average
  (and the sRGB encoding) to the device, on a device not measured to round as Metal does: any
  color attachment but an extended-range one and one resolving into its format's sRGB twin (each
  resolved after the pass), one a fetching pass's sample planes hold as it ends included (the
  planes merged back first). Never, whatever the
  format, on a device whose resolve is measured to round
  as Metal does (`DriverFact::ColorResolveAverageIsMetals`: MoltenVK's resolve is Metal's own).
  Also, on every device, a resolve into an 8-bit UNORM attachment's sRGB twin where the device has
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
- `metal2vulkan-degrade-unmeasured-resolve` (precision). `UnmeasuredResolve`: A multisample color
  resolve of more than 4 samples, of which Apple's GPUs make no texture: no Metal answer is measured
  there; likewise a stencil resolved by `DepthResolvedSample` after `Min` or `Max`, whose pick among
  equal depths is measured at 2 and 4 samples. Cost: the resolve is the device's average, or after
  the pass the rules Metal's resolve follows at 2 and 4 samples, which no Metal answer confirms at
  this count.
- `metal2vulkan-degrade-extended-range-resolve` (precision). `ExtendedRangeResolve`: A multisample
  resolve of an extended-range color attachment whose resolve Metal averages from values finer than
  the codes the attachment holds (`BGRA10_XR`'s alpha, an `_sRGB` twin's color,
  `RGB10A2Unorm_sRGB`'s channels): the crate-owned resolve writes a near-equivalent from the codes
  where a pixel's samples differ, and Metal's code where they hold one. Measured against Metal's
  resolve on the M3 Max under macOS 27.0.1 (4096 texels per format at 2 and 4 samples, each sample
  its own value): `BGRA10_XR`'s alpha Metal's in 84% and 88% of texels and within a code in every
  one; an `_sRGB` twin's color in 81% and 85% of channels and within a code in 99.5%, at most 7
  codes off where the samples' values nearly cancel; `RGB10A2Unorm_sRGB`'s colors in 83% and 87%,
  within 2, and its alpha in 94% and 97%, within one. A linear form's color is Metal's rule, every
  texel. Cost: where a pixel's samples differ, a resolved alpha, or an sRGB color, can differ from
  Metal's by a code (an sRGB color by a few where the samples' values nearly cancel): Metal averages
  values finer than the codes the attachment holds.
- `metal2vulkan-degrade-dont-care-store` (precision). `DontCareStore`: An attachment stored with
  `DontCare` (or the multisampled attachment of `MultisampleResolve`) is Vulkan's `DONT_CARE`, where
  Apple's GPU keeps the earlier texels unless its tiler overflows. Only on a driver measured to keep
  either the earlier texel or the stored one at each texel
  (`DriverFact::DontCareStoreIsOneOfMetals`); every other device stores the attachment, Metal's
  answer where its tiler overflows. Not a texture over a buffer: Metal leaves the buffer's bytes in
  it (on an M3 Max, every texel of 926 such passes alike, and of 36 with a memory barrier between
  two draws, 20 times each, but that the first pass in a process to
  overflow Apple's tiler leaves the rendered texels, which no call can predict or detect), and the
  runtime gives the buffer's bytes, exactly, through an image the pass copies nothing back from
  (`dont-care-stand-in` below, or the texture's shadow), or by handing the attachment to Metal's
  own `DontCare` where the driver stores it so; only where the pass cannot (a draw took a view of
  the texture's own image before its store action was known, the pass executes render commands,
  or the device copies no image of the format) does the pass render in place, announced so. Cost:
  the attachment's texels after the pass are each its earlier texel or the one the pass rendered,
  both Metal's answers, but not always the one Metal's tiler keeps. Within Metal's contract:
  `MTLStoreAction.dontCare` lets the GPU leave arbitrary data ("You need to assume that GPU
  discarded the texture's contents").
- `metal2vulkan-degrade-dont-care-load-tiles` (precision). `DontCareLoadTiles`: An attachment loaded
  with `DontCare` and stored, in a pass that draws, is cleared before the draws to zero (a color) or
  its clear value (depth, stencil) over the render area and the tiles past it a clear reaches, where
  Metal clears only the tiles a draw's primitives touch and keeps the texels of the others. Cost:
  the tiles no draw touches hold zero or the clear value where Metal keeps their texels.
- `metal2vulkan-degrade-fresh-pattern-zeroed` (precision). `FreshPatternZeroed`: A new `Private`
  texture Metal keeps losslessly compressed reads a fixed pattern in Metal, which the device cannot
  write exactly into its image: a NaN depth where a copy into a depth image may store it clamped or
  the image is multisampled, a NaN texel of a multisampled image (whose bits a clear leaves to the
  device), or an image holding other bytes than Metal's texel. Cost: the new texture reads zero
  where Metal reads its fixed pattern until a write.
- `metal2vulkan-degrade-zeroed-index-fetch` (precision). `ZeroedIndexFetch`: An indexed draw from an
  index offset at or past its index buffer's length draws index 0 plus the base vertex at each
  position, a direct draw from a zero-padded copy and an indirect one through the index bounds pass,
  where Metal reads the memory past the buffer. Cost: each index position draws index 0 plus the
  base vertex, where Metal reads whatever memory follows the buffer.
- `metal2vulkan-degrade-implementation-defined-reconvergence` (precision).
  `ImplementationDefinedReconvergence`: A pipeline stage's reconvergence after divergence is
  implementation-defined on the device, where Metal's is maximal
  (`Reconvergence::ImplementationDefined`). Cost: simdgroup operations after divergent control flow
  can see other lanes active than Metal's.
- `metal2vulkan-degrade-unlisted-argument-sampler` (precision). `UnlistedArgumentSampler`: A live
  sampler state the device's sampler tables do not map to it for some commands, which reach every
  other live state by its ID as they run, whenever it was made: one whose element no set-1 sampler
  array can hold, as more states live than it holds (the device's, as the state is made, or a
  pipeline's, as a command is encoded or the state made), each such command buffer counted once
  (`WorkCounts::unlisted_argument_sampler_buffers`); one at an ID neither window of the tables
  reaches (IDs 0 to 32767, and 32768 from the first ID past them, a reservation's); one whose
  Vulkan sampler for an array's use the device cannot make or the use refuses, that array holding
  the neutral at its element; a cubic state, for a command whose stage does not test the states it
  reaches for a cubic filter (a fragment stage never does); and one whose elements could not be
  written. Cost: a function reading the state through its ID in an argument buffer reads the
  neutral sampler, where Metal's reads the state.
- `metal2vulkan-degrade-restricted-depth-upload` (precision). `RestrictedDepthUpload`: An upload
  (`replace_region`, a blit from a buffer) into the 32-bit float depth of a texture's depth image,
  on a device without `VK_EXT_depth_range_unrestricted` enabled, where Vulkan leaves a value outside
  `[0, 1]` undefined, whose driver is not measured to keep one
  (`DriverFact::DepthImageKeepsValuesOutsideUnitRange`). And a render pass's NaN clear depth on such
  a device, which Metal stores as `0x7fc00000`, cleared to 1.0. Cost: a value outside [0, 1] may be
  stored clamped to [0, 1], where Metal's keeps it, so reads, samples, comparisons and depth tests
  of it can differ.
- `metal2vulkan-degrade-host-read-dispatch-arguments` (performance). `HostReadDispatchArguments`: An
  indirect dispatch, or a dispatch after `setStageInRegionWithIndirectBuffer:indirectBufferOffset:`:
  its arguments or its stage-in region are GPU memory the work before it may write, so they are
  known only once that work completed. Cost: one GPU-to-CPU round trip per dispatch: a copy of its
  arguments where it is encoded, then its command buffer's work before it, and every batch submitted
  before that on any queue, completes before the CPU reads them and records the dispatch they state;
  the device's scheduler submits nothing else meanwhile, and a threadgroup size its pipeline state
  has not dispatched makes that VkPipeline then.
- `metal2vulkan-degrade-non-strict-residency` (precision). `NonStrictResidency`: A sparse texture on
  a device whose sparse residency is not `residencyNonResidentStrict`. Cost: a read of a texel no
  mapped tile holds, by a function or a copy, answers the device's value where Metal's reads zero,
  and a write to one may be kept where Metal's drops it.
- `metal2vulkan-degrade-host-read-mapping-arguments` (performance). `HostReadMappingArguments`: An
  indirect mapping update of a sparse texture, whose regions the host reads. Cost: one GPU-to-CPU
  round trip per batch of such updates: the command buffer's work before it, and every batch
  submitted before that on any queue, completes before the CPU reads the regions (through a copy for
  a Private buffer) and binds them; the device's scheduler submits nothing else meanwhile.
- `metal2vulkan-degrade-emulated-sparse-texture` (performance). `EmulatedSparseTexture`: A sparse
  texture the device makes no sparse image of (MoltenVK binds no sparse memory), where no function
  reads an image's own residency (the device does not enable `shaderResourceResidency`): an ordinary
  image whose unmapped tiles the runtime keeps at zero, which gives Metal's answers. Cost: the
  texture holds its whole image's memory from when it is made, where Metal's backs only its mapped
  tiles; each batch that changes its mappings zeroes the tiles it unmaps and rewrites its tile map
  before its work, and each blit that writes it zeroes the unmapped tiles it wrote.
- `metal2vulkan-degrade-shared-in-system-memory` (performance). `SharedInSystemMemory`: A `Shared`
  buffer or heap the program makes with the default CPU cache mode, on a device without unified
  memory (`DeviceProperties::has_unified_memory`), lies in host-cached system memory, which the host
  reads quickly, as a shared resource's on a Mac with a discrete GPU. Cost: the GPU reads and writes
  it across the bus, as a discrete Mac's GPU does a shared resource: coalesced accesses at the bus's
  speed, and scattered writes far slower (4-byte stores 12 bytes apart at 0.08 GB/s on an RTX 3090,
  where device-local memory takes them at 850 GB/s).
- `metal2vulkan-degrade-device-filtered-mipmaps` (precision). `DeviceFilteredMipmaps`:
  `generate_mipmaps` of a filtered format (an integer format's texels are picked as plain Metal
  picks them): each level is the device's linear filter of the one above at the scaled texel
  centres, its blit chain (`vkCmdBlitImage` with `VK_FILTER_LINEAR`) where its blits filter the
  image's format, else the runtime's bilinear (trilinear in 3D) compute filter in `float`: an
  extended-range format, an 8-bit sRGB texture held as its UNORM twin, or a format without blits or
  linear filtering. The compute filter's arithmetic is the device's: its products and sums as it
  contracts them, an 8-bit or extended-range sRGB code decoded and encoded through the MSL transfer
  by its `Pow` (as a shader's read of the UNORM twin decodes it, `srgb-texel-decode`), a half by
  its widening and narrowing, and a NaN its own; `bind::MipmapFilterChain::outside_bound` states
  the bound its levels keep to the filter in IEEE `f32`. Neither is Apple's datapath, whose answers
  also differ between Apple's GPUs (an M3 Max's and an M4's by up to a unit in the last place).
  Announced once per device; no bound against Metal's answer is measured. Cost: each level is the device's linear filter of the one
  above (its blit, or a bilinear filter in float where its blits cannot filter the format), not
  Apple's: a channel can differ from Metal's where the filtered value lies near a rounding boundary,
  and wholly where a box averages infinities, NaNs or a float format's extremes.
- `metal2vulkan-degrade-heap-depth-stencil-apart` (precision). `HeapDepthStencilApart`: A depth or
  stencil texture made in a heap on a device whose driver lays depth and stencil images apart from
  the memory they are bound to (`DriverFact::DepthStencilImagesLieApart`) that no shadow of the
  heap's bytes serves: a multisampled texture, which Vulkan copies to and from no buffer, or one
  whose texels do not fit its bytes. Cost: the texture's texels lie apart from
  the heap's bytes: a write through a buffer
  or texture over its bytes changes none of them, and its writes change none of those resources'
  bytes, where Metal's texture aliases them.
- `metal2vulkan-degrade-heap-texture-host-transfer` (performance). `HeapTextureHostTransfer`: A
  colour texture made in a heap on a device whose driver's compressed images hide writes through an
  alias (`DriverFact::CompressedImagesHideAliasedWrites`), whose image takes host-transfer usage,
  which the driver keeps uncompressed in the heap's bytes
  (`metal2vulkan::device::DriverFact::HostTransferImagesKeepTheirBytes`). Cost: the texture's image
  is kept uncompressed, so writes through the heap's buffers and textures over its bytes show
  in it as Metal's do: a clear of it loses the driver's fast clear (334.5 us where a compressed
  2048x2048 RGBA8Unorm image's took 4.1 us on an RPL-S), while its copies, sampling and blending
  measured no slower.
- `metal2vulkan-degrade-heap-texture-linear` (performance). `HeapTextureLinear`: A colour texture
  made in a heap on such a device whose image takes no host-transfer usage (the device did not
  enable `hostImageCopy`, or an adopted device's consumer did not say so, or the format or shape
  takes none there): a linear image, which the driver does not compress, where one does all the
  optimal image does for the texture. Cost: the texture's image is linear, so writes through the
  heap's buffers and textures over its bytes show in it as Metal's do: its reads and writes go
  without the optimal tiling's locality and compression (3.9 to 9.6 times the host-transfer image's
  time for copies, clears and draws of a 2048x2048 RGBA8Unorm texture on an RPL-S).
- `metal2vulkan-degrade-heap-texture-hides-aliased-writes` (precision).
  `HeapTextureHidesAliasedWrites`: A colour texture made in a heap on such a device whose image
  takes neither host-transfer usage nor linear tiling: an optimal image the driver may compress, a
  near-equivalent recorded open. Cost: the driver may compress the texture's image: a write through
  a buffer or texture over its bytes may leave texels as they were, and its writes may show in those
  resources' bytes compressed, where Metal's texture aliases them byte for byte.
- `metal2vulkan-degrade-heap-texture-compressed` (precision). `HeapTextureCompressed`: A colour
  texture made in a heap on such a device that Metal keeps losslessly compressed too
  (`allowGPUOptimizedContents`, no `ShaderWrite`, `PixelFormatView` or `ShaderAtomic` usage, 16
  texels or more a side), under `HeapCompressionPolicy::Compressed`, the default: the driver's
  compressed optimal image, its fast clears kept. Cost: the driver may compress the texture's image,
  as Metal compresses the texture: a write through a buffer or texture over its bytes may leave
  texels as they were, where Metal's texture reads opaque magenta (0xffff00ff) in every texel of
  drawn content the write covered and the written bytes where it held only a clear (an M3 Max), and
  its writes may show in those bytes compressed otherwise than Metal's;
  HeapCompressionPolicy::Uncompressed keeps the image uncompressed in the heap's bytes, whose clears
  then lose the driver's fast clear.
- `metal2vulkan-degrade-combined-archive-stages` (precision). `CombinedArchiveStages`: A render
  creation that fails on a binary-archive miss, made from one add's vertex stage and another add's
  fragment stage: Metal's vertex key holds the vertex outputs its compiler finds the fragment
  reading, which the runtime's lookup does not decode (`binary_archive::metal_key`). Cost: the
  pipeline is made where Metal may refuse it with a binary-archive miss: Metal misses when the two
  adds' fragment functions read different vertex outputs.
- `metal2vulkan-degrade-shadowed-texture-view` (performance). `ShadowedTextureView`: A texture view
  is an image of its own, a shadow of the viewed image's texels, where that image takes no Vulkan
  view of its format: a view of another component layout on a device whose Vulkan views reinterpret
  no image's components (a portability subset without `imageViewFormatReinterpretation`, or an
  adopted device whose consumer did not say it enabled it), or a view of another format than the
  texture's own and its sRGB twin's of a texture made without `PixelFormatView` usage. A view in
  other channels of its image's bits (an `A8Unorm` view of an `R8Unorm` texture) is the image
  itself to its bindings, through a swizzle, and takes a shadow only for a pass that renders into
  it: Vulkan views the image in no format of the view's. Cost: its texels are copied from its
  texture's image before each use and back after each write.
- `metal2vulkan-degrade-repeated-execution-start` (precision). `RepeatedExecutionStart`: A render
  encoder executes an indirect command buffer's commands from a command it executed them from before
  with another length, each range given (`executeCommandsInBuffer:withRange:`) or read from a
  buffer: Metal's executions then do not each run their own range, and not the same way from run to
  run, on an Apple M3 Max and an Apple M4 alike, so no stable answer exists. Cost: each execution
  runs its own range, as Metal documents, where Metal's GPU may run one execution with the other's
  length, so commands run more or fewer times than the ranges name, differently from run to run.
- `metal2vulkan-degrade-predication-word-past-buffer` (precision). `PredicationWordPastBuffer`: A
  command predication record (`encode_start_if`, a loop's start, a live DoWhile's end) whose
  condition word lies past its buffer's end, past the 256 bytes Metal rounds a lone buffer up to,
  or past the end of the heap the buffer lies in: Metal reads whatever its own address space holds
  there, a neighbouring buffer's bytes or zero, which depends on where Metal placed memory
  unrelated to the buffer. Cost: the record reads its word as zero, where Metal reads what lies at
  that address in its own address space: zero where nothing is there, another buffer's bytes where
  Metal placed one.
- `metal2vulkan-degrade-private-resource-option` (precision). `PrivateResourceOption`: A heap of
  the runtime's memory or an indirect command buffer made with `MTLResourceOptions` bit 24
  (`ResourceOptions::PRIVATE_BIT`), or off macOS a no-copy buffer or heap over the caller's bytes,
  which Metal makes and keeps the GPU's writes from the memory of, is made as without it, and
  reports its options without it, as Metal's does. On macOS a no-copy buffer or heap with the bit
  drops the GPU's writes as Metal's does, its pages imported through a read-only view of them,
  announcing nothing. A buffer of the runtime's memory or a texture descriptor with the bit is
  refused, as Metal ends the process there; a buffer placed in a heap without the bit is made as
  without it, announcing nothing. Cost: the GPU's writes reach the resource's memory, the CPU and
  later command buffers, where Metal's never reach them.
- `metal2vulkan-degrade-optimized-depth-stencil-texture` (precision).
  `OptimizedDepthStencilTexture`: A depth or stencil texture with `allowGPUOptimizedContents` off
  sized or made in a heap (never a device's own texture, whose layout nothing shows), on a device
  whose driver keeps depth and stencil images GPU-optimized whatever their create info
  (`DriverFact::DepthStencilImagesStayGpuOptimized`), that no shadow of Metal's tiles in the heap's
  bytes serves (a multisample texture, a format whose layout is not measured): Metal
  lays the texture out without its compression metadata, and the runtime's image keeps it. Cost: the
  texture's image keeps the GPU-optimized layout Metal
  drops with the property off, so its bytes in the heap are not Metal's layout of its texels (where
  they lie instead, and what moving them costs, heap-texture-untiled-layout and shadowed-texture
  name for the shapes they cover); heap_texture_size_and_align answers Metal's own size and
  alignment for the descriptor wherever the runtime models them, and the larger of the images' and
  Metal's for a texture whose images need more bytes or a coarser alignment that no shadow of the
  heap's bytes holds it to, where a heap sized by Metal's answers holds fewer such textures.
- `metal2vulkan-degrade-narrow-surface-rows` (precision). `NarrowSurfaceRows`: A texture of an
  IOSurface plane of more than one row under 16 bytes a row: Metal's GPU writes its rows 16 bytes
  apart and reads every row past the first as memory-dependent bytes (zeros in a fresh process;
  after a write, bytes neither the surface's nor written). Cost: a GPU read of a row past the first
  reads the bytes the GPU's writes put there, 16 bytes a row apart, where Metal's GPU reads
  memory-dependent bytes; CPU access and the GPU's writes and first row are Metal's.
- `metal2vulkan-degrade-colliding-surface-row-writes` (precision). `CollidingSurfaceRowWrites`: A
  texture of an IOSurface plane whose rows share bytes on Metal's GPU: more than one row of over 16
  bytes whose `bytesPerRow` rounded down to 16 is below the row, so consecutive rows share 1 to 15
  bytes, a whole number of texels. Metal's reads see the sharing and so do the runtime's, through a
  shadow whose two copy regions each way hold it; but where two rows' writes meet the shared bytes
  take whichever of the two rows Metal's command wrote last, and that order is no one order: it
  follows the command, the GPU pitch and the row count, a kernel's winner moving with the
  threadgroup shape. A kernel is the only writer such a plane takes, every blit into one being
  refused. Cost: where two of the plane's rows write the bytes they share the earlier row's stand,
  one of the rows a kernel of Metal's leaves there, whose winner moves with the threadgroup shape;
  reads, CPU access and every other byte are Metal's.
- `metal2vulkan-degrade-absent-argument-read` (precision). `AbsentArgumentRead`: A pipeline state's
  function may read an argument its function constants removed (a texture's read, sample, gather,
  query or atomic, or a sample through a removed sampler; `RuntimeFact::AbsentArgumentRead`), which
  Metal gives no value: its compiler drops or folds such a read, and with its debug layer every such
  read is zero. Cost: such a read takes zero, Metal's answer under its debug layer, and the command
  buffer completes, its execution report stating the read (absent_argument_read); without the layer
  Metal's compiler drops or folds the read instead.
- `metal2vulkan-degrade-mixed-depth-stencil-resolve` (precision). `MixedDepthStencilResolve`: A
  render pass whose one depth/stencil texture stores and resolves its depth
  (`StoreAndMultisampleResolve`) while it resolves its stencil alone (`MultisampleResolve`), which
  Metal's validation stops (`resolve::layout_patterned`). Cost: both aspects resolve and the depth's
  samples are kept, as the store actions document, and the stencil's samples are kept too, where
  Metal's GPU overwrites the depth's samples and its resolve texture with a layout pattern of the
  resolved depth and leaves the stencil's texels (measured on an Apple M4).
- `metal2vulkan-degrade-capped-anisotropy` (precision). `CappedAnisotropy`: A sampler state whose
  `maxAnisotropy`, which Metal's validation stops past 16, plain Metal samples with an anisotropy
  past the device's largest (`maxSamplerAnisotropy`): 17 to 31 in its low five bits, which Metal
  samples with 32. Cost: it filters with the device's largest anisotropy (16 on every venue
  measured), where Metal filters with 32: a footprint more than 16 times as long as it is wide reads
  a coarser level than Metal's (one level coarser at 32 to 1 and beyond).
- `metal2vulkan-degrade-not-mipmapped-anisotropic-sampler` (precision).
  `NotMipmappedAnisotropicSampler`: A sampler state with anisotropy, no mip filter (`NotMipmapped`)
  and different minification and magnification filters, on a device that samples through Metal's
  own sampler (MoltenVK on Apple's GPU): Vulkan has no sampler without mipmapping, so the device's
  sampler takes a nearest mip filter with the level clamp `[0, 0.25]`, which keeps Metal's choice
  of filter, and Metal's anisotropic filter reads no such clamp as Metal's `NotMipmapped`.
  Announced once, as the device makes the state. Cost: a sample at an implicit level or with
  gradients through it filters another anisotropic footprint than Metal's at `maxAnisotropy` 2 to
  16. A constexpr sampler of that shape records `PrecisionNote::NotMipmappedAnisotropicFilter`.
- `metal2vulkan-degrade-attachment-read-in-place` (precision). `AttachmentReadInPlace`: A draw whose
  functions read or write, other than as an attachment, memory an attachment of its render pass
  instance renders, where no copy or snapshot serves the access: a texture reached through a handle
  or an argument buffer, a storage binding of a color attachment's texture, the depth or stencil
  attachment's texture, a texture buffer or a buffer over its bytes, a texture whose copy is not
  served, or an execution's or a capture's command. Announced where the draw records. Cost: the
  access sees the attachment as the device holds it inside its render pass instance, which Vulkan
  leaves undefined (`VUID-vkCmdDraw-None-10795`), where Metal's read at the draw's own pixel gives
  the texel from before the pass.
- `metal2vulkan-degrade-unreadable-handle-texture` (precision). `UnreadableHandleTexture`: A command
  whose function reads texture handles from memory (an argument buffer, or any memory holding a
  texture's resource ID) has a texture the runtime holds as raw blocks for copies made resident, or
  bound in a stage that reads handles: the handle's element is the neutral texture, as the runtime
  does not know which of the textures it reaches the function reads. Cost: a read or sample of the
  texture through its handle reads the neutral texture where Metal's decodes its blocks; a function
  that reads it at a slot is refused (Unsupported(Format)).
- `metal2vulkan-degrade-unnamed-texture-unviewable` (precision). `UnnamedTextureUnviewable`: A
  texture that a heap array's class fits, but whose format the array cannot view (one the device
  cannot sample, or store without a format, or one whose raw texels need its function's decode): the
  device's heap set holds the class's neutral at its ID, which a command that names the texture
  refuses instead. Cost: a function reading the texture through its ID, in a command that does not
  name it (bind it, or make it resident), reads the neutral texture where Metal's reads the texture.
- `metal2vulkan-degrade-unnamed-shadowed-texture` (precision). `UnnamedShadowedTexture`: A shadowed
  texture in the device's heap set. Cost: a function reading the texture through its ID, in a
  command that does not name it, reads the texels an encoder that named it last copied in (undefined
  before any did), where Metal's reads the texels as they stand.
- `metal2vulkan-degrade-unnamed-textures-past-capacity` (precision). `UnnamedTexturesPastCapacity`:
  A live texture whose resource ID is past the elements of a pipeline's set-1 heap arrays (their
  count, the device's update-after-bind limits shared among the pipeline's arrays), so the texture
  table a command binds misses it. Cost: a function of such a pipeline reading the texture through
  its ID reads the neutral texture where Metal's reads the texture; a command that names it is
  refused (Bind(HeapTooLarge)).
- `metal2vulkan-degrade-heap-set-copied` (precision). `HeapSetCopied`: A heap set on a device
  without `descriptorBindingUpdateUnusedWhilePending`, which writes no set a pending command buffer
  binds: each command buffer binds a copy of its own, written as it first binds the set and until it
  commits. Cost: each command buffer's first command that binds the set writes a copy of it (every
  element a texture's ID may name, and every sampler array's, per array); a function reading a
  texture through its ID reads the texture that held the ID as its command buffer committed, and one
  reading a sampler state the state that held it as the command buffer's first batch was submitted
  (the neutral where none did, or where it was released before that submission), where Metal's
  reads the one holding it as the function runs.
- `metal2vulkan-degrade-uncovered-color-kept` (precision). `UncoveredColorKept`: A draw of a render
  pipeline state in a pass with a color at no place of the state's per-sample record, under a state
  color that blends through a destination factor of the fragment's output (`SourceAlpha`,
  `OneMinusSourceAlpha`, `SourceColor`,...) and reads the destination no other way. Cost: the pass's
  color keeps its contents where the draw covers it, as Metal keeps them for an output it computes;
  where Metal's compiler folds the output to a constant that zeroes the factor (an alpha of the
  literal 1 under `OneMinusSourceAlpha`), Metal stores zero there (measured on an Apple M4).
- `metal2vulkan-degrade-drawn-depth-copy` (precision). `DrawnDepthCopy`: A blit copy into a
  multisampled 32-bit float depth texture from a colour one or a buffer, a render pass's separate
  multisampled 32-bit float depth texture copied to and from its scratch image, or a heap's such
  texture held as a shadow of its samples in Metal's tiles, which takes the heap's bytes (then a
  value the draw changes is written back to the heap's bytes changed, announced as the texture is
  made), by the sample-rate `FragDepth` draw, on a device whose copies go by draws and whose driver's fragment
  depth is not measured to store every value unchanged: measured to change some values (MoltenVK,
  NVIDIA's), or not measured at all (any other driver, and lavapipe releases not measured). The
  reason names the values that driver changes, or, on a driver not measured, those Vulkan leaves to
  it (outside `[0, 1]`, denormals, -0). Cost: each sample's depth is written by a fragment-depth
  draw, where Metal copies its bytes: the values the driver's fragment depth changes or may change,
  which the reason names, can land changed; every other value lands as Metal's copy leaves it.
- `metal2vulkan-degrade-undeclared-usage-shadow` (performance). `UndeclaredUsageShadow`: A texture
  bound by a slot as a storage image without `ShaderWrite` usage, or as a sampled one without
  `ShaderRead`, whose image lacks that usage because adding it would cost the image its size or the
  driver's compression: the binding takes an image of its own with both shader usages, a shadow of
  the texture's texels. Cost: its texels are copied into an image of its own before each command
  that binds it beyond its declared usage, and back after each write through it.
- `metal2vulkan-degrade-heap-two-plane-texture` (precision). `HeapTwoPlaneTexture`: A texture of
  `RGB10A8_2P_XR10` or its sRGB twin made in a heap whose planes no shadow of the heap's bytes
  holds: one whose moves pass the device's limits. Its one image holds each texel's colour
  word and alpha byte side by side, more than Metal's two planes take in the heap, so it lies in
  memory of its own. Cost: the texture's size and alignment answers are Metal's and its reads,
  writes and copies as outside a heap, but its texels lie in memory of its own, apart from the
  heap's bytes: a buffer or texture over its range of the heap neither sees nor changes them, where
  Metal's texture keeps its two planes there.
- `metal2vulkan-degrade-heap-two-plane-layout` (precision). `HeapTwoPlaneLayout`: A texture of
  `RGB10A8_2P_XR10` or its sRGB twin made in a heap whose first level's planes lie in the heap's
  bytes, of more than one level, or a 1D array: Metal's paths address the texture's levels past the
  first at bytes that differ by path (its CPU copies at its first level's rows from past that level,
  its blits and an array's GPU writes over the first level's bytes, a 2D texture's GPU writes
  nowhere, its GPU reads at other bytes again), and a 1D array's GPU reads every texel at the
  texture's first word. The runtime keeps the first level's slices in the heap's bytes, as Metal's
  CPU copies and GPU writes lay them, and the other levels in the texture's image alone. Cost: the
  first level's texels lie in the heap's bytes as Metal's CPU copies and GPU writes lay them, and
  read as written; the later levels' texels lie apart from the heap's bytes, and a 1D array's GPU
  reads read each texel, where Metal's paths address those texels at bytes that differ by path.
- `metal2vulkan-degrade-moltenvk-argument-buffers-unread` (precision).
  `MoltenVkArgumentBuffersUnread`: A MoltenVK device at a release whose `pipelineCacheUUID` layout
  the runtime has not read: whether MoltenVK binds through Metal argument buffers is unknown, so a
  device without them is not refused (as at a read release). Cost: whether MoltenVK binds through
  Metal argument buffers is not read at this release: with MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS
  off, Metal's compiler can crash on large fragment functions' pipelines, and argument buffers hold
  fewer resources than Metal's.
- `metal2vulkan-degrade-two-plane-texture-copy` (precision). `TwoPlaneTextureCopy`: A blit between
  two textures of `RGB10A8_2P_XR10` or its sRGB twin whose answer is not Metal's measured one: one
  the runtime copies texel for texel (textures of more than one level or slice, a copy within one
  texture, a level larger than a storage buffer), or one whose rows overlap in the destination, or
  of a texture narrower than 4 texels, where Metal's blit writes in an order that varies or reads
  bytes from outside the planes, or one reading bytes outside the source's texels that an earlier
  blit wrote there (or a heap's texture's), which Metal's blit keeps and reads back. Cost: where
  Metal's answer was measured (2D textures of one level and one slice) the copy moves both planes'
  bytes as Metal's blit does, at its own rows; here it copies both words of each texel, where
  Metal's blit moves bytes at those rows and so loses or moves texels, or it keeps the later row
  where the copy's rows overlap (Metal's blit writes them in an order that varies), or it reads as 0
  the bytes Metal's blit reads from outside the texels: from outside the planes below a width of 4,
  and those an earlier blit left in a row's padding or past a plane, which Metal keeps.
- `metal2vulkan-degrade-extended-range-bits-kept` (precision). `ExtendedRangeBitsKept`: A draw in a
  render pass that loads a color attachment of an extended-range, padded 10-bit or bfloat format
  (`BGR10_XR`, `BGRA10_XR`, `RGB10A8_2P_XR10`, `R10Unorm_X6`, `RG10Unorm_X12` and their sRGB twins,
  `RGB10A2Unorm_sRGB`, `R16Bfloat`, `RG16Bfloat` and `RGBA16Bfloat`) and does not discard it at the
  store, on every device: the device's pass keeps the bytes its draws do not write, where Metal's
  pass writes each 32 x 32 tile a draw reaches back through its tile store. Also, once per device, a
  GPU blit into, out of or between textures of which one is a `BGRA10_XR` or `BGRA10_XR_sRGB`
  texture Metal keeps losslessly compressed: Metal's blit moves the texels through the compression,
  which clears the same bits and clamps the same alpha, where the device's copy keeps the bytes.
  Cost: the texels the pass's draws do not write keep their bytes; Metal's pass rewrites, in each
  tile a draw reaches, the bits their codes leave unused (BGR10_XR's bits 30 and 31 to ones,
  BGRA10_XR's low six bits of each field, the two-plane color word's top two bits and a padded
  word's low six bits to zeros), clamps BGRA10_XR's alpha code to [384, 894], moves a padded sRGB
  twin's code through its table and a bfloat code through half (out of half's range to an infinity
  or zero, a NaN to 0x7fc0): a texel whose bytes no Metal write left (uploaded bytes, an alpha code
  past 894, a bfloat code off half's grid) keeps bytes Metal's pass would change, and a blit
  through a BGRA10_XR texture Metal compresses keeps the bytes Metal's blit rewrites so; an
  extended-range texel reads the same value either way, a bfloat code off half's grid reads its own
  value where Metal's reads the half it was moved to.
- `metal2vulkan-degrade-snorm-minimum-kept` (precision). `SnormMinimumKept`: A draw in a render pass
  that loads a single-sampled color attachment of a signed normalized format (`R8Snorm`, `RG8Snorm`,
  `RGBA8Snorm`, `R16Snorm`, `RG16Snorm`, `RGBA16Snorm`) and does not discard it at the store. Cost:
  the texels the pass's draws do not write keep their bytes; Metal's pass rewrites a channel's
  minimum code (-128, or -32768 for 16 bits) as the code above it (-127, -32767) in each tile a draw
  reaches: a texel whose bytes no Metal write left (uploaded or copied bytes) keeps a code Metal's
  pass would change. Not on a driver whose pass stores the codes as Metal's
  (`DriverFact::TileStoreIsMetals`: MoltenVK on Apple's GPUs, whose pass is a Metal pass).
- `metal2vulkan-degrade-shared-exponent-kept` (precision). `SharedExponentKept`: A draw in a render
  pass that loads a single-sampled `RGB9E5Float` color attachment and does not discard it at the
  store; and, once per device, a GPU blit into, out of or between textures of which one is an
  `RGB9E5Float` texture Metal keeps losslessly compressed, whose compression normalizes every such
  word, where the device's copy keeps the bytes. Cost: the texels the pass's draws do not write keep
  their words; Metal's pass normalizes a word whose largest mantissa is below 256 (its exponent
  above 0) in each tile a draw reaches, and Metal's blit through an RGB9E5Float texture it
  compresses every such word: a texel whose bytes no Metal write left (uploaded or copied words)
  keeps a word Metal would change, of the same value. Not on a driver whose pass stores the words
  as Metal's (`DriverFact::TileStoreIsMetals`), nor, for a blit, where the device's copy of the
  compressed image is Metal's blit (MoltenVK's).
- `metal2vulkan-degrade-two-plane-buffer-blit` (precision). `TwoPlaneBufferBlit`: A blit between a
  buffer and a level of `RGB10A8_2P_XR10` or its sRGB twin whose width is not a multiple of 4,
  where Metal's blit addresses the level's rows `⌊4 w / 16⌋ · 16` bytes apart, short of where they
  lie, and so moves texels into and out of other rows and the bytes between them. A 2D texture or
  array of one level is blitted at Metal's rows, the row whose bytes Metal's GPU leaves where two
  meet standing, the bytes between its rows that its fills and blits with a buffer write held
  apart; announced there are a blit into a texture 32 texels wide or wider whose rows meet (narrower,
  `two-plane-buffer-blit-order`), and a blit out of a texture of the device's memory that reads bytes
  between its rows after a copy from another texture wrote some there. Any other such texture (more
  levels, a view, 1D, 3D, a surface's) moves the texels at the rows the texture holds them in. Cost:
  where rows meet in a blit into a texture 32 texels wide or wider the earlier row's bytes stand,
  where Metal's GPU writes them in an order that varies from run to run, at some sizes never all the
  earlier row's; a blit out of a texture of the device's memory reads the bytes
  between its rows that a copy from another texture wrote as its fills and blits with a buffer
  left them, where Metal's reads the copy's back; and a texture of more than one
  level, a view, a 1D or 3D texture or a surface's moves its texels at the rows it holds them in,
  where Metal's blit addresses its rows floor(4w/16)*16 bytes apart.
- `metal2vulkan-degrade-two-plane-buffer-blit-order` (precision). `TwoPlaneBufferBlitOrder`: A blit
  from a buffer into a level of `RGB10A8_2P_XR10` or its sRGB twin narrower than 32 texels, at a
  width not a multiple of 4, whose rows meet between Metal's groups of 32 rows of the region:
  Metal's GPU writes the two rows there in an order that varies from run to run (the earlier row's
  bytes at 556 of 560 such meetings in our probes), and in its fixed order at every other meeting.
  The runtime keeps the earlier row's bytes there and Metal's order elsewhere: Metal's most frequent
  whole outcome. Cost: where the rows of two of Metal's groups of 32 meet the earlier row's bytes
  stand, where Metal's GPU leaves either row's from run to run.
- `metal2vulkan-degrade-unadvertised-vertex-simdgroups` (validity). `UnadvertisedVertexSimdgroups`:
  A render pipeline state whose vertex function runs simdgroup functions on a driver measured to run
  a vertex function's subgroup operations as Metal does though its `subgroupSupportedStages` leaves
  the vertex stage out (`metal2vulkan::error::DriverDeviation::MetalVertexSimdgroups`): its pipeline
  requires `VkProperty::MetalVertexSimdgroups` in place of the vertex stage's subgroups. Cost: the
  device runs them as Metal does, but its driver leaves the vertex stage out of
  subgroupSupportedStages, so Vulkan's validation layer reports each shader module of the function
  (VUID-RuntimeSpirv-None-06343).
- `metal2vulkan-degrade-set-past-layout-support` (validity). `SetPastLayoutSupport`: A descriptor
  set layout the device's driver refuses through `vkGetDescriptorSetLayoutSupport` though it makes
  and runs it, a heap's set 1 past `maxPerSetDescriptors`
  (`metal2vulkan::error::RuntimeDriverDefect::SetLayoutSupportUnderstated`, MoltenVK): the runtime
  makes it there as the driver runs it. Cost: the device runs the set as Metal runs the argument
  buffer it is encoded in, but its driver's vkGetDescriptorSetLayoutSupport refuses the layout, so
  Vulkan's validation layer reports each such layout
  (VUID-vkCreateDescriptorSetLayout-support-09582).
- `metal2vulkan-degrade-released-command-state` (precision). `ReleasedCommandState`: An indirect
  command naming the ID of a pipeline state the program released is executed where the runtime
  cannot run what Metal runs. A record the GPU wrote runs the state that held the ID as the GPU
  last could write the buffer's records, released since or not, as Metal's does: a buffer the GPU
  may write keeps the programs of the states released after a command buffer that may write it
  was first encoded. Cost: a command whose record the GPU wrote naming a state released before its
  buffer could be written runs nothing, where Metal runs that state's program; one naming the ID
  of the state set on it runs its program, as Metal runs a command the CPU encoded, where for a
  record the GPU wrote while another state held the ID Metal runs that state (nothing, for a
  state of the other kind).
- `metal2vulkan-degrade-ycbcr-filter-not-exact` (precision). `YCbCrFilterNotExact`: A two-plane
  YCbCr texture whose planes' format the device converts itself (its sampler Y′CbCr conversion:
  unpacked 8- or 10-bit codes of a 4:2:0, 4:2:2 or 4:4:4 layout over plane 0, a 4:4:4 one read as
  codes aside, on a device that reports the format's conversion features, Metal's midpoint chroma
  through a linear chroma filter among them for a 4:2:0 or 4:2:2 one, and has no defect in it:
  MoltenVK's 4:2:0 and 4:2:2 formats and lavapipe's and RADV's 10-bit ones aside) reads the device's conversion
  of its planes, announced where the texture is made; so do the samples of a texture whose planes
  are read in place through the device's conversion of a linear image of them (where the device's
  linear tiling of the 4:2:0 format has Metal's reconstruction), the device's filter of the codes
  too, announced where the image is made; and a pipeline stage's filtered sample of a
  texture slot bound to a two-plane YCbCr texture, through Metal's own sampler (MoltenVK), does not
  take Metal's exact filter of the format: the stage records `PrecisionNote::HalfSampleFilter` or
  `FloatSampleFilter` under the slot's bound format. Cost: where the device converts the planes'
  format, the texels are its own Y′CbCr conversion's, at its precision and unclamped where it does
  not clamp, not Metal's fixed-point read; elsewhere the texels hold Metal's reads exactly, but the
  sample filters them as the device does, off the grid Metal sums the format's steps on, so a
  filtered result can miss Metal's by a step of that grid or more. Measured on lavapipe (Mesa
  26.0.8; random planes, every texel), which converts the 8-bit 4:4:4 layout here: each channel
  within 0.04/2048 of Vulkan's conversion past the `half` its image stores, and of Metal's read
  within 6/2048 where the result lies in `[0, 1]`; outside it by what Metal's clamp takes off. A
  device that samples natively answers the device's filter, `metal2vulkan-degrade-device-sampling`;
  on MoltenVK, Metal's own sampler filters the converted texels as an `RGBA16Float` or `RGBA32Float`
  texture's, within one step of the grid (a `half` result within one `half` ulp, measured in plain
  Metal on an Apple M3 Max under macOS 27.0.1 for `YCBCR8_420_2P` and its sRGB twin against such a
  texture of the same texels), announced at a draw that binds such a texture.
- `metal2vulkan-degrade-rotated-load-tiles` (precision). `RotatedLoadTiles`: A render pass that
  draws loads and stores a rotated texture of rotation 4, whose load is the plane transposed and
  whose store is rotation 1's: every tile of the frame lands mirrored, where Metal loads and stores
  only the 32x32 tiles a draw's primitives reach and keeps the texels of the others in place. Cost:
  the tiles no draw reaches land mirrored as the tiles it reaches do, where Metal keeps their texels
  in place.
- `metal2vulkan-degrade-ycbcr-load-tiles` (precision). `YCbCrLoadTiles`: A render pass that
  draws loads and stores a two-plane YCbCr texture (`YCBCR8_420_2P` and its kin over plane 0 of a
  surface): the pass stores every 32x32 tile of the rectangle bounding its draws' scissors, each
  texel's codes re-encoded from the texel it loaded (its chroma the filtered pairs), where Metal
  stores only the tiles a draw's primitives reach and keeps the planes' bytes of the others. Cost:
  in the tiles no draw reaches the planes' codes are re-encoded from the texels the pass loaded,
  the chroma from its filtered pairs, where Metal keeps their bytes.
- `metal2vulkan-degrade-ycbcr-blend-rounding` (precision). `YCbCrBlendRounding`: A draw that
  blends into a color attachment of a two-plane YCbCr texture: the device blends the pass's floats,
  where Metal's tile blends halves, an 8-bit format's in half arithmetic (the source and its alpha
  rounded to `half`, each product, and a fused sum), a 10-bit one's in `float` rounded toward zero
  to `half`. Cost: a blended texel's codes can miss Metal's by one, where Metal's blend in half
  arithmetic rounds otherwise than the device's in float, or over a texel an earlier draw of the
  pass wrote (Metal holds it as a half).
- `metal2vulkan-degrade-encoded-use-rebound` (precision). `EncodedUseRebound`: A command buffer's
  command named a buffer over the caller's host pages (a binding, a blit's operand, an attachment, a
  texture over the pages, an indirect command's resolved buffer) that was rebound
  (`rebind_bytes_no_copy`) after the command was encoded and before the command buffer was first
  submitted, in an encoder the rebind could not encode again. Cost: each of Metal's submissions of
  the command buffer holding such a command runs nothing and the command buffer fails
  (EncodedUseRebound), where Metal resolves the use as it submits the command buffer and reaches the
  pages the buffer lies in then; the rebind encodes compute, blit and render encoders again, but not
  this one. The reason names where the first such submission found the use (the encoder; its own
  commands or work it left for later) and why no rebind encoded or recorded it again.
- `metal2vulkan-degrade-stale-detached-reach` (precision). `StaleDetachedReach`: A command buffer
  that bound a heap set, whose commands may reach any texture through its resource ID without naming
  it, was submitted while a texture's bytes (a no-copy buffer's or heap's, the texture over them or
  in it) were detached (`detach_backing`) after a submitted command buffer made them resident. Cost:
  a read of the texture that does not name it reads zero bytes (the stand-in its detached bytes
  present), where Metal's reads the pages the program gave back, which a correct program never
  relies on.
- `metal2vulkan-degrade-detached-storage-keeps-writes` (precision). `DetachedStorageKeepsWrites`: A
  command buffer that bound a heap set, whose commands may reach any texture through its resource ID
  without naming it, was committed while a texture of storage usage was over detached bytes
  (`detach_backing`; made before the detach or while detached): its stand-in has a sink its
  storage-class elements name. Cost: a read of the texture through a storage-class element (read or
  read_write access) that does not name it reads what an earlier such write left in the stand-in's
  sink, where Metal drops those writes and reads zero; a sampled read, and a size query, are
  Metal's.
- `metal2vulkan-degrade-zero-layer-map-after-release` (precision). `ZeroLayerMapAfterRelease`: A
  draw in a render pass over a rasterization-rate map of no layers that was made after the device
  released a map of one or more layers, or reset from one of layers. Cost: the draws write nothing,
  where Metal draws rows of the pass (a 64-pixel map's rows 0 to 15, at the positions of a pass
  with no map).
- `metal2vulkan-degrade-rate-map-reset-unencoded` (precision). `RateMapResetUnencoded`: A render
  pass over a rasterization-rate map reset (`RasterizationRateMap::reset_using_descriptor`) after
  the pass took it and before its command buffer was submitted, in an encoder the runtime could not
  encode again (it kept no log: a render encoder opened while a buffer of the device was detached,
  or one that left work for a host step). Announced at
  the commit, or at the reset where the command buffer was committed. Cost: the pass draws by the
  map as it took it, where Metal's GPU draws by the map as reset.
- `metal2vulkan-degrade-heap-texture-lossless-layout` (precision). `HeapTextureLosslessLayout`: A
  texture made in a heap over host pages (`new_heap_with_bytes_no_copy`), or a depth or stencil
  texture made in any heap as a shadow of Metal's tiles (on a driver that lays such images apart from
  their memory or keeps them GPU-optimized: MoltenVK), that Metal keeps losslessly compressed: its
  bytes on Metal are no layout of its texels, so the runtime lays them out as its own, its
  uncompressed tiles where it tiles them. A multisample texture so (its sample grid's shorter side
  16 or more) holds its samples in its uncompressed tiles, or, where the device does not move its
  samples to and from a buffer, lies in memory of its own, the heap's bytes keeping what they held.
  Also such a texture whose image lies in a heap over the device's memory: the image, of the
  runtime's usage, need not be compressed as Metal's. Cost: the texture's reads, writes and copies
  are Metal's, but its bytes in the heap are not Metal's compressed bytes (its texels or samples in
  the runtime's layout, or a multisample texture's image in memory of its own, the heap's bytes
  keeping what they held): a buffer or texture over them reads and writes other bytes than Metal's
  would.
- `metal2vulkan-degrade-heap-texture-block-compressed-layout` (precision).
  `HeapTextureBlockCompressedLayout`: A texture of a block-compressed format made in a heap over
  host pages whose blocks' placement is not measured: Metal places a format family's blocks in each
  level by a rule that differs by family and by release, measured at the release every device
  answers as (`MACOS_RELEASE`) for the BC, ETC2, EAC, ASTC and PVRTC families (an Apple M3 Max, and
  an Apple M4 alike but PVRTC), whose textures lie as Metal lays them out; a format of none of them
  (ASTC's 4x2 and 8x4 footprints) lies in the runtime's layout. Cost: the texture's
  reads, writes and copies are Metal's, but its blocks lie in the heap's bytes in the runtime's
  layout, where Metal tiles them: a buffer or texture over them meets the blocks at other bytes than
  on Metal.
- `metal2vulkan-degrade-heap-texture-untiled-layout` (precision). `HeapTextureUntiledLayout`: A
  texture made in a heap over host pages that Metal lays out tiled where the runtime does not: a
  layout not measured (a format whose layout in Metal's heaps is not probed), an image that holds
  other bytes than Metal's elements, moves past the device's limits, or a multisample texture whose
  samples the device does not move to and from a buffer (a 16-bit depth where its multisampled
  copies go by draws whose codes the driver is not measured to keep), in memory of its own. Also a
  texture made in any other heap (over the device's memory) whose bytes there are not Metal's
  layout: its image bound
  there in the driver's layout, on a driver whose colour images are not measured to lie as Metal's
  heap textures (`DriverFact::ColorImagesLieAsMetalHeapTextures`, and
  `DriverFact::MultisampleColorImagesLieAsMetalHeapTextures` for a multisample one) or of a depth or
  stencil texture there; its image a shadow of the bytes in rows; or a layout not measured.
  Cost: the texture's reads, writes and copies are Metal's, but its texels lie in the heap's bytes
  in the runtime's layout, where Metal tiles them: a buffer or texture over them meets the texels at
  other bytes than on Metal.
- `metal2vulkan-degrade-shader-side-depth-test` (performance). `ShaderSideDepthTest`: A render
  pass's draws test the depth in the fragment shader: a fragment function that stores a color
  through the pixel interlock before Vulkan's tests, in a draw whose depth test may drop the
  fragment, tests the fragment's depth itself against a stand-in of the depth, so it stores only
  what passes, as Metal stores only after the tests. Cost: a 32-bit float stand-in image of the
  depth per pass, which the depth is copied into before each run of such draws and back from after
  it (through a staging buffer at one sample, image to image or by a compute split and a
  fragment-depth draw at more), each run in a render pass instance of its own under depth `ALWAYS`
  without writes, its fragment shader testing and writing the stand-in in the pixel interlock; where
  the stencil test acts too, an 8-bit stand-in of the stencil per pass, copied in and back the same
  way (image to image at more than one sample) under the stencil test off.
- `metal2vulkan-degrade-overlapping-texture-copy-as-memmove` (precision).
  `OverlappingTextureCopyAsMemmove`: A blit copy between overlapping regions of one texture's
  texels, other than a region onto itself: Metal's copy reads as `memmove`'s for regions up to a few
  hundred texels across, but past that some texels read what another part of the copy already wrote,
  differently from run to run. Cost: the destination holds the source's texels as they were before
  the copy, memmove's answer, which is Metal's for small regions, where Metal's copy of a larger
  region races and differs from run to run; the source goes through a scratch buffer, two copies
  instead of one.
- `metal2vulkan-degrade-extended-range-blends-stored-codes` (precision).
  `ExtendedRangeBlendsStoredCodes`: A draw whose render pipeline state blends into, or reads (a
  framebuffer fetch, or the interlocked storage form an extended-range blend takes), a color
  attachment of an extended-range format (`BGR10_XR`, `BGRA10_XR`, `RGB10A8_2P_XR10` and their sRGB
  twins) that an earlier draw or execution of render commands of the same render pass may have
  written; an execution of render commands in a pass with such an attachment, whose commands may
  blend onto each other. Cost: a blend into, or a read of, an extended-range color an earlier draw
  of the pass wrote sees the codes the runtime stored at that draw (each channel rounded to its
  10-bit code, the two-plane format's alpha to its byte), where Metal's tile memory holds the color
  as half through the pass and rounds it once at the store: a blended or read value can differ from
  Metal's by a code, and the difference can grow over many draws that blend onto each other.
- `metal2vulkan-degrade-unmeasured-heap-texture-size` (precision). `UnmeasuredHeapTextureSize`: A
  heap texture sized or made (`heap_texture_size_and_align`, `Heap::new_texture`) whose size Metal's
  layout the runtime does not model: a format or a sample count whose layout in Metal's heaps is not
  measured (8 samples); a mipmapped texture Metal compresses and a `Depth32Float_Stencil8` one are
  modelled. Cost: heap_texture_size_and_align answers the
  device's own image's size and alignment, which may be larger or smaller than Metal's, so a heap
  sized by Metal's answers may hold more or fewer such textures and they lie at other offsets.
- `metal2vulkan-degrade-plain-state-unique-identifier` (precision). `PlainStateUniqueIdentifier`: A
  program's call answered the `uniqueIdentifier` of a pipeline state made without
  `supportIndirectCommandBuffers` (`ComputePipelineState::unique_identifier`,
  `RenderPipelineState::unique_identifier`, a tile state's too). Cost: the identifier is one value
  of the runtime's count, unique and fixed for the state's life, where Metal's is an address that
  differs from run to run.
- `metal2vulkan-degrade-unmeasured-unique-identifier` (precision). `UnmeasuredUniqueIdentifier`: A
  program's call answered a `uniqueIdentifier` Metal answers alike in every run of one program
  started alike, which the runtime's is not measured to be: a pipeline state's made with
  `supportIndirectCommandBuffers` (`ComputePipelineState::unique_identifier`,
  `RenderPipelineState::unique_identifier`, an indirect command's state identifier other than 0), or
  an indirect command buffer's made after a command buffer of its device that encoded a render
  encoder was scheduled, or after a resource state encoder opened
  (`IndirectCommandBuffer::unique_identifier`). Cost: the identifier is unique and fixed for its
  holder's life, and a released one is reused as Metal reuses it, but it is not the number Metal
  answers there, which follows how the process was started, what it made and Metal's own encoders'
  allocations.
- `metal2vulkan-degrade-cross-class-texture-neutral` (precision). `CrossClassTextureNeutral`: A draw
  or dispatch that binds, at a texture slot, a texture whose format's numeric class (float, signed
  or unsigned integer) is not the class the function declares there and that its plan has no exact
  answer for (`PlanDiagnostic::CrossClassNeutral`): a depth, stencil, extended-range or YCbCr
  texture, a depth slot, an atomic or `ulong` access, a written storage slot over a swizzled
  texture, or an integer write to an `RGB9E5Float` texture. Cost: the function reads the slot's
  neutral texture (zero texels; zero size), where plain Metal reads the texture in its format's own
  class and hands back its words as the declared type.
- `metal2vulkan-degrade-indirect-nil-buffer-offset` (precision). `IndirectNilBufferOffset`: An
  execution of an indirect command buffer replayed on the host ran a command that holds a nil buffer
  set at an offset whose low 32 bits are not 0. Cost: the function sees no buffer, a null address,
  where Metal's sees the nil buffer's offset as its address; neither holds a buffer, so only the
  address's value differs.
- `metal2vulkan-degrade-compressed-texel-decode` (precision). `CompressedTexelDecode`: A texture of
  a BC format (`BC1_RGBA` to `BC7_RGBAUnorm_sRGB`) made on a device whose driver is not measured to
  decode its texels as Metal does (BC1 to BC3 `DriverFact::ColorBlockDecodeIsMetals`: MoltenVK on
  Apple's GPUs, RADV on the 780M; BC4 and BC5 `DriverFact::ChannelBlockDecodeIsMetals`: MoltenVK on
  Apple's GPUs; BC6H `DriverFact::Bc6hBlockDecodeIsMetals` and BC7
  `DriverFact::Bc7BlockDecodeIsMetals`: MoltenVK on Apple's GPUs, NVIDIA's on the RTX 3090 and RTX
  5080 Laptop GPU, RADV on the 780M, ANV on the Iris Xe of Raptor Lake-P, and lavapipe's BC6H). The
  texture's image is its own BC image, which the device samples through its own decode: no BC
  texture keeps a decoded twin. The bounds are the largest differences from Metal's texels measured
  on random blocks of every format; a driver not named is unmeasured, and validation derives no
  tolerance from them.
  Cost: a read, sample or gather of its texels answers the device's decode of its blocks (an sRGB
  form's code through the device's sRGB decode), where Metal interpolates with 6-bit weights: a BC1
  to BC3 channel up to 3/255 off Metal's on NVIDIA (615.71.09, the RTX 3090 and RTX 5080 Laptop
  GPU; BC3's alpha 6/255) and lavapipe, 1/255 on ANV; a BC4 or BC5 channel up to 7.3/255 (signed
  0.064) on NVIDIA, 1.3/255 (signed 0.013) on ANV, 2.4/255 (signed 0.022) on lavapipe, and on RADV
  (Mesa 26.2.4, the Radeon 780M) a signed texel with an endpoint of -128 up to 1.7/255; a BC7
  channel a `float` ulp on lavapipe; its blocks' bytes, copies and sizes are Metal's.
- `metal2vulkan-degrade-cpu-block-decode` (performance). `CpuBlockDecode`: A CPU copy into a texture
  with a decoded twin (`replace_region`) whose CPU copies go on the host or the staging queue: its
  region is decoded on the CPU, on the device's decode thread, and copied into the twin; the first
  command buffer that may read the texture waits at its submission for that, and the call itself
  where a command buffer that may read it is submitted and not yet complete (which then reads the
  copied texels, as Metal's does) or the texture is untracked. Cost: the written blocks decoded on
  the CPU, on the device's decode thread split over up to 8 threads, and their texels copied into
  the decoded twin on the host or the staging queue: the first command buffer that may read the
  texture waits at its submission for that, and the call waits for it where a command buffer that
  may read the texture is submitted and not yet complete, or the texture is untracked.
- `metal2vulkan-degrade-carried-decoded-twin` (performance). `CarriedDecodedTwin`: A texture of an ASTC
  LDR format (`ASTC_4x4_LDR` to `ASTC_12x12_LDR`, the sRGB forms too) made on a device without that
  ASTC format, of a PVRTC format on a device without it (every device but MoltenVK's, and there
  the four RGB formats), or of an ETC2 or EAC format on a device without them (desktop GPUs and
  lavapipe): its blocks lie in an image of a carrier format whose blocks are also 16 (ASTC,
  `EAC_RGBA8`, `RG11`) or 8 (PVRTC, the other ETC2 and EAC formats) bytes, which copies and
  `get_bytes` move unchanged, and its samples read a decoded twin, Metal's decode of the blocks in
  codes: an ASTC LDR texture's and an EAC 11-bit texture's each nearest Metal's value
  (`decoded-texel-precision`); a PVRTC texture's, and an 8-bit ETC2 or EAC texture's, 8-bit codes,
  exact (its sRGB form's slots decode them through Metal's table, and one twin serves the views of
  the linear and sRGB forms); an ASTC sRGB texture's, its own 8-bit codes, read the same way.
  It is decoded after each write into it (a PVRTC texture's whole levels, since each texel's
  colors come from the blocks around it). An ASTC texture's first view in its other colour
  space (its sRGB form as LDR, or the other way) gives it a second twin of that form's decode, kept
  in step by GPU passes its command buffers' submissions record. Also a texture of a 4:2:2 format
  (`GBGR422`, `BGRG422`) on every device, whose Vulkan formats are viewed only through a sampler
  Y′CbCr conversion: its texels' bytes lie in an `R8G8_UNORM` carrier, and its twin holds its
  texels as 8-bit codes (each texel's own green, its red and blue interpolated with the pair beside
  it), decoded in whole rows: the device's own conversion of its pairs where it converts the format
  with Metal's reconstruction (`decoded-texel-precision`; every decode then a GPU pass, its CPU
  copies through submissions), else Metal's texels, exact. Cost: a carrier image of its
  blocks, at most about 1 byte a texel (a 4:2:2 texture's 2), and a decoded twin of 4 bytes a texel
  (R8G8B8A8; an EAC 11-bit texture's 16-bit codes, 2 bytes a channel, or its floats, 4 bytes a
  channel, where the device does not take those codes), two for an ASTC texture viewed in both
  colour spaces, and a decode after each write into it: a pass on
  the GPU for a blit, a copy from a texture or a CPU copy through a submission; on the CPU, on the
  device's decode thread, where its CPU copies go on the host or the staging queue.
- `metal2vulkan-degrade-decoded-texel-precision` (precision). `DecodedTexelPrecision`: A texture
  with an ASTC LDR decoded twin of 8-bit codes, or an EAC 11-bit one (`R11`, `RG11`) of 16-bit
  normalized codes; or a 4:2:2 texture (`GBGR422`, `BGRG422`) whose twin holds the device's own
  sampler Y′CbCr conversion of its pairs, where the device converts `G8B8G8R8_422_UNORM` or
  `B8G8R8G8_422_UNORM` with Metal's reconstruction (the chroma at the midpoint through a linear
  filter: MoltenVK). Cost: a read, sample or gather of its
  texels answers the nearest 8-bit code to Metal's 12-bit value, at most 1/510 off it; an EAC 11-bit
  texture's, the nearest 16-bit normalized code to Metal's 11-bit value, at most half a 16-bit step
  off it; a 4:2:2 texture's, where the device converts its format, the nearest 8-bit code to the
  device's own Y′CbCr conversion of its pairs, not Metal's decode (Metal reads an EAC 11-bit
  channel as `c / 2047`, signed `c / 1023`; a 4:2:2 chroma channel as `(3 c + n + 2) >> 2` of its
  pair's code and the code beside, whose ties the device's nearest code takes to even). An ASTC
  sRGB texture's twin holds Metal's own 8-bit codes, which its slots read through the twin's sRGB
  view, the device's decode (`srgb-texel-decode`).
- `metal2vulkan-degrade-shared-present-queue` (performance). `SharedPresentQueue`: A command queue
  made for presenting (`Device::new_command_queue_for_presenting`) on a device that reserved no
  `VkQueue` for presenting: its queue family has no queue past the ones the round robin takes
  (MoltenVK's four families of one queue, RADV's and lavapipe's single graphics queue). Cost: it
  submits to a `VkQueue` other command queues share, so a present that blocks under its lock (FIFO
  on a window the compositor does not show, a wait for the queue's idle at a resize) holds their
  submissions until it returns, where Metal's present holds no other queue.
- `metal2vulkan-degrade-heap-texture-larger-than-metal` (precision). `HeapTextureLargerThanMetal`: A
  heap texture sized or made (`heap_texture_size_and_align`, `Heap::new_texture`) whose images on the
  device need more bytes or a coarser alignment than Metal's answer, which the runtime models, and
  which no shadow of the texture's bytes holds it to: not a multisample texture, whose image lies in
  memory of its own, nor one whose planes or Metal's tiles fit its bytes. The answer is then the larger of the two, since no image binds at Metal's offsets.
  Every other such texture answers Metal's numbers and announces nothing here. Not where
  `optimized-depth-stencil-texture` says why. Cost:
  heap_texture_size_and_align answers more bytes or a coarser alignment than Metal's, so a heap
  sized by Metal's answers holds fewer such textures, and they lie at other offsets; each texture
  keeps its own bytes.
- `metal2vulkan-degrade-linear-attachment-copies` (performance). `LinearAttachmentCopies`: A pass
  renders a color attachment whose texture's image is linear (a texture over a buffer's memory, a
  no-copy buffer's host pages included) through an image its texture keeps, where the device renders
  such an image only under `VK_NV_linear_color_attachment`'s rule, that every color attachment of
  the render pass instance be linear, and the pass breaks it: another of its color attachments is
  not linear, or it resolves a multisampled attachment into the texture. Cost: its texels are copied
  into an image the pass renders, and what the pass wrote is merged or copied back after it.
- `metal2vulkan-degrade-linear-attachment-clear-copies` (performance).
  `LinearAttachmentClearCopies`: A pass clears a color attachment whose texture's image is linear
  and which the device renders under `VK_NV_linear_color_attachment`, where the rule allows it in
  place: it clears an image its texture keeps and copies it back after, as clearing a linear image
  of host memory in place measured slower than that copy. Cost: the pass clears an image of the
  device's and copies it back after, as clearing the texture's linear image in place costs more.
- `metal2vulkan-degrade-dont-care-stand-in` (performance). `DontCareStandIn`: A pass renders a color
  attachment that it stores with `DontCare` (or resolves alone, `MultisampleResolve`) and whose
  texture's image lies over the buffer the texture lies in (a linear image over an ordinary buffer's
  memory) through an image its texture keeps, so that the buffer
  keeps its bytes, as Metal's does: rendered in place, the pass would write them. A pass that stores
  the attachment renders it in place, at no cost; so does one that hands it to a driver that stores
  `DONT_CARE` as Metal's own `DontCare` (`DriverFact::DontCareStoreIsMetals`: MoltenVK on the M3
  Max and the M4), as one render pass instance the driver runs unbroken, where Metal keeps the bytes
  itself. Cost: an image of the
  attachment's size, which its texture keeps for the queue, renders the pass; a `Load` copies the
  buffer's texels over the render area into it before the pass, and nothing is copied back.
- `metal2vulkan-degrade-host-deallocator-delayed` (precision). `HostDeallocatorDelayed`: A no-copy
  buffer or heap released with a deallocator while a command buffer that may still reach its pages
  holds them. Metal runs the deallocator at the release, even while a command buffer in flight
  reaches the buffer unnamed (through a loaded pointer, or an indirect command the GPU encoded with
  its address), and that command buffer's later accesses reach nothing. Vulkan frees no memory that
  submitted commands may still use (`VUID-vkFreeMemory-memory-00677`), so the pages stay: with no
  keep-alive (`KeepAlive`), while a command buffer submitted before a texture over them was released
  may still read that texture through its resource ID (its image holds the import); with no
  keep-alive, while a command buffer whose commands may reach the buffer unnamed holds a table of
  the live buffers that lists it (its first submission's and its host steps', until it completes;
  one a rebind encoded again before its first submission, until then); with no keep-alive on
  MoltenVK (`RuntimeDriverDefect::FreedAddressMemoryFailsPendingWork`), while the batches submitted
  before the import's memory went, and the program's own submissions through
  `CommandQueue::with_vk_queue` before it, are in flight, as MoltenVK fails pending work that reads
  device addresses where such memory is freed; with no keep-alive, a detached buffer's or heap's
  rebind or release while the pages its detach let go of are still so held (the deallocator its
  detach held back, which Metal runs at the replace or the release). With a keep-alive the
  deallocator runs at the release, as Metal's, and the import keeps the keep-alive, which keeps the
  pages mapped, until it is freed: not announced. Cost: the deallocator runs once every command
  buffer that may still reach the pages (through a texture over them, or a table of the live buffers
  it holds) lets go of them, or the batches submitted before the import's memory went complete,
  where Metal's runs at the release; the pages stay mapped until then.
- `metal2vulkan-degrade-read-only-pages-written` (precision). `ReadOnlyPagesWritten`: A no-copy
  buffer, heap or rebind was made over host pages the process maps read-only, from the caller's
  dma-buf source (`HostMemory::with_dma_buf`): on Linux, where drivers pin a pointer's imported
  pages for writing, such pages are imported only as dma-bufs of their pages, whose memory the GPU
  writes. Cost: the GPU's writes through it reach the pages, and the CPU then reads them there,
  where after Metal's GPU writes into pages the process maps read-only the CPU reads them as they
  were (an M3 Max under macOS 27.0.1, private and shared memory alike).
- `metal2vulkan-degrade-copied-host-pages` (precision). `CopiedHostPages`: A no-copy buffer, a
  no-copy heap or a buffer's rebind was made over a copy of the caller's host pages in memory of
  the device's own, which no import maps (a heap's textures shadows of its bytes):
  the caller's gate (`HostMemory::with_import_gate`) refused the pages' import, or the import the
  device and the gate admitted failed (out of memory, or another Vulkan error, such as a driver out
  of the mappings the host kernel allows). Pages the process cannot read are never copied: the
  buffer is refused, as their import is. A copy whose own allocation fails refuses the call
  (`OutOfMemory`). A heap's rebind and memory whose GPU writes Metal's private option bit 24 drops
  are never copied. Counted: `WorkCounts::copied_host_buffers_made`, of them
  `copied_host_buffers_refused` and `copied_host_buffers_degraded` (and `copied_heaps_made`,
  `copied_rebind_pages_made`). Cost: the GPU uses a copy of
  the bytes, into which each commit writes every byte the CPU changed in the pages since, and from
  which each batch's completion writes back every byte the GPU changed, where Metal's GPU reads and
  writes the pages themselves: a CPU write the GPU reads mid-command-buffer, or a GPU write the CPU
  reads before the batch completes (behind a shared event's value among them), is not seen, and
  each commit and completion compares every byte of every copy.
- `metal2vulkan-degrade-grouped-completion` (precision). `GroupedCompletion`: Batches of command
  buffers committed on one queue that the scheduler found ready together, or held while two or more
  earlier submissions of their queue ran, went to the driver in one `vkQueueSubmit`, only the last
  signalling its queue's timeline and their events (`CompletionPolicy::Grouped`, the default, on a
  driver whose queue merges a submission that signals nothing into the next one: Mesa's drivers):
  one kernel submission for up to four batches, where each kernel submission costs CPU time for
  every `VkDeviceMemory` of the device. Announced once per device, on the scheduler's thread. Cost:
  each command buffer of the submission but the last is seen complete (its status, completed
  handlers and waits), and the event values it signals are reached, once the last completed, late by
  the GPU time of the batches after it, at most three, never early and in commit order, and a
  submission held while earlier ones of its queue ran late besides by any gap between the last of
  those and its own start on the GPU, at most a completion turn's reaction and its vkQueueSubmit;
  and the submission checks of the batches behind a queue's next one run with its own, before the
  batch before them is submitted; their commands, barriers and GPU times are unchanged;
  `CompletionPolicy::Exact` submits and checks each batch alone.
- `metal2vulkan-degrade-draw-dropped-still-making` (precision). `DrawDroppedStillMaking`: A draw
  whose `VkPipeline` the driver was still making once its command buffer's bound passed
  (`PipelineWaitPolicy::Bounded`, the default: its commit plus
  `DevicePolicy::pipeline_commit_wait`, 4.5 s by default) ran nothing: its recording
  records the barriers and the visibility query around it and no draw, and the command buffer runs
  on without error. Metal makes a render state's pipelines inside `newRenderPipelineState`, so none
  of its draws waits for a compile; the runtime makes them on its pool, a state's predicted one
  among them (the creation waits for it `DevicePolicy::pipeline_creation_wait` at most), so a
  driver's cold compile (seconds on
  MoltenVK with a cold Metal compiler service) cannot hold the command buffer past a guest's
  watchdog. Announced where the draw is recorded: in the program's `commit`, or the command
  buffer's host step. Cost: the draw's commands ran nothing, where Metal's draw renders; the
  command buffers committed after the pipeline is made draw with it;
  `CommandBuffer::dropped_draw` names the first such draw with the error VK_TIMEOUT;
  `PipelineWaitPolicy::Exact` waits until the pipeline is made.
- `metal2vulkan-degrade-draw-dropped-unmade` (precision). `DrawDroppedUnmade`: A draw whose
  `VkPipeline` the driver failed to make after the draw returned (the runtime makes a draw's
  pipeline while the draw goes on, `render/pending.rs`) ran nothing: its recording records the
  barriers and the visibility query around it and no draw, and the command buffer runs on without
  error. Metal makes a render state's pipelines inside `newRenderPipelineState`, which fails there,
  so none of its draws meets such a failure. The notice names the draw, its state, the variant and
  the driver's error. Announced where the draw is recorded: in the program's `commit`, or the
  command buffer's host step. Cost: the draw's commands ran nothing, where Metal's draw renders;
  the draws after it that need the same pipeline make it again; `CommandBuffer::dropped_draw`
  names the first such draw with the driver's error, and `WorkCounts::draws_dropped_unmade` counts
  them.
- `metal2vulkan-degrade-dispatch-dropped-still-making` (precision). `DispatchDroppedStillMaking`: A
  compute dispatch, or a render pass's tile dispatch, whose `VkPipeline` the driver was still
  making once its command buffer's bound passed (`PipelineWaitPolicy::Bounded`, the default: its
  commit plus `DevicePolicy::pipeline_commit_wait`, 4.5 s by default) ran nothing: its recording
  records its barriers (a tile dispatch's copies into
  its scratch and back) and no dispatch, and the command buffer runs on without error. Metal makes
  a compute state's pipeline inside `newComputePipelineState`, so none of its dispatches waits for
  a compile; the runtime makes a dispatch's on its pool, a compute state's predicted one among them
  (the creation waits for it `DevicePolicy::pipeline_creation_wait` at most), so a driver's cold
  compile cannot hold the command
  buffer past a guest's watchdog. Announced where the dispatch is recorded: in the program's
  `commit`, or the command buffer's host step. Cost: the dispatch ran nothing and wrote nothing,
  where Metal's dispatch runs; the command buffers committed after the pipeline is made dispatch
  with it; `CommandBuffer::dropped_draw` names the first such dispatch with the error VK_TIMEOUT;
  `PipelineWaitPolicy::Exact` waits until the pipeline is made.
- `metal2vulkan-degrade-misplaced-notice` (performance). `MisplacedNotice`: In a build without debug
  assertions, a notice announced where the runtime's rule announces none (`DegradeNotice::misplaced`
  names its kind and `NoticePlace`): inside a pipeline pool's piece, or, for a kind announced at
  every occurrence, outside the program's call and outside a host step. Cost: the notice reached the
  message sink from the runtime's notice thread after the call that met it, not in the program's
  order on the thread that met it; Metal's answers are unchanged.
- `metal2vulkan-degrade-depth-denormals` (precision). `DepthDenormals`: A write that may store a
  denormal in the 32-bit float depth of a depth image (a pass's clear to one, a blit's copy into a
  `Depth32Float` or `Depth32Float_Stencil8` texture's depth, a `replace_region` whose bytes hold
  one), on a device whose driver stores one a write on the GPU gives it as a zero (no
  `DriverFact::DepthImageKeepsDenormals`: RADV, ANV; a host copy keeps it) or whose depth test
  compares one otherwise than Metal's (no `DriverFact::DepthTestKeepsDenormals`: every driver
  measured but MoltenVK on an Apple M4). Cost: a denormal depth may read, sample and copy out as a
  zero, or the depth test may compare it as zero (a positive one) or below zero (a negative one),
  where Metal keeps it and compares a positive one above zero and a negative one not below it.
- `metal2vulkan-degrade-signed-depth-test` (precision). `SignedDepthTest`: A depth test that may
  compare a negative value or a NaN stored in the 32-bit float depth of a render target as it is, or
  a pass with depth writes that may store such a value with its sign, where Metal's test compares
  the stored value's magnitude (a NaN above +inf) and its pass with depth writes stores every texel
  no fragment wrote without its sign. Announced at a write that may give a render target's depth
  such a value, or at the pass that loads a new texture's pattern (a NaN with the sign set) before a
  write covered it, on a device that keeps such a value in a depth image:
  `VK_EXT_depth_range_unrestricted` enabled, or `DriverFact::DepthImageKeepsValuesOutsideUnitRange`
  (elsewhere `restricted-depth-upload` is announced). MoltenVK's test is Metal's own
  (`DriverFact::DepthTestComparesMagnitudes`): there it is announced at a pass that loads and stores
  the depth and sets a state with depth writes but draws under none, whose pass keeps the signs.
  Also announced for a NaN clear depth the device does not write before the pass. Cost: a depth test
  against the stored value, or the value a pass with depth writes stores, can differ from Metal's in
  the pass or target named: the device compares a negative value below zero and a NaN by its own
  rule and keeps the value's sign, where Metal compares the value's magnitude, orders a NaN above
  every value and stores the value without its sign; or a NaN clear depth is stored and tested as
  1.0, where Metal stores `0x7fc00000`.
- `metal2vulkan-degrade-first-sparse-resource-wait` (performance). `FirstSparseResourceWait`: The
  device's first sparse texture made (`Heap::new_texture` in a sparse heap) on a device whose
  driver's first sparse resource waits for the work submitted before it on every queue
  (`DriverFact::FirstSparseResourceWaitsForQueues`: ANV through TR-TT, Mesa 26.2.4 on Intel's GPUs):
  `vkCreateImage` sets the driver's sparse translation tables up with a batch on each queue and
  waits for it. Cost: the call returns only once every queue finished the work submitted before it,
  once per device, where Metal's returns at once; what the texture holds and reads is Metal's.
- `metal2vulkan-degrade-driver-pipeline-cache-kept` (performance). `DriverPipelineCacheKept`: A
  device the runtime makes whose driver it cannot make with the driver's internal pipeline cache off
  (`VkDevicePipelineBinaryInternalCacheControlKHR`): its driver lists no `VK_KHR_pipeline_binary`,
  reports no `pipelineBinaryInternalCacheControl`, or the device enables no `VK_KHR_maintenance5`,
  which the extension requires. Cost: the driver keeps its own cache of the device's pipelines
  beside the shader cache's records, with whatever it writes to disk and holds in memory for it; the
  pipelines made are the same.
- `metal2vulkan-degrade-private-properties-unknown` (precision). `PrivatePropertiesUnknown`: Apple's
  private device properties the device has no answer for: no document describes them and nothing the
  device reports maps to them. Cost: the device gives no answer for them (`PrivateAnswer::value` is
  `None`); Metal's measured answers on Apple's GPUs ride beside, never as this device's.
- `metal2vulkan-degrade-unpacked-fill-color` (precision). `UnpackedFillColor`: A color fill
  (`BlitCommandEncoder::fill_texture_color`) whose color is packed in a format Metal packs no color
  in: compressed, 4:2:2, depth, stencil, two-plane, YCbCr, padded 10-bit, `Invalid` or a private
  format. Metal fills bytes of the calling thread's stack there, which hold the last color fill's
  packed texel only where no other call used them since. Cost: every unit of the region holds the
  command buffer's last packed texel over zeros, whatever the color, where Metal's bytes are its
  stack's.
- `metal2vulkan-degrade-texture-buffer-fill-leftover` (precision). `TextureBufferFillLeftover`: A
  fill of a texture buffer (`BlitCommandEncoder::fill_texture`,
  `BlitCommandEncoder::fill_texture_color`) whose 16-byte chunks read bytes of its block past the
  pattern's whole copies, or past the block: most patterns whose length is not a power of two, and
  most regions whose first byte is not at a multiple of 16 of the buffer. Metal reads bytes of the
  calling thread's stack there: zeros in our probes, an earlier fill's bytes in the consumer's
  process. The runtime writes zeros there. Cost: those bytes of each chunk are zeros, where Metal's
  are its stack's.
- `metal2vulkan-degrade-unsized-points` (precision). `UnsizedPoints`: A draw of points that
  rasterizes, of some vertices and instances, under a render pipeline state whose vertex function
  writes no `[[point_size]]` and whose input primitive topology class is not the point class: a
  direct draw, an indirect one, or an indirect command's in an execution. Cost: the points rasterize
  nothing, the vertex function run for each, where what Metal draws is undefined: most of its draws
  draw nothing, and some a few of the points or a box.
- `metal2vulkan-degrade-provoking-vertex-lines` (precision). `ProvokingVertexLines`: A draw of
  lines, or an emulated mesh draw of lines, under Apple's last provoking vertex
  (`setProvokingVertexMode:` 1), of a pipeline state that reads it (a flat input, or a slice or
  viewport its vertex function selects), on a device without `VK_EXT_provoking_vertex`'s last vertex
  (or whose pipelines of one render pass instance share one mode): each line is drawn from its last
  vertex to its first, so a first-vertex pipeline reads Metal's provoking vertex. Cost: each line's
  flat inputs are Metal's, but it can cover other pixels and interpolate its smooth inputs otherwise
  than Metal's line from its first vertex to its last.
- `metal2vulkan-degrade-provoking-vertex-unmatched` (precision). `ProvokingVertexUnmatched`: A draw
  under Apple's last provoking vertex (`setProvokingVertexMode:` 1) the runtime draws otherwise than
  Metal. On a device without `VK_EXT_provoking_vertex`'s last vertex (or whose pipelines of one
  render pass instance share one mode), what no list the runtime draws holds as Metal draws it: a
  draw of triangles, or an emulated mesh draw of triangles, whose fragment function reads flat
  inputs and `[[barycentric_coord]]`, each drawn from its last vertex, its second and third weights
  taken back to Metal's order from the drawn triangle's, Metal's but for rounding; patches, and mesh
  draws through `VK_EXT_mesh_shader`, whose pipeline state reads flat inputs: they read the first
  vertex. On a device with it, a strip whose vertex function computes `[[primitive_id]]` into its
  varying (a device without `geometryShader` and `tessellationShader`), whose IDs read each
  primitive's last vertex; and a slice or viewport a vertex function selects where the pipeline
  selects it in a synthesized geometry stage (a device without `shaderOutputLayer` or
  `shaderOutputViewportIndex`, an index that differs between a primitive's vertices, or weights the
  stage writes), from each primitive's first vertex. And on every device, a draw at a mode past 1,
  which Metal draws a third way and the runtime as mode 1, whose shape and pipeline state Metal draws
  otherwise than mode 1 (a strip's odd triangles, every line's flat inputs and `[[primitive_id]]` 0,
  sized points, the middle vertex of a triangle whose pipeline selects its slice or viewport, mesh
  draws unmeasured): announced at the draw. Cost: flat
  inputs or a computed primitive ID read another vertex than Metal's, or barycentric weights come in
  another order than Metal's or round otherwise.
- `metal2vulkan-degrade-visibility-count-per-color` (precision). `VisibilityCountPerColor`: A
  draw counted by a `Counting` visibility result whose fragment function writes more than one color
  attachment, on ANV (`RuntimeDriverDefect::OcclusionCountPerColorOutput`). Cost: the result is
  Metal's count multiplied by the colors the draw writes; a `Boolean` result is Metal's.
- `metal2vulkan-degrade-float-tininess-after-rounding` (precision). `FloatTininessAfterRounding`: A
  device whose float functions are not Metal's own (not MoltenVK): Metal flushes a `float` product,
  fused multiply-add or `ldexp` whose exact value lies below `2^-126` to the zero of its sign,
  deciding tininess before rounding, and the device rounds before it decides (measured on RADV,
  ANV, NVIDIA and lavapipe, whether they flush denormals or keep them); a `float` `ldexp`'s result
  below `2^-126` is the device's product. Announced once, as the device is made. Cost: a `float`
  product, fused multiply-add or `ldexp` whose exact value lies within half a unit below `2^-126`
  answers `2^-126` of its sign where Metal answers the zero of its sign.
- `metal2vulkan-degrade-shader-cache-entry-not-written` (performance).
  `ShaderCacheEntryNotWritten`: an entry of the device's shader cache directory was not written --
  the directory, a temporary file in it or the rename over the entry failed, on the device's cache
  thread or on the thread that handed the entry over past the queue's bound. A full or read-only
  directory, a cap no entry fits in, or a write another process won. Nothing is ended by it: the
  pipeline state, the translation and the compile are made regardless. Announced once per device,
  naming the directory, its cap and the largest entry it takes; later failures are counted in
  `ShaderCacheStatistics::write_failures`. Cost: what the entry would hold is made again by the
  next process that needs it; what this process computes is unchanged. The entry is a translation,
  a compiled pipeline, a creation's answers or the driver's pipeline cache data.
- `metal2vulkan-degrade-f16-fma-double-rounding` (precision). `PrecisionNote::F16FmaDoubleRounding`:
  Widened half fma rounds twice.
- `metal2vulkan-degrade-f16-int-conv-double-rounding` (precision).
  `PrecisionNote::F16IntConvDoubleRounding`: Widened half-integer conversions round twice.
- `metal2vulkan-degrade-grad-from-bias` (precision). `PrecisionNote::GradFromBias`: Bias sampling
  emulated with gradients may pick another level and another anisotropic footprint than Metal's:
  derivatives hoisted out of a per-fragment texture choice, which scale by `2^B` through every
  sampler, and, on a device that does not sample through Metal's own sampler, a bias whose sum with
  the sampler's own may lie beyond the device's `maxSamplerLodBias`, which Vulkan clamps, or that an
  anisotropic sampler applies to its footprint's minor axis alone. On a device without
  `maintenance8`, a fragment's implicit-level sample with a dynamic texel offset samples with its
  derivatives too, taken before the switch over the offset's values (`docs/FAST-PATHS.md`).
- `metal2vulkan-degrade-device-transcendentals` (precision). `PrecisionNote::DeviceTranscendentals`:
  Transcendentals answer within Apple's stated bound, not Metal's bits: the device's own functions
  where they are Metal's (MoltenVK), and the device's precision for a call no other note bounds.
- `metal2vulkan-degrade-reduction-order` (precision). `PrecisionNote::ReductionOrder`: Float
  reductions use the device's association order.
- `metal2vulkan-degrade-device-denormals` (precision). `PrecisionNote::DeviceDenormals`: `float`
  denormals follow the device: Metal reads an `f32` denormal operand of its float arithmetic,
  comparisons, and functions as the zero of its sign and flushes a denormal result to one, where a
  Vulkan device keeps or flushes them by its own float controls (this translation declares no
  `DenormFlushToZero`). Recorded where the module leaves an operation's denormals to a device not
  measured to read them as Metal does: `+ - *`, the comparisons, `fma`, `dot`, `sqrt`, `rsqrt`,
  `trunc`, `rint` and `ldexp` on `float` (by the device's products) but on a device that takes the
  measured
  `DriverDeviation::MetalFloatDenormals`; and on every device the divisions and the remainder,
  `floor`, `ceil` and `fneg`, the device's maths functions (`fast::fmod`'s fused multiply-add,
  `fabs`, `sign` and `fract` among them), `mix` by `FMix`, `fast::sqrt` by `Sqrt`, `dot` of
  vectors of other than two to four lanes, whose chunks' products the device sums where Metal
  flushes each one below `2^-126`, and a conversion to or from `double`, where Metal's reading is
  not measured. The operations that read a denormal on its bits answer Metal on every device:
  `copysign` and `atan2`; and so do the functions the device answers as Metal's own. Where the
  minimum and maximum
  functions and `saturate` are the device's `NMin`, `NMax` and `NClamp`, their pick of a denormal
  is `NativeMinMax`'s. Off MoltenVK the device's multiply and fused multiply-add, and `ldexp` on
  `float` by that multiply, round a result whose exact value is below `2^-126` up to `2^-126`, which
  Metal flushes (`float-tininess-after-rounding`).
- `metal2vulkan-degrade-pixel-sampler-scaling` (precision). `PrecisionNote::PixelSamplerScaling`:
  A pixel-coordinate sampler's coordinate reaches the device's sampler divided by the texture's
  level-0 size in the shader, on every device, and the device reads its texels from that coordinate
  in its own arithmetic: within a few units in the last place of a texel edge (a nearest sample's
  texel, a gather's block, a linear filter's pair) or of one of the device's sub-texel steps (a
  linear filter's weight) it may read the neighbouring texel, block or step, where Metal reads the
  coordinate as given. Measured: the device's normalized sampler's answer within one `f32` ulp of
  the quotient on MoltenVK and at the quotient on lavapipe; a linear downscale by 8 within 1/255 of
  Metal's on MoltenVK and 2/255 on lavapipe. A constexpr pixel sampler that filters nearest and
  clamps to the edge fetches its texel instead and records nothing; a runtime sampler slot's access
  announces it where the bound sampler states pixel coordinates.
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
  is held at `float` precision between the roundings where its values leave it
  (`HalfChainPolicy::AtExits`, the default; `HalfChainPolicy::EachOperation` rounds each).
  Cost: a sum of mixed signs grouped otherwise, or a multiply-add fused on one side alone, differs
  from Metal's by any amount relative to the result; a held `half` chain by the roundings it
  skips, a value inside it past `half`'s range staying finite.
- `metal2vulkan-degrade-tessellation-winding` (precision). `PrecisionNote::TessellationWinding`: A
  tessellation pipeline's vertex order: Metal's clockwise output winding is declared
  `VertexOrderCcw` and its counter-clockwise one `VertexOrderCw`, as MoltenVK maps them under
  Vulkan's default upper-left domain origin.
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
- `metal2vulkan-degrade-tessellation-triangle-order` (precision).
  `PrecisionNote::TessellationTriangleOrder`: A tessellation pipeline drawn through the device's
  tessellation stages (a device with `tessellationShader`) whose tessellator is not Apple's (every
  driver but MoltenVK on an Apple GPU, `DriverWorkaround::TessellatorNotMetals`): the device's
  tessellator orders each patch's triangles, picks each one's first corner, and joins the patch's
  rings where Vulkan leaves that to the implementation. The domain points are Metal's; a flat
  varying, a value interpolated across another triangulation, and the order a patch's triangles
  blend in may differ. A device without `tessellationShader` draws Metal's order through the
  emulation (`emulated-tessellation-draws`) and records none.
- `metal2vulkan-degrade-unmeasured-group-speculation` (precision).
  `PrecisionNote::UnmeasuredGroupSpeculation`: A function holds a branch whose arms hold a quad or
  simdgroup operation that answers over every lane of the branch's block when Metal's backend
  speculates the arms (a quad shuffle, broadcast or vote, a simdgroup shuffle, `simd_is_first`),
  where no measurement says whether Metal speculates them: an arm of an instruction, size or shape
  no probe covers. The branch keeps the AIR's lanes, so that operation may answer over the lanes
  that take the arm where Metal's answers over the block's. A branch Metal was measured to
  speculate is speculated, and one it was measured to keep is kept, with no note.
- `metal2vulkan-degrade-not-mipmapped-anisotropic-filter` (precision).
  `PrecisionNote::NotMipmappedAnisotropicFilter`: A sample at an implicit level or with gradients
  through a constexpr sampler with anisotropy, no mip filter and different minification and
  magnification filters, on a device that samples through Metal's own sampler (MoltenVK on Apple's
  GPU): no level clamp the device's sampler takes reads Metal's `NotMipmapped` footprint at
  `maxAnisotropy` 2 to 16, so the sample filters another anisotropic footprint than Metal's.
- `metal2vulkan-degrade-device-sample-level` (precision). `PrecisionNote::DeviceSampleLevel`: An
  extended-range texture's (`BGR10_XR`, `BGRA10_XR` and their sRGB twins) fragment sample at an
  implicit level, or with explicit gradients, on every device: its texels are read in the shader, at
  Vulkan's level of detail computed there and Metal's level rule, where a constexpr sampler states a
  mip filter or a runtime sampler's word does. The device takes the level of detail by Vulkan's
  rules, whose scale factor may be approximated; Metal's level of detail from derivatives is not
  measured. The sample may read another level or filter than Metal's near a boundary.
- `metal2vulkan-degrade-unmeasured-fetch-blend` (precision). `PrecisionNote::UnmeasuredFetchBlend`:
  A fragment function reads a color attachment the pipeline blends, on a device that orders
  framebuffer fetch only through the pixel interlock (MoltenVK, RADV, ANV), of a format whose Metal
  blend arithmetic is not built: a format whose channels read as floats but a 16-bit float, a 32-bit
  float, an sRGB and an extended-range one (a 16-bit normalized, 8-bit signed normalized, 10-bit,
  packed float or `A8Unorm` format, among others), an 8-bit unorm one written by an output of fewer
  than four lanes, a 32-bit float one (`R32Float`, `RG32Float`, `RGBA32Float`) written by an output
  of fewer than four lanes, by a module without fast math, or under a blend state the one sweep that
  measured that family does not fit: a color or alpha arithmetic of more than a lone term, a lone
  fused multiply-add, or a minimum or maximum, outside the states measured word for word. A
  one-channel attachment's `SourceColor` over `DestinationColor` is such a state, and so is a blend
  color's difference, which a 16-bit float attachment's blend fits and a 32-bit float one's does
  not; a lone term that reads the blend color, `OneMinusBlendColor` over `Zero` among them, is
  fitted and takes the exact form. Also a 16-bit float or 8-bit unorm one under a blend of the
  second source or beside a second (index 1) output. No Vulkan blend applies to the storage write,
  so the shader computes the pipeline's blend in `float`: each factor, product and sum rounded to
  nearest even, the source and the blend color clamped first to a normalized format's range (a NaN
  to its low end), an output's missing lane zero, and the result stored through the device's
  conversion. Metal's blend arithmetic for these formats is not measured, so the bound is not known.
  A factor of the second source reads the function's second output, clamped as the source. An 8-bit
  sRGB attachment (`R8Unorm_sRGB`, `RG8Unorm_sRGB`, `RGBA8Unorm_sRGB`, `BGRA8Unorm_sRGB`) under a
  blend of the second source, a one- or two-channel one under a lerp, each in a module compiled with
  fast math, and one written by an output of fewer than four lanes, announce it too: the shader
  computes Metal's 8-bit unorm blend's rounded forms over the decoded texel, in the output's type (a
  missing lane zero), and stores the result by the device's sRGB encode (`DevicePackConversion`),
  Metal's blend without fast math at four lanes; under fast math Metal regroups the blend,
  unmeasured.
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
  implementation-defined position within the fragment area being shaded"), a centroid varying
  (anywhere in the fragment's covered area, where Metal, measured on Apple M3 Max, moves it to a
  covered sample's location only under partial coverage), `[[point_coord]]` ("the fragment center,
  or the location of a sample"), or `[[barycentric_coord]]` (its builtin). A function that runs per
  sample under sample shading reads all but the center varying where Vulkan evaluates them for its
  sample, where Metal reads every one as the pixel's in each sample's run (Apple M3 Max at 4x,
  measured). Each is evaluated at a point of the pixel, so it differs from Metal's by at most its
  change over half the pixel's diagonal, and a centroid varying by at most its change across the
  pixel's covered area; exact where the device evaluates a read where Metal does.
- `metal2vulkan-degrade-rasterization-rate-map` (precision). `PrecisionNote::RasterizationRateMap`:
  A render pass drawn by regions over a rasterization-rate map that is not the identity: its
  regions' lines are this crate's tiles' exact lines, where Apple's are its fixed-point ones, so a
  fragment's evaluated point lies within a bounded distance of Metal's.
- `metal2vulkan-degrade-bicubic-filter-arithmetic` (precision).
  `PrecisionNote::BicubicFilterArithmetic`: Metal's bicubic filter, emulated by fetches with Metal's
  weights, the 8- and 16-bit normalized and the extended-range formats' exact sum rounded half up
  onto the format's grid, and the 32-bit float and half-float formats' sums through Metal's
  accumulators, differs from an Apple M3 Max in what is not yet pinned: a 2D normalized sum within
  about `2^-21` of the format's range (`1` unsigned, `2` signed) of a rounding tie: Metal may round
  it to the other side, one grid step away (`1/65535` or `1/32767` for 16-bit, `1/4080` or `1/2032`
  for 8-bit); an extended-range 2D sum (`BGR10_XR`, `BGRA10_XR`, `RGB10A8_2P_XR10` and their sRGB
  twins) within 0.014 of a grid step of a tie likewise (`1/8160` a color lane, `1/8192` an sRGB
  color lane, `1/4080` alpha).
- `metal2vulkan-degrade-sparse-residency` (precision). `PrecisionNote::SparseResidency`: A stage
  samples or gathers a texture whose residency it observes (`sparse_color::resident()`) through the
  slot's sparse form, which the module has where the profile enables `shaderResourceResidency`:
  the word is the device's `OpImageSparse*` residency, whose footprint may count a texel of zero
  weight Metal does not, and an unmapped texel's lanes the device's non-resident value. Where the
  device has no residency of its own, the slot's tile map answers Metal's footprint rule exactly,
  unrecorded.
- `metal2vulkan-degrade-device-nan-encoding` (precision). `PrecisionNote::DeviceNanEncoding`: A
  `float` NaN result's bit pattern follows the device: Metal answers the canonical `0x7fc00000` for
  the NaN results of its float arithmetic and functions, and Vulkan states no encoding, so a device
  may answer another sign or payload (llvmpipe: `0xffc00000`). Also, once per device, a GPU blit
  into, out of or between textures of which one is an `RG11B10Float` texture Metal keeps losslessly
  compressed, whose compression makes a NaN field the quiet NaN (`0x7e0`, `0x3f0`), where the
  device's copy keeps the bits. The runtime announces it too for a plain color fill
  (`BlitCommandEncoder::fill_texture_color` without a pixel format) to a NaN of a half or
  small-float texture the device's clear writes, which converts the NaN as the device likes, where
  Metal's fill keeps a half NaN's sign and payload and makes a small float's all ones. Also a blend
  a fragment function computes into a 16-bit float attachment through the pixel interlock: a NaN
  of the blend is the device's arithmetic's, which the storage image's store converts, where Metal
  stores every NaN of the blend as `0x7e00`.
- `metal2vulkan-degrade-restricted-depth-range` (precision). `PrecisionNote::RestrictedDepthRange`:
  The device compares a depth texture's texels on a device without `VK_EXT_depth_range_unrestricted`
  enabled whose driver is not measured to keep depth values outside `[0, 1]` without it.
- `metal2vulkan-degrade-cube-gradient-footprint` (precision).
  `PrecisionNote::CubeGradientFootprint`: An explicit-gradient sample of a cube texture takes
  Metal's footprint in a form a Vulkan device reads alike, but for what is not reproduced. Cost:
  off MoltenVK the sample takes the device's logarithm of Metal's footprint and, if anisotropic,
  the device's taps, so a level or blend near a boundary can differ from Metal's. Where the shader
  takes Metal's level and taps itself (`PrecisionNote::AnisotropicFilter`) this note is not
  recorded. On a MoltenVK release or GPU whose cube-gradient route is not measured the sample
  takes Vulkan's rule's form, the device's reading of it. Through MoltenVK's vertex and fragment stages
  under macOS 27, an isotropic sample is Metal's, but an anisotropic sample on a cube's `+Y` or
  `-Y` face loses the gradients' `z` component to SPIRV-Cross's `spvGradientCube`, so its ellipse
  lies on the face's diagonal (docs/FAST-PATHS.md's known gaps). A kernel's samples through
  MoltenVK are Metal's own, with no note.
- `metal2vulkan-degrade-half-texel-conversion` (precision). `PrecisionNote::HalfTexelConversion`:
  The module reads, gathers, or samples a texture with a `half` result, the device's `float` texel
  rounded to nearest even on every device: within one `half` ulp of Metal's, but a 32-bit float
  texel's sample past `±65504`, the infinity where Metal saturates. On MoltenVK, whose `float`
  texel is Metal's, announced at a draw or dispatch that binds a texture of a class whose Metal
  `half` differs from it (a 32-bit float texel's sample, a 16-bit normalized texel, a format not
  measured) and a two-plane YCbCr texture on the grid `8192` at a slot sampled through a runtime
  sampler (a sample of one texel rounds to nearest even, one `half` ulp from Metal's ties-away
  rounding at a tie); a heap element or a depth texture wherever it is accessed.
- `metal2vulkan-degrade-half-sample-filter` (precision). `PrecisionNote::HalfSampleFilter`: The
  module samples a texture with a `half` result through a filter (linear minification or
  magnification, or a runtime sampler) that Metal's own sampler (MoltenVK) answers otherwise than
  Metal filters the format bound there: a two-plane YCbCr texture's.
- `metal2vulkan-degrade-srgb-texel-decode` (precision). `PrecisionNote::SrgbTexelDecode`: The module
  reads, samples, or gathers a texture with a `half` or `float` result through the device's sRGB
  decode, or decodes the device's filter of its codes, where the texture may be bound in an 8-bit
  sRGB format. The device's decode is the only form wherever it samples the format (an 8-bit sRGB
  format, or the sRGB form of a BC, PVRTC, ETC2, EAC or ASTC format, through the texture's or its
  decoded twin's sRGB view), so it is announced at a draw or dispatch that binds such a texture,
  at an argument buffer's texture access, and at a render pipeline that reads an sRGB color
  attachment as an input attachment; never on MoltenVK, whose decode is Metal's own. Where the
  device holds the format as its UNORM twin, or a storage texture holds it through its UNORM
  alias, the module decodes the codes (or the device's filter of them) by the MSL transfer
  through the device's `Pow`, announced at every such access on every device.
- `metal2vulkan-degrade-float-sample-filter` (precision). `PrecisionNote::FloatSampleFilter`: The
  module samples a texture with a `float` result through a filter that Metal's own sampler
  (MoltenVK) answers otherwise than Metal filters the format bound there: a two-plane YCbCr
  texture's.
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
  whose conversion the device makes where it is not measured to make Metal's: a normalized, sRGB or
  small float channel on a device other than MoltenVK on an Apple GPU not measured to convert that
  kind as Metal does, an attachment that blends with factors other than one and zero
  (`ChannelWrite::Device`), the alpha of attachment 0 under alpha-to-coverage, a format whose
  conversion Metal was not measured with (the packed 16-bit unorm formats, `A8Unorm`, the packed
  floats). Near a tie between two codes the device may store the other one. Also a `float` texel a
  function writes to a storage texture that may hold a normalized or half-float format, or
  `ARGB8Unorm`, on a device not measured to store Metal's codes (or a 1-bit alpha, on one measured
  to store every other normalized channel's); an integer texel written into a storage texture of
  the float class that may hold a half-float or small-float format, which Metal rounds to nearest
  even and the device's store as it rounds a `float` (toward zero on every measured driver, one
  unit in the last place, a finite value past 65504 stored as 65504); and a fragment function's
  `float` blend into a 16-bit float attachment through the pixel interlock on a device not
  measured to round a `half` store as Metal does, and a blend into an 8-bit unorm attachment through
  the pixel interlock, whose result the storage image's store converts to its code. The recorded bound past one code, measured on
  lavapipe against plain Metal on an Apple M3 Max: a blend into an 8-bit unorm attachment
  (`RGBA8Unorm`, `ARGB8Unorm`) within 2 codes of Metal's; a fragment function that fetches an 8-bit
  sRGB attachment and writes it again, within 2 codes once a later draw reads an earlier one's code
  (with `srgb-texel-decode`); a blend into an 8-bit sRGB attachment, within 4 codes. Another device
  is not bounded past one code (an NVIDIA RTX 5080 Laptop GPU's blend into an 8-bit sRGB attachment
  stored up to 11 codes from Metal's).
  Also a render pass's clear of a normalized, sRGB,
  packed 16-bit, half or small-float color attachment to a color whose conversion is in question
  (a normalized channel within 1/1024 of a midpoint between two codes, an sRGB channel inside its
  range, a value the format does not hold exactly): the device converts the clear color, where
  Metal's clear rounds each channel to nearest, a tie to even, and truncates a small float's and a
  shared exponent's mantissas. Not for a clear by the load operation of an attachment the pass's
  render area covers on a driver whose load operation clears as Metal's `Clear` load action does
  (`DriverFact::ColorClearIsMetals`: MoltenVK on the Apple9 GPUs under macOS 27, measured on the M3
  Max); a clear made before the pass, of an attachment its first instance leaves out, is announced
  there too. Also a plain color fill (`BlitCommandEncoder::fill_texture_color` without a pixel
  format) of a texture of such a format the device renders, which the device's clear of the color
  writes, to a color whose conversion is in question against Metal's fill packing (a normalized
  channel within 1/1024 of a midpoint between two codes, more for a packed 16-bit channel, an sRGB
  channel inside its range, a value a half, small float or shared exponent does not hold exactly):
  Metal's fill rounds a tie half up and takes an sRGB channel's code whose decoded value is
  nearest.
- `metal2vulkan-degrade-device-interpolation` (precision). `PrecisionNote::DeviceInterpolation`: A
  fragment function reads a varying the device interpolates (`center_perspective`, `centroid`,
  `sample` or `no_perspective`, not `flat`) on a device not measured to interpolate as Metal does
  (`DriverDeviation::MetalInterpolation`, which MoltenVK takes: Metal interpolates there). Vulkan
  states no precision for interpolation, and RADV, ANV, lavapipe and NVIDIA's driver each round
  most interpolated lanes otherwise than Metal, up to about `1.5e-4` apart (`3e-4` on lavapipe).
- `metal2vulkan-degrade-quads-split-by-clipping` (precision). `PrecisionNote::QuadsSplitByClipping`:
  A fragment function runs a quad or simdgroup operation on a device that shades a triangle it clips
  as several triangles (`DriverWorkaround::ClippedTrianglesSplitPixelFragments`): a quad a seam
  between the pieces crosses runs once per piece, each run with the other piece's pixels as helper
  invocations, so the operation counts or combines fewer active lanes than Metal's, whose quad holds
  all four.
- `metal2vulkan-degrade-vertex-simdgroup-grouping` (precision).
  `PrecisionNote::VertexSimdgroupGrouping`: A vertex function runs a simdgroup or quad operation on
  a device whose vertex stage takes no required subgroup size and whose driver is measured to group
  vertices otherwise than Apple's GPUs (`DriverWorkaround::VertexGroupingNotApples`: ANV, 8 vertices
  a subgroup, interleaved): the operation runs over the device's group of vertices, so a sum, a
  ballot or a shuffle combines other vertices than on an Apple GPU, whose simdgroup holds 32
  consecutive vertices.
- `metal2vulkan-degrade-fragment-simdgroup-grouping` (precision).
  `PrecisionNote::FragmentSimdgroupGrouping`: A fragment function runs a simdgroup operation (a
  reduction, scan, vote, ballot, shuffle, broadcast or AGX multiply-add) or reads its
  `[[thread_index_in_simdgroup]]` on a device whose fragment subgroups are not Metal's simdgroups
  (the profile does not take `DriverDeviation::MetalFragmentSimdgroups`): the operation runs over
  the pixels the device's rasterizer packs into a subgroup, so a sum, a ballot or a shuffle combines
  other pixels than on an Apple GPU, whose packing is measured, and the lane is the pixel's place in
  the device's subgroup. A device whose fragment subgroup holds fewer than two quads combines the
  one quad's pixels alone, so a reduction ahead of a `discard_fragment` counts no quad the discard
  drops, where an Apple GPU's simdgroup counts the dropped quads of its 8x4 pixel block; the
  device states one fragment subgroup size, so no required size widens it, and no exact form
  exists there.
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
- `metal2vulkan-degrade-device-alpha-coverage` (precision). `PrecisionNote::DeviceAlphaCoverage`:
  A render pipeline covers by alpha (`alphaToCoverageEnabled`, over a four-component float first
  output at attachment 0) on a device not measured to cover as Metal does
  (`DriverDeviation::MetalAlphaCoverage`, which MoltenVK takes on the Apple M3 Max's macOS 27
  identity: Metal covers there). The device converts the alpha to its coverage mask by its own
  algorithm, which Vulkan does not specify ("It is intended that the number of 1's in this value be
  proportional to the alpha value", and "The algorithm may be different at different framebuffer
  coordinates"). Metal, measured over 639 alphas a sample count on an Apple M3 Max under macOS
  27.0.1, covers a pixel at one sample above an alpha of 1/2, and at two and four samples
  `ceil(alpha * n - t)` samples, `t` a 2x2 ordered dither of the pixel's window coordinates (1/8,
  5/8; 7/8, 3/8), either count within an `f32` ulp of a threshold. Recorded bound, pixel by pixel
  against Metal's reading: on lavapipe (Mesa 26.0.8, arm64) every pixel covers `floor(alpha * n)`
  or `ceil(alpha * n)` samples, so within one sample of Metal's and Metal's where `alpha * n` is an
  integer; at one sample, where lavapipe dithers too, a whole fragment is stored or dropped where
  its alpha lies between its own threshold and 1/2 (2,560 of the sweep's 10,224 pixels); at four
  samples a pixel is one sample apart only within an `f32` ulp of a threshold (96 of 10,224). An
  NVIDIA RTX 5080 Laptop GPU stores a fragment of alpha 1/2 at one sample, where Metal drops it;
  every other device and count is unmeasured.
- `metal2vulkan-degrade-rotated-blend-rounding` (precision). `PrecisionNote::RotatedBlendRounding`:
  A fragment function writes an attachment held with its channels rotated (`ARGB8Unorm`) whose blend
  is none of the measured forms (`RotatedBlendForm::Unmeasured`), or the premultiplied form through
  a first output that is not 32-bit float: Apple's blend computes a factor pair apart in its color
  and alpha arithmetic, and the rotation moves Metal's blue into the image's alpha and Metal's alpha
  into its red, so a channel may store one code apart from Metal's.
- `metal2vulkan-degrade-linked-sampler-numbering` (precision).
  `PrecisionNote::LinkedSamplerNumbering`: The sampler resource IDs a pipeline state takes for the
  constexpr samplers of the visible functions its stage links may be other than Metal's: two or more
  linked functions bring constexpr samplers, which Metal numbers in an order the pipeline's inputs
  do not decide; or a function a table can hold that linking did not import could not be planned, so
  its samplers are not known; or a dynamic library's symbol, a binary function or a function-pointer
  group's function brings them, whose numbering is not measured.
- `metal2vulkan-degrade-computed-sample-depth` (precision). `PrecisionNote::ComputedSampleDepth`: A
  fragment function runs its draw's depth test itself at more than one sample on a device whose
  `FragCoord.z` is the pixel centre's there (`VkProperty::CenterFragCoordDepth`), and computes each
  sample's depth from its vertex function's window depth read at the sample, plus `FragCoord.z` less
  that depth read at the centre, which carries the depth bias.
- `metal2vulkan-degrade-argument-pixel-sampler-scaling` (precision).
  `PrecisionNote::ArgumentPixelSamplerScaling`: A sample, gather or comparison through an argument
  buffer's sampler whose word the specialized module cannot read may hand the device a pixel
  coordinate scaled by the texture's size, where the sampler states pixel coordinates, with
  `pixel-sampler-scaling`'s bound. A sample without a comparison through a state that filters
  nearest and clamps to the edge fetches its texel instead. Announced at the first draw or dispatch
  of a pipeline that records it while the device holds a sampler state of pixel coordinates.
- `metal2vulkan-degrade-half-output-precision` (precision). `PrecisionNote::HalfOutputPrecision`: A
  fragment function returns a `half` color output (a scalar or vector of AIR's `half`) to an
  attachment the pipeline has, and either its module is compiled without fast math
  (`air.compile.fast_math_disable`) and that attachment blends, factors one and zero included, with
  some channel written (`StageIo::half_blend_colors`, which the pipeline's stage reads), or its
  module names no math mode (under fast math the pipeline's stage records
  `PrecisionNote::HalfOutputFastMathBlend` instead).
- `metal2vulkan-degrade-extended-range-blend-source` (precision).
  `PrecisionNote::ExtendedRangeBlendSource`: A fragment function's `half` color output (a scalar or
  vector of AIR's `half`) is blended into an extended-range attachment (`BGR10_XR`, `BGRA10_XR`,
  `RGB10A8_2P_XR10`, or an `_sRGB` twin) where Metal may round the blend's source otherwise than the
  function's computed blend: under fast math a lane of the output is not a load; or the module names
  no math mode.
- `metal2vulkan-degrade-discard-drops-simdgroup-stores` (precision).
  `PrecisionNote::DiscardDropsSimdgroupStores`: A fragment function meets the store-dropping defect
  of `DriverWorkaround::DiscardDropsSimdgroupStores` (MoltenVK on an Apple GPU: Apple's compiler
  drops the texture writes of every live thread of a simdgroup holding a discarded thread, where the
  function discards, then writes storage, samples, and calls SPIRV-Cross's `optnone` arithmetic
  helpers), and a discard site that would discard at once cannot wait until after the writes: a
  store, atomic, image write or non-quad subgroup operation follows it, which a predicated discard
  does not yet guard (`lower::fragment::discard`).
- `metal2vulkan-degrade-helper-branch-drops-simdgroup-stores` (precision).
  `PrecisionNote::HelperBranchDropsSimdgroupStores`: A fragment function meets the store-dropping
  defect of `DriverWorkaround::DiscardDropsSimdgroupStores` without a discard: SPIRV-Cross writes an
  operation of the module as an `optnone` call (a sum, difference or product decorated
  `NoContraction`, among others) where only helper invocations (the quad helpers past a primitive's
  edges, or demoted invocations) may run it before a storage write, and keeping them out would not
  be exact (`spirv::helper_branches`): a derivative or subgroup operation reads what they compute
  there or after, an arm leaves its selection otherwise than at its merge, or the call is in a loop
  whose helper invocations may iterate alone.
- `metal2vulkan-degrade-extended-range-float-blend-source` (precision).
  `PrecisionNote::ExtendedRangeFloatBlendSource`: `PrecisionNote::ExtendedRangeBlendSource` for a
  `float` color output (a scalar or vector of AIR's `float`): blended into an extended-range
  attachment where Metal may round the blend's source otherwise than the function's computed blend,
  under fast math a lane of the output that is not a load, or in a module that names no math mode.
- `metal2vulkan-degrade-anisotropic-filter` (precision). `PrecisionNote::AnisotropicFilter`: The
  module samples a texture through an anisotropic sampler with Metal's anisotropic filter emulated
  by the device's samples at explicit levels (`lower::sampler::anisotropic`), on a device without
  `samplerAnisotropy`, every Vulkan sampler being isotropic there, each tap the device's isotropic
  filter: a sample without comparison of a 2D texture or 2D array at the fragment's implicit
  level or with explicit gradients, or of a cube or cube array with explicit gradients, through a
  constexpr sampler of `maxAnisotropy` above 1, or through a runtime sampler slot whose word states
  anisotropy (under that word alone). A device with the feature samples with its own anisotropic
  filter. Cost: a
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
  function writes `[[point_size]]`, or a mesh function `set_point_size`, on a device whose
  `pointSizeRange` ends below Metal's largest point, 511 pixels (`Limits::max_point_size`: 255 on
  ANV, 256 on lavapipe).
- `metal2vulkan-degrade-clipped-coverage-split` (precision). `PrecisionNote::ClippedCoverageSplit`:
  A fragment function compiled for a multisampled pipeline reads `[[sample_mask]]` on a device whose
  rasterizer is not Apple's (`DriverWorkaround::ClippedCoverageNotMetals`): Apple's GPU draws a
  triangle that extends past the viewport as pieces whose shared edges cross pixels, and such a
  pixel's fragments can run once per piece, each reading that piece's coverage, where the device
  hands the pixel's fragment its coverage by the whole primitive (lavapipe by its own pieces).
- `metal2vulkan-degrade-looped-sample-outputs` (precision). `PrecisionNote::LoopedSampleOutputs`: A
  fragment function that reads `[[sample_mask]]`, runs once per covered sample on Metal and writes
  an output runs once per pixel in a multisampled pipeline, its body once per covered sample, on a
  device that does not run it per sample itself (MoltenVK on Apple's GPU runs a memory writer per
  sample itself, as written). The samples Metal writes are written, but each with the last
  pass's outputs, where Metal writes each sample's from that sample's run: equal where every pass
  computes the same outputs. Vulkan hands an invocation under sample shading only its own sample's
  bit of the coverage.
- `metal2vulkan-degrade-half-output-fast-math-blend` (precision).
  `PrecisionNote::HalfOutputFastMathBlend`: A fragment function of a module compiled with fast math
  (`air.compile.fast_math_enable`) returns a `half` color output to an attachment the pipeline has
  that stores some channel (its write mask names one of its format's channels;
  `StageIo::half_fast_colors`, which the pipeline's stage reads, or the interlocked storage form's
  write), where `PrecisionNote::HalfOutputPrecision`'s conditions hold; or
  computes the blend of a 16-bit or 32-bit float attachment it fetches in the interlocked storage
  form (no Vulkan blend applies to a storage write) where Metal's compiler may fold the output's
  own arithmetic into that blend or regroup a sum of it: within 2 `half` units in the last place of
  Metal's at the operands' magnitude, 16 at the stored value's under cancellation. Where the output
  is opaque to the blend and its form fits every measured state, the function's blend is Metal's
  and records nothing; so does the blend of a fetched 8-bit unorm attachment it computes there
  (every factor, operation, write mask and blend color it builds), which Metal reads the output
  into clamped, folding none of it (the storage image's store of its result is the device's
  conversion, `DeviceColorConversion`). On a device that enables `storageInputOutput16` the output
  reaches the device as a 16-bit `half` (`StageIo::half_outputs`); MoltenVK 1.4.2 on an Apple M3
  Max then stores Metal's codes for a loaded `half4` under every blend measured (21, into 16- and
  32-bit float formats).
- `metal2vulkan-degrade-device-sampling` (precision). `PrecisionNote::DeviceSampling`: Native device
  sampling: the module hands a sample, gather, comparison or level query to the device's own
  sampler on a device whose sampler is not Metal's (the module requires
  `VkProperty::DeviceSampling`), its level clamp the descriptor's as given (Metal holds it on a
  `1/64` grid, and its nearest mip filter reads the lower level at a clamp halfway between two,
  where the device may read the upper) and a subnormal coordinate the device's to read (Metal reads it as a zero before
  addressing). That includes a
  linear sample of a two-plane YCbCr texture, whose texels hold Metal's reads exactly: the device's
  filter of them lands within 1.183 codes (1/255) of Metal's sum on the format's grid, so a draw
  binding such a texture makes no pipeline of its own.
- `metal2vulkan-degrade-native-transcendentals` (precision). `PrecisionNote::NativeTranscendentals`:
  Native device maths: the module computes a `float` or `half` math-library function by the device's
  own `GLSL.std.450` instruction (`exp`, `log`, `pow`, `sinh`, `cosh` and `tanh` among them; a
  composition of the device's `Exp2` and `Log2` only for `exp10` and `log10`, which no instruction
  is) where Metal's function is not the device's own; `fast::sqrt` the device's `Sqrt`; the fast
  `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh` and `atanh` the device's instructions, not MSL's
  fast-math formulas (`fast::sin` of `1e7` is the device's sine, where Metal answers `+0`;
  `fast::tanh(45)` is `1`, where Metal answers NaN); `fabs`, `sign` and `fract` the device's
  `FAbs`, `FSign` and `Fract`, their special operands the instruction's (Metal's `sign` of a NaN,
  a zero and an `f32` denormal is `+0`, its `fract` of an infinity `+0`); a `half` function's
  signed zeros and NaN signs its `float` instruction's (`half` `sin(-0)` is `-0`, where Metal
  answers `+0`); `pow`, `powr` and `atan2` C99's special values by the device's comparisons
  around its instruction (`pow(x, 1)` the device's `Pow`, where Metal answers `x`; an `f32`
  denormal operand as the device reads it); and `round` C99's from the device's `Trunc` on every
  device (`round(-0.3)` is `-0`, where Metal answers `+0`).
- `metal2vulkan-degrade-native-division` (precision). `PrecisionNote::NativeDivision`: Native device
  maths: a precise `float` or `half` division, or Metal's fast division or reciprocal, is the
  device's `OpFDiv` where the device's division is not Metal's own: within Vulkan's 2.5 ulp, where
  Metal's precise division is correctly rounded.
- `metal2vulkan-degrade-native-fma` (precision). `PrecisionNote::NativeFma`: Native device maths: a
  `float` fused multiply-add on a device without `OpFmaKHR`, or a `half` one where the device's
  16-bit `Fma` is not measured to be Metal's and the device has no 16-bit `OpFmaKHR` under the
  16-bit float controls, is the device's `GLSL.std.450 Fma` (on the `half` operands held in
  `float`, its result rounded to `half`), where Metal's rounds once.
- `metal2vulkan-degrade-device-texel-conversion` (precision).
  `PrecisionNote::DeviceTexelConversion`: Native device sampling (a texel conversion): the module
  hands a read, sample or gather with a `float` result the device's own conversion of the texel's
  codes, off Apple's GPUs (`VkProperty::MetalSampler` keeps Metal's value). Recorded bound, of a
  texel read from one texel of one level: an 8-, 10- or 16-bit normalized code within one `f32`
  ulp of Metal's `c / (2^b - 1)` where measured (lavapipe multiplies by the reciprocal: some 8-bit
  codes one ulp high; some 10-bit `BGR10A2Unorm`, a 16-bit and 512 `Depth16Unorm` codes one ulp
  low); a 4-, 5- or 6-bit channel of a packed format within one `f32` ulp where the device converts
  it by the reciprocal (lavapipe: some `ABGR4Unorm` codes one ulp high), and within 0.18 of a code
  where it widens a 5- or 6-bit channel by replicating its bits (ANV, NVIDIA; a 4-bit channel's
  replication is exact); a half-float or small-float texel's NaN in the device's encoding, where
  Metal answers `0x7fc00000`; every other class Metal's value.
- `metal2vulkan-degrade-device-vertex-conversion` (precision).
  `PrecisionNote::DeviceVertexConversion`: Native vertex input (a conversion): a render pipeline's
  vertex descriptor gives a normalized format (`UCharNormalized` through `Short4Normalized` of
  either sign, `UChar4Normalized_BGRA`, `Int1010102Normalized`, `UInt1010102Normalized`) to an
  attribute that fixed-function vertex input fetches, so the attribute is the device's `UNORM` or
  `SNORM` conversion of its codes, where Metal's fetch answers the code over its maximum correctly
  rounded (an `Int1010102Normalized` 10-bit field times `float(1/511)`): within one `f32` ulp
  where measured (lavapipe reads some 8-bit unsigned codes one ulp high), but lavapipe's
  per-instance fetch, which reads the most negative code of an 8- or 16-bit signed format
  unclamped (-128/127, -32768/32767) where Metal reads -1. Recorded on the pipeline's vertex stage
  on every device but one whose profile takes `DriverDeviation::MetalNormalizedVertexFetch`
  (MoltenVK on the Apple M3 Max's macOS 27 identity), where only a 16-bit format read as `half`
  records it; one module serves every format the device fetches. A normalized attribute the shader
  pulls from its buffer instead (an offset, stride or step fixed-function input cannot fetch, a
  kernel's `[[stage_in]]` attribute) is the module's product of the code and its maximum's
  reciprocal, rounded once, the signed one clamped to -1: within one `f32` ulp of Metal's quotient
  (some 8-bit unsigned codes one ulp high, some signed, 10-bit and 16-bit codes one ulp toward
  zero; 0, the maximum and -1 exact). The function records it on every device, read as `float` for
  every normalized format but `Int1010102Normalized` (Metal's own product), and read as `half` for
  a 16-bit format (one `half` ulp at four codes).
- `metal2vulkan-degrade-native-half-conversion` (precision). `PrecisionNote::NativeHalfConversion`:
  Native half rounding (a conversion): the module rounds a `half` result it computes in `float` (a
  `half` operation's, a function's, a conversion from `float`'s, a texel's) by the device's
  conversion of the `float` to 16 bits and its widening, on a device with `shaderFloat16` whose
  float controls do not make that conversion Metal's `half(float)` (`RoundingModeRTE`,
  `DenormPreserve` and `SignedZeroInfNanPreserve` for 16 bits: ANV and RADV round through the
  conversion under them, exactly, with no note): NVIDIA, whose driver folds the conversion's round
  trip even under those controls (`DriverWorkaround::HalfConversionPairFolded`), MoltenVK,
  lavapipe, and any device without them. Also recorded where a `half` read from its bits is the
  device's widening, or a `half` encoded to its bits (where they are not the bits it was read from)
  is the device's narrowing, and that conversion is neither under those controls nor measured to be
  Metal's: the conversion on NVIDIA and lavapipe, and `GLSL.std.450 UnpackHalf2x16` and
  `PackHalf2x16` on every device without `shaderFloat16`. A half denormal may read or encode as the
  zero of its sign.
- `metal2vulkan-degrade-native-half-quantize` (precision). `PrecisionNote::NativeHalfQuantize`:
  Native half rounding (a quantize): the module rounds a `half` result it computes in `float` (a
  `half` operation's, a function's, a conversion from `float`'s, a texel's) by the device's
  `OpQuantizeToF16` alone, on every device without `shaderFloat16`, its quantize measured or not: a
  value below the smallest normal `half` may round to a zero where Metal answers the half denormal
  (or `2^-14`, from just below it). An integer's `half` rounds exactly instead.
- `metal2vulkan-degrade-native-nan-encoding` (precision). `PrecisionNote::NativeNanEncoding`: Native
  device maths: a `float` or `half` operation's NaN result is the device's, its sign and payload
  unselected, where Metal answers a NaN the device is not measured to answer (every NaN but
  Metal's canonical one on a device measured to answer that, MoltenVK on an Apple GPU): a `float`
  `fneg` and `fabs`, which keep a NaN operand's payload on Metal, and a `half` `fract`, which
  returns it, among them. Also
  recorded wherever a `half` is read from its bits, on every device, since the read is the
  device's widening (its conversion with `shaderFloat16`, `UnpackHalf2x16` without): a NaN's bits
  the device's, where Metal widens every `half` NaN to `0x7fc00000` and moves one it does not
  widen with its bits. And wherever a `half` is encoded to its bits (a store, a reinterpretation, a
  texel's bits) where they are not the bits it was read from, on every device, since the encode is
  the device's narrowing (its conversion with `shaderFloat16`, `PackHalf2x16` without): a NaN's
  bits the device's, where Metal keeps a `half` NaN's sign and payload.
- `metal2vulkan-degrade-native-min-max` (precision). `PrecisionNote::NativeMinMax`: Native device
  maths: Metal's `fmin`, `fmax`, `min3`, `max3`, `clamp` or `saturate` (its `clamp(x, +0, 1)`) on
  `float` or `half` is the device's `GLSL.std.450` `NMin`, `NMax` or `NClamp`, on a device not
  measured to answer Metal's operand; and `median3` on every device, `min(max(c, min(a, b)),
  max(a, b))` of the device's `NMin` and `NMax`, whose pick of two NaNs and of an `f32` denormal is
  the device's (Metal's `fast::median3` answers `b` for two NaN first operands).
- `metal2vulkan-degrade-native-dot` (precision). `PrecisionNote::NativeDot`: Native device maths:
  Metal's `dot` on `float` and `half` is the device's `OpDot` (a `half` one summed in `float` and
  rounded once to `half`), not Metal's fused chain of its products, on every device but for a
  `float` one where the device's functions are Metal's own (MoltenVK).
- `metal2vulkan-degrade-native-mix` (precision). `PrecisionNote::NativeMix`: Native device maths:
  Metal's `mix(x, y, t)` on `float` and `half` is the device's `FMix` (a `half` one computed in
  `float` and rounded once to `half`), not Metal's `fma(t, y, fma(-t, x, x))`, on every device but
  for a `float` one where the device's functions are Metal's own (MoltenVK).
- `metal2vulkan-degrade-device-pack-conversion` (precision). `PrecisionNote::DevicePackConversion`:
  Native device maths (a conversion): `air.pack` or `air.unpack` of a normalized format is the
  device's: `GLSL.std.450`'s `Pack`/`Unpack` `Unorm4x8`, `Snorm4x8`, `Unorm2x16` and `Snorm2x16`,
  their definition by the device's `NClamp`, multiply, `RoundEven` and reciprocal for `rgb10a2` and
  `rgb565`, and the MSL transfer functions by the device's `Pow` for `unorm4x8_srgb`'s colour. Metal
  rounds the exact product and divides correctly rounded, answers code 0 for a NaN channel, and
  takes an sRGB colour through its twelve-bit table; a packed code is within one of Metal's, an
  unpacked value within 2.5 ulp, a decoded sRGB colour within half a twelve-bit step (`1 / 8190`)
  of Metal's step beyond its `Pow`'s bound. Also a write of an 8-bit sRGB storage texture held
  through its UNORM alias, which encodes by `unorm4x8_srgb`'s pack. Not recorded for a
  `float` conversion of the four named formats on MoltenVK, Metal's own there. The same conversions
  serve an implicit imageblock access of a normalized attachment (a `half` load rounding the
  unpacked `float` once more), a fragment function's read and write of an 8-bit sRGB attachment
  in the interlocked storage form or through its word view, and a snapshot read's decode of a
  normalized code (`c` times the `f32` nearest `1 / D`), noted alike.
- `metal2vulkan-degrade-color-attachment-groups` (precision).
  `PrecisionNote::ColorAttachmentGroups`: A render pipeline whose color attachments the device
  draws in more than one group (more than the lesser of its `maxColorAttachments` and
  `maxFragmentOutputAttachments`, four on a device at Vulkan's floor): each draw runs once per
  group, each group's variant writing its own attachments over its own copy of the depth and
  stencil the pass loads. Every variant's vertex stage is the same module, `Position` invariant,
  and every fragment variant's `FragDepth`, `SampleMask` and `FragStencilRefEXT` are decorated
  `Invariant`; Vulkan promises no more across pipelines of other fragment modules and attachments.
  Groups past the first may differ at discard or depth edges if the driver compiles the variants
  differently.

## 10. Accepted differences

Where the runtime answers otherwise than Metal on purpose, and nothing is slower or approximate
for it (so no degrade notice says so), the difference is listed here. Most are answers Metal leaves
undefined or ends the process for, and answers the runtime adds beyond Metal's API. The rustdoc of
each item named states the rule in full.

**Draws and index buffers**

- **An indexed draw past its index buffer draws.** Metal's API validation stops it; the runtime
  draws as Metal does without validation: a partial last index reads up to the length rounded up
  to 4 bytes, and every later position draws index 0 plus the base vertex.
- **An indexed draw from an index offset at or past the length draws index 0** plus the base vertex
  at each position, where Metal reads whatever memory follows (`zeroed-index-fetch`).
- **An object grid of more than 2^29 threadgroups fails its command buffer by its count, at
  once** (`GpuRecoveryVictim`, Metal's code 1). Metal's GPU runs it for its watchdog's 5 seconds,
  then discards the command buffer from the draw's encoder on, and a command buffer of the queue
  committed while it ran. The runtime fails by the count Metal's GPU ran within the watchdog with
  an object function that does next to nothing (2^29 in 4.98 s, `bind::METAL_OBJECT_GRID_TOTAL`):
  a grid of up to 2^29 runs whatever its function costs, where Metal discards a slow one; the
  queue's other command buffers run. A direct draw fails from its encoder on, as Metal's; an
  indirect one, whose count only its arguments pass reads, draws nothing and fails the command
  buffer while the rest of its encoder and the encoders after it run.
- **A mesh grid of 1,048,577 to 2,097,151 threadgroups draws nothing**, direct, indirect or set by
  an object threadgroup. Metal on the M3 Max draws whole up to 1,048,575 and nothing at exactly
  2^20 or from 2^21 on, and between them is erratic (17 by 61,681 and 65,535 by 24 drew, 1025 by
  1025 and 33 by 32,768 did not); the runtime answers there as at 2^20
  (`bind::METAL_MESH_GRID_TOTAL`).
- **A draw's `VkPipeline` is made while the draw goes on.** A draw whose pipeline is not made yet
  returns once encoded; the pipeline is made on the device's pool of at most four threads, and the
  command buffer's work waits for it in queue order.
- **A draw can take its key's residual form while its own pipeline is made.** A state's creation
  makes its predicted key with every constant buffer slot read through a uniform view read at the
  slot's residual (`RangeAtBindOffset` 0), which reads the same bytes at every bind offset: the
  binder writes the slot's descriptor range and residual from the bind offset alone, whichever form
  the pipeline takes, and the residual is 0 wherever the range starts at the bind offset. A slot
  bound to nothing is read at its residual too (0), so a draw that leaves such slots unbound takes
  the creation's pipeline itself. A draw whose own pipeline (which reads such a slot at the bind
  offset alone, the residual-free form) is not made, and whose key's residual form's is, draws with
  that one at once; its own is made on the pool behind the makings something waits for, and later
  draws take it. Only GPU time moves for the draws before it lands: on ANV the residual form's reads
  are loads where the other form's are pushed constants (`WorkCounts::residual_form_draws`,
  `residual_form_makings`). On MoltenVK every key is its residual form (the two forms measured
  alike there), so a state's draws at every bind offset take the one pipeline its creation made.
- **A dispatch's `VkPipeline` is made while the dispatch goes on**, the same way: one pipeline per
  threadgroup size and edge region, made on the pool. A compute state whose descriptor states
  `requiredThreadsPerThreadgroup` -- the threadgroup every dispatch of the state must have -- makes
  one at its creation: its predicted region's, at that size, each uniform view read at its
  residual; a dispatch of that region takes it, as does one whose key differs only in reading a
  view at its bind offset alone while its own is made
  (`WorkCounts::creation_dispatch_pipelines_made`, `creation_dispatch_pipelines_drawn`). A state
  whose descriptor requires no threadgroup size makes none at its creation: the size is the
  dispatch's own call and nothing in the request answers it, and a pipeline guessed from the state
  instead was taken by no dispatch at all over the whole conformance suite (141 made, 0 taken,
  against the render-side prediction's 764 of 794 in the same run). A tile dispatch's pipeline
  is made on the pool too, its pass recorded once it is made, the making ranked from the dispatch
  as one a committed command buffer waits for (`WorkCounts::tile_pipelines_pooled`); a failed
  making fails the command buffer, as a draw's does, `FailedPipeline` naming the dispatch by its
  order among the encoder's tile dispatch calls.

**Binary archives and dynamic libraries**

- **A binary archive serializes after its file is deleted.** `Device::new_binary_archive` reads the
  file whole, so `BinaryArchive::serialize_to_url` writes what was loaded; Metal's fails there.
- **An archive's file is a path.** `BinaryArchiveDescriptor::url` and `serialize_to_url` take a file
  path, not a URL.
- **A dynamic library's file is its metallib.** `DynamicLibrary::serialize_to_url` writes the bytes
  the library was made from, without a slice of GPU code; the runtime loads either form.

**Resource IDs and identifiers**

- **A gone holder's texture ID reads the neutral texture.** A function reading a texture ID from an
  argument buffer reaches the texture holding the ID as the command buffer runs; once the holder
  is gone, the ID reads the neutral texture, where Metal's read is undefined.
- **Two live textures with one ID: a function reads the holder**, the texture placed there last.
- **An indirect command buffer's ID is Metal's small integer**, from 1, the lowest free, in the
  indirect command buffers' own space; it stays held while committed work that runs or makes the
  buffer resident has not completed.
- **Resource IDs are numbered as Metal numbers them**: textures', sampler states', pipeline states'
  and indirect command buffers' IDs follow Metal's allocator, around reservations
  (`Device::reserve_resource_ids`) too. IDs that end Metal's process, or whose reservation Metal
  never returns from, are refused.
- **The device counts the sampler IDs it holds** (`Device::sampler_ids_live`,
  `Device::sampler_id_pool_size`), runtime extensions for a consumer that models Metal's sampler
  heap.
- **A texture can take no resource ID** (`IdPlacement::Outside`), for a consumer's helper textures
  and its textures over its own images (`Device::new_texture_with_vk_image`), such as swapchain
  images.
- **An indirect command buffer placed at an ID** (`IndirectCommandBufferDescriptor::placement`,
  Apple's private `setResourceIndex:`) takes it as Metal does; an index of 16384 or more is refused.
- **A pipeline state placed before any pipeline ID is made** (`PipelineOptions::placement`) is made,
  where Metal ended the process; an index of 16384 or more is refused.
- **An indirect command names its pipeline state without holding it.** As in Metal, releasing a
  state gives its ID back while a command names it; a CPU-encoded command still runs what the state
  ran, and a GPU-written record runs the ID's holder, or nothing (`released-command-state`).
- **Unique identifiers** (`ComputePipelineState::unique_identifier`,
  `RenderPipelineState::unique_identifier`, `IndirectCommandBuffer::unique_identifier`) are
  nonzero and unique among live objects, laid out by Metal's rule where it was measured
  (`plain-state-unique-identifier`, `unmeasured-unique-identifier` where it differs).
  `IndirectCommandBuffer::identifier_unmeasured` (no Metal counterpart) answers, without
  announcing, whether a buffer's identifier is one that notice names.

**Indirect command buffers**

- **A command's read-back answers Metal's private getters**: `command_type`, the buffer addresses
  and attribute strides, as Metal's `get*` getters answer them. `IndirectComputeCommand` and
  `IndirectRenderCommand` read one field of their record alone (`command_type`, `has_barrier`,
  `dispatch` or `draw`, one buffer slot, `pipeline_unique_identifier`, and the rest), as
  `IndirectCommandBuffer::commands` reads it, without decoding the whole command. Where Metal's
  getters answer other than what the command runs, the read-back answers as they do: a render
  command's `draw` of primitive type 7, 8 or 9 reads as one of points (0), and `command_type` of
  an indexed draw of type 9 reads as none; `commands` keeps the type the command draws.
- **Whether the GPU may have written the records** (`IndirectCommandBuffer::gpu_written`, no Metal
  counterpart) turns true at the first call that lets GPU work write them; from then on an
  execution runs the records as they stand once the work before it completed.
- **What an execution runs, where its records are read**
  (`CommandBuffer::add_executed_commands_handler`, no Metal counterpart): a handler is called each
  time the runtime reads an execution's records, at `commit`, in a host step, or not at all for an
  execution the GPU alone runs.
- **An execution the GPU alone runs orders its commands one after another in a serial encoder**,
  where Metal orders them only where their uses meet: the uses are the records', which the GPU
  writes. In a concurrent encoder its commands run together where no route that wrote the records
  may set a barrier (`IndirectCommandBuffer::command_writes`), and one after another otherwise.
- **Render commands the GPU may write run on the GPU alone** where the commit can tell from what
  the records may hold (`IndirectCommandBuffer::command_writes`) that every command is a draw of one
  primitive type under the encoder's pipeline state, which the buffer inherits with none of its
  buffers: one indirect draw per command, in order, from what a preprocess wrote; and where the
  commands draw points, lines or triangles, the state's vertex function writes no memory and its
  fragment function reads no buffer and no `[[primitive_id]]`, one merged draw for the whole range,
  in the commands' order, whatever its count (`WorkCounts::merged_draw_commands`). Otherwise, and
  where what the records may hold grew by the time the pass's batch is ready to submit, a host step
  records the pass.
- **A checked execution states what its check looks for**
  (`execute_commands_in_buffer_checked`'s and `execute_commands_in_buffer_indirect_checked`'s
  `IndirectCommandCheckNeeds`, no Metal counterpart): draws of points, and commands under states
  that may encode indirect commands. The check is called with each command the runtime records,
  before it records any, wherever it reads the records: at `commit` for the CPU's records, in a
  host step for records the GPU may have written. Where what the records may hold rules out every
  kind the check names (`IndirectCommandCheckNeeds::ALL` names both; a record of no draw and no
  dispatch runs nothing), the check has nothing to see, is not called, and the execution may run
  on the GPU alone, render and compute alike; where that grows by the time the batch is ready, a
  host step records it and calls the check.
- **Command predication** (`encode_start_if`, `encode_start_else`, `encode_end_if` and the
  while and do-while loops) runs Metal's blocks and loops on every device, with or without
  buffer device addresses, its unbalanced records answering `NO` and dropping their ranges, and
  its live blocks left open where a unit ends failing the command buffer as Metal's GPU recovery
  does (`PredicationOpenAtUnitEnd`). Loops nest to any depth and run as many passes as Metal's.
  A loop is laid down as 64 copies (16 inside another loop, fewer where they would lay down more
  than 16384 steps and gated dispatches, and at least one); where it runs past them, a check
  submits its copies again, from where it stopped, until it ends, each time a round trip to the
  host, holding the batches after it on its queue (and those on other queues that use what the
  loop uses), and nothing encoded before the loop runs again. A loop without bound runs on, as
  Metal's does, after the program let go of its queue and device too, until the process ends or
  the device is lost. A loop running again stops, as Metal's does, at the GPU's recovery from a
  fault or a hang of another queue's command buffer (primitive type 9, or one a program commits
  as failed with Metal's code of either, 3 or 2; one behind the loop on its own queue waits for
  it, a known gap in docs/FAST-PATHS.md): a fault's fails its command
  buffer as a victim (`GpuRecoveryVictim`), a hang's ends it with no error, and nothing after the
  loop runs. Work inside a loop the runtime records apart (an indirect dispatch or
  an indirect command buffer's execution a host step records) is recorded again in each such
  run, after another round trip to the host where its first recording waited for the work
  before it. A condition word past its buffer reads
  as Metal reads it (the heap's bytes, or zero), and as zero where Metal reads memory unrelated to
  the buffer (`predication-word-past-buffer`). An unbalanced record that is its unit's first
  drops the unit's dispatches before it as Metal's does: what the unit's encoders recorded before
  it is never submitted.
- **A GPU-written indirect command naming a placed ID runs that ID's live holder**, the state made
  last; once it is released, nothing.
- **An indirect render command keeps a draw's values to 32 bits**, as Metal's records keep most of
  them; a value that cannot fit is refused.
- **An indexed patch command reads its control-point indices as Metal's did**, with its pipeline's
  control points and index type; an index past its buffer reads 0.
- **An indirect render command's index and factor buffers must be resident**: an execution resolves
  them against the resources the encoder made resident, as its kernel buffers.
- **A kernel's patch command scales its factors** by the scale its `draw_patches` passes, as Metal
  does.
- **A render execution records its pass at commit**, from the records as they stand then, as
  Metal's does.
- **A command's refusal is one typed error wherever it is recorded**: the CPU's records are checked
  as `commit` records them, and `commit` returns what a command refuses.
- **A render execution's range read from a buffer is read before its pass**
  (`execute_commands_in_buffer_indirect`).
- **An indirect command's buffer outside the resident resources misses**: its Metal address
  resolves against the buffers and heaps the encoder made resident; outside them the slot reads
  zero, drops writes, and the execution report counts a resident-set miss. Metal leaves the access
  undefined.
- **Indirect commands the GPU may write run in the large-buffer form**: buffer slots at device
  addresses with in-shader bounds checks (`VariantKey::large_buffer_form`), so no buffer a command
  binds needs another compile. Inside a slot's window every access is the CPU form's; outside it,
  two contained answers differ, neither defined by Metal.
- **A blit copy between indirect command buffers of different descriptors copies the commands**;
  a record the destination's descriptor does not allow runs nothing.
- **Render states an indirect command buffer leaves to its commands are the encoder's** on a device
  that does not claim Apple10.
- **A function that sets a render state on a render command is not made** (a kernel's
  `render_command::set_cull_mode` and its kin): the pipeline state is refused, as Metal refuses it
  without Apple10.
- **An indirect command buffer's `allocatedSize` is Metal's** (its layout in 16-KiB pages), not the
  runtime's records' footprint.

**Blits**

- **Multisampled blit copies between depth or stencil and colour formats** copy each sample's
  bytes as plain Metal does (its validation stops them), through `VK_KHR_maintenance8` or
  sample-rate draws. A 16-bit depth crosses as its codes, exact where the driver keeps every code
  (`DriverFact::FragmentDepthKeepsDepth16Codes`: lavapipe's measured releases, and MoltenVK 1.4.2
  on the Apple M3 Max) and refused where it is not measured to. A copy into a 32-bit float depth on
  a driver not measured to store fragment depths unchanged is drawn and announced
  (`drawn-depth-copy`).
- **A blit copy of 24-bit depth between textures of different formats is refused**: Apple's
  reference GPUs make no `Depth24Unorm_Stencil8` texture, so there is no Metal answer to match.

**Render passes**

- **A texture over a buffer that a pass stores `DontCare` keeps the buffer's bytes**, in every
  texel, with or without memory barriers between the pass's draws: Metal's answer, but that the
  first pass in a process to overflow Apple's tiler (some 2 to 4 million triangle-tile entries)
  leaves the rendered texels in every texel, which no call can predict or detect. The runtime does
  not model the overflow, but where it hands the attachment to Metal's own `DontCare` (MoltenVK),
  which decides it. A pass that renders such a texture in place otherwise announces
  `dont-care-store`.

- **Two mesh draws of one pass with no barrier between them run in order.** Where a later draw's
  object or mesh function reads a buffer an earlier draw wrote, and no `memoryBarrier` stands
  between them, Metal orders nothing: on an M3 Max (macOS 27.0.1) the reader read the host's words
  in 30 runs of 30, through an indirect command buffer's two commands. The runtime's emulation of
  mesh draws (MoltenVK, ANV on the Iris Xe) records the reader's dispatches after the writer's, so
  the reader reads the writer's words every run. Nothing is slower or approximate for the order, so
  no notice says so, and every field the pass does determine is Metal's: the conformance case
  `mesh-commands-unordered-draws` compares those exactly and keeps Metal's own texels beside them,
  for the next macOS to be read against.

**GPU hangs and faults** (a draw of private primitive type 9, `DrawPrimitiveType::Nine`)

- **A hang reads `Scheduled` 5.06 ms after its commit and completes no sooner than 27.5 ms after
  that and the GPU reached it**, and its queue runs nothing of what it schedules meanwhile: one of
  Metal's two outcomes. On an M3 Max (macOS 27.0.1, 33 hangs) Metal's hang read `Scheduled` 0.68
  to 13.2 ms in and completed about 27.6 ms after; the queue's later command buffers' work mostly
  ran on the GPU during the hang where it read `Scheduled` within 2 ms (17 of 18), and mostly did
  not where later (56 of 62 ran nothing or in part). Either way they read `Scheduled` from the
  hang's on, as they are committed or their event waits are met, and complete with the hang, its
  signals firing, as the runtime's do (0.04 to 1.5 ms after it on MoltenVK, Metal's 0.01 to
  0.03 ms).
- **A page fault or a victim of the GPU's recovery runs nothing from its encoder on, and
  completes no sooner than 6.86 ms after it read `Scheduled` and the GPU reached the encoder**, its
  event signals firing, the encoders before it run, and its queue's command buffers committed
  meanwhile reading `Scheduled` as they are committed, and running: Metal's steady state. On an M3
  Max (macOS 27.0.1, 164 command buffers: type-9 draws and executions of indirect commands under a
  state that cannot inherit) Metal completed it 6.86 to 10.3 ms after `Scheduled`, and whether the
  encoders after the faulted one ran was a race the GPU's state decides: lost to its recovery
  where the GPU ran just before (127 of 142), run where it was idle before (the first fault after
  3 s idle), and run in part in some shapes (an encoder after an event signal, a command buffer
  committed right after). The runtime does not model the GPU's idle state.
- **A fault that later failing encoders follow fails with the fault's error, in their time**: it
  reads `Scheduled` no sooner than 6.86 ms after the GPU reached the fault, and completes no
  sooner than 6.86 ms after that for each later fault, then 27.5 ms for a hang, past which
  nothing adds; its queue's command buffers committed meanwhile run nothing where a hang follows
  the fault, as during a hang. On an M3 Max (macOS 27.0.1, 88 command buffers) Metal's read
  `Scheduled` 14.3 to 21.0 ms after its commit, once the GPU recovered from the fault, and
  completed 25.8 to 29.1 ms after that where a hang followed the fault. One that hangs first fails
  as the hang alone does (30 of 30), the GPU never reaching the fault.
- **A queue's second caused GPU error revokes it, as Metal's does**, and only an event wait held
  across the revocation is answered unmeasured. Measured on the M3 Max (macOS 27.0.1, about sixty
  command buffers): page faults and hangs (`CommandBufferError::PageFault`, `GpuHang`) count
  against their queue, victims of the GPU's recovery (`GpuRecoveryVictim`) do not; from the
  completion of the queue's second counted command buffer, every one committed on that queue fails
  at its commit with `CommandBufferError::AccessRevoked` (`MTLCommandBufferErrorAccessRevoked`, 4),
  running nothing and firing none of its event signals, for good (two seconds on too), while one
  committed before it runs; the device's other queues, new ones too, run on. The runtime counts the
  faults and hangs it makes Metal's answer for (primitive type 9, an execution of indirect
  commands under a state it cannot inherit) and those a program commits as failed in their place
  (`commit_failed` with Metal's code 3 or 2, `CommandBufferError::Injected`), not the ones the
  driver relays, and decides at the commit: a command buffer committed before the revocation runs
  whatever its event waits, a predicated loop's runs again included (on the M3 Max a loop
  committed before the revocation ran all its passes after it, with no error).

**Heaps and memory**

- **A heap's size and alignment answers are the device's own**
  (`Device::heap_texture_size_and_align`): the requirement of the image the runtime makes. A
  placement heap's `newTextureWithDescriptor:offset:` is held to Metal's own alignment, not that
  answer, for a texture whose image lies in memory of its own (a shadow of the heap's bytes, a
  multisample texture's samples in Metal's tiles): no image binds to the heap's memory there, so
  every offset Metal makes the texture at makes it here, and such a texture is held to no
  alignment bound of the heap's memory base either, whatever alignment it answers. A texture an
  image of does bind into that memory is refused `NotImplemented(HeapTextureAlignment)` where its
  answer passes the base (256 KiB, the largest image alignment measured).
- **A heap's texture answers the size that placed it** (`Texture::allocated_size`, Metal's
  `allocatedSize`): the size query's answer for its descriptor, a view's its texture's, as Metal
  answers; `None` for a texture outside a heap, whose Metal answer the runtime does not model.
- **Metal's own heap answer, for bookkeeping** (`Device::metal_heap_texture_size_and_align`, no Metal
  counterpart): what plain Metal answers, for a consumer that lays out a guest's heap.
- **Where Metal's answer is not its image's, every device answers Metal's**, for the few formats
  the runtime holds in an image of another layout.
- **A multisample texture answers at least Metal's size and alignment.**
- **Metal's answer by usage**: Metal's size depends on whether it compresses the texture, which
  follows its usage bits and shape; the bookkeeping answer follows the measured rule.
- **Apple's private texture usage bits are kept** in `TextureUsage` (`NO_CONCURRENT_ACCESS`,
  `VIDEO`, `BLOCK_WRITES_ONLY`).
- **A heap larger than the device's `maxBufferSize` is refused**: a heap's buffers live in one
  `VkBuffer` over the heap's allocation.
- **A `Shared` heap's textures are linear images** where the device's optimal images take no
  host-visible memory.
- **A sparse texture on a device without shader residency holds all its texels' memory.**
- **A heap's buffer and texture over the same bytes alias them, in the device's layout**, not
  Apple's tiled or compressed one.

**Kernels**

- **An imageblock's length rounds its cell as an M4 does**
  (`ComputePipelineState::imageblock_memory_length_for_dimensions`), never less than what a
  dispatch counts against the device's threadgroup memory.
- **A recursive call past the call stack yields zero**: the stack holds what Metal's byte model
  holds for the pipeline's `max_call_stack_depth`; a call that finds it full returns zero, where
  Metal's answer is corrupt.
- **A simdgroup barrier in divergent code waits for its tangle**, as SPIR-V defines it. The Metal
  Shading Language requires every thread of the SIMD-group to execute a `simdgroup_barrier` that
  any of them executes in a conditional or a loop iteration, so Metal defines no answer there.

**Process exit**

- **On Linux, an exit while the driver runs the runtime's work ends the process at once** with the
  exit's status and one line on standard error: the exit handlers registered before the runtime's
  last registration (a terminal's restore, say) do not run, where Metal's exit runs them.

**Two-plane YCbCr textures**

- **A view of an `_sRGB` twin as a 2D array reads its luma codes**, as its converted format's
  view does, where Metal's reads change from run to run.

**Padded 10-bit textures** (`R10Unorm_X6`, `RG10Unorm_X12` and their `_sRGB` twins)

- **A texture of one in memory of its own round-trips its words at every extent.** Metal makes
  such a texture from every creator, `newSharedTextureWithDescriptor:` among them in the `Private`
  storage that is the only storage any shareable texture is made in, and the runtime makes it from
  each. Metal's own `replaceRegion:` and `getBytes:` on it
  are exact only while it keeps the texture linear, and are silent no-ops where it picks its
  compressed layout instead (`allocatedSize` 128 bytes or more beyond the tight linear size: on an
  M3 Max under macOS 27.0.1, 16x16 and 64x64 under `ShaderRead` alone, where `getBytes:` left its
  destination untouched and a shader read 0 at every texel after a `replaceRegion:`; exact at every
  height to 8 at widths to 1024, and at 16x16 once the usage included `ShaderWrite`). A blit either
  way moves the texels in both layouts. Which extents Metal compresses is an undocumented, per-chip
  choice, so the runtime moves the caller's bytes at every extent rather than predict it.

**Rasterization-rate maps**

- **A map of no layers refuses its coordinate maps, and a mutable map not yet reset its
  physical-to-screen map**, where Metal ends the process. So does the screen-to-physical map of a
  map whose screen width or height is 0, which plain Metal compiles and whose physical-to-screen
  map it answers (an M3 Max under macOS 27.0.1: a SIGSEGV at the first such call on 0 by 64, 64 by
  0 and 0 by 0 alike).
- **A screen past 16384 pixels a side is refused**, where Apple's compile crashes or returns a
  corrupt size.
- **A parameter-data copy Metal makes unsafely is refused**: into a `Private` buffer, or past the
  buffer's length.
- **A physical size past the map's layers has width 0**, but after a reset to fewer layers, which
  keeps the earlier layers' widths, as Metal does. Past layer 1 Metal reads the map's other fields and the
  memory after them (an M3 Max: at layer 2 the vertical extent, at 3 and 4 the screen's sides), and
  a layer near the top of the range ends the process.
- **A mutable map's pass stops at the physical size**, where Metal rasterizes past the screen edge
  (an M3 Max: a 64-pixel half-rate map's pass to 49 by 49 pixels); announced with
  `rasterization-rate-map`.
- **A pass whose map is reset to another screen after it took the map keeps the screen it took**,
  as Metal's does (an M3 Max: over a reset to the identity, it shades to the new screen's 32-pixel
  tiles where the new screen spans one tile fewer, else the whole screen taken); over a map reset
  to one that is not the identity, which no probe measured, it draws by the map as reset within
  the screen taken, announced with `rasterization-rate-map`.
- **A rate-mapped pass's clear covers its attachments**, where Metal's covers the physical size
  rounded up to 32-pixel tiles and leaves the attachments' other pixels as they were (an M3 Max: a
  16-pixel identity map's pass into a 64-pixel attachment cleared 32 by 32 pixels).
- **A rate-mapped pass draws region by region, and refuses a cutting scissor.**
- **A pass over a map of a zero screen extent is refused**, `RegionError::RegionExtent` on that
  axis: the layer has no physical pixel there, so no attachment of the map's physical size can be
  made, and what Metal rasterizes in such a pass is not measured.
