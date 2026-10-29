# Integrating metal2vulkan

The checklist around the library: the tools each piece needs, caching, failures, isolating a
compile, host memory a driver refuses, and checking an integration. The API itself is
[API.md](API.md); the runtime's is [its README](../crates/metal2vulkan-runtime/README.md).

## 1. Tools

The libraries read AIR bitcode (a metallib, a bitcode wrapper or a raw module) and run no external
tool. The command line (feature `cli`) runs two, both optional:

- `llvm-as` (LLVM 21 or newer) to assemble an LLVM text input (`.ll`). Not needed for a metallib
  or bitcode.
- `spirv-val` (SPIRV-Tools) to check each module it writes. When it is missing the tool says so and
  still writes the modules, which the library has verified itself.

Put them on `PATH`, or name each one:

```sh
export METAL2VULKAN_LLVM_AS=/path/to/llvm-as
export METAL2VULKAN_SPIRV_VAL=/path/to/spirv-val
```

The runtime opens the Vulkan loader when it makes a device (or takes the consumer's
`vkGetInstanceProcAddr` when it adopts one). Nothing links against Vulkan at build time.

## 2. Compile

From the command line, see the [README](../README.md#command-line). From Rust, [API.md](API.md)
walks the flow in the order a consumer meets it:

1. [A device](API.md#1-a-device): describe the Vulkan device as created and choose a policy.
2. [A library](API.md#2-a-library): parse a metallib or AIR bitcode.
3. [A compute pipeline](API.md#3-a-compute-pipeline) or
   [a render pipeline](API.md#4-a-render-pipeline), under a `Budget`.
4. [The interface and the plan](API.md#5-the-interface-and-the-plan): the layout, what each slot
   binds, what a dispatch or draw needs.
5. [At bind time](API.md#6-at-bind-time): descriptor writes and push-constant bytes from what is
   bound.

Every `air.simd_*` operation keeps Metal's 32-lane simdgroup; there is no option to use the
driver's subgroup width instead.

A `half` computed in `float` rounds through the device's conversion to 16 bits and back, Metal's
`half(float)` under the float controls the entry declares, where the driver keeps that pair (ANV,
RADV). Elsewhere it rounds through the device's own conversion, or its `OpQuantizeToF16` alone
without `shaderFloat16` (which may flush a value below the smallest normal `half`), and the
pipeline records `NativeHalfConversion` or `NativeHalfQuantize`. NVIDIA's driver (615.71.09)
folds the pair (`DriverWorkaround::HalfConversionPairFolded`), so it takes the native conversion.
A `half` read from memory or from its bits is the device's widening on every device (an
`OpFConvert` of the bits read as a 16-bit float with `shaderFloat16`, `GLSL.std.450
UnpackHalf2x16` without it), and a `half` stored or reinterpreted as bits, unless they are the
bits it was loaded from, the device's narrowing (an `OpFConvert` to 16 bits, `GLSL.std.450
PackHalf2x16` without `shaderFloat16`); the pipeline records `NativeNanEncoding` (a NaN's bits)
and, where that conversion is neither under the float controls nor measured to be Metal's,
`NativeHalfConversion` (a denormal the device may flush).

To read AIR without translating it (list a metallib's functions, decode a module into typed IR,
read its entry points' bindings, canonicalize an entry under function-constant values), depend on
[`metal2vulkan-air`](../crates/metal2vulkan-air). Use the same version as `metal2vulkan`, which
pins it exactly; the translator's input errors are that crate's types.

## 3. Cache

The translator keeps no cache; it gives you keys and encodings to keep one
([API.md section 7](API.md#7-caching)):

- A key is a SHA-256 of the inputs, `cache::SEMANTICS_VERSION` and the device's profile
  fingerprint, known before compiling.
- A decode checks the format and semantics versions, the key, the profile fingerprint and a
  checksum. Every `ArtifactError` is a cache miss: compile again and replace the entry.
- The versions move independently: the crate's (the API), `SEMANTICS_VERSION` (the output; a bump
  misses every cached artifact), `ARTIFACT_FORMAT_VERSION` (the encoding) and `RUNTIME_VERSION`
  (what the runtime does with an unchanged artifact).

### The runtime's shader cache

The runtime keeps every shader it translates on disk, as Metal programs get from the OS, so a
second process does not translate it again. It cannot be turned off.

- **Where.** By default `~/Library/Caches/metal2vulkan` on macOS, `$XDG_CACHE_HOME/metal2vulkan`
  elsewhere, or `~/.cache/metal2vulkan` when `XDG_CACHE_HOME` is unset. `ShaderCacheOptions`
  (in `DeviceOptions`, or `RawDevice` for an adopted device) names another directory. Several
  processes and builds may share it; a build reads only the translations of its own translator
  (its version, `SEMANTICS_VERSION`, `ARTIFACT_FORMAT_VERSION` and translator sources), so an
  update that changed only the runtime keeps every translation.
- **How much.** At most `capacity` bytes on disk (25 GiB by default), least recently used first;
  another build's entries go after 30 days unused. In memory, at most `memory_capacity` bytes
  (256 MiB by default): a pipeline state made again takes its translation from there without
  reading the disk.
- **The driver's pipeline cache.** A device keeps one `VkPipelineCache` per record, made by the
  first pipeline making that names that record -- from the record where one is there, else empty --
  and handed to every later making of the same name until the device goes. A record is named by
  what the driver's conversion of a stage reads (its SPIR-V, layout, specialization and the state
  that reaches the conversion, with the driver's identity) and nothing else, so a state's sweep
  over blend factors, write masks or the topology's class is one record and one cache: its stages
  are converted and compiled once. The data is written back only when a making added to it. After
  an update of metal2vulkan, a pipeline whose SPIR-V and state did not change is made from its
  record; a driver update starts empty. A device keeps at most 256 of these caches; a program with
  more record names than that makes the rest a cache for the one making, as every making once had.
  On drivers that report `pipelineBinaryInternalCacheControl` (NVIDIA's), a device the runtime
  makes turns the driver's own internal cache off; an adopted device keeps what its creator chose.
- **Written in the background.** A creation hands its entries to the device's cache thread and
  does not wait for the directory: the same bytes under the same name, committed by the same
  rename, after the call returns. An entry still queued answers that device's own lookups, and the
  queue is written out before `Device`'s drop returns, so a second process finds everything the
  first one's creations handed over. A process that ends without dropping its devices loses what
  is still queued, which costs the next process those translations and nothing else.
  `Device::shader_cache_statistics` counts the entries queued, those a creation wrote itself past
  the queue's bound, and the bytes written.
- **When it fills or a write fails.** Each device says so once, through its `MessageSink` (or on
  stderr) as a `Warning`: `id_name` `metal2vulkan-shader-cache-evicted` when the directory reaches
  its cap, and the degrade notice
  `metal2vulkan-degrade-shader-cache-entry-not-written` (`DebugMessage::degradation`
  `Degradation::ShaderCacheEntryNotWritten`) when an entry cannot be written. Neither ends any
  work. Match on `id_name`, never on the text. Later events are counted in
  `Device::shader_cache_statistics`.
- **Failures.** Making or adopting a device fails with `Error::ShaderCache` when the directory
  cannot be made or written. After that, a failed read or write only costs a translation.
- **Cold runs.** To measure with nothing cached, give the device an empty directory, and turn the
  driver's own shader cache off (`__GL_SHADER_DISK_CACHE=0` for NVIDIA,
  `MESA_SHADER_CACHE_DISABLE=true` for Mesa). Give each test its own directory.
- **It is a cache.** Deleting the directory, or any file in it, is always safe.

## 4. Handle failures

Every failure is a typed value; match variants, never messages
([API.md section 8](API.md#8-refusals-requested-inputs-and-precision)).

- A refusal names what would change the answer: the feature the device lacks, the pipeline input
  the compile needs, or a limit.
- A deadline or a cancellation (`Error::Budget`, `Error::Cancelled`) is transient: retry later.
  Every other error is the answer for that input and device and may be cached as one
  (`error.retry()`, `error.is_permanent_for()`).
- Never use a module from a compile that returned an error.
- The command line prints `metal2vulkan: FALLBACK (<why>)`, exits 2, and writes a repro bundle
  (the input, the profile, `repro.sh`, `failure.txt`) under `$TMPDIR/metal2vulkan-repros`, or
  `METAL2VULKAN_REPRO_DIR`.

## 5. Bound a compile

Every compile runs in-process under the caller's `Budget`: a wall deadline (`Budget::standard()`
is 20 seconds), a `CancelToken` any thread may trigger, a deterministic work-unit cap and a byte
ledger charged before each input-sized allocation. A panic inside a compile becomes
`Error::Internal`. That is enough for a consumer that trusts its process.

When an abort, a stack overflow or a failed allocation in the compiler must not take the consumer
down, or a hard memory ceiling must hold, compile in a child process:

1. Spawn one worker per compile in its own process group, with the input in a file and the result
   written to a file. Send the library bytes, the entry and descriptor, and the device description
   and policy (with feature `serde` they serialize; equal description and policy give the same
   profile fingerprint and keys).
2. In the child, install a counting global allocator that refuses allocations above a budget, build
   the `Device`, and compile with the same `Budget`.
3. In the parent, poll the deadline, cancellation and the process group's resident memory. On a
   breach, kill the whole group and remove its scratch files. A result that arrives late is a
   timeout.
4. Load the result with `PipelineArtifact::from_bytes` against the parent's own `Device`, and check
   that its key is the one the parent computed (`Device::compute_pipeline_key`,
   `render_pipeline_key`).
5. Classify as in-process: a typed error is the answer; a deadline, cancellation or killed child is
   transient.

The validation crate's worker (`crates/metal2vulkan-validation/src/worker/`) implements this,
under 20 seconds and 500 MiB per attempt.

## 6. Host memory a driver refuses

`Device::new_buffer_with_bytes_no_copy`, `new_heap_with_bytes_no_copy` and the
`rebind_bytes_no_copy` calls import the caller's pages by their pointer
(`VK_EXT_external_memory_host`). Some drivers refuse the pointer of shared memory: RADV (AMD on
Linux) imports a pointer through amdgpu's userptr, which takes anonymous pages only, so it refuses
`MAP_SHARED` mappings of a `memfd`, a shared memory object or any file (QEMU's
`memory-backend-memfd,share=on` guest RAM among them) with `Unsupported`, `HostMemory`. NVIDIA and
lavapipe import the same memory.

A consumer whose memory is such a mapping passes `HostMemory::with_dma_buf(pointer, source)` instead
of `HostMemory::new(pointer)`. The runtime still imports the pointer first, exactly as without a
source, and calls the source only where the device refuses it: for dma-bufs of the pages the range
touches, and again once for each texture's image over its own pages in them. The rustdoc of
`HostMemory::with_dma_buf`, `DmaBufSource` and `DmaBufRequest` is the contract: when the source is
called, the chunks it answers, and what is refused. In short, answer each `DmaBufRequest` with
dma-bufs of exactly the pages asked for, in order, split only at multiples of its `granule`. Where
the granule is less than the length (64 KiB on RADV of Mesa 26.2.4, 4 KiB on 26.0.8), several
chunks are bound to one sparse buffer and must total the length rounded up to the granule: pad the
last with a scratch page of your own, repeated. A texture's request takes one chunk, of the same
file pieces the buffer's chunks hold there.

On Linux, `udmabuf` makes such a dma-buf of the file pieces behind the pages: open `/dev/udmabuf`
(the process needs read and write access to it) and issue `UDMABUF_CREATE_LIST` with one item per
run of contiguous file pages; the `memfd` must be sealed against shrinking (`F_SEAL_SHRINK`, as QEMU
seals it) and not against writing. The consumer knows the pieces behind every byte; the runtime runs
no udmabuf code. Each udmabuf takes at most `list_limit` items and `size_limit_mb` MiB (the module's
parameters under `/sys/module/udmabuf/parameters`, 1024 and 64 by default). Keep each chunk within
the host's limits as they are: read them, never assume them raised. Scattered memory needs many
items: a macOS desktop guest of 14 GiB was measured importing up to 63.3 MiB in one object, in up to
16,201 separate 4 KiB runs, with 19 to 29 of its imports past 1024 runs; at a 64 KiB granule such an
object takes 16 chunks.

A texture over such memory is an image over a dma-buf of its own pages where the device also lists
`VK_EXT_image_drm_format_modifier` and `VK_EXT_queue_family_foreign` (RADV does). RADV (Mesa 26.2.4
and 26.0.8) lays such an image's plane only at an offset in its page that is a multiple of 256
bytes, with rows a multiple of 256 bytes apart: keep a texture's offset and `bytes_per_row` so (a
4 KiB-aligned surface with 256-byte-aligned rows always is), with no byte between its rows (below).
Elsewhere the texture takes a shadow of the bytes, announced (`shadowed-texture`, naming why), and a
rebind that cannot make the image again is refused (`NotImplemented(RebindTextures)`).
`WorkCounts::dma_buf_imports`, `dma_buf_sparse_buffers`, `dma_buf_images` and
`dma_buf_image_fallbacks` count what the route did.

RADV lists every memory a device imported in every kernel submission, and amdgpu's work for a
submission grows with each entry, whether a command uses it or not; ANV on i915 lists every memory
in every execbuffer too, a pointer import (an i915 userptr) costing about three times an ordinary
buffer object. So the runtime releases a no-copy buffer's own import that only blits have read
(copies and fills of the buffer, copies between it and a texture) and no pending command buffer
holds, once it has stayed idle for its window (2 seconds at first, doubled at each import again, up
to 64 seconds), and imports it again at the buffer's next blit, on the thread that encodes it,
which pays that time. Nothing of Metal's observes the release: the buffer's bytes, its Metal
address and its pages stay, and the CPU's writes made meanwhile are read. Two kinds of import are
released:

- An import of dma-bufs. The new import comes from the same dma-bufs, never from the source, so the
  buffer keeps the pages it was made over even after `host_range_remapped`. The runtime holds a
  duplicate descriptor of each released chunk, within a quarter of the process's `RLIMIT_NOFILE`
  soft limit, read once at the process's first such import (an import past it stays imported,
  counted by `WorkCounts::host_imports_kept_for_fds`), and changes no limit: where many no-copy
  buffers stand at once, raise the soft limit toward the hard one at startup (`setrlimit`), as QEMU
  does.
  A blit whose released buffer the device cannot import again (out of memory) is refused.
- An import of the pointer, on a driver whose pointer imports follow their address
  (`DriverFact::PointerImportsFollowTheirAddress`: ANV, Mesa 26.2.4 on the Iris Xe of Raptor
  Lake-P; RADV, Mesa 26.2.4 on the Radeon 780M of Phoenix or Hawk Point and on the RX 7900 XTX; and
  lavapipe). There a live import reads, at each submission, the pages mapped at its pointer then,
  so the new import at the same pointer reads what the kept one would. A blit whose released buffer the device cannot import again is refused,
  typed as the first import would be. Elsewhere a pointer import stays imported.

A buffer stays imported for good once its Metal address is handed out (`gpu_address`, a binding, an
argument buffer or table, an indirect command buffer, `use_resource`), its `VkBuffer` is read, a
texture is made over its bytes, or a second buffer lies in its import; so do a heap's and a
rebind's imports. `WorkCounts::host_imports_released`, `host_imports_reimported`,
`host_imports_resident` and `host_import_fds_kept` count it. For imports of either kind,
`host_imports_read_apart` counts the live ones that a read other than a blit's keeps for good (the
`host_imports_read_apart_*` gauges by reason: the Metal address, the `VkBuffer`, a binding, a
texture, a second buffer, no releasing thread), the `host_imports_kept_*` gauges the live ones kept
whatever reads them, by cause (a heap's, a rebind's, a pointer import on an unmeasured driver, no
stand-in, and `host_imports_kept_for_fds`), and `host_imports_used` the imports each committed
command buffer's commands use, summed over command buffers; against `host_imports_live` they show
how many of the listed imports a workload's commands name. Of a rebind's imports,
`host_imports_rebound_unlisted` counts the live ones no command buffer committed since the rebind
used (no command reaches them unnamed yet), `host_rebind_imports_made` those rebinds made, and
`host_rebind_imports_covered` those whose pages lay within bytes a buffer over host pages lay over
already.

A texture over a no-copy buffer's host pages, imported by their pointer or as dma-bufs (or over a
buffer the runtime placed in fresh host pages), meets a rule Metal does not have. An image bound to
imported memory starts in `VK_IMAGE_LAYOUT_UNDEFINED` (`VUID-VkImageCreateInfo-pNext-01443`), and
its layout transitions "may perform read and write accesses on all memory bound to the image
subresource range"; every image takes one, its first move from the layout it is made in, before any
use, and the host could keep the bytes across it only by reading them before it and writing them
back after, which would undo a CPU write made meanwhile, where Metal keeps every write to bytes a
command buffer does not write.

By default (`HostPageTexturePolicy::InPlace`) the runtime relies on the driver there, checked. A
texture whose declared usage only samples or reads it is a linear image aliasing the pages wherever
the device's own check passed for that kind of pages (pointer imports, dma-buf imports, fresh
pages): the check makes such images while a CPU thread writes beside their texels and passes a kind
only where every first move kept every byte. It runs on a thread of its own, started by the first
such texture, once per driver, driver version and runtime release; with a shader-cache directory
its verdict is kept there and recalled as the next device is made, else it runs once per process.
Until it answers, a texture is a shadow (`WorkCounts::host_texture_shadows_check_pending`), so the
first frames of a first session pay the shadow's copies. A texture that renders or writes
(`RenderTarget`, `ShaderWrite`, `ShaderAtomic`), one the check failed or could not cover, and one
no image in place serves (a pitch the device's linear images do not take --
`host_texture_shadows_refused_pitch`; over dma-buf pages, a source that makes no one dma-buf of
the texture's own pages, as a chunked `udmabuf` import does -- `host_texture_shadows_refused_source`)
is a shadow (`shadowed-texture`, naming why): an optimal image of the runtime's own memory, its
texels copied in from the pages before a use and merged back after a write. `HostPageTexturePolicy::Shadow` (in `DevicePolicy::exact_answers`) keeps every such
texture a shadow and runs no check.

A texture over another buffer's bytes (the runtime's own memory) is a linear image over them where
its rows lie one after another (`bytes_per_row` is its texels' bytes, or it has one row; an array's
slices likewise), the device lays a linear image's rows at that pitch, and the image holds no byte
past its texels; it starts `VK_IMAGE_LAYOUT_PREINITIALIZED`, which keeps its bytes. A texture whose
rows lie with bytes between them is a shadow (`shadowed-texture`), whatever its usage: an image at
that pitch holds the bytes between the rows in its subresource (`VkSubresourceLayout`: "size
includes any extra memory that is required based on rowPitch"), which its layout transitions and a
render pass's load and store operations ("read and write back any memory locations within the image
subresource bound for that attachment"; lavapipe writes back the bytes its 4-texel blocks reach
past a row, a signaling NaN's bits quieted) may write, where Metal leaves them as the program writes
them.

A blit that names a texture over a buffer runs on the buffers' bytes, a region per row where its
rows lie apart, and a buffer-texture copy in a layout Vulkan cannot state moves its rows through a
staging buffer, as does a shadow's copy back where its texels start off their alignment (and a
rotated texture's, and a heap's two-plane or tiled texture's), whether an encoder records it or a
render pass between two of its instances. A driver may record each region as an operation of its
own: on ANV the 1080 rows of a 1920-texel `BGRA8Unorm` texture took 3.1-3.3 ms of CPU and 5.3-7.8 ms
of GPU on the Iris Xe of RPL-P. Where the device is measured to run it faster
(`DriverFact::RowCopiesDispatchFaster`: ANV, there 0.3-0.4 ms and 1.1 ms, and NVIDIA's on the RTX
5080 Laptop GPU, where 1080 rows took 2.0 ms of GPU as regions and 0.4 ms as one dispatch), a copy
of 2 rows or more, each the same whole number of 4-byte words and the same distance after the one
before, is one compute dispatch that writes the same bytes and no byte between the rows.
`WorkCounts::row_copies`, `row_copy_regions`, `row_copy_regions_most` and `row_copies_dispatched`
count these copies, their regions, the most in one, and those dispatched. `WorkCounts::copy_sites`
names the places in the runtime that recorded copy commands, each with its calls, their regions and
the most in one.

A driver's recording CPU for one region of a copy between buffers may grow with its size at some
sizes: on ANV's UHD 770 one region of 8,294,411 bytes took 28-40 ms, one of 8,294,400 bytes
0.006-0.035 ms. So every region over 16 KiB whose size is no multiple of 4 KiB is recorded as two,
its bytes up to the last whole multiple of 4 KiB and the rest, whichever part of the runtime
records it (`WorkCounts::copy_regions_split`): among them a two-plane YCbCr texture's planes copied
into its staging buffer where it converts whole (`WorkCounts::ycbcr_whole_conversions`), and a
staged texture's bytes and rows.

### Copied host buffers

A no-copy buffer whose import the runtime does not make is made over a copy of its bytes instead,
announced (`copied-host-pages`): no import maps the pages, so they take no kernel mapping. That
happens where:

- the caller's gate refuses the import: `HostMemory::with_import_gate(gate)` asks the `ImportGate`
  once, on the thread making the buffer, heap or rebind and holding no lock of the runtime's, just
  before the import, with the pages it would map (`ImportRequest`, the host pages from the first
  byte's through the last's). `false` makes the copy (`WorkCounts::host_imports_refused_by_host`,
  `copied_host_buffers_refused`). A consumer with a budget of mappings (Linux's
  `vm.max_map_count`) refuses there, paying in copies instead of a failed call. It is not asked
  for a buffer or rebind that lies in a live import of the same bytes, a heap's rebind, memory
  whose GPU writes Metal's private option bit 24 drops, or where the device refused the pointer
  before it;
- an import the device and the gate admitted fails (out of memory, another Vulkan error:
  `copied_host_buffers_degraded`).

The copy is host-visible memory of the device's own; the buffer's `contents` is still the pointer.
Each commit writes into the copy every byte the CPU changed in the pages since the copy and the
pages last agreed, and each batch's completion writes back every byte the GPU changed, before its
handlers run or a waiter returns, keeping a byte the CPU changed since (`copied_copy_in_bytes`,
`copied_merge_back_bytes`). Each compares every byte of every live copy, so keep copies few. A
texture over a copied buffer is a shadow of its bytes. Pages the process cannot read are never
copied: the buffer is refused, as their import is.

A no-copy heap's own buffer is copied the same way (`copied_heaps_made`): its buffers lie in the
copy, and its textures are shadows of its bytes. `Buffer::rebind_bytes_no_copy` gives the buffer a
copy of its new pages the same way (`copied_rebind_pages_made`), except over a buffer with a
texture in place over it, which the rebind refuses (`NotImplemented(RebindTextures)`).

## 7. Check the integration

`spirv-val` checks structural validity, not agreement with Metal:

```sh
spirv-val --target-env vulkan1.2 output.spv
```

`device.profile().spirv_validation()` names the environments and relaxations a module compiled
for the device is valid under; the command line runs each.

For a claim about behavior, follow [VALIDATION.md](VALIDATION.md): a synthetic regression test,
the golden output hashes where output must not move, and an authored case run on Metal and Vulkan
where it should.

A host note: RADV under Ubuntu 26.04's stock kernel 7.0.0-38 with linux-firmware 20260319
sometimes loses a GPU write into a newly made buffer under heavy multi-process GPU memory churn
(about 1 in 10^4 to 10^5, with no error); kernel 7.2.6 with linux-firmware 20260916 loses none.

A host note: on a laptop whose NVIDIA GPU powers down when idle (the driver's runtime D3, shown in
`/proc/driver/nvidia/gpus/*/power`), the first GPU work after about 15 seconds idle waits about
1 second while the GPU wakes (970 to 980 ms on an RTX 5080 Laptop GPU, driver 615.71.09). A host
that wants no wake can keep the GPU powered (`on` in its PCI device's `power/control`); the runtime
never needs it.
