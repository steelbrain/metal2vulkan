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
pipeline records `NativeHalfConversion` or `NativeHalfQuantize`; not `NativeHalfConversion` on
MoltenVK on the Apple M3 Max's macOS 27 identity (`DriverDeviation::MetalHalfConversion`), where
every such `half` is a conversion of a `float` the function holds or of a texel, measured to be
Metal's. NVIDIA's driver (615.71.09) folds the pair
(`DriverWorkaround::HalfConversionPairFolded`), so it takes the native conversion. RADV's compiler
fuses a `float` addition, subtraction, multiply or fused multiply-add into the conversion of its
result and rounds the exact result once (`DriverWorkaround::HalfConversionFusedWithArithmetic`), so
a pipeline that rounds such a result to `half` there records `NativeHalfConversion`.
A `half` read from memory or from its bits is the device's widening on every device (an
`OpFConvert` of the bits read as a 16-bit float with `shaderFloat16`, `GLSL.std.450
UnpackHalf2x16` without it), and a `half` stored or reinterpreted as bits, unless they are the
bits it was loaded from, the device's narrowing (an `OpFConvert` to 16 bits, `GLSL.std.450
PackHalf2x16` without `shaderFloat16`); the pipeline records `NativeNanEncoding` (a NaN's bits;
not on MoltenVK on the Apple M3 Max's macOS 27 identity, `DriverDeviation::MetalNanMoves`, where
the `half` moves through the registers alone or into a texture write's texel) and, where that conversion is neither under the
float controls nor measured to be Metal's, `NativeHalfConversion` (a denormal the device may
flush).

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
- **Pipeline records and warming.** A device with `DeviceOptions::pipeline_records` set hands
  that hook a `PipelineRecord` at each pipeline state creation, and keeps each library a record
  names in the directory once, named by its content hash, beside the driver's data: a record holds
  the descriptor and the libraries' hashes, not their bytes, so its size is the descriptor's. The
  libraries count against `capacity` with everything else; a record whose library was evicted
  names one the directory no longer holds. Keep a VM's records in the order they came
  (`to_bytes`), and at its next boot hand them to `Device::warm(records, byte_budget)` on a device
  over the same directory: one background thread makes their translations and predicted
  `VkPipeline`s before the guest asks, below every other making, until the driver data it added
  reaches the budget, and the guest sees nothing of it. Data a making finds in the directory adds
  nothing: where the boot before wrote every record's data, the pass adds 0 bytes and the budget
  does not bind. What it makes is what the guest's own
  creation of each descriptor makes on this build: a record holds a descriptor, never a
  translation, so records kept across an update are warmed by the new translator, and the pass
  reads no translation of another translator and no creation record of another
  `RUNTIME_VERSION`. A record it cannot read is counted and dropped, the device's first said once
  as the `Warning` `metal2vulkan-warm-record-unread`; nothing it meets fails or ends the run.
  `Device::warm_statistics` counts the records made, those that could not be (a stage the compile
  layer compiled already), the libraries written, each warmed record's outcome and the driver
  bytes added.
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
`memory-backend-memfd,share=on` guest RAM among them). NVIDIA and lavapipe import the same memory.
A no-copy buffer or heap over memory the device so refuses, made with no source of dma-bufs of its
pages (below), or with one that fails or whose dma-bufs the device refuses, is a copy of the
pages, announced ([Copied host buffers](#copied-host-buffers),
`WorkCounts::copied_host_buffers_device_refused`); a rebind to such memory without a working
source is refused with `Unsupported`, `HostMemory`.

A device that enables no `VK_EXT_external_memory_host` (a driver that lists none, or a device made
with it in `DeviceOptions::withheld_extensions`) but enables the dma-buf route imports no pointer
at all: memory made with a source is imported from its dma-bufs, its pages rounded out to the host
page. A no-copy buffer or heap over memory made without one, or with one that fails, is a copy of
the pages, as where a driver refuses the pointer above (`copied_host_buffers_device_refused`); a
rebind to such memory is refused (`Unsupported`, naming the extension), as is every no-copy object
on a device with neither route. A guest RAM there (`Device::use_host_ram`) needs a source too, and
its chunks are its dma-bufs; an object whose runs lie in one of its chunks
(`HostMemory::with_runs`) is bound there with no source of its own.

No driver that pins imported pages imports a mapping the kernel cannot pin (`VM_IO` or `VM_PFNMAP`:
a mapping of a dma-buf). NVIDIA's driver fails such an import as out of device memory and then fails
the process's later host-visible allocations too
(`RuntimeDriverDefect::UnpinnablePagesBreakHostMemory`), so on Linux the runtime asks the kernel
first on that driver (one `PROCMAP_QUERY` and one one-byte `process_vm_readv` per mapping) and
never hands it such pages (`WorkCounts::host_imports_unpinnable`): the source is called, and with
none, or where its import fails, the object is a copy of the pages, as where an import fails
([Copied host buffers](#copied-host-buffers)). Pages not mapped readable are refused.

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
file pieces the buffer's chunks hold there. A device that imports a dma-buf for a plain buffer but
reports none for a sparse one asks with a granule of the whole length: one chunk, which one
udmabuf must hold, else the object is refused. ANV of Mesa 26.2.6 on i915 is such a device: it
takes a view of up to `list_limit` runs through its dma-buf, and refuses the pointer of a mapping
of dma-bufs (`VK_ERROR_INVALID_EXTERNAL_HANDLE`), so a view of more runs, mapped through dma-bufs,
is refused there; mapped from its files, its pointer is imported.

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
A driver that lays such a plane out at another offset than its texels' in their page (Honeykrisp,
Mesa 25.3.6 on Apple's M2 Pro, answers offset 0 for a plane 8192 bytes into a 16 KiB page) gets an
image whose plane starts at offset 0, bound at the texels' offset of an import of the pages that
is no dedicated allocation, where it takes such a binding: the image requires no dedicated
allocation (RADV requires one of every such image) and its memory's alignment divides the offset
(`dma_buf_images_at_offset`). Elsewhere the texture takes a shadow of the bytes, announced
(`shadowed-texture`, naming why), and a rebind that cannot make the image again is refused
(`NotImplemented(RebindTextures)`). `WorkCounts::dma_buf_imports`, `dma_buf_sparse_buffers`,
`dma_buf_images`, `dma_buf_images_at_offset` and `dma_buf_image_fallbacks` count what the route
did.

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
- An import of the pointer, on a device whose held pointer imports add to every submission's cost,
  as the device checks once per driver, release and kernel (its import tax check, which holds 32
  imports of its own pages through timed submissions; RADV and ANV on i915 pay it). The new import
  at the same pointer reads the pages the kept one did: `HostMemory::new`'s contract keeps the
  caller's pages mapped there, with the same backing, while the buffer lives. Buffers made before
  the check answered keep their imports. A blit whose released buffer the device cannot import
  again is refused, typed as the first import would be. Where a held import costs nothing (NVIDIA's
  driver on an RTX 5080 Laptop GPU, 615.71.09: 2.1 to 2.2 us a submission with and without;
  MoltenVK 1.4.2 on an M3 Max), a pointer import stays imported: releasing it would save nothing
  and cost the next blit an import.

A buffer stays imported for good once its Metal address is handed out (`gpu_address`, a binding, an
argument buffer or table, an indirect command buffer, `use_resource`), its `VkBuffer` is read, a
texture that reads its memory is made over its bytes (a shadow, or an image in place in an import
of their pointer), or a second buffer lies in its import; so do a heap's and a rebind's imports. A
texture in place over a dma-buf import's bytes lies in an import of a dma-buf of its own pages,
asked of the buffer's source, and leaves the buffer's import releasable: on RADV a guest surface (a
no-copy buffer and a texture in place over it) lists the texture's import alone in each kernel
submission while the buffer is idle. `WorkCounts::host_imports_released`,
`host_imports_reimported`, `host_imports_resident` and `host_import_fds_kept` count it. For imports
of either kind, `host_imports_read_apart` counts the live ones that a read other than a blit's keeps
for good (the
`host_imports_read_apart_*` gauges by reason: the Metal address, the `VkBuffer`, a binding, a
texture, a second buffer, no releasing thread), the `host_imports_kept_*` gauges the live ones kept
whatever reads them, by cause (a heap's, a rebind's, a pointer import on a device whose held
imports cost no submission or made before its check answered, no stand-in, and
`host_imports_kept_for_fds`), and `host_imports_used` the imports each committed
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
texture whose declared usage samples, reads, renders into it or writes it through a shader (of
texels of 1, 2, 4, 8 or 16 bytes) is a linear image aliasing the pages wherever the device makes
one and its own check passed for that kind of pages (pointer imports, dma-buf imports, fresh
pages) and each of its accesses (a texture that renders, and one a shader writes, are checked
apart, and a shader's writes of 1- or 2-byte texels apart from wider ones'), on a
discrete GPU as on any other: the check makes such images while a CPU thread writes beside their
texels and passes a kind only where every first move (and render pass, for one that renders, and
kernel's storage writes, for one a shader writes) kept every byte. It runs on a thread of its own, started by the first such texture, once per driver and
driver version; with a shader-cache directory its verdicts are kept there, named by the driver and
the check's own version (not the runtime release), and recalled as the next device is made, else
it runs once per process. A texture made before its verdict is a shadow, made without waiting for
the check (`WorkCounts::host_texture_shadows_check_pending`), and moves in place once its kind's
verdicts pass, with no copy (`WorkCounts::host_textures_moved_in_place`; its shadow's image goes
once that commit's command buffer completed, `WorkCounts::host_texture_shadows_freed_after_move`), at a quiet commit: one
made while no other command buffer of the device is uncommitted, that waits for no event and binds
no texture by its ID, while none that does is still running (a program that never commits so
keeps them shadows). A program that makes such textures as it starts (a compositor's
framebuffers) and wants them in place from their first use on a driver's first run starts the
check earlier, for instance as it makes the device, with `Device::settle_in_place_checks` and a
zero timeout, which waits for nothing; the same call with a zero timeout later says whether the
verdicts are known. A program that routes each surface either to a texture over its pages or to a
copy of its own asks `Buffer::texture_placement` (or `texture_placement_with_iosurface_plane`)
with the arguments it would make the texture with: the answer (`TexturePlacement`: in place,
pending, a shadow and why, no host pages, another shape) is the making's own decision, so no report
string needs parsing. A texture a shader writes of texels of a width the check stores none of
(`host_texture_shadows_store_width`), one the check failed or could not cover, and one
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
them. A consumer that owns every read of such a buffer's bytes (a cache it fills and reads back by
rows of texels, say) may make it with `BufferDescriptor::with_padding_unobserved()`: a runtime
extension, no Metal counterpart, refused over the caller's memory (`NoCopy` contents). A texture
over it is then a linear image at its pitch wherever the device lays one out so, or else, where
the device makes `DRM_FORMAT_MOD_LINEAR` images (`VK_EXT_image_drm_format_modifier`), one at the
pitch where its in-place check passed for its pooled memory (for a texture that renders, the
check of renders that lets a pass change those bytes, as the declaration does, and no byte past the
last row's pitch) and, for a tracked texture over a `Private` buffer, its rows take no more bytes
than a bound past which such an image renders and samples slower than the shadow (over a `Shared`
or `Managed` buffer, and for an untracked texture, any size)
(`WorkCounts::pooled_pitched_images`; `TextureShadowReason::LinearCost` past it;
`HostPageTexturePolicy::Shadow` turns it off; one made before that check answered is a shadow
that moves to the image at a quiet commit once it passed, as one over host pages moves in place,
`WorkCounts::pooled_textures_moved_in_place`), its subresources holding each row's bytes after its
texels to its pitch (the last row's included) and nothing past the last row's pitch; those bytes
and the bytes between slices may change whenever the texture is made or used, and no other byte
does (`WorkCounts::unobserved_padding_aliases` counts such images; `Buffer::texture_placement`
answers `NotHostPages` for such a texture, else the shadow's cause, and
`Shadow(TextureShadowReason::BytesBetweenRows)` without the declaration).

Such an image is bound at the buffer's memory offset plus the texture's offset, which must be a
multiple of the device's linear-image alignment (`Device::minimum_linear_texture_alignment_for_pixel_format`
answers it: 256 bytes on RADV, 128 on NVIDIA, 64 on lavapipe, 16 on MoltenVK). The runtime starts
each buffer of its own memory on a multiple of 256, so a texture at offset 0 (or any multiple of
256) aliases the buffer where the device lays its rows out; one at another offset, as a decoded
frame's planes CoreVideo lays 32 bytes into their surface (and its second plane 32 bytes past a
multiple of 256 after the first's rows), is a shadow there. A consumer that makes such a buffer
for such textures (a device cache of decoded frames, say) may make it with
`BufferDescriptor::with_aligned_byte(0x20)`: a runtime extension, no Metal counterpart, refused
over the caller's memory (`NoCopy` contents). The buffer's memory then starts so that byte lies on
a multiple of 256 (at most 255 bytes more of it taken), and a texture there, and at any multiple of
256 bytes past it, aliases the buffer as one at offset 0 would. Its length, `gpuAddress` and
Metal's rules of a texture's offset in it stay Metal's; its contents (`contents`) start that
byte's remainder short of a multiple of 256, where Metal's start on a page. Such a buffer is never
parked for another making nor laid in fresh pages.

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

### Unmapping host memory

`VK_EXT_external_memory_host` leaves undefined whether an import of host memory holds its pages,
and so leaves them in the caller's care: "the application must ensure that the imported memory
range remains valid and accessible for the lifetime of the imported memory object"
(`VkImportMemoryHostPointerInfoEXT`), and a buffer bound to the memory may hold its payload until
the buffer is destroyed too (`vkFreeMemory`). So the pages under a no-copy buffer, heap or rebind
are unmapped, remapped or freed only once the runtime's imports of them are freed: the objects go
first, the pages after. The runtime tells when, for each object:

- With a keep-alive (`KeepAlive`, a runtime extension), its drop: after the `vkFreeMemory` of the
  object's import of the pages returned, on every `VkDevice` a recovery made the object on. The
  simplest form is a keep-alive that owns the mapping and unmaps it as it drops. A deallocator
  given with it runs at the release, as Metal's does, possibly while the import lives: that is not
  the moment to unmap.
- With a deallocator alone, its run: after that free.
- With neither, the drop of the last reference frees the import before it returns. A committed
  command buffer that used the object holds a reference until its completed handlers ran, after
  its status reads `Completed` and before its `wait_until_completed` returns; and a texture over
  the bytes, released while a command buffer submitted before may still read it, holds the import
  until that command buffer completed.
- A buffer made over exactly the bytes of another live buffer lies in that one's import: its
  keep-alive or deallocator goes at its release, and the other buffer's comes after the free. Pages
  under several objects go once each of them has signalled.
- A guest RAM (`HostRam`): its keep-alive's drop, once every chunk's import was freed.

An import is freed by the thread that drops the last reference to it; by a queue's thread once the
last command buffer holding it completed (one that used the buffer, a blit's hold on an import the
device releases while idle, or a texture's image over the pages); by a handler thread where a
keep-alive alone waits for the free; and, while the buffer lives, by the device's residency thread
(`metal2vulkan-residency`), which every half second frees the idle imports past their window that
the device releases (above), the next blit importing them again on the thread that encodes it. On
RADV and ANV every submission of the device, from any thread, lists every live import. So pages
unmapped before their objects go meet those threads at any moment, also where the objects are
dropped right after on another thread: on ANV a submission that lists an import of pages gone fails
and the device is lost, and on a Radeon 780M (Linux 7.3.3) an unmap of pages under a live pointer
import oopsed the kernel in amdgpu's notifier (`amdgpu_hmm_invalidate_gfx`), the GPU's work hanging
until the machine was rebooted.

### Copied host buffers

A no-copy buffer whose import the runtime does not make is made over a copy of its bytes instead,
announced (`copied-host-pages`): no import maps the pages, so they take no kernel mapping. That
happens where:

- the caller's gate refuses the import: `HostMemory::with_import_gate(gate)` asks the `ImportGate`
  once, on the thread making the buffer, heap or rebind and holding no lock of the runtime's, just
  before the import and before the runtime asks whether the process may write the pages, with the
  pages it would map (`ImportRequest`, the host pages from the first
  byte's through the last's). `false` makes the copy (`WorkCounts::host_imports_refused_by_host`,
  `copied_host_buffers_refused`). A consumer with a budget of mappings (Linux's
  `vm.max_map_count`) refuses there, paying in copies instead of a failed call. It is not asked
  for a buffer or rebind that lies in a live import or copy of the same bytes, memory whose GPU
  writes Metal's private option bit 24 drops, a rebind whose bytes moved in their page under a texture
  (below), or where the device refused the pointer before it;
- the device refuses the pointer of a no-copy buffer or heap, and no dma-buf of its pages serves
  it (no source given, or one that fails or whose dma-bufs the device refuses): RADV refuses shared
  memory, a `memfd` mapping among it (`Device::imports_shared_host_pointers`,
  `copied_host_buffers_device_refused`); a rebind to such memory is refused instead;
- an import the device and the gate admitted fails, of the pointer or of the dma-bufs of the
  caller's source (out of memory, another Vulkan error: `copied_host_buffers_degraded`);
- on NVIDIA's driver, the pages are a mapping the kernel cannot pin (a dma-buf's), which the
  runtime never hands that driver (`RuntimeDriverDefect::UnpinnablePagesBreakHostMemory`,
  `WorkCounts::host_imports_unpinnable`), and no dma-buf of the caller's source served them
  (`copied_host_buffers_degraded`);
- off macOS, the memory is of Metal's private option bit 24 (`ResourceOptions::PRIVATE_BIT`),
  whose GPU writes Metal drops: no import there keeps the GPU's writes out of the pages, so the
  copy is made without asking the gate and drops them (below; `copied_read_only_memory`). On macOS
  such pages are imported through a read-only view of them instead.

A buffer made over exactly the bytes (the same first byte and length) a live copied buffer was
made over lies in that copy, of either storage mode, as a buffer over a live import's bytes lies in
that import: no second copy, no read of the pages, the gate not asked
(`copied_host_buffers_shared`). So textures of one surface over a new buffer each frame are one
memory with the surface's other buffers while one of them lives, ordered as one resource's
accesses in one command buffer and across queues, as Metal orders them; a buffer made after the
last one over the bytes went makes a new copy. A rebind to such bytes lies in the copy where its
textures were planned at the copy's offset (its first byte) and no image in place rides on it. A
heap's own buffer, and memory whose GPU writes Metal's private option bit 24 drops, keep their own.
A copy and another copy or import of overlapping but different bytes agree GPU to GPU only
through the pages: a read committed before the write completed reads the bytes from before it
([FAST-PATHS](FAST-PATHS.md), known gaps).

`Buffer::host_backing` answers, per buffer, whether it got the copy: `HostBacking::Copied`, else
how its pages are imported (`ImportedByPointer`, `ImportedAsDmaBuf`, `InHostRam`), or `Detached`;
`None` for a buffer over no host memory of the caller's. A buffer placed in a no-copy
heap answers its heap's. The answer changes only at the buffer's (or heap's) rebind, which may move
it between an import and a copy, and at a detach. Unlike a `WorkCounts` delta, it does not race
with buffers another thread makes on the same device.

The copy is host-visible memory of the device's own; the buffer's `contents` is still the pointer.
Each commit writes into the copy every byte the CPU changed in the pages since the copy and the
pages last agreed, and each batch's completion writes back every byte the GPU changed, before its
handlers run or a waiter returns, keeping a byte the CPU changed since (`copied_copy_in_bytes`,
`copied_merge_back_bytes`). Until a command buffer that may write the copy is committed, the copy
itself holds the bytes the pages and it agree on, and nothing else is kept: a copy the GPU only
reads costs its own length and no more. At the first such commit the copy takes a base
(`copied_bases_made`): a 128-bit tag of each 1 KiB block in host memory (a 64th of the copy's
length), and the agreed bytes in coherent device-local memory the host maps where that heap is
1 GiB or more (a resizable BAR's, an RTX 5080 Laptop GPU's) and the runtime's blocks there and the
base stay within half of it, else in host memory, the copy's length again (an RTX 3090 or RTX 2080
without resizable BAR, whose 256 MiB window no base takes, and every device with unified memory)
(`copied_base_host_bytes`, `copied_base_device_bytes`). From then on a commit reads the pages'
blocks and a completion the copy's, each compared with its block's tag, and the base's bytes are
read only for a block both sides changed since they agreed (`copied_base_blocks_read`; uncached,
across the bus, where they lie in the device's memory, and then kept in 1 KiB of host memory for
the base's life, so a block both sides write at every frame crosses the bus once:
`copied_base_host_blocks`). A comparison that would read 32 such blocks or more reads them through
one copy into host memory on the device's staging queue, which waits for no work of the program's
(`copied_base_blocks_read_back`); a base lies in the device's memory only on a device with one. A
block whose tag agrees is taken as unchanged: a changed block is missed with a chance of at most
2^-64 at each comparison. A comparison of a copy with a base of 16 MiB or more splits its blocks
across up to four threads, the calling one among them, which end before it returns
(`copied_compare_threads`). A commit
that finds no host memory for a base fails with `Error::OutOfMemory` and runs nothing
(`copied_bases_refused`); the copy keeps none, and a later commit makes it. An event signal
submitted while a copy is live is seen at its batch's completion, after that write-back: no read
of the event's value shows it earlier, and a wait for it on another queue (a plain `Event`'s too)
is held until then rather than chained on the GPU (`event_signals_behind_merge`). Each
completion reads every byte of each copy its command buffers may have written
(`copied_merges_compared`, `copied_merges_skipped`): those their uses
wrote, and every copy where a function that may store reads addresses or an indirect command
buffer runs, so keep copies few. Where the pages' memory carries a log of the writes to them
(`HostMemory::with_dirty_pages`, `HostRam::with_dirty_pages`, a copy that drops the GPU's writes
included), each commit compares only the pages
the log reports written since the copy's last commit, and the bytes no log keeps
(`copied_copy_in_bytes_skipped`): it asks one stamp of each log for every copy and waits for its
publication (bounded) on the committing thread, and every copy compares every byte at its first
commit and where a log answers `false`. The runtime reports to the log what the log cannot see: a
copy's merge back, as the copy's own write, the GPU's writes into the pages through imports in
place, and its own CPU writes into them. A texture over a copied buffer is a linear image over the
copy, bound in its memory, wherever the device's in-place check passed for the caller's pages, for
the texture's accesses (`copied_textures_in_place`): in fresh pages the runtime imported, an image
made for host allocations, as over the pages, under the fresh pages' check too; in memory the device
allocates, the driver's own linear image where it lays the rows at the texture's pitch, which starts
`PREINITIALIZED` and so keeps the copy's bytes, else a `DRM_FORMAT_MOD_LINEAR` image at that pitch
under the check of the runtime's pooled memory too. Elsewhere it is a shadow of the copy's bytes, its
notice naming why. The copy reads the pages through the kernel
first, and where the kernel cannot read them, with plain loads where the process maps every page
with read permission: on Linux the kernel's read pins each page, which a mapping of a dma-buf
(`VM_PFNMAP`) does not allow. Pages not mapped, or mapped
without read permission, are never copied: the buffer is refused, as their import is.

Pages the process maps read-only (a VMM's alias of `replaceBackingWithRanges:readOnly:`, its
maximum protection lowered to read) are copied the same way on Linux, whatever the gate answers
and whether or not a dma-buf source was given, which is never called for them
(`copied_unwritable_pages`): Linux drivers pin an imported pointer's pages for writing, and a
dma-buf of the pages would take the GPU's writes. The copy drops the GPU's writes,
as Metal drops them: each batch's completion takes the bytes the GPU changed back out of the copy
and writes nothing to the pages (`copied_dropped_bytes`), so later commands of the same command
buffer read the writes back and a later command buffer reads the pages' bytes. The runtime asks
the system whether the process can write the pages as it makes the copy. The copy of memory of
Metal's private option bit 24 drops the GPU's writes the same way, whatever the pages' protection.
A command buffer committed before the writer's batch completed may still read the writes, where
Metal's never does, and a render pass's store-out into such a copy completes, where Metal's faults
its command buffer. The announcement's detail says so.

A no-copy heap's own buffer is copied the same way (`copied_heaps_made`): its buffers lie in the
copy, and its textures are shadows of its bytes. `Buffer::rebind_bytes_no_copy` and
`Heap::rebind_bytes_no_copy` give the buffer or heap a copy of their new pages the same way
(`copied_rebind_pages_made`). The heap's placed buffers lie in the copy at their offsets; its
textures bound to its bytes, and a texture's image in place over the buffer, are made again over
the copy (`copied_rebind_images_made`): where an image in place rides on it, the copy is fresh
pages the runtime maps and imports by pointer, and the image is made as over the caller's pages,
under the device's in-place check for fresh pages; else the copy is memory the device allocates,
and each image a linear image of it that starts `PREINITIALIZED`, so its first move keeps the
copy's bytes, or where the driver lays that image's rows at another pitch, a
`DRM_FORMAT_MOD_LINEAR` image at the texture's pitch, under the check of the runtime's pooled
memory. Textures held as shadows read the copy. A buffer's
rebind to bytes at another offset in their page than the old, under a texture planned at the old
offset, is a copy laid out at the old offset too, the gate unasked (`copied_rebind_pages_moved`).
A rebind is refused (`NotImplemented(RebindTextures)`) only where the device makes no such image
over the copy (its driver lays the linear image out at other pitches and makes no
`DRM_FORMAT_MOD_LINEAR` image at the texture's pitch that the pooled memory's check passed, or
either lies in memory of another type or alignment), or where the buffer's GPU writes are dropped and its bytes moved in their page.

A `Managed` no-copy buffer on a device without unified memory is never such a copy: it is two
copies, as on a Mac with a discrete GPU, the GPU's in device-local memory and the caller's pages
(`HostBacking::DeviceCopy`; the runtime's guide, **Host memory without a copy**), and it holds
the buffer's bytes of device-local memory for its life (`WorkCounts::managed_device_copy_bytes`).
Its pages are imported as above, the gate asked, as the transfers' source and target; where the
gate refuses, the device refuses the pointer or the import fails, they are reached through staging
buffers instead (`managed_staging_bytes`, at most 256 MiB staged ahead, none once their work
completed), no copy of them made and nothing compared at commit or completion. The GPU sees only
the CPU writes `did_modify_range` names, and the CPU sees the GPU's only through a blit's
`synchronize_resource`, as Metal's contract for `Managed` storage on such a Mac says, so a
consumer that wrote the pages without calling `did_modify_range` (relying on a log of their
writes) must call it. Pages the process maps read-only keep the GPU's writes out of them the same
way: the pages are copied in again at the completion of each command buffer that used the
buffer. A rebind whose import is not made, of a buffer whose first pages were imported, makes such
a copy of the new pages for the import (FAST-PATHS.md, known gaps).

### Guest RAM imported whole

A VMM whose device refuses views of guest pages made through dma-bufs (each import of scattered
pages then takes a kernel mapping per page), or whose driver lists every import in every
submission, can give the device the guest's RAM once instead:

- `unsafe HostRam::new(ranges, keep_alive)` takes the RAM's ranges of host memory (whole host pages,
  mapped readable and writable until the keep-alive goes) and `Device::use_host_ram(&ram)` uses it
  on a device, until the returned `HostRamUse` is dropped. The drop waits for the device's thread
  for the RAM (below), which stops within one 2 MiB step of faulting pages in: once it returned,
  the runtime reads, faults in and imports no page of the RAM, and only an import a buffer or a
  pending command buffer still holds keeps pages, until the keep-alive goes. A device may use
  several RAMs whose ranges are apart, and a new one over the same ranges once the earlier use
  ended; ranges that meet a used RAM's are misuse (`Invalid`, `Problem::HostRam`).
- `HostMemory::with_runs(resolver)` gives each no-copy object a `RunResolver`: asked once, at the
  object's making and at each `rebind_bytes_no_copy` to new memory, on the calling thread with no lock of the runtime's held and before the import
  gate, with the object's import window (whole host pages), it answers the host addresses of the
  same bytes in the RAM's ranges, as runs of whole pages in the object's byte order, or `None`.
  It is asked only on a device that uses a RAM, only where the object would import its pages
  itself, never for memory whose GPU writes Metal's private option bit 24 drops. An answer that
  does not cover the window in whole pages of one RAM the device uses is refused (`Invalid`,
  `Problem::HostRam`).
- One run inside one chunk: the device imports the RAM by host pointer in chunks of 64 MiB, never
  across a range's end, each once (`WorkCounts::host_ram_chunks_imported`, `host_ram_chunk_bytes`,
  `host_ram_chunks_live`). A device that refuses pointers to shared memory
  (`Device::imports_shared_host_pointers` false: RADV) imports each chunk of a RAM of shared memory
  (`memory-backend-memfd,share=on`) as a dma-buf where the RAM has a source of them
  (`HostRam::with_dma_buf`, below); with none, it imports no chunk and refuses each object over it
  (`Unsupported`, `HostMemory`), each copy over it goes through the CPU, whenever it comes
  (`copy_from_host_ram` and `copy_to_host_ram`, below), and the device's thread faults none of its
  pages in (`host_ram_chunks_faulted_ahead` counts the chunks it faults in elsewhere). From
  `use_host_ram` on, a thread of the device's imports chunks ahead of the objects over them (`host_ram_chunks_imported_ahead`): the chunk an object last took, then a
  window of chunks past it in the direction the takes go, up the RAM or down it, one chunk after a
  take that starts no run (and the RAM's first chunk before any take), four once a take lands inside
  the window or next to the chunk last taken. So a run of objects over consecutive RAM, up or down,
  as a VMM's 2 MiB heaps over fresh guest RAM are, finds its next chunks imported while it stays
  inside the window. An object over a chunk not imported yet (outside the window, past a burst's
  reach, or the chunk the thread is importing) imports the chunk as it is made, or waits for the
  import of it under way, and so does a copy over it that is not staged through the CPU (below):
  every object over a chunk lies in its one import. Such a chunk's import is of its host pointer,
  and `VK_EXT_external_memory_host` leaves the same host allocation imported twice into a device
  undefined (its issue 6), so no object's run, nor a copy's range, is imported by its own pointer
  beside the chunk's (`ObjectChurn::imports_inside_live` counts an import made over host
  addresses a live import holds). The price is a making that waits for its chunk's import where
  the window has not reached the chunk (the window bench below). Only a device whose chunks are dma-bufs
  (`HostRam::with_dma_buf`, below) imports an object's run alone while its chunk's import is made,
  as one more dma-buf. The thread faults in no more than the window past the objects, and never goes back for
  a chunk a take skipped. What an
  import costs is faulting in the chunk's fresh pages, not the pin. On an RTX 5080 laptop
  (NVIDIA 615.71.09, Linux 7.2), the import of 64 MiB already resident took 2 to 3 ms; the
  fault-in of 64 MiB of fresh `memfd` RAM under `MADV_HUGEPAGE` took 7.5 ms where free memory
  held whole huge pages, and 45 to 70 ms (at worst 0.34 s) with 2 to 3 GB free and the page
  cache filling the rest of memory, where the kernel reclaims and compacts memory for each huge
  page. There, 16 GiB imported whole ahead took 40 to 57 s, during
  which another thread's `mmap` waited up to 1.2 s for the memory map lock the fault-in held, and
  every process on the host stalled on memory for 6.6 to 9.6 s (pressure stall information).
  With the window, 2 GiB of 2 MiB heaps made in bursts of four chunks had 36 chunks faulted in
  ahead, no heap after the first two imported its own chunk, and the host stalled on memory for
  under 1 s; 16 heaps over scattered chunks had one chunk each faulted in ahead. With the window
  following the takes down the RAM, of 256 2 MiB heaps over fresh RAM with free memory holding
  whole huge pages (27 GB free), in two bursts of four chunks up, the same down, and 256
  scattered over 32 chunks 10 ms apart, 2, 8 and 15 makings each imported a chunk, the slowest
  8.2 to 8.7 ms (`tests/host_ram_window_bench.rs`). The thread is
  paced: it runs as the system's batch work (`SCHED_BATCH` on Linux, the utility class on
  Apple's systems), never at a weight the system starves, since it holds a chunk's lock and, as
  it pins, the process's memory map lock; it yields between chunks; and on Linux it faults each
  chunk's pages in 2 MiB at a time before the import pins them: shared memory of base pages (a
  VMM's `memfd`) by a read of each page, whose fault holds its mapping's own lock rather than the
  memory map lock, so no other thread's `mmap`, `munmap` or `mprotect` (a driver's compiler
  growing its heap) waits behind it; private memory, hugetlbfs and kernels before 6.11 by
  `MADV_POPULATE_WRITE`, which holds the memory map lock for each step. Each
  import is made with the device's memory lock released. The thread stops where one
  more import would leave the device less than a quarter of its `maxMemoryAllocationCount`, the
  rest left to the objects over them; once the device's allocations come within that quarter, the
  chunks it imported that no object took are freed again (`host_ram_chunks_given_back`). Only
  `use_host_ram` starts the thread: objects that import their own pages, by pointer or as
  dma-bufs, import nothing ahead. The
  object's `VkBuffer` is made as an import of its own pages would make it and bound in the chunk
  at its run (`chunk_bound_buffers_made`): no import of its own, no gate asked, its `contents`
  still the pointer, and its size reaching no page past its own, so neither its Metal range, a
  robust descriptor range nor a root entry reaches the rest of the chunk. A no-copy heap's own
  buffer, and a buffer's or heap's rebind, take the same path; a buffer's rebind to bytes at
  another offset in their page under a texture takes the copy laid out at the old offset instead.
- `HostRam::with_dma_buf(source)` gives the RAM a `DmaBufSource` of its pages. A device that imports
  dma-bufs then imports each chunk (and each object's run imported alone) as one dma-buf of it, from
  the source, where the device's own check finds that a held host-pointer import adds to the cost of
  every submission (the import tax, RADV's: amdgpu gets every page of every held pointer import
  again in each submission), or where it refuses the chunk's pointer (RADV refuses any shared
  mapping, a `memfd`'s among them); elsewhere (ANV and MoltenVK pay no such tax) the pointer, as
  without a source (`WorkCounts::host_ram_dma_buf_imports`). A dma-buf chunk costs a submission what
  one import does, whatever its length: on the Radeon 780M (Mesa 26.2.6, Linux 7.3.1) one held
  pointer import of 64 MiB added 24.1 us to every submission, and 100 held dma-bufs added 42 us
  (`runtime_lane::submit_per_import`). Each request asks for whole host pages inside one chunk of
  the RAM, by the RAM's own addresses, with a `granule` of its `length`: answer one dma-buf of
  exactly those pages (a `udmabuf` of the `memfd` behind them; `udmabuf`'s default `size_limit_mb`
  is 64, a chunk's size). A texture over an object bound in such a chunk is an image over a dma-buf
  of its own pages, as over an object's own dma-buf imports, asked of the same source. A source that
  fails, or a dma-buf the driver refuses, leaves the chunk to its
  pointer (`WorkCounts::host_ram_dma_buf_imports_failed`); a driver's refusal that lasts (of the
  dma-buf itself, not for want of memory) is remembered for the device's life, and its later
  chunks (and the copies' lists below) take their pointer without asking the source.
- `HostRam::with_largest_dma_buf(bytes)` says how long one dma-buf of the source may be (for
  `udmabuf`, its `size_limit_mb` parameter, read from `/sys/module/udmabuf/parameters/`). On a
  device whose held imports add to the cost of every submission, with a source, each listed
  import costs every submission: an 8 GiB RAM held as 128 dma-bufs of 64 MiB raised a
  submission's median from 14 us to 60 to 88 us on the Radeon 780M (Mesa 26.2.6, Linux 7.3.1;
  `runtime_lane::submit_per_import::submit_per_ram_dma_buf`), as 3 of 2730 MiB to 16 us. There
  `use_host_ram` first asks the driver whether it takes the source's dma-bufs: it asks the source
  for the RAM's first host page and imports that dma-buf, released at once (so that page is
  allocated in the `memfd` where the guest never wrote it), dedicated to nothing as a chunk's
  import is, or, where the driver refuses that, dedicated to the page's buffer
  (`WorkCounts::dma_buf_buffer_imports_dedicated`). Where the driver refuses it in both forms for
  a cause that lasts, or refused one so before, the RAM keeps its 64 MiB chunks, imported ahead,
  each by its pointer. Where
  it takes it, each chunk is as long as that, the device's `maxMemoryAllocationSize`,
  `maxBufferSize` and `maxStorageBufferRange` all take, in whole 64 MiB, and never longer than
  512 MiB (an 8 GiB RAM in 16 chunks at most): a chunk's import pins every page of it, and
  `udmabuf` allocates the pages of the `memfd` the guest never touched, which stay allocated after
  the chunk is freed, so a longer chunk makes more of the guest's untouched RAM resident. Without the call, or where a
  kernel caps it at 64 MiB, the chunks stay 64 MiB. On such a device no chunk is imported ahead. The first object over a chunk
  not imported asks a thread of the device's for it, and that thread imports it, which pins
  every page of the chunk (all of it resident, where 64 MiB chunks pinned only those the objects
  and the window reached; 660 to 692 ms for 2.7 GiB on the Radeon 780M and the RX 7900 XTX). That
  object, and every object over the chunk until the import is done, is bound in an import of its
  own run alone, one dma-buf of the run's pages from the same source, made on the object's thread
  (`run_bound_buffers_made`), so no making waits for the chunk; each is one more listed import
  (`run_imports_live`), a buffer's released after its idle window, as a no-copy buffer's own
  import is, and the buffer bound in the chunk at its next blit, and a heap's, or a buffer's whose
  address was handed out, kept while the object lives. Its uses order against those of the
  objects over the same bytes in the chunk, and of the copies of them, by their host addresses, as
  any two no-copy buffers over shared bytes do; on a driver whose barriers do not order blits
  through two imports of the same pages (`RuntimeDriverDefect::AliasedImportBlitsUnordered`) the
  object imports the chunk instead. The first kernel submission after a chunk's import waits while
  the kernel writes the chunk's page tables, and so does every other submission of the device
  (231 to 275 ms for a 3.94 GiB chunk on the Radeon 780M), so a chunk stays imported once
  imported where the whole RAM fits in half the device's memory heaps with unified memory, or in
  three quarters of its heaps that are not device-local without (an 8 GiB RAM does on the 780M
  and the RX 7900 XTX of a 27 GiB host). Each kept chunk adds about half a microsecond to every
  submission. Where the RAM does not fit, a chunk no buffer bound in it and no pending command
  buffer that used it holds for its idle window (2 s, doubled at each release of the chunk up to
  64 s) is released by the same thread (`host_ram_chunks_released`), and the next object over it
  asks for it again.
- `HostRam::with_dma_buf_of_runs(source, most_runs)` gives the RAM a `DmaBufRunsSource`: one
  dma-buf of a list of runs of its pages, the runs' pages one after another (on Linux,
  `udmabuf`'s `UDMABUF_CREATE_LIST`, whose list holds at most its `list_limit` parameter, 1024 by
  default: pass that as `most_runs`). Only the copies by runs below ask it, and only on a device
  whose held imports add to the cost of every submission and whose chunks of the RAM are its
  dma-bufs (RADV with `with_dma_buf`) and whose barriers order blits through two imports of the
  same pages. There a copy takes its runs of the chunks not imported by address across them,
  those that share a host page as one, and imports each list of at most `most_runs` of them (and
  at most a chunk's and `with_largest_dma_buf`'s bytes) as one dma-buf of the source
  (`WorkCounts::host_ram_run_list_imports`, their time in `host_ram_run_list_import_nanos`): a
  tile of 1,800 pages anywhere in the RAM is one list or two. Each list is kept past its command
  buffer for the copies after it: a copy takes a kept list for each of its runs that one of the
  list's runs holds whole (a device cache filled and downloaded again, a tile copied both ways) and
  imports nothing for those, lists of its own for the rest. The lists kept hold at most 64 MiB and
  are at most 16, the least recently taken let go of first; each goes once no copy took it for 2 s
  and no command buffer holds it, and all go as the device's use of the RAM ends
  (`run_imports_live` counts them). It neither imports those chunks nor asks for them, so copies
  pin the pages they copy while they run or their lists are kept, and none of the RAM the guest
  never touched; a chunk already imported serves its pages as before. Without the source a copy of
  more than 16 ranges of a chunk imports the chunk (or waits for its import under way) and keeps
  it: that pins every page of the chunk, up to 512 MiB, `udmabuf` allocating those the guest never
  touched, for good, and later copies over it import nothing. With it a copy over a chunk not
  imported pays its lists' imports where no kept list holds its pages. On the Radeon 780M (RADV,
  `tests/host_ram_listed_bench.rs`, a guest that wrote half its RAM in 2 MiB extents and scrolls
  tiles of 1,800 of those pages, each filled and downloaded, then one of the last 24 filled and
  downloaded again; then a 5 MiB device cache filled and downloaded 40 times each): the RAM's file
  held 4028 MiB of an 8 GiB guest at the end, where whole-chunk imports would have made all 8192
  MiB resident, and 8300 MiB of a 16 GiB guest, where they would have made 16384 MiB, the same with
  the lists kept as without. A tile laid as a surface's pages (454 to 620 sorted runs, one list)
  took 3.9 to 4.3 ms from the commit to the completion and 0.7 to 0.9 ms to encode where it
  imported its list, and 0.53 to 0.60 ms and 0.6 to 1.0 ms where it took the list kept from its
  fill; a tile of 1,800 pages each anywhere (two lists) 0.83 to 1.15 ms and 1.3 to 1.8 ms
  importing, 0.46 to 0.50 ms and 0.42 to 0.54 ms taking them. The 5 MiB cache filled again took
  0.37 to 0.43 ms and 0.2 to 0.4 ms (3.0 to 3.2 ms and 0.7 to 1.0 ms with no list kept); over an
  imported chunk 0.37 to 0.40 ms and 0.12 to 0.14 ms. The 24 tiles of the scroll that are filled
  again pass the 64 MiB the lists kept hold (about nine such tiles), so a third to a half of their
  fills took a kept list. A copy with a run no kept list holds looks through all of them: 48 to
  63 µs for a surface's tile against nine lists, 250 to 290 µs for 1,800 runs anywhere against 16.
  The GPU's own time was about 0.23 ms for a tile and 0.17 ms for the cache. Each request of a list
  is whole host pages inside one chunk, by the RAM's own addresses, by start and sharing no page,
  with a `granule` of its `length`: answer one dma-buf of exactly those pages in that order, its
  `length` their total. A source that fails, or answers a dma-buf of another length, leaves the
  copy to the chunk.
- Any other answer (several runs, one run crossing a chunk's end, `None`) takes the paths above:
  the gate, then the import or the copy. The runs are kept with the object.
- Elsewhere the chunks are never released while idle. Once the use is dropped, each is freed when
  nothing holds it (no buffer bound in it, no command buffer that used one), and the RAM's
  keep-alive is dropped after the last.
- `HostRam::with_dirty_pages` takes one `DirtyPages` log for the whole RAM, its offsets host
  addresses. A write the runtime reports over an object whose pages are many runs of the RAM
  comes to the log as one call (`DirtyPages::wrote_runs`, or `wrote_runs_as` for a copy's own
  write) carrying the host addresses of every run it reaches, in the object's order; by default
  those call `wrote` or `wrote_as` once a run.
- `BlitCommandEncoder::copy_from_host_ram(ram, runs, destination, offset)` gathers `runs` of a RAM
  the device uses into consecutive bytes of `destination` from `offset`, in the order given, and
  `copy_to_host_ram(source, offset, ram, runs)` scatters them back: for a surface whose pages are
  a scatter of guest pages, with no view of them and no import of its own. Each run is a host
  address range inside the RAM's ranges that starts on a 4-byte word and ends at any byte past
  it: a surface's whole pages, or each of its rows' texels alone, whose padding and edge pages the
  guest's CPU may own; a run that starts where the one before ended joins it, a descending
  neighbour stays apart, and each is cut where a chunk ends. Where every run also ends on a 4-byte
  word (whole pages, or the rows of a surface of 4-byte texels), the runs become a table of runs
  of words, at most 4 KiB each, in the command buffer's upload arena and one compute dispatch for
  each chunk they touch, over the chunk's own `VkBuffer` (more only past the device's storage
  range or workgroup count): never a copy region per run, and no other word of their pages read or
  written (`WorkCounts::host_ram_copies`, `host_ram_copy_bytes`, `host_ram_copy_dispatches`).
  Otherwise (a run that ends off a word, such as a row of 2-byte texels, or a scatter whose runs
  write a byte twice) each run's bytes in a chunk are one region of a `vkCmdCopyBuffer` between the
  import that holds them (whole pages, imported as for any copy) and the buffer, the regions of one
  import one command of at most 1024: no other byte of their pages is read or written, and of runs
  that write a byte twice the later's bytes stay (`WorkCounts::host_ram_copy_regions`). A copy
  takes each chunk it touches as an object over it does, and the thread's window moves past it.
  Over a chunk not imported yet, a copy on a device whose thread imports ahead (every device but
  one whose listed imports cost each submission: RADV with a dma-buf source) is staged through the
  CPU, however few its ranges: an import of a copy's ranges alone pins every page between ranges
  less than 2 MiB apart on the encoding thread (a 24 MiB cache of 2,477 runs over five chunks, its
  runs joined where less than 1 MiB apart, pinned 317 MiB in 61 ms on an RTX 5080 Laptop GPU, where
  the CPU copied the runs in 2.8 ms). A copy where the thread does not import, of at most 16
  ranges, imports those ranges alone, as an object's run is imported (not on MoltenVK or ANV). No
  staged copy makes or waits for an import, no page is pinned for it, and the thread imports each
  such chunk next, so the copies after it go through the imports.
  - A staged gather's work is its batch's host step, off the encoding and committing threads,
    once the batch is otherwise ready to submit (its event waits met): it takes memory of the
    command buffer's own, copies the runs from the pages into it by the CPU, and records the
    GPU's copy from there into the buffer (`WorkCounts::host_ram_copies_staged`,
    `host_ram_staged_fill_nanos`). A rebind before the first submission that encodes its encoder
    again copies the gather through the imports instead. Where a buffer or heap over host pages lies over the runs'
    bytes, or a scatter into those chunks was recorded, the CPU's copy first waits until every
    queue completed the work submitted before it (`host_ram_staged_fills_gated`), and where that
    is so at the encoding the copy takes the imports' path.
  - A staged scatter's GPU copy is recorded by its batch's host step, off the encoding and
    committing threads, into memory of the command buffer's own that the step takes; the work
    after it goes in a batch of its own, whose host step waits until every queue completed the
    work submitted before it, then copies each run's bytes by the CPU into the pages (a byte two
    runs name keeps the later's) before that batch is submitted. So the pages hold the bytes once
    the command buffer completed, and every command after the copy, in its command buffer or a
    later one, sees them (`host_ram_scatters_staged`, `host_ram_staged_drain_nanos`). The cost is
    one wait for the GPU between the copy and the work after it in its command buffer. A rebind,
    before the command buffer's first submission, of a buffer over host pages that the copy's
    encoder also used finds that encoder kept no log to encode it again from
    (`CommandBufferError::EncodedUseRebound`, as for an encoder with an indirect dispatch the
    host reads).

  Any other copy imports the chunk on its encoding thread, or waits for the import under way
  (`host_ram_copy_import_waits`, `host_ram_copy_import_wait_nanos`; every chunk import's time in
  `host_ram_chunk_import_nanos`). Of a RAM with a source of run lists, on a device whose held
  imports add to every submission's cost, a copy over chunks not imported imports lists of its own
  pages instead (`HostRam::with_dma_buf_of_runs`, above), and those chunks not at all. Each copy is
  ordered and seen as a copy between the buffer and a no-copy buffer over the pages: the chunk's
  bytes have a hazard-tracking key placed over its host bytes, as a no-copy buffer's are, so the
  copy is ordered against every other copy of those bytes and every no-copy buffer or heap over the
  RAM's own addresses, in the encoder and across encoders, command buffers and queues (an object
  over another mapping of the same pages is not, as two no-copy buffers over two mappings of one
  page are not). The CPU's writes made before the commit are seen, the GPU's are in the pages at the
  completion, and where the RAM came with a log its scatters are reported to it as the GPU's writes
  in place, so copied buffers and shadows over those pages copy them in at their next commit.
  Refused: a device that imports no host memory (`Unsupported`, naming
  `VK_EXT_external_memory_host`); a RAM the device does
  not use, a run empty, starting off a 4-byte word or outside the RAM, bytes of the buffer that
  start off a 4-byte word of its memory, or, of a copy by regions, a buffer over host pages the
  runs' bytes overlap (`Invalid`, `Problem::HostRam`); bytes past the buffer (`Invalid`,
  `Problem::RangeOutOfBounds`); a chunk the device cannot import (the import's error). A caller
  falls back to its own copy on any of them.
- `BlitCommandEncoder::copy_runs_from_host_ram(ram, runs, destination)` and
  `copy_runs_to_host_ram(source, ram, runs)` are the same copies with each run at a byte of the
  buffer of its own: `runs` are `(buffer byte, host run)` pairs, for a device cache that keeps a
  surface's layout (row `r`'s texels at `base + r × pitch`, the padding between rows the guest's,
  neither read nor written): one call for all its rows, whose dispatches number by the chunks the
  rows touch, where a call a row takes a dispatch or more for each row. The host runs are as above; a run that starts where the one before ended,
  in the RAM and in the buffer, joins it. Where every run also ends on a word and lies at a word
  of the buffer's memory, the runs take the dispatches; otherwise, or where two runs name one
  destination byte (the later run's bytes stay), regions. Hazard tracking takes the buffer's
  bytes from the lowest run's first to the highest's last. Refused as above, and a run whose bytes
  pass the buffer's length (`Invalid`, `Problem::RangeOutOfBounds`, the first such run's); a run
  at a byte of the buffer off a word is copied by regions, not refused.

## 7. Memory another API writes

A video decoder or another driver that writes a runtime buffer in place imports the buffer's
memory. Make the buffer with a descriptor's `exported_as`: `MemoryExport::DmaBuf` for VA-API or
any dma-buf importer, `MemoryExport::OpaqueFd` for CUDA or another `VkDevice` of the same driver
(`Device::new_exportable_buffer` makes the latter too). The descriptor's `aligned_byte` and
`padding_unobserved` act as for any buffer, so a frame whose planes start 32 bytes into their
surface keeps its textures in place:

```rust,ignore
let descriptor = BufferDescriptor::new(length, ResourceOptions::private(), BufferContents::Zero)
    .with_aligned_byte(0x20)
    .with_padding_unobserved()
    .with_exported_as(MemoryExport::DmaBuf);
let frame = device.new_buffer_with_descriptor(descriptor)?;
let exported = frame.export_memory_dma_buf()?;
// A DRM_PRIME_2 surface: the dma-buf `exported.fd` of `exported.allocation_size` bytes, each
// plane at `exported.offset` plus its offset in the frame, DRM_FORMAT_MOD_LINEAR.
```

An opaque file descriptor (`export_memory_fd`) is imported the same way: CUDA's
`cuImportExternalMemory` takes `exported.fd` with `exported.allocation_size` as its size, flagged
`cudaExternalMemoryDedicated` only where `exported.dedicated`, and `cuExternalMemoryGetMappedBuffer`
takes `exported.offset` and `exported.length`; another `VkDevice` imports it with that size and
`exported.memory_type_index`, dedicated to a buffer of the same parameters only where
`exported.dedicated`, and binds that buffer at `exported.offset`. The buffer is its own dedicated
allocation only where the device exports such memory only from one or requires one; elsewhere it
lies at `exported.offset` in a block that other exportable buffers share, and the import reaches the
whole block, of which the other API reads and writes the buffer's bytes alone.

A texture over a plane of the frame is an image of the exported memory in place, made for the
export's handle type (`VkExternalMemoryImageCreateInfo`, as Vulkan requires of an image bound in
exported memory), where all of these hold for the plane's linear image (its format, extent and
usage, that handle type; the plain linear image first, else a `DRM_FORMAT_MOD_LINEAR` one):

- the image format query (`vkGetPhysicalDeviceImageFormatProperties2` with
  `VkPhysicalDeviceExternalImageFormatInfo`) takes the handle type for it, and not in a dedicated
  allocation alone;
- the plane's byte in the memory, `exported.offset` plus the plane's offset in the frame, is a
  multiple of the `alignment` `vkGetImageMemoryRequirements` answers for the image;
- `vkGetImageSubresourceLayout` answers `offset` 0 and a `rowPitch` equal to the plane's row
  pitch, and the image's `size` fits in the buffer from the plane on;
- the device's in-place check of the runtime's memory exported with that handle type passed for
  the texture's access (a texture made before the check answered is a shadow that moves in place
  at a later commit).

Over a dma-buf export where the device takes no such image in the memory itself (the format query
takes the dma-buf handle type only in a dedicated allocation, the image made requires one, or the
memory is the buffer's dedicated allocation), the plane's image is a `DRM_FORMAT_MOD_LINEAR` one
whose plane lies at the plane's byte in the memory with the plane's row pitch, bound to a dedicated
import of that memory's own dma-buf (`vkGetMemoryFdKHR`, as `export_memory_dma_buf` makes one),
so it reads and writes the frame's bytes in place through a second `VkDeviceMemory` of the same
memory, where the driver takes that layout and the same check passed
(`WorkCounts::exported_dma_buf_images`; the import goes with the texture). RADV takes the dma-buf
handle type for no plain linear image, and requires a dedicated allocation of a
`DRM_FORMAT_MOD_LINEAR` one, so a frame's planes take this route there. `export_memory_dma_buf`
answers the same dma-buf as before.

Elsewhere the texture is a shadow, announced with its cause: the memory is an opaque file
descriptor's dedicated allocation, which the buffer takes only where the device exports such
memory only from one or requires one (Vulkan binds no image in memory dedicated to a buffer), the
format query refuses the handle type, the layout differs, or the check did not pass. Measured for
a 3840x2160 NV12 frame (an `R8Unorm` plane, then an `RG8Unorm` 1920x1080 plane, both rows 3840
bytes) made with `aligned_byte(0x20)`, its planes at 256 and 8,294,656 bytes of the memory, each
plane's image answered row pitch 3840 and offset 0, with an alignment of 64 on lavapipe and 256 on
RADV (RX 550, opaque file descriptors). Mesa 26.2.6's RADV lists no
`VK_EXT_image_drm_format_modifier` for the RX 550 (POLARIS12), so it has no image to import its
dma-buf into, and its dma-buf planes are announced shadows.

The export settles the zeros the buffer was made with first, so the decoder may write at once.
After each of its writes, once the consumer waited for them (`vaSyncSurface`), call
`frame.external_write_done()`: Vulkan leaves what another API wrote undefined until the device
acquires the memory from the queue family reserved for it, and the device's next submission records
that acquire before its commands, so the call waits for no GPU work. The call is also what tells a
texture over the frame that its bytes changed: one whose rows no image of the memory holds reads them
through a shadow, which copies them in again at its next use after the call, and keeps what it copied
before where no call came. The decoder writes only while
no command of the runtime's that uses the buffer is in flight, and the buffer outlives every import.
A device that does not export a dma-buf refuses the making (`NotImplemented`,
`RuntimeGap::DmaBufExport`); the consumer then decodes into its own memory and copies.

## 8. Check the integration

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

A host note: on NVIDIA's proprietary driver, drop the runtime's devices before the process exits.
A device kept to the exit (in a static, as Metal's device lives for the process) can end the exit
with `SIGSEGV` inside the driver's own exit handler, which tears the driver's SPIR-V compiler down
while a worker thread of the driver's still runs in it: a null dereference at `0x18` in
`libnvidia-glvkspirv.so`, on a thread the driver started from the thread that made the device (so
it carries that thread's name, a test's under a test harness). No call of the runtime's is in the
driver then, and Vulkan has no call that waits for that worker. On an RTX 3090 (620.70.06), 8
processes at once, each making a device whose in-place check had no record yet
(`Device::settle_in_place_checks`, run once per driver) and keeping it: 6 of 80 exits right after
the check ended faulted, 0 of 80 three seconds after it, and 0 of 80 with the device dropped first;
with no check, 0 of 160. Which other work leaves the worker running is unaudited
([FAST-PATHS](FAST-PATHS.md), known gaps).
