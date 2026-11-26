# metal2vulkan-runtime

The Metal API on Vulkan, with one path for every device.

Its public API so far: the instance and the device, memory blocks (taken, or imported from the
consumer's memory) and the buffers in them, the grants of texture and sampler indices, textures,
drawables' textures and their presents, sampler states, command queues, command buffers and
events, compute and blit encoders, the device's counters, and `floor`, the check of a Vulkan
device against the floor below.

Every fallible call returns an `Error`, and `Error::basis` says where a refusal's belief comes from:
Metal's own refusal, measured or in Apple's words, the device's lack, or this crate's own gap or cap.

## What the consumer implements

Ten of Metal's objects and getters have no counterpart here by design. The consumer implements
them over this crate's items:

- `MTLRenderPassDescriptor`'s read-back of its own fields. `RenderPassDescriptor` is a value the
  consumer fills, so what Metal's descriptor answers after its setters is the consumer's to keep.
- `MTLHeap` and its getters. A heap is a range of a block the consumer took (`HeapRange`); the
  size and alignment of a resource placed in a heap (`Device::heap_texture_size_and_align`,
  `Device::heap_buffer_size_and_align`) are this crate's.
- `MTLComputeCommandEncoder`'s `dispatchType`. `CommandBuffer::compute_command_encoder` takes it
  as its argument and keeps no getter.
- `MTLTexture`'s `cpuCacheMode`, `hazardTrackingMode` and `allowGPUOptimizedContents`.
  `TextureDescriptor` states none of them.
- A pipeline state's reflection (`MTLComputePipelineReflection`, `MTLRenderPipelineReflection`)
  and the options a creation takes: `MTLPipelineOptionFailOnBinaryArchiveMiss`,
  `MTLPipelineOptionArgumentInfo`, `MTLPipelineOptionBufferTypeInfo` and
  `MTLPipelineOptionBindingInfo`. rt builds no pipeline reflection and acts on no
  `MTLPipelineOption`; its consumer builds reflection from the library's AIR. A word with a bit
  Metal defines no option for is the consumer's too: Metal reflects under it as under
  `MTLPipelineOptionBindingInfo`.
- `MTLTexture`'s `parentTexture`, `parentRelativeLevel`, `parentRelativeSlice` and `shareable`.
  `Texture::new_texture_view` keeps the texture whose image the view reads and no getter, and no
  texture here is shareable; the consumer keeps the texture it made a view of and what it shared.
  A view's descriptor (its type, format, levels, slices and combined swizzle) is this crate's.
- The `label` of every object and descriptor. Nothing here keeps a label; the consumer keeps the
  label it set.
- The `device` of a library and of a function. Both are the translator's, which keep no device;
  the consumer knows which device it made each on.
- `MTLBinaryArchive` and a pipeline descriptor's `binaryArchives`. The consumer keeps its archives
  in its own bundle; nothing of an archive reaches this crate.
- `MTLResource`'s `label`, `device`, `cpuCacheMode`, `hazardTrackingMode`, `resourceOptions`,
  `heap`, `heapOffset` and `isAliasable`. No buffer or texture here keeps any of them, and the
  consumer answers each from what it holds: `label` echoes the label the program set;
  `cpuCacheMode`, `hazardTrackingMode` and `resourceOptions` echo the options or descriptor the
  program made the resource with, each default resolved by the consumer; `device` and `heap` are
  the consumer's own objects, the device it made the resource on and the heap it placed it in;
  `heapOffset` is where the consumer's own placement put the resource, and `isAliasable` is what
  the consumer's own record of `makeAliasable` holds. A resource's `storageMode` is this crate's:
  a texture's descriptor states it, and a buffer's is its block's class.

## Instance and device

- `Instance::new` loads the Vulkan loader, the first of the platform's standard names that the
  system's library search finds (`vulkan-1.dll` on Windows, `libvulkan.dylib` then
  `libvulkan.1.dylib` on macOS, `libvulkan.so.1` then `libvulkan.so` elsewhere), and creates the
  instance at the floor's version, with the floor's instance extensions and each extension the
  consumer names that the loader offers: the window system's, and any other it uses. Where the
  loader offers `VK_KHR_portability_enumeration`, it is enabled and portability devices are listed
  too, each checked against the floor as any other. `Instance::extensions` lists those enabled.
  `Instance::physical_devices` lists the physical devices in the loader's order.
- `Instance::with_drivers` (unsafe) is `Instance::new` with the consumer's own Vulkan loader, its
  own Vulkan drivers, or both, for a consumer that ships them (on macOS, the loader and Mesa's Mac
  driver in its bundle). The loader is given by its path and is the one loaded, with no search of
  the standard names. Each driver is given by its `vkGetInstanceProcAddr`: for a driver built for
  the Vulkan loader, the `vk_icdGetInstanceProcAddr` it exports. The instance enables
  `VK_LUNARG_direct_driver_loading` and gives the loader the list in its exclusive mode, so the
  loader loads those drivers alone, none the system installed or its variables name; where the
  loader does not offer the extension, it refuses with `Error::NoDirectDriverLoading`. An
  extension is enabled where the loader or one of the drivers lists it. The runtime searches no
  directory for a loader or a driver, reads no variable and names no driver: the consumer loads
  each driver's library and keeps it loaded while the instance lives.
- `Device::new` checks a physical device against the floor and makes one `VkDevice`. Every
  command queue, every guest process and the compositor share it. It takes the floor's queue
  family, the first that does graphics and compute: one queue for the runtime, and a second for
  presenting where the family has one. Where it has one queue, presenting shares the runtime's
  queue and its lock. `DeviceOptions` carries what the consumer adds: its `BlockMappings` hook,
  and the device extensions it uses beyond the floor's. Each `DeviceExtension` names one and,
  with `DeviceExtension::with_features`, its feature structure: the members the consumer needs,
  and a test of the device's answer. The device enables it where it lists the extension and its
  answer passes, and passes over it otherwise; `Device::extensions` lists those enabled, the
  floor's first. The consumer names an extension's dependencies too. It names the core Vulkan
  features its own work needs as `ConsumerFeature`s (`SamplerYcbcrConversion` for now): the device
  enables each one it reports, once, and `Device::features` lists those enabled. It names the
  directory the device keeps its lowered functions and its pipeline cache in,
  `pipeline_cache_dir`, and the most bytes the lowerings take, `pipeline_cache_cap` (see "The
  cache directory"). With `None`, every boot lowers every function and compiles every pipeline
  cold, so a consumer should pass a directory. It states the memory ceiling it gives its helper
  for a kernel past the stage bound, `stage_memory_ceiling` (see "Kernels past the stage bound").
- `Device::raw` gives the instance, physical device, device and queue family handles, and
  `Device::with_present_queue` runs the consumer's presents on the presenting queue under its lock
  (see "Presenting"). The runtime keeps owning every handle.
- `Device::has_unified_memory` is true where Vulkan's device type is integrated or CPU.
- `Device::max_threadgroup_memory_length` answers `maxThreadgroupMemoryLength`: the device's
  `maxComputeSharedMemorySize` less the 4-byte vote word a kernel's partial-threadgroups variant
  may take beside its threadgroup memory, rounded down to 16, and at most Metal's 32,768. A kernel
  whose threadgroup memory, its own and the lengths set, is within it runs every dispatch, ragged
  grids included. On a device of 32,768 bytes it is 32,752, where a Mac reads 32,768. Every
  dispatch is still checked against the device's own limit, so one past the answer and within
  that limit runs as before. Unaudited: a guest that rounds the answer down to a power of two
  gets 16 KiB there, where a Mac's answer gives it 32 KiB.
- `Device::supports_function_pointers` is true on every device: a kernel's call through a
  visible-function table is a switch over the functions its state links, known when the state is
  made, and needs no Vulkan feature (see "Visible-function tables"). The runtime gives no
  `supportsFunctionPointersFromRender` answer: a render pipeline state that reads a function
  table is refused (the known gaps of render pipeline states).
- `Device::read_write_texture_support` is `MTLReadWriteTextureTier2` on every device above the
  floor: Vulkan requires storage images of tier 2's four-channel and 32-bit formats, and the
  floor's `shaderStorageImageExtendedFormats` its six 8- and 16-bit single-channel ones. The
  device test `read_write_tier` reads and writes each of the 18 through `access::read_write`.
- `Device::argument_buffers_support` is `MTLArgumentBuffersTier2` on every device above the
  floor, which holds buffer device addresses and the set-1 arrays' runtime descriptor arrays,
  partial binding and update-after-bind of sampled and storage images. The device tests
  `argument_tier` read and write the writable textures an argument buffer holds, read through
  an argument buffer a kernel wrote and a Private one, read the texture of an argument buffer a
  pointer member reaches, read buffers through an argument buffer's pointer members and through a
  `gpuAddress` the CPU wrote into a buffer, and answer a buffer argument's pointer converted to
  an integer as its `gpuAddress` plus the offsets. Some forms both tiers promise are
  refused at the making of a state (the known gaps of pipeline states).

### Teardown

The runtime runs four threads per device: the block-maker, which makes the pool's allocations
ahead of need and the runtime's fixed-size internals off every caller's thread (Memory blocks and
buffers) and frees device memory, the submit thread, the completion thread, and the reaper, which
frees the semaphores and command pools the others give up, one at a time, so no driver free runs
on the completion thread. The device memory of a texture's allocation of its own, a surface
plane's import and a copy's staging buffer goes from the reaper or the upload's retirement to the
block-maker, so every free of device memory but a call's undo of what it made and did not use,
and the teardown's, is the block-maker's, and a submission waits behind at most one. The reaper
also lowers the compute pipeline states' variants an encoding call waits for (the
emulated-SIMD-group and the partial-threadgroups ones), and makes the device's pipeline cache when
it starts and writes it to the cache directory once pipeline creation is quiet. While an old
generation of the cache directory is removed, a thread of the store's own does it, and ends with
the removal; the teardown waits for it. `Gauges::threads` holds the OS id of each of these threads while it runs, so a tool
can tell the runtime's own work from the threads a driver starts for itself.
A device is torn down when its last handle, and the last handle of every object made from
it, is dropped; a committed command buffer and the blocks the consumer holds do not keep it alive.
Nothing the runtime keeps for itself counts as a handle of the device: what a command buffer keeps
until it retires of the buffers, textures and sampler states it used is the object alone, without
the device's handle, so the consumer's last drop is the device's last whatever work is in flight.
Teardown stops the submit thread, which submits what is ready and ends what it holds with
`CommandBufferError::NotRun`; the completion thread retires the rest as the GPU completes it and
runs the handlers; the threads are joined and the GPU goes idle; `BlockMappings::unmap` runs for
each of the pool's allocations the consumer mapped. Teardown runs on the thread that drops the
last handle, or,
where that is a handler (scheduled or completed) or a listener, on the reaper once the handler
returns. The reaper joins every other thread before anything of the device goes; where its own
reference is the runtime's last, it frees the blocks and destroys the `VkDevice` on its own
thread, then ends, touching nothing of the device after. A call that holds a
handle of its own gives it up before another thread can drop the last: `commit` gives up the
command buffer's before the command buffer can complete. Then every block is freed and the
`VkDevice` destroyed, once the runtime's last internal reference is gone: on the thread that tore
down, or where a call still held one as it returned, in that call, which runs no consumer code
there. A last handle dropped on the consumer's thread, with no object made from the device still
held, tears the device down in that drop: every gauge reads zero as it returns. A physical device
and a device each hold their instance. Nothing relies on a static destructor or an `atexit`
handler: the consumer drops every handle before the process exits, or, where a handle is still
out, calls `Device::quiesce` and ends with `_exit` (Quiesce and exit).

`Instance::wait_until_devices_torn_down(timeout)` waits for the teardowns that run on the reaper.
It answers true once every device made from the instance has destroyed its `VkDevice`, dropped
everything it held and had the thread that tore it down joined; at once where nothing is pending;
false once the timeout passes first, and at once on a runtime thread (in a handler or a
listener). A consumer whose last handle may drop in a handler or a listener calls it before it
drops the instance or exits. The gauges are no such signal: they read zero once the `VkDevice` is
destroyed, before the instance's handle and the cache store the device held are dropped, and
before the thread that tore it down ends. After a false answer the consumer reads the device's
`Counters`, which it took before its drops: `Gauges::handles_out` is 1 while a handle of the device
is still out (its own, or one a handler or a listener holds) and 0 once the last has dropped; with
0, a thread in `Gauges::threads` is the teardown still running there.

A device the consumer dropped is never handed out again: only `Device::new` makes a `Device`, and
the runtime's threads hold the device's core, never a handle. A call that configures what has
started refuses typed, never answering Ok for nothing: every call of a committed command buffer
(`Refusal::Committed`), a call naming a block whose last handle is gone (`Error::NoSuchBlock`),
and a call that needs a runtime thread after `Device::quiesce` (`Error::Quiesced`).
`DeviceOptions` are read once, by `Device::new`.

### Counters

`Device::counters` returns a handle that stays readable after the device is gone. `metrics()` are
cumulative counts (submissions, passes, barriers, bytes copied, pipelines created, blocks made,
first-use waits, kernels lowered, kernels read from the cache directory, upload waits, pushes,
and the draws and dispatches whose push bytes the command buffer already held), and times in
nanoseconds, in total and longest: among them each `vkQueueSubmit2` call's time inside the
driver (`submit_call_nanos`, with the calls over 100 ms counted) and the submit thread's wait
for the queue's lock before a submission's calls (`queue_lock_wait_nanos`), which on a family of
one queue is also the wait behind the consumer's presents (`Device::with_present_queue`);
a pipeline make's two sides, the time inside `vkCreateComputePipelines` and
`vkCreateGraphicsPipelines` (`pipeline_creation_nanos`) and inside the translator's calls, its
lowerings from AIR to SPIR-V and its links and plans (`translation_nanos`);
the `vkQueueSubmit2` calls that met one of the block-maker thread's frees, made during it or
inside the driver as it began (`submissions_during_frees`), and their time inside the driver
(`submit_call_nanos_during_frees`), the time a driver may have held them behind the frees; the
pool's allocations made within a second after a free of an idle allocation of their class
(`shared_remade_after_frees`, `private_remade_after_frees`), memory given back and taken again;
four histograms of each command buffer's stages, in 23 power-of-two buckets from under 2 us to
2^22 us (about 4.19 s) or more, filled from the times the command buffer already reports: commit
to scheduled (`commit_to_scheduled`, counted as it is scheduled), scheduled to GPU start, the GPU
run, and GPU end to the host time its handlers begin (counted as it retires, for one that
completes with GPU times), each beside its spans' sum in nanoseconds (`commit_to_scheduled_nanos`
and the three like it); of scheduled to GPU start, the part under the GPU spans of other command
buffers retired before it, the runtime's own included (`scheduled_to_gpu_start_behind_nanos`); GPU
end to handlers split at the completion thread's six steps, for all and for those of 1,024 us or
more (`gpu_end_to_handler_steps_nanos`, `gpu_end_to_handler_tail` and its steps); the clock
calibrations and their calls' time (`calibrations`, `calibration_nanos`); the segments the submit thread held for an event's value that neither
the host had set nor a submitted signal promised, by the length of each hold (`event_holds`, in
the same buckets); the groups a queue held without room that went once a completion left room
(`held_starts`), and the GPU's time behind them, from the latest
end of the command buffers submitted before each to the start of its first, on their timestamps
(`held_start_gap_nanos`); the command buffers scheduled while their queue held them for room,
not yet in a group (`held_for_room`); the command buffers of the runtime's own it submitted, the uploads and
the Managed buffers' copy-ins (`runtime_command_buffers`), and the GPU's time they ran, on
timestamps of their own (`runtime_command_buffer_gpu_nanos`);
the copies of textures over a buffer that no linear image holds are split by cause
(`over_copies_pitch` and the rest, in Textures below); the exported blocks a public call handed
out a device address of, each counted at the first (`exported_blocks_addressed`, the exported
block's rule in Textures below), 0 where none went out; the pipeline cache's replacements past its
bound and its blob writes skipped past it (`pipeline_caches_replaced`,
`pipeline_cache_writes_skipped`, The cache directory below); the dispatches that ran a kernel's
emulated-SIMD-group module (`emulated_simdgroup_dispatches`, Compute pipeline states below); the
states' main pipelines their `finish_pipeline` calls made, with their time, and the calls whose
make failed (`finished_pipelines`, `finished_pipeline_nanos`, `failed_finishes`, Compute pipeline
states below); the compiled contents a state's make dropped because an equal one landed in the
device's map first (`kernel_makes_dropped`); the copies of resolve images into the buffers their
textures lie over (`resolve_copies` and `resolve_copy_bytes`, Render encoders below);
`gauges()` are live counts per kind of resource, every one zero after teardown, and
`handles_out`, 1 while a handle of the device is out (Teardown). They are relaxed atomics, always
on. What an encoder's commands count arrives when the encoder ends.

## Memory blocks and buffers

Memory comes in blocks the consumer takes and places buffers in, as a placement heap's buffers sit
in its memory.

- `Device::take_block(class, len)` gives a block of `MemoryClass::Private` (device-local, no host
  mapping) or `MemoryClass::Shared` (host-visible and host-cached, host-coherent too where the
  device has such a type, mapped for the block's life; Shared memory's coherence below) memory, as
  a `Block`: rt's own handle, with no Metal counterpart, which derefs to
  the block's `BlockInfo` (its number, class, length, device address and, for Shared, its host
  address, and the allocation it is a range of with its offset there). One `VkBuffer` spans each
  of the pool's allocations, with every buffer usage and a device address; a block is a range of
  it. Nothing in a block is exportable.
- A block is a range of one of the device's allocations of its class: the length asked rounded up
  to 64 KiB (to `bufferImageGranularity` where that is larger), at an offset that is a multiple of
  that length rounded up to a power of two, up to 2 MiB. A take is the first free range of the
  class's allocations that fits it, in the order the allocations were made: under the pool's lock, with no Vulkan call and no call to the
  consumer's hook (`Metrics::takes_from_free_ranges`). A long allocation, which a take longer than
  16 MiB made for itself, serves only later takes of exactly its length:
  `Metrics::shared_long_reuses` and `private_long_reuses` count them, and
  `Gauges::shared_long_reused_bytes` and `private_long_reused_bytes` the bytes of long allocations
  that hold such a block. Where no free range fits, a take is the start of an allocation made in
  the call, of `MemoryClass::block_len` (16 MiB of either class) or the block's length where that
  is longer, up to the device's `maxMemoryAllocationSize` and `maxBufferSize` (`Limit::BlockLen`
  past them). Its rest serves later takes.
- `Device::new` makes one allocation of each class before it returns. After that the block-maker
  thread makes a class's next allocation of 16 MiB, off every caller's path, while the class holds
  no wholly free unit (an allocation of 16 MiB that no block holds), and at no other time. So a
  take of up to 16 MiB lands in memory made ahead once the block-maker thread has made it, and the
  trade is that a small take that breaks a class's last whole unit sets off an allocation beside
  the consumer's next calls. A take that no free range fits allocates on the calling thread. A
  driver may hold the device's other calls for the whole of an allocation, longer for more bytes:
  on an RTX 3090 (NVIDIA's Linux driver) a pipeline state's creation beside a Shared allocation's
  making took 6 to 8 ms at 16 MiB and 40 to 45 ms at 256 MiB; on an RTX 2080 a compute
  pipeline's creation waited out the whole 3.4 ms of a 16 MiB Shared allocation made beside it.
  It holds another thread's `vkAllocateMemory` too: on the RTX 3090 (620.70.06, a standalone
  program), a 2,621,440-byte exported block's allocation begun 1 ms into a 547 MiB Shared
  allocation took 85.3 ms (57.8 µs alone), begun 1 ms into that allocation's free 111.4 ms, and
  begun 1 ms into a 547 MiB Private allocation 15.3 ms; two exported blocks' allocations begun
  together took 1.7 to 3.0 times their two times alone.
  `Metrics::consumer_allocation_nanos` and `consumer_allocation_max_nanos` count the time of
  allocations made on the consumer's threads, and `maker_allocation_nanos` those of the
  block-maker thread. Eight of the consumer's sites are
  counted apart, each as a `SiteAllocations` (calls, bytes, the longest call and its bytes):
  `texture_allocations` (a texture's allocation of its own), `texture_block_allocations`,
  `ring_block_allocations`, `staging_region_allocations`, `staging_buffer_allocations` (a copy
  larger than the staging region), `zero_source_allocations`, and `shared_take_allocations` and
  `private_take_allocations` (a take of each class that no free range fits). At the three sites
  of takes from the pool, `SiteAllocations::takes_ahead` and `takes_ahead_bytes` count the takes
  that landed instead in a unit the block-maker thread made ahead, each the first to land in it:
  takes that would otherwise have allocated there. A take's parts are timed apart, each as a `Timed` (calls, bytes, total
  and longest time, the longest call's bytes and whether it made an allocation of its own, and
  the calls over 1, 5 and 20 ms): `Metrics::takes` (each take whole, a
  texture block's among them), `take_lock_waits` (its wait for the pool's lock), `take_makes`
  (an allocation it made for itself on the caller's thread: the buffer, `vkAllocateMemory`, the
  bind, the address and a Shared one's map), `take_maps` (the consumer's `BlockMappings::map` of
  it) and `maker_makes` (the allocations made ahead of need, off every take); `zero_writes` times
  `Device::new_buffer`'s host writes of zeros (`memset` through the block's mapping), and
  `long_take_zero_writes` those of them into a take past the 16 MiB unit. An exported block's make
  is timed the same way, whole and by step: `export_makes` (each `Device::new_exported_block` that
  made its block), `export_allocations` (its `vkAllocateMemory`, exportable and dedicated to its
  buffer), `export_handles` (the handle's export, `vkGetMemoryFdKHR` or
  `vkGetMemoryWin32HandleKHR`) and `export_claims` (the block's claim, which enters the device's
  map of claims); the rest of a make is the buffer's creation, the binding and the address. A
  texture's make is timed by step: `image_creates` (`vkCreateImage`, a surface plane's too),
  `image_binds` (`vkBindImageMemory`), `image_view_creates` (every `vkCreateImageView`),
  `texture_place_lock_waits` (its placement's wait for the arrays' lock, at an index) and
  `home_lock_waits` (its entry's wait for the work lock in the next commit's upload, which moves
  it to its home layout). A texture block it takes is among the `takes`; its place in a block,
  found under the texture blocks' lock, is not timed. An exported surface's in-place make adds
  `surface_allocations` (its exportable `vkAllocateMemory`) and `surface_handles` (each export of
  its handle).
- The runtime's fixed-size internals are made by the block-maker thread, never on a caller's
  thread and never inside `Device::new`: its first work once `Device::new` starts it is the zero
  source (64 KiB), the parameter ring's first block (2 MiB) and the staging region (16 MiB). The
  ring's next block is made one ahead: when a take leaves fewer free chunks than a block holds
  (32 of 64 KiB), the block-maker thread makes the next, so the ring holds at most one block more
  than the most chunks in use at once need. A call that needs one of these before it is made
  waits for the make: `Metrics::maker_waits` counts those waits and `maker_wait_max_nanos` the
  longest. Where the block-maker thread's make failed, or the thread had stopped, the call makes
  it on its own thread with the same function, counted at its site. Every device holds these
  18 MiB from its start, used or not, and the ring's second block of 2 MiB from its first take of
  a chunk (a parameter write, or a buffer argument bound to nothing).
- A Private allocation takes the device-local Private type. Where that allocation itself fails
  out of device memory (never a budget poll), it takes the next memory type the resource allows,
  in the device's order: `BlockInfo::device_local` is false for such a block, and
  `Gauges::spilled_bytes` counts the bytes until they are freed. A spilled allocation is freed
  once its last block is given back, so the next tries device-local memory again. Textures are not
  placed in a spilled block: a texture then takes an allocation of its own, which spills in the
  same way where the image allows another type. A Managed buffer's device copy that the consumer
  places in a spilled block is still its second copy, so `did_modify_range` and a GPU write keep
  Metal's two-copy rule.
- Every `VkDeviceMemory` the runtime holds counts against the device's `maxMemoryAllocationCount`:
  each block (texture blocks among them), each imported block, each texture's own allocation, the
  staging region, the staging buffer of a copy larger than the region while it runs, the
  parameter ring's blocks and the zero source. Vulkan leaves an
  allocation past the count undefined and does not promise `VK_ERROR_TOO_MANY_OBJECTS` for it, so
  the runtime counts its own: one more past it is `Limit::MemoryAllocations`, refused before the
  driver is called; `Gauges::blocks` counts those live. The limit is about 4.29 billion on RADV,
  NVIDIA's Linux driver and ANV, 8,401,685 on Intel's Windows driver for the UHD Graphics 630, and
  4,096, Vulkan's minimum, on Mesa's Mac driver.
- A block's bytes are undefined, new or taken again: the runtime zeroes nothing ahead of need, and
  asks the driver for no zeroed memory. A buffer reads zero where
  `Device::new_buffer` zeroes its range (below), as Metal zeroes a buffer from
  `newBufferWithLength:options:`, and Metal owes zeros for that call's buffers alone: not for a
  heap's memory, whose bytes are undefined, as Metal is silent on them. The consumer passes
  `zero_from` for a buffer Metal zeroes, or zeroes the range at its own carve, once.
- A texture the runtime places reads zero until written: `Device::new_texture`'s,
  `new_texture_for_views`'s and `new_managed_texture`'s. The upload of the next commit clears every
  subresource of its image once, ahead of every command buffer committed after the call: a colour
  clear, a depth-stencil clear, or copies from the zero source for a block-compressed format, which
  no clear writes. This is the runtime's choice, not Metal's rule: Metal does not say what a new
  texture holds (`newTextureWithDescriptor:` is "Allocate a new texture with privately owned
  storage."). The runtime chose zero because programs read textures they never wrote, an Apple M3
  Max answered zero for new Shared and Managed textures, and the memory's last bytes would follow
  process history. The clear runs on every device. `Metrics::texture_clears` and
  `texture_clear_bytes` count the clears. A texture in a heap's memory, over a buffer, of a
  surface's plane, or a drawable's is not cleared: it reads its memory's bytes, or the swapchain's.
  A view reads its parent's image.
- `Device::zero_block_range(block, offset, length)` zeroes a range of a Private block the
  consumer holds, where it places a buffer Metal zeroes: by a GPU fill in the upload, ahead of
  every command buffer committed after the call, as a zeroed Private buffer's fill is. The
  consumer asks; the runtime infers nothing and zeroes no Private block at a take. The call
  never waits for the GPU. `Metrics::block_zero_fills` and `block_zero_fill_bytes` count the
  fills. A range past the block (`Refusal::ZeroRangePastBlock`) and a Shared block, whose bytes
  the consumer zeroes through the mapping (`Refusal::ZeroRangeShared`), are refused.
- Dropping a block's last handle gives the block back; `Device::release_block` consumes a handle,
  that drop made explicit. Clones of a handle share one claim, so dropping one of two gives nothing
  back. It takes effect once no buffer, command buffer or upload using the block remains: its
  bytes are then a free range of its allocation at once, as written. A later take that lands
  there has a new number.
  An allocation that holds no block anyone holds is idle. Its idle time starts when its last block
  comes back, and only a take that lands in it again ends it. The idle age is 250 ms, on every
  device: the block-maker thread frees each allocation idle that long, one at a time, after
  `BlockMappings::unmap` of it, once every command buffer committed before it went idle has
  retired. Memory a command buffer reaches by address alone (a heap's buffers through an argument
  buffer, with no handle of them) is so never freed while work committed before its release is
  unretired; a range of it taken again meanwhile still gives such a reader the new block's bytes.
  "Committed before" is the order the runtime is handed: a consumer that forwards a guest's
  releases (of memory, textures or samplers) hands each one over after every commit the guest made
  before it, or the guest's earlier work can meet memory and objects already freed.
  So a program that keeps committing gives back what it stopped using within 250 ms plus the
  frees ahead of it and the work in flight at its release. A command buffer that waits at an event
  (`CommandBuffer::encode_wait_for_event`) holds every such free released after its commit until
  it retires, as a free waits for the command buffers committed before it in commit order. The
  harness's own rows `idle_frees` and `idle_frees_under_commits` hold each free, measured alone,
  to begin within 25 ms of its idle age, with no commits and with one every 2 ms. It frees
  between the allocations it makes, never beside one. Each class keeps a reserve that the idle
  rule does not free: its last idle 16 MiB allocation in the order the allocations were made, its
  wholly free unit. Takes land in the first allocation with room, so they reach the reserve last,
  and a class idle past the idle age holds that one wholly free unit and no other. A long
  allocation, which a take longer than 16 MiB made for itself, is never the reserve.
  `Metrics::allocations_freed`
  counts the frees and `Metrics::idle_frees_by_age` their idle times; `Metrics::idle_reuses_by_age`
  counts idle allocations taken again, by how long they were idle. `Gauges::shared_idle_bytes` and
  `Gauges::private_idle_bytes` count the idle bytes, the reserve's included, and
  `Gauges::shared_idle_age_nanos` and `Gauges::private_idle_age_nanos` the age of the oldest idle
  allocation the rule will free; `Gauges::shared_reserve_bytes` and `Gauges::private_reserve_bytes`
  count the reserve's bytes, and `Gauges::shared_allocation_bytes` and
  `Gauges::private_allocation_bytes` the bytes of each class's live allocations, idle ones
  included. `Gauges::shared_placed_bytes` and `private_placed_bytes` count the bytes of the blocks
  taken and not given back, so a class's allocation bytes less its placed and idle bytes are the
  free ranges between blocks; `shared_buffer_bytes` and `private_buffer_bytes` count live buffers'
  lengths by their block's class, `texture_block_bytes` the placed textures' image sizes and
  `texture_own_bytes` textures' allocations of their own, and `import_bytes`, `export_bytes` and
  `surface_import_bytes` the imports', the exports' (exported blocks, and an opaque handle's
  surfaces in place) and the surface plane imports' bytes; `releases_waiting`
  counts the releases of textures and sampler states that wait for work in flight (Set 1 and
  index grants). The texture blocks
  and the allocations holding the consumer's Private blocks are counted apart, as nothing moves
  a live texture or block and a block or allocation goes back only once it holds nothing:
  `texture_block_held_bytes` counts the texture blocks whole (the one kept empty among them) and
  `texture_block_bytes` what is placed in them; `private_take_held_bytes` counts the Private
  allocations holding a block the consumer took, whole, and `private_take_bytes` those blocks.
  `texture_block_sparse_bytes` with `texture_block_sparse_blocks`, and
  `private_take_sparse_bytes` with `private_take_sparse_allocations`, count the sparse ones among
  each, those holding something that is less than a sixteenth of their length. Both kinds
  together, `private_units` counts the 16 MiB Private units that hold a block (a texture block or
  the consumer's blocks; idle units and long allocations are not counted) and
  `private_unit_used_bytes` the bytes in use inside them, the consumer's blocks' lengths and the
  placed textures' image sizes: against 16 MiB per unit, the rest is free bytes held by what
  lives beside them. Once the last handle
  is gone the block's
  number names nothing
  (`Error::NoSuchBlock`), and numbers are never reused during the device's life. The device's
  teardown frees every allocation, blocks still claimed included, each after the consumer's
  `BlockMappings::unmap` of it returned; a handle dropped after that does nothing.
- `Device::new_buffer(block, offset, length, zero_from)` makes a `Buffer`: a view of the range,
  with no Vulkan object of its own. Its `gpu_address` is the block's address plus the offset, and
  its `contents` the block's mapping plus the offset, or `None` in a Private block. `zero_from`
  zeroes the buffer's bytes from that offset in it to its end: `Some(0)` all of them, as
  `newBufferWithLength:options:` does, and `None` none, for a buffer placed in a heap (Metal does
  not zero heap memory) or one whose bytes the consumer writes itself; an offset at or past the
  length zeroes nothing. A Shared range is zeroed through the mapping, before the call returns; a
  Private one by a GPU fill of exactly the range, recorded into the upload of the next commit,
  which goes ahead of every command buffer committed after the call; the last upload of its
  submission ends with a barrier that orders the fill before every
  later command (a copy into the buffer among them), and the upload holds the block until it
  completes. `Metrics::bytes_zeroed` counts the bytes. A length of 0, a length
  past `MAX_BUFFER_LENGTH` (2 GiB) and a range past the block are refused, as Metal answers nil for
  each.
- `Device::heap_buffer_size_and_align(length, options)` is `heapBufferSizeAndAlignWithLength:options:`:
  the length itself at a multiple of 256, whatever the options, as an Apple M3 Max answers under
  macOS 27.0.1, and the same on every device. The alignment is the runtime's placement rule: the
  runtime takes a buffer at any byte of a block, and every rule a use of a buffer has (an
  argument's alignment, indirect arguments' and indices' places, a fill's words, a copy's texel
  blocks) holds for one at a multiple of 256, which is also no less than Vulkan lets any device
  ask of a buffer's offset.
- A shader reaches a buffer argument's bytes by 32-bit offsets from its bind address, which the
  translator may read as signed, so the consumer states a maximum buffer length (Metal's
  `maxBufferLength`) no larger than `MAX_BUFFER_LENGTH`, and places no buffer longer than that in
  a block, whether Metal's `newBufferWithLength:`, a heap's or a no-copy buffer stands behind it.
- `Device::new_managed_buffer(bytes, copy, length, copied)` makes a Managed buffer whose bytes are
  `length` bytes at `bytes`, a block the consumer holds and an offset in it: the consumer's bytes,
  which `contents` answers where the runtime maps the block (one taken from the pool), and `None`
  where it does not (an imported block, whose pages the consumer reaches through its own mapping).
  The block is a Shared one (`Refusal::ManagedBlockPrivate` for a Private one, whose pages the
  consumer cannot write). Vulkan's device type picks its form, as `Device::has_unified_memory`
  answers it.
  On an integrated or CPU device the buffer is its bytes, as a Shared buffer is, its `gpu_address`
  theirs, `did_modify_range` does nothing, and `copy` is `None` (`Refusal::ManagedCopyOnUnified`
  otherwise). Elsewhere shaders reach a device copy at `gpu_address`, always: `length` bytes at
  `copy`, a Private block the consumer holds and an offset in it, placed by the consumer as it
  places a Private buffer (`Refusal::ManagedCopyMissing` for none, `Refusal::ManagedCopyNotPrivate`
  for a Shared block, `Refusal::ManagedCopyExported` for an exported one). Either offset is any byte, and `gpu_address` is the block's address plus
  it. The copy's rest past its first `copied` bytes is zeroed in the upload, and those bytes are
  named as `did_modify_range` names a range.
  `Buffer::did_modify_range(start, length)` names a range changed (`Refusal::ModifiedRange` past
  the buffer), merged with the buffer's others where they overlap or touch, and submits and
  allocates nothing. The next commit of a command buffer, with commands or none, on any queue,
  takes every range named since, once each: GPU copies from the consumer's block into the device
  copy, recorded at the commit into a command buffer of their own. It goes in the submission of
  the first command buffer committed at or after that commit that the device submits, whichever
  queue and whichever waits held the committing one, as a part of its own ahead of that command
  buffer's commands and outside its waits, after every earlier command on the device, which the
  barrier ending each earlier command buffer orders; the copies record one barrier, after them.
  A command buffer with nothing for the GPU (no commands, or only dispatches that record none: a
  kernel that writes no memory, or a grid with no thread) that reaches the submit thread while
  the copies wait for it takes a submission of them alone, and completes once they have run, its
  GPU start and end one time as for any command buffer with nothing for the GPU. A command buffer
  committed before it never takes it. The copy runs when that submission
  reaches the GPU, so every command buffer committed after a range is named reads the bytes the
  consumer wrote before the naming, as Metal's `didModifyRange:` asks; a write made after the
  commit may reach it, and so may a command buffer committed before that runs after a later one,
  where Metal leaves the range undefined for the GPU.
  `Metrics::managed_bytes_captured` counts the bytes the copies take. A command buffer keeps each
  device copy it uses, and the consumer's blocks its copies read, until it retires. A range named
  and not yet taken keeps neither block: once the buffer's last handle is gone (a command buffer,
  an encoder or a texture that could read the copy would hold the buffer), its ranges are
  forgotten and its blocks go at once, an imported block's release among them, as Metal runs a
  no-copy buffer's deallocator when it deallocates the buffer.

The consumer's ranges follow Metal's retention: it reuses a range only once the work that named
the buffer there before has completed. A command buffer keeps each buffer it uses, and so its
block, until it retires, so a block released meanwhile goes back to the pool only then. Placement is the consumer's, and so
is the slack past a buffer's length: the runtime writes no byte of a block outside the ranges of
the buffers placed in it (a zero fill, a Managed buffer's copy-in and its device copy's zeroing
each cover exactly the buffer's bytes), so what lies between and after them is as the consumer
left it. Vulkan reads an indirect dispatch's arguments only at a multiple of 4 of the block, and an
indexed draw's indices only at a multiple of their size; a dispatch or draw whose arguments or
indices start at a byte that misses that multiple is refused (`Refusal::ArgumentPlace`, in the
known gaps below).

### The consumer's mappings

A consumer that maps the pool's memory itself, into a guest for instance, passes
`DeviceOptions::block_mappings`. Its hooks run per allocation, not per block:

- `BlockMappings::map(allocation)` runs once for every allocation of the pool, before any block of
  it goes out, with the allocation's number, class, length, device address and host mapping, and
  the host time its make began with the make's length (`make_start`, `make_nanos`). It runs where
  the allocation is made: in `Device::new` for each class's first, on the block-maker thread for
  one made ahead of need (`Metrics::allocations_mapped`), and inside the take, on the consumer's
  thread, for a take that no free range fits (`Metrics::allocations_mapped_on_takes`).
  It answers false where the consumer has no place for the allocation: the runtime frees it and
  hands out none of its blocks, and a take that made it returns `Limit::AllocationNotMapped`
  (`Metrics::allocation_map_refusals`).
- `BlockMappings::unmap(allocation)` runs once for every allocation `map` accepted, before the
  allocation is freed, and returns once every mapping the consumer made of it is gone and none of
  its threads touches it. The memory stays mapped and valid until then. It runs on the block-maker
  thread before an idle allocation's free; in `Device::quiesce`, on the calling thread, for every
  allocation still mapped; or at teardown, on the thread that tears down, after the runtime's other
  threads are joined.
- Neither runs under the pool's lock, calls into the runtime or drops a runtime handle. Imported
  blocks are not the pool's, and neither runs for them. An allocation made after a quiesce is
  neither mapped nor unmapped.

The pool's allocations are in the guest's range whole, their free ranges and the runtime's own
staging blocks included; the consumer hands the guest blocks only. A block given back stays mapped
with its allocation: from a block's give-back to the next take that lands on its bytes, they are
the runtime's. A guest kernel that writes a free range damages only its own later buffers.

### Shared memory's coherence

Shared memory is the first host-visible, host-cached memory type a buffer can use, a
host-coherent one first (`floor::MemoryTypes::shared`). Some devices have no type with all
three: Mesa's Intel driver over the i915 kernel driver, on a GPU without a cache it shares with
the CPU, lists a device-local, host-visible, host-coherent type and a device-local, host-visible,
host-cached one. On a type without `VK_MEMORY_PROPERTY_HOST_COHERENT_BIT`, the host's writes
reach the GPU only through `vkFlushMappedMemoryRanges`, and the GPU's writes reach the host only
through `vkInvalidateMappedMemoryRanges`. `Device::new` reads that flag of the Shared type once.
Where the type has it, as on every device of ours, the device keeps no ranges and makes neither
call, which Vulkan calls "unnecessary" there. Where the type lacks it, the device makes both calls
at these points:

- Before each submission, it flushes the Shared memory the submission's command buffers reach, as
  each commit found it: every buffer the command buffer keeps (bound, declared with
  `use_resource`, a blit's or a pass's), the buffer under every texture over a Shared buffer it
  keeps, every heap it uses, and the bytes each Managed copy-in it carries reads. A Managed buffer
  with a device copy is not among them: its commands reach the copy, and only its copy-ins (which
  flush the bytes they read) and its synchronize (a write, below) touch the block of its bytes.
- Before the host can see the GPU's progress, it flushes the Shared memory a command buffer may
  write, then invalidates it. That happens as the command buffer completes, before its status reads
  completed, a wait for it returns or its completed handlers run. It also happens before a shared
  event's value that a GPU signal raised reaches a listener, `SharedEvent::signaled_value` or
  `SharedEvent::wait_until_signaled_value`, and then it covers every command buffer committed and
  not yet completed. A command buffer may write: a blit's destination, a binding its function's
  interface writes, a resource it declares with `use_resource` for writing, a render pass's
  attachments and resolve targets, a visibility result buffer, a synchronize's buffer, the
  whole Shared memory of every heap it uses, and every texture over a Shared buffer in an image
  of its own. The flush comes first because Vulkan leaves a range undefined when the host wrote
  it and it is invalidated without a flush.
- Around its own host accesses: the bytes `Texture::replace_region` stages are flushed before
  their copy, and so are the bytes `Texture::upload_region` copies from a Managed texture's
  named buffer, at the call; the bytes a copy stages for `Texture::get_bytes` are flushed and
  invalidated before it reads them; the zero source of fills is flushed once, when it is made.

Each range is widened to whole multiples of the device's `nonCoherentAtomSize`, inside the
pool's allocation, which the runtime maps whole. `Metrics::shared_flushes` and
`shared_invalidates` count the calls, `shared_flushed_bytes` and `shared_invalidated_bytes` their
bytes, and `shared_coherence_failures` the calls that did not return success; the work goes on
after one. On a device whose Shared type is host-coherent all five stay 0.

On a device whose Shared type is not host-coherent, a range that a command buffer reaches only
through an address, with no binding or `use_resource` declaration naming its buffer, is neither
flushed nor invalidated: the consumer declares what its commands reach. The consumer's own host
reads and writes, through its own mappings (The consumer's mappings) or between those points, are
its own. Imported blocks (below) are host-coherent memory and take no call. Exported blocks and
surfaces are Private memory with no host mapping, and take none either.

### Imported blocks

A no-copy buffer (`newBufferWithBytesNoCopy:length:options:deallocator:`) is the memory it was made
over. The consumer imports that memory as a block and places the buffer in it with
`Device::new_buffer`, as in a block it took. There are three calls: one per handle, and one for
pages that take several dma-bufs; which to try first, and what to do where they refuse, is the
consumer's.

- `Device::import_host_memory(pointer, len, release)` (unsafe) imports host memory by its address
  (`VK_EXT_external_memory_host`), at a multiple of the device's
  `minImportedHostPointerAlignment`, into the first memory type the pointer's query offers that is
  host-visible and host-coherent and that the buffer's requirements list. The block's `contents`
  is the pointer. The pages are pinnable host memory: anonymous memory, or a mapping of a file (a
  memfd's among them). Pages the consumer maps through a dma-buf's fd go through `import_dma_buf`
  instead. On NVIDIA's Linux driver (620.70.06, an RTX 3090), a pointer into a mapping of a
  udmabuf is refused with `VK_ERROR_OUT_OF_DEVICE_MEMORY` in each type its query offers, and the
  process's later allocations fail too: a device-local one after either refusal, and a host-cached
  one after the refusal in the host-cached type.
- `Device::import_dma_buf(fd, len, release)` (unix) imports a dma-buf
  (`VK_EXT_external_memory_dma_buf`) only into the first memory type the fd's query offers, and
  only where that type is host-visible and host-coherent and the buffer's requirements list it;
  anything else is `Limit::ImportType`, with nothing imported. It takes the fd (the consumer's
  dup): after `Ok` the driver owns it; after a refusal or a failed query the runtime closes it, as
  no driver was given it; after a failed import it closes it only where it still names the same
  file, as a driver may have closed it and the number may name another file since. The block's
  `contents` is `None`: the consumer reaches the pages through its own mappings.
- `Device::import_dma_bufs(chunks, release)` (unix) imports several dma-bufs, each fd with its
  length, as one block over them in order, for pages more than one dma-buf holds (a udmabuf takes
  a bounded number of a memfd's ranges). Each length is a multiple of `Device::dma_buf_granule()`,
  where the consumer cuts the pages, the last padded; the granule is `None` where the call refuses
  (`Limit::NoImport`). The block's buffer is one sparse buffer (`VK_BUFFER_CREATE_SPARSE_BINDING_BIT`)
  with each dma-buf's import bound at its offset, so the block's `gpu_address` and its buffers run
  across the dma-bufs as over one range, and its `contents` is `None` as for one dma-buf. Each fd
  takes the dma-buf rule's type of the sparse buffer's types, and every fd the same one. One
  `vkQueueBindSparse` on the runtime's queue binds the imports (`Metrics::sparse_binds`); the call
  does not wait for it. Vulkan orders no sparse bind against command buffers, so the runtime's next
  `vkQueueSubmit2` waits for it, once: a semaphore wait orders every later command in submission
  order. The free, on the block-maker thread, waits for it too before the buffer goes, then frees
  each import, and `release` runs after the last. Each dma-buf's import is one allocation, counted
  in `Gauges::imports` and `Gauges::blocks`. The fds are taken as `import_dma_buf` takes its one.
  While it lives, each import adds to every submission's host time what one `import_dma_buf`
  block adds: on the Radeon 780M (RADV, Mesa 26.2.6, without the validation layer) an empty
  submission took 7.09 µs with no import, 22.18 µs beside one block over 40 dma-bufs (0.377 µs a
  dma-buf) and 22.12 µs beside 40 blocks of one (0.376 µs an import); on the RX 550, 0.353 and
  0.354 µs; on lavapipe none measured. That cost is the driver's: the runtime's own part is the
  next submission's one wait for each bind, and one atomic read a submission on a device that
  offers the call, with blocks live or not.

Each call returns the block and the index of the memory type it imported into, and refuses before
any allocation is attempted.

On Linux 6.11 to 7.1 a udmabuf made over the pages of a large folio is malformed: one of its entries
is a 4 KiB page at an offset past that page's size, and the AMD kernel driver's import of it then
binds other pages than the buffer's. Linux 7.2 fixes it. So a consumer hands `import_dma_buf` or
`import_dma_bufs` a udmabuf over such pages only on Linux 7.2 or later. The runtime does not check
this, and cannot: it reads no kernel version.

The granule is the alignment in which the device binds a sparse buffer's memory, rounded up to the
host's page, offered where the device has `sparseBinding`, the runtime's queue family binds sparse
memory, a sparse buffer of the floor's usage imports a dma-buf with no dedicated allocation, and
such a buffer takes every host-visible, host-coherent type a plain one takes, so that wherever
`import_dma_buf` takes a type for an fd, `import_dma_bufs` takes the same. Measured: 64 KiB on the
Radeon 780M and the RX 550 (RADV, Mesa 26.2.6) and 4 KiB on lavapipe; none on the Iris Xe (ANV,
Mesa 26.2.6) and the RTX 3090 (NVIDIA 620.70.06), whose sparse buffers import no dma-buf. A
device's live sparse buffers stay within its `sparseAddressSpaceSize` (2 GiB on lavapipe), past
which the call is `Limit::SparseAddressSpace`.

The no-copy imports have no read-only form: `import_host_memory` takes pages mapped readable and
writable, and Vulkan defines no read-only import. Metal's private read-only option (bit 24 of
`MTLResourceOptions`, not in the public header) so has no counterpart here: over pages a consumer
imports, a GPU write reaches them, where Metal drops it.

The two rules differ because a failed dma-buf import is not harmless: on NVIDIA's Linux driver
(620.70.06, an RTX 3090) one failed dma-buf import into a later type the query offered broke every
later allocation of that type, the Shared type among them, so the dma-buf rule attempts nothing it
is not sure of. The host-memory rule takes the first host-visible, host-coherent type offered:
where RADV (Mesa 26.2, an RX 7900 XTX) refused the pages of a shared mapping of a memfd at the
import, its Shared allocations and a dma-buf import still worked after it. A device that does not
list a handle's extensions, or whose buffers do not import it, refuses that call with
`Limit::NoImport`, and an address or length off the device's alignment is `Limit::ImportAlignment`.

An imported block is Shared, and is the whole of an allocation of its own, whose number
(`BlockInfo::allocation`) no hook is told of. Each import is one `VkDeviceMemory` allocation and
counts against the device's allocations; `Gauges::imports` counts those live. The block lives
until its last handle and its last buffer, command buffer or upload are gone, and every command
buffer committed before then has retired (one may reach it by address alone); then the block-maker
thread frees its memory, which ends the import, and runs `release`, Metal's deallocator, after it
(at teardown, on the thread that tears down). Neither `BlockMappings` hook runs for an imported
block.

Each live import costs every submission host time in the driver, whether that submission uses it
or not. On Mesa's AMD driver (RADV), every imported or exportable allocation is on a device-wide
list that each `vkQueueSubmit2` hands to the kernel, and the kernel checks every entry at every
submission. The runtime's own pooled blocks are not exportable, so they are not on that list. On
Mesa's Intel driver with i915 (ANV), every allocation, pooled or imported, is in every submission's
list; that is read from Mesa's source, not measured. Measured with 1 MiB imports held idle beside a
recorded play, without the validation layer:

| Device | None added | 450 dma-bufs added (`import_dma_buf`) | 450 host pointers added (`import_host_memory`) | Job |
|---|---|---|---|---|
| Radeon 780M (RADV, Mesa 26.2.6, Linux 7.3.3) | 20.6 µs | 243.1 µs | 1,458.5 µs | m3-max-68519 |
| A second Radeon 780M, same driver | 48.8 µs | 273.7 µs | 1,583.0 µs | m3-max-68535 |
| RTX 2080 (NVIDIA 620.70.06) | 24.9 µs | not imported | 27.5 µs | m3-max-68535 |

Each figure is the time of one `vkQueueSubmit2` call. On the 780M a dma-buf costs about 0.5 µs a
submission, almost all of it in the kernel. A host-pointer import costs 3.2 to 3.4 µs a submission
for 1 MiB, as the kernel walks its pages at each submission. On the RTX 2080, 450 host-pointer
imports add about 0.006 µs each, inside the spread of the runs with none, and its dma-bufs are
refused by the dma-buf rule above. A buffer placed at an offset in a block it shares with others
(`Device::new_buffer`) adds nothing to a submission. So a consumer keeps its imports few and large.
Not measured: the RX 7900 XTX, the Iris Xe, and imports of other sizes.

On a guest's own work the cost shows in the wall time and the submitting thread's CPU time, not in
the score. A guest's compute benchmark ran on a Radeon 780M (RADV, Mesa 26.2.6) twice in each of
two arms. On the device's own answer, unified memory (it is an integrated GPU), the guest makes
no-copy buffers over its own memory, and about 800 imports are live at the most, about 310 of them
dma-bufs. On the other answer, given by a probe for this measurement alone, the guest makes Managed
buffers instead, and about 120 imports are live at the most, one of them a dma-buf. First run, then
second:

| Guest's buffers | Imports live at most | Time in each `vkQueueSubmit2` | Submitting thread's CPU time | Scored pass | Score | The runtime's Shared and Private memory at peak, by its gauges | Jobs |
|---|---|---|---|---|---|---|---|
| No-copy | 802, 795 | 0.832, 0.803 ms | 12.99, 12.74 s | 174.4, 172.9 s | 21,711, 21,996 | 988, 821 MiB | m3-max-68988, m3-max-69040 |
| Managed | 118, 148 | 0.119, 0.114 ms | 2.34, 2.29 s | 162.4, 163.3 s | 22,164, 22,003 | 3.31, 3.34 GiB | m3-max-69016, m3-max-69073 |

Between the arms' means, the submitting thread's CPU time differs by about 10.5 s and the scored
pass by about 10.8 s; the score differs by 1.1%. By the runtime's gauges the Managed arm holds about
2.3 to 2.5 GiB more Shared and Private memory at its peak; that is new host memory, since the
no-copy arm's imports are the guest's own pages. The driver's own total of device memory counts
those imports, and reads 2.36 and 2.25 GiB against 3.66 and 3.69 GiB, about 1.4 GiB more.

A texture over a buffer in a dma-buf's one import (`import_dma_buf`) is in place, as over a
Shared block of the runtime's own (Textures), on a device that offers a linear image for a
dma-buf import without asking a dedicated allocation. Of the devices run, that is lavapipe alone
(jobs m3-max-58579, m3-max-58583, m3-max-58587, m3-max-58589); the others' are in Known gaps of
memory blocks. Its image is
the device's linear image made for the dma-buf (`VkExternalMemoryImageCreateInfo`), bound in the
import's memory at the texture's place, in the import's memory type. It needs three things. First,
the external image query (`vkGetPhysicalDeviceImageFormatProperties2` with
`VkPhysicalDeviceExternalImageFormatInfo` for the dma-buf, of the linear tiling, format, usage and
flags) answers the handle importable, compatible and not dedicated-only, at the texture's
extent. Second, the image meets the layout, alignment and feature tests of any texture over a
buffer in place. Third, the device enables `VK_EXT_queue_family_foreign`. Such an image is made
undefined, as Vulkan requires of an image for imported memory
(`VUID-VkImageCreateInfo-pNext-01443`). Its move home, in the upload ahead of the next commit,
is the acquire from the foreign queue family (`VK_QUEUE_FAMILY_FOREIGN_EXT`) of the image,
preinitialized to general, and of the buffer's range under it, in one barrier. It is counted in
`Metrics::external_acquires`. Vulkan's external resource sharing keeps the bytes through it: "To
make the contents of the underlying memory accessible in the destination instance or API,
applications must ... Acquire exclusive ownership in the destination instance or API". The
acquire's old layout is preinitialized, Vulkan's layout of linear bytes the host wrote, because
Vulkan's note on such an acquire says "the layout specified as part of this transition will be
the true initial layout of the image. The undefined layout specified when creating it is a
placeholder", and no layout is implied for a dma-buf (Table 17 lists Direct3D 11 handles
alone). The buffer's range is in the barrier
because an acquire of the image alone would leave undefined "the contents of any portion of
another resource which aliases memory that is bound to" it. From then on, the image stays in the
general layout, a host-accessible alias of the buffer, as in place over a Shared block. A second
texture over the same bytes acquires them again at its own move home. Anything else is the
counted copy of a texture over a buffer: an image of its own, which the device copies the
buffer's bytes into before the submissions that use it and back after a command writes it
(`Metrics::over_copies`, `Metrics::over_copies_imported`). That covers a texture over host
memory's import, over several dma-bufs, or where a query does not take it (Known gaps of memory
blocks). In both forms the buffer stays the import itself.

### Exported blocks

Metal has no counterpart, as it has none of presenting: this is the consumer's own path for
sharing memory with another API, a video decoder that writes decoded frames into it for one.
`Device::new_exported_block(handle, len)` makes a Private block of `len` bytes rounded up to 64
KiB and returns it with an `Export`: a new handle of its memory, the consumer's to close, the
allocation's size and its memory type index, which an import of the handle names.

- `ExportHandle::OpaqueFd` (unix): an opaque fd, which another API of the same device and driver
  imports (`VK_KHR_external_memory_fd`).
- `ExportHandle::DmaBuf` (unix): a dma-buf, which another driver imports
  (`VK_EXT_external_memory_dma_buf`). It also needs `VK_EXT_queue_family_foreign`, the queue
  family that ownership transfers with another driver name; a device without it refuses with
  `Limit::ExportType`.
- `ExportHandle::OpaqueWin32` (Windows): an NT handle (`VK_KHR_external_memory_win32`).

The block is one form on every device: an allocation of its own in the device's Private memory
type, dedicated to the one buffer that spans it (`VkMemoryDedicatedAllocateInfo`), never a range
of a pooled allocation. Vulkan lets a driver require a dedicated allocation for an export
(`VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT`), so this is the form every driver takes, and
what exportable memory costs each submission falls on these allocations alone. That cost grows
with the exports live, and the consumer's picture set bounds it: the consumer keeps at most 64 of
its device caches exported at once, a few decoders' output pools. On RADV each live export adds
about 0.37 µs to every submission's host time (the Radeon RX 7900 XTX, without the validation
layer: 6.7 µs with none, 30.1 µs with 64); under the validation layer the RTX 2080, the RTX 3090
and the Iris Xe added at most 1.6 µs with 64. The harness's own row `export_submit_cost` times
it alone, without the layer. A handle is offered where the device lists its
extensions, which device creation enables where listed, and its buffers export it
(`vkGetPhysicalDeviceExternalBufferProperties`); elsewhere the call refuses with
`Limit::NoExport`, before anything is allocated.

Each export is one `VkDeviceMemory` allocation and counts against the device's allocations;
`Gauges::exports` counts those live and `Gauges::export_bytes` their allocations' bytes. The block
lives until its last handle and its last buffer, command buffer or upload are gone; then the
block-maker thread frees its memory. An import of the
handle may outlive the block, unused. Neither `BlockMappings` hook runs for an exported block, and
its bytes are undefined until written, as a taken block's are. No image is made in its memory,
which is dedicated to its buffer (`VUID-vkBindImageMemory-memory-10926`): a texture over a buffer
in it is the counted copy, as over host memory's import, and a heap texture in it is
`Gap::HeapTexture`. The `Block` handle states no address (`BlockInfo::gpu_address` is `None`): an
address of the block's memory goes out only as `Buffer::gpu_address` of a buffer in it, which
counts in `Metrics::exported_blocks_addressed` (Textures over a buffer, the exported block's
rule).

Once the other API's work that accessed the block has completed (work that wrote it, or that read it
after a release; the consumer's own wait on the decoder), the consumer calls
`Block::external_writes_done()` before it commits the command buffers that use the block. The call
records the block's acquire from the other API's queue family (`VK_QUEUE_FAMILY_EXTERNAL` for an
opaque handle, `VK_QUEUE_FAMILY_FOREIGN_EXT` for a dma-buf) in the upload, ahead of every command
buffer committed after it, without waiting for the GPU: one buffer barrier, counted in
`Metrics::external_acquires`. A block that is not exported is `Refusal::NotExported`. The contract
has three parts: between `external_writes_done` and the other API's next write, the other API does
not access the block, unless a release (`release_to_external`, below) comes between, from whose
completion it may read the whole block, the runtime's writes and its own earlier ones; when it
writes again, it reads nothing of the block but what it has itself written since and what a release
since handed it; and after each acquire, a byte the other API has not written since the previous
acquire is undefined to the runtime, unless a release came between the two: a release and the
acquire after it transfer ownership, which keeps the bytes the other API did not write (section
7.7.6). The acquire is not followed by a release of its own: without one, Vulkan leaves the block's
contents undefined to the other API's reads (section 7.7.6).

Where the other API is to read the block after the runtime's commands (a video encoder reading the
frames a blit wrote, or a decoder reading again, as a reference, a picture the runtime read), the
consumer calls `BlitCommandEncoder::release_to_external(&buffer)`, with a buffer in the block, after
the encoder's last command that uses it. The call records the whole block's release from the
runtime's queue family to the other API's (the acquire's family) into that command buffer, after
every command before it: one buffer barrier, counted in `Metrics::external_releases`, with no
submission of its own and no wait. The barrier that orders the release before later commands is
owed, and the command buffer's end barrier pays it where nothing follows. A buffer whose block is
not exported is `Refusal::NotExported`. The contract of this direction: the other API reads the
block from that command buffer's completion, which the consumer waits for; from the call until the
consumer's next write into the block through the runtime or the next `external_writes_done`, the
runtime does not read the block and the consumer encodes no read of it. That next write, once the
other API's reads have completed, needs no acquire: after a release, Vulkan lets the releasing queue
use the resource again without one, its contents written anew (section 7.7.6). Where the other API
writes the block in between, `external_writes_done` comes first, as above; where it only reads the
block and the runtime is to read it again, `external_writes_done` comes first too, once those reads
have completed: the acquire after the release, which keeps the block's bytes (section 7.7.6).

### Exported surfaces

Metal has no counterpart: this is the consumer's own path for sharing a picture's planes with
another API, a video decoder that writes decoded pictures or an encoder that reads rendered ones.
`Device::new_exported_surface(&SurfaceSpec { handle, pitch_multiple, start_multiple, planes })`
makes a `Surface`, whose `planes()` are textures as each `PlaneSpec`'s descriptor makes them, and
a `SurfaceExport`: the memory's `Export` (handle, size, memory type), each plane's `PlaneLayout`
(offset and row pitch) and the `SurfaceForm` it took. A plane is what `Buffer::new_texture`
takes: 2D, of one level, sample and slice, of a color format of one texel an element; the call
refuses the rest as that call does. A spec of no plane, more than four, or a multiple of 0 is
`Refusal::SurfacePlanes`.

The layout is one rule in both forms. The planes share one pitch, a multiple of `pitch_multiple`
and of every plane's texel. Plane 0 is at offset 0; each next plane starts at the first multiple
of `start_multiple` and of its texel at or past the previous plane's start plus its rows and its
`rows_below` at the pitch. A start multiple of 1 with no rows below puts each plane directly below
the one before, at the same pitch. A VA-API import asks `pitch_multiple: 256,
start_multiple: 256`, each plane holding the coded picture's rows: the luma's `rows_below` is its
height rounded up to 16 less its height, the chroma's half that rounded height less its own (8 and
4 at 1080p, none at 720p and 2160p). The layout is linear: a dma-buf's import names
`DRM_FORMAT_MOD_LINEAR`.

**In place**, each plane's texture is a linear image in the shared memory, which the other API's
writes reach with no copy. What a consumer passes to get it, by handle:

- `ExportHandle::DmaBuf`: a device with `VK_EXT_image_drm_format_modifier` (an optional extension
  device creation enables where listed) whose `DRM_FORMAT_MOD_LINEAR` entry for each plane's
  format offers sampling, rendering and both copies, exportable and importable as a dma-buf. The
  pitch is the widest of the planes' pitches the driver chooses for an image of the implicit list,
  rounded up to the multiples; each plane's image states its start and that pitch, and the driver
  must take both. Each plane's image lies on its own dedicated import of the surface's dma-buf,
  bound at its first byte, sized to the larger of its requirement and its stated offset plus its
  subresource; the allocation the dma-buf came from is freed once the imports and the consumer's
  handle exist, so a surface is two allocations, counted in `Gauges::surface_imports`. That
  allocation and every import of it take device addresses, which the surface's buffer needs: one
  payload's allocations in one device take one set of flags (ANV, of Mesa 26.2, refused a buffer's
  import with device addresses beside plane imports without them).
- `ExportHandle::OpaqueFd` and `ExportHandle::OpaqueWin32`: each plane's format's `LINEAR` tiling
  offers sampling, rendering and both copies, exportable and importable as the handle, not only
  in an allocation of its own. Rendering there is the colour attachment feature, or, where device
  creation enabled `linearColorAttachment` (Floor, Linear attachments),
  `VK_FORMAT_FEATURE_2_LINEAR_COLOR_ATTACHMENT_BIT_NV`: such a plane renders as a linear colour
  attachment alone (below). The pitch is the driver's own for each plane's `LINEAR` image: the
  planes' pitches must be equal and a multiple of `pitch_multiple` and of the texels, and each
  start a multiple of its image's alignment, past the plane before's requirement. The images are
  bound in one exportable allocation that holds no buffer, counted in `Gauges::exports` and its
  bytes in `Gauges::export_bytes`. Measured for NV12 and P010 at 1080p and 2160p: the RTX 3090
  (NVIDIA 620.70.06) lays a `LINEAR` image's rows at the row rounded up to 128 bytes, at an
  alignment of 128, so those surfaces are in place at a pitch multiple of 1 or 4 with a start
  multiple of 1, and at multiples of 256 the 1080p NV12 surface is copied (`SurfaceCopy::Pitch`:
  its rows are 1,920 bytes) while the 2160p and P010 ones are in place. The Radeon 780M and RX 550 (RADV, Mesa 26.2.6) round rows up to 256, at an
  alignment of 256, and take every one of them in place at either.

A tiling offers a plane's usage where the handle's image format query
(`vkGetPhysicalDeviceImageFormatProperties2` of the plane's format in that tiling, external as the
handle) offers it, and the tiling's own format features (`LINEAR`'s, or the
`DRM_FORMAT_MOD_LINEAR` modifier's) hold what the device's optimal image of the format has for
it: both copies; sampling and its linear filter; storage, its atomics, and its reads and writes
without a format; rendering and blending, as a texture over a buffer in place is tested. Rendering
is the colour attachment feature itself, for which
`VK_FORMAT_FEATURE_2_LINEAR_COLOR_ATTACHMENT_BIT_NV` stands only in a `LINEAR` image's features on
a device that enabled `linearColorAttachment`: Vulkan sets that bit "for renderable color formats
in the linearTilingFeatures" alone, so it holds no dma-buf plane in place.

A plane in place through that bit renders as a linear colour attachment alone, and its image is
marked: Vulkan renders such a format "as long as all the color attachments in the render pass
instance have VK_IMAGE_TILING_LINEAR tiling". A pass over a marked plane begins where every colour
attachment is a marked plane, both planes of a picture as two colour attachments of one pass
among them, with no depth or stencil attachment; any other is refused as it begins,
`Gap::LinearPlanePass` naming its `LinearPlanePassKind`: `OtherColor`, a colour attachment that is
not a marked plane; `DepthStencil`, a depth or stencil attachment, which is not Vulkan's rule but
the driver's fault (an RTX 3090 lost the device rendering a linear colour attachment beside one,
and the runtime recovers from no device loss); `Resolve`, a multisample attachment resolving into
a marked plane. A plane whose tiling has the colour attachment feature itself is not marked.

Storage joins an image's usage where its tiling offers it; a usage a plane's texture asks that its
tiling lacks is `Gap::SurfaceUsage`. A plane given no index whose usage reaches shaders is reached
through its views alone, each with an index of its own, as `Device::new_texture_for_views` makes
a texture: the textures a consumer makes over a plane are its views. A plane's image lists, for views of another format, the
formats of the texture's own list that its tiling samples, and none where the device refuses that
list; a view of another format it does not list is `Gap::TextureView` of
`TextureViewKind::SurfacePixelFormat`. The make submits nothing and waits for nothing: the
images move out of the undefined layout into the general one in the runtime's submission that
carries the next commit, or at once where `Surface::wait_until_ready` asks first, and stay in it
through every copy.

A plane whose texture's usage holds both `ShaderRead` and `RenderTarget` takes, on a device with
the attachment feedback loop, the feedback usage beside the rest where the handle's image format
query (`vkGetPhysicalDeviceImageFormatProperties2` of the plane's format in the handle's tiling,
external as the handle, with the feedback usage added) offers it with the same export features,
and the layout checks above then run on the images with it. A draw reads such a plane as an
attachment of its own pass through the loop, as it reads a texture of an image of its own (Render
encoders): the pass moves the plane's image from the general layout to the feedback layout and
back. Where the query refuses the feedback usage, or the images with it do not hold the layout
and the images without it do, the plane takes none, and a draw reads it through its shadow, the
counted copy a device without the loop takes (Known gaps of render encoders). The image stays a
linear image on the surface's own memory, at the pitch and starts the export states. A Direct3D
11 plane asks its import's query the same.

**The ready point.** That move may discard what the memory holds, so the other API writes an
in-place surface's memory only once it has completed: before the other API's first write, the
consumer calls `Surface::wait_until_ready()` on its own thread, which returns once the move's
submission has completed and has that submission go at once where it is not yet made. The
runtime's own commands need no wait: the move goes ahead of every command buffer committed after
the make. In the copy form the call returns at once. Each call in place is counted in
`Metrics::surface_waits`, with its time in `surface_wait_nanos`, the longest in
`surface_wait_max_nanos`, and in `surface_wait_submissions` the runtime's submissions it waited
for: those not yet complete as it began, the one that moves the planes among them unless it
completed first (a submission made after the make, for any command buffer, takes the moves). A
wait far longer than those submissions' GPU times waited behind work the runtime did not submit.

**The copy form** is taken where any of that fails, and the answer's `SurfaceForm::Copy` names
why (`SurfaceCopy`): `NoModifier`, `Tiling`, `Dedicated`, `Pitch`, `Start` or `MemoryType`. The
surface is then an exported block (as above) laid by the same rule at the tightest pitch the
multiples allow, and each plane's texture is a texture over a buffer in it, an image of its own
that the submissions copy the bytes into and back (`Metrics::over_copies`): in, once after each
acquire or other change of the bytes (Textures over a buffer, the exported block's rule). A start or pitch the
driver cannot take falls to the copy form with no error: the answer's `form` is the only sign.
`Metrics::surfaces_in_place` and `Metrics::surfaces_copied` count each form made.

One contract for both forms, the exported block's above, after the ready point. Once the other API's
work that accessed the surface has completed (work that wrote it, or that read it after a release),
the consumer calls `Surface::external_writes_done()` before it commits the command buffers that use
the surface: in place, each plane's image's acquire from the other API's queue family, the general
layout kept, one barrier in the upload; in the copy form the block's acquire. Where the other API is
to read the surface after the runtime's commands (what they wrote, or a reference picture they
read), the consumer calls `BlitCommandEncoder::release_surface_to_external(&surface)` after the last
command that uses the planes: in place, each plane's image's release, one barrier; in the copy form
the block's release, the planes' images copied back first. Both count in
`Metrics::external_acquires` and `Metrics::external_releases`. From a release until the consumer's
next write into the surface through the runtime or the next `external_writes_done`, the consumer
encodes no read of its planes. The surface's memory is freed once the surface, its planes'
handles and every command buffer that used it are gone. An acquire that no command buffer's
submission has carried yet goes as the last of them lets go of a plane or of the exported block,
in a submission of its own, with no wait, so the memory waits for no later call.

**The surface's buffer.** `Surface::buffer()` answers a buffer over the surface's own memory, the
counterpart of an IOSurface-backed buffer: from the memory's first byte, its bytes at the layout
the `SurfaceExport` states (plane `i`'s row `y` at `planes[i].offset + y * planes[i].row_pitch`).
In place it spans the export's size; in the copy form, its block's length, which holds every
plane and which the export's size passes where the driver's allocation is larger.
Every call answers the same buffer. It is Private, its `contents` `None`: the host reads it
through a copy into a Shared buffer. In place it is made at the first call, never at the
surface's make. Of an opaque handle it is a buffer bound at the first byte of the one allocation
the planes are bound in, which is made with device addresses for it. Of a dma-buf it is a buffer
on one more import of the dma-buf, into the memory type the planes' imports took. The surface
keeps its own fd of the dma-buf for that import. The call imports only where the device's buffers
import a dma-buf (`vkGetPhysicalDeviceExternalBufferProperties`) and the fd's query and the
buffer's requirements list that type. Either form counts in `Gauges::surface_buffers`, and a
dma-buf's also in `Gauges::surface_imports`. The first call records the buffer's acquire from the
other API's queue family. From then on, `external_writes_done` acquires the buffer with the planes
and `release_surface_to_external` releases it with them, writing back first the copies of the
textures the consumer made over it. In the copy form the buffer is the block's, which the planes
are textures over.

Vulkan's memory aliasing makes the buffer and the planes in place aliases that read each other's
writes: a buffer and linear images in the general layout are host-accessible, and two
host-accessible aliases interpret the memory alike. A dma-buf's buffer and planes are distinct
imports of one payload, which an fd's import allows. The section names aliases through export and
import across instances and APIs, and rt takes imports in one instance as those aliases. The
planes already do so, each on an import of its own. The surface, its planes and their views, and
its buffer are one resource in rt's ordering of reads and writes, as a heap's resources are. A
kernel's write through the buffer is ordered before the next kernel's read through a plane, by
the global memory barrier rt records between them. `Surface::buffer`'s doc comment quotes the
sections.

On Windows, `Device::import_d3d11_surface(&[D3d11Plane])` takes a picture's planes the other API
made as Direct3D 11 textures, each by its NT handle (`IDXGIResource1::CreateSharedHandle`), its
own format and the texture rt makes over it, and returns a `Surface` whose planes rt's shaders
read and write in place. Each plane is an optimal image of the Direct3D 11 texture's format, of
the descriptor's width and height, on a dedicated import of its handle, counted in
`Gauges::surface_imports`. The plane's VkImage takes the usage the texture's BindFlags give
(`D3d11Plane::bind_flags`, as the consumer made the texture), and no other: copied from and into,
as Direct3D 11 copies any texture of default usage; sampled for `D3D11_BIND_SHADER_RESOURCE`,
rendered for `D3D11_BIND_RENDER_TARGET`, storage for `D3D11_BIND_UNORDERED_ACCESS`; each where
the descriptor's format holds it and the import query
(`vkGetPhysicalDeviceImageFormatProperties2`) offers it. Vulkan states no rule for an image over
memory another API made with other parameters: its rule for an import of Vulkan's own export is
the same image (`VUID-VkMemoryDedicatedAllocateInfo-image-01876`), and its memory aliasing
promises a consistent reading to images of the same creation parameters. A texture whose uses
ask more (`ShaderWrite` on a texture made without `D3D11_BIND_UNORDERED_ACCESS`) is
`Gap::SurfaceUsage`. The one usage beside them is the attachment feedback loop's, which Direct3D
11's description has no counterpart for: taken where the descriptor's texture is a render target
a draw may read in its own pass. The image is mutable where it lists more than its own format, and
it lists at most two more, the two places it asks beyond the texture Direct3D 11 made: the
descriptor's format where it differs (a chroma plane of `R16Uint` under an `RG8Unorm` texture, the
same bytes per texel in Metal's same view class), and the descriptor's sRGB sibling, which a guest
may view the plane through (`R8Unorm_sRGB` of an 8-bit luma), where the device holds it and the
import query takes the list. A 10-bit luma (`R16Unorm`) lists none and is not mutable. A view of
any other format has no in-place form on the import: it is `Gap::TextureView` of
`TextureViewKind::SurfacePixelFormat`, and the consumer answers it with a counted copy of the
plane (Known gaps of memory blocks). The consumer keeps and closes the handles.

The import submits nothing and waits for nothing. Vulkan 1.4's section 12.5.2 says "When images
created with external memory handle types listed in the external image implied layouts table are
accessed outside of Vulkan, they must be in the layout specified in the table, and can be assumed
to be in that layout when they are accessed in Vulkan after external access", and the table's row
for `VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT` is `VK_IMAGE_LAYOUT_GENERAL`. Each
submission whose command buffers or uploads use a plane acquires it from
`VK_QUEUE_FAMILY_EXTERNAL` in a command buffer of rt's own ahead of everything, and releases it
to that family in another after everything, both in the general layout: one barrier each, counted
in `Metrics::external_acquires` and `Metrics::external_releases`. The acquire follows a global
barrier from and to every stage, so an earlier submission's release on rt's one queue happens
before it (Vulkan 1.4, section 7.7.6). The submission counts in
`Metrics::d3d11_plane_submissions` and the two command buffers in
`Metrics::runtime_command_buffers`. The release takes the submission's last signals, so it adds
no `VkSubmitInfo2`, and no submission or semaphore wait is added. Direct3D 11 thus owns the
planes' memory between rt's submissions, and `Surface::external_writes_done()` and
`BlitCommandEncoder::release_surface_to_external(&surface)` record nothing for such a surface. So
the consumer's precondition is that Direct3D 11's work that accesses a plane has completed before
the consumer commits the command buffers that use it, and that those command buffers have
completed before Direct3D 11 accesses it again. No keyed mutex is used, by rt or on the textures:
each is made with `D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE`, not
`D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX`, whose mutex the device that makes the texture and each
device that opens it must acquire before rendering to it; the consumer orders the two APIs' work
by its waits. A plane rt uses before any Direct3D 11 access has undefined contents, as a Metal
texture nothing wrote does. The surface has no buffer: `Surface::buffer()` is `Gap::D3d11SurfaceBuffer`, as its
planes lie in memories of their own, and `Surface::handle()` is `None`. No move goes ahead of the
planes' first use, so `Surface::wait_until_ready()` returns at once. The planes' imports are freed
as an exported surface's memory is (above); rt keeps no handle of its own, and each import holds
the Direct3D 11 texture's payload until it is freed.

### Known gaps of memory blocks

- On a device whose Shared type is not host-coherent (Shared memory's coherence), the flushes and
  invalidates are that device's cost: before each submission over what its command buffers reach,
  and at each completion and each shared event's GPU-raised value seen over what they may write, on
  the submit and completion threads and in the calls that read an event's value. A submission's
  flush names every Shared range its command buffers reach, written or not, merged, so its cost
  grows with the Shared bytes a submission reaches, not with what its commands wrote. Every device
  of ours has a host-visible, host-coherent, host-cached type, which Shared takes, so none makes
  these calls, and none can show a flush the runtime missed: only a device whose Shared type is
  cached and not coherent can. The device tests run the calls on our devices through a view that
  reads their cached types as not coherent (`--cfg m2v_shared_view`, docs/VALIDATION.md), on memory
  that is coherent in fact; neither the calls' time nor their effect has been measured on a device
  whose memory is not.
- On such a device, a host write to a range a command buffer may write, made while the command
  buffer runs, may be lost, and so may the GPU's writes to the same atoms. The runtime flushes
  each such range just before it invalidates it, and Vulkan states: "If a range of non-coherent
  memory is written by the host and then invalidated without first being flushed, its contents
  are undefined." A host write that lands between the flush and the invalidate is such a write.
  Vulkan's `nonCoherentAtomSize` is "the size and alignment in bytes that bounds concurrent access
  to host-mapped device memory", so the same holds where the host's write and the GPU's touch
  different bytes of one atom.
- On such a device, a Shared buffer that a command buffer reaches only through a residency set is
  neither flushed nor invalidated: the runtime has no residency set, and such a buffer is one the
  commands reach through an address with no declaration naming it.
- On the Radeon 780M (RADV of Mesa 26.2.6) a texture over a buffer in a dma-buf's one import is
  the counted copy (`over_copies_dma_buf`). The external image query refuses the dma-buf handle
  with `LINEAR` tiling (`VK_ERROR_FORMAT_NOT_SUPPORTED`) for RGBA8Unorm and RGBA8Uint, for every
  usage asked: sampled, sampled and storage, either with both copies, and mutable with the sRGB
  view. The same linear image without the handle is offered, with the `LINEAR` features equal to
  the optimal ones, and takes the program's own pitch: 7,680 bytes at 1920 wide, 9,984 at 2448
  wide (rows padded to 256) and 4,096 at 1024 wide, at an alignment of 256 and with no dedicated
  allocation (job m3-max-68787). With `DRM_FORMAT_MOD_LINEAR` tiling the query offers the
  dma-buf image for every one of those usages, importable and not dedicated-only; an image made
  with the program's pitch in its explicit plane layout takes that pitch, and its memory
  requirements require a dedicated allocation (job m3-max-68847). An import of its own for each
  such texture, dedicated to its image, was not tried and is not built. The copies it could
  remove are 3.54 GB of the 40.49 GB a guest's run of the same compute benchmark copies over
  its imports (job m3-max-68603), under 5% of each workload's time (above), so it stays a known
  gap. The runtime asks the `LINEAR` tiling alone.
- On the Iris Xe (ANV, Mesa 26.2.6) the dma-buf query offers a device-local-only type first, so
  its dma-bufs are refused with `Limit::ImportType`, and a consumer imports those pages by pointer.
  Its sparse buffers import no dma-buf either, so `Device::import_dma_bufs` refuses there, as it
  does on the RTX 3090 (NVIDIA 620.70.06).
- Under AMD's open Linux driver (AMDVLK 2025.Q2.1, LLPC) on a Radeon 780M (1002:15bf, and the
  same on a 1002:1900), no no-copy import takes a guest's pages, so the consumer copies them. The
  pages are shared mappings of memfds, and their dma-buf is a udmabuf of the memfd's pages.
  `Device::import_dma_buf` refuses with `Limit::ImportType`: the fd's query offers every memory
  type (0xffff), and the first, the only one the dma-buf rule takes (Imported blocks), is not
  host-visible and host-coherent. `Device::import_host_memory` of the same pages fails in the
  driver, an `Error::Vulkan` of `vkAllocateMemory` with `VK_ERROR_INVALID_EXTERNAL_HANDLE`:
  both AMD Linux drivers on that card import a pointer to private anonymous pages only, and refuse
  shared or file-backed pages such as the guest's.
  `Device::dma_buf_granule` is `None` there, so pages that span more than one dma-buf holds have
  no dma-buf route. Under RADV (Mesa 26.2.6) on the same card the same pages import as dma-bufs
  (`import_dma_buf`, and `import_dma_bufs` where they span more than one holds). In one guest run
  of a GPU compute benchmark from a cold boot on the 1002:15bf, the consumer copied the pages of
  1,693 imports of no-copy buffers and heaps: 1,491 that both calls refused, 165 that span more
  than one dma-buf holds, and 37 read-only ones, which it copies on every device (no import is
  read-only, Imported blocks). The writable copies took part in 17,083 commits, compared 1.82 TB,
  and wrote 0.98 GB to the copies and 60.46 GB back to the pages. The copies' syncs took 230.2 s
  of the consumer's CPU on the committing and completion threads. In a run of the same benchmark
  under RADV on the same card, 1,622 imports were dma-bufs (168 of them over several), 32 were
  read-only copies, and the copies' syncs took 5.7 ms. On a Radeon 780M (1002:1900) under AMD's Windows driver
  (26.11.2, LLPC), in a guest session (not that benchmark run), all 998 of the consumer's
  host-pointer imports took the pages in place, none refused, and its 97 copies were the
  read-only ones.
- An exported surface takes the copy form, its planes copied at each acquire and release, in
  four cases the build found. A copy form's planes are textures over a buffer in the exported
  block, which is Private, so the CPU's `replace_region` and `get_bytes` on them are refused
  (`Refusal::TextureBytesPrivate`):
  - A dma-buf surface on a device without `VK_EXT_image_drm_format_modifier`, or whose
    `DRM_FORMAT_MOD_LINEAR` entry lacks a plane's format or one of its four usages
    (`SurfaceCopy::NoModifier`, `SurfaceCopy::Tiling`): the device offers no in-place form for it.
    On the Radeon RX 550 (Polaris 12, RADV of Mesa 26.2.6) the driver lists no modifier extension,
    and it offers no dma-buf image in `LINEAR` tiling either: its image format query refuses the
    dma-buf handle with `LINEAR` tiling for every usage, though the `LINEAR` format features of
    R8, RG8, R16 and RG16 hold sampling, the colour attachment, both copies and storage (jobs
    m3-max-66247 and m3-max-66261). So neither in-place form exists there; the Radeon 780M and the
    RX 7900 XTX answer that query alike. Each plane is copied in once a change of its bytes, an
    acquire among them (the exported block's rule, Textures over a buffer).
  - A dma-buf surface on NVIDIA (`SurfaceCopy::Tiling`): on the RTX 3090 and the RTX 5080 Laptop
    GPU (NVIDIA 620.70.06) the `DRM_FORMAT_MOD_LINEAR` modifier's features carry
    `VK_FORMAT_FEATURE_2_LINEAR_COLOR_ATTACHMENT_BIT_NV`, not the colour attachment feature, for
    every plane format it is listed for (R8, RG8, R16 and RG16 among them on the RTX 3090, jobs
    m3-max-66251 and m3-max-66264), and that bit holds no modifier's image in place
    (Exported surfaces); the other formats measured have no such modifier there. The opaque fd surface of the same
    planes is in place where its pitch multiple divides the driver's pitch (above), its planes
    marked, and is the NVIDIA route a consumer takes.
  - A layout the device cannot meet (`SurfaceCopy::Pitch`, `SurfaceCopy::Start`): a pitch or
    start the consumer asks that the driver's linear images do not take. A pitch multiple of
    4,096 takes `SurfaceCopy::Pitch` on every device measured: the Radeon 780M, the RTX 3090,
    the Iris Xe, the RX 550 and lavapipe.
  - A plane whose image the device exports only in an allocation of its own
    (`SurfaceCopy::Dedicated`): the external image format query answers
    `VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT` for the plane's tiling and usage, or, on the
    opaque handle's route, the plane's image requires a dedicated allocation. Under AMD's open
    Linux driver (AMDVLK 2025.Q2.1) every route the runtime tries takes it, on the RX 7900 XTX
    and the Radeon 780M (the runtime's surface test, jobs m3-max-70391, m3-max-70425 and
    m3-max-70426), so the conformance cases `surface-textures-*` are refused at `replace_region`
    there; RADV on the same cards answers all ten exactly (jobs m3-max-70513 and m3-max-70514).
- A plane of a Direct3D 11 import (`Device::import_d3d11_surface`) is viewed in place only in the
  image's own format, the descriptor's and the descriptor's sRGB sibling, as its image lists no
  other format (Exported surfaces). A view in another format, which `PixelFormatView` usage lets
  a Metal texture over an IOSurface's plane take, is `Gap::TextureView` of
  `TextureViewKind::SurfacePixelFormat`: the import has no in-place form for it. The consumer
  takes the copy form of the same call for such a texture, a copy of the plane in a texture of
  its own, which it counts.
- On the Iris Xe (ANV, Mesa 26.2.6) a 4K dma-buf surface takes 3.05 ms for P010 and 1.80 ms for
  NV12 from the start of its make to the return of `Surface::wait_until_ready()` (medians of ten),
  where the M3 Max takes 1.62 and 1.22 ms to create an IOSurface of the same planes, make its two
  plane textures and clear both in one waited render pass. The make is 0.25 and 0.21 ms of it; the
  rest is the wait for the submission that moves the planes, the first command buffer to use the
  new memory, which the runtime records and submits on its own thread. That submission reaches
  its first GPU timestamp 2.94 ms after it is submitted for P010 and 1.43 ms for NV12, against
  0.26 and 0.32 ms without the moves, and the delay grows with the size. A fill of a newly
  allocated block of P010's size takes 8.50 ms at its first submission and 4.34 ms at its second,
  3.52 ms of GPU time in both. The device lacks the zero-initialize feature and the runtime writes
  no zeros there. The time is consistent with the kernel making the memory's pages resident at
  their first submission; which kernel work it is was not measured.
- On the UHD 630 under Intel's Windows driver (101.2145) a 4K surface exported as an opaque Win32
  handle takes 8.45 ms for P010 and 4.28 ms for NV12 from the start of its make to the return of
  `Surface::wait_until_ready()` (medians of ten), against the M3 Max's 1.62 and 1.22 ms. The
  driver's `vkAllocateMemory` of the planes' memory, 24,903,680 and 12,451,840 bytes, is 6.99 and
  3.40 ms of it. Of the same size and memory type, in another run, that allocation takes 0.23 and
  0.11 ms without `VkExportMemoryAllocateInfo` and 10.15 and 6.41 ms with it (8.07 and 4.92 ms
  without device addresses): 0.27 to 0.51 ms a MB across the runs. A buffer in the exported
  allocation takes the same GPU time at its first fill as at its second, 14.15 and 14.09 ms at
  P010's size. The ready wait is 1.12 and 0.61 ms.
- A shader uses a buffer's `gpuAddress` directly, unchecked, wherever the program holds one: a
  pointer member of an argument buffer, an address it wrote into a buffer, a pointer it made from
  an integer. The stored word must be a buffer's `gpu_address` plus an offset inside that buffer,
  and a buffer the consumer did not place at the address its guest asked for has a `gpu_address`
  other than that one. A stale address, of a released buffer, can still fault the device. An
  address in an exported block is one the consumer read from rt before the commit of the command
  buffer that writes through it (`Buffer::gpu_address` of a buffer in the block, whose
  `BlockInfo` states no address): a write through an address made any other way does not reach
  the textures over the block, whose copies in owe nothing for it (Textures over a buffer).
- An app buffer too large for the consumer's map budget is copied, on drivers whose dma-buf query
  offers a non-host-visible type first.
- A block freed before a submission used it waits on the block-maker thread for its bind, and a
  ring take that needs a new ring block can wait behind it.
- rt caps a buffer at 2 GiB (`MAX_BUFFER_LENGTH`), where the M3's Metal allows 38.88 GiB
  (`maxBufferLength`, 41,747,087,360 bytes), about twenty times as much: shaders reach a buffer's
  bytes by 32-bit offsets from its bind address, and lifting the cap would put 64-bit offset
  arithmetic on every buffer access. The word rt pushes for a buffer slot is the slot's address
  less 2^31; an access adds its 32-bit offset plus 2^31 to that word, zero-extended, which is the
  offset read as signed in one 32-bit add. In a probe of one particle kernel that form was within
  0.7% of a zero-extended offset (inexact below the bind address) on the RX 7900 XTX and two
  Radeon 780Ms, at the kernel's grid and at four times it, where a sign-extended offset was 6.8
  to 9.5% slower. On the Intel UHD 630's Windows driver the compute harness
  measured three of its 38 workloads slower in this form than in the sign-extended one:
  image_filter by 3.4%, dispatch_chain by 2.6% and burst_160 by 2.2%, the median moving 0.05%.
  Each reaches a buffer through a computed index. On that device (Gen9) the cause is inferred, not
  measured: one more 32-bit add per access, which it does not fold into the index's shift. On the
  Iris Xe (Gen12, Mesa's ANV) the form's median over the harness's 38 workloads is +0.03%, with
  three rows up (alu and long_4cb 14%, image_filter 7.6%) and three down (zero_products 18%,
  vote_loop 9%, frame_5cb 7%). There a change in the code that runs before an otherwise identical
  loop moves the kernel by -18% to +14%, by a mechanism not shown: not the code's placement, not
  the instruction count, not the thread payload and not register numbers. ANV's compiler has no
  64-bit integer add on Gen12, so each access through an address costs, beyond a binding-table
  access, a compare for the carry, an add of the high word and four moves that pack the address
  into the message, and the bias one 32-bit add more: the particle kernel's loop, which reads
  once an iteration, is 31 instructions there. That kernel alone, back to back, takes 1.26 times
  as long per dispatch as a native Vulkan program of the same work (502.0 against 397.0
  microseconds), its output bytes equal to the native program's. Intel's Windows driver on Gen12
  is unaudited. In a probe of constant reads, 32-bit offsets bought 4.3 to 6.6% on AMD cards and
  more on the RTX 3090, where a 64-bit sum was 8.9% slower on the RX 7900 XTX. A longer buffer is
  `Refusal::BufferTooLong`.
- A dispatch whose indirect arguments, or an indexed draw whose indices, start at a byte of their
  block that is not a multiple of 4, or of the index size, is refused (`Refusal::ArgumentPlace`).
  Metal's offset rules keep both at that multiple of the buffer's start, so only a buffer that
  starts off it meets this: a no-copy buffer over a pointer that is not a multiple of 4 (indirect
  arguments) or of the index size (indexed draws), which documented Metal does not allow, its
  pointer being page-aligned. The consumer places every other buffer at an aligned byte. Such a
  buffer is still made and its other uses run; only the dispatch or draw is refused.
- A texture over a buffer in host memory's import (`Device::import_host_memory`) is the counted
  copy of a texture over a buffer on every device (`Metrics::over_copies`,
  `Metrics::over_copies_imported`): its bytes are copied into an image of its own before the
  submissions that use it, and back after a command writes it. This is by handle type, not by
  host: the guest pages of the Mac and Windows hosts, which import by address, and on Linux any
  pages not imported as a dma-buf (on the Iris Xe and the RTX 3090, all of them). An image bound
  in imported memory is made for the import's handle, so "initialLayout must be
  `VK_IMAGE_LAYOUT_UNDEFINED`" (`VUID-VkImageCreateInfo-pNext-01443`), and a move out of the
  undefined layout keeps no bytes. Over a dma-buf, an acquire from the other side keeps them
  (Imported blocks). Host memory has no other side to acquire from: "Importing memory from a host
  pointer shares ownership of the memory between the host and the Vulkan implementation. The
  application can continue to access the memory through the host pointer but it is the
  application's responsibility to synchronize device and non-device access to the payload as
  defined in Host Access to Device Memory Objects" (Vulkan 1.4, Host External Memory). The host
  reaches a linear image only in a host-accessible layout, which such an image cannot be made in.
  The drivers would make that image: for host memory's handle the external image query offers a
  linear image of every format and usage it offers one of without a handle (13 colour formats,
  read, written and rendered, asked of lavapipe, RADV and ANV of Mesa 26.2.6, NVIDIA's 620.70.06
  and Mesa 26.3.1's Mac driver, which makes no linear render target with or without one), and
  each one asked binds in a host-memory import, so the bar is the undefined start that
  `VUID-VkImageCreateInfo-pNext-01443` makes mandatory, not the driver. In a guest walk on an M3
  Max on Mesa's Mac driver (rt at 306d357141, job m3-max-66271), whose guest pages are imported
  by address (20 steps, 5,969 presents in 481 s), these copies came to 4.67 GB, 97.2% of the
  walk's 4.81 GB of over-copies, every one over a no-copy buffer on a guest program's own pages.
  On average that is 9.7% of a 1080p frame's bytes a present, and from 0.44% to 45.7% a present in
  the steps that present more than once. On a discrete card the copy's time follows the link's
  state. On the RTX 5080 Laptop GPU idling at P8, where its driver keeps PCIe at Gen1 x8, the
  runtime's own command buffers moved 553.5 MB in 362.4 ms (1.53 GB/s), so one copy-in of a
  10,240,000-byte texture took about 6.7 ms. At P0, P2 and P4 the same card's link runs Gen5 x8.
  In a 30 s scrolling step of a guest walk (job m3-max-72689) these copies came to 534.8 MB in
  52 copies of 20 textures, and 26 of the step's 50 GPU starts that came 8 ms or more after
  their scheduling fell in 50 ms intervals holding such a copy of 10.24 MB or more.
  Documented Metal asks for no call before a CPU write to a Shared buffer reaches a texture
  over it, so the runtime has no call to hold the copy for.
- A texture over a buffer in a dma-buf's one import is the counted copy too on a device that
  offers no linear image for a dma-buf import without a dedicated allocation (Imported blocks).
  On the Radeon 780M (RADV) the external image query refuses the dma-buf handle with linear
  tiling, and each image of a DRM format modifier it offers asks a dedicated allocation, which
  cannot be bound in the buffer's import: "If image requires a dedicated allocation (as reported
  by vkGetImageMemoryRequirements2 in VkMemoryDedicatedRequirements::requiresDedicatedAllocation
  for image), memory must have been created with VkMemoryDedicatedAllocateInfo::image equal to
  image" (`VUID-vkBindImageMemory-image-01445`). On the Iris Xe (ANV) and the RTX 3090 (NVIDIA)
  the runtime imports no guest dma-buf: the dma-buf rule imports only into the first type the
  fd's query offers, and only where that type is host-visible and host-coherent (on NVIDIA's
  driver one failed import into a later type broke every later allocation of that type, Imported
  blocks), and the first type those devices offer is not. Their pages take the host-memory import
  above. On the Iris Xe the consumer's own no-copy pages stay the counted copy at each submission
  for a second reason too: one dma-buf holds only what udmabuf's stock limit of 1,024 runs
  allows (about 4 MiB of a guest buffer), and the runtime binds several dma-bufs behind one buffer
  only where the device has a granule, which the Iris Xe has not (Imported blocks). Each of these
  is a known gap, reopened only by a measured copied share above 5% of a frame on that card.
- A texture over a buffer in an import over several dma-bufs (`Device::import_dma_bufs`) is the
  counted copy too: the block's buffer is sparse, bound to one import per dma-buf.
- A texture over a buffer in an exported block is the counted copy of a texture over a buffer on
  every device (`Metrics::over_copies`). The runtime makes no image in an exported block's memory,
  which is dedicated to its buffer. A decoded picture in an exported block is then one copy a
  plane into the plane's image, made by the first submission that uses the picture after its
  acquire; later submissions that only read it copy nothing, and a picture the decoder writes
  over before any use is not copied. In the video decodes we replay
  (two planes a picture: 3,110,400 bytes for an 8-bit 1920x1080 picture, 24,883,200 for a 16-bit
  3840x2160 one), that is 2 copies and about 8.6 MB a picture used, and a picture's copies take
  0.288 ms of GPU time on the Radeon 780M, 0.294 ms on the RX 550, 0.520 ms on the Iris Xe
  (Raptor Lake) and 0.653 ms on the RTX 3090. In a 336-second guest walk on an RTX 3090 (rt at
  7096fa7de3, which copied a texture over any import), all 13,613 such copies (228.6 MiB) were of
  textures over imported memory, 1.05 copies and 16 KB a frame on average over 13,005 frames and
  78 copies in the worst one; no texture over a block of the runtime's own was copied.
- On the NVIDIA cards we measured in Thunderbolt enclosures, freeing host-visible memory takes
  about 1.1 ms per MiB and holds any submit of the same process for its length; on an internal
  NVIDIA card the same free takes about 0.5 ms per 16 MiB. rt frees one idle allocation at a time,
  at its idle age, whatever the queues are doing. `Metrics::allocations_freed` counts the frees,
  `Metrics::submissions_during_frees` the submit calls that met one of the block-maker thread's
  frees, and `Metrics::submit_call_nanos_during_frees` their time.
- NVIDIA's Linux driver holds a submit made while another thread is inside `vkAllocateMemory` or
  `vkFreeMemory` until that call returns, device-local memory included, and calls made back to
  back hold it through all of them. So as each of its allocations and frees returns, the
  block-maker thread marks the `vkQueueSubmit2` calls of rt's inside the driver then, and starts
  its next one only once all of them have returned; with none left inside it goes straight on, and
  what it made is handed on as its call returns. A submit waits out at most one of its calls, the
  one it met. Its calls are the makes of the pool's 16 MiB units and of the
  runtime's fixed-size internals, and the frees of idle allocations, of the imported and
  exported blocks given back, of textures' allocations of their own, of surface planes' imports
  and of copies' staging buffers of their own. `Metrics::maker_memory_calls` counts them, `maker_yields` the ones
  after which it waited, and `maker_yield_nanos` the time. A single call still holds a submit for
  its own length: a take longer than a unit allocates its own length on the consumer's thread,
  with no wait, and that allocation goes back to the driver in one free of its whole length. In a
  standalone program on an RTX 3090 (driver 620.70.06), giving back 1 GiB of Shared memory in
  16 MiB frees made back to back held the worst submit for 75.3 ms, and with this wait between
  the frees for 20.6 ms; at every unit length measured the worst hold with the wait was 0.90 to 0.96
  times the longest single free, and with no submit made the wait added nothing. In a 336-second
  guest walk on an RTX 3090 (driver 620.70.06, rt at 7096fa7de3, whose block-maker thread made its
  calls back to back), every submit of 5 ms or more overlapped such calls on other threads for 97
  to 100% of its length: a 64 MiB Shared allocation made inside a take held one submit for
  14.2 ms; frees of imported blocks, 0.4 to 2.9 ms each and 52 in one second, held four submits for
  5.2 to 6.9 ms; and the pool's trim after the workload, five 64 MiB frees of 7.7 to 13.9 ms made
  back to back, held one submit for 48.7 ms. A take that leaves its class no wholly
  free unit also sets the block-maker making the class's next allocation at once, so the take's
  own first submission can wait out that allocation: on the same card (rt at ac1896537f) a
  16 MiB Private allocation of 661 µs held the next submit for 603 µs, and on an RTX 2080 one of
  647 µs held it for 579 µs.
- Under AMD's open Linux driver (AMDVLK 2025.Q2.1) on a Radeon 780M, a 16 MiB allocation takes
  over 5 ms about one time in eight. In one 591-second guest desktop session (rt at 7360a148bb),
  60 of the block-maker thread's 502 allocations took over 5 ms, the longest 129.1 ms, and 8 of
  the 62 made on a caller's thread did, the longest 35.5 ms. The pool keeps one wholly free
  16 MiB unit per class made ahead, so textures made in a burst, faster than one allocation each,
  take their blocks on the call, and that `Device::new_texture` waits for the allocation: 49 of
  the session's 350 texture blocks were made so, the longest in 29.06 ms. The pool holds no more
  memory ahead to avoid it. A take does not wait for an allocation the block-maker thread is
  making, which would inherit that allocation's tail.
- NVIDIA's Linux driver allocates host-visible memory at about 0.14 ms per MiB on an RTX 3090 and
  0.2 ms per MiB on an RTX 2080, in every form we tried (plain, with a device address, dedicated,
  cached or not), against about 0.03 ms per MiB for device-local memory on the 3090. Device-local
  memory made dedicated to its buffer allocates 7 to 30 times as fast there and frees 5 to 20 times
  as fast, with nothing moved to its first use (547 MiB in 0.51 ms against 15.48 ms, freed in
  0.81 against 15.95 ms; job m3-max-65734). The runtime's pool allocations are plain: they hold
  textures, which Vulkan keeps out of memory dedicated to a buffer
  (`VUID-vkBindImageMemory-memory-10926`), and on a Radeon 780M and an Iris Xe a dedicated
  allocation with its first fill was not faster (up to 8.7% slower; jobs m3-max-65776 and
  m3-max-65785). A take longer
  than the pool's 16 MiB allocations allocates on the calling thread and pays it there: in one
  consumer's replay stream on the 3090, six Shared takes of 547 MiB took about 70 ms each, and its
  Shared takes past 16 MiB took 1.46 s of the stream's 210 s. The pool keeps no memory to avoid
  it; `Metrics::consumer_allocation_nanos` counts it. On the 3090 (driver 620.70.06, rt at
  140714ceae), 64 MiB of rt's Shared memory type costs about 12.5 ms in `vkAllocateMemory`, with
  or without a device address, and 2.75 ms more for rt to zero it through its mapping. Host pages
  a process maps itself and imports by host pointer cost about 5.8 ms for 64 MiB there, 5.4 ms of
  it the import, and importing fresh pages so takes 0.35 to 0.54 times as long as allocating them,
  from 16 to 256 MiB on both cards. No host-visible type outside the card's 246 MiB
  device-local heap is cheaper, and a Shared allocation is the driver's own memory on every
  device, so the import is not open to it. A view of guest pages, one mapping a page and none
  populated (29.7 MB in 7,261 mappings), imports by host pointer in 6.6 to 6.8 ms median and
  10.8 ms at most there, on the CPU in the driver with one page fault a page (3.4 to 5.1 ms with
  the pages present), and its free takes 30 to 31 ms, about 1.03 ms per MiB (a standalone program,
  job m3-max-66022). Beside another thread's allocation or free such an import waited up to
  100 ms on the driver's own lock, which the runtime does not choose. In another consumer's replay stream on the 2080 (the
  same driver, rt at c810d2a3df), the first 1138 frames make 304 Shared takes of 9.5 GiB, whose
  allocations took 0.90 to 0.93 s in two plays, under 1% of the play. On three loading frames of
  1.3 to 3.6 s they are 6 to 8.3% of the frame (at most 301 ms of 3.64 s); in the play that
  listed every frame, they are at most 1.7% of any other.
- On NVIDIA's Linux driver each make or free of memory also holds every other thread's submits
  for the length of the call, so a consumer's submits beside the block-maker thread's make of a
  unit, or its free of an idle one, wait for it. reims's measurements on driver 620.70.06, each the
  longest submission beside the call, against an empty submission's 1.76 ms on the RTX 3090 and
  2.08 ms on the RTX 2080:
  - device-local memory: 16 MiB at the baseline; 256 MiB 6.6 to 8 ms beside a make and about
    5 ms beside a free;
  - host-visible, coherent and cached memory (the Shared class): 16 MiB 3.3 ms at the median and
    4.5 ms at most beside a make, and beside a free 4.1 ms at the median, up to 15.4 ms on the
    2080 and 19 ms on the 3090; 64 MiB 15.2 ms beside a make and 69.5 ms beside a free on the
    2080; 256 MiB 51 ms on the 3090 and 58 ms on the 2080 beside a make, and beside a free 93 ms
    at the median, up to 207 ms on the 3090 and 242 ms on the 2080.

  rt's Shared units are the driver's own memory on every device, so these holds are what the
  consumer sees beside each one made or freed.
- On the AMD kernel driver (amdgpu) a Shared allocation's make gathers its pages inside the
  driver's create, under the reservation of the device's page tables, which every submit, import,
  map and close on the device also takes. With the host's free memory low the kernel compacts
  memory first, and those calls wait behind it. On a Radeon 780M (RADV of Mesa 26.2.6, Linux
  7.3.3), with the host's free memory at 303 to 318 MiB before each set of makes, beside 8 GiB of
  resident shared memory and 20.5 to 21.0 GiB of page cache, 16 MiB makes met 5.2 and 6.9
  compaction stalls each on average in two passes, and the longest submit held behind one took
  13.87 ms (a standalone program, job m3-max-66343). In a guest walk on a 780M (rt at 4e9aad67d5, job
  m3-max-66278) the long 16 MiB makes met 5 to 18 compaction stalls each, and 8 of the 9 submits
  of 5.3 to 22.6 ms waited behind a make. The runtime keeps its one unit length on every device,
  since a shorter unit would move makes onto the take path (a take longer than a unit allocates
  on the caller's thread), and it keeps no memory to avoid the call.
- On a full discrete card a Private block the consumer takes goes to another memory type, host
  memory where the card has no other device-local type, and the kernels that read a Managed
  buffer's device copy placed there read it across the host link:
  `Gauges::managed_spilled_bytes` counts those buffers' bytes while they live.
- A new buffer's zeros are the runtime's on every device: a Shared buffer's are written through
  the mapping inside `Device::new_buffer`, which pulls the first touch of every page into the
  call, where Metal pays it at the program's first write. On the Iris Xe a new Shared buffer of
  144 MiB takes 24 to 34 ms to make, almost all of it that zeroing: the driver's
  `vkAllocateMemory` of it takes 0.01 ms. Into the fresh pages of a new allocation the write runs
  at 7.9 to 8.5 GB/s under RADV (Mesa 26.2.6) on the Radeon 780M (Phoenix and Hawk Point), the
  RX 550 and the RX 7900 XTX, against 25.5 to 27.4 GB/s under NVIDIA's driver (620.70.06) on the
  RTX 3090 and RTX 2080, and 24 to 27 GB/s for the same `memset` into fresh pages of the host's
  own memory on those hosts; into pages written before, 73 to 97 GB/s on every one of them.
  So a new long Shared buffer costs at its make what a Mac pays at its first write. In a guest
  on the runtime, making a new 547 MiB Shared buffer and writing it once took 191.2, 152.6 and
  146.3 ms for the first three in a process on an RTX 3090 (job m3-max-65737), and 252.3, 110.0
  and 106.7 ms on a Radeon 780M (job m3-max-65736). On a Mac with a Radeon Pro 5300M (job
  m3-max-65625) the make took 0.179 ms, its first write pass 221.27 ms and a second pass
  18.99 ms. A make plus its first write is at or under the Mac's on both cards, except the
  780M's first in a process.
  `Metrics::long_take_zero_writes` counts the zeroing of takes past the 16 MiB unit. The runtime
  does not use
  `VK_EXT_zero_initialize_device_memory`: under RADV (Mesa 26.2.6) an RX 7900 XTX's memory
  allocated with its flag read other processes' bytes.

## Set 1 and index grants

Shaders reach textures and samplers by index (`docs/RESOURCE-ID.md`). Every pipeline takes set 1,
one descriptor set the device makes at creation, of seven arrays whose element `i` is the object
whose `gpuResourceID` holds index `i`: 65,536 sampled textures (binding 0), 65,536 storage
textures and then 65,536 level elements (binding 1), 8,000 samplers of each use, plain, depth comparison and integer texture
(bindings 2, 3 and 4), and 65,536 uniform and 65,536 storage texel buffers, the texture buffers
(bindings 5 and 6). A texture's index is the same element of both texture arrays: one with a
storage view takes both, any other the sampled one alone, and its storage element stays null. A
texture buffer's index is the same element of both texel-buffer arrays: the uniform one where
shaders read it, the storage one where they write it, the other null; its texture elements stay
null, as a texture's texel-buffer elements do. Every binding is partially bound, updated after bind and
updated while unused elements are pending, so an object's element is written once, as it is
made, with no lock, and never rewritten for layouts.

- Where the device reads attachments (Floor, Attachment reads), binding 0 holds 131,072
  elements: element `i + 65,536` is texture `i`'s second element, which a draw pushes in place of
  the texture's own index while it reads an attachment of its pass through the texture. Through
  the attachment feedback loop it is the texture's view in the feedback layout, written with
  element `i` for a texture whose image a pass may move there (a colour render target not in place
  over a buffer); on a device without the loop it is the texture's view of its image's
  shadow (Render encoders), written at the first draw that reads through it where the texture
  still owns index `i`. It is a null descriptor otherwise, and no grant reaches it.
- Binding 1's elements 65,536 to 131,071 are level elements, in 4,096 runs of 16. A function that
  writes or reads a storage texture at a level other than the constant 0 (MSL
  `write(color, coord, lod)` or `read(coord, lod)` at a level it computes or reads) indexes a
  run: the first dispatch or draw that binds a texture of more than one level to such a slot
  gives the texture a run and writes the run's elements, each a single-level storage view of one
  of the texture's levels, in order from its first (a cube's level as the 2D array of its faces, a
  texture view's from its own first level). The dispatch or draw pushes the texture's level count
  and the run's first element, and an access at level `l` reads or writes element
  `first + min(l, count - 1)` only where `l < count`: a write past the count is dropped and a
  read there answers zeros, and no level reaches another texture's elements. A texture of one
  level pushes its own index, and nil pushes count 0, so its writes are dropped. A later bind of the texture makes nothing: it takes the texture's lock and
  pushes the same two words. The run goes back once the texture is released and its work has
  retired: its views are destroyed and its elements are null again
  (`Gauges::level_elements`). A texture bound so when every run is held, or one of more than 16
  levels, past any texture Metal makes, is `Limit::LevelElements`, and the dispatch or draw
  does not run. No grant and no resource ID reaches the level elements.
  `Metrics::level_field_commands` counts the dispatches and draws that push a texture's level
  count, these and those of a function that asks `get_num_mip_levels` of a storage texture.
- A shader masks the index of a texture ID it reads from an argument buffer to its low 16 bits,
  so a bad ID there reads the element at `index & 0xFFFF`, never one outside the arrays. A slot's
  index is the runtime's own push, unmasked, and no argument buffer's ID reaches the feedback
  elements.
- Every element no object owns is defined. A texture or texel-buffer element is a null descriptor
  (`nullDescriptor`): it reads zeros, sizes 0 and drops writes, as Metal's nil texture does. A
  sampler element is its array's default sampler, nearest and clamped to the edge; the
  comparison array's compares with `NEVER`, so a comparison through it answers 0, as Metal's
  default compare function does. Vulkan has no null sampler. The device writes every element so
  at creation and holds two `VkSampler`s for the defaults (`Gauges::device_samplers`).
- `Device::grant_indices(class, first, count)` grants the consumer the range it names, so the IDs
  it chooses from it on the guest's side reproduce in replay. There are two classes
  (`IndexClass::range`): textures of every usage take 1 to 65,535, one index valid in both
  texture arrays, so a texture of any usage can be made at any index of a texture grant; samplers
  take 1 to 3,999. Index 0 is nil and never granted. A range that is empty or leaves its class's is
  `Limit::IndexRange`, and one holding an index granted already `Refusal::IndicesGranted`. A
  grant lasts as long as the device.
- An index has one owner: the object placed there last, in each class. Making an object at an
  index another object holds, a live one or one the program released that a command buffer not
  yet retired still holds, takes the index at once, as Metal's later texture at a resource ID
  takes the ID: the new object writes its elements (a texture with no storage view writes its
  storage element null where it took the index from another, so nothing stale answers there),
  and the earlier object's going, at its release or once its command buffers retire, touches
  neither the index nor its elements. Taking an index that a released object still holds for a
  command buffer not yet retired writes elements that work in flight may use, a write Vulkan does
  not define (an element of these bindings may change while a command buffer that uses the set is
  pending only where no shader invocation uses it). Not asking for one is the consumer's duty, not
  the program's: Metal hands a released ID out again at once (seen on a Radeon Pro 5300M), so a
  program can be given the ID of an object that work in flight still reads. A consumer keeps a
  released ID back from the program until every command buffer committed before the release has
  retired. The runtime does not check it, and does not refuse the placement: from that retirement
  until the vacate below, the index is still the released object's and a placement is defined;
  the displaced form is for that window. Work in flight may read a released object that still
  holds its index: a texture's index, elements, views and image, a texture view's views, and a
  sampler state's index and device samplers go once every command buffer committed before the release
  has retired, at once where none is in flight, so that work reads them as they were
  (`Gauges::releases_waiting` counts what waits). Work committed after the release holds none of
  them: the harness's own row `releases_under_work` holds what waits, the live textures and the
  texture blocks' bytes flat from its 100th iteration to its 200th, each releasing 16 textures and
  views while three command buffers that write 1 GiB each keep the GPU running, read with that
  work in flight. Metal leaves that read undefined: a program that
  released textures a frame in flight still sampled finished every run with no command buffer
  error on two Macs, with the answer it gave with the textures held on a Radeon Pro 5300M, and one
  a few channel levels off it, differing from run to run, on an Apple M3 Max. The runtime gives
  the first answer. As with memory, a command
  buffer that waits at an event holds every such release after its commit until it retires. A
  sampler state or constexpr sampler whose device samplers equal the displaced one's is safe under
  work in flight: equal create infos share the device's `VkSampler`s, the arrays keep the samplers last written at each index, and
  an element that holds its sampler already is not written, so the placement changes no
  descriptor that pending work reads (`Metrics::sampler_element_writes` counts the elements
  written). A different sampler state stays the consumer's duty.
  Making one at an index not granted is `Refusal::IndicesNotGranted`. Where nothing took it, the
  index is free again once its owner's last handle is gone, and with it the last command buffer
  that used it: at once where nothing was in flight then, and otherwise once every command buffer
  committed before that has retired and the block-maker thread has vacated it, on its first pass
  after the completion thread's round that retired them, behind the frees ahead of it there
  (`Gauges::releases_waiting` reads 0 only after that). Its elements read their default again
  from then, and `Gauges::descriptor_slots` no longer counts it.

## Textures

- `Device::new_texture(descriptor, index)` is `newTextureWithDescriptor:` at `index`: an
  optimal-tiling image of the descriptor in device-local memory, which reads zero until written
  (Memory blocks and buffers, above). The image is placed in a texture block: a Private block the
  device takes from its pool. An image of 16 MiB or less goes at the first place with room at its
  alignment in the 16 MiB texture blocks, in the order they were taken; a longer one in a texture
  block of its own length, which holds no other image, so its memory goes back to the pool with
  it. So a texture makes no allocation of its own. It has one only where the device requires one
  for it (`requiresDedicatedAllocation`; preferring one is no reason), where the blocks' memory
  type does not serve it, or where device-local memory has no texture block left. A place is free
  again once the image is freed; one empty 16 MiB texture block is kept, and another that empties
  goes back to the pool. Nothing moves a live image, so a 16 MiB texture block goes back only once
  its last image is freed (`Gauges::texture_block_sparse_bytes` counts the blocks holding less
  than a sixteenth of their length). `index` is given exactly where shaders reach the texture (`Refusal::TextureIndex`
  otherwise): shader-read or shader-write usage, or render-target usage of a colour pixel format
  (one with no depth or stencil), which a fragment function reads through its attachment, as
  Metal gives every texture a `gpuResourceID`; from a live `IndexClass::Texture` grant, whatever
  the usage. `TextureUsage` with no bits is Metal's unknown usage: shader read and write, render
  target and pixel-format view.
- The image's usage is the descriptor's plus transfer source and destination, and, for a colour
  render target where the device reads attachments, sampled, with the feedback usage where it has
  the attachment feedback loop (Render encoders). Its home layout is general where it has a storage
  element, shader-read-only where shaders only read it, and the attachment layout where only render
  passes use it. Its elements of set 1 are written before the call returns, with that layout, and
  never rewritten; a colour render target shaders neither read nor write leaves both null, so a
  slot that binds it reads Metal's nil texture, and writes only its second element (Set 1). The
  image moves from undefined to home in the upload of the next commit, ahead of every command
  buffer committed after the call, or ahead of its first copy; until then nothing holds it, so a
  texture released first is freed at once, with nothing submitted.
- A texture's image lists the formats Metal views the texture as that the device holds in a
  viewed form of the image format's compatibility class, and is then mutable
  (`VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT` with a `VkImageFormatListCreateInfo`, its own format
  first): its sRGB sibling whatever its usage, as Metal views a texture as its sibling without
  `PixelFormatView` usage, and with that usage every uncompressed colour format of the same bytes
  a texel. A texture of no such format is not mutable. A view of another format takes the image's
  usage that its format's features hold (`VkImageViewUsageCreateInfo`). A texture made for shader
  writes whose own format the device does not store, where a format it lists does (an 8-bit sRGB
  texture and its unorm sibling), takes the storage usage for those views
  (`VK_IMAGE_CREATE_EXTENDED_USAGE_BIT`), where the device's image format query makes that image
  with every usage, flag and sample count it makes the plain one with; in place over a buffer, its
  linear image too where the linear query makes it and the format's linear tiling holds the
  storage, else it stays in place plain. Its home is then the general layout, and its own
  format's views take the usage that format holds. So its view of the stored format has a storage
  element, and `Metrics::extended_view_dispatches` counts the dispatches that write such a view at
  a slot; with no such image it has none (`StorageElementCase::Unextended`). An 8-bit sRGB
  texture's own storage element is then its unorm sibling's view (Known gaps of textures).
- The sampled element's view is the format's, through its component mapping with the
  descriptor's swizzle composed over it; a depth or stencil format's view takes its depth, else
  its stencil. The storage element is written where the usage has shader write and the device
  stores the format through a storage image declared without a format, which no sRGB format is
  on any device; its view maps each channel to itself. A cube's or a cube array's storage view is the 2D array of its faces,
  layer `6 · cube + face`, which a kernel's storage cube writes and reads by face. A multisampled
  texture has no storage element on any device, as Metal's shaders only read one (the Metal
  Shading Language: "Multisampled textures only support the read attribute"). Its image takes no
  storage usage, so its sample counts are the device's for its other usages, whether or not the
  device has `shaderStorageImageMultisample`.
- A format is made where the translator's format table maps it for the usage on the device
  (`vk_format`) to the format itself or to one whose view gives Metal's values (a swizzled or
  rotated view, alpha held in red, an RGBX format, a depth or stencil aspect of a combined
  format), and the device's image format query takes the image. Elsewhere it is
  `Gap::TextureFormat`: the shaders of the identity domains read a texture through its view alone,
  so a format whose texels the shader converts (extended range, bfloat, YCbCr, decoded,
  copies-only, 24-bit depth held as 32-bit float) is not made. Past the query's extent, layers or
  sample counts it is `Limit::TextureSize` or `Limit::TextureSamples`.
- Refused as Metal refuses each descriptor (`Refusal::TextureZeroExtent`,
  `TextureExtentForType`, `TextureCubeNotSquare`, `TextureArrayLength`, `TextureLevelCount`,
  `TextureSampleCount`, `TextureSwizzleWithWrites`). Memoryless storage
  (`Gap::MemorylessTexture`) is not built. A texture buffer's descriptor is
  `Refusal::TextureBufferNotOverBuffer` at every make but `Buffer::new_texture`, which makes texture
  buffers (below). A cube array texture's image
  is cube-compatible with six layers a cube, and its view is a cube array view (`imageCubeArray`,
  in the floor). A 3D texture of `RenderTarget` usage and a colour format is made
  2D-array-compatible (`VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT`, in its image format query too),
  so that a pass renders into one depth plane of a level through a 2D view of it (Render
  encoders).
- `Texture::gpu_resource_id` is its index (0 without one) and the translator's `TextureWord` of
  its format and swizzle (`docs/RESOURCE-ID.md`), so the guest computes the ID itself. An ID of
  index 0 that a shader reads from bytes (an argument buffer's texture) reads the null descriptor,
  as Metal's nil texture; nothing refuses it there.
- `Texture::replace_region` and `Texture::get_bytes` take a Shared or Managed texture
  (`Refusal::TextureBytesPrivate` otherwise), a region inside a level and slice on the format's
  block grid (`Refusal::TextureRegion`), whole blocks ending inside the level's last block past
  its edge too (the one 4 x 4 block of a level of 4 x 2 texels, which Metal copies whole, where
  Apple's pages ask for a size clamped to the edge), and rows and images apart as Metal's calls
  take them, a compressed format's rows being rows of blocks (`Refusal::TextureBytesLayout` where they
  overlap or the bytes are too few). `replace_region` copies the bytes into the staging region
  before it returns, and the upload copies them into the image, leaving home for the transfer
  layout and coming back, ahead of every command buffer committed after the call. It takes the
  bytes as `HostBytes`: from a slice, or from a pointer and a length (`HostBytes::from_raw`, for
  memory another party may write meanwhile, such as a guest's RAM), and reads them in place, each
  byte of the region once, by raw copies, deciding nothing from their values and keeping no
  pointer past the call: the CPU copies the region's bytes once, at the call, as Metal's does.
  `get_bytes` records the copy out into the upload, submits it at once, waits for it and copies
  the bytes out; as in Metal, the program has waited for the work that writes the texture.
- `Device::new_managed_texture(descriptor, index, bytes, places)` makes a Managed texture whose
  bytes the consumer names, on every device: an image as `new_texture` makes, and the consumer's
  bytes in a Shared buffer, at a `BytesPlace` (offset, bytes per row, bytes per image) for each
  level and, within it, each slice, laid out as the consumer chooses. A place whose rows are not a
  whole number of texel blocks apart, whose images are not a whole number of rows apart, whose
  first byte is not a multiple of the block's bytes (4 for depth or stencil), or whose level
  leaves the buffer is refused (`Refusal::ManagedTextureBytes`), as is a buffer the host does not
  reach. `Texture::upload_region(region, level, slice)` copies a region of the named bytes into
  the image by the device, with no CPU copy, ahead of every command buffer committed after the
  call on every queue; the copy reads the bytes when the GPU reaches it. On such a texture `replace_region` writes the named bytes and uploads
  them, and `get_bytes` reads the named bytes, with no submission, as Metal reads a Managed
  texture's CPU copy.
- A heap is a range of a Private block the consumer took at the heap's creation (`HeapRange`:
  block, offset, length). `Device::heap_texture_size_and_align(descriptor)` answers Metal's
  `heapTextureSizeAndAlignWithDescriptor:` with the device's own image requirements, not Metal's:
  the size rounded up to, and the alignment raised to, the device's `bufferImageGranularity`, so
  a buffer placed by the same answers never shares a granularity page with the texture. The
  answers differ by device, so the consumer places by this device's.
  `Device::new_heap_texture(descriptor, index, heap, offset)` binds the texture's image at that
  offset in the heap, where the consumer's own placement put it: no allocation and no texture
  block, the bytes those the memory holds, as Metal does not zero heap textures. Textures may
  alias one another and the heap's buffers. A heap texture whose bytes another resource of its
  heap wrote since its last write moves from undefined before its next write, a GPU command's or
  `Texture::replace_region`'s copy (`Metrics::heap_moves`), as Vulkan's memory aliasing asks. A
  placement past the heap's end, an offset in the block off the image's alignment, or a heap
  range past its block is refused
  (`Refusal::HeapTexturePlace`); the image is never bound outside its heap. The texture holds
  the block, so a released block goes back to the pool only once its textures' images are gone.
  `Device::new_heap_buffer(heap, offset, length)` is `newBufferWithLength:options:offset:` of a
  heap: `Device::new_buffer` at the heap's offset plus `offset` in its block, none of its bytes
  zeroed, and a range past the heap's end refused (`Refusal::HeapBufferPlace`). The heap is the
  one at its offset in its block, and every buffer and texture placed in it takes its id in the
  compute encoders' hazard sets (below). A buffer the consumer places with `Device::new_buffer`
  is no heap's, and keeps its own. Placing costs the runtime a look at what already lives in the
  heap: a new heap buffer looks at the heap's live textures, a new heap texture at all its live
  placements, and a release takes out only its own entry.
- `Buffer::new_texture(descriptor, index, offset, bytes_per_row)` is
  `newTextureWithDescriptor:offset:bytesPerRow:`: a texture whose texels are the buffer's bytes
  from `offset`, rows `bytes_per_row` apart, of the buffer's storage mode. Any pitch of whole texels at least a row long is taken; rows closer than a row or not a
  whole number of texels apart, an offset that is not a whole number of texels, or texels past
  the buffer are refused (`Refusal::BufferTextureLayout`). Its image is the device's own linear
  image, bound at the texture's place in the bytes shaders reach (a Managed buffer's device copy,
  where it has one), made preinitialized so its move home keeps the bytes (in a dma-buf's import,
  made undefined for the dma-buf and acquired preinitialized by its move home: Imported blocks),
  and kept in the general
  layout through every use, copies included, so that it and the buffer stay host-accessible
  aliases, which Vulkan's memory aliasing says interpret their bytes alike: a write through the
  image touches only its texels' bytes, and nothing is copied that the buffer does not copy.
  The bytes of a render-target texture's rows that are not texels may be rewritten with their
  earlier values while a pass renders it: an attachment's store, clear or resolve may read and
  write back any byte of the subresource bound for it. It is made so where the device's linear image of the texture's extent,
  format and usage lays its rows at `bytes_per_row` from the image's start
  (`vkGetImageSubresourceLayout`), the place meets the image's memory alignment and memory type,
  the format's linear tiling has every feature of its optimal tiling that the usage uses
  (copies, which a blit encoder's copies and fills are, sampling and linear filtering, a
  shader's reads, writes and atomics without a format, rendering and blending), and, for a
  texture with render-target usage, the subresource the device advertises (its offset and size)
  ends at or before both the texture's own rows' end (`offset` plus `height` rows of
  `bytes_per_row`) and the buffer's end. A texture of any other usage writes its texels alone,
  and its subresource may end past them inside the buffer's block. `replace_region` writes the buffer's bytes, and copies the
  rows' span into a Managed buffer's device copy as `did_modify_range` does; `get_bytes` reads
  the buffer's bytes, with no submission; a dispatch's use of the texture is a use of the buffer in the
  hazard sets, and a write to it reaches the host as a write to a Shared buffer does. The texture
  holds the buffer.
- A texture buffer (`MTLTextureTypeTextureBuffer`) is made over a buffer alone:
  `Buffer::new_texture` of its descriptor is one `VkBufferView` of the bytes shaders reach from
  `offset`, its width's texels long, with no image and no copy. Metal's
  `newTextureWithDescriptor:` of a texture buffer's descriptor allocates the bytes itself; here it
  is `Refusal::TextureBufferNotOverBuffer`, and the consumer places a buffer of the texels' bytes,
  as it places every buffer, and makes the texture buffer over it. `offset` is a multiple of
  `Device::minimum_texture_buffer_alignment(pixel_format)`, `minimumTextureBufferAlignmentForPixelFormat:`
  (`Refusal::TextureBufferAlignment` otherwise): the device's texel-buffer offset alignment for
  the format's texel, the larger of the uniform and the storage texel buffer's (Vulkan 1.3's
  `…TexelBufferOffsetAlignmentBytes`, or the texel's bytes where the device's
  `…TexelBufferOffsetSingleTexelAlignment` says one texel suffices), at most 256, so it divides
  the multiple of 256 every buffer starts at. Metal documents the alignment for the row's bytes
  too; a texture buffer's one row reads no other, so any `bytes_per_row` that covers the row is
  taken (`Refusal::BufferTextureLayout` below it, or for texels past the buffer). The format is one
  the device holds directly, of one colour texel an element, whose buffer features hold both the
  uniform and the storage texel buffer (`Gap::TextureFormat` otherwise): the view is of a buffer
  of both usages, so its format takes both. A width past `maxTexelBufferElements` is
  `Limit::TextureSize`. Its view is the element of the uniform texel-buffer array at its index
  where shaders read it, and of the storage one where they write it (the known gaps say where a
  write or a read through it without a format is refused). Its texels are the
  buffer's bytes: the host, a blit or a shader writing the buffer writes them, `replace_region`
  and `get_bytes` write and read them there, and its uses are uses of the buffer in the hazard
  sets. Its storage mode is the buffer's. The view goes once the texture buffer is gone and its
  elements are null.
- Where the device's linear image does not take the texture (another pitch, an offset off the
  image's alignment, a format or usage the linear tiling lacks, a spilled block), the texture is
  an optimal image of its own, and the device copies the bytes shaders reach into it and back,
  one `vkCmdCopyBufferToImage` or `vkCmdCopyImageToBuffer` region a texture, the buffer's rows
  `bytes_per_row` apart, with no CPU copy. Each copy counts in `Metrics::over_copies` and its
  bytes in `Metrics::over_copy_bytes` and `Metrics::bytes_copied`; one over a buffer in an
  imported block counts in `Metrics::over_copies_imported` and `Metrics::over_copy_bytes_imported`
  too. Each copy also counts in the pair of the cause the texture's making found, the first that
  holds: the buffer's memory (`over_copies_host_memory`, `_dma_buf`, `_dma_bufs`, `_dedicated`
  for an exported block's, `_surface` for an exported surface's buffer), a spilled block
  (`_spilled`), no linear image of the format, usage and size (`_linear`), a format feature the
  linear tiling lacks (`_linear_features`), then the linear image as made: its row pitch or
  start (`_pitch`), a render target's reach (`_render_reach`), the place off its memory
  alignment (`_alignment`) and its other memory requirements (`_image_memory`), each with its
  `over_copy_bytes_` field. `Counters::over_copy_kinds(from)` names each kind of copied texture
  once, at its first copy (an `OverCopyKind`: its `OverCause`, pixel format, width, height and
  usage), the first 256 kinds in that order; once it holds 256, a copy of a kind not among them
  counts in `Metrics::over_copies_unkept_kind`. When is decided by the buffer's storage:
  - Over a Shared buffer (or a Managed one without a device copy), once a submission: the upload
    that leads it copies the bytes into every such texture a segment of it uses before any command
    of the segment wrote the buffer, between one pair of barriers. Inside a segment a command that
    uses the texture after a command wrote the buffer copies them in again first; a segment that
    leans on the leading copy for a buffer an earlier segment of the held group wrote leads a new
    group. A command that wrote the texture leaves it to go back to the buffer before the next
    command of the segment that reaches the buffer, and at the segment's end, whose end barrier
    then reaches the host. Metal reads Shared memory when the command buffer executes; every member
    of a group was committed before the group was submitted and none has been seen complete, so a
    CPU write between two members' commits races an uncompleted command buffer, and Metal does not
    define which bytes it sees; the bytes at the group's execution are one of the allowed outcomes.
    A guest that waits for a completion before writing has, by that fact, a command buffer not in
    a held group.
  - Over a buffer in an exported block (`Device::new_exported_block`), the Shared rules inside a
    segment, and the leading copy only once after each event that changes the bytes: the
    texture's making, the block's acquire (`Block::external_writes_done`, which
    `Surface::external_writes_done` makes in the copy form) or a zero fill of its range
    (`Device::zero_block_range`), each ahead of every command buffer committed after it, and a
    submitted command's write of the texture's bytes or another texture's copy back over them. A
    write through an address outside the command's bound slots and declared resources changes
    the bytes of every such texture whose block's address a public call handed out before that
    submission, and of no other: `Buffer::gpu_address` of a buffer whose bytes shaders reach in
    the block, and a texture buffer's
    storage element at an index over the block's bytes; the block's own `BlockInfo` states no
    address. An exported surface's block in the copy form is the surface's own, so only
    `Surface::buffer`'s `gpu_address` hands it out; the planes' resource IDs reach their own
    images. Each such block counts once in `Metrics::exported_blocks_addressed`. A command
    recorded while no texture over a buffer in an image of its own lived is untracked, and changes
    every such texture's bytes. The upload that leads the next submission that uses the texture
    makes the copy; a submission that only reads the texture copies nothing. The block is Private,
    so no host write reaches it, and the other API's writes are the consumer's to announce by the
    acquire.
  - Over a Managed buffer's device copy, only at Metal's own Managed events, each texture keeping
    one flag of each kind: in, after a `did_modify_range` over its range, by the commit that
    captures the range, after its copies into the device copy; and at its making, or where device
    work wrote the buffer (a binding whose compiled interface writes it, a blit or fill into it),
    by the upload that leads the next submission using it, in the order submissions go to the
    device's one `VkQueue`: a command buffer of another queue that passes a group held takes the
    copy ahead of that group. A capture over its range settles that copy only for the command
    buffers committed at or after the capture; one committed before it takes its own. Both copy
    from the device copy. Back, where device work
    wrote the texture, before the next reader of the device copy (a binding that reaches the
    buffer, a blit from or into it, a copy into another texture over a range that meets its, a
    `synchronize_resource`, a capture of a range over it, ahead of its copies): a reader in the
    write's own command buffer, recorded after the write, makes it in its own commands; any other
    reader, whichever was encoded first, takes it in the upload that leads its submission, which
    runs before the submission's commands, only once the write's segment was submitted, so a
    reader committed before the write on its queue runs ahead of it and leaves it owed, and a
    reader whose group held holds the write ahead of it leads its own group, the write submitted
    first. The write's submission also owes each other texture over a range that meets its a copy
    in. A texture released while its copy back is owed leaves its image with the buffer until a reader takes it
    or the buffer is released, which drops it uncopied; `Gauges::textures` counts it meanwhile.
    A host write the program does not tell with `did_modify_range` reaches such a texture only
    at a later event that copies it in; Metal leaves what the texture shows then undefined. An M3
    Max on macOS 27.0.1 shows the untold write
    (`crates/metal2vulkan-conformance/probes/managed_texture_stale_write.m`), and the macOS 27
    SDK says Managed storage has no effect on Apple silicon; the documented contract, not that
    answer, is the rule here.
- `Device::minimum_linear_texture_alignment(pixel_format)` answers
  `minimumLinearTextureAlignmentForPixelFormat:` with the device's linear row rounding: the smallest
  power of two from the format's texel bytes to 4,096 at which the device lays the rows of a 3-row
  linear image of each width of 1, 3, 17, 64, 65, 100, 129, 1,000 and 1,920 texels at the row
  rounded up to it; the largest over shader read, read and render, and read and write, where the
  device makes such images; 16 where no candidate holds, and a texture of that format over a buffer
  is then never in place. Each image is made, read and destroyed, never bound, once per format for
  the device's life. The answer leaves out the image's memory alignment
  (`VkMemoryRequirements::alignment`): a texture whose pitch is its row rounded up to the answer is
  in place, at the widths tried, only where its place in the buffer's memory is a multiple of that
  alignment too, and copied elsewhere. NVIDIA gives the same number for both, as RADV (Mesa 26.2.6)
  on the RX 550 does for texels of up to 4 bytes (256), where it answers 512 and 1,024 for 8- and
  16-byte texels over a 256-byte alignment; the UHD 630 rounds rows to 64 bytes and aligns its
  linear images' memory to 4,096; ANV (Mesa 26.2.6) on the Iris Xe of RPL-P lays each row at the row
  itself with its memory aligned to the texel, so it answers the texel's bytes (4 for `RGBA8Unorm`,
  1 for `R8Unorm`), below Metal's 16. The answer promises nothing for widths not tried.
- The staging region is 16 MiB of the Shared type in an allocation of its own, made by the
  block-maker thread as its first work and kept for the device's life, so no copy allocates
  memory. Its bytes are never zeroed:
  each copy's span is written, by the host or by the copy, before anything reads it. It is a ring: each copy takes a span at a multiple of 256 bytes,
  free again once the submission that carries the copy completes and no `get_bytes` reads it. The
  uploads not yet submitted go, without a wait, once their copies' spans take 8 MiB, so the GPU
  copies while the host writes the next. Where the ring has no room, the call waits for its oldest
  span's submission, submitting the uploads first where the span is in one
  (`Metrics::upload_waits`); where
  another call still writes that span's bytes, or holds them for a `get_bytes` to read, it sleeps
  until that call releases the span (`Metrics::staging_sleeps`). A
  copy larger than 16 MiB has a staging buffer of its own, which counts against the 64 MiB of the
  uploads not yet submitted. `Gauges::staging_bytes` counts the region once made and each such
  buffer until the upload that copies through it is gone.
- `Texture::new_texture_view(pixel_format, texture_type, levels, slices, index)` is
  `newTextureViewWithPixelFormat:textureType:levels:slices:`, and with the texture's own type and
  every level and slice `newTextureViewWithPixelFormat:`. A view is a texture over the parent's
  image with image views and elements of set 1 of its own, at its own index: no image, no memory
  and no copy (no submission, block or copied byte). It keeps its parent, and its descriptor is
  what Metal's view reports: the extent of its first level, its levels and its slices. Copies,
  `replace_region` and `get_bytes` through it reach the parent's levels and slices under it, and a
  view of a Managed texture names the parent's bytes of them; its hazards are its parent's. Built:
  the parent's own pixel format, over every slice in the parent's type, a slice subrange of a 1D
  or 2D array or of a cube array's cubes, a 2D texture as a 2D array of its one slice, one slice
  of a 2D array as a 2D texture, a cube's or a cube array's faces as a 2D texture (one face) or a
  2D array (any faces), six of a cube array's faces as a cube, and a cube as a cube array of one
  cube, each over any level subrange; a view of a view is a view of the first parent. Refused, as
  Metal's validation
  stops them: an empty range of levels or slices, or one past the parent's
  (`Refusal::TextureViewRange`), a slice count the view's type does not take
  (`Refusal::TextureViewSlices`), and another pixel format of a texture made without
  `PixelFormatView` usage (`Refusal::TextureViewFormat`). A view of another format of a texture
  without `PixelFormatView` usage that the image's own Vulkan format holds, its component mapping
  alone another (an `A8Unorm` view of an `R8Unorm` texture, both `R8_UNORM`), which Metal's
  validation stops, is made, as plain Metal makes it; one that needs another Vulkan format stays
  refused, as making every image mutable for it would cost every texture.
- `Device::new_texture_for_views(descriptor)` has no Metal counterpart of its own: a texture as
  `new_texture` makes it, of any usage, with no index. Shaders reach it only through its views,
  each at an index of its own, which a view of a texture shaders reach takes as above. It is the
  one image under textures the consumer makes as its views: the textures Metal makes over one
  IOSurface plane (`newTextureWithDescriptor:iosurface:plane:`), each with a `gpuResourceID` of
  its own, are its views at their own indices, two such textures two views. Copies,
  `replace_region` and `get_bytes` reach it as they reach any texture. What differs: Metal gives
  every texture an index, and this one has none, so its `gpu_resource_id` answers index 0, which
  a shader reads as Metal's nil texture, and where its usage reaches shaders a bind of it to a
  function's texture slot (`set_texture`, `set_vertex_texture`, `set_fragment_texture`) is
  `Refusal::TextureWithoutIndex`: bind a view. `new_texture` still refuses a texture shaders
  reach made without an index (`Refusal::TextureIndex`).
  `Device::new_heap_texture_for_views(descriptor, heap, offset)` is the same texture placed in a
  heap, as `newTextureWithDescriptor:offset:` places one: its image is bound at `offset` in the
  heap by `new_heap_texture`'s rule, with no allocation and no texture block, the bytes those the
  heap's memory holds, and its views take the heap's id in the hazard sets. It differs from
  Metal's heap texture as `new_texture_for_views` differs from Metal's texture: no index of its
  own.
- A texture or view whose own pixel format is block-compressed (BC1 to BC7, and the ASTC, ETC2
  and EAC formats where the device makes them) holds its blocks in an image of that format, and a
  read, sample or gather of its texels is the device's decode of them (an sRGB form's code through
  the device's sRGB decode), where Metal interpolates a block's endpoints with 6-bit weights: the
  static note `PrecisionNote::CompressedTexelDecode` (`Texture::precision_note`), decided by the
  pixel format alone on every device. Its blocks' bytes, copies and sizes are Metal's. The
  largest differences measured are in docs/API.md's catalogue
  (`metal2vulkan-degrade-compressed-texel-decode`).
- `Device::pixel_format_info(raw)` is `MTLPixelFormatGetInfoForDevice` (Apple SPI): Metal's
  description of a raw pixel format value, its name, Apple's flags word, cast class and four
  words (a block's bytes and extent for a format of blocks), as an Apple M3 Max answers it under
  macOS 27.0.1, `PixelFormatInfo::INVALID` for a value it describes no format for.
  `Device::texture_level_info(query)` is `MTLGetTextureLevelInfoForDeviceWithOptions`: Metal's
  layout of one level of a texture, its 13 words, computed from that description. Both answer the
  same on every device and ask Vulkan nothing: a format described there may be one this device
  makes no texture of (`Gap::TextureFormat`), and the flags are Apple's GPU's and promise nothing
  about this device (a write to a texture with no storage element here is `Gap::StorageElement`).
- A texture's handle holds the device; what a command buffer keeps of it does not (Teardown).
  Once its last handle and the last command buffer that used it are gone, its elements read null
  and its index is free at once; the reaper frees its image once the last upload that moves or
  copies it has retired, and a view's image views once the view is gone. `Gauges::textures`
  counts images and views until then, `Gauges::descriptor_slots` the indices held.

### Known gaps of textures

- A texture buffer with a swizzle other than each channel's own is `Gap::TextureBufferSwizzle`:
  Metal applies it to the reads, and neither a buffer view nor a `gpuResourceID` carries one.
- A view of a texture buffer (`newTextureViewWithPixelFormat:` and its kin) is `Gap::TextureView`
  of `TextureViewKind::TextureBuffer`, and a blit copy of one is `Gap::TextureCopy` between
  textures and `Gap::TextureBytes` with a buffer: no copy reads a buffer view, and none is built
  as the buffer copy it would be.
- A texture buffer's format needs both the uniform and the storage texel buffer in its buffer
  features: its view is of a buffer of both usages, and without `VK_KHR_maintenance5`, which the
  floor does not require, the view takes its buffer's usage. A format that lacks one is
  `Gap::TextureFormat` (`BGRA8Unorm` on KosmicKrisp, whose storage texel buffers lack it).
- `replace_region` and `get_bytes` of a texture without named bytes go through the staging region:
  a copy Metal does not make, counted in `Metrics::bytes_copied`.
- A blit copy between textures of a compressed region of whole blocks that ends inside one
  level's last block past its edge and not inside the other's (the one 4 x 4 block of a level of
  4 x 2 texels into the second block of a level of 8 x 4, or the reverse) is `Gap::TextureCopy`.
  Metal takes it, with its validation layer and without, and copies the block whole (probe
  `compressed_copies_of_small_levels.m` in the conformance crate, an Apple M3 Max, job
  m3-max-73696); one `vkCmdCopyImage` region has one extent, which stops at an image's edge only
  where that image's region does ("Copy regions for each image must be aligned to a multiple of
  the texel block extent in each dimension, except at the edges of the image").
- A texture made for shader writes whose format the device does not store through a storage
  image without a format (`RGB9E5Float` or `BGR10A2Unorm` where the device lacks their storage,
  an 8-bit sRGB format whose image is not extended for its unorm twin), or holds through a view
  of another (an `A8Unorm` held in red, the extended-range formats), has no storage element; its
  view of a format the device stores has one where its image is extended for it (above). A draw or dispatch whose function writes such
  a texture bound to a slot, or reads it there through its storage image (`access::read_write`),
  is `Gap::StorageElement`. So is, with no slot, a dispatch whose kernel, or a draw whose vertex
  or fragment function, may reach a texture through its storage image beyond its slots (an
  argument buffer's `access::write` or `access::read_write` texture, a resource ID) after the
  compute or render encoder's `use_resource` declared a write of such a texture in its encoder.
  A write through a resource ID with no such declaration is outside documented Metal
  (`useHeap:` declares its heap's resources read only) and is dropped, and a storage read there
  answers zeros.
- An 8-bit sRGB texture (`RGBA8Unorm_sRGB`, `BGRA8Unorm_sRGB` and the other 8-bit sRGB formats)
  made for shader writes, whose image is extended for its unorm twin (above), has the twin's
  storage view for its storage element, on every device, as no device stores an sRGB format
  through its own view here. A dispatch whose kernel writes it at a slot the kernel does not read
  runs the kernel's sRGB variant, lowered at the first such dispatch from the state's function
  with those slots and kept in the device's map of contents (1,024 entries, the least recently
  used forgotten past them): the variant encodes each write there by the Metal Shading Language's
  transfer (8.7.7, alpha linear) through the device's conversion, under the static note
  `PrecisionNote::DevicePackConversion`, which a state whose kernel is a function writing a
  texture slot it does not read announces from its creation. Its other uses:
  - a draw whose vertex or fragment function writes it at a slot, and a draw or dispatch that
    reads it at a slot through its storage image (`access::read_write`; Metal's read-write tiers
    list no sRGB format), are `Gap::StorageElement`, and so is, with no slot, a draw or dispatch
    after `use_resource` declared a write of it (above);
  - a write that reaches it through an argument buffer or a resource ID with no such declaration
    (after `use_heap` alone) is not encoded: the format of the texture such a write reaches is
    unknown when its function is compiled, so the write stores each channel's linear code
    (`round(255 · v)`) in the twin's bytes, where Metal stores its sRGB encode, and a storage read
    there answers the stored codes as linear values. It is not refused: nothing the runtime is
    told names the textures such an argument reaches, and a refusal by the declarations of reads
    or by the bindings would refuse programs Metal runs.
- A texture whose storage element the device cannot read without a format (its format lacks
  `VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT`, which a device without
  `shaderStorageImageReadWithoutFormat` answers per format: on ANV before Gfx12.5, `BGRA8Unorm`,
  `BGR10A2Unorm` and `RG11B10Float` among others) is written as any other. A draw or dispatch
  whose function reads it at a slot through its storage image (`access::read_write`) is
  `Gap::StorageElement`. Such a format is outside Metal's read-write tiers, whose reads Metal's
  documentation defines. A read of one through an argument buffer or a resource ID is not refused,
  as no declaration tells it from a sample: it reads the texture's storage element without a
  format, outside Vulkan's valid usage (`VUID-vkCmdDispatch-OpTypeImage-07028`), and Vulkan does
  not define what that read returns.
- A texture buffer's storage texel buffer is declared without a format too, but at a slot a
  texture atomic updates (`R32ui` or `R32i`), and its answers are its format's buffer features,
  of which Vulkan requires neither `VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT` nor the
  read bit, whatever the device's features and the format. A draw or dispatch whose function
  writes it at a slot declared without a format where they lack the write bit, or reads it there
  through `access::read_write` where they lack the read bit, is `Gap::StorageElement`
  (`VUID-vkCmdDispatch-OpTypeImage-07029`, `-07030`), whatever its format; so is, with no slot,
  one that may reach it beyond its slots after `use_resource` declared a write of it where they
  lack the write bit. A write through an argument buffer or a resource ID with no such
  declaration is outside documented Metal and is not refused: where they lack the write bit it
  writes the texture buffer's storage element without a format, outside Vulkan's valid usage
  (`VUID-vkCmdDispatch-OpTypeImage-07029`), as such a read is (`-07030`), and Vulkan does not
  define what it does. The UHD Graphics 630's Windows
  driver (101.2145) answers neither bit for any format: there a texture buffer a function writes
  or reads through `access::read_write` is refused, but at a slot a texture atomic updates, and
  one it reads through `access::read` runs.
- An upload into a 32-bit float depth (`Depth32Float`, or the depth of `Depth32Float_Stencil8`)
  by `replace_region`, `upload_region` or a blit's `copy_from_buffer` is the device's copy into
  the depth aspect, on every device, under the static note `PrecisionNote::RestrictedDepthUpload`
  (`Texture::FLOAT_DEPTH_UPLOAD_NOTE`). A value within `[0, 1]` is copied exactly, except a
  denormal or `-0.0`, which a device may store as `+0.0`. Metal keeps every `float` there bit for
  bit, and its depth comparisons compare it unclamped: measured on an
  Apple M3 Max under macOS 27.0.1, and an AMD Radeon Pro 5300M and an Intel UHD Graphics 630
  under macOS 26.7.1, for negative values, values above 1, denormals, infinities and NaNs. Vulkan
  takes no value outside `[0, 1]` into a depth aspect without `VK_EXT_depth_range_unrestricted`
  (`VUID-vkCmdCopyBufferToImage-pRegions-07931`), which the floor does not require and the
  runtime enables on no device, and the runtime reads no uploaded value: so on every device,
  whether or not it offers the extension, a value outside `[0, 1]` is the device's own, and so is
  every read, sample, comparison and copy of it. Measured: NVIDIA's driver (620.70.06, an RTX
  3090) keeps every value bit for bit, as Metal does; RADV and ANV (Mesa 26.2.6, a Radeon 780M and
  an Iris Xe) clamp it to `[0, 1]`, store a NaN of either sign, a denormal and `-0.0` as `+0.0`,
  and compare what they stored, so a shadow map uploaded outside `[0, 1]` compares otherwise
  there.
- Their bytes are not copied for a format of depth and stencil both or a multisample texture
  (`Gap::TextureBytes`).
- `Device::pixel_format_info` answers as an Apple M3 Max under macOS 27.0.1, which describes
  none of the values from 630 to 645 (`BGRA10Unorm`, its packed, sRGB, PQ and HLG forms,
  `R10UintX6`, `RG10UintX12` and the rest); an Apple M4 under that release describes them.
- A heap texture whose bytes another resource of its heap wrote moves out of
  `VK_IMAGE_LAYOUT_UNDEFINED` before its next use that writes it (`Metrics::heap_moves`), as
  Vulkan's memory aliasing asks: "each of those image subresources must have its layout
  transitioned from `VK_IMAGE_LAYOUT_UNDEFINED` to a valid layout before it is used". What remains:
  a read of such a texture before its next write takes no move; a write rt cannot name, through a
  resource ID or an address outside the encoder's slots and `use_resource` declarations, leaves no
  texture stale; a declared write (`use_resource` with a usage that writes) counts as a write of
  the whole resource; a texture placed over another's bytes moves home from undefined at its
  making, which on a device whose move writes memory may change a live alias's bytes; a move is
  of the whole image, so where an alias's write met only part of it, the move takes the rest too;
  and a draw's write of a texture that nothing earlier in its command buffer wrote, nor an alias
  of it, is checked where the pass's part of the command buffer starts, so its move may go ahead
  of a read of an alias there.
- A heap buffer's use that may write it (a bound slot its kernel or draw does not read only, or a
  declared write) makes every heap texture over its bytes stale, though the use wrote other bytes
  or none: that texture's next write moves its whole image from undefined, so on a device whose
  move resets an image, what the texture held where that write does not reach is lost.
- A texture's memory is device-local; where the device has none left, the call fails
  (`Error::Vulkan`).
- A texture over a buffer other than 2D of one level, sample and slice, or of a depth, stencil
  or compressed format, is not made (`Gap::TextureOverBuffer`).
- A texture over a buffer whose pitch, place, memory type or format features the device's
  linear image does not meet is copied between the buffer and an image of its own: the copies
  Metal does not make, counted (`Metrics::over_copies`). A program that rounds its pitch past the
  device's own (`Device::minimum_linear_texture_alignment`), or offsets its texture off the
  image's memory alignment, takes them: NVIDIA lays a linear image's rows at the row
  rounded to 128 bytes, AMD to 256, so a pitch rounded to 256 is copied on NVIDIA; the Iris Xe
  lays them at the row itself (the answer is the texel's bytes), so a pitch rounded past the row
  is copied there. In a recorded play of a compute workload on the RTX 5080 Laptop, 80 textures of
  2448x3264 RGBA8 at 9,984 bytes a row, against the device's linear row of 9,856, took 206 copies
  of 3,835,938,720 bytes in three frames, 95.2% of the play's pitch-copy bytes; their cost on those
  frames was not resolved: the means read 0.9% more GPU busy time and 2.2% more wall time, with
  the two pairs of opposite sign (job m3-max-68398). In a recorded play of a compute benchmark on
  the Radeon 780M, with the no-copy buffers the program may own imported as udmabufs over a memfd,
  the copies of textures over those imports cost each workload this share of its timed GPU time,
  against the same play over the runtime's own Shared memory (job m3-max-68694): the leading
  uploads' own GPU time exactly, and the copies inside a command buffer at most the time those
  command buffers lose against the other play. A background blur: 600 copies of 4,976,640,000
  bytes, at most 1.94% of 5.238 s. An image workload: 240 copies of 11,955,164,160 bytes, 2.99%
  exactly and at most 6.94% of 5.044 s. A second image workload: 60 copies of 3,150,950,400
  bytes, at most 2.68% of 5.294 s. A path tracer: 83 copies of 174,063,616 bytes, at most 2.23% of
  5.221 s. The benchmark's other workloads took none. A render
  target over a buffer is copied on NVIDIA, whose linear tiling does not take it in place. A
  render target whose advertised subresource ends past its own rows or its buffer is copied too:
  llvmpipe (lavapipe) advertises a linear image's height rounded up to 4 rows and stores whole
  blocks of 4 by 4 texels, so a render target over a buffer whose height is not a multiple of 4
  is copied there. So is one that ends at its buffer's last texel with a pitch wider than its
  row, on the devices we measured but the Iris Xe: their subresource ends at the last row's
  pitch. In the replay streams we counted, every render target over a buffer starts at its
  buffer's first byte and its rows end at the buffer's end.
- Such a texture over a Managed buffer's device copy takes its copy in only from the upload
  that leads a submission. A command that writes the buffer, or another texture over a range
  that meets its, reaches it from the next submission: a later command of the same command
  buffer, or of a command buffer submitted with it, reads the texture's older texels, where a
  discrete Mac's GPU reads the written ones. A command buffer that wrote such a texture and is
  never committed leaves the texture's entry with the buffer until the buffer is released.
- A `replace_region` on a Managed texture made after a commit and before the GPU reaches that
  command buffer's upload is seen by it; Metal on a two-copy GPU sees it only until the command
  buffer is scheduled.
- Texture views that change what the replay streams' views keep are not built
  (`Gap::TextureView`, naming the kind): another pixel format the texture's image does not list,
  another texture type than the casts built above, some of a 2D multisample array's slices, and a
  cube view of a 2D array whose image is not cube-compatible (`TextureViewKind::CubeOfArray`): a
  square 2D array of six slices or more is made cube-compatible where the device's image format
  query takes the flag for its format and usage, and its cube views of six slices and cube array
  views of a multiple of six are built; a non-square array's image is not. A view of a pixel
  format Metal does not
  view the texture as is `Refusal::TextureViewFormat`.
- A swizzle reaches a texture's sampled element alone, as Apple defines it for reads and
  samples. A swizzled texture, or a view made with a swizzle or of a swizzled texture, has a
  storage element and attachment views that map each channel to itself, as Vulkan requires of
  a storage image and an attachment: a write and a render pass store the texels unswizzled.
  A draw or dispatch whose function declares `access::read_write` at a slot it reads, bound to
  such a texture, is `Gap::SwizzledReadWrite`, found when the pipeline state or a binding
  changes: its read would answer the texel unswizzled, where Metal reads it through the swizzle. A texture's
  own descriptor takes a swizzle only without shader-write and render-target usage
  (`Refusal::TextureSwizzleWithWrites`), so this reaches views alone.
- A kernel, a vertex function or a fragment function samples a 2D texture slot through a bicubic
  constexpr sampler, at normalized or at pixel coordinates, by the arm its slot's pushed word
  selects, which the runtime writes from the bound texture's format and swizzle
  (`metal2vulkan::metal::format::bicubic_texels_word`), the same word for a dispatch and for a draw.
  A dispatch or a draw whose slot holds a texture of a format with no such word (a compressed, YUV,
  stencil, integer or extended-range format, or one of 8-bit sRGB codes such as `RGBA8Unorm_sRGB`,
  whose taps Metal filters decoded) is `Gap::BicubicTexels`, naming the slot and the format, found
  when the pipeline state or a binding changes: each such format's arm is built when a guest binds
  one. A bicubic sample of a 1D texture is refused at creation in every stage (the identity domains'
  form, below). Only a render state whose functions sample through a bicubic sampler lists a slot to
  check and fill; a draw of any other state does neither. Basis: this is NOT documented Metal. The
  MSL spec lists only nearest and linear filters, and 'bicubic' appears nowhere in the spec or the
  feature-set tables. It is built because a guest already uses the bicubic constexpr sampler in
  kernels, which is why the kernel half exists. Filling the field for a draw completes that one form
  and adds none.
- An opaque-black border sampled through a view whose component mapping is not the identity (a
  swizzle, or the mapping a pixel format takes: `A8Unorm` is held in `R8_UNORM` read through
  `(0, 0, 0, r)` on every device, as rt uses Vulkan 1.3 and takes no format of
  `VK_KHR_maintenance5`, so `A8_UNORM` is not used; where indirect command buffers enable
  `VK_KHR_maintenance5`, that changes nothing here) is the device's answer on a device that
  does not report `borderColorSwizzleFromImage`
  (`VkPhysicalDeviceBorderColorSwizzleFeaturesEXT`, `VK_EXT_border_color_swizzle`): Vulkan leaves
  that texel undefined, and the NVIDIA driver reads `(0, 0, 0, 1)` through such a view, where
  Metal reads the border through the swizzle. A sampler in the arrays meets every view, so no
  sampler's create info names a view's mapping, and no exact form exists on such a device. The
  sampler state carries the static note `PrecisionNote::SwizzledOpaqueBlackBorder`
  (`SamplerState::precision_note`), on every device (Sampler states); so does a pipeline whose
  function samples through a constexpr sampler of such a border, among its `precision_notes`.
- A drawable's texture (see "Presenting") has no view of another pixel format
  (`Gap::TextureView`), and a draw that reads it as an attachment of its own pass is `Gap::Draw`
  of `AttachmentRead`: the swapchain's image lists no other format and has no feedback usage.
  Heap textures are placed only in Private heaps (`Gap::HeapTexture` for a Shared block).
- A texture of Metal's 4:2:2 chroma-pair formats, `GBGR422` and `BGRG422`, is
  `Gap::TextureFormat` at every make and at `heap_texture_size_and_align`. Vulkan's 4:2:2 formats
  (`G8B8G8R8_422_UNORM`, `B8G8R8G8_422_UNORM`) are viewed only through a sampler Y′CbCr
  conversion, whose chroma reconstruction is the implementation's, so the translator's format
  table holds these two as a decoded `RGBA8Unorm` twin, which is not a form the runtime makes (a
  format held otherwise than through its view, above). The conformance cases `chroma_pairs` and
  `heap-sizes-422` record the refusal on every venue (`known-gaps-rt.tsv`).

## Sampler states

- `Device::new_sampler_state(descriptor, index)` is `newSamplerStateWithDescriptor:` at `index`
  of a live `IndexClass::Sampler` grant. `SamplerDescriptor` is the translator's, and so is each
  device sampler: `metal::sampler::vk_sampler` of the use, its create info chained by
  `ash_interop::sampler_create_info`. A state takes a device sampler for each array it can be
  read through: the plain array always, the comparison array where its compare function is not
  `Never` (else that element keeps the array's default), and the integer-texture array, nearest
  with the integer border colour. All three are written at `index` before the call returns.
- Device samplers are shared: equal create infos, across uses and states, are one `VkSampler`,
  and a border colour no address mode reads is left out of a create info, so a nearest state
  clamped to the edge takes one sampler for its plain and integer-texture uses. They count
  against the device's `maxSamplerAllocationCount`, the arrays' two defaults among them; past it
  a state is `Limit::DeviceSamplers`. None is made at bind time. `Gauges::device_samplers` counts
  them, `Gauges::samplers` the states.
- `SamplerState::gpu_resource_id` is its index and the translator's `SamplerWord` of its
  descriptor (`docs/RESOURCE-ID.md`).
- A state whose descriptor reads the border (an address mode of `ClampToZero` or
  `ClampToBorderColor`, both of which read the descriptor's colour) with an opaque-black border
  carries the static note `PrecisionNote::SwizzledOpaqueBlackBorder` (`SamplerState::precision_note`),
  decided by its descriptor alone on every device: sampled through a view whose component mapping
  is not the identity, its border is the device's answer where the device does not report
  `borderColorSwizzleFromImage` (Known gaps of textures; Floor, Border swizzle). Other states
  carry none.
- A state's handle holds the device; what a command buffer keeps of it does not (Teardown).
  Once its last handle and the last command buffer that used it are gone, its elements read the
  defaults again and its index is free; a device sampler no other state holds goes to the reaper.
- Anisotropy is the device's own filter (`samplerAnisotropy` is in the floor). A cubic filter is
  `Gap::CubicSampler`: the runtime makes no cubic sampler, so no live sampler's resource ID carries
  the sampler word's cubic bit (30), and a shader does not test an argument buffer's sampler for
  it. A descriptor the translator makes no device sampler of is `Error::Translation`.
- A state bound with LOD clamps (`set_sampler_state_with_lod_clamps`, and the render encoder's
  `set_vertex_sampler_state_with_lod_clamps` and `set_fragment_sampler_state_with_lod_clamps`:
  Metal's `lodMinClamp:lodMaxClamp:` forms, whose range forms are loops of them) samples as Metal
  answers on an Apple M3 Max (conformance cases `sampler-bind-lod-clamps-kernel` and `-fragment`):
  the pair replaces the state's clamps, a lower bound above the upper reads every sample at the
  upper's level, as the clamps `(max, max)`, and a plain bind afterwards reads the state's own
  clamps again. A NaN bound, which Metal documents no answer for, binds the state as it is.
- Where the pair reads alike with the state's own (each bound held to `[0, 15]` is the state's own
  so held, `metal::sampler::lod_clamps_read_alike`, as a clamp of 1000 over a state's `FLT_MAX`
  is), the state itself is bound. Otherwise the slot takes the state's variant of the pair: the
  state's descriptor with the pair as its clamps, its device samplers from the same shared
  samplers (`Limit::DeviceSamplers` bounds them), at one of 4,000 variant elements past the 4,000
  indices of each sampler array (Set 1 and index grants), whose `gpuResourceID` the slot pushes.
  No grant and no argument buffer's index reaches a variant element. A state keeps its 16 most
  recently bound variants for its later binds; a new one past them takes the place of the least
  recently bound one no command buffer holds, and where command buffers hold all 16, the new one
  is held by the binding command buffer alone and goes when it retires. Past 4,000 live variants
  device-wide a bind is `Limit::LodClampVariants`, and nothing is bound. The
  `lod_clamp_variants` gauge counts the live variants.

## Queues, command buffers and events

- `Device::new_command_queue(max)` is `newCommandQueueWithMaxCommandBufferCount:`
  (`newCommandQueue` is 64). Every queue submits to the device's one `VkQueue`.
  `CommandQueue::command_buffer` waits, as Metal does, while `max` of the queue's command buffers
  are made and not yet completed.
- A `CommandBuffer` records at once, with no replay log. Event waits and signals split it into
  segments, one `VkCommandBuffer` each where it has commands. Each ends with one global memory
  barrier: its writes are visible to every later command, and later commands run after its own,
  as Metal orders a queue's command buffers. The barrier also reaches the host where the segment
  wrote Shared memory (a Shared buffer bound or declared for writing, a write through an address
  outside the dispatch's resources, or a blit copy or fill into a Shared buffer) or signals an
  event, so the host reads those writes once the
  command buffer completes or it sees the event, after the flushes and invalidates of Shared
  memory's coherence (Memory blocks and buffers); a segment that wrote only Private memory leaves
  the host out. No segment starts with a barrier of its own; a blit copy records its own two
  (see "Blit encoders").
  `enqueue` takes the command buffer's place in its queue; `commit` hands it to the submit
  thread and returns. A committed command buffer holds no device handle.
- The submit thread passes what is ready to `vkQueueSubmit2` in the device's commit order, each
  queue's command buffers in groups: a submission holds at most sixteen command buffers of one
  queue, ahead of them the uploads of their commits (zero fills, texture copies and new textures'
  moves to their home layouts), and only its last submit info signals the progress semaphore.
  The calls made between two commits record into an upload of their own, which goes in the first
  submission whose last command buffer was committed after them: ahead of every command buffer
  committed after the call, and never in a submission whose command buffers were all committed
  before it, whenever the submit thread runs.
  A commit only takes its number; the first call after it begins the next upload, and the
  submission ends the last upload it carries with the one barrier that makes their writes visible
  and records the moves home of every texture they keep in one barrier. Consecutive command
  buffers that no event signal separates share one submit info, the uploads with the
  first: the GPU runs them in order either way, and a driver may pay for each info (the UHD 630's
  paid about 160 us apiece). `Metrics::submit_infos` counts them. The uploads not yet submitted
  record at most 4,096 calls and keep at most 64 MiB on their own account: the staging buffers of
  copies larger than the staging region, and the textures and blocks released while they held
  them. A call that finds them past either submits them and waits for them to complete before
  recording, so nothing grows without a commit. The release of a texture or a block they hold
  submits them at once, with no wait, so its memory goes back once they complete, with no later
  call, as Metal frees a released resource once no work uses it: at most one submission more for
  each upload that holds a released object. `Metrics::held_releases` and `held_release_bytes`
  count every release they held, and `held_release_starts` those that were the first since they
  came to hold the object, each a submission it started. While
  a queue has room on the device's `VkQueue`, each of its command buffers goes at once, alone. It
  has room while fewer than two of its submissions are not yet seen complete, and while fewer than
  three are whose expected GPU runs (below) behind the oldest, with the next's, sum to 16 ms or
  less; a run not yet known counts as past 16 ms. So a queue whose run is not yet known (none of
  its command buffers has retired with its GPU times) has room for two submissions, and its third
  command buffer waits for one seen complete. A queue keeps on the `VkQueue` no more than about 16
  ms of work behind what runs, and one behind it always, and a command buffer of another queue
  committed later waits behind no more of it: for work within the 16 ms bound, about 16 ms of
  it; behind longer command buffers, up to two whole command buffers of it (Known gaps, below).
  Without room, the ones committed meanwhile are held
  in one group of at most sixteen, each scheduled as it joins; one the group does not take is held
  too, not yet in a group, scheduled as it would be in one (its waits met, in its queue's order,
  its scheduled handlers run: `Metrics::held_for_room`), and so are its queue's later command
  buffers. The held work goes when a completion leaves room (the completion thread wakes the
  submit thread as it retires; no commit or
  wait is needed), when the host is to wait for one of them (`wait_until_completed`, or
  `CommandQueue::send_now`, the hint for a wait the runtime does not see), or at teardown. Waiting
  for a place in the queue sends nothing. A segment that signals an event ends its group, which goes
  at once as it is staged, and so does a presenting command buffer's last (see "Presenting"); so a
  command buffer that signals an event, in any segment, is staged only once its queue has room for
  it, as one held for room behind it is, and a wait for its signal waits with it, while a
  presenting one has its queue go at once, through to it. The
  driver runs a submission's command buffers as one kernel job, which a job timeout ends as a whole
  (Linux's amdgpu ends a job past its timeout and resets the ring), so a group is bounded by its
  expected GPU run: a queue's expected run follows its command buffers'
  own GPU times as each retires (a longer one raises it at once, a shorter one halves the way toward
  it), and a command buffer joins a group only while the group's expected runs, summed, stay within
  16 ms. One whose queue has no run yet, or that would take the sum past the bound, goes alone, the
  group held going first. The bound is tested again as the group goes, with the expected run as it
  then stands: a group past it goes member by member, one `vkQueueSubmit2` each, each signalling
  its own progress value, so each completes at its own end, not the last's. At most four command
  buffers join a group while none of the queue's retires, since
  they join on an expected run no completion has updated: in
  a replay stream on an RTX 3090, the most GPU time over four consecutive command buffers of a queue
  was 385.1 ms, within a quarter of amdgpu's default job timeout of 2,000 ms, and over sixteen
  1,536.6 ms. The thread holds a segment whose event waits are not met, and its queue's later
  command buffers behind it, while the segments before it run, so the host can see a signal the
  command buffer makes before its wait. Other queues go on. Each queue groups on its own count:
  a command buffer of a queue with nothing running is submitted at once, ahead of every command
  buffer committed after it on any queue, its part before an event wait included (of the groups
  that go at once, the one committed first goes first: on an M3 Max, a copy on another queue,
  committed after a command buffer that fills, waits for the event the copy then signals, and
  fills again, read the first fill and not the second, job m3-max-68673), while one its queue
  holds, in a group or for room, may be passed by another queue's command buffer committed after
  it, which only a fence or an event
  then orders after it. Metal's hazard tracking orders such a pair against work already submitted, and no
  more. Recorded on an M3 Max with macOS 27.0.1, with tracked buffers and no fence, a blit fill on
  one queue and a copy of what it filled on another, committed after it, five runs each: where the
  fill's queue had nothing ahead of it, the copy read the fill, starting 0.004 to 0.065 ms after
  the fill ended (job m3-max-52432, and with compute dispatches, job m3-max-52186); where the fill
  was committed behind one command buffer of long work on its queue (job m3-max-52432) or three
  (job m3-max-52415), the copy read none of it, and behind one it reached the GPU about 1 ms
  before any of that queue's work. On a 2019 MacBook Pro (job 64081), on its Radeon Pro 5300M, a
  second process's unrelated read completed 0.99 to 1.97 ms after its commit, beside about 160 ms
  of another process's work already submitted (8 busy command buffers, and a ninth committed by
  the call that submitted them), about 167 ms before that ninth; and a read of that process's
  shared surface completed after the surface's writer and 19.06 to 19.26 ms before the first of 4
  command buffers the writer's process had committed after its write (committed: their `commit`
  had returned, which does not show Metal had submitted them). On the same Mac's Intel UHD 630
  the two processes' command buffers ran first in, first out in all 18 rounds, controls
  included, so a read after its writer there is not told apart from that order (its busy command
  buffers ran 16.6 falling to 10.8 ms, short of the 20 ms aimed at). The room above follows the
  5300M's order, which these two GPUs alone were measured for: where Macs differ, the runtime
  takes the faster. `Scheduled` means the last segment is in
  a submission: passed to `vkQueueSubmit2`, or held, in a group or for room; a command buffer that presents a
  drawable is never held, and is scheduled once its submission has passed `vkQueueSubmit2` (see
  "Presenting"). One that waits for an event's value whose signal is submitted, of an `Event` or
  a `SharedEvent`, is scheduled so too, while its commands wait on the GPU for the signaller's:
  every queue submits on the device's one `VkQueue`, so the signal's submission is earlier in its
  submission order, and the barrier ending each segment orders every later command after the
  signaller's, so no semaphore wait is submitted for an event's value. Apple's
  MTLCommandBuffer.h describes `Scheduled` as "All dependencies have been resolved and the command
  buffer has been scheduled for execution" (lines 47-48), and `encodeWaitForEvent:value:` as a
  command that "pauses execution of this command buffer until the specified event reaches a given
  value" (lines 405-406): execution pauses after scheduling. The M3 Max does more for a shared
  event: recorded there, a command buffer behind a signal still running reads `Committed` with no
  handler where it waits for an `MTLSharedEvent`, and its queue's next command buffer with it,
  and `Scheduled` with its handler run where it waits for an `MTLEvent`. No documented Metal
  states that hold, and rt keeps none, so do not bring it back from a recording: it is an accepted
  difference (docs/API.md). A wait for a value that neither the host has set nor a submitted
  signal promises is held by the submit thread, its command buffer `Committed`, until one does:
  on the one `VkQueue` every queue submits to, a wait submitted ahead of its signal would hold the
  signal too. `Metrics::event_holds` counts those holds by length. A command buffer whose
  segments hold no commands, no event wait and no event signal takes no submission: the submit
  thread schedules it as it reaches it, and it completes with its queue's command buffer before
  it, in flight, or at once where that one has completed. Where that one is in a group held, it
  joins the group, which goes at once, so its completion waits only on submitted work. One that
  a Managed buffer's copy-in waits for as it reaches it ("Memory blocks and buffers" above) takes
  the copy-in as its submission instead, and completes after it.
- A command buffer's scheduled handlers run on the submit thread, once it is scheduled and before
  its submission goes to the driver (one that presents a drawable: after, before the completion
  thread learns of it; one that ends without a submission: once it ends, before it retires), with
  no lock of the runtime's held. In the handler its status is `Scheduled` and its kernel times are
  set, and it runs before the command buffer's completed handlers. `wait_until_scheduled` returns
  once the scheduled handlers have returned, as Metal's does: Apple's page for
  `waitUntilScheduled` names both events, the command buffer scheduled and its scheduled
  handlers invoked. Every queue's later submissions wait while it runs, those of command buffers
  held in a group or for room included, so a scheduled handler that blocks outside the runtime
  until such a command buffer completes never returns, where Metal, running it on a thread of its
  own, would go on: a known gap. While the completion thread runs a round it sees no completion,
  so held work waits for the round's end: a completed handler or listener that runs longer than
  the work in flight leaves the GPU idle for the rest, where Metal, running it on a thread of its
  own, would go on: a known gap. A scheduled handler may run while another command buffer's
  completed handler runs on the completion thread. A scheduled handler may read the command buffer's status
  and times, enqueue command buffers, make objects that take no upload, and do the consumer's own
  work. Each call that may wait for the submit thread is `Refusal::InScheduledHandler` in it, on
  every device: `commit`, the waits (`wait_until_scheduled`, `wait_until_completed`,
  `wait_until_signaled_value`), the uploads and readbacks (`replace_region`, `upload_region`,
  `get_bytes`, `zero_block_range`), and the objects that take an upload (`new_managed_buffer`, and
  `new_buffer` zeroing a buffer in a Private block). Metal runs a scheduled handler on a thread of
  its own, where those calls run. The submit thread wakes no other thread for a scheduled handler.
- The completion thread retires command buffers as the progress semaphore reaches them, a
  submission's in commit order: the status, the blocks they kept, their recorders and their place
  in the queue, then the completed handlers. Completed handlers and listeners run on it, in
  turn, so a slow completed handler delays the later command buffers' retirement, status and
  handlers, in that order, and the held work their completions would send (above). A wait in one for what that thread has not yet done is `Refusal::WaitOnOwnThread`, and one
  already met answers at once: `wait_until_completed` for a command buffer not yet completed (its
  own handler's command buffer has), `wait_until_scheduled` for one not yet scheduled,
  `wait_until_signaled_value` for a value not yet reached (a timed wait too: it is refused, never
  false at its timeout, since the thread that would observe the value is the one waiting), and
  `CommandQueue::command_buffer` on a queue whose maximum count are made and not yet completed,
  which a scheduled handler answers the same way. No wait of the runtime's hangs by construction.
  A command buffer of a group is seen complete with the group's last: up to the GPU time of the
  fifteen after it late. Only a queue without room holds a group; one whose
  submissions have completed sends its next command buffer alone. Without a wait or a hint, the lateness is bounded only while the queue has no room,
  and the completion that leaves room sends the group (the measured lateness is under
  "Known gaps"). A command buffer ended while its queue holds a group joins it, and retires in its
  place; the group goes at once.
- A command buffer's times, its GPU times included, are nanoseconds of one host clock,
  `host_time()`: `CLOCK_MONOTONIC` on Linux and other Unix systems, `CLOCK_MONOTONIC_RAW` on
  macOS and the performance counter on Windows, each the host time domain the device calibrates
  its own clock against (`HOST_DOMAIN`). On macOS that clock counts while the system sleeps, where
  Metal's (`mach_absolute_time`) does not. Metal gives seconds. `kernel_start_time` is the commit, or
  once it is scheduled, the `kernel_end_time` of its queue's command buffer before it where that
  is later, as Metal schedules a queue's command buffers in order; `kernel_end_time` is the time
  it was scheduled. `gpu_start_time` and `gpu_end_time` are two
  timestamp queries, one written where its first commands begin and one as the last command of
  its last segment, after the copies back of the images its commands wrote and its end barrier,
  so the two span all the GPU work done for it, read once it completes with no wait, and placed
  on the host clock by the device's last
  calibration: one `vkGetCalibratedTimestampsEXT` call reading the GPU's clock and the host's
  together, made as the device starts and again at a completion once 100 ms have passed since
  the last, never for each command buffer. Neither is ever later than the host time the
  completion thread observed the command buffer complete: where the calibration would place the
  end after it, the calibration's host time moves earlier by the excess, for that command buffer
  and every later one, so the two times move together, `gpu_end_time - gpu_start_time` stays the
  two timestamps' own span, and later command buffers keep their order. Mesa's drivers read the
  two clocks in turn, in the order asked, the GPU's first, so a calibration's host time trails
  its GPU tick by the time between the reads, and each completion bounds it from above. A command
  buffer with nothing for the GPU, whether it took no submission or a Managed copy-in's, writes no
  query: its `gpu_start_time` and `gpu_end_time` are one time, never 0: the latest `gpu_end_time`
  of its queue's command buffers before it, or its `kernel_end_time` where that is later, the shape Metal
  reports for a command buffer that does no work. A command buffer's GPU times so fall between
  `host_time()` before its commit, within the calibration's deviation and the clocks' drift over
  100 ms, and `host_time()` in its completed handlers. The two queries are a pair from the device's timestamp
  pools, which a command buffer takes as it records its first commands and gives back as they are
  read at its completion, or as it is dropped uncommitted. A pool holds 4,096 queries, the pairs
  of 2,048 command buffers; the device makes one only once every pair of its pools is taken, and
  keeps it until the device drops (`Gauges::timestamp_pools`), so a queue makes no Vulkan object
  and `new_command_queue` fails on nothing but a count of 0. Each pool is reset on the host as it
  is made (`hostQueryReset`), and a pair as its command buffer retires, those of the two its
  submissions wrote, so no command buffer the GPU runs resets a query. A command buffer with no
  commands, or whose last segment has none, records a `VkCommandBuffer` for its timestamps. One
  that ends with an error has GPU times of 0. The runtime's own command buffers, which no Metal
  command buffer is (the uploads that lead a submission or go alone, and the Managed buffers'
  copy-ins), take a pair each from the same pools, written as their first command and their
  last, and read as they retire: `Metrics::runtime_command_buffers` counts them and
  `Metrics::runtime_command_buffer_gpu_nanos` sums their spans, which no Metal command buffer's
  GPU times hold.
- `Event` and `SharedEvent` are timeline semaphores whose values are Metal's values. A host set
  (`SharedEvent::set_signaled_value`) raises a floor and never signals the semaphore. A wait
  submits no semaphore wait: it is held until the floor meets it or a signal of its value is
  submitted, earlier on the one `VkQueue`, whose segment's end barrier orders the waiter's commands
  after the signaller's. A GPU signal not above the highest value submitted or set is left out, so
  the semaphore only rises and never races a host set.
  `signaled_value` is the higher of the semaphore and the floor. `notify_listener` runs its
  listener on the completion thread when the value is reached, at a signal in the middle of a
  command buffer as well as at its end; `wait_until_signaled_value` wakes then too.
- `Fence` is `MTLFence`, and `update_fence` and `wait_for_fence` on compute and blit encoders are
  `updateFence:` and `waitForFence:`, on one queue or across queues. They record nothing: every
  queue's command buffers go to the device's one `VkQueue`, and each encoder's writes reach every
  command submitted after it through the barrier at its end or at its segment's end, so a wait
  sees every update submitted before it. A command buffer the submit thread holds in a group while
  its queue's submissions run is submitted before a command buffer of another queue that waits on
  a fence it updates, as Metal would have submitted it at its commit. Metal evaluates a wait at
  its kernel submission against the updates already submitted: an update in a command buffer held
  at an event wait, or behind one held so or enqueued and not committed, is not waited for, and a
  waiter on another queue runs at once, without its writes, as on an M3 Max (the conformance
  crate's `probes/cross_queue_fence_held_update.m`, jobs m3-max-52186 and m3-max-52188, untracked
  buffers). A waiter with nothing for the GPU (an encoder that only waits) completes no sooner
  than the last submission that updates its fence, as Metal's does (job m3-max-52194); its GPU
  times are one time, as for any command buffer with nothing for the GPU, where Metal reports a
  span over the update's run. Work with no fence between it is ordered as well, where Metal may
  run it at once.
- `CommandQueue::abandon` ends every command buffer of the queue held at an event wait or not yet
  scheduled with `NotRun` and runs its handlers, as a Mac drops a dead process's held work, each
  retiring in the queue's order. Segments already submitted still complete, and so does a group
  held, which those it ended join and which goes at once; the queue stays usable.
- At a loss of the device (or a failed submission, which the runtime takes as one), the submit
  thread ends what it holds, the completion thread ends what is in flight once the device is idle
  (at once on a lost device), and what waited for that work goes as at its completion (memory given
  back is freed, and an import's release runs), host waits for pending signals end false, and every
  later commit ends at once. `CommandBufferError::DeviceLost { in_flight }` tells the command
  buffers that had passed `vkQueueSubmit2` from the rest. `Device::loss()` names the first loss's
  cause, a `DeviceLoss` of the Vulkan call that failed and its `VkResult`, from before any command
  buffer ends `DeviceLost`; a later loss does not change it. A loss that the consumer's own Vulkan
  call on the device returns (an acquire, a present or another swapchain call) can come before any
  call of the runtime's meets it: the consumer reports it with `Device::report_loss`, a
  `DeviceLoss` of its call's name and result, which the runtime takes as given and which ends the
  device's work by the same path at once, rather than at the runtime's next submission or wait;
  `Device::loss()` then names the reported call, unless a loss came first. Teardown then completes,
  and a new device can be made on the same instance.

### Quiesce and exit

`Device::quiesce(timeout)` is for a process that must end while a handle of the device is still
out. With every wait bounded by the timeout, it stops the submit thread (ready work is submitted,
held work ends `NotRun`) and the block-maker, waits for the GPU to complete what was submitted, lets the
completion thread retire it and end, and then runs `BlockMappings::unmap` on the calling thread for
every allocation still mapped. It frees nothing: every block's memory and mapping stays valid until
the process ends. After it, no runtime thread touches a block; later calls that need a thread
return `Error::Quiesced`. It answers true once the GPU went idle and every thread ended in time,
and false at once in a handler or a listener, whose thread cannot wait for itself. A second call
returns at once, with the first call's answer, or false while the first still runs. After a true
answer, the last handle dropped on the consumer's thread tears the device down in that drop:
every gauge reads zero as it returns. After a false answer a runtime thread can still be in the driver (the completion
thread in `vkWaitSemaphores` with no timeout where the GPU did not go idle, which on NVIDIA's
driver lasts until a later submission that after quiesce the runtime never makes, or in its
retirement calls once a handler that held it returns, and the reaper or the block-maker in a
pipeline creation or a memory call that outlasted the timeout). Quiesce leaves such a thread
running and keeps its handle: the device's last handle joins it, with no deadline, before it
destroys anything the thread can reach, so a drop after a false answer waits for that thread.
After a false answer the consumer keeps the device and ends with `_exit`, or drops its last handle
on its own thread, which joins with no deadline; a handle still held by a handler is then freed
only by the process's end, as the reaper that would tear the device down may have ended.

The consumer drops the device's last handle before the process exits. A process that cannot,
because a handle is still out, quiesces and ends with `_exit`, whatever quiesce answered: `exit()`
with a live device is supported on no device. The evidence: on an RTX 3090 (driver 620.70.06), 3 of
9 of a consumer's power-offs that called `exit()` after quiesce answered true ended in a
segmentation fault inside `exit()`, in a thread of the driver's own reading near a null pointer
while the driver's exit handler waited on a lock, with every runtime thread already ended (an RTX
2080: 0 of 4).

### Known gaps

- A lost device is not recovered. After a loss, the device and everything made on it stay dead:
  its buffers, textures, heaps, pipeline states, queues and events, and every device address and
  resource ID it handed out. Every later commit ends `DeviceLost`. The consumer may make a new device
  on the same instance and make its resources again there; nothing carries over, and an address or
  resource ID from the lost device names nothing on the new one. Vulkan gives an imported allocation
  no way to take a chosen device address (`VUID-VkMemoryAllocateInfo-opaqueCaptureAddress-03333`),
  so a guest's addresses cannot be kept across a new device. A failed `vkQueueSubmit2` of any
  result, out-of-memory included, ends the device; `Device::loss()` names the call and result,
  and, at the runtime's own `ERROR_DEVICE_LOST` on a device that lists `VK_EXT_device_fault`, the
  driver's fault report as given (`DeviceLoss::fault`), each address placed among the runtime's own
  ranges (`DeviceFault::places`, under Fault reports), and the consumer places them among its own
  blocks and buffers.
- Command buffers committed while an earlier one of their queue runs are reported complete
  together, at the last one's end, where Metal reports each at its own. Measured with the harness
  row `long_4cb` (four command buffers committed together, two runs in flight): on an RTX 2080,
  at about 12 ms of GPU time each, the third was reported with the fourth, about 12 ms after its
  own end, and the second 0.8 ms after its own at the median (25 ms at the 90th percentile); on an
  RX 550, at about 7 ms each, the third about 7.7 ms late and the second 0.9-1.4 ms at the median
  (15 ms at the 90th). The last command buffer's own report is not late: on the 2080 it came
  97.2-97.5 ms after its commit. The harness prints each command buffer's lateness, from its
  `gpu_end_time` to its completed handler (`late_us`).
- Every queue's command buffers go to the device's one `VkQueue`, which takes one
  `vkQueueSubmit2` call at a time (Vulkan requires a queue's host access to be externally
  synchronized), all of them made by the device's one submit thread. A command buffer is
  scheduled (its status, its scheduled handlers, `wait_until_scheduled`) by that thread once its
  earlier calls have returned, whichever queue they were for, so one long driver call delays
  every queue's scheduling by its length. Metal promises no time between a command buffer's
  dependencies resolving and its scheduling. Measured on an Iris Xe (RPL-P, Mesa's ANV, job
  m3-max-52564): the device's first submission, one command buffer filling a new 256 MiB block
  192 times, spent 27 to 30 ms in the call, and another queue's command buffer whose wait that
  submission's signal met, committed 0.05 ms after it, was scheduled 0.02 ms after the call
  returned, where an Apple M3 Max schedules it while the signalling command buffer still runs.
  The call's time grows with the new block it is the first to use: 2.1 and 2.5 ms with a 1 MiB
  block, 9.1 ms with 64 MiB and 28.4 ms with 256 MiB (job m3-max-52745); the probe's later
  submissions spent at most 0.18 ms each.
- On the Mac host (KosmicKrisp, Mesa 26.3.1, on an Apple M3 Max under macOS 27.0.1), the first
  command buffer committed after an idle pause starts later the more memory the device holds,
  and the time it adds is spent in the driver's `vkQueueSubmit2`. Measured with a bare Vulkan
  program on a device of its own (no runtime code), one 256-byte fill committed after a 2 s pause,
  arms alternating, as p50 / p90 / max of commit to GPU start: with nothing held beyond its own
  objects, 8.49 / 10.74 / 11.06 ms, 0.18 / 0.31 / 0.71 ms of it in the call; holding 6
  allocations of 64 MiB, 18.45 / 23.88 / 24.44 ms, 12.32 / 14.66 / 14.68 ms of it in the call
  (14 each, job m3-max-69600). An earlier job read 5.75 / 8.62 / 9.32 ms with nothing held, 8.12
  to 9.43 ms at the median with 48 MiB held as 6 x 8, 48 x 1 or 1 x 48 MiB, and 20.55 / 25.73 /
  28.43 ms with 6 x 64 MiB, 14.03 / 20.80 / 22.27 ms of it in the call (12 each, m3-max-69540).
  After a 0.1 s pause the held 6 x 64 MiB cost nothing measurable: 0.77 / 0.85 / 0.86 ms against
  0.79 / 0.86 / 0.87 ms with nothing held (12 each, m3-max-69600). Read from the driver's source,
  not instrumented: each `VkDeviceMemory` is a Metal heap in one residency set of the device,
  added to the device's Metal queue, and every `vkQueueSubmit2` commits that set and calls its
  `requestResidency`. Metal itself pays a like delay for 384 MiB in a residency set added to a
  queue. In the same job, a Metal program holding 6 x 64 MiB in such a set, committed once and
  never requested, started the same fill 19.44 / 26.15 / 26.25 ms after its commit, whose call took
  0.03 ms; requesting residency before each commit put 13.18 / 15.53 / 17.49 ms in the calls and
  left the start at 20.96 / 23.74 / 26.46 ms; the same buffers in no set read 7.85 / 9.29 /
  11.52 ms, and an empty set 8.37 / 10.07 / 10.12 ms (14 each). A Metal 4 queue, the kind the
  driver uses, read alike: 18.86 / 23.09 / 23.35 ms with the set added, and 19.96 / 21.44 / 25.25
  ms with the request, 12.94 ms of it in the calls at the median. So at this size and pause the
  delay is Metal's own for bytes in a queue's residency set, where the driver keeps all of a
  device's memory; the driver's request in each submission moves it into `vkQueueSubmit2`, on
  the runtime's one submit thread (above). The machine was shared during every run: other
  sessions were active on it.
- A command buffer committed while another queue's long command buffers run waits behind up to
  two whole command buffers of that queue. A queue always has room for two submissions, whatever
  their GPU runs (Queues, above), and the runtime never cuts a command buffer, so the room's 16 ms
  bounds only work shorter than that. Counted, not timed, on lavapipe: a queue that commits three
  command buffers of 30 ms each holds two of them, 60 ms, on the `VkQueue` when an idle queue's
  command buffer is committed (job m3-max-69402). The runtime takes no queue priority. On an Intel
  Mac (macOS 26.7.1), Metal's UHD 630 also runs the later command buffer after the running ones in
  commit order (167 to 222 ms later) when no queue is marked, and starts it 0.4 ms after its commit
  beside the running command buffer when the busy queue is marked to a lower GPU priority (8 of 8,
  job m3-max-68129); its Radeon Pro 5300M ran both queues' work at once (job m3-max-68137).
- On NVIDIA each submission pays the driver's latency per submission. The runtime submits each
  command buffer as soon as it can, so short command buffers committed back to back reach the GPU
  in several submissions: on an RTX 2080 (harness row `frame_5cb`: five short command buffers a
  run, two runs in flight) a run takes about 2.5 submissions, and each submission boundary leaves
  the GPU idle for about 20-25 µs.
- One command buffer whose own GPU run passes the driver's job timeout still ends there: the
  runtime does not cut a command buffer's work into shorter jobs, and joins none to it (the
  grouping bound above). Linux's amdgpu ends a job past its timeout, 2 s by default (Linux 7.3.3),
  and resets the ring, where Metal runs the command buffer within its own watchdog's longer bound.
- A command buffer whose timestamps are not available when its submission signals reports no GPU
  times: its `gpu_start_time` and `gpu_end_time` are 0, and `Metrics::stamps_not_available`
  counts it, where Metal reports the times of every command buffer that completes. A signal says
  nothing about a query's availability: Vulkan defines a dependency on the timestamp writes only
  for a read with `VK_QUERY_RESULT_WAIT_BIT`, and the completion thread reads without it, so it
  never waits on a query.
- GPU start and end times are the device's timestamps. On Mesa's Apple-GPU driver a timestamp
  reads the end of the Metal encoder that holds it: the driver writes it with Metal's relaxed
  timestamp granularity into the compute encoder open there, which holds the compute and blit
  work up to the next render pass. So a command buffer's `gpu_start_time` is the end of its
  compute and blit work before its first render pass: one with no render pass reports
  `gpu_start_time` equal to its `gpu_end_time`, a length of zero, and one that dispatches before
  its first render pass reports a start after those dispatches. Its `gpu_end_time` is its
  work's end. An upload or a copy-in holds no render pass, so there its span, in
  `Metrics::runtime_command_buffer_gpu_nanos`, is 0. Measured on an M3 Max with Mesa 26.3.1, in Vulkan alone, around a 16 ms dispatch:
  two timestamps read one value with a pipeline barrier or a buffer fill between them too, 43-56
  µs apart with an empty render pass after the dispatch, and the end one came 51-213 µs before
  the host's calibration after its fence wait. Vulkan allows it: a timestamp may be latched at
  any logically later stage. The two timestamps
  also cost each committed command buffer about 93 µs there: on an M3 Max with Mesa 26.3.1, a
  burst of command buffers of two dispatches each took 122-124 µs a command buffer with them and
  29.5-29.8 µs without. The driver writes a timestamp outside a render pass with a one-thread
  dispatch in a compute encoder, making the encoder where none is open, and resolves each on the
  GPU as the encoder ends. On NVIDIA's driver 620.70.06 the start timestamp costs a command
  buffer about 2.6 µs and the end one about 0.3 µs: on an RTX 2080, a burst of 2,000 command
  buffers of one dispatch each took 23.88 µs a command buffer with the start alone, 21.57 with
  the end alone and 21.29 with neither, medians of 40 runs. Writing the start at the
  bottom-of-pipe stage costs the same, 23.68.
- On lavapipe a command buffer's completion waits for all the device's later work, of every
  queue: the driver's read of query results waits for the whole device to idle
  (`lvp_GetQueryPoolResults`, with or without `VK_QUERY_RESULT_WAIT_BIT`), and the completion
  thread reads each command buffer's two timestamps before its completed handlers run, and each
  upload's and copy-in's as it retires. A short
  command buffer committed beside another queue's long one completes once the long one ends.
  Measured with Vulkan alone on Mesa 26.2.6: a read of a completed submission's timestamps
  returned 500.4 ms later, once a later batch that waited on the host could run. It delays and
  never hangs: the runtime submits no wait that only the host can meet (it holds such a segment
  on the host), so the device always reaches idle. A command buffer waiting on an event that
  another queue's completed handler set completed 2.2 ms after its commit there.
- A command buffer that finds every pair of the device's timestamp pools taken, where the device
  then refuses a new pool, runs untimed: its `gpu_start_time` and `gpu_end_time` are 0, and
  `Metrics::untimed_command_buffers` counts it. Once the device refuses a pool it asks for none
  again. A command buffer whose pair reads as not available once its submission signalled has
  times of 0 too, and `Metrics::stamps_not_available` counts it: a submission the device cancelled
  at a reset, or a driver that makes timestamps available only after the signal. A command buffer
  retires only once a host wait has seen its submission's progress value: with one value pending
  the completion thread asks only that wait, and with more, where the semaphore's counter read
  runs ahead of the wait, `Metrics::waits_behind_counter` counts the round, and nothing retires
  until the wait agrees. The device tests, the harness and the replay player
  report both counts. Vulkan states no most queries a pool may hold or most pools a device may
  make; the runtime's 4,096 a pool is the smallest met: Mesa's Apple-GPU driver makes each pool a
  Metal counter heap, of at most 4,096 entries and at most 32 live in a process, so there at most
  65,536 command buffers of a process are timed at once, fewer where other devices of the process
  hold heaps.
- On NVIDIA's Linux driver 620.70.06 (an RTX 3090), a process that makes and destroys a Vulkan
  instance and a device over and over gets `VK_ERROR_INCOMPATIBLE_DRIVER` from its 27th
  `vkCreateInstance`, with no runtime in the process: a bare loop made and destroyed 26, twice
  alike. `Instance::new` fails there too. The runtime destroys each instance and device it makes:
  under the validation layer on lavapipe, 36 device tests in one process made 34 instances and
  destroyed 34, made 36 devices, and left no object undestroyed.
- Values the runtime reads back from GPU-written memory decide nothing beyond the reading command
  buffer's own results (`src/readback.rs`; a rule test holds every such read there). The sites:
  the two timestamp queries of a command buffer, which decide only its `gpu_start_time`, through
  arithmetic defined for every value and never later than its `gpu_end_time`, read with no wait
  once a host wait has seen its submission's progress value, so a pair the device never
  wrote (a submission it cancelled at a reset) reads as no times at once; and
  `Texture::get_bytes`'s copy out of its staging span, whose bytes reach only the caller's
  buffer, at offsets and lengths from the call's checked region. The parameter ring
  and an indirect dispatch's arguments are read only by the GPU: the runtime writes the ring and
  never reads it back, and never reads the arguments.
- Guest processes are not isolated from each other. One device serves every guest process, and
  shaders use raw device addresses, so a guest shader can read and write any memory on the
  device: other processes' buffers and textures, and the host's own resources there. The runtime
  does not isolate processes: one guest process can read and overwrite another's Shared buffer,
  Private buffer and texture through addresses and resource IDs it makes up, with no fault. The
  consumer treats every byte on the device as
  guest-writable and keeps nothing there that it parses or trusts.
- On NVIDIA, a GPU fault in a guest's kernel hangs that command buffer's completion, where Metal
  ends it with an error. A fault (an access to an address nothing maps) loses the device on NVIDIA's
  driver and on RADV, and one device serves every guest process, so every guest's later work ends
  `DeviceLost`; a second Vulkan device, such as another process's, keeps running. NVIDIA's driver
  (620.70.06; an RTX 2080 and an RTX 3090) tells only a later submission or `vkDeviceWaitIdle`: the
  faulting submission's semaphore never signals, and an unbounded wait for it had not returned after
  60 s. A wait on the faulting work returns only after something submits on the device: a later
  submission meets the loss, the completion thread wakes, and everything ends as a loss, so every
  guest process sees the device lost. A submission soon after the fault may still succeed: where
  the fault leaves a context hung, a submission can succeed until the driver's own context-switch
  timeout, which on an RTX 3090 came 4.3 s after the fault in one play and 10.1 s in another. The
  hang lasts indefinitely only when nothing else submits. RADV (a 780M and an RX 7900 XTX)
  signals the semaphore within about 2 s and refuses the next submission; Intel's Windows driver (a
  UHD 630) does so at once; ANV (an Iris Xe) takes the store with no fault. On RADV (a 780M), the
  command buffers of the submission a reset cut can report completed, since the reset signals its
  semaphore before any call reports the loss: the loss is reported by the next call, where Metal
  ends the faulting command buffer with an error. On a device whose driver reports no loss for
  such a read, it can stall the device with no error, and the runtime's waits end only on a loss
  the driver reports: on an RTX 3090 a read of freed memory hung its context (Xid 109) for 10
  minutes with no loss reported until its process was killed (job m3-max-62569). The runtime
  refuses a kernel that stores through
  a pointer to an address nothing maps, and took a store 64 TiB past a buffer without a fault. A
  bad address that faults ends the GPU's work for every guest process on NVIDIA and RADV: the
  consequence of one device and direct addresses, which the owner accepted.
- A pointer the guest supplies, loaded from memory or made from an integer, null included, is a
  device address used as given: no binding bounds its accesses. Metal leaves an access through one
  that names no live buffer undefined; here it reads or writes whatever the device holds at that
  address, or faults, and a fault ends as the device ends it, up to a device loss that every guest
  process shares (above). What is bounded is a buffer argument's access at a byte offset that is
  not a translation-time constant: outside its buffer a load reads zero and a store does nothing
  (below). An access at a constant offset is not checked.
- No guest word indexes a set 1 array past its length. A slot's texture or sampler, and a
  constexpr sampler, are read at the runtime's own placement (texture indices 1 to 65,535, sampler
  indices 1 to 3,999; a placement outside them is refused). An argument buffer's texture is read at
  its resource ID's index masked to its low 16 bits, read or sampled, and an argument buffer's
  sampler at its resource ID's index where that is below 4,000 and at element 0 (the default
  sampler) otherwise, so a bad ID reads an element of the arrays, never one past them. Metal leaves
  a resource ID that names no live resource of the access undefined: an index within an array that
  no live object owns reads its null descriptor or default sampler, with no fault.
- An indirect dispatch's threadgroup counts are the GPU's to read, so a count above the device's
  `maxComputeWorkGroupCount` is not refused; Vulkan leaves such a dispatch undefined. A direct
  dispatch past it is split into dispatches of at most that many, each told its first
  threadgroup.
- `setBytes` takes up to 64 KiB, one ring chunk (`Limit::BytesLength`).
- A buffer argument bound to nothing reads the nil buffer: 64 KiB of zeros, one ring chunk the
  device keeps from its first such dispatch or draw, its address the argument's pushed value, as
  Metal's nil buffer reads zeros. Its pushed buffer is 0 bytes long, so every checked access of
  it (below) is outside. Reads past the 64 KiB and writes through a nil argument at a constant
  offset aren't covered.
- A buffer argument's access at a byte offset that is not a translation-time constant is checked
  against its buffer, from the buffer's start to its end, as Metal's domain is the buffer: a
  dispatch or draw pushes, beside the argument's address, the bind offset and the buffer's length
  (a `set_bytes` copy: 0 and its length; nil: 0 and 0). An offset before the bind offset reads and
  writes the buffer's bytes there. Outside the buffer, a load reads zero, from 16 bytes of zeros
  the device keeps in a ring chunk of their own, taken at the first dispatch or draw that checks a
  load, which nothing is bound to and nothing writes; a store or an atomic does nothing, and an
  atomic answers 0. This holds on every device. An access at a constant offset is not checked,
  before the bind offset or past it: a draw or dispatch that binds a buffer shorter than the
  function's constant-offset accesses to it meets the device's own out-of-bounds behaviour, and
  where nothing maps there that is a fault, which loses the device on NVIDIA's driver and on RADV
  (above). An access through a pointer loaded from memory or made from an integer is not checked
  either: no binding bounds it. The check's measured cost, in GPU time and in cold pipeline
  makes, is in docs/API.md, section 5 (`PushField::SlotBounds`).
- A dispatch's threadgroup memory, the kernel's own and the lengths set, each rounded up to 16
  bytes, is at most the device's `maxComputeSharedMemorySize` (`Limit::ThreadgroupMemory`), where
  Metal's is its `maxThreadgroupMemoryLength`, which `Device::max_threadgroup_memory_length`
  answers with room for a partial threadgroup's vote word. A kernel's access past its threadgroup
  memory is undefined, as in Metal, and the runtime does not confine it.
- Threadgroup memory, a function's declared threadgroup variables and its `[[threadgroup(n)]]`
  arguments, reads zero in every threadgroup until the program writes it, as an Apple GPU reads it
  (the translator's null initializers, docs/API.md, "Threadgroup memory before the program writes
  it").
- On the Intel UHD 630 under Windows (driver 101.2145), a kernel whose thread memory is in the
  private arena can return 1 or 2 wrong results in 512 from one long dispatch of its whole grid,
  intermittently: 2 failing runs of 31 alone at one build and 0 of 31 at the next (m3-max-68291,
  m3-max-68314), and 1 failing suite run of 11 (m3-max-68142). No other device the suites run on
  shows it, no split dispatch has shown it, and a kernel that checks each of its reads met no wrong
  one in 5 runs (m3-max-68367). The cause is not found.

## Compute encoders

- `CommandBuffer::compute_command_encoder(DispatchType)` is `computeCommandEncoderWithDispatchType:`.
  While it is open, the encoder holds the command buffer's recording: another encoder, an encode
  call, `enqueue` and `commit` are refused (`Refusal::EncoderOpen`). `end_encoding`, or dropping
  the encoder, gives it back.
- `set_texture` (128 slots, `Refusal::TextureArgumentIndex` past them;
  `Refusal::TextureWithoutIndex` for a texture reached through its views alone, which has no
  index a slot could push) and `set_sampler_state` (16 slots, `Refusal::SamplerArgumentIndex`)
  bind as Metal's calls do; `use_resource` takes a
  texture as it takes a buffer. A dispatch pushes, for each texture and sampler slot its kernel
  reads, the bound object's `gpuResourceID` (0 for none), whose index names its elements of set
  1; for a texture slot an integer write saturates through, the channel widths of the bound
  texture's format (`metal::format::integer_channels_word`); and for each constexpr sampler it
  samples through, the index the consumer gave it at the state's creation. Nothing is written
  into set 1 at bind or dispatch time, and every texture stays in its home layout.
- `set_visible_function_table` (`setVisibleFunctionTable:atBufferIndex:`; the range call is a
  loop of it) binds a table's words at a buffer index, through the buffer path: the dispatch
  pushes their address, and the command buffer keeps them until it retires. `None` unbinds the
  slot, which then reads the nil buffer's zeros, a null table.
- `set_stage_in_region` (`setStageInRegion:`) sets the region a kernel's `[[grid_origin]]` and
  `[[grid_size]]` read, pushed at each later dispatch. A dispatch of a kernel that reads it on an
  encoder that set none is `Gap::NoStageInRegion`: what Metal's kernel reads then is not
  established. A kernel's `[[stage_in]]` attributes, under its descriptor's
  `stageInputDescriptor`, are fetched in the shader from the buffer bound at each layout's
  `bufferIndex`, at its pushed address, for every step function a kernel may use (`Constant`,
  `ThreadPositionInGridX` and `Y`, and their `Indexed` forms at step rate 1, whose indices are
  read from the buffer at `indexBufferIndex`). A thread-position step's element is the region's
  origin plus the thread's position over the step rate, clamped to the region's last element, so a
  thread past the region, whose fetch Metal leaves undefined, reads that element (the Metal
  recordings of `stage-in-region-clamps`, Apple M3 Max). Each fetch keeps its reach check: an
  element past the first 2^32 bytes from its bind offset reads zero.
- A kernel whose private memory passes the translator's 16 KiB floor (its arrays a thread
  indexes at run time: Metal holds such a kernel's thread memory itself) keeps it in the runtime's
  device memory, the private arena (`RuntimeFact::PrivateArena`): one slot of the kernel's bytes,
  rounded up to 16, for each invocation of the dispatch, at the address the dispatch pushes
  (`PushField::PrivateArenaAddress`), its invocation's slot at its index in the Vulkan dispatch.
  The arena is one Private block of the pool, never zeroed (Metal's private memory is undefined
  until the thread writes it). The device keeps the last block it took while a command buffer
  keeps it, and a dispatch that needs no more takes it again; one that needs more takes a block of
  its need rounded up to a power of two. A dispatch's slots take at most a sixteenth of the
  device's largest device-local heap, within its longest block, or one threadgroup's slots where
  those are more: a dispatch past that runs as several, each of as many whole threadgroups as the
  budget holds, its region carrying its first threadgroup, as a count past
  `maxComputeWorkGroupCount` does. Each piece is ordered after the commands before it by a
  barrier, but the first in a block the dispatch took new, which nothing reached before; a
  dispatch of a kernel without an arena takes no such barrier.
  `Metrics::private_arena_dispatches`, `private_arena_pieces` and `private_arena_blocks` count
  them, and `private_arena_allocations` the takes that allocated. An indirect dispatch of such a kernel
  is `Gap::IndirectPrivateArena`: the GPU reads its counts, so the runtime can neither size its
  arena nor split it.
- `use_heap` is `useHeap:`, for the block a heap's memory is: the command buffer keeps the block
  until it retires, as it keeps a used buffer's block, so a heap given back while the work runs
  stays valid under it. It declares no access (Metal states no retention for it; keeping is the
  safe reading).
- A dispatch of a kernel that writes no memory (the translator's `PipelineInterface::writes_memory`:
  no store to a buffer or through a device address, no texture write and no atomic update, in the
  kernel or any function it calls) records nothing, binds nothing and reads no indirect buffer,
  once its threadgroup size is checked: it has no effect a program can observe.
  `Metrics::dispatches_without_effects` counts them. Its GPU time and its command buffer's are
  then shorter than Metal's.
- Each dispatch records at once. What it touches is its pipeline's bound buffer and texture
  slots, with the access the translator states for each (a texture's sample and read are reads),
  the `use_resource` declarations, and an indirect dispatch's arguments buffer (a read). The
  hazard sets (resources read and resources written since the last barrier) are kept by resource,
  a buffer or a texture being one resource, and every resource placed in one heap
  (`Device::new_heap_buffer`, `Device::new_heap_texture`) being that heap, whatever its tracking
  mode and whatever bytes each reaches: Metal tracks a heap's resources as one ("tracking is done
  at the granularity of the heap", `MTLHeap.h`'s `hazardTrackingMode`), so a write through one
  orders every later use of another. They are a table indexed by
  a small resource id with a generation counter, so clearing them is one increment. They carry
  across the encoders of a command buffer until a barrier.
- A `Serial` encoder places one global memory barrier before a dispatch that reads or writes a
  resource in the written set, or writes one in the read set. A pipeline that reads through an
  address outside its slots conflicts with any write; one that writes so conflicts with
  everything and owes a barrier before the next dispatch. Every resource takes part whatever its
  tracking mode: serial dispatch is serial order in Metal. A `Concurrent` encoder places barriers
  only at `memory_barrier`, and begins with one where anything was read or written since the last.
- An encoder that wrote anything owes one barrier at its end. The next command records it, or the
  segment's own end barrier pays it, so a command buffer of one writing encoder has one barrier.
- A barrier within a segment orders dispatches after dispatches: compute shader writes before it,
  compute shader reads and writes after it, and indirect argument reads only where the dispatch
  after it is indirect. An indirect dispatch after a write that no barrier since has ordered
  before indirect argument reads gets one that does, in either kind of encoder. The other
  commands a segment records that touch memory, blit copies, order themselves against everything
  before and after them; the barrier at a segment's end takes every command and the host.
- A dispatch whose kernel reaches a buffer bound where its address, its `gpuAddress` plus the
  offset, is not a multiple of the kernel argument's alignment (`air.arg_type_align_size`, at
  most 16) is refused (`Refusal::BufferOffsetAlignment`) before anything is recorded, as Metal's
  API validation refuses the bind offset. Plain Metal runs such a dispatch at addresses of its
  own; `docs/API.md` section 10 says which.
- Parameters: fields in the push window, `[0, W)`, are pushed, `W` the device's
  `maxPushConstantsSize` up to 256 bytes (`DeviceProfile::push_window`: 256 on every lab device,
  128 on a device that reports Vulkan's least): the dispatch region and up to 24 buffer addresses
  below the kernel's tail at 256 bytes, 8 at 128 (`docs/API.md` section 5), each
  kernel pushing only the bytes it reads. The tail placement's price, against the build before
  it: `wide_bindings` is 4.6% slower on an RTX 3090 and 6.9% slower on an Iris Xe and 9.3% faster
  on a Radeon 780M; `dispatch_chain` 3.0% slower on the RTX 3090; `bind_2000` 25.4% slower in
  GPU time on the RTX 3090 with host time level; the other eight harness rows level on all three
  cards (jobs m3-max-58429, m3-max-58430, m3-max-58431). In a states-only play of all 34
  recorded streams on lavapipe (jobs m3-max-58471, m3-max-58477, m3-max-58478, m3-max-58483), no
  recorded kernel keeps a field in its block: 171 kernel stages push past 232 bytes, over
  2,805,608 dispatches, with 0 fields in a block. The room the tail takes moves fields into a
  block for 41 fragment stages only: 374 fields over 23,690 draws. `wide_bindings`' loss is a
  harness row's, not a recorded guest's. Fields past the window go to a 16-byte
  aligned block in the parameter ring, whose address is a field in the window. A kernel whose
  buffers' bounds do not all fit beside its addresses rings one: one of 12 buffers pushes all of
  its fields, and one of 24 or 30 pushes 22 addresses beside the block's. `Counters::metrics`'
  `ring_writes` counts each block and each `set_bytes` copy.
  `set_bytes` copies its bytes into the ring at the call, as Metal copies them. The ring is 2 MiB
  blocks of the floor's ring memory type (device-local, host-visible, coherent), split into
  64 KiB chunks that a command buffer takes as it records and gives back when it retires; a block
  is made when no chunk is free and kept for the device's life.
- The command buffer keeps each pipeline state, buffer, texture and sampler state an encoder
  sets, binds, declares or copies, once, until it retires, as Metal's command buffer retains each
  object once, so their memory and their elements of set 1 stay theirs while work may read them.
  The first use of an object in a command buffer takes its one reference; a later bind of it, and
  every binding slot that holds it, borrows that one and changes no count. The completion thread
  releases them as the command buffer retires, and the lists keep their capacity for the queue's
  next command buffer.

## Blit encoders

- `CommandBuffer::blit_command_encoder` is `blitCommandEncoder`. While it is open it holds the
  command buffer's recording, as a compute encoder does.
- `copy_from_texture` is `copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:`
  `toTexture:destinationSlice:destinationLevel:destinationOrigin:`, a `TexturePlace` holding each
  side's slice, level and origin. It records one `vkCmdCopyImage` between the two images, at once,
  with nothing staged and no byte copied beyond the region. Both textures have one pixel format
  and one sample count, as Metal's validation requires; a compressed region is whole blocks or
  reaches the level's edge on both sides, and whole blocks may end inside the level's last block
  past its edge, as Metal copies the one 4 x 4 block of a level of 4 x 2 texels whole where
  Apple's page asks for the size clamped to the edge (probe `compressed_copies_of_small_levels.m`
  in the conformance crate): the command's extent stops at the edge, both sides' at the same
  one. A 3D texture's image copies into or out of a 2D slice. A format of depth and stencil both
  copies its pixels whole, as Metal's copy takes no plane option: each region names both
  aspects. A size of 0 on an axis a texture lacks copies one
  texel there; on another axis the copy copies nothing.
- A copy records two barriers. The first moves each texture's level and slice from its home
  layout to the transfer layout after every earlier command, and makes every earlier write visible
  to every later command; it pays a barrier an encoder owed, and the hazard sets start again. The
  second moves them home once the copy has run, and holds one global memory barrier beside the
  image barriers, so its writes are visible to every later command through any resource over the
  same bytes, a heap's aliases among them, not through the image alone (Vulkan's memory aliasing
  asks a barrier between two aliases' uses to reach every byte they share). A copy so needs no
  hazard tracking and leaves none: a command buffer of one copy has three barriers, the segment's
  end among them. The encoder owes nothing at its end.
- Refused: what Metal's validation refuses, textures of two pixel formats or two sample counts
  (`Refusal::TextureCopyMismatch`), and a region outside either texture or off its block grid
  (`Refusal::TextureRegion`), a size past the block the level's edge rounds up to among them
  (8 x 4 at a level of 4 x 2, which the validation layer stops and plain Metal answers
  differently by each call). A size of no whole blocks that stops at the source level's edge
  and inside the destination's level is off the destination's grid too: 4 x 2 texels of a
  compressed level of 4 x 2 into the second block of a level of 8 x 4, the size Apple's page
  asks for. It stays refused for three facts:
  - the validation layer stops it ("size.height(2) must be a multiple of MTLPixelFormatBC1_RGBA
    blockHeight(4)"), though Apple's page asks for the clamped size;
  - plain Metal lands the block by no one rule: from a level of 4 x 2 at the block the copy
    names, from a level of 2 x 1 at the level's first block (probe
    `compressed_copies_of_small_levels.m`, an Apple M3 Max, job m3-max-73696);
  - Vulkan's prose asks a region of no whole blocks to match each image's edge ("Copy regions
    for each image must be aligned to a multiple of the texel block extent in each dimension,
    except at the edges of the image, where region extents must match the edge of the image"),
    though in Vulkan 1.4.366 only the source's extent has a valid usage of that
    (`VUID-vkCmdCopyImage-srcImage-01728`), and the destination's extent is bounded only by its
    level (`VUID-vkCmdCopyImage-dstOffset-00150`).
- Not built (`Gap::TextureCopy`): what one `vkCmdCopyImage` in the floor does not make (a
  multisample texture, one pixel format held in two Vulkan formats by the textures' usages, a 1D
  texture with another type, and whole blocks that end inside one level's last block past its
  edge and not inside the other's, as the 4 x 4 block of a level of 4 x 2 texels into a level of
  8 x 4, which plain Metal copies whole: one extent stops at one image's edge alone).
- Within one level and slice of one image (a texture into itself, or a texture and its view,
  which shares its image), regions apart copy with one `vkCmdCopyImage`, that subresource in the
  general layout: one barrier moves it there and one home, the same three barriers as any copy.
  The same region copies nothing and records no command, every texel already its own.
- Overlapping regions there, which Metal leaves undefined (Apple's `copy(from:...to:...)` page:
  "unexpected behavior"), copy as if the source were read whole before any texel is written,
  which is what an Apple M3 Max gives. No `vkCmdCopyImage` reads a texel it writes, so the region
  is cut into slabs along one axis it is shifted on, the axis that takes the fewest. Each slab is
  no thicker than the shift there, and the slab at the far end of the shift goes first. Each slab
  is one `vkCmdCopyImage`, with an execution barrier between two slabs. A shift of one texel takes
  one slab and one barrier per texel of the region's extent on that axis.
- Regions of two textures in place over one buffer whose bytes meet copy the same way, each shift
  that from a destination row to a source row on its bytes, by the row pitch the device reports
  for the linear images: in rows along y, in whole texels along x. Regions on the same bytes copy
  nothing. Where the rows meet shifted both ways, or not shifted, along each axis (a destination
  row across the end of its own source row and the start of the next), no cut along one axis
  keeps the order, and the copy is `Gap::OverlappingCopy` (below).
- Two textures whose memory meets where one is placed in a heap (`Device::new_heap_texture`),
  which Metal leaves undefined as it does any aliased heap resources' contents: the device lays
  that texture's texels out as it chooses, so no copy command is known to read no byte it writes,
  and the copy is `Gap::OverlappingCopy`. Two slices or levels of one heap texture are one
  image's, and copy as any texture's.
- `copy_from_buffer` is `copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:`
  `sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:` and
  `copy_from_texture_to_buffer` is `copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:`
  `sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:`, a
  `BufferLayout` holding the offset and the two strides. `copy_from_buffer_with_options` and
  `copy_from_texture_to_buffer_with_options` are their options forms, a `BlitOption` naming the
  aspect a copy of a depth and stencil format moves (`MTLBlitOptionDepthFromDepthStencil` or
  `MTLBlitOptionStencilFromDepthStencil`, which Metal takes of a format of that aspect alone too),
  one region of that aspect; the forms without an option are those of `None`. A compressed
  region takes whole blocks ending inside the level's last block past its edge as
  `copy_from_texture` does, each way: plain Metal copies the block of a level of 4 x 2 texels at
  4 x 4 from a buffer too, where Metal's validation layer stops it (`(destinationOrigin.y +
  sourceSize.height)(4) must be <= height(2)`), and the runtime does what plain Metal does.
  `MTLBlitOptionRowLinearPVRTC` is not offered, as no PVRTC texture is made. Each records
  `vkCmdCopyBufferToImage` or `vkCmdCopyImageToBuffer` against the bytes shaders reach (a Managed
  buffer's device copy, where the device keeps one), one region an image of the region, each at its
  own offset, so an image stride of no whole number of rows copies too. The bytes between the
  region's rows and images are kept. Every copy of a colour aspect from a texture to bytes (these, a
  synchronize, `get_bytes` and a texture over a buffer's copy back) whose row stride in bytes is not
  a multiple of M, the largest power of two up to 16 dividing its row's bytes, is two regions, every
  block column but the last and then the last, on every device: the UHD 630's Windows driver
  (101.2145) writes such a copy's rows floor(stride / M) × M bytes apart as one region where its
  offset is a multiple of M. It records the same two barriers as a copy between textures, the
  second's global memory barrier making a write into either visible to every later command, and a
  write into a Shared buffer marks the segment, whose end barrier then reaches the host. A command
  buffer of one copy into a Shared buffer has three barriers.
- A region whose bytes meet those of a texture in place over the buffer (`Buffer::new_texture`),
  either way, copies as two textures over one buffer do: in slabs, each shift that from a
  destination row to a source row on its bytes, the region's rows its `bytes_per_row` apart and
  the texture's at the row pitch the device reports, as from the source before the copy, an
  execution barrier between two slabs. The region on the texture's own texels copies nothing.
  Where the rows meet in no order a cut along one axis keeps, and where the region's bytes meet
  the memory of a texture placed in a heap, the copy is `Gap::OverlappingCopy` (below). Metal
  leaves such a copy undefined; the answer between a buffer and its texture was not recorded on
  a Mac, and is the one an Apple M3 Max gives between two textures over one buffer.
- Refused (`Refusal::BufferCopyLayout`), as Metal's validation stops them: an offset or row stride
  of no whole number of a format's texels, rows or images closer than the region's, and a region
  past the buffer's end. Metal reads no row stride of a region one row high and no image stride of a
  region one image deep, and neither does the runtime. Not built (`Gap::BufferCopyAlignment`),
  though Metal takes them: an image's offset in the `VkBuffer` shaders reach that is no multiple of
  a texel block's bytes, or of 4 for a depth or stencil aspect, and a row stride of no whole number
  of a compressed format's blocks; also what `replace_region` and `get_bytes` refuse
  (`Gap::TextureBytes`), but for a plane option's copy of a format of depth and stencil both.
- `copy_from_buffer_to_buffer` is `copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:`:
  one `vkCmdCopyBuffer` region between the bytes shaders reach (a Managed buffer's device copy,
  where the device keeps one), between the same two barriers as a fill, a copy into a Shared
  buffer marking the segment as a copy into one does. A size of 0 copies nothing. Any offsets and
  any size copy, each byte of the range and no other: Apple's page for the call asks for
  multiples of 4 on macOS, and plain Metal copies such a range exactly (an Apple M3 Max on macOS
  27.0.1, its validation layer on as well, which took the copies without a word, job
  m3-max-69613; an AMD Radeon Pro 5300M and an Intel UHD Graphics 630 on macOS 26.7.1, job
  m3-max-69614; probe `blit_buffer_copy_unaligned.m` in the conformance crate). Refused, as
  Metal's validation stops it: a range past either buffer's end (`Refusal::BufferCopyRange`).
  Refused too: a source and a destination that share bytes, one buffer's overlapping ranges or
  two buffers that overlap in one block (`Refusal::BufferCopyOverlap`), which `vkCmdCopyBuffer`
  forbids and the runtime does not emulate.
- `fill_buffer` is `fillBuffer:range:value:`: one `vkCmdFillBuffer` of the byte in each word,
  between the same two barriers, a fill of a Shared buffer marking the segment as a copy into
  one does. Refused (`Refusal::FillRange`), as Metal's validation stops them: an empty range and
  one past the buffer's end. Not built (`Gap::FillAlignment`): a range that does not start and end
  on whole 4-byte words of the bytes shaders reach, which Metal fills byte by byte.
- `synchronize_resource` is `synchronizeResource:`, counted in `Metrics::synchronizations` and
  never a submission of its own. Of a Managed buffer with a device copy it records one
  `vkCmdCopyBuffer` of the copy into the consumer's bytes between the same two barriers; of a
  Managed texture with named bytes, on every device, one `vkCmdCopyImageToBuffer` of each level
  and slice into its place in them, between one pair of barriers. Either marks the segment so its
  end barrier reaches the host: the host reads the GPU's writes before the call once the command
  buffer completes or it sees an event signaled after it. Of any other buffer, which the host
  reads where shaders write it, and of a texture without named bytes, which has none the host
  reads apart from its image, it records nothing. Metal's validation stops it on a resource
  that is not Managed; the runtime keeps no storage mode for those and does not refuse it.
- `generate_mipmaps` is `generateMipmapsForTexture:`: each level below the texture's first made
  from the level above it, over each slice, by one `vkCmdBlitImage` a level with the linear
  filter, the device's filter under the static note `PrecisionNote::DeviceFilteredMipmaps`
  (`BlitCommandEncoder::GENERATE_MIPMAPS_NOTE`; Apple states no filter). A view's levels and
  slices are its own. The levels leave home behind the copy's first barrier, with one barrier
  between each level and the next, and come back in the last, which holds a copy's global memory
  barrier. A texture of one level records
  nothing. Refused (`Refusal::MipmapFormat`), as Apple defines mipmaps of colour-renderable,
  filterable formats alone: a depth or stencil, compressed, YCbCr or integer format. Not built
  (`Gap::MipmapFilter`): an image the device's blits do not filter linearly (its format query),
  one that holds its pixel format through a conversion (extended-range, YCbCr), and a view in
  another format than its image's.
- The slice-and-level and whole-texture forms of `copyFromTexture`,
  `synchronizeTexture:slice:level:` (no recorded stream calls it), and the other blit commands
  are not built yet.

### Known gaps of blit encoders

- A copy whose source and destination share memory in no order a copy command keeps is not
  built (`Gap::OverlappingCopy`), and each refusal counts in
  `Metrics::overlapping_copies_refused`: no Vulkan copy command reads bytes it writes
  (`VUID-vkCmdCopyImage-pRegions-00124`, `VUID-vkCmdCopyBufferToImage-pRegions-00173`,
  `VUID-vkCmdCopyImageToBuffer-pRegions-00184`). Two shapes of rows in place over one buffer,
  between two textures or a buffer and a texture, are cut along no one axis: a destination row
  across the end of its own source row and the start of the next, and rows of two pitches that
  meet at their starts. Each is built, as a cut on two levels (rows, then the texels within
  them), when a replay stream or a walk counts one.
- A copy between a texture placed in a heap and a texture or a buffer whose memory meets its
  range is refused alike (`Gap::OverlappingCopy`): the device lays the heap texture's texels out
  as it chooses, and Metal leaves the contents of aliased heap resources undefined, so nothing is
  emulated.

## Compute pipeline states

- `Device::new_compute_pipeline_state(&ComputePipelineDescriptor, sampler_ids)` is
  `newComputePipelineStateWithDescriptor:options:reflection:error:` without reflection. The
  translator compiles the kernel on the calling thread, which waits for it, within its 20-second
  budget per lowering, for what device creation enabled and under its identity domains: each
  buffer argument is the guest address of the bound buffer plus its offset, and each texture and
  sampler argument the bound object's resource ID, pushed; set 1 is the only descriptor set a
  kernel reads. A refusal of the translator is `Error::Translation`, whose `functions` names the
  kernel by its own name in its library and whose text prints it, for the program's log as the
  text of Metal's `NSError` is; nothing decides by it. A kernel the consumer compiled has no name
  there.
- rt builds no pipeline reflection and acts on no `MTLPipelineOption`; its consumer builds
  reflection from the library's AIR. The options it leaves to the consumer are
  `MTLPipelineOptionFailOnBinaryArchiveMiss`, `MTLPipelineOptionArgumentInfo`,
  `MTLPipelineOptionBufferTypeInfo` and `MTLPipelineOptionBindingInfo`.
- `Device::translator` (no Metal counterpart) is the translator device creation built. A state of
  a kernel function lowers each module, the plain one and its variants, on it under
  `LoweringInputs::kernel_variant` of the inputs its `kernel_inputs_read` gives the descriptor's,
  so a consumer that lowers a variant there gets the module the state runs.
- What a state compiles is its content, never the state: its descriptor's kernel with its
  constant values, linked functions and dynamic libraries, the descriptor's fields that change
  the compile (the translator's `Device::compute_pipeline_key`), and its
  `requiredThreadsPerThreadgroup`. Every state of one content shares one compiled kernel: its
  translation, its shader modules (a module past the stage bound as its words, with no shader
  module: Kernels past the stage bound) and their `VkPipeline`s, its emulated-SIMD-group variant
  and its precision notes. The device's map holds the last 1,024 contents used, so a state of a
  content it holds costs a lookup and its samplers' numbering, and compiles nothing. Past them the
  map forgets the least recently used content, which every state that took it keeps; a compiled
  kernel goes once its last state and the map have let it go. Each state numbers its own
  samplers. `Gauges::kernels` counts the compiled kernels alive and `Gauges::kernel_entries` the
  map's entries.
- A content the map does not hold is lowered from the device's stage entries (the translator's
  `StageEntries`), which every compute and render creation of the device reads: a module is
  decoded once for the device and a stage specialized once for its function, constants, links
  and call-stack depth, not once a content. They hold at most 32 specialized entries in 64 MiB
  and 4 module decodes in 32 MiB, and forget the least recently used past them;
  `Gauges::stage_entries` counts what they hold and `Gauges::stage_entry_bytes` its bytes.
- A state numbers samplers as Metal's pipeline creation does (`NumberedSampler`, in Metal's
  order): each constexpr sampler its kernel uses, then one where it reads a texture and one where
  it writes one. `sampler_ids` takes that list once the kernel is lowered and answers an index
  for each, which the consumer numbers as Metal does, from its sampler grants
  (`Refusal::SamplerIds` for another count; an error it answers is the call's). A constexpr
  sampler the kernel samples through takes, before the call returns, the device sampler of each
  use the kernel has (`CombinedSampler::constexpr_sampler`, normalized; a pixel sampler's
  coordinate is scaled in the shader) at its index of the three sampler arrays, shared by create
  info with the sampler states'. States that number one constexpr sampler at one index share its
  elements; a constexpr sampler of other state words, or a sampler state, made at that index takes
  it, as any object made at a held index does (above), and a later state that numbers the
  constexpr sampler there takes it back, with every element the constexpr sampler holds. The last
  of its states to go writes the arrays' defaults back and frees the index, where the constexpr
  sampler still owns it. The consumer keeps a constexpr sampler's index from the program's sampler
  states while a state numbering it there lives; rt does not check it. The read and write samplers
  take nothing at their indices. Once `sampler_ids` has answered, a call that fails holds none of
  the indices.
- Every kernel is lowered at creation, its variants with it. The translator answers, lowering nothing
  (`Device::kernel_observes_partition`), whether it observes its threadgroup partition (it holds
  threadgroup memory, a barrier, fence or atomic of threadgroup or SIMD-group scope, a SIMD-group
  or quad function, or reads a builtin of the partition; the translator decides from its AIR),
  unless the cache directory holds the kernel's partial-threadgroups variant or the answer a
  device before it kept there, either of which answers it. A
  kernel that does not observe it is lowered as that variant, which runs a direct dispatch in
  workgroups of the runtime's, whatever its threadgroup size:
  - a grid of a kernel that is image-only (`CompiledFunction::is_image_only`: it samples,
    gathers or reads a texture, and loads, stores and updates atomically nothing through a
    `device` pointer; `constant` loads and texture writes are allowed) in workgroups of 16 by 16
    by 1 where it is one deep, on one `VkPipeline`, the state's main one (below), and of its
    tile (below) where it is deeper, on one `VkPipeline` per tile, made at the kernel's first
    dispatch at that tile, which waits for it, each thread's position the workgroup's origin plus
    its place in it, with no decode. Those pipelines are kept with the kernel's others, at most
    one per tile of the 45 a rule's tile can be. A tile deeper than 64 runs as one row of its 256
    threads (below);
  - a grid of any other such kernel in tiles: workgroups of 256 threads on one `VkPipeline`, the
    state's main one, each one tile of the grid, which its region carries. A thread's position is
    the tile's origin plus its index masked and shifted by the tile's sides, with no division.

  A tile has power-of-two sides whose product is 256. The translator's `pipeline::grid_tile`
  picks a tiling kernel's from the grid's extents alone: launching the fewest threads past the
  grid, then with the fewest tiles partly past it, then the shallowest, then (where the grid has
  two rows or more) the height nearest 2, then the widest. An image-only kernel's tile is picked
  from the grid and the call's threadgroup size: the fewest threads launched past the grid, then
  the fewest tiles partly past it, then the depth nearest the threadgroup's (the shallower of two
  as near), then the most square in `x` and `y`, then the wider. So threadgroups of 8x8x4 over a
  grid that 8x8x4 tiles divide run in 8x8x4, and threadgroups one deep in the square 16x16x1,
  the state's main pipeline, wherever 16x16x1 tiles divide the grid. Where the workgroups
  do not divide the grid, the dispatch's regions are flagged ragged (the translator's
  `NONUNIFORM`), and an invocation past the grid returns before the kernel's body: only there does
  a thread compare its position with the grid's last thread, inside one branch on the flag. Every
  other dispatch, on the same pipeline, branches past that test. An indirect dispatch runs at its
  own threadgroup size, unflagged. A kernel that observes its partition is lowered plainly and
  as its partial-threadgroups variant at the state's creation, through the cache directory: its
  whole `dispatch_threads` grids, `dispatch_threadgroups` and
  `dispatch_threadgroups_with_indirect_buffer` run the plain module at their own size, and a
  `dispatch_threads` grid that is not a whole number of threadgroups runs the variant at its size,
  its regions flagged ragged (`Metrics::partial_variant_ragged_grids` counts those dispatches;
  `Metrics::partial_variant_whole_grids` stays 0). A ragged grid whose threadgroup memory, the
  lengths set with the variant's own, passes the device's `maxComputeSharedMemorySize` once the
  variant's predicated walk adds its 4-byte vote word (a barrier in a loop; the plain module holds
  none) is refused as `Limit::ThreadgroupMemory`. The translator gives a kernel no order of the
  accesses Apple's SIMD-group lockstep orders (docs/API.md, accepted differences), so only the
  kernel's own barriers, fences, atomics of threadgroup or SIMD-group scope, SIMD-group and quad
  functions, threadgroup memory and builtins of the partition observe it.
- A threadgroup deeper than 64, the floor's `maxComputeWorkGroupSize[2]`, runs as one row of its
  threads, a workgroup of `x·y·z` by 1 by 1, each thread's position its index unflattened in
  Metal's order (the translator's `SpecKey::DeepThreadgroup`), on every device alike; any other
  threadgroup is its own workgroup. A kernel's threadgroup memory, barriers, SIMD groups and grid
  are the same either way, an indirect dispatch's included. So every threadgroup Metal allows,
  up to 1,024 threads on any axis, runs on every device, and `MetalDeviceProperties` on the
  runtime's profile reports 1,024 on each axis of `max_threads_per_threadgroup`. The form follows
  from the size, which keys a `VkPipeline` already, so it adds none, and the most a kernel makes
  is the same with it or without: for its direct dispatches, one for the tiles of a kernel that
  runs its grid in tiles, 46 for an image-only one (the square one and one per tile, three of
  whose tiles are deeper than 64), and for one that observes its partition one per threadgroup
  size and threadgroup memory lengths; for any kernel's indirect dispatches, one per size and
  lengths.
- A kernel whose SIMD groups run one per subgroup of 32 (pinned where the device lists compute in
  `requiredSubgroupSizeStages` with 32 between its sizes, or the device's own 32), or one per
  32-lane cluster of a 64- or 128-lane subgroup, runs only workgroups of whole subgroups. So does
  a kernel whose every cross-lane operation is a quad operation, on a device that lists compute in
  `requiredSubgroupSizeStages` and whose narrowest subgroup is below 32 lanes and at least 4: it
  pins no size, takes its quads from the subgroup at the size the device gives
  (`ALLOW_VARYING_SUBGROUP_SIZE`), and runs only workgroups of whole subgroups of
  `maxSubgroupSize`.
- Such a kernel also has an emulated-SIMD-group variant, whose SIMD groups exchange through
  threadgroup memory between workgroup barriers. A threadgroup that is not whole subgroups (one of
  80 or of 1 thread, or of 32 on a device of 64-lane subgroups) runs it at its own size, the last
  SIMD group partly filled, as Metal runs it. The state's creation lowers it, through the cache
  directory, whether or not a dispatch needs it. In a threadgroup of one thread the module's
  exchanges pass no barrier. `Metrics::emulated_simdgroup_dispatches` counts the dispatches that
  ran it.
- The dispatch region (two `uvec4`) gives every direct dispatch's kernel Metal's grid: its first
  threadgroup (0, or a split piece's), the threadgroup size packed as
  `(x-1) | (y-1) << 10 | (z-1) << 20`, and Metal's threads per grid; an indirect dispatch pushes a
  zero grid, and its kernel takes the grid from its threadgroup count. Outside the tiles the packed
  size is the pipeline's own workgroup size, so the kernel's `MultiRegion` constant stays at its
  default, 0, and the kernel takes the size from its workgroup size, with no decode.
- What a state's creation is given it makes there, as Metal compiles a state when it is made, up
  to the driver's create of its main pipeline: every module (the kernel's, and its
  partial-threadgroups and emulated-SIMD-group variants), with every refusal and Metal's own
  errors, and the samplers' numbering; then it answers. The main pipeline is the one `VkPipeline`
  whose key the descriptor holds: a tiling kernel's tiles' or an image-only kernel's square
  workgroups', or, where the kernel takes no threadgroup memory length from the encoder, an
  observing kernel's two, its plain module's of a whole grid and its partial-threadgroups
  variant's of a ragged grid, at the descriptor's
  `requiredThreadsPerThreadgroup`, or, where it sets none and the state declares a bound of
  threads (the descriptor's `maxTotalThreadsPerThreadgroup` or the kernel's
  `[[max_total_threads_per_threadgroup(n)]]`), at `[N, 1, 1]`, `N` the state's
  `maxTotalThreadsPerThreadgroup`. The device's own limit is no declared bound, and a kernel's
  bound of Metal's own 1024 is not told apart from none. The consumer's `finish_pipeline` call
  makes it after the answer (below); a kernel past the stage bound makes it at the creation
  (Kernels past the stage bound, below). A kernel's `VkPipeline` at a threadgroup size,
  threadgroup memory lengths and an explicit imageblock's size only a dispatch gives is made at
  the consumer's `prepare_dispatch` call (below) or at the dispatch: one per module, threadgroup
  size, threadgroup memory lengths (each rounded up to a multiple of 16 bytes) and imageblock
  size, of which a module keeps 64 (Known gaps of pipeline states, below), on the encoding thread
  that first dispatches any of its states with them, which waits for it. Nothing the runtime records depends on when a compile finishes: there is one recorded
  form, made after the wait. Each such wait is counted in `Metrics::first_use_waits`, with its
  time in `first_use_wait_nanos` and the longest in `first_use_wait_max_nanos`, and again by where
  it waited, each with its count, time and longest: `first_use_variant_waits` (a kernel's pipeline
  for a new workgroup size, threadgroup memory lengths, imageblock size or tiles), `first_use_line_waits` (a
  render content's line-class pipeline), `first_use_point_waits` (its point-class pipeline),
  `first_use_raster_waits` (its pipeline under the depth clip mode `Clamp` or the triangle fill
  mode `Lines`) and `first_use_unfinished_waits` (a state's main pipeline, compute or render,
  before its `finish_pipeline` made it).
  No dispatch lowers a function. `Counters::variant_makes(from)` answers the kept makes of
  kernels' `VkPipeline`s numbered `from` or later, the last 256 to end on the device, each a
  `VariantMake`: its key (the `DispatchVariant` its dispatches answer), who started it
  (`VariantStart`: a `finish_pipeline` call, the state's creation for a kernel past the stage
  bound, a `prepare_dispatch` call, or a dispatch's first use), when it began and ended in
  `host_time`, whether it made the pipeline, and the first-use waits that waited for it, with
  when the earliest began. A wait on a make another thread started found that make in flight.
  Neither type has a Metal counterpart.
- `ComputePipelineState::finish_pipeline()` and `RenderPipelineState::finish_pipeline()` have no
  Metal counterpart. Metal compiles a state when it is made, and the program waits for that
  answer. Here the creation answers once everything that can refuse is done, and the driver's
  create of the state's main pipelines is this call's, so a consumer can answer the program first
  and let the driver's compile run beside the program's own work. rt starts no thread for it: the
  consumer calls it on a thread of its own once it has answered. It makes each main pipeline
  through the device's pipeline cache where it is neither made nor being made, so a second call
  makes nothing, and so does a call for a state with no main pipeline or for a kernel past the
  stage bound. A dispatch or a draw that needs a main pipeline first waits for the make in flight,
  or makes the pipeline itself where none is, a first-use wait (`first_use_unfinished_waits`),
  and records what it would have recorded with it made: a state never finished runs the same, its
  first use making the pipeline. `Metrics::finished_pipelines` counts what the calls made, with
  their time in `finished_pipeline_nanos`, the driver's side of a state's make, which the consumer
  reads beside the time of its own two calls; `failed_finishes` counts the calls whose make
  failed (Known gaps of pipeline states, below). Its only error is `Error::Vulkan`.
- `ComputePipelineState::prepare_dispatch(Dispatch, threads_per_threadgroup,
  threadgroup_memory_lengths, imageblock_size)` has no Metal counterpart: it is a hook for a
  consumer's pre-pass. Metal compiles a compute state whole when the state is made; here the
  state's creation makes every module, and a kernel's `VkPipeline`, but for the state's main one,
  waits for the dispatch's threadgroup size, threadgroup memory lengths and imageblock size. A
  consumer that reads a command buffer's dispatches before it encodes them may pass each one's own
  arguments: its grid as its call took it (`Dispatch::Threads` for `dispatchThreads:`,
  `Dispatch::Threadgroups` for `dispatchThreadgroups:`, `Dispatch::Indirect` for the indirect
  form), its threadgroup size, the encoder's `setThreadgroupMemoryLength:atIndex:` settings, in
  stream order, and the encoder's last `setImageblockWidth:height:` size, `[1, 1]` where it set
  none, on threads of its own; rt
  starts none. The call decides what the dispatch runs as the dispatch decides it (one function
  serves both), then makes that module's `VkPipeline`. It returns at once where that is made, or
  another thread is making it. A make runs outside the kernel's lock, its variant marked
  pending, so makes of different variants run at once; only a dispatch waits, and only for its
  own variant's make, a first-use wait. The call makes nothing where the dispatch would refuse
  its threadgroup size, its variant or its threadgroup memory, where the imageblock size is one
  the encoder refuses (`Refusal::ImageblockSize`) for a kernel that reaches an imageblock, or
  where the dispatch would record nothing (a grid
  with no threads, a kernel that writes no memory), and reports no error: the dispatch reports
  its own. The one error it reports is a make refused for the stage bound (`Error::StageUnfilled`,
  or `Gap::StageSize`; Kernels past the stage bound, below), which the dispatch reports too until
  the consumer's helper fills the entry. What the dispatch checks of the encoder's bindings is not checked. A make that failed
  is made again by the next call or dispatch that needs it. What a dispatch records is the same
  whether or not its pipeline was made ahead. `Metrics::made_ahead_variants` counts what the
  calls made, with their time in `made_ahead_variant_nanos`.
- `ComputePipelineState::dispatch_variant`, beside it with the same arguments, has no Metal
  counterpart either: it answers a `DispatchVariant`, the key of the `VkPipeline` the dispatch
  runs, decided by the same function, so a pre-pass can call `prepare_dispatch` once for each key
  it has not met. Two keys are equal exactly when their dispatches run one `VkPipeline`: of one
  compiled content (the states of one content share theirs), the same module, workgroup size,
  threadgroup memory lengths and, for a kernel that reaches an explicit imageblock, imageblock
  size. What the decision does not read does not change the key: an observing kernel's grid
  beyond whether its last threadgroups are partial, a tiling kernel's direct grid and threadgroup
  size, the imageblock size of a kernel that reaches none. Keys of different contents never compare equal. With the
  key it answers whether that `VkPipeline` is made, so a pre-pass can leave a made one alone:
  read from the kernel's first eight variants with no lock, and past them under the short lock of
  the kernel's list, which no make holds while the driver compiles. It may answer false for a key
  whose make ends after the read, and never answers true for one not made. It answers None where
  `prepare_dispatch` makes nothing whatever was made before, makes nothing, and allocates nothing
  where it answers a key. A kernel holds at most 31
  threadgroup memory lengths, the indices `setThreadgroupMemoryLength:atIndex:` sets; a module
  with more is refused at the state's creation (`KernelInputKind::Specialization`).
- Each lowering of a state's creation is one translation within its own 500 MiB ceiling, one
  after another, so the width of the consumer's creating threads multiplies the bound:
  creations on 4 threads at once hold at most 2 GiB of translation.
- That ceiling is on the bytes a translation holds live, and the high-water of what it freed can
  stay resident under the process's allocator once it returns, which is the consumer's to return:
  on glibc 2.44, resident memory stayed within about 10 MiB of the live bytes while an attempt ran,
  and about 140 MiB of an attempt that peaked near 250 MiB stayed resident after it until
  `malloc_trim(0)` returned it.
- `ComputePipelineState::precision_notes` carries the translator's notes of where the kernel's
  answers may differ from Metal's on this device, fixed per operation: those of the modules made
  with the state, its variants among them. A kernel compiled
  with fast math carries `FastMathRewrites`: a driver may fold its arithmetic differently per
  `VkPipeline`, so its answers in a tile and at another size may differ in the last place. A
  kernel that is a function writing a texture slot it does not read carries
  `DevicePackConversion`, for an 8-bit sRGB texture a dispatch may bind there, whose writes its
  sRGB variant encodes (the known gaps of textures): the note is fixed at creation and never
  decided at dispatch.
- `ComputePipelineState::texture_accesses` answers each texture slot the kernel uses, by index,
  with the access a dispatch of the state marks in its hazards for the texture bound there, read
  from the same interface, as a `BindingAccess` (`MTLBindingAccess`): `ReadOnly` where the
  kernel samples, reads or only queries the texture, `WriteOnly` where it writes it and does not
  read it, `ReadWrite` where it does both, an atomic included. Each variant the state lowers
  takes those slots and accesses or is refused, so no dispatch marks others. The access is what the kernel's operations do, not the argument's `access::`
  qualifier, which Metal's reflection answers (`MTLBinding.access`): a `read_write` slot the
  kernel only writes is `WriteOnly`. A texture an argument buffer, a heap or a resource ID
  reaches is in no slot.
- `ComputePipelineState::buffer_accesses` answers each buffer slot the kernel uses, by index,
  with the access a dispatch of the state marks in its hazards for the buffer bound there, read
  from the same interface, as a `BindingAccess`: `WriteOnly` where the kernel writes the buffer
  and does not read it, `ReadWrite` where it does both (an atomic, or a store of part of a word,
  which reads the word), `ReadOnly` otherwise. The access is what the kernel's operations do, not
  the argument's declaration, which Metal's reflection answers (`MTLBinding.access`: `ReadOnly`
  for a `constant` or `const device` buffer, `ReadWrite` for a `device` one): a `device` buffer
  the kernel only writes is `WriteOnly`. A buffer an argument buffer, a heap or an address
  reaches is in no slot, and neither is a function table.
- A state answers Metal's `staticThreadgroupMemoryLength`
  (`ComputePipelineState::static_threadgroup_memory_length`), rounded up to 16 bytes as Metal
  reports it, without the threadgroup memory the runtime adds beside it, and `threadExecutionWidth`
  (`thread_execution_width`), 32 on every device: a kernel that observes its SIMD group runs it 32
  wide, and one that does not cannot tell its width. `imageblockMemoryLengthForDimensions:`
  (`imageblock_memory_length_for_dimensions`) answers 0 for a kernel that declares no imageblock, as
  Metal does; one that declares an imageblock, or whose compile kept no reflection to tell, answers
  `Gap::ImageblockLength`: the one recording of Metal's answer for such a kernel
  (`pipeline-facts-compute-imageblock`, 256 bytes at 8 by 4 by 3) settles no rule.
  `requiredThreadsPerThreadgroup` (`required_threads_per_threadgroup`) is
  the descriptor's size, `[0, 0, 0]` where it set none, as the M3 recordings
  `pipeline-facts-compute-plain` and `pipeline-facts-compute-described` answer, and
  `supportIndirectCommandBuffers` (`support_indirect_command_buffers`) is false: a descriptor that
  asks for indirect command buffers is refused at the making (`Gap::IndirectCommandBuffers`,
  `IndirectCommandsKind::ComputeCommands`).

### The cache directory

Every `VkPipeline` of a device, compute or render, is made through one `VkPipelineCache` of the
device's, internally synchronized, which threads that create at once share.
`DeviceOptions::pipeline_cache_dir` names a directory, made where it is missing, that keeps it and
the lowerings from one device to the next:

- `pipelines-<header>`: the pipeline cache's blob, its data as `vkGetPipelineCacheData` gave it
  after a magic, its length and its checksum, named by the hex of the whole 32-byte pipeline
  cache header the device's driver writes, opaque bytes: a device or driver version whose header
  differs in any byte, its `pipelineCacheUUID` included, has a blob of its own. Devices with one
  header, in one process or several, start from it, and each replaces it. The directory keeps at
  most four blobs: a device's seed removes the oldest past that, by last write, its own counted
  among the four. Taking a blob writes nothing to it. A device whose driver's data of an empty
  cache does not begin with a version-one header (a driver may cache nothing; lavapipe in
  Mesa 26.2.4 gives 20 bytes) keeps no blob: it reads none and writes none, and its cache lives
  with the device.
- `translations/`: each function's lowering, a kernel's or a render pipeline's vertex or fragment
  function's, named by its `FunctionCompileKey`, which covers everything that changes the
  lowering: the function's AIR, entry, constants and linked functions, its lowering inputs (a
  kernel's variant, or what its use in the render pipeline gives), the device profile's
  fingerprint and the translator's semantics version. A state whose lowering is there reads it in
  place of lowering (`Metrics::translations_loaded`, beside `Metrics::translations`); a kernel
  variant the translator refused is kept too, and so is its answer that a kernel observes its
  threadgroup partition, which a state on the directory reads in place of asking it again.
- `renders/`: each render content's record, 122 bytes, named by the hex of the translator's key of
  its descriptor (`Device::queried_render_descriptor_key`), which reads no function: the
  `FunctionCompileKey` of its vertex and fragment functions' lowerings. A state of a content the
  device does not hold reads its record and the two lowerings it names
  (`Metrics::render_records_read`), so neither function is decoded or specialized. The name covers
  everything that changes either lowering: every field of the descriptor, the fields the functions'
  specialization reads among them; each function by its library's AIR hash (its content), its entry,
  its constant values as supplied, its linked functions and dynamic libraries; the device profile's
  fingerprint; and the translator's semantics version. The record holds the translator's semantics
  and artifact format versions and the descriptor's key again under a checksum, so one of another
  version or another content is a miss and is removed. A record naming a lowering the directory no
  longer holds is a miss too, counted (`Metrics::render_records_stale`): the state decodes,
  specializes and lowers as with no record, and keeps the record again. Records are written and
  counted as lowerings are, within the cap below.
- `stages/`: the cache data of each `VkPipeline` of a kernel past the stage bound, one file per
  pipeline, which the consumer's helper compiled (Kernels past the stage bound, below). At most
  256 MiB and 32 files, apart from the blobs and the lowerings' cap.

**The pipeline cache.** The reaper makes an empty cache first when it starts, which a creation that
comes before it waits for (microseconds). Then, off every guest thread, it reads the blob, makes a
cache from it and makes that the cache creations take; the empty one is destroyed once no creation
holds it. So no creation waits for the blob's read or the driver's parse of it, which for a blob
of 32 MiB take tens of milliseconds: a creation meanwhile goes through the empty cache
(`Metrics::creations_before_seed` counts them), and what those compiled is not in this run's blob: a
later run compiles those pipelines again, once. A blob that does not check, whose data does not
begin with the device's pipeline cache header (whole, the `pipelineCacheUUID` a new driver version
changes included), or that the driver refuses, is removed, and the empty cache stays. Once pipeline
creation has been quiet for 20 ms (no creation running and none ending) after a creation, the reaper
looks at the cache: it reads the cache's size (`vkGetPipelineCacheData` with no buffer), and where
it grew by more than an eighth since the last write, writes the cache's data over the blob: a
temporary file renamed over it, never appended to. So it writes at most once a quiet period, a run
that makes no new pipeline writes nothing, and a device's writes over its life come to a few times
the blob's final size. The look's reads run under a lock every creation holds shared, which the look
takes exclusive by a try, so none of them runs during a creation (which Vulkan allows, and during
which Intel's Windows driver 101.2145 for the UHD 630 has faulted reading the data): a look that
finds a creation begun since creation was quiet is skipped until creation is quiet again
(`Metrics::pipeline_cache_looks_skipped`), and a creation that begins during a look waits for that
look alone (`Metrics::pipeline_cache_lock_waits` counts those waits,
`pipeline_cache_lock_wait_nanos` their time and `pipeline_cache_lock_wait_max_nanos` the longest).
The device's drop writes the blob once more where the size changed at all. A crash between writes
loses only the pipelines made since the last; no creation waits for a write.

The cache holds at most 128 MiB when the reaper looks: one long recorded session's 448 pipelines
write 62.6 MB of data on KosmicKrisp (the Mac host's driver) and 1.5-11.1 MB on NVIDIA's, RADV's
and ANV's drivers (m3-max-67847, 67727, 67726, 67729). One past it is replaced by a fresh, empty
cache that creations take from then on, and is destroyed once no creation holds it, each
replacement counted in `Metrics::pipeline_caches_replaced`. The next write replaces the blob with
the fresh cache's data, so the blob holds at most 128 MiB too, with no second file and no
generations; a directory keeps at most four blobs, one per kind of device or driver version, so
the blobs take at most 512 MiB of disk. A write of data past 128 MiB, the teardown's of a cache
that grew past it after the reaper's last look, is skipped and counted in
`Metrics::pipeline_cache_writes_skipped`; the blob keeps what it held. Each write is the whole
blob: a cold run with the bound lifted on KosmicKrisp wrote its blob 9 times, 271.9 MB in all, for
a 62.6 MB blob (m3-max-68006). A full cache costs host memory from device start: on KosmicKrisp,
seeding from a 59.6 MiB blob raised the process's footprint by 119.3 MiB before any make, about
twice the blob (m3-max-67977), so a cache at the bound is expected to hold about 256 MiB there.
`Gauges::pipeline_cache_bytes` is the cache's size at the reaper's last look, and is zero once the
device is torn down. With no directory the cache is the same, made empty, and
nothing is written.

**The lowerings** live in two generations, `g<N>/` (young) and `g<N-1>/` (old), each holding at most
half of `DeviceOptions::pipeline_cache_cap` (128 MiB where it is `None`: half of it is about seven
times the 9.16 MB a long recorded session's 282 lowerings fill). A lookup tries young, then old, and
a hit in old moves the file into young by a rename. A new file goes into young, written by the
reaper off every guest thread: a state hands its lowering over and returns, and where it would take
the lowerings waiting for the reaper past 8 MiB it is dropped and counted
(`Metrics::cache_writes_dropped`), never waited for, and lowered again by a later device. A lowering
larger than 8 MiB by itself is queued where none waits, so the heaviest kernels are kept too. The write that takes young past half the cap starts
the next generation: young becomes old, and the old one is removed whole, by a thread the store
starts for it. So a file stays for at least half the cap, and at most the whole cap, of files
written or moved after its last use, until a device of a newer translator opens the directory
(below). Half the cap is what the store is sure to keep: a set of
functions used over and over that is larger than half the cap is partly removed at each new
generation and lowered again. On disk the lowerings take at most the cap, and while a removal runs,
also what is left of the generation it removes; the blobs are beside them. A file larger than half
the cap is not kept. `Metrics::cache_bytes_written` counts the bytes the device wrote there,
lowerings and blobs.

The bytes young holds are counted in `state`, a file of 40 bytes beside the generations, read and
written under its lock, so devices open on one directory, in one process or several, share its
files and one count; they should pass one cap, since each starts a generation at half its own.
After the count, `state` records the newest translator semantics and artifact format versions a
device opened the directory under. Every file's name holds the semantics version, and a lowering
decodes under its own format version alone, so a device under a newer translator reads no file
an older one wrote. So a device whose translator is newer than the recorded one, in either
version, retires both generations at its open: young moves two generations on, counted empty, the
store's thread removes the two it found, and `state` records the higher of each version. A device
at the recorded versions or older ones retires nothing and records nothing. So two builds of
different versions on one directory, at once or in turn, keep each other's files after the newer
one's first open. A `state` of 24 bytes, as an earlier version of the runtime wrote it, reads as
older than every translator: its first open retires both generations once. An earlier version
writes only those 24 bytes and leaves the recorded versions as they are.
Opening the directory reads `state` and checks for a generation a removal left unfinished, and for
the `pipelines/` directories under young and old in which an earlier version kept one file of
pipeline data per `VkPipeline`, which the store's thread then removes; it lists no directory and
keeps nothing per file in memory.

The costs do not grow with the number of files. Measured with 100,000 and 1,000,000 files of 100
bytes (release build, btrfs on Linux / NTFS on Windows): opening the directory 10-11 µs / 106-129
µs, with no resident growth on Linux; a hit in young 1.8-2.0 µs / 49-50 µs; a miss 3.6-4.4 µs /
30-35 µs; a hit in old, which renames the file into young, 15-21 µs / 244-249 µs; a write 15-17
µs / 253-330 µs. The store's own thread removes a dead generation of 49,800 / 499,800 files in
0.42 / 4.6 s on btrfs and 4.0 / 43 s on NTFS.

Each write is a temporary file renamed over its name, and each file checks itself (the
translator's self-checking encoding, the blob's checksum), so a file cut short, by a crash or a
power loss, is a miss and is removed: nothing unchecked reaches the driver. A failed read or
write, a full disk included, is a miss too, and never fails a call; in a directory the device can
read and not write, its hits stay where they are and it writes nothing. A `state` that is missing
or does not check is rebuilt from the generations' names, with young counted full.

### Kernels past the stage bound

A driver's compiler takes memory that grows faster than the stage, in the process that creates the
pipeline. The translator states each module's weight (`ModuleStats::entry_weight`: the entry's
instructions, each call counted as its callee's body and each subgroup instruction as 64), and the
stage bound is `COMPILED_STAGE_WEIGHT`, 39,014: the heaviest kernel measured to compile in its creating process
within 512 MiB of that process's resident growth on every driver measured. Made cold (no pipeline
cache, the driver's shader cache off) and alone in a process of its own, a real program's kernel
of that weight grew the process by 350 MiB in 6.8 s on ANV (Mesa 26.2.6, the Iris Xe of RPL-P;
job m3-max-66048), 148 MiB in 1.8 s on NVIDIA's driver (620.70.06, an RTX 3090; job
m3-max-66050) and 80 MiB in 0.55 s on RADV (Mesa 26.2.6, a Radeon 780M; job m3-max-66049). Above
it, on the same Iris Xe, an older build's kernels of entry weights 73,814, 75,886 and 99,109 took
peak resident sets of 1.76, 2.24 and 3.65 GiB in 45 s, 47 s and 132 s (job m3-max-26147). No
Vulkan query states how large a stage a device's compiler takes, so the bound is one for every
device; it rises when a heavier kernel is measured to compile within 512 MiB. A subgroup
instruction counts as 64 because a driver's compile grows with one many times faster than with any
other instruction: a kernel of entry weight 24,814 holding 3,772 subgroup instructions passed
6 GiB compiling on an RTX 3090 (NVIDIA 620.70.06; job m3-max-67936), where the kernel of 39,014
above grew the process by 148 MiB. A kernel past the bound is never compiled in the process that
runs the program: each of its `VkPipeline`s is compiled in a helper process the consumer starts,
caps and times, and is created here from that compile's cache data alone.

A kernel past the bound only because it emulates its SIMD groups in place of the device's widest
subgroup mode is planned again keeping that mode pinned: the re-plan takes the pinned module where
it weighs less than the emulated one by the stage weight; otherwise the emulated one goes on. The
one taken compiles in the process where it is under the bound, and in the helper where it is not.

A fill's memory follows the threadgroup size its pipeline is made at, as the workgroup size is part
of what the driver compiles. On the Iris Xe of RPL-P (ANV, Mesa 26.2.6) one emulated module of
weight 46,491 took 794 MiB in 18.5 s at 1 thread a threadgroup and 2,630 MiB in 57.7 s at 32, the
state's execution width; at 1,024, its `max_total_threads_per_threadgroup`, the helper passed the
consumer's 3,072 MiB cap after 46 s and the pipeline was refused (job m3-max-68473, twice of two).
So a kernel whose fill fits at its execution width can be refused at a larger threadgroup.

Of the private corpus's kernels, compiled for every committed profile with no device (job
m3-max-68532; 11,253 or 11,254 kernels weighed a profile, 11,097 on `runtime-floor`), 26 pass the
bound on some profile: 25 on each NVIDIA, AMD and Intel profile, 26 on `moltenvk-m3` and the three
lavapipe profiles, and 24 on `runtime-floor`. Twenty-four are twelve pairs: two libraries hold the
same twelve kernels, one key of each pair from each library, at equal weights but for one pair, 2
apart. Their weights on the Intel and lavapipe profiles, then on the NVIDIA and AMD profiles
(`runtime-floor` weighs each 8 to 17 above the first, `moltenvk-m3` 2 or 3 above the second):

| Kernels | Intel, lavapipe | NVIDIA, AMD |
|---|---|---|
| b89e2e87392d4527, 21ab571c48816f4a | 171,236 | 205,223 |
| c9ac4cd28a074782, 12f7547e1a367fca | 116,124 | 139,779 |
| ffe1c513f3097c07, 44de5d661ac9f084 | 115,445 | 139,234 |
| 63f3407fadae3c59, 06c69bd9d4785c5d | 114,806 | 138,519 |
| fa025740f639f290, b1bdbf2280554df2 | 112,021 | 137,254 |
| e01f66611db622cf, 1ea600db518b8efe | 109,954 | 135,729 |
| c21fc56a2e064db6, aec6a678f3c13513 | 61,753, 61,751 | 71,477, 71,475 |
| c789b188f267c141, 55aef611427f981e | 60,869 | 70,593 |
| d34b295f5b14bc68, a15721e0338a51bc | 59,259 | 71,970 |
| f10654d23bd63211, 081b131a2661e77a | 58,461 | 71,527 |
| a03b1caa0bf2c0ae, 5b09c39e51c47af9 | 46,491 | 56,163 |
| bdff31bae4f6e84a, 444fb5872c37b388 | 46,483 | 56,155 |

The other two pass it on fewer profiles: efdb5f0a79f197f9 weighs 39,347 on the Intel profiles,
40,872 on the lavapipe ones, 41,157 on the AMD ones and `nvidia-rtx50-linux` and
`nvidia-rtx5080-laptop`, 41,180 on `nvidia-pascal-linux` and 42,142 on `moltenvk-m3`, and is not
past the bound on `runtime-floor`; 69a35b6f29ddeaa0 weighs 42,445 on `moltenvk-m3` and the lavapipe
profiles and is past it on no other profile.

The heaviest of them, b89e2e87392d4527, and d34b295f5b14bc68 with its pair a15721e0338a51bc (the
pair's modules differ by 199 words at equal weights), were filled once each in the helper under the
consumer's 3,072 MiB cap, from an empty driver shader cache and an empty cache directory, at the
state's execution width, 32, and at 1,024, its `max_total_threads_per_threadgroup`. The peak is the
helper process's resident set. On the RTX 3090 and the RX 7900 XTX the module asked for is the
partial-threadgroup one with a required subgroup size of 32; on the Iris Xe it is the emulated one.

| Card | Kernel | Module words, weight | Width 32 | Width 1,024 |
|---|---|---|---|---|
| RTX 3090 (NVIDIA 620.70.06; job m3-max-68749) | b89e2e87392d4527 | 462,898, 213,304 | made: 1,350 MiB, 32.8 s | made: 1,991 MiB, 48.5 s |
| | d34b295f5b14bc68 | 157,833, 76,943 | made: 787 MiB, 13.5 s | made: 1,034 MiB, 22.9 s |
| | a15721e0338a51bc | 158,032, 76,943 | made: 788 MiB, 13.5 s | made: 1,033 MiB, 22.8 s |
| RX 7900 XTX (RADV, Mesa 26.2.6; job m3-max-68750) | b89e2e87392d4527 | 462,898, 213,304 | made: 1,192 MiB, 15.7 s | made: 1,401 MiB, 17.9 s |
| | d34b295f5b14bc68 | 157,833, 76,943 | made: 527 MiB, 4.3 s | made: 618 MiB, 5.2 s |
| | a15721e0338a51bc | 158,032, 76,943 | made: 530 MiB, 4.3 s | made: 621 MiB, 5.2 s |
| Iris Xe of RPL-P (ANV, Mesa 26.2.6; job m3-max-68748) | b89e2e87392d4527 | 906,327, 171,236 | refused: past the cap after 190.3 s, peak 3,129 MiB | not made: the helper ended on SIGSEGV after 48.8 s at 1,291 MiB |
| | d34b295f5b14bc68 | 307,371, 59,259 | refused: past the cap after 35.7 s, peak 3,112 MiB | refused: past the cap after 50.3 s |
| | a15721e0338a51bc | 307,570, 59,259 | refused: past the cap after 35.8 s, peak 3,098 MiB | refused: past the cap after 56.8 s, peak 3,104 MiB |

So on the RTX 3090 and the RX 7900 XTX these kernels are made at 1,024 threads a threadgroup within
the cap, and on the Iris Xe none of them is made at either width.

The Iris Xe's end of b89e2e87392d4527 at 1,024 is a fault in ANV's compiler: a read at a null
base plus 0x40 inside a recursive function of `libvulkan_intel.so`, under the helper's
`vkCreateComputePipelines` (Mesa 26.2.6), at 1,289 MiB after 45 to 47 s, read with a debugger in
two asks (job m3-max-68793). Job m3-max-68748 ended the same fill eight times, its first after
48.8 s and seven re-asks after 1.2 to 1.3 s each; the five ends read (both asks of m3-max-68793,
and the first and two re-asks of m3-max-68748 from their cores, job m3-max-68824) fault at
`libvulkan_intel.so + 0xc863` with the same stack: the re-asks read the driver's own cache
(inferred: same fault offset, 1.2 s against 46 s; not shown by a file listing). Such an end is
kept as `StageBreach::Ended` (below).

Render stages have a bound of their own, 99,109, as no helper compiles a render stage: a render
pipeline with a stage past it is refused (Known gaps of render pipeline states), and one at or
under it compiles in the process that runs the program with no cap, past the kernels' bound too.
The two numbers measure two things: where a kernel leaves the process, and where a render stage is
refused because nothing else can compile it. They are one number again when a helper compiles
render stages. `Metrics::render_makes_past_kernel_bound` counts the render pipelines translated
with a stage past the kernels' bound, and `Metrics::render_weight_past_kernel_bound_max` holds the
heaviest such stage's weight.

- The make of such a `VkPipeline` (at the state's creation, a `prepare_dispatch` call or a
  dispatch) reads the pipeline's entry under `stages/` in the cache directory. With data, it
  makes a `VkPipelineCache` of that entry alone and a shader module of the kernel's words,
  creates the pipeline through them with
  `VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT`, and destroys both: the device's own
  pipeline cache and its blob never hold the pipeline, and the kernel holds no shader module,
  only its words, of which each make, two at once included, makes its own. Vulkan lets a module
  go while pipelines made of it are in use. A driver may keep a copy of a module's code for the
  module's life: destroying the module of a kernel of 7,966,928 bytes of words, its pipeline
  kept, gave back 7.6 MiB of heap on an RTX 3090 (NVIDIA 620.70.06; job m3-max-67862).
  `Metrics::pipelines_from_stage_fills` counts these creates, which `pipelines_created` counts
  too.
- With no entry, the make is refused with `Error::StageUnfilled`, carrying a `StageRequest`. Two
  requests of one pipeline compare equal and hash alike (by the pipeline's stage key, the name of
  its entry), so a consumer joins the makes, the `prepare_dispatch` calls and the dispatches that
  meet one pipeline into one helper run. The consumer starts its helper process, whose main calls
  `serve_stage` with an instance made as its own process's and the `DeviceOptions` the device was
  opened with, and hands the helper's two pipes to `Device::fill_stage` on a thread of its own.
  That call writes the request, reads the helper's cache data and keeps it as the entry
  (`Metrics::stage_fills`), and answers the helper's own times, from its host clock
  (`StageSpans`: its device's creation and its one `vkCreateComputePipelines` call); the make
  then succeeds when asked again. A request holds the module's
  words specialized for the pipeline (its specialization constants fixed at their values, as
  every compute pipeline's module is), the pipeline's entry point and pinned subgroup size, and the device's
  `deviceUUID`, `driverUUID`, pipeline cache header and enables, so the helper opens the same
  device with the same features; a helper whose device differs is answered `StageFill::NoDevice`
  or `StageFill::Enables`, and one of another build of the runtime `StageFill::Version`.
- rt starts, caps, times and kills no process, and asks no helper again. A consumer that ends its
  helper for its memory cap or its time limit keeps that with `Device::record_stage_breach`
  (`StageBreach::Memory`, `StageBreach::Time`), and one whose helper ended before its reply
  (`StageFill::Ended`) with no end of the consumer's, as a helper whose driver faults in the
  compile does, keeps that as `StageBreach::Ended`: the first such end is the answer. A compile
  the driver failed is kept as `StageBreach::Compile` by `fill_stage`. A breach is an entry of the same directory, and refuses the pipeline's makes
  (`Gap::StageSize`, `StageGap::Breach`) with no helper asked again. So does cache data the driver
  answers compile-required for (`StageBreach::CompileRequired`): it is not filled a second time.
  The stage key holds the driver's pipeline cache header, so a new driver version asks again.
- A memory breach also holds the memory ceiling the consumer gives its helper,
  `DeviceOptions::stage_memory_ceiling` (bytes, 0 by default), as the device that kept it was
  given. That ceiling is a fact about the consumer's helper: rt only stores it with a memory breach
  and compares it at a make, and takes no other path by it, on any device. On a device given a
  higher ceiling than the breach's, its entry is a miss and is removed, and the make is refused
  with `Error::StageUnfilled` again: the consumer runs its helper under the higher ceiling, and the
  fill, or a breach kept under that ceiling, takes the entry's place. At the breach's ceiling or
  below, it refuses as before. A memory breach entry that holds the breach's byte alone, with no
  ceiling after it, counts as kept under 0. Time, compile and compile-required breaches hold no
  ceiling.
- An ended breach holds the version of rt's helper code it was kept under (4 bytes), which rises
  with each change to what `serve_stage` does. Under a later version its entry is a miss and is
  removed, and the make is refused with `Error::StageUnfilled` again, so a fix of rt's helper asks
  the helper again; a driver's fix asks again through the driver's header in the key.
- The entries are files of the cache directory only: nothing of an entry stays in memory past
  the call that reads or writes it. Each holds its key again under a magic, its length and its
  checksum; one that does not check, names another key or holds data that does not begin with the
  device's header is a miss and is removed. A hit marks its file used. `stages/` holds at most
  256 MiB and 32 entries; each write removes the entries past that, the least recently used first.
- A device with no cache directory, and one whose driver writes no pipeline cache header
  (lavapipe in Mesa 26.2.4), keeps no entry: the make is refused (`Gap::StageSize`,
  `StageGap::NoCacheDirectory` or `StageGap::NoPipelineCache`).

Measured with one kernel of entry weight 842,031, whose request is 15,969,892 bytes, each step in a
process of its own with the drivers' own disk caches off. The helper's fill (its instance, its
device and the compile) took 53.6 s on a Radeon 780M (RADV, Mesa 26.2.6) and 129.3 s on an RTX
2080 (NVIDIA 620.70.06), and kept an entry of 5,866,121 and 7,594,869 bytes. A make in a new
process, the device's blobs removed, created the pipeline from the entry alone in 3 ms and 33 ms;
with the entry removed too, the make was refused unfilled. A create with the flag through an empty
cache is answered compile-required on both.

### Visible-function tables

- `ComputePipelineState::function_handle` is `functionHandleWithFunction:`: the handle of a
  function the descriptor's `linked` lists in `functions` or `binary_functions`, the
  translator's `FunctionHandle`, numbered as `LinkedFunctions::function_handle` numbers them;
  `None`, Metal's nil, for a private function or one the state does not link.
- `ComputePipelineState::new_visible_function_table` is `newVisibleFunctionTableWithDescriptor:`
  (`VisibleFunctionTableDescriptor::function_count`): a table of null entries, its words laid out
  as the translator states (`bind::function_table_header`, then one word per entry at
  `bind::function_table_entry_offset`) in a Shared block of its own, which the runtime writes
  through its mapping. `VisibleFunctionTable::set_function` and `set_functions`
  (`setFunction:atIndex:`, `setFunctions:withRange:`) write the handles' words there, so a
  dispatch calls the entries as they stand when it runs: an entry set after the dispatch is
  encoded and before its command buffer is committed is the one it calls. An index at or past the
  table's size is `Refusal::FunctionTableIndex`, and nothing is written. `gpu_resource_id`
  (`gpuResourceID`) is the device address of the words.
- A call through an entry calls the function with that word among the linked functions of the
  dispatch's state; a null entry, a table of no entries and an unbound slot are what the kernel's
  `is_null_visible_function_table`, `size` and null-pointer checks read. A table or handle another
  state answered calls whichever function has the word in the dispatch's state, or nothing, and
  yields zero.
- The device test `function_tables` dispatches through tables of two orders, one with a null
  entry, one of no entries, an unbound slot and an entry set after encoding, and compares every
  word the kernel writes.

### Known gaps of pipeline states

- A driver's create of a state's main pipeline can fail after the state has answered, where Metal
  compiles the state before it answers and a program never meets a state it was given that cannot
  run. Vulkan lets `vkCreateComputePipelines` and `vkCreateGraphicsPipelines` fail with
  `VK_ERROR_OUT_OF_HOST_MEMORY`, `VK_ERROR_OUT_OF_DEVICE_MEMORY`, `VK_ERROR_UNKNOWN` or
  `VK_ERROR_VALIDATION_FAILED` (and `VK_ERROR_INVALID_SHADER_NV`, of an extension the runtime does
  not enable). `finish_pipeline` returns the failure as `Error::Vulkan`, keeps nothing and counts
  it (`Metrics::failed_finishes`); the first dispatch or draw that needs the pipeline makes it
  again, and where that make fails too, that dispatch or draw returns the `Error::Vulkan`. A
  kernel past the stage bound makes its main pipeline at the state's creation (Kernels past the
  stage bound, above): a make the stage bound refuses fails the creation, and any other failure
  there is the first dispatch's in the same way, uncounted by `failed_finishes`.
- One very large app's pipelines can pass the pipeline cache's 128 MiB on KosmicKrisp (the Mac
  host's driver), whose blob takes about 140 KB a pipeline against 3-25 KB on NVIDIA's, RADV's and
  ANV's drivers for the same pipelines, so about 960 pipelines fill it there. A guest boot's set
  (32.3 MB of blob after one boot on that host) and one long recorded session's 448 pipelines
  (62.6 MB) fit. Past the bound the cache is replaced by an empty one, so a launch after the
  replacement makes those pipelines again. On an M3 Max (KosmicKrisp in Mesa 26.3.1), with the
  bound at 32 MiB, re-making those 448 pipelines in the same process took 1,669-2,370 ms after a
  replacement, with 45-60 cache hits (m3-max-67741, m3-max-67789), against 80.4-81.0 ms with no
  replacement, with 448 hits (m3-max-67847); a launch there made its set in 0.06 s from a full blob,
  and in 0.83-2.03 s around a replacement. The bound's host memory on that host: a full cache's
  seed holds about twice its data from device start (above).
- Ray tracing's bound acceleration structure is not built (ray tracing is out of scope,
  docs/REFUSALS.md's `RayTracing` row). The runtime makes no acceleration structure, no encoder
  binds one (`setAccelerationStructure:atBufferIndex:`), and nothing supplies the instance count
  of a bound instance structure that the translator's intersection reads. Device creation enables
  neither `rayQuery` nor `accelerationStructure`, so a function that takes an acceleration
  structure is refused when its state is made, as the translator refuses it on a device without
  them (`Unsupported{RayTracing}` naming the two features, the translator's
  `tests::engine_ray_tracing`).
- A sample at a texel offset the shader computes, on a device without `maintenance8` (Chosen by a
  query): the sample branches over the offset's values, one branch per combination of the 16
  values of each lane the shader computes (256 for a 2D offset), each with its value as
  `ConstOffset`, where a
  device with the feature takes the offset as the sample's own `Offset` operand. The branch costs
  compile time: on NVIDIA's driver one fragment function's 255 branches took 193 s to create on an
  RTX 3090, against 80 ms with the operand. Intel's Windows driver for the UHD 630 (101.2145)
  does not list `VK_KHR_maintenance8`, so it takes the branch; its compile time there is not
  measured. Test: `tests::filter_nest`.
- A narrow integer's high bits (an 8- or 16-bit value held in a 32-bit word) are cleared by
  `OpBitwiseAnd` by the width's mask on every device. NVIDIA's driver 615.71.09 on Ampere GA10x
  dropped that mask after a signed clamp to exactly the narrow range, so such a kernel answered the
  clamp's sign-extended words there; 620.70.06 answers right, and no card the runtime is run on has
  that driver on that architecture. The other form, `OpBitFieldUExtract(x, 0, N)`, loses the
  device on Intel's Windows driver 101.2145 (UHD 630) at a desktop's first pipelines, so it is not
  used. Test: `tests::narrow`'s `every_device_zero_extends_a_narrow_clamped_sum`.
- Some forms are not lowered by the translator's identity domains, which every state is translated
  under, among them forms argument buffers take under `MTLArgumentBuffersTier1` or
  `MTLArgumentBuffersTier2`, which `Device::argument_buffers_support` answers: a state over a
  function that holds one is `Error::Translation` of `Unsupported{IdentityDomains}`, not
  implemented, whose symbol names the first form it holds. By form:
  - an object or mesh function ("a function of a stage other than a kernel, vertex or fragment
    function"): no test yet;
  - a value from the encoder that the function reads and the identity domains do not source ("an
    encoder value the function reads as F", `F` the `PushField`: a texture buffer's bind offset,
    `SlotBindOffset`; the draw's `InstanceCount` or `BaseVertexInstance`; a stage-in stride,
    `StageInStride`; and the others the translator pushes beside the identity fields): no test
    yet;
  - a buffer argument not reached through its device address ("a buffer argument not reached
    through its device address"): no test yet;
  - a texture argument of a type the identity arrays hold no element of, or whose bound texture's
    state beyond its resource ID the function reads ("texture argument S: a type the identity
    arrays hold no element of, or state of the bound texture beyond its resource ID"), a cube
    read by face through `access::read` or `access::sample` among them: the golden table
    (`tests/golden_hashes.tsv`) holds the refusal's class for `engine/texture_ops`' `cube_faces`,
    and no test asserts the text yet. A storage cube written or read by face is lowered: the
    translator test `a_storage_cube_written_by_face_is_the_2d_array_element_of_its_faces`;
  - a texture argument sampled through a sampler the identity sampling does not take ("a texture
    argument sampled through a sampler the identity sampling does not take"), and a sampler
    argument whose minimum level or emulated filter the function specializes ("a sampler argument
    whose minimum level or emulated filter the function specializes"): no test yet;
  - other state of a bound texture or sampler beyond its resource ID ("state of a bound texture
    or sampler beyond its resource ID (F)", `F` the `RuntimeFact`), among them an argument
    buffer's texture that the function only queries (`HeapNumericType`):
    `tests::engine_heap_groups`'s
    `a_query_only_argument_buffer_texture_is_refused_naming_its_numeric_type` asserts the text
    for a 2D texture and a cube array, and the golden table holds the refusal's class for
    `engine/tail_cases`' `tail_heap_cube_queries`;
  - program-scope constant data past 64 KiB read through memory ("constant data of N bytes read
    through memory"): `tests::engine_identity`'s
    `a_constant_blob_past_the_modules_bound_is_refused_naming_its_size` asserts the text;
  - a resource bound through a descriptor ("a resource bound through a descriptor"): no test yet;
  - with no symbol, a storage texture access without a format that the device's
    `shaderStorageImageReadWithoutFormat` or `shaderStorageImageWriteWithoutFormat` does not take
    and the formats' own bits do not grant: no test yet;
  - a bicubic sample of a 1D texture ("a bicubic filter of a 1D texture"): the translator test
    `a_bicubic_kernel_reads_each_slots_format_treatment_from_its_pushed_field` asserts the text;
  - an argument buffer's texture of a type the identity arrays hold no element of (a texture
    buffer's parts, a cube read as an array, level views, a two-plane, extended-range, RGBX or
    `ulong` form, or a storage texture with a format), one read through another numeric type than its
    own or as the raw texels of an extended-range member, a read of one that decodes an
    extended-range member, and a handle that reads its table entry beyond its element: no test
    yet;
  - a sequentially consistent fence (`atomic_thread_fence` of `memory_order_seq_cst` beyond the
    thread, or a `seq_cst` barrier whose memory scope is the device), which the translator
    orders through a word the identity domains do not reach ("a sequentially consistent
    fence"): the translator test
    `a_seq_cst_fence_is_refused_naming_its_form_under_the_identity_domains` asserts the text.

  The plays' counts of the pipeline states they skipped read 0, a refusal under the identity
  domains among them, on five recorded streams: three played to their end (1,047, 1,100 and
  1,129 frames) and two to their line-width stop (frames 379 and 394), in jobs m3-max-51725,
  -51726, -51757, -51772, -51839 and -51840. No tool translates every pipeline state of the
  recorded streams under the identity domains without a GPU, so the count is of these streams
  to these frames.
- A 64-bit texture atomic (`atomic_min` or `atomic_max` on a `ulong` texture), which Metal's
  feature set tables give the Apple8 family and later ("GPU devices in the Apple8 family support
  64-bit atomic minimum and maximum using ulong, on both buffers and textures, only on macOS"),
  is refused on every device: a state over a function that calls one is `Error::Translation` of
  `Unsupported{ImageAtomics}`, not implemented. Its one exact form is a 64-bit image atomic
  (`shaderImageInt64Atomics`), which device creation does not enable; no form over 32-bit
  atomics is exact.
- A derivative in a kernel (`dfdx`, `dfdy`, `fwidth` and their `fast_` spellings) is refused on
  every device: a state over a function that calls one is `Error::Translation` of
  `Unsupported{ComputeDerivatives}`, needing `computeDerivativeGroupLinear`. The translator lowers
  it on a device that enables that feature (`VK_KHR_compute_shader_derivatives`, whose linear
  quads are Metal's four consecutive threads), and device creation does not enable it (Not
  enabled, below), so the lowering does not run under rt.
- Every `VkPipeline` is made through the device's `VkPipelineCache` (The cache directory, above),
  which is empty on a host's first run. Cold, the driver's time for a recorded session's render
  pipelines made through it, against the same pipelines made past any cache (states-only plays):
  5.8% slower on an RTX 3090 (NVIDIA 620.70.06, job m3-max-58076) and 6.8% slower on a second RTX
  3090 with a second session (m3-max-58098); 9.8% faster on an Iris Xe (ANV, m3-max-58178), whose
  driver shares compiled stages between the pipelines made through one cache; level on a Radeon
  780M (RADV, m3-max-58159), 0.4% slower, within the plays' spread. Warm, in a second process on
  the directory a first one filled, with the driver's own disk cache on: the second session's 274
  render pipelines took 21.4 against 21.0 ms through the cache against past it on the second RTX
  3090 (m3-max-58285), 5.5 against 21.5 ms on
  the Iris Xe and 3.6 against 8.1 ms on the Radeon 780M; with the driver's disk cache off, 5.6
  against 4,280.6 ms on the Iris Xe and 3.7 against 845.8 ms on the Radeon 780M.
- What a state's creation is given is made there, as Metal compiles a state when it is made:
  every lowering, a kernel's variants among them. Each `VkPipeline` whose key the descriptor
  holds (a kernel's, Compute pipeline states above; a render content's triangle class, and its
  line or point class where `inputPrimitiveTopology` names that class) is made after the answer,
  by the consumer's `finish_pipeline` call, or by the first dispatch or draw that needs it before
  that call has made it. What only a dispatch or a draw gives waits for it: a kernel's
  `VkPipeline` at a threadgroup size, threadgroup memory lengths and an explicit imageblock's size
  is made at the consumer's `prepare_dispatch` call or at the dispatch, a render content's line or point class, unless its
  descriptor names that class, at its first draw of it (one `VkPipeline` for every class needs
  `dynamicPrimitiveTopologyUnrestricted`, which the floor does not require, and no second form is
  chosen by it), and its `VkPipeline` under the depth clip mode `Clamp` or the triangle fill mode
  `Lines` at its first draw under it (Known gaps of
  render pipeline states). Each such make at a dispatch or a draw is a first-use wait, and on a
  cold cache it compiles in that call, where Metal changes these with no compile: on a cold play,
  `Metrics::first_use_waits` against `Metrics::made_ahead_variants` reads what the consumer's
  pre-pass left to the encoding threads.
- Each compute `VkPipeline` is created from a shader module of its own, made of the kernel's
  words with every specialization constant the fixed constant of its value
  (`metal2vulkan::pipeline::specialize`) and destroyed once the create returns: the driver is
  handed no `VkSpecializationInfo`, every array length is a plain constant, and the workgroup's
  size is the entry point's `LocalSize` literals, so no driver can size a workgroup or an array
  from a constant's default. A render pipeline's stages are made the same way once, with the
  content: each stage's module holds its sample count and amplification count fixed, and no
  graphics `VkPipeline` takes a specialization either. Tests: `tests::deep_threadgroups`, and
  every draw test for the render stages.
- A kernel's module keeps 64 `VkPipeline`s of the threadgroup sizes, threadgroup memory lengths,
  imageblock sizes and tiles its dispatches give: the first 8 it made, and past them the 56
  dispatched last. A make past them evicts the least recently dispatched ones no command buffer
  not yet retired dispatched, and none where work holds every one, so a command buffer that
  dispatches more keys keeps each variant until it retires, and the kernel's next make after that
  evicts the idle ones past 64. A variant evicted and dispatched again is made again, through the
  device's pipeline cache, a first-use wait where Metal makes nothing. One small kernel's variant,
  made, evicted and made again (release build, no validation layer, the drivers' disk caches off,
  job m3-max-70404): 313 µs and then 104 µs on lavapipe (llvmpipe, LLVM 23.1.4), 2,727 µs and then
  5.1 µs on an RTX 3090.
  `Gauges::kernel_variants` counts the live variants. A dispatch's threadgroup memory length is
  rounded up to a multiple of 16 bytes, the only lengths Metal takes, so lengths within one
  multiple run one variant; each threadgroup size is a variant of its own, as a `VkPipeline`
  fixes its workgroup size (30,343 sizes at 1,024 threads), and so is each imageblock size.
  Tests: `tests::kernel_variants`.
- A compute kernel with an explicit imageblock runs at the size the compute encoder's
  `set_imageblock_size` (`setImageblockWidth:height:`) set last, 1 by 1 where it set none, as
  Metal's does. The size is a specialization of the kernel's `VkPipeline`, as a threadgroup memory
  length is, so each size a kernel runs at is one more `VkPipeline`, made at the consumer's
  `prepare_dispatch` call or at the dispatch, a first-use wait (`first_use_variant_waits`), and
  `finish_pipeline` makes none for such a kernel. Metal changes the size with no compile: on an
  Apple M3 Max, a dispatch's first use of a size cost what its second did (medians of 25 runs at 4
  by 4, 16 by 16 and 32 by 32: 202, 228 and 312 µs against 205, 207 and 228 µs, a dispatch at a
  size already used 222 µs; job m3-max-69983). Metal's rule for the size, powers of two of at most
  1024 cells, holds a kernel to 66 sizes. The Metal Shading Language specification
  supports imageblocks on macOS for Apple silicon only (sections 5.6 and 6.14), so a device that
  answers `supportsImageBlocks` NO gives a guest no imageblock to use. Over 34 recorded guest
  streams, one of them an arm64 macOS 27 boot, two such kernels are made, no dispatch binds one, and
  no stream calls `setImageblockWidth:height:`. One recording, the M3's, makes 10 states of
  explicit-imageblock kernels and dispatches none (jobs m3-max-58471, m3-max-58477, m3-max-58478
  and m3-max-58483, and m3-max-57983 for the count). The conformance case
  `compute-imageblock-sizes` holds one encoder's dispatches before any size and at two sizes
  equal to Metal's.
- An image-only kernel's first dispatch over a grid more than one deep at a tile no state of its
  content has run makes that tile's `VkPipeline` on the calling thread, which waits for it
  (`Metrics::first_use_waits`), through the device's pipeline cache. In each of the two recorded
  streams measured, on a device that holds 1,024 threads along `z`, that made 50 pipelines, all in
  one frame: the frame where the stream's image-only dispatches over such grids start (frame 224
  of one stream and 312 of the other). The 50th, of tile 1x1x256, runs as one row, so a device
  that holds 64 makes the same 50.
- A kernel's variants are lowered with its state, whether or not a dispatch needs them: an
  observing kernel's partial-threadgroups variant, which its dispatches run, and the
  emulated-SIMD-group variant of a kernel whose
  tier runs only whole SIMD groups, which only a threadgroup that is not whole subgroups runs. So
  a state's creation lowers up to three functions where Metal compiles one, each through the
  cache directory. For the observing kernel the device tests run, a grid template over
  threadgroup memory, one lowering took 0.61 to 0.63 ms on an RX 7900 XTX in a release build; a
  larger kernel's lowering takes longer.
- On Intel's Windows driver (101.2145) for the UHD 630, the driver's compile of a tiling kernel's
  `VkPipeline`, the state's main one, took medians of 6,680 to 6,853 µs for the harness's
  `create_state` kernel, in three processes of 40 compiles each with no driver cache hit. The
  workgroup accounts for 1.1 to 1.3 ms of it: the runtime makes the pipeline at 256 threads, the
  tile it runs every grid of such a kernel in, against a pipeline at the dispatch's threadgroup,
  here one thread. The decode of the pushed tile accounts for 0.4 to 0.55 ms, and the module's
  device addresses, against storage-buffer descriptors, for 0.37 to 0.47 ms. The runtime's pipeline
  layout and its pipeline cache cost nothing measurable there. The cost is paid once per driver
  cache: the driver keeps compiled pipelines across processes, and the same modules compiled again
  in a new process took 45 to 48 µs.
- A kernel that reads anything the encoder does not supply yet, an intersection-function table, a
  visible-function table an argument buffer holds, or a sampler or texture state the translator
  would take from the bound object rather than from its resource ID, is refused
  (`Gap::KernelInput`, whose `KernelInputKind` names what the runtime found: `Slot` for an
  intersection-function table). So is a state for indirect command buffers
  (`Gap::IndirectCommandBuffers`, `IndirectCommandsKind::ComputeCommands`).
- A visible-function table's words take a Shared block of their own, 64 KiB at the least, for a
  table of any size.
- A state whose kernel the consumer compiled (`StageSource::Compiled`) holds no linked functions
  and answers no handle (`function_handle` is `None` for every function), so its tables hold only
  null entries.
- Three forms of a kernel's `[[stage_in]]` are refused, each as `Gap::StageIn` naming its
  `StageInKind`: `set_stage_in_region_with_indirect_buffer`
  (`setStageInRegionWithIndirectBuffer:indirectBufferOffset:`, a region the GPU reads),
  `IndirectRegion`; an attribute the kernel reads through a thread-position layout of
  `MTLBufferLayoutStrideDynamic`, whose stride the `setBuffer:offset:attributeStride:atIndex:`
  family would bind (the compute encoder has none of it), `DynamicStride`, at the state's
  creation; and one it reads through an indexed step at a step rate other than 1, whose Metal
  answer is not measured, `IndexedStepRate`, at the state's creation.
  A kernel that encodes indexed triangle draws into an indirect command buffer is built (Indirect
  command buffers, below); one that encodes anything else the translator refuses
  (`Unsupported{GpuEncodedIndirectCommands, NotImplemented}`).
- A threadgroup that is not whole SIMD groups of a kernel whose SIMD groups run one per subgroup
  of 32 is refused (`Gap::PartialSimdgroups`) where the state has no emulated-SIMD-group module:
  for a kernel the consumer compiled (`StageSource::Compiled`), and where the translator refuses
  the emulated variant. Any other failure of a variant's lowering, this one's or the
  partial-threadgroups one's, is reported as itself (`Error::Translation`, or `Gap::KernelInput`
  for an input the compute encoder does not supply). The emulated module's scratch counts in the
  dispatch's threadgroup memory (`Limit::ThreadgroupMemory`).
- `dispatch_threads` with a grid that is not a whole number of threadgroups on some axis is
  refused (`Gap::PartialThreadgroups`) at a threadgroup that is not whole subgroups of a kernel
  that runs only whole subgroups, and for a kernel that observes its threadgroup
  partition and whose partial-threadgroups variant the translator refuses. It refuses the
  variant of a kernel whose SIMD groups it emulates: one with SIMD-group operations where the
  device's compute subgroups are neither 32 lanes nor 64 or 128 split into 32-lane clusters. It
  refuses the variant of a kernel whose barriers in a
  loop need a vote word that does not fit beside its threadgroup memory in
  `maxComputeSharedMemorySize`, which `Device::max_threadgroup_memory_length` leaves room for: a
  kernel that takes more threadgroup memory than that answer, as one built for a Mac's 32,768
  bytes does on a device of 32,768, keeps this refusal for its ragged grids. The answer does not
  count an emulated SIMD group's scratch, whose bytes vary by kernel, so a kernel at the answer
  whose SIMD groups are emulated can still pass the device's limit (`Limit::ThreadgroupMemory`).
- A kernel that observes its threadgroup partition makes a `VkPipeline` per threadgroup size it
  is dispatched at: the state's main pipeline at the descriptor's
  `requiredThreadsPerThreadgroup` where the kernel takes no threadgroup memory length from the
  encoder, and otherwise at a `prepare_dispatch` call or its first dispatch at that size.
- A kernel that observes its threadgroup partition runs in workgroups of its own threadgroup
  size, never in tiles. On Intel's Windows driver (101.2145) for the UHD 630, a kernel that
  touches a buffer per thread through its device address runs slower in narrow square workgroups
  than in tiles or in wide workgroups. Over a 2048x2048 grid of `RGBA16Float` texels, a kernel
  reading one texel and storing 16 bytes per thread took 7,775 µs in 16x16 workgroups, against
  3,721 µs in tiles and 3,735 µs in 128x2 workgroups; one reading 16 bytes and writing one texel
  took 6,554 µs, against 4,634 and 4,735 µs. Bound through storage-buffer descriptors, the two
  took 3,585 and 4,605 µs in 16x16 workgroups; the runtime binds buffers only by address. In the
  two recorded streams measured, kernels that observe their partition, store to device memory and
  run threadgroups 32 threads wide or narrower ran 2.0% and 3.4% of the threads dispatched.
- A kernel that does not observe its threadgroup partition and is not image-only runs a grid more
  than one deep in the tiles its regions carry, as it does a grid one deep. On an RX 550 (RADV,
  Mesa 26.2.6), a kernel storing 4 bytes per thread through its device address took 2.6, 9.4 and
  30.2 µs over grids of 4x16x312, 24x24x252 and 256x256x8 in tiles, and 2.2, 8.6 and 25.0 µs on a
  pipeline that took the tile as a specialization constant, which the runtime does not make. On
  Intel's Windows driver (101.2145) for the UHD 630, the same kernel over the 24x24x252 grid took
  93.6 and 79.7 µs in tiles, the medians of two runs (within one run, one form's rounds ranged up
  to 2.8 times apart); in the first, a pipeline at its 8x8x4 threadgroup with the buffer through a
  storage-buffer descriptor, which the runtime does not make, took 0.97 of the tiles' time.
- Such a kernel runs a grid one high and one deep in tiles too: its position comes from the pushed
  tile in all three dimensions and is tested against the grid in all three. On an RX 550 (RADV,
  Mesa 26.2.6), a kernel of 2^20 threads in a line, each reading 16 bytes 64 times from a 64 KiB
  `constant` table and storing 4 bytes, took 2,380.8 µs in tiles in the first of three runs. In
  workgroups of the kernel's 256x1 threadgroup it took 0.90 of its time in tiles, through
  storage-buffer descriptors 0.89, and with both 0.89; the runtime makes neither. Its reads take
  their 64-bit addresses from two adds with a carry each, as the RX 550's loads through an address
  need, where reads through a descriptor take a 32-bit offset. An Iris Xe (Mesa 26.2.6) pays
  neither: there the kernel took the same GPU time in tiles, in 256x1 workgroups and through
  descriptors. In the two recorded
  streams measured, kernels that do not observe their partition ran 0.00% and 0.59% of their
  threads over grids one high and one deep.
- A dispatch of a size other than the descriptor's `requiredThreadsPerThreadgroup`, which Metal's
  validation stops, runs at the size it gives.
- NVIDIA 615.71.09 on GA10x gives sign-extended bits for a mask over a signed clamp, to exactly N
  bits' range, of a sum or difference of two values sign-extended from N bits; not reproduced on
  620.70.06.
- A kernel past the stage bound is made only from its helper's compile (Kernels past the stage
  bound, above). On a device with no cache directory, or whose driver writes no pipeline cache
  header (lavapipe in Mesa 26.2.4), on a driver that answers compile-required for its helper's
  cache data, and where the helper passed the consumer's cap or time, it is refused
  (`Gap::StageSize`) where Metal makes it. A dispatch of it waits for the consumer's helper, where
  Metal compiles in the program's process.
- On a host whose Vulkan driver is ANV, Mesa's Intel driver, a guest's kernels of entry weight
  186,065 to 424,132 are refused (`Gap::StageSize`) where Metal makes them: their helper's compile
  passes the consumer's 3 GiB cap. On the Iris Xe of RPL-P under the packaged Mesa 26.2.6, the cold
  launch of a 3D modelling program made 2 of its 8 heavy kernels (entry weights 121,523 and
  128,372, helper peaks 2,620 and 2,763 MiB) and refused 6 (186,065 to 424,132, helper peaks 3,091
  to 3,334 MiB; job m3-max-65807). Each refused make returns its error to the guest's Metal call:
  the synchronous form answers nil with the error, the asynchronous forms call their handler with
  it. A program that waits for a kernel with no exit for a failed make waits forever: that program
  showed its first sample line 143 s after launch and never advanced, with no error shown, while
  the guest stayed usable. Under an 8 GiB cap, the helpers of the two heaviest (355,578 and
  424,132) passed 8 GiB 34 s and 51 s into their compiles (job m3-max-65470). A Mesa
  26.2.6 build that adds count prints and a heap census let the lighter one run on: it was past
  18 GiB resident and unfinished at 840 s (job m3-max-65511). The stock driver's time past 8 GiB is
  not measured. Both costs are in Mesa's Intel compiler and grow with the product of two of the
  kernel's sizes (source lines of Mesa 26.3.1). Its liveness analysis gives every block six bitsets
  over all virtual registers (`src/intel/compiler/brw/brw_analysis_liveness.cpp`, lines 240-247):
  4,972 MiB for the heavier kernel's first dead-code pass and 9,549 MiB for its next. Its register
  allocator's interference graph holds a bit for each pair of nodes and doubles its node capacity
  once spill nodes pass it (`src/util/register_allocate.c`, lines 504-508 and 633-639): 3,124 MiB
  for the lighter kernel's 228,938 nodes, then 12,498 MiB.
- A loop asks the driver nothing about unrolling unless its AIR does (`#pragma unroll`,
  `#pragma nounroll`) or the translator wrote the loop itself (`DontUnroll`), so the driver's
  choice decides, and it can differ from Metal's compiler's. RADV (Mesa 26.2.6) unrolls a
  128-iteration loop of one 16-byte device read fully. On an RX 7900 XTX, with neighbouring
  threads 976 bytes apart in a 64 KiB table, that loop took 207 to 209 µs unrolled against 105 to
  113 µs rolled; over 16 KiB and 256 KiB tables the unrolled loop was 7 to 17% faster.
- A `half` `fma` on a device that computes `half` in 16 bits under the 16-bit float controls
  (`shaderFloat16`, with 16-bit `RoundingModeRTE`, `DenormPreserve` and
  `SignedZeroInfNanPreserve`) is the device's 16-bit `Fma` (`PrecisionNote::NativeFma`). Vulkan
  gives `Fma` the precision of a multiply followed by an add, each rounded, where Metal's rounds
  once. RADV (Mesa 26.2.6) on an RX 7900 XTX does not round it once: over a chain of 1,024
  dependent `half` `fma`s, 1,622,016 lanes ended up to 9 `half` ulp from Metal's. NVIDIA's driver
  on an RTX 3090 and Intel's Windows driver on a UHD 630 matched Metal in every lane of the same
  chains. Elsewhere a `half` `fma` is the device's `float` `Fma` on the operands, rounded to
  `half`, within one `half` ulp.
- A compute or render pipeline state answers no `gpuResourceID`
  (`MTLComputePipelineState.gpuResourceID`, `MTLRenderPipelineState.gpuResourceID`): neither type
  has the call, so the runtime gives no word for a program to write into an argument buffer's
  `compute_pipeline_state` or `render_pipeline_state` member, and the consumer answers the
  property itself. A function that reads such a member is refused at its state's making
  (`Error::Translation`): the only use of the member the translator admits is an indirect
  command's pipeline setter, which is `Unsupported{GpuEncodedIndirectCommands, NotImplemented}`
  (`tests::engine_indirect_commands`).
- A kernel keeps the reconvergence mode (`MaximallyReconvergesKHR`, where the device lists
  `shaderMaximalReconvergence`) when one of its SIMD-group operations stands in a block the
  translator does not prove uniform for the threadgroup, as in a loop whose trip count is a count
  read from a buffer at run time less a per-thread offset. AMD's open Linux driver (AMDVLK
  2025.Q2.1, LLPC) on a Radeon 780M runs such kernels slower with the mode than the same modules
  without it. Mesa's RADV (26.2.6) on the same chip runs the session below in the same compute time
  with the mode on such kernels alone as with it on every kernel (21.05 s against 21.03 s and
  21.06 s to frame 349, m3-max-72436). In one recorded session on that card, five kernels of that
  shape keep the mode. Under AMDVLK the play's compute GPU time to frame 349 was 59.22 s
  (m3-max-72427), 5.8% above the 55.98 s of the same play with the mode removed from every kernel, a
  measurement and not a form the runtime has (m3-max-72248). In one span of that play, where the
  five took 4.40 s timed per dispatch with the same five modules (m3-max-72362), compute was 11.67 s
  against 9.03 s without the mode. Test: the translator's
  `tests::engine_reconvergence_together::a_shuffle_outside_uniform_control_flow_keeps_the_mode`.
- AMD's open Linux driver (AMDVLK 2025.Q2.1, LLPC) on a Radeon 780M may compile a kernel whose
  thread-local array of 256 or 512 bytes is indexed at run time with that array in scratch memory,
  where Mesa's RADV (26.2.6) on the same chip keeps it in registers. In one recorded session on
  that card, one kernel's module came to 148 VGPRs, 528 bytes of scratch and 512 scratch
  instructions under AMDVLK, against 224 VGPRs and no scratch under RADV (m3-max-72775). Per
  dispatch, the kernel's states that AMDVLK compiled with scratch ran 2 to 9 times as long as
  under RADV, and those it compiled without scratch ran at 0.86 to 0.93 times. The array's
  storage class does not change it: the same modules with every such array a `Function` variable
  instead of a `Private` one compiled to the same registers and scratch (m3-max-72798). Neither
  the array's size nor the module's register count separates the modules AMDVLK spills from
  those it does not: a module holding the same 512-byte array, all eight of its access chains
  indexed at run time, compiled to 174 VGPRs and no scratch.

## Render pipeline states

- `Device::new_render_pipeline_state(&RenderPipelineDescriptor, sampler_ids)` is
  `newRenderPipelineStateWithDescriptor:error:` without reflection. The translator lowers the
  vertex and fragment functions on the calling thread, which waits for them, within its 20-second
  budget per lowering, for what device creation enabled and under its identity domains, as for a
  compute state. A refusal of the translator is `Error::Translation`, whose `functions` names the
  vertex function and then the fragment function, as for a compute state.
- rt builds no pipeline reflection and acts on no `MTLPipelineOption`; its consumer builds
  reflection from the library's AIR, as for a compute state.
- `RenderPipelineState::texture_accesses(stage)` answers each texture slot the vertex
  (`Stage::Vertex`) or fragment (`Stage::Fragment`) function uses, by index, with the access a
  draw of the state marks in its hazards for the texture bound there, as a compute state's
  `texture_accesses` answers a kernel's; nothing for another stage.
- `RenderPipelineState::imageblock_sample_length` answers Metal's `imageblockSampleLength`: the
  color attachments' texels as Metal packs them, each set format's in index order at its
  component's alignment, depth and stencil excluded (measured on an M3 Max and an M4:
  `[R8Unorm, R32Float]` is 8). No explicit imageblock adds to it, as a state whose fragment
  function reaches one is not made. `imageblock_memory_length_for_dimensions` answers
  `imageblockMemoryLengthForDimensions:`: that length times the width, the height and the raster
  sample count, none rounded, the depth ignored. A color format with no measured place in Metal's
  record is `Gap::ImageblockLength`. The M3 measurement the rule rests on is the conformance
  suite's recordings `pipeline-facts-render-plain`, `pipeline-facts-render-two-attachments`,
  `pipeline-facts-render-multisampled`, `pipeline-facts-render-multisampled-direct`,
  `pipeline-facts-render-narrow-attachments` and `pipeline-facts-render-depth-stencil` (an Apple
  M3 Max, macOS 27.0.1 build 26A434, three runs each): `RGBA8Unorm` answers 4 and 128 at
  `[8, 4, 3]`; `RGBA8Unorm` then `RGBA16Float` 12 and 384; the same at four samples 12 and 1536,
  with indirect-command support or without; `R8Unorm` then `RG16Float` 6 and 192 (`RG16Float`
  at byte 2, its component's alignment, where packing tight would answer 5 and aligning to 4
  would answer 8); `RGBA8Unorm` then `RGBA16Float` with a `Depth32Float_Stencil8` depth and
  stencil attachment 12 and 384, as without it. `support_indirect_command_buffers` is the
  descriptor's: a state made with it pulls every vertex attribute its vertex function reads, and
  its commands run in an execution of an indirect command buffer (Indirect command buffers,
  below). The runtime makes no tile render state, so it has no tile getters
  (`maxTotalThreadsPerThreadgroup`, a tile threadgroup's, `threadgroupSizeMatchesTileSize`,
  `requiredThreadsPerTileThreadgroup`).
- A state numbers its functions' samplers as a compute state numbers its kernel's:
  `sampler_ids` takes the vertex function's, then the fragment function's, each in Metal's order,
  and answers the index the consumer gives each. Each constexpr sampler a function samples
  through takes its device samplers at its index before the call returns, and each draw pushes
  the index.
- What a state compiles is its content, never the state: its descriptor, with its functions'
  constant values and linked functions, and every field (the translator's
  `Device::queried_render_descriptor_key`). Every state of one content shares one compiled content: its
  translation, its shader modules and their `VkPipeline`s, and its precision notes. Render
  contents and compute kernels share the device's one map of the last 1,024 contents used, so a
  state of a content it holds costs the key, which reads no function, and a lookup.
  `Gauges::render_contents` counts the compiled render contents alive, and
  `Gauges::kernel_entries` the map's entries, compute and render together. Two descriptors that
  differ only in a field their compile does not read are two contents, each compiled.
- A render pipeline's two functions push their values (each buffer's address, each constexpr
  sampler's index) in one range of the device's push window (`W` bytes, 256 or 128, as for a
  kernel above), dense by link: the vertex function's
  from 0 (from 24 for a post-tessellation function, whose control stage pushes the factors'
  address at 0), ending by `W - 8` (248 or 120), the fragment function's from the vertex function's end, each value at
  the next offset aligned to it (a framebuffer fetch's element index, four bytes, before the
  eight-byte values, where the four bytes below one are free), so only the values the pair uses
  take space. Compute and render pipelines share the device's one pipeline layout and its one
  push range. A function whose values do not fit its share pushes the address of a parameter
  block first and reads the rest from the block, as a kernel does: a draw writes each such block
  into the parameter ring where it fills its pushed values anew (the state, a binding or a pushed
  value changed since the encoder's last draw), and `ring_writes` counts it. The values only a
  checked access reads (a checked slot's bounds and the zero range's address) come last, so they
  are the first to move to the block.
- A `VkPipeline` bakes what Vulkan 1.3 cannot leave dynamic: the content, the topology class
  (point, line or triangle) of the draws it serves, and the depth clip and triangle fill modes. It
  takes as dynamic state the viewports and scissors with their count, cull mode, front face, primitive topology within its class, primitive restart, depth test, write and
  compare, stencil test, operations, masks and reference, depth bias and blend constants, and a
  state without rasterization its discard, which keeps its attachment formats and that state in
  the pipeline. Vertex buffer strides are the descriptor's, so the content's, and baked. It
  renders dynamically into its attachments' formats, with no render pass.
- Every `[[stage_in]]` attribute of a state made without `supportIndirectCommandBuffers` is
  fetched by fixed-function vertex input, the form native Vulkan uses, with no second form (a state
  made with it fetches every attribute in the shader: Indirect command buffers, below): the vertex descriptor's formats, offsets, strides and steps
  are the `VkPipeline`'s vertex input state. Of Metal's vertex descriptor:
  - formats: each maps to the Vulkan format of its layout and conversion, taken where the
    device's format query reports `VERTEX_BUFFER` for it: an 8- or 16-bit integer format read
    into a `float` attribute in its `USCALED` or `SSCALED` form, an unnormalized signed-integer
    format into an attribute declared `int`, `UChar4Normalized_BGRA` as `B8G8R8A8_UNORM`, the
    10:10:10:2 formats as `A2B10G10R10`, `FloatRG11B10` as `B10G11R11_UFLOAT_PACK32`. The
    validation crate's device profiles of RADV, ANV, NVIDIA and lavapipe report the bit for
    every one of them but `FloatRGB9E5`, which only lavapipe's reports;
  - steps: per vertex, and per instance at rate 1, map to the vertex and instance input rates,
    and per instance at a rate above 1 to the instance input rate with the rate as its divisor
    where the device has the step rates (Floor, Step rates): Vulkan fetches element
    `firstInstance + (instanceIndex - firstInstance) / divisor`, counting from the draw's base
    instance as Metal does (the conformance case `vertex-step-rate-instances`); the constant step is a vertex-rate binding of stride 0, on every device and with no divisor:
    Vulkan reads a vertex-rate attribute at `vertexIndex * stride` past the bound offset, so every
    vertex and instance reads the layout's first element, and no rule but the portability
    subset's bounds a binding's stride from below;
  - offsets and strides: the descriptor's, baked, each a multiple of the format's component size
    and within the device's offset and stride limits; on a portability-subset device the constant
    step's stride of 0 is below `minVertexInputBindingStrideAlignment` and refused
    (`VertexInputLayout`).
- What has no fixed-function equivalent on the floor is refused at creation by the translator,
  `Error::Translation` naming what it needs, and never pulled into the shader:
  - a per-instance step rate above 1 on a device without the step rates, or past its
    `maxVertexAttribDivisor` (`VertexStepRate`, naming `vertexAttributeInstanceRateDivisor` or
    the limit): Vulkan divides the instance index only with that feature and a rate within that
    limit;
  - a format the device's query does not report for `VERTEX_BUFFER` (`Format`, with the format),
    and `FloatRGB9E5` on every device (`Format`, not implemented: of those profiles only
    lavapipe's reports the format, and its fetch answers 0 for the fourth component, where Metal
    and Vulkan give 1);
  - an offset or stride that is not a multiple of the format's component size, which Metal takes,
    and one past `maxVertexInputAttributeOffset` or `maxVertexInputBindingStride`, and more
    attributes or layouts than `maxVertexInputAttributes` or `maxVertexInputBindings`
    (`VertexInputLayout`, with the limit where one is passed);
  - the per-patch and per-patch-control-point steps for a vertex function that is not
    post-tessellation (`Tessellation`).
- The triangle class's `VkPipeline` is a main pipeline of the state, made through the device's
  pipeline cache by its `finish_pipeline` call or by the first draw that needs it before then
  (Compute pipeline states, `finish_pipeline`), and so is the line or the point class's where the
  descriptor's `inputPrimitiveTopology` names `MTLPrimitiveTopologyClassLine` or
  `MTLPrimitiveTopologyClassPoint`; otherwise each is
  made by the content's first draw of lines or of points, which waits for it
  (`Metrics::first_use_waits`, `first_use_line_waits`, `first_use_point_waits`). Of the two replay
  streams' 537 render contents drawn, 527 draw only triangles and 2 lines and triangles; a content
  drawn only as lines (8 of them) makes a triangle `VkPipeline` it never uses. A draw under the
  depth clip mode `Clamp` or the triangle fill mode `Lines` takes its class's `VkPipeline` for that
  state, which the content's first such draw makes and waits for (`first_use_raster_waits`). The
  fill mode is a triangle's alone, in Metal and in Vulkan, so a draw of lines or points takes the
  clip mode alone, and a triangle's edges under `Lines` are drawn by Bresenham's rule as lines are.
- A point is as wide as the `PointSize` the vertex module writes: the function's `[[point_size]]`,
  stored as Metal's largest point, 511 pixels, where it is past it, which the device draws through
  `largePoints` (Floor) and clamps to its `pointSizeRange` (`PrecisionNote::DevicePointSize` where
  the range ends below 511), or 1.0 for a function without one, and 1.0 too where the function's
  size is not greater than 0: a zero, a negative or a NaN, which Vulkan's `PointSize` does not take
  (Metal's answer for them is undocumented and unmeasured). Metal draws a point of a function
  without `[[point_size]]` only under a state whose descriptor names the point class, 1 pixel wide;
  under any other class it rasterizes nothing and runs the vertex function all the same (Metal
  probes, an M4), so such a draw records its points behind an empty scissor: the vertex function
  runs, its stores land, and no fragment runs. A state whose vertex function writes `[[point_size]]`
  under the line or the triangle class is refused at its make (`Refusal::PointSizeTopology`), as
  Metal's creation refuses it. A point near the viewport's edge or the clip volume is clipped or
  culled by the device's own rule (its `pointClippingBehavior` and the driver's handling of the
  viewport), as Metal states no rule for it (docs/API.md, section 10).
- A line is drawn by Bresenham's rule, which is Metal's: the pixels whose diamond the line exits
  (`bresenhamLines`, Floor). The line class's `VkPipeline` chains
  `VK_LINE_RASTERIZATION_MODE_BRESENHAM_EXT`, except for a state that enables alpha to coverage or
  alpha to one, which Vulkan's Bresenham mode does not take: its lines are drawn by the device's
  default rule, the rectangle of the line's width on a device whose `strictLines` is set. Either
  way a line's pixels are the device's: Vulkan's Bresenham mode lets a fragment lie up to one
  pixel in x or y from the diamond-exit rule's, and a draw of lines, or of triangles under the
  `Lines` fill mode, carries `PrecisionNote::DeviceLineRasterization`
  (`RenderCommandEncoder::LINE_NOTE`). Where a line only touches a pixel's diamond at its vertex,
  which pixel it covers is the device's own (docs/API.md, section 10).
- A post-tessellation vertex function's state draws its patches through the device's tessellator
  (`tessellationShader`, Floor). Its one `VkPipeline` is the patch class's, its main pipeline:
  the translator's synthesized vertex stage (each patch one vertex of a `PATCH_LIST` of one control
  point, at its position), its synthesized control stage, the function as the evaluation stage,
  and the fragment function. The control stage reads each patch's factors from the encoder's
  tessellation factor buffer through their device address, which a draw pushes beside the
  functions' values; it binds no descriptor, and nothing about the factors happens on the CPU at a
  draw. The evaluation stage reads its control points' and per-patch attributes from their buffers
  through the slots' pushed addresses, at the descriptor's layout strides, which the state fixes.
  The partition modes, windings and maximum factors are the control stage's own, each as Metal
  states it (the translator's control stage), with its precision notes.
- Refused at creation for a post-tessellation vertex function, naming what is not built: a
  per-instance factor step function (`PerInstance`, `PerPatchAndPerInstance`) and an enabled
  factor scale (`Gap::RenderPipeline` of `Tessellation`), whose instance stride and scale no draw
  pushes; a control-point index type (`Error::Translation`, `Tessellation` not implemented: the
  address form reads no control-point index buffer); a layout whose stride is set per bind
  (`DrawInput`). No replay stream's tessellation state uses any of them.

### Known gaps of render pipeline states

- A render pipeline whose vertex or fragment stage weighs more than 99,109, the render stages'
  bound (Kernels past the stage bound), is refused (`Gap::StageSize`, `StageGap::RenderStage`)
  where Metal makes it: no helper compiles a render stage.
- A render pipeline whose heaviest stage weighs more than the kernels' bound
  (`COMPILED_STAGE_WEIGHT`, 39,014) and at most 99,109 compiles in the process that runs the
  program, with no cap, where a kernel that heavy compiles only in the consumer's capped helper
  (`Metrics::render_makes_past_kernel_bound` counts them). No render stage that heavy is measured.
  The heaviest render stage measured of a real program, of entry weight 35,063 beside a vertex
  stage of 431 (39,424 as lavapipe's profile weighs it), made cold and alone in a process of its
  own, grew the process by 316 MiB in 3.8 s on ANV (Mesa 26.2.6, the Iris Xe of RPL-P; job
  m3-max-66048), 180 MiB in 2.5 s on NVIDIA's driver (620.70.06, an RTX 3090; job m3-max-66050)
  and 58 MiB in 0.6 s on RADV (Mesa 26.2.6, a Radeon 780M; job m3-max-66049).
- Refused at creation by `Gap::RenderPipeline`, naming its `RenderPipelineKind`: depth and
  stencil attachment formats that differ, as of separate depth and stencil textures
  (`DepthStencilFormats`: Vulkan's pipeline takes one format for both,
  `VUID-VkGraphicsPipelineCreateInfo-renderPass-06589`), a `rasterSampleCount` that is not a
  power of two from 1 to 64 (`SampleCount`: Vulkan's rasterization takes one of its sample
  count bits), vertex amplification, a post-tessellation vertex function's per-instance factor
  step function or enabled factor scale (`Tessellation`;
  a descriptor's tessellation state with any other vertex function is ignored, as Metal ignores it),
  stages the translator synthesizes beside the functions' and a tessellation's, a specialization
  other than the sample and
  amplification counts, dynamic state beyond Vulkan 1.3's (a vertex buffer layout whose stride is
  set per bind among it), a feature or extension device creation does not enable, and a function
  that reads what a draw does not supply (`DrawInput`: a function table, a pointer or handle translation table,
  a parameter block past the push range, a push field or bind-time fact the encoder does not fill,
  an unnormalized constexpr sampler). Of the replay streams' contents that translate, every one
  reads only its slots' addresses, resource IDs and constexpr sampler indices. The
  `completionHandler:` forms are not built.
- A fragment function with imageblock data, an input or an output declared `[[imageblock_data]]`,
  is refused at creation on every device, an `Error::Translation` naming `Imageblocks`
  (`NotImplemented`): the translator has no form for it. The runtime has refused such a function on
  every device all along, worded otherwise before: the only form the translator had bound a storage
  image through a descriptor, which the identity domains the runtime compiles under hold none of.
  The states-only plays of the 35 replay streams on lavapipe refuse none of their render or compute
  states (jobs m3-max-63105 to 63108, and m3-max-63132 and m3-max-63156 for the two streams that
  ran past those jobs' 300-second limit).
- On a device that does not list `VK_EXT_shader_stencil_export`, a fragment function's `[[stencil]]`
  output is drawn by bits (`RuntimeFact::StencilBitDraws`). Each draw through its state is drawn
  once under the stencil reference 0, then once for each bit of the low 8 of the write masks of
  the faces whose operations replace, each keeping only the fragments whose output holds that bit
  (`PushField::StencilBit`), under the reference 255, that bit's write mask, `Replace` kept and
  every other operation `Keep`, and no depth write; where the state writes the depth under a
  compare other than `Always` and `Never`, the first draw writes none and a last draw writes it.
  That is up to 10 draws where Metal draws once (`Metrics::stencil_bit_draws`), and a draw whose
  state replaces no bit, or tests no stencil, is drawn once. Two fragments of one draw on one
  sample store the OR of their outputs where Metal stores the later one, and under a depth test
  that writes, each fragment's stencil operation is chosen against the depth from before the
  draw: the static note `PrecisionNote::StencilOutputOverlap` on the fragment stage of every such
  function. Refused at creation there, an `Error::Translation` naming `StencilExport`: a function
  that also writes a colour or memory, one whose returns cannot discard, a mesh pipeline, and a
  state made for indirect command buffers. Refused at a draw, `Gap::Draw` of
  `DrawKind::StencilOutput`: a stencil compare other than `Always` and `Never` on a face, whose
  answer the output decides, and, where the state replaces a bit, a draw inside a `Counting`
  visibility result's segment, whose count every one of the draws would add to. A colour beside
  the output can be built under this form when a guest needs it: every draw through the one
  pipeline, the colour written the same each time, with blending off on every attachment, the
  logic operation off and no colour input. Metal's feature set tables give stencil
  feedback to the Apple5 family and later and to Metal 3 and Metal 4. NVIDIA's 615 and 620 drivers
  on the RTX 2080, RTX 3090 and RTX 5080 Laptop list none (`vulkaninfo`, jobs m3-max-48727, 48729
  and 48731); RADV on the RX 7900 XTX, the 780Ms and the RX 550, ANV on the Iris Xe, lavapipe,
  KosmicKrisp and the UHD 630's Windows driver list it (48727 to 48731, 48736, 48737 and 61508).
  A WebGL 2 page reaches it: a browser's GL-on-Metal layer copies stencil for `blitFramebuffer`
  with `STENCIL_BUFFER_BIT` by a draw whose fragment function writes `[[stencil]]`, on a Metal
  device that it does not take for an Intel or NVIDIA GPU.
- On a device without `shaderQuadControl` (Quad control, under Floor), a fragment function whose
  group operations read, combine or count other lanes (every SIMD-group and quad group function, and
  the SIMD-group matrix functions) is refused at creation, an `Error::Translation` naming
  `SubgroupOperations` and `shaderQuadControl`: Metal runs the helpers of a partly covered quad, and
  they take part in those operations, which Vulkan gives only under `RequireFullQuadsKHR`. Of the
  venues run, KosmicKrisp on the M3 Max and the UHD 630's Windows driver lack the feature
  (`vulkaninfo`, jobs m3-max-61179 and 61190). No replay stream has one: no fragment module of the
  26 streams' libraries calls a SIMD-group, quad or SIMD-group matrix function (each library whose
  bytes name one disassembled with `llvm-dis`, jobs m3-max-52530, 52531, 52532, 52557 and 52604).
  Their libraries that call quad functions, 2 to 5 a stream, call them from no fragment function.
- On a device without `shaderResourceMinLod` (Min-LOD clamps, under Floor), a function that samples
  with a `min_lod_clamp` other than 0 is refused at creation, an `Error::Translation` naming
  `MinLodClamp` and `shaderResourceMinLod`. Of the venues dumped, the UHD 630's Windows driver
  alone does not report it (`vulkaninfo`, job m3-max-48736); RADV on the RX 7900 XTX, the 780Ms
  and the RX 550, NVIDIA's driver on the RTX 2080, RTX 3090 and RTX 5080 Laptop, ANV on the Iris
  Xe, lavapipe and KosmicKrisp on the M3 Max report it (m3-max-48727 to 48731, 48737 and 61508).
- On a device without `alphaToOne` (Alpha to one, under Floor), a render pipeline whose descriptor
  sets `alphaToOneEnabled` is refused at creation, an `Error::Translation` naming `AlphaToOne` and
  `alphaToOne`: the translator has no other form, and Metal has no query for it. A Radeon under
  AMD's own drivers reports no `alphaToOne`: AMD's open Linux driver (AMDVLK 2025.Q2.1) on the
  Radeon 780M (`vulkaninfo`, job m3-max-70426), and AMD's Windows driver by a user's report. No replay stream sets it: 0 of the 58,976 render states of the nine streams'
  states-only plays on lavapipe (job m3-max-66153).
- On a device without `fragmentShaderBarycentric` (Barycentrics, under Floor), a fragment function
  that reads `[[barycentric_coord]]` is refused at creation, an `Error::Translation` naming
  `Barycentrics` and `fragmentShaderBarycentric`. Of the venues run, lavapipe, ANV on the
  Iris Xe, RADV on the RX 550, the UHD 630's Windows driver and KosmicKrisp on the M3 Max list no
  `VK_KHR_fragment_shader_barycentric` (`vulkaninfo`, jobs m3-max-61287, 61291, 61292 and 61296);
  NVIDIA's driver on the RTX 3090, RTX 2080 and RTX 5080 Laptop, and RADV on the RX 7900 XTX and
  the 780M, list it and report the feature (m3-max-61287 to 61290 and 61296). No replay stream has
  one: the states-only plays of the 36 streams on lavapipe refuse none of their render states
  (m3-max-60927 to 60930 and 60945).
- On a device without `fragmentShaderPixelInterlock` (Pixel interlock, under Floor), a fragment
  function's raster order group on memory it reads and writes is refused at creation, an
  `Error::Translation` naming `RasterOrderGroups` and `fragmentShaderPixelInterlock`. Of the venues
  run, KosmicKrisp on the M3 Max and RADV on the RX 550 list neither the extension nor the feature
  (vulkaninfo, jobs m3-max-61291 and m3-max-61296).
- A fragment function that reads `[[sample_mask, post_depth_coverage]]`, the coverage after the
  early depth and stencil tests, is refused at creation on every device, an `Error::Translation`
  naming `PostDepthCoverage` (`NotImplemented`). Vulkan's one form for it is
  `VK_EXT_post_depth_coverage`, which the floor does not enable, and the translator keeps no
  lowering for it. The runtime has refused it on every device all along, worded before as the
  extension the device lacked. If a guest meets it, it is built on that extension, with its tests.
- The depth clip mode `Clamp` and the triangle fill mode `Lines`, which Vulkan 1.3 bakes into a
  `VkPipeline`, are each a `VkPipeline` of the draw's class that a content's first draw under them
  makes, which Metal does not: that draw waits for it (`Metrics::first_use_raster_waits`).
- Under the depth clip mode `Clamp`, the depth a fragment function writes is the device's depth
  clamp's, clamped to the viewport's range. Metal stores a negative NaN there as 0, below a range
  above 0; the translator's `LoweringInputs::depth_clamp_range` variant, which stores it as Metal
  does, is not drawn.
- On a driver that keeps the topology class in a pipeline (dynamicPrimitiveTopologyUnrestricted
  false; RADV), a render pipeline state's first draw in a second topology class compiles a second
  pipeline in full when the driver's cache is cold: 225 ms for the heaviest pipeline measured on a
  780M, 0.9 to 1.5 ms for the others, nothing once the driver's cache holds it. A driver without
  the restriction serves it from its cache (0.1 ms on ANV). Linking from pipeline libraries would
  avoid it; the floor does not have them.
- Each render content is one whole `VkPipeline`, compiled with no shared library parts, so a cold
  make pays the content's whole compile. A recorded session's heaviest content, a fragment
  function of about 30,800 SPIR-V instructions, takes 1,408 ms of driver time cold on an RTX 3090
  (NVIDIA 620.70.06, job m3-max-58251), and 1,335 ms on a second RTX 3090 (m3-max-58134). Made
  without depth bias's enable and the feedback loop's enable as dynamic state, its time changes by
  under 1% (m3-max-58251). It is paid once per cold cache: warm, with the driver's disk
  cache on, that session's 383 render pipelines take 34.5 ms of driver time through rt's cache on
  the second RTX 3090 (m3-max-58285). A warm make in a new process reads each render content's
  record (The cache directory), so it decodes and specializes no function: on lavapipe, a recorded
  session's warm render state creation is 31 to 44% faster with the records than without them
  (m3-max-58317).
- Mesh render pipelines are not built. No call makes a state of a `MTLMeshRenderPipelineDescriptor`
  (`newRenderPipelineStateWithMeshDescriptor:`), and the render encoder has no object or mesh
  binding and no mesh draw (`drawMeshThreadgroups:`, `drawMeshThreads:` and the indirect form), so
  no type here names them and the consumer refuses them itself. `new_render_pipeline_state` takes
  a vertex function: an object or mesh function in its place is the translator's refusal
  (`Error::Translation`). The translator's mesh lowering (docs/API.md, "Mesh pipelines") is only
  compiled and validated: it has never been compared with Metal on a GPU, and rt makes no
  `VkPipeline` of it. No replay stream or live session has used one; harvested guest
  functions declare them, and validation's committed object and mesh cases qualify on Metal alone
  (its `mesh_render_pipeline` requirement).
- On a Vulkan driver built on Metal (Mesa's KosmicKrisp on the Mac), a fragment function that
  declares `[[early_fragment_tests]]` loses every texture write in a pipeline of more than one
  sample: it runs, and its buffer writes land, but none of its texture writes does. This is
  Metal's own defect, and the driver passes it on: it turns the module's `EarlyFragmentTests`
  into Metal's `[[early_fragment_tests]]`. Metal itself, on an Apple M3 Max under macOS 27.0.1,
  writes none of 512 texels at 2 or at 4 samples, while the same function without the attribute
  writes every one. Vulkan requires those writes, and no Vulkan query tells such a driver apart,
  so rt emits the mode on every device. The device test `tests::lost_writes` accepts exactly this
  outcome there and prints it.

## Render encoders

- `CommandBuffer::render_command_encoder(&RenderPassDescriptor)` is
  `renderCommandEncoderWithDescriptor:`. While it is open it holds the command buffer's recording,
  as a compute encoder does. It begins one dynamic rendering (`vkCmdBeginRendering`, in the floor)
  into the pass's attachments, and `end_encoding` ends it (`vkCmdEndRendering`): one Vulkan
  rendering a Metal render encoder, with no render pass object or framebuffer.
- `RenderPassDescriptor` is `MTLRenderPassDescriptor`: eight color attachments
  (`MAX_COLOR_ATTACHMENTS`), a depth and a stencil attachment, `renderTargetWidth`,
  `renderTargetHeight`, `defaultRasterSampleCount` and `renderTargetArrayLength`. Each attachment is one level of its texture
  and, from its `slice`, the pass's layers (below): a 2D texture, a 2D
  array, a cube or a cube array, or a 2D multisample texture or array, of `RenderTarget` usage;
  or a colour attachment's `depthPlane` of a level of a 3D texture of that usage, in a pass of one
  layer, through a 2D view of that plane of its image (Vulkan's
  `VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT`, whose barriers name the image's layers as
  `VK_REMAINING_ARRAY_LAYERS` from 0 and move every plane of the level, and whose load, store and
  clear act on the plane alone: the level's other planes keep their texels);
  every attachment of a pass has one sample count, and a draw's pipeline state that count as its
  `rasterSampleCount` (`DrawKind::AttachmentFormats` otherwise). The device's counts are
  `Device::supports_texture_sample_count`'s. A multisample attachment is stored as its samples:
  a shader reads them through `texture2d_ms` or `depth2d_ms`. The texture keeps an
  image view of that level and those layers with the identity swizzle (a 2D view of one layer, a
  2D array view of more), made at the first pass that renders into it and freed with the texture
  (`Gauges::attachment_views`).
- A pass with no attachment is `renderTargetWidth` by `renderTargetHeight` pixels at
  `defaultRasterSampleCount` samples: one dynamic rendering of no attachment, whose draws'
  pipeline states have that `rasterSampleCount` and no attachment format, and write their answers
  to memory. Vulkan takes the samples from the pipeline (`framebufferNoAttachmentsSampleCounts`
  the translator's query for its state). A side of 0 or a count of 0, Metal's default, is
  `Refusal::NoAttachmentPass`, as plain Metal makes no encoder of either (an Apple M3 Max, job
  m3-max-62174: nil for a side of 0, the process ended for a count of 0); a side past the device's
  `maxFramebufferWidth` or `maxFramebufferHeight`, `Gap::RenderPass` of `NoAttachment` (every
  venue dumped reports 16384 or more, Metal's largest side).
- A pass of `renderTargetArrayLength` N of 2 or more is one rendering of `layerCount` N: every
  attachment and every resolve texture renders N slices from its own `slice` (a cube's faces are
  its slices), and the rendering's load, store and resolve act on all N, so a layer no draw
  reaches carries the pass's clear. At 0 and 1 the pass renders the attachment's slice alone, as
  Metal's does. A vertex function's `[[render_target_array_index]]` has one form, in every pass:
  it stores `Layer`, its index where that is below the pass's length and 0 at or past it, the
  length each draw pushes (`renderTargetArrayLength`, at least 1); so at 0 and 1 every index
  renders the attachment's slice, as Metal's does. A fragment function reads the index as the
  vertex function wrote it, through a flat varying, so it reads 3 where the vertex function wrote
  3 into a pass of one layer. Device creation enables `shaderOutputLayer` (Floor). Metal's rules
  (the conformance case `layered-render-target-array-length`, recorded on an Apple M3 Max, macOS
  27.0.1): at 0 and 1 the index is ignored, at 2 or more index `i` renders slice `slice + i`, and
  the load and store cover every layer.
- Load actions are one to one: `Load`, `Clear` and `DontCare` are Vulkan's load operations of
  those names. `Store`, `DontCare` and `Unknown` store: a pass may end its rendering before the
  encoder ends and begin it again, loading what it stored (below), so a `DontCare` store keeps the
  texels, one of the outcomes Metal allows. Vulkan fixes the store as the rendering begins, so an
  `Unknown` store action stores, and the encoder's `set_color_store_action`,
  `set_depth_store_action` and `set_stencil_store_action` change nothing the device does: on a
  tile-based device this stores a target Metal would have discarded. A clear color is a `float` in a
  float, normalized or sRGB channel, which the device converts to the format by its own rules
  (Vulkan clamps a normalized value and asks for rounding to nearest; Metal rounds to nearest
  even); an integer channel's is truncated toward zero and saturated to the channel's width, NaN
  0, as Metal's. A rotated format's clear color, and `A8Unorm`'s held in `R8_UNORM`, go through
  the format's rotation. A clear stencil keeps its low 8 bits.
- A store action that resolves (`MultisampleResolve`, `StoreAndMultisampleResolve`, and `Unknown`
  where the attachment names a resolve texture) resolves by Vulkan's own resolve, in the same
  rendering: the attachment's `VkRenderingAttachmentInfo` names a view of the resolve texture's
  `resolve_level` and `resolve_slice` and a resolve mode, so no pass, copy or barrier of its own
  is added; a resolve texture whose home is shader-read-only moves to the attachment layout and
  back with the pass, as an attachment does. Both actions also store the samples, as `DontCare`
  keeps them: a rendering the encoder ends early and begins again resolves again as it ends, and
  its last resolve holds the whole pass. A color attachment resolves by `AVERAGE`, the mode
  Vulkan requires of a float, normalized or sRGB format, under the static note
  `PrecisionNote::DeviceRoundedResolve` (`RenderPassDescriptor::precision_note`) on every device:
  the device rounds the average, and decodes and encodes an sRGB channel, by its own rules. A
  depth attachment resolves by its `depth_resolve_filter`: `Sample0` by
  `VK_RESOLVE_MODE_SAMPLE_ZERO_BIT`, which Vulkan requires, and `Min` and `Max` by their modes
  where `supportedDepthResolveModes` has them; a stencil attachment by `Sample0` alone. A depth
  and a stencil attachment that both resolve resolve into one view, paired as the device's
  `independentResolve` and `independentResolveNone` allow. The resolve texture takes the
  attachment's pixel format and one sample, and its level the attachment level's width and
  height. Metal resolves into a texture of any usage, and so does the runtime into a one-sample
  colour texture, 2D, a 2D array, a cube or a cube array, whose image is optimal: such a texture
  takes the attachment usage on every device where its format renders, whatever Metal's usage.
  So does the linear image of a texture in place over a buffer of the runtime's own memory, where
  the device's linear tiling renders its format and its image format query makes the image with
  the usage, which left its layout as it was on every venue counted. Where such a texture's linear
  image takes none, the texture stays in place and the resolve is a counted copy, a known gap: the
  pass resolves into the texture's resolve image, a one-sample optimal image of its format and
  size made at the first pass that resolves into it and freed with it (`Gauges::resolve_images`
  and `resolve_image_bytes`), and the pass's end copies the render area's texels into the buffer
  at the texture's pitch, leaving the bytes past each row's texels as they were. That is one copy
  per resolve, 1,440,000 bytes for a 600 x 600 `BGRA8Unorm_sRGB` pass, and none on the texture's
  later uses (`Metrics::resolve_copies` and `resolve_copy_bytes`, in `bytes_copied` too); Metal
  makes no such copy. A linear image takes no attachment usage where the device's format query
  offers no linear image of the format with it (measured on the RTX 3090 and on KosmicKrisp, for
  every format, where a pitch of the row rounded up to the alignment answer, 128 and 16, lies in
  place), and where its subresource reaches past its own rows or its buffer, as a resolve "may
  read and write back any memory locations within the image subresource bound for that
  attachment" (Vulkan's multisample resolve operations; measured on every venue counted where a
  buffer ends at the texture's last texel and its rows are padded). A resolve into a texture made
  without `RenderTarget` usage whose image has no attachment usage and that has no resolve image
  is a known gap (`RenderPassKind::ResolveUsage`): a depth or stencil texture, a colour texture of
  another type or of a format the device's optimal image does not render, a texture in place over
  a dma-buf's import, and a view of another pixel format of a texture in place over a buffer
  whose linear image takes no attachment usage (a view of its own format resolves as the texture
  does, through its resolve image; where the linear image takes the usage, a view resolves in
  place). Refused as
  Metal refuses it: a resolving action with no resolve texture (`Refusal::ResolveTexture`, also
  where an encoder sets it later) and an integer format's resolve (`Refusal::ResolveFormat`).
  `Metrics::resolves` counts the attachments passes resolved.
- A depth attachment is loaded, cleared and stored as a color attachment is, its clear depth a
  `float`: an optimal image in its real layout, of the Vulkan format the translator's format table
  chooses by the device's format query (`Depth32Float` is `D32_SFLOAT`, `Depth32Float_Stencil8`
  `D32_SFLOAT_S8_UINT`). Draws test and write it as their depth-stencil state says, through
  dynamic state: a pipeline bakes the depth format alone, so a change of depth-stencil state or
  depth bias makes no pipeline. A state that tests or writes depth carries the static note
  `PrecisionNote::DepthDenormals` (`DepthStencilState::precision_note`): every driver measured but
  MoltenVK on an Apple M4 compares a denormal depth otherwise than Metal, and RADV and ANV store
  one a GPU write gives as a zero. A negative or NaN depth, which Metal's test compares by its
  magnitude, takes none: no clear, viewport or upload gives one, and Vulkan leaves a depth past
  the viewport's range undefined. In a pass without a depth attachment Vulkan skips the depth
  test, where Metal's documentation states no answer. `set_depth_bias` takes the bias, slope
  scale and clamp as given; a draw of triangles is biased where the bias or the slope scale is not
  zero and a draw of lines never, as Metal biases triangles alone, and the clamp bounds the bias as
  Metal's does (a positive clamp from above, a negative one from below, 0 not at all), under the
  static note `PrecisionNote::DeviceDepthBias`
  (`RenderCommandEncoder::DEPTH_BIAS_NOTE`): the slope measure, and a fixed-point attachment's
  unit, are the device's; Apple states neither. Where a pass
  has both a depth and a stencil attachment they are one texture's same level and slice, a
  format of depth and stencil, rendered through one view: Vulkan's rendering takes one view for
  both (`VUID-VkRenderingInfo-pDepthAttachment-06085`). A pipeline state's depth and stencil
  formats are the pass's, as its color formats are; the format query decides which depth and
  stencil formats a device renders, a texture of `RenderTarget` usage in one it does not is not
  made, and a pipeline state of one is the translator's refusal.
- A pass has two areas. Metal's render area is the smallest attachment's level, cut to
  `renderTargetWidth` and `renderTargetHeight` where they are set: every scissor, the default
  viewport and scissor, the areas a draw reads of its own attachments and the shadows are cut to
  it, so no draw rasterises past it. Vulkan's render area, where the load, store and resolve
  operations act, is Metal's rounded up to 32 pixels a side and cut to the smallest attachment's
  level, in every pass, with no call, barrier or pass of its own. A Mac clears that band too:
  Apple's page for the `Clear` load says "The GPU writes a value to every pixel in the attachment
  at the start of the render pass", and on an M3 Max and an M4 a `Clear` pass that draws cleared
  its area rounded up to 32 pixels a side, clipped to the attachment. So a `Clear` pass's band
  holds its clear value, a `Load` keeps the band, and a `DontCare` leaves it undefined, as Metal
  allows any value there. A resolving pass resolves the band too, from the samples its load left
  there (Known gaps of render encoders).
- A pass orders itself as a blit's copy does, with two barriers. The first runs it after every
  earlier command, makes their writes visible to it, and moves each attachment whose home layout
  is shader-read-only to the attachment layout; one whose home is general or the attachment layout
  renders there. It pays a barrier an encoder owed, and the hazard sets start again. A pass that
  opens its command buffer (its command buffer's first command, or the first after an event's
  signal or wait) and moves no attachment records no first barrier: every command before it is
  in a command buffer that ended with a barrier ordering it before every later command on the
  queue. The second, after the rendering ends, moves those attachments home and makes the pass's
  writes visible to every later command. Where no attachment moves, it is owed instead: the next
  command records it first, reaching every later command, and where the pass is the last command
  before the command buffer's end, the end's barrier is it. A command buffer of one pass whose
  attachments render at home has one barrier, its end's, one pass in `Metrics::passes`, and
  copies nothing.
- Refused, as Metal's validation stops them: an attachment's texture made without
  `RenderTarget` usage (`Refusal::AttachmentUsage`), a level or slice past its texture's
  (`Refusal::AttachmentPlace`), a color attachment of a depth or stencil format, a depth
  attachment of a format without depth or a stencil attachment of a format without stencil
  (`Refusal::AttachmentFormat`). A refused descriptor
  leaves the command buffer open for another encoder. A pass may use all eight color attachments:
  the floor's `maxColorAttachments` is 8.
- An encoder holds Metal's render state: a pipeline state, the vertex and fragment functions'
  buffers (`set_vertex_buffer`, `set_vertex_bytes` and their offsets, the fragment's alike),
  textures and sampler states, up to 16 viewports and scissor rectangles, the cull mode, the
  front-facing winding, a depth-stencil state (`Device::new_depth_stencil_state`), the stencil
  references, the depth bias and the blend color, Metal's defaults where it set none. A vertex buffer slot serves
  the vertex function's buffer argument of its index and the vertex descriptor's layout of its
  index alike, a `set_vertex_bytes` copy too, read through the parameter ring's buffer.
- An encoder also holds the alpha test reference of OpenGL on Metal
  (`set_alpha_test_reference_value`, Apple's `setAlphaTestReferenceValue:` SPI): 1 until the
  encoder sets it, a reference set in an earlier encoder not kept, as an Apple M3 Max under macOS
  27.0.1 keeps it. A state with the private alpha test (`PrivateRenderState::alpha_test_enabled`,
  a compare function other than `Never` or `Always`) reads it at each draw, pushed as Metal's
  8-bit unorm code (`metal2vulkan::bind::alpha_test_code`), and its fragment function compares its
  color 0's alpha's code with it; a fragment that fails writes nothing.
- An encoder also holds the line width (`set_line_width`, Apple's `setLineWidth:` SPI): 1 until
  the encoder sets it, every width kept. Lines draw at width 1 alone; a draw of triangles runs at
  any width, as Metal's filled triangles do.
- `draw_primitives` and `draw_indexed_primitives` are one `vkCmdDraw` or `vkCmdDrawIndexed` each.
  A draw pushes its functions' fields in the one push range (the addresses of their buffer slots
  plus offsets, the resource IDs of their textures and sampler states, the state's constexpr
  sampler indices), binds the vertex buffers its vertex descriptor reads and the index buffer,
  and sets the dynamic state that changed since the encoder's last draw: all of it at the first.
  A draw or a dispatch whose push bytes the command buffer already holds pushes nothing: the push
  stays across pipelines, which share the one layout, so 2,000 draws of unchanged bindings are one
  push and 2,000 `vkCmdDraw`s, as Metal's bindings set once stay for the draws after them.
  `Counters::metrics`' `pushes` counts each push recorded and `pushes_held` each one skipped.
  A draw takes no barrier, as Metal orders none between the draws of one encoder, so a pass of
  any number of draws records the barriers a pass of one does. A draw of no vertices, indices or instances
  records nothing.
- `set_tessellation_factor_buffer` is `setTessellationFactorBuffer:offset:instanceStride:`, and
  `draw_patches` is `drawPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:instanceCount:baseInstance:`
  with a post-tessellation vertex function's state: one `vkCmdDraw` of `patchCount` vertices from
  `patchStart`, each patch one vertex of the state's patch list. The draw pushes the factor
  buffer's address plus its offset with the functions' fields, filled once per change of the
  state or a binding as every field is, so a patch draw records what a plain draw of the same
  bindings records. The factor buffer counts among the draw's reads as an index buffer does (a
  Managed one's device copy brought up to date first), and the command buffer keeps it until it
  retires. The draw's `numberOfPatchControlPoints` changes nothing: the function's own count is
  what Metal draws. A patch draw with a state that is not post-tessellation, or a draw of
  primitives with one that is, is `Refusal::PatchDraw`; a patch draw with no factor buffer
  `Refusal::NoTessellationFactorBuffer`; factors at an odd byte
  `Refusal::TessellationFactorOffset` (an offset 2 past a multiple of 4, which Metal's validation
  refuses, is read at that byte, as plain Metal reads it).
- `draw_primitives_indirect` is `drawPrimitives:indirectBuffer:indirectBufferOffset:`,
  `draw_indexed_primitives_indirect` is
  `drawIndexedPrimitives:indexType:indexBuffer:indexBufferOffset:indirectBuffer:indirectBufferOffset:`
  and `draw_patches_indirect` is
  `drawPatches:patchIndexBuffer:patchIndexBufferOffset:indirectBuffer:indirectBufferOffset:`: one
  `vkCmdDrawIndirect`, or `vkCmdDrawIndexedIndirect` for the indexed draw, on the buffer at its
  offset, which the GPU reads as Vulkan's own command, as Metal's
  `MTLDrawPrimitivesIndirectArguments`, `MTLDrawIndexedPrimitivesIndirectArguments` and
  `MTLDrawPatchIndirectArguments` lie as `VkDrawIndirectCommand` and `VkDrawIndexedIndirectCommand`
  do. A patch draw's patch count is the draw's vertex count unchanged, each patch one vertex as a
  direct patch draw's. The base instance is the device's own under the floor's
  `drawIndirectFirstInstance`. The arguments' buffer counts among the draw's reads as an index
  buffer does, and the command buffer keeps it until it retires; the pass's first barrier, and the
  encoder's memory barrier after a draw that wrote, make earlier writes visible to the GPU's read.
  The runtime never reads the arguments, so a count of 0 is recorded and draws nothing. An offset
  that is not a multiple of 4 is `Refusal::IndirectOffset`, as Metal's headers require; one that
  leaves fewer than the arguments' 16 bytes in the buffer (20 for an indexed draw) is
  `Refusal::IndirectArgumentsPastEnd`, the runtime's own check, as Vulkan's indirect draws read
  inside the buffer; arguments at a byte of the block's allocation that is not a multiple of 4 are
  `Refusal::ArgumentPlace`; an indexed draw's index offset that is not a multiple of the index size
  or is at or past the buffer's end is `Refusal::IndexBufferOffset`.
- A viewport is Vulkan's with its height negated (from `y + height`): framebuffer rows, the
  fragment position and derivatives follow Metal's clip space, whose y points up, and Metal's
  front-facing winding is Vulkan's front face of the same name. A viewport of no width or height,
  or with a NaN origin x, draws nothing, as Metal's. Metal clips a draw to the pixels its viewport
  covers, so the scissor is Metal's rectangle cut to them; a rectangle past Metal's render area,
  which Metal's validation stops, is cut to it, as plain Metal cuts it. A stencil reference
  saturates at 255, as Metal stores it.
- `set_viewports` and `set_scissor_rects` are `setViewports:count:` and
  `setScissorRects:count:`, 1 to 16 each, recorded with their count (`vkCmdSetViewportWithCount`,
  `vkCmdSetScissorWithCount`, so no pipeline is made for a count). A vertex function's
  `[[viewport_array_index]]` selects a viewport per primitive: the translator stores it to
  `ViewportIndex` (Floor: `multiViewport`, `shaderOutputViewportIndex`), 0 for an index at or past
  the encoder's count, which a draw pushes, as Metal draws it through viewport 0. A draw of any
  other function draws through viewport 0. Vulkan takes as many scissors as viewports: each
  viewport's is the rectangle of its index, the one rectangle where one is set, and none past
  the rectangles set where several are, each cut to the pixels its viewport covers. The index is
  stored as the function writes it, as a layer index is (Layered rendering, below).
- A draw may read an attachment of its own pass through a texture, as Metal reads a render
  target, where the device reads attachments (Floor, Attachment reads): through the attachment
  feedback loop where the device has it, else through a copy. The texture reaches the
  attachment's level and layer, the fragment function reads it and nothing writes it, and its
  image is a colour render target not in place over a buffer, which one attachment of the pass
  renders: its home is general where it has a storage view (made for shader writes, or of an
  unknown usage, where the device stores its format), shader-read-only where shaders read it,
  else the attachment layout, and its image takes the sampled and feedback usages where the
  device's image format query offers them beside the rest, the storage usage included, at the
  texture's sample count; where it refuses them, or offers fewer sample counts with them (the
  sampled usage's can be fewer than the attachment's), the image is made without them and a
  draw that reads it is `Gap::Draw`. A pass
  that reads no attachment through it records no move of it. Vulkan's feedback layout asks the attachment and sampled usages beside it and
  bars none (VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-07006); in it the image is only
  rendered and sampled. Through the feedback loop, at the pass's first such draw the rendering
  ends, one barrier moves the whole image to the feedback layout (the attachment's level and layer
  from the layout the pass renders it in, the rest from home), and the rendering begins again
  with the attachment in that layout; the pass's second barrier moves the whole image home. Until then every draw of the
  pass reads the image's textures through their second elements (Set 1), a texture that reaches
  no rendered level among them. Every render pipeline is made with the feedback loop's colour
  flag (`VK_PIPELINE_CREATE_COLOR_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT`), which lets a draw read an
  attachment in the feedback layout and asks nothing of a draw that reads none, so no draw records
  a state for the loop. Each draw that
  reads an attachment through the loop, through a texture or by a fetch (below), counts in
  `Metrics::attachment_loop_reads`, its barriers in `Metrics::barriers`.
- Through a copy, on a device without the feedback loop, and for a surface plane whose image took
  no feedback usage (Exported surfaces), the draw reads the image's shadow: an image of the same
  format, extent, levels and layers, made at the first draw that reads the image and freed with it
  (`Gauges::shadows` and `shadow_bytes`). A texture that reaches the attachment's level and layer
  reads the shadow through its second element; one that reaches neither reads the image at home.
  Where the draw reads what the shadow does not hold, the rendering ends, one barrier orders every
  earlier command before the copies and moves what they copy to the transfer layouts, the copies
  are recorded (`Metrics::attachment_read_copies` and `attachment_read_copy_bytes`), one barrier
  moves it all back, and the rendering begins again. A slot the fragment function reads only at its
  own pixel reads the draw's scissor, a pixel wider each way, clamped into its level; any other
  slot reads every level and layer its texture reaches. What is new to the shadow is copied as one
  area where it is one (a wider scissor's band), else as the whole read, with no look-ahead. At the
  pass's first draw, where the pass cleared the attachment and every read of its level and layer
  lies in it alone and inside Metal's render area, the shadow's level and layer is cleared to
  the pass's clear colour between the same barriers instead of copied (the shadow is of the
  attachment's format, so the clear writes the texels the pass's did); a draw of the pass before
  it, of any pipeline, leaves the copy.
- A fragment function's read of a colour attachment (`[[color(n)]]`, framebuffer fetch) reads the
  attachment's texture at the fragment's own pixel, through its second element, whose index the
  draw pushes (`PushField::FetchedColor`): in the feedback layout, or through the image's shadow,
  copied at the draw's scissor a pixel wider each way, as above. Metal orders the read after every
  earlier fragment of the pass at the pixel, with no barrier from the program, so a fetching draw
  after a draw of the pass pays the barrier `memory_barrier` owes (the rendering ends, one global
  barrier, and it begins again), and through a copy copies what it reads into the shadow again: a
  pass of N fetching draws records N - 1 such barriers, or N - 1 copies. The texture is a 2D
  texture (a 2D multisample one in a multisampled pass) whose first level and slice the pass
  renders, made with an index, of a colour render target's image, read by shaders or not, with a
  storage view or not: a texture of its own image, or a surface plane's (Exported surfaces).
- `memory_barrier` after a draw of the encoder that wrote memory ends the rendering, records one
  barrier and begins the rendering again, loading what it stored. After draws that only rendered,
  in a pass that moved an image to the feedback layout, the barrier is owed to the next draw that
  reads an attachment: the rendering ends, one global barrier is recorded and it begins again,
  which orders every pixel the draw reads, its own and any other. Elsewhere it records nothing:
  everything before the pass is ordered before it, and a move orders everything before it.
  On a device without the feedback loop it records nothing either, and after draws every shadow
  holds nothing from then: the next draw that reads an attachment copies what it reads again,
  after what the earlier draws rendered. A draw that needs a copy first, of a texture over a
  buffer in an image of its own or over a
  Managed buffer's device copy, records it the same way, between the rendering's end and its new
  beginning. Fences record nothing, as in a compute encoder.
- Refused, as Metal's validation stops them: a draw with no pipeline state
  (`Refusal::NoPipeline`), a vertex buffer the vertex descriptor reads that is not bound
  (`Refusal::NoBufferAtIndex`) or bound at an offset past its length
  (`Refusal::VertexBufferOffset`), an index offset that is not a multiple of the index size or
  indices past the buffer (`Refusal::IndexBufferOffset`), a viewport field NaN or infinite but its
  origin's x (`Refusal::ViewportNotFinite`), and a stencil test in a pass without a stencil
  attachment (`Refusal::StencilAttachment`).

### Known gaps of render encoders

- A draw that reads its own pass's attachment through an argument buffer or a resource ID reads a
  descriptor whose layout differs from the image's, which Vulkan leaves undefined
  (`VUID-vkCmdDraw-imageLayout-00344`); such draws count in
  `Metrics::unslotted_attachment_reads`, and the image is back in its home layout once the pass
  ends.
- An indexed indirect draw's index count is the GPU's to read, so a count that runs past the index
  buffer is not refused, as a direct draw's is (`Refusal::IndexBufferOffset`, which Metal's
  validation stops). The index buffer is bound from the indices to the end of the block's
  `VkBuffer`, so such a draw reads the block's later bytes as indices, and a count past the block's
  end is outside Vulkan's valid usage without `robustBufferAccess2`, which the floor does not
  require (`VUID-vkCmdDrawIndexedIndirect-robustBufferAccess2-08798`).
- Refused by `Gap::Draw`, naming its `DrawKind`: the triangle fill mode `Lines` on a device
  without `fillModeNonSolid` (KosmicKrisp on the M3 Max, of the venues dumped, m3-max-48737) and
  the fill mode of raw value 2, which no SDK header declares (`FillMode`,
  `set_triangle_fill_mode`), no viewport or scissor rectangle, or more than 16 (`Viewports`,
  `set_viewports`, `set_scissor_rects`), and a `Boolean` or `Counting` visibility result mode in a
  pass whose `visibility_result_type` is `Accumulate` (`AccumulatedVisibilityResults`,
  `set_visibility_result_mode`; counted in `Metrics::accumulated_visibility_refused`), each
  refused where it is set, the encoder's state unchanged; the replay streams set Metal's defaults
  alone, which those setters take as nothing; beside `Accumulate`, while a `Boolean` or `Counting`
  mode's query counts the draws, an execution of an indirect command buffer over a range a buffer
  holds (`VisibilityResultInterrupt`, `execute_commands_in_buffer_with_range`; counted in
  `Metrics::visibility_interrupts_refused`), refused where it is called: its count kernel ends the
  rendering, and both need a count carried across two queries with the parts added, which is not
  built; a draw of lines or a line strip while the encoder's line width
  (`set_line_width`, which keeps every width) is not 1 (`LineWidth`): Metal draws such a line as
  a parallelogram that width wide, and Vulkan's wider lines are a device feature (`wideLines`)
  the floor does not require; a
  viewport Vulkan's is not (a
  negative width, a depth range past `[0, 1]`, bounds past the device's viewport limits), a
  pipeline state whose formats are not the pass's attachments', a draw that binds a texture
  whose image the pass renders, where neither the feedback loop nor a copy reads it: on a device
  without the attachment reads, from the vertex function at the rendered level and layer, written, of a
  depth or stencil attachment, of an image over a buffer in place or whose format the device
  takes neither the feedback usage nor a shadow for, or of an image two attachments of the pass
  render; and, on a device that does not report
  `supportsNonZeroFirstInstance` (lavapipe and the UHD 630 among the lab's), a draw at a base
  instance other than 0 through a state that steps a layout per instance at a rate above 1
  (`DividedBaseInstance`: Vulkan's divisor takes `firstInstance` 0 alone there,
  VUID-vkCmdDraw-pNext-09461); the same draw from base instance 0 runs. An indirect draw through
  such a state, whose base instance the GPU reads, is refused alike. A patch draw through a patch
  index buffer is `PatchIndexBuffer`: the control stage takes each patch's index from its position,
  and no replay stream's patch draw names one. Every attachment the replay streams read through a
  texture is a colour render target of `Private` storage, read from the fragment function; and their
  stepped draws (a WebGL stream's, at rates 2 and 3) are all at base instance 0. A draw that reads
  its own target over a buffer in place, a linear image on the buffer's memory that never leaves the
  general layout, is refused on every device (`AttachmentRead`): no play of the replay streams met
  it.
- A fetch (`[[color(n)]]`) orders the fragments of earlier draws before it, not the fragments of its
  own draw that cover the same pixel, which Metal orders in primitive order: a fragment may read the
  texel from before an earlier fragment of its draw wrote it
  (`PrecisionNote::UnorderedFetchWithinDraw`). Exact for a draw whose primitives do not overlap at a
  pixel. A write the fragment function would compute over its own read of the attachment, where no
  Vulkan blend, write mask or logic operation gives Metal's, is not built: the translator refuses
  each such state at creation on every device, an `Error::Translation` (the conformance ledger's 22
  such rows). Each is built on the runtime's own fetch form, the attachment's pushed texture
  (`PushField::FetchedColor`), if a guest meets it:
  - a padded extended-range attachment's blend (`R10Unorm_X6`, `RG10Unorm_X12`, their `_sRGB`
    twins and `RGB10A2Unorm_sRGB`), refused naming `FramebufferFetch`, not implemented;
  - a write of some of the codes one word of an extended-range attachment holds (`BGR10_XR`, the
    two-plane forms' colour word and their `_sRGB` twins, under a mask of some of those codes),
    refused naming `FramebufferFetch`, not implemented;
  - a blend no Vulkan state gives on an attachment whose image holds its channels rotated
    (`ARGB8Unorm` reading destination alpha), refused naming `FramebufferFetch`, not implemented;
  - a word attachment written under a mask of some of its channels, which takes its format's own
    attachment and is refused where the device lacks that format;
  - a logic operation on a device without `logicOp`, refused naming
    `PrivateRenderState(LogicOperation)` and `logicOp`.

  A raster order group on the colour it reads, or on a colour output of an attachment it reads,
  takes the same form on every device, the pixel interlock's included: the interlock cannot order
  the attachment's own write, which happens after the function. A fetch of an attachment the draw
  cannot read through is
  `Gap::Draw` of `FetchedAttachment`: a drawable's texture (its image has no feedback usage and no
  second element; no boot or replay stream fetched one), a texture made with no index, one that is
  no 2D texture at its first level and slice (a 2D array texture among them, of one slice too, as a
  layered pass renders into), a texture of an image another attachment of the pass shares (two
  planes of one surface), and every attachment on a device without the attachment reads.
- A pass over a surface plane that renders as a linear colour attachment alone (Exported
  surfaces: an opaque handle's plane in place through
  `VK_FORMAT_FEATURE_2_LINEAR_COLOR_ATTACHMENT_BIT_NV`, every plane format measured on the RTX 3090
  and the RTX 5080 Laptop GPU) is refused as it begins, `Gap::LinearPlanePass`, where it sets a
  colour attachment that is not such a plane (`OtherColor`), a depth or stencil attachment
  (`DepthStencil`, the driver's fault: an RTX 3090 lost the device on that shape, and the runtime
  recovers from no device loss), or resolves a multisample attachment into one (`Resolve`). A
  pass of such planes alone renders.
- A surface plane whose texture's usage holds `ShaderRead` and `RenderTarget`, where the handle's
  image format query (`vkGetPhysicalDeviceImageFormatProperties2`: the plane's format, the
  handle's linear tiling, external as the handle, with the feedback usage added) refuses the
  feedback usage, or where the images with it do not hold the export's layout, is read as an
  attachment of its own pass through its shadow, the copy a device without the feedback loop
  takes, which Metal does not make (`Metrics::attachment_read_copies` and
  `attachment_read_copy_bytes`, in `bytes_copied` too).
- A draw that reaches an attachment of its own pass through an argument buffer's texture is not
  seen: it reads through the texture's own element, in its home layout, while the pass may have
  moved the image to the feedback layout, or, without the feedback loop, while the pass renders
  the level and layer it reads.
- A vertex fetch outside its buffer returns what the device gives: the floor enables no buffer
  robustness, so there is no bounds guarantee beyond the device's. A post-tessellation function's
  control points and a kernel's stage-in are fetched in the shader instead, checked as a buffer
  argument's access is (see "Known gaps").
- On the Iris Xe (ANV, Mesa 26.2.6) a viewport whose far edge lies inside a pixel loses that
  pixel's row or column even where its pixel centre is inside the viewport, which Metal draws (the
  conformance case `fractional-viewport-coverage`, recorded on an Apple M3 Max). ANV clamps the
  hardware scissor to the viewport and keeps the far edge as an integer: the last row is
  `MAX2(vp->y, vp->y + vp->height) - 1` and the last column `vp->x + vp->width - 1`, each
  truncated (`update_scissors` in Mesa's `src/intel/vulkan/genX_gfx_state.c`, lines 2164 to 2169),
  though the device reports 13 bits of viewport sub-pixel precision (`viewportSubPixelBits`). So a
  viewport ending at 7.7 draws through row 6, and row 7, whose centre 7.5 is inside it, is not
  drawn: a filled draw through a viewport from y -0.3 to 7.7 left row 7 undrawn on the Iris
  (m3-max-63429). Vulkan gives no allowance for it: a pixel is covered where its sample point lies
  inside the clipped primitive ("Basic Polygon Rasterization"), and the scissor test holds the
  application's rectangle ("Scissor Test"). No Vulkan control or query reaches that scissor; it is
  a driver bug, and the row stays undrawn on that driver.
- AMD's open Linux driver (AMDVLK 2025.Q2.1) on the RX 7900 XTX and the Radeon 780M differs from
  Metal in the same case as the Iris Xe, with the same count of differing elements (480), the
  same first texel and the same largest difference: where the far pixel's centre lies inside the
  viewport it is not drawn, which Metal draws and the two rules above require (the 780M reports 8
  bits of viewport sub-pixel precision, job m3-max-70426); where the centre lies outside the
  viewport or on its edge, the difference is inside the accepted difference of docs/API.md
  section 10. RADV on the same cards draws Metal's pixels (jobs m3-max-70513 and m3-max-70514).
  The driver's source was not read; it is a driver bug, and the pixels stay undrawn on that
  driver.
- AMD's open Linux driver (AMDVLK 2025.Q2.1) on the RX 7900 XTX resolves a
  colour attachment at a slice above 0 of a multisample array texture into the right slice of the
  resolve texture, but from the samples of the array's slice 0: in the conformance case
  `layered-resolve-places`, which renders slice 1 and resolves it into slice 1, all 64 texels of
  that slice read 0 where Metal holds the drawn and cleared values (job m3-max-70391 on the RX 7900
  XTX), and the resolve texture's other slices keep their values. RADV on the same card resolves
  it exactly. The runtime resolves a 2D view of the attachment's slice into a 2D view of the resolve
  slice, and Vulkan's resolve combines the samples of the attachment's own view ("Render Pass
  Multisample Resolve Operations"); a standalone Vulkan program with no runtime code gives the same
  slice-0 samples under that driver on the RX 7900 XTX and exact slices under RADV (job
  m3-max-70937). No Vulkan query reaches it and the runtime has no other form: it is a driver bug,
  and the slice stays wrong on that driver. No replay stream resolves at all: 0 resolve attachments
  in the 26 streams (jobs m3-max-70810 to m3-max-70812).
- Visibility results (`set_visibility_result_mode`, Metal's `setVisibilityResultMode:offset:`):
  each `Boolean` or `Counting` mode set counts the samples its later draws pass, up to the next
  mode set or the encoder's end, in one precise occlusion query (`occlusionQueryPrecise`, the
  floor's), begun in the pass's rendering before the mode's first draw and ended there as the
  mode ends; a mode with no draw counts none. After the rendering ends, the pass copies each
  count, 64 bits, into the visibility result buffer (`visibility_result_buffer`, Metal's
  `visibilityResultBuffer`) at its offset, one `vkCmdCopyQueryPoolResults` for each run of
  offsets 8 bytes apart, before the pass's second barrier, which orders the copies before every
  later command; the host sees them once the command buffer completes, as a blit's write. Every
  other byte of the buffer keeps its value, as Metal's pass keeps it; a pass whose encoder
  enables no mode leaves the buffer as it was under either `visibility_result_type` (an Apple M3
  Max, macOS 27.0.1; the conformance cases `visibility-disabled-reset` and
  `visibility-disabled-accumulate`). `Disabled` takes any offset, with or without a buffer. A
  `Boolean` mode writes the count too, which is nonzero exactly where a sample passed (Metal
  documents a nonzero word; an Apple M3 Max writes 1), and the count is the device's own
  (`RenderCommandEncoder::VISIBILITY_NOTE`, `PrecisionNote::DeviceOcclusionCount`). Metal's own
  refusals, each where the mode is set, the encoder's mode unchanged: a mode with no visibility
  result buffer (`Refusal::NoVisibilityResultBuffer`), at an offset that is not a multiple of 8
  or whose 8 bytes run past the buffer (`Refusal::VisibilityResultOffset`), and at an offset a
  mode of the encoder took before (`Refusal::VisibilityResultOffsetTaken`: Apple documents each
  offset as set once per pass). A command buffer takes its queries from pools of its own, 256
  queries each, which the device lends it and takes back as it retires, each query reset on the
  host (`hostQueryReset`); the device keeps at most 16 free and destroys the rest.
  `Gauges::occlusion_pools` counts the pools alive, the free ones and those command buffers hold,
  and reads 0 once the device drops. A command buffer gives its pools back before its
  `wait_until_completed` returns, so passes that set a mode, each waited for before the next,
  hold one pool between them however many run.
- While a `Boolean` or `Counting` mode's query counts the draws (from the mode's first draw to the
  next mode set or the encoder's end), a command that would end the pass's rendering is refused,
  `Gap::Draw` of `VisibilityResultInterrupt`, the encoder as it was, and counted in
  `Metrics::visibility_interrupts_refused`: a memory barrier after a draw that wrote memory; an
  execution of an indirect command buffer over a range a buffer holds, whose count kernel runs
  outside the rendering; a
  draw that moves an attachment to the feedback layout, pays the barrier owed before an
  attachment read, or copies into an attachment's shadow what it reads; and, whether or not a
  copy is owed, a draw that looks at textures over a buffer in images of their own (one bound or
  declared, or, while such a texture lives, a draw through addresses outside its resources, an
  execution of an indirect command buffer, or a draw while such a texture's image is ahead of its
  buffer) or that reaches a Managed buffer with textures over its device copy. Vulkan's query
  begins and ends inside one rendering, and carrying a count across the rendering's end into a
  second query, with the parts added, is not built. A mode's first draw may still end the
  rendering, before the query begins, and so may every command after a mode set before its first
  draw.
- `use_resource` is `useResource:usage:stages:`, its plural and their deprecated forms without
  stages: the command buffer keeps the buffer or the texture until it retires, and nothing is
  recorded. The stages are not read: every resource is resident to every function, and the pass's
  barriers order its draws after every earlier command and before every later one. Every later
  draw of the encoder reaches a declared resource with its declared access, as it reaches a bound
  one: a declared texture over a buffer in an image of its own takes its copies at the draw, and
  a declared write of a Shared buffer, or of a texture over one, reaches the host once the command
  buffer completes. A declared write of a texture with no storage element refuses the encoder's
  later draws that may reach one beyond their slots (the known gaps of textures).
- Not built, and so not methods: the store action options setters, tile functions,
  `setVertexBuffer:offset:attributeStride:atIndex:`, `setTessellationFactorScale:` (read only by a
  state that enables the scale, which is refused) and `drawIndexedPatches:` (whose control-point
  index buffer only a state of a control-point index type reads, which is refused).
  `useHeap:stages:` is `use_heap`, in the compute encoder's form: the command buffer keeps the
  block and declares no access.
- Refused by `Gap::RenderPass`, naming its `RenderPassKind`: a depth and a stencil attachment
  that are not one texture's same level and slice (`DepthStencilViews`: Vulkan's rendering takes
  one view for both), a `Clear` load of a depth attachment whose clear depth is outside `[0, 1]`
  or NaN (`ClearDepth`: Vulkan clears no such depth without `VK_EXT_depth_range_unrestricted`,
  `VUID-VkClearDepthStencilValue-depth-00022`, which the floor does not require), attachments of
  two sample counts (`SampleCounts`: Vulkan renders one count into every attachment of a pass,
  `VUID-VkRenderingInfo-multisampledRenderToSingleSampled-06857`), a resolve the runtime does not
  build (`Resolve`: into a resolve texture of another pixel format, of more than one sample, or
  whose level is not the attachment level's size; out of an attachment of one sample; of an
  extended-range, bfloat or YCbCr format, whose image Vulkan's average would read in its own
  encoding; or of a depth and a stencil attachment into two views), a resolve into a texture made
  without `RenderTarget` usage whose image has no attachment usage (`ResolveUsage`: Metal
  resolves into a texture of any usage; Vulkan only into a view with the attachment usage,
  `VUID-VkRenderingInfo-colorAttachmentCount-09476`, which a one-sample colour texture of an
  optimal image takes where its format renders, and a texture's linear image in place over a
  buffer of the runtime's own memory where the device's linear tiling renders its format (where
  it does not, or where its subresource reaches past its own rows or its buffer, the pass resolves
  into the texture's resolve image and copies it into the buffer, above), and a depth or stencil
  texture does not, nor a texture in place over a dma-buf's import, nor a view of another format
  of a texture in place whose linear image takes no attachment usage; the usage grows a 1920 x 1080 `RGBA8Unorm`
  optimal image of one level 1.0056 times on the Radeon 780M and changes no linear image's layout,
  `RenderPassKind::ResolveUsage` has the measures),
  a depth or stencil resolve the device's resolve modes do not offer (`ResolveFilter`: `Min` or
  `Max` where `supportedDepthResolveModes` lacks it, the stencil filter `DepthResolvedSample`,
  which no Vulkan resolve mode is, and two aspects paired where `independentResolve` or
  `independentResolveNone` does not allow it), a store action an encoder sets that does not
  resolve an attachment whose pass left it `Unknown` with a resolve texture (`LateStore`: the
  rendering began resolving, as Vulkan fixes the resolve as it begins), a 1D texture's
  attachment, a 3D texture's depth or stencil attachment, a 3D resolve texture whatever its
  `resolveDepthPlane` (`TextureType`; a resolve texture of another type is resolved into as at
  plane 0, as Metal with its validation layer off does, probe `resolve_depth_planes.m`), the
  `CustomSampleDepthStore` store action, a clear of a format only a shader clears (extended
  range, bfloat, YCbCr), a `renderTargetArrayLength` past the device's `maxFramebufferLayers`
  (`ArrayLength`: Vulkan renders at most that many layers,
  `VUID-VkRenderingInfo-layerCount-07817`, and requires 256 of it, where Metal's arrays hold 2048
  slices), and a pass with no attachment past the device's `maxFramebufferWidth` or
  `maxFramebufferHeight` (`NoAttachment`). The descriptor has no field for a
  rasterization rate map, tile sizes, imageblock and threadgroup memory, sample buffer
  attachments or store action options. The replay
  streams' passes render into one or two color attachments and a stencil attachment or a
  `Depth32Float` depth attachment, each a level of a 2D texture of one sample, with `Load`,
  `Clear` and `DontCare` loads and `Store`, `DontCare` and `Unknown` stores.
- Programmable sample positions are not built. The descriptor has no field for
  `setSamplePositions:count:`, and the runtime answers no `programmableSamplePositionsSupported`.
  Every multisample pass rasterizes at Vulkan's standard sample locations
  (`standardSampleLocations`, under Floor), which are Metal's default positions at 1, 2 and 4
  samples. Metal rasterizes a pass at the positions the program sets; a pass whose set differs
  from the defaults samples here at the defaults, with no refusal. A consumer that answers the
  query YES owns counting and dropping a set that differs from the defaults. Vulkan's form for
  such positions is `VK_EXT_sample_locations`, which device creation does not enable.
- Layered rendering refuses: in a pass of `renderTargetArrayLength` 2 or more, an attachment or a
  resolve texture with fewer slices from its own `slice` than the pass's length
  (`Refusal::AttachmentPlace`, naming the last slice the pass would render), a 3D texture's
  depth planes, which are no attachment's layers (`Gap::RenderPass`, `TextureType`), and a draw
  that reads one of its pass's attachments through a texture on a device without the attachment
  feedback loop (`Gap::Draw`, `AttachmentRead`: an attachment's shadow holds one layer, and no 3D
  texture's depth plane, which a pass of one layer refuses the same way). A vertex function's `[[render_target_array_index]]` and
  `[[viewport_array_index]]` are stored as the function writes them. Metal requires the same value
  at every vertex of a primitive; where a program breaks that, the layer or viewport is the
  device's choice. An index the function declares and never writes, which Metal gives no value,
  is stored as 0, the same at every vertex.
- A `Clear` pass of one layer that encodes no draw clears Metal's render area rounded up to 32
  pixels a side, where a Mac clears the whole level and slice of a texture it keeps losslessly
  compressed, which no Vulkan query tells. A pass whose attachments differ in size reaches the
  smallest attachment's level alone. What a Mac's resolve writes into the resolve texture past
  Metal's render area is unmeasured; here a resolving pass resolves that band too, so after a
  `Load` pass the band of the resolve texture takes the resolve of the samples the attachment held
  there.
- An attachment whose home layout is shader-read-only (a texture of shader-read and render-target
  usage) is moved to the attachment layout and back by every pass that renders into it. A pass
  that reads it through a texture moves the whole image to the feedback layout at its first
  reading draw and home at its end, also where the next pass over it reads it so again. Their
  cost, measured in the harness (release; Mesa 26.2.6, NVIDIA 620.70.06): a pass that clears a
  `BGRA8Unorm` target of shader-read and render-target usage and draws over it once, the
  fragment function reading its own pixel of the target, against the same pass reading another
  texture of the same size; GPU time a pass, the median over four rounds of each round's median.
  The moves are timed alone by timestamps around their barriers: the move in, and how much more
  the pass's last barrier takes moving the image home from the feedback layout than from the
  attachment layout.

  | Device | 3648 × 2038: reading the target / another texture | 1920 × 1080: the same | The move in, 3648 × 2038 / 1920 × 1080 | The move home |
  |---|---|---|---|---|
  | RX 7900 XTX (RADV) | 54.3 / 20.3 µs (+34.0) | 20.4 / 13.9 µs (+6.5) | 19.6 / 8.0 µs | nothing more |
  | RTX 3090 | 66.6 / 59.4 µs (+7.2) | 37.9 / 30.9 µs (+7.0) | 7.3 / 7.3 µs | nothing more |
  | RX 550 (RADV) | 1,122.9 / 516.5 µs (+606.4) | 320.2 / 150.2 µs (+170.1) | 370.6 / 106.4 µs | nothing more |
  | Iris Xe (ANV) | 449.1 / 561.7 µs (−112.7) | 143.1 / 133.5 µs (+9.6) | 2.1 / 2.0 µs | 7.0 / 7.1 µs more |

  On RADV the cost is the move in; the rest of the difference, about 14 µs on the XTX and
  236 µs on the RX 550 at 3648 × 2038, is the draw's own, slower in the feedback layout. On the
  Iris Xe at 3648 × 2038 the pass reading another texture is the slower one, so that difference
  is not the moves'; their timed cost there is about 9 µs. On RADV the move in costs more the
  more the image's last rendering left compressed: at 3648 × 2038 it takes 203.3 µs on the
  RX 550 and 5.2 µs on the XTX in a pass that loads an image the last pass moved home from the
  feedback layout with no draw between, 370.6 and 19.6 µs after the pass's clear (the table's),
  and 477.3 and 22.7 µs after a draw over the image in the pass.
- Where the device has the feedback loop, a colour render target's image takes the feedback usage
  whether or not a draw reads it. Under AMD's open Linux driver (AMDVLK 2025.Q2.1) on the Radeon
  780M that usage costs draws that read no attachment GPU time: 64 draws over a 512 × 512
  `RGBA16Float` target of shader-read and shader-write usage (the harness's `draw_full`) take
  413.8 µs against 404.7 µs on an image without it (2.2%, m3-max-72026 and m3-max-72033), and the
  same draws into a target of render-target usage alone take the same time either way. The loop's
  flag on every render pipeline costs no time measured there (m3-max-72033) or on RADV
  (m3-max-72020).
- On a device that does not report `attachmentFeedbackLoopLayout` (expected of the UHD 630
  among the lab's, Windows driver 101.2145: no public report of that card on Windows lists
  `VK_EXT_attachment_feedback_loop_layout`), a draw reads an attachment through a copy into the image's shadow, which Metal does not make: at
  the pass's first draw that reads the image, at the first after a memory barrier after draws,
  and where a draw reads what the shadow does not hold, the rendering ended and begun again
  around the copies and their two barriers; where the draw is the pass's first and reads only
  what its pass cleared, a clear of the shadow takes the copies' place. Its cost, measured on the
  UHD 630 in the harness (release; GPU time a pass, the median over four rounds of each round's
  median) by a pass that loads its `BGRA8Unorm` target, ends for a copy of all of it and begins
  again, against the same pass without the copy: 39.0 µs a MiB at 1920 × 1080 and 53.5 µs a MiB
  at 3648 × 2038, and about 3 to 8 µs a copy beyond its bytes. Two replay streams, counted before
  a first read of what its pass cleared became a clear (so at most this many copies), make 68.7
  copies and 298.0 MiB a frame (r3) and 65.5 copies and 93.4 MiB a frame (b3): about 12 to 16.5
  ms a frame on the UHD 630 for r3 and 4 to 5.5 ms for b3. A pass that clears its target and
  reads it at its first draw, through the shadow's clear, takes 1,095.3 µs at 1920 × 1080
  and 4,200.8 µs at 3648 × 2038, against 644.2 µs and 2,341.2 µs for the same pass reading
  another texture. At 3648 × 2038 a pass that reads its target after a barrier takes 4,836.4 µs,
  and one that loads its target and reads it at its first draw 3,849.6 µs.

## Indirect command buffers

`Device::new_indirect_command_buffer` (`newIndirectCommandBufferWithDescriptor:maxCommandCount:options:`)
makes a buffer of indexed draw commands on a device that enables `VK_EXT_device_generated_commands`
and `VK_KHR_maintenance5` (`floor::INDIRECT_COMMANDS`, in the Floor's extensions). The runtime builds
the commands a kernel or a render pipeline state's function encodes: indexed triangle draws
(`MTLIndirectCommandTypeDrawIndexed` alone) that inherit the executing encoder's pipeline state
(`inheritPipelineState`) and bind their own vertex and fragment buffers (`inheritBuffers` false), at
most 31 of each stage. What a draw writes into the commands, an execution in a later encoder of its
command buffer reads: a render pass's end orders its writes before every later command.

- Its records are a Private block in the layout `metal2vulkan::pipeline::IndirectDrawLayout` names:
  a 24-byte header (the capacity, the vertex and fragment slots a command holds, the stride, and
  the index address a reset command binds), then one command a stride: an 8-byte word a slot, the
  index buffer token's 16 bytes, 4 bytes for the sequence index and the indexed draw's 20 bytes
  (152 bytes for 7 vertex and 7 fragment slots, 536 at most). `gpu_resource_id` is the header's
  device address, which the consumer writes into an argument buffer's `command_buffer` member; a
  function's `set_vertex_buffer`, `set_fragment_buffer`, `draw_indexed_primitives` and `reset` of a
  `render_command` write the command's fields through it, under the identity domains. `size` is
  the `maxCommandCount`.
- A new buffer's commands are copies of the device's template of reset commands of its layout (up
  to 64 KiB of commands, each drawing nothing with every slot the nil buffer's; 16 layouts kept, the
  least recently used forgotten first), in the upload of the next commit.
  `BlitCommandEncoder::reset_commands_in_buffer` copies the same commands over its range, and
  `optimize_indirect_command_buffer` records nothing.
- `RenderCommandEncoder::execute_commands_in_buffer` runs the range in parts, in order, the last
  part the rest: one `vkCmdExecuteGeneratedCommandsEXT` a part, a sequence a command, with the
  encoder's pipeline state and render state: a push constant token for each slot word the state's
  functions read, the index buffer token, a sequence index token and the indexed draw token. A
  part is the most commands whose preprocess bytes, as
  `vkGetGeneratedCommandsMemoryRequirementsEXT` answers for them with the inherited pipeline, stay
  at or under 2 MiB, one rule on every device (known gaps, below): the device is asked from one
  command up, doubling while the answer stays at or under the bound, then again at fewer, in
  proportion to the last answer, until it is. Before each part the execution pushes the part's
  first command, which the vertex function adds to the sequence index. The device keeps 64 command
  layouts, one per shape of tokens, and 256 execution shapes, each with its part, one per state,
  raster state (the depth clip and triangle fill modes, which choose the state's pipeline), layout
  and capacity.
- `execute_commands_in_buffer_with_range` reads the range (`MTLIndirectCommandBufferExecutionRange`)
  where a buffer holds it when the command runs, the one form for every range the GPU reads,
  location 0 included: the rendering ends, the device's count kernel (the translator's
  `OwnKernel::ExecutionCounts`, made at its first use and counted in `pipelines_created`) runs one
  workgroup a part, reads the range and writes each part's count into the command buffer's ring,
  one global barrier follows, and the rendering begins again, loading what it stored. The parts
  run from command 0, each through the range's last command (its `sequenceCountAddress`), none
  past it and none wholly before its location; in the part that holds the location, the vertex
  function returns before its body for each command before it. A range past the buffer's
  commands is clamped to them. A range the CPU gives (`execute_commands_in_buffer`) runs no
  kernel.
- A state made with `supportIndirectCommandBuffers` pulls every vertex attribute its vertex function
  reads, at its layout's stride from the descriptor, so a command binds no vertex buffer of its
  own, and its vertex function tests its command against the execution's range; a direct draw of
  the state tests it against a range of every command.
- A command's buffer slots are not bounds-checked: an execution pushes their bounds as 2^31 - 1
  bytes before the address and `u32::MAX` bytes after it. An access through one past the buffer's
  end reaches the memory there, as an access through a pointer loaded from memory does. An access
  at a negative offset that stays inside the buffer reads the buffer's own bytes.
- After an execution the push bytes and the index buffer are undefined, as Vulkan leaves them, and
  the next draw pushes and binds its own; the pipeline and the vertex buffers stay bound.
- Each execution takes a preprocess region when it is recorded, an idle one of the device's pool
  or one made then, its parts' preprocess one after another, each sized by
  `vkGetGeneratedCommandsMemoryRequirementsEXT` with the inherited pipeline at its part, and
  gives it back when its command buffer completes. Its sizes are those of a token per slot word
  (known gaps, below): on the 2080 a part of 64 commands of four slot words takes 69,888 bytes,
  1,092 a command, and 17 tokens take about 592 bytes a command at 1,024 commands or more. The
  pool keeps at most 64 MiB of idle regions,
  the oldest freed past that, and the block-maker thread frees each region idle for 250 ms, the
  idle age, as it frees idle allocations: at rest the pool holds nothing.
  `Metrics::preprocess_regions_remade` counts the regions made within a second after one was freed
  past the bound. Nothing waits and no execution is refused for the pool's sake. The gauges
  `indirect_command_buffers`, `preprocess_used_bytes` and `preprocess_idle_bytes` count them, each
  zero once the device is dropped.
- Of `VK_KHR_maintenance5` the runtime uses one structure, `VkBufferUsageFlags2CreateInfoKHR`, for
  the preprocess buffers' `VK_BUFFER_USAGE_2_PREPROCESS_BUFFER_BIT_EXT`.

### Known gaps of indirect command buffers

- An execution over a range a buffer holds runs its parts from command 0: each part wholly before
  the range's location runs no command but costs its execution (below), and the part that holds
  the location runs its commands before it, at most a part's less one, each costing its vertex
  work and drawing nothing.
- An execution over a range a buffer holds ends the rendering for its count kernel and begins it
  again: one more dispatch, one barrier, and the attachments stored and loaded. With the probe's
  form of the split, Apple's rendering sample's frame (11 such executions) took 1,003.4 us of GPU
  time against 931.8 without it on an RTX 2080, and 617.0 against 588.5 on a Radeon RX 7900 XTX
  (job 63481, medians of 9). On an RTX 3090 (NVIDIA's 620.70.06) the splits and the parts at the
  bound cost 164.0 us of the light frame's GPU time together (below): it took 1,862.8 us against
  1,698.8 at one part an execution with each part's count written as the command buffer records
  and no split (a probe, job m3-max-71730, medians of 9 turns at P0); timestamps around each split
  put it at 12.7 us. Running the count kernels instead ahead of each render encoder's rendering,
  in a Vulkan command buffer of their own before the rendering's (a form measured and not taken),
  made the light frame slower on the 3090: 1,793.2 and 1,880.5 us against 1,686.8 and 1,703.3,
  and the same draws direct, which run no execution, took 1,088.5 and 1,122.3 us against 1,070.1
  and 1,070.1, so a command buffer boundary ahead of each of the frame's 6 render encoders'
  renderings added 18.4 to 52.2 us a frame (job m3-max-71784: the two builds alternated, each
  pair at one clock state, medians of 9 turns). No form adds a command buffer boundary per render
  encoder. While a visibility result's query counts the draws (after a draw under a `Boolean` or
  `Counting` mode) the rendering cannot end, so such an execution is refused,
  `Gap::Draw` of `DrawKind::VisibilityResultInterrupt`, counted in
  `Metrics::visibility_interrupts_refused`; one before the mode's first draw runs.
- A state made with `supportIndirectCommandBuffers` pays one 8-byte load a vertex on its direct
  draws too: its vertex function reads the execution range.
- A state made with `supportIndirectCommandBuffers` whose vertex descriptor has a layout of
  `MTLBufferLayoutStrideDynamic` that its vertex function reads is refused at its making
  (`Gap::RenderPipeline`, `RenderPipelineKind::DrawInput`): its shader fetch would read the
  stride each bind sets (`setVertexBuffer:offset:attributeStride:atIndex:`, a command's included)
  from a push field the encoder does not fill.
- An execution over a range a buffer holds takes a preprocess region for the buffer's whole
  capacity, whatever the range, held until the command buffer completes: 17 tokens take about 592
  bytes a command on NVIDIA's driver (an RTX 2080) and 776 on RADV (a Radeon 780M), at 1,024
  commands or more.
- Every execution runs in parts of at most 2 MiB of preprocess bytes, on every device: half of
  about 4 MiB, past which one execution did not draw where it was measured on RADV, a driver bug,
  and since a wrong guess loses the guest's device, the bound keeps that margin. On a Radeon RX
  7900 XTX (Mesa 26.2.6) an execution of N commands whose generated commands pass about 4 MiB (296
  bytes a command, 14,170 commands) drew only its first N - 14,170 commands, or none; on a Radeon
  780M (Mesa 26.2.6) one execution of 8,162 commands of 17 tokens (about 6.3 MB) lost the device.
  Mesa 26.3.1's source chains the generated commands into further command buffers by the count of
  sequences alone (65,536), not by their bytes. NVIDIA's driver draws one execution of a buffer's
  whole capacity right, so there each part past the first is paid for the other driver's limit,
  about 10.5 us of GPU time a part: Apple's rendering sample's frame (45,024 commands of 17 tokens
  in 11 executions, each run through its range's last command) took 2,598.9 us in 182 parts,
  1,648.8 in 91, 1,175.6 in 46, 907.2 in 16 and 870.5 in 11 (one part an execution), against
  391.4 drawn directly (job 63460, an RTX 2080 on NVIDIA's 620.70.06, medians of 9). On the XTX a
  part costs about 2.4 us. The frame's parts at the bound took 931.8 us on the 2080 and 588.5 on
  the XTX (job 63481). A part runs its execution whatever its count: on an RTX 3090 a part with
  count 0 (wholly past its range's last command) costs about 10 us. There the light frame's 11
  executions take 17 parts at the bound, 6 of them with count 0, which timestamps around each put
  at 62.7 us together (a probe, job m3-max-71730).
- On NVIDIA's driver an execution costs GPU time beyond the draws it runs, each
  `vkCmdExecuteGeneratedCommandsEXT` call most of all, measured on one card, an RTX 3090 (NVIDIA's
  620.70.06), against a Radeon RX 7900 XTX (RADV, Mesa 26.2.6).
  - A guest's frame of this shape at a held camera pose took 3.446 ms of GPU time with its
    executions against 3.034 ms with the same draws direct on the 3090, 13.6% over (the
    consumer's jobs m3-max-64069 and m3-max-64113), and 4.305 ms against 4.372 on the XTX, 1.5%
    under (m3-max-64076 and m3-max-64084).
  - The light frame, a timing test's at its heaviest scale: 11 executions over ranges a buffer
    holds in 6 render passes, of buffers of 8,162, 785 and 289 commands, 2,132 commands in range,
    each command an indexed strip of 2,260 quads over a 32 by 16 pixel cell that binds 7 vertex
    and 7 fragment buffers, 5 and 4, or 4 and none (17, 12 or 7 tokens). It took 1,868.8 us
    of GPU time against 1,105.9 with the same draws direct on the 3090, 69.0% over (job
    m3-max-71730, every turn at P0, the SM at 1,950 MHz), and 719.7 against 558.4 on the XTX,
    28.9% over (job m3-max-71686, every turn at about 3,000 MHz): medians of 9 alternated turns of
    two frames in flight, each pair at one clock state, both arms' targets equal to the CPU's
    texel for texel.
  - The 3090's 735.5 us over direct in that frame (a probe, job m3-max-71730, medians of 9 turns,
    each pair at one clock state): about 300 us (41%) is each execute's own preprocessing inside
    the rendering: the same draws in 22 execute calls took 311.2 us more than in 11, about 28 us a
    call, and with each part preprocessed apart (`vkCmdPreprocessGeneratedCommandsEXT`) its
    executes took 294.9 us less while its preprocesses took 218.1. The splits and the parts at the
    bound take 164.0 us (22%), the commands' 17, 12 or 7 tokens against 7 for every command take
    63.7 (9%), and the remaining about 190 us (26%) is per-draw cost inside the executes, which no
    piece measured isolates. Sizing each part by the buffer's capacity rather than its range costs
    8.0 us. The same probe on the XTX (job m3-max-71686, medians of the 2 to 7 of its 9 pairs at
    one clock state): about 4.0 us a call, about 80 for the splits and the parts at the bound, and
    40.3 for the tokens.
  - Explicit preprocessing gains on NVIDIA and loses on RADV: with each part preprocessed apart, in
    a rendering ended for it, the light frame took 1,896.7 us against 1,909.8 on the 3090 (job
    m3-max-71730) and 808.1 against 745.1 on the XTX (job m3-max-71686, 5 and 3 of 9 pairs at one
    clock state), each with timestamps around each piece, so it is not the runtime's form.
- A command's buffer slots reach its functions through the push bytes alone, so each slot a
  command binds needs its word in the state's push window, `W` bytes (256, or 128 on a device
  whose `maxPushConstantsSize` is 128), which the vertex and fragment
  functions and all their other fields share. The translator places the vertex function's fields
  from byte 0 to byte `W - 8` (the last 8 bytes are kept for the fragment function's parameter block
  address), in a fixed order:
  1. its parameter block's address (8 bytes), where any of its fields spill;
  2. any other leading address (8 bytes, so each costs one slot word);
  3. the execution range's address (8 bytes), then the command's sequence word and its part's
     first command (4 bytes each), then any other leading word (4 bytes each);
  4. its buffer slots' words, in slot order (8 bytes each);
  5. its other fields, including, for each slot it reads at an offset that is not a
     translation-time constant, that slot's bounds and the zero range's address (8 bytes each);
  6. 24 bytes of gap and a 4-byte tail word, or its last bounds field as the tail.
  From the first field that does not fit, its fields sit in its parameter block. So a vertex
  function with no other leading address pushes 25 slot words ((248 - 16 - 28) / 8, rounded down)
  only where it pushes nothing past them but one 4-byte word or one bounds field. Anything more
  makes it spill, the block address joins, and it pushes 24: a slot read at a variable offset adds
  its bounds and the zero range's address after the slot words, so 25 no longer fit. The fragment
  function's fields run from where the vertex function's end to byte `W`, in the same order and with
  the same tail, except that a lone field of 8 bytes or less takes no tail. From code: a fragment
  function whose only field is one slot word fits beside a vertex function of 24 or 25 words. Its
  room past one slot is not measured.
- On a device whose push window is 128 bytes (its `maxPushConstantsSize` is 128), the vertex
  function's fields end by byte 120, so a command reaches 9 vertex slots ((120 - 16 - 28) / 8,
  rounded down) where the vertex function pushes nothing past them but one 4-byte word or one
  bounds field, and 8 where its fields spill, against 25 and 24 at 256 bytes (from code: the
  translator's `lower::push` test `an_indirect_draw_s_vertex_function_pushes_25_or_9_slot_words`).
  A fragment function whose only field is one slot word fits beside a vertex function of 8 or 9
  words. A state whose vertex function reads more slots than that is refused at its execution
  (`IndirectCommandsKind::DrawInput`), where at 256 bytes it executes. No lab card reports under
  256: this form runs on lavapipe with its translator told of 128 bytes (the runtime's tests'
  `push_window_128`), and on no card. An execution whose command binds a slot whose word sits in a
  parameter block is refused (`IndirectCommandsKind::DrawInput`). Measured on an RTX 3090 and a
  Radeon RX 7900 XTX (the consumer's jobs m3-max-63155, 63163 and 63164, the runtime at 173861f4f7):
  a vertex function reading N buffer slots with a fragment function reading 1 draws for N up to 24,
  and is refused as `DrawInput` for N from 25 to 29; with 28 slots read, commands that bind up to 24
  vertex slots draw. The token count binds first only on lavapipe in a window of 256 bytes (13 slot
  words, below).
- Each slot word a command holds is a push constant token of its own, on every device, where a run
  of words at consecutive offsets could share one: on NVIDIA's driver (620.70.06, an RTX 2080) a
  24-byte push constant token in an execution of 96 commands or more carried command 31's bytes in
  every later command, and three 8-byte tokens of the same bytes carried each command's own (a
  standalone program, which draws right either way on RADV and lavapipe). The cost:
  - A command takes at most `maxIndirectCommandsTokenCount` less 3 slot words, refused past that
    (`IndirectCommandsKind::Tokens`). That binds before the push bytes only on lavapipe in a window
    of 256 bytes, at 13. On NVIDIA's driver (29) and RADV (125), and in a window of 128 bytes on
    every device, the push bytes above run out first.
  - That program's part of 64 commands of three vertex words and one fragment word takes 69,888
    preprocess bytes on the 2080, against 57,600 with the vertex words in one token (21% more);
    567,808 against 557,568 on lavapipe; and 19,712 either way on the XTX.
  - On an RX 7900 XTX (RADV), one execution of 512 such commands in parts of 64: the execute call
    took 338,352 user-space instructions against 339,490 with one token for each run of words
    (0.34% fewer), and the command buffer 271.6 us of GPU time against 280.3 us, which reads as
    not worse, since the runs-of-words form ran first in the same hold (job m3-max-63201, medians
    of 9).
  - NVIDIA's driver has no right form to compare against.
- On Mesa's CPU device (lavapipe, Mesa 26.2.6) the size
  `vkGetGeneratedCommandsMemoryRequirementsEXT` reports for a draw layout is quadratic in the
  command count and wraps at 2^32 (8,519,680 bytes at 16,384 commands of four buffers), a driver
  bug. The part search asks from one command up, doubling, so it asks at no count the wrap
  reaches, and the parts are small: 64 commands of four buffers take 557,568 bytes (8,712 a
  command; 134,750,208 bytes at 1,024 commands).
- On the Iris Xe (ANV, Mesa 26.2.6), which does not list `VK_EXT_device_generated_commands`, an
  indirect command buffer is refused (`Gap::IndirectCommandBuffers`,
  `IndirectCommandsKind::Extension`).
- On the UHD 630 (its Windows driver), which does not list `VK_EXT_device_generated_commands`, an
  indirect command buffer is refused the same way.
- On the Mac host (KosmicKrisp, Mesa 26.3.1, on an Apple M3 Max), which does not list
  `VK_EXT_device_generated_commands`, an indirect command buffer is refused the same way.
- On AMD's open Linux driver (AMDVLK 2025.Q2.1) on the Radeon 780M, which lists
  `VK_KHR_maintenance5` but not `VK_EXT_device_generated_commands` (`vulkaninfo`, job
  m3-max-70426), an indirect command buffer is refused the same way.
- Refused by `Gap::IndirectCommandBuffers`, naming its `IndirectCommandsKind`: command types other
  than indexed draws alone (`CommandTypes`), commands that set their own pipeline state
  (`CommandPipelines`, on every device), commands that inherit the encoder's buffers
  (`InheritedBuffers`), more than 31 buffers of a stage (`BindCount`), a count of 0 or one past
  `maxIndirectSequenceCount` (`Capacity`), a compute state for indirect command buffers
  (`ComputeCommands`), an execution with a state made without `supportIndirectCommandBuffers`
  (`UnflaggedState`), one whose slots take more tokens than `maxIndirectCommandsTokenCount`
  (`Tokens`, which only lavapipe's 16 reach before the push bytes run out, in a window of 256
  bytes), and one whose functions
  read what a command cannot carry: a slot the command holds outside the push bytes (at most 24
  vertex slots once the vertex function spills, 8 in a window of 128 bytes, above), a fixed-function vertex binding, or a
  vertex layout that must draw from base instance 0 (`DrawInput`).
- Commands draw with the executing encoder's render states whatever the descriptor's other
  `inherit*` properties say. Not built: commands the CPU encodes (`indirectRenderCommandAtIndex:`),
  `copyIndirectCommandBuffer:sourceRange:destination:destinationIndex:`, compute commands, and a
  function's draws other than indexed triangles, which the translator refuses
  (`Unsupported{GpuEncodedIndirectCommands, NotImplemented}`).

## Presenting

The consumer presents on the runtime's own device, in its process: it owns the surface, the
swapchain, the acquires, the presents, their pacing and the window's recovery, and the runtime
renders into the swapchain's images. Nothing is exported, imported or copied.

- `Device::new_drawable_texture` (unsafe) is the texture a `CAMetalDrawable` answers, over one of
  the swapchain's images, which the runtime never destroys. The consumer gives the swapchain's
  create info and a texture descriptor: a 2D texture of one level of the swapchain's image extent
  (else `Refusal::DrawableDescriptor`, as for a swapchain of more than one array layer), of a pixel
  format the runtime holds in the swapchain's image format (`Refusal::DrawableFormat`;
  `BGRA8Unorm` in `B8G8R8A8_UNORM`), and of a usage whose image usage the swapchain's images have:
  transfer source and destination always, as the CPU and blits reach any Metal texture, and
  sampled, storage and colour attachment for shader-read, shader-write and render-target usage
  (`Refusal::DrawableUsage`, naming what they lack). Its home layout, views, elements of set 1 at
  its index and hazards are a texture's of its usage. That the image is the swapchain's is the
  call's safety contract: the runtime cannot check it.
- `Texture::drawable_acquired` (unsafe) is the runtime's part of `nextDrawable`, called once the
  consumer has acquired the image with a fence (`vkAcquireNextImageKHR`) and waited for the fence.
  The image moves to its home layout in the upload of the next commit, ahead of every command buffer
  committed after the call, as a new texture's does: from undefined the first time, and from the
  present layout after a present, which keeps its texels as Vulkan keeps a presented image's. A
  second acquire before a present is `Refusal::DrawableAcquired`.
- `CommandBuffer::present_drawable` (unsafe) is the runtime's part of `presentDrawable:`: once
  every command of the command buffer has run, wherever it was encoded, the image is in the present
  layout, moved in the last segment's end barrier, and the consumer's binary semaphore is
  signalled, in that segment's submit info beside the progress semaphore. The group with it goes
  to the driver at once. The command buffer is scheduled once that submission has passed
  `vkQueueSubmit2`: its status, its kernel times, its scheduled handlers and `wait_until_scheduled`
  all follow the call, so a present made from then waits on a signal already submitted, as Vulkan
  requires. A present from the command buffer's own scheduled handler is valid, but that handler
  runs on the submit thread, whose later submissions wait while a present blocks: the consumer
  presents from its own thread. The drawable is presented at the commit, the presentation engine's
  until the next acquire. A command buffer may present several drawables, each with a semaphore of
  its own; the same semaphore twice is `Refusal::DrawableSemaphoreRepeated`. A present of a
  drawable not acquired since its last present, or that a command buffer (this one too) already
  presents, is `Refusal::DrawableNotAcquired`; of a texture not over a swapchain's image,
  `Refusal::NotDrawable`. A command buffer dropped uncommitted leaves its drawable acquired. That
  the semaphore is unsignalled with no signal or wait pending is the call's safety contract.
- The consumer's order: one drawable texture per swapchain image; then each frame the acquire and
  its fence, `drawable_acquired`, the encoders into the texture, `present_drawable` with the
  image's semaphore, `commit`, `wait_until_scheduled`, and, where the status is not `Error`,
  `vkQueuePresentKHR` waiting on the semaphore inside `with_present_queue`.

### Destroying a swapchain

`vkDestroySwapchainKHR` needs every use of the swapchain's images complete. The runtime uses an
image in the command buffers that encode into or present its drawable, and in the move home each
`drawable_acquired` queues for the runtime's submission of the next commit. Before the consumer
destroys a swapchain:

1. Every command buffer that encoded into or presented one of its drawables has completed
   (`wait_until_completed` has returned for it), and so has each of the consumer's presents.
2. A drawable acquired since the commit of the last command buffer that presented it (one whose
   presenting command buffer was dropped uncommitted): a command buffer with commands committed
   after the acquire has completed. Its submission, or an earlier one, carried the move home.
3. The drawables' textures over its images are dropped: the runtime records nothing more for
   their images. Their views go once the runtime frees them, which may be after the swapchain:
   Vulkan lets a view outlive its image, as a view's parent is the device.

### Known gaps of presenting

- The move home after an acquire waits on no semaphore: the consumer acquires with a fence and
  waits for it first. A semaphore wait would go in a submission that carries its group's other
  command buffers, holding them behind the window system's release of the image.
- A presenting command buffer that ends without running (`NotRun`, `DeviceLost`) before its
  submission is never scheduled and signals nothing: its status goes to `Error`, its scheduled
  handlers run as it retires, and the image stays acquired, possibly not in the present layout.
  The consumer presents nothing and makes its swapchain again.

## One path for every device

Every difference between devices comes from a Vulkan query: a feature, a property, a format, a
limit or a memory type. The crate holds:

- no vendor, device or driver identity;
- no environment gate;
- no runtime policy switch;
- no check that runs beside the program's work.

The test `tests::one_path` reads the crate's sources and fails on the first three.

## Encoder calls

An encoder call executes no locked instruction on its success path: no atomic read-modify-write,
no mutex and no reference count change. The one exception is a fence: the first time a command
buffer's encoders update or wait on a fence, the command buffer keeps it, which is one reference
count change. Each encoder holds the device once; a command borrows it,
the pipeline state and the bound resources, and counts into the encoder's own tally, which the
device's counters take once, at the encoder's end. What a draw or a dispatch learns from its
bindings (the vertex buffers it binds, the alignments it checks, whether it reads an attachment,
reaches a Managed buffer's device copy or writes what the host reads) it works out again only
after the pipeline state or a binding changes. Whether a declaration of the encoder covers an
attachment its pass holds out of its home layout (`Metrics::unslotted_attachment_reads`) is
settled as each declaration and each move of an attachment is made, and a draw tests that answer
alone. On RADV a recorded command lies in write-combining memory, and a locked instruction between
two of them flushes it: a draw took about 490 ns more on a Radeon RX 7900 XTX with one.

## Floor

The runtime requires Vulkan 1.3 and every extension, feature, limit, queue family and memory type
below, of the instance and of every device. `PhysicalDevice::check` decides for one physical
device, with no unsafe call: it returns what the device offers above the floor
(`floor::Capabilities`), or `floor::BelowFloor`, a typed refusal that names every item the device
lacks (`floor::Missing`: the version, an extension, a feature, a limit, the queue family, a
memory type, a time domain or a set layout). `Device::new` makes the last choice, of memory
types, on the created device, and asks the last item there: that the device supports both of the
runtime's descriptor set layouts, the empty set 0 and set 1 of the arrays
(`vkGetDescriptorSetLayoutSupport`, which Vulkan answers only for a created device; the limits
alone do not promise it). A device that does not is refused with `Error::BelowFloor`,
`Missing::SetLayout`.

On macOS the driver that meets the floor is Mesa's Vulkan driver for Apple GPUs (KosmicKrisp):
Mesa 26.3.1 on an Apple M3 Max offers every item, `nullDescriptor` and both set layouts among them
(job m3-max-47313), `VK_EXT_line_rasterization`'s `bresenhamLines` too (m3-max-60379), and the
conformance suite's rt side runs on it (m3-max-57732). A consumer
ships it and loads it through `Instance::with_drivers`. MoltenVK is
below the floor: it reports no `nullDescriptor`, and its `vkGetDescriptorSetLayoutSupport`, which
answers by a count of descriptors alone, refuses set 1 (`Missing::SetLayout`; MoltenVK 1.4.2, job
m3-max-43653).

### Extensions

Presenting runs on the runtime's own device; nothing is exported or imported.

| Extension | Where | Used for |
|---|---|---|
| `VK_KHR_surface` | instance | presenting |
| `VK_KHR_swapchain` | device | presenting |
| `VK_EXT_robustness2` | device | null descriptors, and image reads out of range answered by the device |
| `VK_EXT_calibrated_timestamps` | device | GPU times on the host clock: the device calibrates its own clock (`VK_TIME_DOMAIN_DEVICE_EXT`) with the host's (`floor::TIME_DOMAINS`) |
| `VK_EXT_line_rasterization` | device | lines drawn by Bresenham's rule, through `bresenhamLines` below; every venue's driver lists this name (RADV, ANV, lavapipe, NVIDIA's, KosmicKrisp, and the UHD 630's Windows driver, a Vulkan 1.3 device) |
| `VK_KHR_shader_maximal_reconvergence` | device, where listed | SIMD groups that reconverge as Metal's do |
| `VK_KHR_shader_subgroup_uniform_control_flow` | device, where listed | SIMD groups' reconvergence where maximal reconvergence is not offered |
| `VK_KHR_shader_quad_control` | device, where listed | a fragment function's group operations over other lanes, through the quad-control feature below |
| `VK_EXT_fragment_shader_interlock` | device, where listed | a fragment function's raster order groups on memory it reads and writes, through the pixel-interlock feature below |
| `VK_EXT_shader_stencil_export` | device, where listed | a fragment function's `[[stencil]]` output as the fragment's stencil reference (`FragStencilRefEXT`); a device without it draws the output by bits (the known gaps); RADV, ANV, lavapipe, KosmicKrisp and the UHD 630's Windows driver list it, NVIDIA's 615 and 620 drivers do not |
| `VK_KHR_portability_enumeration` | instance, where offered | listing portability devices, each checked against the floor as any other |
| `VK_KHR_portability_subset` | device, where listed | Vulkan requires it enabled where a device lists it; the device is asked for the portability features below |
| `VK_EXT_attachment_feedback_loop_layout` | device, with the feedback loop | a draw's read of an attachment of its own pass through a texture, in the feedback layout, with the colour flag every render pipeline takes (below) |
| `VK_KHR_vertex_attribute_divisor` | device, where listed | a vertex layout stepped per instance at a rate above 1, through the step-rate feature below |
| `VK_EXT_vertex_attribute_divisor` | device, where listed | the same, on a device that lists only this extension, which the KHR was promoted from with the same feature and pipeline structures |
| `VK_EXT_external_memory_host` | device, where listed | importing host memory by its address (`Device::import_host_memory`) |
| `VK_KHR_external_memory_fd` | device, where listed | importing a dma-buf by its fd, with the extension below, and exporting a block as an opaque fd or, with it, a dma-buf (`Device::new_exported_block`) |
| `VK_EXT_external_memory_dma_buf` | device, where listed | importing a dma-buf (`Device::import_dma_buf`) or several behind one buffer (`Device::import_dma_bufs`), and exporting a block as one |
| `VK_KHR_external_memory_win32` | device, where listed | exporting a block as an NT handle (`Device::new_exported_block`) |
| `VK_EXT_queue_family_foreign` | device, where listed | the foreign queue family that a dma-buf export's ownership transfers name |
| `VK_EXT_image_drm_format_modifier` | device, where listed | a dma-buf surface's planes in place, as `DRM_FORMAT_MOD_LINEAR` images (`Device::new_exported_surface`) |
| `VK_NV_linear_color_attachment` | device, where listed | an opaque handle's surface planes in place whose `LINEAR` tiling renders as a linear colour attachment alone, through the linear-attachment feature below |
| `VK_EXT_device_fault` | device, where listed | a lost device's fault report, read once at its first loss (`DeviceLoss::fault`), through the fault-report feature below |
| `VK_KHR_maintenance8` | device, where listed | a sample at a texel offset the shader computes, with the offset as the sample's own `Offset` operand, through the optional feature `maintenance8` below |
| `VK_EXT_custom_border_color` | device, where reported from the image | Vulkan requires it of a device that enables the border swizzle below; none of its features is enabled |
| `VK_EXT_border_color_swizzle` | device, where reported from the image | an opaque-black border sampled through a view whose component mapping is not the identity, defined and read through that mapping (Border swizzle, below) |
| `VK_EXT_device_generated_commands` | device, with indirect command buffers | an indirect command buffer's execution, through `deviceGeneratedCommands`, enabled where the device reports it with the limits an execution needs (Indirect command buffers, above) |
| `VK_KHR_maintenance5` | device, with indirect command buffers | the preprocess buffers' usage flag (`VkBufferUsageFlags2CreateInfoKHR`), through `maintenance5`, enabled with the extension above alone |
| `VK_KHR_fragment_shader_barycentric` | device, where reported | a fragment function's `[[barycentric_coord]]`, through the barycentric feature below; `Device::extensions` names it only where that feature is enabled |

### Features

| Feature | Structure | Used for |
|---|---|---|
| `bufferDeviceAddress` | `VkPhysicalDeviceVulkan12Features` | a buffer's `gpuAddress` is its device address, used directly by shaders |
| `shaderInt64` | `VkPhysicalDeviceFeatures` | 64-bit addresses in shaders |
| `runtimeDescriptorArray` | `VkPhysicalDeviceVulkan12Features` | one descriptor array per image class: sampled textures, read-write textures, samplers |
| `descriptorBindingPartiallyBound` | `VkPhysicalDeviceVulkan12Features` | array slots that hold nothing |
| `descriptorBindingUpdateUnusedWhilePending` | `VkPhysicalDeviceVulkan12Features` | writing a new slot while work that uses the array runs |
| `descriptorBindingSampledImageUpdateAfterBind` | `VkPhysicalDeviceVulkan12Features` | the sampled-texture and sampler arrays |
| `descriptorBindingStorageImageUpdateAfterBind` | `VkPhysicalDeviceVulkan12Features` | the read-write texture array |
| `shaderSampledImageArrayDynamicIndexing` | `VkPhysicalDeviceFeatures` | indexing the sampled-texture and sampler arrays by a `gpuResourceID` |
| `shaderStorageImageArrayDynamicIndexing` | `VkPhysicalDeviceFeatures` | indexing the read-write texture array by a `gpuResourceID` |
| `shaderSampledImageArrayNonUniformIndexing` | `VkPhysicalDeviceVulkan12Features` | a `gpuResourceID` that differs between threads |
| `shaderStorageImageArrayNonUniformIndexing` | `VkPhysicalDeviceVulkan12Features` | a `gpuResourceID` that differs between threads |
| `descriptorBindingUniformTexelBufferUpdateAfterBind` | `VkPhysicalDeviceVulkan12Features` | the read texture-buffer array |
| `descriptorBindingStorageTexelBufferUpdateAfterBind` | `VkPhysicalDeviceVulkan12Features` | the written texture-buffer array |
| `shaderUniformTexelBufferArrayDynamicIndexing` | `VkPhysicalDeviceVulkan12Features` | indexing the read texture-buffer array by a `gpuResourceID` |
| `shaderStorageTexelBufferArrayDynamicIndexing` | `VkPhysicalDeviceVulkan12Features` | indexing the written texture-buffer array by a `gpuResourceID` |
| `shaderUniformTexelBufferArrayNonUniformIndexing` | `VkPhysicalDeviceVulkan12Features` | a texture buffer's `gpuResourceID` that differs between threads |
| `shaderStorageTexelBufferArrayNonUniformIndexing` | `VkPhysicalDeviceVulkan12Features` | a texture buffer's `gpuResourceID` that differs between threads |
| `shaderStorageImageWriteWithoutFormat` | `VkPhysicalDeviceFeatures` | writes to array elements of any pixel format |
| `nullDescriptor` | `VkPhysicalDeviceRobustness2FeaturesEXT` | every texture element no texture owns: a nil, freed or unowned index reads zeros, sizes 0 and drops writes, as Metal's nil texture does. The robust buffer access beside it stays off |
| `samplerAnisotropy` | `VkPhysicalDeviceFeatures` | a sampler state's `maxAnisotropy`, which the device's own filter takes |
| `shaderStorageImageExtendedFormats` | `VkPhysicalDeviceFeatures` | a read-write texture of a format Vulkan lists among the storage image extended formats (`RG16Float`, `RG11B10Float`, `R8Unorm` and the rest), declared with its format where a shader reads it; Vulkan 1.4 and Roadmap 2022 require it |
| `samplerMirrorClampToEdge` | `VkPhysicalDeviceVulkan12Features` | a sampler state's `MTLSamplerAddressModeMirrorClampToEdge`; Vulkan 1.4 and Roadmap 2022 require it |
| `shaderImageGatherExtended` | `VkPhysicalDeviceFeatures` | a gather at an offset the shader computes; Vulkan 1.4 requires it |
| `shaderInt16` | `VkPhysicalDeviceFeatures` | `short` and `ushort` in shaders |
| `shaderInt8` | `VkPhysicalDeviceVulkan12Features` | `char` and `uchar` in shaders |
| `storageBuffer16BitAccess` | `VkPhysicalDeviceVulkan11Features` | 16-bit values in buffers |
| `storageBuffer8BitAccess` | `VkPhysicalDeviceVulkan12Features` | 8-bit values in buffers |
| `scalarBlockLayout` | `VkPhysicalDeviceVulkan12Features` | buffer and push-constant blocks laid out as Metal lays them, where std430 cannot |
| `uniformBufferStandardLayout` | `VkPhysicalDeviceVulkan12Features` | uniform blocks in std430 layout |
| `vulkanMemoryModel` | `VkPhysicalDeviceVulkan12Features` | the translator's atomics and barriers |
| `vulkanMemoryModelDeviceScope` | `VkPhysicalDeviceVulkan12Features` | atomics and barriers of device scope |
| `subgroupSizeControl` | `VkPhysicalDeviceVulkan13Features` | a kernel's SIMD groups at Metal's width of 32, where the device lists compute in `requiredSubgroupSizeStages` |
| `computeFullSubgroups` | `VkPhysicalDeviceVulkan13Features` | a kernel's subgroups required full, so each SIMD group is one subgroup |
| `robustImageAccess` | `VkPhysicalDeviceVulkan13Features` | an image write or atomic out of range dropped and a read out of range zero, with no guard in the shader |
| `robustImageAccess2` | `VkPhysicalDeviceRobustness2FeaturesEXT` | an image read or fetch out of range, past the view's levels included, answering Metal's texel (zero, alpha 1 for a format without alpha), with no check in the shader; a texel buffer's read is still checked in the shader |
| `independentBlend` | `VkPhysicalDeviceFeatures` | each color attachment of a render pipeline blended by its own state, as Metal's are |
| `fragmentStoresAndAtomics` | `VkPhysicalDeviceFeatures` | a fragment function's writes and atomics on device memory |
| `logicOp` | `VkPhysicalDeviceFeatures` | a render pipeline's logic operation (private render state), the device's own; without it the state is refused (Known gaps of render encoders) |
| `bresenhamLines` | `VkPhysicalDeviceLineRasterizationFeaturesEXT` | a line drawn by Metal's rule, the pixels whose diamond it exits (`VK_LINE_RASTERIZATION_MODE_BRESENHAM_EXT`), where Vulkan's default rule is the device's own: the rectangle of the line's width on a device whose `strictLines` is set (lavapipe, NVIDIA's driver); Vulkan 1.4 requires it |
| `fullDrawIndexUint32` | `VkPhysicalDeviceFeatures` | an indexed draw's 32-bit indices past `2^24 - 1`; Vulkan 1.4 and Roadmap 2022 require it |
| `imageCubeArray` | `VkPhysicalDeviceFeatures` | a cube array texture and a view as one, a cube array image view; Vulkan 1.4 and Roadmap 2022 require it |
| `depthBiasClamp` | `VkPhysicalDeviceFeatures` | a depth bias's clamp (`set_depth_bias`), the device's own; Vulkan 1.4 and Roadmap 2022 require it |
| `drawIndirectFirstInstance` | `VkPhysicalDeviceFeatures` | an indirect draw's base instance, which the GPU reads from Metal's arguments; Vulkan 1.4 requires it |
| `depthClamp` | `VkPhysicalDeviceFeatures` | the depth clip mode `Clamp` (`set_depth_clip_mode`), a pipeline's depth clamp, the device's own; Vulkan 1.4 and Roadmap 2022 require it |
| `shaderClipDistance` | `VkPhysicalDeviceFeatures` | a vertex or post-tessellation vertex function's `[[clip_distance]]` |
| `tessellationShader` | `VkPhysicalDeviceFeatures` | a post-tessellation vertex function's patches, drawn through the device's tessellator |
| `shaderTessellationAndGeometryPointSize` | `VkPhysicalDeviceFeatures` | a post-tessellation vertex function's `[[point_size]]`; without it the device draws every such point at size 1 |
| `shaderOutputLayer` | `VkPhysicalDeviceVulkan12Features` | a vertex function's `[[render_target_array_index]]`, stored to `Layer` in every pass; every venue's driver reports it (RADV on the 780M, the RX 550 and the RX 7900 XTX, NVIDIA's on the RTX 3090 and RTX 2080, lavapipe, KosmicKrisp on the M3 Max, and the UHD 630's Windows driver, a Vulkan 1.3 device; jobs m3-max-61359 to 61362 and 61366) |
| `multiViewport` | `VkPhysicalDeviceFeatures` | several viewports and scissor rectangles (`set_viewports`, `set_scissor_rects`); every venue's driver reports it (jobs m3-max-48727 to 48731, 48736, 48737 and 61508) |
| `shaderOutputViewportIndex` | `VkPhysicalDeviceVulkan12Features` | a vertex function's `[[viewport_array_index]]`, stored to `ViewportIndex`; every venue's driver reports it (jobs m3-max-48727 to 48731, 48736, 48737 and 61508, and the probes 61359 to 61362 and 61366) |
| `largePoints` | `VkPhysicalDeviceFeatures` | a draw of points at the `[[point_size]]` its vertex function writes, up to the device's `pointSizeRange`; without it a device draws only points of size 1.0; Vulkan 1.4 and Roadmap 2022 require it |
| `dualSrcBlend` | `VkPhysicalDeviceFeatures` | a fragment function's second output for colour attachment 0 (`[[color(0), index(1)]]`) as the blend's second source (`Source1Color`, `Source1Alpha` and their complements) |
| `sampleRateShading` | `VkPhysicalDeviceFeatures` | a fragment function that reads its `[[sample_id]]`, or an interpolant at a sample, run once per covered sample as Metal runs it; Vulkan 1.4 requires it |
| `vertexPipelineStoresAndAtomics` | `VkPhysicalDeviceFeatures` | a vertex or post-tessellation vertex function's writes and atomics on device memory |
| `occlusionQueryPrecise` | `VkPhysicalDeviceFeatures` | a `Boolean` or `Counting` visibility result (`set_visibility_result_mode`), a precise occlusion query's count; Roadmap 2022 requires it, and every venue's device profile reports it |
| `multiview` | `VkPhysicalDeviceVulkan11Features` | Vulkan 1.1 requires it; the translator's multiview forms |
| `shaderDrawParameters` | `VkPhysicalDeviceVulkan11Features` | a vertex function's `[[base_vertex]]` and `[[base_instance]]` as the draw's own `BaseVertex` and `BaseInstance`, so no draw pushes its bases; Vulkan 1.1 requires it where the device lists `VK_KHR_shader_draw_parameters`, Roadmap 2024 of every device |
| `shaderSubgroupExtendedTypes` | `VkPhysicalDeviceVulkan12Features` | SIMD-group operations on 8-, 16- and 64-bit values; Vulkan 1.2 requires it |
| `shaderDemoteToHelperInvocation` | `VkPhysicalDeviceVulkan13Features` | a discarded fragment demoted to a helper, so its quad's derivatives stay defined; Vulkan 1.3 requires it |
| `shaderTerminateInvocation` | `VkPhysicalDeviceVulkan13Features` | a fragment's termination; Vulkan 1.3 requires it |
| `shaderZeroInitializeWorkgroupMemory` | `VkPhysicalDeviceVulkan13Features` | threadgroup memory zeroed at its declaration where the translator zeroes it; Vulkan 1.3 requires it |
| `timelineSemaphore` | `VkPhysicalDeviceVulkan12Features` | submission and completion |
| `hostQueryReset` | `VkPhysicalDeviceVulkan12Features` | a command buffer's timestamp and occlusion queries reset on the host as it retires |
| `synchronization2` | `VkPhysicalDeviceVulkan13Features` | barriers, layout changes and submission |
| `maintenance4` | `VkPhysicalDeviceVulkan13Features` | nothing of the runtime's own; Vulkan 1.3 requires it, and the translator's description states it enabled, so its modules are valid under it (`spirv-val --allow-localsizeid`) |
| `dynamicRendering` | `VkPhysicalDeviceVulkan13Features` | render passes |
| `pipelineCreationCacheControl` | `VkPhysicalDeviceVulkan13Features` | a kernel past the stage bound created from its helper's cache data alone, never compiled in the process (Kernels past the stage bound); Vulkan 1.3 requires it |

### Portability features

A device that lists `VK_KHR_portability_subset` may offer less than Vulkan defines. The floor asks
such a device for these, and device creation enables them; a device that does not list the
extension offers each of them by Vulkan's definition and is not asked.

| Portability feature | Structure | Used for |
|---|---|---|
| `imageViewFormatSwizzle` | `VkPhysicalDevicePortabilitySubsetFeaturesKHR` | a texture view's component mapping: its format's and Metal's swizzle |
| `mutableComparisonSamplers` | `VkPhysicalDevicePortabilitySubsetFeaturesKHR` | the comparison sampler array, which takes each sampler state's sampler as it is made |
| `multisampleArrayImage` | `VkPhysicalDevicePortabilitySubsetFeaturesKHR` | `MTLTextureType2DMultisampleArray` |
| `imageView2DOn3DImage` | `VkPhysicalDevicePortabilitySubsetFeaturesKHR` | a 3D texture's depth plane as a colour attachment: a 2D view of a 2D-array-compatible 3D image |

### Step rates

Metal's per-instance step at a rate above 1 has one form, a vertex binding's divisor. Where the
device lists `VK_KHR_vertex_attribute_divisor` or `VK_EXT_vertex_attribute_divisor` and reports
the feature below, device creation enables them, and the translator builds the step where the rate is within
`maxVertexAttribDivisor`. Elsewhere it refuses the step at creation (`VertexStepRate`). Where
the device does not report `supportsNonZeroFirstInstance` (the EXT has no such property, and
its devices take `firstInstance` 0 alone), a draw through such a state at a base
instance other than 0 is refused (render encoders' known gaps); no other feature or answer
changes.

| Step-rate feature | Structure | Used for |
|---|---|---|
| `vertexAttributeInstanceRateDivisor` | `VkPhysicalDeviceVertexAttributeDivisorFeaturesKHR` | a vertex binding's divisor other than 1 |

### Min-LOD clamps

A sample's `min_lod_clamp` other than 0 has one form, SPIR-V's `MinLod` image operand. Where the
device reports the feature below, device creation enables it, and the translator builds the
sample; elsewhere it refuses the function at creation, an `Error::Translation` naming
`MinLodClamp` and `shaderResourceMinLod` (Known gaps of render pipeline states).

| Min-LOD feature | Structure | Used for |
|---|---|---|
| `shaderResourceMinLod` | `VkPhysicalDeviceFeatures` | a sample's `min_lod_clamp`, the device's own level-of-detail clamp |

### Fill modes

The triangle fill mode `Lines` has one form, a pipeline's `polygonMode` `LINE`. Where the device
reports the feature below, device creation enables it, and the render encoder sets the mode;
elsewhere `set_triangle_fill_mode` refuses it, `Gap::Draw` of `DrawKind::FillMode` (Known gaps of
render encoders).

| Fill-mode feature | Structure | Used for |
|---|---|---|
| `fillModeNonSolid` | `VkPhysicalDeviceFeatures` | the triangle fill mode `Lines`, each triangle's edges drawn as lines |

### Alpha to one

A render pipeline's `alphaToOneEnabled` has one form, the pipeline's own `alphaToOneEnable`. Where
the device reports the feature below, device creation enables it, and the translator builds the
state; elsewhere it refuses the state at creation, an `Error::Translation` naming `AlphaToOne` and
`alphaToOne` (Known gaps of render pipeline states).

| Alpha-to-one feature | Structure | Used for |
|---|---|---|
| `alphaToOne` | `VkPhysicalDeviceFeatures` | a render pipeline's `alphaToOneEnabled`, each sample's alpha 1 after the coverage is taken |

### Quad control

A fragment function whose group operations read, combine or count other lanes has one form: its
quads run whole under `RequireFullQuadsKHR`, helpers included, as Metal runs them. Where the
device lists `VK_KHR_shader_quad_control` and reports the feature below, device creation enables
both, and the translator builds the function; elsewhere it refuses the function at creation, an
`Error::Translation` naming `SubgroupOperations` (Known gaps of render pipeline states).

| Quad-control feature | Structure | Used for |
|---|---|---|
| `shaderQuadControl` | `VkPhysicalDeviceShaderQuadControlFeaturesKHR` | a fragment function's quads run whole for its group operations over other lanes |

### Pixel interlock

A fragment function's raster order group on memory the function reads and writes (a buffer, a
texture or an argument buffer's member) has one form: its body runs inside the ordered pixel
interlock (`PixelInterlockOrderedEXT`), after every earlier fragment of its pixel, as Metal orders
the group. Where the device lists `VK_EXT_fragment_shader_interlock` and reports the feature below,
device creation enables both, and the translator builds the function; elsewhere it refuses the
function at creation, an `Error::Translation` naming `RasterOrderGroups` (Known gaps of render
pipeline states). The interlock chooses no other form: a framebuffer fetch, a stencil output, an
imageblock and a blend the function computes take the forms, or the refusals, they take on a device
without it, and a group on a colour attachment alone takes the fetch's own ordering (Known gaps of
render encoders). No other feature of the extension is enabled.

| Pixel-interlock feature | Structure | Used for |
|---|---|---|
| `fragmentShaderPixelInterlock` | `VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT` | a fragment function's raster order groups on memory it reads and writes |

### Barycentrics

A fragment function that reads `[[barycentric_coord]]` has one form: it reads the device's own
weights, `BaryCoordKHR` or `BaryCoordNoPerspKHR`. Where the device lists
`VK_KHR_fragment_shader_barycentric` and reports the feature below, device creation enables both,
and the translator builds the function; elsewhere it enables neither, and the translator refuses
the function at creation, an `Error::Translation` naming `Barycentrics` (Known gaps of render
pipeline states). So `Device::extensions` names the extension only with its feature.

| Barycentric feature | Structure | Used for |
|---|---|---|
| `fragmentShaderBarycentric` | `VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR` | a fragment function's `[[barycentric_coord]]`, read from the device's own weights |

### Limits

The descriptor limits are what the runtime's fixed arrays need: 65,536 sampled textures, 65,536
storage textures and 65,536 level elements, 65,536 read and 65,536 written texture buffers, and three arrays of 8,000
samplers, one per use (plain, depth comparison, integer texture): 4,000 indices and 4,000 variant
elements each (Set 1 and index grants). A read texture buffer (a uniform texel buffer) counts
against the sampled-image limits and a written one (a storage texel buffer) against the
storage-image limits. Vulkan requires at least 500,000 of each texture and sampler limit and of
the two resource totals on a device that supports `descriptorIndexing`, as every lab device does.
4,000 is Vulkan's minimum `maxSamplerAllocationCount`. Of the lab's devices, the least
`maxPerStageDescriptorUpdateAfterBindSamplers` and `maxDescriptorSetUpdateAfterBindSamplers` is
500,000 (MoltenVK on an Apple M3 Max) and the least `maxUpdateAfterBindDescriptorsInAllPools`
1,048,576 (Intel UHD Graphics 630 on Windows). The compute limits let a threadgroup take
Metal's 1,024 threads in any shape up to 1,024 by 1,024 by 64. `maxColorAttachments` holds
Metal's eight color attachments on every device; Vulkan 1.3 requires 4 of it, and Vulkan 1.4 8.
`maxFragmentOutputAttachments` holds a fragment function's output to each of them; Vulkan 1.3 and
1.4 require 4 of it, and Roadmap 2026 8.
A Boolean limit's minimum is 1,
`VK_TRUE`; `timestampValidBits` is the queue family's. No `maxPushConstantsSize` is asked: the
push window, and the pipeline layout's one push range, is the device's `maxPushConstantsSize` up to
256 bytes (`metal2vulkan::device::DeviceProfile::push_window`), at least Vulkan's least, 128, and a
function's fields past it go to a parameter block whose address the window holds. On a device
below 256 an indirect command reaches fewer vertex slots (Known gaps of indirect command buffers).

| Limit | Minimum | Used for |
|---|---|---|
| `maxPerStageDescriptorUpdateAfterBindSampledImages` | `131072` | the sampled-texture and read texture-buffer arrays |
| `maxDescriptorSetUpdateAfterBindSampledImages` | `131072` | the sampled-texture and read texture-buffer arrays |
| `maxPerStageDescriptorUpdateAfterBindStorageImages` | `196608` | the storage-texture array with its level elements, and the written texture-buffer array |
| `maxDescriptorSetUpdateAfterBindStorageImages` | `196608` | the storage-texture array with its level elements, and the written texture-buffer array |
| `maxPerStageDescriptorUpdateAfterBindSamplers` | `24000` | the three sampler arrays; the lab's least is 500,000 |
| `maxDescriptorSetUpdateAfterBindSamplers` | `24000` | the three sampler arrays; the lab's least is 500,000 |
| `maxSamplerAllocationCount` | `4000` | Vulkan's minimum: live samplers |
| `maxPerStageUpdateAfterBindResources` | `327680` | the four texture and texture-buffer arrays, the level elements among them, in one stage; a fragment stage needs its `maxColorAttachments` more |
| `maxUpdateAfterBindDescriptorsInAllPools` | `351680` | all seven arrays; the lab's least is 1,048,576 |
| `maxComputeWorkGroupInvocations` | `1024` | Metal's most threads per threadgroup |
| `maxComputeWorkGroupSize[0]` | `1024` | a threadgroup's width |
| `maxComputeWorkGroupSize[1]` | `1024` | a threadgroup's height |
| `maxComputeWorkGroupSize[2]` | `64` | a threadgroup's depth; a deeper one runs as one row |
| `timestampComputeAndGraphics` | `1` | GPU times of command buffers |
| `standardSampleLocations` | `1` | Metal's sample positions: those of 1, 2 and 4 samples are Vulkan's standard locations |
| `timestampValidBits` | `36` | GPU times of command buffers |
| `maxColorAttachments` | `8` | every color attachment of a render pass descriptor |
| `maxFragmentOutputAttachments` | `8` | a fragment function's output to every color attachment |

### Queue family and memory types

The runtime submits to the first queue family with `VK_QUEUE_GRAPHICS_BIT` and
`VK_QUEUE_COMPUTE_BIT` (`floor::Capabilities::queue_family`). It allocates buffers from three memory
types, each the first with its flags that a buffer of `floor::BUFFER_USAGE` can use: every buffer
usage Vulkan 1.3 defines, and device addresses. Of the types with Shared's flags, the first that is
also host-coherent is taken where the device has one; where the type taken is not, the runtime
flushes and invalidates Shared memory (Shared memory's coherence). Vulkan guarantees such a buffer a device-local
type, so of the three only Shared and the ring can be missing on a conforming device. Vulkan answers which memory types a buffer can use
only for a created device, so `PhysicalDevice::check` finds a type with each set of flags, and
`Device::new` chooses on the device (`Device::memory_types` answers `floor::MemoryTypes`). A device
without one is refused with `floor::Missing::MemoryType`, which names the flags it lacks and every
memory type the device offers buffers, by index and flags: from `PhysicalDevice::check`, every type
the device lists.

| Memory type | Flags | Used for |
|---|---|---|
| Private | `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` | Private storage, which only the GPU reads and writes |
| Shared | `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT`, `VK_MEMORY_PROPERTY_HOST_CACHED_BIT` | Shared storage, which the host maps and reads |
| ring | `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`, `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT`, `VK_MEMORY_PROPERTY_HOST_COHERENT_BIT` | the parameter ring, which the host writes with no flush and the GPU reads |

Vulkan's device type, not the Shared type's flags, decides a Managed buffer's form: on a device
that is neither integrated nor a CPU, shaders reach a device copy in Private memory (Memory blocks
and buffers). A 780M's Shared type lacks `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`, and its Managed
buffers are their Shared ranges all the same.

On a discrete GPU the Shared type is system memory, as Metal keeps a shared resource on a discrete
Mac's GPU: shaders read a Shared buffer across the host link, and Private and Managed storage put
the GPU's bytes in device-local memory. A kernel reading 64 MiB of a buffer once, in order, took
this GPU time (the median of 10 runs, in each of two passes):

| GPU | Shared | Private |
|---|---|---|
| RTX 3090 | 33,720 µs (2.0 GB/s) | 160 µs |
| RTX 2080 | 33,783 µs (2.0 GB/s) | 340 µs |
| RX 550 | 23,117 µs (2.9 GB/s) | 1,654 µs |
| RX 7900 XTX | 9,320 µs (7.2 GB/s) | 162 to 249 µs |

A kernel reading 64 MiB and writing 64 MiB once, in order (`scale_offset`, one thread a
16-byte vector each way), with both buffers Shared or both Private, took this GPU time (the
mean of two processes' median runs), the rate counting the bytes read and written:

| GPU | Shared | Private | Shared over Private |
|---|---|---|---|
| RX 7900 XTX | 11,528 µs (11.6 GB/s) | 167 µs | 69 times |
| RTX 3090, two hosts | 40,201 and 50,426 µs (3.3 and 2.7 GB/s) | 160 µs | 252 and 316 times |
| RTX 2080 | 50,383 µs (2.7 GB/s) | 343 µs | 147 times |
| RX 550 | 33,172 µs (4.0 GB/s) | 1,653 µs | 20 times |

A 3x3 filter over a 1920x1080 image of `float4` texels in a buffer (`binomial3x3`, one thread a
texel, nine reads and one write) took 2.8 ns a thread with Shared buffers on the RX 7900 XTX, 8.3
on the RX 550, 10.0 and 12.5 on the RTX 3090 and 12.3 on the RTX 2080, against 0.015 to 0.40 ns
with Private: 20 times (RX 550) to 305 times (RTX 3090) the Private time.

Scattered reads widen the gap (RTX 3090: 2.9 s against 5.1 ms for the same 64 MiB). The
device-local host-visible type is no alternative: the host reads it at 18 MB/s (RX 7900 XTX) and
37 MB/s (RX 550), against about 5,500 MB/s from the Shared type, and where resizable BAR is off it
is a 246 MiB heap (both NVIDIA boards above).

The replay streams recorded on a discrete GPU bind no Shared buffer. Their buffers are Managed,
apart from 1.8 to 3% Private, so the host link's rate above is not on their path.

### Chosen by a query

Six features choose between two forms by the device's answer: 16-bit floats, storage-image reads
without a format, 64-bit atomics, 64-bit floats, SIMD groups and run-time texel offsets. Beside
them, a device without the
attachment feedback loop reads attachments through a copy, the one form left there, which is a
known gap (Known gaps of render encoders). `floor::Capabilities` holds each answer the floor
reads, and the features and extensions to enable: the floor's, and each optional one the device
has.

| Feature | Queries | Used for |
|---|---|---|
| SIMD groups | `minSubgroupSize`, `maxSubgroupSize` and `requiredSubgroupSizeStages` (`VkPhysicalDeviceVulkan13Properties`); `subgroupSupportedStages` and `subgroupSupportedOperations` (`VkPhysicalDeviceVulkan11Properties`); `shaderMaximalReconvergence` (`VK_KHR_shader_maximal_reconvergence`) and `shaderSubgroupUniformControlFlow` (`VK_KHR_shader_subgroup_uniform_control_flow`) | the translator's choice of how a kernel's SIMD groups run (its tier), and their reconvergence mode: maximal where offered, else subgroup-uniform control flow where offered |

The other six are one feature each:

| Optional feature | Structure | Used for |
|---|---|---|
| `shaderFloat16` | `VkPhysicalDeviceVulkan12Features` | `half` arithmetic in 16 bits |
| `shaderStorageImageReadWithoutFormat` | `VkPhysicalDeviceFeatures` | reads of read-write textures of any format; where it is false, each format's `VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT` answers |
| `shaderBufferInt64Atomics` | `VkPhysicalDeviceVulkan12Features` | 64-bit atomics on buffers |
| `shaderSharedInt64Atomics` | `VkPhysicalDeviceVulkan12Features` | 64-bit atomics on threadgroup memory |
| `shaderFloat64` | `VkPhysicalDeviceFeatures` | 64-bit floats |
| `shaderMaximalReconvergence` | `VkPhysicalDeviceShaderMaximalReconvergenceFeaturesKHR` | SIMD groups that reconverge as Metal's do |
| `shaderSubgroupUniformControlFlow` | `VkPhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR` | SIMD groups' reconvergence where maximal reconvergence is not offered |
| `maintenance8` | `VkPhysicalDeviceMaintenance8FeaturesKHR` | a sample at a texel offset the shader computes, with the offset as the sample's own `Offset` operand; where it is false, the sample branches over the offset's values, each branch with its value as `ConstOffset`, which costs compile time (Known gaps of pipeline states) |

`sparseBinding` (`VkPhysicalDeviceFeatures`) chooses no form: device creation enables it only
where the device offers imports over several dma-bufs (`Device::import_dma_bufs`, Imported
blocks), and only that call makes a sparse buffer. Elsewhere the call refuses.

### Attachment reads

A draw that reads, through a texture, an attachment its own pass renders (Metal's read of a
render target) is built where the device's descriptor limits hold the sampled-texture array's
second elements beside the floor's arrays: 65,536 sampled images more than the limits above name
(`maxPerStageDescriptorUpdateAfterBindSampledImages`,
`maxDescriptorSetUpdateAfterBindSampledImages`, `maxPerStageUpdateAfterBindResources` and
`maxUpdateAfterBindDescriptorsInAllPools`); a device without them refuses such a draw
(`Gap::Draw` of `DrawKind::AttachmentRead`), and is otherwise the same
(`floor::Capabilities::attachment_reads`). The draw reads through the attachment feedback loop
where the device also lists the extension marked "with the feedback loop" above and reports the
feature below, which device creation then enables, and neither otherwise
(`floor::Capabilities::feedback_loop`). A device that reads attachments without them reads
through a copy of the image into its shadow (Render encoders), a known gap. AMD's Windows driver
is expected to take the loop, not verified in the lab (public reports of it list
`VK_EXT_attachment_feedback_loop_layout` from 2.0.250 on).

| Attachment-read feature | Structure | Used for |
|---|---|---|
| `attachmentFeedbackLoopLayout` | `VkPhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT` | the feedback layout and usage |

### Border swizzle

Where the device lists `VK_EXT_border_color_swizzle` and reports `borderColorSwizzleFromImage`,
device creation enables the extension and both its features, with `VK_EXT_custom_border_color`,
which the extension requires (none of its features), and none of them otherwise
(`floor::BORDER_SWIZZLE_EXTENSIONS`). No form differs: no sampler's create info chains
`VkSamplerBorderColorComponentMappingCreateInfoEXT` on any device, since a sampler in the arrays
meets every view and that structure names one mapping. With the feature, an opaque-black border
sampled through a view whose component mapping is not the identity (a texture's swizzle, or the
mapping its pixel format takes) reads through that mapping, as Metal's border reads through the
swizzle. Without it Vulkan leaves that texel undefined (Known gaps of textures).

| Border-swizzle feature | Structure | Used for |
|---|---|---|
| `borderColorSwizzle` | `VkPhysicalDeviceBorderColorSwizzleFeaturesEXT` | Vulkan requires it of a device that lists the extension; enabled with the one below |
| `borderColorSwizzleFromImage` | `VkPhysicalDeviceBorderColorSwizzleFeaturesEXT` | an opaque-black border through a swizzled view, read through the view's mapping with no mapping on the sampler |

### Linear attachments

Where the device lists `VK_NV_linear_color_attachment`, device creation enables it, and its
feature below where the device reports it, which Vulkan requires of a device that lists it. It
adds no form: an exported surface's plane takes one of the surface's two forms by the device's
queries, and with the feature an opaque handle's `LINEAR` image whose format's linear tiling
features carry `VK_FORMAT_FEATURE_2_LINEAR_COLOR_ATTACHMENT_BIT_NV` in place of the colour
attachment bit holds the plane in place, rendered only beside other such planes (Exported
surfaces). The RTX 3090 and the RTX 5080 Laptop GPU (NVIDIA 620.70.06) list it; their linear
tiling carries that bit, not the colour attachment bit, for every plane format measured, and so
does their `DRM_FORMAT_MOD_LINEAR` modifier for each of those formats it is listed for.

| Linear-attachment feature | Structure | Used for |
|---|---|---|
| `linearColorAttachment` | `VkPhysicalDeviceLinearColorAttachmentFeaturesNV` | a `LINEAR` surface plane rendered as a linear colour attachment |

### Fault reports

Where the device lists `VK_EXT_device_fault`, device creation enables it, and its feature below,
which Vulkan requires of a device that lists it. It adds no form and nothing beside the work: at
the device's first loss of its own whose result is `ERROR_DEVICE_LOST`, the runtime reads the
driver's report once (`vkGetDeviceFaultInfoEXT`) into `DeviceLoss::fault`. The RTX 3090 (NVIDIA
620.70.06) lists it; after an MMU fault (Xid 13) of a read through a freed buffer's address it
reported the read's page and an instruction pointer.

The report places each address it gives (`DeviceFault::places`, one line each by `FaultPlace`'s
`Display`) among the runtime's own ranges a shader reaches by device address: the pool's
allocations, the parameter ring's blocks, imports, exports, surfaces' buffers and imports over
several dma-bufs, and indirect command buffers' preprocess regions. It names the ranges that meet the address, live or among the last 1,024 freed
(each free's time, the command buffers then in flight and, for the pool, the submission its free
waited for), and the nearest range below and above. Each range is noted as it is made and as it
is freed; nothing else runs before the loss. Images, which Vulkan gives no device address, and the
zero source and staging memory, which only copies reach, are not placed.

| Fault-report feature | Structure | Used for |
|---|---|---|
| `deviceFault` | `VkPhysicalDeviceFaultFeaturesEXT` | the fault report of a lost device |

### Not enabled

The translator's profile reads each feature below where a device enables it, and device creation
never enables it, so the translator takes the form without it on every device: one form, or a
typed refusal. Every other feature the profile reads is enabled where the device reports it, and
the translator's description of the device states every feature and extension device creation
enables that the translator names.

| Feature | Structure | Without it |
|---|---|---|
| `computeDerivativeGroupLinear` | `VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR` | a derivative in a kernel is refused (`ComputeDerivatives`) |

The test `tests::feature_enables` holds this to the code: on every committed device profile, and
under `--cfg m2v_cpu_icd` on the machine's CPU device, it derives the translator's profile with and
without each feature the device reports that creation does not enable, and fails on one that
changes the profile and is in neither this table nor `Enabled, not stated`, and on a row whose
feature creation enables.

### Enabled, not stated

Device creation enables each item below where its own rule says, and the translator's description
of the device never states it, though the translator names it. Every other feature and extension
creation enables that the translator names, the description states.

| Item | Kind | Why it is not stated |
|---|---|---|
| `maintenance5` | feature (`VkPhysicalDeviceMaintenance5FeaturesKHR`) | enabled with indirect command buffers alone. Stated, the translator would hold `A8Unorm` in `A8_UNORM` where a device has them and in `R8_UNORM` read as `(0, 0, 0, r)` where it does not, a second form chosen by a query; it is held in `R8_UNORM` on every device (Known gaps of textures) |
| `VK_KHR_maintenance5` | extension | as `maintenance5` |

`tests::feature_enables` holds this table to the code: an item creation enables that the
translator names reaches the description unless a row here names it, and a row names an item
creation enables, that the translator names, and that the description does not state.

The test `tests::floor` holds these tables equal to the code, checks that each missing item is
named, and, under `--cfg m2v_gpu` or `--cfg m2v_cpu_icd`, checks every device the machine's
Vulkan loader lists.
