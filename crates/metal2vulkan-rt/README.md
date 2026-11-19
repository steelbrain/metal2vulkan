# metal2vulkan-rt

The Metal API on Vulkan, with one path for every device. It replaces `metal2vulkan-runtime` and
takes its name when it does; until then it is unpublished.

Its public API so far: the instance and the device, memory blocks and the buffers in them, the
grants of texture and sampler indices, textures, sampler states, command queues, command buffers
and events, compute and blit encoders, the device's counters, and `floor`, the check of a Vulkan
device against the floor below.

## What the consumer implements

Six of Metal's objects and getters have no counterpart here by design. The consumer implements
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
  `MTLPipelineOption`; its consumer builds reflection from the library's AIR. An option not
  named here is not the consumer's by this line.
- `MTLTexture`'s `parentTexture`, `parentRelativeLevel`, `parentRelativeSlice` and `shareable`.
  `Texture::new_texture_view` keeps the texture whose image the view reads and no getter, and no
  texture here is shareable; the consumer keeps the texture it made a view of and what it shared.
  A view's descriptor (its type, format, levels, slices and combined swizzle) is this crate's.

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
  cold, so a consumer should pass a directory.
- `Device::raw` gives the instance, physical device, device and queue family handles, and
  `Device::with_present_queue` runs the consumer's presents on the presenting queue under its lock.
  The runtime keeps owning every handle.
- `Device::has_unified_memory` is true where Vulkan's device type is integrated or CPU.

### Teardown

The runtime runs four threads per device: the block-maker, the submit thread, the completion
thread, and the reaper, which frees the semaphores and command pools the others give up, one at a
time, so no driver free runs on the completion thread and a submission waits behind at most one,
lowers the compute pipeline states' variants an encoding call waits for (the emulated-SIMD-group
and the partial-threadgroups ones), and makes the device's pipeline cache when it starts and writes
it to the cache directory once pipeline creation is quiet. While an old generation of the cache
directory is removed, a thread of the store's own does it, and ends with the removal; the teardown
waits for it. `Gauges::threads` holds the OS id of each of these threads while it runs, so a tool
can tell the runtime's own work from the threads a driver starts for itself.
A device is torn down when its last handle, and the last handle of every object made from
it, is dropped; a committed command buffer and the blocks the consumer holds do not keep it alive.
Teardown stops the submit thread, which submits what is ready and ends what it holds with
`CommandBufferError::NotRun`; the completion thread retires the rest as the GPU completes it and
runs the handlers; the threads are joined and the GPU goes idle; `BlockMappings::unmap` runs for
each of the pool's allocations the consumer mapped. Teardown runs on the thread that drops the
last handle, or,
where that is a handler (scheduled or completed) or a listener, on the reaper once the handler
returns. A call that holds a
handle of its own gives it up before another thread can drop the last: `commit` gives up the
command buffer's before the command buffer can complete. Then every block is freed and the
`VkDevice` destroyed, once the runtime's last internal reference is gone: on the thread that tore
down, or where a call still held one as it returned, in that call, which runs no consumer code
there. Every gauge reads zero only once the `VkDevice` is destroyed, so a consumer whose last
handle dropped in a handler waits for that before it drops its instance or exits. A physical device and a device each hold their instance. No thread of the runtime outlives its device, and nothing relies on a
static destructor or an `atexit` handler: the consumer drops every handle, or calls
`Device::quiesce`, before the process exits, since a driver may unmap device memory inside
`exit()`.

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
and the draws and dispatches whose push bytes the command buffer already held);
`gauges()` are live counts per kind of resource, every one zero after teardown. They are relaxed
atomics, always on. What an encoder's commands count arrives when the encoder ends.

## Memory blocks and buffers

Memory comes in blocks the consumer takes and places buffers in, as a placement heap's buffers sit
in its memory.

- `Device::take_block(class, len)` gives a block of `MemoryClass::Private` (device-local, no host
  mapping) or `MemoryClass::Shared` (host-visible, host-coherent and host-cached, mapped for the
  block's life) memory, as a `Block`: rt's own handle, with no Metal counterpart, which derefs to
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
  thread makes a class's next allocation of 16 MiB, off every caller's path, while the class's free
  bytes are fewer than 8 MiB, and at no other time: a take that leaves more free sets nothing off
  beside the consumer's next calls. A take that no free range fits allocates on the calling
  thread; the pool keeps 8 MiB ahead per class. A driver may hold the device's other calls for the whole of an
  allocation, longer for more bytes: on an RTX 3090 (NVIDIA's Linux driver) a pipeline state's
  creation beside a Shared allocation's making took 6 to 8 ms at 16 MiB and 40 to 45 ms at 256
  MiB; on an RTX 2080 a compute pipeline's creation waited out the whole 3.4 ms of a 16 MiB Shared
  allocation made beside it. `Metrics::consumer_allocation_nanos` and
  `consumer_allocation_max_nanos` count the time of allocations made on the consumer's threads,
  and `maker_allocation_nanos` those of the block-maker thread.
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
  each block (texture blocks among them), each texture's own allocation, the staging region (from
  the first texture copy on), the staging buffer of a copy larger than the region while it runs, the
  parameter ring's blocks and the zero source. One more past it is `Limit::MemoryAllocations`,
  refused before the driver is called; `Gauges::blocks` counts those live. The limit is about 4.29
  billion on RADV, NVIDIA's Linux driver and ANV, 8,401,685 on Intel's Windows driver for the UHD
  Graphics 630, and 4,096 at Vulkan's minimum.
- A block's bytes are undefined, new or taken again: the runtime zeroes nothing ahead of need,
  and nothing the driver or the memory type does counts. A buffer reads zero where
  `Device::new_buffer` zeroes its range (below), as Metal zeroes a buffer from
  `newBufferWithLength:options:`, and Metal owes zeros for that call's buffers alone: not for a
  heap's memory or a texture's, whose bytes are undefined, as Metal is silent on them. The
  consumer passes `zero_from` for a buffer Metal zeroes, or zeroes the range at its own carve,
  once.
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
  `BlockMappings::unmap` of it, whatever the device's queues are doing, so a program that keeps
  committing gives back what it stopped using within 250 ms plus the frees ahead of it. It frees
  between the allocations it makes, never beside one. Each class keeps a reserve that the idle
  rule does not free: the newest idle 16 MiB allocation with free bytes while the class's free
  bytes need it to stay at 8 MiB or more. An idle allocation's bytes are all free, so the reserve
  is one 16 MiB allocation of each class at most. A long allocation, which a take longer than
  16 MiB made for itself, is never the reserve.
  `Metrics::allocations_freed`
  counts the frees and `Metrics::idle_frees_by_age` their idle times; `Metrics::idle_reuses_by_age`
  counts idle allocations taken again, by how long they were idle. `Gauges::shared_idle_bytes` and
  `Gauges::private_idle_bytes` count the idle bytes, the reserve's included, and
  `Gauges::shared_idle_age_nanos` and `Gauges::private_idle_age_nanos` the age of the oldest idle
  allocation the rule will free; `Gauges::shared_reserve_bytes` and `Gauges::private_reserve_bytes`
  count the reserve's bytes, and `Gauges::shared_allocation_bytes` and
  `Gauges::private_allocation_bytes` the bytes of each class's live allocations, idle ones
  included. Once the last handle is gone the block's
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
  Private one by a GPU fill of exactly the range, recorded into the upload, which goes ahead of
  every command buffer committed after the call and ends with a barrier that orders the fill
  before every later command (a copy into the buffer among them), and holds the block until it
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
  which `contents` answers, in a block the host maps (`Refusal::ManagedBlockNotShared` for a
  Private one). Vulkan's device type picks its form, as `Device::has_unified_memory` answers it.
  On an integrated or CPU device the buffer is its bytes, as a Shared buffer is, its `gpu_address`
  theirs, `did_modify_range` does nothing, and `copy` is `None` (`Refusal::ManagedCopyOnUnified`
  otherwise). Elsewhere shaders reach a device copy at `gpu_address`, always: `length` bytes at
  `copy`, a Private block the consumer holds and an offset in it, placed by the consumer as it
  places a Private buffer (`Refusal::ManagedCopyMissing` for none, `Refusal::ManagedCopyNotPrivate`
  for a Shared block). Either offset is any byte, and `gpu_address` is the block's address plus
  it. The copy's rest past its first `copied` bytes is zeroed in the upload, and those bytes are
  named as `did_modify_range` names a range.
  `Buffer::did_modify_range(start, length)` names a range changed (`Refusal::ModifiedRange` past
  the buffer), merged with the buffer's others where they overlap or touch, and submits nothing.
  The next commit of a command buffer with commands, on any queue, takes the bytes of every range
  named since: on the committing thread, before `commit` returns, it copies each once into a
  64 MiB staging ring of the Shared type (made at the device's first `did_modify_range`, on its
  thread, so no commit waits for the allocation; never zeroed, as each span is written before it
  is copied), and the copies from the ring into the device copies go
  ahead of that command buffer's commands, after every earlier command on the device, which the
  barrier ending each earlier command buffer orders; the copies record one barrier, after them. So
  the bytes are taken at the commit; Metal takes them shortly after, before the command buffer is
  scheduled. A write or a `did_modify_range` made while a command buffer runs does not reach it.
  Commits take turns while they copy, so command buffers keep the order of their captures. A
  commit that finds the ring full waits for its oldest copy's submission to complete. Where that
  copy is not yet submitted, the held groups go at once; once every command buffer committed
  before is submitted, the commit's own ring-full goes in a copy submission of its own, which the
  commit waits for before it reuses the ring, so the ring is all the staging a commit holds,
  whatever it names, and the wait runs at the GPU's copy speed. Where a command buffer committed
  before waits on the host instead (an event wait the host has not met, or a command buffer
  enqueued ahead of it and not committed), nothing can go before the host acts: the rest of the
  commit's bytes go into Shared staging blocks of its own from the pool (at most 256 MiB each),
  which its command buffer gives back as it retires, the pool's idle rule then deciding when their
  memory is freed. Every byte is still the commit's. `Metrics::managed_bytes_captured`,
  `capture_waits`, `capture_streams`, `capture_overflows` and `capture_overflow_bytes` count them,
  `capture_copy_nanos` and `capture_wait_nanos` time the commits' host copies and waits, and
  `Gauges::capture_bytes` counts the ring and the staging blocks alive. A command buffer
  keeps each device copy it uses, and so the copy's block, until it retires.

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
  it goes out, with the allocation's number, class, length, device address and host mapping. It
  runs where the allocation is made: in `Device::new` for each class's first, on the block-maker
  thread for one made ahead of need (`Metrics::allocations_mapped`), and inside the take, on the
  consumer's thread, for a take that no free range fits (`Metrics::allocations_mapped_on_takes`).
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
with its allocation, and the block-maker thread zeroes free Shared bytes while the consumer's
mappings of them stand: from a block's give-back to the next take that lands on its bytes, they are
the runtime's. A guest kernel that writes a free range damages only its own later buffers.

### Known gaps of memory blocks

- A shader uses a buffer's `gpuAddress` directly, unchecked: a stale one, of a released buffer, can
  still fault the device.
- A dispatch whose indirect arguments, or an indexed draw whose indices, start at a byte of their
  block that is not a multiple of 4, or of the index size, is refused (`Refusal::ArgumentPlace`).
  Metal's offset rules keep both at that multiple of the buffer's start, so only a buffer that
  starts off it meets this: a no-copy buffer over a pointer that is not a multiple of 4 (indirect
  arguments) or of the index size (indexed draws), which documented Metal does not allow, its
  pointer being page-aligned. The consumer places every other buffer at an aligned byte. Such a
  buffer is still made and its other uses run; only the dispatch or draw is refused.
- On the NVIDIA cards we measured in Thunderbolt enclosures, freeing host-visible memory takes
  about 1.1 ms per MiB and holds any submit of the same process for its length; on an internal
  NVIDIA card the same free takes about 0.5 ms per 16 MiB. rt frees one idle allocation at a time,
  at its idle age, whatever the queues are doing, so a submit can meet a free and wait for the
  rest of it. `Metrics::allocations_freed` counts the frees, and
  `Metrics::submissions_during_frees` the submits made while one was in progress.
- NVIDIA's Linux driver also holds a submit made while another thread is inside
  `vkAllocateMemory` until the allocation returns, device-local memory included. A take that
  leaves its class fewer than 8 MiB free sets the block-maker making the class's next allocation
  at once, so the take's own first submission can wait out that allocation. On an RTX 3090
  (driver 620.70.06, rt at ac1896537f) a 16 MiB Private allocation of 661 µs held the submit of
  the first command buffer to read a new 16 MiB Private buffer for 603 µs; on an RTX 2080 one of
  647 µs held it for 579 µs; the other submits took a median of 2 to 3 µs. A submit waits at most
  the rest of one allocation, and the pool makes one per 16 MiB the consumer takes, about what the
  take would spend allocating on its own thread.
- A commit that fills the capture ring waits for each ring-full's copy into the device copies,
  which crosses the card's link to the host. On an RTX 3090 in a Thunderbolt 3 enclosure whose
  link runs at 20 Gb/s (rt at ac1896537f), a ring-full of 63.3 MiB took 33.4 ms on the GPU (1.99
  GB/s) of the 33.6 ms the commit waited; the runtime's own part (its submission, the queue ahead
  of the copy and the commit's wake) was about 0.2 ms. In one consumer's replay stream on that
  card, 258 ring-fulls took 7.3 s of a 98 s play. Over the internal link of an RTX 5080 Laptop GPU
  the same kind of copy, 64 MiB from Shared memory into a device copy, took 3.9 to 4.5 ms on the GPU
  (15 to 17 GB/s, rt at 5ad1438f7b), about an eighth of the time.
- NVIDIA's Linux driver allocates host-visible memory at about 0.14 ms per MiB on an RTX 3090 and
  0.2 ms per MiB on an RTX 2080, in every form we tried (plain, with a device address, dedicated,
  cached or not), against about 0.03 ms per MiB for device-local memory on the 3090. A take longer
  than the pool's 16 MiB allocations allocates on the calling thread and pays it there: in one
  consumer's replay stream on the 3090, six Shared takes of 547 MiB took about 70 ms each, and its
  Shared takes past 16 MiB took 1.46 s of the stream's 210 s. The pool keeps no memory to avoid
  it; `Metrics::consumer_allocation_nanos` counts it. On the 3090 (driver 620.70.06, rt at
  140714ceae), 64 MiB of rt's Shared memory type costs about 12.5 ms in `vkAllocateMemory`, with
  or without a device address, and 2.75 ms more for rt to zero it through its mapping. The old
  runtime imports host pages it mapped itself instead: about 5.8 ms for 64 MiB, 5.4 ms of it the
  import. Importing fresh pages by host pointer takes 0.35 to 0.54 times as long as allocating
  them, from 16 to 256 MiB on both cards. No host-visible type outside the card's 246 MiB
  device-local heap is cheaper, and a Shared allocation is the driver's own memory on every
  device, so the import is not open to it. In another consumer's replay stream on the 2080 (the
  same driver, rt at c810d2a3df), the first 1138 frames make 304 Shared takes of 9.5 GiB, whose
  allocations took 0.90 to 0.93 s in two plays, under 1% of the play. On three loading frames of
  1.3 to 3.6 s they are 6 to 8.3% of the frame (at most 301 ms of 3.64 s); in the play that
  listed every frame, they are at most 1.7% of any other.
- On a full discrete card a Private block the consumer takes goes to another memory type, host
  memory where the card has no other device-local type, and the kernels that read a Managed
  buffer's device copy placed there read it across the host link:
  `Gauges::managed_spilled_bytes` counts those buffers' bytes while they live.

## Set 1 and index grants

Shaders reach textures and samplers by index (`docs/RESOURCE-ID.md`). Every pipeline takes set 1,
one descriptor set the device makes at creation, of five arrays whose element `i` is the object
whose `gpuResourceID` holds index `i`: 65,536 sampled textures (binding 0), 65,536 storage
textures (binding 1), and 4,000 samplers of each use, plain, depth comparison and integer texture
(bindings 2, 3 and 4). A texture's index is the same element of both texture arrays: one with a
storage view takes both, any other the sampled one alone, and its storage element stays null. Every binding is partially bound, updated after bind and
updated while unused elements are pending, so an object's element is written once, as it is
made, with no lock, and never rewritten for layouts.

- Where the device reads attachments (Floor, Attachment reads), binding 0 holds 131,072
  elements: element `i + 65,536` is texture `i`'s second element, which a draw pushes in place of
  the texture's own index while it reads an attachment of its pass through the texture. Through
  the attachment feedback loop it is the texture's view in the feedback layout, written with
  element `i` for a texture whose image a pass may move there (a shader-read colour render target
  with no storage view); on a device without the loop it is the texture's view of its image's
  shadow (Render encoders), written at the first draw that reads through it where the texture
  still owns index `i`. It is a null descriptor otherwise, and no grant reaches it.
- A shader masks the index of a texture ID it reads from an argument buffer to its low 16 bits,
  so a bad ID there reads the element at `index & 0xFFFF`, never one outside the arrays. A slot's
  index is the runtime's own push, unmasked, and no argument buffer's ID reaches the feedback
  elements.
- Every element no object owns is defined. A texture element is a null descriptor
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
  `Limit::IndexRange`, and one holding an index granted already `Refusal::IndicesGranted`.
  `Device::release_indices` returns a grant; one not granted is `Refusal::IndicesNotGranted`, and
  one where an object still holds an index `Refusal::IndicesInUse`.
- An index has one owner: the object placed there last, in each class. Making an object at an
  index another object holds, a live one or one the program released that a command buffer not
  yet retired still holds, takes the index at once, as Metal's later texture at a resource ID
  takes the ID: the new object writes its elements (a texture with no storage view writes its
  storage element null where it took the index from another, so nothing stale answers there),
  and the earlier object's going, at its release or once its command buffers retire, touches
  neither the index nor its elements. No work in flight may read a displaced or released object
  through the index: the program's duty, as it is Metal's, which the runtime does not check. A
  sampler state or constexpr sampler whose device samplers equal the displaced one's is safe under
  work in flight: equal create infos share the device's `VkSampler`s, the arrays keep the samplers last written at each index, and
  an element that holds its sampler already is not written, so the placement changes no
  descriptor that pending work reads (`Metrics::sampler_element_writes` counts the elements
  written). A different sampler state stays the program's duty.
  Making one at an index not granted is `Refusal::IndicesNotGranted`. Where nothing took it, the
  index is free again once its owner's last handle is gone, and with it the last command buffer
  that used it; its elements read their default again from then.

## Textures

- `Device::new_texture(descriptor, index)` is `newTextureWithDescriptor:` at `index`: an
  optimal-tiling image of the descriptor in device-local memory, its contents undefined, as Metal's
  are. The image is placed in a texture block: a Private block the device takes from its pool, at
  the first place with room at the image's alignment, so a texture makes no allocation. It has an
  allocation of its own only where the device requires one for it (`requiresDedicatedAllocation`;
  preferring one is no reason), where it is larger than a Private block (64 MiB), or where the
  blocks' memory type does not serve it. A place is free again once the image is freed; one empty
  texture block is kept, and another that empties goes back to the pool. `index` is given exactly
  where shaders reach the texture, shader-read or shader-write usage (`Refusal::TextureIndex`
  otherwise), from a live `IndexClass::Texture` grant, whatever the usage. `TextureUsage` with no
  bits is Metal's unknown
  usage: shader read and write, render target and pixel-format view.
- The image's usage is the descriptor's plus transfer source and destination. Its home layout is
  general where it has a storage element, shader-read-only where shaders only read it, and the
  attachment layout where only render passes use it. Its elements of set 1 are written before the
  call returns, with that layout, and never rewritten. The image moves from undefined to home in
  the upload of the next submission, ahead of every command buffer committed after the call, or
  ahead of its first copy; until then nothing holds it, so a texture released first is freed at
  once, with nothing submitted.
- A texture's image lists the formats Metal views the texture as that the device holds in a
  viewed form of the image format's compatibility class, and is then mutable
  (`VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT` with a `VkImageFormatListCreateInfo`, its own format
  first): its sRGB sibling whatever its usage, as Metal views a texture as its sibling without
  `PixelFormatView` usage, and with that usage every uncompressed colour format of the same bytes
  a texel. A texture of no such format is not mutable. A view of another format takes the image's
  usage that its format's features hold (`VkImageViewUsageCreateInfo`).
- The sampled element's view is the format's, through its component mapping with the
  descriptor's swizzle composed over it; a depth or stencil format's view takes its depth, else
  its stencil. The storage element is written where the usage has shader write and the device
  stores the format through a storage image declared without a format; its view maps each
  channel to itself.
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
  `TextureSampleCount`, `TextureSwizzleWithWrites`). Texture buffers (`Gap::TextureBuffer`),
  cube array textures (`Gap::CubeArrayTexture`: their view needs Vulkan's `imageCubeArray`, which
  the floor does not require, and no replay stream makes one) and Memoryless storage
  (`Gap::MemorylessTexture`) are not built.
- `Texture::gpu_resource_id` is its index (0 without one) and the translator's `TextureWord` of
  its format and swizzle (`docs/RESOURCE-ID.md`), so the guest computes the ID itself.
- `Texture::replace_region` and `Texture::get_bytes` take a Shared or Managed texture
  (`Refusal::TextureBytesPrivate` otherwise), a region inside a level and slice on the format's
  block grid (`Refusal::TextureRegion`), and rows and images apart as Metal's calls take them,
  a compressed format's rows being rows of blocks (`Refusal::TextureBytesLayout` where they
  overlap or the bytes are too few). `replace_region` copies the bytes into the staging region
  before it returns, and the upload copies them into the image, leaving home for the transfer
  layout and coming back, ahead of every command buffer committed after the call. `get_bytes`
  records the copy out into the upload, submits it at once, waits for it and copies the bytes
  out; as in Metal, the program has waited for the work that writes the texture.
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
  alias one another and the heap's buffers. A placement past the heap's end, an offset in the
  block off the image's alignment, or a heap range past its block is refused
  (`Refusal::HeapTexturePlace`); the image is never bound outside its heap. The texture holds
  the block, so a released block goes back to the pool only once its textures' images are gone.
- `Buffer::new_texture(descriptor, index, offset, bytes_per_row)` is
  `newTextureWithDescriptor:offset:bytesPerRow:`: a texture whose texels are the buffer's bytes
  from `offset`, rows `bytes_per_row` apart, of the buffer's storage mode. Any pitch of whole texels at least a row long is taken; rows closer than a row or not a
  whole number of texels apart, an offset that is not a whole number of texels, or texels past
  the buffer are refused (`Refusal::BufferTextureLayout`). Its image is the device's own linear
  image, bound at the texture's place in the bytes shaders reach (a Managed buffer's device copy,
  where it has one), made preinitialized so its move home keeps the bytes, and kept in the general
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
- Where the device's linear image does not take the texture (another pitch, an offset off the
  image's alignment, a format or usage the linear tiling lacks, a spilled block), the texture is
  an optimal image of its own, and the device copies the bytes shaders reach into it and back,
  one `vkCmdCopyBufferToImage` or `vkCmdCopyImageToBuffer` region a texture, the buffer's rows
  `bytes_per_row` apart, with no CPU copy. Each copy counts in `Metrics::over_copies` and its
  bytes in `Metrics::bytes_copied`. When is decided by the buffer's storage:
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
  - Over a Managed buffer's device copy, only at Metal's own Managed events, each texture keeping
    one flag of each kind: in, after a `did_modify_range` over its range, by the commit that
    captures the range, after its copies into the device copy; and where device work wrote the
    buffer (a binding whose compiled interface writes it, a blit or fill into it), by the upload
    that leads the next submission using it; both from the device copy. Back, where device work
    wrote the texture, before the next reader of the device copy (a binding that reaches the
    buffer, a blit from or into it, a copy into another texture over a range that meets its, a
    `synchronize_resource`, a capture of a range over it, ahead of its copies): a
    reader recorded after the write makes it in its own commands, and the upload that leads a
    submission, which runs before the submission's commands, only once the write's segment was
    submitted; that submission also owes each other texture over a range that meets its a copy
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
- The staging region is 16 MiB of the Shared type in an allocation of its own, made at the first
  copy and kept for the device's life, so no copy allocates memory. Its bytes are never zeroed:
  each copy's span is written, by the host or by the copy, before anything reads it. It is a ring: each copy takes a span at a multiple of 256 bytes,
  free again once the submission that carries the copy completes and no `get_bytes` reads it. The
  upload is submitted, without a wait, once its copies' spans take 8 MiB, so the GPU copies while
  the host writes the next. Where the ring has no room, the call waits for its oldest span's
  submission, submitting the upload first where the span is in it (`Metrics::upload_waits`); where
  another call still writes that span's bytes, or holds them for a `get_bytes` to read, it sleeps
  until that call releases the span (`Metrics::staging_sleeps`). A
  copy larger than 16 MiB has a staging buffer of its own, which counts against the upload's
  64 MiB. `Gauges::staging_bytes` counts the region once made and each such buffer until the
  upload that copies through it is gone.
- `Texture::new_texture_view(pixel_format, texture_type, levels, slices, index)` is
  `newTextureViewWithPixelFormat:textureType:levels:slices:`, and with the texture's own type and
  every level and slice `newTextureViewWithPixelFormat:`. A view is a texture over the parent's
  image with image views and elements of set 1 of its own, at its own index: no image, no memory
  and no copy (no submission, block or copied byte). It keeps its parent, and its descriptor is
  what Metal's view reports: the extent of its first level, its levels and its slices. Copies,
  `replace_region` and `get_bytes` through it reach the parent's levels and slices under it, and a
  view of a Managed texture names the parent's bytes of them; its hazards are its parent's. Built:
  the parent's own pixel format, over every slice in the parent's type, a slice subrange of a 1D or
  2D array, or one slice of a 2D array as a 2D texture, each over any level subrange; a view of a
  view is a view of the first parent. Refused, as Metal's validation stops them: an empty range of
  levels or slices, or one past the parent's (`Refusal::TextureViewRange`), and another pixel format
  of a texture made without `PixelFormatView` usage (`Refusal::TextureViewFormat`).
- `Device::pixel_format_info(raw)` is `MTLPixelFormatGetInfoForDevice` (Apple SPI): Metal's
  description of a raw pixel format value, its name, Apple's flags word, cast class and four
  words (a block's bytes and extent for a format of blocks), as an Apple M3 Max answers it under
  macOS 27.0.1, `PixelFormatInfo::INVALID` for a value it describes no format for.
  `Device::texture_level_info(query)` is `MTLGetTextureLevelInfoForDeviceWithOptions`: Metal's
  layout of one level of a texture, its 13 words, computed from that description. Both answer the
  same on every device and ask Vulkan nothing: a format described there may be one this device
  makes no texture of (`Gap::TextureFormat`), and the flags are Apple's GPU's.
- A texture holds the device. Once its last handle and the last command buffer that used it are
  gone, its elements read null and its index is free at once; the reaper frees its image once the
  last upload that moves or copies it has retired, and a view's image views once the view is
  gone. `Gauges::textures` counts images and views until then, `Gauges::descriptor_slots` the
  indices held.

### Known gaps of textures

- `replace_region` and `get_bytes` of a texture without named bytes go through the staging region:
  a copy Metal does not make, counted in `Metrics::bytes_copied`.
- A texture whose format the device does not store through a storage image without a format has
  no storage element, so a shader write through its index is dropped, as a write to a nil
  texture is.
- `replace_region` takes no bytes into a 32-bit float depth (`Gap::FloatDepthUpload`): Metal
  keeps any `float` there, Vulkan copies no depth outside `[0, 1]` into a depth aspect without
  `VK_EXT_depth_range_unrestricted` (`VUID-vkCmdCopyBufferToImage-pRegions-07931`), which the floor
  does not require, and no recorded stream uploads one.
- Their bytes are not copied for a format of depth and stencil both or a multisample texture
  (`Gap::TextureBytes`).
- `Device::pixel_format_info` answers as an Apple M3 Max under macOS 27.0.1, which describes
  none of the values from 630 to 645 (`BGRA10Unorm`, its packed, sRGB, PQ and HLG forms,
  `R10UintX6`, `RG10UintX12` and the rest); an Apple M4 under that release describes them.
- A texture's memory is device-local; where the device has none left, the call fails
  (`Error::Vulkan`).
- A texture over a buffer other than 2D of one level, sample and slice, or of a depth, stencil
  or compressed format, is not made (`Gap::TextureOverBuffer`).
- A texture over a buffer whose pitch, place, memory type or format features the device's
  linear image does not meet is copied between the buffer and an image of its own: the copies
  Metal does not make, counted (`Metrics::over_copies`). A program that rounds its pitch past the
  device's own (`Device::minimum_linear_texture_alignment`), or offsets its texture off the
  image's memory alignment, takes them: NVIDIA lays a linear image's rows at the row
  rounded to 128 bytes, AMD to 256, so a pitch rounded to 256 is copied on NVIDIA. A render
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
  another texture type (but a 2D view of one slice of a 2D array), and some of a cube's faces or a
  cube array's cubes. A view of a pixel format Metal does not view the texture as is
  `Refusal::TextureViewFormat`.
- A swizzle reaches a texture's sampled element alone, as Apple defines it for reads and
  samples. A swizzled texture, or a view made with a swizzle or of a swizzled texture, has a
  storage element and attachment views that map each channel to itself, as Vulkan requires of
  a storage image and an attachment: a write and a render pass store the texels unswizzled.
  A draw or dispatch whose function declares `access::read_write` at a slot it reads, bound to
  such a texture, is `Gap::SwizzledReadWrite`, found when the pipeline state or a binding
  changes: its read would answer the texel unswizzled, where Metal reads it through the swizzle. A texture's
  own descriptor takes a swizzle only without shader-write and render-target usage
  (`Refusal::TextureSwizzleWithWrites`), so this reaches views alone.
- Textures over a swapchain image are not built yet. Heap textures are placed only in Private
  heaps (`Gap::HeapTexture` for a Shared block).

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
- A state holds the device. Once its last handle and the last command buffer that used it are
  gone, its elements read the defaults again and its index is free; a device sampler no other
  state holds goes to the reaper.
- Anisotropy is the device's own filter (`samplerAnisotropy` is in the floor). A cubic filter is
  `Gap::CubicSampler`. A descriptor the translator makes no device sampler of is
  `Error::Translation`.
- A state bound with LOD clamps (`set_sampler_state_with_lod_clamps`, and the render encoder's
  `set_vertex_sampler_state_with_lod_clamps` and `set_fragment_sampler_state_with_lod_clamps`:
  Metal's `lodMinClamp:lodMaxClamp:` forms, whose range forms are loops of them) is bound as it
  is where its samples read alike with its clamps replaced by these: each bound, held to `[0,
  15]`, is the state's own so held (`metal::sampler::lod_clamps_read_alike`). A sample's level
  lies in `[0, 14]` (Metal's largest textures have 15 levels), so a clamp of 1000 over a state's
  `FLT_MAX`, which a replay stream binds, reads as the state's own. Clamps that read otherwise
  are `Gap::LodClamps`, and nothing is bound.

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
  command buffer completes or it sees the event; a segment that wrote only Private memory leaves
  the host out. No segment starts with a barrier of its own; a blit copy records its own two
  (see "Blit encoders").
  `enqueue` takes the command buffer's place in its queue; `commit` hands it to the submit
  thread and returns. A committed command buffer holds no device handle.
- The submit thread passes what is ready to `vkQueueSubmit2` in the device's commit order, each
  queue's command buffers in groups: a submission holds at most sixteen command buffers of one
  queue, ahead of them the upload (zero fills, texture copies and new textures' moves to their home
  layouts), and only its last submit info signals the progress semaphore. Consecutive command
  buffers that no event wait or signal separates share one submit info, the upload with the first:
  the GPU runs them in order either way, and a driver may pay for each info (the UHD 630's paid
  about 160 us apiece). `Metrics::submit_infos` counts them. The upload records at most 4,096 calls
  and keeps at most 64 MiB on its own account: the staging buffers of copies larger than the staging
  region, and the textures and blocks released while it held them. A call that finds it past either
  submits it and waits for it to complete before recording, so nothing grows without a commit. While
  fewer than three of a queue's submissions are not yet seen complete, each of its command buffers
  goes at once, alone. While three are, the ones committed meanwhile are held in one group, each
  scheduled as it joins, and the group goes when it holds sixteen, when a completion leaves fewer
  than three in flight (the completion thread wakes the submit thread as it retires; no commit or
  wait is needed), when the host is to wait for one of them (`wait_until_completed`, or
  `CommandQueue::send_now`, the hint for a wait the runtime does not see), or at teardown. Waiting
  for a place in the queue sends nothing. A segment that signals an event ends its group, which goes
  at once; one that waits on an event's semaphore leads a new group, the one held going first. The
  driver runs a submission's command buffers as one kernel job, which a job timeout ends as a whole
  (Linux's amdgpu ends a job past its timeout and resets the ring), so a group is bounded by its
  expected GPU run, as the old runtime's are: a queue's expected run follows its command buffers'
  own GPU times as each retires (a longer one raises it at once, a shorter one halves the way toward
  it), and a command buffer joins a group only while the group's expected runs, summed, stay within
  16 ms. One whose queue has no run yet, or that would take the sum past the bound, goes alone, the
  group held going first. The bound is tested again as the group goes, with the expected run as it
  then stands: a group past it goes member by member, one `vkQueueSubmit2` each, only the last
  signalling. At most four command buffers join a group while none of the queue's retires (four, the
  old runtime's bound on a group), since they join on an expected run no completion has updated: in
  a replay stream on an RTX 3090, the most GPU time over four consecutive command buffers of a queue
  was 385.1 ms, within a quarter of amdgpu's default job timeout of 2,000 ms, and over sixteen
  1,536.6 ms. The thread holds a segment whose event waits are not met, and its queue's later
  command buffers behind it, while the segments before it run, so the host can see a signal the
  command buffer makes before its wait. Other queues go on. `Scheduled` means the last segment is in
  a submission: passed to `vkQueueSubmit2`, or in a group held. A command buffer whose segments hold
  no commands, no event wait and no event signal takes no submission: the submit thread schedules it
  as it reaches it, and it completes with its queue's command buffer before it, in flight, or at
  once where that one has completed. Where that one is in a group held, it joins the group, which
  goes at once, so its completion waits only on submitted work.
- A command buffer's scheduled handlers run on the submit thread, once it is scheduled and before
  its submission goes to the driver (one that ends without a submission: once it ends, before it
  retires), with no lock of the runtime's held. In the handler its status is `Scheduled` and its
  kernel times are set, and it runs before the command buffer's completed handlers. The queue's
  later submissions wait while it runs, and it may run while another command buffer's completed
  handler runs on the completion thread. A scheduled handler may read the command buffer's status
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
  turn. A wait in one for what that thread has not yet done is `Refusal::WaitOnOwnThread`, and one
  already met answers at once: `wait_until_completed` for a command buffer not yet completed (its
  own handler's command buffer has), `wait_until_scheduled` for one not yet scheduled,
  `wait_until_signaled_value` for a value not yet reached (a timed wait too: it is refused, never
  false at its timeout, since the thread that would observe the value is the one waiting), and
  `CommandQueue::command_buffer` on a queue whose maximum count are made and not yet completed,
  which a scheduled handler answers the same way. No wait of the runtime's hangs by construction.
  A command buffer of a group is seen complete with the group's last: up to the GPU time of the
  fifteen after it late. Only a queue already three submissions behind holds a group; one whose
  submissions have completed sends its next command buffer alone. Without a wait or a hint, the lateness is bounded only while three submissions are
  in flight, and the completion that leaves fewer sends the group (the measured lateness is under
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
  timestamp queries, one written where its first commands begin and one where its last segment
  ends, read once it completes with no wait, and placed on the host clock by the device's last
  calibration: one `vkGetCalibratedTimestampsEXT` call reading the GPU's clock and the host's
  together, made as the device starts and again at a completion once 100 ms have passed since
  the last, never for each command buffer. A command buffer that took no submission writes no
  query: its `gpu_start_time` and `gpu_end_time` are one time, never 0: the latest `gpu_end_time`
  of its queue's command buffers before it, or its `kernel_end_time` where that is later, the shape Metal
  reports for a command buffer that does no work. A command buffer's GPU times so fall between
  `host_time()` before its commit and after its completion, within the calibration's deviation
  and the clocks' drift over 100 ms. The two queries are a pair from the device's timestamp
  pools, which a command buffer takes as it records its first commands and gives back as they are
  read at its completion, or as it is dropped uncommitted. A pool holds 4,096 queries, the pairs
  of 2,048 command buffers; the device makes one only once every pair of its pools is taken, and
  keeps it until the device drops (`Gauges::timestamp_pools`), so a queue makes no Vulkan object
  and `new_command_queue` fails on nothing but a count of 0. Each pool is reset on the host as it
  is made (`hostQueryReset`), and a pair as its command buffer retires, those of the two its
  submissions wrote, so no command buffer the GPU runs resets a query. A command buffer with no
  commands, or whose last segment has none, records a `VkCommandBuffer` for its timestamps. One
  that ends with an error has GPU times of 0.
- `Event` and `SharedEvent` are timeline semaphores whose values are Metal's values. A host set
  (`SharedEvent::set_signaled_value`) raises a floor and never signals the semaphore; a wait the
  floor meets is submitted with no semaphore wait, and a GPU signal not above the highest value
  submitted or set is left out, so the semaphore only rises and never races a host set.
  `signaled_value` is the higher of the semaphore and the floor. `notify_listener` runs its
  listener on the completion thread when the value is reached, at a signal in the middle of a
  command buffer as well as at its end; `wait_until_signaled_value` wakes then too.
- `Fence` is `MTLFence`, and `update_fence` and `wait_for_fence` on compute and blit encoders are
  `updateFence:` and `waitForFence:`. They record nothing: a command buffer's encoders run in
  their order and a queue's command buffers in their commit order, and each encoder's writes reach
  the commands after it through the barrier at its end or at its segment's end, so a wait after an
  update already sees them. Work with no fence between it is ordered as well, where Metal may run
  it at once. A fence keeps the first queue whose encoder uses it.
- `CommandQueue::abandon` ends every command buffer of the queue held at an event wait or not yet
  scheduled with `NotRun` and runs its handlers, as a Mac drops a dead process's held work, each
  retiring in the queue's order. Segments already submitted still complete, and so does a group
  held, which those it ended join and which goes at once; the queue stays usable.
- At a loss of the device (or a failed submission, which the runtime takes as one), the submit
  thread ends what it holds, the completion thread ends what is in flight once the device is idle
  (at once on a lost device), host waits for pending signals end false, and every later commit
  ends at once. `CommandBufferError::DeviceLost { in_flight }` tells the command buffers that had
  passed `vkQueueSubmit2` from the rest. Teardown then completes, and a new device can be made on
  the same instance.

### Quiesce and exit

`Device::quiesce(timeout)` is for a process that exits with the device alive. With every wait
bounded by the timeout, it stops the submit thread (ready work is submitted, held work ends
`NotRun`) and the block-maker, waits for the GPU to complete what was submitted, lets the
completion thread retire it and end, and then runs `BlockMappings::unmap` on the calling thread for
every allocation still mapped. It frees nothing: every block's memory and mapping stays valid until
the process ends. After it, no runtime thread touches a block; later calls that need a thread
return `Error::Quiesced`. It answers true once the GPU went idle and every thread ended in time,
and false at once in a handler or a listener, whose thread cannot wait for itself. A second call
returns at once. After a false answer a runtime thread can still be in the driver (the completion
thread in `vkWaitSemaphores` with no timeout where the GPU did not go idle, which on NVIDIA's
driver lasts until a later submission that after quiesce the runtime never makes, or in its
retirement calls once a handler that held it returns, and the reaper or the block-maker in a
pipeline creation or a memory call that outlasted the timeout), so a process that gets false ends
without running exit handlers (`_exit`): a known gap on NVIDIA.

The consumer quiesces before it calls `exit()`. Quiesce can also run in an `atexit` handler, on
whichever thread calls `exit()`: it joins a thread only once that thread has ended, reads no
thread-local that `exit()` tears down, and takes no lock a consumer's thread can hold. Such a
handler is a fallback, not the contract: `exit()` runs handlers in the reverse order of their
registration, so it runs before the driver's own exit teardown only where it was registered after
the driver loaded, and no driver promises when it registers. On NVIDIA's driver (an RTX 2080 and
an RTX 3090), a handler registered after `Device::new` exited cleanly in 50 runs of 50; one
registered before `Instance::new` crashed in 50 of 50. The handler parks the consumer's own threads
that touch blocks first.

### Known gaps

- Command buffers committed while an earlier one of their queue runs are reported complete
  together, at the last one's end, where Metal reports each at its own. Measured with the harness
  row `long_4cb` (four command buffers committed together, two runs in flight): on an RTX 2080,
  at about 12 ms of GPU time each, the third was reported with the fourth, about 12 ms after its
  own end, and the second 0.8 ms after its own at the median (25 ms at the 90th percentile); on an
  RX 550, at about 7 ms each, the third about 7.7 ms late and the second 0.9-1.4 ms at the median
  (15 ms at the 90th). The last command buffer's own report is not late: on the 2080 it came
  97.2-97.5 ms after its commit, where the old runtime's came 94.9-96.4 ms after. The harness prints each command buffer's lateness, from
  its `gpu_end_time` to its completed handler (`late_us`).
- On NVIDIA, short command buffers committed back to back reach the GPU in more submissions than
  the old runtime made of them, and such a burst runs behind it. The cost is NVIDIA's latency per
  submission: the runtime submits sooner, each command buffer as soon as it can. On an RTX 2080
  (harness row `frame_5cb`: five short command buffers a run, two runs in flight) a run takes
  about 2.5 submissions against the old runtime's 2.0, each submission boundary leaves the GPU
  idle for about 20-25 µs, and the row's throughput is about 8% behind. Latency from idle (one
  run in flight) is equal. A recorded replay stream on the same card plays 1.5 times as fast as on
  the old runtime; it was recorded before the recorder's completion fix, so that figure compares
  the two runtimes on the same stream and no more.
- One command buffer whose own GPU run passes the driver's job timeout still ends there: the
  runtime does not cut a command buffer's work into shorter jobs, and joins none to it (the
  grouping bound above). Linux's amdgpu ends a job past its timeout, 2 s by default (Linux 7.3.3),
  and resets the ring, where Metal runs the command buffer within its own watchdog's longer bound
  (the old runtime's known gap in `docs/FAST-PATHS.md` names the same limit).
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
  work's end. Measured on an M3 Max with Mesa 26.3.1, in Vulkan alone, around a 16 ms dispatch:
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
  thread reads each command buffer's two timestamps before its completed handlers run. A short
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
  does not isolate processes, and the old runtime does not either: measured, one guest process
  read and overwrote another's Shared buffer, Private buffer and texture through addresses and
  resource IDs it made up, with no fault. The consumer treats every byte on the device as
  guest-writable and keeps nothing there that it parses or trusts.
- On NVIDIA, a GPU fault in a guest's kernel hangs that command buffer's completion, where Metal
  ends it with an error. A fault (an access to an address nothing maps) loses the device on NVIDIA's
  driver and on RADV, and one device serves every guest process, so every guest's later work ends
  `DeviceLost`; a second Vulkan device, such as another process's, keeps running. NVIDIA's driver
  (620.70.06; an RTX 2080 and an RTX 3090) tells only a later submission or `vkDeviceWaitIdle`: the
  faulting submission's semaphore never signals, and an unbounded wait for it had not returned after
  60 s. The hang lasts until the device's next submission on any queue: that submission meets the
  loss, the completion thread wakes, and everything ends as a loss. Where something submits on the
  device every frame, the hang lasts about a frame and then every guest process sees the device
  lost; it lasts indefinitely only when nothing else submits. RADV (a 780M and an RX 7900 XTX)
  signals the semaphore within about 2 s and refuses the next submission; Intel's Windows driver (a
  UHD 630) does so at once; ANV (an Iris Xe) takes the store with no fault. On RADV (a 780M), the
  command buffers of the submission a reset cut can report completed, since the reset signals its
  semaphore before any call reports the loss: the loss is reported by the next call, where Metal
  ends the faulting command buffer with an error. The old runtime took a store through a pointer to
  an address nothing maps, and one 64 TiB past a buffer, without a fault on all six GPUs, since it
  does not hand kernels raw addresses; the new runtime refuses the first kernel and took the second
  without a fault. A bad address that the old runtime survived ends the
  GPU's work for every guest process under the new runtime on NVIDIA and RADV: the consequence of
  one device and direct addresses, which the owner accepted.
- An index past a set 1 array's length is a fault of the same kind. A kernel indexes the arrays
  with a resource ID's index as the guest gave it, with no clamp: an index of 65,536 or more used
  through a texture array, or 4,000 or more through a sampler array, reads a descriptor from past
  the array. Vulkan leaves that undefined,
  and Metal leaves a resource ID that names no live resource of the access undefined; here it can
  fault as a bad address does, with the same consequences. An index within an array that no live
  object owns reads its null descriptor or default sampler, with no fault.
- Presenting (the binary semaphore a command buffer signals for a present) comes with presenting.
- A fence orders the encoders of one queue: an encoder of another queue that updates or waits on
  it is refused (`Gap::FenceOtherQueue`), as Metal's fence orders one queue and `MTLEvent` orders
  queues.
- An indirect dispatch's threadgroup counts are the GPU's to read, so a count above the device's
  `maxComputeWorkGroupCount` is not refused; Vulkan leaves such a dispatch undefined. A direct
  dispatch past it is split into dispatches of at most that many, each told its first
  threadgroup.
- `setBytes` takes up to 64 KiB, one ring chunk (`Limit::BytesLength`).
- A buffer argument bound to nothing reads the nil buffer: 64 KiB of zeros, one ring chunk the
  device keeps from its first such dispatch or draw, its address the argument's pushed value, as
  Metal's nil buffer reads zeros. Reads past the 64 KiB and writes through a nil argument aren't
  covered.
- A dispatch's threadgroup memory, the kernel's own and the lengths set, each rounded up to 16
  bytes, is at most the device's `maxComputeSharedMemorySize` (`Limit::ThreadgroupMemory`), where
  Metal's is its `maxThreadgroupMemoryLength`. A kernel's access past its threadgroup memory is
  undefined, as in Metal, and the runtime does not confine it.

## Compute encoders

- `CommandBuffer::compute_command_encoder(DispatchType)` is `computeCommandEncoderWithDispatchType:`.
  While it is open, the encoder holds the command buffer's recording: another encoder, an encode
  call, `enqueue` and `commit` are refused (`Refusal::EncoderOpen`). `end_encoding`, or dropping
  the encoder, gives it back.
- `set_texture` (128 slots, `Refusal::TextureArgumentIndex` past them) and `set_sampler_state`
  (16 slots, `Refusal::SamplerArgumentIndex`) bind as Metal's calls do; `use_resource` takes a
  texture as it takes a buffer. A dispatch pushes, for each texture and sampler slot its kernel
  reads, the bound object's `gpuResourceID` (0 for none), whose index names its elements of set
  1; for a texture slot an integer write saturates through, the channel widths of the bound
  texture's format (`metal::format::integer_channels_word`); and for each constexpr sampler it
  samples through, the index the consumer gave it at the state's creation. Nothing is written
  into set 1 at bind or dispatch time, and every texture stays in its home layout.
- `set_stage_in_region` (`setStageInRegion:`) sets the region a kernel's `[[grid_origin]]` and
  `[[grid_size]]` read, pushed at each later dispatch. A dispatch of a kernel that reads it on an
  encoder that set none is `Gap::NoStageInRegion`: what Metal's kernel reads then is not
  established. A kernel with a `[[stage_in]]` argument, which fetches by the region, is
  `Gap::KernelInput` at its state's creation.
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
  a buffer or a texture being one resource, in a table indexed by
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
- Parameters: fields in the push window, `[0, 256)` on every device, are pushed: the dispatch
  region and up to 28 buffer addresses, each kernel pushing only the bytes it reads. Fields past it
  go to a 16-byte aligned block in the parameter ring, whose address is a field in the window: a
  kernel with more than 28 buffers. `Counters::metrics`' `ring_writes` counts each block and each
  `set_bytes` copy.
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
  reaches the level's edge on both sides; a 3D texture's image copies into or out of a 2D
  slice. A size of 0 on an axis a texture lacks copies one texel there; on
  another axis the copy copies nothing.
- A copy records two barriers. The first moves each texture's level and slice from its home
  layout to the transfer layout after every earlier command, and makes every earlier write visible
  to every later command; it pays a barrier an encoder owed, and the hazard sets start again. The
  second moves them home once the copy has run, its writes visible to every later command. A copy
  so needs no hazard tracking and leaves none: a command buffer of one copy has three barriers,
  the segment's end among them. The encoder owes nothing at its end.
- Refused: what Metal's validation refuses, textures of two pixel formats or two sample counts
  (`Refusal::TextureCopyMismatch`), and a region outside either texture or off its block grid
  (`Refusal::TextureRegion`). Not built (`Gap::TextureCopy`): what one `vkCmdCopyImage` in the
  floor does not make (a multisample texture, a format of depth and stencil both, one pixel format
  held in two Vulkan formats by the textures' usages, a 1D texture with another type), and one
  level and slice of a texture into itself.
- `copy_from_buffer` is `copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:`
  `sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:` and
  `copy_from_texture_to_buffer` is `copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:`
  `sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:`, a
  `BufferLayout` holding the offset and the two strides; their options forms with no option are
  the same calls. Each records `vkCmdCopyBufferToImage` or `vkCmdCopyImageToBuffer` against the
  bytes shaders reach (a Managed buffer's device copy, where the device keeps one), one region an
  image of the region, each at its own offset, so an image stride of no whole number of rows
  copies too. The bytes between the region's rows and
  images are kept. It records the same two barriers as a copy between textures; the second also
  makes a write into the buffer visible to every later command, and a write into a Shared buffer
  marks the segment, whose end barrier then reaches the host. A command buffer of one copy into a
  Shared buffer has three barriers.
- Refused (`Refusal::BufferCopyLayout`), as Metal's validation stops them: an offset or row
  stride of no whole number of a format's texels, rows or images closer than the region's, and a
  region past the buffer's end. Metal reads no row stride of a region one row high and no image
  stride of a region one image deep, and neither does the runtime. Not built
  (`Gap::BufferCopyAlignment`), though Metal takes them: an image's offset in the `VkBuffer`
  shaders reach that is no multiple of a texel block's bytes, or of 4 for a depth or stencil
  aspect, and a row stride of no whole number of a compressed format's blocks; also what `replace_region` and
  `get_bytes` refuse (`Gap::TextureBytes`, and `Gap::FloatDepthUpload` into a texture).
- `copy_from_buffer_to_buffer` is `copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:`:
  one `vkCmdCopyBuffer` region between the bytes shaders reach (a Managed buffer's device copy,
  where the device keeps one), between the same two barriers as a fill, a copy into a Shared
  buffer marking the segment as a copy into one does. A size of 0 copies nothing. Refused, as
  Metal's validation stops them: a range past either buffer's end (`Refusal::BufferCopyRange`),
  and on macOS an offset or a size that is not a multiple of 4 (`Refusal::BufferCopyUnaligned`).
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
  between each level and the next, and come back in the last. A texture of one level records
  nothing. Refused (`Refusal::MipmapFormat`), as Apple defines mipmaps of colour-renderable,
  filterable formats alone: a depth or stencil, compressed, YCbCr or integer format. Not built
  (`Gap::MipmapFilter`): an image the device's blits do not filter linearly (its format query),
  one that holds its pixel format through a conversion (extended-range, YCbCr), and a view in
  another format than its image's.
- The slice-and-level and whole-texture forms of `copyFromTexture`,
  `synchronizeTexture:slice:level:` (no recorded stream calls it), and the other blit commands
  are not built yet.

## Compute pipeline states

- `Device::new_compute_pipeline_state(&ComputePipelineDescriptor, sampler_ids)` is
  `newComputePipelineStateWithDescriptor:options:reflection:error:` without reflection. The
  translator compiles the kernel on the calling thread, which waits for it, within its 20-second
  budget per lowering, for what device creation enabled and under its identity domains: each
  buffer argument is the guest address of the bound buffer plus its offset, and each texture and
  sampler argument the bound object's resource ID, pushed; set 1 is the only descriptor set a
  kernel reads. A refusal of the translator is `Error::Translation`.
- rt builds no pipeline reflection and acts on no `MTLPipelineOption`; its consumer builds
  reflection from the library's AIR. The options it leaves to the consumer are
  `MTLPipelineOptionFailOnBinaryArchiveMiss`, `MTLPipelineOptionArgumentInfo`,
  `MTLPipelineOptionBufferTypeInfo` and `MTLPipelineOptionBindingInfo`.
- What a state compiles is its content, never the state: its descriptor's kernel with its
  constant values, linked functions and dynamic libraries, the descriptor's fields that change
  the compile (the translator's `Device::compute_pipeline_key`), and its
  `requiredThreadsPerThreadgroup`. Every state of one content shares one compiled kernel: its
  translation, its shader modules and their `VkPipeline`s, its emulated-SIMD-group variant and
  its precision notes. The device's map holds the last 1,024 contents used, so a state of a
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
- Every kernel is lowered once at creation. The translator answers, lowering nothing
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
    by 1 where it is one deep, on one `VkPipeline` made with the state, and of its tile (below)
    where it is deeper, on one `VkPipeline` per tile, made at the kernel's first dispatch at that
    tile, which waits for it, each thread's position the workgroup's origin plus its place in it,
    with no decode. Those pipelines are kept with the kernel's others, at most one per tile of the
    45 a rule's tile can be. A tile deeper than 64 runs as one row of its 256 threads (below);
  - a grid of any other such kernel in tiles: workgroups of 256 threads on one `VkPipeline`, made
    with the state, each one tile of the grid, which its region carries. A thread's position is
    the tile's origin plus its index masked and shifted by the tile's sides, with no division.

  A tile has power-of-two sides whose product is 256. The translator's `pipeline::grid_tile`
  picks a tiling kernel's from the grid's extents alone: launching the fewest threads past the
  grid, then with the fewest tiles partly past it, then the shallowest, then (where the grid has
  two rows or more) the height nearest 2, then the widest. An image-only kernel's tile is picked
  from the grid and the call's threadgroup size: the fewest threads launched past the grid, then
  the fewest tiles partly past it, then the depth nearest the threadgroup's (the shallower of two
  as near), then the most square in `x` and `y`, then the wider. So threadgroups of 8x8x4 over a
  grid that 8x8x4 tiles divide run in 8x8x4, and threadgroups one deep in the square 16x16x1,
  the pipeline made with the state, wherever 16x16x1 tiles divide the grid. Where the workgroups
  do not divide the grid, the dispatch's regions are flagged ragged (the translator's
  `NONUNIFORM`), and an invocation past the grid returns before the kernel's body: only there does
  a thread compare its position with the grid's last thread, inside one branch on the flag. Every
  other dispatch, on the same pipeline, branches past that test. An indirect dispatch runs at its
  own threadgroup size, unflagged. A kernel that observes its partition is
  lowered plainly: its whole grids, `dispatch_threadgroups` and indirect dispatches run that
  module at their own size, and a `dispatch_threads` grid that is not a whole number of
  threadgroups runs its partial-threadgroups variant, which the reaper lowers for the first such
  grid, which waits for it (a first-use wait). One that holds no threadgroup memory (which any
  variant of it observes the partition through) and runs a lockstep barrier where each SIMD group
  lies within one row (`CompiledFunction::runs_lockstep_barriers_in_rows`) has its one-row variant
  of partial threadgroups asked too (`LoweringInputs::one_row`, whose lockstep order leaves out the
  barriers of pairs that meet only where a SIMD group spans two rows). A kernel whose lockstep
  barriers all order such pairs is not asked: its own module runs them only where its threadgroup
  size lets a SIMD group span two rows, a test the driver folds, so the variant would leave out
  nothing more. Only the order's own barriers are left out: a
  kernel that observes the partition otherwise (a barrier of its own, where a thread may read
  what another row's thread stored before it; a fence or an atomic of threadgroup or SIMD-group
  scope; a SIMD-group or quad function; a builtin of the partition) observes it in that variant
  too, and keeps its own size. Where the translator answers that the variant does not observe
  the partition, it is lowered with the state, its tiles' `VkPipeline` made with
  it, and a direct dispatch whose threadgroups keep each SIMD group within one row (none has more
  than one row, or every one's rows are a multiple of 32 threads wide, the grid's edge included)
  runs it in tiles, as a kernel that does not observe its partition runs one; any other dispatch
  runs the plain module at its own size, its order kept.
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
  size and threadgroup memory lengths, and one more per size for grids that are not whole
  threadgroups, with one for its one-row module's tiles where it has that module; for any
  kernel's indirect dispatches, one per size and lengths.
- A kernel whose SIMD groups run one per subgroup of 32 (pinned where the device lists compute in
  `requiredSubgroupSizeStages` with 32 between its sizes, or the device's own 32), or one per
  32-lane cluster of a 64- or 128-lane subgroup, runs only workgroups of whole subgroups.
- Such a kernel also has an emulated-SIMD-group variant, whose SIMD groups exchange through
  threadgroup memory between workgroup barriers. A threadgroup that is not whole subgroups (one of
  80 or of 1 thread, or of 32 on a device of 64-lane subgroups) runs it at its own size, the last
  SIMD group partly filled, as Metal runs it. The reaper lowers it, through the cache directory,
  for the first such dispatch, which waits for it: a first-use wait. A kernel no such dispatch
  reaches lowers no variant. In a threadgroup of one thread the module's exchanges pass no
  barrier.
- The dispatch region (two `uvec4`) gives every direct dispatch's kernel Metal's grid: its first
  threadgroup (0, or a split piece's), the threadgroup size packed as
  `(x-1) | (y-1) << 10 | (z-1) << 20`, and Metal's threads per grid; an indirect dispatch pushes a
  zero grid, and its kernel takes the grid from its threadgroup count. Outside the tiles the packed
  size is the pipeline's own workgroup size, so the kernel's `MultiRegion` constant stays at its
  default, 0, and the kernel takes the size from its workgroup size, with no decode.
- A compiled kernel makes one `VkPipeline` per module, threadgroup size and threadgroup memory
  lengths, on the encoding thread that first dispatches any of its states with them, which waits
  for it. Nothing the runtime records
  depends on when a compile finishes: there is one recorded form, made after the wait. Each such
  wait is counted in `Metrics::first_use_waits`, with its time in `first_use_wait_nanos` and the
  longest in `first_use_wait_max_nanos`.
- `ComputePipelineState::precision_notes` carries the translator's notes of where the kernel's
  answers may differ from Metal's on this device, fixed per operation: those of the modules made
  with the state, and the emulated-SIMD-group module's once it is lowered. A kernel compiled
  with fast math carries `FastMathRewrites`: a driver may fold its arithmetic differently per
  `VkPipeline`, so its answers in a tile and at another size may differ in the last place.

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

**The pipeline cache.** The reaper makes an empty cache first when it starts, which a creation that
comes before it waits for (microseconds). Then, off every guest thread, it reads the blob and makes
a cache from it, merges into that cache what creations compiled in the empty one meanwhile
(`vkMergePipelineCaches`), and makes it the cache creations take; the empty one is destroyed once no
creation holds it. So no creation waits for the blob's read or the driver's parse of it, which for a
blob near 32 MiB take tens of milliseconds, or for the merge, which takes hundreds on some drivers:
a creation during it goes through the empty cache. `Metrics::creations_before_seed` counts the
creations that went through the empty cache; those that end once the merge has begun are not in
this run's blob, and the next start compiles their pipelines again. A blob that does not check, whose data does not begin with the
device's pipeline cache header (whole, the `pipelineCacheUUID` a new driver version changes
included), or that the driver refuses, is removed, and the empty cache stays. Once pipeline creation
has been quiet for 20 ms (no creation running and none ending) after a creation, the reaper reads
the cache's size (`vkGetPipelineCacheData` with no buffer), and where it grew by more than an eighth
since the last write, writes the cache's data over the blob: a temporary file renamed over it, never
appended to. So it writes at most once a quiet period, a run that makes no new pipeline writes
nothing, and a device's writes over its life come to a few times the blob's final size. The device's
drop writes it once more where the size changed at all. A crash between writes loses only the
pipelines made since the last; no creation waits for a size query or a write.

The cache holds at most 32 MiB when the reaper looks (about five times the 6.25 MB of pipeline data
a long recorded session made): one past it is replaced by a fresh, empty cache that creations take
from then on, and is destroyed once no creation holds it. The next write replaces the blob with
the fresh cache's data, so the blob holds at most 32 MiB too, with no second file and no
generations. `Gauges::pipeline_cache_bytes` is the cache's size at the reaper's last look, and is
zero once the device is torn down. With no directory the cache is the same, made empty, and
nothing is written.

**The lowerings** live in two generations, `g<N>/` (young) and `g<N-1>/` (old), each holding at most
half of `DeviceOptions::pipeline_cache_cap` (128 MiB where it is `None`: half of it is about seven
times the 9.16 MB a long recorded session's 282 lowerings fill). A lookup tries young, then old, and
a hit in old moves the file into young by a rename. A new file goes into young, written by the
reaper off every guest thread: a state hands its lowering over and returns, and where the lowerings
waiting for the reaper hold 8 MiB it is dropped and counted (`Metrics::cache_writes_dropped`), never
waited for, and lowered again by a later device. The write that takes young past half the cap starts
the next generation: young becomes old, and the old one is removed whole, by a thread the store
starts for it. So a file stays for at least half the cap, and at most the whole cap, of files
written or moved after its last use. Half the cap is what the store is sure to keep: a set of
functions used over and over that is larger than half the cap is partly removed at each new
generation and lowered again. On disk the lowerings take at most the cap, and while a removal runs,
also what is left of the generation it removes; the blobs are beside them. A file larger than half
the cap is not kept. `Metrics::cache_bytes_written` counts the bytes the device wrote there,
lowerings and blobs.

The bytes young holds are counted in `state`, a file of 24 bytes beside the generations, read and
written under its lock, so devices open on one directory, in one process or several, share its
files and one count; they should pass one cap, since each starts a generation at half its own.
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

### Known gaps of pipeline states

- No translation or `VkPipeline` compile runs ahead of the call or the dispatch that needs it: a
  state reads only its lowerings from the cache directory, each variant is lowered at the first
  dispatch that needs it, and each `VkPipeline` past the first tile is made at its first dispatch,
  through the device's pipeline cache.
- An image-only kernel's first dispatch over a grid more than one deep at a tile no state of its
  content has run makes that tile's `VkPipeline` on the calling thread, which waits for it
  (`Metrics::first_use_waits`), through the device's pipeline cache. In each of the two recorded
  streams measured, on a device that holds 1,024 threads along `z`, that made 50 pipelines, all in
  one frame: the frame where the stream's image-only dispatches over such grids start (frame 224
  of one stream and 312 of the other). The 50th, of tile 1x1x256, runs as one row, so a device
  that holds 64 makes the same 50.
- A kernel's variants are lowered at the first dispatch that needs each, which waits for the
  lowering and its pipeline (a first-use wait): an observing kernel's partial-threadgroups
  variant at its first `dispatch_threads` grid that is not a whole number of threadgroups, and
  the emulated-SIMD-group variant at the first threadgroup that is not whole subgroups. For the
  observing kernel the device tests run, a grid template over threadgroup memory, one lowering
  took 0.61 to 0.63 ms on an RX 7900 XTX in a release build; a larger kernel's lowering takes
  longer.
- On Intel's Windows driver (101.2145) for the UHD 630, the driver's compile of a tiling kernel's
  `VkPipeline`, made with the state, took 1.44 to 1.51 times the old runtime's for the harness's
  `create_state` kernel: medians of 6,680 to 6,853 µs against 4,473 to 4,761 µs, in three
  processes of 40 compiles each with no driver cache hit. The workgroup accounts for 1.1 to 1.3 ms
  of it: the runtime makes the pipeline at 256 threads, the tile it runs every grid of such a
  kernel in, where the old runtime made it at the dispatch's threadgroup, here one thread. The
  decode of the pushed tile accounts for 0.4 to 0.55 ms, and the module's device addresses, where
  the old runtime bound storage-buffer descriptors, for 0.37 to 0.47 ms. The runtime's pipeline
  layout and its pipeline cache cost nothing measurable there. The cost is paid once per driver
  cache: the driver keeps compiled pipelines across processes, and the same modules compiled again
  in a new process took 45 to 48 µs.
- A kernel that reads anything the encoder does not supply yet, a function table, a
  `[[stage_in]]` fetch, constant data, or a sampler or texture state the translator would take
  from the bound object rather than from its resource ID, is refused (`Gap::KernelInput`). So is
  a state for indirect command buffers (`Gap::IndirectCommandBuffers`), and a kernel that encodes
  commands into one, which the translator refuses
  (`Unsupported{GpuEncodedIndirectCommands, NotImplemented}`).
- A threadgroup that is not whole SIMD groups of a kernel whose SIMD groups run one per subgroup
  of 32 is refused (`Gap::PartialSimdgroups`) where the state has no emulated-SIMD-group module:
  for a kernel the consumer compiled (`StageSource::Compiled`), and where the translator refuses
  the emulated variant. The emulated module's scratch counts in the dispatch's threadgroup memory
  (`Limit::ThreadgroupMemory`).
- `dispatch_threads` with a grid that is not a whole number of threadgroups on some axis is
  refused (`Gap::PartialThreadgroups`) at a threadgroup that is not whole subgroups of a kernel
  that runs only whole subgroups, and for a kernel that observes its threadgroup
  partition and whose partial-threadgroups variant the translator refuses. It refuses the
  variant of a kernel whose SIMD groups it emulates: one with SIMD-group operations where the
  device's compute subgroups are neither 32 lanes nor 64 or 128 split into 32-lane clusters, and
  one with a lockstep pair in divergent flow where the device lacks
  `VK_KHR_shader_maximal_reconvergence`. It refuses the variant of a kernel whose barriers in a
  loop need a vote word that does not fit beside its threadgroup memory in
  `maxComputeSharedMemorySize`.
- A kernel that observes its threadgroup partition makes a `VkPipeline` per threadgroup size it
  is dispatched at, but for the grids its one-row module runs in tiles.
- A kernel that observes its threadgroup partition runs in workgroups of its own threadgroup
  size, never in tiles, but for the grids its one-row module runs. On Intel's Windows driver (101.2145) for the UHD 630, a kernel that
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
  1.26 and 1.59 times the old runtime's time in two runs (medians of 93.6 against 74.0 µs and
  79.7 against 50.0 µs; within one run, one form's rounds ranged up to 2.8 times apart). The old
  runtime's form, the 8x8x4 threadgroup as the workgroup and the buffer through a storage-buffer
  descriptor, took 0.97 of the tiles' time in this runtime in the first of those runs.
- Such a kernel runs a grid one high and one deep in tiles too: its position comes from the pushed
  tile in all three dimensions and is tested against the grid in all three. On an RX 550 (RADV,
  Mesa 26.2.6), a kernel of 2^20 threads in a line, each reading 16 bytes 64 times from a 64 KiB
  `constant` table and storing 4 bytes, took 1.18 times the old runtime's time (the median of
  three pairs of runs; 2,380.8 against 2,018.3 µs in the first), and 1.18 again with the table
  `Managed` and its first 256 bytes written before each run; the matrix's ceiling for both rows on
  that GPU is 1.20. In workgroups of the kernel's 256x1 threadgroup it took 0.90 of its
  time in tiles, through storage-buffer descriptors 0.89, and with both 0.89; the runtime makes
  neither. Its reads take their 64-bit addresses from two adds with a carry each, as the RX 550's
  loads through an address need, where the old runtime's reads through a descriptor take a 32-bit
  offset. An Iris Xe (Mesa 26.2.6) pays neither: there the kernel took 1.03 times the old
  runtime's GPU time, the same in 256x1 workgroups or through descriptors. In the two recorded
  streams measured, kernels that do not observe their partition ran 0.00% and 0.59% of their
  threads over grids one high and one deep.
- A dispatch of a size other than the descriptor's `requiredThreadsPerThreadgroup`, which Metal's
  validation stops, runs at the size it gives.
- NVIDIA 615.71.09 on GA10x gives sign-extended bits for a mask over a signed clamp, to exactly N
  bits' range, of a sum or difference of two values sign-extended from N bits; not reproduced on
  620.70.06.
- A kernel whose translated entry weighs more than 99,109 instructions (each call counted as its
  callee's body) is made on no device: that is the heaviest stage measured to compile within
  host-sized memory on the slowest compiler measured (ANV, Mesa 26.2.6, on the Iris Xe of RPL-P).
  One that emulated its SIMD groups only in place of the 32-lane pin is translated again pinned,
  and every other is refused (`Error::Translation`: `Feature::StageSize`, `Needs::StageBound`)
  where Metal makes it.
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

## Render pipeline states

- `Device::new_render_pipeline_state(&RenderPipelineDescriptor, sampler_ids)` is
  `newRenderPipelineStateWithDescriptor:error:` without reflection. The translator lowers the
  vertex and fragment functions on the calling thread, which waits for them, within its 20-second
  budget per lowering, for what device creation enabled and under its identity domains, as for a
  compute state. A refusal of the translator is `Error::Translation`.
- rt builds no pipeline reflection and acts on no `MTLPipelineOption`; its consumer builds
  reflection from the library's AIR, as for a compute state.
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
  sampler's index) in one range of the floor's 256 bytes, dense by link: the vertex function's
  from 0, the fragment function's from the vertex function's end rounded up to 8, so only the
  values the pair uses take space. Compute and render pipelines share the device's one pipeline
  layout and its one push range. A pair whose values do not fit is refused,
  `Error::Translation` with `maxPushConstantsSize`'s limit; no pair in the replay streams comes
  near it (the most is 248 bytes).
- A `VkPipeline` bakes what Vulkan 1.3 cannot leave dynamic: the content and the topology class
  (line or triangle) of the draws it serves. It takes as dynamic state the viewport, scissor, cull
  mode, front face, primitive topology within its class, primitive restart, depth test, write and
  compare, stencil test, operations, masks and reference, depth bias and blend constants, and a
  state without rasterization its discard, which keeps its attachment formats and that state in
  the pipeline. Vertex buffer strides are the descriptor's, so the content's, and baked. It
  renders dynamically into its attachments' formats, with no render pass.
- Every `[[stage_in]]` attribute is fetched by fixed-function vertex input, the form native
  Vulkan uses, with no second form: the vertex descriptor's formats, offsets, strides and steps
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
- The triangle class's `VkPipeline` is made with the state, through the device's pipeline
  cache; the line class's is made by the content's first draw of lines, which waits for it (`Metrics::first_use_waits`). Of the two replay streams' 537 render contents
  drawn, 527 draw only triangles and 2 lines and triangles; a content drawn only as lines (8 of
  them) makes a triangle `VkPipeline` it never uses.
- A post-tessellation vertex function's state draws its patches through the device's tessellator
  (`tessellationShader`, Floor). Its one `VkPipeline` is the patch class's, made with the state:
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

- Refused at creation by `Gap::RenderPipeline`, naming its `RenderPipelineKind`: depth and
  stencil attachment formats that differ, as of separate depth and stencil textures
  (`DepthStencilFormats`: Vulkan's pipeline takes one format for both,
  `VUID-VkGraphicsPipelineCreateInfo-renderPass-06589`), a `rasterSampleCount` that is not a
  power of two from 1 to 64 (`SampleCount`: Vulkan's rasterization takes one of its sample
  count bits), vertex amplification, a post-tessellation vertex function's per-instance factor
  step function or enabled factor scale (`Tessellation`;
  a descriptor's tessellation state with any other vertex function is ignored, as Metal ignores it),
  stages the translator synthesizes beside the functions' and a tessellation's, framebuffer fetch, an imageblock, a
  fragment function's stencil output, alpha to one, constant data the binder supplies, a
  specialization other than the sample and amplification counts, dynamic state beyond Vulkan
  1.3's (a vertex buffer layout whose stride is set per bind among it), a
  feature or extension device creation does not enable, and a function that reads what a draw
  does not supply (`DrawInput`: a function table, a table the binder builds, a parameter block
  past the push range, a push field or bind-time fact the encoder does not fill, an unnormalized
  constexpr sampler). Of the replay streams' contents that translate, every one reads only its
  slots' addresses, resource IDs and constexpr sampler indices. A state for indirect
  command buffers is `Gap::IndirectCommandBuffers`. The `completionHandler:` forms are not built.
- Metal's triangle fill mode and depth clip mode, which Vulkan 1.3 bakes into a `VkPipeline`, are
  their defaults.
- On a driver that keeps the topology class in a pipeline (dynamicPrimitiveTopologyUnrestricted
  false; RADV), a render pipeline state's first draw in a second topology class compiles a second
  pipeline in full when the driver's cache is cold: 225 ms for the heaviest pipeline measured on a
  780M, 0.9 to 1.5 ms for the others, nothing once the driver's cache holds it. A driver without
  the restriction serves it from its cache (0.1 ms on ANV). Linking from pipeline libraries would
  avoid it; the floor does not have them.

## Render encoders

- `CommandBuffer::render_command_encoder(&RenderPassDescriptor)` is
  `renderCommandEncoderWithDescriptor:`. While it is open it holds the command buffer's recording,
  as a compute encoder does. It begins one dynamic rendering (`vkCmdBeginRendering`, in the floor)
  into the pass's attachments, and `end_encoding` ends it (`vkCmdEndRendering`): one Vulkan
  rendering a Metal render encoder, with no render pass object or framebuffer.
- `RenderPassDescriptor` is `MTLRenderPassDescriptor`: eight color attachments
  (`MAX_COLOR_ATTACHMENTS`), a depth and a stencil attachment, `renderTargetWidth` and
  `renderTargetHeight`. Each attachment is one level and slice of its texture: a 2D texture, a 2D
  array, a cube or a cube array, or a 2D multisample texture or array, of `RenderTarget` usage;
  every attachment of a pass has one sample count, and a draw's pipeline state that count as its
  `rasterSampleCount` (`DrawKind::AttachmentFormats` otherwise). The device's counts are
  `Device::supports_texture_sample_count`'s. A multisample attachment is stored as its samples:
  a shader reads them through `texture2d_ms` or `depth2d_ms`. The texture keeps an
  image view of that level and layer with the identity swizzle, made at the first pass that
  renders into it and freed with the texture (`Gauges::attachment_views`).
- Load actions are one to one: `Load`, `Clear` and `DontCare` are Vulkan's load operations of
  those names. `Store`, `DontCare` and `Unknown` store: a pass may end its rendering before the
  encoder ends and begin it again, loading what it stored (below), so a `DontCare` store keeps the
  texels, one of the outcomes Metal allows. Vulkan fixes the store as the rendering begins, so an
  `Unknown` store action stores, and the encoder's `set_color_store_action`,
  `set_depth_store_action` and `set_stencil_store_action` change nothing the device does: on a
  tile-based device this stores a target Metal would have discarded. A resolving action set there
  is refused (`Gap::RenderPass` of `Resolve`). A clear color is a `float` in a
  float, normalized or sRGB channel, which the device converts to the format by its own rules
  (Vulkan clamps a normalized value and asks for rounding to nearest; Metal rounds to nearest
  even); an integer channel's is truncated toward zero and saturated to the channel's width, NaN
  0, as Metal's. A rotated format's clear color, and `A8Unorm`'s held in `R8_UNORM`, go through
  the format's rotation. A clear stencil keeps its low 8 bits.
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
  test, where Metal's documentation states no answer. `set_depth_bias` takes the bias and slope
  scale as given; a draw of triangles is biased where either is not zero and a draw of lines
  never, as Metal biases triangles alone, under the static note `PrecisionNote::DeviceDepthBias`
  (`RenderCommandEncoder::DEPTH_BIAS_NOTE`): the slope measure, and a fixed-point attachment's
  unit, are the device's; Apple states neither. Where a pass
  has both a depth and a stencil attachment they are one texture's same level and slice, a
  format of depth and stencil, rendered through one view: Vulkan's rendering takes one view for
  both (`VUID-VkRenderingInfo-pDepthAttachment-06085`). A pipeline state's depth and stencil
  formats are the pass's, as its color formats are; the format query decides which depth and
  stencil formats a device renders, a texture of `RenderTarget` usage in one it does not is not
  made, and a pipeline state of one is the translator's refusal.
- The render area is the smallest attachment's level, cut to `renderTargetWidth` and
  `renderTargetHeight` where they are set.
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
  textures and sampler states, one viewport and scissor rectangle, the cull mode, the
  front-facing winding, a depth-stencil state (`Device::new_depth_stencil_state`), the stencil
  references, the depth bias and the blend color, Metal's defaults where it set none. A vertex buffer slot serves
  the vertex function's buffer argument of its index and the vertex descriptor's layout of its
  index alike, a `set_vertex_bytes` copy too, read through the parameter ring's buffer.
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
- A viewport is Vulkan's with its height negated (from `y + height`): framebuffer rows, the
  fragment position and derivatives follow Metal's clip space, whose y points up, and Metal's
  front-facing winding is Vulkan's front face of the same name. A viewport of no width or height,
  or with a NaN origin x, draws nothing, as Metal's. Metal clips a draw to the pixels its viewport
  covers, so the scissor is Metal's rectangle cut to them; a rectangle past the render area,
  which Metal's validation stops, is cut to it, as plain Metal cuts it. A stencil reference
  saturates at 255, as Metal stores it.
- A draw may read an attachment of its own pass through a texture, as Metal reads a render
  target, where the device reads attachments (Floor, Attachment reads): through the attachment
  feedback loop where the device has it, else through a copy. The texture reaches the
  attachment's level and layer, the fragment function reads it and nothing writes it, and its
  image is a shader-read colour render target with no storage view, which one attachment of the
  pass renders. Through the feedback loop, at the pass's first such draw the rendering ends, one
  barrier moves the whole image to the feedback layout (the attachment's level and layer from the
  attachment layout, the rest from home), and the rendering begins again with the attachment in
  that layout; the pass's second barrier moves the whole image home. Until then every draw of the
  pass reads the image's textures through their second elements (Set 1), a texture that reaches
  no rendered level among them. Every render pipeline takes the feedback loop's enable as dynamic
  state, which the encoder sets at its first draw and where it changes: colour for a draw that
  reads an attachment, none otherwise, so a pass of no such draw sets it once.
- Through a copy, on a device without the feedback loop, the draw reads the image's shadow: an
  image of the same format, extent, levels and layers, made at the first draw that reads the
  image and freed with it (`Gauges::shadows` and `shadow_bytes`). A texture that reaches the
  attachment's level and layer reads the shadow through its second element; one that reaches
  neither reads the image at home. Where the draw reads what the shadow does not hold, the
  rendering ends, one barrier orders every earlier command before the copies and moves what they
  copy to the transfer layouts, the copies are recorded (`Metrics::attachment_read_copies` and
  `attachment_read_copy_bytes`), one barrier moves it all back, and the rendering begins again.
  A slot the fragment function reads only at its own pixel reads the draw's scissor, a pixel
  wider each way, clamped into its level; any other slot reads every level and layer its texture
  reaches. What is new to the shadow is copied as one area where it is one (a wider scissor's
  band), else as the whole read, with no look-ahead. At the pass's first draw, where the pass
  cleared the attachment and every read of its level and layer lies in it alone and inside the
  render area, the shadow's level and layer is cleared to the pass's clear colour between the
  same barriers instead of copied (the shadow is of the attachment's format, so the clear writes
  the texels the pass's did); a draw of the pass before it, of any pipeline, leaves the copy.
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

- Refused by `Gap::Draw`, naming its `DrawKind`: a draw of points, a depth bias clamp other
  than 0 (`DepthBiasClamp`, set by `set_depth_bias`), a triangle fill mode but `Fill`
  (`FillMode`, `set_triangle_fill_mode`), the depth clip mode `Clamp` (`DepthClamp`,
  `set_depth_clip_mode`), a count of viewports or scissor rectangles other than 1 (`Viewports`,
  `set_viewports`, `set_scissor_rects`), each refused where it is set, the encoder's state
  unchanged; the replay streams set Metal's defaults alone, which those setters take as nothing; a
  viewport Vulkan's is not (a
  negative width, a depth range past `[0, 1]`, bounds past the device's viewport limits), a
  pipeline state whose formats are not the pass's attachments', a draw that binds a texture
  whose image the pass renders, where neither the feedback loop nor a copy reads it: on a device
  without the attachment reads, from the vertex function at the rendered level and layer, written, of a
  depth or stencil attachment, of an image with a storage view or over a buffer in place, or of
  an image two attachments of the pass render; and, on a device that does not report
  `supportsNonZeroFirstInstance` (lavapipe and the UHD 630 among the lab's), a draw at a base
  instance other than 0 through a state that steps a layout per instance at a rate above 1
  (`DividedBaseInstance`: Vulkan's divisor takes `firstInstance` 0 alone there,
  VUID-vkCmdDraw-pNext-09461); the same draw from base instance 0 runs. An indirect draw, whose
  base instance the CPU does not see, is not built. A patch draw through a patch index buffer is
  `PatchIndexBuffer`: the control stage takes each patch's index from its position, and no replay
  stream's patch draw names one. The replay streams draw lines and triangles
  under depth biases with no clamp; every attachment they read through a texture is
  a shader-read colour render target of `Private` storage, read from the fragment function; and
  their stepped draws (a WebGL stream's, at rates 2 and 3) are all at base instance 0.
- A draw that reaches an attachment of its own pass through an argument buffer's texture is not
  seen: it reads through the texture's own element, in its home layout, while the pass may have
  moved the image to the feedback layout, or, without the feedback loop, while the pass renders
  the level and layer it reads.
- A vertex fetch outside its buffer returns what the device gives: the floor enables no buffer
  robustness, so there is no bounds guarantee beyond the device's.
- Not built, and so not methods: indirect draws (of patches too), visibility results,
  `useResource:`, `useHeap:stages:`, the store action options setters, tile functions,
  `setVertexBuffer:offset:attributeStride:atIndex:`, `setTessellationFactorScale:` (read only by a
  state that enables the scale, which is refused) and `drawIndexedPatches:` (whose control-point
  index buffer only a state of a control-point index type reads, which is refused). Where a replay stream calls
  `useHeap:stages:` (`command-replay shapes` counts a stream's calls of each selector), it is
  built in the compute encoder's form, `use_heap`: the command buffer keeps the block and declares
  no access.
- Refused by `Gap::RenderPass`, naming its `RenderPassKind`: a depth and a stencil attachment
  that are not one texture's same level and slice (`DepthStencilViews`: Vulkan's rendering takes
  one view for both), a `Clear` load of a depth attachment whose clear depth is outside `[0, 1]`
  or NaN (`ClearDepth`: Vulkan clears no such depth without `VK_EXT_depth_range_unrestricted`,
  `VUID-VkClearDepthStencilValue-depth-00022`, which the floor does not require), attachments of
  two sample counts (`SampleCounts`: Vulkan renders one count into every attachment of a pass,
  `VUID-VkRenderingInfo-multisampledRenderToSingleSampled-06857`), a resolve (`Resolve`: a
  resolve texture, or a store action that resolves), a 1D or 3D texture's attachment, the
  `CustomSampleDepthStore` store action, a clear of a format only a shader clears (extended
  range, bfloat, YCbCr), and a pass with no attachment. The descriptor has no field for a
  visibility result buffer, a rasterization rate map, tile sizes, imageblock and threadgroup
  memory, sample buffer attachments, layered rendering or store action options. The replay
  streams' passes render into one or two color attachments and a stencil attachment or a
  `Depth32Float` depth attachment, each a level of a 2D texture of one sample, with `Load`,
  `Clear` and `DontCare` loads and `Store`, `DontCare` and `Unknown` stores.
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

  The same passes on the old runtime (`metal2vulkan-runtime`), timed in the same runs, reading
  the target / another texture, with the difference:

  | Device | 3648 × 2038, this runtime | 3648 × 2038, the old runtime | 1920 × 1080, this runtime | 1920 × 1080, the old runtime |
  |---|---|---|---|---|
  | RX 550 (RADV) | 1,122.9 / 516.5 µs (+606.4) | 1,553.0 / 1,135.7 µs (+417.3) | 320.3 / 150.2 µs (+170.1) | 452.2 / 320.8 µs (+131.4) |
  | RX 7900 XTX (RADV) | 54.3 / 19.1 µs (+35.2) | 54.4 / 37.6 µs (+16.8) | 20.4 / 13.1 µs (+7.2) | 31.6 / 20.1 µs (+11.5) |

  A pass that reads its target takes no longer here than on the old runtime, on either card at
  either size. On the RX 550 the difference from the plain pass is larger here, because the old
  runtime's plain pass is 2.2 times slower.
- On a device that does not report `attachmentFeedbackLoopLayout` and
  `attachmentFeedbackLoopDynamicState` (the UHD 630 among the lab's, Windows driver 101.2145), a
  draw reads an attachment through a copy into the image's shadow, which Metal does not make: at
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
  another texture. Against the old runtime in the same runs, at 3648 × 2038, this form reads the
  target faster after a barrier (4,836.4 µs against 5,429.4 µs) and at the first read of a pass
  that loads it (3,849.6 µs against 4,218.2 µs), and within 1% at the first read of a pass that
  clears it (4,200.8 µs against 4,159.4 µs; at 1920 × 1080, 1,095.3 µs against 1,109.1 µs).

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
no mutex and no reference count change. Each encoder holds the device once; a command borrows it,
the pipeline state and the bound resources, and counts into the encoder's own tally, which the
device's counters take once, at the encoder's end. What a draw or a dispatch learns from its
bindings (the vertex buffers it binds, the alignments it checks, whether it reads an attachment,
reaches a Managed buffer's device copy or writes what the host reads) it works out again only
after the pipeline state or a binding changes. On RADV a recorded command lies in write-combining
memory, and a locked instruction between two of them flushes it: a draw took about 490 ns more on
a Radeon RX 7900 XTX with one.

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

### Extensions

Presenting runs on the runtime's own device; nothing is exported or imported.

| Extension | Where | Used for |
|---|---|---|
| `VK_KHR_surface` | instance | presenting |
| `VK_KHR_swapchain` | device | presenting |
| `VK_EXT_robustness2` | device | null descriptors, and image reads out of range answered by the device |
| `VK_EXT_calibrated_timestamps` | device | GPU times on the host clock: the device calibrates its own clock (`VK_TIME_DOMAIN_DEVICE_EXT`) with the host's (`floor::TIME_DOMAINS`) |
| `VK_KHR_shader_maximal_reconvergence` | device, where listed | SIMD groups that reconverge as Metal's do |
| `VK_KHR_shader_subgroup_uniform_control_flow` | device, where listed | SIMD groups' reconvergence where maximal reconvergence is not offered |
| `VK_KHR_portability_enumeration` | instance, where offered | listing portability devices, each checked against the floor as any other |
| `VK_KHR_portability_subset` | device, where listed | Vulkan requires it enabled where a device lists it; the device is asked for the portability features below |
| `VK_EXT_attachment_feedback_loop_layout` | device, with the feedback loop | a draw's read of an attachment of its own pass through a texture, in the feedback layout (below) |
| `VK_EXT_attachment_feedback_loop_dynamic_state` | device, with the feedback loop | the feedback loop's enable as dynamic state of every render pipeline |
| `VK_KHR_vertex_attribute_divisor` | device, where listed | a vertex layout stepped per instance at a rate above 1, through the step-rate feature below |
| `VK_EXT_vertex_attribute_divisor` | device, where listed | the same, on a device that lists only this extension, which the KHR was promoted from with the same feature and pipeline structures |

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
| `shaderStorageImageWriteWithoutFormat` | `VkPhysicalDeviceFeatures` | writes to array elements of any pixel format |
| `nullDescriptor` | `VkPhysicalDeviceRobustness2FeaturesEXT` | every texture element no texture owns: a nil, freed or unowned index reads zeros, sizes 0 and drops writes, as Metal's nil texture does. The robust buffer access beside it stays off |
| `samplerAnisotropy` | `VkPhysicalDeviceFeatures` | a sampler state's `maxAnisotropy`, which the device's own filter takes |
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
| `tessellationShader` | `VkPhysicalDeviceFeatures` | a post-tessellation vertex function's patches, drawn through the device's tessellator |
| `dualSrcBlend` | `VkPhysicalDeviceFeatures` | a fragment function's second output for colour attachment 0 (`[[color(0), index(1)]]`) as the blend's second source (`Source1Color`, `Source1Alpha` and their complements) |
| `vertexPipelineStoresAndAtomics` | `VkPhysicalDeviceFeatures` | a vertex or post-tessellation vertex function's writes and atomics on device memory |
| `multiview` | `VkPhysicalDeviceVulkan11Features` | Vulkan 1.1 requires it; the translator's multiview forms |
| `shaderDrawParameters` | `VkPhysicalDeviceVulkan11Features` | a vertex function's `[[base_vertex]]` and `[[base_instance]]` as the draw's own `BaseVertex` and `BaseInstance`, so no draw pushes its bases; Vulkan 1.1 requires it where the device lists `VK_KHR_shader_draw_parameters`, Roadmap 2024 of every device |
| `shaderSubgroupExtendedTypes` | `VkPhysicalDeviceVulkan12Features` | SIMD-group operations on 8-, 16- and 64-bit values; Vulkan 1.2 requires it |
| `shaderDemoteToHelperInvocation` | `VkPhysicalDeviceVulkan13Features` | a discarded fragment demoted to a helper, so its quad's derivatives stay defined; Vulkan 1.3 requires it |
| `shaderTerminateInvocation` | `VkPhysicalDeviceVulkan13Features` | a fragment's termination; Vulkan 1.3 requires it |
| `shaderZeroInitializeWorkgroupMemory` | `VkPhysicalDeviceVulkan13Features` | threadgroup memory zeroed at its declaration where the translator zeroes it; Vulkan 1.3 requires it |
| `timelineSemaphore` | `VkPhysicalDeviceVulkan12Features` | submission and completion |
| `hostQueryReset` | `VkPhysicalDeviceVulkan12Features` | a command buffer's timestamp queries reset on the host as it retires |
| `synchronization2` | `VkPhysicalDeviceVulkan13Features` | barriers, layout changes and submission |
| `dynamicRendering` | `VkPhysicalDeviceVulkan13Features` | render passes |

### Portability features

A device that lists `VK_KHR_portability_subset` may offer less than Vulkan defines. The floor asks
such a device for these, and device creation enables them; a device that does not list the
extension offers each of them by Vulkan's definition and is not asked.

| Portability feature | Structure | Used for |
|---|---|---|
| `imageViewFormatSwizzle` | `VkPhysicalDevicePortabilitySubsetFeaturesKHR` | a texture view's component mapping: its format's and Metal's swizzle |
| `mutableComparisonSamplers` | `VkPhysicalDevicePortabilitySubsetFeaturesKHR` | the comparison sampler array, which takes each sampler state's sampler as it is made |
| `multisampleArrayImage` | `VkPhysicalDevicePortabilitySubsetFeaturesKHR` | `MTLTextureType2DMultisampleArray` |

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

### Limits

The descriptor limits are what the runtime's fixed arrays need: 65,536 sampled textures, 65,536
storage textures and three arrays of 4,000 samplers, one per use (plain, depth comparison,
integer texture). Vulkan requires at least 500,000 of each texture limit and of the two resource
totals on a device that supports `descriptorIndexing`, as every lab device does. 4,000 is Vulkan's minimum `maxSamplerAllocationCount`. The compute limits let a threadgroup take
Metal's 1,024 threads in any shape up to 1,024 by 1,024 by 64. `maxColorAttachments` holds
Metal's eight color attachments on every device; Vulkan 1.3 requires 4 of it, and Vulkan 1.4 8.
A Boolean limit's minimum is 1,
`VK_TRUE`; `timestampValidBits` is the queue family's.

| Limit | Minimum | Used for |
|---|---|---|
| `maxPerStageDescriptorUpdateAfterBindSampledImages` | `65536` | the sampled-texture array |
| `maxDescriptorSetUpdateAfterBindSampledImages` | `65536` | the sampled-texture array |
| `maxPerStageDescriptorUpdateAfterBindStorageImages` | `65536` | the storage-texture array |
| `maxDescriptorSetUpdateAfterBindStorageImages` | `65536` | the storage-texture array |
| `maxPerStageDescriptorUpdateAfterBindSamplers` | `12000` | the three sampler arrays |
| `maxDescriptorSetUpdateAfterBindSamplers` | `12000` | the three sampler arrays |
| `maxSamplerAllocationCount` | `4000` | Vulkan's minimum: live samplers |
| `maxPerStageUpdateAfterBindResources` | `131072` | both texture arrays in one stage; a fragment stage needs its `maxColorAttachments` more |
| `maxUpdateAfterBindDescriptorsInAllPools` | `143072` | all five arrays |
| `maxPushConstantsSize` | `256` | per-command values: a compute kernel's dispatch region and 28 buffer addresses, anything larger in a block whose address is pushed; a render pipeline's vertex and fragment functions' values in the same 256 bytes, dense by link, a pair that does not fit refused |
| `maxComputeWorkGroupInvocations` | `1024` | Metal's most threads per threadgroup |
| `maxComputeWorkGroupSize[0]` | `1024` | a threadgroup's width |
| `maxComputeWorkGroupSize[1]` | `1024` | a threadgroup's height |
| `maxComputeWorkGroupSize[2]` | `64` | a threadgroup's depth; a deeper one runs as one row |
| `timestampComputeAndGraphics` | `1` | GPU times of command buffers |
| `standardSampleLocations` | `1` | Metal's sample positions: those of 1, 2 and 4 samples are Vulkan's standard locations |
| `timestampValidBits` | `36` | GPU times of command buffers |
| `maxColorAttachments` | `8` | every color attachment of a render pass descriptor |

### Queue family and memory types

The runtime submits to the first queue family with `VK_QUEUE_GRAPHICS_BIT` and
`VK_QUEUE_COMPUTE_BIT` (`floor::Capabilities::queue_family`). It allocates buffers from three memory
types, each the first with its flags that a buffer of `floor::BUFFER_USAGE` can use: every buffer
usage Vulkan 1.3 defines, and device addresses. Vulkan guarantees such a buffer a device-local
type, so of the three only Shared and the ring can be missing on a conforming device. Vulkan answers which memory types a buffer can use
only for a created device, so `PhysicalDevice::check` finds a type with each set of flags, and
`Device::new` chooses on the device (`Device::memory_types` answers `floor::MemoryTypes`).

| Memory type | Flags | Used for |
|---|---|---|
| Private | `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` | Private storage, which only the GPU reads and writes |
| Shared | `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT`, `VK_MEMORY_PROPERTY_HOST_COHERENT_BIT`, `VK_MEMORY_PROPERTY_HOST_CACHED_BIT` | Shared storage, which the host maps and reads |
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

Five features choose between two forms by the device's answer: 16-bit floats, storage-image reads
without a format, 64-bit atomics, 64-bit floats and SIMD groups. Beside them, a device without the
attachment feedback loop reads attachments through a copy, the one form left there, which is a
known gap (Known gaps of render encoders). `floor::Capabilities` holds each answer the floor
reads, and the features and extensions to enable: the floor's, and each optional one the device
has.

| Feature | Queries | Used for |
|---|---|---|
| SIMD groups | `minSubgroupSize`, `maxSubgroupSize` and `requiredSubgroupSizeStages` (`VkPhysicalDeviceVulkan13Properties`); `subgroupSupportedStages` and `subgroupSupportedOperations` (`VkPhysicalDeviceVulkan11Properties`); `shaderMaximalReconvergence` (`VK_KHR_shader_maximal_reconvergence`) and `shaderSubgroupUniformControlFlow` (`VK_KHR_shader_subgroup_uniform_control_flow`) | the translator's choice of how a kernel's SIMD groups run (its tier), and their reconvergence mode: maximal where offered, else subgroup-uniform control flow where offered |

The other four are one feature each:

| Optional feature | Structure | Used for |
|---|---|---|
| `shaderFloat16` | `VkPhysicalDeviceVulkan12Features` | `half` arithmetic in 16 bits |
| `shaderStorageImageReadWithoutFormat` | `VkPhysicalDeviceFeatures` | reads of read-write textures of any format; where it is false, each format's `VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT` answers |
| `shaderBufferInt64Atomics` | `VkPhysicalDeviceVulkan12Features` | 64-bit atomics on buffers |
| `shaderSharedInt64Atomics` | `VkPhysicalDeviceVulkan12Features` | 64-bit atomics on threadgroup memory |
| `shaderFloat64` | `VkPhysicalDeviceFeatures` | 64-bit floats |
| `shaderMaximalReconvergence` | `VkPhysicalDeviceShaderMaximalReconvergenceFeaturesKHR` | SIMD groups that reconverge as Metal's do |
| `shaderSubgroupUniformControlFlow` | `VkPhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR` | SIMD groups' reconvergence where maximal reconvergence is not offered |

### Attachment reads

A draw that reads, through a texture, an attachment its own pass renders (Metal's read of a
render target) is built where the device's descriptor limits hold the sampled-texture array's
second elements beside the floor's arrays: 131,072 sampled textures where the limits above name
65,536 (`maxPerStageDescriptorUpdateAfterBindSampledImages`,
`maxDescriptorSetUpdateAfterBindSampledImages`, `maxPerStageUpdateAfterBindResources` and
`maxUpdateAfterBindDescriptorsInAllPools`); a device without them refuses such a draw
(`Gap::Draw` of `DrawKind::AttachmentRead`), and is otherwise the same
(`floor::Capabilities::attachment_reads`). The draw reads through the attachment feedback loop
where the device also lists the two extensions marked "with the feedback loop" above and reports
each feature below, which device creation then enables, and none otherwise
(`floor::Capabilities::feedback_loop`). A device that reads attachments without them reads
through a copy of the image into its shadow (Render encoders), a known gap.

| Attachment-read feature | Structure | Used for |
|---|---|---|
| `attachmentFeedbackLoopLayout` | `VkPhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT` | the feedback layout and usage |
| `attachmentFeedbackLoopDynamicState` | `VkPhysicalDeviceAttachmentFeedbackLoopDynamicStateFeaturesEXT` | the enable as dynamic state |

The test `tests::floor` holds these tables equal to the code, checks that each missing item is
named, and, under `--cfg m2v_gpu` or `--cfg m2v_cpu_icd`, checks every device the machine's
Vulkan loader lists.
