# metal2vulkan-rt

The Metal API on Vulkan, with one path for every device. It replaces `metal2vulkan-runtime` and
takes its name when it does; until then it is unpublished.

Its public API so far: the instance and the device, memory blocks and the buffers in them, the
grants of texture and sampler indices, textures, sampler states, command queues, command buffers
and events, the device's counters, and `floor`, the check of a Vulkan device against the floor
below.

## Instance and device

- `Instance::new` loads the Vulkan loader and creates the instance at the floor's version, with
  the floor's instance extensions and each extension the consumer names that the loader offers:
  the window system's, and any other it uses. Where the loader offers
  `VK_KHR_portability_enumeration`, it is enabled and portability devices are listed too, each
  checked against the floor as any other. `Instance::extensions` lists those enabled.
  `Instance::physical_devices` lists the physical devices in the loader's order.
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
  directory the device keeps its compiled kernels in, `pipeline_cache_dir` (see "The cache
  directory").
- `Device::raw` gives the instance, physical device, device and queue family handles, and
  `Device::with_present_queue` runs the consumer's presents on the presenting queue under its lock.
  The runtime keeps owning every handle.
- `Device::has_unified_memory` is true where Vulkan's device type is integrated or CPU.

### Teardown

The runtime runs four threads per device: the block-maker, the submit thread, the completion
thread, and the reaper, which frees the semaphores and command pools the others give up, one at a
time, so no driver free runs on the completion thread and a submission waits behind at most one,
lowers the compute pipeline states' emulated-SIMD-group variants after their creation returned,
and writes the pipeline data to the cache directory when it is due.
A device is torn down when its last handle, and the last handle of every object made from
it, is dropped; a committed command buffer and the blocks the consumer holds do not keep it alive.
Teardown stops the submit thread, which submits what is ready and ends what it holds with
`CommandBufferError::NotRun`; the completion thread retires the rest as the GPU completes it and
runs the handlers; the threads are joined and the GPU goes idle; `BlockMappings::unmap` runs for
each block the consumer was handed. Teardown runs on the thread that drops the last handle, or,
where that is a handler or a listener, on the reaper once the handler returns. A call that holds a
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
(`Refusal::Committed`), the release of a block not held (`Error::NoSuchBlock`), and a call that
needs a runtime thread after `Device::quiesce` (`Error::Quiesced`). `DeviceOptions` are read once,
by `Device::new`.

### Counters

`Device::counters` returns a handle that stays readable after the device is gone. `metrics()` are
cumulative counts (submissions, passes, barriers, bytes copied, pipelines created, blocks made,
first-use waits, kernels lowered, kernels read from the cache directory, and upload waits);
`gauges()` are live counts per kind of resource, every one zero after teardown. They are relaxed
atomics, always on.

## Memory blocks and buffers

Memory comes in blocks the consumer takes and places buffers in, as a placement heap's buffers sit
in its memory.

- `Device::take_block(class, len)` gives a block of `MemoryClass::Private` (device-local, no host
  mapping) or `MemoryClass::Shared` (host-visible, host-coherent and host-cached, mapped for the
  block's life) memory, as a `BlockInfo`: its number, class, length, device address and, for
  Shared, its host address. One `VkBuffer` spans each block, with every buffer usage and a device
  address. Nothing in a block is exportable.
- Up to `BLOCK_LEN` (256 MiB) a block is pooled. The block-maker thread keeps one block of each
  class ready, made off every caller's path, so a take waits only when none is ready. Past
  `BLOCK_LEN` a block is dedicated: the length rounded up to 64 KiB, made in the call, up to the
  device's `maxMemoryAllocationSize` and `maxBufferSize` (`Limit::BlockLen` past them).
- Every `VkDeviceMemory` the runtime holds counts against the device's
  `maxMemoryAllocationCount`: each block, each texture's own allocation, the staging region (from
  the first texture copy on), the staging buffer of a copy larger than the region while it runs,
  the parameter ring's blocks and the zero source. One more past it is `Limit::MemoryAllocations`,
  refused before the driver is called; `Gauges::blocks` counts those live. The limit is about 4.29 billion on RADV,
  NVIDIA's Linux driver and ANV, 8,401,685 on Intel's Windows driver for the UHD Graphics 630, and
  4,096 at Vulkan's minimum.
- Every block reads zero when it is handed out, new or taken again, as Metal's new buffers do. A
  Shared block is zeroed through its mapping, which also makes every page resident before anyone
  holds it; a Private block by a GPU fill.
- `Device::release_block` gives a block back. It takes effect once no buffer of the block remains.
  A pooled block is then zeroed and kept, with the same number, address and mapping, while its
  class has fewer than two idle blocks; the block-maker thread frees the rest. A dedicated block is
  always freed. Block numbers are never reused during the device's life.
- `Device::new_buffer(block, offset, length, zeroed)` makes a `Buffer`: a view of the range, with
  no Vulkan object of its own. Its `gpu_address` is the block's address plus the offset, and its
  `contents` the block's mapping plus the offset, or `None` in a Private block. `zeroed` zeroes the
  range, as `newBufferWithLength:options:` does: through the mapping for Shared, before the call
  returns; in Private by a GPU fill of exactly the range, recorded into the upload, which goes
  ahead of every command buffer committed after the call and ends with a barrier that makes the
  fills visible to every later command, and holds the block until it completes. A buffer
  placed in a heap passes false, since Metal does not zero heap memory. A length of 0, a length
  past `MAX_BUFFER_LENGTH` (2 GiB) and a range past the block are refused, as Metal answers nil for
  each.
- A shader reaches a buffer argument's bytes by 32-bit offsets from its bind address, which the
  translator may read as signed, so the consumer states a maximum buffer length (Metal's
  `maxBufferLength`) no larger than `MAX_BUFFER_LENGTH`, and places no buffer longer than that in
  a block, whether Metal's `newBufferWithLength:`, a heap's or a no-copy buffer stands behind it.

The consumer's ranges follow Metal's retention: it reuses a range only once the work that named
the buffer there before has completed. A command buffer keeps each block it uses until it retires,
so a block released meanwhile goes back to the pool only then.

### The consumer's mappings

A consumer that maps blocks itself, into a guest for instance, passes `DeviceOptions::block_mappings`.
Before the runtime frees a block it handed out, it calls `BlockMappings::unmap(block)`, which
returns once every mapping the consumer made of it is gone and none of its threads touches the
block. The memory stays mapped and valid until then. It runs once per block: on the block-maker
thread; in `Device::quiesce`, on the calling thread, for every block still handed out; or at
teardown, on the thread that tears down, after the runtime's other threads are joined. It never calls
into the runtime. Blocks the consumer never took are freed without it, and so is every block freed
after a quiesce.

## Set 1 and index grants

Shaders reach textures and samplers by index (`docs/RESOURCE-ID.md`). Every pipeline takes set 1,
one descriptor set the device makes at creation, of five arrays whose element `i` is the object
whose `gpuResourceID` holds index `i`: 65,536 sampled textures (binding 0), 4,096
storage-capable textures (binding 1), and 4,000 samplers of each use, plain, depth comparison and
integer texture (bindings 2, 3 and 4). Every binding is partially bound, updated after bind and
updated while unused elements are pending, so an object's element is written once, as it is
made, with no lock, and never rewritten for layouts.

- Every element no object owns is defined. A texture element is a null descriptor
  (`nullDescriptor`): it reads zeros, sizes 0 and drops writes, as Metal's nil texture does. A
  sampler element is its array's default sampler, nearest and clamped to the edge; the
  comparison array's compares with `NEVER`, so a comparison through it answers 0, as Metal's
  default compare function does. Vulkan has no null sampler. The device writes every element so
  at creation and holds two `VkSampler`s for the defaults (`Gauges::device_samplers`).
- `Device::grant_indices(class, first, count)` grants the consumer the range it names, so the IDs
  it chooses from it on the guest's side reproduce in replay. The three classes are disjoint
  (`IndexClass::range`): storage-capable textures (shader-write or unknown usage) take 1 to 4,095,
  one index valid in both texture arrays; other textures 4,096 to 65,535; samplers 1 to 3,999.
  Index 0 is nil and never granted. A range that is empty or leaves its class's is
  `Limit::IndexRange`, and one holding an index granted already `Refusal::IndicesGranted`.
  `Device::release_indices` returns a grant; one not granted is `Refusal::IndicesNotGranted`, and
  one where an object still lives `Refusal::IndicesInUse`.
- An index holds one object at a time: making one where another lives is `Refusal::IndexInUse`,
  and at an index not granted `Refusal::IndicesNotGranted`. The index is free again once the
  object's last handle is gone, and with it the last command buffer that used it; its elements
  read their default again from then.

## Textures

- `Device::new_texture(descriptor, index)` is `newTextureWithDescriptor:` at `index`: an
  optimal-tiling image of the descriptor in device-local memory of its own (a dedicated
  allocation), its contents undefined, as Metal's are. `index` is given exactly where shaders
  reach the texture, shader-read or shader-write usage (`Refusal::TextureIndex` otherwise), from
  a live grant of the class the usage picks: `IndexClass::StorageTexture` with shader-write
  usage, else `IndexClass::Texture`. `TextureUsage` with no bits is Metal's unknown usage: shader
  read and write, render target and pixel-format view.
- The image's usage is the descriptor's plus transfer source and destination. Its home layout is
  general where it has a storage element, shader-read-only where shaders only read it, and the
  attachment layout where only render passes use it. Its elements of set 1 are written before the
  call returns, with that layout, and never rewritten. The image moves from undefined to home in
  the upload of the next submission, ahead of every command buffer committed after the call, or
  ahead of its first copy; until then nothing holds it, so a texture released first is freed at
  once, with nothing submitted.
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
  `TextureSampleCount`, `TextureSwizzleWithWrites`). Texture buffers (`Gap::TextureBuffer`) and
  Memoryless storage (`Gap::MemorylessTexture`) are not built.
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
- The staging region is 16 MiB of the Shared type in an allocation of its own, made at the first
  copy, its pages written then, and kept for the device's life, so no copy allocates memory and
  the host writes resident pages. It is a ring: each copy takes a span at a multiple of 256 bytes,
  free again once the submission that carries the copy completes and no `get_bytes` reads it. The
  upload is submitted, without a wait, once its copies' spans take 8 MiB, so the GPU copies while
  the host writes the next. Where the ring has no room, the call waits for its oldest span's
  submission, submitting the upload first where the span is in it (`Metrics::upload_waits`). A
  copy larger than 16 MiB has a staging buffer of its own, which counts against the upload's
  64 MiB.
- A texture holds the device. Once its last handle and the last command buffer that used it are
  gone, its elements read null and its index is free at once; the reaper frees its image once the
  last upload that moves or copies it has retired. `Gauges::textures` counts images until then,
  `Gauges::descriptor_slots` the indices held.

### Known gaps of textures

- `replace_region` and `get_bytes` go through the staging region: a copy Metal does not make,
  counted in `Metrics::bytes_copied`.
- A texture whose format the device does not store through a storage image without a format has
  no storage element, so a shader write through its index is dropped, as a write to a nil
  texture is.
- `replace_region` takes no bytes into a 32-bit float depth (`Gap::FloatDepthUpload`): Metal
  keeps any `float` there, Vulkan copies no depth outside `[0, 1]` into a depth aspect without
  `VK_EXT_depth_range_unrestricted` (`VUID-vkCmdCopyBufferToImage-pRegions-07931`), which the floor
  does not require, and no recorded stream uploads one.
- Their bytes are not copied for a format of depth and stencil both or a multisample texture
  (`Gap::TextureBytes`).
- A texture's memory is device-local; where the device has none left, the call fails
  (`Error::Vulkan`).
- Texture views, heap textures and textures over a swapchain image are not built yet.

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

## Queues, command buffers and events

- `Device::new_command_queue(max)` is `newCommandQueueWithMaxCommandBufferCount:`
  (`newCommandQueue` is 64). Every queue submits to the device's one `VkQueue`.
  `CommandQueue::command_buffer` waits, as Metal does, while `max` of the queue's command buffers
  are made and not yet completed.
- A `CommandBuffer` records at once, with no replay log. Event waits and signals split it into
  segments, one `VkCommandBuffer` each where it has commands. Each ends with one global memory
  barrier: its writes are visible to every later command, and later commands run after its own,
  as Metal orders a queue's command buffers. The barrier also reaches the host where the segment
  wrote Shared memory (a Shared buffer bound or declared for writing, or a write through an address
  outside the dispatch's resources) or signals an event, so the host reads those writes once the
  command buffer completes or it sees the event; a segment that wrote only Private memory leaves
  the host out. No segment starts with a barrier.
  `enqueue` takes the command buffer's place in its queue; `commit` hands it to the submit
  thread and returns. A committed command buffer holds no device handle.
- The submit thread passes what is ready to `vkQueueSubmit2` in the device's commit order, each
  queue's command buffers in groups: a submission holds at most four command buffers of one
  queue, ahead of them the upload (zero fills, texture copies and new textures' moves to their
  home layouts), and only its last submit info signals the progress semaphore. The upload records
  at most 4,096 calls and keeps at most 64 MiB on its own account: the staging buffers of copies
  larger than the staging region, and the textures and blocks released while it held them. A call
  that finds it past either submits it and waits for it to complete before recording, so nothing
  grows without a commit. While fewer than two of a queue's submissions are not yet seen complete,
  each of its command buffers goes at once, alone. While two are, the ones committed meanwhile
  are held in one group, each scheduled as it joins, and the group goes when it holds four, when
  a completion leaves fewer than two in flight (the completion thread wakes the submit thread as
  it retires; no commit or wait is needed), when the host is to wait for one of them
  (`wait_until_completed`, or `CommandQueue::send_now`, the hint for a wait the runtime does not
  see), or at teardown. Waiting for a place in the queue sends nothing. A segment that signals an
  event ends its group, which goes at once; one that waits on an event's semaphore leads a new
  group, the one held going first. The thread holds a segment whose event waits are not met, and
  its queue's later command buffers behind it, while the segments before it run, so the host can
  see a signal the command buffer makes before its wait. Other queues go on. `Scheduled` means
  the last segment is in a submission: passed to `vkQueueSubmit2`, or in a group held.
- The completion thread retires command buffers as the progress semaphore reaches them, a
  submission's in commit order: the status, the blocks they kept, their recorders and their place
  in the queue, then the handlers. Scheduled and completed handlers run on it, every device's in
  turn, so no handler waits for a command buffer. A command buffer of a group is seen complete
  with the group's last: up to the GPU time of the three after it late. Only a queue already two
  submissions behind holds a group; one whose submissions have completed sends its next command
  buffer alone. Without a wait or a hint, the lateness is bounded only while two submissions are
  in flight, and the completion that leaves fewer sends the group. The worst case measured: four
  command buffers of 29 ms each in one submission on an RX 550 (the harness row `long_4cb`), the
  first seen complete 87 ms after it ended. A command buffer ended while its queue holds a group
  joins it, and retires in its place.
- A command buffer's times, its GPU times included, are nanoseconds of one host clock,
  `host_time()`: `CLOCK_MONOTONIC` on Linux and other Unix systems, `CLOCK_MONOTONIC_RAW` on
  macOS and the performance counter on Windows, each the host time domain the device calibrates
  its own clock against (`HOST_DOMAIN`). On macOS that clock counts while the system sleeps, where
  Metal's (`mach_absolute_time`) does not. Metal gives seconds. `kernel_start_time` is the commit
  and `kernel_end_time` the time it was scheduled. `gpu_start_time` and `gpu_end_time` are two
  timestamp queries, one written where its first commands begin and one where its last segment
  ends, read once it completes with no wait, and placed on the host clock by the device's last
  calibration: one `vkGetCalibratedTimestampsEXT` call reading the GPU's clock and the host's
  together, made as the device starts and again at a completion once 100 ms have passed since
  the last, never for each command buffer. A command buffer's GPU times so fall between
  `host_time()` before its commit and after its completion, within the calibration's deviation
  and the clocks' drift over 100 ms. Each queue has a pool of two queries per command buffer it may have made at once, all reset on the
  host as it is made (`hostQueryReset`); a command buffer's pair is reset on the host as it
  retires, those of the two its submissions wrote, so no command buffer the GPU runs resets a
  query. A command buffer with no commands, or whose last segment has none, records a
  `VkCommandBuffer` for its timestamps. One that ends with an error has GPU times of 0.
- `Event` and `SharedEvent` are timeline semaphores whose values are Metal's values. A host set
  (`SharedEvent::set_signaled_value`) raises a floor and never signals the semaphore; a wait the
  floor meets is submitted with no semaphore wait, and a GPU signal not above the highest value
  submitted or set is left out, so the semaphore only rises and never races a host set.
  `signaled_value` is the higher of the semaphore and the floor. `notify_listener` runs its
  listener on the completion thread when the value is reached, at a signal in the middle of a
  command buffer as well as at its end; `wait_until_signaled_value` wakes then too.
- `CommandQueue::abandon` ends every command buffer of the queue held at an event wait or not yet
  scheduled with `NotRun` and runs its handlers, as a Mac drops a dead process's held work, each
  retiring in the queue's order. Segments already submitted still complete, and so does a group
  held; the queue stays usable.
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
every block still handed out. It frees nothing: every block's memory and mapping stays valid until
the process ends. After it, no runtime thread touches a block; later calls that need a thread
return `Error::Quiesced`. It answers true once the GPU went idle and every thread ended in time. A
second call returns at once.

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

- Values the runtime reads back from GPU-written memory decide nothing beyond the reading command
  buffer's own results (`src/readback.rs`; a rule test holds every such read there). The sites:
  the two timestamp queries of a command buffer, which decide only its `gpu_start_time`, through
  arithmetic defined for every value and never later than its `gpu_end_time`; and
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
- Presenting (the binary semaphore a command buffer signals for a present) comes with presenting.
- An indirect dispatch's threadgroup counts are the GPU's to read, so a count above the device's
  `maxComputeWorkGroupCount` is not refused; Vulkan leaves such a dispatch undefined. A direct
  dispatch past it is split into dispatches of at most that many, each told its first
  threadgroup.
- `setBytes` takes up to 64 KiB, one ring chunk (`Limit::BytesLength`).
- A dispatch's threadgroup memory, the kernel's own and the lengths set, each rounded up to 16
  bytes, is at most the device's `maxComputeSharedMemorySize` (`Limit::ThreadgroupMemory`), where
  Metal's is its `maxThreadgroupMemoryLength`.

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
  before indirect argument reads gets one that does, in either kind of encoder. A segment records
  nothing else that touches memory; the barrier at a segment's end takes every command and the
  host.
- Parameters: fields in the push window, `[0, 256)` on every device, are pushed: the dispatch
  region and up to 28 buffer addresses, each kernel pushing only the bytes it reads. Fields past it
  go to a 16-byte aligned block in the parameter ring, whose address is a field in the window: a
  kernel with more than 28 buffers. `Counters::metrics`' `ring_writes` counts each block and each
  `set_bytes` copy.
  `set_bytes` copies its bytes into the ring at the call, as Metal copies them. The ring is 2 MiB
  blocks of the floor's ring memory type (device-local, host-visible, coherent), split into
  64 KiB chunks that a command buffer takes as it records and gives back when it retires; a block
  is made when no chunk is free and kept for the device's life.
- The command buffer keeps each pipeline state it dispatches with, and each texture and sampler
  state an encoder binds or declares, until it retires, so their elements of set 1 stay theirs
  while work may read them.

## Compute pipeline states

- `Device::new_compute_pipeline_state(&ComputePipelineDescriptor, sampler_ids)` is
  `newComputePipelineStateWithDescriptor:options:reflection:error:` without reflection. The
  translator compiles the kernel on the calling thread, which waits for it, within its 20-second
  budget per lowering, for what device creation enabled and under its identity domains: each
  buffer argument is the guest address of the bound buffer plus its offset, and each texture and
  sampler argument the bound object's resource ID, pushed; set 1 is the only descriptor set a
  kernel reads. A refusal of the translator is `Error::Translation`.
- A state numbers samplers as Metal's pipeline creation does (`NumberedSampler`, in Metal's
  order): each constexpr sampler its kernel uses, then one where it reads a texture and one where
  it writes one. `sampler_ids` takes that list once the kernel is lowered and answers an index
  for each, which the consumer numbers as Metal does, from its sampler grants
  (`Refusal::SamplerIds` for another count; an error it answers is the call's). A constexpr
  sampler the kernel samples through takes, before the call returns, the device sampler of each
  use the kernel has (`CombinedSampler::constexpr_sampler`, normalized; a pixel sampler's
  coordinate is scaled in the shader) at its index of the three sampler arrays, shared by create
  info with the sampler states'. States that number one constexpr sampler at one index share its
  elements (another sampler there is `Refusal::IndexInUse`); the last of them to go writes the
  arrays' defaults back and frees the index. The read and write samplers take nothing at their
  indices. Once `sampler_ids` has answered, a call that fails holds none of the indices.
- Every kernel is lowered first as its partial-threadgroups variant. A kernel that does not
  observe its threadgroup partition (it holds no threadgroup memory, barrier, fence or atomic of
  threadgroup or SIMD-group scope, SIMD-group or quad function, and reads no builtin of the
  partition; the translator decides from its AIR) keeps that module alone, which runs its grid in
  tiles: every direct dispatch runs in workgroups of 256 threads on one `VkPipeline`, made with the
  state, each workgroup one tile of the grid. The translator's `pipeline::grid_tile` picks the
  tile from the grid's extents alone: power-of-two sides whose product is 256, launching the fewest
  threads past the grid, then with the fewest tiles partly past it, then the shallowest, then
  (where the grid has two rows or more) the height nearest 2, then the widest. Each region carries
  the tile, and a thread's position is the tile's origin plus its index masked and shifted by the
  tile's sides, with no division. Where the tile does not divide the grid, an invocation past the
  grid returns before the kernel's body. An indirect
  dispatch runs at its own threadgroup size. A kernel that observes its partition is lowered a
  second time, plainly: its whole grids, `dispatch_threadgroups` and indirect dispatches run that
  module at their own size, and a `dispatch_threads` grid that is not a whole number of
  threadgroups runs the variant.
- A kernel whose SIMD groups run one per subgroup of 32 (pinned where the device lists compute in
  `requiredSubgroupSizeStages` with 32 between its sizes, or the device's own 32), or one per
  32-lane cluster of a 64- or 128-lane subgroup, runs only workgroups of whole subgroups.
- Such a kernel also has an emulated-SIMD-group variant, whose SIMD groups exchange through
  threadgroup memory between workgroup barriers. The reaper lowers it once the state is made,
  through the cache directory, and a threadgroup that is not whole subgroups (one of 80 or of 1
  thread, or of 32 on a device of 64-lane subgroups) runs it at its own size, the last SIMD group
  partly filled, as Metal runs it. A dispatch of such a size before the reaper is done waits for
  the lowering, which the reaper takes next: a first-use wait. In a threadgroup of one thread the
  module's exchanges pass no barrier.
- The dispatch region (two `uvec4`) gives every direct dispatch's kernel Metal's grid: its first
  threadgroup (0, or a split piece's), the threadgroup size packed as
  `(x-1) | (y-1) << 10 | (z-1) << 20`, and Metal's threads per grid; an indirect dispatch pushes a
  zero grid, and its kernel takes the grid from its threadgroup count. The packed size is always
  the pipeline's own workgroup size, so the kernel's `MultiRegion` constant stays at its default,
  0, and the kernel takes the size from its workgroup size, with no decode.
- A state makes one `VkPipeline` per module, threadgroup size and threadgroup memory lengths, on
  the encoding thread that first dispatches with them, which waits for it. Nothing the runtime records
  depends on when a compile finishes: there is one recorded form, made after the wait. Each such
  wait is counted in `Metrics::first_use_waits`, with its time in `first_use_wait_nanos` and the
  longest in `first_use_wait_max_nanos`.
- `ComputePipelineState::precision_notes` carries the translator's notes of where the kernel's
  answers may differ from Metal's on this device, fixed per operation: those of the modules made
  with the state, and the emulated-SIMD-group module's once it is lowered. A kernel compiled
  with fast math carries `FastMathRewrites`: a driver may fold its arithmetic differently per
  `VkPipeline`, so its answers in a tile and at another size may differ in the last place.

### The cache directory

`DeviceOptions::pipeline_cache_dir` names a directory, made where it is missing, that the device
keeps two things in, for compute pipeline states:

- `translations/`: each kernel lowering, named by its `FunctionCompileKey`, which covers
  everything that changes the lowering: the kernel's AIR, entry, constants and linked functions,
  its lowering inputs (the variant), the device profile's fingerprint and the translator's
  semantics version. A state whose lowering is there reads it in place of lowering
  (`Metrics::translations_loaded`, beside `Metrics::translations`); a variant the translator
  refused is kept too. The files take at most 256 MiB; past it the least recently used go.
- `pipelines-<pipelineCacheUUID>`: the `VkPipelineCache` data, which the driver keys by each
  pipeline's module and specialization. Every `VkPipeline` is made through the device's
  `VkPipelineCache`, which starts from this file. The reaper writes it once a `VkPipeline` was
  made and 10 seconds have passed since the last write, and the device's end writes it a last
  time. Data past 256 MiB is not written, and a file past it is not read: the cache starts empty.

Each write is a temporary file renamed over its name, and each file checks itself (the
translator's self-checking encoding, the pipeline data's checksum), so a file cut short is a
miss. A failed read or write is a miss too, and never fails a call. Devices open on one directory
share its files, each keeping the cap over the files it knows. With no directory, nothing is kept
on disk; the `VkPipelineCache` still serves the device's own pipelines.

### Known gaps of pipeline states

- No translation or `VkPipeline` compile runs ahead of the call or the dispatch that needs it, but
  the emulated-SIMD-group variant's lowering on the reaper: a state reads only its lowerings from
  the cache directory, and each `VkPipeline` past the first tile is made at its first dispatch,
  through the pipeline data.
- A kernel that observes its threadgroup partition is lowered twice at creation (its variant and
  its plain module). For the observing kernel the device tests run, a grid template over
  threadgroup memory, the second lowering took 0.61 to 0.63 ms and the whole creation 2.7 to
  2.9 ms on an RX 7900 XTX in a release build; a larger kernel's lowering takes longer.
- A kernel that reads anything the encoder does not supply yet, a function table, a
  `[[stage_in]]` fetch, constant data, or a sampler or texture state the translator would take
  from the bound object rather than from its resource ID, is refused (`Gap::KernelInput`). So is
  a state for indirect command buffers (`Gap::IndirectCommandBuffers`).
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
  is dispatched at.
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

## One path for every device

Every difference between devices comes from a Vulkan query: a feature, a property, a format, a
limit or a memory type. The crate holds:

- no vendor, device or driver identity;
- no environment gate;
- no runtime policy switch;
- no check that runs beside the program's work.

The test `tests::one_path` reads the crate's sources and fails on the first three.

## Floor

The runtime requires Vulkan 1.3 and every extension, feature, limit, queue family and memory type
below, of the instance and of every device. `PhysicalDevice::check` decides for one physical
device, with no unsafe call: it returns what the device offers above the floor
(`floor::Capabilities`), or `floor::BelowFloor`, a typed refusal that names every item the device
lacks (`floor::Missing`: the version, an extension, a feature, a limit, the queue family, a
memory type or a time domain). `Device::new` makes the last choice, of memory types, on the
created device.

### Extensions

Presenting runs on the runtime's own device; nothing is exported or imported.

| Extension | Where | Used for |
|---|---|---|
| `VK_KHR_surface` | instance | presenting |
| `VK_KHR_swapchain` | device | presenting |
| `VK_KHR_workgroup_memory_explicit_layout` | device | threadgroup memory laid out as Metal lays it |
| `VK_EXT_robustness2` | device | null descriptors |
| `VK_EXT_calibrated_timestamps` | device | GPU times on the host clock: the device calibrates its own clock (`VK_TIME_DOMAIN_DEVICE_EXT`) with the host's (`floor::TIME_DOMAINS`) |
| `VK_KHR_shader_maximal_reconvergence` | device, where listed | SIMD groups that reconverge as Metal's do |
| `VK_KHR_shader_subgroup_uniform_control_flow` | device, where listed | SIMD groups' reconvergence where maximal reconvergence is not offered |
| `VK_KHR_portability_enumeration` | instance, where offered | listing portability devices, each checked against the floor as any other |
| `VK_KHR_portability_subset` | device, where listed | Vulkan requires it enabled where a device lists it; the device is asked for the portability features below |

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
| `nullDescriptor` | `VkPhysicalDeviceRobustness2FeaturesEXT` | every texture element no texture owns: a nil, freed or unowned index reads zeros, sizes 0 and drops writes, as Metal's nil texture does. The robust buffer and image access beside it stay off |
| `samplerAnisotropy` | `VkPhysicalDeviceFeatures` | a sampler state's `maxAnisotropy`, which the device's own filter takes |
| `shaderInt16` | `VkPhysicalDeviceFeatures` | `short` and `ushort` in shaders |
| `shaderInt8` | `VkPhysicalDeviceVulkan12Features` | `char` and `uchar` in shaders |
| `storageBuffer16BitAccess` | `VkPhysicalDeviceVulkan11Features` | 16-bit values in buffers |
| `storageBuffer8BitAccess` | `VkPhysicalDeviceVulkan12Features` | 8-bit values in buffers |
| `scalarBlockLayout` | `VkPhysicalDeviceVulkan12Features` | buffer and push-constant blocks laid out as Metal lays them, where std430 cannot |
| `uniformBufferStandardLayout` | `VkPhysicalDeviceVulkan12Features` | uniform blocks in std430 layout |
| `workgroupMemoryExplicitLayout` | `VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR` | threadgroup blocks with an explicit layout |
| `workgroupMemoryExplicitLayoutScalarBlockLayout` | `VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR` | threadgroup blocks in scalar layout |
| `workgroupMemoryExplicitLayout8BitAccess` | `VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR` | 8-bit values in threadgroup blocks |
| `workgroupMemoryExplicitLayout16BitAccess` | `VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR` | 16-bit values in threadgroup blocks |
| `vulkanMemoryModel` | `VkPhysicalDeviceVulkan12Features` | the translator's atomics and barriers |
| `vulkanMemoryModelDeviceScope` | `VkPhysicalDeviceVulkan12Features` | atomics and barriers of device scope |
| `subgroupSizeControl` | `VkPhysicalDeviceVulkan13Features` | a kernel's SIMD groups at Metal's width of 32, where the device lists compute in `requiredSubgroupSizeStages` |
| `computeFullSubgroups` | `VkPhysicalDeviceVulkan13Features` | a kernel's subgroups required full, so each SIMD group is one subgroup |
| `robustImageAccess` | `VkPhysicalDeviceVulkan13Features` | an image write or atomic out of range dropped and a read out of range zero, with no guard in the shader |
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

### Limits

The descriptor limits are what the runtime's fixed arrays need: 65,536 sampled textures, 4,096
read-write textures and three arrays of 4,000 samplers, one per use (plain, depth comparison,
integer texture). 4,000 is Vulkan's minimum `maxSamplerAllocationCount`. The compute limits let a threadgroup take
Metal's 1,024 threads in any shape up to 1,024 by 1,024 by 64. A Boolean limit's minimum is 1,
`VK_TRUE`; `timestampValidBits` is the queue family's.

| Limit | Minimum | Used for |
|---|---|---|
| `maxPerStageDescriptorUpdateAfterBindSampledImages` | `65536` | the sampled-texture array |
| `maxDescriptorSetUpdateAfterBindSampledImages` | `65536` | the sampled-texture array |
| `maxPerStageDescriptorUpdateAfterBindStorageImages` | `4096` | the read-write texture array |
| `maxDescriptorSetUpdateAfterBindStorageImages` | `4096` | the read-write texture array |
| `maxPerStageDescriptorUpdateAfterBindSamplers` | `12000` | the three sampler arrays |
| `maxDescriptorSetUpdateAfterBindSamplers` | `12000` | the three sampler arrays |
| `maxSamplerAllocationCount` | `4000` | Vulkan's minimum: live samplers |
| `maxPerStageUpdateAfterBindResources` | `69632` | both texture arrays in one stage; a fragment stage needs its `maxColorAttachments` more |
| `maxUpdateAfterBindDescriptorsInAllPools` | `81632` | all five arrays |
| `maxPushConstantsSize` | `256` | per-command values: a dispatch region and 28 buffer addresses; anything larger goes in a block whose address is pushed |
| `maxComputeWorkGroupInvocations` | `1024` | Metal's most threads per threadgroup |
| `maxComputeWorkGroupSize[0]` | `1024` | a threadgroup's width |
| `maxComputeWorkGroupSize[1]` | `1024` | a threadgroup's height |
| `maxComputeWorkGroupSize[2]` | `64` | a threadgroup's depth |
| `timestampComputeAndGraphics` | `1` | GPU times of command buffers |
| `timestampValidBits` | `36` | GPU times of command buffers |

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

### Chosen by a query

Five features choose between two forms by the device's answer, and no other feature does: 16-bit
floats, storage-image reads without a format, 64-bit atomics, 64-bit floats and SIMD groups.
`floor::Capabilities` holds each answer the floor reads, and the features and extensions to
enable: the floor's, and each optional one the device has.

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

The test `tests::floor` holds these tables equal to the code, checks that each missing item is
named, and, under `--cfg m2v_gpu` or `--cfg m2v_cpu_icd`, checks every device the machine's
Vulkan loader lists.
