# metal2vulkan-rt

The Metal API on Vulkan, with one path for every device. It replaces `metal2vulkan-runtime` and
takes its name when it does; until then it is unpublished.

Its public API so far: the instance and the device, memory blocks and the buffers in them, command
queues, command buffers and events, the device's counters, and `floor`, the check of a Vulkan
device against the floor below.

## Instance and device

- `Instance::new` loads the Vulkan loader and creates the instance at the floor's version, with
  the floor's instance extensions and each extension the consumer names that the loader offers:
  the window system's, and any other it uses. `Instance::extensions` lists those enabled.
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
runs the handlers; the threads are joined and the GPU goes idle. Then every block is freed,
`BlockMappings::unmap` first for each block the consumer was handed, and the `VkDevice` is
destroyed. Teardown runs on the thread that drops the last handle, or, where that is a handler or
a listener, on the reaper once the handler returns; every gauge reads zero only once
the `VkDevice` is destroyed, so such a consumer waits for that before it drops its instance or
exits. A physical device and a device each hold their instance. No thread of the runtime outlives its device, and nothing relies on a
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
first-use waits, kernels lowered and kernels read from the cache directory);
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
  placed in a heap passes false, since Metal does not zero heap memory. A length of 0 and a range
  past the block are refused, as Metal answers nil for both.

The consumer's ranges follow Metal's retention: it reuses a range only once the work that named
the buffer there before has completed. A command buffer keeps each block it uses until it retires,
so a block released meanwhile goes back to the pool only then.

### The consumer's mappings

A consumer that maps blocks itself, into a guest for instance, passes `DeviceOptions::block_mappings`.
Before the runtime frees a block it handed out, it calls `BlockMappings::unmap(block)`, which
returns once every mapping the consumer made of it is gone and none of its threads touches the
block. The memory stays mapped and valid until then. It runs once per block: on the block-maker
thread; in `Device::quiesce`, on the calling thread, for every block still handed out; or at
teardown, on the thread that ends it, after the runtime's other threads are joined. It never calls
into the runtime. Blocks the consumer never took are freed without it, and so is every block freed
after a quiesce.

## Queues, command buffers and events

- `Device::new_command_queue(max)` is `newCommandQueueWithMaxCommandBufferCount:`
  (`newCommandQueue` is 64). Every queue submits to the device's one `VkQueue`.
  `CommandQueue::command_buffer` waits, as Metal does, while `max` of the queue's command buffers
  are made and not yet completed.
- A `CommandBuffer` records at once, with no replay log. Event waits and signals split it into
  segments, one `VkCommandBuffer` each where it has commands. Each ends with one global memory
  barrier: its writes are visible to every later command and to the host, and later commands run
  after its own, as Metal orders a queue's command buffers. No segment starts with a barrier.
  `enqueue` takes the command buffer's place in its queue; `commit` hands it to the submit
  thread and returns. A committed command buffer holds no device handle.
- The submit thread passes whatever is ready to one `vkQueueSubmit2`, in the device's commit
  order, ahead of it the upload of zero fills, and the last submit info signals the progress
  semaphore. It holds a segment whose event waits are not met, and its queue's later command
  buffers behind it, while the segments before it run, so the host can see a signal the command
  buffer makes before its wait. Other queues go on. `Scheduled` means the last segment has passed
  `vkQueueSubmit2`.
- The completion thread retires command buffers as the progress semaphore reaches them: the
  status, the blocks they kept, their recorders and their place in the queue, then the handlers.
  Scheduled and completed handlers run on it, every device's in turn, so no handler waits for a
  command buffer.
- A command buffer's times are nanoseconds of one host clock, `host_time()`: `CLOCK_UPTIME_RAW`
  on macOS (the clock of `mach_absolute_time`), `CLOCK_MONOTONIC_RAW` elsewhere on Unix, and the
  performance counter on Windows. Metal gives seconds. `kernel_start_time` is the commit and
  `kernel_end_time` the return of the `vkQueueSubmit2` that passed the last segment.
  `gpu_end_time` is when the completion thread saw the command buffer complete, so a queue's end
  times never decrease in completion order; `gpu_start_time` is that less the GPU's own duration
  of the command buffer, exact from two timestamp queries: one written where its first commands
  begin, one where its last segment ends, read once it completes, with no wait. Each queue has a
  pool of two queries per command buffer it may have made at once; a command buffer's first
  `VkCommandBuffer` resets its pair. A command buffer with no commands, or whose last segment has
  none, records a `VkCommandBuffer` for its timestamps. One that ends with an error has GPU times
  of 0.
- `Event` and `SharedEvent` are timeline semaphores whose values are Metal's values. A host set
  (`SharedEvent::set_signaled_value`) raises a floor and never signals the semaphore; a wait the
  floor meets is submitted with no semaphore wait, and a GPU signal not above the highest value
  submitted or set is left out, so the semaphore only rises and never races a host set.
  `signaled_value` is the higher of the semaphore and the floor. `notify_listener` runs its
  listener on the completion thread when the value is reached, at a signal in the middle of a
  command buffer as well as at its end; `wait_until_signaled_value` wakes then too.
- `CommandQueue::abandon` ends every command buffer of the queue that is held or not yet submitted
  with `NotRun` and runs its handlers, as a Mac drops a dead process's held work. Segments already
  submitted still complete; the queue stays usable.
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
  arithmetic defined for every value and never later than its `gpu_end_time`. The parameter ring
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
- Each dispatch records at once. What it touches is its pipeline's bound buffer slots, with the
  access the translator states for each, the `use_resource` declarations, and an indirect
  dispatch's arguments buffer (a read). The hazard sets (resources read and resources written
  since the last barrier) are kept by resource, a buffer being one resource, in a table indexed by
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
- Parameters: fields in the push window, `[0, 128)` on every device, are pushed; fields past it go
  to a 16-byte aligned block in the parameter ring, whose address is a field in the window.
  `set_bytes` copies its bytes into the ring at the call, as Metal copies them. The ring is 2 MiB
  blocks of the floor's ring memory type (device-local, host-visible, coherent), split into
  64 KiB chunks that a command buffer takes as it records and gives back when it retires; a block
  is made when no chunk is free and kept for the device's life.
- The command buffer keeps each pipeline state it dispatches with until it retires.

## Compute pipeline states

- `Device::new_compute_pipeline_state(&ComputePipelineDescriptor)` is
  `newComputePipelineStateWithDescriptor:options:reflection:error:` without reflection. The
  translator compiles the kernel on the calling thread, which waits for it, within its 20-second
  budget per lowering, for what device creation enabled and under its identity domains: each
  buffer argument is the guest address of the bound buffer plus its offset, pushed. A refusal of
  the translator is `Error::Translation`.
- Every kernel is lowered first as its partial-threadgroups variant. A kernel that does not
  observe its threadgroup partition (it holds no threadgroup memory, barrier, fence or atomic of
  threadgroup or SIMD-group scope, SIMD-group or quad function, and reads no builtin of the
  partition; the translator decides from its AIR) keeps that module alone: every direct dispatch
  runs in tiles of 256 threads, 256 by 1 for a grid 256 threads wide or wider and, for a narrower
  one, as wide as the grid rounded down to a power of two, so at most nine `VkPipeline`s; the
  256-wide one is made with the state. An indirect dispatch runs at its own threadgroup size. A
  kernel that observes its partition is lowered a second time, plainly: its whole grids,
  `dispatch_threadgroups` and indirect dispatches run that module at their own size, and a
  `dispatch_threads` grid that is not a whole number of threadgroups runs the variant.
- A kernel whose SIMD groups run one per subgroup of 32 (pinned where the device lists compute in
  `requiredSubgroupSizeStages` with 32 between its sizes, or the device's own 32) has an
  emulated-SIMD-group variant, whose SIMD groups exchange through threadgroup memory between
  workgroup barriers. The reaper lowers it once the state is made, through the cache directory,
  and a threadgroup that is not whole SIMD groups runs it at its own size, the last SIMD group
  partly filled, as Metal runs it. A dispatch of such a size before the reaper is done waits for
  the lowering, which the reaper takes next: a first-use wait. In a threadgroup of one thread the
  module's exchanges pass no barrier.
- The dispatch region (two `uvec4`) gives every direct dispatch's kernel Metal's grid: its first
  threadgroup (0), the threadgroup size packed as `(x-1) | (y-1) << 10 | (z-1) << 20`, and Metal's
  threads per grid; an indirect dispatch pushes a zero grid, and its kernel takes the grid from its
  threadgroup count.
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
- A kernel that reads anything the encoder does not supply yet, a texture, a sampler, a function
  table, a `[[stage_in]]` fetch or constant data, is refused (`Gap::KernelInput`). So is a state
  for indirect command buffers (`Gap::IndirectCommandBuffers`).
- A threadgroup that is not whole SIMD groups of a kernel whose SIMD groups run one per subgroup
  of 32 is refused (`Gap::PartialSimdgroups`) where the state has no emulated-SIMD-group module:
  for a kernel the consumer compiled (`StageSource::Compiled`), and where the translator refuses
  the emulated variant. The emulated module's scratch counts in the dispatch's threadgroup memory
  (`Limit::ThreadgroupMemory`).
- `dispatch_threads` with a grid that is not a whole number of threadgroups on some axis is
  refused (`Gap::PartialThreadgroups`) at a threadgroup that is not whole SIMD groups of a kernel
  whose SIMD groups run one per subgroup of 32, and for a kernel that observes its threadgroup
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
lacks (`floor::Missing`: the version, an extension, a feature, a limit, the queue family or a
memory type). `Device::new` makes the last choice, of memory types, on the created device.

### Extensions

Presenting runs on the runtime's own device; nothing is exported or imported.

| Extension | Where | Used for |
|---|---|---|
| `VK_KHR_surface` | instance | presenting |
| `VK_KHR_swapchain` | device | presenting |
| `VK_KHR_workgroup_memory_explicit_layout` | device | threadgroup memory laid out as Metal lays it |
| `VK_KHR_shader_maximal_reconvergence` | device, where listed | SIMD groups that reconverge as Metal's do |
| `VK_KHR_shader_subgroup_uniform_control_flow` | device, where listed | SIMD groups' reconvergence where maximal reconvergence is not offered |

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
| `synchronization2` | `VkPhysicalDeviceVulkan13Features` | barriers, layout changes and submission |
| `dynamicRendering` | `VkPhysicalDeviceVulkan13Features` | render passes |

### Limits

The descriptor limits are what the runtime's fixed arrays need: 65,536 sampled textures, 4,096
read-write textures and 4,000 samplers. 4,000 is Vulkan's minimum `maxSamplerAllocationCount`, so
every slot can hold a live sampler on every device. The compute limits let a threadgroup take
Metal's 1,024 threads in any shape up to 1,024 by 1,024 by 64. A Boolean limit's minimum is 1,
`VK_TRUE`; `timestampValidBits` is the queue family's.

| Limit | Minimum | Used for |
|---|---|---|
| `maxPerStageDescriptorUpdateAfterBindSampledImages` | `65536` | the sampled-texture array |
| `maxDescriptorSetUpdateAfterBindSampledImages` | `65536` | the sampled-texture array |
| `maxPerStageDescriptorUpdateAfterBindStorageImages` | `4096` | the read-write texture array |
| `maxDescriptorSetUpdateAfterBindStorageImages` | `4096` | the read-write texture array |
| `maxPerStageDescriptorUpdateAfterBindSamplers` | `4000` | the sampler array |
| `maxDescriptorSetUpdateAfterBindSamplers` | `4000` | the sampler array |
| `maxSamplerAllocationCount` | `4000` | a live sampler in every slot |
| `maxPerStageUpdateAfterBindResources` | `69632` | both texture arrays in one stage; a fragment stage needs its `maxColorAttachments` more |
| `maxUpdateAfterBindDescriptorsInAllPools` | `73632` | all three arrays |
| `maxPushConstantsSize` | `128` | per-command values; anything larger goes in a block whose address is pushed |
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
