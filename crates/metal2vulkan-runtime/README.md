# metal2vulkan-runtime

The Metal API on Vulkan. A program calls it the way it calls Apple's Metal (a device, command
queues, command buffers, the encoders, buffers, textures, heaps, pipeline states, events), and the
runtime runs the work on a Vulkan device through [`ash`](https://crates.io/crates/ash). It
translates each shader from Metal AIR to SPIR-V, owns the command buffers, submission and
barriers, and tracks hazards as Metal does.

> **Unstable.** Names and shapes may change before the runtime's API is frozen. Where the runtime
> cannot give Metal's answer exactly it refuses with a typed error, or announces a recorded
> near-equivalent; it never silently runs something else.

- Objects keep Metal's names without the `MTL` prefix (`Device`, `CommandQueue`, `Buffer`, ...).
  Methods are Metal's selectors in snake case: `newBufferWithLength:options:` is
  `Device::new_buffer`. Each item's rustdoc names its Metal counterpart and what differs or is
  refused.
- Rust ownership stands for retain and release. Handles are cheap to clone, `Send + Sync`, and
  equal when they are the same object; encoders are owned values that `end_encoding` consumes.
- Where Metal raises, ends the process, answers nil or fills an `NSError`, a call returns a typed
  `Error` ([Errors](#errors)).

## The crates

| Crate | Directory | What it is |
|---|---|---|
| `metal2vulkan-runtime` | `crates/metal2vulkan-runtime` (this crate) | The Metal API on Vulkan, built only on the translator's public API |
| `metal2vulkan` | [`crates/metal2vulkan`](../metal2vulkan) | The translator: AIR to SPIR-V for a described Vulkan device, and its compile-and-plan API ([docs/API.md](../../docs/API.md)) |
| `metal2vulkan-air` | [`crates/metal2vulkan-air`](../metal2vulkan-air) | The AIR front end |

The three are released together at one version. A program using the runtime depends on the
runtime and on `metal2vulkan`: the runtime's calls take the translator's descriptors and Metal's
value types, which it does not re-export (`metal2vulkan::pipeline::RenderPipelineDescriptor`,
`metal2vulkan::metal::PixelFormat`, ...).

## Depending on it

Not on crates.io yet. Depend on both crates from the same git revision (or checkout):

```toml
[dependencies]
metal2vulkan-runtime = { git = "https://github.com/steelbrain/metal2vulkan", rev = "<sha>" }
metal2vulkan = { git = "https://github.com/steelbrain/metal2vulkan", rev = "<sha>" }
# Only for the Vulkan interop calls (raw handles, adopting a VkBuffer or VkImage):
ash = { version = "0.38", default-features = false, features = ["std"] }
```

- Rust 1.98 or newer.
- Nothing links against Vulkan at build time. The runtime opens the Vulkan loader when it makes a
  device, or takes the consumer's `vkGetInstanceProcAddr` when it adopts one. A device needs
  Vulkan 1.2 and `timelineSemaphore`.
- A program that makes a `VkInstance` of its own beside a runtime device calls
  `program_instance_made()` right after it. A layer registers an exit handler as an instance is
  made (the validation layer's forgets every device it knows), and exit handlers run in the reverse
  order of their registration: unreported, that handler runs before the runtime's at exit, and a
  call the runtime's threads still make into the layer ends the process. The instance of a device
  the runtime adopts (`Device::from_raw`) needs no call.
- Shaders come in as a metallib (`Device::new_library_with_data`); there is no MSL compiler. Like
  Metal, `new_library_with_data` refuses bare AIR bitcode. For AIR bitcode, parse it with
  `metal2vulkan::library::Library::from_air` and pass the result to
  `Device::new_library_with_compile_library`. A Core Image kernel library (`metallib -cikernel`)
  loads as Metal loads it: an executable library whose kernels are extern functions
  (`Library::extern_function_names`), which Core Image links into an ordinary kernel library of
  its own through Metal; that library's kernel is what a pipeline state compiles.
- Every shader the runtime translates is kept on disk, by default in the platform's per-user cache
  directory, and in memory ([HOWTO](../../docs/HOWTO.md#the-runtimes-shader-cache)).
  `ShaderCacheOptions` names another directory; give each test its own.

## A first dispatch

From [`examples/compute_dispatch.rs`](examples/compute_dispatch.rs):

```rust,ignore
let device = Device::system_default()?;
let library = device.new_library_with_data(&metallib_bytes)?;
let function = library.new_function_with_name("scale_and_offset")?;
let state =
    device.new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())?;

let input = device.new_buffer_with_bytes(&input_bytes, ResourceOptions::shared())?;
let output = device.new_buffer((COUNT * 4) as u64, ResourceOptions::shared())?;

let queue = device.new_command_queue()?;
let command_buffer = queue.command_buffer()?;
let mut encoder = command_buffer.compute_command_encoder()?;
encoder.set_compute_pipeline_state(&state)?;
encoder.set_buffer(Some(&input), 0, 0)?;
encoder.set_buffer(Some(&output), 0, 1)?;
encoder.dispatch_threads(Size::new(COUNT as u64, 1, 1), Size::new(64, 1, 1))?;
encoder.end_encoding()?;
command_buffer.commit()?;
command_buffer.wait_until_completed()?;
// `output.contents()` now holds the kernel's results.
```

Creating a pipeline state translates its functions, each under the 20-second translation budget.
A function the device cannot run exactly is refused there, typed, naming what is missing. A stage
function made by `Library::new_function_with_name` whose function declares function constants is
refused before anything is translated (`Invalid`, `Problem::UnspecializedFunction`), as Metal ends
its process for it; make it with `new_function_with_name_constant_values` (no values are needed)
or `new_function_with_descriptor`, and a constant without a value is undefined (read as
[docs/API.md](../../docs/API.md)'s accepted differences, "Function constants", say).

A `[[visible]]` function a stage calls but does not define comes from the functions it links:
`Function::linked_function` of each, in the stage's `LinkedFunctions`
(`function.function_descriptor().with_linked(linked)`, `MTLLinkedFunctions`). A call resolves to
the linked function of its name, `Function::name`: the specialized name
`Library::new_function_with_descriptor` made it with (`FunctionDescriptor::specialized_name`,
Metal's `specializedName`), else its own, whatever the definition's type: the arguments pass by
position ([docs/API.md](../../docs/API.md) section 3, "Linked functions: types"). A linked
function computes with the constant values it was made with, and a constant it was given no value
for reads as undefined, as Metal's does; one
made by `new_function_with_name` that reaches a constant of its module refuses the pipeline state
(`Compile`, `LinkProblem::UnresolvedConstant`), as Metal fails its creation. A function in the
stage's `binary_functions` is one a visible-function table calls: no call by name resolves to it
(`Compile`, `NeedsInput(LinkedFunction)`, as Metal's creation fails), and beside a function of its
name in `functions` the call calls that one while a table holds both, each with its handle. A
function there that was not made with `FunctionOptions::COMPILE_TO_BINARY` fails the creation as
plain Metal's does (`MetalError`, code 2), and one added to a made state
(`*_with_additional_binary_functions`) fails that call (`MetalError`, code 1). A function Metal's
stitcher made links the same way, privately or not, from the stitched library's bytes
(`Device::new_library_with_data`); the runtime makes no stitched library itself
(`newLibraryWithStitchedDescriptor:` is not built).

Creating a render pipeline state also makes the `VkPipeline` its first draw is predicted to take,
before it returns, where Metal compiles. The prediction comes from the descriptor and the device
alone: the pass of its attachments, its input primitive class, each vertex-descriptor layout's
buffer bound at an aligned offset (a layout of `MTLBufferLayoutStrideDynamic` at the stride that
packs its attributes), Metal's default sampler at each runtime sampler slot, no texture, and
Metal's default encoder state. A pipeline's key holds no pass's resolves, which render pass
compatibility ignores, so a draw in a pass that resolves takes the predicted one. A point-class
state's prediction takes the variant its draws of points take under Metal's guard band, and a
multisampled state that blends into or reads an extended-range color (`BGRA10_XR` and its kin) the
variant that holds the color's `half` shadow, as its draws in a pass that clears the color do. A
draw whose key differs makes its own `VkPipeline` at that draw. Tessellated and mesh states,
separate depth and stencil formats and fragment functions that write depth make nothing at
creation.

Where the device links its render pipelines from graphics pipeline libraries
(`VK_EXT_graphics_pipeline_library` with fast linking; MoltenVK makes them whole),
a linked `VkPipeline` is the device's, found by exactly what its linking takes: its four
libraries and its pipeline layout. Render states whose whole key is equal share it even where
each is a compiled pipeline of its own, whose stages come out as the same words (descriptors that
differ only in the call stack depth of a function that recurses through no visible function, for
one): the first links it, and each later one takes it with no driver call
(`WorkCounts::linked_pipelines_shared`).

## Examples

Each example runs on the system default device and uses only public API. Without an argument it
runs the repository's own fixture
[`runtime_examples.metal`](../metal2vulkan-validation/fixtures/public/examples/runtime_examples.metal),
assembled from its committed LLVM text by `llvm-as` (LLVM 21 or newer) on `PATH`. With an argument
it runs that metallib instead, for example one built with
`xcrun -sdk macosx metal -o runtime_examples.metallib runtime_examples.metal`.

| Example | What it shows |
|---|---|
| [`compute_dispatch`](examples/compute_dispatch.rs) | A library from bytes, a compute pipeline state, buffers, one dispatch, the result read on the CPU |
| [`render_to_texture`](examples/render_to_texture.rs) | A render pipeline state, a render pass that clears and draws into a `Private` texture, and a blit that reads it back |
| [`indirect_command_buffer`](examples/indirect_command_buffer.rs) | Compute commands encoded on the CPU, read back as typed records, and executed after `use_resource` |
| [`heap_placement`](examples/heap_placement.rs) | A placement heap, textures at offsets `heap_texture_size_and_align` answers, and fences between the encoders that use them |
| [`adopted_buffer`](examples/adopted_buffer.rs) | A `Shared` buffer over a `VkBuffer` and `VkDeviceMemory` the caller made, the shape of a dma-buf import |

```sh
cargo run -p metal2vulkan-runtime --example compute_dispatch
cargo run -p metal2vulkan-runtime --example render_to_texture -- runtime_examples.metallib
```

## Opening a device

**Made by the runtime.** `Device::system_default()` is `MTLCreateSystemDefaultDevice()`: the first
physical device the runtime can open, in the order discrete, integrated, virtual, CPU.
`Device::all()` is `MTLCopyAllDevices()`. Their `_with` forms take `DeviceOptions`, the
Vulkan-side choices Metal has no counterpart for:

| `DeviceOptions` field | What it chooses |
|---|---|
| `policy` | The translator's `DevicePolicy` ([docs/API.md](../../docs/API.md#the-policy)): containment, and the fast or exact form of each accuracy choice ([FAST-PATHS](../../docs/FAST-PATHS.md)). |
| `kinds` | Which `DeviceKind`s qualify; empty admits every kind. |
| `loader` | A Vulkan loader library to open instead of the default ones (the platform's loader by name, then on macOS `$VULKAN_SDK/lib`, `/opt/homebrew/lib` and `/usr/local/lib`). If none opens, `Error::LoaderNotFound` lists each path tried. |
| `layers`, `layer_settings` | Instance layers, in order, and their settings. |
| `instance_extensions`, `device_extensions` | Extensions beyond those the runtime enables (a swapchain, for example). |
| `optional_device_extensions` | Device extensions each device enables only where it can (it lists one, the instance has what it requires, and the device has its features), never refused for one; `Device::enables_extension` says which. A listed extension's feature structure is enabled with it where the runtime knows one (`VK_KHR_video_maintenance1` and `2`'s, `VK_KHR_unified_image_layouts`' with `unifiedImageLayoutsVideo` where the device has it), here and for `device_extensions`. |
| `extra_queues` | Queues each device makes for the consumer beside its own, one per `VkQueueFlags` entry (a video decode queue, for example), of the first family whose flags contain the entry's; `DeviceHandles::extra_queues` returns them. A device with no queue for an entry is made all the same, says so (`metal2vulkan-extra-queues-absent`), and leaves it out. The runtime never submits to them. Empty by default. |
| `reserved_queues` | How many queues of its own queue families each device reserves for the consumer out of the round robin its command queues share, which shrinks by each (on an RTX 5080, 13 `VkQueue`s rather than 14); `DeviceHandles::reserved_queues` returns them. The round robin keeps its first four and presenting its two, so a device with none to spare past those (MoltenVK's four families of one queue, lavapipe's one queue) reserves fewer, is made all the same, and says so (`metal2vulkan-reserved-queues-absent`). No command queue, presenting or not, and no internal copy ever submits to them. 0 by default. |
| `queue_limits` | Fewer `VkQueue`s for each device than its queue plan takes (`QueueLimits`: the most for the round robin its command queues share, at least one, and for presenting queues), so making and dropping a device costs less: on an RTX 5080, whose plan takes 18 queues, a device of 17 took 445 ms to make and 110 ms to drop, and one of 2 (one round-robin queue and the staging queue) 145 ms and 44 ms. Past the limits command queues share `VkQueue`s (one waits behind another's whole command buffer), and presenting and unshared ones may share. Where the limits change a placement from what the plan gives, the device says so in an informational message (`metal2vulkan-queue-limits-reached`), at the first such change alone, and counts each later one (`WorkCounts::queue_limits_reached_withheld`). A device with no queue to spare (RADV's, lavapipe's one graphics queue) takes its plan. `None` (the default) is the plan. For tests and tools that make many devices. |
| `instance_api_version` | The instance's `apiVersion` (1.2 when unset); a version outside 1.2 to the loader's is refused, never clamped. |
| `withheld_features`, `withheld_extensions`, `withhold_generated_commands`, `withhold_view_reinterpretation` | Parts of the recommended enables to leave off; the device then compiles for what is enabled. |
| `messages` | A `MessageSink` for the validation layer's messages and the runtime's notices; without one, notices go to stderr. `MessageSink::with_least_severity` skips messages below a severity (never a degrade notice), and the runtime then formats none of its own lines it would skip. |
| `shader_cache` | Where translated shaders are kept, and how much. |
| `gpu_times_from_host` | Answer command buffers' GPU times from the host clock, sparing two timestamps per command buffer. |
| `announce_each` | Degradation kinds to announce at every occurrence instead of once. |
| `lowering_refuses` | Degradation kinds that refuse a render encoder's call when its lowering after the call meets them (`RenderLoweringPolicy::Worker`), failing its command buffer, instead of being announced. Empty by default. |
| `record_executions` | Keep, per command buffer, which pipeline each dispatch and draw ran (`CommandBuffer::executions`). Off by default. |

**A driver short of resources.** Where the driver refuses a device with its queue plan's `VkQueue`s
for want of its resources (`vkCreateDevice` answering `VK_ERROR_TOO_MANY_OBJECTS`,
`VK_ERROR_INITIALIZATION_FAILED`, `VK_ERROR_OUT_OF_DEVICE_MEMORY` or `VK_ERROR_OUT_OF_HOST_MEMORY`:
on NVIDIA's driver, other devices of the GPU, in this process or others, holding what it needs,
such as its channels), the runtime asks once more with its least
plan's: a round robin of one `VkQueue` and none reserved for presenting, the reserved, extra,
staging and fill queues kept (`queue_limits` of one and none, without that option's messages). A
device made so announces `metal2vulkan-degrade-least-queue-plan` once, as it is made: its command
queues share that one `VkQueue`. A device the driver refuses again, or whose least plan takes no
fewer queues, is refused, `Error::DeviceRefused`. On an RTX 2080 with a 256 MiB BAR (NVIDIA
620.70.06), sixteen processes started 3 s apart, each making one device: seven devices were made
with 17 `VkQueue`s and two more with 2 (one round-robin queue and the staging queue). The driver
refused the other seven with both plans, and the kernel logged that it could not allocate a
channel's instance block. In one process, the driver refused the eighth device with either plan, on
that RTX 2080 and on an RTX 3090.

While a handle lives, equal options give the same device for a physical device, as Metal has one
`MTLDevice` per GPU. Nothing is read from the environment but the platform's default cache
directory. `Device::registry_ids_with` lists each qualifying physical device's `registryID`,
preferred first, without making any device, and `Device::with_registry_id` makes only the device
of one ID (runtime extensions). `Device::identities_with` reads the identity of each of those
physical devices, in the same order, with no device made, and `Device::system_default_identity_with`
the first of them, the one `Device::system_default_with` makes its device of: its name, kind, IDs,
`deviceUUID`, `registryID`, Vulkan version, the size of its largest `DEVICE_LOCAL` memory heap
(`recommended_max_working_set_size`, named after Metal's nearest property, whose estimate it is
not), and its PCI bus address and DRM render node where the physical device lists
`VK_EXT_pci_bus_info` and `VK_EXT_physical_device_drm`, which a consumer addresses the same GPU
through another API by. The same two are
`DeviceProperties::pci_bus` and `DeviceProperties::drm_render` on a device that exists; both are
`None` on macOS, where MoltenVK lists neither extension.

**Adopted from the consumer.** `unsafe Device::from_raw(RawDevice, DevicePolicy)` works on an
instance and device the consumer made and keeps. `RawDevice` takes the `vkGetInstanceProcAddr`,
the instance and its version, the physical device, the device, the queue family and index, and
exactly what the device enabled (an `EnableSet`), plus what the runtime may use outside that
vocabulary (calibrated timestamps, sparse features, view format reinterpretation, host image copy)
and the same per-device choices as `DeviceOptions`. Dropping the runtime device destroys only what
the runtime made. The program's last drop of the device and its objects, on a thread of the
program's, returns once that is done: it stops the runtime's own background work on the device (its
check of textures over host pages, its warm pass) and waits for it, at most a round of the check or
the record the pass is making, and for the release a handler thread makes of a no-copy buffer's
pages that a keep-alive alone waits for (their free, then the keep-alive's drop), so a keep-alive's
drop must not wait for the thread that drops the device.

**At the process's exit.** On NVIDIA's proprietary driver, drop the runtime's devices before the
process exits. A device kept to the exit can end it with `SIGSEGV` inside the driver's own exit
handler, which tears the driver's SPIR-V compiler down while a worker thread of the driver's still
runs in it, where the device's check of textures over host pages ended a few seconds or less before
the exit: on an RTX 3090 (620.70.06), 6 of 80 exits right after the check, 0 of 80 three seconds
after it, and 0 of 80 with the device dropped first
([HOWTO](../../docs/HOWTO.md#8-check-the-integration)).

**The Vulkan objects underneath** (no Metal counterpart; each call is `unsafe`):

- `Device::raw()` gives the entry, instance, physical device, `ash::Device`, queue families, and
  the extra queues the options asked for (`ExtraQueue`: flags, family, `VkQueue`), which the
  consumer alone submits to. The runtime's own buffers and images are shared among its own queue
  families only, so work on an extra queue of another family uses the consumer's own objects.
  The reserved queues (`ReservedQueue`: family, `VkQueue`) are of the runtime's own families, so
  the consumer's work there uses the runtime's buffers and images with no ownership transfer
  (while they live, ordered against the runtime's work through semaphores, a runtime image left
  in `GENERAL`), and waits behind no command queue's command buffers.
  `CommandQueue::with_vk_queue` lends a queue's `VkQueue`, with the `VkDevice` it belongs to and
  its family, under the runtime's lock: the consumer's submissions there sit in that command
  queue's order, and in that of any other command queue sharing the `VkQueue`. From its first call the command queue keeps that `VkQueue` for its life.
- `Texture::vk_image`, `Buffer::vk_buffer`, `Event::vk_semaphore` and `SharedEvent::vk_semaphore`
  give an object's handle. Handing out a texture's image, or a buffer's `VkBuffer` whose zeros or
  bytes still wait for the device's next submission (a new `Private` buffer's), first makes that
  work, the handle's own alone, by a submission it waits for, so the consumer's own work on the
  handle comes after it: a buffer's on the device's staging queue where it reserved one (a copy
  engine; or, on a device where no memory the host caches holds a texture's image, such as an
  RX 7900 XTX under RADV, a queue of compute work beside its own), behind no other work of the
  program's, else on the device's first `VkQueue`, behind what was submitted there before.
  A texture over a buffer's bytes that aliases them gives a linear image (the same one for textures
  of one image over the same bytes of one import of host pages, which share it), which over a
  no-copy buffer's or dma-buf host pages may be of `VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT`
  (`DRM_FORMAT_MOD_LINEAR`, its one memory plane at the texture's `bytes_per_row`): ask its layout
  of memory plane 0.
- `CommandBuffer::encode_wait_for_vk_semaphore` and `encode_signal_vk_semaphore` order a command
  buffer against the consumer's own work.
- An event's timeline semaphore, plain or shared, orders the consumer's submissions against
  command buffers' `encodeWaitForEvent:` and `encodeSignalEvent:` of the event. Its counter is a
  signal index, not the event's value: `vk_wait(value)` answers the counter a submission waits for
  (or that the value is reached, or not yet signalled), and `unsafe reserve_vk_signal(value)`
  reserves the counter one submission signals, after the counter it names, or answers that Metal
  drops the signal (`Event` and `SharedEvent` have both). A command buffer's wait for a plain
  event's reserved signal goes to its `VkQueue` with a Vulkan wait for the counter, maybe before
  the consumer's submission is made, so the consumer signals from a queue no such command buffer
  is ahead of (an extra or reserved queue). A shared event's value may be set from the host
  (`setSignaledValue:`), which no Vulkan wait can answer, so a command buffer's wait for its
  reserved signal is held until the device's watcher saw the signal land: an event the host never
  sets avoids that hold as a plain one. The device watches for the consumer's signals landing and
  keeps the event's value in step.
- `Device::new_texture_with_vk_image` makes a `Private` texture over the consumer's `VkImage` (a
  swapchain image, for presenting). Placed `IdPlacement::Outside`, it takes no resource ID, so the
  program's textures are numbered as with no window.
- `Device::new_buffer_with_vk_buffer` makes a buffer over the consumer's `VkBuffer` and memory,
  such as an imported dma-buf, made with `Device::vk_buffer_usage()`. The runtime never frees it.
- `Device::new_texture_with_vk_image_plane` makes a `Private` texture over one plane of the
  consumer's multi-planar `VkImage` (`R8`/`RG8` of NV12, `R16`/`RG16` of P010), as
  `newTextureWithDescriptor:iosurface:plane:` does of an IOSurface plane: the consumer's image is
  `CONCURRENT` across the runtime's queue families and its own, `MUTABLE_FORMAT` and
  `EXTENDED_USAGE`, rests in `GENERAL`, and outlives the texture; the runtime views the plane in
  place and never moves its layout, takes ownership or frees it. A copy, fill, mipmap generation
  or render pass of the plane is refused (`RuntimeGap::VkImagePlaneCopy`).
- `Buffer::vk_host_memory` answers where a no-copy buffer's bytes lie in the device's import of
  the consumer's pages (memory, offset, size, type, `HOST_ALLOCATION` handle type), where they are
  imported by their pointer in one allocation, so the consumer binds a buffer of its own usage
  (a video decode's bitstream source) to the same bytes; `None` for copied, dma-buf, read-only or
  detached pages.
- `Device::new_exportable_buffer` makes a buffer whose memory another API imports (CUDA's
  `cuImportExternalMemory`, another `VkDevice`), in memory allocated exportable as an opaque file
  descriptor: its own dedicated allocation where the device needs one (it exports such memory
  only so, or requires one; a driver's preference alone does not make one), else a block
  that exportable buffers share; every other buffer stays in the device's other pooled blocks.
  `Buffer::export_memory_fd` answers a new descriptor each call, the consumer's to close, with the
  allocation's size, the buffer's offset and length in it, whether it is a dedicated allocation
  and its memory type. An import of a shared block reaches the whole block, of which the buffer
  is its offset and length. The buffer must outlive every access through an import; an import
  may outlive it unused, since each descriptor holds a reference to the allocation's payload. On
  a device loss the memory's contents are lost for the imports too, and a device made again exports new allocations. A
  device without `VK_KHR_external_memory_fd`, or that exports no buffer's memory so, refuses it
  (`RuntimeGap::ExportableMemory`).
- A buffer descriptor's `exported_as` (`MemoryExport::OpaqueFd` or `MemoryExport::DmaBuf`) makes
  such a buffer of either handle type with the descriptor's other fields: `aligned_byte` lays that
  byte on a multiple of 256 bytes of the exported memory, and `padding_unobserved` lets textures
  over it alias it with rows apart. A texture over such a buffer is a linear image made for the
  memory's handle type, as Vulkan requires of an image bound in exported memory; such an image
  starts `UNDEFINED`, so it takes the memory in place where the device's in-place check of that
  exported memory passed, and a texture made before the check answered is a shadow that moves in
  place at a later commit (docs/HOWTO.md, section 7, says where each plane must lie). Over a
  dma-buf export where no image binds in the memory itself (RADV's 780M), the texture's image is a
  `DRM_FORMAT_MOD_LINEAR` one bound to a dedicated import of that memory's own dma-buf, in place
  as well (`WorkCounts::exported_dma_buf_images`). An opaque file descriptor's memory the driver
  exports only as the buffer's dedicated allocation takes no image: its textures are shadows,
  announced. Dma-buf-exportable buffers share blocks of their own kind, or
  take a dedicated allocation where the device needs one. `Buffer::export_memory_dma_buf`
  answers a new dma-buf each call, with what `export_memory_fd` answers; its `offset` is the
  buffer's first byte, so a VA-API `DRM_PRIME_2` surface's plane lies at `offset` plus the plane's
  offset in the buffer. A device without `VK_KHR_external_memory_fd`,
  `VK_EXT_external_memory_dma_buf` and `VK_EXT_queue_family_foreign`, which the runtime enables
  wherever listed, or that exports no buffer's memory as a dma-buf, refuses it
  (`RuntimeGap::DmaBufExport`). `NoCopy` contents with `exported_as` are refused
  (`Problem::ExportedNoCopy`).
- `Buffer::external_write_done` says the other API's writes to an exported buffer completed (the
  consumer waited for them: `vaSyncSurface`, a CUDA stream's synchronize). Vulkan needs an acquire
  from the queue family reserved for that API before the device reads them
  (`VK_QUEUE_FAMILY_FOREIGN_EXT` for a dma-buf, `VK_QUEUE_FAMILY_EXTERNAL` for an opaque
  descriptor): the device's next submission records it, of the whole `VkBuffer`, before any of its
  commands, so the call waits for no GPU work; a host-visible buffer's acquire is made at once and
  waited for alone. A texture over the buffer that reads it through a shadow (rows no image of the
  memory holds, such as a decoder's pitch) copies the bytes in again at its next use in a command
  buffer committed after the call; without the call, it may keep reading what it copied in before.
  Call it once after each external write, while none of the runtime's commands that use the buffer
  is in flight. Metal has no counterpart.
- `Device::new_exportable_shared_event` makes a shared event whose timeline semaphore another API
  imports (CUDA's external semaphores, another `VkDevice`), made exportable as an opaque file
  descriptor; `SharedEvent::export_semaphore_fd` answers a new descriptor each call. Its counter
  is the signal index the event's interop hooks name: the importer waits for the counter
  `vk_wait` answers and signals only one `reserve_vk_signal` reserved. The event outlives every
  import and every signal and wait an importer submitted; on a device loss its state is lost for
  the imports too. A device without `VK_KHR_external_semaphore_fd`, or that exports no timeline
  semaphore so, refuses it (`RuntimeGap::ExportableSemaphore`).

## The memory model

- **Storage modes** keep Metal's meaning. `DeviceProperties::has_unified_memory` answers from the
  device type: yes for an integrated GPU or a CPU implementation, no for a discrete GPU. Without
  unified memory a `Managed` buffer has two copies, as on a Mac with a discrete GPU
  (`did_modify_range` and a blit's `synchronize_resource` move the bytes), a `Managed` no-copy
  buffer's included (its CPU copy the caller's pages; **Host memory without a copy**, below), and
  a `Shared` resource lies in host-cached system memory (announced once,
  `shared-in-system-memory`). On a device whose family of transfers alone has a second queue
  (NVIDIA's copy engine), the copies in after `did_modify_range`, and new buffers' zeros and
  staged bytes, run on that queue before the command buffer that takes them, rather than on the
  queue its kernels run on, where they come to 1 MiB or more
  (`WorkCounts::fill_queue_submissions`). There a command buffer takes and waits for only the
  fills of memory its commands may reach (`WorkCounts::fill_queue_waits_unused` counts those
  that waited for fills they reach none of).
- **Past device-local memory.** Where no device-local memory type a `Private` buffer or heap, a
  texture's image or (without unified memory) a `Managed` buffer's own bytes takes has room, it
  lies in system memory, as Metal makes such resources past a discrete GPU's memory rather than refusing
  them (announced once, `device-memory-overflow`; counted in
  `WorkCounts::device_memory_overflows`); a `Managed` no-copy buffer then lies in the caller's
  pages, as a `Shared` one does (`WorkCounts::managed_device_copies_in_place`). Where the device lists
  `VK_EXT_pageable_device_local_memory`, the runtime enables it and the driver may page memory out
  to make room instead. A resource Metal itself refuses (longer than `max_buffer_length`, past a
  texture's limits) is refused as before.
- **New resources** read what plain Metal's measured answer is: a new buffer reads zero. A
  host-visible buffer longer than 32 MiB lies, where the device imports host pointers, in fresh
  anonymous pages imported by their pointer, zero as mapped, so that nothing writes its zeros; a
  recycled one is set to zero on the host. A new `Shared` or `Managed` texture reads zero: where
  it copies on the host, its image is zeroed on the host as it is made, or, where the device has
  `VK_EXT_zero_initialize_device_memory` and the image lies in memory no resource used, starts in
  the driver's zeros. A new
  `Private` texture answers as the Mac the device presents (`MacClass`, which
  `DeviceOptions::mac_class` declares, by default the class of the device's GPU: an Apple silicon
  Mac on an Apple GPU, a Mac with an Intel or AMD GPU on every other;
  `DeviceProperties::lossless_compression` states it): an Apple silicon Mac's reads zero or that
  Mac's fixed pattern of a compressed texture; a Mac with an Intel or AMD GPU's holds what its
  memory held, written by nothing (`WorkCounts::private_textures_unfilled`), or on an Apple GPU,
  whose own compressed texture would read Apple's pattern, zero
  (`WorkCounts::private_textures_zero_filled`) (docs/API.md, accepted differences). A new
  resource's first GPU work (its image's first layout, its zeros) runs at the start of the next
  command buffer committed on any queue, or on the fill queue (above); a command buffer on
  another queue waits for it only where its commands, or its own first work, may reach the new
  resource's memory (`WorkCounts::first_work_waits_unused` counts those that waited for first
  work they reach none of).
- **The Metal address space.** Every buffer and heap has a `gpu_address`, and textures, samplers,
  pipeline states and indirect command buffers a `gpu_resource_id`. Translated shaders read both
  from argument buffers, as on Metal 3. A consumer may pin either (`AddressPlacement`,
  `IdPlacement`, `Device::reserve_resource_ids`); a pin the space cannot honour is
  `Error::PlacementConflict`. The runtime's own buffers (its in-place check's among them), and
  the storage of objects Metal shows no address of (a texture buffer the device makes, a visible
  function table), lie outside the space, so no pin meets them and no buffer's address depends
  on them.
- **Host memory without a copy.** `unsafe Device::new_buffer_with_bytes_no_copy` imports host
  memory, given as a `HostMemory`, through `VK_EXT_external_memory_host`; memory the device cannot
  import exactly is refused, but where the driver refuses the pointer itself (below).
  `unsafe Device::new_heap_with_bytes_no_copy` is its heap twin.
  `Buffer::rebind_bytes_no_copy` and `Heap::rebind_bytes_no_copy` move either to another range,
  keeping the objects, addresses and IDs. A rebind is refused (`RebindProblem::EncodedUses`) while
  a command buffer not yet committed, or an open encoder, used the bytes; a render encoder whose
  attachment lies in them uses them from its open. Where a driver refuses the pointer (RADV
  refuses shared mappings, a `memfd`'s among them), memory made with `HostMemory::with_dma_buf` is
  imported from dma-bufs of its pages instead
  ([HOWTO](../../docs/HOWTO.md#6-host-memory-a-driver-refuses)), and a no-copy buffer or heap
  over memory made without a source is a copy of its pages, as below. A device that enables no
  `VK_EXT_external_memory_host` but imports dma-bufs imports only those: memory made with a
  source is imported from its dma-bufs, and a no-copy buffer or heap over memory made without
  one is a copy of its pages, as where a driver refuses the pointer (a rebind to such memory is
  refused, `Unsupported`, naming the extension); on NVIDIA's driver the runtime
  never hands the driver pages the kernel cannot pin (a dma-buf's mapping): they take the source,
  and else, or where its import fails, a copy, as below. A no-copy buffer's import of such
  dma-bufs that only blits have read may be released while idle and imported again at its next
  blit, which then costs time: RADV lists every import in every submission. Nothing of Metal's
  observes it. A no-copy buffer, heap, or buffer's or heap's rebind whose import the caller's gate
  refuses (`HostMemory::with_import_gate`), or whose admitted import fails, and a no-copy buffer
  or heap whose pointer the device refuses, or on a device that imports no pointer, with no
  dma-buf of its pages serving it (no source given, or one that failed or whose dma-bufs the
  device refused), is made over a copy of
  its bytes in memory of the device's own, announced (`copied-host-pages`,
  [HOWTO](../../docs/HOWTO.md#copied-host-buffers)), wherever the process maps the pages
  readable (a dma-buf's mapping among them); over pages not mapped readable it is refused. A
  rebind's textures follow it onto the copy. A buffer over exactly the bytes a live copy covers
  lies in that copy, as one over a live import's bytes lies in that import.
  Without unified memory a `Managed` no-copy buffer is two copies instead
  (`HostBacking::DeviceCopy`): the GPU's in device-local memory of the runtime's own, which every
  command reads and writes, and the caller's pages, which `contents` answers. The pages are
  copied in as the buffer is made, by `did_modify_range` (its range) and by a rebind (whole),
  before any command of a command buffer committed after the call; a CPU write no
  `did_modify_range` names is not seen by the GPU, as on a Mac with a discrete GPU. A blit's
  `synchronize_resource` copies the GPU's copy out to the pages in its command buffer's order.
  The copies run from an import of the pages where the device makes one, else (the gate refused
  it, the device refused the pointer) through staging buffers held until their work completed,
  at most 256 MiB staged ahead (`WorkCounts::managed_staging_bytes`): never through a copy of the
  pages as well. Over pages the process maps read-only the GPU's writes are dropped: nothing is
  copied out, and the pages are copied in again at the completion of each command buffer that
  used the buffer (`managed_writes_undone`). A detach keeps the GPU's copy, which commands go on
  reading and writing; `contents` answers the detached pages' pointer, as every detached no-copy
  buffer's does (`Buffer::detach_backing`). The device-local memory it holds is counted in
  `managed_device_copy_bytes`.
  `CommandBuffer::used_resources` lists what a command buffer uses until it completes, so a
  program knows when it may take memory away. The pages under a no-copy object are unmapped only
  after the object goes and its keep-alive is dropped (or, given none, its deallocator ran), never
  before: Vulkan requires imported pages to stay mapped while their import lives, and the
  runtime's threads free imports and submit work over them at any moment
  ([HOWTO](../../docs/HOWTO.md#unmapping-host-memory)). A no-copy buffer made over part of a live
  one's bytes, or over the same bytes as memory of another class, is bound inside the live one's
  import (one live import of a page: Vulkan leaves the same pages imported twice undefined) and
  holds it, so the live one's keep-alive is dropped, or, given none, its deallocator runs, only
  once both are gone; with a keep-alive its deallocator still runs at its release, as Metal's
  (`WorkCounts::host_releases_held_inside`).
- **Hazards** are tracked as Metal tracks them. A heap's resources are untracked by default, so
  encoders that share them order themselves with fences. Commands that reach a resource through an
  address or ID in memory need it resident (`use_resource`, `use_heap`), as on Metal. The runtime
  orders a tracked buffer by the bytes a command reaches: those its translation proves a shader
  reads or writes, else from the bound offset to the buffer's end. Vulkan's synchronization
  validation takes a shader's buffer as accessed over its whole descriptor, which spans those
  bytes widened to 16-byte granules from the device's offset alignment, so it reports hazards
  between commands whose bytes are apart, as it does between untracked uses Metal leaves to the
  program; no order of either changes a result.
- **Textures over buffers** share the buffer's bytes. Over a no-copy buffer's host pages or such
  a long buffer's fresh pages, an image bound to them would start `UNDEFINED`, and Vulkan lets its
  first move, and a render pass's load and store, rewrite any byte bound to it: by default
  (`HostPageTexturePolicy::InPlace`) a texture there that samples, reads, renders into it or is
  written through a shader (of texels of 1, 2, 4, 8 or 16 bytes, the narrow ones checked apart)
  is an image in place wherever the
  device makes it and its own check showed those operations keeping every byte for that kind of
  pages (but the rows' padding of a texture over a buffer a kernel writes, which an x86 Mac's GPU
  may change too: docs/API.md, section 10), on a discrete GPU as on any other (run once per driver and release on a thread of its
  own, kept in the shader cache). The runtime keeps a copy and announces it (`shadowed-texture`,
  naming why) for every other texture there (under `HostPageTexturePolicy::Shadow`, one a shader
  writes of texels of a width the check stores none of, one made before the check answered or
  whose check did not
  pass), where the device's linear images cannot hold Metal's layout, and where the rows lie with
  bytes between them (but over a buffer whose maker declared them unobserved,
  `BufferDescriptor::padding_unobserved`, below). A texture over a two-copy `Managed` no-copy
  buffer's bytes is over the GPU's copy, as over a `Managed` buffer of the runtime's own. On a
  discrete GPU, a `Managed` texture over the bytes of a no-copy buffer that lies in its pages
  (not an IOSurface plane's) that only samples or reads, whose
  policy takes the image in place, over memory that carries a log of its writes
  (`HostMemory::with_dirty_pages`), tracked and laid out as one copy, is instead such a copy in
  the device's memory, announced nowhere, whatever the check answered: the GPU's copy of the two
  a `Managed` resource is on a Mac with a discrete GPU, its copies in only the rows the log
  reports changed (`WorkCounts::device_copy_textures`). A `Shared` texture and an IOSurface's
  stay in place there, read across the bus as a discrete Mac's GPU reads them. Exact either way;
  integrated GPUs, CPU devices and memory without a log keep the image in place.
- **Two-plane YCbCr textures** (`YCBCR8_420_2P` and its kin over a surface's planes) hold an
  image of the texels Metal reads, converted from the planes before a command buffer's first use
  where they changed. Where `has_unified_memory`, a function that only samples or gathers a
  `YCBCR8_420_2P` or `YCBCR10_420_2P` texture reads the planes' codes in place instead and converts them in the
  shader, exactly as the image holds them (`WorkCounts::ycbcr_textures_in_place`,
  `ycbcr_planes_bound`; a texture that keeps the image counts why, `ycbcr_images_kept_*`).

## Runtime extensions

Calls Metal does not have, for a consumer that emulates a Metal device:

- `Device::new_command_queue_unshared(count)` makes a command queue that shares its `VkQueue` with
  no other where the device has `VkQueue`s to share, for one whose command buffers must never wait
  behind another queue's (a compositor's); a device with one graphics queue (RADV, ANV, lavapipe)
  has none to spare, so there the other queues' batches are held back from that `VkQueue` while 2
  ms or more of their expected GPU work is in flight on it (`UnsharedIsolationPolicy::Bounded`, the
  default), and its command buffers wait behind about one batch of theirs, the whole of a long
  one (docs/FAST-PATHS.md, known gaps; counted:
  `WorkCounts::unshared_command_buffers_behind_other_queues`, and its wait's tallies). Under `QueueSharingPolicy::Adaptive` (the default), where the device's own check
  finds its GPU idle between consecutive work of two `VkQueue`s (NVIDIA's driver), the other
  command queues move among the `VkQueue`s of their family at their idle points and share them
  ([docs/API.md](../../docs/API.md#the-policy)); this one keeps the `VkQueue` it takes for its
  life, no queue moves onto it, and a queue already there leaves at its next idle point. A
  shared queue waits behind another's long command buffer once per queue that goes long: the
  first long command buffer of a queue it shares a `VkQueue` with holds its command buffers for
  that command buffer's whole length (beside 30 ms command buffers on an RTX 5080, a compositor's
  64 completions over 10 ms and one of 30 ms). Make a queue that must never wait so (a
  compositor's) with this call: on that GPU, none of its completions took over 10 ms in any run.
  Under every policy it takes a `VkQueue` no other command queue holds where one is free, and no
  command queue made later takes that `VkQueue` while another is free of unshared queues (every
  command queue takes, as it is made, the `VkQueue` the fewest live queues hold); where none is
  free it shares the one the fewest hold, counted
  (`WorkCounts::unshared_queues_sharing_a_vk_queue`). A count of 0 and a lost device are refused
  as `new_command_queue_with_max_command_buffer_count` refuses them.

- `CommandBuffer::commit_failed(CommitFailure)` commits a command buffer as failed, running the
  encoders Metal would still run and keeping its signals, as a consumer measured on an M3 Max.
- `CommandBuffer::set_submission_check` runs a check as the command buffer is submitted, and fails
  it as Metal fails a submission. It runs on the thread that submits it: the scheduler's, one in
  `commit_and_submit`, or the one that makes the pass of a `commit_and_submit` left for a render
  encoder's end, so it must take no lock that thread holds and wait for no GPU work. It
  runs once the command buffer's event waits are met as Metal submits it: an `Event` wait for a
  signal already submitted is met at once and waits on the `VkQueue`; a `SharedEvent` wait that
  only a GPU signal still running reaches is held until the signal lands (measured on an M3 Max,
  the conformance case `event-wait-behind-a-running-signal`).
- `CommandBuffer::add_submission_handler` is called when the runtime takes the command buffer's
  last batch for its queue (submits it, or holds it for one kernel submission with later ones;
  what another queue's command buffers had held then goes to the driver before it runs, and the
  completion of one that needs no GPU work, taken with no submission behind no work in flight,
  never waits for it);
  `add_executed_commands_handler` each time an execution's indirect records are read.
- `CommandBuffer::add_completion_seen_handler` is called once the command buffer completed, its
  status, error and GPU times final, on the runtime's thread that saw it complete (its queue's
  waiter) where its transition to `Completed` runs there, before its completed handlers and
  before `wait_until_completed` returns: one with no completed handler then finishes with no
  handler thread woken. One with completed handlers too has its status made final and these run
  there, and the rest of its transition (its completed handlers on) goes to its queue's handler
  thread, where its queue's later transitions follow it; unless another queue's command buffer
  may complete after it (it signals an event at its end). It runs on the queue's handler thread
  where the transition does (an earlier transition of the queue still there, a command buffer that
  signals an event at its end with completed handlers, or the transition of another queue's
  command buffer whose GPU signal it waited for not yet at that one's handlers: it completes after
  that one, as Metal's completion follows the GPU's). It must be short and never
  block; a wait in it for what has not happened is refused
  (`DeadlockCause::CompletionSeenHandler`).
- `CommandBuffer::commit_and_submit` commits as `commit` does, then submits on the calling thread
  where no other submission pass is under way: the ready command buffers in commit order, this
  one among them, each needing no host step and no switch of a storage form, up to the pass's
  bound (under `CompletionPolicy::Grouped`, on a driver whose batches may wait in a group, one
  goes alone where its `VkQueue` runs at most one earlier submission not seen complete, and is
  left to the scheduler's thread to group where two or more run; one held back, another queue's waiting for its check or an event, holds back only the
  later commits that wait behind it, as on the scheduler's thread), their submission checks run
  inside the call just before each submission (unless one waits for a shared event's GPU signal
  to land, which the scheduler's thread waits for; a plain event's wait goes to the `VkQueue`, as
  Metal submits an `MTLEvent` wait at once) and their submission handlers after it, in submission
  order with every other command buffer's. What it leaves goes to the scheduler's thread, which
  it wakes, as it does after a batch it submitted alone where the batch may wait in a group. No
  wake and no thread hop lies on its way to the `VkQueue`, where `commit`'s submission is the
  scheduler thread's, as Metal's is its own thread's
  (`WorkCounts::scheduler_passes_at_commit`, `batches_submitted_at_commit`,
  `batches_checked_at_commit`). Under `RenderLoweringPolicy::Worker`, one whose last render
  encoder's end the lowering thread has not lowered returns as `commit` does, and the thread that
  runs its commit once the end is lowered (the lowering thread, or one waiting for the command
  buffer) makes that pass, its checks and handlers with it.
- `RenderCommandEncoder::set_color_resolve_target`, `set_depth_resolve_target` and
  `set_stencil_resolve_target` change where a pass resolves until it ends (the last target set
  wins), as Metal's private calls do.
- The four draw calls, and an indirect render command's two, take a `DrawPrimitiveType`, which
  every `PrimitiveType` converts into, and
  whose `LineLoop`, `TriangleFan`, `Quads` and `QuadStrip` are the values 5 to 8 Metal's draws take
  besides its header's (`DrawPrimitiveType::from_raw`): each draws as the list plain Metal draws,
  in Metal's vertex order, direct, indexed and indirect, and quads and a quad strip count one
  `[[primitive_id]]` a quad, as Metal does, where the list draws two triangles. Its `Nine`, value
  9, draws nothing and fails the command buffer as Metal's GPU does: a hang
  (`CommandBufferError::GpuHang`), a page fault or a victim of the GPU's recovery
  (`GpuRecoveryVictim`), by the draw's form, running nothing from its encoder on (docs/API.md,
  "Accepted differences", for the timing and the race over the encoders after a fault); one that
  fails at more than one encoder fails with the first one's error, in the time Metal's GPU takes
  to reach and recover from each;
  a queue's second hang or page fault revokes it, as Metal's does: its later command buffers fail
  at their commit (`AccessRevoked`). A kernel's draw of a value past the header's draws points,
  as Metal's does.
- `TextureDescriptor::address_mode_map` presents a texture as sampling by other address modes than
  its sampler's.
- Resource ID reservations, placements and counts (`Device::reserve_resource_ids`,
  `IdPlacement`, `Device::sampler_ids_live`), and the private getters of indirect commands.
- `execute_commands_in_buffer_checked` and `execute_commands_in_buffer_indirect_checked` (render and
  compute encoders) take an `IndirectCommandCheck` and what it looks for
  (`IndirectCommandCheckNeeds`). Wherever the runtime reads an execution's records, at `commit` for
  the CPU's or in the host step for records the GPU may have written, it calls the check with each
  command it will record, before it records any; an execution whose records cannot hold what the
  check looks for never calls it, and runs on the GPU alone where it can.
- `IndirectCommandBuffer::gpu_written` says whether a route that may write a buffer's records met
  it: its command buffer made it resident for writing (`ResourceUsage::WRITE`) and set, in any
  encoder, before or after, a pipeline state whose functions may call an indirect-command operation
  (`indirect_command_writes`, read from their AIR), or executed commands whose records may name
  one; or a blit copied or reset commands into it. Only then does an execution of it wait for the
  work before it and read its records on the host. A buffer made resident for writing beside
  functions that call no such operation runs its executions from the CPU's records, as one made
  resident for reading does (`WorkCounts::indirect_write_residencies_unmarked`). What its records
  may hold (`command_writes`), which decides whether an execution runs on the GPU alone, takes what
  every such state of the command buffer may write, whichever encoder set it.
- `HostMemory::with_dma_buf` and its `DmaBufSource`: dma-bufs of a no-copy object's pages, for a
  device that refuses their pointer.
- `HostMemory::with_dirty_pages` and its `DirtyPages` (with `DirtyStamp`): the consumer's log of
  the writes to a no-copy object's pages, by which every copy of them into memory of the device's
  own (a shadow of a texture over them, a copied buffer or heap over them) copies in only the
  pages changed since its last copy, on every device. The runtime's own writes into the pages are
  reported back: a copy's (a shadow's copy back, a copied buffer's merge back) naming the copy as
  their writer (`DirtyPages::wrote_runs_as`), and the GPU's writes into pages imported in place
  and the runtime's CPU writes into them as the consumer's (`DirtyPages::wrote_runs`). Each write
  is one call to each log, carrying every run of it there (an object over a guest RAM's scattered
  pages is many runs of the RAM's log); by default those call `wrote_as` or `wrote` once a run.
  A command that replaces every texel of such a shadow is merged back, as one that leaves some is:
  copied in (the changed pages), kept, and only the words it changed written into the pages.
- `HostMemory::with_import_gate` and its `ImportGate` (with `ImportRequest`): the consumer's
  gate over a no-copy buffer's, heap's or rebind's import of its pages; a refusal makes a copy of
  them.
- `Buffer::host_backing` (with `HostBacking`): what the GPU uses for a no-copy buffer's bytes now,
  or a no-copy heap's buffer's: the caller's pages, imported by their pointer, as dma-bufs or in a
  guest RAM's chunk, a copy of them (`copied-host-pages`), which Metal never makes, or, for a
  `Managed` one without unified memory, a copy in device-local memory (`DeviceCopy`); a rebind
  may change it ([HOWTO](../../docs/HOWTO.md#copied-host-buffers)).
- `BufferDescriptor::padding_unobserved` (`with_padding_unobserved`): the maker of a buffer of
  the runtime's own memory declares that nothing reads the bytes between a texture's rows over it
  (each row's after its texels to its pitch, the last row's included) or between its slices, so
  such a texture is a linear image aliasing the buffer where the device lays one out at its
  pitches, or else a `DRM_FORMAT_MOD_LINEAR` image at them where the device's in-place check
  passed for its pooled memory (`WorkCounts::pooled_pitched_images`), not a shadow, and those
  bytes may change whenever it is made or used; no other byte does
  (`WorkCounts::unobserved_padding_aliases`). Such a buffer lies in a plain block of the
  runtime's memory, where on a unified-memory device whose other buffers lie in blocks allocated
  exportable (docs/FAST-PATHS.md, the `Private` row) a block's making holds the process's
  submissions on RADV (`WorkCounts::plain_buffer_blocks_made`). No Metal counterpart; refused
  (`PaddingUnobservedNoCopy`) with `NoCopy` contents.
- `BufferDescriptor::aligned_byte` (`with_aligned_byte`): a buffer of the runtime's own memory
  laying one of its bytes on a multiple of 256 bytes of its memory, so a texture whose texels start
  there (a decoded frame's plane 32 bytes into its surface, say) is a linear image aliasing the
  buffer where the device lays one out, not a shadow, on a device whose linear images start on a
  multiple of more than the texture's offset (RADV's 256). Its contents start that byte's remainder
  short of a multiple of 256, not on a page as Metal's do; its length and `gpuAddress` are Metal's.
  No Metal counterpart; refused (`AlignedByteNoCopy`) with `NoCopy` contents.
- `Buffer::texture_placement` and `texture_placement_with_iosurface_plane` (with
  `TexturePlacement` and `TextureShadowReason`): where a texture `new_texture` or
  `new_texture_with_iosurface_plane` makes of the same arguments lies if made now: in place over
  host pages, a shadow pending the device's in-place check, a shadow and why, no host pages, or
  another shape. The making's own route answers it, after every check the making runs before it
  makes anything, so the same arguments are refused with the same errors; an answer in place runs
  the image's making up to its bind (one or two unbound `VkImage`s, released), so a pitch or an
  image the device refuses answers the shadow the texture then is.
  `Device::settle_in_place_checks` starts the check and says when its verdicts are known (a zero
  timeout waits for nothing). `Buffer::tiled_texture_placement` answers it for
  `new_tiled_texture`'s arguments.
- `HostRam`, `Device::use_host_ram` (with `HostRamUse`) and `HostMemory::with_runs` (with
  `RunResolver`): a guest's RAM imported whole, in 64 MiB chunks, for a device that refuses views
  of guest pages made through dma-bufs; a no-copy object whose pages the consumer resolves to one
  run inside one chunk is bound in the chunk, with no import of its own
  ([HOWTO](../../docs/HOWTO.md#guest-ram-imported-whole)). `HostRam::with_dma_buf` gives the RAM a
  source of dma-bufs of its pages: a device whose own check finds held pointer imports taxing
  every submission (RADV), or that refuses the RAM's pointer, imports each chunk as one dma-buf.
  `HostRam::with_dma_buf_of_runs` gives it a source of one dma-buf of a list of runs (udmabuf's
  `UDMABUF_CREATE_LIST`): there a copy by runs over a chunk not imported imports lists of its own
  pages in place of the chunk, which pins every page of it, each list kept past its command buffer
  for the copies of the same pages after it (at most 64 MiB and 16 lists, each released once no
  copy took it for 2 s).
- `BlitCommandEncoder::copy_from_host_ram` and `copy_to_host_ram`: runs of a used guest RAM
  (host addresses from a 4-byte word to any byte) gathered into consecutive bytes of a buffer,
  or scattered back, from the device's chunk imports: runs that start and end on a word (whole
  pages, or rows of 4-byte texels) by one compute dispatch per chunk the runs touch, other runs by
  a copy region per run in a chunk; over a chunk not imported yet, where the device's thread
  imports ahead, through memory of the command buffer's own that the CPU copies from or into
  the pages; neither touches another byte of their pages; with no view
  of the pages and no import of their own; ordered and seen as a copy between the buffer and a
  no-copy buffer over the pages
  ([HOWTO](../../docs/HOWTO.md#guest-ram-imported-whole)). `copy_runs_from_host_ram` and
  `copy_runs_to_host_ram` take each run at a byte of the buffer of its own, so a buffer that
  keeps a surface's layout (rows at a pitch, the padding between them untouched) is one call for
  every row.
- `HostBytes`: bytes at a pointer that may change while a call reads them (a guest's memory),
  which `Texture::replace_region_with_host_bytes` and `BufferContents::CopyHost` copy once through
  the pointer, where the slice forms would need the caller's own copy first.
- `DeviceOptions::pipeline_records` (and `RawDevice::pipeline_records`) is called once per
  pipeline state creation (compute, render, mesh and tile, and each
  `*_with_additional_binary_functions`) with a `PipelineRecord`: the creation's descriptor, every
  field, its libraries named by content hash and kept once in the shader cache directory. A
  compute state's direct dispatch of a new shape whose `VkPipeline`s it made (its threadgroup,
  its grid's ragged edge and its threadgroup memory lengths) hands it one more record, the
  creation's descriptor with that dispatch, at most 16 per state.
  `PipelineRecord::to_bytes` and `from_bytes` carry it to a later process; `from_bytes` answers
  `None` for a record of another format, so a consumer keeps records across updates and drops
  those. Equal creations make equal records.
- `Device::warm(records, byte_budget)` stands in for Metal's own pipeline caches across
  processes. It returns at once and, on one background thread of the device's (the utility class
  on macOS, its pipeline makings ranked after every other making of the process), makes what each
  record's creation makes, in order: its translations into the shader cache and its predicted
  `VkPipeline`s' data into the driver's records, so the program's later creation of the same
  descriptor finds them, and for a dispatch's record that dispatch's `VkPipeline`s, planned with
  nothing bound, so the program's later dispatch of that shape finds their data. It stops once the driver's pipeline cache data its makings added reaches
  `byte_budget` bytes; data a making finds in the directory already adds nothing, so over the
  directory whose creations made the records the budget binds only where a making writes new
  data. It takes no identifier, resource ID or sampler number, makes no neutral,
  announces no degradation and keeps nothing for the program's kept variants, and it reads no
  translation of another translator and no creation record of another `RUNTIME_VERSION` (a record
  holds a descriptor, never a translation), so no answer the program gets changes. Nothing fails:
  a record it cannot read (another format, damaged, or a library the directory no longer holds)
  is counted and dropped, the device's first announced as the warning
  `metal2vulkan-warm-record-unread`, and a failed creation is counted and dropped. A mesh render
  pipeline's record is not made (its creation keeps nothing past the state), and a layout that
  bakes an immutable sampler is translated only. `Device::warm_statistics` counts the records
  made, each record's outcome, the dispatches' `VkPipeline`s made and the driver bytes added.

[docs/API.md section 10](../../docs/API.md#10-accepted-differences) lists where the runtime
answers otherwise than Metal on purpose.

## Errors

Every failure is a value of `metal2vulkan_runtime::Error`. Match its variants; never read the
message text. The ones a consumer meets most:

| Variant | Means |
|---|---|
| `Invalid(Invalid)` | Misuse plain Metal itself refuses, ending its process or failing the call: `call` names the method, `problem` the typed `Problem`. A program that runs on Metal never makes it. |
| `Nil(Nil)` | What plain Metal answers nil for, the process going on. |
| `MetalError(MetalError)` | What plain Metal answers nil for with an `NSError`: Metal's `domain`, `code` and description as measured, for a consumer to pass on. |
| `Unproven(Unproven)` | What the runtime cannot prove plain Metal's answer for. A consumer that answers as Metal stops its guest, naming the problem. |
| `Unsupported(Unsupported)` | The device lacks what an exact answer needs: `feature` names the Metal feature, `needs` what would change the answer. |
| `NotImplemented(RuntimeGap)` | Plain Metal the runtime does not build yet. Metal takes the input, so a later build may too. |
| `AppleOnly(AppleOnly)` | An Apple-private call off Metal's default with no Vulkan form, or whose meaning on Apple's GPU is unmeasured. Calls that state Metal's default are accepted. |
| `Compile(...)` | A library or pipeline the translator refused, over budget or cancelled ([docs/API.md section 8](../../docs/API.md#8-refusals-requested-inputs-and-precision)). |
| `OutOfMemory`, `DeviceLost`, `Vulkan`, `NoDevice` | The device's own failures. `NoDevice`: the loader opened but found no usable device. A device holding `maxMemoryAllocationCount` allocations live has the next refused (`Vulkan`, `VK_ERROR_TOO_MANY_OBJECTS`) without a call of the driver, counted in `WorkCounts::memory_allocations_refused_at_limit`, once the frees its releases posted to its free thread (which count until they run) completed. |
| `LoaderNotFound(LoaderNotFound)` | The Vulkan loader did not open; `tried` lists each path. |
| `DriverSetting(DriverSetting)` | A driver setting the runtime cannot run under, such as MoltenVK with `MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS` off. |
| `DeviceRefused(DeviceRefused)` | The driver refused to make the device for want of its resources (`VK_ERROR_TOO_MANY_OBJECTS`, `VK_ERROR_INITIALIZATION_FAILED`, `VK_ERROR_OUT_OF_DEVICE_MEMORY` or `VK_ERROR_OUT_OF_HOST_MEMORY`), with its queue plan's `VkQueue`s and then with its least plan's; `attempts` names each. Other devices of the GPU, in any process, hold what it lacks; once fewer are live it may be made. |
| `WaitWouldDeadlock(DeadlockCause)` | A command buffer wait that would deadlock, refused at once instead of hanging. |
| `ShaderCache(...)` | The shader cache directory cannot be made or written. |

```rust,ignore
match device.new_render_pipeline_state(&descriptor, &PipelineOptions::default()) {
    Ok(state) => use_it(state),
    Err(Error::Unsupported(gap)) => report_device_gap(gap.feature, gap.needs),
    Err(Error::NotImplemented(gap)) => report_runtime_gap(gap),
    Err(Error::AppleOnly(call)) => report_apple_only(call),
    Err(other) => return Err(other.into()),
}
```

A refused call records nothing. A command buffer that fails while it runs ends with status
`Error` and an `error()`; `metal_code()` gives Metal's code, and `origin()` whether Metal or the
runtime failed it. What a translated shader reports (a resident-set miss, for one) is a fact of
the run, not an error: `CommandBuffer::execution_report` reads it once the command buffer
completed.

A draw whose pipeline the driver did not make after the draw returned (the runtime makes a draw's
pipeline while the draw goes on) runs nothing, and its command buffer runs on without error:
`CommandBuffer::dropped_draw` names the first, counted: `WorkCounts::draws_dropped_unmade`. A
dispatch's such failure fails its command buffer (`CommandBufferError::PipelineCreation`,
`CommandBuffer::failed_pipeline`).

A draw or dispatch (a compute encoder's, a render pass's tile dispatch, an execution replay's
draw) whose pipeline the driver is still making `DevicePolicy::pipeline_commit_wait` (4.5 s by
default) after its command buffer's commit runs nothing too (`PipelineWaitPolicy::Bounded`, the
default; a render or compute state's creation likewise waits `DevicePolicy::pipeline_creation_wait`,
4.5 s by default, at most for the pipeline it hands to the runtime's pool), its command buffer
running on, and the command buffers after it draw and dispatch with the pipeline once made. One
call so waits for the driver's makings at most: a state's creation, its creation wait; a `commit`,
until its commit plus its commit wait (its commands share that one deadline, and the commit records
there only what it can, an execution of render commands the CPU writes among them); neither waits
on the other's bound, so a host step that creates a state and commits waits 9 s at most by default,
under a guest's 10 s FIFO check. Made until made under either value, in the call that makes them: a
draw that records nothing, a draw in a pass of custom sample positions or recorded again into
another instance of its pass (in the draw's call or an execution's replay), a device replay's draws
(in its `commit`), and a making that failed, made again by its command's recording. `dropped_draw`
names the first with the error `VK_TIMEOUT`, counted: `WorkCounts::draws_dropped_still_making` and
`dispatches_dropped_still_making`, announced `draw-dropped-still-making` and
`dispatch-dropped-still-making`. `PipelineWaitPolicy::Exact` waits until the pipeline is made.

## A lost device

A device the runtime made survives the loss of its `VkDevice` (`VK_ERROR_DEVICE_LOST`), as Metal's
device survives a GPU recovery. The first call that needs the device after the loss makes a new
`VkDevice` of the same physical device, options, queue families and extensions; the calls made
meanwhile wait for it.

- The command buffers in flight fail: the oldest incomplete one with `GpuHang`, the rest with
  `GpuRecoveryVictim`. Until the recovery, a command buffer committed on the hang's queue runs
  nothing and completes with no error, as on Metal's hang's queue. One made before the recovery
  and committed after fails with `GpuRecoveryVictim`. A queue's second hang revokes it, as
  Metal's does: a command buffer committed on it from then on, before the recovery too, fails
  with `AccessRevoked`.
- The `Device` and its `CommandQueue`s go on: what they make from then on is made on the new
  `VkDevice`. A `Device` call that the loss meets while it runs is not refused for it: it runs
  again on the new `VkDevice`. One that takes host memory, a deallocator or a range token, which a
  second run could not have, runs on the new `VkDevice` where the loss came before it took them,
  and completes on the lost one where the loss came after, its object then one made before the
  recovery.
- The recovery carries the events, the uses of guest RAM (`HostRamUse`), the no-copy buffers, the
  `Private` buffers and the buffers in the runtime's own pages to the new `VkDevice`: the same
  objects, at the same addresses. An event keeps the value the host saw and its listeners'
  notifications. A buffer in the runtime's own pages keeps its bytes and its `contents` pointer:
  a long host-visible buffer's fresh pages, and, on a device whose checks found that a held import
  of host pages costs its submissions nothing and that textures over such pages take in place
  what they take over the device's own memory, the blocks of pages the `Shared` and `Managed`
  (unified memory) buffers lie in, imported again on the new `VkDevice`. A no-copy buffer is
  made again over the same pages, so it keeps its bytes and its pointer (a two-copy `Managed` one
  makes its GPU copy again and copies the pages in: what the GPU wrote and no synchronize copied
  out is lost, as a `Private` buffer's contents are); its import gate, run
  resolver and dma-buf source are asked again, on the thread that recovers, and must not call
  into the device. Its deallocator runs at its release once no `VkDevice` imports its pages. A
  `Private` buffer's contents are lost: it reads zero.
- Any other object made before the recovery is refused where it meets the new `VkDevice` (an
  encoder's bind, a pipeline made from its functions) and in its own calls that need its lost one:
  `Error::NotImplemented(RuntimeGap::MadeBeforeRecovery)`. That covers pooled `Shared` and
  `Managed` buffers in memory the device allocated, whose `contents` pointer is the lost memory's
  mapping (on a device that imports no host pointer, or whose checks found that a held import
  costs its submissions or that textures over such pages take less, or had not answered; and a
  `Managed` buffer without unified memory), detached no-copy buffers, heaps, textures, pipeline
  states, libraries and exportable memory or events. Make it again.
  Metal's objects go on. The recovery's notice counts the buffers and events it carried and left.
- The lost `VkDevice` is destroyed once the objects left on it went and, where `Device::raw`
  answered its handles, a later `raw` call answered the new one's: `WorkCounts::vk_devices_live`
  reads 2 until then. A carried object holds nothing of it once none of its command buffers
  records. `vk_devices_made` and `device_recoveries` count the rest.
- Each recovery is announced (`device-recovered`), naming the hang's command buffer.
- `Device::raw` answers the new `VkDevice`'s handles once a later call asks; the handles it
  answered before stay valid until then, and the lost `VkDevice` stays alive until then, so the
  consumer destroys what it made with those handles (a swapchain, an import) first.
- `CommandQueue::with_vk_queue` gives its closure the `VkDevice` the lent `VkQueue` belongs to,
  read under the same lock: from the recovery on, the new one, whatever `Device::raw` answered
  before. A consumer whose command pool, fence or semaphore was made on another `VkDevice` makes
  it again on that one before it submits there.
- A device the consumer made (`Device::from_raw`) is not made again: it answers `DeviceLost` from
  the loss on.

`Device::recover_for_test` runs the same recovery with no GPU fault, for tests of what survives
one.

## Degrade notices

Where the runtime gives Metal's answer only at a cost Metal does not pay (a copy, a scratch
resource, an extra pass), or gives a recorded near-equivalent, the device announces it once per
kind: through `DeviceOptions::messages` as a `DebugMessage` whose `degradation` holds a
`DegradeNotice`, else on stderr. A notice is never fatal. `Degradation::code()` is a stable
`metal2vulkan-degrade-<kind>` string and `category()` says whether it costs time and memory or
exactness; decode a notice by its `Degradation`, never its text.
[docs/API.md section 9](../../docs/API.md#9-degrade-notices) lists every kind.

```rust,ignore
let mut options = DeviceOptions::default();
options.messages = Some(MessageSink::new(|message: &DebugMessage| {
    if let Some(notice) = &message.degradation {
        log_degradation(notice.degradation, &notice.object, &notice.reason);
    }
}));
let device = Device::system_default_with(&options)?;
```

## Line widths

Apple's private `setLineWidth:` (`RenderCommandEncoder::set_line_width`) takes every width. As
measured on an Apple M3 Max (macOS 27.0.1), Metal draws `w' = floor(16 w) / 16` clamped to
`[0, 16]` (NaN and negative widths as 0), the encoder's from 1 at its start: at `w' = 1` its
ordinary lines, at every other `w'` the parallelogram Vulkan's non-strict wide lines may be (its
minor-axis edges `w'` long through the endpoints, the ends not extended), at 0 as at 1/16. A strip
or loop draws one parallelogram per segment; triangles in fill mode `Lines` draw each edge so,
facing as the triangle does and culled by it, their flat inputs the provoking vertex's.
The runtime draws each draw at `w' != 1` that draws a primitive by one of three routes, counted
(`WorkCounts::device_wide_line_draws`, `line_band_draws`, `nearest_width_line_draws`); a direct
draw whose vertices or indices complete no primitive of its type draws nothing, as Metal's:

| Device | Route |
|---|---|
| `wideLines`, `w'` in `lineWidthRange` and above 0 | Its own lines, announced `device-wide-lines` (`WideLinePolicy::Device`, the default): no device's are taken as Metal's, a driver's parallelograms included, whose ends the driver places; under `WideLinePolicy::Exact` (`DevicePolicy::exact_answers()`) the band |
| No `wideLines` (MoltenVK), `w'` outside `lineWidthRange`, or `w' = 0` | The band |
| What neither draws | Its own lines at the width nearest Metal's they draw (clamped into `lineWidthRange` and put on the nearer step of `lineWidthGranularity`'s grid from its bottom, never 0; 1 without `wideLines`), announced `lines-at-nearest-width` |

The band is the pipeline's band variant, through a tessellation control and evaluation stage the
translator adds (docs/API.md section 4): two triangles over each line's parallelogram, and for a
triangle in fill mode `Lines` its three edges' parallelograms, which face as the triangle does under
the draw's cull mode, front face and depth bias. A strip, loop or fan is drawn as the list of its
primitives. The band's depth bias is the device's, of each band triangle's own slope and depths,
where Metal biases the edges by the triangle's: announced `wide-edge-depth-bias` under a slope
factor, or a constant factor over a float depth. Neither route draws, so the nearest width takes: a
mesh draw's lines and edges and a tessellated patch draw's edges the device's lines do not draw (1
pixel wide on MoltenVK), lines and edges on a device without `tessellationShader` whose lines do not
draw them or of a pipeline state whose band variant the translator refuses (a vertex function that
selects a slice or viewport, barycentric weights, synthesized geometry, multiview; the state's first
band asks), a draw whose own band variant the translator refuses where the state's first band
compiled (a varying or push data the draw's variant adds past the evaluation stage's outputs or
`maxPushConstantsSize`; the state's later draws of the primitive take the nearest width too), an
amplified band, and every wide line in an execution or a rasterization-rate-mapped pass, whose width
Metal's answer is not measured for. Refused `NotImplemented(RuntimeGap::WideLines)`: a strip's band
whose list of primitives the runtime cannot draw.
docs/FAST-PATHS.md records the choice and the known gaps.

## GPU families

`DeviceProperties` answers Metal's device queries from the Vulkan device.
`Device::supports_family` answers yes for a GPU family only where every row of Apple's feature set
tables for it (features, limits and pixel-format capabilities) is exact on the device;
`Device::family_blockers` names each blocking row and why: not built yet (`RowState::Gap`), not
possible on this device (a refusal), or Metal's own answer is no. Run it on the device for its
list.

`Device::supports_32bit_msaa` (`supports32BitMSAA`) answers yes where the device makes a
multisample texture of each 32-bit integer and float color format at every sample count its
textures take, 4 among them, and resolves the float ones, each planned as `Device::new_texture`
plans it within the device's own limits for the format. Vulkan guarantees an integer format only
`framebufferIntegerColorSampleCounts`, one sample on some drivers, but a driver may take more for
one format (RADV does), so the answer follows the device's per-format limits. A float resolve is
the device's own average, announced as `device-rounded-resolve` where it is not Metal's.

`DeviceProperties::supports_pull_model_interpolation` (`supportsPullModelInterpolation`) answers
yes where the device has `sampleRateShading` (and, on MoltenVK, the portability subset's
`shaderSampleRateInterpolationFunctions`): an `interpolant` input is then read at its centroid, a
sample and an offset through Vulkan's interpolation functions; without them only
`interpolate_at_center` compiles. An offset is read within the device's range and on its grid
(`device-interpolation-offset`); on RADV a perspective interpolant read at a sample or an offset
is the driver's extrapolation from the pixel center, off by the value's curvature across the
distance (`extrapolated-interpolation`); and a draw at custom sample positions through a
function that reads at a sample is refused (`RuntimeGap::InterpolateAtSampleWithPositions`).

`Device::minimum_linear_texture_alignment_for_pixel_format`
(`minimumLinearTextureAlignmentForPixelFormat:`) answers, per pixel format, the alignment of a
texture's offset and `bytesPerRow` in a buffer at which the device's images hold the texture in
place over host pages, at least the 16 bytes Metal answers on an Apple GPU. The runtime asks the
driver for every format at the first query of any by making those images (none is bound; 17 to
23 ms for all of them on the devices measured), keeps those answers for the device's life, and
writes them as one record per driver in the shader cache's content directory. The M3
Max's MoltenVK answers 16 for every format; RADV answers 256 and lavapipe 64 for each format
whose texture over a buffer they can hold in place, and 16 for the rest (the two-plane YCbCr,
packed and extended-range formats among them). A texture at a smaller
multiple of 16 is still made, as a shadow where the device's images do not take its layout
(`shadowed-texture`). The private `deviceLinearTextureAlignmentBytes` and
`linearTextureAlignmentBytes` answer the largest of the formats' answers.
`Device::linear_texture_bytes_per_row` (a runtime extension, no Metal counterpart) answers, for a
2D texture's descriptor, the row pitch the driver lays its own linear image's rows at, which a
texture over a buffer of the runtime's own memory must meet exactly to be that image, with no
check of the device's: a pitch rounded up to the alignment above can miss it, as that alignment
is where images over host pages lie.

Where Metal's answers differ by macOS release (which pixel formats it describes, how a heap
texture's blocks lie, its private device properties),
every device answers as one release, `MACOS_RELEASE` (macOS 27.0.1, measured on an Apple M3 Max),
on every host, whatever release the host runs. What a device can do, its GPU family, features and
limits, follows its GPU.

## Out of scope

What an ordinary Metal program on Linux needs belongs here; what exists only because the work comes
from a paravirtualized guest belongs to that consumer. So the runtime has none of these, and a
consumer builds them on public calls:

- IOSurface and Mach VM: use host memory without a copy, textures over buffers, and adopted
  `VkBuffer`s and `VkImage`s. Importing a file descriptor or dma-buf is plain Vulkan on the
  runtime's device.
- Scattered backing (`replaceBackingWithRanges:`, `detachBacking`): remap memory to be contiguous.
- Apple's serialized pipeline descriptors and guest-visible identifiers (`resourceIndex`,
  `uniqueIdentifier`, private device answers): keep them beside the runtime's objects.
- Presentation (`CAMetalLayer`): render into a texture over a swapchain image and present with
  plain Vulkan.
- Threads, clocks, preferences and other OS services.

Plain Metal the runtime does not build yet is refused typed where a program reaches it
(`RuntimeGap` lists every gap; [docs/METAL-COVERAGE.md](../../docs/METAL-COVERAGE.md) maps the
whole Metal API).

## Further reading

- [docs/RUNTIME-PORTING.md](../../docs/RUNTIME-PORTING.md): Metal's calls, as Objective-C or
  metal-rs makes them, mapped to the runtime's.
- [docs/API.md](../../docs/API.md): the translator's API, the degrade notices, and the accepted
  differences.
- [docs/HOWTO.md](../../docs/HOWTO.md): integration, the shader cache, failures.
- [docs/REFUSALS.md](../../docs/REFUSALS.md): every refusal of something Metal accepts.
- The rustdoc (`cargo doc -p metal2vulkan-runtime --open`): every item with its Metal counterpart.

## License

Dual-licensed, as the rest of the repository: the GNU General Public License, version 2 or (at
your option) any later version, or the GNU Lesser General Public License, version 3 or (at your
option) any later version, at your choice (`GPL-2.0-or-later OR LGPL-3.0-or-later`).
[LICENSE](LICENSE) is the notice; [LICENSE-GPL-2.0](LICENSE-GPL-2.0),
[LICENSE-LGPL-3.0](LICENSE-LGPL-3.0) and [LICENSE-GPL-3.0](LICENSE-GPL-3.0) (which the LGPL v3
incorporates) are the texts.
