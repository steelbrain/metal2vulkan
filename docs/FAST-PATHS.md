# Fast paths: the accuracy ledger

**The GPU's native arithmetic is the contract.** Where the device has a native form of an
operation (its `float` and `half` arithmetic and functions, NaN bits, minimum and maximum, rounding,
texel and colour conversions, sampler and filters, resolves), that form is the only one: no path
emulates Metal's bits. What the device lacks is still implemented or emulated, as capability.

**An accuracy note is logged, never fatal.** Each difference from Metal is recorded in the
pipeline's artifact as a `PrecisionNote` and announced as `metal2vulkan-degrade-<code>` when the
runtime makes the pipeline state; each note's rustdoc and docs/API.md section 9 state its bound.
The work keeps running; only what Metal itself refuses ends it. On MoltenVK on an Apple GPU, whose
sampler and `float` functions are Metal's own, most rows do not apply.

Each row of the first table is one place where the default trades exactness against Metal for
speed (pipeline-creation ("making") time or GPU time) outside the device's native arithmetic: the fast form is the
default, the accurate form stays behind its `DevicePolicy` switch (the runtime's
`DeviceOptions::policy`; `DevicePolicy::exact_answers()` selects every one), and both are kept. The
other tables record the device gaps, the switches whose default is Metal's, the costs of matching
Metal and the known gaps. This file is a ledger, not a journal: rows are edited in place, with no
history. A change that adds, removes or changes a fast default, its bound or its switch updates its
row in the same commit. The test `tests::fast_paths` fails when a `DevicePolicy` accuracy choice,
one of its values, or a precision note a fast default records is missing here, or when this file
names a policy value that no longer exists.

## Fast defaults

Costs are measured on the RTX 5080 Laptop GPU (NVIDIA 615.71.09) with `scripts/making-bench`:
cold, the command queue made first, the driver's disk cache off; the making of the state's
pipeline, then GPU time per draw, the default against the accurate form alone. The workloads are
three consumer pipelines: A2X (`a2x-s4`, `a2x-s96`), Xgc14 (`xgc14-bgra8`) and Xgc15
(`xgc15-rgba8`). A cell that names another source took its numbers there.

| What it is | Default (fast) | Accurate switch | Recorded bound (note, code) | Cost of the accurate form (5080) |
|---|---|---|---|---|
| A blend into a packed 16-bit attachment the device renders through its word view (`A1BGR5Unorm`, `ABGR4Unorm` on the RTX 5080; `WordAttachment`), which the function computes in the interlocked storage form, reads the attachment's codes, where Metal's tile holds the pass's clear and each earlier blend's result as a `half`, finer; under fast math it also sums a few states otherwise than Metal. Every device rendering such an attachment. | The in-shader blend over the codes (no switch) | **Not built yet:** the extended-range attachments' `half` shadow (`render/extended_shadow.rs`, `VariantKey::extended_shadows`) taken by word attachments: the runtime's shadow start and end, its load of the codes and the clear's halves, and the word's shadowed variant in `lower::fragment::fetch`, for a single-sampled pass and a multisampled one (an extended-range color's shadow holds per sample, from the pass's clear). Fast math's regrouped sums have no model | One code: without fast math exact over a loaded attachment (the 2,891 states of the M3 sweep, every channel), under fast math 3,527 of 94,732,288 channels a code apart; after a clear or an earlier blend in the pass, a code apart where the two readings store different codes. `DeviceColorConversion` (`device-color-conversion`) | Not measured. Expected from the shadow's form: an `R16G16B16A16_SFLOAT` image of the pass's extent per such color (8 bytes a pixel, 16.6 MB at 1920x1080), set at the draw it starts by a fill or a load dispatch, and an 8-byte storage read and write per blended fragment inside the interlock beside the word's |
| A heap's colour texture that Metal keeps losslessly compressed (`allowGPUOptimizedContents`, no `ShaderWrite`, `PixelFormatView` or `ShaderAtomic` usage, 16 texels or more a side), on a driver whose compressed images may keep their texels apart from the memory they are bound to (`DriverFact::CompressedImagesHideAliasedWrites`: ANV), is the driver's compressed optimal image, its compression and fast clears kept. A texture Metal keeps uncompressed takes the uncompressed image under either value. The runtime's choice: no pipeline differs. | `heap_compression = HeapCompressionPolicy::Compressed` | `heap_compression = HeapCompressionPolicy::Uncompressed`: an image the driver keeps uncompressed in the heap's bytes, as for a texture Metal keeps uncompressed: host-transfer usage where the device enabled `hostImageCopy` (`heap-texture-host-transfer`), else a linear image (`heap-texture-linear`), else the optimal one (`heap-texture-hides-aliased-writes`) | A write through a buffer or another texture over the texture's bytes may leave its texels as they were or change them otherwise than as written, where Metal's texture reads opaque magenta (`0xffff00ff`) in every texel of drawn content the write covers and the written bytes where it held only a clear; the texture's own writes may show in those bytes compressed. Within Metal's contract: Metal documents such a texture's layout as its own (`allowGPUOptimizedContents` lets it "adjust the private layout"), so no write through another view over its bytes has a defined result there either. `HeapTextureCompressed` (`heap-texture-compressed`), announced once per device where such a texture is made; `runtime_lane::heap_textures` | ANV (Iris Xe of RPL-P; `heap_textures::heap_compression_costs`), a 2048x2048 `RGBA8Unorm` render target in a `Private` placement heap: a pass that clears it 26.9 us against 676.2-676.5 us; a copy into another such texture 647.7-649.1 us against 673.9-693.2 us. The compressed image lies in the heap only where the heap's answer holds it (`heap_answers = HeapAnswerPolicy::Device`, the next row). This 2048x2048 texture's does not under either answer policy, so it takes the linear image under either compression value: a clear 963.5 us against 958.4 us under `heap_answers = HeapAnswerPolicy::Metal`, 955.3 us against 971.4 us under the default, the copy 488-496 us |
| A heap texture whose image of the route the device takes for it needs more bytes or a coarser alignment of the heap than Metal's answer (each of a guest compositor's 22 heap shapes, `RGBA16Float` 2D arrays of one slice and 2D textures to 456x255: on ANV, whose compressed images align to 64 KiB; on the RTX 3090 at 1024; on the RX 7900 XTX at 256 to 64 KiB, where Metal answers 128 for each but the 299x299 array's 16 KiB): the size query (`heapTextureSizeAndAlign`) answers the larger of Metal's answer and that image's, and the texture lies in that image in the heap's bytes. A `Shared` texture's answer also holds the linear image a heap over host pages binds for it. Where that image's alignment is past 64 KiB, or past both 4 KiB and the texture's size, the texture's image that a 4 KiB alignment holds takes its place (host-transfer usage, else linear), the answer that one's; else Metal's answer, as under the switch. Metal's measured answers (4:2:2, one-plane YCbCr, two-plane extended-range, `X32_Stencil8`, `BlockWritesOnly` textures), multisample textures and textures whose image lies in memory of its own keep Metal's answer. The runtime's choice: no pipeline differs. | `heap_answers = HeapAnswerPolicy::Device` | `heap_answers = HeapAnswerPolicy::Metal`: Metal's answer; the texture lies in an image that fits it (host-transfer usage, else linear: `heap-texture-host-transfer`, `heap-texture-linear`), else is a shadow of the heap's bytes in Metal's tiles, copied in before and out after the commands that use it (`shadowed-texture`) | A heap holds fewer such textures than Metal's: of the 22 shapes, the small arrays answer 4096 bytes where Metal answers 128 to 2176 (on ANV and the XTX; the XTX's 1x1 answers 256), and the 22 answers take 3769962 bytes of heaps placed one after another on ANV and 4155498 on the XTX, against Metal's 3656426 (`heap_texture_images`, `HEAP-COUNT`). A program that places a heap's resources by sizes of its own rather than the query's may overlap the texture with its neighbour; Metal's answer differs by GPU too (65536 on an AMD Radeon Pro 5300M, 4096 on an Intel UHD 630, 128 on an M3). `HeapTextureLargerThanMetal` (`heap-texture-larger-than-metal`), announced once per device; `tests::heap`, `heap_shadow_costs::heap_texture_images` | Of the 22 shapes made in heaps under the switch (`heap_texture_images`), on ANV (Iris Xe of RPL-P) 14 lie in an image put in place of their first one and 7 are shadows in Metal's tiles; on the RX 7900 XTX (RADV) 1 and 8, every other answer Metal's; under the default none on either. ANV (`heap_shadow_costs::heap_texture_shadow_costs`), a 1x1 `RGBA16Float` array held as a shadow: a frame that clears it and blits it 34.7 us in the heap against 8.2 us outside |
| Batches of command buffers committed on one queue that the scheduler finds ready together (a backlog) go to the driver in one `vkQueueSubmit`, at most four, only the last signalling the queue's timeline, on a device whose own check finds a submission that signals nothing merged into the next (`submit_check.rs`, once per driver, release and kernel: one `vkQueueSubmit` of four empty infos against four calls of one, merged where the one costs at most 0.6 of the four; Mesa 26.2.6's RADV on the Radeon 780M, the RX 7900 XTX and the RX 550 at 0.33-0.38, ANV on the Iris Xe of RPL-P at 0.39, MoltenVK 1.4.2 on the M3 Max at 0.98; a device whose check has not answered, or that does not merge, takes the ungrouped form under either value of the switch): one kernel submission, where ANV on i915 lists every `VkDeviceMemory` in each (the consumer's guest drag on the Iris Xe: about 300 entries per execbuffer, 260 of them its host imports, at 112 to 128 execbuffers a frame). A batch's event signals ride to the group's end, which signals each event's semaphore to the highest index the batches held signal it to (an event's indices rise in commit order). A batch that signals a binary semaphore, takes a fence, binds a sparse update, does first work, names the caller's wrapped image, or waits for anything not seen complete (an event, another queue's work) goes alone, after the batches held: a batch with a wait leads a group, never joins one. So that checked command buffers (the consumer checks each) are found ready together, the submission checks of up to three batches right behind a line's head that goes run with the head's, in commit order, none committed after another queue's batch: such a check runs before the previous batch's `vkQueueSubmit` rather than after it, in the scheduler pass that submits its own batch. A pass that finds no batch ready sends the batches held, unless two or more earlier submissions of their `VkQueue` are not seen complete: the GPU runs those first in any case, so the batches stay held, and those committed meanwhile join them, until one submission is left in flight (a completion turn wakes the scheduler at its value), the group fills, a later batch sends it, a head waits for the GPU, or a thread waits for one of their command buffers' completion, which sends them at once; nor are they kept where the scheduler's thread runs next a submission handler of another queue's command buffer: they go first, as Metal schedules a queue's command buffers whatever another queue's handler does (an M3 Max, probe `cross_queue_handler_waits`); a batch that ends a command buffer with a switch of storage forms riding on it, which settles by its submission's outcome, is never kept so. A batch a committing thread submits (`CommandBuffer::commit_and_submit`) goes alone where its `VkQueue` runs at most one earlier submission not seen complete, as the pass that held it would send the group at once; where two or more run it is left to the scheduler's thread, which holds it. A held batch that ends its command buffer is `Scheduled` as it is held, its submission handlers run first, then its scheduled handlers, and its kernel end time set, as an M3 Max schedules a command buffer committed behind running work 0.04 to 0.18 ms after its commit. The runtime's: no pipeline differs. | `completion = CompletionPolicy::Grouped` | `completion = CompletionPolicy::Exact`: one submission per batch, each signalling its own values; each line's head checked alone | Each command buffer of a submission but the last is seen complete (status, completed handlers, waits), and the event values it signals are reached (by the GPU waits on other queues, the host's waits, value reads and listeners), once the last completed: late by the GPU time of at most three later batches (about 0.1 to 0.2 ms each in the consumer's compositor frames on the Iris Xe; about 20 ms behind a 20 ms batch), never early, in commit order. Metal's contract states no bound on when completion is seen (an M3 Max reports a command buffer complete, its handlers, status, waits and event listeners alike, 0.04 to 0.10 ms after its GPU end), and no member but the first waits, so no wait holds a member back: a valid program sees only later completion, as of slower GPU work. A batch kept while earlier work runs is seen complete within the same bound, and, where the one submission left in flight ends before its group reaches the GPU, late besides by that gap: at most a completion turn's reaction and the group's `vkQueueSubmit`. Commands, barriers and GPU times are unchanged. `GroupedCompletion` (`grouped-completion`), announced once per device at its first such submission; `runtime_lane::grouped_submissions` | ANV (Iris Xe of RPL-P; `frame_costs::frame_costs_over_imports`, the checked device, 67 command buffers a frame, GPU at 467-783 MHz), with 520 host imports live, where the Exact form's submissions fall behind the commits as the guest's do: 32.0-33.2 kernel submissions a frame against 67.0, 3.4-3.5 ms of execbuffer ioctl a frame against 6.9-7.9 ms, the scheduler thread's CPU 61-63 us a command buffer against 121-136 us, a frame through completion 4.7-5.9 ms against 8.6-9.7 ms; in the guest's shape with 520 (a second queue taking every 20th command buffer, each signalling a shared event, every 10th waiting for one): 26.2-26.7 kernel submissions a frame against 67.0, 2.8 ms of ioctl a frame against 6.9-7.0 ms, the scheduler 52-53 us a command buffer against 122-123 us, a frame through completion 6.0 ms against 8.7 ms; with 260, where it keeps up, 64.9-66.3 kernel submissions a frame against 67.0, the CPU and wall times alike. The cost (`grouped_submissions::grouped_completion_delays`, 8 command buffers of 0.3-0.5 ms of GPU work ready together): the host saw each but a group's last complete later by the later ones' GPU time, the last 0.1-0.3 ms later than alone (the completion turn's per-command-buffer work for the group, done at its end). Keeping a group while earlier work runs (UHD 770 of RPL-S, ANV, the guest's shape with 520 host imports, a submission handler on every command buffer and a 16 MiB fill ahead of each, about 1.3 ms of GPU each; the kept form against the batches sent as found): committed every 1 ms, 18.1-18.2 groups kept a frame, 24.9-25.0 kernel submissions a frame against 57.9, 2.18-2.90 ms of execbuffer ioctl a frame against 4.64-5.43 ms, the scheduler thread's CPU 67-82 us a command buffer against 113-118 us, the GPU idle within a frame's span 0.61 ms against 0.68-0.69 ms, a frame through completion 89.5-89.6 ms alike, completion seen later (GPU end to completed handler: mean 1.82-1.86 ms against 0.43-0.46 ms, p50 1.57-1.60 against 0.11-0.13, p99 4.35-4.37 against 4.09-4.11, max 4.52-4.55 against 4.18-4.23); committed at once, 18.0 kernel submissions a frame against 23.4; with no fill, 18.0 against 20.6, the GPU idle 0.59 ms against 0.58-0.66 ms. In the consumer's guest drag on the Iris Xe (i9-13900H; three passes each way): 24.4-27.5 kernel submissions a frame against 42.4-43.8, the scheduler thread 0.4-1.7 ms of CPU a frame less, 15.7-17.3 frames a second against 16.4-17.6: the frame rate alike within noise, as the scheduler thread does not bound that frame. A batch a committing thread submits alone (RADV, Mesa 26.2.6, RX 7900 XTX, whose check finds submissions merged; two rounds each against the form that left every batch that may wait in a group to the scheduler's thread, `tests::completion_latency`): a checked fill's `commit_and_submit` to its submission handler p50 30.4-32.8 us, p90 32.3-35.9, p99 48.8-57.8 against 36.2-55.7, 56.4-176.3 and 112.1-198.5, the call returning after its submission (32-34 us against 4-8); committed 2 ms apart between two shapes of `commit`, its commit to submission p50 31.9-36.1 us, p90 102.6-135.7 against 39.2-50.1 and 152.1-169.7, and the `commit`s beside it alike (p50 32.1-47.3 us against 38.9-46.7, the scheduler thread's wake p50 4.0-4.5 us against 4.1), as the committing thread wakes that thread after a batch it submitted alone, as the batch's hand-off to it would have. On lavapipe, whose check finds no merge, the two forms alike, the scheduler's thread woken alike (a checked fill's p50 12.5-13.4 us against 12.4-13.4). |
| Batches of command buffers on one `VkQueue`, on a driver measured to start no kernel submission of a queue before the one before it completed (ANV, RADV and NVIDIA's; `batch_order.rs`), are ordered by their ends: each ends with a barrier ordering its work before every later command of its `VkQueue` (the exact form's end orders it before the host's reads and writes alone), so a batch that depends on an earlier one there begins with no barrier of its own, often no command buffer of barriers, and its first encoder's GPU start is written at the bottom of the pipe, once everything before it completed. Metal starts an independent command buffer committed behind running work while that work runs (an M3 Max under macOS 27.0.1, probe `gpustart`); those drivers ran two batches of independent work one after the other under every shape measured (no end barrier, the host's or the widened one; in one `vkQueueSubmit` or two, merged into one kernel submission or not), where two dispatches of one command buffer ran side by side. The runtime's: no pipeline differs. | `batch_order = BatchOrderPolicy::Ends` | `batch_order = BatchOrderPolicy::Exact`: a barrier for each dependency of a batch's work on an earlier batch's of its `VkQueue`, the end's barrier to the host's reads and writes alone, the GPU start written at the top of the pipe; every other driver takes it under either value | None: on those drivers each batch starts once the one before it completed under either value, so the work's order, its results and its GPU times are the same, and a batch's GPU start is the end of the work before it, when such a driver starts it. Not announced; `runtime_lane::commit_work` and `runtime_lane::gpu_times` test both values | UHD 770 (ANV, RPL-S), the consumer's frame shape over 520 host imports, 67 command buffers a frame, Grouped completion, the widened end barrier against the exact form (a probe build of each): with no fill ahead of each command buffer, 67.0 command buffers submitted a frame against 129.3-129.7, the GPU's span a frame 1.84-1.90 ms against 1.99-2.06 ms, a frame through completion 2.13-2.22 ms against 2.27-2.38 ms; with a 1 MiB fill ahead of each, 134.0 against 262.9, 8.03-8.22 ms against 8.32-8.42 ms, 9.00-9.15 ms against 9.25-9.36 ms. Radeon 780M (RADV, Mesa 26.0.8), this switch: with no fill, 67.0 against 128.9-129.0, a frame through completion 48.3-48.6 ms against 42.6-47.0 ms; with a 16 MiB fill committed at once, 134.0 against 251.0-251.1, 50.6-50.8 ms against 50.8-57.3 ms; committed every 1 ms, 134.0 against 249.7-250.2, 71.3-83.6 ms against 106.3-113.6 ms (each kernel submission there leaves about 2.5 ms of the GPU idle over 520 host imports, which sets these frames) |
| Command queues on a device whose GPU idles when consecutive work comes from another `VkQueue` share `VkQueue`s by choice (`queue_share.rs`): the runtime's own self-check, not a `DriverFact` row, measures that idling once per driver, release and runtime on a thread of its own and keeps its verdict (`Idles`, `DoesNotIdle`, `Unchecked`) in the shader cache's directory, as the in-place check keeps its own (`queue_switch_check.rs`: 48.7 to 51.3 us a switch on the RTX 5080 Laptop GPU, NVIDIA 615.71.09, `Idles` above 20 us). A command queue that makes a command buffer while it holds no other uncompleted one moves onto the busy `VkQueue` of its family whose oldest batch has run under 0.5 ms, the most batches in flight first, so short command buffers of several queues run with no switch between them. A queue one of whose command buffers ran on the GPU for over 2 ms, by its own timestamps (not the host's view of how long it ran as its `VkQueue`'s oldest, which also holds its wait for the GPU to start it and the waiter's lateness to see it complete), or held its `VkQueue` for over 2 ms from the submission of a batch that waited on the GPU for another `VkQueue`'s work (an `MTLEvent`'s signal, the tracker's wait, new resources' first work) to its own GPU end, is long: no queue joins its `VkQueue` while it lives, and it moves once more, at its next idle point, to an idle `VkQueue` no long or unshared queue holds. A queue whose command buffers must never wait behind another queue's (a compositor's beside an application's) is made with `Device::new_command_queue_unshared`: it never moves and no queue joins its `VkQueue`. Under either value, as a queue is made, it takes the round robin's `VkQueue` the fewest live command queues hold, of those no unshared queue holds, in turn among equals, and an unshared queue one no command queue holds where one is free (one pass over the `VkQueue`s' counts; one that finds none shares the one the fewest hold, `WorkCounts::unshared_queues_sharing_a_vk_queue`). Presenting queues and a queue that lent its `VkQueue` (`with_vk_queue`) never move. Before the verdict, and where it is not `Idles` or there are no two `VkQueue`s of one family (MoltenVK, RADV, lavapipe), every queue keeps the `VkQueue` it took at its creation; so does every queue of a device whose command buffers write no GPU timestamps it reads (`gpu_times_from_host`, or a driver whose timestamps wait for every queue's), which does not share and is counted (`WorkCounts::queue_sharing_untimed`). The runtime's: no pipeline differs. | `queue_sharing = QueueSharingPolicy::Adaptive` | `queue_sharing = QueueSharingPolicy::PerQueue`: each command queue keeps the `VkQueue` it took at its creation (the fewest live queues' as above), and the check does not run | A command buffer waits behind the command buffers of the queues sharing its `VkQueue`, where Metal's queues wait for each other only inside one command. The residual stall: once per logical queue that goes long, for a queue that joined its `VkQueue` during its short phase, about one long command buffer's length: the command buffers committed behind that queue's first long one there wait for its whole length (and those behind its later ones while it holds work, until its next idle point). Measured on the RTX 5080 Laptop GPU beside an application that alternates draws with 30 ms command buffers: a shared compositor's commits at 4,800 a second had 64 completions over 10 ms, at most 26.4 to 30.1 ms, in each of 3 runs of each of two such shapes (one join each), against none over 10 ms under `PerQueue` in those 6 runs (at most 5.6 to 9.7 ms); with 2 ms of draws between long command buffers, none over 10 ms in 3 runs (at most 5.7 to 9.8 ms) against 7, 0 and 0 under `PerQueue` (at most 6.9 to 11.2 ms). On an unshared queue, 0 over 10 ms in every run. A command buffer's GPU start time includes the other queues' work before it. `QueueSharing` (`queue-sharing`), at the device's first join | Command buffers a second on the RTX 5080 Laptop GPU, one-draw command buffers, each thread its own queue, the default against `PerQueue`, runs after a lock hold's first: 8 threads with 3 in flight 50.7k to 54.7k against 27.4k to 28.0k (5 runs each); 8 threads with 1 in flight 47.0k to 50.1k against 13.9k to 14.3k (5); 2 threads with 3 in flight 28.0k to 53.7k against 21.4k to 28.0k (10 each, in turn, the default above `PerQueue` in each pair; no thread's queue marked long), and beside another process's 30 ms command buffers 5.1k to 6.7k against 4.0k to 4.1k (3 each; the host's view marked a thread's queue long in 1 of 3 runs, which then ran at `PerQueue`'s 4.1k); 2 threads with 1 in flight 16.9k to 20.7k against 10.1k to 11.5k (5); one thread over 16 queues in turn 21.7k to 43.3k against 10.2k to 11.1k, over 2,048 20.0k to 39.2k against 10.2k to 11.2k (4). Beside an application whose queue never shared with the compositor (isolation, 300 and 30 dispatches), both 0 over 10 ms |
| A chain of `half` additions, subtractions and products on a device without `shaderFloat16` (vk12-floor, SwiftShader, Pascal; capability emulation: the module computes each in `float` and rounds it by the device's `OpQuantizeToF16`): an operation whose flags hold `contract` and `reassoc` and whose result only such operations read is held at `float` precision unrounded, so the chain rounds where a value leaves it (a store, an output, a conversion, a comparison, a call, a phi, a vector move, an operation without both flags), one quantize per exit instead of one per operation (`lower::arith::half_chain`). A device with `shaderFloat16` computes `half` alike under either value. | `half_chains = HalfChainPolicy::AtExits` | `half_chains = HalfChainPolicy::EachOperation`: every operation's result rounded to `half`, as the AIR computes it | The result may differ from the rounded operations' by the roundings the chain skips, each within half a unit in the last place of a `half` at its step, compounded along the chain; a value inside the chain past `half`'s range (65504) stays finite where the rounded operations reach infinity. Within the AIR's flags, which let a compiler regroup and fuse such operations. `FastMathRewrites` (`fast-math-rewrites`) | Not timed. Size on vk12-floor (the fixtures' SPIR-V words, the default against the accurate form): `hfb_filter_bank` 121,270 against 131,766 (1,024 quantizes against 2,464), `grd_half_many_samples` 82,019 against 84,627, `nan_half_fast` 9,808 against 9,832 |
| A texture that samples, reads, renders into it or is written through a shader (`ShaderWrite` or `ShaderAtomic` usage, of texels of 1, 2, 4, 8 or 16 bytes) over host pages (a no-copy buffer's, or a no-copy heap's buffer's, imported by their pointer or as dma-bufs; or a buffer in fresh pages; or a copied host buffer's copy of them, `copied.rs`, its image bound in the copy's memory where the check passed for the pages, and for the copy's fresh pages or, for a `DRM_FORMAT_MOD_LINEAR` image in memory the device allocates, its pooled memory, `WorkCounts::copied_textures_in_place`) is a linear image aliasing the pages in place, where the device makes the image (its format and usage linear, at the pages' pitch) and its own check shows that the operations Vulkan lets read and write every byte bound to it leave every byte alone while the CPU writes beside it: the image's first move (`VUID-VkImageCreateInfo-pNext-01443` makes such an image start `UNDEFINED`), and for one that renders a render pass's load and store operations and its moves into `ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT` and back, and for one a shader writes a kernel's storage writes of its texels (which of a linear image a driver may widen past a texel), each access checked apart (a shader's writes of 1- or 2-byte texels apart from those of wider ones, which a driver may store through a wider access), and each apart for texels that share a line of the device's memory (`nonCoherentAtomSize`, "the size and alignment in bytes that bounds concurrent access to host-mapped device memory") with bytes beside them and for texels that share none (their start, their end and, with bytes between rows, each row's start and end on a line; `in_place::RowEnds`): a byte changed beside texels that share none fails both, one changed beside texels that share one fails those alone (the Iris Xe, ANV, Mesa 26.2.6, whose clear and store, and draws, write back whole the 64-byte line that holds a row's last texels, fails renders, and writes, for rows that end inside a line, a byte between rows changed, and passes both for rows that end on one: a 3840x2160 `BGRA8Unorm` surface's 15360-byte rows are 240 lines each). A kernel's writes of a texture over a buffer's rows (no IOSurface plane's) whose rows, each to its pitch, start and end on a line, its buffer holding the last row to its pitch, take a verdict of their own from the same rounds, which a round that changed only bytes of its rows' padding (after a row's last texel, before its pitch, the last row's included) passes (`in_place::Padding`, `Access::BufferWrites`): Metal leaves those bytes to the GPU there (an Intel UHD Graphics 630 under macOS 26.7.1 wrote older values back over the CPU's writes to them under a kernel's writes of a texture over a buffer, the last row's included, where an M3 Max under macOS 27.0.1 and a Radeon Pro 5300M kept them, and all three kept them under draws, clears and stores and over IOSurfaces; `crates/metal2vulkan-conformance/probes/row_padding_writes.m`), and the rule follows the x86 Macs' envelope (docs/API.md, section 10, **Kernels**), so the Iris Xe's written textures over a buffer take the image in place whatever their rows' ends. A deliberate, scoped exception to relying on documented Vulkan behaviour alone, the owner's rulings (2026-10-26, and 2026-10-27 and 2026-11-07 for render targets and written textures: the GPU works on the caller's pages in place as a Mac's does, whatever a linear image costs it, on a discrete GPU too; `in_place.rs`). A texture over a buffer of the runtime's pooled memory whose rows the device's own linear image lays out otherwise (a padded pitch the driver does not take: ANV lays linear rows tight; rows apart only over a buffer that declares them unobserved, `BufferDescriptor::padding_unobserved`) is, by the same rule, a `DRM_FORMAT_MOD_LINEAR` image at Metal's pitch bound in the buffer's memory, at any size, where the device makes one (no external memory, which Vulkan allows: `VUID-VkImageCreateInfo-tiling-02261`, `VUID-vkBindImageMemory-image-01445`), made `UNDEFINED` (`VUID-VkImageCreateInfo-initialLayout-12478`), whose first move the check's pooled class judges (`PagesClass::Pooled`: a buffer of that memory, its images at Metal's pitch alone; `WorkCounts::pooled_pitched_images`); a texture made before that class's verdict is a shadow that moves to that image once the verdict passes, as one over host pages moves in place (below; `WorkCounts::pooled_textures_moved_in_place`, `pooled_texture_shadows_freed_after_move`). A texture over a buffer whose memory the runtime exports (`exported_memory.rs`: a block that exportable buffers of one handle type share) is, by the same rule, the device's linear image or that `DRM_FORMAT_MOD_LINEAR` image made for the memory's handle type, which an image bound there must name (`VUID-vkBindImageMemory-memory-02728`), and so made `UNDEFINED` (`VUID-VkImageCreateInfo-pNext-01443`), whose first move the check's exported classes judge, one per handle type (`PagesClass::ExportedFd`, `PagesClass::ExportedDmaBuf`: an exportable buffer of that handle type, its textures taking the product's image), a texture made before their verdicts moving to it once they pass as over the pooled memory; a shadow, its notice naming why, where the memory is a buffer's dedicated allocation (`VUID-vkBindImageMemory-memory-10926`) or the device's image format query (`VkPhysicalDeviceExternalImageFormatInfo`) takes the handle type for no such image without a dedicated allocation. A texture over a buffer in a block of the runtime's buffers allocated exportable as an opaque file descriptor and never exported (`memory.rs`'s "Buffers in exportable blocks") is the device's linear image made for that handle type, made `UNDEFINED`, and never a `DRM_FORMAT_MOD_LINEAR` one (RADV asks a dedicated allocation of one made for any handle type), whose first move and accesses the check's class of such blocks judges (`PagesClass::FdBlocks`: a buffer in such a block, its textures taking the product's images, made only where such a block holds every image a texture over a plain block's buffer that keeps its padding takes in place, `texture/fd_blocks_cover.rs`); the device's buffers lie there only where those verdicts give every texture over one what it takes over a plain block's buffer (`in_place::fd_blocks_take`), in plain blocks until they are known, and a buffer that declares its padding unobserved lies in a plain block wherever, so that a texture over it whose rows lie apart keeps its `DRM_FORMAT_MOD_LINEAR` image. A texture that renders over a buffer that declares its padding unobserved, its rows each to its pitch sharing no line with bytes past them, takes the verdict of renders that lets a byte of its rows' padding change (`Access::UnobservedRenders`, `Padding::Unobserved`), which the renders' own rounds give as the writes' give theirs: the declaration lets a pass's load and store write those bytes (the Iris Xe's clear and store write back the 64-byte line that holds a row's last texels whole), and a byte past the last row's pitch, or a texel, fails it. The check runs once per driver and driver version (its record named by the driver and the check's own version, not the runtime release), per class of pages, access and row ends, on a thread of its own, the verdicts it measured kept in the shader cache's directory with those it could not check for a cause that lasts for the driver (the device lacks an extension, a feature, an import, a format or an image of the shape), while one it could not check for a cause that may pass (pages or memory not made, an import refused, its writer thread not started, a move or the stores that failed) is checked again by the next device, its reason noted; a texture made before its verdict is a shadow, made without waiting for the check (its rounds take seconds on some devices and builds; `host_texture_shadows_check_pending`), and moves in place once its class's verdicts pass, at the commit of the device's only uncommitted command buffer, with no copy (`host_textures_moved_in_place`), its shadow's image freed once the command buffer that took the move completed (`host_texture_shadows_freed_after_move`), or, where a view made before the move presents it, a shadow made over it holds it as its truth, or the texture is untracked, once those and the texture went. A texture a shader writes of texels of a width no check stores (none of 1, 2, 4, 8 and 16 bytes), a check that failed or made no image, and a format, usage or pitch no image takes keep the shadow (`shadowed-texture`, its reason named, `WorkCounts::host_texture_shadows_*`, those that render and those a shader writes counted apart in `host_texture_shadows_render_target` and `host_texture_shadows_shader_write`). On a discrete GPU a `Shared` texture and an IOSurface's are in place too, read across the bus, never copied to the device's memory for speed; a `Managed` no-copy buffer there is two copies (`managed_copy.rs`, the **Costs of matching Metal** bullet), and a texture over its bytes is over its GPU copy, no texture over host pages; a `Managed` texture over the bytes of a no-copy buffer that lies in its pages there (its GPU copy found no room, or Metal's private option bit 24 drops its GPU writes) that only samples or reads, over memory that carries a log of its writes, is a device copy instead (`WorkCounts::device_copy_textures`), the GPU's copy of the two Metal keeps of a `Managed` resource on a Mac with a discrete GPU, its copies in only the rows the log reports changed. | `host_page_textures = HostPageTexturePolicy::InPlace` | `host_page_textures = HostPageTexturePolicy::Shadow`: every texture over host pages a shadow, its texels copied in before each command buffer that uses it and merged back after each write (the host-pages bullet under **Costs of matching Metal**) | Exact where the driver keeps what its check showed: the check covers texels, the bytes between rows and those after the image, of 1, 2, 4, 8 and 16-byte texels and a 3840x2160 surface, three rounds each, for an image that reads and one that renders (its pass a clear at load and a store, then a draw over its last column in a pass that loads and stores it; a resolve into it is not among the rounds), the CPU writing beside the texels sweep after sweep until each step completed, so that a device that writes back bytes it read beside them, wherever in the step, puts back an older value the check finds, and, for one a shader writes, of 4, 8 and 16-byte texels, rows apart (of widths no multiple of 4 and odd heights among them) or tight, and 1920x1080, and apart of 1- and 2-byte texels, rows apart at a 256-byte pitch (of rows that end inside a 4-byte word and odd heights) or tight, and 1024x64, each texel stored by a kernel through an unsigned-integer view (a program's atomics, and its stores through a float view, are not among the rounds); a driver whose operations changed a byte outside every round's window would undo a CPU write made meanwhile. A kernel's writes of a texture over a buffer whose padding is free may change that padding, as an x86 Mac's GPU's do (above). Not announced where the check passed. A CPU write reaches the image's reads as it reaches the buffer's (Host Write Ordering Guarantees) | See **Costs of matching Metal** (host pages): the shadow copies a sampled 3840x2160 surface in whole each command buffer, and a render target's shadow copies it in, keeps it and merges it back at each render pass instance |
| A render or compute state's creation hands the driver's making of its predicted pipeline (`pipeline/predicted.rs`, `pipeline/dispatch_making.rs`) to the runtime's pool, kept by the state as a draw's or dispatch's making is, and waits for it `pipeline_creation_wait` at most; a committed command buffer's recordings wait until its commit plus `pipeline_commit_wait` at most, one deadline its commands share, for its draws' and dispatches' (a compute encoder's, a render pass's tile dispatches) pipelines still being made (the scheduler wakes at the bound, `StepWaits::until`), an execution's replayed draws (`render/pending.rs`) among them, whose pipelines the commit's settle hands to the pool (a device replay's, one recording for every command, waits until made under either value), then record each such command as nothing: a draw's barriers and visibility query, a dispatch's barrier (a tile dispatch's copies into its scratch and back), no draw or dispatch, the command buffer running on without error. The making goes on, and the command buffers after it draw and dispatch with the pipeline once made. Metal compiles inside `newRenderPipelineState`, so no draw of a made state waits or is dropped; a host that creates a state inside a command buffer's encoding (the consumer creates a guest's state at an execution's first lookup of it) held that execution through the driver's whole compile: 11.59 s inside one execution on an M2 Pro (MoltenVK 1.4.2, a cold Metal compiler service; a sample of such a stall finds the compile's thread waiting in `MVKMetalCompiler::compile` for Metal's compiler), past the guest's 10 s FIFO check, which ended the guest's task. Each bound is 4.5 s: a creation and a commit may fall in one step of the guest's FIFO (a task's jobs run one after another, and a draw's own variant waits under its command buffer's bound), and no runtime call waits for both, so one step waits 9 s at most for makings, under that 10 s check. A creation waits on no command buffer's deadline and a `commit` on no creation's; a `commit` waits until its own deadline at most (it records there what it can, an execution of render commands the CPU writes among them), every other recording on the scheduler's thread, a host step's on one of the steps' threads. Made until made under either value, in the call that makes them: a draw that records nothing, one in a pass of custom sample positions and one recorded again into another instance of its pass (each in the draw's call or an execution's replay), a device replay's draws (in its `commit`), and a failed making made again (`PipelineWaitPolicy`'s docs). The runtime's: no pipeline differs. | `pipeline_waits = PipelineWaitPolicy::Bounded`, `pipeline_creation_wait = 4.5 s`, `pipeline_commit_wait = 4.5 s` | `pipeline_waits = PipelineWaitPolicy::Exact`: the creation makes its predicted pipeline before it returns (a render state's first key on its own thread, the rest on the pool, joined; a compute state's on its own thread), an execution's settle makes its replayed draws' pipelines at the commit, and a command buffer waits for each draw's and dispatch's pipeline until it is made | A draw whose pipeline was still being made `pipeline_commit_wait` after its command buffer's commit renders nothing in that command buffer (only there: later ones draw it), where Metal's renders: content drawn once (a layer, a glyph cache) stays without it until redrawn. A creation returns after `pipeline_creation_wait` at most with its making in flight (`WorkCounts::creations_left_making`). `DrawDroppedStillMaking` (`draw-dropped-still-making`), at each such draw where `DeviceOptions::announce_each` asks, else once per device; `CommandBuffer::dropped_draw` names the first with `VK_TIMEOUT`; `WorkCounts::draws_dropped_still_making`; `deferred_calls::a_draw_whose_pipeline_is_still_being_made_at_its_bound_runs_nothing` and `under_the_exact_wait_a_held_making_holds_its_command_buffer_until_it_lands`. A dispatch whose pipeline was still being made then writes nothing in that command buffer, where Metal's runs: `DispatchDroppedStillMaking` (`dispatch-dropped-still-making`), `WorkCounts::dispatches_dropped_still_making`, the first named by `dropped_draw`; `deferred_calls::a_dispatch_whose_pipeline_is_still_being_made_at_its_bound_runs_nothing`, `a_compute_state_s_creation_waits_its_bound_for_its_predicted_making` and `a_replayed_draw_whose_pipeline_is_still_being_made_at_its_bound_runs_nothing` | No GPU cost. The exact form holds the creating thread, and the command buffer, for the whole compile: up to 11.59 s on the M2 Pro's cold boot (that boot's 2,483 render state creations, translation included: 300 over 16 ms, 151 over 100 ms, 14 over 250 ms, 3 over 1 s, the longest 11.59 s, 5.82 s and 1.17 s), where the default holds a creation for 4.5 s at most and a command buffer for 4.5 s from its commit, which drops the draws of the compiles that outlast those (2 of that boot's 2,483 were over 4.5 s) for the command buffers committed before they land; bounds of 1 s dropped 39 draws in a cold boot on the M2 Pro, and the guest's menu bar, drawn once, stayed half drawn |
| A line Apple's `setLineWidth:` widens (`w' = floor(16 w) / 16`, other than 1), and a triangle's edge in fill mode `Lines`, is drawn by the device's own lines at `w'` wherever it draws that width (`wideLines`, `w'` in `lineWidthRange` and above 0), where Metal draws the parallelogram Vulkan's non-strict wide lines may be (`render/line_width.rs`). No device's are taken as it, announced on every device: a strict device's rectangle with square ends (lavapipe, NVIDIA), lines Vulkan does not state as parallelograms (ANV, and every width below 1), a width rounded to the granularity, and a driver's parallelograms (`strictLines` unset, `nonStrictWideLinesUseParallelogram` set), whose ends the driver places (RADV on the Radeon RX 7900 XTX draws 21 of `runtime_lane::line_widths`'s 48 scattered lines at width 2.5 a row or a column off Metal's at an end). A mesh draw's lines and edges and a tessellated patch draw's edges take the device's lines under either value, the band not drawing them (at the nearest width they draw where they do not draw `w'`, announced `lines-at-nearest-width`: a known gap below). The runtime's: no pipeline's code differs. | `wide_lines = WideLinePolicy::Device` | `wide_lines = WideLinePolicy::Exact`: a draw of lines or edges the device's lines draw is the pipeline's band variant (`VariantKey::line_band`, `pipeline::link::band`) on a device with `tessellationShader` (the device's lines where it has none, or where the translator refuses the state's band or the draw's): two triangles over each line's parallelogram, and each triangle's three edges' parallelograms facing as the triangle, through a tessellation control and evaluation stage; every other width other than 1 is the band under either value | The lines can cover pixels the parallelogram does not, and miss others, at each end and along each edge: up to half the width past each end (square ends), a rounded width, and a row or a column at an end (a driver's parallelograms). `DeviceWideLines` (`device-wide-lines`), at each such draw where `DeviceOptions::announce_each` asks, else once per device; `WorkCounts::device_wide_line_draws`, `line_band_draws`, `nearest_width_line_draws`; `runtime_lane::line_widths` (the band matches Metal's maps of lines and of triangles' edges, their facing, culling, flat inputs, primitive IDs and depth on MoltenVK under both values, which has no `wideLines`; a band of edges under a depth bias is announced `wide-edge-depth-bias`) | Not measured. Expected from the band's form: a second pipeline variant made for each render state that draws such lines or edges (its two synthesized stages), each line tessellated as a patch of two control points into four corners and each triangle as a patch of three into 21 triangles (12 degenerate), and a strip, loop or fan drawn as the list of its primitives |
| A blit or fill through a `BGRA10_XR` or `BGRA10_XR_sRGB` texture Metal keeps losslessly compressed (`allowGPUOptimizedContents`, 16 texels or more a side, neither `ShaderWrite` nor `PixelFormatView` usage, one sample, not over a buffer) is the device's copy, which keeps the bytes it moves, on every device: Metal's GPU blit into, out of or between such textures moves the texels through its compression, which writes each 16-bit field's low six bits cleared and the alpha code clamped to `[384, 894]`, and its fill clears the low six bits alone (an Apple M3 Max under macOS 27.0.1, every direction, storage mode, usage and size that decides the compression; CPU copies keep the bytes on Metal too). A pass's draws through such an attachment are the device-gap row below. The runtime's choice: no pipeline differs (`compressed_blit.rs`). | `extended_range_blits = ExtendedRangeBlitPolicy::Kept` | `extended_range_blits = ExtendedRangeBlitPolicy::Canonical`: after a blit writes such a texture's region, or a blit out of one writes a buffer's words, a texture over a buffer's or another texture's, a compute dispatch (`bind::extended_canonical_module`, `extended_canonical.rs`) rewrites the texels it wrote in Metal's encoding: in place in a buffer's bytes, an image's through a scratch buffer, a copy out and back around the dispatch; a fill's texel is encoded before it is written. A buffer whose words the device cannot bind (an offset or pitch off 4 bytes, a region past `maxStorageBufferRange`) and a texture over a buffer whose rows share bytes keep the bytes under either value, announced | Bytes no Metal write left (uploaded, CPU-copied or kernel-written texels: a field's low six bits, an alpha code outside `[384, 894]`) keep what Metal's blit would rewrite; a texel reads the same value either way, its low six bits unread and its alpha read clamped. `ExtendedRangeBitsKept` (`extended-range-bits-kept`), once per device; under `Canonical` not for the blits it makes exact; `runtime_lane::compressed_blits` (both values on MoltenVK), the conformance cases `compressed-blits-bgra10xr` and `compressed-blits-bgra10xr-srgb` | Measured, `Canonical` against `Kept`, on `Private` `BGRA10_XR` textures Metal keeps compressed (the sRGB twin and fills not measured), 256x256 and 2048x2048, one whole-texture blit a command buffer: p50 of 300 blits a run, the two values interleaved in one process, three runs a host. RTX 5080 Laptop (NVIDIA 615.71.09, the command buffer's GPU time): a buffer into the texture 23.5 against 11.5 us and 761 against 244 us from a `Private` buffer, 42 against 30 us and 1,945 against 1,176 us from a `Shared` one; the texture into a `Private` buffer 14.1 against 11.2 us and 287 against 246 us, into a `Shared` one, whose words the dispatch rewrites in system memory, 66 against 27 us and 3,386 against 1,182 us; texture into texture 22.5 against 9.7 us and 630 against 80 us. Apple M3 Max (MoltenVK 1.4.2, whose GPU times are the host's, so per blit of 16 in one command buffer), the same blits: into the texture +38 to +57 us and +536 to +548 us, out of it into a buffer +15 us and +84 to +94 us, texture into texture +46 to +48 us and +535 to +541 us. Every figure is above its `Kept` one in every run, by more than the runs' spread (under 1 us on the 5080): measurable at both sizes on both hosts. Into an image a blit adds a copy out into a scratch buffer of the region's size, the dispatch, a copy back and the barriers between them; into a buffer the dispatch and a barrier. The scratch is made without zeros, since the copy out writes every byte the dispatch and the copy back read; the figures above were taken while it was zeroed first, by a command buffer of its own (a fill on the 5080, a copy on MoltenVK). CPU from the command buffer's making through its commit, a blit alone: +1 to +3 us on the M3 Max, +1 to +127 us on the 5080, whose host other builds loaded; 16 blits into 2048x2048 images in one command buffer hold 16 scratch buffers of 32 MiB, and encoding each costs 141 to 165 us (5080) and 286 to 356 us (M3 Max), new memory blocks among it, against 2 to 6 us under `Kept`. Nothing for a texture Metal keeps uncompressed or of another format |
| A colour attachment of a texture's own image (not over a buffer's bytes, not memoryless, resolving nothing) that a pass stores `DontCare` is stored as the driver stores it: `DONT_CARE` on a driver measured to keep one of Metal's two answers or to store as Metal's own (`DriverFact::DontCareStoreIsOneOfMetals`, `DontCareStoreIsMetals`), else `STORE`, the rendered texels (`render_target.rs`). Metal's documentation leaves the contents open (`MTLStoreAction.dontCare`: "You need to assume that GPU discarded the texture's contents"); Apple's GPU leaves the texels the attachment held before the pass, the pass's clear and draws both gone (the conformance case `pass-sequence-color-actions`, an M3 Max under macOS 27.0.1, eight runs alike), or the rendered ones where its tiler overflows. A pass MoltenVK runs as one instance is handed to Metal's own `DontCare` under either value. | `dont_care_colors = DontCareColorPolicy::Driver` | `dont_care_colors = DontCareColorPolicy::Kept`: the pass renders through a backing its command queue keeps, one image of the attachment's shape, into which a `Load` copies the texels first, and nothing is copied back, so the texture keeps its earlier texels exactly (`kept_bytes.rs`, `WorkCounts::kept_texels_passes`); where a format makes no such image, the pass stores as under `Driver`, announced | Under `Driver`, the attachment holds the rendered texels (`STORE`: NVIDIA, lavapipe) or what the driver's `DONT_CARE` leaves (RADV, ANV: measured to be one of Metal's two answers), where Metal's holds the earlier texels unless its tiler overflows; where the pass also loads it `DontCare` and the runtime clears it (`unstored_dont_care_loads`, below), a tile no draw touches holds zero or the clear value. `DontCareStore` (`dont-care-store`), once per device; under `Kept` not for the passes it keeps; `runtime_lane::dont_care_stores` (both values), the conformance case `pass-sequence-color-actions` | Measured, `Kept` against the same pass stored `Store`, one pass of one draw over an `RGBA8Unorm` texture a command buffer, commit to completion, the median of 300 interleaved (a scratch measurement, `rows3_cost`): RTX 5080 Laptop (NVIDIA 615.71.09), a `Load` pass 19 µs more at 1920x1080 and 10 µs at 256x256 (16 µs and 7 µs of GPU time), a `Clear` pass 3 µs; lavapipe (llvmpipe, LLVM 23.1.4), a `Load` pass 0.5 to 0.8 ms more at 1920x1080 and 30 µs at 256x256, a `Clear` pass within the runs' spread. Memory: one backing image of the attachment's shape per command queue |
| An attachment (not memoryless) that a pass loads `DontCare` and does not store, in a pass whose draws reach a tile, is cleared over the render area (`clear_reach.rs`) on a driver measured to run that clear no slower than its `DONT_CARE` load (`DriverFact::UnstoredClearNoSlower`: MoltenVK on the M3 Max, RADV on the Radeon 780M), and left `DONT_CARE` on every other driver; an aspect of separate depth and stencil textures likewise, each aspect apart, in the scratch image of the combined format the pass renders them through (`depth_stencil_scratch.rs`). Metal's draws read zero (a colour) or the pass's clear value (depth, stencil) of such an attachment wherever they read it, never its earlier texels (an M3 Max under macOS 27.0.1: an `RG16Float` second colour a framebuffer fetch reads and a `Stencil8` a stencil test reads, in passes split and whole; a separate `Depth32Float` and `Stencil8` read by a depth and a stencil test, each aspect so loaded alone or both; `runtime_lane::dont_care_unstored`); Vulkan's `DONT_CARE` load leaves the contents undefined. | `unstored_dont_care_loads = UnstoredDontCarePolicy::Driver` | `unstored_dont_care_loads = UnstoredDontCarePolicy::Cleared`: cleared on every device, by the pass's first render pass instance's load operation, or before the pass where that instance leaves the attachment out (separate depth and stencil textures: by the scratch image's load operation of the aspect); Metal's answer wherever a draw reads it | On a driver without the fact, the draws read what its `DONT_CARE` load leaves (lavapipe: the attachment's earlier texels). `DontCareUnstoredLoad` (`dont-care-unstored-load`), once per device; `runtime_lane::dont_care_unstored` (both values: Metal's answer under `Cleared` and on a driver with the fact, lavapipe's earlier texels under `Driver`; separate depth and stencil textures alike, but what lavapipe's `DONT_CARE` leaves in the scratch image is printed, not checked) | Measured, `CLEAR` against `DONT_CARE`, the GPU's span per pass, medians of 15 frames in each of two rounds, eight 3648x2038 passes a frame (a `BGRA8Unorm` colour loaded and stored, an `RG16Float` colour and a `Depth32Float_Stencil8` loaded by the form and stored `DontCare`, one draw over the whole target writing all three; a probe whose runtime still left such a load `DONT_CARE`, without the validation layer): lavapipe (Mesa 26.2.6, llvmpipe on LLVM 23.1.4) 10480 and 10744 µs against 7280 and 7259, 3.2 to 3.5 ms more, its clear a write of every texel; ANV (Mesa 26.2.6) on the Iris Xe (RPL-P) 1936 and 1936 µs against 1907 and 1151, slower in both rounds; RADV (Mesa 26.2.6) on the Radeon 780M 861 and 865 µs against 1147 and 988, faster; MoltenVK 1.4.2 on the M3 Max 218 and 218 µs against 224 and 217, apart by less than one form's two rounds; NVIDIA's 615.71.09 on the RTX 5080 Laptop GPU 74.4 and 74.6 µs against 71.2 and 71.4, 3.2 µs more. Memory: none |
| A linear sample of a two-plane YCbCr texture whose planes a function reads in place (`YCBCR8_420_2P`, `YCBCR10_420_2P` through a uniform texel buffer over their codes, on an integrated GPU or a CPU device; `lower::sampler::in_place`), through a sampler that addresses no border on either plane axis, with no texel offset placed by a parent's size, is Vulkan's Y′CbCr model of the codes, as the device's own conversion with a linear chroma filter computes it: the four luma codes around the point and the four chroma pairs around the same point of the chroma plane (Metal's midpoint siting) blended bilinearly, then the surface's range and matrix (Metal's coefficients) in `float`, each channel clamped into `[0, 1]`: eight fetches. A runtime sampler's word decides as the function runs. A nearest sample, a border mode, a parent-placed offset and a gather are Metal's under either value. | `ycbcr_in_place_reads = YCbCrInPlacePolicy::Model` | `ycbcr_in_place_reads = YCbCrInPlacePolicy::Exact`: every tap Metal's texel (its chroma weighted 9/3/3/1 and rounded to a code, the fixed-point conversion), and the sample Metal's sum of the four taps' steps on the grid `2048`: twenty fetches and four conversions | The model reconstructs each channel at the point where Metal converts four taps and sums them: within the `float` arithmetic's rounding of the model (checked against an `f64` model on the CPU interpreter, `engine_ycbcr_in_place`), it differs from Metal's by the chroma the taps reconstruct, by Metal's roundings to a code and to `1/2048`, and where a tap's channel leaves `[0, 1]`, which Metal clamps before it sums. `HalfSampleFilter` / `FloatSampleFilter` under the bound converted 4:2:0 format, announced `ycbcr-filter-not-exact` | Radeon 780M (RADV): a 3840 x 2160 `YCBCR10_420_2P` frame sampled linearly into a 3840 x 2160 `BGRA8Unorm` target, GPU time per frame (median of 60): 6.4-6.9 ms under the exact form (`runtime_lane::video_frames`) |
| A vertex buffer layout a draw fetches in fixed function, on a driver whose fixed-function vertex input takes an element's displacement modulo `2^32` (`RuntimeDriverDefect::VertexFetchWrapsPast4GiB`: lavapipe), where the draw's indices (`uint` indices, a per-vertex layout) or its indirect arguments (any layout but a constant one) state how far it fetches: the layout keeps the fixed-function fetch, so an element `2^32` bytes or more past the buffer's bind offset reads the bytes at its displacement's low 32 bits, where Metal reads zero past a buffer that ends before them (`render/vertex_reach.rs`). A layout whose reach the draw's own values bound (its last vertex without indices, `65535` plus its base vertex with `ushort` indices, its base instance plus its last instance over the step rate) is pulled where that reach passes `2^32`, under either value, and reads zero. Other drivers read zero (RADV on the 780M, MoltenVK 1.4.2: `runtime_lane::far_vertex_fetches`). | `vertex_reach = VertexReachPolicy::Bounded` | `vertex_reach = VertexReachPolicy::Exact`: such a layout is pulled too, wherever its stride and furthest attribute could reach past `2^32` from the largest index or argument (on lavapipe, every `uint`-indexed draw's per-vertex layout of a stride above 0, and every indirect draw's layout but a constant one), the plan's pulled-vertex-buffer variant (`WorkCounts::far_vertex_fetch_pulled_draws`) | Under `Bounded`, on lavapipe, an element of such a draw at or past `2^32` reads the wrapped bytes: the conformance case `vertex-fetch-past-4gib-small-buffer` replayed under the default policy differs in 8 elements (its two far `uint`-indexed draws at stride 4), and passes under `Exact`. `VertexFetchMayWrap` (`vertex-fetch-may-wrap`), once per device, for a draw left so; `runtime_lane::far_vertex_fetches` (both values) | Measured on lavapipe (Mesa 26.2.4, LLVM 23.1.1) on a Core Ultra 9 275HX shared with other jobs (`runtime_lane::far_vertex_fetches::far_reach_costs`): a state drawn without indices and then with `uint` indices makes one pipeline under `Bounded` and two under `Exact`, the pulled variant made at the first `uint`-indexed draw; a pass of one draw of 65,536 `uint`-indexed points, from the encoder's making to the wait's return, takes 2.3 to 2.8 times as long pulled (medians of 20 passes: 0.94 against 2.41 ms beside two builds, 3.5 to 4.8 against 9.7 to 12.0 ms at a load average of 59 on 24 threads). On every other driver neither value changes a draw |

All at once: `DevicePolicy::exact_answers()` sets every switch above; each row states its own cost.

Speed choices that change no answer (the roll pass's floor; a specialization measured free) are not
rows.

## Not speed choices: device gaps

No switch gives Metal's answer here: the device lacks what an exact form needs, or no exact form is
known. The rest of the differences are docs/API.md section 9's precision notes.

| What it is | Why there is no switch | Note, code |
|---|---|---|
| The maths library (`sin`, `exp`, `log`, `pow`, `sinh`, `cosh`, `tanh`, `atan2`...) on `float` and `half`, fast divisions and Metal's fast reciprocal, a `half` fused multiply-add, and a `float` one without `OpFmaKHR` are the device's instructions on every device whose `float` functions are not Metal's own (every device but MoltenVK, whose functions are Metal's in a fast-math entry; a `half` function is the `float` instruction rounded on every device); Metal's precise division (an `fdiv` without `arcp`, or on `half`) is the device's `OpFDiv` on every device. `fabs`, `sign` and `fract` are `FAbs`, `FSign` and `Fract`, their special operands (a NaN, an infinity, a signed zero, an `f32` denormal) the instruction's; the fast `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh` and `atanh` are `Sin`, `Cos`, `Tan`, `Sinh`, `Cosh`, `Tanh` and `Atanh`, not MSL Table 8.2's formulas (`fast::sin(1e7)` is the device's sine, Metal's `+0`; `fast::tanh(45)` is `1`, Metal's NaN); a `half` function's signed zeros and NaN signs are its `float` instruction's (`half` `sin(-0)` is `-0`, Metal's `+0`), and on every device a `half` `fmod`'s zero remainder is its `float` remainder's (`fmod(-2, 0.5)` is `-0`, Metal's `+0`); `pow`, `powr` and `atan2` add only C99's special values that `GLSL.std.450` leaves undefined, by the device's comparisons (`pow(x, 1)` is the device's `Pow`, Metal's `x`; a NaN operand's NaN is the device's sum of the operands); `round` is C99's from `Trunc` on every device (`round(-0.3)` is `-0`, Metal's `+0`). | The device's arithmetic is the contract: no routine computes Metal's answer, and no select reads a special operand as Metal does, where the device has an instruction. | Vulkan's precision table (`Sin` 2^-11 absolute in [-π, π], `Exp` 3 + 2·\|x\| ulp, `Log` 3 ulp or 2^-21 absolute in [0.5, 2], `Pow` from `Exp2(y · Log2 x)`, `FAbs`, `FSign` and `Fract` correct or correctly rounded, `Atan` 4096 ulp) against Metal's 4-6 ulp; division 2.5 ulp; `fma` one `half` ulp (a `float` one may round twice). `NativeTranscendentals` (`native-transcendentals`), `NativeDivision` (`native-division`), `NativeFma` (`native-fma`) |
| `dot`, `mix` and `fast::sqrt` are the device's `OpDot`, `FMix` and `Sqrt` on every device, not Metal's fused chain, its two fused multiply-adds and `x · rsqrt(x)`: a `float` one on MoltenVK reaches Metal's own `dot`, `mix` and `fast::sqrt`, a `half` one is the instruction on its `float` container, rounded once; a `dot` of other than two to four lanes sums the `OpDot` of each chunk of up to four lanes. | The device's order is the contract. | `float` `dot` of n lanes within (3n − 1)·2^-24·Σ\|aᵢbᵢ\|, `mix` within 2^-21 of its two terms, `fast::sqrt` 4.5 ulp. `NativeDot` (`native-dot`), `NativeMix` (`native-mix`), `NativeTranscendentals` (`native-transcendentals`); a `dot` of other than two to four lanes within Vulkan's precision of its chunks' `OpDot` and sum, `DeviceArithmetic` (`device-arithmetic`) |
| A filter emulated by fetches sums its taps in the device's arithmetic on every device: Metal's bicubic filter on every format (the Catmull-Rom weights computed in `f32` from the fraction, the sixteen taps summed by the device's multiply and add, a 32-bit float format's or a level under four texels' bilinear filter by its `FMix`), and an extended-range texture's linear, 3D and linear-mip-filter samples (`FMix` of the decoded taps at the `f32` fractions); a `half` result is the device's narrowing. Metal's fixed-point weight table, its exact grid sums rounded half up, its float accumulators and its sixty-fourth level fractions are not reproduced. The footprint, the addressing, the filter each format selects, the level and the format's clamp stay Metal's. | The device's arithmetic is the contract (rule 1); the device has no cubic filter on any committed profile (`VK_EXT_filter_cubic`), and no format holds the extended-range codes. | A few grid steps (`1/4080` at 8 bits), up to `2.1e-3` on an extended-range sRGB bicubic sample (`xr_bicubic::WEIGHTS_BOUND`), up to `1/64` of the levels' difference under a linear mip filter. `BicubicFilterArithmetic` (`bicubic-filter-arithmetic`) |
| Fast-math sums, differences and products group, and a multiply contracts into an addition, as the device's compiler groups and fuses them, on every device, MoltenVK included: each operation is emitted in the AIR's order, one carrying both `contract` and `reassoc` undecorated and any other `NoContraction`, a `half` one on the device's 16-bit arithmetic too (its product a plain `OpFMul` when undecorated). Metal's compiler takes its own grouping (its rank order of a sum's leaves, its pairing of a product's lane-uniform factors) and its own fusion pairs. On MoltenVK an undecorated 16-bit `half` operation is computed in `float`: Metal's fast compile folds its conversions away. | The device's grouping is the contract: what the AIR's flags allow, the device's compiler chooses; nothing reproduces Metal's choice. | Unbounded relative to the result where a sum of mixed signs is grouped otherwise, or a multiply-add is fused on one side alone; each operation kept apart is exact as written; a MoltenVK `half` one left to the compiler by the rounding it skips. `FastMathRewrites` (`fast-math-rewrites`) |
| `air.pack` and `air.unpack` of a normalized format are the device's conversion: `GLSL.std.450`'s `Pack`/`Unpack` `Unorm4x8`, `Snorm4x8`, `Unorm2x16` and `Snorm2x16`, and for `rgb10a2` and `rgb565` their definition by the device's `NClamp`, multiply, `RoundEven` (a pack) and multiply by the reciprocal (an unpack); `unorm4x8_srgb`'s colour is the MSL transfer functions by the device's `Pow` around the 8-bit instruction, as is an 8-bit sRGB storage texture's write through its UNORM alias. A `float` one of the four named formats on MoltenVK reaches Metal's own; on MoltenVK on the Apple M3 Max's macOS 27 identity (`DriverDeviation::MetalPackConversions`) so do `rgb565` in `float` and, in `half`, `unorm4x8`, `snorm4x8`, `rgb10a2` and `rgb565` both ways and `unorm2x16` and `snorm2x16` packed, measured to answer Metal's at every code and every `half` (`runtime_lane::pack_conversions`). The same conversions serve an implicit imageblock access of a normalized attachment (a `half` load rounding the unpacked `float` once more), the interlocked storage form's 8-bit sRGB attachment and sRGB word, and a snapshot's normalized codes (`c` times the `f32` nearest `1/D`). So does every conversion of an extended-range, padded 10-bit or bfloat texel, which no Vulkan format holds (`BGR10_XR`, `BGRA10_XR`, `RGB10A8_2P_XR10`, `R10Unorm_X6`, `RG10Unorm_X12`, their sRGB twins, `RGB10A2Unorm_sRGB`, the bfloat formats; every device): a read `float(c - 384)` (a padded code's `float(c)`) times the `f32` nearest `1/510` (`1/1023`), a write `RoundEven` of the clamped value's `f32` product plus 384, an sRGB twin's colour the MSL transfer by the device's `Pow` on the magnitude, a `half` result the device's narrowing, a bfloat NaN as it is. | The device's conversion is the contract. | A packed code within one of Metal's, apart only where `clamp(c)·scale` lies within its rounding of a half-integer (`Round`'s tie is the device's); an unpacked value within 2.5 ulp of the correctly rounded quotient; a decoded sRGB colour within half a twelve-bit step (`1/8190`) of Metal's table step `k/4095`, beyond its `Pow`'s bound; a NaN channel's code the device's (Metal's: 0). An extended-range or padded value within one `f32` ulp of Metal's quotient, a code within one of Metal's exact product, an sRGB twin's decoded value within half a `2^-13` step of Metal's table (`R10Unorm_X6_sRGB`'s and `RGB10A2Unorm_sRGB`'s colour 1.64 steps) and its code within one (theirs within two below `0.0031308`, where Metal rounds the value to `2^-13` before its table), a `half` result within one `half` ulp (Metal rounds ties away from zero, a bfloat sample's `half` toward zero), a bfloat NaN the device's (Metal's: `0x7fc00000`, a write `0x7fc0`). `DevicePackConversion` (`device-pack-conversion`) |
| Off MoltenVK a `float` multiply, a fused multiply-add by `OpFmaKHR`, and a `float` `ldexp` (whose result below `2^-126` is that multiply's product) round before Metal's flush: Metal flushes a result whose exact value lies below `2^-126` before rounding it, so a result whose exact value lies within half a unit below `2^-126` answers `2^-126` of its sign where Metal answers the zero of its sign. | Every device measured off MoltenVK (RADV, ANV, NVIDIA, lavapipe) rounds first; its multiply is the contract. | Announced once per device as it is made: `float-tininess-after-rounding` |
| `f32` denormals in a `float` operation on a device not measured to read them as Metal does (`DriverDeviation::MetalFloatDenormals`: lavapipe, and every unmeasured device), in `floor`, `ceil`, `fneg` and `fabs` on a device not measured to read them as Metal's do (`DriverDeviation::MetalUnaryDenormals`: every device but MoltenVK on the Apple M3 Max's macOS 27 identity), and on every device in a nearest pixel sample's texel `floor(x)` (which Metal's sampler reads from a subnormal `x` itself, a negative one as texel -1, where MoltenVK's `floor` reads it as the zero of its sign: recorded only where the `s` or `t` mode is repeat, clamp to zero or clamp to the border, which read texel -1 apart from texel 0, through a constexpr sampler of such a mode, through a runtime slot under a pipeline made for a nearest pixel state and announced where the bound state's mode is such, through an argument buffer as `argument-pixel-sampler-scaling`), in the maths functions (`sign` and `fract` among them, and the zero and sign tests of `pow`, `powr` and `atan2`) and the device's order of the rows above, and in a conversion to or from `double`. Metal reads them as zero and flushes a denormal result. | The device's reading is the contract: Vulkan lets each device keep or flush them, and Metal runs no `double`, so its reading of a conversion is not measured. | `DeviceDenormals` (`device-denormals`) |
| Samples, gathers, depth comparisons and level queries are the device's own sampler's on every device whose sampler is not Metal's (every device but MoltenVK): its filter, footprint, level, anisotropy, sRGB decode and comparison, its level clamp (the descriptor's as given, which Metal holds on a `1/64` grid) and its reading of a subnormal coordinate (Metal's sampler reads a subnormal normalized coordinate as the zero of its sign before addressing, measured on an Apple M3 Max on `R32Float`, and addresses a negative subnormal pixel coordinate as texel -1, as an Apple M3 Max with macOS 27.0.1 answers the conformance case `pixel-samplers-subnormal-border`). A sample's or gather's residency on a device with `shaderResourceResidency` is the device's `OpImageSparse*` word and non-resident value (`SparseResidency`); Metal's footprint rule answers it only where the device has no residency and the slot's tile map does. A two-plane YCbCr texture's linear sample is the device's filter of its texels (the device's own Y′CbCr conversion's where it converts the format, next row; Metal's reads exactly elsewhere); where the device's memory is the host's, a sample that reads the planes in place is Metal's filter exactly, and a gather Metal's taps. An 8-bit sRGB format the device cannot sample (`R8_SRGB` on lavapipe, `R8G8_SRGB` on lavapipe, ANV and NVIDIA) is held as its UNORM twin, the device lacking its decode: a read, a sample, a fetch, a gather and a bicubic tap decode what the twin's view reads (one texel's codes, or the device's filter of them) by the MSL transfer through the device's `Pow`. | The device's sampler is the contract. | Linear filter within one code (1/255) on the measured devices, sRGB within 1.83, a two-plane YCbCr texture's within 1.183; levels, anisotropy, cubes, 3D, gathers and comparisons unmeasured. A twin's texel: the transfer within the device's `Pow` (MoltenVK's within 1.83e-7 through a storage alias, `runtime_lane::srgb_storage`) where Metal's table rounds it to a twelve-bit step (up to 1/8190 apart); a filtered sample decodes the filtered codes, up to 0.286 off for a blend of codes 0 and 255 (`SrgbTexelDecode`). `DeviceSampling` (`device-sampling`) |
| A pixel-coordinate sampler's linear sample, gather, comparison and level query is the device's normalized sampler's at the coordinate divided by the texture's level-0 size in the shader, on every device, MoltenVK included, where no other form takes it: through a runtime sampler slot or an argument buffer's sampler, whose state the function does not know, and through a constexpr one of an array, a 3D texture or a texture whose residency the function observes. The device reads its texels and weights from that coordinate in its own arithmetic, where Metal's sampler reads the coordinate as given on its own sub-texel grid. A constexpr pixel sampler's sample or gather of a 1D or 2D texture that is no array hands the coordinate itself, moved by its offset, to the device's unnormalized sampler at level 0 (a nearest sample with an offset its texel's centre `floor(x) + 0.5 + k`), which reads it as Metal's does on MoltenVK; a nearest sample elsewhere hands the device's normalized sampler its texel's centre `floor(x) + 0.5` over the size, where the device reads exactly Metal's texel `floor(x)` (a runtime slot's behind its `NearestPixelSampler` specialization, an argument buffer's by its state's word); a constexpr sampler that filters nearest and clamps to the edge fetches its texel (`sampler.pixel_fetch`). | The device's sampler is the contract. Vulkan's unnormalized samplers take no offset operand, comparison, array, 3D image, mip level or anisotropy, and a runtime or argument buffer's state is not known when the function is made, so the shader scales the coordinate there; no path reproduces Metal's sub-texel grid. | The device answers as its normalized sampler at a coordinate within one `f32` ulp of the quotient on MoltenVK (10 of 800 cells of each mirror-clamp linear configuration move by it) and at the quotient itself on lavapipe (`runtime_lane::pixel_samplers`: 800 coordinates at and around each texel edge of a 5x3 texture, every filter and address mode); on MoltenVK a linear sample through a runtime or argument buffer's sampler, or of an array, reads another weight step than Metal's at under 1% of the coordinates at and around each texel edge of textures 258, 21 and 37 texels wide, where `x - 0.5` lies at a half step of Metal's `1/256` grid (`runtime_lane::pixel_edges`); a linear downscale by 8 of textures 1000 to 1920 texels wide within 1/255 of Metal's on MoltenVK and 2/255 on lavapipe (`runtime_lane::pixel_downscales`); other devices unmeasured. Every nearest access of those edge probes reads Metal's texel on MoltenVK and on lavapipe, and every constexpr sample and gather of a 1D or 2D texture answers Metal's on MoltenVK (`runtime_lane::pixel_edges`); elsewhere a constexpr linear sample is the device's own filter in unnormalized coordinates (`device-sampling`), which on lavapipe differs from its normalized filter in the last bits of 79 to 107 of 800 cells of clamp to zero and to the border (`runtime_lane::pixel_samplers`). `PixelSamplerScaling` (`pixel-sampler-scaling`), `ArgumentPixelSamplerScaling` (`argument-pixel-sampler-scaling`) |
| A two-plane YCbCr texture's texels are the device's own sampler Y′CbCr conversion of its planes where the device converts their format (`ycbcr_device.rs`): unpacked 8- or 10-bit codes of a 4:2:0, 4:2:2 or 4:4:4 layout over plane 0 that form one image (no chroma codes 0, no rows past the buffer; a 4:4:4 layout read as codes aside, whose identity conversion converts nothing), on a device that enables `samplerYcbcrConversion` (a 4:4:4 layout also needs Vulkan 1.3 or `VK_EXT_ycbcr_2plane_444_formats`), reports the multi-planar format's conversion and, for a 4:2:0 or 4:2:2 layout, Metal's reconstruction (midpoint chroma through a linear chroma filter), and has no defect in the format; the planes are copied into an image of it and sampled once per texel into the texture's image (`bind::ycbcr_sample_module`), through the refresh's indirect arguments. The conversion takes the surface's matrix and range (the identity model for the code-reading formats), and Metal's reconstruction of a subsampled layout; a device's cosited or nearest reconstruction would move what the codes mean to Metal (cosited, up to 1340/2048 of random planes on lavapipe), a capability it lacks, so such a layout converts through Metal's arithmetic. A render pass's load stays Metal's arithmetic, which our store inverts, so a loaded texel no draw changes keeps its codes, as `MTLLoadActionLoad` keeps the attachment's contents. lavapipe (Mesa 26.0.8) converts the 8-bit 4:4:4 layout (its 4:2:0 and 4:2:2 ones are cosited only, and it reads 10-bit codes as 16-bit words, `RuntimeDriverDefect::TenBitCodesReadAsWords`); MoltenVK 1.4.2 converts none (its 4:2:0 and 4:2:2 reconstruction is misplaced, `RuntimeDriverDefect::TwoPlaneChromaMisplaced`, and it reports no 4:4:4 two-plane format). RADV (Mesa 26.2.4) reads 10-bit codes as 16-bit words too (measured on a linear image of the 4:2:0 format on the Radeon 780M), so its 10-bit layouts convert through Metal's arithmetic; so do NVIDIA's (615.71.09, RTX 5080 Laptop GPU), whose conversion reads a 10-bit code's six unused low bits as its fraction, and which converts the 8-bit 4:2:0, 4:2:2 and 4:4:4 layouts. Elsewhere, and for packed codes, chroma codes 0 and plane 1, Metal's fixed-point conversion stays as capability (`WorkCounts::ycbcr_device_conversions`). | The device's conversion is the contract: Vulkan states the reconstruction's weights and the model's matrices, the precision of each step at least the format's, and the clamp of the result into `[0, 1]` the implementation's. | Each channel within 0.04/2048 of Vulkan's conversion past the `half` its image stores on lavapipe's 8-bit 4:4:4 layout, and within 0.034/2048 on NVIDIA's 8-bit layouts; of Metal's read within 6/2048 where the result lies in `[0, 1]` on lavapipe, 12.4/2048 on NVIDIA's 4:2:0 (19.1 through an `_sRGB` twin's curve; Metal rounds each reconstructed chroma to a code), and unclamped outside it; a code-reading format's codes the nearest to Vulkan's reconstruction, NVIDIA's Metal's (random planes, every texel, `tests/ycbcr_device.rs`). `YCbCrFilterNotExact` (`ycbcr-filter-not-exact`) |
| A texel's `float` value is the device's conversion of its codes on every device whose texture unit is not Metal's (every device but MoltenVK), and a `half` result the device's `float` texel rounded to nearest even on every device: on MoltenVK the `float` texel is Metal's, whose `half` Metal rounds otherwise for a 32-bit float texel's sample (toward zero, saturating at `±65504`) and the 16-bit normalized grids (three steps round twice through `float`); a two-plane YCbCr field on the grid `8192` keeps Metal's ties away from zero in one texel's read, sample and gather, but in a sample through a runtime sampler slot, whose `half` is the device's rounding of the step (toward zero on MoltenVK 1.4.2 over the Apple M3 Max, measured over 8,409 channels by `runtime_lane::ycbcr_textures`; unmeasured on other Apple GPUs). An sRGB texel (the 8-bit sRGB formats, and the sRGB forms of BC1 to BC3, BC7, PVRTC, ETC2, EAC and ASTC, a decoded twin's through its `R8G8B8A8_SRGB` view) is the device's sRGB decode through the texture's sRGB view, in every read, sample, gather, bicubic tap and input-attachment read, where Metal decodes each code through its twelve-bit table; so is a resolve's into an 8-bit UNORM attachment's sRGB twin, which decodes the samples and encodes their average through the twin's sRGB views where Metal's resolve takes its own tables (`DeviceRoundedResolve`). An 8-bit sRGB storage texture held through its UNORM alias (Vulkan has no sRGB storage view) reads its codes by the MSL transfer through the device's `Pow`, and writes them by that transfer and `PackUnorm4x8` (`DevicePackConversion`). | The device's conversion is the contract. | One `f32` ulp for 8- and 16-bit codes, and for 10-bit ones but where the device replicates their bits to 16 (NVIDIA's RTX 5080 Laptop GPU: 0.015 of a code); 0.18 of a code for 5- and 6-bit packed channels; `half` within one ulp, but a 32-bit float texel's sample past `±65504` the infinity of its sign where Metal saturates (MoltenVK included); NaN bits the device's. An sRGB decode: lavapipe's within 5.25 steps of Metal's table (0.00128) at every code, MoltenVK's Metal's at every code (`runtime_lane::srgb_decode`); Vulkan sets no precision for it, and other devices are not measured. `DeviceTexelConversion` (`device-texel-conversion`), `HalfTexelConversion` (`half-texel-conversion`, also recorded for heap textures and unmeasured formats), `SrgbTexelDecode` (`srgb-texel-decode`, under the bound format) |
| A texture of a BC format (`BC1_RGBA` to `BC7_RGBAUnorm_sRGB`) is the device's decode of its own BC image on every device: where its driver is not measured to decode the format as Metal does (BC1 to BC3 off MoltenVK and RADV on the 780M and the RX 7900 XTX; BC4 and BC5 off MoltenVK; BC7 on lavapipe), its reads, samples and gathers answer the device's interpolation of the blocks, where Metal's weights are 6-bit (an sRGB form's code then through the device's sRGB decode, the texel-conversion row). No BC texture keeps a decoded twin. | The device has the format: its decode is the contract. | A BC1 to BC3 channel within 4/255 of Metal's on NVIDIA (BC3's alpha 6/255), 3/255 on lavapipe, 1/255 on ANV; a BC4 or BC5 channel within 7.4/255 (signed 0.065) on NVIDIA, 1.3/255 (signed 0.013) on ANV, 2.4/255 (signed 0.022) on lavapipe, a signed one with an endpoint of -128 within 1.7/255 on RADV; a BC7 channel a `float` ulp on lavapipe; other drivers unmeasured. `CompressedTexelDecode` (`compressed-texel-decode`) |
| A vertex attribute that fixed-function vertex input fetches is the device's conversion of its format: a normalized one (`UCharNormalized` through `Short4Normalized` of either sign, `UChar4Normalized_BGRA`, `Int1010102Normalized`, `UInt1010102Normalized`) through its own Vulkan `UNORM` or `SNORM` vertex format, where Metal divides each code by its maximum correctly rounded (an `Int1010102Normalized` 10-bit field it multiplies by `float(1 / 511)` in `float`) and rounds the quotient once to a `half` attribute; a narrow float's (a `half` format, `FloatRG11B10`) read as `float` by the device's widening, NaNs included. The format is `VertexAttribute` state that no instruction names, so every format the device fetches shares one module, and the render pipeline's link notes the conversion on its vertex stage (`pipeline::link::vertex`). A normalized attribute the shader pulls instead (an offset, stride or step fixed-function input cannot fetch, a kernel's `[[stage_in]]` attribute) is the module's product of the code and its maximum's reciprocal, `float(c) · float(1 / (2^b - 1))`, the signed one clamped to -1 (`lower::stage_io::fetch`), noted by the function on every device where it misses Metal's fetch. **Exempt:** a normalized attribute on a profile that takes `DriverDeviation::MetalNormalizedVertexFetch`, MoltenVK on Apple's GPUs of the Apple M3 Max's macOS 27 identity, which hands the format to Metal's own fetch and was measured to answer Metal's `float` at every code of all 19 normalized formats, per vertex and per instance (`runtime_lane::normalized_vertex_fetch`): its pipeline records no note and requires `VkProperty::MetalNormalizedVertexFetch`, but for a 16-bit format read as `half`, whose `float` the module rounds to `half` again. | The device's conversion is the contract. | One `f32` ulp at some codes on a driver that multiplies by the reciprocal: llvmpipe, read at every code (Mesa 26.0.8 on LLVM 21.1.8, arm64), is within one ulp per vertex at every normalized format (127/255 as `0x3EFEFF00` where Metal reads `0x3EFEFEFF`; `Int1010102Normalized` Metal's), and per instance reads the most negative code of an 8- or 16-bit signed format unclamped (-128/127, -32768/32767) where Metal reads -1; RADV's unsigned formats are Metal's at the sites measured. A 16-bit format read as `half`: one `half` ulp at codes 65455 and 65519 unsigned, 32759 and -32759 signed, on a device whose `float` is Metal's (the quotient rounded to `half` twice). The signed formats are **unaudited** on RADV, ANV and NVIDIA. A pulled attribute's product, on every device: one `f32` ulp from Metal's quotient at 126 of the 256 8-bit unsigned codes (high), and toward zero at 16 signed 8-bit codes, 24 unsigned 10-bit codes, 512 unsigned and 1,536 signed 16-bit codes; `Int1010102Normalized` read as `float`, and an 8- or 10-bit format read as `half`, Metal's at every code; a 16-bit format read as `half`, one `half` ulp at the same four codes. A narrow float's NaN is the device's widening's, where Metal answers `0x7fc00000`. `DeviceVertexConversion` (`device-vertex-conversion`), `DeviceNanEncoding` (`device-nan-encoding`) |
| A texture of an ASTC LDR format on a device without that ASTC format reads a decoded twin of 8-bit codes, each channel the code nearest Metal's 12-bit value (`C >> 4` over 4095); an sRGB form's twin holds Metal's own 8-bit codes, which its slots read through the twin's `R8G8B8A8_SRGB` view, the device's sRGB decode (the texel-conversion row). A texture of an EAC 11-bit format (`R11`, `RG11`, unsigned and signed) on a device without ETC2 reads one of 16-bit normalized codes, each nearest Metal's value (`c / 2047`, signed `c / 1023`). | The device lacks the format; the twin's codes are the compact form the device samples. | Within 1/510 of Metal's value; an EAC 11-bit channel within half a 16-bit step (1/131070, signed 1/65534). `decoded-texel-precision`, announced where a command first reads the twin (a sampled binding of the texture, or a heap set holding it), not where the texture is made |
| A texture of a 4:2:2 format (`GBGR422`, `BGRG422`) reads a decoded twin of 8-bit codes on every device (`chroma_pairs.rs`): where the device converts the pairs' format, `G8B8G8R8_422_UNORM` or `B8G8R8G8_422_UNORM`, with Metal's reconstruction (its optimal tiling reports `SAMPLED_IMAGE`, `TRANSFER_DST`, `MIDPOINT_CHROMA_SAMPLES` and a linear chroma filter; an even width), each decode pass copies the written rows' pairs into an image of that format and samples it through the device's own sampler Y′CbCr conversion (the identity model at full range) into the twin (`ycbcr_device.rs`, `bind::chroma_sample_module`), each channel the nearest 8-bit code to the device's value; elsewhere the twin holds Metal's decode, `(3c + n + 2) >> 2` of the texel's pair `c` and the pair beside it `n`, as capability. MoltenVK 1.4.2 converts both formats; lavapipe (Mesa 26.0.8) filters their chroma nearest only and RADV reports neither format, so theirs hold Metal's decode (`WorkCounts::chroma_device_twins`). | Vulkan views a 4:2:2 format only through a sampler Y′CbCr conversion, which no pipeline's layout bakes: the twin is the form every slot samples, and where the device converts the format the conversion is the contract. | Green the texel's own code; red and blue within half a code of Vulkan's `(3c + n) / 4` on MoltenVK 1.4.2 (the nearest code; random pairs, every texel of 40 x 24, `tests/ycbcr_device.rs`), where Metal's `(3c + n + 2) >> 2` rounds that sum's ties up. `DecodedTexelPrecision` (`decoded-texel-precision`), announced where a command first reads the twin (a sampled binding of the texture, or a heap set holding it), not where the texture is made |
| A two-plane YCbCr texture whose planes are read in place (`texture/ycbcr_image.rs`) is sampled through the device's own sampler Y′CbCr conversion where the device converts their 4:2:0 format as a linear image with Metal's reconstruction (its linear tiling of `G8_B8R8_2PLANE_420_UNORM` or `G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16` reports `SAMPLED_IMAGE`, `MIDPOINT_CHROMA_SAMPLES` and a linear chroma filter, and its driver has no defect in the format): the planes are also a linear image of the format bound over their bytes in memory the runtime allocated (`texture/ycbcr_planes.rs`; an even size, the driver's planes at the surface's offsets and row pitches), and a slot whose every sample the device's sampler takes (no border address mode, no texel offset placed by a parent's size, no gather, which `VUID-RuntimeSpirv-OpTypeSampledImage-12206` keeps from a conversion's sampler) binds it as a combined image sampler whose immutable sampler bakes the conversion (`bind::YCbCrPlanesSampler`), each sample the device's at level 0, a nearest one at the centre of the luma texel Metal reads. Elsewhere the texel buffer read stays, the shader converting and filtering the codes, as capability (a linear sample Vulkan's model by default, Metal's fixed-point conversion and exact filter under `YCbCrInPlacePolicy::Exact`: the fast-defaults row): on a device whose conversion is not Metal's reconstruction (lavapipe's 4:2:0 conversion is cosited only, MoltenVK 1.4.2's misplaced, `TwoPlaneChromaMisplaced`; RADV and NVIDIA read 10-bit codes as words, `TenBitCodesReadAsWords`), over fresh pages, a no-copy buffer's pages or the caller's `VkBuffer` (an image bound to imported memory starts `UNDEFINED`, and the in-place check covers that first move for single-plane images alone), where the driver lays the planes out otherwise (RADV's linear rows lie 256 bytes apart), and for a slot that samples with a border or a parent-placed offset, or gathers (`WorkCounts::ycbcr_planes_images`, `ycbcr_planes_images_kept_*`). | Where the device has Metal's reconstruction, its conversion and filter are the contract; a device without it reads Metal's texels through the texel buffer, which is the capability it lacks. | The device's conversion and filter of the codes, not Metal's. A bare Vulkan probe on the Radeon 780M (RADV, Mesa 26.2.4, 8-bit codes) read each luma centre within 0.0003 codes of Vulkan's model and points between texels within 0.66 codes, where Metal's conversion then filter differs by up to 54/255 between texels of random planes (the device reconstructs the chroma at the point); the runtime's path is not measured on a device that takes it. GPU time of that probe, 3840 x 2160 composited into 1680 x 1050: 248 µs against the texel buffer's 340, one pass. `YCbCrFilterNotExact` (`ycbcr-filter-not-exact`), announced where the image is made |
| `generate_mipmaps` of a filtered format is the device's filter of each level at the scaled texel centres: its blit chain (`vkCmdBlitImage` with `VK_FILTER_LINEAR`) where its blits filter the image's format, else the runtime's bilinear (trilinear in 3D) compute filter in `float` (an extended-range format, an 8-bit sRGB texture held as its UNORM twin, a format without blits or linear filtering, and a 3D texture on a driver whose blits do not filter a volume, `RuntimeDriverDefect::VolumeBlitsUnfiltered`: MoltenVK's), by the device's arithmetic: its products and sums as it contracts them, an 8-bit or extended-range sRGB code through the MSL transfer by its `Pow`, a half by its widening and narrowing, a NaN its own. An integer format's texels are picked as plain Metal picks them, exactly. | The device's filter is the contract; Apple's datapath is not modelled, and its answers differ between Apple's GPUs. | `DeviceFilteredMipmaps` (`device-filtered-mipmaps`) |
| A draw biasing a `Depth16Unorm` depth by a constant, or by a slope alone, biases by Metal's constant `c` in the device's own unit `r` on every device: Metal biases by `c + 1` units of 1/65535 for `c` of 0 or more and `c - 1` below (one unit for a slope alone), added to the depth rounded toward zero, where Vulkan's `r` of a 16-bit unorm depth is the implementation's, at most `2 × 2^-16`, and the device rounds the sum its own way. MoltenVK hands Metal its constant (`DriverDeviation::MetalDepth16BiasedStores`). | The device's depth bias is the contract. | Another code than Metal's for a biased depth: one unit's difference or more. `UnormDepthBias` (`unorm-depth-bias`) |
| A render pass that loads and stores an extended-range, padded 10-bit or bfloat color attachment (`BGR10_XR`, `BGRA10_XR`, `RGB10A8_2P_XR10`, `R10Unorm_X6`, `RG10Unorm_X12` and their sRGB twins, `RGB10A2Unorm_sRGB`, `R16Bfloat`, `RG16Bfloat`, `RGBA16Bfloat`) keeps the bytes its draws do not write, on every device, MoltenVK included (such an attachment is an image of an unsigned integer format there too): Metal's pass writes each 32 x 32 tile a draw reaches back through its tile store, the unused bits of the codes made canonical, `BGRA10_XR`'s alpha code clamped to `[384, 894]`, a padded sRGB twin's codes through Metal's table and a bfloat code through `half` (a NaN to `0x7fc0`). A single-sampled signed normalized or `RGB9E5Float` attachment likewise keeps its minimum codes and unnormalized words, but on MoltenVK, whose pass is a Metal pass (`DriverFact::TileStoreIsMetals`). | The device's load and store are the contract: no pass rewrites the texels its draws do not write. | Bytes no Metal write left (uploads, copies) differ in the tiles a draw reaches; an extended-range texel reads the same value either way, a bfloat code off `half`'s grid reads its own value. `ExtendedRangeBitsKept` (`extended-range-bits-kept`), `SnormMinimumKept` (`snorm-minimum-kept`), `SharedExponentKept` (`shared-exponent-kept`) |
| Depth values outside [0, 1] stored clamped on a device without `VK_EXT_depth_range_unrestricted` (RADV, ANV and MoltenVK, whatever a driver is measured to keep): a depth comparison the device makes records the note. | Vulkan gives a depth image no such value without the extension, by a copy or a render write: enable it. | `RestrictedDepthRange` (`restricted-depth-range`) |
| A 32-bit float depth outside [0, 1] uploaded into a depth image, on a device without `VK_EXT_depth_range_unrestricted` (ANV on the Iris Xe of RPL-P, which stores it clamped; MoltenVK on the M3 Max, which exposes no such extension, whatever its driver keeps): a blit from a buffer, `replace_region`, a blit from a colour texture, and a heap's depth texture whose shadow copies the heap's bytes into its image: each clamps the value into [0, 1], a NaN to 1.0, before the image takes it (`replace_region` on the CPU, `bind::clamped_depth`; a blit by a dispatch of `bind::depth_clamp_module`; a heap texture's copies in, `TiledLevelMove::clamps_depth`), and a heap texture's copies back then write the clamped value, in texels no command wrote too. | Vulkan takes no such copied value without the extension (`VUID-vkCmdCopyBufferToImage-pRegions-07931`), the device stores one clamped, and a depth image holds a depth by no other means. | `RestrictedDepthUpload` (`restricted-depth-upload`) |
| A multisampled 32-bit float depth written by a blit copy from a buffer or a colour texture, or a separate depth texture's copies to and from its pass's scratch image, on a device whose depth/colour copies are missing or wrong (lavapipe, MoltenVK, NVIDIA's GA10x, a device without `VK_KHR_maintenance8`): the values its driver's fragment depth changes (MoltenVK's, NVIDIA's), or on a driver not measured may change (outside [0, 1], denormals, -0), land changed. | Vulkan writes a multisampled depth image's samples by no other means there but a clear; a driver measured to store every fragment depth unchanged (lavapipe's measured releases) is exact. | `DrawnDepthCopy` (`drawn-depth-copy`) |
| Signed zeros, infinities and NaNs where the device lacks `SignedZeroInfNanPreserve` at the operation's width. | A device capability. | `DeviceSignedZeroInfNan` (`device-signed-zero-inf-nan`) |
| Every device but MoltenVK interpolates varyings by its own rounding, up to about `1.5e-4` from Metal's (`3e-4` on lavapipe) on the probes. MoltenVK hands the varyings to Metal. On RADV a perspective input read at a sample or an offset is further off: the row below. | The exact path needs Metal's interpolation formula, which is not known. | `DeviceInterpolation` (`device-interpolation`) |
| On RADV, a perspective input read at a sample or an offset (`interpolate_at_sample`, `interpolate_at_offset`, and a sample-qualified varying a function looping over its covered samples reads at each pass's sample) is the driver's value at the pixel center extrapolated along the quad's coarse derivatives, a first-order answer, where Metal interpolates in perspective at the location. It is off by the value's curvature across the distance: on the pull-model lane's triangle (clip `w` 1, 2 and 4 over a 6x6 target) up to about 1.0 in a lane whose vertices hand 3, -2 and 0.5, on the Radeon 780M with Mesa 26.2.6. A no-perspective input is exact there. | The device's interpolation is the only form (Vulkan states no precision for it); computing the perspective division in the shader would replace it. | `ExtrapolatedInterpolation` (`extrapolated-interpolation`) |
| `interpolate_at_offset` reads at the offset moved to the device's origin and clamped to its `[minInterpolationOffset, maxInterpolationOffset]`, which Vulkan requires, and the device rounds it to its `subPixelInterpolationOffsetBits`; Metal reads at the offset as given, from the pixel's corner, at any distance and on no grid (an Apple M3 Max). Within the range the read lies within half a grid step of Metal's (MoltenVK: Metal's, its range `[-1/16, 15/16]` from the corner); past it, at the range's nearer end. | Vulkan states the range and the grid; no exact form reads past them. | `DeviceInterpolationOffset` (`device-interpolation-offset`) |
| Every device but MoltenVK on Apple's GPUs of the Apple M3 Max's macOS 27 identity covers by alpha (`alphaToCoverageEnabled`) by its own algorithm: lavapipe takes a 2x2 ordered dither at one sample too, where Metal covers a pixel there only above an alpha of 1/2, so a whole fragment is stored or dropped between a pixel's threshold and 1/2; at four samples it is Metal's but within an `f32` ulp of a threshold, one sample apart. MoltenVK hands the field to Metal. | The device's coverage is its native answer, and Vulkan specifies no algorithm for it; Metal's dither computed in the shader would emulate the device's own coverage. | `DeviceAlphaCoverage` (`device-alpha-coverage`) |
| A fast-math function's `half` color into a blend: within three `half` ulps (16-bit float) or two `float` ulps (32-bit float). A fetched 16-bit float attachment's blend, computed in the pixel interlock, where Metal folds the output's own arithmetic into it or regroups it: within 2 `half` ulps at the operands' magnitude, 16 at the stored value's under cancellation; exact elsewhere. | Metal's compiler regroups the blend per pipeline, following its passes over the whole function: a form the blend state decides is Metal's only where it is measured, and a lane of the output the compiler can fold into the blend has no form at all. Two 16-bit float states are measured, and a fetched attachment's blend takes Metal's forms there; `SourceAlphaSaturated` over `OneMinusSourceAlpha` shows why the rest are not, Metal rounding the destination's product first at one and four channels and the source's at two. A device that enables `storageInputOutput16` receives the output as a 16-bit `half` (`StageIo::half_outputs`): MoltenVK 1.4.2 on the M3 Max then stores Metal's codes at every word of `half-output-blends-*` (a loaded `half4`, 21 blends into 16- and 32-bit float formats). | `HalfOutputFastMathBlend` (`half-output-fast-math-blend`) |
| A fetched blended attachment of a format whose Metal blend arithmetic is not built (16-bit normalized, 8-bit signed normalized, 10-bit, packed float, `A8Unorm`; an 8-bit unorm one through fewer than four output lanes; a 32-bit float one (`R32Float`, `RG32Float`, `RGBA32Float`); a 16-bit float or 8-bit unorm one under a blend of the second source), on a device that orders fetch only through the pixel interlock (MoltenVK, RADV, ANV): the shader blends in `float`, each step rounded, the source and blend color clamped to a normalized format's range, the device's conversion storing the result. An 8-bit sRGB one under a blend of the second source, a one- or two-channel one under a fast-math lerp, and one written by an output of fewer than four lanes: the 8-bit unorm blend's rounded forms over the decoded texel, encoded by Metal's table (Metal's without fast math, at four lanes). Bound not measured. | No Vulkan blend applies to the interlock's storage write, and Metal's blend arithmetic for these formats is not measured, so no exact form is known. A 32-bit float attachment's blend in `float` is the device's own arithmetic, as its blend unit would compute it (every committed profile but the Vulkan 1.2 floor blends the three formats; a pipeline whose attachment is not fetched takes that blend). | `UnmeasuredFetchBlend` (`unmeasured-fetch-blend`) |
| A framebuffer fetch on a device with neither rasterization-order color attachment access nor `fragmentShaderPixelInterlock` (Honeykrisp on Apple's M2 Pro; RADV on the RX 550) reads the attachment as an input attachment, ordered after every earlier draw by a barrier between the draws (`WorkCounts::fetch_draw_barriers`) but not after an overlapping earlier fragment of its own draw, which Metal orders in primitive order. Exact for a draw whose primitives do not overlap at a pixel. | Vulkan orders an input attachment read after an overlapping earlier fragment of the same draw only through rasterization-order attachment access, and a storage read only inside the pixel interlock; the device has neither. Drawing each primitive as a draw of its own behind a barrier would order them, at a draw and a barrier per primitive; that form is not built. | `UnorderedFetchWithinDraw` (`unordered-fetch-within-draw`) |
| A color blended into an extended-range attachment (`BGR10_XR`, `BGRA10_XR`, `RGB10A8_2P_XR10` and their sRGB twins) or a padded 10-bit one (`R10Unorm_X6`, `RG10Unorm_X12`, their sRGB twins, `RGB10A2Unorm_sRGB`) is blended in the shader over an ordered read of the attachment (inside the pixel interlock, or over its rasterization-order input attachment), in the device's arithmetic: `S·Fs op D·Fd` in `float`, in the blend equation's order, `1 - x` in `float`, `SourceAlphaSaturated` `min(sa, 1 - da)`, the source, second source and blend color clamped to the format's range first (a NaN to its low end; a padded format's not clamped, as Metal clamps none there), then encoded as the kernel write encodes. The device's compiler may contract and regroup it: MoltenVK on the M3 Max and NVIDIA's driver on an RTX 5080 compute `S·F + D·(1 - F)` as `D + F·(S - D)`. Every device. Measured against Metal by the conformance suite's extended-range and padded blend cases: a stored channel within two codes of Metal's on a linear format and within six on an sRGB twin near zero, and where the device's compiler regroups a padded format's blend, an infinite second source stores the end code its sign picks where Metal's sum is NaN, stored as code 0. | No device's blend unit reads the format's raw codes, so the shader blends (capability), and its arithmetic is the device's: Metal's compiler fuses the blend into the function, regroups it and rounds each step to `half`, a form no device has natively. | `ExtendedRangeBlendSource` (`extended-range-blend-source`), `ExtendedRangeFloatBlendSource` (`extended-range-float-blend-source`) |
| Float simdgroup reductions sum in the device's association order. | Apple's order is not modelled. | `ReductionOrder` (`reduction-order`) |
| A vertex or fragment function keeps more than 16 KiB a thread of private memory, up to the 192 KiB Metal holds, on a device whose Function storage was not measured: it runs on the driver's scratch, which past what the driver holds answers wrongly or hangs the GPU with no error. | Vulkan states no amount of Function storage; only a measurement of the device does. No other form is exact: a fragment function's helper invocations may not write the memory it would move to, and a vertex function's, a compute pre-pass, is not built. | `FunctionStorageAssumed` (`function-storage-assumed`) |
| A render pipeline of more color attachments than the device draws at once (the lesser of `maxColorAttachments` and `maxFragmentOutputAttachments`, four at Vulkan's floor; Metal takes eight) draws each draw once per group of that many, each group in a render pass of its own over a twin of the pass's depth and stencil. Every variant's vertex stage is one module, `Position` invariant, and every fragment variant's `FragDepth`, `SampleMask` and `FragStencilRefEXT` are `Invariant`; groups past the first may differ at discard or depth edges if the driver compiles the variants differently. | The device holds no more attachments in one subpass. Vulkan's invariance rules make only identical pipelines repeat exactly, and the groups' fragment modules and attachments differ. | `ColorAttachmentGroups` (`color-attachment-groups`) |
| On NVIDIA's driver (615.71.09 on an RTX 5080 Laptop GPU) one `vkQueueSubmit` in every 1024 that signal on a `VkQueue` takes 2 to 4 ms in the driver (1.97 to 4.06 ms measured), which holds the command buffer whose batch it falls on, and the ones committed behind it, from the GPU that long. | The cost is the driver's, and every batch signals its completion. The scheduler's thread takes the stall while idle, by empty submissions, once one was seen (`DriverFact::SignallingSubmitsStallEvery1024`, `device/submit_stalls.rs`; a call in which the calling thread spent 1 ms or more of its CPU, and once one came where the count put it, only there; a call that only waits that long, as beside other processes' GPU work, is none): it falls on work only before then, after a use of the queue the runtime cannot count, or on a queue that never idles 1 ms. Whether the driver's stall also holds other threads' semaphore calls is unsettled: a probe polling the timeline from a second thread during 10 stalls saw 5 of its 9 slow calls (up to 7.1 ms) overlap one in one run, and none of its 3 in another; in `host_ab present` the completion handlers of command buffers that completed during a stall taken idle ran 126 us (p50, 341 at most) after their GPU end, as elsewhere. | None (timing only; `WorkCounts::submit_stalls_on_work`, `WorkCounts::submit_stalls_idle`) |
| Command buffers that share no resource run one after another, where Metal overlaps them while the GPU has room. | The runtime orders nothing between them and runs at each driver's floor for one completion signal per command buffer; a signal shared by several would hold an earlier one's completion for the later ones' work. | None (timing only) |
| On MoltenVK a light queue's command buffer waits for the latest one another queue committed, where Metal waits for none: MoltenVK fences every encoder of every queue device-wide when argument buffers and a residency set are on. | The runtime needs argument buffers, and MoltenVK has no setting for its residency set; `runtime_lane::queue_storms` judges the runtime against Metal under those fences. | None (timing only) |
| A `Boolean` visibility result where a sample passed is the device's occlusion query result without `VK_QUERY_CONTROL_PRECISE_BIT` (ORed over a split pass's render pass instances, and under `Accumulate` into the word the buffer held), which Vulkan states only as nonzero; where none passed it is 0 (under `Accumulate`, the word). An Apple M3 Max writes 1, Metal on the AMD Radeon Pro 5300M and the Intel UHD Graphics 630 (macOS 26.7.1) the samples that passed. | Metal's contract is nonzero, and the device's query answers it natively: no dispatch rewrites the word to 1. | `DeviceBooleanVisibility` (`device-boolean-visibility`) |
| A `Counting` visibility result of a draw writing more than one color attachment, on ANV (Mesa 26.2.4): Metal's count multiplied by the colors written, not divided back. `Boolean` results keep their contract (zero where no sample passed). | The driver counts each sample once per color output (`RuntimeDriverDefect::OcclusionCountPerColorOutput`), no documented Vulkan form counts it once, and a refusal would withhold the Mac2 and Metal3 families. | `VisibilityCountPerColor` (`visibility-count-per-color`) |
| A visibility result of a draw whose fragment function declares `[[early_fragment_tests]]` and writes its sample mask (the pipeline's private sample mask or coverage, or its own `[[sample_mask]]` return), on a device without `earlyFragmentMultisampleCoverageAfterSampleCounting` (ANV on the Iris Xe of RPL-P): the count may leave out the samples the mask drops, which Metal counts; a `Boolean` result may be 0 where Metal's passed. | Vulkan lets such a device count after the fragment shader's mask, every form that masks the colors alone (the `SampleMask` output, a discard) is a fragment operation it may count after, and only a second draw of each counted draw without the mask would count as Metal does. | `VisibilityPastEarlyMask` (`visibility-past-early-mask`) |
| On MoltenVK a no-copy buffer or heap released with no keep-alive while batches are in flight runs its deallocator once the batches submitted before its release completed, and the program's own submissions through `CommandQueue::with_vk_queue` before it, where Metal runs it at the release. | MoltenVK 1.4.1 fails pending work that reads device addresses where memory a device-address buffer lay in is freed (`RuntimeDriverDefect::FreedAddressMemoryFailsPendingWork`), so freed memory waits for the batches in flight (`WorkCounts::freed_memory_parked`). | `HostDeallocatorDelayed` (`host-deallocator-delayed`) |
| On MoltenVK a `Shared` or `Managed` texture whose format packs more than one texel into a block (every block-compressed family) keeps its image out of host-transfer memory, so its CPU copies take submissions where another driver's take none. The stored texels are Metal's either way; what changes is the work, in `WorkCounts::command_buffers_begun_batch_work` and the `command_buffers_begun` and `command_buffers_submitted` that carry it (`bc1-rgba`, four 16 x 16 `BC1_RGBA` levels: `command_buffers_begun` 8 on the bind-time path and 15 off it; `fresh-texture-bc1-rgba-managed`: 3 and 5). A `replaceRegion` of such a texture stages its bytes for the device's next submission and waits for none (`texture/first_move.rs`). Its CPU **reads** come out of a host shadow of the image's texels, made at the first read of the texture by the submission that read waits for either way and read from by every read after it: exact (it holds the image's own bytes, and a CPU write writes both), bounded (`host_shadow::SHADOW_BYTES`, past which a read takes the submission path), and lazy, so a texture nobody reads back holds nothing. It is dropped wherever anything but a CPU write may have written the texels: every commit of the program's, its own Vulkan submissions, another `VkImage`, a detach, or the program taking the image's handle (`texture/host_shadow.rs`; `WorkCounts::texel_shadows_made`, `texel_shadow_reads`, `texel_shadows_dropped`). Measured on an M3 Max: `compressed-moves-bc1-rgba` reads `internal_submissions` 5 where it read 56 and `command_buffers_begun` 30 where it read 81, for 1,168 bytes of shadow over its five textures and 56 reads out of them. | MoltenVK 1.4.2 binds such an image through a `flushToDevice` that `replaceRegion:`s every subresource at one block's bytes a row instead of the row of blocks, tripping AGX's `bytes_per_row >= used_bytes_per_row` assertion (`RuntimeDriverDefect::HostTransferImagesCopiedAtBind`). The copy is the driver's own and no call of the program's asks for it, so the only way not to provoke it is not to ask for the image; every other driver keeps the host-transfer path, as does a device without `VK_EXT_host_image_copy`. A write cannot leave the GPU's copy of the texels unwritten, so only reads come off the host shadow. | None (cost only, and exact) |
| On a GPU without unified memory whose types for host-transfer images lie in a heap with no room left, a `Shared` or `Managed` texture made then takes no host-transfer usage: its image lies in device-local memory where that image has room, and its CPU copies take submissions where those of a texture made while the heap had room take none (`memory::host_image_types`, `texture/host_copy.rs`). An RTX 3090 without resizable BAR (NVIDIA 620.70.06) places such images in its 246 MiB BAR heap alone, which every process's devices share, beside a device-local heap of 24 GiB; a color texture there copies on NVIDIA's staging queue where its image has both transfer usages (`texture/staging_copy.rs`), so this reaches its depth and stencil textures and the color textures it does not stage. The texels are Metal's either way. | The heap's size is the device's (resizable BAR is the platform's setting). The other memory with room is system memory, where the GPU would reach the image across the bus for as long as it lives (`device-memory-overflow`) to spare the CPU copies their submissions. | None (cost only, and exact) |
| On a driver that does not take a set-0 buffer binding whose entry point uses it through several variables as a **dynamic** descriptor, a draw or dispatch of such a variant that binds it at another offset than its predecessor did allocates and writes a descriptor set of its own, where a dynamic binding would let it bind its predecessor's set again at its own offset. That saving needs the binding's range to stay as its offset moves (`bind::instantiate::descriptor_range`): a slot read to its buffer's end has a range that moves with its offset, so its commands write a set each in either form. Such a binding is a slot reached through its word view and a narrow or `uvec4` unit view (`memory::views`), as `record_stride_not_a_multiple_of_four`'s `recs` is. The stored values are the same either way; what changes is the set a command allocates and writes (`WorkCounts::set_zero_sets_reused` counts none of them). The device answers for itself, by creating one pipeline of exactly that shape and destroying it (`pipeline::objects::takes_multi_variable_dynamic`, `bind::DynamicViewsModule`, asked once and only where a layout has such a binding), so a driver without the defect pays nothing. A binding the device takes still turns dynamic only while the layout's dynamic-buffer budget lasts (`maxDescriptorSetUniformBuffersDynamic`, `maxDescriptorSetStorageBuffersDynamic`), spent in binding order: on a Radeon 780M (RADV, Mesa 26.2.4, eight dynamic storage buffers) a layout's storage bindings past the eighth get none, so commands that bind them at moving offsets keep a set each wherever set 0 is not pushed. A device whose set 0 is pushed (`VK_KHR_push_descriptor` with an eligible layout) never reaches the choice. | Nothing in Vulkan bounds the variables at a binding, dynamic or not, so there is no exact form to switch to: the driver either compiles the shape or does not. MoltenVK's SPIRV-Cross declares one variable's local alone -- read in 1.4.1's source and measured on 1.4.2 (`driverVersion` 10402) on an Apple M3 Max, where `vkCreateComputePipelines` answers `VK_ERROR_INITIALIZATION_FAILED` for a kernel of three and of four variables at one dynamic binding. The check proves the driver compiles the shape, not that it applies the dynamic offset to every variable at it; no device is measured to get that wrong, and RADV (Mesa 26.2.4, Radeon 780M, at offset 0) and lavapipe (Mesa, arm64, at moving offsets) are measured to get it right, `record_stride_not_a_multiple_of_four` answering Metal's `sizeof` and both its narrow stores with its three-variable binding dynamic. | None (cost only, and exact) |

## Accuracy switches whose default is Metal's

- `render_lowering = RenderLoweringPolicy::AtCall` (the default) lowers each render encoder call
  inside the call, which returns every refusal. `render_lowering = RenderLoweringPolicy::Worker`
  logs the call once its checks at the call pass, and lowers it later: in batches of 32 calls on
  the device's lowering thread while the program goes on (a program 16 batches or 4 ends ahead
  waits), the encoder's end there too, and what that thread has not taken on the program's thread
  at the next call that follows the end in its command buffer. `commit` returns after its checks
  at the call, its own work run after the end on the thread that lowers it, and later commits of
  the device behind it. A refusal only the lowering finds then fails the command buffer, naming
  the call (`CommandBufferError::CallRefused`), rather than returning from it (docs/API.md section
  1). The runtime's choice: no pipeline differs.
- `unshared_isolation = UnsharedIsolationPolicy::Bounded` (the default) holds other command
  queues' batches back from a `VkQueue` that a queue made with `Device::new_command_queue_unshared`
  shares with them (`unshared_hold.rs`; a device with one graphics queue: RADV, ANV, lavapipe):
  another queue's batch goes there only while less than 2 ms of their expected GPU work is in
  flight on it, its queue's expected run being the host's view of its recent batches' runs, raised
  at once by a longer one and halved toward a shorter one; a batch of a queue with no run seen yet
  holds the others back until it completes. So a compositor's command buffer waits behind about
  one batch of an application's, as Metal runs it beside the application's work, and the next
  batch goes in before the one ahead ends. `unshared_isolation = UnsharedIsolationPolicy::Off`
  submits every ready batch at once: peak rate with no hold, the unshared queue's command buffer
  behind every batch submitted before it. Not an accuracy choice: it trades the other queues'
  submission ahead for the unshared queue's wait. The runtime's choice: no pipeline differs.
  Measured with the runtime's `queue_share` example (`frames --unshared`: an application keeping
  three command buffers in flight beside a compositor committing one draw every 16 ms), `Off`
  against `Bounded`. On the Iris Xe (ANV, Mesa 26.2.6, a quiet host; m3-max-16880, -16885), with
  30 ms command buffers, the compositor's median wait fell from 75 to 17.5 ms and the application
  ran 33.2-33.5 command buffers a second under both (the GPU idle 0.12% against 0.15-0.16%); with
  5 ms ones, from 12.6 to 4.7 ms, at 196.4-198.6 a second under both. On the Radeon 780M (RADV, Mesa 26.2.6;
  m3-max-16900), with 30 ms ones, the median fell from 71-84 to 18 ms and the worst from 94-114 to
  35-40 ms, about one command buffer. The cost: no throughput measurable on a quiet host, but the
  hold leaves 2 ms of slack where `Off` keeps two batches queued, so under the host's CPU load the
  GPU waits for the scheduler's thread more often: on the Iris at a load of about 10, with 5 ms
  command buffers, 181.8 a second against 189.3 (the GPU idle 8.9% against 4.5%; m3-max-16885),
  and on the 780M 192.4 against 203.7 (m3-max-16900). The 780M's rate otherwise follows its
  command buffers' own GPU time, which moved 4.1-4.8 ms between runs of either value on a host
  other jobs loaded. The default on every driver with one graphics queue.
- `pointer_domain`, `handle_domain`, `containment` and `isolation` choose how resources resolve and
  how guests are contained, not an answer of Metal's arithmetic or sampling.

## Costs of matching Metal

Where matching what Metal observably does costs time, the runtime matches it and the cost is
recorded here.

- **`commit` returns at once.** Metal's `commit` returns before its command buffer is scheduled.
  The runtime's commit hands the batches to the device's scheduler thread, which submits them
  (`scheduler/inbox.rs`), so no driver call holds `commit`. Where that thread sleeps, a commit's
  batches wait for its wake: three command buffers committed back to back and waited for take a
  median 90 to 160 µs longer on ANV (Iris Xe), 5 µs on the RTX 5080 Laptop GPU and none on the M3
  Max. From idle it is longer: on the RTX 5080 Laptop GPU (NVIDIA 615.71.09), with one fill
  committed every 2 ms, a commit's batch was watched 140 to 165 µs after the call at the median
  against 5 µs for `vkQueueSubmit` itself, and a parked thread's wake took 80 to 105 µs at the
  median on that host (the queue waiters', `tests::completion_latency`). On the Iris Xe, frames of
  24 command buffers committed back to back (`runtime_lane::commit_costs`, `frames`) take a median
  5.7 to 6.0 ms against 3.2 to 4.5 ms with the submissions on the committing thread, the GT held at
  its efficient 300 MHz by its governor (GuC SLPC). A submission check or submission handler,
  which run on that thread, holds back every other command buffer's submission while it runs.
  `CommandBuffer::commit_and_submit`, a runtime extension, is the inline form: it submits on the
  committing thread where no other pass is under way, and runs the submission checks and
  submission handlers of what it submits there.
- **A present that samples a guest plane copies the whole plane in, and Metal requires it.** Where
  a window's sink draws a guest's 3840x2160 plane into its swapchain texture, it does so in its own
  command buffer, and Metal shows a command buffer any CPU write made since its commit -- probed on
  an Apple M3 Max, a write made while an earlier command buffer ran reached the next, already
  committed one. So the copy in is required rather than redundant, and carrying a shadow's standing
  across the encoders of one command buffer, which is sound, does not reach it. The sink's texture
  is also a different `MTLTexture` over the same plane, so it takes its own shadow image and its own
  first copy in, while the guest's own pass into that plane pays only its damage rectangle. That is
  8,294,400 texels per present. Neither route out is open: aliasing the plane as a linear image
  needs a row pitch the guest chooses and whose memory stays in place, and refreshing only a damaged
  rectangle needs a rectangle no present carries -- a surface present hands over a whole surface and
  nothing more -- while recovering the damage by comparing frames is the copy skipping that is ruled
  out.

- **A two-plane YCbCr texture is read in place on an integrated GPU and converted on a discrete
  one.** Metal reads a `YCBCR8_420_2P` or `YCBCR10_420_2P` texture's texels from its surface's
  planes. On a device whose memory is the host's (Vulkan's `deviceType` an integrated GPU or a CPU),
  a function that only samples such a texture (nearest or linear, no comparison, anisotropy or
  `min_lod_clamp`), gathers it or reads its size reads the planes' codes in place, through a uniform texel
  buffer over their bytes, and converts and filters them in the shader (a linear sample by
  Vulkan's model, or as Metal does under `YCbCrInPlacePolicy::Exact`: the fast-defaults row); every other
  device, and every other use (a `read()`), converts the texture's image from the planes
  before the command buffer's first use (the refresh compares every word first) and reads the
  image, which holds Metal's texels exactly, or the device's own Y′CbCr conversion of them where
  it converts the format (the device-gaps row), and whose linear sample is the device's filter
  (`device-sampling`; on MoltenVK `ycbcr-filter-not-exact`). Neither is a switch: the device's kind
  picks the cheaper. GPU time of a 3840 x 2160 texture whose planes lie in host memory composited
  into 1680 x 1050 (µs, best of 3 x 40, unchanged planes; Mesa 26.2.4, MoltenVK 1.4.2):

  | Device | Codes | In place (Metal's filter), 1 pass | Image, 1 pass | In place (Metal's filter), 3 passes | Image, 3 passes |
  |---|---|---|---|---|---|
  | Radeon 780M (RADV) | 8-bit | 340 | 1178 | 1027 | 3013 |
  | Radeon 780M (RADV) | 10-bit | 457 | 1509 | 1371 | 3344 |
  | Radeon RX 7900 XTX (RADV) | 8-bit | 1819 | 1754 | 5460 | 1876 |
  | Radeon RX 7900 XTX (RADV) | 10-bit | 3614 | 3479 | 10843 | 3613 |
  | Apple M3 Max (MoltenVK) | 8-bit | 127 | 259 | 375 | 634 |
  | Apple M3 Max (MoltenVK) | 10-bit | 125 | 364 | 378 | 737 |

  A discrete GPU reads the host's planes across its bus at every pass, where its image lies in its
  own memory; changed planes add the conversion of every texel to the image's form (1055 to 1252
  µs on the 780M, 1790 to 3636 on the 7900 XTX, 350 on the M3 Max). Where the device converts the
  planes' 4:2:0 format as a linear image with Metal's reconstruction and the planes lie as that
  image's would, the in-place form samples them through the device's conversion (the device-gaps
  row); elsewhere it converts as Metal does. A texture whose planes' first
  byte misses `minTexelBufferOffsetAlignment`, whose planes exceed `maxTexelBufferElements` or reach
  past their buffer, or which has a swizzle keeps the image on every device
  (`WorkCounts::ycbcr_images_kept_*`).
- **Idle dma-buf imports are released.** RADV lists every imported memory in every kernel
  submission, so a no-copy buffer's dma-buf import that only blits have read is released after its
  idle window (2 s, doubled at each import again up to 64 s). The encoding thread pays its import
  again at the first blit after the window (HOWTO.md section 6). A texture in place over such a
  buffer lies in an import of its own (`dma_buf.rs`) and leaves the buffer's import to this
  release; a blit of the texture runs on the buffer's bytes, so it imports the buffer again as any
  blit does. On the Radeon 780M (`runtime_lane::submit_per_surface`, three runs), 128 surfaces of
  64 KiB, each a buffer and a texture in place over it, list 256 dma-bufs in each kernel
  submission (115 to 129 µs in the ioctl) while their buffers are in use, and 128 once the buffers
  sat idle past their window (58 to 67 µs); on the RX 7900 XTX 108 to 109 µs and 59 to 70 µs. For
  the same reason a guest RAM
  there (`Device::use_host_ram`, with a source of dma-bufs) is imported in chunks longer than 64
  MiB, as long as one dma-buf of the source holds (`HostRam::with_largest_dma_buf`) within the
  device's allocation, buffer and storage-range limits, none ahead of the objects. Each live chunk
  costs every submission about half a microsecond (an 8 GiB RAM in 2 MiB pages on the Radeon 780M,
  `runtime_lane::submit_per_import::submit_per_ram_chunk_length`: a submission's median 14 to 15 µs
  with none, 59 to 91 µs with 128 chunks of 64 MiB, 36 to 50 µs with 64 of 128 MiB, 32 to 33 µs with
  32 of 256 MiB, 24 µs with 16 of 512 MiB, 19 µs with 8 of 1 GiB). The chunks stop at 512 MiB,
  trading that speed for memory, not accuracy (no switch): a chunk's import pins every page of it,
  udmabuf allocating the pages the guest never touched, which stay allocated in the guest's file
  after the chunk is freed. In that probe, one 2 MiB object in each chunk left all 8 GiB resident at
  every length; a 16 GiB guest in chunks of about 2.5 GiB had 4.4 GiB more of its RAM resident than
  without the source. The same guest's QEMU had 0.45 GiB more resident at its desktop in 512 MiB
  chunks, with at most 11 to 12 live, and 0.94 GiB more in 256 MiB chunks, with at most 15 to 18
  (0.15 and 0.25 GiB more at the end of its run; the means of two boots each). A 16 GiB RAM in 512
  MiB chunks lists 32 at most. The first submission that lists a RAM's chunks took 52 to 56 ms for
  the 8 GiB at every length. A chunk, once imported, stays imported where the whole RAM fits in half
  the device's memory heaps with unified memory, or in three quarters of those that are not
  device-local without: the first kernel submission after a chunk's import waits while the kernel
  writes its page tables, and every other submission of the device with it (231 to 462 ms for a 3.94
  GiB chunk on the Radeon 780M: a gather after the idle window took 475 to 480 ms where its chunk
  was released and imported again, 2.6 to 2.8 ms where it was kept). Where the RAM does not fit, a
  chunk no buffer nor pending command buffer holds is released after the same window
  (`WorkCounts::host_ram_chunks_released`), as RADV lists every import in every kernel submission,
  which fails once they pass the kernel's GTT. The first object over a chunk not imported asks a
  thread of the device's for it, which imports it off the object's thread (it pins every page of the
  chunk: 660 to 692 ms for 2.7 GiB on the Radeon 780M and the RX 7900 XTX), and the objects over it
  meanwhile are each bound in an import of its own run, one more listed import apiece: a buffer's is
  released after its idle window and the buffer bound in the chunk at its next blit, but a heap's,
  or a buffer's whose address was handed out, stays while it lives. A copy by runs over a chunk not
  imported, of a RAM with a source of run lists (`HostRam::with_dma_buf_of_runs`, the consumer's
  choice), imports lists of its own pages instead, kept past its command buffer for the copies of
  the same pages after it (at most 64 MiB and 16 lists, each released once no copy took it for
  2 s), and not the chunk: so the copies make none of the guest's untouched RAM resident, and a
  copy whose pages no kept list holds pays its lists' imports (on the Radeon 780M a tile of 1,800
  pages laid as a surface's took 3.9 to 4.3 ms from the commit to the completion where it imported
  its list, 0.53 to 0.60 ms where it took a kept one, 0.37 to 0.40 ms over an imported chunk; a
  guest's scroll of such tiles over half its RAM left 4028 MiB of an 8 GiB guest resident, where
  whole chunks would have made 8192 MiB, and 8300 MiB of 16384 for a 16 GiB guest).
- **A pass in color groups runs once per group** (the `ColorAttachmentGroups` row above), the later
  groups' draws again over a twin of the depth and stencil, made for the pass and loaded by a copy
  or a clear (`render/color_groups.rs`). On lavapipe restricted to four colors
  (`runtime_lane::color_groups::color_group_costs`), 16 fullscreen draws into five 512²
  `RGBA16Float` colors took 1.04 times the same draws into four from commit to completion, and 1.51
  times with a loaded `Depth32Float_Stencil8` the draws write.
- **A draw that reads its own pass's attachment reads a copy, a snapshot or a split.** A sampled
  slot binding of a texture the render pass instance renders reads a copy taken before the
  instance (`texture/attached_reads.rs`): the texels around the draw's scissors where it reads at
  its own pixels alone, else every texel or what the instances since drew; at a later command
  buffer's first copy, where only the runtime's commands write the texture (a tracked `Private`
  one's image, or a tracked `Private` buffer's bytes), only what changed since the last command
  buffer left the copy standing: nothing, or the texels inside the scissors of the draws of a
  pass that loads it (`shadow_standing.rs`, `WorkCounts::shadow_texels_unskipped_stale_within`).
  The copy stays where the device has the feedback loop too: read in place, the draw would race
  its own and earlier draws' writes of the attachment, a data race Vulkan forbids; none where every read
  of the slot reaches only levels before the lowest one the instance renders, which it reads in
  the texture's own image through a view of those levels (`render/attached_levels.rs`,
  `WorkCounts::attached_read_slots_narrowed`), or only levels past the highest one, through a view
  of the levels up to the highest it reads (`attached_read_slots_past`). After a barrier inside
  the instance, an own-pixel read takes a snapshot drawn in its subpass (`render/snapshot.rs`,
  docs/API.md "Snapshot reads") or splits the instance; both are exact. A private texture of the
  runtime's own whose image lacks `STORAGE` (the device lays it out otherwise with it) splits at
  its first such read, as a later fetch would render it through a stand-in its snapshot does not
  see, and takes its storage form at a later commit (`texture/storage_form.rs`,
  `WorkCounts::storage_forms_taken`, `fetched-target-storage`): its later passes' readers take the
  snapshot. Over a texture over its
  buffer's bytes (a linear image over an ordinary buffer's packed rows or over host pages, or the
  shadow a texture over host pages that renders is where it is one) it takes the snapshot at any
  area, where the device makes that image with feedback loop usage
  (`WorkCounts::feedback_loop_alias_images`, `feedback_loop_shadow_images`; NVIDIA's drivers make
  the linear one without, and such a texture's readers split there), one over host pages with the
  storage usage too, which it takes where the device's check passed for a shader's writes there
  (docs/API.md): on the Iris Xe (`runtime_lane::attachment_read_costs::snapshot_crossover`,
  a 1920x1080 surface) 0.56 to 0.99 times the split's GPU time through scissors of 960x540 and up,
  at most 1.08 times through the whole target and 1.2 times through a 320x180 scissor it covers
  half of or more.
  Elsewhere the area rule (`render/snapshot.rs`) weighs each form at the share of its scissors least
  favourable to it, as a draw's recording cannot know what it covers: the snapshot draw at 20 bytes
  of the split's copy for each pixel its scissors bound, the split at its copy (a one-level,
  one-slice color's texels around the draw's scissors, a pixel wider each way, as the copy is cut;
  else every texel of the level), each beside a reader's fixed cost of a mebibyte of the copy; it
  takes the form whose cost there is the smaller multiple of the other's. A 4-byte color's reader
  takes the snapshot through scissors whose rectangle holds about 1024x768 pixels or more, and
  splits through smaller ones. On a device that measures readers (`occlusionQueryPrecise`, queries
  reset on the host: the Iris Xe, the Radeon 780M and the RX 7900 XTX among those that take the
  snapshot path), each reader the rule weighs records its drawing commands inside a precise
  occlusion query, and from the pass after one completed it is weighed at the pixels it covered
  there instead (`render/coverage.rs`): the snapshot where 20 bytes of the copy for each of them,
  for each color, cost no more than the copy. A reader whose coverage changes takes the form its
  last count chose until a pass after the change completes; both forms are exact. The query costs
  about a microsecond a reader on the Iris Xe; on the Radeon 780M passes that measured took 1.00 to
  1.01 times those that did not, on the RX 7900 XTX 1.00 to 1.08 times (its times for one build vary
  up to 1.4 times between runs). Through no scissor over private `BGRA8Unorm` targets of 1280x720 to
  3662x2039 (`runtime_lane::attachment_read_costs::snapshot_area_costs`, the split on a device that
  withholds `fragmentShaderPixelInterlock`), a reader covering 1.5% to 12.5% of the target took 0.09
  to 0.63 times the split's GPU time on the Iris Xe, 0.10 to 0.67 on the Radeon 780M and 0.26 to
  0.65 on the RX 7900 XTX; one covering half of it or all of it takes the split from its second
  pass, 0.99 to 1.01 times, 0.95 to 1.03 and 0.83 to 0.91. Over a private 1920x1080 target
  (`runtime_lane::attachment_read_costs::snapshot_crossover`, against the build before the area
  rule), readers covering 1.5% or 12.5% of a scissor of 960x540 or more take the snapshot, 0.37 to
  0.85 times the GPU time on the Iris Xe, 0.31 to 1.26 on the Radeon 780M (1.15 to 1.26 for one over
  12.5% of a scissor its pass filled whole, as under the area rule alone) and 0.68 to 1.05 on the RX
  7900 XTX, and those covering half of it or all of it the split, 1.00 to 1.03 times, 1.00 and 0.74
  to 1.02; through a 320x180 scissor, 0.75 to 0.80 times covering half or all, and 1.05 to 1.07
  times covering 1.5% or 12.5%, where both builds take the snapshot and the query is the difference
  (on the Radeon 780M 0.55 to 0.71 and 1.04, on the RX 7900 XTX 0.90 to 0.91 and 1.04 to 1.05).
  Other devices are not measured.
- **A texture over a no-copy buffer's host pages, or over a host-visible buffer longer than 32
  MiB, that a shader writes of texels of a width the check stores none of (none of 1, 2, 4, 8 and
  16 bytes; or whose class's check has not passed for an access of its, or whose format, usage or
  pitch no image takes, or under
  `HostPageTexturePolicy::Shadow`), and a texture over a buffer of the runtime's memory whose rows
  lie with bytes between them,** is a shadow (`shadowed-texture`, its reason named): its texels
  copied in before each use and merged back after each write (only those inside the draws'
  scissors for a pass that takes it as a loaded attachment alone). One over host pages that
  samples, reads, renders or is written is an image in place where its class's check passed for
  each of its accesses (the row above, `in_place.rs`). An image bound to host pages starts
  `UNDEFINED` (`VUID-VkImageCreateInfo-pNext-01443`), and its first move may read and write every
  byte bound to it, which the host could keep only by undoing a CPU write made meanwhile, where
  Metal keeps it (the device's check is what takes one in place); an image over padded rows holds
  the bytes between them, which a render pass's load and store operations, and a driver's storage
  writes wider than a texel, may rewrite too (the checks of an image that renders and of one a
  shader writes cover them). The shadows
  over host pages are counted by why (`WorkCounts::host_texture_shadows_*`; a first session's
  textures made before the check answered, `host_texture_shadows_check_pending`). A host-visible
  buffer
  longer than half a pooled block (32 MiB) lies in fresh anonymous pages imported by their pointer
  where the device imports them, its submission checks found that a held pointer import adds nothing
  to every submission's cost (`submit_check.rs`'s import tax: MoltenVK and ANV, not RADV, where one
  such 64 MiB buffer raised a commit's p50 from 10.7 to 36.6 us on the Radeon 780M), and the memory
  class's best type for the buffer takes such an import (`fresh_pages.rs`; on a discrete card the
  write-combined class's best type is a device-local host-visible one that a host-pointer import
  cannot bind, so that buffer is placed as ever and cleared on the host), which read zero as mapped,
  so its creation writes none of its bytes, and the CPU's first touch of its pages is paid where the
  program makes it, as on Metal (host_ab on the M3 Max, 64 MiB: made in 0.01-2.2 ms against 3.2-4.7
  ms of allocation and memset, its first CPU write 2.4-6.5 ms against 0.7-1.4 ms, Metal's 3.7-7.4
  ms): such a buffer's textures that render or write are shadows for the same reason
  (`WorkCounts::shadows_over_fresh_pages`).
  On the UHD 770 (ANV) a 1080p `BGRA8Unorm` surface's pass
  that loads, draws and stores (`guest_frame_costs`) took 1.9 ms from commit to completion, 1.4 ms
  of it the copies (copy in 0.21 ms, keep 0.60, merge 0.60), against 0.4 ms into a `Private`
  texture; a linear image over the pages took 2.6-3.1 ms, its draws slow in host memory. Twelve such
  passes in one command buffer took 21 ms against the linear image's 27.5 ms, and 0.4-0.5 ms more of
  the process's CPU a frame (1.2 ms against 0.7): about 20 µs a pass on the encoding thread, 10 µs
  on the queue's waiter and 10 µs on the recorder and reaper together. On ANV making a 1307x512 tile
  took about 135 µs as a shadow against 41 µs as an image; a frame sampling four `BGRA8Unorm` tiles
  over no-copy pages (`surface_texture_costs::read_only_padded_costs`) took 3.2-5.3 ms with the
  textures made once and 3.4-7.1 ms with new ones each frame, where images over the rows took
  3.0-6.1 and 4.6-8.6 ms. An ordinary buffer's packed rows avoid it.
- **A view of another format without `PixelFormatView` usage** is a shadow image copied at each use
  (`shadowed-texture-view`): about 50 to 95 µs a command buffer for a 64x64 texture. Making the
  texture with `PixelFormatView` usage avoids it. A view in other channels of its image's bits (an
  `A8Unorm` view of an `R8Unorm` texture) binds the image itself through a swizzle, and only a pass
  that renders into it renders through a shadow.
- **A pass that loads a texture over a buffer and does not store it** renders through an image its
  texture keeps, the buffer's texels copied in first (`dont-care-stand-in`), so the buffer keeps its
  bytes as Metal's does: about 10 µs a pass on the Iris Xe (ANV). A pass that clears and stores
  `DontCare` costs no more, and one that stores renders in place.
- **The interlocked pass on the M4.** Every device splits a pass the pixel interlock would split,
  about a quarter faster than kept whole on the M3 Max (`DriverFact::InterlockedPassSplitNoSlower`);
  the M4 ran that pass about 8% faster kept whole.
- **A fetching draw in the pixel interlock waits for the draw before it.** Between two draws of a
  render pass instance where the later one reaches a color in the fetch's interlocked storage form
  (or an extended-range color's `half` shadow) that the earlier one wrote, the runtime records a
  barrier from the fragment shader's writes (`render_target::between_storage_draws`). Vulkan orders
  such draws by the interlock alone, but RADV on the Radeon 780M does not. On the Iris Xe the
  barrier costs about 15,000 GPU cycles a draw, 9 to 20 µs at the clocks measured: a 1920x1080
  pass of 27 such draws took 1.46 against 1.08 million cycles without it
  (`runtime_lane::tile_path_costs`, `wide blended in the function`). Not on a driver measured to
  order them (`DriverFact::InterlockOrdersDraws`: ANV on the Iris Xe;
  `runtime_lane::interlocked_draw_order`). The imageblock's storage and a depth stand-in keep
  theirs on every device. On MoltenVK on Apple's GPUs, which drops the barrier and whose such
  draws race under load (`DriverFact::InterlockedDrawsRaceUnderLoad`), the later draw begins a
  render pass instance of its own instead (a known gap below).
- **A heap texture Metal keeps uncompressed is a shadow of Metal's tiles where no image in place
  lays its elements there, under `heap_answers = HeapAnswerPolicy::Metal`.** Under the default
  `HeapAnswerPolicy::Device` such a texture answers its device's image and lies in it (the Fast
  defaults row), with no shadow, wherever that image's alignment is within the row's bounds; what
  follows is the switch's, and that of a texture past the bounds whose 4 KiB image does not fit
  either. Metal lays such a texture's bytes out in its tiles
  (`texture/heap_tiled.rs`). Where the device's image of it takes more of the heap than Metal's
  answer, an image that fits takes its place in the heap's bytes where Metal compresses the
  texture (its bytes are no layout of its texels) or its layout is not modelled
  (`heap-texture-host-transfer`, `heap-texture-linear`); of a texture Metal keeps uncompressed,
  only a linear image whose every element lies where Metal's tiles place it does, announced with
  no layout (a level of one element or one row: a 1x1 texture, a one-level 1D one;
  `texture/heap.rs`, `rows_lie_in_metal_tiles`). Every other such texture is a shadow of Metal's
  tiles (`shadowed-texture`). On ANV (Iris Xe of RPL-P, Mesa 26.2.6), of the shapes a guest's
  compositor made in its heaps (`runtime_lane::heap_shadow_costs`), those are the `RGBA16Float`
  arrays of 3x3 to 10x10; the ones Metal compresses and the 1x1 lie in place. A 1x1 one
  measured as a shadow there took 35.9 µs of GPU time a frame that clears it in a pass and
  blits it, against 8.1 µs in place and 12.1 µs outside a heap, and 16.9 µs a frame that only
  blits it, against 3.7 and 3.9 (`heap_shadow_costs`); the shapes from 3x3 to 10x10 are not
  measured apart.
- **A shadow over bytes another writer may write is merged back, not copied back.** Metal writes
  only the texels a command writes, and another writer writing bytes no command touches while a
  command buffer runs is valid under its contract. On an M3 Max under macOS 27.0.1, a CPU write to
  a texel no command touched, made 6 to 23 ms into a kernel's 58 to 68 ms run that wrote another
  texel of a texture over a `Shared` or `Managed` buffer, stayed every time; so did another
  queue's fill of such a texel of an untracked `Private` buffer, run inside a kernel's or a render
  pass's 61 ms run (5 times of 6; the sixth ran after it). Another queue's fill of a tracked
  buffer ran after the kernel or the pass ended, every time, as the runtime's tracker orders it.
  So a shadow over the bytes of a `Shared` or `Managed` buffer, host pages, the caller's
  `VkBuffer` or an untracked buffer is kept after its copy in, and the bytes the commands changed
  are merged back in place of the copy back (`shadow.rs`, `shadow_merge.rs`): two copies of the
  texture's image into a buffer and a merge, in place of one copy back, each cut to the rectangle
  the commands write in where they bound it. A tracked `Private` buffer's, which no other writer
  writes while the command buffer runs, is copied back. Where the copy back is a dispatch
  (`DriverFact::ShadowCopyBackDispatchFaster`: ANV, whose `vkCmdCopyImageToBuffer` is slow), so is
  the first copy, and the merge reads the texels the commands left from the image through an
  unsigned-integer view in place of the second. Where the copy in and the first copy are also one
  dispatch (`DriverFact::ShadowCopyInDispatchFaster`, measured faster on the UHD 770 of RPL-S
  alone), that dispatch reads the texels from the buffer once and stores them in the image and in
  the scratch: on the UHD 770 a 1920x1080 pass that loads took 1.43 ms of GPU time against
  1.64-1.67 ms, twelve 17.2 ms against 20.0 ms, a clearing pass 0.89 against 1.04 ms; on the Iris
  Xe of RPL-P twelve took a median 5% longer, so the Iris keeps the two. Where
  the merge reads the image and the copy in takes no first copy (the Iris), the first copy is a
  copy of the image's texels into a second image at the same coordinates (`vkCmdCopyImage`), which
  the merge reads beside it through the same view: the same bits, as a copy between formats of
  one compatibility class moves them. On the Iris Xe a 1920x1080 `BGRA8Unorm` image copied into
  another in 67-73 µs of GPU time, against 236-253 µs into a buffer and 270 µs for the dispatch
  it replaces; over a 1920x1080 surface of anonymous pages (`guest_frame_costs`, three rounds),
  twelve passes in as many command buffers took 7.2-8.2 ms of GPU time a frame against 9.2 ms,
  twelve in one 10.4-12.9 against 14.5-16.2 ms, and one 0.76-0.83 against 0.92-1.42 ms. That
  image takes the scratch buffer's place, 8.3 MB at 1920x1080, and is its queue's: a later keep of
  the queue of the same view and extent, whatever its texture, takes it once the encoder that
  took it ended (or the run of passes it began, below), without waiting for the command buffers
  that used it, which the merge's closing barrier and the image's tracked uses order
  (`texture/kept_image.rs`); at most two that no open encoder holds stay for each view and
  extent, and none untaken over 256 takes. On the Iris Xe
  (`surface_texture_costs::surface_texture_cost_parts`, four shadowed textures a frame, 1,582,016
  texels), over anonymous pages, merged back, passes that draw took 5.0-5.4 ms a frame from commit
  to completion, and under scissors (790,448 texels) 3.6-3.9 ms; over tracked `Private` buffers,
  copied back, 3.1 ms and 2.2-2.4 ms. Render passes one after another in one batch over one
  such texture, each loading and storing it as a color, with nothing between them that may touch
  its bytes, are one run (`pass_runs.rs`): the first pass is kept before and merged after, in
  place, as a pass alone is, and the texture is kept again before the second and merged after
  the last, so a pass that no other follows pays for no run. 700 such passes in one command
  buffer, each blending one rectangle over a 1024x1024 `BGRA8Unorm` texture over host pages with
  rows of 15360 bytes, took 5 shadow copies against 1,401 and kept 2,097,152 texels against
  734,003,200; from commit to completion, 32.7-35.6 ms against 69.3-99.9 ms on the M3 Max
  (MoltenVK 1.4.2) and 6.5-9.1 ms against 40.6-43.0 ms on the RTX 5080 Laptop GPU (NVIDIA
  615.71.09); with a 1x1 render encoder of no attachment between every two, 61.5-67.2 ms against
  107.4-118.6 ms and 9.3-12.2 ms against 44.5-47.1 ms (the second to fourth of four such command
  buffers in a process).
- **A `Managed` no-copy buffer is two copies on a device without unified memory**
  (`managed_copy.rs`, `HostBacking::DeviceCopy`), as Metal keeps a managed resource on a Mac with
  a discrete GPU: device-local memory of the buffer's size beside the caller's pages
  (`WorkCounts::managed_device_copy_bytes`), and copies across the bus, as GPU transfers from the
  pages' import, else through staging buffers of at most 32 MiB, 256 MiB staged at most before the
  call submits them and waits: the pages whole at the making and at each rebind, a
  `did_modify_range`'s range before the next submission, the buffer whole at each
  `synchronize_resource` of it (of a texture over it, the texels' bytes)
  (`WorkCounts::managed_bytes_copied_in`, `managed_bytes_copied_out`, `managed_staging_bytes`). A
  kernel reading one 256 MiB such buffer and writing another (`examples/managed_bandwidth.rs`,
  release) ran at 864.6-866.8 GB/s on the RX 7900 XTX (RADV), where the buffer lay in its pages
  before at 11.3-11.4 GB/s, and at 395.6-396.0 GB/s on the RTX 2080 (NVIDIA's driver), 2.7 GB/s
  before; a `Shared` no-copy buffer, which stays in its pages, ran at 11.3-11.4 and 2.7 GB/s both
  times, and a `Private` one at 865.0-867.3 and 395.6-396.0 GB/s.
- **A command buffer's copies in and new buffers' fills come after the work before it**, as Metal's
  copy of a `Managed` buffer's modified range and a new buffer's contents come before the commands
  that take them and after those before. On a device whose family of transfers alone has two queues
  or more (NVIDIA's copy engine), a batch's fills of 1 MiB or more in all run on that family's
  second queue, in a submission of their own after the batches before it, which the batch waits for
  (`device/staging.rs`, `WorkCounts::fill_queue_submissions`); smaller ones, and every fill
  elsewhere, run at the start of the batch's own work on its queue. Beside another queue's kernels,
  a frame that copied a `Managed` buffer's 1 MiB in took 1.80 to 1.84 ms (p50) through the fill
  queue against 3.53 to 3.60 ms at the batch's start on the RTX 3090 and RTX 2080, and 64 MiB 35.9
  to 37.8 against 40.6 to 49.4 ms (3.31 to 3.39 against 5.90 to 5.92 ms on the RTX 5080 Laptop GPU);
  alone, from 1 MiB up, the fill queue's frames took at most 1.04 times as long. That 1.04 is the
  RTX 3090's at 1 MiB in two runs of three (0.93 in the third) and the RTX 2080's at 64 MiB in one
  run of two, 0.06 and 1.3 ms a frame, traded for the halved frame beside the kernels: the swept
  size from which the fill queue was within 1% alone in every run is 256 MiB, and the fills under it
  would lose that gain. Under 1 MiB the fill queue's frames took 0.51 to 2.03 times as long as the
  batch's start's, where two devices that both copied at the batch's start differed by 0.31 to 1.85
  times (`tests::managed_copies_bench`): no gain past that spread, so smaller fills keep the batch's
  start. From 1 MiB up the other queue's kernels' share of their rate moved by -11 to +8 points on
  the RTX 3090 and RTX 2080 and by +1 to +21 on the RTX 5080 Laptop GPU. A batch takes and waits
  on the GPU for the fills of memory its commands may reach alone (every one where a command may
  reach memory no use names), and internal submissions for the fill queue's last, until the host
  saw it complete (`WorkCounts::fill_queue_waits_unused`). On the RTX 3090 a compositor's frame of
  one dispatch, beside an app on another queue that makes two 4 MiB `Private` buffers before each
  of its command buffers, takes 2.10 ms (p50; 2.21 p99), as when the app makes nothing, and none
  of its frames waits for the app's fills; when every batch waited for the fill queue's last
  submission it took 9.0 ms (p50), and 424 to 429 of its frames in 4 s waited for fills they reach
  none of (`tests::fill_waits_bench`).
- **A no-copy buffer's deallocator runs in its release.** On an M3 Max under macOS 27.0.1 an unused
  no-copy buffer's deallocator ran inside its final release, on the releasing thread, every time.
  So a release that frees the buffer's import while a deallocator waits for the free frees it
  there and then runs the deallocator, before returning. The free waits on the driver's locks a
  submission holds (ANV takes its device's lock in each execbuffer and in `vkFreeMemory`). On the
  Iris Xe (`submit_per_import::submit_per_import_release_cost`, 520 host imports live, four
  releases a frame) a free in the release took 115-128 µs (p50) on the releasing thread, 472-691
  µs a frame. A release whose keep-alive (a runtime extension) alone waits for the free goes to a
  handler thread instead, the release returning in 2.7-4.0 µs, 64-145 µs a frame: the consumer's
  guest gives its transfer buffers keep-alives.
- **A `Shared` buffer over the caller's pages is read and written across the bus on a discrete
  GPU**, as a Mac with a discrete GPU reads its `Shared` memory: the GPU reaches the pages through
  their import, or through a copied host buffer's copy in host-visible memory (`copied.rs`), never
  through a copy in its own memory. A kernel that reads one 256 MiB buffer and writes another
  (read plus write) ran at 5.3 GB/s over a `Shared` no-copy buffer and 328 GB/s over a `Private`
  one on the RTX 5080 Laptop GPU (NVIDIA 615.71.09), where a Radeon Pro 5300M under macOS 26.7.1
  ran 6.1 and 96.1.
- **Emulated tessellation's scan.** On a device without `tessellationShader` a patch draw first
  runs a one-workgroup scan that places each patch's triangles in Metal's order, patch after
  patch: about 6-60 ms of GPU time per 10,000-patch draw at factors 4 to 64 on the Radeon 780M and
  the RTX 3090, 0.6-5 ms on lavapipe (`emulated-tessellation-draws`). A parallel form is not built.

## Known gaps

Open differences from Metal that no note announces yet, and runtime costs still above the
driver's. Each is open work.

- **A two-copy buffer over read-only pages keeps the GPU's writes until the command buffer that
  made them completed** (`managed_copy.rs`, `WorkCounts::managed_writes_undone`): its pages are
  copied in whole again at that completion, so the command buffer's later commands, and a
  command buffer submitted before that completion, read the writes, which Metal drops.
- **A `synchronize_resource` of a two-copy buffer encoded before a rebind copies to the pages
  the buffer lay in when it was encoded** (`managed_copy.rs`), not to the new ones.

- **A heap texture Metal keeps uncompressed lies in its driver's layout, not Metal's tiles, where
  its image fits Metal's answer in the heap's memory** (`texture/heap.rs`,
  `heap-texture-untiled-layout`): on every driver whose colour images are not measured to lie as
  Metal's heap textures (`DriverFact::ColorImagesLieAsMetalHeapTextures`: every one but MoltenVK
  on the M3 Max), its optimal image, or on ANV the route a driver that hides aliased writes takes
  (host-transfer usage, a linear image, or the optimal image, `heap-texture-host-transfer`,
  `heap-texture-linear`, `heap-texture-hides-aliased-writes`), is bound in the heap's bytes as
  it is. A buffer or another texture over those bytes then meets its texels where the driver
  lays them, where Metal's tiles place them on Metal (`texture/heap_tiled.rs`). Only where
  that image does not fit Metal's answer does a shadow of Metal's tiles, or a linear image that
  lays every element there, take its place (the costs above).

- **On RADV, the first submission after the GPU idled into runtime suspend waits for the kernel to
  resume it**: where the GPU idles past its autosuspend delay while a program runs (5 s on the RX
  7900 XTX, Mesa 26.2.6), the next submission of any kind, a command buffer on the program's queue
  or a texture's copy on the staging queue, waits for the kernel's resume of the device: 4.73 to
  4.88 s there, where the kernel's log of a resume spent about 3 s on the display firmware's
  errors with no display attached (`Wait for DMUB auto-load failed`, then `DMCUB error` lines).
  A commit returns at once (0.12 to 0.52 ms), its batches handed to the scheduler thread, and its
  wait carries the resume (4.83 to 4.86 s). A `replace_region` into a `Shared` texture also
  returns in its awake time, its copy queued for the next submission and the device left
  suspended. A `get_bytes` as the first call after the suspend, or after such a `replace_region`,
  waits for the resume (4.73 to 4.88 s at 1920 x 1080 and at 4096 x 4096, where a `get_bytes`
  right after the device woke took 3.2 to 4.0 ms and 18.7 to 20.8 ms). The runtime starts no
  wake ahead and holds off no suspend: what first needs the GPU after the delay pays the resume
  (`tests::cpu_copy_times`' measurement after a runtime suspend, run by name).

- **On RADV every live import adds to every kernel submission** (`runtime_lane::submit_per_import`,
  `runtime_lane::submit_per_surface`): RADV lists each imported or exportable `VkDeviceMemory` in
  each kernel submission's list of buffer objects, and the kernel's work per submission is per
  entry. On the Radeon 780M and the RX 7900 XTX (Mesa 26.2.6, Linux 7.3.3, a raw Vulkan probe) a
  `vkQueueSubmit` of one command buffer took 7 to 8 µs (median) beside no import and 229 to 234 µs
  beside 512 dma-buf imports of 1 MiB, 0.43 to 0.44 µs each (64 of 1 MiB: 32 µs; 64 of 16 MiB: 45
  to 80 µs); 64 pointer imports of 1 MiB took 191 to 200 µs. Memory at a low `VK_EXT_memory_priority`
  priority, and memory with no buffer device address, listed as many objects and took the same
  time, and neither device lists `VK_EXT_pageable_device_local_memory`. Submitting command buffers
  in one call makes one kernel submission where Mesa can merge them (no signal between them), so
  each call apart lists every import again. The runtime keeps fewer imports live where nothing of
  Metal's changes (idle imports released, a guest RAM in chunks, above), but a texture in place
  over a dma-buf keeps its image's import, which RADV takes only as a dedicated allocation, while
  the texture lives: on the 780M, 128 such textures listed 128 dma-bufs in each kernel submission
  with their buffers released (58 to 67 µs in the ioctl, against 10 to 12 µs with none).

- **A draw that reads and writes a color in the pixel interlock after an earlier draw of its
  instance wrote it begins a render pass instance of its own on MoltenVK on Apple's GPUs**
  (`DriverFact::InterlockedDrawsRaceUnderLoad`, `WorkCounts::instance_splits_interlocked_writes`,
  `render-pass-split`). MoltenVK encodes no Metal barrier inside a render pass there, so the
  barrier other drivers take orders nothing, and the raster order group it puts the storage in
  did not order such draws while other processes' work ran on the GPU. Each instance stores and
  loads the pass's attachments, a tile flush a draw. On the M3 Max (MoltenVK 1.4.2, macOS
  27.0.1; `runtime_lane::tile_path_costs`, release, four runs each) `wide blended in the
  function` takes 11.5 to 12.9 ms a frame against 1.8 ms with the barrier (243 instances a frame
  against 9), `extended range` 11.8 to 12.4 ms against 2.5 to 2.6 ms (234 against 9), and
  `extended range, wide attached` 10.7 to 12.4 ms against 2.5 to 2.7 ms; every other case keeps
  its instances and its time.

- **A make whose own submission is the first to meet a lost `VkDevice` is refused
  `DeviceLost`** (`device/line.rs`): where a call's own work on the device (a staging upload of a
  buffer's or texture's bytes, an internal submission) finds the `VkDevice` lost before any wait
  did, that call answers `DeviceLost`, and the next call makes the new `VkDevice`; Metal's object
  creation never fails because of a hang. Where a wait found the loss first, the call waits for
  the recovery and goes on.

- **A texture over host pages whose buffer ends inside its last row's pitch is a shadow**
  (`InPlaceRefusal::Placement`, `WorkCounts::host_texture_shadows_refused_placement`,
  `shadowed-texture`), where Metal reads it in place: a linear image's last row takes the whole
  pitch, so the image's memory requirements reach past the buffer's bytes, which are not the
  runtime's to bind. So over the caller's pages imported by their pointer and over a copied host
  buffer's copy alike, whose bytes are the buffer's length (ANV on the Iris Xe, RADV). The shadow
  copies the texels in before each command buffer that uses the texture and merges them back
  after each write.
- **A copied host buffer costs O(its size) at every batch's completion that may have written it, and
  at every commit where its pages carry no log** (`copied.rs`, `copied-host-pages`), and again at
  the submission of each batch that waits for an event value its commit found not reached
  (`command_buffer/copies_again.rs`, `copied_copy_ins_after_waits`; none for a command buffer
  without such a wait): until a command
  buffer that may write the copy is committed, a commit compares the pages with the copy a page at a
  time, and a page that differs takes the pages' whole; from then on each comparison reads one side
  whole (the pages at a commit, the copy at a completion) and compares each 1 KiB block's tag with
  its base's (below), reading the other side's block only where the tag changed; either on up to
  eight threads for 4 MiB or more (`copied_compare_threads`). On a Ryzen 7 7840HS (the RTX 3090's
  and RTX 2080's hosts), its comparisons on four threads, a 256 MiB copy that agrees took 5.8-6.1 ms
  each way, 5.8-6.5 ms where 1% of its pages changed on one side or both, and 25.7-26.4 ms in and
  25.9-27.3 ms back where every byte changed on both sides (the byte-for-byte snapshot this
  replaced, interleaved with them: 14.4-15.8, 15.0-16.4, 143-152 ms in and 204-215 ms back;
  `tests/copied_bench.rs`); with other work running, 2,049 copies of 680 MiB that agree take 29-30
  ms each way, each on one thread. As long at each completion of a command buffer whose functions
  read addresses and may store (which may write every copy; a draw whose vertex and fragment
  functions store nothing writes none) or that executes an indirect command buffer. Where the pages'
  memory carries a log of their writes (`HostMemory::with_dirty_pages`, `HostRam::with_dirty_pages`,
  a copy that drops the GPU's writes included), the commit compares only the pages it reports
  written: 0.2 ms for 680 MiB where none changed, 4.2 ms where 1% did. A completion whose command
  buffers' uses name the copies they write compares those alone. A copy is made only where the
  caller's gate refuses an import, an admitted import fails, or the GPU's writes to the pages must
  be dropped (pages the process cannot write, and memory of Metal's private option bit 24 where the
  system maps no read-only view), never for other memory by default.
- **A kernel that rewrites a copied host buffer pays its merge back** (`copied.rs`,
  `tests/copied_pass_bench.rs`): a completion compares each 16 bytes of each block the GPU changed
  with the agreed bytes and stores only those the GPU changed and the CPU did not, so that no byte
  the CPU may be writing is stored over: all 16 in one store where every one is to be written, else
  in one masked store on a host with AVX-512BW, else in up to eight stores of 8, 4, 2 and 1 bytes. A
  float a kernel rewrites changes some of its bytes, so most of a rewritten buffer's 16s take the
  masked or the smaller stores. On the 7840HS, on four threads, every float of 256 MiB changed as
  `x + i` (135 MB of changed bytes) merged back in 25.5-27.5 ms (4.9-5.3 GB/s of changed bytes), and
  in 40.1-47.4 ms (2.8-3.3 GB/s) without its masked stores (270-289 ms with the snapshot); on the
  RTX 5080 Laptop GPU's host, an Intel Core Ultra 9 275HX without AVX-512, in 41.9-44.8 ms (3.0-3.2
  GB/s; 389-423 ms with the snapshot). One command buffer a pass of a kernel that reads and rewrites
  a copied buffer, its comparisons on four threads, on the RTX 2080 host (an eGPU link of 2.7 GB/s
  each way): 64 MiB in 63.5-64.1 ms, its commit 1.8-1.9 ms, and 256 MiB in 240-250 ms, its commit
  5.5-7.0 ms, where the device's own `Shared` buffer takes 50.6 and 202 ms (137-165 and 526-632 ms
  with the snapshot); on the RTX 5080 Laptop GPU's host 64 MiB in 15.0-15.5 ms, its commit 2.1 ms,
  and 256 MiB in 60.4-61.8 ms, its commit 8.3-8.5 ms, where its own `Shared` buffer takes 3.5 and
  12.7-13.6 ms (69.6-70.8 and 287-307 ms with the snapshot).
- **A copied host buffer that a command buffer may write finds changed bytes by a keyed tag**
  (`copied.rs`'s `tag`, `copied-host-pages`): from the first commit whose command buffer may write
  the copy (its uses wrote one of the copy's keys, or it may write memory no use names), a
  comparison takes a 1 KiB block whose 128-bit tag agrees with its base's as unchanged and reads
  none of its bytes. The tag is two NH sums of the block (RFC 4418, the second keyed by the first's
  key shifted by four words) under a key drawn at random once per process: a changed block keeps its
  tag with a chance of at most 2^-64 per block per comparison, whatever its bytes. A CPU write so
  missed is not copied in at that commit, a GPU write not merged back at that completion (nor
  dropped, for a copy that drops them), and a CPU write to a block the GPU changed, so missed at
  that completion, is undone where the GPU did not change its byte (the agreed byte stored over it).
  A copy no command buffer may write compares bytes. No switch compares the bytes instead: that
  needs the agreed bytes on the host again, the copy's length (`copied_base_host_bytes`,
  `copied_base_device_bytes`).
- **A written copied host buffer's base costs its length again, in the device's memory where a
  resizable BAR has room, else in the host's** (`copied.rs`, `Device::copied_base`): a copy holds
  nothing beyond itself until a command buffer that may write it is committed; then a 64th of its
  length in host memory for its tags, and the agreed bytes in coherent device-local memory the host
  maps where that heap is 1 GiB or more (a resizable BAR's: the RTX 5080 Laptop GPU's, the RX 7900
  XTX's) and the runtime's blocks there and the base stay within half of it. Elsewhere the bytes lie
  in host memory: on the RTX 3090 and RTX 2080 hosts, whose device-local host-visible heap is the
  246 MiB BAR window, a written 256 MiB copy holds 260 MiB of host memory beyond itself (the
  snapshot this replaced: 256 MiB for every copy, written or not), its base made in 72 ms on the
  7840HS; and on every device with unified memory. A base in device-only memory, read and written
  through the device's copy engine, is not built. Where the base's bytes lie in the device's memory
  (only on a device with a staging queue, `device/staging.rs`), the host reads them uncached, with
  streaming loads on an x86-64 host with SSE4.1: on the RTX 5080 Laptop GPU's host 9-16 µs a 1 KiB
  block while its GPU is awake (plain loads 36-38 µs) and 170-206 µs after three seconds idle, four
  threads no faster than one (`tests/copied_base_device.rs`). A comparison reads them only for a
  block both sides changed since they agreed, and keeps each such block in 1 KiB of host memory for
  the base's life (`copied_base_blocks_read`, `copied_base_host_blocks`, counted in
  `copied_base_host_bytes`), so the host holds the base's bytes of every block both sides wrote, up
  to the copy's length again. A comparison that would read 32 such blocks or more reads them all
  through one copy into host memory on the staging queue, which waits for no work of the program's,
  runs at most 16 blocks apart copied as one, its buffer freed before the comparison returns
  (`copied_base_blocks_read_back`); the host reads fewer. Merges back of a 256 MiB copy whose N
  blocks, every other one, both sides changed took, host reads against the copy: on the 5080's host
  9.1-9.4 against 9.6-10.2 ms at 8, 9.2-10.0 against 9.4-9.9 at 32, 10.8-12.0 against 9.0-9.4 at
  128, 24-35 against 10.6-10.9 at 1024 and 69-120 against 15.6-16.7 at 4096, and after three seconds
  idle 130-218 against 11.1 ms at 651; on the RX 7900 XTX's host (other work running) 6.7-7.2
  against 6.9-8.3 at 32, 7.5-8.1 against 7.2-7.8 at 128 and 24.7-29.2 against 15.8-16.6 at 4096
  (`copied_read_back_bench`). A copy that finds the 5080 powered down waits for it to wake:
  0.94-1.28 s in 3 of 12 copies made after three seconds or more idle. With every float of 256 MiB
  rewritten by the GPU and a byte of every block by the CPU, the first merge back took 187-192 ms on
  the 5080's host and 167-228 ms on the XTX's, 176-292 ms on either after three seconds idle, and
  each one after it 60-68 ms and 33-64 ms (the snapshot: 403-439 ms each time on the 5080's host,
  where the host's own reads of all the blocks after three seconds idle take about 54 s).
- **Two memories over overlapping host bytes agree GPU to GPU only through the pages when one is
  a copied host buffer** (`copied.rs`, `copied-host-pages`): a copy and another copy or an import
  of pages that overlap without being the same first byte and length (a rebind's bytes inside a
  surface's, a guest RAM chunk's import over a copy's pages, a rebind whose texture moved in its
  page, a buffer made after the range was reported remapped), and a copy beside a live import of
  the same bytes for the other storage mode (a `Managed` buffer's import, then a `Shared` buffer
  over them whose import the gate refused). The writer's bytes reach the pages
  at its completion, and the reader's copy at its commit, so a read committed before the write
  completed reads the bytes from before it: in one command buffer, and in one queue's command
  buffers in flight together. A reader held by an event wait its commit found not reached copies in
  again once the wait is met (`command_buffer/copies_again.rs`). Of two such copies written in
  flight, the first's write stays where
  Metal keeps the later. Buffers over exactly the bytes a live copy covers lie in that copy and are
  one memory (`copied_host_buffers_shared`).
- **A copied host buffer's merge keeps the CPU's byte where the CPU and the GPU both wrote it**
  between a commit and the completion: a merge writes only the bytes one side changed, so a GPU
  write never rewrites a byte beside it, but a byte both wrote keeps the CPU's where on Metal the
  later write wins; one the CPU writes while the completion's merge runs keeps the CPU's or the
  GPU's (the merge stores it where it read it unchanged).
- **On MoltenVK a kernel's out-of-bounds write through a buffer reaches other bytes of the
  buffer's memory allocation**, where Metal's GPU faults past the pages of the resource: on the
  M3 Max (MoltenVK 1.4.2) a dispatch writing three pages through a one-page buffer of the
  runtime's own changed 32,720 bytes of the buffer made after it, and through a buffer bound in a
  chunk of guest RAM (`Device::use_host_ram`) as many bytes of the chunk's next pages. A no-copy
  buffer's own import ends at its last page, so nothing past it changed. On the RTX 5080 laptop
  (NVIDIA 615.71.09) and lavapipe the same write left the chunk's pages untouched.
- **A shadowed texture over an IOSurface plane's bytes is copied in whole for every command
  buffer that samples it**, where it is a shadow: one that only samples or reads is an image in
  place where its class's check passed (`in_place.rs`), so this holds for a plane that renders or
  writes, one whose check has not passed, one whose pitch no image takes, and every one under
  `HostPageTexturePolicy::Shadow`. A binding bounds nothing it reads -- only a pass's color attachment
  does (`ShadowUse::loads`, its damage rectangle) -- so a copy in for a sampled texture moves
  every texel: 8,294,400 for one 3840x2160 plane. A sampler's reach is unknowable, so that is not
  slack. Within one command buffer that is paid once, its later encoders taking the
  image the first left holding the truth (`shadow::ShadowCarry`), because Metal shows every
  encoder of a command buffer the bytes as of its commit. Across command buffers it is not: Metal
  shows a later command buffer a CPU write made since the commit, so each one that samples the
  plane copies it in again, and nothing short of knowing which of its pages changed could tell
  that from a plane nobody wrote (a present's own copy in is one Metal requires, under **Costs of
  matching Metal**). Three figures divide such a plane's first copies in on a run and say what the
  carry is worth there: the texels it took off, those a copy in moved behind an earlier command
  buffer's standing, and those it moved behind none at all
  (`WorkCounts::shadow_texels_first_copy_in_carried_large_surfaces`,
  `shadow_texels_first_copied_in_large_surfaces_stood_earlier`,
  `shadow_texels_first_copied_in_large_surfaces_unstood`). A compositor that submits one command
  buffer per use of the plane has the whole of it in the second: the carry takes nothing off it.
  Not announced.

- **Command queues live past the device's `VkQueue`s share one, so one's command buffers wait
  behind another's whole command buffer where Metal's wait only inside one command.** Measured on Metal's
  own API on an Apple M3 Max: a side command queue completed one command buffer behind a single
  2.50 s dispatch, waiting its whole duration, and 11,335 to 12,992 behind the same 2.5 s of work
  as 256 dispatches, none waiting past 3.8 ms. So Metal holds an unrelated queue for one long
  command and for nothing else, while one `VkQueue` runs its submissions in order and cannot
  schedule a sharing queue out of them. A device now lends the round robin every queue its families
  have past the two reserved for presenting, at least four and at most sixteen, which on a graphics
  family of sixteen gives fourteen, and a new command queue takes the one the fewest live queues
  hold, so queues share only while more are live than `VkQueue`s, whatever queues came and went
  before (a guest's processes make about 350 in a 250 s walk); a device whose families have none
  to spare -- MoltenVK's four families of one queue, a single-graphics-queue device -- still
  shares. Counted: the queues made past the round robin's `VkQueue`s
  (`WorkCounts::queues_sharing_a_vk_queue`), the most the program kept live at once
  (`command_queues_live_most`), and unshared queues that found no `VkQueue` free of other queues
  (`unshared_queues_sharing_a_vk_queue`). Not announced.

- **Off macOS, GPU writes into read-only memory are dropped only at the writer's completion.**
  Metal drops what a shader or the blit engine writes into memory whose backing is read-only
  (no-copy memory of its private option bit 24, or pages replaced read-only): a kernel's or a draw
  stage's stores, through a bound slot or an address loaded out of an argument buffer, a blit fill
  and a visibility result complete, and neither a later command buffer nor the CPU sees them,
  though a later command of the same command buffer may read them back from the GPU's caches; what
  its atomics do there is unmeasured. A render pass's store-out over such memory
  (`storeAction = Store`, or a multisample primary stored there) ends its command buffer in error
  (`MTLCommandBufferErrorDomain` / `kIOGPUCommandBufferCallbackErrorPageFault`), while attachments
  at `storeAction = DontCare` or `loadAction = Load` complete; resolving *into* such memory is
  unknown (no page fault in 520 processes, almost all of them recovery victims). On macOS the
  runtime imports such pages through a read-only view of them (`read_only_view.rs`), which takes the
  GPU's writes nowhere as Metal's memory does. Elsewhere no import keeps them out (ANV lost the
  device and RADV failed every submission over read-only imported pages), so the memory is a copy
  of the pages whose GPU writes each batch's completion takes back out (`copied.rs`, announced as
  `copied-host-pages`): the CPU and a command buffer committed after that completion read the
  pages' bytes, as Metal's do, but one committed before it, while the writer is in flight, reads
  the writes, and a render pass's store-out into the copy completes where Metal's faults its
  command buffer. The copy costs what any copied host buffer costs (the first gap above). On the
  RTX 5080 Laptop GPU, commit to completion, median of 300: beside a live 16 MiB copy whose pages
  carry no log, a 256-byte fill into a writable 16 MiB no-copy buffer took 1.5 ms (49 µs beside a
  64 KiB copy, into a 64 KiB buffer), as every commit compares the copy, and the same fill into the
  copy 2.6 ms, its completion restoring it; into the 64 KiB copy 53 µs.

- **A function linked from a module that declares function constants** is refused
  (`Unsupported{Linking; NotImplemented}` at a surviving call to the stub, `canon::link`), keyed on
  the module declaring any constant at all. Metal is narrower: it refuses only an *unresolved*
  constant the linked code reaches. Measured on an Apple M3 Max under macOS 27.0.1, one compute
  pipeline per shape, the visible function in `MTLLinkedFunctions.functions` and the linked module
  declaring `[[function_constant(0)]]`: the function not reading it creates and runs, from
  `newFunctionWithName:` and from `newFunctionWithName:constantValues:` alike; the function reading
  it creates and runs from `newFunctionWithName:constantValues:` and is refused from
  `newFunctionWithName:` with "cannot lower module with unresolved function constants". So three of
  those four shapes are ours to close, and a specialized `MTLFunction` is how Metal resolves them
  before linking. The values are already at the site (`LinkedName::constants`); closing it means
  binding them and folding the linked module before the import, then refusing only a constant the
  reached code reads with no value bound. A dynamic library of such a module is blocked the same
  way, and the same device answers that `newDynamicLibrary:` refuses the module whose exported code
  reads the constant and accepts the one where nothing reads it; a pipeline importing from the
  accepted library is **unmeasured**.

- **A sample or gather in an AIR 1.x library** (deployment target macOS 10.12 or earlier, iOS 10
  or earlier) is refused (`Unsupported{NotRepresentable{HandleAsPointer}}`). From AIR 2.0 a
  `constexpr sampler` is the module global `@__air_sampler_state`, whose packed word the ABI
  decodes; before it the same sampler is the word cast inline at the call
  (`inttoptr (i64 ... to ptr addrspace(2))`), in a packing that is not AIR 2.0's -- the default
  sampler is `-9223372036854251447` at AIR 1.11 against the `-9188470239253757879` AIR 2.0 stores
  -- and that layout is not pinned, so the call is refused rather than guessed. Reads, writes,
  queries, atomics, render targets, function constants and argument buffers all translate at
  every AIR version this toolchain emits (1.6 and up); pinning the older sampler word would close
  the sample and gather too.

- **A sparse tile mapped after a write covered it while unmapped** reads zero for us and Metal's
  unwritten pattern (`0xFFFFFFFF`) for Metal; an unmap-and-map round trip loses the texels the same
  way. The runtime zeroes a sparse texture's unmapped texels and every newly bound block, and
  `BlitEncoder::scrub` has already marked the whole destination of the blit written, so nothing
  holds the tile unwritten; keeping the dropped part of a write unwritten in `FreshUnwritten` would
  reach it. No case compares it, because Metal means macOS means MoltenVK, which binds no sparse
  memory: confirming it needs a venue with `sparseBinding`, a sparse-binding queue family and
  `sparseResidencyImage2D` (the three `SparseNeed` checks), and the Metal half stays
  `probes/sparse_unmapped_blit.m`'s.

- **A sparse mapping update of a slice past the texture's own slices** maps nothing for us, where
  plain Metal takes the call and reaches real memory: it binds one page, keeps it through an unmap
  of the same slice, and part of that page lands inside an in-bounds mapped tile, whose texels then
  read Metal's unwritten pattern (`0xFFFFFFFF`). The runtime maps none (`UnitGrid::clipped_size`),
  so it loses no texel, and counts the pages Metal's heap counts, on every level and the tail,
  given back on an unmap (`UnitGrid::clipped_region`,
  `probes/sparse_slice_past_levels_usedsize.m`). The address Metal's out-of-bounds
  slice produced is Apple's own page layout for the chip and no rule for it is measured, so the
  exact form cannot be derived yet; the conformance case
  `sparse-mapping-a-slice-past-the-textures` holds every side of the difference visible, with its
  four rows of `crates/metal2vulkan-conformance/known-gaps.tsv`.

- **A linear sample of a converted two-plane YCbCr texture over a P3 surface** (`DCI_P3`,
  `P3_D65`, each code read alone): the runtime filters the codes as the code-reading format's
  (`RGB8_420_2P` and its kin), unannounced; Metal's filter of such a texture is not measured.

- **A two-plane YCbCr texture over plane 1 of a three-plane surface** (`y420`, `v0a8`): Metal
  reads plane 2's pairs as its chroma; the surface's facts (`IoSurfaceYCbCr`) name plane 1 alone,
  so the runtime reads chroma codes 0 there, as over plane 1 of a two-plane surface.

- **A mipmapped two-plane 3D texture's reads** (`RGB10A8_2P_XR10` and its sRGB twin): Metal's
  read and sample take bytes within the texture's first 4 KiB that vary from run to run, where a
  shader here reads its first texel, as of a one-level one (`lower::texture::emit::planes_volume`).

- **A draw under Apple's last provoking vertex on a device without it** (MoltenVK), whose pipeline
  state reads flat inputs or selects a slice or viewport (or a strip's whose fragment function reads
  barycentrics): it draws the list of its primitives led by their last vertices
  (`render/provoking.rs`), where Metal's mode costs nothing. A direct draw without indices draws the
  list the CPU writes into the upload arena (4 bytes an index); every other draw the list a run
  writes, one dispatch for a run of draws over one index buffer (`index_rewrite.rs`): the index
  gather pass's for a direct indexed draw of lines or triangles and for an indirect draw but an
  indexed strip's (an indirect run's after one plan dispatch), the index rewrite pass's for a direct
  indexed strip of up to a few thousand positions. On MoltenVK on the M3 Max that is 0.16, 0.24,
  0.65 and 0.71 us more frame time per direct, indexed, indexed strip and indirect draw of two
  triangles than at mode 0 (1.58, 1.72, 1.73 and 1.71 us; host_ab's `provoking`, 2000 draws, 3
  rounds), and 2.5 us per indexed indirect draw against mode 0's 4.1 (a run of the index bounds
  pass, which a gathered list does not need). A draw no run takes (a longer strip, an indexed indirect strip,
  indices an earlier draw of the encoder wrote) takes the index rewrite pass's three dispatches
  between barriers before its render pass: there 77 us per draw.

- **An indirect render command's draw of primitive type 9 that its execution's call cannot see**:
  one the CPU wrote after the call, or among a range the execution reads from a buffer. It runs
  nothing, counted as a record fault in the execution report (`RECORD`), where Metal's GPU hangs
  or faults the command buffer at it (the M3 Max, macOS 27.0.1: one written after the call, after
  the commit, or among a range read from a buffer failed as one the call sees, a direct one hung
  27.6 to 28.7 ms after `Scheduled`, an indexed one faulted 8.7 to 10.6 ms after it). One among a
  range given at the call, as the records stand then, fails the command buffer as the encoder's
  draw does (`render/executions.rs`); failing it from inside the host step that meets the record,
  its earlier encoders run, is not built.

- **A command buffer committed behind a running predicated loop on the loop's own queue waits for
  the loop to end**, as every batch of a queue waits for the ones before it, where Metal's GPU
  runs it meanwhile: a fault or hang in it ends the loop there on Metal (`recovery.rs`), and in
  the runtime it waits, for good behind a loop without bound. A fault or hang on another queue
  ends the loop as Metal's does.

- **What a no-copy buffer reads after a remap of its pages under it is unspecified**, where Metal's
  buffer keeps the pages it was made over. `HostMemory::new`'s contract forbids replacing the pages
  under a live buffer (`mmap` with `MAP_FIXED`), and no notice tells of it. Under a live import of
  the pointer, `VK_EXT_external_memory_host` leaves undefined whether the import holds a reference
  to its pages, so what later command buffers read and write is the driver's, and differs between
  drivers. Under a released one (`host_residency.rs`), the next blit imports the new pages. On RADV
  (Mesa 26.2.4, the Radeon 780M and the RX 7900 XTX) a file's pages mapped so over a live pointer
  import of anonymous pages lose the device: its next submission is rejected whatever it uses ("The
  CS has been rejected"): amdgpu's userptr takes only pages with no file behind them, and each
  submission lists every memory the device imported. No exact form exists once the pages are gone,
  and `Device::host_range_remapped` does not release such an import. An import of dma-bufs
  (`HostMemory::with_dma_buf`) keeps its pages on every driver, but is taken only where the device
  refuses the pointer.
- **On a device with one graphics queue, a compositor's command buffers wait behind another
  queue's long batch**: RADV, ANV and lavapipe expose a single graphics `VkQueue`, so every command
  queue, `Device::new_command_queue_unshared`'s included, submits to it, in order
  (`WorkCounts::unshared_queues_sharing_a_vk_queue` counts each unshared queue so placed). Other
  queues' batches are held back to about 2 ms of expected work ahead of an unshared queue's
  (`unshared_isolation = UnsharedIsolationPolicy::Bounded`, above), but a batch already submitted
  runs to its end: a command buffer committed behind another queue's 200 ms one starts after it,
  where Metal runs the compositor's beside it, and one behind a batch longer than its queue's
  recent ones waits for the rest of that one too. On the Iris Xe (ANV, Mesa 26.2.6, i915) work of
  another `VkDevice` (its own kernel context) ran only between submissions, never inside one,
  whatever its global priority (`HIGH` is refused without privilege; `MEDIUM` against `LOW`
  answered as equal priorities did): a light submission waited the whole of a 440 ms command
  buffer, and 14 ms where the same work was 32 submissions. Neither a second `VkDevice` for such
  queues nor splitting long command buffers into submissions is built. Each command buffer whose
  first batch goes behind another command queue's batch not yet seen complete is counted
  (`WorkCounts::command_buffers_behind_other_queues`, those whose wait ran over 16 and 33 ms, the
  longest wait, and the same four for unshared queues), its wait timed on the host from its
  submission to that batch's observed completion: an upper bound; for an unshared queue's waits
  over 16 ms, also the batches in flight ahead of each and its longest single run
  (`WorkCounts::unshared_waits_over_16ms_batches_ahead` and kin).
- **On Asahi Linux, the kernel can keep a freed dma-buf import, and its pages, past the process's
  exit**: the asahi DRM driver (its source at tag `asahi-6.19.14-2`; Honeykrisp in Mesa 25.3.6)
  makes an imported object without the driver's close callback, so a freed import stays mapped in
  the GPU's address space until a later bind maps over its range, and whatever is still mapped
  when the device's GPU VM is torn down is queued for a cleanup that never runs. Those objects,
  their dma-bufs and the pages under them stay allocated. Honeykrisp imports each dma-buf twice
  (`vkGetMemoryFdPropertiesKHR` binds one too). The runtime frees every import it made
  (`tests::dma_buf_leak_probe`); no documented Vulkan call makes the driver unbind one, so the
  gap stays until the kernel changes. `scripts/dma-buf-leak/` holds a C reproducer and the
  bpftrace scripts that show which layer keeps them.
- **A no-copy buffer's pointer import that stays live costs RADV's every kernel submission per
  page**: amdgpu gets every page of each live pointer import again in each command submission
  (the RX 7900 XTX under Linux 7.0.0: the scheduler thread's CPU per command buffer of a
  guest-shaped frame 18 us with none, 328 us with 275 imports of 512 KiB, 1144 us with 275 of
  2 MiB). An idle one is released where the device's check finds a held pointer import costs every
  submission (`submit_check.rs`, the import tax), but one whose Metal address was handed out, or one
  made before the check answered, stays live. 275 kept dma-buf imports of 512 KiB each
  cost 54-56 us per command buffer there (no other size measured).
- **A detach or rebind between a command buffer's submissions.** A command buffer's later
  submission, behind an event wait, reaches a no-copy buffer detached or rebound after its first
  submission through an address (made resident, or unnamed) through the pages its first
  submission found, where Metal's reads the new pages or zeros; those pages stay imported, and
  their deallocator waits (announced), until the command buffer completes. Its commands that
  named the buffer are encoded again over the new pages, as Metal runs them, and fail with code 8
  where it is detached by then, as Metal's; those of an encoder that cannot be encoded again (a
  compute encoder its unit of command predication reaches past, which is one whose unit laid a
  step before it opened or one of its own that the unit's later encoders read, and one whose unit
  laid a loop or left work out; work on the GPU alone) fail `EncodedUseRebound` (announced).
- **A copied rebind under a texture in place, where the device's check did not pass fresh pages
  and its own linear image lies at another pitch, is refused** (`NotImplemented(RebindTextures)`).
  A rebind whose pages the gate refuses, or whose bytes moved in their page, copies them, and the
  texture's image is made again over the copy: in fresh pages for host allocations, at the texels'
  pitch, where the check passed for fresh pages (it checks them wherever the device takes them,
  also where each held pointer import costs every submission, as on RADV); else a linear image of
  the driver's own over memory the device allocates, which must lie at the texels' pitch. So on a
  device whose check failed fresh pages, or that takes none (it imports no host pointer), a texture
  whose rows lie at another pitch than the driver's linear image's is refused. Metal accepts the
  rebind.
- **A packed 16-bit attachment's blend on a device that does not render the format**
  (`A1BGR5Unorm`, `ABGR4Unorm` on the RTX 5080). Its word view (`WordAttachment`) takes writes
  (blended and partly masked ones through the pixel interlock, the row above), color inputs and
  resolves; a blend of an output of fewer than four lanes is `NotImplemented`, and a blend of the
  second source, or a blend or partial mask on a device without the interlock or on one that
  orders fetch through rasterization-order input attachments (lavapipe: the function blends and
  merges no packed word over its input attachment), is refused as the device lacking the format.
- **A blended or fetched one- or two-channel 8-bit sRGB attachment on a device that does not
  render the format** (`R8Unorm_sRGB`, `RG8Unorm_sRGB` on the RTX 5080 and ANV). Its word view
  takes writes (partly masked ones through the pixel interlock, or over the attachment's input
  attachment on a device that orders fetch through them, lavapipe) and resolves; a blend other
  than one and zero is refused as the device lacking the format, and a color input is
  `NotImplemented(Format)`. A device that orders fetch only through the pixel interlock and renders
  any color format through its word view (these, the packed 16-bit ones, `RGB9E5Float`) answers
  the `ProgrammableBlending` family row `NotImplemented` and `isFramebufferReadSupported` no.
- **A half output, blend, partial mask or color input of an `RGB9E5Float` attachment on a device
  that does not render the format** (the RTX 5080). Its word view takes a `float` output's writes,
  clears and resolves; a `half` output and a color input are `NotImplemented(Format)`, and a blend
  or a mask of some of its channels is refused as the device lacking the format.
- **ASTC textures on a device without ASTC** (and Apple's 4x2 and 8x4 footprints on every
  device). Making a heap's or a sparse heap's such texture (`RuntimeGap::DecodedHeapTexture`) and
  a view of another decode (an LDR or sRGB texture's HDR form, an HDR texture's LDR or sRGB form)
  are refused. The heap size query answers one: its footprint is Metal's blocks, which no image of
  the device decides. On every device, a kernel's or a fragment function's storage slot over an
  ASTC texture binds the slot's neutral (`astc-storage-neutral`): its writes leave the blocks as
  they were, where Metal stores undefined bytes into them, and its reads answer zero, where Metal
  reads the blocks' decode. A resolve of an ASTC attachment is refused, unmeasured.
- **The 26 formats `metal::format::footprint` answers for** (`ARGB8Unorm_sRGB`, Apple's nine
  one-plane `YCBCR*` formats, the ten padded 10- and 12-bit display formats, the four `RGB10A8_2P`
  ones and the two packed 10-bit ones). `heapTextureSizeAndAlignWithDescriptor:` answers Metal's
  exact size and alignment for every one, because the footprint is arithmetic over the descriptor
  and no image decides it.
  - **Twenty are held for copies alone** (`metal::format::copies_only`: `ARGB8Unorm_sRGB`, the
    one-plane `YCBCR*` and the padded ones): a texture of each is made, its bytes in `R16_UINT` or
    `R32_UINT`, and host copies and blits move them. A read, a write, a render pass, a view, a
    texel buffer's read and a sparse texture of one stay refused `NotImplemented(Format)`: what a
    function reads from one is not measured. `ARGB8Unorm_sRGB`'s reads need a shader-side decode
    of its codes by the device's transfer (`pack::srgb_decode_texel`), as a UNORM twin's are, as
    the only read of it: no Vulkan format applies the sRGB transfer to the three channels Metal's
    does (every `_SRGB` format encodes the word's low three and leaves the fourth linear, and
    Metal's alpha is the low byte here, which a view's component mapping cannot move because
    format conversion runs before the swizzle).
  - **The four `RGB10A8_2P` formats** stay refused `NotImplemented(Format)` at every use: Metal
    lays them out in two planes (their footprint), as `RGB10A8_2P_XR10`'s, whose blits between
    textures move both planes (`two_planes.rs` in the runtime); one bytes image of a texel's word
    would not hold the second plane, and the two-plane holding is not built for them.
  - **The two packed 10-bit ones**: plain Metal lays out no image storage for them and makes a
    texture from every creator that needs none -- a placement heap's offset, the shared creator in
    `Private`, a buffer's and an IOSurface plane's own bytes, and `Memoryless` -- so the refusal
    from each of those is this crate's own and not Metal's nil; the device's own creator and an
    automatic heap, the two that would have to lay the image out, answer nil (an Apple M3 Max,
    macOS 27.0.1, `conformance/probes/packed10_texture_alone.m`). Their reads are a stream of
    10-bit codes that no Vulkan format holds, and none of that is built. The code layout is
    derived in `metal/format/packed_codes.rs` and is **not** held to a texel: the two measured
    anchors fix codes 0 and 1 only, and a padded 32-bit word and an unpadded bitstream first
    differ at `R10Unorm_PACKED`'s texel 3, which nothing has measured.
  - A **multisample** descriptor of any of the 26 is refused, its layout unmeasured: Metal's
    validation may stop such a descriptor, so none was asked.
- **`BGRA4Unorm` as a render target on a device that does not render its word.** Its texel is
  `A4R4G4B4_UNORM_PACK16`'s word bit for bit (core 1.3, or `VK_EXT_4444_formats`), and a device
  exposing neither is refused the version or the extension at the format's first use. The two
  core-1.0 4-bit words hold the same bits with the channels one place over, but only a rotating
  view reads them back and what Metal's opaque-black border reads through this format's rotation
  is unmeasured (`ARGB8Unorm`'s was measured, and is not alpha), so no fallback candidate is
  taken. Its write conversion, its filter and its channel order are measured (an Apple M3 Max
  under macOS 27.0.1, `probes/bgra4_write_and_mipmap.m`): blue in bits 0 to 3, then green, red and
  alpha, read out of Metal's own stored words; each channel `ChannelWrite::HalfUnorm`'s rule, as
  its four siblings; `generateMipmapsForTexture:` the device's blit chain of its word, or the
  compute filter (`bind::mipmap_filter`) where the device's blits cannot filter it.
  What is still open is the **render target** on a device whose `A4R4G4B4_UNORM_PACK16` carries no
  `COLOR_ATTACHMENT` -- both MoltenVK profiles and the RTX 5080 -- where the format is refused
  rather than rendered through the view of its word. Metal takes both a `[[color(0)]]` fragment
  function and a source-over blend on an attachment of it, so the word view's refusal of a
  fetch and of any blend but one and zero would be our gap and not Metal's; and because
  `framebuffer_fetch::renders_through_words` is an `any` over every format, taking the fallback
  would turn MoltenVK's `ProgrammableBlending` family row into a gap and block every Apple family
  there. The fallback therefore waits on a packed-16 word fetch in `lower::fragment::fetch`
  (`metal::format::word` has no entry for the format meanwhile). Six committed profiles render it
  directly (RADV, lavapipe and ANV), where the measured write conversion applies.
- **PVRTC textures on a device without PVRTC.** One without a decoded twin (a sparse heap's, or a
  heap's over the caller's host pages or laid out apart from its bytes) holds its blocks for copies
  alone: a command that reads it is refused.
- **ETC2 and EAC textures on a device without them.** A sparse heap's such texture
  (`RuntimeGap::DecodedHeapTexture`) and a view of another decode (an EAC 11-bit format's other
  signedness) are refused; one without a decoded twin (a heap's over the caller's host pages or
  laid out apart from its bytes) holds its blocks for copies alone, and a command that reads it is
  refused.
- **A level the shader computes from derivatives.** An extended-range texture's fragment sample at
  an implicit level or with explicit gradients takes Vulkan's level of detail computed in the
  shader on every device, where Metal's is not measured, announced as `DeviceSampleLevel`.
- **An anisotropic cube sample on a `y` face through MoltenVK's vertex and fragment stages under
  macOS 27.** SPIRV-Cross's `spvGradientCube`, which MoltenVK 1.4.1 and 1.4.2 apply, writes a
  `+Y` or `-Y` face's gradients as `(d_x, d_z, d_x)`, so Metal loses the `z` gradient. The module hands such a
  face Metal's length of the two tangents on both (`CubeGradientRoute::AppleProjected`): isotropic
  samples, and every sample on the `x` and `z` faces, are Metal's, but an anisotropic one on a `y`
  face filters along the face's diagonal, announced as `CubeGradientFootprint`
  (`cube-gradient-footprint`). A kernel's samples are Metal's own. A form exact inside the face,
  sampling a `y` face through a 2D-array view of the cube, needs a second binding of each such cube
  from the binder and the runtime; it is not built.
- **A group operation in a branch arm whose speculation no probe covers.** Metal's backend runs a
  branch's short arms of plain computation in every lane of the branch's block; the module does the
  same for the arm shapes measured on Apple's GPU (`air::canon::speculate`). A branch of another
  shape whose arms hold a quad or simdgroup operation keeps the AIR's lanes, where Metal's may be
  the block's, announced as `UnmeasuredGroupSpeculation`.
- **A render pass's attachment read another way than a sampled slot binding.** Vulkan forbids
  reading an attachment in its own render pass instance (`VUID-vkCmdDraw-None-10795`), and with a
  feedback loop, reading the pixels the instance writes unordered (a data race); a sampled
  slot binding reads a copy or a snapshot (Costs above). Every other such read still reads the
  attachment in its instance, announced as `AttachmentReadInPlace`: a texture reached through the
  heap set (a handle, or an argument buffer's texture), a storage binding, a depth or stencil
  attachment a draw samples, a texture buffer or a buffer over the same buffer's bytes, a texture
  whose shadow is not served (`texture/usage_shadow.rs`), and an execution's command
  (`render/executions.rs`). In an instance that holds the attachment in the feedback loop layout,
  those reads name `VK_IMAGE_LAYOUT_GENERAL` in their descriptors, against the layout the instance
  holds.
- **Compute private memory past a device's Function storage is refused.** A kernel, object or mesh
  function whose private memory (its `alloca`s, a call stack among them, and constants held as
  `Private` variables) passes what the device holds is refused (`Unsupported{PrivateMemory}`, or
  `Unsupported{Recursion}` for a call stack): past 16 KiB a thread on a device not measured, past
  its measured value on a measured one. So is a vertex or fragment function past a measured
  device's value, or past 192 KiB elsewhere (below that it runs with `FunctionStorageAssumed`, the
  device-gap row above). Vulkan states no limit on the driver's scratch, so the lane
  `runtime_lane::private_memory` measures it per device and `device::function_storage` keys it on
  the driver, vendor and device ID (a GPU that reports several read by the one measured, the Radeon
  780M of Hawk Point as Phoenix's): 96 KiB on the Radeon 780M and RX 7900 XTX, 32 KiB on the Iris
  Xe, the UHD 770 and the RTX 5080 Laptop GPU, 128 KiB on the RTX 3090, 192 KiB on the M3 Max and
  M4 under MoltenVK, 48 KiB on lavapipe; 16 KiB elsewhere. An exact form for compute, private memory
  in a runtime buffer with a slot per resident threadgroup, is not built.
- **An object grid past the device's limits where the device does not fold it.** With
  `VK_EXT_mesh_shader` but without device addresses, `shaderInt64`, `multiDrawIndirect` or
  `drawIndirectCount` (no committed profile), an indirect draw's object grid past the task stage's
  limits fails its command buffer (`CommandBufferError::MeshGridPastLimits`, drawing nothing) and
  a direct one is refused (`Unsupported{MeshShading}` naming the limit). The emulation (MoltenVK,
  ANV on the Iris Xe) refuses a direct object grid past the compute stage's limits or whose work
  buffer passes `maxStorageBufferRange` (`Unsupported{MeshShading}` naming the limit), and fails
  the command buffer of an indirect one past the compute limits with
  `CommandBufferError::MeshGridPastLimits` (ANV's y and z axes stop at 65,535; MoltenVK's limits
  pass every grid Metal runs).
- **A mesh render pipeline's amplified draws.** A mesh descriptor's `maxVertexAmplificationCount`
  above 1 is made, as Metal makes it, and its draws of one amplification (Metal's default) draw as
  a pipeline without amplification does; a draw of more than one is refused
  (`Unsupported{VertexAmplification, NotImplemented}`). An M3 Max runs the object function once
  for the draw and each mesh threadgroup once per amplification (`mesh_amplification.m`). No form
  is built: a draw per amplification runs the object function again and needs the mesh function
  to add a view mapping's offsets, and Vulkan multiview may run the task shader once per view.
  Open: the form that runs the object stage once and the mesh stage per amplification.
- **An object threadgroup that sets a mesh grid Metal launches nothing for keeps its own writes.**
  On the M3 Max such a threadgroup's device writes (an atomic add before the set) do not land
  either, while the draw's other threadgroups run; the runtime runs the threadgroup and launches
  nothing from it.
- **A fragment function's imageblock write that its draw's tests may drop after it, where no
  form tests first.** Such a draw runs the tests before the write, early or in the shader against a
  depth stand-in. Refused as not implemented: a writing function under alpha to coverage; where
  the tests act, a draw under a visibility result mode, a depth format other than `Depth32Float`,
  a stencil attachment alone, and at more than one sample a stencil test on a device without
  multisampled storage images (MoltenVK) or a depth test where `FragCoord.z` is unmeasured (ANV);
  and on MoltenVK at more than one sample, whose early tests drop image writes
  (`DriverWorkaround::EarlyTestsDropMultisampledImageWrites`), a function declaring early tests
  that discards, returns depth, stencil or a sample mask, or writes other memory. A form testing
  at the function's entry against the stand-in is not built.
- **Threads per tile past the pass's tile.** `dispatch_threads_per_tile` refuses threads per tile
  wider or taller than the pass's tile when they are within Metal's limit for the state
  (`DrawProblem::ThreadsPerTilePastTile`). Plain Metal runs such a dispatch over every tile of the
  pass from that tile's origin, so a pixel inside the threads per tile of more than one tile's
  origin runs once per tile, on overlapping threadgroups the runtime does not model; a dispatch of
  the tile's own threads would store once where Metal stores repeatedly, so no narrower dispatch
  serves it.
- **A vertex function that selects its slice, on a device without `shaderOutputLayer` and
  `geometryShader`.** Metal renders each primitive into the slice its first vertex selects. Vulkan
  stores `Layer` from a vertex function only with `shaderOutputLayer`, and the generated geometry
  stages that store it otherwise need `geometryShader`; a device with neither refuses the pipeline,
  `Unsupported(LayeredRendering)` needing either feature (among the committed profiles
  `vk12-floor`, `roadmap-2022` and `lunarg-desktop-baseline-2023`). So does a vertex function that
  selects its viewport there with `multiViewport`. One whose slice or viewport differs among a
  primitive's vertices needs the stages even with the outputs (refused needing `geometryShader`
  without them), except on MoltenVK, which hands the indices to Metal. An exact form, a
  layered pass drawn once per layer keeping the primitives whose first vertex selects it, is not
  built.
- **Commit to completion.** A command buffer that is submitted takes longer from commit to
  completion through the runtime than a bare driver submission. On MoltenVK most of the difference
  is the two GPU timestamps each timed command buffer writes: on an M4 a bare barrier took 14.7 µs
  alone and 167.8 µs with them. `DeviceOptions::gpu_times_from_host` (off by default, announced)
  answers GPU times from the host clock and spares them. An empty command buffer is not submitted
  (`WorkCounts::command_buffers_unsubmitted`): its commit and wait took 2.6 µs on an RTX 5080 and
  1.8 µs on an M3.
- **The kernel start of a command buffer whose first command waits for an event** is its first
  batch's submission, after the signal (`gpu_time.rs`). Metal's is the time it began scheduling
  the command buffer, 13 to 39 us after its commit on the M3 Max under macOS 27.0.1, however long
  the wait; both show only with the kernel end, after the signal.
- **A texture made while a batch is submitted waits for that submission on a device without
  `descriptorBindingUpdateUnusedWhilePending`** (the `vk12-floor` profile), where its making
  registers a first move (a `Private` texture's): the registration takes the first-move order,
  which the batch holds from its take until its `vkQueueSubmit` returns. Metal's making waits for
  nothing. Elsewhere the batch holds the texture slots' writes instead
  (`WorkCounts::heap_slot_writes_deferred`, which stays 0 on such a device).
- **A shadow's copy back covers every texel it moves, where no merge serves it**: an image's
  shadow (a view of a layout the device's views do not reinterpret), a turned texture's and a
  two-plane one's. After a pass that draws, that is every texel inside the draws' scissors, where
  Metal's store writes back only the 32x32 tiles a draw's primitives touch: another writer's write
  during the pass to an undrawn texel in an untouched tile there can be undone where Metal keeps
  it. A shadow over a buffer's bytes another writer may write is merged back instead, only the
  words the commands changed (`shadow.rs`), and no write reaches the bytes between rows; a tracked
  `Private` buffer's, whose bytes no other writer writes during the pass, is copied back.
- **A merge through a copy rewrites the texture's bytes it did not change**: where the texture's
  first byte is not a whole word of its buffer, its bytes go through a staging buffer, or its last
  word runs past the buffer, the merge copies the bytes from the texture's first to its last into
  its scratch, merges there, and copies them back (`shadow_merge.rs`). Another writer's write to
  them between the two copies is undone.
- **A linear image over a `Shared` or `Managed` buffer's packed rows moves out of
  `PREINITIALIZED` at its first use**, a transition Vulkan lets read and write every byte bound to
  it: a CPU write to those bytes during it could be undone, where Metal keeps it (not observed).
- **A cold pipeline making on NVIDIA grows with a module's sampling sites.** NVIDIA's compiler takes
  about 1 s for macOS 26's largest CoreAnimation fragment (210,033 words) on the RTX 5080 Laptop
  GPU, once per cold cache; states that share the module then take 13 to 16 ms, and a warm cache
  2 ms. The module inlines a few sampling helpers five or six times each; keeping one copy of each
  would cut it. It is not built.
- **A cold making of a large module takes its driver up to seconds, once per module and
  driver.** A compositing fragment module's fragment shader library, cold with the driver's disk
  cache off
  (203,016 words, macOS 26.7.2's fixed-function compositing shader, which a uniform buffer's state
  drives): ANV on the Iris Xe (Mesa 26.2.6) 3.3 to 3.7 s at a load of about 16, its program 29,101
  SIMD8 instructions with 34 spills; RADV on the Radeon 780M 1.3 to 3.0 s; the RTX 3090's driver
  (620.70.06) 4.5 to 7.4 s. A state over another attachment format takes the first's record and
  is found in it in milliseconds (`PassShape::without_formats`). The time is the driver's own work
  on what the shader computes, not the module's size: spirv-opt's `-O` (186,314 words) and its
  frozen specialization constants (180,402) leave ANV's NIR within 0.2% and its program within 9
  instructions (29,110, 36 spills), its time within the run's noise, and holding `half` values in
  16 bits across moves, phis and selects adds spills (103).
  `VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT` leaves ANV's program and time, and the 3090's
  time, as they were; RADV takes 0.9 to 1.0 s with it, at a draw cost not measured. A convolution
  kernel a macOS 12.7.6 guest dispatches at login is the compute case: its dispatches' 12
  variants (22,874 to 29,651 words, `float` arithmetic) each compile on ANV into one SIMD8 program
  of 6,222 to 8,925 instructions with 70 to 221 spills, the driver unrolling every loop (1,280
  fused multiply-adds, 340 threadgroup loads, 116 barriers) with its accumulators live at once.
  Made alone with the driver's disk cache off, the 23,108-word variant takes 0.27 to 0.34 s, and
  11 variants made at once 0.31 to 0.53 s each, about 0.5 s for all 11; the 138,047-word
  compositing fragment module of the same login takes 2.0 to 2.4 s alone, 2.4 to 2.5 s each two
  at once and 2.7 to 3.3 s each four at once. The driver's time grows with the host's load: on
  the same Iris Xe's host (20 logical CPUs), 20 threads kept busy beside it take that fragment
  making to 6.3 s and 30 to 7.9 s, and the 11 kernel makings at once to 0.8 to 1.4 s each; a
  cold login on that host while it ran other work (a load of 31 at the login's start) made the
  fragment module in 4.87 s and the kernels in 1.5 to 4.2 s each, 10 or 11 at once. What is built
  around the driver's time:
  - **Each making once per driver.** A making's pipeline cache data is kept as its record
    (`shader_cache/driver.rs`), and a later process over the same content directory finds the
    pipeline in it: the consumer's warm boot of the 26.7.2 guest made the 203,016-word module's
    state in 3.1 ms. A record is named by the driver's identity (vendor, device, driver version
    and `pipelineCacheUUID`) besides the making's own content, so a driver update, another device
    or a removed content directory makes every making cold again; Vulkan's cache data is the
    driver's own and holds the compiled shaders, so no machine's records serve another's.
  - **Made again before the program asks.** A consumer that keeps the pipeline records
    (`DeviceOptions::pipeline_records`, descriptors and the libraries they name, not driver data)
    and hands them to a later process's `Device::warm` has that device make the same translations
    and predicted `VkPipeline`s on a thread of its own, below every other making
    (`warm/pass.rs`): after a driver update, the cold makings run there, not at the program's
    calls. A record is a state's creation, and the pass makes what a creation predicts; a compute
    state whose descriptor states no `requiredThreadsPerThreadgroup` predicts no pipeline, so
    each direct dispatch of a new shape whose pipelines its program made (a threadgroup, a
    grid's ragged edge, its threadgroup memory lengths: up to 16 a state) hands the hook one more
    record, and the pass makes that dispatch's pipelines too (`pipeline/dispatch_record.rs`,
    `WarmStatistics::dispatch_pipelines_warmed`), planned with nothing bound. A dispatch whose
    bindings specialize its pipelines otherwise (a bound texture's format a variant reads), and
    an indirect dispatch or one of a threadgroup range, still makes its own at its first
    dispatch.
  - **No call waits the whole making.** Under `PipelineWaitPolicy::Bounded` (the default) a
    creation waits `pipeline_creation_wait` (4.5 s) at most for its predicted making and returns
    with it in flight, a committed command buffer waits for its commands' makings until its commit
    plus `pipeline_commit_wait` (4.5 s) at most, and a dispatch hands its pipeline's making to the
    device's pool and goes on (the ledger's `pipeline_waits` row, `pipeline/dispatch_making.rs`).
  - **Off the guest's command path.** The consumer creates a guest's states on a compile thread
    of its own: in the 12.7.6 login above, the guest's context recorded 1 to 197 command buffers
    a second through the fragment module's 4.87 s making, and another state was created and made
    meanwhile.

  A making with no record still costs its driver's time once per module and driver: no
  unoptimized first form shortens it where the driver ignores
  `VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT`, and a state's pipeline is linked from its
  libraries with no link-time optimization already (the link takes 0.0 ms of the fragment
  module's 4.87 s).
- **On NVIDIA, a command buffer submitted while the runtime makes or frees a GPU pool block
  completes as that call ends.** A queue's first lease of an index or mesh records pool makes its
  block as the command buffer encodes where the device holds no spare of that kind, a block
  another queue let go (`gpu_pools.rs`): a device makes no more blocks of a kind than the most
  queues that held one at once, and none as queues come and go or rest. A block is a sixty-fourth
  of the device-local heap. On the RTX 5080 Laptop GPU (NVIDIA 615.71.09, Linux), measured with a
  Vulkan device of its own, `vkAllocateMemory` of such a block's memory takes about 0.16 ms a MiB
  whether the GPU is idle or another process's work runs (medians of 12: 5.5 ms for 32 MiB, 11.0
  for 64, 20.4 for 128 and 40.6 for the block's 254.7 MiB; beside the other work 5.5, 10.8, 21.7
  and 41.4) and `vkFreeMemory` about 0.035 ms a MiB (1.5, 2.6, 4.5 and 9.0 ms), and a
  `vkQueueSubmit` another thread calls during either returns only once it does (12 of 12 at each
  size). In the runtime a block's making took 3.1 to 35.7 ms beside another process's GPU work,
  and a `vkQueueSubmit` the scheduler's thread called meanwhile returned 0.01 to 0.25 ms after the
  allocation did (25 of 40 rounds of a probe). The runtime watches a batch for its completion from
  its submission's return on, so a command buffer submitted then completes, and runs its completed
  handlers, as the making ends, and the submissions behind it on the scheduler's thread wait as
  long. Frames submitted every 16.7 ms were late at each making of 254.7 MiB by 28.6 to 48.6 ms
  with the making on the frame's own thread before its submission (23 of 23 over 1 ms, alone and
  beside the other work), and still by up to 67.5 ms with it on another thread at a moment spread
  over the frame or right after a frame's submission (11 of 23 over 1 ms either way): a block made
  ahead of its lease on a thread of the device's own would hold the frames it meets as long, so
  none is. A smaller block would shorten the wait in proportion and hold fewer indices and records
  for each command buffer. Metal's command buffers wait for no such making. Not announced.
- **On NVIDIA, a command buffer submitted while the runtime makes or frees a memory block completes
  as that call ends; emptied blocks are kept so that a returning working set makes none again.**
  The runtime's memory blocks (`memory.rs`: pooled blocks of 64 MiB, images' included (the row on
  plain blocks below), blocks of their own for placements past half a pooled block, and the blocks
  exportable buffers share) are made with
  `vkAllocateMemory` and freed with `vkFreeMemory`, which hold every submission on NVIDIA's driver
  (the row above has their times). A block that empties is kept for the next placements: past its
  pool's two pooled ones, while the empty blocks kept in its heap stay within an eighth of the
  heap, at most 2 GiB (`KEPT_EMPTY_BYTES`); a kept block of its own takes the next placement of its
  pool at most a quarter shorter. Each goes once it stayed empty for 10 s (`KEPT_EMPTY_FOR`), but
  one pooled block per pool, at a release in its pool or within a second at the device's next
  creation call, and every kept block of a memory type goes before a block of that type fails to be
  made. The cost is that memory, held from a working set's dip until 10 s after it, which other
  processes cannot use meanwhile: up to 2 GiB in each of the RTX 5080 Laptop GPU's heaps (its 16
  GiB of VRAM and its system heap), 512 MiB on a card of 4 GiB, 2 GiB across a Radeon 780M's
  heaps of 11 and 5.5 GiB (RADV), whose unified memory makes them one budget (`memory.rs`'s
  `kept_budget`), and 2 GiB, wired, in the M3 Max's one heap. No
  switch: no answer of Metal's changes. On the RTX 5080 Laptop GPU (NVIDIA 615.71.09), a clip
  start's burst (`tests::clip_bursts`: 3062 buffers, 1175 textures and 34 `Private` exportable 4K
  NV12 pictures made at once while another thread commits a fill every 2 ms, three clips 3 s apart,
  two runs each) took 41 and 40 allocations again for the second and third clips where a pool freed
  every emptied block past its two and each picture had an allocation of its own, freed with it, in
  89 to 102 ms, and 2 to 5 of the 37 to 41 commits meanwhile waited 1 ms or more to be scheduled
  (longest 8.8 to 9.4 ms); with the blocks kept they took none, in 64 to 71 ms, and no commit
  waited 1 ms (longest 0.39 to 0.67 ms), 16 empty blocks (1 GiB) kept between the clips. A first
  clip still makes its blocks: 16 allocations, against 44 when each picture had an allocation of
  its own, with 3 to 5 commits waiting 1 ms or more (longest 10.9 to 19.6 ms): its blocks'
  `vkAllocateMemory` holds every submission. A pool that grew, and one near its next block (it
  holds a pooled block and lacks room for three quarters of one, its first growth among them),
  gets one block made ahead (`memory/ready.rs`; the pools of exportable buffers, which hold the
  pictures, among them), one `vkAllocateMemory`, right after a burst of the device's submissions
  ended. It is counted in bytes with the pool's free room: made again once the block made ahead
  and that room come to less than a block, a placement of a quarter of a block or more starting
  its making at once rather than at the next creation call. A creation call that needs one while
  its making still waits for that end makes it at once and waits for it, as for a block made in
  place, and one that finds none made waits for the ready thread to make it first. A pool near its
  next block keeps that block at rest for as long as it stays so, 64 MiB a pool at most, counted
  against the kept empty blocks' budget. A heap under 1 GiB (an NVIDIA GPU's 246 MiB
  device-local host-visible heap without Resizable BAR, its BAR) is the exception, so that a
  device takes none of it that no resource asked for: its pooled blocks are the largest power of
  two within a sixteenth of it (8 MiB there; `memory::pooled_block_bytes`), none is made before a
  placement needs it (no room kept at creation calls, none made ahead), and the placement that
  finds no room makes its block itself, on its own thread, an encoding's or a commit's included
  (the texture and sampler tables of heap sets, at their first use, are the encodings' own
  placements there). So the same placements there make about eight times as many blocks, each a
  `vkAllocateMemory` that holds the device's submissions, and none is made ahead of its need.
  On an RTX 2080 with 256 MiB of BAR1 (NVIDIA 620.70.06), a device and its first command queue
  take 2 to 3 MiB of it (seven devices of one process took 18 MiB), where a block of 64 MiB
  made for room took 67 MiB a device and the heap held three. Metal's command buffers wait for no
  allocation. Not announced.
- **On NVIDIA, each `vkFreeMemory` holds the device's other submissions while it runs, an import
  of host pages' included, and a program's own release of a no-copy buffer frees its import
  unpaced.** On an RTX 3090 (NVIDIA 620.70.06, Linux), `tests::import_free_hold_bench` commits a
  one-word fill every 250 us on one thread and waits for it, while another thread frees runs of
  32 imports of 512 KiB of host pages by their pointer, four of 64 MiB (a buffer in fresh pages),
  32 dma-buf imports of 512 KiB (udmabuf, each dedicated to a buffer as a no-copy buffer's is),
  and 32 plain host-visible allocations of 512 KiB and four of 64 MiB, eight runs of each. The
  fill took 0.13 to 0.14 ms from commit to completion (median, two runs) beside no free. Beside
  frees one right after another it took 0.46 to 0.50 ms beside the 512 KiB pointer imports (each
  free 0.26 to 0.28 ms), 0.47 to 0.54 ms beside the dma-buf imports (0.27 to 0.39 ms), 2.6 to 2.9
  ms beside the 64 MiB imports (2.2 to 2.4 ms), 1.2 to 1.6 ms beside the plain 512 KiB
  allocations (1.1 to 1.2 ms) and 70 to 79 ms beside the plain 64 MiB ones (28 to 30 ms): a
  submission that meets a run of frees waits for more than the one it meets. Paced, each free
  after a wait as long as the frees before it took (`step_aside::pace`), it took 0.29 to 0.32,
  0.23 to 0.25, 0.10 to 0.13, 0.17 to 0.22 and 0.15 to 0.19 ms, the runs taking 1.2 to 2.2 times
  as long; one free still holds a submission that meets it for its whole length (p99 2.6 to 3.0
  ms beside the 64 MiB imports, 18 to 22 ms beside the plain 64 MiB allocations). An RTX 2080 on
  the same driver, on a host loaded by other jobs, freed the 512 KiB pointer imports in 0.10 to
  0.12 ms and the dma-buf ones in 0.69 to 0.81 ms (median), and pacing lowered the fill's median
  beside every kind's runs there too. A probe outside the runtime that freed 32 pointer imports of
  512 KiB in turn beside one fill submitted 4,800 times a second measured 0.52 to 0.90 ms a free
  on the RTX 3090, and the fills' latency (median) rose from 0.12 to 3.9 and 8.3 ms with the GPU
  otherwise idle, and from 0.74 to 11.4 and 12.7 ms beside 30 ms command buffers on another queue.
  The runtime's own threads pace their runs of frees so: the free thread (its blocks, a buffer's
  fresh pages among them), the residency thread's releases of idle imports, a guest RAM's
  give-backs and idle releases, and the submission checks' import tax (`step_aside.rs`,
  `WorkCounts::memory_frees_paced`). A program's own release of a no-copy buffer frees its import
  in the release, as Metal lets the buffer go there, so a program that releases many at once holds
  the device's other submissions for as long as their frees take together; so does a completed
  command buffer's handler that frees the imports its blits held last, unpaced, as a wait there
  would delay the completed handlers behind it. Not announced.
- **With unified memory, `Private` memory lies in the system heap's types before the device-local
  ones, its buffers' blocks there allocated exportable but for buffers that declare their padding
  unobserved: about 1 µs a submission for each such block held.** Where the device has unified memory and a heap that is not
  device-local, `Private` and `Memoryless` placements take that heap's types first (`memory.rs`'s
  `candidates`; `Memoryless` its lazily allocated types before all): RADV backs a Radeon 780M's
  device-local types from the 2 GiB carve-out the kernel reserves for every process's GPU memory,
  and once that is full the kernel evicts to make room during the submissions of every process
  holding memory there (the consumer's scroll: the submitting thread 2.28 s in
  `drm_exec_lock_obj` over 39 episodes), while `VK_EXT_memory_budget` reports the heap's budget of
  11 GiB, not the carve-out's, so no budget keeps placements within it. On a Radeon 780M (RADV
  PHOENIX, Linux), with 1.6 GiB of the carve-out held by other processes: copies of 256 MiB ran at
  23.9 and 28.8 GB/s (medians of two rounds) in the device-local type and 28.5 and 29.0 GB/s in
  the system heap's; a scroll of 259 tiles (2.3 GiB of the runtime's blocks) beside 1 GiB of
  another allocation's device-local memory held median frames of 1.31 and 1.29 ms (frames 300 and
  600) with its placements device-local and 1.87 and 1.30 ms in the system heap's types, and the
  same scroll alone 2.87 and 4.87 ms against 1.33 and 1.31 ms; no eviction traced in any
  (`amdgpu_bo_move`). The system heap is the smaller of the two (5.53 GiB), and with a budget of
  kept empty blocks per heap, its eighth (707 MiB) kept all but 3 of a video clip's 16 blocks
  (`tests::clip_bursts`): each later clip made 4 again and took 237 and 245 ms to make where it
  had taken 132 and 126 ms with the placements device-local; with unified memory the heaps keep
  one budget of an eighth of them all (the row above), which keeps every block: the later clips
  made none and took 122 and 119 ms. There the blocks of the runtime's buffers (the program's,
  those of buffer descriptors and its own; not a heap's, nor a buffer that declares its padding
  unobserved, `BufferDescriptor::padding_unobserved`, which lies in a plain block wherever, as a
  texture over it whose rows lie apart may take a `DRM_FORMAT_MOD_LINEAR` image at Metal's pitch
  there, which RADV makes over exportable memory only as a dedicated allocation) are allocated
  exportable as an opaque file descriptor and never exported (`memory.rs`'s "Buffers in
  exportable blocks"), every image over one the device's linear image made for it, where its
  policy takes textures in place by its check and that check of such blocks gave every texture
  over one what it takes over a plain block's buffer (`in_place::fd_blocks_take`,
  `texture/fd_blocks_cover.rs`, the in-place row above; plain blocks until it answered,
  `WorkCounts::fd_blocks_awaited`, the verdicts from the check's record or from a check a texture
  over host pages or `Device::settle_in_place_checks` started, a buffer's making starting none):
  on a Radeon 780M (RADV PHOENIX, Mesa
  26.2.4) a plain block's `vkAllocateMemory` holds every submission of the process while the
  kernel fills its pages, under the reservation the process's buffer objects share, and an
  exportable block, a buffer object of its own reservation, holds none. With a 64 MiB block of
  the system heap made about every 0.8 s beside a command buffer committed about every
  millisecond (`tests::reclaim_submit_bench`), 11 of 7050 turnarounds took 8 ms or more on a
  quiet host and 12 of 13701 on one whose free memory lay in single pages (2 of 17 ms or more,
  the slowest 63.9 ms, the scheduler's thread waiting 305 ms on that reservation in all), against
  0 of 7153 and 0 of 14266 with the blocks exportable as an opaque file descriptor (the slowest
  2.7 and 4.5 ms, 2.6 ms waited). Through the runtime (24 `Private` buffers of 32 MiB a phase,
  its pools growing by 64 MiB blocks made ahead; two runs each), the base took 13 and 12
  turnarounds of 8 ms or more on the quiet host and 103 and 17 on the fragmented one (19 and 4 of
  17 ms or more, the slowest 40.4 and 49.0 ms, the scheduler's thread waiting 350 and 311 ms on
  the kernel's lock); with the check's verdicts known first, as from its record, the blocks
  exportable and none in either phase (the slowest 7.1 ms, 4 and 10 ms waited). Until the
  verdicts are known the blocks are plain, as the base's. In the arm whose 64 `Private` buffers of 32 MiB were made
  before the phases, in plain blocks, the first turnaround took 57.0 ms and none after it 1.4 ms;
  with 64 exportable blocks held, the first took 2.4 to 8.1 ms. The trade: each such block held
  costs every
  submission, which names each, about 1 µs: with 64 held, `vkQueueSubmit` took 0.175 and
  0.168 ms where it took 0.105 and 0.109 ms with none, and a turnaround's median 0.341 ms
  against 0.294 ms. Two queues over one such allocation run beside each other as over a plain
  one (a fill of 96 MiB eight times on a compute queue, a copy of 64 KiB on the graphics queue a
  millisecond later: 96 of 100 copies completed while the fills ran, against 97 and 98). A driver
  that treated such memory as its private memory would behave as today plus that cost. Plain
  blocks remain, their making holding the submissions as before, for as long as the next row
  lets it (`WorkCounts::plain_buffer_blocks_made`): those of the buffers that declare their padding
  unobserved, of images and of heaps, and of the few buffers the runtime places apart. A memory
  type whose buffers lie in both kinds keeps a pool of each, each keeping a block partly filled
  or empty (the row above), so up to 64 MiB more stays held than with one; the exportable pool
  keeps its block first, and only while the runtime's buffers lie there. A two-plane
  texture's planes over a buffer in such a block are read through the texel buffer, not an image
  of the device's conversion (`WorkCounts::ycbcr_planes_images_kept_exportable_block`): the
  check covers single-plane images alone; over a buffer that declares its padding unobserved,
  in a plain block, they take the conversion image as before. Devices whose
  every type is device-local (ANV, MoltenVK, lavapipe) and those without unified memory place as
  before, in plain blocks. No switch: no answer of Metal's changes. Not announced.
- **On RADV, a command buffer submitted while the runtime makes a block of plain memory waits for
  that making; pooled blocks stay 64 MiB.** A block allocated with no export (`memory.rs`: the
  blocks of images, of heaps, of buffers that declare their padding unobserved, and of every
  buffer where the device's buffers lie in plain blocks; not the runtime's own pages, which it
  imports) shares the reservation of the process's buffer objects on RADV, and its
  `vkAllocateMemory` holds every submission of the process while the kernel fills its pages (the
  row above). `amdgpu_gem_create_ioctl` reserves the root page directory of the process's GPU
  address space for such a buffer object (`AMDGPU_GEM_CREATE_VM_ALWAYS_VALID`) and makes the object
  on that reservation, and `ttm_bo_init_reserved` fills its pages before the root is released
  (`ttm_bo_validate`, `ttm_tt_populate`, `ttm_pool_alloc`; Linux 7.3.3, `amdgpu_gem.c`,
  `ttm_pool.c`). The pages come from the page allocator in runs of up to 4 MiB, and a run of 2 MiB
  (order 9, the order amdgpu names beneficial) is asked with `__GFP_RETRY_MAYFAIL`, so it may
  reclaim and compact memory until it succeeds or fails. Every kernel submission of the process,
  and every buffer object it makes, opens or imports, locks that reservation and waits meanwhile:
  beside a consumer's guest on a Radeon 780M whose host was short of free memory (RADV PHOENIX,
  Mesa 26.2.6), one making held the submissions 228 ms, 222 ms of it one compaction for one 2 MiB
  run. Every block of 2 MiB or more takes such runs, so no block size bounds the hold below one
  run's, and an image binds one memory object, so an image of 2 MiB or more takes them whatever
  the blocks. A smaller allocation moves the hold into direct reclaim under the same reservation:
  its runs of more than a page are asked with no reclaim, so on a host whose free memory lies in
  single pages it takes them a page at a time, reclaiming as it goes. On a Radeon 780M
  (`tests::reclaim_submit_bench`, 768 MiB of allocations a phase, two runs of each arm, the
  host's free memory in single pages), allocations of 1 MiB scanned 40,438 and 66,941 pages in
  direct reclaim a phase and the host's tasks all stalled on memory 3.5 and 1.7 s; allocations of
  4 MiB scanned 252 and 2,147 pages, stalled 1.1 and 1.0 s; with nothing made, 4,132 pages and
  0.8 s. The 4 MiB allocations held 33 and 0 turnarounds of 8 ms or more (the slowest 23.5 ms, the
  slowest making 8.0 ms, far short of the guest's), the 1 MiB ones none (the slowest 6.0 ms), so
  the measurement shows no long hold that smaller allocations remove. A 1 MiB allocation took a
  median of 0.4 to 0.6 ms, and holding many costs a submission nothing: with 2,752 of 1 MiB held,
  `vkQueueSubmit` took 0.096 to 0.110 ms, with 43 of 64 MiB 0.096 to 0.109 ms, with none 0.097 to
  0.107 ms. Pooled blocks are 64 MiB on every device, images' included. A driver's image takes
  more than its texels (RADV on a Radeon 780M and ANV on an Iris Xe place a `Private`
  `RGBA8Unorm` texture of 1024x1024 in 4 MiB and 16 KiB, and one of 2048x2048 in 16 MiB and
  64 KiB, at an alignment of 64 KiB), so a block of images holds three such 2048x2048 textures
  there, not four (`tests::blocks_ahead`): one block size for every pool, at the cost of a
  fourth such texture's block. On a Radeon 780M (RADV PHOENIX, Mesa
  26.2.6), the kernel's page pools emptied before each run (through debugfs, the measurement's
  method alone: with freed pages in them a block of 16 to 64 MiB took 0.1 to 0.3 ms), a `Private`
  texture made every 16.7 ms beside a command buffer committed about every millisecond
  (`tests::reclaim_submit_bench`, two runs of each arm), a 64 MiB making took 10 to 12 ms on a
  quiet host and 10 to 31 ms on one whose free memory lay in single pages, 90 to 96% of it on a
  CPU inside `vkAllocateMemory` (the kernel's page allocation, clearing and compaction; its
  mapping 0.02 to 0.1 ms), and each held the submissions as long: with textures of 1024x1024, of
  mixed sizes and in bursts, 12 to 14 turnarounds of 8 ms or more a phase (up to 5 of 17 ms or
  more, the slowest 32.1 ms), 13 to 15 makings a phase holding 125 to 201 ms in all. Smaller
  blocks shorten each hold and add makings: with blocks of 17 MiB, 0 to 8 turnarounds of 8 ms or
  more a phase (none of 17 ms, the slowest 14.5 ms), 45 to 55 makings a phase holding 138 to
  230 ms in all, and each texture past 17 MiB in a block of its own. A block's own allocation
  is its only one: the runtime does not first fill its pages through an exportable allocation of
  its size, freed at once. On a Radeon 780M under memory pressure most such fills missed (the
  kernel handed the block other pages), so that the kernel compacted for both allocations: in a
  scroll of a consumer's guest, every frame gap of 100 ms or more lay within one block making's
  direct compaction (46 to 164 ms), and with no fill there were none. No other allocation avoids
  the hold for these blocks. RADV makes every memory neither exported nor imported a buffer
  object of the process's reservation, a dedicated allocation (`VkMemoryDedicatedAllocateInfo`)
  among them (Mesa 26.2.6, `radv_device_memory.c`). An image bound to exportable or imported memory
  names its handle type (`VUID-vkBindImageMemory-memory-02728`,
  `VUID-vkBindImageMemory-memory-02989`), RADV requires a
  dedicated allocation of every image made for an external handle type in other than linear
  tiling (`radv_device.c`), and such an image binds only the memory made for it
  (`VUID-vkBindImageMemory-image-01445`), which a host pointer's import may not be
  (`VUID-VkMemoryAllocateInfo-pNext-02806`). So a block that holds images of optimal tiling, or the
  `DRM_FORMAT_MOD_LINEAR` images over buffers that declare their padding unobserved, can be
  neither exportable memory nor imported pages, and an image of optimal tiling lies in no host
  pointer's import at all. A dedicated exportable allocation per image would add each image to
  the buffer objects every submission names (RADV lists every one not of the reservation in each,
  `radv_amdgpu_bo.c`, whatever the device enables), about 1 µs a submission each on a Radeon 780M
  (the row above); buffers in a host pointer's import cost every submission 24.1 µs per 64 MiB
  held there, the kernel getting their pages again in each (`submit_check.rs`). Its device-local
  types lie in the carve-out while it has room and in system memory past it, and Vulkan reports
  neither the carve-out's size nor its room (the row above), so no placement there is decided from
  the device's answers. No switch: no answer of Metal's changes. Not announced.

- **Making and releasing an event on NVIDIA take 0.1 to 0.6 ms of the calling thread.**
  `new_event` and `new_shared_event` each make one timeline semaphore (`vkCreateSemaphore`,
  `event.rs`), and an event's release destroys it where it goes. On the RTX 5080 Laptop GPU both
  grow with the semaphores live: made and released back to back, one or two live, a median of 98
  to 119 µs each; 1,000 made, then released, 143 to 262 µs a making and 495 to 579 µs a release.
  Through MoltenVK 1.4.2 on an M3 Max a making takes 2.0 to 4.6 µs, and Metal's own `newEvent`
  and `newSharedEvent` 0.8 to 1.9 µs. The release does not post its destroy to the device's free
  thread (`free_thread.rs`): NVIDIA's driver runs a making behind another thread's destroy, so
  there a posted destroy took a making and release back to back from 203-238 µs to 390-507 µs.
  Not announced.
- **An intrinsic declaration from macOS 12's and 13's Metal compiler** whose type fits none of the
  forms `intrinsics::classify` derives from its name is refused with
  `InvalidInput(Intrinsic(SignatureMismatch))` (`intrinsics::bind`). AIR 2.5 (Metal compiler
  31001.723) declares the indirect command setters `air.set_kernel_buffer_compute_command` and
  `air.set_vertex_buffer_render_command` without the attribute stride, which decode takes as MSL's
  plain setter; whether the refused kernels are those setters, and so translate, is unverified.
- **A direct call whose arguments do not line up with its callee's parameters** is refused with
  `InvalidInput(Canon(CallTypeMismatch))` (`canon::post_check`): LLVM leaves such a call undefined,
  and what Metal's back end makes of it is not documented, so no exact form is known.
- **A loop that sums its counter fails to make through MoltenVK.** Apple's MSL front end rewrites a
  loop whose running sum adds its 32-bit counter (`1 + 2 + ... + n`) into a closed form in 33-bit
  integers, which the M3 Max's back end cannot legalize where it lands in the kernel's constant
  program: MoltenVK, whose SPIRV-Cross writes such a loop in that shape, fails the pipeline's making
  (`XPC_ERROR_CONNECTION_INTERRUPTED`), and the runtime returns the failure
  (`CommandBufferError::PipelineCreation` for a dispatch, or `Error::Vulkan` at an indirect
  command's commit; a draw whose pipeline it made after the draw returned runs nothing,
  `CommandBuffer::dropped_draw`).
  Metal runs the AIR, which gets no such rewrite (case `authored-four-iteration-loop`). No form of
  the loop is known that avoids the rewrite for every such loop.
- **An anisotropic sampler without a mip filter whose min and mag filters differ, through Metal's
  own sampler (MoltenVK).** Vulkan has no sampler without mipmapping, so the runtime hands MoltenVK
  a nearest mip filter with a level clamp: `[0, 0]` where the two filters agree, which is Metal's
  `NotMipmapped` bit for bit, and `[0, 0.25]` where they differ, which chooses Metal's filter, but
  at `maxAnisotropy` 2 to 16 no clamp reads Metal's `NotMipmapped` answer. Announced as
  `NotMipmappedAnisotropicSampler` for a sampler state and `NotMipmappedAnisotropicFilter` for a
  constexpr sampler.
- **Dynamic texel offsets past 512 arms in one function, on a device without the `Offset`
  operand.** Without `maintenance8` (a sample) or `shaderImageGatherExtended` (a gather) a dynamic
  texel offset is a switch over the values Metal reads, each arm the access with that value as
  `ConstOffset` (16 arms a dynamic lane), exactly. A function whose accesses would take more than
  512 arms together is refused (`TexelOffset` or `GatherExtended`): a driver's making grows faster
  than the arms (on ANV 512 arms make in 0.63 s, 1536 in 5.9 s). Applying the offset in the shader
  is exact only where the shader filters, so a filtered sample past the bound has no exact form yet.
- **A fetched attachment's blend through the pixel interlock (RADV, ANV, NVIDIA, MoltenVK).** No
  Vulkan blend applies to the storage form's write, so the function computes the blend: the
  device's arithmetic for an extended-range attachment (the device-gaps table); Metal's for a
  16-bit float attachment; and for an 8-bit unorm one (`R8Unorm`, `RG8Unorm`, `RGBA8Unorm`, `BGRA8Unorm`) of a
  four-lane output under any factor, operation, write mask and blend color, and for an 8-bit sRGB
  one the device renders (`R8Unorm_sRGB`, `RG8Unorm_sRGB`, `RGBA8Unorm_sRGB`, `BGRA8Unorm_sRGB`)
  the same blend over the decoded texel, encoded by the device's conversion; the generic `float` blend,
  announced (`UnmeasuredFetchBlend`, the row above), for every other format whose channels read as
  floats, a 32-bit float one's included, and for a blend of the second source (an sRGB one's in the 8-bit unorm blend's rounded
  forms, as are a one- or two-channel sRGB one's fast-math lerps and an output of fewer than four
  lanes). Open: the 16-bit normalized, 10-bit unorm and 8-bit snorm formats, which a source-over
  survey on the M3 fits to the 8-bit model (with their clamps) but whose factor sets are not
  measured.
- **A blended signed normalized attachment on lavapipe.** lavapipe blends no snorm format
  (`R8Snorm`, `RG8Snorm`, `RGBA8Snorm`, `R16Snorm`, `RG16Snorm`, `RGBA16Snorm`: no
  `COLOR_ATTACHMENT_BLEND`), so a pipeline that blends one is refused as the device lacking the
  format. The function's own blend over the attachment's input attachment, as an extended-range
  attachment takes there, is not built: Metal's snorm blend is measured for source-over alone
  (the row above).
- **Format-less storage reads on a device without `shaderStorageImageReadWithoutFormat` (ANV).**
  There a read without a format goes by the format's own `STORAGE_READ_WITHOUT_FORMAT` bit
  (Vulkan 1.3 or `VK_KHR_format_feature_flags2`), which the 16-bit packed formats,
  `B8G8R8A8_UNORM`, `A2R10G10B10_UNORM_PACK32`, `A2B10G10R10_UNORM_PACK32` and
  `B10G11R11_UFLOAT_PACK32` lack in the committed ANV profiles: a fetch through the pixel interlock
  of a 16-bit packed color attachment (`B5G6R5Unorm`, `A1BGR5Unorm`, `ABGR4Unorm`, `BGR5A1Unorm`),
  which no SPIR-V format names, is refused (`FramebufferFetch`). A read of an argument buffer's
  writable texture is refused there whatever its format (`ArgumentBufferTextures`): the heap
  views its members for the accesses the device's features grant, for every pipeline alike, and a
  check of each member's bit against the pipeline's heap reads is not built. A fetch through a
  16-bit unsigned view that packs the channels in the shader, and heap storage classes typed per
  format, are not built either.
- **The formats from 630 to 645 under macOS 27.0.1.** The macOS 27.0.1 level answers as an Apple M3
  Max, which describes none of them (`BGRA10Unorm`, its packed, sRGB, PQ and HLG forms,
  `R10UintX6`, `RG10UintX12` and the rest): a texture of one is stopped and the linear alignment
  query refused, as on the M3 Max. An Apple M4 under 27.0.1 describes and makes them; no level the
  runtime has does.
- **Tessellation's triangle order.** On a device with `tessellationShader` but MoltenVK on an
  Apple GPU, the device's tessellator orders a patch's triangles, picks their first corners and
  joins its rings itself, announced as `TessellationTriangleOrder`
  (`tessellation-triangle-order`): every patch differs on the RTX 3090, 55-95% on RADV and
  lavapipe.
- **lavapipe aborts on the owned fixture `mesh_order`'s mesh functions.** Compiling
  `mesh_order_mesh` or `mesh_order_solo` ends the process inside lavapipe's compiler ("LLVM ERROR:
  Cannot emit physreg copy instruction"); the validation case `mesh_order_mesh-payload-mesh-draw`
  runs on a GPU only. The construct behind the abort is not isolated yet.
- **A record the GPU wrote before its ID changed holder, in a buffer the GPU wrote again since.**
  A record the GPU wrote names the state that held its ID as the GPU last could write the
  buffer's records (`released_programs.rs`). Where a command buffer wrote the record, another
  state took its ID, and a later command buffer that may write the buffer completed without
  rewriting it, the record runs and answers the later holder, where Metal's runs the state its
  write resolved. No note announces it. The exact form needs the GPU's write to carry its
  resolution: a word of the record the translated setter fills from a per-dispatch stamp of the
  device's IDs, which the host step, the getter and the GPU-alone preprocess resolve.
- **A private logic operation other than copy and no-op outside the forms built.** On a device
  without `logicOp`, the fragment function computes the operation through framebuffer fetch
  (MoltenVK's pixel interlock, or rasterization-order input attachments), but a pipeline is
  refused (`PrivateRenderField::LogicOperation`) where an attachment is `RGB10A2Uint` or a packed
  16-bit format (whose Metal operation is not a per-channel rule of either width), color 0 is
  operated under alpha to coverage, an integer output is narrower
  than 32 bits, or, through an input attachment, the attachment holds its channels rotated or
  Metal's alpha in red; on a device that orders fetch in neither form, every such pipeline. On
  any device it is refused where an attachment is sRGB, a float attachment blends under Vulkan's
  operation, a format's operation was not measured, or the function writes an attachment itself
  beside Vulkan's operation (a storage attachment, a shadow stencil, an imageblock or a merged
  word) or the function's (a shadow stencil, an imageblock or a merged word).
- **Vulkan's logic operation on a 10-bit or signed normalized format's own codes.** Metal
  operates on an `RGB10A2Unorm` or `BGR10A2Unorm` channel's codes widened to 16 bits and narrows
  the result back, and reads a signed normalized channel's lowest code (`0x80`, or `0x8000`) as
  the code above it and stores a result of the lowest code as the code above (probe
  `logic_operation` on an Apple M3 Max, macOS 27.0.1), as the fragment function's operation
  does; Vulkan's `logicOp`, which a device that has it draws with, operates on the stored codes
  themselves and stores its result as it is, so such a draw can store another code (the
  logic-operation lane on lavapipe: its xor and equivalence draws into `RGB10A2Unorm` store the
  per-channel texel, xor's green 938 where Metal's rule gives 937; its xor of `0xe0` over an
  `RGBA8Snorm` channel's `0x80` stores `0x60` where Metal's gives `0x61`). Vulkan's operation is
  the device's native form there and stays: the pipeline's fragment stage records
  `PrecisionNote::NativeLogicOperation` (`native-logic-operation`), which the runtime announces,
  wherever Vulkan's `logicOp` operates on such an attachment the function writes. A signed
  channel the draw leaves unwritten keeps its lowest code in either form, as any draw's
  (`snorm-minimum-kept`). An integer output past the channel's range, which Metal saturates
  before the operation, is a value Vulkan leaves undefined ("Any value that cannot be represented
  in the attachment's format is undefined", the Fragment Output Interface); the lane's
  out-of-range draws on lavapipe and on an RTX 3090 store Metal's saturated codes, so no note is
  recorded for it.

- **A reset compute pipeline descriptor keeps its thread shape on Metal and not here.**
  `MTLComputePipelineDescriptor.reset` restores the label, the buffer mutability,
  `supportIndirectCommandBuffers`, `supportAddingBinaryFunctions` and `maxCallStackDepth`, and
  leaves `threadGroupSizeIsMultipleOfThreadExecutionWidth`, `maxTotalThreadsPerThreadgroup` and
  `requiredThreadsPerThreadgroup` as the program set them (measured on an Apple M3 Max, macOS
  27.0.1; the tile and mesh descriptors' resets restore every field). The runtime's counterpart is
  `ComputePipelineDescriptor::new`, a fresh value that resets all nine, so a consumer that resets
  loses the thread shape where Metal keeps it. Nothing announces it.
  `compute-descriptor-reset` holds it.

- **Blending into an integer-format attachment is not compared, because Metal ends the process.**
  `-[MTLRenderPipelineDescriptorInternal validateWithDevice:]` fails an assertion and the process
  ends on SIGABRT — `Blending is enabled for render target 0; however, the pixelformat
  MTLPixelFormatR32Uint for this render target is not blendable` — measured on an Apple M3 Max,
  macOS 27.0.1 for `R32Uint`, `RGBA8Uint` and `RGBA8Sint`, with blending **off** on the same
  format and blending **on** over `RGBA8Unorm` creating normally as the two controls. So Metal has
  no answer here to be conformant with and no case may ask for one. What the translator does with
  the neighbouring shape is compared: alpha-to-coverage over an integer attachment 0 is a typed
  `InvalidDescriptor` (`reads_alpha_to_coverage`), which is Metal's own pipeline error.

- **`rasterSampleCount` disagreeing with the attachment's sample count is not compared.** Plain
  Metal creates the pipeline and draws something that looks undefined — a 1-sample pipeline into a
  4-sample attachment left samples 1 to 3 at neither the clear nor the fragment — while Metal's
  API validation layer ends the process at `setRenderPipelineState:` (`the texture sample count
  (4) does not match the renderPipelineState colorSampleCount (1)`), both measured on an Apple M3
  Max, macOS 27.0.1. A plain-Metal answer that looks undefined is not one to pin as a conformance
  expectation, and a case of it would end a recording run.


- **A blend's result on RADV, by 1 to 64 codes** (`half-output-blends-*`, `blend-sweep-operations`,
  `blend-interaction-format-unorm8` and `-srgb8`, `argb8-targets-*`, `a2c-on-*`; 24 cases). The
  one-code rows are what a driver's own rounding of a normalized blend produces, which Vulkan does
  not pin as closely as Metal does, and those are a candidate for an announced precision note rather
  than a fix. The largest, 64 codes at `blended/sas-1msa`, is **too large for rounding** and is a
  separate open question.

- **One line texel at a diamond vertex on the RTX 5080** (`primitive-type-restart-indices`): all
  four `Line` and `LineStrip` cells differ from the M3 Max's recording by two texels on an RTX 5080
  Laptop GPU (615.71.09), and by none on llvmpipe (LLVM 23.1.1, 256 bits). The line of vertices 0
  and 1 runs exactly through (3.5, 1.0), a vertex of the diamonds of the two pixels either side of
  it: Metal draws the lower one and this driver draws the upper one. Vulkan's Bresenham diamond is
  open (`|p - x| + |q - y| < 0.5`), so a line that only touches a vertex exits no diamond and
  neither pixel is required; both choices conform and there is no Vulkan control over the tie, so
  this is the driver's and not ours. Announced (`device-line-ties`) at each draw the device's lines
  draw at width 1 on a driver without `DriverFact::LineTiesAreMetals` (MoltenVK on the M3 Max has
  it). lavapipe (Mesa 26.2.4) draws this case's line as Metal does but leaves out the first pixel
  of the line lane's strip, which starts on such a vertex, so it announces too; RADV and ANV unrun,
  so announced there.

- **A line band's endpoint half a subpixel step off the grid off the M3 Max**
  (`runtime_lane::line_widths`, the lines offset from rows and columns of centres by
  `(r - 32) / 512`): the band leaves its corners' snap to 1/256 pixel to the device's rasterizer,
  where Metal snaps the endpoints first, a half step rounded up. NVIDIA's 615.71.09 on the RTX 5080
  Laptop GPU and lavapipe (Mesa 26.2.4) draw those lines a row lower than Metal's, as a half step
  rounded down gives. Vulkan leaves that rounding to the implementation. Announced (`device-subpixel-snap`) at
  each band draw on a driver without `DriverFact::SubpixelSnapIsMetals` (MoltenVK on the M3 Max has
  it); RADV and ANV unrun, so announced there.

- **Lines under Apple's `setLineWidth:` at a width other than 1 that neither route draws as Metal**
  (`render/line_width.rs`). Drawn by the device's own lines at the nearest width they draw (1 pixel
  without `wideLines`), announced `lines-at-nearest-width`: a mesh draw's lines and edges and a
  tessellated patch draw's edges where the device's own lines do not draw the width (every such draw
  on MoltenVK), lines and edges on a device without `tessellationShader` whose lines do not draw
  them or of a pipeline state whose band variant the translator does not build (a vertex function
  that selects a slice or viewport, barycentric weights, synthesized geometry, multiview), a draw
  whose own band variant the translator refuses where the state's first band compiled (a varying or
  push data the draw's variant adds past the evaluation stage's outputs or `maxPushConstantsSize`)
  and the state's later draws of its primitive, an amplified band, and any wide line in an execution
  or a rasterization-rate-mapped pass. A band of a triangle's edges takes the device's depth bias of
  each band triangle, where Metal biases by the triangle's slope and largest depth exponent:
  announced `wide-edge-depth-bias` under a slope factor or a constant over a float depth. Refused
  (`RuntimeGap::WideLines`): a strip's band whose list of primitives the runtime cannot draw. Not
  announced: the band is cut at the near and far planes where its triangles are, which can differ
  from Metal's clipped segment by part of a pixel there, and a triangle that crosses the near plane
  is drawn as its edges' visible parts, with no edge along the plane (Metal's unmeasured); what a
  line's band's fragments read (varyings and depth across the width) is not compared with Metal's;
  and at width 1 in a multisampled pass the runtime's lines are its ordinary ones (Bresenham's where
  the device offers them) where Metal draws the parallelogram at the sample positions (MoltenVK's
  are Metal's own).

- **A heap multisample texture's sample moves on a GPU of 20 GiB or less**: a command queue keeps
  the images such moves go through up to a sixty-fourth of the device's largest device-local heap
  (`gpu_pools::device_share`, `texture/copy_images.rs`). A 4096 x 2160 `Depth32Float_Stencil8`
  texture of 4 samples -- the consumer's surface -- needs 295 MiB of them per blit, over the
  255 MiB a 16 GiB GPU keeps, so every copy makes all 16 again (measured on an RTX 5080 Laptop
  GPU, 615.71.09; `heap_sample_moves` prints each copy's images made and lent, and its 256 x 256
  shape, 2 MiB of images, makes none after the first). The images are correct and bounded; the
  cost is the making. Raising the share, or holding the bound to at least one copy's images,
  would reach it.

- **A folded object grid's cost per Metal threadgroup**: a device that runs object functions
  through `VK_EXT_mesh_shader` takes one task workgroup per Metal object threadgroup
  (`bind/mesh_arguments.rs`, the folded object form), and an RTX 5080 Laptop GPU (615.71.09) takes
  2.4 to 2.7 microseconds over each: 4,587,520 of `mesh_object_grid`'s threadgroups in 12.0 s and
  4,194,307 in 11.1 s. Metal's own GPU ran the 33.5 million-threadgroup grids of the same fixture
  inside its 5 s watchdog, about sixteen times faster per threadgroup, so those grids are past any
  bounded wait here and `mesh_draws::object_grids_run_as_metal_runs_them` now measures the device's
  cost and skips them, printing both. Nothing is wrong with the words the smaller grids leave; the
  cost is the dispatch.

- **A dma-buf import on NVIDIA**: the driver enables `VK_EXT_external_memory_dma_buf` and refuses
  to import a small udmabuf of host pages dedicated to nothing, `vkAllocateMemory` answering
  `ERROR_OUT_OF_DEVICE_MEMORY` (615.71.09, measured: one 4 KiB page and a 16 KiB window refused,
  with 15,291 MiB of the card free for 8 KiB; 64 KiB taken). It takes the same 4 KiB page
  imported dedicated to its buffer (`VkMemoryDedicatedAllocateInfo`), so an import whose memory
  backs one buffer alone for its life (a no-copy buffer's own import of one chunk, a copy's list
  of a guest RAM's runs, the page a RAM's use asks the driver with) is made so where the driver
  refuses the first form (`memory.rs`'s `DmaBufFor::Buffer`,
  `WorkCounts::dma_buf_buffer_imports_dedicated`). A guest RAM's chunk or an object's run inside
  one cannot be: its objects' buffers are bound in it, and Vulkan binds only the dedicated buffer
  in a dedicated allocation (`VUID-vkBindBufferMemory-memory-01508`). A chunk is 64 MiB or more,
  which the driver takes so (measured: a 64 MiB chunk imported dedicated to nothing on the RTX 5080
  Laptop, its RAM's first page dedicated to its buffer); an object's run alone of a few pages the
  driver may refuse so, and the object then takes the runtime's other paths (`host_ram.rs`'s
  `bind_in_chunk`). It lists a udmabuf of host pages as two memory types, 0 and 4 (host-visible,
  coherent and cached, the `Shared` class's), and refuses every import as 4 as out of device
  memory (620.70.06 on an RTX 2080 and an RTX 3090, measured). Once it refused an import as 4,
  620.70.06 refuses every later allocation of device-local memory exportable as an opaque file
  descriptor or a dma-buf, as out of device memory, for the device's life, where it made each
  before and makes them after a successful import as 0 (measured on an RTX 3090,
  `tests::export_after_dma_buf`): an exportable buffer then overflowed into system memory. So a
  dma-buf import tries the system-memory types the host maps (host-visible, not device-local)
  after every other type, in every form (`memory.rs`'s `dma_buf_candidates`; the runtime maps no
  dma-buf import), and NVIDIA's udmabuf imports never try type 4 where type 0 takes them.
  Device-local types keep their class's order, so ANV's, RADV's and lavapipe's imports are as
  before. 615.71.09 also refuses
  most udmabufs of scattered pages dedicated to nothing, of 16 KiB to 63 MiB, and takes them
  dedicated to their buffer (measured on an RTX 5080 Laptop GPU: 4346 of the leak probe's 6000
  imports). So an import tries first the form and type its kind (what it backs, and its class)
  last imported as (`memory.rs`'s `worked_first`), every form and type still tried after it, and
  the imports of a kind none has yet succeeded as run one at a time
  (`Memory::dma_buf_first_imports`). A failed import is where the driver's defects lie:
  - it **closes the descriptor it was handed**, which the Vulkan specification gives it only on a
    successful import: `F_GETFD` on the descriptor reads `EBADF` the instant the call returns
    (615.71.09 and 620.70.06). 620.70.06 at times closes the number twice within the one failed
    call (measured on an RTX 2080, eight threads importing: 3 of 10 failed imports), and the
    second close closed whatever another thread had opened under the number meanwhile. An import
    hands the driver a clone of the descriptor at a number kept from the process's other
    descriptors while the import runs (`handed_descriptor.rs`), so neither close reaches another
    descriptor, and after a failure closes the clone where the driver left it open. Past 64
    imports in flight at once, or where the process holds the 64 numbers below its descriptor
    limit (or below 65536) itself, the clone takes the lowest free number, and a failed import
    lets it go rather than close it: one descriptor per such failure.
  - with four or eight threads importing, a failed import at times never returns (620.70.06: 3 of
    7 four-thread runs on an RTX 2080 alone, and on an RTX 3090; 615.71.09: both probes in their
    first run on an RTX 5080 Laptop GPU with the type order alone, and 4 of 10 runs with no
    host-visible type tried, failed imports dedicated to nothing in every one): it spins inside
    `vkAllocateMemory` holding the driver's lock, the other threads' makings and frees waiting
    for it, the GPU idle. With no import as type 4 tried, 8 of 8 four-thread runs ended on the
    RTX 2080; with the type order, 8 of 8 on the RTX 2080 and 8 of 8 on the RTX 3090; with the
    form and type order and a kind's first imports one at a time, 6 of 6 four- and eight-thread
    runs on the RTX 5080 Laptop GPU, in 8 to 9 s each. The first
    imports of each kind on a device still try a refused form or type, one at a time, so the
    stall stays **open** for them.

  RADV's imports take the kept numbers too (measured on a Radeon 780M, Mesa 26.2.6, by the
  dma-buf probes). A refusal for want of memory is that import's alone, never remembered as a
  refusal (the form and type that worked go first, above): only a refusal that lasts (the driver
  refuses the dma-buf itself, `ERROR_INVALID_EXTERNAL_HANDLE`, or imports it as no memory type
  its buffer takes) is counted (`WorkCounts::host_ram_dma_buf_imports_failed`) and remembered
  for the device's life, so that the device's later chunks and copies' lists take their pointer
  without asking the source, and a RAM's use that asks the driver first, by an import of the
  RAM's first page, keeps its 64 MiB chunks imported ahead by their pointer, never the long
  chunks made for dma-bufs.

- **A pointer import of pages the kernel cannot pin on NVIDIA** (a dma-buf's mapping, `VM_PFNMAP`)
  fails as out of device memory (615.71.09), where Vulkan requires
  `VK_ERROR_INVALID_EXTERNAL_HANDLE`, and leaves the process's later host-visible allocations failing
  (`RuntimeDriverDefect::UnpinnablePagesBreakHostMemory`, the runtime's `driver_defects.rs`). The
  runtime keeps such pages from that driver, asking the kernel before it hands their pointer over
  (`host_memory.rs`), at a `PROCMAP_QUERY` and a one-byte `process_vm_readv` a mapping per import:
  a no-copy buffer, heap or rebind over them takes its dma-buf source (whose import of a udmabuf
  NVIDIA fails, above), and with none, or where its import fails, a copy of them
  (`copied-host-pages`), at a copy's cost; a guest RAM's chunk over them is not imported. Which later allocations the driver fails, and for how long, is
  **unaudited** past the shapes measured there.

- **On NVIDIA's proprietary driver, a process that exits with a device live can fault in the
  driver's own worker thread during its exit handler**: the exit handler of `libEGL_nvidia.so`
  tears the driver's SPIR-V compiler down while a worker thread of the driver's still runs in it, a
  null dereference at `0x18` in `libnvidia-glvkspirv.so` (`SIGSEGV`, `SEGV_MAPERR`) on a thread the
  driver started from the thread that made the device, so it carries that thread's name. No thread
  or call of the runtime's is in the driver then, and Vulkan has no call that waits for the
  driver's worker. Measured on an RTX 3090 (620.70.06), 8 processes at once, each making a device
  with no record of its in-place check (`Device::settle_in_place_checks`) and keeping it to the
  exit: 6 of 80 exits right after the check ended faulted, 0 of 80 three seconds after it, 0 of 80
  where the device was dropped before the exit, and 0 of 160 with no check (exits right after the
  device's creation and 6 s after it). What other work leaves the worker running is unaudited.
  Drop the runtime's devices before the process exits on that driver
  ([HOWTO](HOWTO.md#8-check-the-integration)).

- **Reflection after a compute descriptor reset on RADV**: `max_total_threads_per_threadgroup`
  answers 0 where Metal answers 96, `required_threads_per_threadgroup` `0x0x0` where Metal answers
  `8x6x2`, and `threadgroup_size_is_multiple_of_execution_width` false where Metal answers true
  (`compute-descriptor-reset`). Both RADV chips, the RTX 5080 and llvmpipe, exact on MoltenVK.

These were found by replaying the conformance suite on a driver that is not Metal: RADV first, then
the whole 1,099-case set on an RTX 5080 Laptop GPU (615.71.09) and on llvmpipe (Mesa 26.2.4).
**ANV is unaudited** for every one
of them. On a card without unified memory the runtime announces `shared-in-system-memory` for a
`Shared` buffer, which no recording taken on Metal holds: each case that makes one declares it
(`replay::shared_memory_notice`), which loosens the notices site alone, so every value still
compares with Metal's.

- **A texture over an IOSurface plane whose GPU rows overlap is served only as a shadow.** A plain
  plane of more than one row of over 16 bytes whose `bytesPerRow` rounded down to 16 is below the
  row is made: its rows share 1 to 15 bytes, a whole number of texels, between consecutive rows
  alone, and its shadow's copies hold the sharing with two regions a slice each way, which
  announces `colliding-surface-row-writes` (`linear_plan.rs`,
  `buffer_texture::plane_copy_regions`). So is a colour plane whose rows are wider than its
  `bytesPerRow`, at multiples of 16, its rows sharing any whole number of texels on the GPU and the
  CPU alike. Still refused
  `NotImplemented(OverlappingSurfaceRows)`: a plane a render pass stores into, where what Metal
  leaves of the shared bytes is unmeasured; a two-plane YCbCr texture's rows of codes, which no row
  rule holds and which overlap by any amount across any number of rows, and the other formats whose
  rows no row rule bounds (a two-plane extended-range one's, a 4:2:2 one's pairs, a depth or stencil
  plane's elements), which two regions do not express; a blit that would write such a plane's rows
  as bytes (`blit_texture.rs`: a copy into it from a buffer, and one between two textures over
  buffers, refused whichever side the plane is on, the destination having no one order and what
  Metal reads out of a source plane there being unmeasured), which would write every shared byte
  from one fixed row where Metal has no one order: the row Metal leaves there follows the command,
  the GPU pitch and the row count, one shape of `copyFromTexture:toTexture:` gives three whole-plane
  outcomes from run to run, and a kernel's winner moves with the threadgroup shape (an Apple M3 Max
  under macOS 27.0.1, one case a process, `probes/overlapping_plane_row_writes.m`); and a copy
  between such a plane and an image texture, either way, whose one `bufferRowLength` states no
  sharing. A blit out of such a plane into a buffer is served: its rows are read, not written, and
  each of the caller's rows takes the plane's bytes at its own GPU row, so a byte two rows share is
  read once for each of them, which is what Metal's `copyFromTexture:toBuffer:` reads
  (`probes/overlapping_plane_row_reads.m`). A read is not the mirror of a write here -- a write must
  settle which of two rows leaves the shared bytes behind and Metal settles it no one way, while a
  read settles nothing. A rotated plane never reaches any of this: its
  `bytesPerRow` must be a multiple of 16, so its GPU rows lie at the pitch itself and share no
  byte.
- **Some textures over an IOSurface plane past one 2D image of its rows are refused.** Over a
  plane of a surface with planes Metal makes a texture of any type, extent and levels, but its
  GPUs lay out alike only a texture buffer (at any offset and pitch, its functions from the offset
  rounded down to 16), a 1D texture whose row fits and a 2D array of one slice
  (`probes/surface_texture_envelope.m`, an AMD Radeon Pro 5300M, an Intel UHD 630 and an Apple M3
  Max). Of the rest, a colour 2D texture of several levels, a 3D and a 2D multisample one lie in
  memory of their own (announced `surface-texels-apart`): the M3 Max lays them out in no one
  place, their first level and slice among them, where the x86 Macs keep those over the plane. A
  colour 2D array of several slices over a plane at an offset and pitch that are multiples of 16
  keeps its first slice in place and its others in a tail of its own (announced
  `surface-texels-tail`), as the M3 Max lays those in no one place. A colour 2D texture whose rows
  are wider than the plane's `bytesPerRow`, and a colour 1D texture wider than one, over a plane at
  multiples of 16 lie as the M3 Max lays them out, rows `bytesPerRow` apart on every path, a 2D
  texture's overlapping (announced `colliding-surface-row-writes`). A BC1 or BC4 texture (8-byte
  blocks) at an offset or pitch off 16 that 8 divides lies at the plane's own offset and pitch on
  every path, as the x86 Macs' blits and samples take it. Wider rows of another format or off 16,
  those other shapes of another format, an array off 16, a BC texture of 16-byte blocks at an
  offset or pitch off 16, and one of 8-byte blocks off 8 or under 16 bytes a row are
  `NotImplemented(SurfacePlaneShape)`. A cube or cube array over a plane is `Invalid`, as
  plain Metal asserts on it on the M3 Max. Texels past the surface's end, which Metal lays out in
  no one place, lie in a tail of the texture's own (announced `surface-texels-tail`), but a
  two-plane YCbCr, two-plane extended-range, 4:2:2, depth, stencil, BC or rotated texture's, which
  are `NotImplemented(TexelsPastBufferLength)`. A texture taller than its plane, inside the surface,
  is held as the M3 Max lays it out (its rows on at the plane's pitch); the x86 Macs' GPUs keep it
  to the plane, which the runtime cannot, as `IoSurfacePlane` gives no plane height.
- **A shadow's merge back to host pages stores each changed word alone after a command that
  leaves texels it does not write** (a kernel's or a blit's partial writes among them; a render
  pass that loads and stores its attachment inside its scissors, and a command that replaces
  every texel over logged pages, store whole 256-byte blocks). Over PCIe each separate run of
  changed words past the first in a 64-byte host line costs about 180 ns, serialized: on an RTX
  5080, 4 MiB with every other word changed took 88 to 102 ms (11 ms with each line's runs stored
  in separate passes), against 0.4 ms in whole blocks. Blocks stay out there: they rewrite
  unchanged bytes a CPU may be writing meanwhile, and Metal keeps such a write beside a kernel's
  (on an M3 Max, a kernel writing one texel lost none of a CPU thread's writes to the texels
  around it in about 1.1e9 checks), while a loading and storing pass rewrites the whole 32x32
  tile it draws in (`shadow.rs`, **Stored attachments**). A blit's partial writes keep the
  kernel's rule. Not announced.
- **An execution of indirect commands costs a decode and a recording per command at every commit
  where it runs no kept recording** (`kept_commands.rs`). An execution of commands the CPU encoded
  runs each 256-command chunk that no write reached since an earlier execution recorded it, under
  the same inputs, from that recording, reading no record (`WorkCounts::indirect_chunks_kept`,
  `indirect_commands_kept`). A chunk is recorded to keep from its second execution with no write
  reaching it since (`WorkCounts::kept_contexts_made`); a range's first execution, and one every
  chunk of which a write reached since, records as one that keeps nothing, so a range executed
  once pays nothing for keeping. A host step of a render execution of records the GPU may write,
  whose commands inherit the pipeline state and whose inherited tables bind buffers alone, keeps
  its chunks so too (`render/executions/kept.rs`), where two steps read its records alike: a step
  whose records hold, chunk by chunk, the bytes a kept recording was made from or else those the
  last step read, compared whole (`WorkCounts::host_step_records_compared`,
  `host_step_compare_nanos`), runs each kept one whose inputs are equal
  (`WorkCounts::host_step_chunks_kept`) and records apart and keeps the others; one whose records
  changed records inline as before, from its records read once into the copy the next step
  compares with, forgetting the recordings whose records changed
  (`host_step_chunks_recorded_again`), but a chunk whose records changed at two steps in a row
  reads them in place and is copied at one step in eight only, each chunk at a step of its own
  (`host_step_chunk_copies_put_off`), so records that change at each step cost a copy of an
  eighth of them a step, and a chunk whose records stop changing is copied again within eight
  steps, then compared at the next. Draws that read textures through the heap (set 1's heap set
  and the device's texture table) are kept too, at `commit` and at a host step, where the device
  has `descriptorBindingUpdateUnusedWhilePending`, the draw writes no set elements of its own (a
  function table's), its set holds no shadowed texture, decoded twin or raw read, it reads no
  sampler table, and its tables bind no texture its heap part reads; the rest are counted by why
  (`WorkCounts::replayed_heap_draws_unshared`, `replayed_heap_draws_writing`,
  `replayed_heap_draws_noted`, `replayed_heap_draws_over_sampler_tables`,
  `replayed_heap_draws_table_textures`). A sampler state made or released records the kept
  chunks again. The step's round trip stays. Every other execution reads
  and records each of its commands at each commit: one of a buffer the GPU may write (from the
  call at which a route that may write its records met it, or a blit into it, on) at its commit,
  one of a buffer of mesh or patch draws alone (neither `DRAW` nor `DRAW_INDEXED` among its command
  types), another host step, a range read from a buffer, one with a check that may see
  something among its commands, a visibility result, tile threadgroup memory, a rasterization-rate
  map, a memoryless attachment, a shadowed texture, or an inherited table that binds anything, one
  in a command buffer with an executed-commands handler, a compute one inside command predication,
  and one of a range holding a chunk whose last recording was not plain (a mesh or patch draw, a
  draw that splits its instance or needs work around it, one that meets a notice its device
  announces at each occurrence), until a write reaches that chunk. Not announced.

- **On ANV below Xe-HP, a device holds at most 16,384 command buffers begun, in flight, in
  recording or kept** (Mesa 26.2.4 and 26.2.6, `anv_device.c`): each takes a 64 KiB block of the
  device's 1 GiB binding-table pool, freed only as the command buffer is begun again, reset or
  freed. An encoder's recording past them fails with `OutOfMemory(Device)` or `EncoderAbandoned`
  where Metal runs it, and a submission that begins command buffers past them crashed in the
  driver where it recorded a clear. A render pass whose first instance records inline takes one
  command buffer (`render/inline_first.rs`; a pass beside a depth/stencil attachment it loads or
  clears and stores among them), and one whose draws record in a secondary takes two (a pass
  that stores its depth or stencil `DontCare`, Metal's default for them, or loads one
  `DontCare`, a multisampled, layered or resolved pass, and the other kinds
  `WorkCounts::passes_not_inline_*` count). A batch begins one more before each of the first
  2,048 encoders of its command buffer that needs a barrier, and none past them (`encoder.rs`,
  `WorkCounts::encoders_barriered_at_open`). On ms01's Iris Xe one command buffer of 10,000 render
  passes, each a draw beside a cleared depth target, runs in 12,049 command buffers
  (`tests::many_passes`); one of such passes recorded inline encoded 16,173 and refused to open
  the next, and one of passes that take a secondary failed at the end of its 8,087th, the
  spares holding 424 command buffers. A command buffer of passes recorded inline, its targets'
  first clears in its submission (`texture/first_move.rs`), ran with 14,000 passes, and with
  14,300 its submission went past the pool and crashed in the driver as it recorded those
  clears: such a command buffer of more than about 14,100 encoders can end the process there,
  where Metal runs it. The spare command pools keep at most 4,096 command buffers (`spares.rs`),
  which leaves the rest to the command buffers committed and in recording.

- **A tiled texture's later levels in a format `generate_mipmaps` leaves or refuses**
  (`Buffer::new_tiled_texture`, more than one level, `shadow.rs`'s **Copies taken at use**,
  `LaterLevels::Kept`): its copy taken at use moves the bytes into its first level alone, and its
  later levels keep what they hold (zero from the making, or what was written into them), where
  Metal makes them from the first. Such a format is one `generate_mipmaps` leaves as it is (a depth
  or stencil format) or refuses on the device (`mipmap_plan.rs`: a compressed, YCbCr or RGBX
  format, or one neither the device's blits filter nor the compute filter computes), or one whose
  compute filter or picks need a scratch past the device's limits. Every other format's later
  levels are made as `generate_mipmaps` makes them: the blit chain, the compute filter or the
  integer picks.

- **A tiled texture's take in an encoder whose later part alone runs nothing**
  (`Buffer::new_tiled_texture`, more than one level, `shadow.rs`'s **Copies taken at use**,
  `State::fail_part`): a take whose encoder runs nothing is given back, so the next use takes the
  copy. An encoder split into parts at a host step (an execution of indirect commands the GPU may
  write) gives back every take of its own where any of its parts runs nothing, so a take that ran
  in an earlier part, where a failure of a later batch fails only a later one, is taken again at
  the next use: that use reads the bytes as they are then, where Metal's levels keep the copy that
  ran.

- **A tiled texture's take in the work apart of a predicated loop's run again**
  (`Buffer::new_tiled_texture`, more than one level, `shadow.rs`'s **Copies taken at use**,
  `command_buffer/rerun.rs`): each run again records the laying's work apart again (an indirect
  dispatch, or an execution of indirect commands) as a new recording, which decides its first use
  of the texture by the texture's mark, as a use encoded then would. Where `Texture::did_modify_data`
  or a `replace_region` of the first level is called while the loop still runs again, the next run
  again takes the copy: it copies the bytes into the levels mid-loop, over what the earlier runs
  wrote into them, and clears the mark, so the first use encoded after the call takes no copy and
  reads the levels as the loop left them, where Metal's takes the copy of the bytes as they are
  then. A recording of a run again carries no take, so a failure of the loop's command buffer gives
  none back.

- **A depth or stencil loaded `DontCare` and stored holds its clear value in the tiles a draw
  touches even where no draw of the pass tests or writes it** (`clear_reach.rs`), where Metal keeps
  its texels there: a draw over the whole target under a depth test `Always` without depth writes
  and the default stencil state left a `Depth32Float`, a `Depth32Float_Stencil8` and separate
  `Depth32Float` and `Stencil8` textures their earlier texels in every tile (an M3 Max under macOS
  27.0.1, with and without API validation). Which tiles Metal clears follows which draws access
  the aspect; the runtime's clear follows the draws' rectangles alone.
