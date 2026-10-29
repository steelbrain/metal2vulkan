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
| A blend into a packed 16-bit attachment the device renders through its word view (`A1BGR5Unorm`, `ABGR4Unorm` on the RTX 5080; `WordAttachment`), which the function computes in the interlocked storage form, reads the attachment's codes, where Metal's tile holds the pass's clear and each earlier blend's result as a `half`, finer; under fast math it also sums a few states otherwise than Metal. Every device rendering such an attachment. | The in-shader blend over the codes (no switch) | **Not built yet:** the extended-range attachments' `half` shadow (`render/extended_shadow.rs`, `VariantKey::extended_shadows`) taken by word attachments: the runtime's shadow start and end, its load of the codes and the clear's halves, and the word's shadowed variant in `lower::fragment::fetch`, for a single-sampled pass (a multisampled one keeps the codes, as an extended-range one does). Fast math's regrouped sums have no model | One code: without fast math exact over a loaded attachment (the 2,891 states of the M3 sweep, every channel), under fast math 3,527 of 94,732,288 channels a code apart; after a clear or an earlier blend in the pass, a code apart where the two readings store different codes. `DeviceColorConversion` (`device-color-conversion`) | Not measured. Expected from the shadow's form: an `R16G16B16A16_SFLOAT` image of the pass's extent per such color (8 bytes a pixel, 16.6 MB at 1920x1080), set at the draw it starts by a fill or a load dispatch, and an 8-byte storage read and write per blended fragment inside the interlock beside the word's |
| A heap's colour texture that Metal keeps losslessly compressed (`allowGPUOptimizedContents`, no `ShaderWrite`, `PixelFormatView` or `ShaderAtomic` usage, 16 texels or more a side), on a driver whose compressed images may keep their texels apart from the memory they are bound to (`DriverFact::CompressedImagesHideAliasedWrites`: ANV), is the driver's compressed optimal image, its compression and fast clears kept. A texture Metal keeps uncompressed takes the uncompressed image under either value. The runtime's choice: no pipeline differs. | `heap_compression = HeapCompressionPolicy::Compressed` | `heap_compression = HeapCompressionPolicy::Uncompressed`: an image the driver keeps uncompressed in the heap's bytes, as for a texture Metal keeps uncompressed: host-transfer usage where the device enabled `hostImageCopy` (`heap-texture-host-transfer`), else a linear image (`heap-texture-linear`), else the optimal one (`heap-texture-hides-aliased-writes`) | A write through a buffer or another texture over the texture's bytes may leave its texels as they were or change them otherwise than as written, where Metal's texture reads opaque magenta (`0xffff00ff`) in every texel of drawn content the write covers and the written bytes where it held only a clear; the texture's own writes may show in those bytes compressed. Within Metal's contract: Metal documents such a texture's layout as its own (`allowGPUOptimizedContents` lets it "adjust the private layout"), so no write through another view over its bytes has a defined result there either. `HeapTextureCompressed` (`heap-texture-compressed`), announced once per device where such a texture is made; `runtime_lane::heap_textures` | ANV (Iris Xe of RPL-P; `heap_textures::heap_compression_costs`), a 2048x2048 `RGBA8Unorm` render target in a `Private` placement heap: a pass that clears it 26.9 us against 676.2-676.5 us; a copy into another such texture 647.7-649.1 us against 673.9-693.2 us |
| Batches of command buffers committed on one queue that the scheduler finds ready together (a backlog) go to the driver in one `vkQueueSubmit`, at most four, only the last signalling the queue's timeline, on a driver measured to merge a submission that signals nothing into the next (`DriverFact::SubmissionsWithoutSignalsMerge`: Mesa 26.2.4's ANV on the Iris Xe of RPL-P and the UHD 770 of RPL-S, RADV on the Radeon 780M and the RX 7900 XTX, and llvmpipe; every other driver, measured or not, takes the ungrouped form under either value of the switch): one kernel submission, where ANV on i915 lists every `VkDeviceMemory` in each (the consumer's guest drag on the Iris Xe: about 300 entries per execbuffer, 260 of them its host imports, at 112 to 128 execbuffers a frame). A batch's event signals ride to the group's end, which signals each event's semaphore to the highest index the batches held signal it to (an event's indices rise in commit order). A batch that signals a binary semaphore, takes a fence, binds a sparse update, does first work, names the caller's wrapped image, or waits for anything not seen complete (an event, another queue's work) goes alone, after the batches held: a batch with a wait leads a group, never joins one. So that checked command buffers (the consumer checks each) are found ready together, the submission checks of up to three batches right behind a line's head that goes run with the head's, in commit order, none committed after another queue's batch: such a check runs before the previous batch's `vkQueueSubmit` rather than after it, in the scheduler pass that submits its own batch. A pass that finds no batch ready sends the batches held, unless two or more earlier submissions of their `VkQueue` are not seen complete: the GPU runs those first in any case, so the batches stay held, and those committed meanwhile join them, until one submission is left in flight (a completion turn wakes the scheduler at its value), the group fills, a later batch sends it, a head waits for the GPU, or a thread waits for one of their command buffers' completion, which sends them at once; a batch that ends a command buffer with a switch of storage forms riding on it, which settles by its submission's outcome, is never kept so. A held batch that ends its command buffer is `Scheduled` as it is held, its submission handlers run first, then its scheduled handlers, and its kernel end time set, as an M3 Max schedules a command buffer committed behind running work 0.04 to 0.18 ms after its commit. The runtime's: no pipeline differs. | `completion = CompletionPolicy::Grouped` | `completion = CompletionPolicy::Exact`: one submission per batch, each signalling its own values; each line's head checked alone | Each command buffer of a submission but the last is seen complete (status, completed handlers, waits), and the event values it signals are reached (by the GPU waits on other queues, the host's waits, value reads and listeners), once the last completed: late by the GPU time of at most three later batches (about 0.1 to 0.2 ms each in the consumer's compositor frames on the Iris Xe; about 20 ms behind a 20 ms batch), never early, in commit order. Metal's contract states no bound on when completion is seen (an M3 Max reports a command buffer complete, its handlers, status, waits and event listeners alike, 0.04 to 0.10 ms after its GPU end), and no member but the first waits, so no wait holds a member back: a valid program sees only later completion, as of slower GPU work. A batch kept while earlier work runs is seen complete within the same bound, and, where the one submission left in flight ends before its group reaches the GPU, late besides by that gap: at most a completion turn's reaction and the group's `vkQueueSubmit`. Commands, barriers and GPU times are unchanged. `GroupedCompletion` (`grouped-completion`), announced once per device at its first such submission; `runtime_lane::grouped_submissions` | ANV (Iris Xe of RPL-P; `frame_costs::frame_costs_over_imports`, the checked device, 67 command buffers a frame, GPU at 467-783 MHz), with 520 host imports live, where the Exact form's submissions fall behind the commits as the guest's do: 32.0-33.2 kernel submissions a frame against 67.0, 3.4-3.5 ms of execbuffer ioctl a frame against 6.9-7.9 ms, the scheduler thread's CPU 61-63 us a command buffer against 121-136 us, a frame through completion 4.7-5.9 ms against 8.6-9.7 ms; in the guest's shape with 520 (a second queue taking every 20th command buffer, each signalling a shared event, every 10th waiting for one): 26.2-26.7 kernel submissions a frame against 67.0, 2.8 ms of ioctl a frame against 6.9-7.0 ms, the scheduler 52-53 us a command buffer against 122-123 us, a frame through completion 6.0 ms against 8.7 ms; with 260, where it keeps up, 64.9-66.3 kernel submissions a frame against 67.0, the CPU and wall times alike. The cost (`grouped_submissions::grouped_completion_delays`, 8 command buffers of 0.3-0.5 ms of GPU work ready together): the host saw each but a group's last complete later by the later ones' GPU time, the last 0.1-0.3 ms later than alone (the completion turn's per-command-buffer work for the group, done at its end). Keeping a group while earlier work runs (UHD 770 of RPL-S, ANV, the guest's shape with 520 host imports, a submission handler on every command buffer and a 16 MiB fill ahead of each, about 1.3 ms of GPU each; the kept form against the batches sent as found): committed every 1 ms, 18.1-18.2 groups kept a frame, 24.9-25.0 kernel submissions a frame against 57.9, 2.18-2.90 ms of execbuffer ioctl a frame against 4.64-5.43 ms, the scheduler thread's CPU 67-82 us a command buffer against 113-118 us, the GPU idle within a frame's span 0.61 ms against 0.68-0.69 ms, a frame through completion 89.5-89.6 ms alike, completion seen later (GPU end to completed handler: mean 1.82-1.86 ms against 0.43-0.46 ms, p50 1.57-1.60 against 0.11-0.13, p99 4.35-4.37 against 4.09-4.11, max 4.52-4.55 against 4.18-4.23); committed at once, 18.0 kernel submissions a frame against 23.4; with no fill, 18.0 against 20.6, the GPU idle 0.59 ms against 0.58-0.66 ms. In the consumer's guest drag on the Iris Xe (i9-13900H; three passes each way): 24.4-27.5 kernel submissions a frame against 42.4-43.8, the scheduler thread 0.4-1.7 ms of CPU a frame less, 15.7-17.3 frames a second against 16.4-17.6: the frame rate alike within noise, as the scheduler thread does not bound that frame |
| Batches of command buffers on one `VkQueue`, on a driver measured to start no kernel submission of a queue before the one before it completed (ANV, RADV and NVIDIA's; `batch_order.rs`), are ordered by their ends: each ends with a barrier ordering its work before every later command of its `VkQueue` (the exact form's end orders it before the host's reads and writes alone), so a batch that depends on an earlier one there begins with no barrier of its own, often no command buffer of barriers, and its first encoder's GPU start is written at the bottom of the pipe, once everything before it completed. Metal starts an independent command buffer committed behind running work while that work runs (an M3 Max under macOS 27.0.1, probe `gpustart`); those drivers ran two batches of independent work one after the other under every shape measured (no end barrier, the host's or the widened one; in one `vkQueueSubmit` or two, merged into one kernel submission or not), where two dispatches of one command buffer ran side by side. The runtime's: no pipeline differs. | `batch_order = BatchOrderPolicy::Ends` | `batch_order = BatchOrderPolicy::Exact`: a barrier for each dependency of a batch's work on an earlier batch's of its `VkQueue`, the end's barrier to the host's reads and writes alone, the GPU start written at the top of the pipe; every other driver takes it under either value | None: on those drivers each batch starts once the one before it completed under either value, so the work's order, its results and its GPU times are the same, and a batch's GPU start is the end of the work before it, when such a driver starts it. Not announced; `runtime_lane::commit_work` and `runtime_lane::gpu_times` test both values | UHD 770 (ANV, RPL-S), the consumer's frame shape over 520 host imports, 67 command buffers a frame, Grouped completion, the widened end barrier against the exact form (a probe build of each): with no fill ahead of each command buffer, 67.0 command buffers submitted a frame against 129.3-129.7, the GPU's span a frame 1.84-1.90 ms against 1.99-2.06 ms, a frame through completion 2.13-2.22 ms against 2.27-2.38 ms; with a 1 MiB fill ahead of each, 134.0 against 262.9, 8.03-8.22 ms against 8.32-8.42 ms, 9.00-9.15 ms against 9.25-9.36 ms. Radeon 780M (RADV, Mesa 26.0.8), this switch: with no fill, 67.0 against 128.9-129.0, a frame through completion 48.3-48.6 ms against 42.6-47.0 ms; with a 16 MiB fill committed at once, 134.0 against 251.0-251.1, 50.6-50.8 ms against 50.8-57.3 ms; committed every 1 ms, 134.0 against 249.7-250.2, 71.3-83.6 ms against 106.3-113.6 ms (each kernel submission there leaves about 2.5 ms of the GPU idle over 520 host imports, which sets these frames) |
| A chain of `half` additions, subtractions and products on a device without `shaderFloat16` (vk12-floor, SwiftShader, Pascal; capability emulation: the module computes each in `float` and rounds it by the device's `OpQuantizeToF16`): an operation whose flags hold `contract` and `reassoc` and whose result only such operations read is held at `float` precision unrounded, so the chain rounds where a value leaves it (a store, an output, a conversion, a comparison, a call, a phi, a vector move, an operation without both flags), one quantize per exit instead of one per operation (`lower::arith::half_chain`). A device with `shaderFloat16` computes `half` alike under either value. | `half_chains = HalfChainPolicy::AtExits` | `half_chains = HalfChainPolicy::EachOperation`: every operation's result rounded to `half`, as the AIR computes it | The result may differ from the rounded operations' by the roundings the chain skips, each within half a unit in the last place of a `half` at its step, compounded along the chain; a value inside the chain past `half`'s range (65504) stays finite where the rounded operations reach infinity. Within the AIR's flags, which let a compiler regroup and fuse such operations. `FastMathRewrites` (`fast-math-rewrites`) | Not timed. Size on vk12-floor (the fixtures' SPIR-V words, the default against the accurate form): `hfb_filter_bank` 121,270 against 131,766 (1,024 quantizes against 2,464), `grd_half_many_samples` 82,019 against 84,627, `nan_half_fast` 9,808 against 9,832 |
| A texture that only samples or reads (no `RenderTarget`, `ShaderWrite` or `ShaderAtomic` usage) over host pages (a no-copy buffer's, or a no-copy heap's buffer's, imported by their pointer or as dma-bufs; or a buffer in fresh pages) is a linear image aliasing the pages in place, where the device's own check shows that the image's first move, which Vulkan lets read and write every byte bound to it (`VUID-VkImageCreateInfo-pNext-01443` makes such an image start `UNDEFINED`), leaves every byte alone while the CPU writes beside it: a deliberate, scoped exception to relying on documented Vulkan behaviour alone, the owner's ruling (`in_place.rs`). The check runs once per driver, release and runtime, per class of pages, on a thread of its own, its verdict kept in the shader cache's directory; a texture made before its class's verdict is a shadow. A texture whose usage renders or writes, a check that failed or made no image, and a pitch no image takes keep the shadow (`shadowed-texture`, its reason named, `WorkCounts::host_texture_shadows_*`). | `host_page_textures = HostPageTexturePolicy::InPlace` | `host_page_textures = HostPageTexturePolicy::Shadow`: every texture over host pages a shadow, its texels copied in before each command buffer that uses it and merged back after each write (the host-pages bullet under **Costs of matching Metal**) | Exact where the driver keeps what its check showed: the check covers texels, the bytes between rows and those after the image, of 1, 2, 4, 8 and 16-byte texels and a 3840x2160 surface, three rounds each; a driver whose first move changed a byte outside every round's window would undo a CPU write made meanwhile. Not announced where the check passed. A CPU write reaches the image's reads as it reaches the buffer's (Host Write Ordering Guarantees) | See **Costs of matching Metal** (host pages): the shadow copies a sampled 3840x2160 surface in whole each command buffer |
| A render or compute state's creation hands the driver's making of its predicted pipeline (`pipeline/predicted.rs`, `pipeline/dispatch_making.rs`) to the runtime's pool, kept by the state as a draw's or dispatch's making is, and waits for it `pipeline_creation_wait` at most; a committed command buffer's recordings wait until its commit plus `pipeline_commit_wait` at most, one deadline its commands share, for its draws' and dispatches' (a compute encoder's, a render pass's tile dispatches) pipelines still being made (the scheduler wakes at the bound, `StepWaits::until`), an execution's replayed draws (`render/pending.rs`) among them, whose pipelines the commit's settle hands to the pool (a device replay's, one recording for every command, waits until made under either value), then record each such command as nothing: a draw's barriers and visibility query, a dispatch's barrier (a tile dispatch's copies into its scratch and back), no draw or dispatch, the command buffer running on without error. The making goes on, and the command buffers after it draw and dispatch with the pipeline once made. Metal compiles inside `newRenderPipelineState`, so no draw of a made state waits or is dropped; a host that creates a state inside a command buffer's encoding (the consumer creates a guest's state at an execution's first lookup of it) held that execution through the driver's whole compile: 11.59 s inside one execution on an M2 Pro (MoltenVK 1.4.2, a cold Metal compiler service; a sample of such a stall finds the compile's thread waiting in `MVKMetalCompiler::compile` for Metal's compiler), past the guest's 10 s FIFO check, which ended the guest's task. Each bound is 4.5 s: a creation and a commit may fall in one step of the guest's FIFO (a task's jobs run one after another, and a draw's own variant waits under its command buffer's bound), and no runtime call waits for both, so one step waits 9 s at most for makings, under that 10 s check. A creation waits on no command buffer's deadline and a `commit` on no creation's; a `commit` waits until its own deadline at most (it records there what it can, an execution of render commands the CPU writes among them), every other recording on the scheduler's thread. Made until made under either value, in the call that makes them: a draw that records nothing, one in a pass of custom sample positions and one recorded again into another instance of its pass (each in the draw's call or an execution's replay), a device replay's draws (in its `commit`), and a failed making made again (`PipelineWaitPolicy`'s docs). The runtime's: no pipeline differs. | `pipeline_waits = PipelineWaitPolicy::Bounded`, `pipeline_creation_wait = 4.5 s`, `pipeline_commit_wait = 4.5 s` | `pipeline_waits = PipelineWaitPolicy::Exact`: the creation makes its predicted pipeline before it returns (a render state's first key on its own thread, the rest on the pool, joined; a compute state's on its own thread), an execution's settle makes its replayed draws' pipelines at the commit, and a command buffer waits for each draw's and dispatch's pipeline until it is made | A draw whose pipeline was still being made `pipeline_commit_wait` after its command buffer's commit renders nothing in that command buffer (only there: later ones draw it), where Metal's renders: content drawn once (a layer, a glyph cache) stays without it until redrawn. A creation returns after `pipeline_creation_wait` at most with its making in flight (`WorkCounts::creations_left_making`). `DrawDroppedStillMaking` (`draw-dropped-still-making`), at each such draw where `DeviceOptions::announce_each` asks, else once per device; `CommandBuffer::dropped_draw` names the first with `VK_TIMEOUT`; `WorkCounts::draws_dropped_still_making`; `deferred_calls::a_draw_whose_pipeline_is_still_being_made_at_its_bound_runs_nothing` and `under_the_exact_wait_a_held_making_holds_its_command_buffer_until_it_lands`. A dispatch whose pipeline was still being made then writes nothing in that command buffer, where Metal's runs: `DispatchDroppedStillMaking` (`dispatch-dropped-still-making`), `WorkCounts::dispatches_dropped_still_making`, the first named by `dropped_draw`; `deferred_calls::a_dispatch_whose_pipeline_is_still_being_made_at_its_bound_runs_nothing`, `a_compute_state_s_creation_waits_its_bound_for_its_predicted_making` and `a_replayed_draw_whose_pipeline_is_still_being_made_at_its_bound_runs_nothing` | No GPU cost. The exact form holds the creating thread, and the command buffer, for the whole compile: up to 11.59 s on the M2 Pro's cold boot (that boot's 2,483 render state creations, translation included: 300 over 16 ms, 151 over 100 ms, 14 over 250 ms, 3 over 1 s, the longest 11.59 s, 5.82 s and 1.17 s), where the default holds a creation for 4.5 s at most and a command buffer for 4.5 s from its commit, which drops the draws of the compiles that outlast those (2 of that boot's 2,483 were over 4.5 s) for the command buffers committed before they land; bounds of 1 s dropped 39 draws in a cold boot on the M2 Pro, and the guest's menu bar, drawn once, stayed half drawn |

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
| Fast-math sums, differences and products group, and a multiply contracts into an addition, as the device's compiler groups and fuses them, on every device, MoltenVK included: each operation is emitted in the AIR's order, one carrying both `contract` and `reassoc` undecorated and any other `NoContraction`. Metal's compiler takes its own grouping (its rank order of a sum's leaves, its pairing of a product's lane-uniform factors) and its own fusion pairs. | The device's grouping is the contract: what the AIR's flags allow, the device's compiler chooses; nothing reproduces Metal's choice. | Unbounded relative to the result where a sum of mixed signs is grouped otherwise, or a multiply-add is fused on one side alone; each operation kept apart is exact as written. `FastMathRewrites` (`fast-math-rewrites`) |
| `air.pack` and `air.unpack` of a normalized format are the device's conversion: `GLSL.std.450`'s `Pack`/`Unpack` `Unorm4x8`, `Snorm4x8`, `Unorm2x16` and `Snorm2x16`, and for `rgb10a2` and `rgb565` their definition by the device's `NClamp`, multiply, `RoundEven` (a pack) and multiply by the reciprocal (an unpack); `unorm4x8_srgb`'s colour is the MSL transfer functions by the device's `Pow` around the 8-bit instruction, as is an 8-bit sRGB storage texture's write through its UNORM alias. A `float` one of the four named formats on MoltenVK reaches Metal's own. The same conversions serve an implicit imageblock access of a normalized attachment (a `half` load rounding the unpacked `float` once more), the interlocked storage form's 8-bit sRGB attachment and sRGB word, and a snapshot's normalized codes (`c` times the `f32` nearest `1/D`). So does every conversion of an extended-range, padded 10-bit or bfloat texel, which no Vulkan format holds (`BGR10_XR`, `BGRA10_XR`, `RGB10A8_2P_XR10`, `R10Unorm_X6`, `RG10Unorm_X12`, their sRGB twins, `RGB10A2Unorm_sRGB`, the bfloat formats; every device): a read `float(c - 384)` (a padded code's `float(c)`) times the `f32` nearest `1/510` (`1/1023`), a write `RoundEven` of the clamped value's `f32` product plus 384, an sRGB twin's colour the MSL transfer by the device's `Pow` on the magnitude, a `half` result the device's narrowing, a bfloat NaN as it is. | The device's conversion is the contract. | A packed code within one of Metal's, apart only where `clamp(c)·scale` lies within its rounding of a half-integer (`Round`'s tie is the device's); an unpacked value within 2.5 ulp of the correctly rounded quotient; a decoded sRGB colour within half a twelve-bit step (`1/8190`) of Metal's table step `k/4095`, beyond its `Pow`'s bound; a NaN channel's code the device's (Metal's: 0). An extended-range or padded value within one `f32` ulp of Metal's quotient, a code within one of Metal's exact product, an sRGB twin's decoded value within half a `2^-13` step of Metal's table (`R10Unorm_X6_sRGB`'s 1.64 steps) and its code within one, a `half` result within one `half` ulp (Metal rounds ties away from zero, a bfloat sample's `half` toward zero), a bfloat NaN the device's (Metal's: `0x7fc00000`, a write `0x7fc0`). `DevicePackConversion` (`device-pack-conversion`) |
| Off MoltenVK a `float` multiply, a fused multiply-add by `OpFmaKHR`, and a `float` `ldexp` (whose result below `2^-126` is that multiply's product) round before Metal's flush: Metal flushes a result whose exact value lies below `2^-126` before rounding it, so a result whose exact value lies within half a unit below `2^-126` answers `2^-126` of its sign where Metal answers the zero of its sign. | Every device measured off MoltenVK (RADV, ANV, NVIDIA, lavapipe) rounds first; its multiply is the contract. | Announced once per device as it is made: `float-tininess-after-rounding` |
| `f32` denormals in a `float` operation on a device not measured to read them as Metal does (`DriverDeviation::MetalFloatDenormals`: lavapipe, and every unmeasured device), in `floor`, `ceil` and `fneg` on every device, in the maths functions (`fabs`, `sign` and `fract` among them, and the zero and sign tests of `pow`, `powr` and `atan2`) and the device's order of the rows above, and in a conversion to or from `double`. Metal reads them as zero and flushes a denormal result. | The device's reading is the contract: Vulkan lets each device keep or flush them, and Metal runs no `double`, so its reading of a conversion is not measured. | `DeviceDenormals` (`device-denormals`) |
| Samples, gathers, depth comparisons and level queries are the device's own sampler's on every device whose sampler is not Metal's (every device but MoltenVK): its filter, footprint, level, anisotropy, sRGB decode and comparison, its level clamp (the descriptor's as given, which Metal holds on a `1/64` grid) and its reading of a subnormal coordinate (which Metal reads as the zero of its sign before addressing). A sample's or gather's residency on a device with `shaderResourceResidency` is the device's `OpImageSparse*` word and non-resident value (`SparseResidency`); Metal's footprint rule answers it only where the device has no residency and the slot's tile map does. A two-plane YCbCr texture's linear sample is the device's filter of its texels (the device's own Y′CbCr conversion's where it converts the format, next row; Metal's reads exactly elsewhere); where the device's memory is the host's, a sample that reads the planes in place is Metal's filter exactly. An 8-bit sRGB format the device cannot sample (`R8_SRGB` on lavapipe, `R8G8_SRGB` on lavapipe, ANV and NVIDIA) is held as its UNORM twin, the device lacking its decode: a read, a sample, a fetch, a gather and a bicubic tap decode what the twin's view reads (one texel's codes, or the device's filter of them) by the MSL transfer through the device's `Pow`. | The device's sampler is the contract. | Linear filter within one code (1/255) on the measured devices, sRGB within 1.83, a two-plane YCbCr texture's within 1.183; levels, anisotropy, cubes, 3D, gathers and comparisons unmeasured. A twin's texel: the transfer within the device's `Pow` (MoltenVK's within 1.83e-7 through a storage alias, `runtime_lane::srgb_storage`) where Metal's table rounds it to a twelve-bit step (up to 1/8190 apart); a filtered sample decodes the filtered codes, up to 0.286 off for a blend of codes 0 and 255 (`SrgbTexelDecode`). `DeviceSampling` (`device-sampling`) |
| A pixel-coordinate sampler's sample, gather, comparison and level query is the device's sampler's at the coordinate divided by the texture's level-0 size in the shader, on every device, MoltenVK included: the device reads its texels from that coordinate in its own arithmetic, where Metal reads the coordinate as given (its nearest texel `floor(x)`, its filter's footprint from `x`). A constexpr pixel sampler that filters nearest and clamps to the edge fetches its texel (`sampler.pixel_fetch`), as does an argument buffer's sample without a comparison through a state that does. | The device's sampler is the contract. Vulkan's unnormalized samplers take no gather, offset, comparison, array, 3D image or level of detail, so the shader scales the coordinate; nothing reproduces Metal's position at a texel edge. | The device answers as its normalized sampler at a coordinate within one `f32` ulp of the quotient on MoltenVK (10 of 800 cells of each mirror-clamp configuration move by it; a nearest sample at a texel edge then reads the neighbouring texel) and at the quotient itself on lavapipe (`runtime_lane::pixel_samplers`: 800 coordinates at and around each texel edge of a 5x3 texture, every filter and address mode); a linear downscale by 8 of textures 1000 to 1920 texels wide within 1/255 of Metal's on MoltenVK and 2/255 on lavapipe (`runtime_lane::pixel_downscales`); other devices unmeasured. `PixelSamplerScaling` (`pixel-sampler-scaling`), `ArgumentPixelSamplerScaling` (`argument-pixel-sampler-scaling`) |
| A two-plane YCbCr texture's texels are the device's own sampler Y′CbCr conversion of its planes where the device converts their format (`ycbcr_device.rs`): unpacked 8- or 10-bit codes of a 4:2:0, 4:2:2 or 4:4:4 layout over plane 0 that form one image (no chroma codes 0, no rows past the buffer; a 4:4:4 layout read as codes aside, whose identity conversion converts nothing), on a device that enables `samplerYcbcrConversion` (a 4:4:4 layout also needs Vulkan 1.3 or `VK_EXT_ycbcr_2plane_444_formats`), reports the multi-planar format's conversion and, for a 4:2:0 or 4:2:2 layout, Metal's reconstruction (midpoint chroma through a linear chroma filter), and has no defect in the format; the planes are copied into an image of it and sampled once per texel into the texture's image (`bind::ycbcr_sample_module`), through the refresh's indirect arguments. The conversion takes the surface's matrix and range (the identity model for the code-reading formats), and Metal's reconstruction of a subsampled layout; a device's cosited or nearest reconstruction would move what the codes mean to Metal (cosited, up to 1340/2048 of random planes on lavapipe), a capability it lacks, so such a layout converts through Metal's arithmetic. A render pass's load stays Metal's arithmetic, which our store inverts, so a loaded texel no draw changes keeps its codes, as `MTLLoadActionLoad` keeps the attachment's contents. lavapipe (Mesa 26.0.8) converts the 8-bit 4:4:4 layout (its 4:2:0 and 4:2:2 ones are cosited only, and it reads 10-bit codes as 16-bit words, `RuntimeDriverDefect::TenBitCodesReadAsWords`); MoltenVK 1.4.2 converts none (its 4:2:0 and 4:2:2 reconstruction is misplaced, `RuntimeDriverDefect::TwoPlaneChromaMisplaced`, and it reports no 4:4:4 two-plane format). RADV (Mesa 26.2.4) reads 10-bit codes as 16-bit words too (measured on a linear image of the 4:2:0 format on the Radeon 780M), so its 10-bit layouts convert through Metal's arithmetic. Elsewhere, and for packed codes, chroma codes 0 and plane 1, Metal's fixed-point conversion stays as capability (`WorkCounts::ycbcr_device_conversions`). | The device's conversion is the contract: Vulkan states the reconstruction's weights and the model's matrices, the precision of each step at least the format's, and the clamp of the result into `[0, 1]` the implementation's. | Each channel within 0.04/2048 of Vulkan's conversion past the `half` its image stores on lavapipe's 8-bit 4:4:4 layout; of Metal's read within 6/2048 where the result lies in `[0, 1]`, and unclamped outside it (random planes, every texel, `tests/ycbcr_device.rs`). `YCbCrFilterNotExact` (`ycbcr-filter-not-exact`) |
| A texel's `float` value is the device's conversion of its codes on every device whose texture unit is not Metal's (every device but MoltenVK), and a `half` result the device's `float` texel rounded to nearest even on every device: on MoltenVK the `float` texel is Metal's, whose `half` Metal rounds otherwise for a 32-bit float texel's sample (toward zero, saturating at `±65504`) and the 16-bit normalized grids (three steps round twice through `float`); a two-plane YCbCr field on the grid `8192` keeps Metal's ties away from zero in one texel's read, sample and gather. An sRGB texel (the 8-bit sRGB formats, and the sRGB forms of BC1 to BC3, BC7, PVRTC, ETC2, EAC and ASTC, a decoded twin's through its `R8G8B8A8_SRGB` view) is the device's sRGB decode through the texture's sRGB view, in every read, sample, gather, bicubic tap and input-attachment read, where Metal decodes each code through its twelve-bit table; so is a resolve's into an 8-bit UNORM attachment's sRGB twin, which decodes the samples and encodes their average through the twin's sRGB views where Metal's resolve takes its own tables (`DeviceRoundedResolve`). An 8-bit sRGB storage texture held through its UNORM alias (Vulkan has no sRGB storage view) reads its codes by the MSL transfer through the device's `Pow`, and writes them by that transfer and `PackUnorm4x8` (`DevicePackConversion`). | The device's conversion is the contract. | One `f32` ulp for 8- and 16-bit codes, 0.18 of a code for 5- and 6-bit packed channels; `half` within one ulp, but a 32-bit float texel's sample past `±65504` the infinity of its sign where Metal saturates (MoltenVK included); NaN bits the device's. An sRGB decode: lavapipe's within 5.25 steps of Metal's table (0.00128) at every code, MoltenVK's Metal's at every code (`runtime_lane::srgb_decode`); Vulkan sets no precision for it, and other devices are not measured. `DeviceTexelConversion` (`device-texel-conversion`), `HalfTexelConversion` (`half-texel-conversion`, also recorded for heap textures and unmeasured formats), `SrgbTexelDecode` (`srgb-texel-decode`, under the bound format) |
| A texture of a BC format (`BC1_RGBA` to `BC7_RGBAUnorm_sRGB`) is the device's decode of its own BC image on every device: where its driver is not measured to decode the format as Metal does (BC1 to BC3 off MoltenVK and RADV on the 780M; BC4 and BC5 off MoltenVK; BC7 on lavapipe), its reads, samples and gathers answer the device's interpolation of the blocks, where Metal's weights are 6-bit (an sRGB form's code then through the device's sRGB decode, the texel-conversion row). No BC texture keeps a decoded twin. | The device has the format: its decode is the contract. | A BC1 to BC3 channel within 3/255 of Metal's on NVIDIA (BC3's alpha 6/255) and lavapipe, 1/255 on ANV; a BC4 or BC5 channel within 7.3/255 (signed 0.064) on NVIDIA, 1.3/255 (signed 0.013) on ANV, 2.4/255 (signed 0.022) on lavapipe, a signed one with an endpoint of -128 within 1.7/255 on RADV; a BC7 channel a `float` ulp on lavapipe; other drivers unmeasured. `CompressedTexelDecode` (`compressed-texel-decode`) |
| A vertex attribute that fixed-function vertex input fetches is the device's conversion of its format: a normalized one (`UCharNormalized` through `Short4Normalized` of either sign, `UChar4Normalized_BGRA`, `Int1010102Normalized`, `UInt1010102Normalized`) through its own Vulkan `UNORM` or `SNORM` vertex format, where Metal divides each code by its maximum correctly rounded (an `Int1010102Normalized` 10-bit field it multiplies by `float(1 / 511)` in `float`) and rounds the quotient once to a `half` attribute; a narrow float's (a `half` format, `FloatRG11B10`) read as `float` by the device's widening, NaNs included. The format is `VertexAttribute` state that no instruction names, so every format the device fetches shares one module, and the render pipeline's link notes the conversion on its vertex stage (`pipeline::link::vertex`). A normalized attribute the shader pulls instead (an offset, stride or step fixed-function input cannot fetch, a kernel's `[[stage_in]]` attribute) is the module's product of the code and its maximum's reciprocal, `float(c) · float(1 / (2^b - 1))`, the signed one clamped to -1 (`lower::stage_io::fetch`), noted by the function on every device where it misses Metal's fetch. **Exempt:** a normalized attribute on a profile that takes `DriverDeviation::MetalNormalizedVertexFetch`, MoltenVK on Apple's GPUs of the Apple M3 Max's macOS 27 identity, which hands the format to Metal's own fetch and was measured to answer Metal's `float` at every code of all 19 normalized formats, per vertex and per instance (`runtime_lane::normalized_vertex_fetch`): its pipeline records no note and requires `VkProperty::MetalNormalizedVertexFetch`, but for a 16-bit format read as `half`, whose `float` the module rounds to `half` again. | The device's conversion is the contract. | One `f32` ulp at some codes on a driver that multiplies by the reciprocal: llvmpipe, read at every code (Mesa 26.0.8 on LLVM 21.1.8, arm64), is within one ulp per vertex at every normalized format (127/255 as `0x3EFEFF00` where Metal reads `0x3EFEFEFF`; `Int1010102Normalized` Metal's), and per instance reads the most negative code of an 8- or 16-bit signed format unclamped (-128/127, -32768/32767) where Metal reads -1; RADV's unsigned formats are Metal's at the sites measured. A 16-bit format read as `half`: one `half` ulp at codes 65455 and 65519 unsigned, 32759 and -32759 signed, on a device whose `float` is Metal's (the quotient rounded to `half` twice). The signed formats are **unaudited** on RADV, ANV and NVIDIA. A pulled attribute's product, on every device: one `f32` ulp from Metal's quotient at 126 of the 256 8-bit unsigned codes (high), and toward zero at 16 signed 8-bit codes, 24 unsigned 10-bit codes, 512 unsigned and 1,536 signed 16-bit codes; `Int1010102Normalized` read as `float`, and an 8- or 10-bit format read as `half`, Metal's at every code; a 16-bit format read as `half`, one `half` ulp at the same four codes. A narrow float's NaN is the device's widening's, where Metal answers `0x7fc00000`. `DeviceVertexConversion` (`device-vertex-conversion`), `DeviceNanEncoding` (`device-nan-encoding`) |
| A texture of an ASTC LDR format on a device without that ASTC format reads a decoded twin of 8-bit codes, each channel the code nearest Metal's 12-bit value (`C >> 4` over 4095); an sRGB form's twin holds Metal's own 8-bit codes, which its slots read through the twin's `R8G8B8A8_SRGB` view, the device's sRGB decode (the texel-conversion row). A texture of an EAC 11-bit format (`R11`, `RG11`, unsigned and signed) on a device without ETC2 reads one of 16-bit normalized codes, each nearest Metal's value (`c / 2047`, signed `c / 1023`). | The device lacks the format; the twin's codes are the compact form the device samples. | Within 1/510 of Metal's value; an EAC 11-bit channel within half a 16-bit step (1/131070, signed 1/65534). `decoded-texel-precision`, announced where such a texture is made |
| A texture of a 4:2:2 format (`GBGR422`, `BGRG422`) reads a decoded twin of 8-bit codes on every device (`chroma_pairs.rs`): where the device converts the pairs' format, `G8B8G8R8_422_UNORM` or `B8G8R8G8_422_UNORM`, with Metal's reconstruction (its optimal tiling reports `SAMPLED_IMAGE`, `TRANSFER_DST`, `MIDPOINT_CHROMA_SAMPLES` and a linear chroma filter; an even width), each decode pass copies the written rows' pairs into an image of that format and samples it through the device's own sampler Y′CbCr conversion (the identity model at full range) into the twin (`ycbcr_device.rs`, `bind::chroma_sample_module`), each channel the nearest 8-bit code to the device's value; elsewhere the twin holds Metal's decode, `(3c + n + 2) >> 2` of the texel's pair `c` and the pair beside it `n`, as capability. MoltenVK 1.4.2 converts both formats; lavapipe (Mesa 26.0.8) filters their chroma nearest only and RADV reports neither format, so theirs hold Metal's decode (`WorkCounts::chroma_device_twins`). | Vulkan views a 4:2:2 format only through a sampler Y′CbCr conversion, which no pipeline's layout bakes: the twin is the form every slot samples, and where the device converts the format the conversion is the contract. | Green the texel's own code; red and blue within half a code of Vulkan's `(3c + n) / 4` on MoltenVK 1.4.2 (the nearest code; random pairs, every texel of 40 x 24, `tests/ycbcr_device.rs`), where Metal's `(3c + n + 2) >> 2` rounds that sum's ties up. `DecodedTexelPrecision` (`decoded-texel-precision`), announced where such a texture is made |
| A two-plane YCbCr texture whose planes are read in place (`texture/ycbcr_image.rs`) is sampled through the device's own sampler Y′CbCr conversion where the device converts their 4:2:0 format as a linear image with Metal's reconstruction (its linear tiling of `G8_B8R8_2PLANE_420_UNORM` or `G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16` reports `SAMPLED_IMAGE`, `MIDPOINT_CHROMA_SAMPLES` and a linear chroma filter, and its driver has no defect in the format): the planes are also a linear image of the format bound over their bytes in memory the runtime allocated (`texture/ycbcr_planes.rs`; an even size, the driver's planes at the surface's offsets and row pitches), and a slot whose every sample the device's sampler takes (no border address mode, no texel offset placed by a parent's size) binds it as a combined image sampler whose immutable sampler bakes the conversion (`bind::YCbCrPlanesSampler`), each sample the device's at level 0, a nearest one at the centre of the luma texel Metal reads. Elsewhere the texel buffer read stays, Metal's fixed-point conversion and exact filter in the shader, as capability: on a device whose conversion is not Metal's reconstruction (lavapipe's 4:2:0 conversion is cosited only, MoltenVK 1.4.2's misplaced, `TwoPlaneChromaMisplaced`; RADV reads 10-bit codes as words, `TenBitCodesReadAsWords`), over fresh pages, a no-copy buffer's pages or the caller's `VkBuffer` (an image bound to imported memory starts `UNDEFINED`, and the in-place check covers that first move for single-plane images alone), where the driver lays the planes out otherwise (RADV's linear rows lie 256 bytes apart), and for a slot that samples with a border or a parent-placed offset (`WorkCounts::ycbcr_planes_images`, `ycbcr_planes_images_kept_*`). | Where the device has Metal's reconstruction, its conversion and filter are the contract; a device without it reads Metal's texels through the texel buffer, which is the capability it lacks. | The device's conversion and filter of the codes, not Metal's. A bare Vulkan probe on the Radeon 780M (RADV, Mesa 26.2.4, 8-bit codes) read each luma centre within 0.0003 codes of Vulkan's model and points between texels within 0.66 codes, where Metal's conversion then filter differs by up to 54/255 between texels of random planes (the device reconstructs the chroma at the point); the runtime's path is not measured on a device that takes it. GPU time of that probe, 3840 x 2160 composited into 1680 x 1050: 248 µs against the texel buffer's 340, one pass. `YCbCrFilterNotExact` (`ycbcr-filter-not-exact`), announced where the image is made |
| `generate_mipmaps` of a filtered format is the device's filter of each level at the scaled texel centres: its blit chain (`vkCmdBlitImage` with `VK_FILTER_LINEAR`) where its blits filter the image's format, else the runtime's bilinear (trilinear in 3D) compute filter in `float` (an extended-range format, an 8-bit sRGB texture held as its UNORM twin, a format without blits or linear filtering), by the device's arithmetic: its products and sums as it contracts them, an 8-bit or extended-range sRGB code through the MSL transfer by its `Pow`, a half by its widening and narrowing, a NaN its own. An integer format's texels are picked as plain Metal picks them, exactly. | The device's filter is the contract; Apple's datapath is not modelled, and its answers differ between Apple's GPUs. | `DeviceFilteredMipmaps` (`device-filtered-mipmaps`) |
| A draw biasing a `Depth16Unorm` depth by a constant, or by a slope alone, biases by Metal's constant `c` in the device's own unit `r` on every device: Metal biases by `c + 1` units of 1/65535 for `c` of 0 or more and `c - 1` below (one unit for a slope alone), added to the depth rounded toward zero, where Vulkan's `r` of a 16-bit unorm depth is the implementation's, at most `2 × 2^-16`, and the device rounds the sum its own way. MoltenVK hands Metal its constant (`DriverDeviation::MetalDepth16BiasedStores`). | The device's depth bias is the contract. | Another code than Metal's for a biased depth: one unit's difference or more. `UnormDepthBias` (`unorm-depth-bias`) |
| A render pass that loads and stores an extended-range, padded 10-bit or bfloat color attachment (`BGR10_XR`, `BGRA10_XR`, `RGB10A8_2P_XR10`, `R10Unorm_X6`, `RG10Unorm_X12` and their sRGB twins, `RGB10A2Unorm_sRGB`, `R16Bfloat`, `RG16Bfloat`, `RGBA16Bfloat`) keeps the bytes its draws do not write, on every device, MoltenVK included (such an attachment is an image of an unsigned integer format there too): Metal's pass writes each 32 x 32 tile a draw reaches back through its tile store, the unused bits of the codes made canonical, `BGRA10_XR`'s alpha code clamped to `[384, 894]`, a padded sRGB twin's codes through Metal's table and a bfloat code through `half` (a NaN to `0x7fc0`). A single-sampled signed normalized or `RGB9E5Float` attachment likewise keeps its minimum codes and unnormalized words, but on MoltenVK, whose pass is a Metal pass (`DriverFact::TileStoreIsMetals`). | The device's load and store are the contract: no pass rewrites the texels its draws do not write. | Bytes no Metal write left (uploads, copies) differ in the tiles a draw reaches; an extended-range texel reads the same value either way, a bfloat code off `half`'s grid reads its own value. `ExtendedRangeBitsKept` (`extended-range-bits-kept`), `SnormMinimumKept` (`snorm-minimum-kept`), `SharedExponentKept` (`shared-exponent-kept`) |
| Depth values outside [0, 1] stored clamped (RADV, ANV) on a device without `VK_EXT_depth_range_unrestricted`. | Vulkan leaves them undefined without the extension: enable it. | `RestrictedDepthRange` (`restricted-depth-range`) |
| A multisampled 32-bit float depth written by a blit copy from a buffer or a colour texture, or a separate depth texture's copies to and from its pass's scratch image, on a device whose depth/colour copies are missing or wrong (lavapipe, MoltenVK, NVIDIA's GA10x, a device without `VK_KHR_maintenance8`): the values its driver's fragment depth changes (MoltenVK's, NVIDIA's), or on a driver not measured may change (outside [0, 1], denormals, -0), land changed. | Vulkan writes a multisampled depth image's samples by no other means there but a clear; a driver measured to store every fragment depth unchanged (lavapipe's measured releases) is exact. | `DrawnDepthCopy` (`drawn-depth-copy`) |
| Signed zeros, infinities and NaNs where the device lacks `SignedZeroInfNanPreserve` at the operation's width. | A device capability. | `DeviceSignedZeroInfNan` (`device-signed-zero-inf-nan`) |
| Every device but MoltenVK interpolates varyings by its own rounding, up to about `1.5e-4` from Metal's (`3e-4` on lavapipe). MoltenVK hands the varyings to Metal. | The exact path needs Metal's interpolation formula, which is not known. | `DeviceInterpolation` (`device-interpolation`) |
| Every device but MoltenVK on Apple's GPUs of the Apple M3 Max's macOS 27 identity covers by alpha (`alphaToCoverageEnabled`) by its own algorithm: lavapipe takes a 2x2 ordered dither at one sample too, where Metal covers a pixel there only above an alpha of 1/2, so a whole fragment is stored or dropped between a pixel's threshold and 1/2; at four samples it is Metal's but within an `f32` ulp of a threshold, one sample apart. MoltenVK hands the field to Metal. | The device's coverage is its native answer, and Vulkan specifies no algorithm for it; Metal's dither computed in the shader would emulate the device's own coverage. | `DeviceAlphaCoverage` (`device-alpha-coverage`) |
| A fast-math function's `half` color into a blend: within three `half` ulps (16-bit float) or two `float` ulps (32-bit float). A fetched 16-bit float attachment's blend, computed in the pixel interlock, where Metal folds the output's own arithmetic into it or regroups it: within 2 `half` ulps at the operands' magnitude, 16 at the stored value's under cancellation; exact elsewhere. | Metal's compiler regroups the blend per pipeline, following its passes over the whole function: a form the blend state decides is Metal's only where it is measured, and a lane of the output the compiler can fold into the blend has no form at all. Two 16-bit float states are measured, and a fetched attachment's blend takes Metal's forms there; `SourceAlphaSaturated` over `OneMinusSourceAlpha` shows why the rest are not, Metal rounding the destination's product first at one and four channels and the source's at two. A device that enables `storageInputOutput16` receives the output as a 16-bit `half` (`StageIo::half_outputs`): MoltenVK 1.4.2 on the M3 Max then stores Metal's codes at every word of `half-output-blends-*` (a loaded `half4`, 21 blends into 16- and 32-bit float formats). | `HalfOutputFastMathBlend` (`half-output-fast-math-blend`) |
| A fetched blended attachment of a format whose Metal blend arithmetic is not built (16-bit normalized, 8-bit signed normalized, 10-bit, packed float, `A8Unorm`; an 8-bit unorm one through fewer than four output lanes; a 32-bit float one (`R32Float`, `RG32Float`, `RGBA32Float`); a 16-bit float or 8-bit unorm one under a blend of the second source), on a device that orders fetch only through the pixel interlock (MoltenVK, RADV, ANV): the shader blends in `float`, each step rounded, the source and blend color clamped to a normalized format's range, the device's conversion storing the result. An 8-bit sRGB one under a blend of the second source, a one- or two-channel one under a fast-math lerp, and one written by an output of fewer than four lanes: the 8-bit unorm blend's rounded forms over the decoded texel, encoded by Metal's table (Metal's without fast math, at four lanes). Bound not measured. | No Vulkan blend applies to the interlock's storage write, and Metal's blend arithmetic for these formats is not measured, so no exact form is known. A 32-bit float attachment's blend in `float` is the device's own arithmetic, as its blend unit would compute it (every committed profile but the Vulkan 1.2 floor blends the three formats; a pipeline whose attachment is not fetched takes that blend). | `UnmeasuredFetchBlend` (`unmeasured-fetch-blend`) |
| A color blended into an extended-range attachment (`BGR10_XR`, `BGRA10_XR`, `RGB10A8_2P_XR10` and their sRGB twins) is blended in the shader, inside the pixel interlock (on lavapipe over its input attachment), in the device's arithmetic: `S·Fs op D·Fd` in `float`, in the blend equation's order, `1 - x` in `float`, `SourceAlphaSaturated` `min(sa, 1 - da)`, the source, second source and blend color clamped to the format's range first (a NaN to its low end), then encoded as the kernel write encodes. Every device. Not measured against Metal. | No device's blend unit reads the format's raw codes, so the shader blends (capability), and its arithmetic is the device's: Metal's compiler fuses the blend into the function, regroups it and rounds each step to `half`, a form no device has natively. | `ExtendedRangeBlendSource` (`extended-range-blend-source`), `ExtendedRangeFloatBlendSource` (`extended-range-float-blend-source`) |
| Float simdgroup reductions sum in the device's association order. | Apple's order is not modelled. | `ReductionOrder` (`reduction-order`) |
| A vertex or fragment function keeps more than 16 KiB a thread of private memory, up to the 192 KiB Metal holds, on a device whose Function storage was not measured: it runs on the driver's scratch, which past what the driver holds answers wrongly or hangs the GPU with no error. | Vulkan states no amount of Function storage; only a measurement of the device does. No other form is exact: a fragment function's helper invocations may not write the memory it would move to, and a vertex function's, a compute pre-pass, is not built. | `FunctionStorageAssumed` (`function-storage-assumed`) |
| A render pipeline of more color attachments than the device draws at once (the lesser of `maxColorAttachments` and `maxFragmentOutputAttachments`, four at Vulkan's floor; Metal takes eight) draws each draw once per group of that many, each group in a render pass of its own over a twin of the pass's depth and stencil. Every variant's vertex stage is one module, `Position` invariant, and every fragment variant's `FragDepth`, `SampleMask` and `FragStencilRefEXT` are `Invariant`; groups past the first may differ at discard or depth edges if the driver compiles the variants differently. | The device holds no more attachments in one subpass. Vulkan's invariance rules make only identical pipelines repeat exactly, and the groups' fragment modules and attachments differ. | `ColorAttachmentGroups` (`color-attachment-groups`) |
| On NVIDIA's driver (615.71.09 on an RTX 5080 Laptop GPU) one `vkQueueSubmit` in every 1024 that signal on a `VkQueue` takes 2 to 4 ms in the driver (1.97 to 4.06 ms measured), which holds the command buffer whose batch it falls on, and the ones committed behind it, from the GPU that long. | The cost is the driver's, and every batch signals its completion. The scheduler's thread takes the stall while idle, by empty submissions, once one was seen (`DriverFact::SignallingSubmitsStallEvery1024`, `device/submit_stalls.rs`; a call of 1 ms or more, and once one came where the count put it, only there): it falls on work only before then, after a use of the queue the runtime cannot count, or on a queue that never idles 1 ms. Whether the driver's stall also holds other threads' semaphore calls is unsettled: a probe polling the timeline from a second thread during 10 stalls saw 5 of its 9 slow calls (up to 7.1 ms) overlap one in one run, and none of its 3 in another; in `host_ab present` the completion handlers of command buffers that completed during a stall taken idle ran 126 us (p50, 341 at most) after their GPU end, as elsewhere. | None (timing only; `WorkCounts::submit_stalls_on_work`, `WorkCounts::submit_stalls_idle`) |
| Command buffers that share no resource run one after another, where Metal overlaps them while the GPU has room. | The runtime orders nothing between them and runs at each driver's floor for one completion signal per command buffer; a signal shared by several would hold an earlier one's completion for the later ones' work. | None (timing only) |
| On MoltenVK a light queue's command buffer waits for the latest one another queue committed, where Metal waits for none: MoltenVK fences every encoder of every queue device-wide when argument buffers and a residency set are on. | The runtime needs argument buffers, and MoltenVK has no setting for its residency set; `runtime_lane::queue_storms` judges the runtime against Metal under those fences. | None (timing only) |
| A `Counting` visibility result of a draw writing more than one color attachment, on ANV (Mesa 26.2.4): Metal's count multiplied by the colors written, not divided back. `Boolean` results stay exact. | The driver counts each sample once per color output (`RuntimeDriverDefect::OcclusionCountPerColorOutput`), no documented Vulkan form counts it once, and a refusal would withhold the Mac2 and Metal3 families. | `VisibilityCountPerColor` (`visibility-count-per-color`) |
| On MoltenVK a no-copy buffer or heap released with no keep-alive while batches are in flight runs its deallocator once the batches submitted before its release completed, and the program's own submissions through `CommandQueue::with_vk_queue` before it, where Metal runs it at the release. | MoltenVK 1.4.1 fails pending work that reads device addresses where memory a device-address buffer lay in is freed (`RuntimeDriverDefect::FreedAddressMemoryFailsPendingWork`), so freed memory waits for the batches in flight (`WorkCounts::freed_memory_parked`). | `HostDeallocatorDelayed` (`host-deallocator-delayed`) |
| On MoltenVK a `Shared` or `Managed` texture whose format packs more than one texel into a block (every block-compressed family) keeps its image out of host-transfer memory, so its CPU copies take submissions where another driver's take none. The stored texels are Metal's either way; what changes is the work, in `WorkCounts::command_buffers_begun_batch_work` and the `command_buffers_begun` and `command_buffers_submitted` that carry it (`bc1-rgba`, four 16 x 16 `BC1_RGBA` levels: `command_buffers_begun` 8 on the bind-time path and 15 off it; `fresh-texture-bc1-rgba-managed`: 3 and 5). A `replaceRegion` of such a texture stages its bytes for the device's next submission and waits for none (`texture/first_move.rs`). Its CPU **reads** come out of a host shadow of the image's texels, made at the first read of the texture by the submission that read waits for either way and read from by every read after it: exact (it holds the image's own bytes, and a CPU write writes both), bounded (`host_shadow::SHADOW_BYTES`, past which a read takes the submission path), and lazy, so a texture nobody reads back holds nothing. It is dropped wherever anything but a CPU write may have written the texels: every commit of the program's, its own Vulkan submissions, another `VkImage`, a detach, or the program taking the image's handle (`texture/host_shadow.rs`; `WorkCounts::texel_shadows_made`, `texel_shadow_reads`, `texel_shadows_dropped`). Measured on an M3 Max: `compressed-moves-bc1-rgba` reads `internal_submissions` 5 where it read 56 and `command_buffers_begun` 30 where it read 81, for 1,168 bytes of shadow over its five textures and 56 reads out of them. | MoltenVK 1.4.2 binds such an image through a `flushToDevice` that `replaceRegion:`s every subresource at one block's bytes a row instead of the row of blocks, tripping AGX's `bytes_per_row >= used_bytes_per_row` assertion (`RuntimeDriverDefect::HostTransferImagesCopiedAtBind`). The copy is the driver's own and no call of the program's asks for it, so the only way not to provoke it is not to ask for the image; every other driver keeps the host-transfer path, as does a device without `VK_EXT_host_image_copy`. A write cannot leave the GPU's copy of the texels unwritten, so only reads come off the host shadow. | None (cost only, and exact) |
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
  Max. On the Iris Xe, frames of 24 command buffers committed back to back
  (`runtime_lane::commit_costs`, `frames`) take a median 5.7 to 6.0 ms against 3.2 to 4.5 ms with
  the submissions on the committing thread, the GT held at its efficient 300 MHz by its governor
  (GuC SLPC). A submission check or submission handler, which run on that thread, holds back every
  other command buffer's submission while it runs.
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
  `min_lod_clamp`) or reads its size reads the planes' codes in place, through a uniform texel
  buffer over their bytes, and converts and filters them in the shader as Metal does; every other
  device, and every other use (a `read()`, a gather), converts the texture's image from the planes
  before the command buffer's first use (the refresh compares every word first) and reads the
  image, which holds Metal's texels exactly, or the device's own Y′CbCr conversion of them where
  it converts the format (the device-gaps row), and whose linear sample is the device's filter
  (`device-sampling`; on MoltenVK `ycbcr-filter-not-exact`). Neither is a switch: the device's kind
  picks the cheaper. GPU time of a 3840 x 2160 texture whose planes lie in host memory composited
  into 1680 x 1050 (µs, best of 3 x 40, unchanged planes; Mesa 26.2.4, MoltenVK 1.4.2):

  | Device | Codes | In place, 1 pass | Image, 1 pass | In place, 3 passes | Image, 3 passes |
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
  again at the first blit after the window (HOWTO.md section 6).
- **A pass in color groups runs once per group** (the `ColorAttachmentGroups` row above), the later
  groups' draws again over a twin of the depth and stencil, made for the pass and loaded by a copy
  or a clear (`render/color_groups.rs`). On lavapipe restricted to four colors
  (`runtime_lane::color_groups::color_group_costs`), 16 fullscreen draws into five 512²
  `RGBA16Float` colors took 1.04 times the same draws into four from commit to completion, and 1.51
  times with a loaded `Depth32Float_Stencil8` the draws write.
- **A draw that reads its own pass's attachment reads a copy, a snapshot or a split.** A sampled
  slot binding of a texture the render pass instance renders reads a copy taken before the
  instance (`texture/attached_reads.rs`): the texels around the draw's scissors where it reads at
  its own pixels alone, else every texel or what the instances since drew. After a barrier inside
  the instance, an own-pixel read takes a snapshot drawn in its subpass (`render/snapshot.rs`,
  docs/API.md "Snapshot reads") or splits the instance; both are exact. Over a texture whose image
  lies over its buffer's bytes (a linear image over an ordinary buffer's packed rows) it takes the
  snapshot at any area, where
  the device makes that linear image with feedback loop usage
  (`WorkCounts::feedback_loop_alias_images`; NVIDIA's drivers make it without, and a surface's
  readers split there): on the Iris Xe (`runtime_lane::attachment_read_costs::snapshot_crossover`,
  a 1920x1080 surface) 0.56 to 0.99 times the split's GPU time through scissors of 960x540 and up,
  at most 1.08 times through the whole target and 1.2 times through a 320x180 scissor it covers
  half of or more.
  Elsewhere the rule weighs the snapshot draw at 20 bytes of the split's copy for each pixel its
  scissors bound, against a copy of every texel of the color (an upper bound of the split's copy,
  which is cut to the scissors or to what changed for a one-level, one-slice texture), which fits
  the RX 7900 XTX and the RTX 5080 Laptop GPU; on the Radeon 780M the split is faster from about
  14.5%, 17% and 42% of the target for 4-, 8- and 16-byte texels, so the rule costs up to 36% there
  (`RGBA16Float` at 30% of a 1920x1080 target), and on the Iris Xe from an eighth of the reader's
  scissor covered, whatever its area, so the rule costs up to 2.1 times there (a 320x180 scissor
  covered whole over a private 1920x1080 `BGRA8Unorm` texture).
- **A texture over a no-copy buffer's host pages, or over a host-visible buffer longer than 32
  MiB, that renders or writes (or whose class's check has not passed, or whose pitch no image
  takes, or under `HostPageTexturePolicy::Shadow`), and a texture over a buffer of the runtime's
  memory whose rows lie with bytes between them,** is a shadow (`shadowed-texture`, its reason
  named): its texels copied in before each use and merged back after each write (only those inside
  the draws' scissors for a pass that takes it as a loaded attachment alone). One over host pages
  that only samples or reads is an image in place where its class's check passed (the row above,
  `in_place.rs`). An image bound to host pages starts `UNDEFINED`
  (`VUID-VkImageCreateInfo-pNext-01443`), and its first move may read and write every byte bound
  to it, which the host could keep only by undoing a CPU write made meanwhile, where Metal keeps it
  (the device's check is what takes a sampled one in place); an image over padded rows holds the
  bytes between them, which a render pass's load and store operations may rewrite too. The shadows
  over host pages are counted by why (`WorkCounts::host_texture_shadows_*`; a first session's
  textures made before the check answered, `host_texture_shadows_check_pending`). A host-visible
  buffer
  longer than half a pooled block (32 MiB) lies in fresh anonymous pages imported by their pointer
  where the device imports them and the memory class's best type for the buffer takes such an
  import (`fresh_pages.rs`; on a discrete card the write-combined class's best type is a
  device-local host-visible one that a host-pointer import cannot bind, so that buffer is placed
  as ever and cleared on the host), which read zero as mapped, so its creation
  writes none of its bytes, and the CPU's first touch of its pages is paid where the program makes
  it, as on Metal (host_ab on the M3 Max, 64 MiB: made in 0.01-2.2 ms against 3.2-4.7 ms of
  allocation and memset, its first CPU write 2.4-6.5 ms against 0.7-1.4 ms, Metal's 3.7-7.4 ms):
  such a buffer's textures that render or write are shadows for the same reason
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
  theirs on every device.
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
  took it ended, without waiting for the command buffers that used it, which the merge's closing
  barrier and the image's tracked uses order (`texture/kept_image.rs`); at most two that no open
  encoder holds stay for each view and extent, and none untaken over 256 takes. On the Iris Xe
  (`surface_texture_costs::surface_texture_cost_parts`, four shadowed textures a frame, 1,582,016
  texels), over anonymous pages, merged back, passes that draw took 5.0-5.4 ms a frame from commit
  to completion, and under scissors (790,448 texels) 3.6-3.9 ms; over tracked `Private` buffers,
  copied back, 3.1 ms and 2.2-2.4 ms.
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
- **Emulated tessellation's scan.** On a device without `tessellationShader` a patch draw first
  runs a one-workgroup scan that places each patch's triangles in Metal's order, patch after
  patch: about 6-60 ms of GPU time per 10,000-patch draw at factors 4 to 64 on the Radeon 780M and
  the RTX 3090, 0.6-5 ms on lavapipe (`emulated-tessellation-draws`). A parallel form is not built.

## Known gaps

Open differences from Metal that no note announces yet, and runtime costs still above the
driver's. Each is open work.

- **A copied host buffer costs O(its size) at every commit and every batch's completion**
  (`copied.rs`, `copied-host-pages`): each compares every byte of every live copy with the bytes
  the copy and the pages last agreed on, whichever buffers the work names, where a page log or the
  work's uses could bound it: a live 64 MiB copy added 45 to 68 ms to each empty command buffer's
  commit and wait (0.2 ms without) on an RTX 5080 laptop (Intel Core Ultra 9 275HX). A copy is
  made only where the caller's gate refuses an import or an admitted import fails, never by
  default.
- **A copied host buffer's merge keeps the CPU's byte where the CPU and the GPU both wrote it**
  between a commit and the completion: bytes are merged one at a time, so a GPU write never
  rewrites a byte beside it, but a byte both wrote keeps the CPU's, where on Metal the later write
  wins. A heap's rebind is never copied, nor a buffer's rebind with an image in place over the
  buffer: a refused or failed import of either still refuses the call.
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

- **Command queues past the device's `VkQueue`s share one, so one's command buffers wait behind
  another's whole command buffer where Metal's wait only inside one command.** Measured on Metal's
  own API on an Apple M3 Max: a side command queue completed one command buffer behind a single
  2.50 s dispatch, waiting its whole duration, and 11,335 to 12,992 behind the same 2.5 s of work
  as 256 dispatches, none waiting past 3.8 ms. So Metal holds an unrelated queue for one long
  command and for nothing else, while one `VkQueue` runs its submissions in order and cannot
  schedule a sharing queue out of them. A device now lends the round robin every queue its families
  have past the two reserved for presenting, at least four and at most sixteen, which on a graphics
  family of sixteen gives fourteen; a device whose families have none to spare -- MoltenVK's four
  families of one queue, a single-graphics-queue device -- still shares, and the sharing is counted
  (`WorkCounts::queues_sharing_a_vk_queue`). Not announced.

- **GPU writes into read-only memory reach it off macOS.** Metal drops what a shader or the blit
  engine writes into memory whose backing is read-only (no-copy memory of its private option bit 24,
  or pages replaced read-only): a kernel's or a draw stage's stores, through a bound slot or an
  address loaded out of an argument buffer, a blit fill and a visibility result complete, and
  neither a later command buffer nor the CPU sees them, though a later command of the same command
  buffer may read them back from the GPU's caches; what its atomics do there is unmeasured. A render
  pass's store-out over such memory (`storeAction = Store`, or a multisample primary stored there)
  ends its command buffer in error (`MTLCommandBufferErrorDomain` /
  `kIOGPUCommandBufferCallbackErrorPageFault`), while attachments at `storeAction = DontCare` or
  `loadAction = Load` complete; resolving *into* such memory is unknown (no page fault in 520
  processes, almost all of them recovery victims). On macOS the runtime imports such pages through a
  read-only view of them (`read_only_view.rs`), which takes the GPU's writes nowhere as Metal's
  memory does. Elsewhere the import is of the caller's own pages and the GPU's writes reach them:
  honouring bit 24 there means a read-only import (`Device::no_copy_options`, `host_lie`'s
  `drops_writes`) and measuring what the drivers then do, since ANV lost the device and RADV failed
  every submission over read-only imported pages. Announced meanwhile as `private-resource-option`
  and `read-only-pages-written`.

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
  read Metal's unwritten pattern (`0xFFFFFFFF`). The runtime maps and counts none
  (`UnitGrid::clipped_size`), so it loses no texel and no page. The address Metal's out-of-bounds
  slice produced is Apple's own page layout for the chip and no rule for it is measured, so the
  exact form cannot be derived yet; the conformance case
  `sparse-mapping-a-slice-past-the-textures` holds every side of the difference visible, with its
  four rows of `crates/metal2vulkan-conformance/known-gaps.tsv`.

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

- **A no-copy buffer imported by its pointer follows a remap of its pages** on a driver whose
  pointer imports follow their address (`DriverFact::PointerImportsFollowTheirAddress`: ANV on
  i915, RADV, lavapipe): pages mapped over the buffer's addresses after it was made (`mmap` with
  `MAP_FIXED`) are what its later command buffers read and write, where Metal's buffer keeps the
  pages it was made over. `HostMemory::new`'s contract forbids such a remap under a live buffer,
  and no notice tells of it. On RADV (Mesa 26.2.4, the Radeon 780M and the RX 7900 XTX) a file's
  pages mapped so over a live pointer import of anonymous pages lose the device: its next
  submission is rejected whatever it uses ("The CS has been rejected"): amdgpu's userptr takes
  only pages with no file behind them, and each submission lists every memory the device
  imported. No exact form exists once the pages are gone, and `Device::host_range_remapped` does
  not release such an import. An import of dma-bufs (`HostMemory::with_dma_buf`) keeps its pages
  on every driver, but is taken only where the device refuses the pointer.
- **A no-copy buffer's pointer import that stays live costs RADV's every kernel submission per
  page**: amdgpu gets every page of each live pointer import again in each command submission
  (the RX 7900 XTX under Linux 7.0.0: the scheduler thread's CPU per command buffer of a
  guest-shaped frame 18 us with none, 328 us with 275 imports of 512 KiB, 1144 us with 275 of
  2 MiB). An idle one is released where `DriverFact::PointerImportsFollowTheirAddress` holds (the
  780M and the RX 7900 XTX under Mesa 26.2.4), but one whose Metal address was handed out, or any
  on a device not measured with the fact, stays live. 275 kept dma-buf imports of 512 KiB each
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
- **A fetched or resolved packed 16-bit attachment on a device that does not render the format**
  (`A1BGR5Unorm`, `ABGR4Unorm` on the RTX 5080). Its word view (`WordAttachment`) takes writes
  (blended and partly masked ones through the pixel interlock, the row above); a color input or
  resolve is `NotImplemented(Format)`, a blend of an output of fewer than four lanes
  `NotImplemented`, and a blend of the second source, or a blend or partial mask on a device
  without the interlock, is refused as the device lacking the format.
- **A blended, fetched or resolved one- or two-channel 8-bit sRGB attachment on a device that
  does not render the format** (`R8Unorm_sRGB`, `RG8Unorm_sRGB` on the RTX 5080 and ANV). Its
  word view takes writes (partly masked ones through the pixel interlock, or over the attachment's
  input attachment on a device that orders fetch through them, lavapipe); a blend other than one
  and zero is refused as the device lacking the format, and a color input or resolve is
  `NotImplemented(Format)`. A device that orders fetch only through the pixel interlock and renders
  any color format through its word view (these, the packed 16-bit ones, `RGB9E5Float`) answers
  the `ProgrammableBlending` family row `NotImplemented` and `isFramebufferReadSupported` no.
- **A half output, blend, partial mask, color input or resolve of an `RGB9E5Float` attachment on
  a device that does not render the format** (the RTX 5080). Its word view takes a `float`
  output's writes and clears; a `half` output, a color input and a resolve are
  `NotImplemented(Format)`, and a blend or a mask of some of its channels is refused as the
  device lacking the format.
- **ASTC textures on a device without ASTC.** Making a heap's or a sparse heap's such texture
  (`RuntimeGap::DecodedHeapTexture`) and a view of another decode (its HDR form) are refused. The
  heap size query answers one: its footprint is Metal's blocks, which no image of the device
  decides.
- **The 32 formats this crate holds no image of** (`metal::format::footprint`: ASTC's 4x2 and 8x4
  footprints, `ARGB8Unorm_sRGB`, Apple's nine one-plane `YCBCR*` formats, the ten
  padded 10- and 12-bit display formats, the four `RGB10A8_2P` ones and the two packed 10-bit
  ones). `heapTextureSizeAndAlignWithDescriptor:` answers Metal's exact size and alignment for
  every one, because the footprint is arithmetic over the descriptor and no image decides it. A
  texture, a view, a copy, a read and a write of one stay refused `NotImplemented(Format)`: no
  Vulkan format holds the bytes (an ASTC 4x2 or 8x4 block) or their reads and writes are not
  measured yet. Plain Metal lays out no image storage for the two packed 10-bit
  ones and makes a texture from every creator that needs none -- a placement heap's offset, the
  shared creator in `Private`, a buffer's and an IOSurface plane's own bytes, and `Memoryless` --
  so the refusal from each of those is this crate's own and not Metal's nil; the device's own
  creator and an automatic heap, the two that would have to lay the image out, answer nil (an Apple
  M3 Max, macOS 27.0.1, `conformance/probes/packed10_texture_alone.m`). Their reads are a stream of
  10-bit codes that no Vulkan format holds, and none of that is built: no image is made, so
  sampling, a blit, `getBytes:` and a render pass of one are all unbuilt behind the same refusal.
  The code layout is derived in `metal/format/packed_codes.rs` and is **not** held to a texel: the
  two measured anchors fix codes 0 and 1 only, and a padded 32-bit word and an unpadded bitstream
  first differ at `R10Unorm_PACKED`'s texel 3, which nothing has measured. A **multisample** descriptor of one is refused there too, its layout
  unmeasured: Metal's validation may stop such a descriptor, so none was asked.
  `ARGB8Unorm_sRGB` is the one of them whose gap is not the bytes: they are `ARGB8Unorm`'s, which
  this crate holds, and no Vulkan format applies the sRGB transfer function to the three channels
  Metal's does -- every Vulkan `_SRGB` format encodes the word's low three and leaves the fourth
  linear, and Metal's alpha is the low byte here, which a view's component mapping cannot move
  because format conversion runs before the swizzle. Holding it needs a shader-side decode of its
  codes by the device's transfer (`pack::srgb_decode_texel`), as a UNORM twin's are, as the only read of it.
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
  reading an attachment in its own render pass instance (`VUID-vkCmdDraw-None-10795`); a sampled
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
- **Commit to completion.** A command buffer takes longer from commit to completion through the
  runtime than a bare driver submission: on a quiet Radeon 780M (RADV) an empty command buffer took
  51.6 µs against a bare barrier's 45.1 µs. On MoltenVK most of the rest is the two GPU timestamps
  each timed command buffer writes: on an M4 a bare barrier took 14.7 µs alone and 167.8 µs with
  them. `DeviceOptions::gpu_times_from_host` (off by default, announced) answers GPU times from the
  host clock and spares them.
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
- **A private logic operation other than copy and no-op without Vulkan's form.** On a device
  without `logicOp` (MoltenVK), and on any device where an attachment is sRGB, a float attachment
  blends, a format's operation was not measured or the function writes an attachment itself, the
  pipeline is refused (`PrivateRenderField::LogicOperation`), where the function could apply
  Metal's operation to the codes it reads and writes in the pixel interlock.

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
  this is the driver's and not ours. RADV and ANV unrun.

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
  to import a udmabuf of host pages, `vkAllocateMemory` answering `ERROR_OUT_OF_DEVICE_MEMORY` for
  8 KiB with 15,291 MiB of the card free (615.71.09). It also **closes the descriptor it was
  handed**, which the Vulkan specification gives it only on a successful import: `F_GETFD` on the
  descriptor reads `EBADF` the instant the call returns. Closing it again aborted the process on
  Rust's I/O safety check, so an import now hands the driver a clone and never closes a clone it
  has handed over (`memory.rs`), which lets go of one descriptor per failed import. Giving a
  well-behaved driver its descriptor back would need a measured driver fact; the pages' other route
  (importing the pointer) is the one NVIDIA takes for writable pages and is unaffected. A device
  that imports dma-bufs (RADV) is **unaudited** for the clone.

- **Reflection after a compute descriptor reset on RADV**: `max_total_threads_per_threadgroup`
  answers 0 where Metal answers 96, `required_threads_per_threadgroup` `0x0x0` where Metal answers
  `8x6x2`, and `threadgroup_size_is_multiple_of_execution_width` false where Metal answers true
  (`compute-descriptor-reset`). Both RADV chips, the RTX 5080 and llvmpipe, exact on MoltenVK.

These were found by replaying the conformance suite on a driver that is not Metal: RADV first, then
the whole 1,099-case set on an RTX 5080 Laptop GPU (615.71.09) and on llvmpipe (Mesa 26.2.4).
**ANV is unaudited** for every one
of them, and on a card without unified memory the comparison is not yet readable at all: the runtime
announces `shared-in-system-memory` for every `Shared` buffer and no recording taken on Metal could
declare it, which counts 699 of the 5080's cases divergent with 655 of them differing in nothing
else.

- **A texture over an IOSurface plane whose GPU rows overlap is served only as a shadow.** A plain
  plane of more than one row of over 16 bytes whose `bytesPerRow` rounded down to 16 is below the
  row is made: its rows share 1 to 15 bytes, a whole number of texels, between consecutive rows
  alone, and its shadow's copies hold the sharing with two regions a slice each way, which
  announces `colliding-surface-row-writes` (`linear_plan.rs`,
  `buffer_texture::plane_copy_regions`). Still refused
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
