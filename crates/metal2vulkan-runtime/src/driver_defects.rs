//! The defects of a device's driver in commands the runtime records for itself, keyed on the
//! driver's identity and GPU: `VkPhysicalDeviceDriverProperties`'s `driverID` and
//! `VkPhysicalDeviceProperties`'s `deviceID`, never its release. A device's defects are in its
//! properties ([`DeviceProperties::driver_defects`](super::DeviceProperties::driver_defects)), and
//! what a defect breaks is taken another way where an exact one exists, else refused there, typed
//! ([`Needs::RuntimeDriverDefect`]), never run wrong unannounced: only where a refusal would
//! withhold a feature of the families the device claims (a row of the feature table,
//! `families/features.rs`) does it run, announced as a degradation.
//!
//! Each defect lists the drivers that have it and, for each, the GPUs a probe found clean
//! (`deviceID`, the one it was measured under for a GPU that reports several:
//! `DeviceDescription::canonical_device_id`). Every other GPU of the driver has it, measured broken
//! or not measured at all, so a GPU nobody probed takes the exact path, and one probed clean then
//! needs only its `deviceID` in the list, not new code.
//!
//! None is a translation's (`DriverWorkaround` covers those), so none is in the compile profile,
//! a cache key or a digest.
//!
//! [`Needs::RuntimeDriverDefect`]: metal2vulkan::error::Needs::RuntimeDriverDefect

use metal2vulkan::device::DeviceDescription;
use metal2vulkan::error::RuntimeDriverDefect;
use metal2vulkan::vk::DriverId;
use std::collections::BTreeSet;

/// One driver that has a defect.
pub(super) struct Affected {
    pub(super) defect: RuntimeDriverDefect,
    pub(super) driver: DriverId,
    /// The GPUs (`deviceID`) a probe found without the defect; every other GPU of the driver
    /// has it.
    pub(super) clean: &'static [u32],
}

/// Every known defect, by driver.
///
/// `MultisampledDepthStencilColourCopies`:
/// - Mesa's lavapipe (`MESA_LLVMPIPE`): `handle_copy_image` in `lvp_execute.c` copies between a
///   depth/stencil and a colour format of different texel sizes (a combined `D32_SFLOAT_S8_UINT`
///   image and `R32_SFLOAT` or `R8_UNORM`) through `texture_map`, which maps sample 0 only, so the
///   other samples are never written. Read in Mesa 26.0.8's source and in main's of 2026-10-05,
///   and measured on 26.0.8 (`runtime_lane::separate_samples`, the render census): known broken
///   through 26.2.x.
/// - MoltenVK (`MOLTENVK`): `MVKCmdCopyImage::encode` copies between a depth/stencil and a colour
///   format through a temporary buffer sized per texel with no sample count, and Metal blits no
///   multisampled texture to or from a buffer, so samples land wrong. Read in 1.4.1's source and
///   measured on 1.4.1 (driverVersion 10401): known broken through 1.4.1.
///
/// `MultisampledStencilColourCopies` (split from the above):
/// - NVIDIA's (`NVIDIA_PROPRIETARY`): on the RTX 3090 (`deviceID` 0x2204, Ampere's GA10x),
///   `vkCmdCopyImage` between the stencil aspect of a 4-sample `D32_SFLOAT_S8_UINT` image and a
///   4-sample 8-bit colour image raises a graphics exception (Xid 13, "WIDTH CT Violation" or "3D
///   HEIGHT CT Violation") and loses the device: into the stencil at 4x1, 4x2 and 1x4, out of it
///   at 1x4, whether the colour image is a colour attachment, a storage image or both, while 2x2,
///   the depth aspect, and an `S8_UINT` image copy. Valid usage: the validation layer reports
///   nothing. Measured on 615.71.09 with a bare-Vulkan program (the probe):
///   known broken through 615.71.09. The RTX 5080 Laptop GPU (Blackwell, 0x2c19) on the same
///   release copies them (measured clean). Every other NVIDIA GPU and release takes the exact path
///   too until a probe finds it clean, where before only the 3090's generation did and Ada and
///   Pascal, unprobed, took the copy.
///
/// `NormalizedDepthPastUnitRange`:
/// - MoltenVK (`MOLTENVK`): Metal's rasterizer on Apple's GPUs reads and stores the depths of a
///   sloped triangle whose normalized depths reach past 1 about 2e-4 off (0.968567252 for
///   0.96875), with or without clip distances and in both depth clip modes, where Metal's own
///   viewport of the range draws the same depths exactly. Measured in Metal on an M4 with macOS
///   27.0.1 (a probe: the vertex stage writing the compensated depth into a viewport of `[0, 1]`,
///   against a native viewport of `[0.5, 2]`), and through the runtime's compensation on 1.4.1
///   (driverVersion 10401) on an M3 Max and an M4 (`runtime_lane::viewport_ranges`): known broken
///   through 1.4.1. The rasterizer is Metal's, so no MoltenVK release is expected to fix it, but a
///   probe of a later one decides; MoltenVK's other GPUs (Intel and AMD Macs) are unprobed and
///   refused alike.
///
/// `ImageSizeShortOfCompressionData`:
/// - Mesa's ANV (`INTEL_OPEN_SOURCE_MESA`) on Intel's Xe-LP GPUs: a compressed
///   image's compression data lies last in its memory, at a multiple of 4096 bytes, `size / 256`
///   bytes long (`add_compression_control_buffer` in `anv_image.c`), and the image's size ends
///   there, while the aux map gives each 64-KiB block of texels a 256-byte chunk of it (Mesa's
///   `intel_aux_map.h`) and the GPU writes it back in 64-byte lines. A 4x4 `R8G8B8A8_UNORM`
///   image asks 4112 bytes at 65536; a 1D image bound at byte 4112, written after it in the same
///   command buffer, read its old contents back at bytes 4112 to 4159, while one bound at 4160 or
///   4352 kept what was written. Measured on 26.2.4 (driverVersion 0x6802004) on Raptor Lake-P's
///   Iris Xe (`deviceID` 0xa7a0; `runtime_lane::texture_blits`, and a bare-Vulkan program) and
///   Raptor Lake-S's UHD 770 (0xa780); read in main's source of 2026-10-07: known broken
///   through 26.2.4. Nothing is refused: the runtime rounds each image's size up to whole
///   256-byte chunks there (`texture::ImagePlacement`). Every ANV GPU and release takes the
///   rounding, at most 255 bytes an image, until a probe finds it clean. Mesa's source pads the
///   data to the map's own granules on Xe-LPG (Meteor Lake, Arrow Lake) and keeps none in memory on
///   Xe-HPG and Xe2, but none of those is measured, so the Xe-LP list (70 `deviceID`s) that
///   narrowed it before is gone.
///
/// `DepthStencilResolveFirstSubresource`:
/// - Mesa's lavapipe (`MESA_LLVMPIPE`): `resolve_ds` in `lvp_execute.c` blits the depth/stencil
///   attachment's image into the resolve attachment's with the blit's `box.z` and `level` left
///   0, where `resolve_color` sets them from each view's `base_array_layer` and
///   `base_mip_level`, so a resolve from slice 1 of a multisampled array takes slice 0's samples,
///   and one into slice or level 1 writes slice or level 0. Read in 25.2.0's, 26.0.8's, 26.2.4's
///   and main's source of 2026-10-07, and measured on 26.0.8 and 26.2.4
///   (`runtime_lane::layered_passes`, whose slices read back by resolves all held slice 0's):
///   known broken through 26.2.4. Nothing is refused but a memoryless attachment at another
///   layer (`resolve.rs`).
///
/// `PipelineCacheDataCompiledAtLoad`:
/// - MoltenVK (`MOLTENVK`): `vkCreatePipelineCache` with initial data reads each shader module's
///   converted MSL back (`MVKPipelineCache::readData`) through
///   `MVKShaderLibraryCache::addShaderLibrary`, whose `MVKShaderLibrary` constructor decompresses
///   the MSL and calls `compileLibrary` at once, waiting for Metal's compile: every library in
///   the data is compiled before the call returns, whether a pipeline of the process uses it or
///   not. Read in 1.4.1's source (`MVKShaderModule.mm`); seen by the Vulkan consumer on an M3 Max
///   (1.4.1, driverVersion 10401) as `MTLCompilerService` rebuilding about 115 libraries at every
///   device creation from an 8.9 MB entry of 131 shaders, about 28 s for one process: known
///   broken through 1.4.1. Nothing is refused, and nothing branches on it any more: the device
///   keeps one cache per record, made once from that record (`shader_cache/driver.rs`), so a load
///   compiles the family's own stages alone and compiles them once for the device, not once a
///   making. Measured at 1.4.2 on an M3 Max: the record of 32 blend states over one
///   vertex-and-fragment pair is the 1522 bytes of one state's, since the data holds converted
///   libraries and no pipeline state, and its load is 2 of Metal's compiles and 50 us against the
///   64 compiles and 2639 us of loading the 32 records a cache per making wrote.
///
/// `LazilyAllocatedVolumePlaneViews`:
/// - MoltenVK (`MOLTENVK`): `MVKImageViewPlane::newMTLTexture` makes a 2D view of a 3D image
///   (the image `VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT`) as a 2D array texture over the
///   image's memory through its placement heap (`getHeapAllocation`), with no check that there
///   is one; `MVKDeviceMemory::ensureMTLHeap` makes none for memory of Metal's memoryless
///   storage, which is what a lazily allocated memory type is, so the view's first command
///   buffer dereferences a null heap and the process takes a `SIGSEGV`
///   (`runtime_lane::memoryless::a_memoryless_3d_depth_plane_renders` on an M4, 1.4.1,
///   driverVersion 10401, where the other types' passes ran: their views either are the
///   memoryless texture itself or have no texture, `LazilyAllocatedSubresourceViews`). Read in
///   1.4.1's source and main's of 2026-10-05 (`fae55a18`, unchanged): known broken through
///   1.4.1. Nothing is refused: such an image takes device-local memory
///   (`ImagePlan::memory_class`).
///
/// `CompressedImageCopiesRewriteTexels`:
/// - MoltenVK (`MOLTENVK`): a copy between a buffer and an image is a Metal blit of the image's
///   Metal texture, which Metal keeps losslessly compressed where its usage allows
///   (`texture/heap_tiled.rs`'s `metal_compresses`; an image of neither `STORAGE` usage nor
///   `MUTABLE_FORMAT`), and Metal's blits of a compressed `RG11B10Float` or `RGB9E5Float`
///   texture write other bytes than they read (`compressed_blit.rs`: NaNs made quiet, shared
///   exponents renormalized), so a CPU copy of a `Shared` texture of either format reads them
///   back so, where Metal's `replaceRegion` and `getBytes` keep the bytes. Measured on
///   1.4.1 (driverVersion 10401) on an M4 under macOS 27.0.1
///   (`runtime_lane::compressed_blits`): known broken through 1.4.1. The
///   compression is Metal's, so no MoltenVK release is expected to fix it, but a probe of a
///   later one decides. Nothing is refused: the image of such a texture a CPU copy reaches, and
///   of a device's own one without `allowGPUOptimizedContents` (whose GPU blits keep the bytes
///   on Metal), is made of `PixelFormatView` usage (`compressed_blit::copy_exact_usage`), which
///   Metal keeps uncompressed, so its copies keep the bytes as every other device's do.
///
/// `MultisampledCopiesConvertFloatNans`:
/// - Mesa's RADV (`MESA_RADV`): `gfx_or_compute_copy_image` in `radv_meta_copy.c` gives the
///   source's blit surface the destination's format wherever the source is not DCC-compressed or
///   the two formats are DCC-compatible (every pair on GFX11, `radv_dcc_formats_compatible`),
///   and a multisampled copy is then a draw (`radv_gfx_copy_image`) that fetches each sample in
///   that format and exports it: a NaN in a 16-bit float channel lands as `0xfe00`, its sign and
///   payload lost, while a single-sampled copy kept them. Read in 26.2's source (2026-10-09), and
///   measured on 26.2.4 (driverVersion 0x6802004) on the Radeon 780M (`deviceID` 0x15bf: every
///   NaN sample of a 4x `R16Float` and `RGBA16Float` texel copied from an unsigned-integer scratch
///   held `0xfe00`, a scratch not mutable to the attachment's format alike): known broken through
///   26.2.4. The path is every RADV GPU's, so every device has it. Nothing is refused: a blit copy
///   between multisampled images of such a format runs through views of them in the
///   unsigned-integer format of its texel size where both take one (`sample_planes::BitsCopy`),
///   which the runtime's multisampled color render targets of such formats take where it costs the
///   image nothing (`nan_clear::view_candidate`), and is the device's copy, announced
///   (`device-nan-encoding`), elsewhere.
///
/// `ReadsCommandBuffersAfterTheirSignal`:
/// - MoltenVK (`MOLTENVK`): on a device with `MTLSharedEvent` the GPU signals a timeline
///   semaphore (`encodeSignalEvent`), so the host sees the value before the Metal command
///   buffer's completed handlers run, and `MVKCommandEncoder::finishQueries` adds one that calls
///   into every query pool the command buffer reset or began, through raw pointers it does not
///   retain. A command buffer released at the value crashed there (`SIGSEGV` at that handler's
///   load of a released `MVKQueryPool`'s vtable;
///   `runtime_lane::visibility::command_buffers_released_at_completion_leave_nothing_read_later`
///   passed 0 of 8 runs on an M4). MoltenVK signals a fence from the last of those handlers. Read
///   in 1.4.1's source and measured on 1.4.1 (driverVersion 10401); read again in 1.4.2's, where
///   it is unchanged: known broken through 1.4.2.
///
///   **It covers the query pools of a command buffer that recorded a query command, and nothing
///   else.** Every `addCompletedHandler:` of 1.4.2's source was read, and `finishQueries` is the
///   only handler that reads a Vulkan object of the command buffer: the temporary-buffer handlers
///   reach the `VkCommandPool`'s encoding pool, which the runtime never destroys or releases
///   before the device is torn down; `MVKImagePlane::pullFromDeviceOnCompletion` reads a `VkImage`
///   but is registered only for an image memory barrier naming `HOST_READ` or `MEMORY_READ` into
///   `GENERAL`, which the runtime never makes (`image_barrier.rs`); and the rest read MoltenVK's
///   own objects or ones it retains for the handler. The handler is registered only where the
///   encoder collected an activated query (`MVKCommandEncoder::addActivatedQueries`).
///
///   Nothing is refused: a command buffer's last batch signals a fence there too, and the command
///   buffer finishes once both are signaled (`command_buffer/fences.rs`). Every command buffer
///   does, not only the ones the narrower scope above can reach: a MoltenVK device the runtime
///   creates relays its queues' failures, which needs a fence for every batch (`device/relay.rs`),
///   and the host sees the fence sooner than the value there anyway (measured on an M3 Max at
///   1.4.2, `fences.rs`). Every other driver ends a command buffer at its value alone: Vulkan
///   makes the value enough, and their fences are signaled with it, so a fence there only added a
///   `vkCreateFence` to the commit path, which NVIDIA's 615.71.09 held for up to 1.9 s while other
///   processes' work ran.
///
/// `WaitForAnyObjectLosesASignal`:
/// - MoltenVK (`MOLTENVK`): a host wait for any of several objects to be signaled, which the
///   driver serves by registering a wait on each of them in turn (`vkWaitSemaphores` with
///   `VK_SEMAPHORE_WAIT_ANY_BIT`, `mvkWaitSemaphores` in `MVKSync.mm`, over the shared events its
///   timeline semaphores are signaled through; `vkWaitForFences` with `waitAll` false has the same
///   shape), loses a signal that lands on an object it registered already while it is still
///   registering a later one, and sleeps on to its timeout although the condition it waited for
///   was met. A wait that names one object loses none. Measured on 1.4.1 (driverVersion 10401): a
///   completion waiter waiting for any of the device's queue timelines to reach its batch's value
///   saw the batch only when its 100 ms slice ran out, where a wait naming that one timeline
///   returns at the signal (the Vulkan consumer's guest presented at about a quarter of Metal's
///   rate meanwhile). Known broken through 1.4.1; 1.4.2 is unprobed and takes it. The scheduler's
///   wait on the GPU is a wait of that shape and is not measured apart. Nothing is refused: no
///   host wait of the runtime names several objects there. Each `VkQueue`'s completion waiter
///   waits for its queue's first pending batch, for that batch's fence or else for that one
///   timeline value (`command_buffer/completion.rs`), and the scheduler's thread sleeps on its
///   condition variable, woken by the completion turns, in place of waiting on the GPU for the
///   signals its heads wait for to land (`scheduler::waits_on_gpu`). Every other driver's thread
///   waits on the GPU, one `vkWaitSemaphores` for any of those event values or the nudge's next,
///   so a signal wakes it from the driver with no completion turn and no waker between.
///
/// `ImplicitPreprocessAfterExecutionSet`:
/// - Mesa's RADV (`MESA_RADV`): `vkCmdExecuteGeneratedCommandsEXT` of a layout not made for
///   explicit preprocessing dispatches the layout's preprocessing pipeline itself
///   (`radv_prepare_dgc`), and `radv_emit_compute_pipeline` skips a pipeline equal to
///   `emitted_compute_pipeline`. The execution's own commands then bind each sequence's pipeline
///   of the execution set on the GPU, which that field does not see, so the next execution's
///   preprocessing in the command buffer, with nothing dispatched between, runs the shader the
///   execution set bound last with the preprocessing's arguments. A range of two commands, an
///   execution each, faulted (`[gfxhub] page fault`, client `TCP`, permission faults) and timed out
///   the ring (`runtime_lane::generated_executions`, both of its tests that execute, on every run),
///   where explicit preprocessing runs them right. Read in 25.0.0's,
///   26.0.8's and 26.1.5's source (the field is tracked alike); 26.2.0 tracks a dirty bit instead,
///   which the preprocessing's own bind sets. Measured on 26.0.8 (driverVersion 0x6800008) on the
///   Radeon 780M (`deviceID` 0x15bf) and the RX 7900 XTX (0x744c): known broken through 26.0.8.
///   Measured clean on 26.2.4 (driverVersion 0x6802004) on both. Nothing is refused: the runtime
///   preprocesses every execution of the range apart, before the first runs
///   (`generated_execute.rs`). Every RADV GPU and release but the measured identity found clean,
///   the 780M at 26.2.4 (Phoenix's 0x15bf, and Hawk Point's 0x1900, the same GPU), takes it; the
///   RX 7900 XTX is no measured identity, so it takes it at every release.
///
/// `ExplicitPlaneRowsPastHeight`:
/// - Mesa's lavapipe (`MESA_LLVMPIPE`): a `DRM_FORMAT_MOD_LINEAR` image made for host allocations
///   with an explicit plane at a texture's packed rows, reported at `rows × rowPitch` bytes in its
///   subresource and within its memory requirements, is rendered in blocks of 4 rows: a pass that
///   loads, blends and stores a 658x126 `RGBA16Float` tile (rows of 5264 bytes) rewrote rows 126
///   and 127 past it (signaling NaNs quieted), and one over a 1308x510 `BGRA8Unorm` tile (rows of
///   5232 bytes) wrote rows 510 and 511 past the no-copy buffer's mapping, and the process took a
///   `SIGSEGV` (`runtime_lane::pitched_surfaces`, packed odd-height layouts). Its own linear images
///   report the rounded rows in their subresource. Measured on 26.2.4 (llvmpipe, LLVM 23.1.1, 256
///   bits; gem10): known broken through 26.2.4; 26.0.8 unprobed. Nothing is refused: no such
///   image of a height that is no multiple of 4 is made there (`dma_buf.rs`'s
///   `explicit_plane_rows`), and the texture takes its shadow.
///
/// `OcclusionCountPerColorOutput`:
/// - Mesa's ANV (`INTEL_OPEN_SOURCE_MESA`): a precise occlusion query counts each passing sample
///   once per color output the fragment shader writes, where Vulkan's Sample Counting counts it
///   once; write masks of 0, the query's command buffer and linked pipelines do not change it, and
///   an imprecise query reads nonzero. Measured on 26.2.4 (driverVersion 0x6802004) on the Iris
///   Xe (`deviceID` 0xa7a0; `runtime_lane::color_groups`): known broken through 26.2.4. Nothing
///   is refused, since a refusal would withhold the Mac2 and Metal3 families: the draw runs,
///   announced (`Degradation::VisibilityCountPerColor`).
///
/// `FreedAddressMemoryFailsPendingWork`:
/// - MoltenVK (`MOLTENVK`): every encoder whose pipeline reads device addresses makes every buffer
///   of `SHADER_DEVICE_ADDRESS` usage resident (`MVKDevice::encodeGPUAddressableBuffers`), in
///   Metal command buffers that retain none of it (`retainedReferences` off where the device keeps
///   its residency set), and `vkFreeMemory` lets go of the memory's Metal objects at once: a
///   Metal command buffer encoded before and not yet run fails `Invalid Resource` (Metal's code
///   9), its device then lost. Destroying the buffer alone, its memory kept, fails nothing.
///   Read in 1.4.1's source, and measured on 1.4.1 (driverVersion 10401) on an M3 Max with macOS
///   27.0.1: a bare-Vulkan program freeing an unused buffer's memory while a dispatch through a
///   device address waits behind fills fails at once, and `runtime_lane::freed_memory`: known
///   broken through 1.4.1. Nothing is refused: the runtime frees such memory once the batches
///   submitted before it completed (`freed_memory.rs`), the program's own submissions through
///   `CommandQueue::with_vk_queue` among them (a mark after each use signals a timeline of its
///   own).
///
/// `DestroyedAddressBufferFailsSubmission`:
/// - MoltenVK (`MOLTENVK`): on an Apple GPU, device memory of private or shared storage is a Metal
///   placement heap (`MVKDeviceMemory::ensureMTLHeap`), so each `VkBuffer` bound to it is a Metal
///   buffer of its own (`MVKBuffer::getMTLBuffer`), which `vkDestroyBuffer` releases at once
///   (`MVKBuffer::detachMemory`). A `vkQueueSubmit` encodes its Metal command buffer on the calling
///   thread, and an encoder whose pipeline reads device addresses lists every buffer of
///   `SHADER_DEVICE_ADDRESS` usage (`MVKDevice::encodeGPUAddressableBuffers`): a buffer destroyed
///   on another thread after that listing and before the commit fails the submission, its device
///   then lost, though no command of it refers to the buffer, which Vulkan allows
///   (VUID-vkDestroyBuffer-buffer-00922 asks only that the commands that refer to it completed).
///   Read in 1.4.1's source, and measured on 1.4.1 (driverVersion 10401) on an M3 Max with macOS
///   27.0.1: a bare-Vulkan program destroying such a buffer on another thread while a dispatch
///   through a device address is inside its `vkQueueSubmit` behind fills, 2000 other such buffers
///   alive, lost the device in the second round of 200, where the same destroy after the
///   `vkQueueSubmit` returned failed none of 10 rounds: known broken through 1.4.1. Nothing is
///   refused: the runtime destroys a block's buffer of addresses under the write lock of the gate
///   its batches' and its own submissions' `vkQueueSubmit` hold for reading (`freed_memory.rs`),
///   so none is inside one. The program's own submissions through `CommandQueue::with_vk_queue`
///   are not waited for.
///
/// `PushedBufferSizesUnset`:
/// - MoltenVK (`MOLTENVK`): a shader's runtime array length of a buffer (`OpArrayLength`) reads
///   the buffer's size from the stage's implicit buffer of sizes, which
///   `MVKCommandEncoderState::bindDescriptorSets` fills for a set of discrete descriptors (a push
///   descriptor set layout is one: `pickArgumentBufferMode` takes no argument buffer for it) and
///   `MVKCommandEncoderState::pushDescriptorSet` never fills: it writes the pushed descriptors
///   alone. A pushed buffer's length then reads 0, or what an earlier bind left there, and the
///   accesses bounded by it go nowhere: on an M3 Max with macOS 27.0.1, kernels that push set 0
///   read the null entry for every resource id (`runtime_lane::resource_ids`), stored no
///   normalized storage write (`runtime_lane::norm_storage_writes`), and wrote nothing through a
///   device without robustness's in-shader bounds (`runtime_lane::executions`), where binding set
///   0 passes each. Read in 1.4.1's source and measured on 1.4.1 (driverVersion 10401): known
///   broken through 1.4.1. Nothing is refused: the device takes no push command there
///   (`device/mod.rs`), and every draw and dispatch binds set 0 from its command buffer's pools,
///   as on a device without the command.
///
/// `PushedDescriptorsStaleAfterPipelineBinds`:
/// - Mesa's ANV (`INTEL_OPEN_SOURCE_MESA`): draws of one layout that push set 0 alike, a
///   pipeline reading a uniform buffer's constants at fixed offsets (its range starting at the
///   bind offset, `SpecKey::RangeAtBindOffset`) and then one reading it at the pushed residual
///   plus the offset through the same descriptor, which the second draw's push leaves out as
///   unchanged, so that no push follows the bind: the second pipeline reads zeros. Pushing set 0
///   whole after the bind reads the bytes, and so does pushing one write that did not change
///   (set 0's first). Measured on 26.2.4 (driverVersion 0x6802004) on the Iris Xe (`deviceID`
///   0xa7a0; `runtime_lane::unorm_depth_bias`, 3 of 3 runs wrong without the whole push, 3 of 3
///   right with it) and on the UHD 770 (`deviceID` 0xa780; the same lane, 3 of 3 runs wrong with
///   no push after the bind, 3 of 3 right with the one write): known broken through 26.2.4.
///   Nothing is refused: the first draw after each pipeline bind pushes at least one write, those
///   that changed or else set 0's first (`render/bound.rs`).
///
/// `ImmutableSamplersLostAfterCopies`:
/// - Mesa's lavapipe (`MESA_LLVMPIPE`): a compute dispatch recorded after a copy in the same
///   `VkCommandBuffer` samples a combined image sampler binding whose pipeline layout bakes an
///   immutable sampler as Repeat. A kernel's variant for a 4:2:0 texture, whose layout bakes a
///   clamp-to-edge sampler (`runtime_lane::ycbcr_textures`'s
///   `a_constexpr_sampler_takes_a_ycbcr_map_through_one_variant`), read the texels clamped at its
///   first dispatch, recorded in a `VkCommandBuffer` of its own while its pipeline was being made,
///   and Repeat-addressed at every later one, recorded after the texture's shadow copy: the same
///   pipeline, layout and pushed descriptors (with the immutable sampler pushed or not), and the
///   validation layer reported nothing. Measured on 26.0.8 (driverVersion 0x6800008, Ubuntu's
///   26.0.8-1ubuntu0.3 on LLVM 21.1.8): known broken through 26.0.8. Measured clean on 26.2.4
///   (0x6802004, LLVM 23.1.1), the same calls recorded alike. Mesa 26.1.0's release notes list
///   "lavapipe: rework immutable samplers", likely the fix (unverified: its change was not read).
///   Nothing is refused: an encoder's dispatch of a variant that bakes samplers records in a
///   `VkCommandBuffer` of its own there (`compute.rs`), as one whose pipeline is still being made
///   does (`pending_dispatch.rs`); a host step's dispatch (an indirect dispatch's, an indirect
///   command buffer's execution) records in place. Graphics draws are unmeasured. Every lavapipe
///   release takes it: 26.2.4, measured clean, has no committed profile to be a measured identity.
///
/// `SetLayoutSupportUnderstated`:
/// - MoltenVK (`MOLTENVK`): `MVKDevice::getDescriptorSetLayoutSupport` sums the bindings'
///   `descriptorCount` and answers `sum < maxPerSetDescriptors` (1212 on Apple's GPUs: four times
///   a stage's Metal argument-table slots), so a layout of exactly `maxPerSetDescriptors` is
///   refused, where Vulkan guarantees `VK_TRUE`, and so is every layout past it, though the driver
///   encodes an update-after-bind set in a Metal argument buffer and reports
///   `maxDescriptorSetUpdateAfterBind*` limits of 500000 and 1000000. The driver makes such a
///   layout, and a set of it is written and read as Metal reads the argument buffer: a bare
///   Vulkan probe (one binding of 65536 sampled images, `UPDATE_AFTER_BIND_POOL`, element 65535
///   read in a dispatch) read it right where the query answered `VK_FALSE` at 1212, 1213 and
///   65536 descriptors, and the runtime lanes that read textures through their IDs past element
///   1211 (`runtime_lane::resource_ids`) are Metal's. Read in 1.4.1's and 1.4.2's source (the
///   function is the same) and measured on 1.4.2 (driverVersion 10402) on an M3 Max: known broken
///   through 1.4.2. Vulkan-ValidationLayers 1.4.363 asks the query for every layout past
///   `maxPerSetDescriptors` and reports one it refuses
///   (`VUID-vkCreateDescriptorSetLayout-support-09582`). Nothing is refused: a set layout the
///   driver refuses is made there, announced (`set-past-layout-support`, `pipeline/objects.rs`);
///   on every other driver a set layout its driver refuses is refused.
///
/// `LazilyAllocatedSubresourceViews`:
/// - MoltenVK (`MOLTENVK`): a view of some of an image's levels or layers is a Metal texture view
///   of the image's texture (`newTextureViewWithPixelFormat:textureType:levels:slices:`, in
///   `MVKImageViewPlane::newMTLTexture` at 1.4.1 and `newMTLTextureFromBaseMTLTexture` at
///   1.4.2), and a pass attaches that view (`MVKImageView::populateMTLRenderPassAttachmentDescriptor`).
///   An image in lazily allocated memory is a texture of Metal's memoryless storage, of which
///   Metal makes no view: its API validation stops the call ("cannot create View from Memoryless
///   texture."), and without it the call answers nil. A Metal probe on an M3 Max under macOS
///   27.0.1 had nil for a 2D texture's level, a 1D array's, a 2D array's and a 2D multisample
///   array's slice, and a cube's and a cube array's face, each made of a private texture, while a
///   pass of the memoryless texture's own level or slice drew, and resolved, as of a private one.
///   So the view has no texture, and a pass attaching it renders without it and resolves nothing.
///   1.4.2 reports it as an error ("Attempting to initialize an MVKImageViewPlane with a nil
///   MTLTexture.", `MVKImageViewPlane::initializeMTLTexture`), measured on 1.4.2 (driverVersion
///   10402) on an M3 Max (`runtime_lane::memoryless`'s slice, face and level passes, and
///   `a_memoryless_multisample_array_slice_resolves`); 1.4.1 makes the same call and keeps its nil
///   silently (read in its source), so its passes of such views lost the attachment unannounced.
///   Known broken through 1.4.2. Nothing is refused: such an image of more than one level or layer
///   takes device-local memory (`ImagePlan::memory_class`); one of a single level and layer is
///   viewed as its texture itself.
///
/// `HostTransferImagesCopiedAtBind`:
/// - MoltenVK (`MOLTENVK`): an image with host-transfer usage lies in host-visible memory, which
///   the driver takes as a linear array of the image's texels: `MVKImageMemoryBinding::flushToDevice`
///   `replaceRegion:`s every subresource whose layout is `UNDEFINED`, `GENERAL` or `PREINITIALIZED`
///   out of that memory, through `MVKImagePlane::getSubresourceLayout`'s own `rowPitch`, and
///   `MVKImageMemoryBinding::bindDeviceMemory` ends in that call. So every such image is copied
///   into whole as its memory is bound, out of bytes nothing has written. For an image whose
///   texel block holds more than one texel Metal's own check of that copy fails:
///   `agxaAssertBufferIsValid`, reached from
///   `AGX::Texture<(AGXTextureMemoryLayout)3, AGX::G15::Encoders, AGX::G15::Classes>::writeRegionCommon`
///   through `-[AGXG15XFamilyTexture replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:]`,
///   prints `AGX: Texture read/write assertion failed: bytes_per_row >= used_bytes_per_row` once
///   for each such image bound. Measured on 1.4.2 (driverVersion 10402) on an M3 Max under macOS
///   27.0.1, by a backtrace off that `fprintf` through `vkBindImageMemory` into
///   `Device::host_placed_image`: `metal2vulkan-conformance`'s `bc1-rgba` reported it 4 times, once
///   a texture, while a record of the same case — plain Metal making the same descriptor and the
///   same `replaceRegion:` the case asks for — reported it **0** times, so the copy is the
///   driver's own and no call of the program's provokes it. Known broken through 1.4.2. Nothing is
///   refused: such an image takes no host-transfer usage there (`Device::host_placed_image`), so it
///   stays in memory of its own class and its CPU copies go through a submission; on every other
///   driver a block-compressed texture's CPU copies stay on the host.
///
/// `TwoPlaneChromaMisplaced`:
/// - MoltenVK (`MOLTENVK`): a sampler Y′CbCr conversion of `G8_B8R8_2PLANE_420_UNORM`,
///   `G8_B8R8_2PLANE_422_UNORM` or their 10-bit formats, whose optimal tiling reports midpoint
///   and cosited chroma, a linear chroma filter, separate reconstruction and forced explicit
///   reconstruction, reconstructs the chroma off where Vulkan places it (20.2.8, "Chroma
///   Reconstruction"). A bare Vulkan probe on an M3 Max under macOS 27.0.1 sampled 40 x 24
///   images at every luma texel's centre and at 2048 random points, the planes random and, for
///   the weights, one chroma pair alone set: under a nearest sampler filter, a linear chroma
///   filter and midpoint chroma, the weights of a chroma sample have the right shape (1/4, 3/4,
///   3/4, 1/4 across and down) but centred one luma texel up and to the left of the midpoint's
///   (chroma sample `k` at luma `2k` where Vulkan's lies at `2k + 1`); under a linear sampler
///   filter the chroma is filtered twice (3/16, 3/16, 10/16, 10/16, 3/16, 3/16 across) and
///   shifted alike; forced explicit reconstruction changes nothing, and cosited chroma reads
///   weights of no interpolation. A texel's channels lay up to 1000/2048 from Vulkan's
///   conversion (4:2:0, the identity model) and 366/2048 (4:2:2), the luma exact. Its 4:2:2
///   single-plane formats (`G8B8G8R8_422_UNORM`, `B8G8R8G8_422_UNORM`) reconstruct as Vulkan
///   states. Measured on 1.4.2 (driverVersion 10402): known broken through 1.4.2. Nothing is
///   refused: a two-plane 4:2:0 or 4:2:2 texture converts through Metal's arithmetic there
///   (`ycbcr_device.rs`), as on a device without the conversion.
///
/// `TenBitCodesReadAsWords`:
/// - Mesa's lavapipe (`MESA_LLVMPIPE`): a sampler Y′CbCr conversion of
///   `G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16` and its 4:4:4 format reads each code's 16-bit
///   word as a 16-bit unsigned normalized value, `word / 65535`, its six unused low bits included,
///   where Vulkan converts the 10-bit code as `c / (2^10 - 1)` ("Fixed-Point Data Conversions";
///   the format "has a 10-bit G component in the top 10 bits of each 16-bit word ... with the
///   bottom 6 bits of each word unused"). A bare Vulkan probe read luma code 1023 (low bits 0) as
///   0.999039 through the identity model, and the runtime's conversion of random words through it
///   (`runtime_lane::ycbcr_textures::retyped_views_read_their_luma_codes`) read codes one low
///   wherever the word's low bits put it below the code's value. Measured on 26.0.8 (LLVM
///   21.1.8): known broken through 26.0.8. Nothing is refused: a 10-bit two-plane texture converts
///   through Metal's arithmetic there (`ycbcr_device.rs`).
/// - Mesa's RADV (`MESA_RADV`): the same read. A bare Vulkan probe of a linear
///   `G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16` image on a Radeon 780M read luma code 1023 (low
///   bits 0) as 0.999039 through the identity model. Measured on 26.2.4 (linear tiling; the
///   optimal tiling's conversion of the format was not measured): known broken through 26.2.4.
///   Nothing is refused: a 10-bit two-plane texture converts through Metal's arithmetic there, and
///   its planes read in place through the texel buffer (`ycbcr_device.rs`,
///   `texture/ycbcr_planes.rs`).
/// - NVIDIA's (`NVIDIA_PROPRIETARY`): on the RTX 5080 Laptop GPU (`deviceID` 0x2c19) under Linux,
///   a sampler Y′CbCr conversion of `G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16` and its 4:2:2 and
///   4:4:4 formats (optimal tiling, the image's own format, the surface's matrix and range) reads
///   each code's six unused low bits too, where Vulkan has them "unused by format conversion" in
///   a view of the image's own format ("If the VkImage format and VkImageView format are the same,
///   these bits will be unused by format conversion and this will have no effect"): measured
///   through the runtime's conversion of random planes (`tests::ycbcr_device`), every texel lay
///   within 0.0001/2048 of Vulkan's conversion of the codes where the words' low bits were 0, and
///   up to 6.43/2048 from it where they were random, within 0.034/2048 there of the word read as
///   the code with six bits of fraction (`word / 64`) under a video range; its 8-bit formats lay
///   within 0.034/2048 of Vulkan's. Measured on 615.71.09: known broken through 615.71.09. Every
///   NVIDIA GPU takes it until a probe finds one clean. Nothing is refused: a 10-bit two-plane
///   texture converts through Metal's arithmetic there (`ycbcr_device.rs`).
///
/// `UnpinnablePagesBreakHostMemory`:
/// - NVIDIA's (`NVIDIA_PROPRIETARY`): on the RTX 5080 Laptop GPU (`deviceID` 0x2c19) under Linux,
///   `vkAllocateMemory` importing the pointer of a mapping of a udmabuf of a `memfd`'s pages
///   (`VM_PFNMAP`, which the kernel's `get_user_pages` refuses) answers
///   `VK_ERROR_OUT_OF_DEVICE_MEMORY` on both memory types `vkGetMemoryHostPointerPropertiesEXT`
///   allows (2 and 3), where Vulkan requires `VK_ERROR_INVALID_EXTERNAL_HANDLE` of a pointer
///   "platform-specific restrictions" keep from being imported ("Host External Memory"), and
///   later allocations of the process fail as out of device memory too, where Vulkan's
///   out-of-memory errors "do not modify any currently existing Vulkan objects". After such an
///   import tried as type 2, then 3: every host-pointer import measured (a `memfd`'s shared pages,
///   8 KiB to 8 MiB at any offset; 8 MiB of private anonymous pages; 8 MiB 512 MiB away from the
///   mapping; 40 tries over 3 s; on a second device made after it) and a plain 64 KiB
///   allocation of host-visible type 2, while a device-local one allocates. After the runtime's
///   own import of such pages (type 3, then 2): an 8 MiB import 8 times in a row, and a guest
///   RAM's chunk at each of 7 objects over it, while each object's own one-page import succeeded.
///   Neither the pointer's properties query alone, a failed import of the udmabuf itself, nor a
///   mapping of one never imported broke anything. Measured on 615.71.09 with the runtime's
///   device tests: known broken through 615.71.09; every other NVIDIA GPU, unprobed, takes the
///   same path. Nothing is refused that the driver would import: on Linux the runtime asks the
///   kernel whether it pins each mapping over the pages before it hands the driver their pointer
///   (`host_memory.rs`'s `refuse_unpinnable`), and never hands it pages it cannot pin: a no-copy
///   buffer, heap or rebind over them takes the caller's dma-buf source, and else, or where its
///   import fails, a copy of them, as where a pointer import fails (`copied.rs`,
///   `copied-host-pages`); a RAM's chunk over them is not imported, and its objects take their
///   own paths to the same end.
pub(super) const AFFECTED: &[Affected] = &[
    Affected {
        defect: RuntimeDriverDefect::HostTransferImagesCopiedAtBind,
        driver: DriverId::MOLTENVK,
        clean: &[],
    },
    Affected {
        defect: RuntimeDriverDefect::MultisampledDepthStencilColourCopies,
        driver: DriverId::MESA_LLVMPIPE,
        clean: &[],
    },
    Affected {
        defect: RuntimeDriverDefect::MultisampledDepthStencilColourCopies,
        driver: DriverId::MOLTENVK,
        clean: &[],
    },
    Affected {
        defect: RuntimeDriverDefect::MultisampledStencilColourCopies,
        driver: DriverId::NVIDIA_PROPRIETARY,
        // The RTX 5080 Laptop GPU, measured clean.
        clean: &[0x2c19],
    },
    Affected {
        defect: RuntimeDriverDefect::NormalizedDepthPastUnitRange,
        driver: DriverId::MOLTENVK,
        clean: &[],
    },
    Affected {
        defect: RuntimeDriverDefect::ImageSizeShortOfCompressionData,
        driver: DriverId::INTEL_OPEN_SOURCE_MESA,
        clean: &[],
    },
    Affected {
        defect: RuntimeDriverDefect::DepthStencilResolveFirstSubresource,
        driver: DriverId::MESA_LLVMPIPE,
        clean: &[],
    },
    Affected {
        defect: RuntimeDriverDefect::LazilyAllocatedVolumePlaneViews,
        driver: DriverId::MOLTENVK,
        clean: &[],
    },
    Affected {
        defect: RuntimeDriverDefect::PipelineCacheDataCompiledAtLoad,
        driver: DriverId::MOLTENVK,
        clean: &[],
    },
    Affected {
        defect: RuntimeDriverDefect::CompressedImageCopiesRewriteTexels,
        driver: DriverId::MOLTENVK,
        clean: &[],
    },
    Affected {
        defect: RuntimeDriverDefect::MultisampledCopiesConvertFloatNans,
        driver: DriverId::MESA_RADV,
        clean: &[],
    },
    Affected {
        defect: RuntimeDriverDefect::ReadsCommandBuffersAfterTheirSignal,
        driver: DriverId::MOLTENVK,
        clean: &[],
    },
    Affected {
        defect: RuntimeDriverDefect::WaitForAnyObjectLosesASignal,
        driver: DriverId::MOLTENVK,
        clean: &[],
    },
    Affected {
        defect: RuntimeDriverDefect::ExplicitPlaneRowsPastHeight,
        driver: DriverId::MESA_LLVMPIPE,
        clean: &[],
    },
    Affected {
        defect: RuntimeDriverDefect::OcclusionCountPerColorOutput,
        driver: DriverId::INTEL_OPEN_SOURCE_MESA,
        clean: &[],
    },
    Affected {
        defect: RuntimeDriverDefect::ImplicitPreprocessAfterExecutionSet,
        driver: DriverId::MESA_RADV,
        // The Radeon 780M, measured clean.
        clean: &[0x15bf],
    },
    Affected {
        defect: RuntimeDriverDefect::FreedAddressMemoryFailsPendingWork,
        driver: DriverId::MOLTENVK,
        clean: &[],
    },
    Affected {
        defect: RuntimeDriverDefect::DestroyedAddressBufferFailsSubmission,
        driver: DriverId::MOLTENVK,
        clean: &[],
    },
    Affected {
        defect: RuntimeDriverDefect::PushedBufferSizesUnset,
        driver: DriverId::MOLTENVK,
        clean: &[],
    },
    Affected {
        defect: RuntimeDriverDefect::PushedDescriptorsStaleAfterPipelineBinds,
        driver: DriverId::INTEL_OPEN_SOURCE_MESA,
        clean: &[],
    },
    Affected {
        defect: RuntimeDriverDefect::ImmutableSamplersLostAfterCopies,
        driver: DriverId::MESA_LLVMPIPE,
        clean: &[],
    },
    Affected {
        defect: RuntimeDriverDefect::SetLayoutSupportUnderstated,
        driver: DriverId::MOLTENVK,
        clean: &[],
    },
    Affected {
        defect: RuntimeDriverDefect::LazilyAllocatedSubresourceViews,
        driver: DriverId::MOLTENVK,
        clean: &[],
    },
    Affected {
        defect: RuntimeDriverDefect::TwoPlaneChromaMisplaced,
        driver: DriverId::MOLTENVK,
        clean: &[],
    },
    Affected {
        defect: RuntimeDriverDefect::TenBitCodesReadAsWords,
        driver: DriverId::MESA_LLVMPIPE,
        clean: &[],
    },
    Affected {
        defect: RuntimeDriverDefect::TenBitCodesReadAsWords,
        driver: DriverId::MESA_RADV,
        clean: &[],
    },
    Affected {
        defect: RuntimeDriverDefect::TenBitCodesReadAsWords,
        driver: DriverId::NVIDIA_PROPRIETARY,
        clean: &[],
    },
    Affected {
        defect: RuntimeDriverDefect::UnpinnablePagesBreakHostMemory,
        driver: DriverId::NVIDIA_PROPRIETARY,
        clean: &[],
    },
];

/// The defects of the driver and device `description` names, its GPU read by the `deviceID` it
/// was measured under (`DeviceDescription::canonical_device_id`).
pub(super) fn of(description: &DeviceDescription) -> BTreeSet<RuntimeDriverDefect> {
    among(
        AFFECTED,
        description.driver_id,
        description.canonical_device_id(),
    )
}

/// The defects of `table` that the driver `driver` has on the GPU `device`: each of the driver's
/// but those a probe found clean on that GPU.
pub(super) fn among(
    table: &[Affected],
    driver: Option<DriverId>,
    device: u32,
) -> BTreeSet<RuntimeDriverDefect> {
    table
        .iter()
        .filter(|affected| Some(affected.driver) == driver)
        .filter(|affected| !affected.clean.contains(&device))
        .map(|affected| affected.defect)
        .collect()
}
