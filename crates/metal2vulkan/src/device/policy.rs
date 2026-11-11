//! `DevicePolicy`: the consumer's product choices that, together with a `DeviceDescription`,
//! decide a `DeviceProfile`.

use crate::device::profile::{GuestIsolation, HandleDomain, PointerDomain};
use std::time::Duration;

/// Policy inputs of `Device::new`. Every field but `heap_compression`, `completion`,
/// `batch_order`, `render_lowering`, `host_page_textures`, `pipeline_waits`,
/// `pipeline_creation_wait`, `pipeline_commit_wait`, `wide_lines`, `extended_range_blits`,
/// `queue_sharing`, `dont_care_colors` and `unstored_dont_care_loads` can change emitted code, so
/// the profile carries it (containment as the encoding it selects) and its fingerprint covers it;
/// those thirteen are the runtime's alone (a heap's compressed texture's image, when the host sees a command buffer
/// complete, how a batch begins after the earlier ones of its queue, where a render encoder's calls
/// are lowered, what holds a sampled texture over host pages, whether and how long a pipeline the
/// driver is still making is waited for, which lines draw a wide line, what a blit through a
/// compressed extended-range texture writes, which `VkQueue` a command queue submits to, what
/// a colour attachment stored `DontCare` keeps, and what a pass reads of an attachment it loads
/// `DontCare` and does not store), and no profile carries them.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DevicePolicy {
    /// How pointer values resolve to resources (v1 ships `ResidentTable` only).
    pub pointer_domain: PointerDomain,
    /// How resource handles resolve (v1 ships `Table` only).
    pub handle_domain: HandleDomain,
    /// Whether accesses are contained to the resources the binder supplied.
    pub containment: ContainmentPolicy,
    /// Whether guest shaders are isolated from each other's resources.
    pub isolation: GuestIsolation,

    /// Whether a heap's colour texture that Metal keeps losslessly compressed, on a driver whose
    /// compressed images hide writes through an alias, is the driver's compressed image (the
    /// default) or one the driver keeps uncompressed in the heap's bytes. The runtime's choice
    /// alone: no emitted code changes.
    pub heap_compression: HeapCompressionPolicy,
    /// Whether command buffers committed on one `VkQueue` that the runtime's scheduler finds ready
    /// together go to the driver in one kernel submission, each seen complete once the last of
    /// them completed (the default, on a driver that merges such submissions), or each in a
    /// submission of its own, seen complete at its own end. The runtime's choice alone: no
    /// emitted code changes.
    pub completion: CompletionPolicy,
    /// Whether a batch of command buffers on one `VkQueue` begins after the batches before it
    /// there by their ends (the default, on a driver measured to run no two kernel submissions of
    /// one queue side by side), or by a barrier for each access that depends on theirs. The
    /// runtime's choice alone: no emitted code changes.
    pub batch_order: BatchOrderPolicy,
    /// Whether a render encoder's call is lowered inside the call, which returns every refusal
    /// (the default), or after it returned, a refusal only the lowering finds failing its command
    /// buffer. The runtime's choice alone: no emitted code changes.
    pub render_lowering: RenderLoweringPolicy,
    /// Whether a texture that only samples or reads, over a no-copy buffer's host pages or over a
    /// buffer the runtime placed in fresh host pages, is an image aliasing those pages in place
    /// (the default, where the device's own check shows its first move keeps every byte), or a
    /// shadow the encoders copy the texels into and back. The runtime's choice alone: no emitted
    /// code changes.
    pub host_page_textures: HostPageTexturePolicy,
    /// Whether a chain of fast-math `half` additions, subtractions and products, on a device
    /// without `shaderFloat16`, rounds where a value leaves it (the default) or at each operation.
    /// Changes emitted code: the profile carries it.
    pub half_chains: HalfChainPolicy,
    /// Whether a function that reads a two-plane YCbCr texture's planes in place (a uniform
    /// texel buffer over their codes, on a device whose memory is the host's) samples them by
    /// Vulkan's Y′CbCr model in the device's arithmetic, as the device's own conversion would
    /// (the default), or converts and filters every tap as Metal does, exactly. Changes emitted
    /// code: the profile carries it.
    pub ycbcr_in_place_reads: YCbCrInPlacePolicy,
    /// Whether a render or compute state's creation, and a committed command buffer, wait a
    /// bound for a pipeline the driver is still making, the command buffer then dropping the
    /// draws and dispatches whose pipeline is not made (the default), or until it is made. The
    /// bounds are the two fields below. The runtime's choice alone: no emitted code changes.
    pub pipeline_waits: PipelineWaitPolicy,
    /// Under [`PipelineWaitPolicy::Bounded`], the longest a render or compute state's creation
    /// waits for the pipelines it hands to the runtime's pool, one deadline for all of them from
    /// the wait's start: then it returns, the makings going on. 4.5 s by default, so a host call that
    /// both creates a state and commits a command buffer (a guest's FIFO step may) waits 9 s at
    /// most for the driver's makings, under a guest's 10 s check of its FIFO. Unread under
    /// `Exact`. The runtime's alone: no emitted code changes.
    pub pipeline_creation_wait: Duration,
    /// Under [`PipelineWaitPolicy::Bounded`], how long after its commit a command buffer waits
    /// for the pipelines of its commands the driver is still making: one deadline, its commit plus
    /// this, which its draws, compute dispatches, tile dispatches and execution replays share,
    /// wherever they record (in its `commit`, or on the runtime's scheduler thread), past which
    /// each such command records as nothing. 4.5 s by default (as the creation wait, above). Unread
    /// under `Exact`. The runtime's alone: no emitted code changes.
    pub pipeline_commit_wait: Duration,
    /// Whether a line Apple's `setLineWidth:` widens, at a width the device's own lines draw (which
    /// no device's are taken to draw as Metal's: strict lines, lines that are not parallelograms, a
    /// width off the device's granularity or below 1, a driver's parallelograms, whose ends the
    /// driver places), is the device's own line at that width (the default) or the pipeline's band
    /// variant, two triangles that cover Metal's parallelogram. The runtime's choice alone: no
    /// emitted code of a pipeline changes, the band being a variant of its own.
    pub wide_lines: WideLinePolicy,
    /// Whether a blit or fill through a `BGRA10_XR` or `BGRA10_XR_sRGB` texture Metal keeps
    /// losslessly compressed keeps the texels' bytes (the default) or writes Metal's canonical
    /// encoding of them. The runtime's choice alone: no emitted code changes.
    pub extended_range_blits: ExtendedRangeBlitPolicy,
    /// Whether command queues share `VkQueue`s, each moving to a busy one of its family as it
    /// makes a command buffer while it holds no other uncompleted one (the default, on a device
    /// whose own check finds its GPU idle when consecutive work comes from another `VkQueue`), or
    /// each keeps the `VkQueue` it took at its creation. The runtime's choice alone: no emitted
    /// code changes.
    pub queue_sharing: QueueSharingPolicy,
    /// Whether a colour attachment of a texture's own image that a pass stores `DontCare` is
    /// stored as the driver stores it (the default) or renders through a backing so that the
    /// texture keeps the texels it held before the pass, as Apple's GPU leaves them. The runtime's
    /// choice alone: no emitted code changes.
    pub dont_care_colors: DontCareColorPolicy,
    /// Whether an attachment a pass loads `DontCare` and does not store, in a pass whose draws
    /// reach a tile, is cleared over the render area on a driver measured to run that clear no
    /// slower than its `DONT_CARE` load and left `DONT_CARE` on every other (the default), or
    /// cleared on every driver, so the pass's draws read Metal's answer wherever they read it.
    /// The runtime's choice alone: no emitted code changes.
    pub unstored_dont_care_loads: UnstoredDontCarePolicy,
}

impl DevicePolicy {
    /// The untrusted-guest production policy: containment and isolation on, the driver's
    /// compressed image of a heap's texture Metal keeps compressed, command buffers ready together
    /// in one kernel submission, batches of one queue ordered by their ends where the driver runs
    /// them one after another, and a sampled
    /// texture over host pages in place where the device's check passed, a pipeline still
    /// being made waited for a bound (4.5 s at a state's creation, and 4.5 s from a commit), a
    /// wide line the device's own wherever it draws the width, command queues sharing
    /// `VkQueue`s where the device's GPU idles between work of different ones, a colour stored
    /// `DontCare` stored as the driver stores it, and an attachment loaded `DontCare` and not
    /// stored cleared where its driver clears it no slower.
    pub const UNTRUSTED: DevicePolicy = DevicePolicy {
        pointer_domain: PointerDomain::ResidentTable,
        handle_domain: HandleDomain::Table,
        containment: ContainmentPolicy::Contained,
        isolation: GuestIsolation::On,
        heap_compression: HeapCompressionPolicy::Compressed,
        completion: CompletionPolicy::Grouped,
        batch_order: BatchOrderPolicy::Ends,
        render_lowering: RenderLoweringPolicy::AtCall,
        host_page_textures: HostPageTexturePolicy::InPlace,
        half_chains: HalfChainPolicy::AtExits,
        ycbcr_in_place_reads: YCbCrInPlacePolicy::Model,
        pipeline_waits: PipelineWaitPolicy::Bounded,
        pipeline_creation_wait: Duration::from_millis(4500),
        pipeline_commit_wait: Duration::from_millis(4500),
        wide_lines: WideLinePolicy::Device,
        extended_range_blits: ExtendedRangeBlitPolicy::Kept,
        queue_sharing: QueueSharingPolicy::Adaptive,
        dont_care_colors: DontCareColorPolicy::Driver,
        unstored_dont_care_loads: UnstoredDontCarePolicy::Driver,
    };

    /// This policy with Metal's answer wherever the choices offer it: a heap's texture whose bytes
    /// alias the heap's
    /// as written ([`HeapCompressionPolicy::Uncompressed`]), each command buffer seen complete
    /// at its own end ([`CompletionPolicy::Exact`]), each batch ordered after the earlier ones of
    /// its queue by
    /// the accesses it depends on alone ([`BatchOrderPolicy::Exact`]), each render encoder
    /// call lowered inside the call, which returns every refusal
    /// ([`RenderLoweringPolicy::AtCall`]), every texture over host pages a shadow, whose
    /// copies touch its texels alone ([`HostPageTexturePolicy::Shadow`]), each `half`
    /// operation rounded on a device without `shaderFloat16` ([`HalfChainPolicy::EachOperation`]),
    /// every two-plane YCbCr sample of planes read in place Metal's conversion and filter of its
    /// taps ([`YCbCrInPlacePolicy::Exact`]),
    /// a render state's predicted pipeline made at its creation, each draw's pipeline waited
    /// for until it is made ([`PipelineWaitPolicy::Exact`]), every wide line Metal's
    /// parallelogram where the device's own line differs ([`WideLinePolicy::Exact`]), and every
    /// blit and fill through a compressed `BGRA10_XR` texture Metal's canonical encoding of its
    /// texels ([`ExtendedRangeBlitPolicy::Canonical`]), every command queue on the `VkQueue`
    /// it took at its creation ([`QueueSharingPolicy::PerQueue`]), and every colour attachment of a
    /// texture's own image stored `DontCare` keeping its earlier texels
    /// ([`DontCareColorPolicy::Kept`]), and every attachment loaded `DontCare` and not stored
    /// cleared in a pass whose draws reach a tile ([`UnstoredDontCarePolicy::Cleared`]). Every
    /// other field is kept.
    pub const fn exact_answers(self) -> DevicePolicy {
        DevicePolicy {
            heap_compression: HeapCompressionPolicy::Uncompressed,
            completion: CompletionPolicy::Exact,
            batch_order: BatchOrderPolicy::Exact,
            render_lowering: RenderLoweringPolicy::AtCall,
            host_page_textures: HostPageTexturePolicy::Shadow,
            half_chains: HalfChainPolicy::EachOperation,
            ycbcr_in_place_reads: YCbCrInPlacePolicy::Exact,
            pipeline_waits: PipelineWaitPolicy::Exact,
            wide_lines: WideLinePolicy::Exact,
            extended_range_blits: ExtendedRangeBlitPolicy::Canonical,
            queue_sharing: QueueSharingPolicy::PerQueue,
            dont_care_colors: DontCareColorPolicy::Kept,
            unstored_dont_care_loads: UnstoredDontCarePolicy::Cleared,
            ..self
        }
    }
}

impl Default for DevicePolicy {
    /// [`DevicePolicy::UNTRUSTED`].
    fn default() -> DevicePolicy {
        DevicePolicy::UNTRUSTED
    }
}

/// The containment choice. The encoding (robustness or in-shader checks) is selected by
/// `Device::new` from the enabled robustness features; the policy only says whether to contain.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ContainmentPolicy {
    /// Every access stays inside the resources the binder supplied (the default).
    Contained,
    /// No containment. Only valid with `GuestIsolation::Off`; recorded in the fingerprint.
    UnsafeOff,
}

/// The image of a heap's colour texture that Metal keeps losslessly compressed (with
/// `allowGPUOptimizedContents`, no `ShaderWrite`, `PixelFormatView` or `ShaderAtomic` usage, and a
/// level 0 of 16 texels or more a side), on a driver whose compressed images may keep their texels
/// apart from the memory they are bound to (`DriverFact::CompressedImagesHideAliasedWrites`: ANV
/// on Intel's GPUs). Metal's own texture shows no aliased write as written either: an M3 Max read
/// opaque magenta (`0xffff00ff`) in every texel of drawn content a buffer's fill over the
/// texture's bytes covered, and the fill's bytes where the texture held only a clear. A texture
/// Metal keeps uncompressed takes the uncompressed image under either value. The runtime's choice
/// alone: no emitted code changes.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum HeapCompressionPolicy {
    /// The driver's compressed optimal image (the default), its fast clears, compression and
    /// tiling kept: a write through a buffer or texture over its bytes may leave its texels as
    /// they were or change them otherwise than Metal's, and its writes may show in those bytes
    /// compressed. Announced `heap-texture-compressed` where such a texture is made.
    #[default]
    Compressed,
    /// An image the driver keeps uncompressed in the heap's bytes, as for a texture Metal keeps
    /// uncompressed: host-transfer usage where the device enabled `hostImageCopy`, else a linear
    /// image, else the optimal one (each announced as such), so a write through a buffer over the
    /// texture's bytes shows in its texels as written.
    Uncompressed,
}

/// When the host sees a command buffer complete, on a driver that merges a submission signalling
/// nothing into the next one in the same `vkQueueSubmit` (Mesa's common queue,
/// `vk_queue_submits_merge`: ANV, RADV, lavapipe, NVK, Turnip and the other Mesa drivers of a kernel
/// submission), where each kernel submission costs CPU time for every `VkDeviceMemory` of the
/// device (ANV on i915 lists each in every execbuffer). Elsewhere every command buffer goes in a
/// submission of its own under either value. The runtime's choice alone: no emitted code changes.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CompletionPolicy {
    /// Up to four batches of command buffers on one `VkQueue` that the scheduler finds ready
    /// together, none of which signals a binary semaphore, takes a fence, moves the caller's
    /// images or does first work, and none but the first of which waits for anything not seen
    /// complete, go in one `vkQueueSubmit`, and only the last signals, the queue's timeline and
    /// every event they signal (the default): one kernel submission for all of them. The commands,
    /// their barriers and their GPU times are unchanged; each but the last is seen complete (its
    /// status, handlers and waits) once the last completed, and an event's values they signal are
    /// reached then, late by the GPU time of the batches after it, never early and in commit
    /// order. The submission checks of up to three batches right behind a queue's next batch run
    /// with its own, in commit order, so checked command buffers are found ready together: such a
    /// check runs before the previous batch's submission rather than after it. Announced
    /// `grouped-completion` at the first such submission.
    #[default]
    Grouped,
    /// Each batch in a submission of its own that signals its own values, so each command buffer
    /// is seen complete, and its event signals land, at its own end, as Metal's are; each batch's
    /// submission checks run once the batch before it was submitted.
    Exact,
}

/// How a batch of command buffers on one `VkQueue` (the work the runtime submits for a command
/// buffer between its event waits and signals) begins after the earlier batches there. Metal
/// orders a command buffer after the earlier ones of its queue only where its work depends on
/// theirs, so an independent one may run beside the one before it; it writes its `GPUStartTime`
/// as its work begins. The runtime's choice alone: no emitted code changes.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BatchOrderPolicy {
    /// By their ends (the default) on a driver measured to run no two kernel submissions of one
    /// queue side by side, with or without a barrier between them, where it runs the same work
    /// recorded in one command buffer side by side (ANV, RADV and NVIDIA's driver): every batch
    /// ends with a barrier that orders all its commands before every command after it on its
    /// queue, so a later batch plans no barrier for the accesses of earlier ones, and its first
    /// encoder's GPU start time is written once the work before it ended, at the bottom of the
    /// pipe. Each batch then still starts once the one before it ended, as those drivers start it
    /// anyway. On any other driver, as [`Exact`](Self::Exact).
    #[default]
    Ends,
    /// A barrier at a batch's start for each access that depends on an earlier batch's, as for
    /// the accesses within a batch, and its first encoder's GPU start time written as it begins,
    /// at the top of the pipe, where the driver may run it beside the batch before it.
    Exact,
}

/// Where a render encoder's calls are lowered: where each call's commands, descriptors, pipelines
/// and resources are worked out and recorded. Metal's encoder checks a call as it is made and
/// returns; its GPU work, and any failure found there, comes after. The runtime's choice alone: no
/// emitted code changes.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RenderLoweringPolicy {
    /// After the call returned: the call makes Metal's checks of its arguments and
    /// of the encoder's state, copies what it was lent and holds what it names, and returns; the
    /// encoder lowers it later, in order, before its command buffer's submission (on the
    /// runtime's lowering thread while the program goes on, the encoder's end too, and the rest
    /// at what follows the end in the command buffer; `commit` makes its checks and returns, its
    /// own work run after the end, and later commits behind it). A refusal only
    /// the lowering finds fails the command buffer
    /// (`CommandBufferError::CallRefused`, naming the call), which then runs nothing of that
    /// encoder or of any after it.
    Worker,
    /// Inside the call, as each call is made (the default): every refusal, the lowering's too,
    /// returns from the call itself.
    #[default]
    AtCall,
}

/// What holds a texture over host pages: over a no-copy buffer's pages (imported by their pointer
/// or as dma-bufs) or over a buffer the runtime placed in fresh host pages it imported. Vulkan
/// makes an image bound to imported memory start `UNDEFINED`
/// (`VUID-VkImageCreateInfo-pNext-01443`), and lets its layout transitions, the first among them,
/// and a render pass's load and store operations read and write every byte bound to it, and a
/// linear image's storage writes may reach more than their texels, where Metal keeps every CPU
/// write to the bytes no command writes. The runtime's choice alone: no emitted code changes.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum HostPageTexturePolicy {
    /// A linear image aliasing the pages (the default), on a device whose own check, run once per
    /// driver and release, shows that the first move of such an image leaves every byte alone
    /// while the CPU writes beside it, for a texture that renders, its render passes' load and
    /// store operations and its moves into the feedback loop layout and back too, and for one a
    /// shader writes (`ShaderWrite` or `ShaderAtomic`, of texels of 1, 2, 4, 8 or 16 bytes, those
    /// of 1 or 2 checked apart), a kernel's storage writes of its texels; and where the device
    /// makes the image (its format and usage linear) and lays it out at the buffer's row pitch.
    /// Elsewhere the shadow below. A texture made before the check's verdict is known is a shadow
    /// too.
    #[default]
    InPlace,
    /// A shadow on every device: an image of the runtime's own memory the encoders copy the
    /// texels into before each use and merge back after each write, whose copies touch the
    /// texels alone.
    Shadow,
}

/// How a chain of `half` additions, subtractions and products rounds on a device without
/// `shaderFloat16`, where a module computes each in `float` on its operands held widened and
/// rounds its result by the device's `OpQuantizeToF16` (`PrecisionNote::NativeHalfQuantize`):
/// capability emulation, as the device has no 16-bit arithmetic. A device with `shaderFloat16`
/// computes `half` as it does under either value. Changes emitted code: the profile carries it.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum HalfChainPolicy {
    /// An operation whose flags hold `contract` and `reassoc`, and whose result only such
    /// operations on `half` read, is held at `float` precision unrounded (the default): the chain
    /// rounds where a value leaves it, a quantize at each of its exits instead of one per
    /// operation. A result may differ from the rounded operations' by the roundings the chain
    /// skips, and a value past `half`'s range inside the chain stays finite. Recorded
    /// `PrecisionNote::FastMathRewrites`.
    #[default]
    AtExits,
    /// Every operation's result rounded to `half`, as the AIR computes it.
    EachOperation,
}

/// How a function samples a two-plane YCbCr texture (`YCBCR8_420_2P`, `YCBCR10_420_2P`) whose
/// planes it reads in place: a uniform texel buffer over the planes' codes, on a device whose
/// memory is the host's (an integrated GPU, a CPU device), where no image of the device's samples
/// them (`bind::YCbCrInPlace`). No sampler of the device reads a texel buffer, so the function
/// filters and converts the codes itself either way. Changes emitted code: the profile carries
/// it.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum YCbCrInPlacePolicy {
    /// A linear sample by Vulkan's Y′CbCr model, as the device's own sampler Y′CbCr conversion of
    /// the planes' multi-planar format with a linear chroma filter computes it (the default): the
    /// luma codes blended bilinearly at the sample's point, the chroma pairs blended bilinearly at
    /// the same point of the chroma plane (Metal's midpoint siting), then the surface's range and
    /// matrix (Metal's coefficients) in `float`, each channel clamped into `[0, 1]`, alpha 1:
    /// eight fetches. Metal converts each of the four taps (its chroma weighted 9/3/3/1 and
    /// rounded to a code, each channel clamped) and sums their steps of `1/2048`: the answers
    /// differ by the chroma the taps reconstruct, by those roundings, and where a tap's channel
    /// leaves `[0, 1]`, recorded `PrecisionNote::HalfSampleFilter` or `FloatSampleFilter` and
    /// announced `ycbcr-filter-not-exact`. A nearest sample (one tap), a sample through a border
    /// address mode, one whose texel offset Metal places by a parent's size, and a gather are
    /// Metal's under either value.
    #[default]
    Model,
    /// Every tap Metal's texel (its luma code, its chroma pairs weighted 9/3/3/1 and rounded to a
    /// code, the range's and the matrix's fixed-point conversion), and a linear sample Metal's sum
    /// of the four taps' steps on the grid `2048`, exactly: twenty fetches and four conversions a
    /// linear sample.
    Exact,
}

/// How long the runtime waits for a draw's `VkPipeline` the driver is still making. Metal makes a
/// render state's pipeline inside `newRenderPipelineState`, so no command buffer of Metal's waits
/// for a compile, and no draw of a state that was made is dropped. The runtime makes a state's
/// pipelines on its pool while the draws that need them go on (a draw's own key, and a state's
/// predicted key), so a command buffer committed before one is made waits for it. A driver's cold
/// compile (MoltenVK on a cold Metal compiler service) can take seconds, past a guest's watchdog,
/// where a host that creates a state lazily inside a command buffer's encoding would wait inside
/// it. The runtime's choice alone: no emitted code changes.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PipelineWaitPolicy {
    /// The default. A render or compute state's creation hands the making of its predicted
    /// pipeline to the runtime's pool and waits for it [`DevicePolicy::pipeline_creation_wait`] at
    /// most (4.5 s by default), then returns with it still being made; a committed command buffer
    /// waits for its draws' and dispatches' (a compute encoder's, a render pass's tile
    /// dispatches, an execution replay's draws) pipelines still being made until its commit plus
    /// [`DevicePolicy::pipeline_commit_wait`] at most (4.5 s by default), one deadline its
    /// commands share, then records those commands as nothing, the command buffer running on,
    /// each counted, the first named, announced `draw-dropped-still-making` or
    /// `dispatch-dropped-still-making`. Later command buffers draw and dispatch with each pipeline
    /// once made.
    ///
    /// What one call blocks for, waiting for the driver's makings: a state's creation, its
    /// creation wait; a `commit`, which records there what it can (a pass whose pipelines are
    /// made, an execution of render commands the CPU writes, which waits for its pipelines), until
    /// its own deadline at most (the moment it committed plus its commit wait), whatever earlier
    /// command buffers wait for (their deadlines are earlier); every other recording runs on the
    /// runtime's scheduler thread, which no program call waits for. A creation waits on no command buffer's
    /// deadline and a commit on no creation's, so a host call that creates a state and commits a
    /// command buffer waits their sum at most: 9 s by default. Made until made, under this value
    /// too, where the call that makes them waits for them: a draw that records nothing (no
    /// vertices or instances, or culled whole), a draw in a pass of custom sample positions, and
    /// one recorded again into another instance of its pass (an extended-range color's `half`
    /// shadow starting), each in the draw's call or in an execution's replay; a device replay's
    /// draws (an execution of render commands the GPU may write, recorded on the GPU), in its
    /// `commit`; and a making that failed, made again by its command's recording.
    #[default]
    Bounded,
    /// A render or compute state's creation makes its predicted pipeline before it returns, as
    /// Metal compiles there, and a committed command buffer waits for each of its draws' and
    /// dispatches' pipelines until it is made: no command is dropped for its making's time.
    Exact,
}

/// Which lines draw a line Apple's `setLineWidth:` widens (a width other than 1, quantized as Metal
/// quantizes it) where the device's own lines draw that width, which no device's are taken to draw
/// as Metal's parallelogram: strict lines (rectangles), lines not stated as parallelograms
/// (`nonStrictWideLinesUseParallelogram` unset or unknown), a width off the device's
/// `lineWidthGranularity` (which rounds it) or below 1 (where Vulkan promises no shape), and a
/// driver's parallelograms, whose ends and endpoints' snap are the driver's. Where they cannot draw
/// it at all (no `wideLines`, or a width outside `lineWidthRange`), and at width 0, the band draws
/// it under either value. The runtime's choice alone: no emitted code changes.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum WideLinePolicy {
    /// The device's own line at the width (the default): its rectangle, its Bresenham columns, its
    /// width rounded to the granularity, or its parallelogram's ends, which can cover pixels
    /// Metal's parallelogram does not and miss others. Announced `device-wide-lines` at the first
    /// such draw.
    #[default]
    Device,
    /// The pipeline's band variant (needs `tessellationShader`): each line two triangles over
    /// Metal's parallelogram. On a device without `tessellationShader`, the device's own line,
    /// announced as under `Device`.
    Exact,
}

/// What a blit or fill writes through a `BGRA10_XR` or `BGRA10_XR_sRGB` texture Metal keeps
/// losslessly compressed (`allowGPUOptimizedContents`, the shorter side of level 0 16 texels or
/// more, neither `ShaderWrite` nor `PixelFormatView` usage, one sample, not over a buffer). Metal's
/// GPU blit that reads or writes such a texture (a buffer into it, it into a buffer, a texture into
/// it or out of it) moves the texels through its compression, which writes their canonical
/// encoding: each 16-bit field's low six bits cleared and the alpha code clamped to `[384, 894]`;
/// its fill clears the low six bits alone. The texel reads the same value either way: the low six
/// bits are not read, and a read clamps the alpha. Measured on an Apple M3 Max under macOS 27.0.1:
/// random texels through each direction, both formats, every storage mode, the usages and sizes
/// that decide the compression; CPU copies (`replaceRegion`, `getBytes`) keep the bytes on Metal
/// too. The runtime's choice alone: no emitted code changes.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ExtendedRangeBlitPolicy {
    /// The device's copy, which keeps the bytes it moves (the default): no work past the copy.
    /// Announced `extended-range-bits-kept` at the first such blit or fill.
    #[default]
    Kept,
    /// Metal's canonical encoding: after a blit writes such a texture's region, or a blit out of
    /// one writes a buffer's or another texture's, a compute dispatch rewrites the texels it
    /// wrote (an image's through a scratch buffer, a copy in and out around the dispatch; a
    /// buffer's or a texture over a buffer's in place); a fill's texel is encoded before it is
    /// written. A blit out of one into a buffer whose bytes the device cannot bind there (an
    /// offset or pitch off 4 bytes, a region past `maxStorageBufferRange`), or into a texture over
    /// a buffer whose rows share bytes, keeps the bytes, announced as under `Kept`.
    Canonical,
}

/// Which `VkQueue` a command queue submits to. A `VkQueue` runs its submissions in order, so two
/// command queues on one wait for each other's whole command buffers, where Metal's queues wait
/// for each other only inside one command. NVIDIA's driver (615.71.09 on the RTX 5080 Laptop GPU)
/// leaves the GPU idle 56 to 62 us whenever consecutive work comes from another `VkQueue`, which
/// held a device of a `VkQueue` per command queue near 30,000 one-draw command buffers a second,
/// the GPU idle at the switches 60% of its time. Presenting queues, and queues made with
/// `new_command_queue_unshared`, keep their `VkQueue` under either value. The runtime's choice
/// alone: no emitted code changes.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum QueueSharingPolicy {
    /// The default, on a device whose own check, run once per driver and release, finds the GPU
    /// idle when consecutive work comes from another `VkQueue` of one family; elsewhere, and before
    /// the check answered, as [`PerQueue`](Self::PerQueue). A command queue that makes a command
    /// buffer while it holds no other uncompleted one takes the busiest `VkQueue` of its family
    /// whose oldest batch has run under 0.5 ms, else stays. A queue one of whose command buffers
    /// ran on the GPU for over 2 ms, by its own timestamps, is long from then on: at its next such
    /// point it takes an idle `VkQueue` no long queue holds and stays there, and no queue moves
    /// onto a long queue's `VkQueue`. So the first long command buffer of a queue that shared holds
    /// the command buffers the queues sharing with it committed behind it, once per queue that goes
    /// long. Announced `queue-sharing` at the first move onto a `VkQueue` with another queue's work
    /// in flight. A device whose command buffers write no GPU timestamps the host reads (the
    /// runtime's `gpu_times_from_host`, or a driver whose timestamps wait for every queue's) does
    /// not share, and the runtime counts it (`WorkCounts::queue_sharing_untimed`).
    #[default]
    Adaptive,
    /// Each command queue on the `VkQueue` it took at its creation, round robin, for its life:
    /// command queues share one only past the `VkQueue`s the device lends the round robin.
    PerQueue,
}

/// What a colour attachment of a texture's own image (not a texture over a buffer's bytes, not
/// memoryless, resolving nothing) keeps after a render pass stores it `MTLStoreActionDontCare`.
/// Metal's documentation leaves it open (`MTLStoreAction.dontCare`: "You need to assume that GPU
/// discarded the texture's contents"); what Apple's GPU leaves is measured: the texels the
/// attachment held before the pass, the pass's clear and its draws both gone (the conformance
/// case `pass-sequence-color-actions`, an Apple M3 Max under macOS 27.0.1, eight runs alike), or
/// the rendered texels where its tiler overflows. Vulkan's `DONT_CARE` leaves the texels
/// undefined. On a driver that stores `DONT_CARE` as Metal's own (MoltenVK on Apple's GPUs), a
/// pass recorded as one render pass instance is handed to it under either value. The runtime's
/// choice alone: no emitted code changes.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DontCareColorPolicy {
    /// As the driver stores it (the default): `DONT_CARE` on a driver measured to keep one of
    /// Metal's two answers (the earlier texels or the rendered ones) or to store as Metal's own,
    /// else `STORE`, the rendered texels, Metal's answer where its tiler overflows. No work past
    /// the pass. Announced `dont-care-store` at the first such pass a driver does not store as
    /// Metal's own.
    #[default]
    Driver,
    /// The texels the attachment held before the pass: the pass renders through a backing its
    /// command queue keeps, one image of the attachment's shape, into which a `Load` copies the
    /// texels first, and nothing is copied back. A `Load` pass costs the copy (on an RTX 5080
    /// Laptop GPU about 19 µs more a pass at 1920x1080 than one stored `Store`, 10 µs at
    /// 256x256); a `Clear` or `DontCare` load about none. Where a format makes no such image, the
    /// pass stores as under `Driver`, announced as there.
    Kept,
}

/// What a render pass's draws read of an attachment (not memoryless) the pass loads
/// `MTLLoadActionDontCare` and does not store (`MTLStoreActionDontCare`), in a pass whose draws
/// reach a tile. Metal's documentation leaves it open; what Apple's GPU reads is measured: zero for
/// a colour and the pass's clear value for depth and stencil, wherever a draw reads it (an Apple
/// M3 Max under macOS 27.0.1: a framebuffer fetch and a stencil test, in passes split and whole;
/// a depth and a stencil test of separate depth and stencil textures, each aspect apart). Vulkan's
/// `DONT_CARE` load leaves the contents undefined. Separate depth and stencil textures render
/// through a scratch image of the pass's own, whose load of each such aspect this decides. The
/// runtime's choice alone: no emitted code changes.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum UnstoredDontCarePolicy {
    /// Cleared on a driver measured to run the clear no slower than its `DONT_CARE` load
    /// (`DriverFact::UnstoredClearNoSlower`), Metal's answer there at no cost; `DONT_CARE`
    /// elsewhere (the default), whose draws read what the driver leaves (on lavapipe, the
    /// attachment's earlier texels), announced `dont-care-unstored-load` at the first such pass.
    #[default]
    Driver,
    /// Cleared on every driver by the pass's load operation over the render area (or before the
    /// pass, where its first render pass instance leaves the attachment out), to zero for a colour
    /// and to the attachment's clear value for depth and stencil: Metal's answer wherever a draw
    /// reads it. On lavapipe a 3648x2038 pass of an `RG16Float` colour and a
    /// `Depth32Float_Stencil8` so loaded costs about 3.3 ms more of its CPU's time.
    Cleared,
}
