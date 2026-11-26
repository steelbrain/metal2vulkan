//! Render pipeline states: the translator's render pipeline of a descriptor's content, its shader
//! modules, and its `VkPipeline`s, one for each topology class drawn.
//!
//! What a state compiles is its content, never the state: the translator's key of its descriptor
//! (`Device::queried_render_descriptor_key`: every field of the descriptor, each function by its AIR,
//! entry, constant values, linked functions and dynamic libraries, with the device's profile and
//! the translator's semantics version), which reads no function, as a compute state's key reads
//! none. Every state of one content shares one [`RenderCompiled`]: the translation, its shader
//! modules and their `VkPipeline`s. The device's one map of contents (`kernel_map`), which compute
//! states use too, holds each content's, so a state of a content the map holds costs the key and a
//! lookup. A content the map does not hold is decoded, keyed for its compile
//! (`Device::render_variant_key`) and lowered, and its record, the keys of its two lowerings, is
//! kept in the cache directory under the descriptor's key (`crate::cache`): a device after it on
//! the directory reads the record and the two lowerings in place of decoding, keying and lowering
//! the content. The cost of keying on the descriptor: two descriptors that differ only in a field
//! their compile does not read are two contents, each compiled.
//!
//! What a `VkPipeline` bakes is what Vulkan 1.3 cannot leave dynamic: the content, the topology
//! class (point, line or triangle) of the draws it serves, and the encoder's depth clip mode and
//! triangle fill mode ([`Raster`]). Everything Metal sets on the
//! encoder that Vulkan 1.3 makes dynamic without a feature is dynamic state: the viewports and
//! scissors with their count,
//! cull mode, front face, the primitive topology within its class and primitive restart, the
//! depth test, write and compare, the stencil test, operations, masks and reference, depth bias
//! and blend constants. Vertex buffer strides are the descriptor's, so the content's: they are
//! baked, and a layout whose stride is set per bind (`MTLBufferLayoutStrideDynamic`), which the
//! replay streams' never are, is refused. A draw under the depth clip mode `Clamp` or the triangle
//! fill mode `Lines` takes a `VkPipeline` of its class made for that raster state, by the first
//! such draw of the content, which waits for it, as a draw of a class not made yet does: Metal
//! sets both per encoder and compiles nothing for them. The fill mode is a triangle's
//! alone, in Metal and in Vulkan, so the line and point classes take the clip mode alone.
//!
//! What a draw supplies is the content's [`Draws`]: each function's slots with their accesses,
//! its push fields (addresses, resource IDs and indices of the slots, in the floor's one push
//! range, the fragment function's after the vertex function's), and the vertex buffers its
//! fixed-function vertex input reads. A state numbers its functions' samplers as a compute state
//! numbers its kernel's, the vertex function's first.
//!
//! A descriptor names no topology class unless it sets `inputPrimitiveTopology`, and the replay
//! streams' never do, while nearly every content draws only triangles (527 of the 537 contents
//! gb and b2 draw). So the triangle class's `VkPipeline` is the content's main one, which Metal
//! compiles with the state, and so is the line or the point class's where the descriptor names
//! that class; otherwise each comes with the draws, made by the first draw of lines or of points,
//! as only a draw gives the class (one `VkPipeline` for several classes would need a dynamic
//! topology across classes, which Vulkan 1.3 does not give). The cost: a content drawn only as
//! lines or points makes a triangle `VkPipeline` it never uses (8 contents of the 537). The state's
//! make does everything up to the driver's create of the main ones and answers; the consumer's
//! [`RenderPipelineState::finish_pipeline`] call makes them after, on a thread of its own, or else
//! the first draw that needs each, which waits for it ([`Main`]).
//!
//! A point's size is the `PointSize` the vertex module writes: the function's `[[point_size]]`,
//! at most Metal's 511 pixels, or 1.0 for a function without one, which the device draws with
//! `largePoints` (Floor), clamped to its `pointSizeRange` (`PrecisionNote::DevicePointSize` where
//! the range ends below 511). Metal draws a point of a function without a size only under the point
//! class; under any other class it rasterizes nothing and runs the vertex function all the same,
//! so the encoder draws such points behind an empty scissor.
//!
//! A line is drawn by Bresenham's rule, which is Metal's (`bresenhamLines`, Floor): the line
//! class's `VkPipeline` chains `VK_LINE_RASTERIZATION_MODE_BRESENHAM_EXT`, except under alpha to
//! coverage or alpha to one, which the mode does not take, where the device's default rule draws
//! it. Either way its pixels are the device's, within Vulkan's allowance of the diamond-exit rule,
//! which a draw of lines records (`RenderCommandEncoder::LINE_NOTE`).
//!
//! A content whose vertex function is post-tessellation draws patches alone, so its one
//! `VkPipeline` is the patch class's, its main one: the translator's synthesized vertex and
//! control stages, the function as the evaluation stage, and a `PATCH_LIST` of one control point
//! per patch. Its control stage reads each patch's factors through their address, which a draw
//! pushes ([`FieldValue::TessellationFactors`]); the evaluation stage reaches its control points'
//! and patches' buffers through their pushed addresses, at the strides of the descriptor's layouts
//! ([`FieldValue::Word`]). A per-instance factor step function and an enabled factor scale, which
//! the replay streams' states never use, are refused ([`RenderPipelineKind::Tessellation`]).

use crate::cache::{self, Lowerings};
use crate::counters::{Counters, FirstUse, Gauge, Metric};
use crate::depth_stencil::EVERY_FRAGMENT;
use crate::device::{lock, Core};
use crate::error::{Error, Gap, Refusal, RenderPipelineKind, StageGap};
use crate::hazard::Access;
use crate::kernel_map::KernelKey;
use crate::pass::Pass;
use crate::pipeline::{
    access, loaded, lower, made_through_cache, number_samplers, of_the_device, texture_access,
    translating, FieldValue, Numbered,
};
use crate::reaper::{Free, Job};
use crate::ring::CHUNK;
use crate::sampler::constexpr_notes;
use crate::times::host_time;
use ash::vk;
use metal2vulkan::cache::{Hash256, VariantKey};
use metal2vulkan::device::EnabledCaps;
use metal2vulkan::metal::{BindingAccess, PrimitiveTopologyClass};
use metal2vulkan::pipeline::{
    specialize, BufferKind, CombinedSampler, NumberedSampler, PrecisionNote, PushField, PushLoc,
    RenderPipeline, RenderPipelineDescriptor, RenderPipelineKeys, RuntimeFact, SamplerRef,
    SlotClass, SlotInterface, SlotRef, SpecKey, Stage, StageEntries, StageSource, VertexStride,
};
use metal2vulkan::Budget;
use std::ffi::CString;
use std::sync::{Arc, Condvar, Mutex, OnceLock, PoisonError};

/// `MTLRenderPipelineState`.
///
/// A cheap handle: clones share one state. A command buffer that draws with it keeps it until it
/// retires. States of one content (the functions, their constant values and linked functions, and
/// the descriptor's fields that change the compile) share its translation and `VkPipeline`s, so a
/// guest that makes one state many times compiles it once; each numbers its own samplers. It
/// answers no `gpuResourceID`, which Metal's state has (`MTLRenderPipelineState.gpuResourceID`).
#[derive(Clone)]
pub struct RenderPipelineState {
    pub(crate) inner: Arc<RenderInner>,
}

pub(crate) struct RenderInner {
    /// What the state's content compiles to, shared with every state of that content.
    pub(crate) compiled: Arc<RenderCompiled>,
    /// The samplers the state numbers and the indices the consumer gave them.
    pub(crate) numbered: Numbered,
}

/// The vertex function's index in a draw's tables, and the fragment function's.
pub(crate) const VERTEX: usize = 0;
pub(crate) const FRAGMENT: usize = 1;

/// Where a draw's field goes: a byte offset of the push range, or of a function's parameter block
/// ([`VERTEX`] or [`FRAGMENT`]'s), which a draw writes into the parameter ring.
#[derive(Clone, Copy)]
pub(crate) enum DrawPlace {
    Push(u32),
    Block(usize, u32),
}

/// What a draw of a content supplies, from the translator.
pub(crate) struct Draws {
    /// Each buffer slot a function uses: its function ([`VERTEX`] or [`FRAGMENT`]), its index and
    /// its access.
    pub(crate) buffers: Vec<(usize, u32, Access)>,
    /// Each texture slot a function uses, likewise.
    pub(crate) textures: Vec<(usize, u32, Access)>,
    /// Each texture slot a function reads through its storage image (`TextureSlot::storage`, a
    /// `read_write` slot it reads), with its function, which takes no swizzle: a swizzled texture
    /// bound there is `Gap::SwizzledReadWrite`.
    pub(crate) storage_reads: Vec<(usize, u32)>,
    /// Each texture slot whose storage image a function declares without a format, with its
    /// function (`crate::pipeline::Interface`'s `formatless`).
    pub(crate) formatless: Vec<(usize, u32)>,
    /// The fragment function's texture slots it reads only at the fragment's own pixel
    /// (`StageModule::own_pixel_textures`), sorted: what a draw copies of an attachment it reads
    /// through one on a device without the attachment feedback loop (`crate::shadow`).
    pub(crate) own_pixel: Vec<u32>,
    /// The colour attachments the fragment function reads at its own pixel (`[[color(n)]]`)
    /// through the element a draw pushes for each (`PushField::FetchedColor`), sorted: each an
    /// attachment read of the pass through its texture's second element, ordered after every
    /// earlier draw of the pass (`crate::render_encoder`).
    pub(crate) fetched: Vec<u32>,
    /// Each texture slot a function samples through a bicubic constexpr sampler, with its
    /// function: the slots whose `PushField::BicubicTexels` a draw writes from the bound texture,
    /// and whose texture a draw's checks refuse where the filter's treatment of its format is not
    /// established (`Gap::BicubicTexels`). Empty for every other content.
    pub(crate) bicubic: Vec<(usize, u32)>,
    /// Each texture slot a function writes or reads at a level other than the constant 0, with its
    /// function: the slots whose `PushField::TextureLevelBase` a draw writes, the first element
    /// of the bound texture's run of level elements, which a draw's checks hold first. Empty for
    /// every other content.
    pub(crate) levels: Vec<(usize, u32)>,
    /// Each fixed-function vertex binding: its binding and the vertex buffer index it reads.
    pub(crate) bindings: Vec<(u32, u32)>,
    /// Whether a function reads through an address that is not one of its slots.
    pub(crate) reads_unbound: bool,
    /// Whether a function writes through an address that is not one of its slots.
    pub(crate) writes_unbound: bool,
    /// Whether a function may reach a texture through its storage image through something other
    /// than its slots (`PipelineInterface::unbound_storage_textures`): an argument buffer's
    /// `access::write` or `access::read_write` texture, a resource ID.
    pub(crate) unbound_storage: bool,
    /// Whether a function writes memory (`PipelineInterface::writes_memory`).
    pub(crate) writes_memory: bool,
    /// The bytes of the push range the functions read, from 0.
    pub(crate) push_len: u32,
    /// Each field: where it goes, the tables it reads, and its value.
    pub(crate) fields: Vec<(DrawPlace, usize, FieldValue)>,
    /// Each function's parameter block, where its fields spill past its share of the push range:
    /// its length, and the push offset of its address (`PushField::ArgumentTableAddress`).
    pub(crate) blocks: [Option<(u32, u32)>; 2],
    /// Whether a draw must be at base instance 0: a binding has a divisor other than 1 and the
    /// device does not report `supportsNonZeroFirstInstance` (VUID-vkCmdDraw-pNext-09461).
    pub(crate) first_instance_zero: bool,
    /// Whether the vertex function is post-tessellation: the content draws patches alone, through
    /// its patch-class `VkPipeline`, and reads the encoder's tessellation factor buffer.
    pub(crate) patches: bool,
    /// Whether the vertex function writes `[[viewport_array_index]]`
    /// (`GraphicsState::writes_viewport_index`): its primitives reach every viewport the encoder
    /// set, where any other function's reach viewport 0.
    pub(crate) selects_viewport: bool,
    /// The push offset of the bit the fragment function keeps its fragments for
    /// (`PushField::StencilBit`), where its stencil output is drawn by bits
    /// (`RuntimeFact::StencilBitDraws`): each draw takes the draws of
    /// [`BitDraws`](crate::depth_stencil::BitDraws), each pushing its bit there.
    pub(crate) stencil_bit: Option<u32>,
}

/// The state Vulkan 1.3 lets a draw set, which every render `VkPipeline` takes as dynamic.
const DYNAMIC: [vk::DynamicState; 17] = [
    vk::DynamicState::VIEWPORT_WITH_COUNT,
    vk::DynamicState::SCISSOR_WITH_COUNT,
    vk::DynamicState::CULL_MODE,
    vk::DynamicState::FRONT_FACE,
    vk::DynamicState::PRIMITIVE_TOPOLOGY,
    vk::DynamicState::PRIMITIVE_RESTART_ENABLE,
    vk::DynamicState::DEPTH_TEST_ENABLE,
    vk::DynamicState::DEPTH_WRITE_ENABLE,
    vk::DynamicState::DEPTH_COMPARE_OP,
    vk::DynamicState::STENCIL_TEST_ENABLE,
    vk::DynamicState::STENCIL_OP,
    vk::DynamicState::STENCIL_COMPARE_MASK,
    vk::DynamicState::STENCIL_WRITE_MASK,
    vk::DynamicState::STENCIL_REFERENCE,
    vk::DynamicState::DEPTH_BIAS,
    vk::DynamicState::DEPTH_BIAS_ENABLE,
    vk::DynamicState::BLEND_CONSTANTS,
];

/// What one render pipeline content compiles to, shared by every state of it and held by the
/// device's map of contents until the map forgets it.
pub(crate) struct RenderCompiled {
    core: Arc<Core>,
    /// Each stage's module (its specialization constants fixed), entry point and stage.
    stages: Vec<Module>,
    /// The fixed-function state every `VkPipeline` of the content takes.
    fixed: Fixed,
    /// The create flags every `VkPipeline` of the content takes: the feedback loop's colour flag
    /// where device creation enabled the attachment feedback loop (`Core::feedback_loop`), and
    /// none otherwise. A state is made before any draw binds its textures, and which texture a
    /// draw binds decides whether it reads an attachment of its pass, so every pipeline takes it;
    /// a draw that reads none is valid with it.
    pub(crate) flags: vk::PipelineCreateFlags,
    /// The content's main `VkPipeline`: the triangle class's, or a post-tessellation vertex
    /// function's patch class's, its only class, at the default [`Raster`].
    primary: Main,
    /// The line or the point class's `VkPipeline` at the default raster state, by its list
    /// topology, a main one too, where the descriptor names that class
    /// (`inputPrimitiveTopology`).
    named: Option<(vk::PrimitiveTopology, Main)>,
    /// The content's other `VkPipeline`s, each by its class's list topology and its raster state,
    /// each made by the first draw that takes it: the line and the point class's at the default
    /// raster state where the descriptor names neither, and those under other raster states. At
    /// most 7: the triangle or patch class's three other raster states, and two of each other
    /// class.
    later: Mutex<Vec<(vk::PrimitiveTopology, Raster, vk::Pipeline)>>,
    /// Whether a draw of points rasterizes them: the vertex function writes `[[point_size]]`, or
    /// the descriptor names the point class, under which Metal draws a point of a function that
    /// writes no size 1 pixel wide (the module writes `PointSize` 1.0). Under any other class
    /// Metal rasterizes no point of such a function and runs its vertex function all the same
    /// (`metal2vulkan::pipeline::GraphicsState::writes_point_size`).
    rasterizes_points: bool,
    /// The translator's notes of where the pipeline's answers may differ from Metal's, and its
    /// functions' constexpr samplers' (`constexpr_notes`).
    notes: Vec<PrecisionNote>,
    /// The translator's content digest of the pipeline (`RenderPipeline::content_digest`).
    digest: Hash256,
    /// What a draw supplies.
    pub(crate) draws: Draws,
    /// The samplers a state numbers, the vertex function's then the fragment function's, each in
    /// Metal's order, and each constexpr sampler a function samples through, by its place there,
    /// with its use.
    numbered: Vec<NumberedSampler>,
    constexprs: Vec<(usize, CombinedSampler)>,
    /// Metal's `imageblockSampleLength` of the content: its color attachments' texels as Metal
    /// packs them (`metal::format::imageblock_sample_length`); `None` where a color format has no
    /// measured place in Metal's record.
    imageblock_sample: Option<u32>,
    /// The descriptor's `rasterSampleCount`.
    samples: u32,
    /// The descriptor's `supportIndirectCommandBuffers`.
    indirect: bool,
    /// The content's number, which no other content of the process takes: the key of its
    /// executions' shapes (`crate::indirect`), which outlive no content they name.
    pub(crate) id: u64,
}

/// What the render encoder sets that a `VkPipeline` bakes: `setDepthClipMode:` and
/// `setTriangleFillMode:`, in `VkPipelineRasterizationStateCreateInfo`, which only
/// `VK_EXT_extended_dynamic_state3` makes dynamic, and the floor does not require it. The default
/// is Metal's, `Clip` and `Fill`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct Raster {
    /// `MTLDepthClipModeClamp`: `depthClampEnable`.
    pub(crate) clamp: bool,
    /// `MTLTriangleFillModeLines`: `polygonMode` `LINE`.
    pub(crate) lines: bool,
}

/// One of a content's main `VkPipeline`s ([`RenderCompiled::primary`],
/// [`RenderCompiled::named`]): made by a state's [`RenderPipelineState::finish_pipeline`] call,
/// or by the first draw that needs it, which waits for a make in flight. It is made by one thread
/// at a time; a make that failed keeps nothing, and the next call or draw makes it again.
#[derive(Default)]
pub(crate) struct Main {
    /// The pipeline once made, which a draw reads with no lock.
    made: OnceLock<vk::Pipeline>,
    /// Whether a thread is making it.
    making: Mutex<bool>,
    /// Signalled when a make ends, made or failed: a draw waiting for it looks again.
    ended: Condvar,
}

/// A make of a [`Main`] in flight: when it ends, made or not, panicking included, the cell is no
/// longer being made, and the draws waiting for it wake.
struct Making<'a>(&'a Main);

impl Drop for Making<'_> {
    fn drop(&mut self) {
        *lock(&self.0.making) = false;
        self.0.ended.notify_all();
    }
}

impl Main {
    /// The pipeline, where it is made.
    fn get(&self) -> Option<vk::Pipeline> {
        self.made.get().copied()
    }

    /// Makes the pipeline with `make` on the calling thread, where it is neither made nor being
    /// made: its nanoseconds where this made it, none where it did not need to, or the make's
    /// error, which keeps nothing.
    pub(crate) fn make_ahead(
        &self,
        make: impl FnOnce() -> Result<vk::Pipeline, Error>,
    ) -> Result<Option<u64>, Error> {
        if self.made.get().is_some() {
            return Ok(None);
        }
        let mut making = lock(&self.making);
        if *making || self.made.get().is_some() {
            return Ok(None);
        }
        *making = true;
        drop(making);
        let began = host_time();
        self.make(make)?;
        Ok(Some(host_time().saturating_sub(began)))
    }

    /// The pipeline: read with no lock where it is made; else waited for where another thread is
    /// making it, and made with `make` on the calling thread where none is or that make failed. A
    /// call that did not find it made is a first-use wait ([`FirstUse::Unfinished`]).
    pub(crate) fn pipeline(
        &self,
        counters: &Counters,
        make: impl FnOnce() -> Result<vk::Pipeline, Error>,
    ) -> Result<vk::Pipeline, Error> {
        if let Some(pipeline) = self.get() {
            return Ok(pipeline);
        }
        let began = host_time();
        let mut making = lock(&self.making);
        while *making {
            making = (self.ended.wait(making)).unwrap_or_else(PoisonError::into_inner);
        }
        let made = self.get();
        if made.is_none() {
            *making = true;
        }
        drop(making);
        let pipeline = match made {
            Some(pipeline) => pipeline,
            None => self.make(make)?,
        };
        counters.first_use_wait(FirstUse::Unfinished, host_time().saturating_sub(began));
        Ok(pipeline)
    }

    /// Runs `make`, the cell marked as being made by the caller, and keeps what it made.
    fn make(
        &self,
        make: impl FnOnce() -> Result<vk::Pipeline, Error>,
    ) -> Result<vk::Pipeline, Error> {
        let _making = Making(self);
        let pipeline = make()?;
        // Only the thread that marked the cell sets it, and it was empty when marked.
        let _ = self.made.set(pipeline);
        Ok(pipeline)
    }
}

/// The next content's [`RenderCompiled::id`].
static CONTENT_IDS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// One stage's shader module and what a `VkPipeline` of it takes.
pub(crate) struct Module {
    pub(crate) module: vk::ShaderModule,
    pub(crate) entry: CString,
    pub(crate) stage: vk::ShaderStageFlags,
    pub(crate) flags: vk::PipelineShaderStageCreateFlags,
    /// The subgroup size the translator pins for the stage (`StageModule::required_subgroup_size`:
    /// Metal's 32, on a device that lists the stage in `requiredSubgroupSizeStages`), which its
    /// SIMD-group operations are lowered for; `None` where it pins none.
    pub(crate) subgroup_size: Option<u32>,
}

/// Each of `stages`' required subgroup size as Vulkan chains it
/// (`VkPipelineShaderStageRequiredSubgroupSizeCreateInfo`), where the translator pins one.
pub(crate) fn subgroup_sizes(
    stages: &[Module],
) -> Vec<Option<vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo<'static>>> {
    (stages.iter())
        .map(|stage| {
            (stage.subgroup_size).map(|size| {
                vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
                    .required_subgroup_size(size)
            })
        })
        .collect()
}

/// The create info of each of `stages`, with its required subgroup size of `subgroups`
/// ([`subgroup_sizes`]) chained where it has one. A stage's module holds its specialization
/// constants fixed ([`specialize`]), so no create info takes a specialization.
pub(crate) fn stage_infos<'a>(
    stages: &'a [Module],
    subgroups: &'a mut [Option<vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo<'static>>],
) -> Vec<vk::PipelineShaderStageCreateInfo<'a>> {
    (stages.iter())
        .zip(subgroups.iter_mut())
        .map(|(stage, subgroup)| {
            let info = vk::PipelineShaderStageCreateInfo::default()
                .flags(stage.flags)
                .stage(stage.stage)
                .module(stage.module)
                .name(&stage.entry);
            match subgroup {
                Some(subgroup) => info.push_next(subgroup),
                None => info,
            }
        })
        .collect()
}

/// The fixed-function state of a content's `VkPipeline`s, as Vulkan takes it: the descriptor's
/// vertex layout, attachments, blending, sample count and private sample mask, as the translator
/// states them.
struct Fixed {
    bindings: Vec<vk::VertexInputBindingDescription>,
    /// Each instance-rate binding's divisor other than 1, a per-instance step rate the translator
    /// states only where device creation enabled the step-rate feature.
    divisors: Vec<vk::VertexInputBindingDivisorDescriptionKHR>,
    attributes: Vec<vk::VertexInputAttributeDescription>,
    discard: bool,
    samples: vk::SampleCountFlags,
    alpha_to_coverage: bool,
    /// `alphaToOneEnabled`, the device's own (`alphaToOne`, which device creation enables where
    /// the device reports it; the translator refuses the state on a device without it).
    alpha_to_one: bool,
    /// The descriptor's private sample mask and sample coverage as one mask of the samples the
    /// draws write (`GraphicsState::sample_mask`), Vulkan's `pSampleMask`; `None` where it keeps
    /// every sample or the fragment function applies it itself.
    sample_mask: Option<u32>,
    blends: Vec<vk::PipelineColorBlendAttachmentState>,
    /// The logic operation the translator states (private render state), the device's own
    /// (`logicOp`, which the floor requires); `None` with the operation off, or copy or no-op,
    /// which the attachments' states already draw.
    logic_op: Option<vk::LogicOp>,
    /// Each colour attachment's format by Metal index, `UNDEFINED` for an absent one.
    colors: Vec<vk::Format>,
    /// A patch list's control points per patch (`patchControlPoints`), 0 but for a
    /// post-tessellation vertex function's content.
    patch_control_points: u32,
    /// The depth and the stencil attachment's formats, `UNDEFINED` for none.
    depth: vk::Format,
    stencil: vk::Format,
}

impl crate::Device {
    /// `newRenderPipelineStateWithDescriptor:error:`, without reflection.
    ///
    /// A state's content is its descriptor: its functions with their constant values and linked
    /// functions, and every field. Every state of one content shares its translation, its shader
    /// modules and their `VkPipeline`s: a state of a content the device holds (the last 1,024
    /// contents used, compute and render together) costs the translator's key of the descriptor,
    /// which reads no function, and a lookup. For a
    /// content the device does not hold, the translator lowers each function on the calling
    /// thread, which waits for it, within its 20-second budget per lowering, and its shader
    /// modules are made here. The call answers before the driver's create of the content's main
    /// `VkPipeline`s, the triangle class's, and the line or the point class's where the
    /// descriptor's `inputPrimitiveTopology` names that class, which the consumer's
    /// [`finish_pipeline`](RenderPipelineState::finish_pipeline) call makes after it, on a thread
    /// of its own, or else the first draw that needs each; any other class's comes with the draws.
    /// Where the device has a cache directory ([`DeviceOptions::pipeline_cache_dir`]), a lowering
    /// a device made before is read from it in place of lowering again, and each `VkPipeline` is
    /// made through the pipeline data kept for it there; a content a device made before is read
    /// by its record there, kept under the translator's key of the descriptor, which names its two
    /// lowerings, so its functions are neither decoded nor specialized
    /// ([`Metrics::render_records_read`](crate::Metrics::render_records_read)). The
    /// `completionHandler:` forms are not built (no replay stream calls them).
    ///
    /// [`DeviceOptions::pipeline_cache_dir`]: crate::DeviceOptions::pipeline_cache_dir
    ///
    /// Built: one vertex and at most one fragment function, colour attachments of any format the
    /// translator maps for the device, a depth and a stencil attachment of a format the device
    /// renders depth or stencil in (its format query, through the translator's mapping; another
    /// is the translator's refusal), the raster sample count with the private sample mask and
    /// sample coverage, the private logic operation, and every blend state, write mask and
    /// vertex descriptor of static strides Vulkan's fixed-function state takes with
    /// what device creation enables; what else the translator states is refused at the creation,
    /// by kind.
    ///
    /// The state numbers samplers as Metal's pipeline creation does
    /// (`PipelineInterface::numbered_samplers`), as
    /// [`new_compute_pipeline_state`](Self::new_compute_pipeline_state) numbers a kernel's: once
    /// the functions are lowered, `sampler_ids` takes the vertex function's, then the fragment
    /// function's, each in Metal's order, and answers the sampler index the consumer gives each.
    /// Each constexpr sampler a function samples through takes the device sampler of each use it
    /// has at its index before the call returns, and every draw pushes the index.
    ///
    /// Err: [`Error::Translation`], the translator's typed refusal or a descriptor Metal refuses,
    /// naming the vertex and the fragment function;
    /// [`Gap::RenderPipeline`] for what the runtime does not build, naming it;
    /// [`Refusal::PointSizeTopology`] for a vertex function that writes `[[point_size]]` under
    /// the line or the triangle class; [`Refusal::SamplerIds`] where `sampler_ids` answers another
    /// count; [`Refusal::IndicesNotGranted`] for a constexpr sampler's index;
    /// [`Limit::DeviceSamplers`](crate::Limit::DeviceSamplers); [`Error::Vulkan`] where a shader
    /// module cannot be made.
    pub fn new_render_pipeline_state(
        &self,
        descriptor: &RenderPipelineDescriptor,
        sampler_ids: impl FnOnce(&[NumberedSampler]) -> Result<Vec<u32>, Error>,
    ) -> Result<RenderPipelineState, Error> {
        let stages = std::iter::once(&descriptor.vertex).chain(&descriptor.fragment);
        (self.render_pipeline_state(descriptor, sampler_ids))
            .map_err(|error| error.of_stages(stages))
    }

    /// [`new_render_pipeline_state`](Self::new_render_pipeline_state), its translator's error
    /// naming no function.
    fn render_pipeline_state(
        &self,
        descriptor: &RenderPipelineDescriptor,
        sampler_ids: impl FnOnce(&[NumberedSampler]) -> Result<Vec<u32>, Error>,
    ) -> Result<RenderPipelineState, Error> {
        refuse(descriptor)?;
        let core = self.core();
        let key = KernelKey {
            pipeline: core.translator.queried_render_descriptor_key(descriptor),
            required: None,
            srgb: 0,
            words: 0,
        };
        let compiled = match core.kernels.find_render(&key) {
            Some(compiled) => compiled,
            None => {
                let budget = Budget::standard();
                let content = key.pipeline.digest();
                let lowered = match recorded(core, descriptor, content) {
                    Some(lowered) => lowered,
                    None => {
                        let entries = &core.entries;
                        let keys = translating(core, || {
                            (core.translator).render_variant_key(
                                descriptor,
                                &VariantKey::new(),
                                &budget,
                                entries,
                            )
                        })
                        .map_err(Error::translation)?;
                        let lowered = lower_stages(core, descriptor, &keys, &budget, entries);
                        core.note_entries();
                        let (lowered, lowerings) = lowered?;
                        cache::keep_record(core, content, &lowerings);
                        lowered
                    }
                };
                let compiled = RenderCompiled::of(core, &lowered, &budget)?;
                (core.kernels).keep_render(key, Arc::new(compiled), &core.counters)
            }
        };
        let numbered = number_samplers(
            core,
            compiled.numbered.clone(),
            &compiled.constexprs,
            sampler_ids,
            draw_input,
        )?;
        Ok(RenderPipelineState {
            inner: Arc::new(RenderInner { compiled, numbered }),
        })
    }
}

/// Whether a content's lines are drawn by the device's default rule rather than Bresenham's:
/// its state enables alpha to coverage or alpha to one, which Vulkan's Bresenham mode does not
/// take (`VUID-VkGraphicsPipelineCreateInfo-lineRasterizationMode-02766`).
fn lines_by_device_rule(fixed: &Fixed) -> bool {
    fixed.alpha_to_coverage || fixed.alpha_to_one
}

/// The samplers a state of a content numbers and the constexpr samplers its functions sample
/// through ([`RenderCompiled::numbered`]).
type Numbering = (Vec<NumberedSampler>, Vec<(usize, CombinedSampler)>);

/// [`Gap::RenderPipeline`] of [`RenderPipelineKind::DrawInput`].
fn draw_input() -> Error {
    Error::Unsupported(Gap::RenderPipeline {
        kind: RenderPipelineKind::DrawInput,
    })
}

/// The samplers a state of `pipeline` numbers and the constexpr samplers its functions sample
/// through ([`RenderCompiled::numbered`]), and what a draw of it supplies.
///
/// Err: [`Gap::RenderPipeline`] of [`RenderPipelineKind::DrawInput`] for a pipeline that reads
/// what a draw does not supply: a function table, a pointer or handle translation table, a parameter block
/// longer than a ring chunk, push bytes past the device's push `window`
/// (`DeviceProfile::push_window`), a push field or bind-time fact the encoder does not fill, or an
/// unnormalized constexpr sampler.
fn draws(
    pipeline: &RenderPipeline,
    nonzero_first_instance: bool,
    window: u32,
) -> Result<(Draws, Numbering), Error> {
    let translated = &pipeline.interface;
    if translated.needs_pointer_translation
        || translated.needs_handle_translation
        || !pipeline.layout.sets.iter().all(of_the_device)
    {
        return Err(draw_input());
    }
    let table = |stage| match stage {
        Stage::Vertex => Ok(VERTEX),
        Stage::Fragment => Ok(FRAGMENT),
        _ => Err(draw_input()),
    };
    // The slot's address, rounded down to a multiple of its argument's alignment.
    let mut aligned = Vec::new();
    for dependency in &pipeline.dependencies {
        match dependency.fact {
            RuntimeFact::AddressAlignment {
                slot:
                    SlotRef {
                        stage,
                        class: SlotClass::Buffer,
                        index,
                    },
                align,
            } if align.is_power_of_two() => {
                aligned.push((table(stage)?, index, u64::from(align) - 1))
            }
            RuntimeFact::DeviceAddresses { .. } => {}
            // The draw's base vertex and base instance: under the floor's `shaderDrawParameters`
            // the `BaseVertex` and `BaseInstance` built-ins, which are the draw's own Vulkan
            // parameters. The push field the translator places without the feature is a push
            // field the encoder does not fill, refused below.
            RuntimeFact::BaseVertexInstance => {}
            // The rasterization sample count (`get_num_samples()`, the mask of a sample index):
            // the `SpecKey::SampleCount` constant, which the state's creation sets to its
            // `rasterSampleCount`, the count of every pass it draws in (`renders_into`).
            RuntimeFact::SampleCount => {}
            // The pulled attributes and their layouts' strides of a post-tessellation function and
            // of a state made for indirect command buffers, which pulls every attribute: the
            // content's own descriptor, whose projection link checked. A stride set per bind is
            // refused at its push field below.
            RuntimeFact::VertexFetch { .. } | RuntimeFact::VertexBufferStride { .. } => {}
            // A colour attachment the fragment function reads, through the element its push field
            // names (`FieldValue::FetchedColor`); one read through an input attachment or a storage
            // image is a descriptor, which the translator refuses under the identity domains.
            RuntimeFact::FramebufferFetch { .. } => {}
            // A vertex function's `[[render_target_array_index]]`, in its one form: the variant
            // that writes `Layer`, which reads the pass's length from its push field
            // (`FieldValue::RenderTargetArrayLength`). The variant that writes none renders every
            // index into the pass's first layer, so a draw of it in a layered pass is not
            // supplied.
            RuntimeFact::RenderTargetArrayLength {
                layer_dropped: false,
            } => {}
            // A vertex function's `[[viewport_array_index]]`, which stores `ViewportIndex` 0 at or
            // past the encoder's viewport count, read from its push field
            // (`FieldValue::ViewportCount`).
            RuntimeFact::ViewportCount => {}
            // A fragment function's stencil output on a device without stencil export, which each
            // draw takes by bits through its push field (`Draws::stencil_bit`).
            RuntimeFact::StencilBitDraws => {}
            // `RuntimeFact` is non-exhaustive: a fact this match does not name is one a draw does
            // not supply.
            _ => return Err(draw_input()),
        }
    }
    let mut buffers = Vec::new();
    let mut textures = Vec::new();
    let mut storage_reads = Vec::new();
    let mut formatless = Vec::new();
    for (slot, used) in &translated.slots {
        let at = table(slot.stage)?;
        match (slot.class, used) {
            (SlotClass::Buffer, SlotInterface::Buffer(buffer))
                if buffer.kind == BufferKind::Buffer =>
            {
                buffers.push((at, slot.index, access(buffer.access)));
            }
            (SlotClass::Texture, SlotInterface::Texture(texture)) => {
                textures.push((at, slot.index, texture_access(texture.access)));
                if texture.storage && texture.access.read {
                    storage_reads.push((at, slot.index));
                }
                if texture.storage && !texture.declared_format {
                    formatless.push((at, slot.index));
                }
            }
            (SlotClass::Sampler, SlotInterface::Sampler(_)) => {}
            _ => return Err(draw_input()),
        }
    }
    // The samplers a state numbers: the vertex function's, then the fragment function's.
    let mut numbered = Vec::new();
    let mut firsts = [0; 2];
    for (stage, samplers) in &translated.numbered_samplers {
        firsts[table(*stage)?] = numbered.len();
        numbered.extend(samplers.iter().copied());
    }
    let mut constexprs = Vec::new();
    let mut fields = Vec::new();
    let mut fetched = Vec::new();
    let mut bicubic = Vec::new();
    let mut levels = Vec::new();
    let mut push_len = 0;
    let mut blocks = [None; 2];
    let mut stencil_bit = None;
    for placement in &translated.push {
        let at = table(placement.stage)?;
        if let Some((offset, size)) = placement.push_range {
            push_len = push_len.max(offset + size);
        }
        let mut address = None;
        for placed in &placement.fields {
            let offset = match placed.loc {
                PushLoc::Push { offset } => DrawPlace::Push(offset),
                PushLoc::ArgumentTable { offset } => DrawPlace::Block(at, offset),
                _ => return Err(draw_input()),
            };
            let value = match placed.field {
                // The function's parameter block's address, which a draw writes once the block is
                // in the ring.
                PushField::ArgumentTableAddress => match offset {
                    DrawPlace::Push(pushed) => {
                        address = Some(pushed);
                        continue;
                    }
                    DrawPlace::Block(..) => return Err(draw_input()),
                },
                PushField::SlotGuestBase(SlotRef {
                    stage,
                    class,
                    index,
                }) => {
                    let of = table(stage)?;
                    let value = match class {
                        SlotClass::Buffer => FieldValue::BufferAddress {
                            slot: index,
                            mask: (aligned.iter())
                                .find(|&&(table, slot, _)| table == of && slot == index)
                                .map_or(0, |&(_, _, mask)| mask),
                        },
                        SlotClass::Texture => FieldValue::TextureId(index),
                        SlotClass::Sampler => FieldValue::SamplerId(index),
                    };
                    fields.push((offset, of, value));
                    continue;
                }
                PushField::TextureIntegerChannels(SlotRef {
                    stage,
                    class: SlotClass::Texture,
                    index,
                }) => {
                    fields.push((offset, table(stage)?, FieldValue::IntegerChannels(index)));
                    continue;
                }
                // How Metal's bicubic filter treats the bound texture's texels, the word a
                // dispatch writes for its kernel's slot (`FieldValue::BicubicTexels`).
                PushField::BicubicTexels(SlotRef {
                    stage,
                    class: SlotClass::Texture,
                    index,
                }) => {
                    let of = table(stage)?;
                    fields.push((offset, of, FieldValue::BicubicTexels(index)));
                    bicubic.push((of, index));
                    continue;
                }
                // The bound texture's level count, and the first element of the run that holds
                // its levels, the words a dispatch writes for its kernel's slot.
                PushField::TextureLevels(SlotRef {
                    stage,
                    class: SlotClass::Texture,
                    index,
                }) => {
                    fields.push((offset, table(stage)?, FieldValue::TextureLevels(index)));
                    continue;
                }
                PushField::TextureLevelBase(SlotRef {
                    stage,
                    class: SlotClass::Texture,
                    index,
                }) => {
                    let of = table(stage)?;
                    fields.push((offset, of, FieldValue::TextureLevelBase(index)));
                    levels.push((of, index));
                    continue;
                }
                PushField::ConstexprSampler(sampler) => {
                    let SamplerRef::Constexpr(words) = sampler.sampler else {
                        return Err(draw_input());
                    };
                    let first = firsts[at];
                    let place = (numbered[first..].iter())
                        .position(|&n| n == NumberedSampler::Constexpr(words))
                        .ok_or_else(draw_input)?
                        + first;
                    if sampler.unnormalized {
                        return Err(draw_input());
                    }
                    constexprs.push((place, sampler));
                    FieldValue::ConstexprSampler(place)
                }
                // A pulled layout's stride, the descriptor's: one the content fixes. A stride set
                // per bind (`MTLBufferLayoutStrideDynamic`) is a push field the encoder does not
                // fill, nor a command's token.
                PushField::VertexBufferStride(layout) => {
                    let stride = (pipeline.vertex_input.pulled.iter())
                        .find(|pulled| pulled.buffer_index == layout)
                        .map(|pulled| pulled.stride);
                    match stride {
                        Some(VertexStride::Static(stride)) => FieldValue::Word(stride),
                        _ => return Err(draw_input()),
                    }
                }
                PushField::SlotBounds(SlotRef {
                    stage,
                    class: SlotClass::Buffer,
                    index,
                }) => {
                    fields.push((offset, table(stage)?, FieldValue::BufferBounds(index)));
                    continue;
                }
                PushField::ZeroAddress => FieldValue::ZeroAddress,
                PushField::TessellationFactorAddress => FieldValue::TessellationFactors,
                // The encoder's reference, which a fragment function under the private alpha test
                // compares its color 0's alpha with.
                PushField::AlphaTestReference => FieldValue::AlphaTestReference,
                // The element a fragment function reads a colour attachment through, which the
                // draw pushes for the pass's attachment there.
                PushField::FetchedColor { color, .. } if at == FRAGMENT => {
                    fetched.push(color);
                    FieldValue::FetchedColor(color)
                }
                // The pass's length, below which the vertex function's index selects its layer.
                PushField::RenderTargetArrayLength if at == VERTEX => {
                    FieldValue::RenderTargetArrayLength
                }
                // The encoder's viewport count, below which the vertex function's index selects
                // its viewport.
                PushField::ViewportCount if at == VERTEX => FieldValue::ViewportCount,
                // The execution range and the command a vertex function of a state made for
                // indirect command buffers reads (`crate::indirect`).
                PushField::IndirectRange if at == VERTEX => FieldValue::IndirectRange,
                PushField::IndirectSequence if at == VERTEX => FieldValue::IndirectSequence,
                PushField::IndirectBase if at == VERTEX => FieldValue::IndirectBase,
                // The bit a fragment function whose stencil output is drawn by bits keeps its
                // fragments for: every fragment, but in a draw's bit draws, which write their own
                // bit over it. One in the parameter block is not supplied.
                PushField::StencilBit if at == FRAGMENT => match offset {
                    DrawPlace::Push(pushed) => {
                        stencil_bit = Some(pushed);
                        FieldValue::Word(EVERY_FRAGMENT)
                    }
                    DrawPlace::Block(..) => return Err(draw_input()),
                },
                _ => return Err(draw_input()),
            };
            fields.push((offset, at, value));
        }
        // The block, where there is one, is in one ring chunk, and its address in the range. A
        // placement with none leaves its table's: the address form's control stage states its
        // fields as a second placement of the vertex stage's, after the vertex function's.
        match (placement.argument_table_size, address) {
            (0, None) => {}
            (len @ 1.., Some(address)) if u64::from(len) <= CHUNK && blocks[at].is_none() => {
                blocks[at] = Some((len, address))
            }
            _ => return Err(draw_input()),
        }
    }
    if push_len > window {
        return Err(draw_input());
    }
    let bindings = (pipeline.vertex_input.bindings.iter())
        .map(|binding| (binding.binding, binding.buffer_index))
        .collect();
    let divided = (pipeline.vertex_input.bindings.iter()).any(|binding| binding.divisor != 1);
    // Only a fragment stage states them.
    let mut own_pixel: Vec<u32> = (pipeline.stages.iter())
        .flat_map(|stage| stage.own_pixel_textures.iter().copied())
        .collect();
    own_pixel.sort_unstable();
    own_pixel.dedup();
    fetched.sort_unstable();
    fetched.dedup();
    let draws = Draws {
        buffers,
        textures,
        storage_reads,
        formatless,
        own_pixel,
        fetched,
        bicubic,
        levels,
        bindings,
        reads_unbound: translated.reads_unbound_resources,
        writes_unbound: translated.writes_unbound_resources,
        unbound_storage: translated.unbound_storage_textures,
        writes_memory: translated.writes_memory,
        push_len,
        fields,
        blocks,
        first_instance_zero: divided && !nonzero_first_instance,
        patches: pipeline.graphics.tessellation.is_some(),
        selects_viewport: pipeline.graphics.writes_viewport_index,
        stencil_bit,
    };
    Ok((draws, (numbered, constexprs)))
}

/// What the descriptor asks that the runtime does not build. Its tessellation state is not
/// among them: Metal reads it only for a post-tessellation vertex function, and ignores it for any
/// other, as the translator does; the translation states such a function (`unbuilt`).
fn refuse(d: &RenderPipelineDescriptor) -> Result<(), Error> {
    let kind = if !(d.raster_sample_count.is_power_of_two() && d.raster_sample_count <= 64) {
        RenderPipelineKind::SampleCount
    } else if d.max_vertex_amplification_count != 1 {
        RenderPipelineKind::VertexAmplification
    } else {
        return Ok(());
    };
    Err(Error::Unsupported(Gap::RenderPipeline { kind }))
}

/// What the translator's pipeline states that the runtime does not build, where it states one;
/// `enabled` is what device creation enabled.
fn unbuilt(pipeline: &RenderPipeline, enabled: &EnabledCaps) -> Option<RenderPipelineKind> {
    use metal2vulkan::vk::ShaderStage;
    let g = &pipeline.graphics;
    let requirements = &pipeline.requirements;
    // A tessellation state the control stage reads beside the factors' address: the instance
    // stride of a per-instance step function, or the scale.
    let pushes =
        |field| (pipeline.interface.push.iter()).any(|p| p.fields.iter().any(|f| f.field == field));
    let tessellated = pushes(PushField::TessellationFactorInstanceStride)
        || pushes(PushField::TessellationFactorScale);
    // A post-tessellation function's pipeline: its synthesized vertex and control stages, and the
    // function's evaluation stage.
    let stages: &[ShaderStage] = match g.tessellation {
        Some(_) => &[
            ShaderStage::VERTEX,
            ShaderStage::TESSELLATION_CONTROL,
            ShaderStage::TESSELLATION_EVALUATION,
            ShaderStage::FRAGMENT,
        ],
        None => &[ShaderStage::VERTEX, ShaderStage::FRAGMENT],
    };
    let synthesized = g.line_band.is_some()
        || (pipeline.stages.iter()).any(|stage| !stages.contains(&stage.stage));
    let unenabled = !(requirements.features).is_subset(&enabled.features)
        || !(requirements.extensions).is_subset(&enabled.extensions);
    let separate = matches!(
        (g.depth_format, g.stencil_format),
        (Some(depth), Some(stencil)) if depth != stencil
    );
    Some(if separate {
        RenderPipelineKind::DepthStencilFormats
    } else if tessellated {
        RenderPipelineKind::Tessellation
    } else if synthesized {
        RenderPipelineKind::SynthesizedStages
    } else if !(g.dynamic_states.iter()).all(|state| DYNAMIC.contains(&dynamic(*state))) {
        RenderPipelineKind::DynamicState
    } else if unenabled {
        RenderPipelineKind::DeviceFeature
    } else {
        return None;
    })
}

/// The ash form of a translator dynamic state, as the pipeline holds it: the viewport and the
/// scissor are the with-count states, which set them with their count, as a pipeline holds no
/// state in both forms (VUID-VkGraphicsPipelineCreateInfo-pDynamicStates-04132 and -04133).
fn dynamic(state: metal2vulkan::vk::DynamicState) -> vk::DynamicState {
    match vk::DynamicState::from_raw(state.as_raw()) {
        vk::DynamicState::VIEWPORT => vk::DynamicState::VIEWPORT_WITH_COUNT,
        vk::DynamicState::SCISSOR => vk::DynamicState::SCISSOR_WITH_COUNT,
        other => other,
    }
}

impl Fixed {
    /// The fixed-function state the translator states for `pipeline`.
    fn of(pipeline: &RenderPipeline) -> Fixed {
        let g = &pipeline.graphics;
        let bindings = (pipeline.vertex_input.bindings.iter())
            .map(|binding| vk::VertexInputBindingDescription {
                binding: binding.binding,
                // A stride set per bind is dynamic state (`VERTEX_INPUT_BINDING_STRIDE`).
                stride: match binding.stride {
                    VertexStride::Static(stride) => stride,
                    VertexStride::PerBind => 0,
                },
                input_rate: vk::VertexInputRate::from_raw(binding.input_rate.as_raw()),
            })
            .collect();
        let divisors = (pipeline.vertex_input.bindings.iter())
            .filter(|binding| binding.divisor != 1)
            .map(|binding| vk::VertexInputBindingDivisorDescriptionKHR {
                binding: binding.binding,
                divisor: binding.divisor,
            })
            .collect();
        let attributes = (pipeline.vertex_input.attributes.iter())
            .map(|attribute| vk::VertexInputAttributeDescription {
                location: attribute.location,
                binding: attribute.binding,
                format: vk::Format::from_raw(attribute.format.as_raw()),
                offset: attribute.offset,
            })
            .collect();
        let factor =
            |factor: metal2vulkan::vk::BlendFactor| vk::BlendFactor::from_raw(factor.as_raw());
        let op = |op: metal2vulkan::vk::BlendOp| vk::BlendOp::from_raw(op.as_raw());
        let blends = (g.color_blend_attachments().into_iter())
            .map(|(mask, blend)| {
                let state = vk::PipelineColorBlendAttachmentState::default()
                    .color_write_mask(vk::ColorComponentFlags::from_raw(mask.as_raw()));
                match blend {
                    Some(blend) => state
                        .blend_enable(true)
                        .src_color_blend_factor(factor(blend.src_color))
                        .dst_color_blend_factor(factor(blend.dst_color))
                        .color_blend_op(op(blend.color_op))
                        .src_alpha_blend_factor(factor(blend.src_alpha))
                        .dst_alpha_blend_factor(factor(blend.dst_alpha))
                        .alpha_blend_op(op(blend.alpha_op)),
                    None => state,
                }
            })
            .collect();
        let colors = (g.color_attachments.iter())
            .map(|attachment| {
                attachment
                    .as_ref()
                    .map_or(vk::Format::UNDEFINED, |attachment| {
                        vk::Format::from_raw(attachment.format.as_raw())
                    })
            })
            .collect();
        let format = |format: Option<metal2vulkan::vk::Format>| {
            format.map_or(vk::Format::UNDEFINED, |format| {
                vk::Format::from_raw(format.as_raw())
            })
        };
        Fixed {
            bindings,
            divisors,
            attributes,
            patch_control_points: g.tessellation.map_or(0, |t| t.patch_control_points),
            discard: g.rasterizer_discard,
            samples: vk::SampleCountFlags::from_raw(g.rasterization_samples.as_raw()),
            alpha_to_coverage: g.alpha_to_coverage,
            alpha_to_one: g.alpha_to_one,
            sample_mask: g.sample_mask,
            blends,
            logic_op: (g.logic_op).map(|op| vk::LogicOp::from_raw(op.as_raw())),
            colors,
            depth: format(g.depth_format),
            stencil: format(g.stencil_format),
        }
    }
}

/// `descriptor` with each `Function` stage lowered under `keys` through the cache directory
/// ([`lower`]), and the keys of the lowerings, which the content's record keeps
/// ([`cache::keep_record`]).
fn lower_stages(
    core: &Core,
    descriptor: &RenderPipelineDescriptor,
    keys: &RenderPipelineKeys,
    budget: &Budget,
    entries: &StageEntries,
) -> Result<(RenderPipelineDescriptor, Lowerings), Error> {
    let mut lowered = descriptor.clone();
    let mut lowerings = Lowerings {
        vertex: None,
        fragment: None,
    };
    if let (StageSource::Function(function), Some(inputs)) =
        (&descriptor.vertex, &keys.vertex_inputs)
    {
        let function = lower(core, function, inputs, budget, entries);
        let function = function.map_err(Error::translation)?;
        lowerings.vertex = Some(*function.key().digest());
        lowered.vertex = StageSource::Compiled(function);
    }
    // The keys' fragment inputs start its push fields at the vertex function's end, in the
    // floor's one range.
    if let (Some(StageSource::Function(function)), Some(inputs)) =
        (&descriptor.fragment, &keys.fragment_inputs)
    {
        let function = lower(core, function, inputs, budget, entries);
        let function = function.map_err(Error::translation)?;
        lowerings.fragment = Some(*function.key().digest());
        lowered.fragment = Some(StageSource::Compiled(function));
    }
    Ok((lowered, lowerings))
}

/// `descriptor` with each `Function` stage the lowering its content's record names, where the
/// cache directory holds the record of `content`, the descriptor's key, and every lowering it
/// names (the key holds each stage's kind, so the record names a lowering for each `Function`
/// stage the compile lowered, as [`lower_stages`] kept them): read with no decode or
/// specialization of a function, counted
/// ([`Metrics::render_records_read`](crate::Metrics::render_records_read), each lowering in
/// `translations_loaded`). None where it holds no record, and the content takes the path of a
/// content with no record; a record that names a lowering the directory no longer holds is
/// counted ([`Metrics::render_records_stale`](crate::Metrics::render_records_stale)).
fn recorded(
    core: &Core,
    descriptor: &RenderPipelineDescriptor,
    content: &Hash256,
) -> Option<RenderPipelineDescriptor> {
    let lowerings = core.caches.record(content)?;
    let read = |key: Option<Hash256>| key.map(|key| loaded(core, &key));
    let (vertex, fragment) = (read(lowerings.vertex), read(lowerings.fragment));
    if matches!(vertex, Some(None)) || matches!(fragment, Some(None)) {
        core.counters.count(Metric::RenderRecordsStale);
        return None;
    }
    let mut lowered = descriptor.clone();
    if let Some(Some(vertex)) = vertex {
        lowered.vertex = StageSource::Compiled(vertex);
        core.counters.count(Metric::TranslationsLoaded);
    }
    if let Some(Some(fragment)) = fragment {
        lowered.fragment = Some(StageSource::Compiled(fragment));
        core.counters.count(Metric::TranslationsLoaded);
    }
    core.counters.count(Metric::RenderRecordsRead);
    Some(lowered)
}

impl RenderCompiled {
    /// The translation of `lowered`, a descriptor whose `Function` stages are lowered
    /// ([`lower_stages`], [`recorded`]), with its shader modules and no `VkPipeline` made.
    fn of(
        core: &Arc<Core>,
        lowered: &RenderPipelineDescriptor,
        budget: &Budget,
    ) -> Result<RenderCompiled, Error> {
        let translator = &core.translator;
        let pipeline = translating(core, || translator.compile_render_pipeline(lowered, budget))
            .map_err(Error::translation)?;
        // Metal makes no pipeline of a vertex function that writes `[[point_size]]` under the line
        // or the triangle class.
        let class = lowered.input_primitive_topology;
        if pipeline.graphics.writes_point_size
            && matches!(
                class,
                PrimitiveTopologyClass::Line | PrimitiveTopologyClass::Triangle
            )
        {
            return Err(Error::Refused(Refusal::PointSizeTopology { class }));
        }
        if let Some(kind) = unbuilt(&pipeline, translator.profile().enabled()) {
            return Err(Error::Unsupported(Gap::RenderPipeline { kind }));
        }
        // No helper compiles a render stage: one past the kernels' bound compiles here, counted,
        // and one past the render stages' own bound is refused.
        let heaviest = (pipeline.stages.iter())
            .map(|stage| stage.stats.entry_weight)
            .max();
        if let Some(weight) = heaviest.filter(|&weight| weight > crate::stage::bound()) {
            core.counters.render_past_kernel_bound(weight);
        }
        if let Some(weight) = heaviest.filter(|&weight| weight > crate::stage::RENDER_STAGE_WEIGHT)
        {
            return Err(Error::Unsupported(Gap::StageSize {
                weight,
                cause: StageGap::RenderStage,
            }));
        }
        let nonzero_first_instance = translator
            .profile()
            .limits()
            .supports_non_zero_first_instance;
        let window = translator.profile().push_window();
        let (draws, (numbered, constexprs)) = draws(&pipeline, nonzero_first_instance, window)?;
        let samples = lowered.raster_sample_count;
        let mut stages = Vec::new();
        let mut notes: Vec<_> = constexpr_notes(translator.profile(), &constexprs).collect();
        for stage in &pipeline.stages {
            let mut values = Vec::new();
            for constant in &stage.specialization.constants {
                let value: u32 = match (constant.key, constant.size) {
                    (SpecKey::SampleCount, 4) => samples,
                    (SpecKey::AmplificationCount, 4) => 1,
                    _ => {
                        release(core, &stages);
                        return Err(Error::Unsupported(Gap::RenderPipeline {
                            kind: RenderPipelineKind::Specialization,
                        }));
                    }
                };
                values.push((constant.id, u64::from(value)));
            }
            // The stage's constants fixed at the content's values, so the driver is handed no
            // specialization, as a kernel's module is.
            let words = match specialize(&stage.spirv, &values) {
                Ok(words) => words,
                Err(error) => {
                    release(core, &stages);
                    return Err(Error::translation(error));
                }
            };
            // The translator's entry points hold no NUL; one that did would name no entry point.
            let Ok(entry) = CString::new(stage.entry_point.as_str()) else {
                release(core, &stages);
                return Err(Error::Unsupported(Gap::RenderPipeline {
                    kind: RenderPipelineKind::SynthesizedStages,
                }));
            };
            let flags = vk::PipelineShaderStageCreateFlags::from_raw(stage.stage_flags.as_raw());
            let kind = vk::ShaderStageFlags::from_raw(stage.stage.as_raw());
            let info = vk::ShaderModuleCreateInfo::default().code(&words);
            // SAFETY: the words are the translator's SPIR-V with its constants fixed, valid where
            // the translator's are, for the features device creation enabled.
            let module = match unsafe { core.device.create_shader_module(&info, None) } {
                Ok(module) => module,
                Err(result) => {
                    release(core, &stages);
                    return Err(Error::Vulkan {
                        call: "vkCreateShaderModule",
                        result,
                    });
                }
            };
            stages.push(Module {
                module,
                entry,
                stage: kind,
                flags,
                subgroup_size: stage.required_subgroup_size,
            });
            notes.extend(stage.precision_notes.iter().copied());
        }
        notes.sort_unstable();
        notes.dedup();
        core.counters.rise(Gauge::RenderContents);
        // A state whose fragment function reaches an imageblock is not made (`unbuilt`), so its
        // sample is its colors alone.
        let colors = (lowered.color_attachments).map(|color| color.map(|c| c.pixel_format));
        let imageblock_sample = metal2vulkan::metal::format::imageblock_sample_length(&colors);
        // A descriptor that names the line or the point class (`inputPrimitiveTopology`) gives
        // that class's key: its `VkPipeline` is a main one too, made as the primary class's is.
        let named = match (draws.patches, lowered.input_primitive_topology) {
            (false, PrimitiveTopologyClass::Line) => Some(vk::PrimitiveTopology::LINE_LIST),
            (false, PrimitiveTopologyClass::Point) => Some(vk::PrimitiveTopology::POINT_LIST),
            _ => None,
        };
        Ok(RenderCompiled {
            core: Arc::clone(core),
            stages,
            fixed: Fixed::of(&pipeline),
            flags: match core.feedback_loop {
                true => vk::PipelineCreateFlags::COLOR_ATTACHMENT_FEEDBACK_LOOP_EXT,
                false => vk::PipelineCreateFlags::empty(),
            },
            primary: Main::default(),
            named: named.map(|list| (list, Main::default())),
            later: Mutex::new(Vec::new()),
            rasterizes_points: pipeline.graphics.writes_point_size
                || lowered.input_primitive_topology == PrimitiveTopologyClass::Point,
            notes,
            digest: pipeline.content_digest(),
            draws,
            numbered,
            constexprs,
            imageblock_sample,
            samples,
            indirect: lowered.support_indirect_command_buffers,
            id: CONTENT_IDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        })
    }

    /// Makes the content's main `VkPipeline`s on the calling thread where they are neither made
    /// nor being made ([`RenderPipelineState::finish_pipeline`]): the primary class's, then the
    /// named class's, each counted as made, both tried where the first fails. Err: the first
    /// make's error, the call counted as failed.
    fn finish(&self) -> Result<(), Error> {
        let primary = (
            self.class(vk::PrimitiveTopology::TRIANGLE_LIST),
            &self.primary,
        );
        let named = (self.named.as_ref()).map(|(list, named)| (*list, named));
        let mut failed = None;
        for (class, main) in std::iter::once(primary).chain(named) {
            match main.make_ahead(|| self.make(class, Raster::default())) {
                Ok(Some(nanos)) => self.core.counters.finished(nanos),
                Ok(None) => {}
                Err(error) => failed = failed.or(Some(error)),
            }
        }
        match failed {
            Some(error) => {
                self.core.counters.finish_failed();
                Err(error)
            }
            None => Ok(()),
        }
    }

    /// The list topology of `topology`'s class, which its `VkPipeline` is made with: a
    /// post-tessellation vertex function's content draws the patch class alone.
    fn class(&self, topology: vk::PrimitiveTopology) -> vk::PrimitiveTopology {
        use vk::PrimitiveTopology as T;
        match (self.draws.patches, topology) {
            (true, _) => T::PATCH_LIST,
            (false, T::LINE_LIST | T::LINE_STRIP) => T::LINE_LIST,
            (false, T::POINT_LIST) => T::POINT_LIST,
            (false, _) => T::TRIANGLE_LIST,
        }
    }

    /// Whether a draw of points rasterizes them (`rasterizes_points`): else the encoder draws them
    /// behind an empty scissor, its vertex function run.
    pub(crate) fn rasterizes_points(&self) -> bool {
        self.rasterizes_points
    }

    /// Whether the content's `VkPipeline`s discard every primitive (Metal's rasterization
    /// disabled): a dynamic state each bind sets.
    pub(crate) fn discards(&self) -> bool {
        self.fixed.discard
    }

    /// The `VkPipeline` of the content for draws of `topology`'s class, a point, line or triangle
    /// one, or for a post-tessellation vertex function's content the patch one, which the encoder
    /// draws alone with it, under `raster`, whose fill mode a line or point class does not take:
    /// one not made yet is made by its first draw, which waits for it, or for the make in flight
    /// of a main one, a first-use wait ([`Metrics::first_use_waits`](crate::Metrics::first_use_waits)),
    /// counted as a main pipeline's before its `finish_pipeline` made it, as the line or point
    /// class's at the default raster state, and as a raster state's otherwise.
    ///
    /// Err: [`Error::Vulkan`] where the pipeline cannot be made.
    pub(crate) fn pipeline(
        &self,
        topology: vk::PrimitiveTopology,
        raster: Raster,
    ) -> Result<vk::Pipeline, Error> {
        use vk::PrimitiveTopology as T;
        let class = self.class(topology);
        let polygons = matches!(class, T::TRIANGLE_LIST | T::PATCH_LIST);
        let raster = Raster {
            clamp: raster.clamp,
            lines: raster.lines && polygons,
        };
        let counters = &self.core.counters;
        if polygons && raster == Raster::default() {
            return (self.primary).pipeline(counters, || self.make(class, raster));
        }
        if let Some((list, named)) = &self.named {
            if (*list, raster) == (class, Raster::default()) {
                return named.pipeline(counters, || self.make(class, raster));
            }
        }
        let mut later = lock(&self.later);
        if let Some(&(.., made)) = (later.iter()).find(|&&(c, r, _)| (c, r) == (class, raster)) {
            return Ok(made);
        }
        let began = host_time();
        let made = self.make(class, raster)?;
        later.push((class, raster, made));
        let site = match (class, raster == Raster::default()) {
            (T::LINE_LIST, true) => FirstUse::Lines,
            (T::POINT_LIST, true) => FirstUse::Points,
            _ => FirstUse::Raster,
        };
        (self.core.counters).first_use_wait(site, host_time().saturating_sub(began));
        Ok(made)
    }

    /// Whether the content's `VkPipeline`s render into `pass`'s attachments: its colour formats are
    /// the pass's colour attachments', by index, an absent one where the pass has none, its
    /// depth and stencil formats the pass's depth and stencil attachments', or none without one,
    /// and its raster sample count the attachments'.
    pub(crate) fn renders_into(&self, pass: &Pass) -> bool {
        let fixed = &self.fixed;
        let format = |t: Option<&crate::pass::Target>| {
            t.map_or(vk::Format::UNDEFINED, |t| t.attachment.format)
        };
        fixed.colors.len() == pass.colors.len()
            && (fixed.colors.iter().zip(&pass.colors)).all(|(&f, t)| f == format(t.as_ref()))
            && fixed.depth == format(pass.depth.as_ref())
            && fixed.stencil == format(pass.stencil.as_ref())
            && fixed.samples == pass.samples
    }

    /// A `VkPipeline` of the content for draws of the class of `topology`, a list topology, under
    /// `raster`, made through the device's pipeline data, rendering dynamically into the
    /// descriptor's attachment formats. A draw sets its own topology within the class as dynamic
    /// state. A patch list takes the translator's control points per patch.
    fn make(&self, topology: vk::PrimitiveTopology, raster: Raster) -> Result<vk::Pipeline, Error> {
        let core = &self.core;
        let fixed = &self.fixed;
        // A stage at the subgroup size the translator pins, as a kernel's is: one it pins none for
        // runs at the device's own size.
        let mut subgroups = subgroup_sizes(&self.stages);
        let stages = stage_infos(&self.stages, &mut subgroups);
        let mut divided = vk::PipelineVertexInputDivisorStateCreateInfoKHR::default()
            .vertex_binding_divisors(&fixed.divisors);
        let mut vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&fixed.bindings)
            .vertex_attribute_descriptions(&fixed.attributes);
        // The divisor structure has at least one divisor where it is chained.
        if !fixed.divisors.is_empty() {
            vertex_input = vertex_input.push_next(&mut divided);
        }
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(topology);
        let tessellation = vk::PipelineTessellationStateCreateInfo::default()
            .patch_control_points(fixed.patch_control_points);
        // The draw sets its viewports and scissors with their count (`VIEWPORT_WITH_COUNT`).
        let viewport = vk::PipelineViewportStateCreateInfo::default();
        // `Clamp` keeps what lies past the near and far planes and clamps its depth to the
        // viewport's range, as Vulkan's depth clamp does (`depthClamp`, Floor); `Lines` draws each
        // triangle's edges as lines 1 pixel wide (`fillModeNonSolid`, which the encoder asks of
        // the device before it sets the mode).
        let polygon_mode = match raster.lines {
            true => vk::PolygonMode::LINE,
            false => vk::PolygonMode::FILL,
        };
        let mut rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(raster.clamp)
            .polygon_mode(polygon_mode)
            .rasterizer_discard_enable(fixed.discard)
            .line_width(1.0);
        // Lines, of the line class or a triangle's edges under `Lines`, draw by Bresenham's rule,
        // Metal's (`bresenhamLines`, Floor), except under alpha to coverage or alpha to one, which
        // Vulkan's Bresenham mode does not take
        // (`VUID-VkGraphicsPipelineCreateInfo-lineRasterizationMode-02766`; the runtime enables no
        // sample shading, the third it names): such a state draws its lines by the device's
        // default rule (`lines_by_device_rule`). Either way a draw of lines records
        // `RenderCommandEncoder::LINE_NOTE`.
        let mut line = vk::PipelineRasterizationLineStateCreateInfoEXT::default()
            .line_rasterization_mode(vk::LineRasterizationModeEXT::BRESENHAM);
        let draws_lines = topology == vk::PrimitiveTopology::LINE_LIST || raster.lines;
        if draws_lines && !lines_by_device_rule(fixed) {
            rasterization = rasterization.push_next(&mut line);
        }
        // The private sample mask, which the translator states (`GraphicsState::sample_mask`)
        // where Vulkan's mask keeps the samples Metal's keeps. Its one word is the first 32
        // samples'; Vulkan's
        // `pSampleMask` "has a length equal to ⌈rasterizationSamples / 32⌉ words", and a second
        // keeps every sample past them.
        let words = fixed.sample_mask.map(|mask| [mask, u32::MAX]);
        let count = fixed.samples.as_raw().div_ceil(32) as usize;
        let mut multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(fixed.samples)
            .alpha_to_coverage_enable(fixed.alpha_to_coverage)
            .alpha_to_one_enable(fixed.alpha_to_one);
        if let Some(words) = &words {
            multisample = multisample.sample_mask(&words[..count.min(words.len())]);
        }
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default();
        let mut blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&fixed.blends);
        if let Some(op) = fixed.logic_op {
            blend = blend.logic_op_enable(true).logic_op(op);
        }
        // A state without rasterization takes its discard as dynamic state, which the encoder sets
        // at each bind: a static discard would leave the fragment shader and output state out of
        // the pipeline, its attachment formats and the dynamic depth, stencil and blend state above
        // with them, which every draw records and its pass renders into.
        let states: Vec<vk::DynamicState> = (DYNAMIC.iter().copied())
            .chain(
                fixed
                    .discard
                    .then_some(vk::DynamicState::RASTERIZER_DISCARD_ENABLE),
            )
            .collect();
        let dynamic_states = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&states);
        let mut rendering = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&fixed.colors)
            .depth_attachment_format(fixed.depth)
            .stencil_attachment_format(fixed.stencil);
        let mut info = vk::GraphicsPipelineCreateInfo::default()
            .flags(self.flags)
            .push_next(&mut rendering)
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&blend)
            .dynamic_state(&dynamic_states)
            .layout(core.layout);
        if topology == vk::PrimitiveTopology::PATCH_LIST {
            info = info.tessellation_state(&tessellation);
        }
        made_through_cache(core, "vkCreateGraphicsPipelines", |cache| {
            // SAFETY: the modules, the layout and the cache are the device's; each module holds
            // its constants fixed; the formats are the attachments' the translator maps for the
            // device, for the features device creation enabled.
            unsafe { core.device.create_graphics_pipelines(cache, &[info], None) }
        })
    }
}

/// Gives `stages`' modules to the reaper: a creation that failed after making them.
fn release(core: &Core, stages: &[Module]) {
    for stage in stages {
        core.reaper
            .give(Job::Free(Free::ShaderModule(stage.module)));
    }
}

impl RenderPipelineState {
    /// No Metal counterpart: makes the state's main `VkPipeline`s on the calling thread, through
    /// the device's pipeline cache, once the state's creation has answered. Metal compiles a
    /// render state when the state is made. Here the creation does everything up to the driver's
    /// create (the lowerings, every refusal and Metal's own errors, the shader modules and the
    /// samplers' numbering) and answers, and this call runs the driver's create, so a consumer
    /// can answer the program before the driver's compile ends and let that compile overlap the
    /// program's own work. The runtime starts no thread for it: the consumer calls it on a thread
    /// of its own after it has answered.
    ///
    /// The main pipelines are the triangle class's (a post-tessellation vertex function's patch
    /// class's, its only class) and, where the descriptor's `inputPrimitiveTopology` names the
    /// line or the point class, that class's, each at the default depth clip and fill modes. Each
    /// is made where it is neither made nor being made, so a second call makes nothing. A draw
    /// that needs one first waits for the make in flight, or makes it itself where none is: a
    /// first-use wait
    /// ([`Metrics::first_use_unfinished_waits`](crate::Metrics::first_use_unfinished_waits)),
    /// after which it records what it records with the pipeline made here. So a state this is
    /// never called for draws as one it was called for, its first draw making the pipeline.
    ///
    /// Counted as [`Metrics::finished_pipelines`](crate::Metrics::finished_pipelines), with their
    /// time.
    ///
    /// Err: [`Error::Vulkan`] where a driver's create failed, which Metal would have answered at
    /// the creation (Vulkan lets it fail for exhausted host or device memory, or with
    /// `VK_ERROR_UNKNOWN` or `VK_ERROR_VALIDATION_FAILED`), counted in
    /// [`Metrics::failed_finishes`](crate::Metrics::failed_finishes), the other main pipeline
    /// still made: the failed one is not kept, the first draw that needs it makes it again, and
    /// that draw answers the driver's error where that make fails too.
    pub fn finish_pipeline(&self) -> Result<(), Error> {
        self.inner.compiled.finish()
    }

    /// No Metal counterpart: the translator's notes of where this pipeline's answers may differ
    /// from Metal's, of its modules ([`PrecisionNote`]), and
    /// [`PrecisionNote::SwizzledOpaqueBlackBorder`] where a function samples through a constexpr
    /// sampler whose address mode reads an opaque-black border (decided by the sampler's own
    /// constants, as [`SamplerState::precision_note`](crate::SamplerState::precision_note) is by
    /// its descriptor); empty where they match. A draw's lines carry their own note
    /// ([`RenderCommandEncoder::LINE_NOTE`](crate::RenderCommandEncoder::LINE_NOTE)).
    pub fn precision_notes(&self) -> Vec<PrecisionNote> {
        self.inner.compiled.notes.clone()
    }

    /// No Metal counterpart: the translator's content digest of the state's pipeline
    /// ([`RenderPipeline::content_digest`]). It covers each stage's module (its SPIR-V words and
    /// everything else the translator states of it), the layout, the vertex input, the
    /// fixed-function state, the interface, the runtime dependencies, the requirements and the
    /// variant bits, and no version of either crate: two builds that translate the state's content
    /// to the same pipeline answer one digest, and one SPIR-V word changed answers another. The
    /// states of one content answer one digest; the samplers' numbering, each state's own, is not
    /// in it. Computed once per content, when its translation is made.
    pub fn content_digest(&self) -> Hash256 {
        self.inner.compiled.digest
    }

    /// No Metal counterpart as such: each texture slot `stage`'s function uses
    /// ([`Stage::Vertex`] or [`Stage::Fragment`]; none for another stage), by index, ascending,
    /// with the access a draw of the state marks in its encoder's hazards for the texture bound
    /// there, read from the same interface, as
    /// [`ComputePipelineState::texture_accesses`](crate::ComputePipelineState::texture_accesses)
    /// answers a kernel's: what the function's operations do, not the argument's `access::`
    /// qualifier. Metal's counterpart is reflection's `MTLBinding.access` of a texture binding in
    /// `vertexBindings` or `fragmentBindings`, the declared access; the consumer builds
    /// reflection itself.
    pub fn texture_accesses(&self, stage: Stage) -> Vec<(u32, BindingAccess)> {
        let table = match stage {
            Stage::Vertex => VERTEX,
            Stage::Fragment => FRAGMENT,
            _ => return Vec::new(),
        };
        (self.inner.compiled.draws.textures.iter())
            .filter(|&&(at, _, _)| at == table)
            .map(|&(_, slot, access)| (slot, access.binding()))
            .collect()
    }

    /// `imageblockSampleLength`: the bytes of one sample of the state's imageblock, its color
    /// attachments' texels as Metal packs them, each set format's in index order at its
    /// component's alignment, depth and stencil excluded
    /// (`metal2vulkan::metal::format::imageblock_sample_length`, measured on an M3 Max and an M4).
    /// A state whose fragment function reaches an imageblock is not made
    /// ([`Gap::RenderPipeline`] of kind `Imageblock`), so no explicit imageblock adds to it.
    ///
    /// Err: [`Gap::ImageblockLength`] for a color format that has no measured place in Metal's
    /// record (no direct Vulkan format, or no whole-byte component or packed word).
    pub fn imageblock_sample_length(&self) -> Result<u32, Error> {
        (self.inner.compiled.imageblock_sample).ok_or(Error::Unsupported(Gap::ImageblockLength))
    }

    /// `imageblockMemoryLengthForDimensions:`: the sample length
    /// ([`RenderPipelineState::imageblock_sample_length`]) times the dimensions' width and height
    /// and the state's raster sample count, the depth ignored, none rounded (measured on an M3 Max
    /// and an M4).
    ///
    /// Err: as [`RenderPipelineState::imageblock_sample_length`].
    pub fn imageblock_memory_length_for_dimensions(
        &self,
        dimensions: [u32; 3],
    ) -> Result<u64, Error> {
        let [width, height, _] = dimensions.map(u64::from);
        let samples = u64::from(self.inner.compiled.samples);
        let sample = u64::from(self.imageblock_sample_length()?);
        Ok(sample
            .saturating_mul(width)
            .saturating_mul(height)
            .saturating_mul(samples))
    }

    /// `supportIndirectCommandBuffers`: the descriptor's. A state made with it pulls every vertex
    /// attribute its vertex function reads, and its vertex function tests its command against an
    /// execution's range, which a direct draw pushes as every command
    /// ([`RenderCommandEncoder::execute_commands_in_buffer`](crate::RenderCommandEncoder::execute_commands_in_buffer)).
    pub fn support_indirect_command_buffers(&self) -> bool {
        self.inner.compiled.indirect
    }

    /// Whether two handles are one state.
    pub(crate) fn is(&self, other: &RenderPipelineState) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Whether `core` is the device this state was made on.
    pub(crate) fn is_of(&self, core: &Arc<Core>) -> bool {
        Arc::ptr_eq(&self.inner.compiled.core, core)
    }
}

impl Drop for RenderInner {
    fn drop(&mut self) {
        self.numbered.release(&self.compiled.core);
    }
}

impl Drop for RenderCompiled {
    fn drop(&mut self) {
        // No state holds it, so no command buffer that used it is left: each kept its state until
        // it retired.
        self.core.counters.fall(Gauge::RenderContents);
        // Only what was made: a main pipeline no call or draw made, or whose make failed, is none.
        let named = (self.named.iter()).filter_map(|(_, named)| named.get());
        let later = self.later.get_mut().unwrap_or_else(|e| e.into_inner());
        let later = later.iter().map(|&(.., pipeline)| pipeline);
        for pipeline in (self.primary.get().into_iter()).chain(named).chain(later) {
            (self.core.reaper).give(Job::Free(Free::Pipeline(pipeline)));
        }
        release(&self.core, &self.stages);
    }
}

impl crate::kept::Uncount for RenderPipelineState {
    unsafe fn uncounted(&self) -> std::mem::ManuallyDrop<RenderPipelineState> {
        // SAFETY: the caller keeps the state alive past the copy's last use.
        let inner = unsafe { crate::kept::Uncount::uncounted(&self.inner) };
        std::mem::ManuallyDrop::new(RenderPipelineState {
            inner: std::mem::ManuallyDrop::into_inner(inner),
        })
    }
}

impl crate::kept::Handle for RenderPipelineState {
    /// The state itself: it holds no handle of the device.
    type Kept = RenderPipelineState;

    fn object(&self) -> &RenderPipelineState {
        self
    }

    fn address_of(object: &RenderPipelineState) -> usize {
        Arc::as_ptr(&object.inner) as *const () as usize
    }
}
