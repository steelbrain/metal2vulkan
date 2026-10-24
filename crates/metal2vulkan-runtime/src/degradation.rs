//! Degrade notices. Where the runtime gives Metal's answer at a cost Metal does not pay (a copy, a
//! staging or scratch resource, an extra pass), gives a recorded near-equivalent of it, or gives it
//! through use of the device its driver does not advertise, it says so once per device and kind
//! ([`Degradation`]): through the device's message sink when it has one
//! (`DeviceOptions::messages`), as a warning whose `id_name` is the kind's code and whose
//! [`DebugMessage::degradation`] carries the [`DegradeNotice`]; else on stderr, the same text.
//! Nothing queries them (the user, 2026-10-05: downgrades are allowed, but each must be visible at
//! least once and decodable, without log spam).
//!
//! - **One closed vocabulary.** [`Degradation`] lists every kind once, with its code, category
//!   and cost, in `degradations!`; its precision kinds include one per
//!   [`PrecisionNote`](metal2vulkan::pipeline::PrecisionNote), named alike, which the macro maps both
//!   ways, so a note without a kind (or two kinds of one note) does not compile. docs/API.md's
//!   catalogue lists every code, and every retired one, closed both ways by a test, and every
//!   code ever issued is in `tests/degradation_codes.txt`, live or retired, so a kind removed
//!   without retiring its code fails that test.
//! - **Codes are a contract.** `metal2vulkan-degrade-<kind>`, never renamed or reused; a new
//!   degradation is a new kind. A kind that no longer fires is retired before the API freezes:
//!   its code stays in [`Degradation::RETIRED_CODES`], with the notices that replace it (or, where
//!   the degradation is gone, why none does), so no kind takes it.
//! - **The payload.** The object degraded (the first one the device met), the reason (with that
//!   object's numbers), and the kind's fixed cost. A consumer decodes a notice by its
//!   [`Degradation`], never by parsing its text.
//! - **Once, unless the device's options ask for each.** A bit per kind and device, set by the
//!   first announcement; a later occurrence costs one atomic load. A kind in
//!   `DeviceOptions::announce_each` (or `RawDevice::announce_each`) is announced at every
//!   occurrence instead, the same notice through the same sink, on the thread whose call
//!   announces it: a bit per kind beside those, read first. For a kind a variant's making announces
//!   (a stage's precision notes, specialized or not, `ycbcr-filter-not-exact`, a shader's color
//!   conversions, reconvergence that is not Metal's, unadvertised vertex simdgroups, exact forms
//!   dropped to fit a compiler word bound), an occurrence
//!   is each dispatch or draw that binds the variant (an indirect command's too, where its
//!   execution records it, and an execution the GPU alone runs, once), announced there with the
//!   making's object and reason and not at the making, since the device memoizes a variant and
//!   makes it once. A variant keeps the kinds its making may announce as a mask (`Kinds`), which a
//!   bind ANDs with the device's before it describes any.
//! - **Where.** Each site announces once the command or object it degrades can no longer be
//!   refused. A pipeline stage's precision notes, and reconvergence that is not Metal's, are
//!   announced where the runtime makes a variant's objects (`pipeline/objects.rs`), for compute and
//!   render pipeline states alike, but for a note the stage records only under some
//!   specializations, announced where the runtime makes a `VkPipeline` whose specialization (the
//!   bound formats' classes and samplers' words) takes it; a render pipeline's vertex simdgroups on
//!   a driver that runs them without advertising the vertex stage, where the runtime makes a
//!   render variant, before its shader modules (`pipeline/render.rs`, from the pipeline's
//!   requirements); the exact forms a pipeline's translation dropped to fit the device's compiler
//!   word bound, where the runtime makes a variant of any pipeline state, before its shader modules
//!   (`pipeline/render.rs` for a render or mesh render state, `pipeline/mod.rs` for a compute or
//!   tile render state, from the pipeline's dependencies); a
//!   render pass's degradations where its encoder ends (`render.rs`), from the pass's layout, its
//!   render pass instances and the storage it kept; a rasterization-rate map that is not the
//!   identity where the device makes it (`rate_map.rs`); a draw's use of an attachment's memory in
//!   place where the draw records (`render.rs`); a draw's copy of its index buffer or arguments
//!   where the copy records (`draw_source.rs`), and its zeroed index positions where it records
//!   (`render.rs`, `index_bounds.rs`); work a host step records (an indirect command buffer's
//!   execution, an indirect dispatch) where its command buffer commits and where the step records
//!   it (`deferred.rs`); a `Shared` buffer or heap in system memory where the device makes it
//!   (`buffer.rs`, `heap.rs`); a texture whose blocks a carrier holds, which a command reading
//!   texture handles reaches, once the command is encoded (`pvrtc.rs`, from `compute.rs` and
//!   `render/arguments.rs`); a Metal level below the host's release where the device is made
//!   (`device/mod.rs`, from `metal_level.rs`); native device maths that rounds a tiny product
//!   before its flush where the device is made (`device/float_tininess.rs`); a sampler state whose
//!   anisotropy the device filters otherwise than Metal where the device makes it (`sampler.rs`); a
//!   BC1 to BC5 texture whose texels the device decodes otherwise than Metal where the device makes
//!   it (`texture.rs`, `texture/shared.rs`, `texture/heap.rs`).

use super::attachment_stand_in::{Cause, StandIn};
use super::color_resolve;
use super::device::{DebugMessage, Device, MessageKind, MessageSeverity};
use super::error::Error;
use super::fragment_imageblock::PassImageblock;
use super::pipeline::objects::{set_layout_support, SetSupport};
use super::render_pass::{ColorTarget, PassLayout, Target};
use super::render_target::{self, depth_stencil_steps, HandedDontCare, PassShape};
use super::render_values::{LoadAction, StoreAction};
use super::resolve::ResolveMode;
use super::resource::{ObjectId, ResourceKey};
use super::texture::heap_tiled::Untiled;
use super::texture_descriptor::TextureDescriptor;
use super::tracker::Use;
use super::values::ResourceOptions;
use crate::error::EncoderCall;
use metal2vulkan::bind::PlanDiagnostic;
use metal2vulkan::cache::SpecializationKey;
use metal2vulkan::device::{DriverFact, FunctionStorage, ResolvePolicy, VkProperty};
use metal2vulkan::metal::PixelFormat;
use metal2vulkan::pipeline::{
    ComputePipeline, PrecisionNote, Reconvergence, RenderPipeline, SlotRef, SpecKey, StageModule,
};
use metal2vulkan::vk::{ImageAspectFlags, ShaderStage};
use std::cell::Cell;
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicU64, Ordering};

/// The pixel format of the texture a draw or dispatch binds at a texture slot, where it binds
/// one: what a specialized notice names ([`Device::degrade_specialized`]).
pub(crate) type BoundFormats<'a> = &'a dyn Fn(SlotRef) -> Option<PixelFormat>;

/// The texture slots a clause of `clauses` tests by their bound format's class
/// (`SpecKey::TexelFormat`, or `SpecKey::UnmeasuredTexels` on a device that samples through
/// Metal's own texture unit, or the bound texture's own, `SpecKey::BoundTexelFormat`), each once,
/// in order.
fn tested_slots<'a>(
    clauses: impl IntoIterator<Item = &'a metal2vulkan::pipeline::SpecClause>,
) -> std::collections::BTreeSet<SlotRef> {
    clauses
        .into_iter()
        .flat_map(|clause| &clause.all)
        .filter_map(|test| match test.key {
            SpecKey::TexelFormat(slot)
            | SpecKey::UnmeasuredTexels(slot)
            | SpecKey::BoundTexelFormat(slot) => Some(slot),
            _ => None,
        })
        .collect()
}

/// The format a draw or dispatch binds at each texture slot a specialized note tests
/// (`VariantObjects::tested_formats`), kept past the bind's tables.
pub(crate) type TestedFormats = Vec<(SlotRef, Option<PixelFormat>)>;

/// The texture slots a specialized note of `stages` tests, each once, in order: those whose
/// formats [`Device::degrade_specialized`] may name.
pub(crate) fn tested_slots_of(stages: &[StageModule]) -> Vec<SlotRef> {
    tested_slots(
        stages
            .iter()
            .flat_map(|stage| &stage.specialized_notes)
            .flat_map(|note| &note.when),
    )
    .into_iter()
    .collect()
}

/// Whether a degradation costs time and memory, or exactness.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DegradationCategory {
    /// Metal's answer, at a cost Metal does not pay: copies, staging or scratch resources,
    /// extra passes or prepasses, a slower variant.
    Performance,
    /// A recorded near-equivalent: Metal's answer is not given exactly.
    Precision,
    /// Metal's answer, exactly, through use of the device that its driver does not advertise
    /// but runs as Metal does, measured (a `metal2vulkan::error::DriverDeviation`, or a
    /// `metal2vulkan::error::RuntimeDriverDefect` of a query that refuses what the driver runs):
    /// Vulkan does not allow the use on the device as the driver describes it, so the validation
    /// layer reports it.
    Validity,
}

/// The cost of every precision-note kind.
const NOTE_COST: &str = "results can differ from Metal's as the note's documentation states";

/// Defines [`Degradation`] from its kinds: the runtime's own (`code`, category, cost), then one
/// per precision note (named as the note, with its code).
macro_rules! degradations {
    (
        runtime {
            $(
                $(#[doc = $doc:literal])*
                $kind:ident = $code:literal, $category:ident, $cost:literal;
            )*
        }
        notes {
            $(
                $(#[doc = $note_doc:literal])*
                $note:ident = $note_code:literal;
            )*
        }
    ) => {
        /// A kind of degradation the runtime announces (the module docs). Its
        /// [`code`](Degradation::code) is a stable contract: never renamed or reused, and a
        /// retired kind's stays in [`Degradation::RETIRED_CODES`].
        #[non_exhaustive]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub enum Degradation {
            $($(#[doc = $doc])* $kind,)*
            $(
                #[doc = concat!(
                    "A pipeline stage records [`PrecisionNote::",
                    stringify!($note),
                    "`](metal2vulkan::pipeline::PrecisionNote::",
                    stringify!($note),
                    ")."
                )]
                $(#[doc = $note_doc])*
                $note,
            )*
        }

        impl Degradation {
            /// Every kind, in declaration order.
            pub const ALL: &'static [Degradation] =
                &[$(Degradation::$kind,)* $(Degradation::$note,)*];

            /// The kind's code, `metal2vulkan-degrade-<kind>`: the notice's `id_name`.
            pub const fn code(self) -> &'static str {
                match self {
                    $(Degradation::$kind => concat!("metal2vulkan-degrade-", $code),)*
                    $(Degradation::$note => concat!("metal2vulkan-degrade-", $note_code),)*
                }
            }

            /// Whether the kind costs time and memory, or exactness.
            pub const fn category(self) -> DegradationCategory {
                match self {
                    $(Degradation::$kind => DegradationCategory::$category,)*
                    $(Degradation::$note => DegradationCategory::Precision,)*
                }
            }

            /// What the kind costs, the same for every occurrence.
            pub const fn cost(self) -> &'static str {
                match self {
                    $(Degradation::$kind => $cost,)*
                    $(Degradation::$note => NOTE_COST,)*
                }
            }

            /// The precision note the kind announces, for a note's kind.
            pub const fn precision_note(self) -> Option<PrecisionNote> {
                match self {
                    $(Degradation::$kind => None,)*
                    $(Degradation::$note => Some(PrecisionNote::$note),)*
                }
            }

            /// The kind announcing `note`.
            pub const fn from_precision_note(note: PrecisionNote) -> Degradation {
                match note {
                    $(PrecisionNote::$note => Degradation::$note,)*
                }
            }
        }
    };
}

degradations! {
    runtime {
        /// A 2D texture over a buffer is a shadow image: the buffer is over host pages (a no-copy
        /// buffer's, imported by their pointer or as dma-bufs), where an image bound to them would
        /// start `UNDEFINED` (`VUID-VkImageCreateInfo-pNext-01443`) and Vulkan lets its layout
        /// transitions (its first move among them) read and write any byte bound to it, where
        /// Metal keeps every CPU write; or its rows (or an array's slices) lie with the buffer's
        /// bytes between them, which an image over them would hold in its subresource, where
        /// Vulkan lets the image's layout transitions read and write any byte of its subresources,
        /// and a render pass's load and store operations read and write back any byte of an
        /// attachment's image subresource, and Metal leaves them as the program writes them; or no
        /// linear image of the device lays its texels out as Metal does in the buffer's memory,
        /// holding no byte past them; or the buffer is over the caller's `VkBuffer`, whose memory
        /// backs no image of the runtime's. Also a heap's texture over host pages imported as
        /// dma-bufs where the device makes no image over a dma-buf of the texture's own pages (the
        /// notice names why). Also a heap's depth or stencil texture on a device whose driver lays
        /// such images apart from their memory
        /// ([`metal2vulkan::device::DriverFact::DepthStencilImagesLieApart`]): its image is a
        /// shadow of the heap's bytes in its range. Also a heap's 2D texture of one level of
        /// `RGB10A8_2P_XR10` or its sRGB twin: its image is a shadow of its two planes in the
        /// heap's bytes, where Metal lays them out. Also a texture of a plain format in a heap over
        /// host pages: its image is a shadow of its elements tiled in the heap's bytes, where Metal
        /// lays them out.
        ShadowedTexture = "shadowed-texture", Performance,
            "its texels are copied into an image before each use and back after each write \
             (over a buffer's bytes but a tracked private buffer's, kept before a write and merged \
             back after it)";
        /// A texture buffer is viewed through a staging buffer: its texels start where the
        /// device views none (`minTexelBufferOffsetAlignment`).
        StagedTextureBuffer = "staged-texture-buffer", Performance,
            "its texels are copied into a staging buffer before each use and back after each \
             write";
        /// One command writes a buffer's bytes through a shadowed texture or a staged texture
        /// buffer and also another way (`shadow_merge.rs`).
        ShadowMerge = "shadow-merge", Performance,
            "the shadow's texels are kept in a scratch buffer of one to three times the texture's \
             bytes (or an image of the texture's size) before the command, and after it a \
             dispatch merges the bytes it changed through the texture into the buffer's, in place \
             where the texture's bytes start and end in whole words of the buffer: one to three \
             copies and a dispatch in place of one copy back";
        /// A draw's index buffer or indirect arguments start where Vulkan reads neither.
        CopiedDrawBytes = "copied-draw-bytes", Performance,
            "they are copied into a buffer of the command buffer before the pass, and the draw \
             reads the copy";
        /// An indexed draw reads index positions past where Metal's fetch stops reading its
        /// index buffer (its length rounded up to 4 bytes), for which Vulkan states no index
        /// (`draw_source.rs`): a direct draw whose positions run past there, on every device; or an
        /// indirect draw that stays one draw (a driver measured to read zero past the bound range,
        /// `DriverFact::RobustIndexFetchReadsZero`) on a device that enables `robustBufferAccess2`
        /// but not `maintenance5`, whose bound range runs to the end of the index buffer's
        /// `VkBuffer`, on past the bytes into its padding or into a heap's bytes after a placed
        /// buffer.
        IsolatedIndexBytes = "isolated-index-bytes", Performance,
            "its whole indices inside the buffer are copied into a buffer of the command buffer \
             before the pass, between two barriers, and the draw reads the copy: a direct draw's \
             at its start, every position past them zero";
        /// An indexed indirect draw reads arguments or indices a draw of its encoder wrote, with
        /// no memory barrier in the encoder before it, a race Metal does not order
        /// (`draw_source.rs`); or a draw's vertex stage reads a buffer of its vertex table a
        /// fragment stage of its render pass wrote (`render/pass_start_reads.rs`).
        UnorderedDrawBytes = "unordered-draw-bytes", Performance,
            "they are copied into a buffer of the command buffer before the pass, between two \
             barriers, and the draw reads the bytes from before the write, which Metal's draw \
             read in every run measured";
        /// A draw biases a `Depth16Unorm` depth by a constant, or by a slope alone, which brings
        /// Metal's constant unit, on a device whose driver is not measured to store Metal's codes
        /// so (`DriverDeviation::MetalDepth16BiasedStores`, `render_state.rs`), through a
        /// pipeline whose fragment stage does not add the bias itself (more than one sample,
        /// early fragment tests, the draw's tests run in the shader, a function that returns its
        /// depth or reads its position's depth).
        UnormDepthBias = "unorm-depth-bias", Precision,
            "a biased depth can store a code next to Metal's: the device rounds the biased sum \
             its own way, and without depthBiasExact counts the bias in its own unit";
        /// A draw tests or writes a `Depth16Unorm` depth on a device whose driver is not measured
        /// to convert a depth to Metal's code (`DriverDeviation::MetalDepth16Stores`,
        /// `render_state.rs`), through a pipeline whose fragment stage cannot write the code
        /// itself (more than one sample, early fragment tests, the draw's tests run in the shader,
        /// a function that reads its position's depth and returns none), or through one that
        /// does, in depth clamp mode, whose range the device clamps the depth to.
        UnormDepthRounding = "unorm-depth-rounding", Precision,
            "a depth near a code's midpoint can store the code below Metal's: the device converts \
             the depth to the 16-bit code its own way, where Metal rounds to nearest, a tie up";
        /// A draw tests or writes a `Depth16Unorm` depth through a fragment stage that writes each
        /// depth as Metal's code, on a device whose driver is not measured to store Metal's codes
        /// (`DriverDeviation::MetalDepth16Stores`, or for a biased draw
        /// `DriverDeviation::MetalDepth16BiasedStores`; `RuntimeFact::Depth16Codes`,
        /// `render_state.rs`); a pipeline without a fragment function draws through a fragment
        /// stage of its own.
        ShaderUnormDepth = "shader-unorm-depth", Performance,
            "the fragment stage writes each depth as Metal's code, about 50 more instructions \
             per fragment, which keeps the device from testing the depth before the stage runs";
        /// A draw biases a `Depth16Unorm` depth by a slope through a fragment stage that adds
        /// Metal's bias itself (`RuntimeFact::Depth16Codes`, `render_state.rs`).
        UnormDepthSlope = "unorm-depth-slope", Precision,
            "the fragment stage takes the depth's slope from the depths of its 2x2 quad, where \
             Metal takes the triangle's own: a biased depth near a code's midpoint can store \
             the code next to Metal's";
        /// A draw of points keeps Vulkan's clipping of points, where Metal's guard band cannot be
        /// emulated: in a rate-mapped pass, on a device without `shaderClipDistance` or with too
        /// few clip distances beside the vertex function's own, or through a viewport whose
        /// widened Vulkan viewport would leave the device's limits or not be a `float` exactly
        /// (`render_state.rs`).
        PointGuardBand = "point-guard-band", Precision,
            "a point whose center lies past its viewport, within Metal's guard band of 1.5 times \
             the viewport's half-size, is culled where Metal draws it, and a wide point can lose \
             pixels at a fractional viewport edge by the device's own clipping";
        /// A draw whose fragment function writes its depth, in depth clamp mode through a range
        /// above 0, keeps the device's clamp of that depth, where the function cannot clamp it
        /// itself under a Vulkan clamp that holds 0: on a device without
        /// `VK_EXT_depth_clamp_control` whose vertex stage cannot compensate the depth exactly, a
        /// mesh pipeline there, through viewports of different clamps or depth ranges, under a
        /// depth bias that saturates or offsets the range, or past the device's clip distances
        /// (`render_state.rs`). And a draw whose fragment function writes its depth under a
        /// stencil depth-failure operation: a positive NaN depth, which fails Metal's depth
        /// test and runs that operation, discards the fragment instead.
        NanDepth = "nan-depth", Precision,
            "a NaN depth the fragment function writes: a negative one in depth clamp mode stores \
             the depth clamp range's low end, where Metal stores 0; a positive one runs no \
             stencil operation, where Metal runs the depth-failure one";
        /// A draw of triangles in depth clip mode under a depth bias of a constant or slope
        /// factor other than 0, on a device that enabled `VK_EXT_depth_range_unrestricted` but
        /// lacks `depthClamp`, `VK_EXT_depth_clip_enable` or `VK_EXT_depth_clamp_control`
        /// (`render_state.rs`, `announce_unclamped_bias`): such a device stores a depth the bias
        /// takes past `[0, 1]` unclamped, where Metal clamps it.
        BiasedDepthPastUnitRange = "biased-depth-past-unit-range", Precision,
            "a depth a bias takes past [0, 1] in depth clip mode is stored unclamped, where Metal \
             stores 0 or 1";
        /// A draw of lines or triangles whose vertex stage clamps each vertex's depth as Metal
        /// does, through a viewport depth range with a negative depth in depth clip mode or past
        /// `[0, 1]` in depth clamp mode (`render_state.rs`, `VertexClamp`): Metal clips a
        /// primitive at its near and far planes and at its guard band first, and clamps the
        /// depths of the pieces' vertices.
        ClippedDepthPieces = "clipped-depth-pieces", Precision,
            "a primitive that crosses Metal's near or far plane, or its guard band of 1.5 times \
             the viewport's half-size, stores the interpolation of its own vertices' clamped \
             depths, where Metal clamps the depths of the pieces it clips the primitive into";
        /// A draw of lines or triangles through such a range whose vertex stage cannot clamp
        /// each vertex's depth: its fragment function reads its position's depth, a mesh
        /// pipeline, on a device without `shaderClipDistance` or past its clip distances, or
        /// through viewports whose depths need different compensations on a device that takes
        /// the ranges natively (`render_state.rs`, `VertexClamp`); and a draw of triangles in depth
        /// clamp mode under a depth bias that moves the depth through a range inside `[0, 1]`,
        /// where Metal adds the bias to each vertex's depth clamped to `[0, 1]`, which a primitive
        /// past its near or far plane shows.
        SampleDepthClamp = "sample-depth-clamp", Precision,
            "a sloped primitive's depth is clamped at each sample, where Metal clamps each \
             vertex's depth before the interpolation, at 0 in depth clip mode and to [0, 1] in \
             depth clamp mode";
        /// A draw of triangles in depth clamp mode through a viewport depth range whose larger
        /// depth lies past 1, under a depth bias of factors, whose Vulkan clamp holds at most 1
        /// (`render_state.rs`, `VertexClamp`).
        DepthBiasPastOne = "depth-bias-past-one", Precision,
            "a depth the bias moves past 1 stores 1, where Metal clamps it to the range's larger \
             depth";
        /// A draw's vertex stage reads in place bytes a fragment stage of its render pass wrote,
        /// through an argument buffer or a resident resource, of a buffer another stage of the
        /// pass or the vertex function writes, or in an execution of render commands
        /// (`render/pass_start_reads.rs`).
        InPlaceVertexReads = "in-place-vertex-reads", Precision,
            "the device may order the fragments' write before the read, where Metal's vertex \
             stages read the bytes from before the pass's fragments in every run measured";
        /// A draw's object or mesh stage reads in place bytes a fragment stage of its render pass
        /// wrote (through an argument buffer or a resident resource, or of a buffer another stage
        /// of the pass or the stage's function writes; an emulated stage's where its work records
        /// after the pass's earlier render pass instances), or a draw's vertex, object or mesh
        /// stage writes bytes a fragment stage of its render pass read or wrote
        /// (`render/pass_start_reads.rs`): Metal's GPUs run every draw's vertex, object and mesh
        /// functions before any draw's fragments of the pass.
        FragmentsBeforeGeometry = "fragments-before-geometry", Precision,
            "the device may run the pass's earlier fragments before the draw's stage, and does \
             across a split of the pass (a command's barrier, a memory barrier): the read sees \
             their write, their read misses the write, or the write outlasts theirs, where \
             Metal's fragments run after every draw's vertex, object and mesh functions in every \
             run measured";
        /// An indexed indirect draw whose index fetch nothing bounds (the device lacks
        /// `robustBufferAccess2`, its driver is not measured to read zero past the bound range,
        /// `DriverFact::RobustIndexFetchReadsZero`, or the index offset lies at or past the
        /// buffer's length) is bounded on the GPU before the pass (`index_bounds.rs`).
        IndirectIndexBoundsPass = "indirect-index-bounds-pass", Performance,
            "two dispatches before the pass write two draws' arguments and the indices past the \
             buffer into the command buffer's index pool (a sixty-fourth of the device's largest \
             device-local heap), between barriers, two for a run of such draws over one arguments \
             and one index buffer, and the draw is two indirect draws; a draw the pool cannot \
             hold fails the command buffer";
        /// A mesh render pipeline's draws on a device without `VK_EXT_mesh_shader`
        /// (`render/mesh.rs`).
        EmulatedMeshDraws = "emulated-mesh-draws", Performance,
            "the object and mesh functions run as compute dispatches before the pass, between \
             barriers, writing records a synthesized vertex stage draws; a draw whose mesh \
             threadgroups the GPU decides is scanned first and reserves its records in the \
             command buffer's records pool (a sixty-fourth of the device's largest device-local \
             heap), and a draw the pool cannot hold fails the command buffer";
        /// A tessellation pipeline's patch draws on a device without `tessellationShader`
        /// (`render/tessellation_scan.rs`).
        EmulatedTessellationDraws = "emulated-tessellation-draws", Performance,
            "each patch draw runs a scan dispatch before the pass, between barriers: one \
             workgroup that processes the draw's patches, instance by instance, into records in \
             order, so its GPU time grows with the draw's patches times its instances, apart \
             from its draw; the draw then draws the tessellator's triangles with the \
             post-tessellation vertex function as a plain vertex function, up to 2^30 triangles \
             per indirect draw; an indirect draw reserves its records in the command buffer's \
             records pool (a sixty-fourth of the device's largest device-local heap), and a draw \
             the pool cannot hold fails the command buffer";
        /// A render pipeline state whose draws run a synthesized geometry stage
        /// (`RuntimeFact::SynthesizedGeometry`): its fragment function reads
        /// `[[barycentric_coord]]` on a device without `fragmentShaderBarycentric`, or its vertex
        /// function selects its primitive's render target slice or viewport on a device that
        /// cannot store `Layer` or `ViewportIndex` from a vertex function.
        SynthesizedGeometryStage = "synthesized-geometry-stage", Performance,
            "each draw runs a geometry stage of its topology class between the vertex and \
             fragment stages, which emits each primitive's vertices with the barycentric weights \
             or the first vertex's slice and viewport";
        /// A pipeline state (render, mesh render, tile render or compute) whose largest stage
        /// module, under the consumer's exact forms, exceeded the device's compiler word bound
        /// ([`metal2vulkan::pipeline::RuntimeFact::ExactFormsOverCompilerLimit`];
        /// [`metal2vulkan::device::DriverWorkaround::RegisterAllocatorOverflowsLargeModules`],
        /// ANV's register allocator faults compiling a module past it), so the translation dropped
        /// the consumer's exact forms until it fit: native sampling first, then every
        /// module-growing exact answer native. Announced where the runtime makes a variant whose
        /// pipeline the fit reduced, the base variant or another (`pipeline/render.rs` for a
        /// render or mesh render state, `pipeline/mod.rs` for a compute or tile render state, from
        /// the pipeline's dependencies), naming the state's kind and the forms dropped; the
        /// reduced stages announce their own precision notes besides.
        ExactFormsOverCompilerLimit = "exact-forms-over-compiler-limit", Precision,
            "the dropped exact forms' results can differ from Metal's as their notes' \
             documentation states, where the device's compiler would fault on the exact module";
        /// Visibility results start where Vulkan writes no query result.
        LandedVisibilityResults = "landed-visibility-results", Performance,
            "they land in an upload region and are copied on into the buffer after the pass";
        /// A blit copies between overlapping ranges of one buffer, which Vulkan's copy forbids.
        OverlappingBufferCopy = "overlapping-buffer-copy", Performance,
            "the range goes through a scratch buffer of its size, which its creation clears by \
             a fill it waits for: two copies instead of one";
        /// A blit fills bytes that are not whole 4-byte words at multiples of 4.
        SplitBufferFill = "split-buffer-fill", Performance,
            "the words are filled and the up to 3 bytes at each end copied from a 1-KiB \
             pattern buffer the encoder makes once";
        /// A blit copies between a buffer and a texture in a layout Vulkan cannot state.
        StagedTextureCopy = "staged-texture-copy", Performance,
            "the rows go through a tightly packed scratch buffer: two copies instead of one, \
             and for a single-sample texture the scratch's creation clears it by a fill it \
             waits for";
        /// A blit copies texel rows of a texture over a buffer to or from bytes of the same
        /// buffer that they share, which Vulkan's copy forbids.
        OverlappingTexelCopy = "overlapping-texel-copy", Performance,
            "the rows go through a tightly packed scratch buffer, which its creation clears by \
             a fill it waits for: two copies instead of one";
        /// A dispatch is more than one Vulkan dispatch: its grid is not a whole number of
        /// threadgroups, or a group count is past the device's `maxComputeWorkGroupCount`.
        SplitDispatch = "split-dispatch", Performance,
            "one Vulkan dispatch per region (the interior and each partial edge, per chunk of \
             group counts), each on its own specialized pipeline, a barrier between each two";
        /// A dispatch whose threadgroups are not whole subgroups of the kernel's pinned
        /// subgroup size runs on the kernel's emulated-simdgroups variant.
        EmulatedSimdgroups = "emulated-simdgroups", Performance,
            "a second compile of the kernel whose simdgroup operations go through threadgroup \
             memory between barriers";
        /// A pass of separate depth and stencil textures, or of a stencil-only texture alone at
        /// custom sample positions, renders into a scratch image of a combined format
        /// (`depth_stencil_scratch.rs`).
        SeparateDepthStencil = "separate-depth-stencil", Performance,
            "each loaded aspect is copied into the scratch image before the pass and each stored \
             one back after it, through a staging buffer, or a multisampled colour image for \
             multisampled textures, or sample-rate draws where the device's copies of those are \
             missing or wrong";
        /// A pass resolves a combined depth/stencil attachment one aspect at a time, on a
        /// device that resolves the two aspects only together or only in one mode, or where its
        /// aspects resolve into different textures (separate textures).
        OneAspectResolve = "one-aspect-resolve", Performance,
            "a scratch image per aspect, the second aspect resolved after the pass in a render \
             pass of its own from the stored samples, and each resolved aspect copied into the \
             resolve texture after the pass through a staging buffer";
        /// A pass resolves into a texture that cannot be an attachment (its usage lacks
        /// `RenderTarget`), or a layered pass into a texture whose slices from the resolve slice
        /// hold fewer layers than the pass's.
        ResolveScratch = "resolve-scratch", Performance,
            "the pass resolves into a scratch image, whose render area is copied into the \
             texture after the pass";
        /// A multisample color resolve drawn after the pass by Metal's exact rules, where the
        /// device's `AVERAGE` resolve is not measured to be Metal's, under
        /// [`metal2vulkan::device::ResolvePolicy::Exact`] or from a fetching pass's sample planes
        /// (`color_resolve.rs`); or a
        /// stencil resolved by `DepthResolvedSample` after `Min` or `Max`, which no Vulkan
        /// resolve mode states, or a `Min` or `Max` depth on a device without the mode, drawn
        /// after the pass (`depth_stencil_resolve.rs`).
        ShaderResolve = "shader-resolve", Performance,
            "one more render pass instance after the pass reads the stored samples (or the \
             sample planes) and writes the resolve texture, where the device would resolve in \
             the pass; where the attachment's image takes no input attachment usage, a copy of \
             its samples over the render area into a scratch image first";
        /// A fragment function writes a float to a normalized, sRGB or half-float color
        /// attachment on a device whose own conversion is not Metal's
        /// (`StageModule::rounded_colors`).
        ShaderColorConversion = "shader-color-conversion", Performance,
            "the fragment function rounds each such channel to the code Metal stores before it \
             writes it: about 35 more instructions per normalized or half-float channel, and \
             about 70 and one read of a 4 KiB table per sRGB channel";
        /// A fragment function writes a `half` to a half-float color attachment as the bits of
        /// the half Metal stores, through a 16-bit output, on a device that does not store a
        /// 32-bit output's NaN as Metal does (`StageModule::half_bit_colors`).
        ShaderHalfColorBits = "shader-half-color-bits", Performance,
            "the fragment function writes each such output as its half's bits through a 16-bit \
             output: about 20 more instructions per output, and the attachment takes no Vulkan \
             blend of one and zero";
        /// A Metal render pass runs as several Vulkan render pass instances (`render.rs`): a
        /// framebuffer fetch through the pixel interlock keeps its attachment out of its draws'
        /// subpass, and the pass's draws leave different attachments out, or its last draws
        /// leave out an attachment that resolves, which resolves in one more instance; or a
        /// fetch through input attachments reads an attachment whose image takes no input
        /// attachment usage (a caller's) after other draws recorded, and the stand-in that has
        /// it makes it an input attachment of the later draws' subpass alone; or a tile dispatch
        /// runs between two instances; or some of its draws render some of its layers (vertex
        /// amplification as multiview, or a view mapping that moves a draw to one layer), each run
        /// of them in an instance over those layers; or a memory barrier over buffers or textures
        /// orders its draws after it behind those before it, which a barrier between instances
        /// does. Not where the interlock alone splits the pass, on a device whose driver ran such a
        /// pass no slower split than whole (`DriverFact::InterlockedPassSplitNoSlower`): its faster
        /// form.
        RenderPassSplit = "render-pass-split", Performance,
            "each instance after the first loads the attachments and each before the last \
             stores them, with a barrier from everything before to everything after between \
             them: on a tiler, a tile flush per split";
        /// A draw under vertex amplification that Vulkan multiview cannot express (view
        /// mappings that offset the viewport or name layers out of order, more amplifications
        /// than the device's views, or a pipeline that chooses or reads its render target
        /// slice) draws one amplification at a time (`bind::DrawCommands::amplifications`).
        AmplificationDraws = "amplification-draws", Performance,
            "one Vulkan draw per amplification, each after the amplification's push data, on \
             the pipeline state's variant that reads it, where multiview would draw them all \
             at once";
        /// A memoryless attachment renders into a backing image of the device's own memory,
        /// because a framebuffer fetch through the pixel interlock reads it or the pass splits
        /// (`attachment_stand_in.rs`); not for a split `render-pass-split` leaves unannounced. A
        /// color's backing is one its command queue keeps for its passes of the attachment's
        /// format, samples and size (`attachment_stand_in::pooled_backing`).
        MemorylessBacked = "memoryless-backed", Performance,
            "a scratch image, a color's kept by its command queue for its passes of that shape \
             and a depth or stencil one's per pass: memory, and the bandwidth to write and read \
             it, that Metal does not spend on a memoryless texture";
        /// A memoryless texture buffer (`newTextureWithDescriptor:` of a `TextureBuffer`
        /// descriptor of `MTLStorageModeMemoryless`, which plain Metal makes), whose texels are a
        /// buffer's bytes (`buffer_texture.rs`): the buffer is a `Private` one of the device's,
        /// since no Vulkan buffer takes memory lazily. The texture answers the descriptor's
        /// memoryless mode, as Metal's does, and the runtime refuses every use of it, as Metal's
        /// validation stops every use of a memoryless texture but as a render pass's attachment,
        /// which no texture buffer is (`Texture::check_not_memoryless`).
        MemorylessTextureBuffer = "memoryless-texture-buffer", Performance,
            "a Private buffer of the texture's row of texels: memory Metal spends none of on a \
             memoryless texture. Nothing reaches those bytes";
        /// A color render target that a framebuffer fetch reads lacks the usage the fetch's form
        /// needs, so its pass renders into a scratch image that has it
        /// (`attachment_stand_in.rs`): `STORAGE` in the pixel interlock, or for its sample
        /// planes sampled usage with a view of its texel size's unsigned-integer format, either
        /// of which the runtime gives a texture only where it leaves the image's layout (its
        /// compression) alone; or input attachment usage, which a caller's image may lack.
        /// Announced once, at the first such pass: a runtime texture may take its storage form
        /// later (`fetched-target-storage`), its later passes then fetching it in place. Only a
        /// fetch's scratch: the notice names the draw that fetched it first (its fragment function,
        /// and the color's blend state and write mask).
        FetchThroughScratch = "fetch-through-scratch", Performance,
            "a scratch image per pass, the texture copied into it before the pass on a `Load` \
             and back after it on a `Store`";
        /// A runtime texture that a pass rendered into a scratch image for want of `STORAGE`
        /// (`fetch-through-scratch` through the pixel interlock) took its storage form: its image
        /// made again with `STORAGE` and its texels copied in once, at the commit of the device's
        /// only uncommitted command buffer (`texture/storage_form.rs`); with a submission check,
        /// once that command buffer's checks ran at its submission and every encoder of it ran.
        /// Announced once per device, whatever `DeviceOptions::announce_each` says, before that
        /// command buffer is scheduled: on the scheduler's thread once it submitted it, outside any
        /// program call or host step, or in the call that makes a command buffer meanwhile.
        FetchedTargetStorage = "fetched-target-storage", Performance,
            "the texture's image laid out as one with `STORAGE` usage from then on (on MoltenVK \
             `ShaderWrite`, without lossless compression), and one copy of its texels, where \
             Metal's texture keeps its usage";
        /// A depth/stencil render target whose image was not made for the pass's custom sample
        /// positions (a caller's image without
        /// `VK_IMAGE_CREATE_SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_BIT_EXT`, which Vulkan needs to
        /// rasterize a depth/stencil attachment at them), or was on a driver that keeps a depth
        /// rendered at them per sample only through a copy
        /// ([`metal2vulkan::device::DriverFact::PositionedDepthReadAtStandardPositions`]: NVIDIA's,
        /// measured), renders into a scratch image made for them (`attachment_stand_in.rs`).
        PositionsThroughScratch = "positions-through-scratch", Performance,
            "a scratch image per pass, each aspect copied into it before the pass on a `Load` \
             and back after it on a `Store`";
        /// A framebuffer fetch through the pixel interlock in a multisampled pass, on a device
        /// without multisampled storage images, reads and writes the attachment as its sample
        /// planes (`sample_planes.rs`): a single-sampled array image of one layer per sample.
        SamplePlanes = "sample-planes", Performance,
            "a planes image per fetched attachment and pass; a compute dispatch copies the \
             samples into it before the draws that fetch (on a `Load`, or after other draws \
             rendered it) and a sample-rate pass copies them back after those draws (before \
             other draws render it, and where the pass keeps or resolves it)";
        /// A render pass's draws test and write the stencil in the fragment shader, on a device
        /// without `VK_EXT_shader_stencil_export` (`shadow_stencil.rs`): the function's
        /// `[[stencil]]` output is Metal's stencil reference, which only that extension hands the
        /// fixed-function test.
        ShaderSideStencil = "shader-side-stencil", Performance,
            "an 8-bit shadow image of the stencil per pass, which the stencil is copied into \
             before each run of such draws and back from after it (through a staging buffer at \
             one sample), each run in a render pass instance of its own, its fragment shader \
             testing and writing the shadow in the pixel interlock";
        /// A tile dispatch stages its pass's attachments through storage views in their staging
        /// formats, which it takes of a scratch image of each attachment's format: each is
        /// copied into its scratch before the dispatch and back after it
        /// (`render/tile_dispatch.rs`).
        TileDispatchScratch = "tile-dispatch-scratch", Performance,
            "a scratch image per staged attachment and pass, over the pixels the dispatch \
             covers, which each dispatch copies the attachment into before it and back from \
             after it, between barriers from everything before to everything after: memory and \
             bandwidth that Metal's tile memory does not spend";
        /// A render pass's draws' fragment functions declare an explicit imageblock, which they
        /// read and write in a storage image of the render area (`fragment_imageblock.rs`).
        FragmentImageblockStorage = "fragment-imageblock-storage", Performance,
            "an R32Uint array image per pass over the render area, a layer per 4 bytes of a \
             pixel's imageblock and per sample, cleared before the pass, which the draws read \
             and write in the pixel interlock: memory and bandwidth that Metal's tile memory \
             does not spend";
        /// A command buffer uses a texture over the caller's image whose home layout is not
        /// `GENERAL` (`Device::new_texture_with_vk_image`): every image the runtime uses rests in
        /// `GENERAL`.
        WrappedImageLayoutTransitions = "wrapped-image-layout-transitions", Performance,
            "two whole-image barriers per batch that uses it: into GENERAL before its first \
             encoder, and back to its home layout at its end";
        /// An execution of an indirect command buffer (`executeCommandsInBuffer:withRange:`) whose
        /// commands are known only once the work before it completed: an encoder made the buffer
        /// resident for writing, so the GPU may write its records; it reads its range from a
        /// buffer; or it lies inside a block of command predication. The runtime records such an
        /// execution on the CPU then where the GPU cannot run it alone. A render execution: where
        /// its records may hold, at the commit or as its batch is ready, more than draws of one
        /// primitive type under the encoder's state, or what its check looks for
        /// (`render/executions/gpu.rs`); its notice names the first rule that kept it from the GPU
        /// alone. A compute execution: with a caller's check (`execute_commands_in_buffer_checked`)
        /// that may see what it looks for (`IndirectCommandCheckNeeds`: a state that may encode
        /// indirect commands, the encoder's where the buffer inherits it, else one its records may
        /// name), at the commit or as its batch is ready; with its range read from a buffer; of a
        /// buffer that inherits the encoder's buffers; where its records may name a pipeline state whose
        /// kernel observes its threadgroup partition or fetches `[[stage_in]]` attributes, or a
        /// state the CPU set that the program released since; and, on a device without
        /// `VK_EXT_device_generated_commands`, where the host step is faster: where its commands
        /// run with no barrier between them and its commands times the pipeline states its records
        /// may name past four pass 256 (the states: every live one made with
        /// `supportIndirectCommandBuffers`, or the encoder's where the buffer inherits it), where
        /// they run with one and its commands times the states past twelve pass 512, and where its
        /// commands times its states pass 32,768. Every other execution runs on the GPU alone and
        /// announces nothing.
        HostReplayedIndirectCommands = "host-replayed-indirect-commands", Performance,
            "one GPU-to-CPU round trip per execution: its command buffer's work before it, and \
             every batch submitted before that on any queue, completes before the CPU records \
             its commands from their records and submits the rest; the device's scheduler \
             submits nothing else while it records them, and the first command of a threadgroup \
             size or threadgroup lengths its pipeline state has not dispatched makes that \
             VkPipeline then. The commands run in the large-buffer form (their range-limited \
             buffer slots at device addresses, with in-shader bounds checks), whose variants the \
             commit compiles for every pipeline state a record may name (or, for an execution \
             committed before the buffer was made resident for writing, the use_resource call \
             that made it so); without device addresses, a command's first buffer shape \
             compiles its variant then";
        /// A command buffer's GPU times on a device that calibrates its timestamps against no
        /// host clock the runtime reads (no `VK_KHR_calibrated_timestamps` or
        /// `VK_EXT_calibrated_timestamps` with the device's time domain and a monotonic host
        /// one), whose calibration failed as it completed, or whose calibration placed them outside
        /// the command buffer's life (before its first submission, after its completion) by more
        /// than its stated deviation and the clocks' drift allow (`gpu_time.rs`).
        UncalibratedGpuTimes = "uncalibrated-gpu-times", Precision,
            "the end is the host time the runtime saw the command buffer complete, late by that \
             latency, and the start the GPU's exact duration before it";
        /// A command buffer's GPU times on a device whose timestamp pools had no slot free at
        /// the command buffer's commit, and which could make no pool more: every slot was held
        /// by a command buffer in flight (`gpu_time.rs`).
        UntimedCommandBuffer = "untimed-command-buffer", Precision,
            "its GPU start and end times are refused (`Unsupported`, `GpuTimes`), where Metal \
             gives them; its kernel times stay";
        /// A command buffer's GPU times on a device told to answer them from the host
        /// (`DeviceOptions::gpu_times_from_host`, `RawDevice::gpu_times_from_host`), or whose
        /// driver's timestamps wait for every queue's (`DriverFact::TimestampsWaitForEveryQueue`:
        /// MoltenVK 1.4.1 on Apple's GPUs, 1.4.2 on the M3 Max), which writes no GPU timestamps
        /// (`gpu_time.rs`). Every
        /// answer is one Metal could give: seconds of Metal's clock (`mach_absolute_time`'s, on
        /// macOS); its commit, kernel start, kernel end, GPU start, GPU end and the moment its
        /// completion is observable in that order; and a queue's command buffers one after
        /// another, none starting before the one before it ended. The start can be early by the
        /// time from the submission's end to the GPU's start, or late by the time the runtime took
        /// to see the queue's command buffer before it complete; the end is late by the time it
        /// took to see this one complete.
        GpuTimesFromHost = "gpu-times-from-host", Precision,
            "its GPU start is its kernel end time (or its queue's latest GPU end before it) and \
             its GPU end the host time the runtime saw it complete: a window its work ran in, \
             which holds the submission's and the completion's latencies, not the GPU's own start \
             and end";
        /// A multisample color resolve averages through Vulkan's `AVERAGE` mode, which leaves
        /// the rounding of the average (and the sRGB encoding) to the device, on a device not
        /// measured to round as Metal does: under [`metal2vulkan::device::ResolvePolicy::Device`],
        /// the default, any color attachment but one a fetching pass's sample planes hold as it
        /// ends (resolved from them by Metal's rule); under
        /// [`metal2vulkan::device::ResolvePolicy::Exact`] a format without Metal's closed rule at
        /// its count (the 16-bit float, `RG11B10Float` and `RGB9E5Float` formats past 4 samples,
        /// and a format no rule covers), or an attachment whose image the exact resolve can
        /// neither read nor copy: an image without input attachment usage of a format the device
        /// does not copy (`color_resolve.rs`). Not where `DriverFact::ColorResolveAverageIsMetals`
        /// holds, for any format: MoltenVK's resolve is Metal's own.
        /// Measured against Metal's rules at 2 and 4 samples
        /// (`runtime_lane::resolve_rounding::the_device_resolve_rounds_as_announced`, which holds
        /// the bound) on RADV (Mesa 26.0.8, the Radeon 780M and the RX 7900 XTX), ANV (Mesa
        /// 26.2.4, the Iris Xe), NVIDIA (615.71, the RTX 5080 Laptop GPU) and lavapipe (Mesa
        /// 26.2.4, at 4 samples): every unorm, snorm and packed 16-bit unorm channel within one
        /// code of Metal's, an sRGB channel within two (NVIDIA's at 4 samples; one elsewhere).
        /// Each rounds an 8-bit unorm tie down (RADV and ANV: 32512 of `R8Unorm`'s 65536 pairs of
        /// samples one code below at 2 samples); an sRGB or packed channel lands above Metal's
        /// code in some texels and below in others. A float channel's average differs in its
        /// last bits, counted and not bounded (RADV: 8565 of `R32Float`'s 65536 channels at 2
        /// samples).
        /// The device also decides what a NaN or infinite sample makes of the average, where
        /// Metal's resolve counts a NaN of a `half` or small-float channel as 0 and an infinite
        /// `half` as 65536 of its sign, and makes any NaN of a 32-bit float channel `0x7fc00000`;
        /// lavapipe's average keeps a `half` NaN.
        DeviceRoundedResolve = "device-rounded-resolve", Precision,
            "a resolved channel can differ from Metal's by one code (an sRGB channel by two) \
             where the average lies near a rounding boundary, a float channel in its last bits, \
             and wholly where a sample is a NaN or an infinity: Metal's resolve counts a half or \
             small-float NaN as 0 and a half infinity as 65536 of its sign, and makes any 32-bit \
             float NaN 0x7fc00000, where the device's average may keep them";
        /// A multisample color resolve of more than 4 samples, of which Apple's GPUs make no
        /// texture: no Metal answer is measured there (`color_resolve.rs`); likewise a stencil
        /// resolved by `DepthResolvedSample` after `Min` or `Max` (`depth_stencil_resolve.rs`),
        /// whose pick among equal depths is measured at 2 and 4 samples.
        UnmeasuredResolve = "unmeasured-resolve", Precision,
            "the resolve applies the rules Metal's resolve follows at 2 and 4 samples (or the \
             device's average, for a format without them), which no Metal answer confirms at \
             this count";
        /// A multisample resolve of an extended-range color attachment whose resolve Metal
        /// averages from values finer than the codes the attachment holds (`BGRA10_XR`'s alpha,
        /// an `_sRGB` twin's color, `RGB10A2Unorm_sRGB`'s channels;
        /// `bind::ColorResolveFormat::extended_range`): the crate-owned resolve
        /// (`color_resolve.rs`) writes a near-equivalent from the codes where a pixel's samples
        /// differ, and Metal's code where they hold one. Measured against Metal's resolve on the
        /// M3 Max under macOS 27.0.1 (our probe `xr`, 4096 texels per format at 2 and 4 samples,
        /// each sample its own value): `BGRA10_XR`'s alpha Metal's in 84% and 88% of texels and
        /// within a code in every one; an `_sRGB` twin's color in 81% and 85% of channels and
        /// within a code in 99.5%, at most 7 codes off where the samples' values nearly cancel;
        /// `RGB10A2Unorm_sRGB`'s colors in 83% and 87%, within 2, and its alpha in 94% and 97%,
        /// within one (`runtime_lane::extended_range_resolve`). A linear form's color is Metal's
        /// rule, every texel.
        ExtendedRangeResolve = "extended-range-resolve", Precision,
            "where a pixel's samples differ, a resolved alpha, or an sRGB color, can differ from \
             Metal's by a code (an sRGB color by a few where the samples' values nearly cancel): \
             Metal averages values finer than the codes the attachment holds";
        /// An attachment stored with `DontCare` (or the multisampled attachment of
        /// `MultisampleResolve`) is Vulkan's `DONT_CARE`, where Apple's GPU keeps the earlier
        /// texels unless its tiler overflows. Only on a driver measured to keep the earlier texel
        /// or the stored one at each texel (`DriverFact::DontCareStoreIsOneOfMetals`); every other
        /// device stores it (`STORE`), Metal's answer where its tiler overflows
        /// (`render_target.rs`). Not a texture over a buffer the pass renders through an image it
        /// copies nothing back from (a shadow, a stand-in: `kept_bytes.rs`), which keeps the
        /// buffer's bytes, Metal's answer; only one whose image the pass renders in place.
        DontCareStore = "dont-care-store", Precision,
            "the attachment's texels after the pass are each its earlier texel or the one the \
             pass rendered, both Metal's answers, but not always the one Metal's tiler keeps";
        /// An attachment loaded with `DontCare` and stored, in a pass that draws, is cleared
        /// before the draws to zero (a color) or its clear value (depth, stencil) over the render
        /// area and the tiles past it a clear reaches (`clear_reach.rs`), where Metal clears only
        /// the tiles a draw's primitives touch and keeps the texels of the others.
        DontCareLoadTiles = "dont-care-load-tiles", Precision,
            "the tiles no draw touches hold zero or the clear value where Metal keeps their \
             texels";
        /// A new `Private` texture Metal keeps losslessly compressed reads a fixed pattern in
        /// Metal (`texture/heap_tiled.rs`'s `fresh_pattern`), which the device cannot write
        /// exactly into its image: a NaN depth where a copy into a depth image may store it
        /// clamped or the image is multisampled, a NaN texel of a multisampled image (whose bits
        /// a clear leaves to the device), or an image holding other bytes than Metal's texel. The
        /// texture reads zero there. Announced where a command first reads a texel of that content
        /// no clear, resolve or blit covered (a pass that loads it, a blit or a shader that reads
        /// it), not as the texture is made (`fresh_unwritten.rs`).
        FreshPatternZeroed = "fresh-pattern-zeroed", Precision,
            "the new texture reads zero where Metal reads its fixed pattern until a write";
        /// An indexed draw from an index offset at or past its index buffer's length draws index
        /// 0 plus the base vertex at each position, a direct draw from a zero-padded copy
        /// (`draw_source.rs`) and an indirect one through the index bounds pass
        /// (`index_bounds.rs`), where Metal reads the memory past the buffer.
        ZeroedIndexFetch = "zeroed-index-fetch", Precision,
            "each index position draws index 0 plus the base vertex, where Metal reads whatever \
             memory follows the buffer";
        /// A pipeline stage's reconvergence after divergence is implementation-defined on the
        /// device, where Metal's is maximal
        /// ([`Reconvergence::ImplementationDefined`](metal2vulkan::pipeline::Reconvergence)).
        ImplementationDefinedReconvergence = "implementation-defined-reconvergence", Precision,
            "simdgroup operations after divergent control flow can see other lanes active than \
             Metal's";
        /// A live sampler state the device's sampler tables do not map to it for some commands
        /// (`heap_sets.rs`), which reach every other live state by its ID as they run: one whose
        /// element no set-1 sampler array can hold, as more states live than it holds (each
        /// takes the lowest element no live state holds, `resource_ids.rs`), the device's as the
        /// state is made, or a pipeline's as a command is encoded or the state made
        /// (`sampler.rs`, `compute.rs`, `heap_sets.rs`), each such command buffer counted once
        /// (`WorkCounts::unlisted_argument_sampler_buffers`); one at an ID neither window of the
        /// tables reaches (IDs 0 to 32767, and 32768 from the first ID past them, a
        /// reservation's); one whose Vulkan sampler for an array's use the device cannot make or
        /// the use refuses (a feature the device lacks), that array holding the neutral at its
        /// element; a cubic state, for a command whose stage does not test the states it reaches
        /// for a cubic filter (a fragment stage never does); and one whose elements could not be
        /// written. Where it happens: past the device's update-after-bind sampler limit or the
        /// heap capacity of a pipeline (65536 on the M3 Max under MoltenVK), past 32768 live
        /// states above a reservation, a cubic state live while a render pipeline samples
        /// argument samplers, or a device's sampler limit (`maxSamplerAllocationCount`).
        UnlistedArgumentSampler = "unlisted-argument-sampler", Precision,
            "a function reading the state through its ID in an argument buffer reads the \
             neutral sampler, where Metal's reads the state";
        /// A `Depth32Float` texture the program only reads or samples, on a device whose driver
        /// is not measured to keep every `float` in a depth image (`device/driver_facts.rs`: a
        /// denormal, and without `VK_EXT_depth_range_unrestricted` a value outside `[0, 1]`), is
        /// held in an `R32_SFLOAT` image with a `D32_SFLOAT` twin for the comparisons the device
        /// makes (`image_plan.rs`).
        DepthTwin = "depth-twin", Performance,
            "a second image of its size, which every copy into it writes too; a copy into it \
             from a depth texture without a twin goes through a scratch buffer";
        /// An upload (`replace_region`, a blit from a buffer) into the 32-bit float depth of a
        /// texture's depth image, on a device without `VK_EXT_depth_range_unrestricted` enabled,
        /// where Vulkan leaves a value outside `[0, 1]` undefined, whose driver is not measured to
        /// keep one (`DriverFact::DepthImageKeepsValuesOutsideUnitRange`): a texture without a
        /// twin. And a render pass's NaN clear depth on such a device, which Metal stores as
        /// `0x7fc00000`, cleared to 1.0 (`nan_clear.rs`).
        RestrictedDepthUpload = "restricted-depth-upload", Precision,
            "a value outside [0, 1] may be stored clamped to [0, 1], where Metal's keeps it, so \
             reads, samples, comparisons and depth tests of it can differ";
        /// An indirect dispatch (`dispatchThreadgroupsWithIndirectBuffer:...`,
        /// `dispatchThreadsWithIndirectBuffer:...`; `indirect_dispatch.rs`), or a dispatch after
        /// `setStageInRegionWithIndirectBuffer:indirectBufferOffset:`: its arguments or its
        /// stage-in region are GPU memory the work before it may write, so they are known only once
        /// that work completed.
        HostReadDispatchArguments = "host-read-dispatch-arguments", Performance,
            "one GPU-to-CPU round trip per dispatch: a copy of its arguments where it is \
             encoded, then its command buffer's work before it, and every batch submitted \
             before that on any queue, completes before the CPU reads them and records the \
             dispatch they state; the device's scheduler submits nothing else meanwhile, and a \
             threadgroup size its pipeline state has not dispatched makes that VkPipeline then";
        /// A sparse texture (`sparse/texture.rs`) on a device whose sparse residency is not
        /// `residencyNonResidentStrict`.
        NonStrictResidency = "non-strict-residency", Precision,
            "a read of a texel no mapped tile holds, by a function or a copy, answers the \
             device's value where Metal's reads zero, and a write to one may be kept where \
             Metal's drops it";
        /// An indirect mapping update of a sparse texture (`sparse/indirect.rs`), whose regions the
        /// host reads.
        HostReadMappingArguments = "host-read-mapping-arguments", Performance,
            "one GPU-to-CPU round trip per batch of such updates: the command buffer's work \
             before it, and every batch submitted before that on any queue, completes before \
             the CPU reads the regions (through a copy for a Private buffer) and binds them; \
             the device's scheduler submits nothing else meanwhile";
        /// A sparse texture (`sparse/emulated.rs`) the device makes no sparse image of (MoltenVK
        /// binds no sparse memory), where no function reads an image's own residency (the device
        /// does not enable `shaderResourceResidency`): an ordinary image whose unmapped tiles the
        /// runtime keeps at zero, which gives Metal's answers.
        EmulatedSparseTexture = "emulated-sparse-texture", Performance,
            "the texture holds its whole image's memory from when it is made, where Metal's backs \
             only its mapped tiles; each batch that changes its mappings zeroes the tiles it \
             unmaps and rewrites its tile map before its work, and each blit that writes it \
             zeroes the unmapped tiles it wrote";
        /// A `Shared` buffer or heap the program makes with the default CPU cache mode, on a
        /// device without unified memory (`DeviceProperties::has_unified_memory`), lies in
        /// host-cached system memory, which the host reads quickly (`memory.rs`'s
        /// `MemoryClass::Shared`), as a shared resource's on a Mac with a discrete GPU.
        SharedInSystemMemory = "shared-in-system-memory", Performance,
            "the GPU reads and writes it across the bus, as a discrete Mac's GPU does a shared \
             resource: coalesced accesses at the bus's speed, and scattered writes far slower \
             (4-byte stores 12 bytes apart at 0.08 GB/s on an RTX 3090, where device-local \
             memory takes them at 850 GB/s)";
        /// `generate_mipmaps` through the device's blit chain, which the runtime's check accepted
        /// for the texture's pixel format within the spread Metal's own GPUs show
        /// (`mipmap.rs`): Metal has no one answer, an Apple M3 Max's and an Apple M4's differing by
        /// up to 1 unit in the last place.
        MipmapsWithinSpread = "mipmaps-within-spread", Precision,
            "each level is the device's linear blit of the one above, which on the check's inputs \
             lay within 1 unit in the last place (2 for 32-bit float) of an Apple M3 Max's or an \
             Apple M4's answer, not at either's exactly; other inputs are not checked";
        /// A depth or stencil texture made in a heap on a device whose driver lays depth and
        /// stencil images apart from the memory they are bound to
        /// ([`metal2vulkan::device::DriverFact::DepthStencilImagesLieApart`]) that no shadow of the
        /// heap's bytes serves: a multisampled texture, which Vulkan copies to and from no buffer,
        /// one whose texels do not fit its bytes, or one with a depth twin. The texture aliases
        /// none of the heap's other resources, where Metal's lies in the heap's bytes.
        HeapDepthStencilApart = "heap-depth-stencil-apart", Precision,
            "the texture's texels lie apart from the heap's bytes: a write through a buffer or \
             texture over its bytes changes none of them, and its writes change none of those \
             resources' bytes, where Metal's texture aliases them";
        /// A colour texture made in a heap on a device whose driver's compressed images hide
        /// writes through an alias
        /// ([`metal2vulkan::device::DriverFact::CompressedImagesHideAliasedWrites`]), whose image
        /// takes host-transfer usage, which the driver keeps uncompressed in the heap's bytes
        /// ([`metal2vulkan::device::DriverFact::HostTransferImagesKeepTheirBytes`]).
        HeapTextureHostTransfer = "heap-texture-host-transfer", Performance,
            "the texture's image is kept uncompressed, so writes through the heap's buffers and \
             textures over its bytes show in it as Metal's do: a clear of it loses the driver's \
             fast clear (334.5 us where a compressed 2048x2048 RGBA8Unorm image's took 4.1 us on \
             an RPL-S), while its copies, sampling and blending measured no slower";
        /// A colour texture made in a heap on such a device whose image takes no host-transfer
        /// usage (the device did not enable `hostImageCopy`, or an adopted device's consumer did
        /// not say so, or the format or shape takes none there): a linear image, which the driver does not compress, where one does all the
        /// optimal image does for the texture.
        HeapTextureLinear = "heap-texture-linear", Performance,
            "the texture's image is linear, so writes through the heap's buffers and textures \
             over its bytes show in it as Metal's do: its reads and writes go without the \
             optimal tiling's locality and compression (3.9 to 9.6 times the host-transfer \
             image's time for copies, clears and draws of a 2048x2048 RGBA8Unorm texture on an \
             RPL-S)";
        /// A colour texture made in a heap on such a device whose image takes neither
        /// host-transfer usage nor linear tiling: an optimal image the driver may compress, a
        /// near-equivalent recorded open.
        HeapTextureHidesAliasedWrites = "heap-texture-hides-aliased-writes", Precision,
            "the driver may compress the texture's image: a write through a buffer or texture \
             over its bytes may leave texels as they were, and its writes may show in those \
             resources' bytes compressed, where Metal's texture aliases them byte for byte";
        /// A colour texture made in a heap on such a device that Metal keeps losslessly
        /// compressed too (`allowGPUOptimizedContents`, no `ShaderWrite`, `PixelFormatView` or
        /// `ShaderAtomic` usage, 16 texels or more a side), under
        /// [`metal2vulkan::device::HeapCompressionPolicy::Compressed`], the default: the
        /// driver's compressed optimal image, its fast clears kept.
        HeapTextureCompressed = "heap-texture-compressed", Precision,
            "the driver may compress the texture's image, as Metal compresses the texture: a \
             write through a buffer or texture over its bytes may leave texels as they were, \
             where Metal's texture reads opaque magenta (0xffff00ff) in every texel of drawn \
             content the write covered and the written bytes where it held only a clear (an M3 \
             Max), and its writes may show in those bytes compressed otherwise than Metal's; \
             HeapCompressionPolicy::Uncompressed keeps the image uncompressed in the heap's bytes, \
             whose clears then lose the driver's fast clear";
        /// A render creation that fails on a binary-archive miss, made from one add's vertex
        /// stage and another add's fragment stage: Metal's vertex key holds the vertex outputs
        /// its compiler finds the fragment reading, which the runtime's lookup does not decode
        /// (`binary_archive::metal_key`).
        CombinedArchiveStages = "combined-archive-stages", Precision,
            "the pipeline is made where Metal may refuse it with a binary-archive miss: Metal \
             misses when the two adds' fragment functions read different vertex outputs";
        /// A texture view is an image of its own, a shadow of the viewed image's texels, where
        /// that image takes no Vulkan view of its format: a view of another component layout on
        /// a device whose Vulkan views reinterpret no image's components (a portability subset
        /// without `imageViewFormatReinterpretation`, or an adopted device whose consumer did not
        /// say it enabled it), or a view of another format than the texture's own and its sRGB
        /// twin's of a texture made without `PixelFormatView` usage (`texture/view_shadow.rs`).
        /// A view in other channels of its image's bits (an `A8Unorm` view of an `R8Unorm`
        /// texture) is the image itself to its bindings, through a swizzle, and takes a shadow
        /// only for a pass that renders into it: Vulkan views the image in no format of the
        /// view's.
        ShadowedTextureView = "shadowed-texture-view", Performance,
            "its texels are copied from its texture's image before each use and back after each \
             write";
        /// `generate_mipmaps` computed as Metal's filter (`mipmap_filter.rs`) where an Apple M3
        /// Max's answer may differ from the chain's (`MipmapFilterChain::unlike_an_m3`): a 2D
        /// texture, array or cube of three levels or more whose M3 Max passes are not known (a
        /// compressed texture's pass from a level more than 128 times wider than tall or taller
        /// than wide), an `A1BGR5Unorm` 2D texture, an `ABGR4Unorm` one of three levels or more, or
        /// a 3D texture of a format summed as half floats, under macOS 26.7's rule
        /// (`MipmapRule::AppleM3MaxPasses`). Under macOS 27.0.1's every level is the M3 Max's.
        MipmapsAsAppleM4 = "mipmaps-as-apple-m4", Precision,
            "each level is the Apple M3 Max's but where its answer is not known: below the first \
             level of a 2D chain whose M3 Max passes are not known an Apple M4's filter, and a \
             few measured values of the 4-bit and 1-bit formats and of 3D half-float sums one \
             unit off";
        /// A render encoder executes an indirect command buffer's commands from a command it
        /// executed them from before with another length, each range given
        /// (`executeCommandsInBuffer:withRange:`) or read from a buffer
        /// (`executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:`;
        /// `render/executions.rs`): Metal's executions then do not each run their own range, and
        /// not the same way from run to run, on an Apple M3 Max and an Apple M4 alike, so no
        /// stable answer exists.
        RepeatedExecutionStart = "repeated-execution-start", Precision,
            "each execution runs its own range, as Metal documents, where Metal's GPU may run \
             one execution with the other's length, so commands run more or fewer times than \
             the ranges name, differently from run to run";
        /// A command predication record (`encode_start_if`, a loop's start, a live DoWhile's
        /// end) whose condition word lies past its buffer's end, past the 256 bytes Metal rounds
        /// a lone buffer up to, or past the end of the heap the buffer lies in
        /// (`predication.rs`): Metal reads whatever its own address space holds there, a
        /// neighbouring buffer's bytes or zero, which depends on where Metal placed memory
        /// unrelated to the buffer.
        PredicationWordPastBuffer = "predication-word-past-buffer", Precision,
            "the record reads its word as zero, where Metal reads what lies at that address in its \
             own address space: zero where nothing is there, another buffer's bytes where Metal \
             placed one";
        /// A heap of the runtime's memory or an indirect command buffer made with
        /// `MTLResourceOptions` bit 24
        /// ([`ResourceOptions::PRIVATE_BIT`](crate::ResourceOptions::PRIVATE_BIT)), or off macOS a
        /// no-copy buffer or heap over the caller's bytes, which Metal makes and keeps the GPU's
        /// writes from the memory of (the M3 Max under macOS 27.0.1, `read_only_view.rs`: what a
        /// shader or the blit engine writes reaches neither the CPU nor a later command buffer,
        /// which read the memory's bytes -- a blit's fill, a kernel's stores, a vertex- or
        /// fragment-stage store, and a store through an address loaded out of an argument buffer,
        /// each completing with no error; a later command of the same command buffer may read them,
        /// as the GPU's caches hold them, which is not stable; while what a render pass's own
        /// machinery writes **faults** the command buffer -- a colour attachment over that memory,
        /// a resolve into it, a depth or stencil attachment in a heap of the bit, and a blit into a
        /// texture over or in it), is
        /// made as without it, and reports its options without it, as Metal's does. On macOS a
        /// no-copy buffer or heap with the bit drops the GPU's writes as Metal's does, its pages
        /// imported through a read-only view of them (`read_only_view.rs`), announcing nothing.
        /// A buffer of the runtime's memory or a texture descriptor with the bit is refused, as
        /// Metal ends the process there; a buffer placed in a heap without the bit is made as
        /// without it, as Metal's is, announcing nothing.
        PrivateResourceOption = "private-resource-option", Precision,
            "the GPU's writes reach the resource's memory, the CPU and later command buffers, \
             where Metal's never reach them";
        /// A depth or stencil texture with `allowGPUOptimizedContents` off sized or made in a heap
        /// (never a device's own texture, whose layout nothing shows), on a device whose driver
        /// keeps depth and stencil images GPU-optimized whatever their create info
        /// ([`metal2vulkan::device::DriverFact::DepthStencilImagesStayGpuOptimized`]), that no
        /// shadow of Metal's tiles in the heap's bytes serves (a multisample texture, a depth twin,
        /// a format whose layout is not measured; `texture/heap.rs`): Metal lays the texture out
        /// without its compression metadata, and the runtime's image keeps it.
        OptimizedDepthStencilTexture = "optimized-depth-stencil-texture", Precision,
            "the texture keeps its GPU-optimized layout: heap_texture_size_and_align answers \
             Metal's size for the texture with the property on, larger than Metal's with it off, \
             so a heap sized by Metal's answers holds fewer such textures";
        /// A texture of an IOSurface plane of more than one row under 16 bytes a row
        /// (`Buffer::new_texture_with_iosurface_plane`, `linear_plan.rs`): Metal's GPU writes its
        /// rows 16 bytes apart and reads every row past the first as memory-dependent bytes
        /// (zeros in a fresh process; after a write, bytes neither the surface's nor written).
        NarrowSurfaceRows = "narrow-surface-rows", Precision,
            "a GPU read of a row past the first reads the bytes the GPU's writes put there, 16 \
             bytes a row apart, where Metal's GPU reads memory-dependent bytes; CPU access and \
             the GPU's writes and first row are Metal's";
        /// A pipeline state's function may read an argument its function constants removed (a
        /// texture's read, sample, gather, query or atomic, or a sample through a removed
        /// sampler; `RuntimeFact::AbsentArgumentRead`), which Metal gives no value: its compiler
        /// drops or folds such a read, and with its debug layer every such read is zero (measured
        /// on an M3 Max and an M4).
        AbsentArgumentRead = "absent-argument-read", Precision,
            "such a read takes zero, Metal's answer under its debug layer, and the command \
             buffer completes, its execution report stating the read (absent_argument_read); \
             without the layer Metal's compiler drops or folds the read instead";
        /// A render pass whose one depth/stencil texture stores and resolves its depth
        /// (`StoreAndMultisampleResolve`) while it resolves its stencil alone
        /// (`MultisampleResolve`), which Metal's validation stops (`resolve::layout_patterned`).
        MixedDepthStencilResolve = "mixed-depth-stencil-resolve", Precision,
            "both aspects resolve and the depth's samples are kept, as the store actions \
             document, and the stencil's samples are kept too, where Metal's GPU overwrites the \
             depth's samples and its resolve texture with a layout pattern of the resolved depth \
             and leaves the stencil's texels (measured on an Apple M4)";
        /// A draw whose functions read or write, other than as an attachment, memory an
        /// attachment of its render pass instance renders, where no copy or snapshot serves the
        /// access (`texture/attached_reads.rs`): a texture reached through a handle or an
        /// argument buffer, a storage binding of a color attachment's texture, the depth or
        /// stencil attachment's texture, a texture buffer or a buffer over its bytes, a texture
        /// whose copy is not served, or an execution's or a capture's command. Announced where the
        /// draw records; the payload names its call.
        AttachmentReadInPlace = "attachment-read-in-place", Precision,
            "the access sees the attachment as the device holds it inside its render pass \
             instance, which Vulkan leaves undefined (`VUID-vkCmdDraw-None-10795`), where Metal's \
             read at the draw's own pixel gives the texel from before the pass";
        /// A sampler state whose `maxAnisotropy`, which Metal's validation stops past 16, plain
        /// Metal samples with an anisotropy past the device's largest (`maxSamplerAnisotropy`):
        /// 17 to 31 in its low five bits, which Metal samples with 32 (`sampler.rs`).
        CappedAnisotropy = "capped-anisotropy", Precision,
            "it filters with the device's largest anisotropy (16 on every venue measured), where \
             Metal filters with 32: a footprint more than 16 times as long as it is wide reads a \
             coarser level than Metal's (one level coarser at 32 to 1 and beyond)";
        /// A sampler state with anisotropy, no mip filter (`NotMipmapped`) and different
        /// minification and magnification filters, on a device that samples through Metal's own
        /// sampler (MoltenVK on Apple's GPU, `DriverDeviation::MetalSampler`): Vulkan has no
        /// sampler without mipmapping, so the device's sampler takes a nearest mip filter with the
        /// level clamp `[0, 0.25]`, which keeps Metal's choice of filter, and Metal's anisotropic
        /// filter reads no such clamp as Metal's `NotMipmapped` (`sampler.rs`). Announced once,
        /// where the device makes the state; the payload names its anisotropy and filters. A
        /// constexpr sampler of that shape records `NotMipmappedAnisotropicFilter` instead.
        NotMipmappedAnisotropicSampler = "not-mipmapped-anisotropic-sampler", Precision,
            "a sample at an implicit level or with gradients through it filters another \
             anisotropic footprint than Metal's at `maxAnisotropy` 2 to 16";
        /// A command whose function reads texture handles from memory (an argument buffer, or
        /// any memory holding a texture's resource ID) has a texture the runtime holds as raw
        /// blocks for copies (a PVRTC texture on a device without its format, `pvrtc.rs`)
        /// made resident, or bound in a stage that reads handles: the handle's element is the
        /// neutral texture, as the runtime does not know which of the textures it reaches the
        /// function reads.
        UnreadableHandleTexture = "unreadable-handle-texture", Precision,
            "a read or sample of the texture through its handle reads the neutral texture where \
             Metal's decodes its blocks; a function that reads it at a slot is refused \
             (Unsupported(Format))";
        /// A texture that a heap array's class fits, but whose format the array cannot view (one
        /// the device cannot sample, or store without a format, or one whose raw texels need its
        /// function's decode): the device's heap set holds the class's neutral at its ID, which a
        /// command that names the texture refuses instead (`heap_sets.rs`).
        UnnamedTextureUnviewable = "unnamed-texture-unviewable", Precision,
            "a function reading the texture through its ID, in a command that does not name it \
             (bind it, or make it resident), reads the neutral texture where Metal's reads the \
             texture";
        /// A shadowed texture (a buffer's texels or a view's, which a shadow image holds as the
        /// encoders that name it copy them in, `shadow.rs`) in the device's heap set
        /// (`heap_sets.rs`). An encoder names it by binding it or making it resident, and so does
        /// each later compute or render encoder of a command buffer that made it resident
        /// (Metal's residency lasts the command buffer). Announced at the first command that binds
        /// a heap set holding its view while neither naming it nor making it resident; one that
        /// does has its shadow copied in, and a set no command binds announces nothing.
        UnnamedShadowedTexture = "unnamed-shadowed-texture", Precision,
            "a function reading the texture through its ID, in a command that does not name it, \
             reads the texels an encoder that named it last copied in (undefined before any \
             did), where Metal's reads the texels as they stand";
        /// A live texture whose resource ID is past the elements of a pipeline's set-1 heap
        /// arrays (their count, the device's update-after-bind limits shared among the
        /// pipeline's arrays), so the texture table a command binds misses it (`heap_sets.rs`).
        UnnamedTexturesPastCapacity = "unnamed-textures-past-capacity", Precision,
            "a function of such a pipeline reading the texture through its ID reads the neutral \
             texture where Metal's reads the texture; a command that names it is refused \
             (Bind(HeapTooLarge))";
        /// A heap set on a device without `descriptorBindingUpdateUnusedWhilePending`, which
        /// writes no set a pending command buffer binds: each command buffer binds a copy of its
        /// own, written as it first binds the set and until it commits (`heap_sets.rs`).
        HeapSetCopied = "heap-set-copied", Precision,
            "each command buffer's first command that binds the set writes a copy of it (every \
             element a texture's ID may name, and every sampler array's, per array); a function \
             reading a texture through its ID reads the texture that held the ID as its command \
             buffer committed, and one reading a sampler state the state that held it as the \
             command buffer's first batch was submitted (the neutral where none did, or where it \
             was released before that submission), where Metal's reads the one holding it as \
             the function runs";
        /// A draw of a render pipeline state in a pass with a color at no place of the state's
        /// per-sample record, under a state color that blends through a destination factor of
        /// the fragment's output (`SourceAlpha`, `OneMinusSourceAlpha`, `SourceColor`, ...) and
        /// reads the destination no other way (`color_reformat.rs`).
        UncoveredColorKept = "uncovered-color-kept", Precision,
            "the pass's color keeps its contents where the draw covers it, as Metal keeps them \
             for an output it computes; where Metal's compiler folds the output to a constant that \
             zeroes the factor (an alpha of the literal 1 under `OneMinusSourceAlpha`), Metal \
             stores zero there (measured on an Apple M4)";
        /// A blit copy into a multisampled 32-bit float depth texture from a colour one, a render
        /// pass's separate multisampled 32-bit float depth texture copied to and from its scratch
        /// image (`depth_stencil_scratch.rs`), or a heap's such texture held as a shadow of its
        /// samples in Metal's tiles, which takes the heap's bytes (`texture/heap_tiled.rs`; then a
        /// value the draw changes is written back to the heap's bytes changed, announced as the
        /// texture is made), by the sample-rate `FragDepth` draw (`sample_copy.rs`,
        /// `depth_copy.rs`), on a device whose copies go by draws and whose
        /// driver's fragment depth is not measured to store every value unchanged
        /// (`depth_copy::drawn`): measured to change some values (MoltenVK, NVIDIA's), or not
        /// measured at all (any other driver, and lavapipe releases not measured). Both copies
        /// take the one decision and announce through `depth_copy::announce_changes`. The reason
        /// names the values that driver changes, or, on a driver not measured, those Vulkan leaves
        /// to it (outside `[0, 1]`, denormals, -0).
        DrawnDepthCopy = "drawn-depth-copy", Precision,
            "each sample's depth is written by a fragment-depth draw, where Metal copies its \
             bytes: the values the driver's fragment depth changes or may change, which the \
             reason names, can land changed; every other value lands as Metal's copy leaves it";
        /// A texture bound by a slot as a storage image without `ShaderWrite` usage, or as a
        /// sampled one without `ShaderRead`, whose image lacks that usage because adding it
        /// would cost the image its size or the driver's compression: the binding takes an image
        /// of its own with both shader usages, a shadow of the texture's texels
        /// (`texture/usage_shadow.rs`). Also a texture without `RenderTarget` usage that a render
        /// pass renders as an attachment, through a scratch image that has it
        /// (`attachment_stand_in.rs`).
        UndeclaredUsageShadow = "undeclared-usage-shadow", Performance,
            "its texels are copied into an image of its own before each command that binds it \
             beyond its declared usage, and back after each write through it";
        /// A texture of `RGB10A8_2P_XR10` or its sRGB twin made in a heap (`texture/heap.rs`)
        /// whose planes no shadow of the heap's bytes holds: one whose moves pass the device's
        /// limits (`texture/heap_two_planes.rs`). Its one image holds each texel's
        /// colour word and alpha byte side by side, more than Metal's two planes take in the
        /// heap, so it lies in memory of its own.
        HeapTwoPlaneTexture = "heap-two-plane-texture", Precision,
            "the texture's size and alignment answers are Metal's and its reads, writes and \
             copies as outside a heap, but its texels lie in memory of its own, apart from the \
             heap's bytes: a buffer or texture over its range of the heap neither sees nor \
             changes them, where Metal's texture keeps its two planes there";
        /// A texture of `RGB10A8_2P_XR10` or its sRGB twin made in a heap whose first level's
        /// planes lie in the heap's bytes (`texture/heap_two_planes.rs`), of more than one level,
        /// or a 1D array: Metal's paths address the texture's levels past the first at bytes that
        /// differ by path (its CPU copies at its first level's rows from past that level, its
        /// blits and an array's GPU writes over the first level's bytes, a 2D texture's GPU
        /// writes nowhere, its GPU reads at other bytes again), and a 1D array's GPU reads every
        /// texel at the texture's first word. The runtime keeps the first level's slices in the
        /// heap's bytes, as Metal's CPU copies and GPU writes lay them, and the other levels in the
        /// texture's image alone.
        HeapTwoPlaneLayout = "heap-two-plane-layout", Precision,
            "the first level's texels lie in the heap's bytes as Metal's CPU copies and GPU \
             writes lay them, and read as written; the later levels' texels lie apart from the \
             heap's bytes, and a 1D array's GPU reads read each texel, where Metal's paths \
             address those texels at bytes that differ by path";
        /// A MoltenVK device at a release whose `pipelineCacheUUID` layout the runtime has not
        /// read (`device/argument_buffers.rs`): whether MoltenVK binds through Metal argument
        /// buffers is unknown, so a device without them is not refused (as at a read release).
        /// Announced when the device is made.
        MoltenVkArgumentBuffersUnread = "moltenvk-argument-buffers-unread", Precision,
            "whether MoltenVK binds through Metal argument buffers is not read at this release: \
             with MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS off, Metal's compiler can crash on large \
             fragment functions' pipelines, and argument buffers hold fewer resources than \
             Metal's";
        /// A blit between two textures of `RGB10A8_2P_XR10` or its sRGB twin
        /// (`two_planes.rs`) whose answer is not Metal's measured one: one the runtime copies
        /// texel for texel (textures of more than one level or slice, a copy within one
        /// texture, a level larger than a storage buffer), or one whose rows overlap in the
        /// destination, or of a texture narrower than 4 texels, where Metal's blit writes in an
        /// order that varies or reads bytes from outside the planes, or one reading bytes outside
        /// the source's texels that an earlier blit wrote there (or a heap's texture's), which
        /// Metal's blit keeps and reads back. Into a heap's texture whose planes lie in the
        /// heap's bytes (`two_planes/heap_bytes.rs`): a copy whose alpha bytes pass the heap's end,
        /// which Metal's blit writes past it, one reading a source's alpha bytes past its heap's
        /// end, and one whose bytes pass a storage buffer, copied texel for texel.
        TwoPlaneTextureCopy = "two-plane-texture-copy", Precision,
            "where Metal's answer was measured (2D textures of one level and one slice) the \
             copy moves both planes' bytes as Metal's blit does, at its own rows; here it \
             copies both words of each texel, where Metal's blit moves bytes at those rows and \
             so loses or moves texels, or it keeps the later row where the copy's rows overlap \
             (Metal's blit writes them in an order that varies), or it reads as 0 the bytes \
             Metal's blit reads from outside the texels: from outside the planes below a width \
             of 4, and those an earlier blit left in a row's padding or past a plane, which \
             Metal keeps";
        /// A draw in a render pass that loads a color attachment of an extended-range or padded
        /// 10-bit format (`BGR10_XR`, `BGRA10_XR`, `RGB10A8_2P_XR10`, `R10Unorm_X6`,
        /// `RG10Unorm_X12` and their sRGB twins, and `RGB10A2Unorm_sRGB`) and does not discard it
        /// at the store, where the
        /// pass's load pass (`render/unused_bits_load.rs`) does not take it (`render.rs`): a
        /// multisampled attachment, a texture held as a shadow of another image or of a heap's
        /// bytes, a sparse one, or one whose image the device does not copy both ways (a padded
        /// format takes no multisample texture). Metal's pass, once it draws, writes each 32 x 32 tile a
        /// draw reaches back with the bits its codes leave unused made canonical, the texels no
        /// fragment writes and the channels a write mask leaves out too (on an Apple M4 and an
        /// Apple M3 Max), where such an attachment keeps the bytes its draws do not write.
        /// Announced at every draw, an indirect command buffer's executed commands' too, naming
        /// every such attachment of the pass.
        ExtendedRangeBitsKept = "extended-range-bits-kept", Precision,
            "the texels the pass's draws do not write keep their bytes; Metal's pass keeps their \
             codes but rewrites the bits they leave unused in each tile a draw reaches \
             (BGR10_XR's bits 30 and 31 to ones, BGRA10_XR's low six bits of each field to \
             zeros, the two-plane color word's top two bits to zeros) and clamps BGRA10_XR's \
             alpha code to [384, 894]: a texel whose bytes no Metal write left (uploaded bytes, \
             or an alpha code past 894) keeps bytes Metal's pass would change";
        /// A draw in a render pass that loads a single-sampled color attachment of a signed
        /// normalized format (`R8Snorm`, `RG8Snorm`, `RGBA8Snorm`, `R16Snorm`, `RG16Snorm`,
        /// `RGBA16Snorm`) and does not discard it at the store (`render.rs`). Metal's pass, once
        /// it draws, writes each 32 x 32 tile a draw reaches back through its channels' values,
        /// so a channel's minimum code (-128, or -32768), which reads as -1.0 as the code above
        /// it does, is stored as that code (-127, or -32767), in the texels no fragment writes
        /// too (measured on an Apple M3 Max under macOS 27.0.1: every such code of each tile a
        /// draw reaches, none in other tiles, none in a pass that draws nothing or through a
        /// blit), where the device's pass keeps the bytes its draws do not write. Announced at
        /// every draw, an indirect command buffer's executed commands' too, naming every such
        /// attachment of the pass. Not where `DriverFact::TileStoreIsMetals` holds (MoltenVK's
        /// pass is a Metal pass, whose tile store rewrites the codes as Metal's does), nor under
        /// [`metal2vulkan::device::TileStorePolicy::Exact`] for an attachment the load pass
        /// (`render/unused_bits_load.rs`) rewrites.
        SnormMinimumKept = "snorm-minimum-kept", Precision,
            "the texels the pass's draws do not write keep their bytes; Metal's pass rewrites a \
             channel's minimum code (-128, or -32768 for 16 bits) as the code above it (-127, \
             -32767) in each tile a draw reaches: a texel whose bytes no Metal write left \
             (uploaded or copied bytes) keeps a code Metal's pass would change";
        /// A draw in a render pass that loads a single-sampled `RGB9E5Float` color attachment and
        /// does not discard it at the store (`render.rs`). Metal's pass, once it draws, writes
        /// each 32 x 32 tile a draw reaches back through its texels' values, so a word whose
        /// largest mantissa is below 256 under an exponent above 0 is stored normalized: its
        /// mantissas shifted up a bit as its exponent steps down one, until the largest reaches
        /// 256 or the exponent 0 (measured on an Apple M3 Max under macOS 27.0.1: the format
        /// scout's draw over uploaded words, 22 of 143 untouched texels of the reached tile
        /// normalized, none other changed), where the device's pass keeps the bytes its draws do
        /// not write. Announced at every draw, an indirect command buffer's executed commands'
        /// too, naming every such attachment of the pass. Not where `DriverFact::TileStoreIsMetals`
        /// holds (MoltenVK's pass is a Metal pass), nor under
        /// [`metal2vulkan::device::TileStorePolicy::Exact`] for an attachment the load pass
        /// (`render/unused_bits_load.rs`) rewrites.
        SharedExponentKept = "shared-exponent-kept", Precision,
            "the texels the pass's draws do not write keep their words; Metal's pass normalizes \
             a word whose largest mantissa is below 256 (its exponent above 0) in each tile a \
             draw reaches: a texel whose bytes no Metal write left (uploaded or copied words) \
             keeps a word Metal's pass would change, of the same value";
        /// A blit between a buffer and a level of `RGB10A8_2P_XR10` or its sRGB twin whose width
        /// is not a multiple of 4 (`two_planes.rs`), where Metal's blit addresses the level's rows
        /// `⌊4 w / 16⌋ · 16` bytes apart, short of where they lie, and so moves texels into and
        /// out of other rows and the bytes between them. A 2D texture or array of one level is
        /// blitted at Metal's rows (`two_planes/buffer_rows.rs`), the row whose bytes Metal's GPU
        /// leaves where two meet standing; announced here are a blit into a texture 32 texels wide
        /// or wider whose rows meet, which keeps the earlier row's bytes where Metal's GPU writes
        /// the two in an order that varies from run to run (narrower, `two-plane-buffer-blit-order`),
        /// and a blit out of a texture of the
        /// device's memory that reads bytes between its rows after a copy from another texture
        /// wrote some there, which reads them as its fills and blits with a buffer left them (held
        /// apart, `Texture::between_rows`). Any other such texture (more levels, a view, 1D, 3D,
        /// a surface's) moves the texels at the rows the texture holds them in, announced.
        TwoPlaneBufferBlit = "two-plane-buffer-blit", Precision,
            "where rows meet in a blit into a texture 32 texels wide or wider the earlier row's \
             bytes stand, where Metal's GPU writes them in an order that varies from run to run, \
             at some sizes never all the earlier row's; a blit out of a \
             texture of the device's memory reads the bytes between its rows that a copy from \
             another texture wrote as its fills and blits with a buffer left them, where Metal's \
             reads the copy's back; and a texture of more than one level, a \
             view, a 1D or 3D texture or a surface's moves its texels at the rows it holds them in, \
             where Metal's blit addresses its rows floor(4w/16)*16 bytes apart";
        /// A blit from a buffer into a level of `RGB10A8_2P_XR10` or its sRGB twin narrower than 32
        /// texels, at a width not a multiple of 4, whose rows meet between Metal's groups of 32
        /// rows of the region (`two_planes/buffer_rows.rs`): Metal's GPU writes the two rows there
        /// in an order that varies from run to run (the earlier row's bytes at 556 of 560 such
        /// meetings in our probes), and in its fixed order at every other meeting. The runtime
        /// keeps the earlier row's bytes there and Metal's order elsewhere: Metal's most frequent
        /// whole outcome.
        TwoPlaneBufferBlitOrder = "two-plane-buffer-blit-order", Precision,
            "where the rows of two of Metal's groups of 32 meet the earlier row's bytes stand, \
             where Metal's GPU leaves either row's from run to run";
        /// A render pipeline state whose vertex function runs simdgroup functions on a driver
        /// measured to run a vertex function's subgroup operations as Metal does though its
        /// `subgroupSupportedStages` leaves the vertex stage out
        /// ([`metal2vulkan::error::DriverDeviation::MetalVertexSimdgroups`]): its pipeline
        /// requires `VkProperty::MetalVertexSimdgroups` in place of the vertex stage's subgroups.
        UnadvertisedVertexSimdgroups = "unadvertised-vertex-simdgroups", Validity,
            "the device runs them as Metal does, but its driver leaves the vertex stage out of \
             subgroupSupportedStages, so Vulkan's validation layer reports each shader module of \
             the function (VUID-RuntimeSpirv-None-06343)";
        /// A descriptor set layout the device's driver refuses through
        /// `vkGetDescriptorSetLayoutSupport` though it makes and runs it, a heap's set 1 past
        /// `maxPerSetDescriptors` ([`metal2vulkan::error::RuntimeDriverDefect::SetLayoutSupportUnderstated`]):
        /// the runtime makes it there as the driver runs it (`pipeline/objects.rs`).
        SetPastLayoutSupport = "set-past-layout-support", Validity,
            "the device runs the set as Metal runs the argument buffer it is encoded in, but its \
             driver's vkGetDescriptorSetLayoutSupport refuses the layout, so Vulkan's validation \
             layer reports each such layout (VUID-vkCreateDescriptorSetLayout-support-09582)";
        /// An indirect command naming the ID of a pipeline state the program released is
        /// executed where the runtime cannot run what Metal runs (`indirect_command_buffer.rs`,
        /// `released_programs.rs`). A command holds the program of the state the CPU set on it,
        /// and runs it while its record names the state's ID, whoever took the ID since, as
        /// Metal's command the CPU encoded does; a record the GPU wrote runs the state that held
        /// the ID as the GPU last could write the buffer's records, released since or not, as
        /// Metal's GPU write resolves the ID to the state that took it last and its record runs
        /// that state after: those are announced nowhere. Announced: a record the GPU wrote whose
        /// ID's holder then was a state of the command's kind the program released, whose program
        /// the buffer does not keep (released before any command buffer that may write the
        /// buffer's records was encoded, or past the ID's last four holders), which Metal runs, so
        /// the command runs nothing; and in a host step of records the GPU may write, a record
        /// naming the ID of the state the CPU set, which the program released, where the GPU could
        /// write the records while another state held the ID, as the step cannot tell the GPU's
        /// write of that ID from the CPU's, where Metal runs the other state for the GPU's. A
        /// state of the other kind holding the ID runs nothing, as on Metal, unannounced.
        /// Announced by the execution alone, where the runtime records it: in `commit` of its
        /// command buffer, on the committing thread, for one recorded from the CPU's records; in
        /// the host step, on the scheduler's thread, for one recorded there (records the GPU may
        /// write, a range a buffer holds, or a render pass whose pipelines were still being made
        /// at the commit). Reading the commands (`IndirectCommandBuffer::commands`) announces
        /// nothing.
        ReleasedCommandState = "released-command-state", Precision,
            "a command whose record the GPU wrote naming a state released before its buffer \
             could be written runs nothing, where Metal runs that state's program; one naming the \
             ID of the state set on it runs its program, as Metal runs a command the CPU encoded, \
             where for a record the GPU wrote while another state held the ID Metal runs that \
             state (nothing, for a state of the other kind)";
        /// A pipeline stage's filtered sample of a texture slot bound to a two-plane YCbCr
        /// texture does not take Metal's exact filter of the format: the stage records
        /// `PrecisionNote::HalfSampleFilter` or `FloatSampleFilter` under the slot's bound format
        /// (a format whose filter is not measured, a swizzle that moves its channels, a linear
        /// mip filter, or a sample the emulation does not reach), or, on a device that samples
        /// through Metal's own sampler under `YCbCrFilterPolicy::Device` (no slot variant), under
        /// any two-plane YCbCr format bound there, whose converted texels Metal's sampler filters
        /// as an `RGBA16Float` or `RGBA32Float` texture's: within one step of the format's grid of
        /// Metal's filter of the format, a `half` result within one `half` unit in the last place
        /// (plain Metal on an Apple M3 Max under macOS 27.0.1, `YCBCR8_420_2P` and its `_sRGB`
        /// twin against such a texture of the same texels). Announced, in place of those notes'
        /// own notices for the slot, where the specialized pipeline is made, or, without the
        /// variant, at a draw or dispatch that binds such a texture there (through a runtime
        /// sampler, whatever its word states), naming the slot and the format.
        YCbCrFilterNotExact = "ycbcr-filter-not-exact", Precision,
            "the texels hold Metal's reads exactly, but the sample filters them as the device \
             does, off the grid Metal sums the format's steps on, so a filtered result can miss \
             Metal's by a step of that grid or more";
        /// A render pass that draws loads and stores a rotated texture of rotation 4, whose load
        /// is the plane transposed and whose store is rotation 1's (`rotated.rs`): every tile of
        /// the frame lands mirrored, where Metal loads and stores only the 32x32 tiles a draw's
        /// primitives reach and keeps the texels of the others in place.
        RotatedLoadTiles = "rotated-load-tiles", Precision,
            "the tiles no draw reaches land mirrored as the tiles it reaches do, where Metal keeps \
             their texels in place";
        /// A render pass that draws loads and stores a two-plane YCbCr texture
        /// (`metal::format::ycbcr_renders`) whose draws reach the tiles their primitives touch
        /// (`ycbcr.rs`): the pass stores every 32x32 tile of the rectangle bounding its draws'
        /// scissors, each texel's codes re-encoded from the texel it loaded (its chroma the
        /// filtered pairs), where Metal stores only the tiles a draw's primitives reach (the M3
        /// Max under macOS 27.0.1) and keeps the planes' bytes of the others. Announced at the
        /// pass's end, naming the attachment.
        YCbCrLoadTiles = "ycbcr-load-tiles", Precision,
            "in the tiles no draw reaches the planes' codes are re-encoded from the texels the \
             pass loaded, the chroma from its filtered pairs, where Metal keeps their bytes";
        /// A draw that blends into a color attachment of a two-plane YCbCr texture
        /// (`metal::format::ycbcr_renders`, `ycbcr.rs`): the device blends the pass's floats,
        /// where Metal's tile blends halves, an 8-bit format's in half arithmetic (the source and
        /// its alpha rounded to `half`, each product, and a fused sum), a 10-bit one's in `float`
        /// rounded toward zero to `half` (the M3 Max under macOS 27.0.1, every code of 4096
        /// blended texels matched by those rules). Announced at the draw, naming the attachment.
        YCbCrBlendRounding = "ycbcr-blend-rounding", Precision,
            "a blended texel's codes can miss Metal's by one, where Metal's blend in half \
             arithmetic rounds otherwise than the device's in float, or over a texel an earlier \
             draw of the pass wrote (Metal holds it as a half)";
        /// A command buffer's command named a buffer over the caller's host pages (a binding, a
        /// blit's operand, an attachment, a texture over the pages, an indirect command's
        /// resolved buffer) that was rebound (`rebind_bytes_no_copy`) after the command was
        /// encoded and before the command buffer was first submitted
        /// (`CommandBufferError::EncodedUseRebound`, `command_buffer/first_submission.rs`), in an
        /// encoder the rebind could not encode again (a command predication's compute encoder
        /// that shares its unit of predication with other encoders, holds a loop or leaves work
        /// out; a render pass a host step records, or whose end captured its draws' tile reach; work
        /// recorded on the GPU alone that no longer runs there; a call that answers otherwise than
        /// at its encoding; `replay.rs`). Its reason names where the first such submission found
        /// the use (the encoder, and its own commands or work it left for later) and why no rebind
        /// encoded or recorded it again. Announced once, as the first such submission is made, on
        /// the scheduler's thread, which makes it: `DeviceOptions::announce_each` does not repeat
        /// it, as no program call or host step meets it.
        EncodedUseRebound = "encoded-use-rebound", Precision,
            "each of Metal's submissions of the command buffer holding such a command runs \
             nothing and the command buffer fails (EncodedUseRebound), where Metal resolves the \
             use as it submits the command buffer and reaches the pages the buffer lies in then; \
             the rebind encodes compute, blit and render encoders again, but not this one";
        /// A command buffer that bound a heap set, whose commands may reach any texture through
        /// its resource ID without naming it, was submitted while a texture's bytes (a no-copy
        /// buffer's or heap's, the texture over them or in it) were detached
        /// (`detach_backing`) after a submitted command buffer made them resident. Announced
        /// once per device, at the first such submission, on the thread that makes it, with the
        /// first texture's ID ([`DegradeNotice::object_id`]), whatever
        /// `DeviceOptions::announce_each` says; each texture counted once
        /// ([`Device::degraded_objects`](crate::Device::degraded_objects)).
        StaleDetachedReach = "stale-detached-reach", Precision,
            "a read of the texture that does not name it reads zero bytes (the stand-in its \
             detached bytes present), where Metal's reads the pages the program gave back, \
             which a correct program never relies on";
        /// A command buffer that bound a heap set, whose commands may reach any texture through
        /// its resource ID without naming it, was committed while a texture of storage usage
        /// was over detached bytes (`detach_backing`; made before the detach or while
        /// detached): its stand-in has a sink its storage-class elements name. Announced once
        /// per device, in the first such `commit`, with the first texture's ID
        /// ([`DegradeNotice::object_id`]), or for each texture, in its first such `commit`,
        /// where `DeviceOptions::announce_each` asks; each texture counted once
        /// ([`Device::degraded_objects`](crate::Device::degraded_objects)).
        DetachedStorageKeepsWrites = "detached-storage-keeps-writes", Precision,
            "a read of the texture through a storage-class element (read or read_write access) \
             that does not name it reads what an earlier such write left in the stand-in's \
             sink, where Metal drops those writes and reads zero; a sampled read, and a size \
             query, are Metal's";
        /// A device made, with no level named in its options, on a macOS host whose release has
        /// no measured [`MetalLevel`](crate::MetalLevel): it answers as the newest measured
        /// release below the host's (`metal_level.rs`). Announced when the device is made; the
        /// payload names the host's release and the level.
        MetalLevelBelowHost = "metal-level-below-host", Precision,
            "where Metal's answers differ by release (which pixel formats a texture descriptor \
             takes and Metal describes, whether a run of compute encoders' first indirect \
             dispatch of another dispatch type ends the process), the device gives the older \
             release's: it can stop or refuse a texture or a dispatch the host's Metal makes, and \
             never makes one the older release stops";
        /// A render pass that loads a color attachment of an extended-range, padded 10-bit or
        /// bfloat format (`BGR10_XR`, `BGRA10_XR`, `RGB10A8_2P_XR10`, `R10Unorm_X6`,
        /// `RG10Unorm_X12` and their sRGB twins, `RGB10A2Unorm_sRGB`, `R16Bfloat`, `RG16Bfloat`
        /// and `RGBA16Bfloat`), or under
        /// [`metal2vulkan::device::TileStorePolicy::Exact`] of a
        /// signed normalized format or `RGB9E5Float`, stores it and draws, whose load pass
        /// (`render/unused_bits_load.rs`) rewrites every 32 x 32 tile of its render area: Metal's
        /// pass rewrites only the tiles a draw's primitives reach (on
        /// an Apple M4 and an Apple M3 Max, alike), and every tile only where a color
        /// attachment of the pass is cleared and stored (announced where none is, or where the
        /// pass renders more than one layer). Announced at the pass's end, naming its
        /// attachments the load pass took.
        UnreachedTilesRewritten = "unreached-tiles-rewritten", Precision,
            "in tiles no draw reaches the pass rewrites what Metal keeps: the unused bits made \
             canonical (BGR10_XR's bits 30 and 31 ones, BGRA10_XR's low six bits of each field \
             and the two-plane color word's top two bits zeros) and BGRA10_XR alpha codes \
             clamped to [384, 894], so an out-of-range raw alpha reads back changed, a padded \
             word's low six bits zeroed (an sRGB twin's code moved through Metal's table), a \
             bfloat code moved through half (out of half's range to an infinity or zero, a NaN \
             to 0x7fc0) and, under the exact tile store, a signed normalized minimum code raised \
             and an RGB9E5Float word normalized; the tiles a draw reaches are Metal's";
        /// A draw in a render pass over a rasterization-rate map of no layers that was made after
        /// the device released a map of one or more layers, or reset from one of layers
        /// (`rate_mapped_pass.rs`). The draw
        /// writes nothing, where Metal draws: an Apple M4 and an Apple M3 Max drew a 64-pixel
        /// map's physical rows 0 to 15 over a 64-pixel target at the positions of a pass with no
        /// map, whatever map was released. Announced at the draw.
        ZeroLayerMapAfterRelease = "zero-layer-map-after-release", Precision,
            "the draws write nothing, where Metal draws rows of the pass (a 64-pixel map's rows 0 \
             to 15, at the positions of a pass with no map)";
        /// A render pass over a rasterization-rate map that was reset
        /// (`RasterizationRateMap::reset_using_descriptor`) after the pass took it and before
        /// its command buffer was submitted, in an encoder the runtime could not encode again
        /// (it kept no log: a render encoder opened while a buffer of the device was detached,
        /// or one that left work for a host step;
        /// `rate_mapped_pass.rs`). The pass draws by the map as it took it, where Metal's GPU
        /// reads the map as it runs the pass (the M3 Max under macOS 27.0.1). Announced at the
        /// commit, or at the reset where the command buffer was committed.
        RateMapResetUnencoded = "rate-map-reset-unencoded", Precision,
            "the pass draws by the map as it took it, where Metal's GPU draws by the map as \
             reset";
        /// A texture made in a heap over host pages (`new_heap_with_bytes_no_copy`), or a depth
        /// or stencil texture made in any heap as a shadow of Metal's tiles (on a driver that
        /// lays such images apart from their memory or keeps them GPU-optimized: MoltenVK), that
        /// Metal keeps losslessly compressed (`allowGPUOptimizedContents`, the shorter side of
        /// level 0 16 texels or more, and a colour format of neither `ShaderWrite` nor
        /// `PixelFormatView` usage, or a depth or stencil format): its bytes on Metal are no
        /// layout of its texels, so the runtime lays them out as its own, its uncompressed tiles
        /// where it tiles them (`texture/heap_tiled.rs`). A multisample texture so (its sample
        /// grid's shorter side 16 or more) holds its samples in its uncompressed tiles, or, where
        /// the device does not move its samples to and from a buffer, lies in memory of its own,
        /// the heap's bytes keeping what they held. Also such a texture whose image lies in a heap
        /// over the device's memory: the image, of the runtime's usage, need not be compressed as
        /// Metal's.
        /// Announced once per device, in the call that made the first such texture, with its ID
        /// ([`DegradeNotice::object_id`]), or for each texture where `DeviceOptions::announce_each`
        /// asks; each texture counted
        /// ([`Device::degraded_objects`](crate::Device::degraded_objects)).
        HeapTextureLosslessLayout = "heap-texture-lossless-layout", Precision,
            "the texture's reads, writes and copies are Metal's, but its bytes in the heap are \
             not Metal's compressed bytes (its texels or samples in the runtime's layout, or a \
             multisample texture's image in memory of its own, the heap's bytes keeping what \
             they held): a buffer or texture over them reads and writes other bytes than Metal's \
             would";
        /// A texture of a block-compressed format made in a heap over host pages whose
        /// placement the device's Metal level does not measure: Metal places a format family's
        /// blocks in each level by a rule that differs by family and by release
        /// (`texture/heap_tiled.rs`), measured for BC, ETC2, EAC and ASTC under macOS 27.0.1 (an
        /// Apple M3 Max and an Apple M4 alike) and 26.7 (an Apple M3 Max), and for PVRTC under
        /// 27.0.1 (an Apple M3 Max), where those lie as Metal lays them out, and not for PVRTC
        /// under 26.7, whose blocks lie in the runtime's layout. Its size and alignment answers
        /// are its device's. Announced once per device, in the call that made the first such
        /// texture, with its ID ([`DegradeNotice::object_id`]), or for each texture where
        /// `DeviceOptions::announce_each` asks; each texture counted
        /// ([`Device::degraded_objects`](crate::Device::degraded_objects)).
        HeapTextureBlockCompressedLayout = "heap-texture-block-compressed-layout", Precision,
            "the texture's reads, writes and copies are Metal's, but its blocks lie in the \
             heap's bytes in the runtime's layout, where Metal tiles them: a buffer or texture \
             over them meets the blocks at other bytes than on Metal";
        /// A texture made in a heap over host pages that Metal lays out tiled where the runtime
        /// does not (`texture/heap_tiled.rs`): a layout not measured (a format whose layout in
        /// Metal's heaps is not probed), an image that holds other bytes than Metal's elements,
        /// moves past the device's limits, or a multisample texture whose samples the device does
        /// not move to and from a buffer (a 16-bit depth where its multisampled copies go by
        /// draws whose codes the driver is not measured to keep), in memory of its own. Also a
        /// texture made in any other heap (over the device's
        /// memory) whose bytes there are not Metal's layout (`texture/heap.rs`): its image bound
        /// there in the driver's layout, on a driver whose colour images are not measured to lie as
        /// Metal's heap textures
        /// ([`metal2vulkan::device::DriverFact::ColorImagesLieAsMetalHeapTextures`], and
        /// [`metal2vulkan::device::DriverFact::MultisampleColorImagesLieAsMetalHeapTextures`] for
        /// a multisample one) or of a depth or stencil texture there; its image a shadow of the
        /// bytes in rows; or a layout not measured. The notice's reason names which (the first
        /// texture's). Announced once per device, in the call that made the first such texture,
        /// with its ID ([`DegradeNotice::object_id`]), or for each texture where
        /// `DeviceOptions::announce_each` asks; each texture counted
        /// ([`Device::degraded_objects`](crate::Device::degraded_objects)).
        HeapTextureUntiledLayout = "heap-texture-untiled-layout", Precision,
            "the texture's reads, writes and copies are Metal's, but its texels lie in the \
             heap's bytes in the runtime's layout, where Metal tiles them: a buffer or texture \
             over them meets the texels at other bytes than on Metal";
        /// A render pass's draws test the depth in the fragment shader (`depth_stand_in.rs`): a
        /// fragment function that stores a color through the pixel interlock before Vulkan's tests,
        /// in a draw whose depth test may drop the fragment, tests the fragment's depth itself
        /// against a stand-in of the depth, so it stores only what passes, as Metal stores only
        /// after the tests. (Where the draw's stencil test acts too, it runs that first, against a
        /// stand-in of the stencil.)
        ShaderSideDepthTest = "shader-side-depth-test", Performance,
            "a 32-bit float stand-in image of the depth per pass, which the depth is copied into \
             before each run of such draws and back from after it (through a staging buffer at \
             one sample, image to image or by a compute split and a fragment-depth draw at \
             more), each run in a render pass instance of its own under depth `ALWAYS` without \
             writes, its fragment shader testing and writing the stand-in in the pixel \
             interlock; where the stencil test acts too, an 8-bit stand-in of the stencil per \
             pass, copied in and back the same way (image to image at more than one sample) \
             under the stencil test off";
        /// A blit copy between overlapping regions of one texture's texels, other than a region
        /// onto itself (`blit_texture.rs`): Metal's copy reads as `memmove`'s for regions up to a
        /// few hundred texels across, but past that some texels read what another part of the
        /// copy already wrote, differently from run to run. The runtime gives
        /// `memmove`'s answer every time: the source regions go through a scratch buffer, read
        /// whole before any destination texel is written.
        OverlappingTextureCopyAsMemmove = "overlapping-texture-copy-as-memmove", Precision,
            "the destination holds the source's texels as they were before the copy, memmove's \
             answer, which is Metal's for small regions, where Metal's copy of a larger region \
             races and differs from run to run; the source goes through a scratch buffer, two \
             copies instead of one";
        /// `generate_mipmaps` computed as Metal's filter (`mipmap_filter.rs`) whose levels take a
        /// rounding the chain does not fit at sums a hair from a rounding boundary
        /// (`MipmapFilterChain::near_ties`): a texture of a format with an 8-bit sRGB colour
        /// channel (`R8Unorm_sRGB`, `RG8Unorm_sRGB`, `RGBA8Unorm_sRGB`, `BGRA8Unorm_sRGB`), whose
        /// filtered levels take it; under macOS 26.7's rule (`MipmapRule::AppleM3MaxPasses`), of an
        /// extended-range sRGB format (`BGR10_XR_sRGB`, `BGRA10_XR_sRGB`), whose filtered levels
        /// and levels read from kept values take it; or of a 16-bit float format (`R16Float`,
        /// `RG16Float`, `RGBA16Float`) with a level read from the level above's kept values (a 2D
        /// chain's even level below a square of a power of two, a pass's second level).
        /// `RGB10A2Unorm`, `BGR10A2Unorm`, `BGR10_XR`, `BGRA10_XR`, `ABGR4Unorm` and `A1BGR5Unorm`
        /// are exact.
        MipmapTieRounding = "mipmap-tie-rounding", Precision,
            "each level is Metal's but at a value whose sum lies a hair from a rounding boundary, \
             which may be one unit off: an 8-bit sRGB colour code of a filtered level (1 of \
             5,331,270 measured values), an extended-range sRGB code, a 16-bit float read from \
             kept values one unit in the last place (8 of 538,088)";
        /// A draw whose render pipeline state blends into, or reads (a framebuffer fetch, or the
        /// interlocked storage form an extended-range blend takes), a color attachment of an
        /// extended-range format (`BGR10_XR`, `BGRA10_XR`, `RGB10A8_2P_XR10` and their sRGB
        /// twins) that an earlier draw or execution of render commands of the same render pass
        /// may have written (`render.rs`); an execution of render commands in a pass with such
        /// an attachment, whose commands may blend onto each other. Metal's tile memory holds
        /// such an attachment as `half` through the pass (a clear's, a load's, a write's or a
        /// blend's value rounded to `half`) and rounds it to its codes once, at the store (on an
        /// Apple M4; the M3 Max's run of the same probes), where the runtime stores the codes at
        /// each draw, so a later blend or read sees the stored codes.
        ExtendedRangeBlendsStoredCodes = "extended-range-blends-stored-codes", Precision,
            "a blend into, or a read of, an extended-range color an earlier draw of the pass \
             wrote sees the codes the runtime stored at that draw (each channel rounded to its \
             10-bit code, the two-plane format's alpha to its byte), where Metal's tile memory \
             holds the color as half through the pass and rounds it once at the store: a \
             blended or read value can differ from Metal's by a code, and the difference can \
             grow over many draws that blend onto each other";
        /// A heap texture sized or made (`heap_texture_size_and_align`, `Heap::new_texture`) whose
        /// size Metal's layout the runtime does not model (`texture/heap_tiled.rs`): a format or a
        /// sample count whose layout in Metal's heaps is not measured at the device's Metal level
        /// (8 samples; a PVRTC format at macOS 26.7's level, whose blocks' placement only macOS
        /// 27.0.1's measures; a mipmapped texture Metal compresses and a
        /// `Depth32Float_Stencil8` one are).
        /// The runtime answers its device's image's size and alignment there, which can differ from
        /// Metal's either way.
        UnmeasuredHeapTextureSize = "unmeasured-heap-texture-size", Precision,
            "heap_texture_size_and_align answers the device's own image's size and alignment, \
             which may be larger or smaller than Metal's, so a heap sized by Metal's answers \
             may hold more or fewer such textures and they lie at other offsets";
        /// A program's call answered the `uniqueIdentifier` (`unique_identifier.rs`) of a
        /// pipeline state made without `supportIndirectCommandBuffers`
        /// (`ComputePipelineState::unique_identifier`, `RenderPipelineState::unique_identifier`,
        /// a tile state's too). Metal's is an address of its allocations that differs from run
        /// to run; the runtime's is one value of a count past every other identifier, with
        /// Metal's relations: one value for the state's life, another state's another.
        PlainStateUniqueIdentifier = "plain-state-unique-identifier", Precision,
            "the identifier is one value of the runtime's count, unique and fixed for the \
             state's life, where Metal's is an address that differs from run to run";
        /// A program's call answered a `uniqueIdentifier` (`unique_identifier.rs`) Metal answers
        /// alike in every run of one program started alike, which the runtime's is not measured
        /// to be: a pipeline state's made with `supportIndirectCommandBuffers`
        /// (`ComputePipelineState::unique_identifier`, `RenderPipelineState::unique_identifier`,
        /// an indirect command's state identifier other than 0), or an indirect command buffer's
        /// made after a command buffer of its device that encoded a render encoder was scheduled,
        /// or after a resource state encoder opened (`IndirectCommandBuffer::unique_identifier`).
        /// Metal's is an address of its own allocation: a state's follows how the process was
        /// started (an M3 Max's first such compute state answered `0x1000000280` from a test
        /// binary, `0x100` more from a shell), the states' functions and what the process made; a
        /// buffer's lies past the ranges Metal's own encoders took, a render pass's run by its
        /// targets' sizes and what the device drew before. The runtime's keeps the relations a
        /// program can rely on (a compute state's from one step into Metal's range, no workload's
        /// first). A buffer made before answers Metal's number, quietly, past the pieces Metal's
        /// blit, compute and render encoders took.
        UnmeasuredUniqueIdentifier = "unmeasured-unique-identifier", Precision,
            "the identifier is unique and fixed for its holder's life, and a released one is \
             reused as Metal reuses it, but it is not the number Metal answers there, which \
             follows how the process was started, what it made and Metal's own encoders' \
             allocations";
        /// A draw or dispatch that binds, at a texture slot, a texture whose format's numeric
        /// class (float, signed or unsigned integer) is not the class the function declares there
        /// and that its plan has no exact answer for (`PlanDiagnostic::CrossClassNeutral`): a
        /// depth, stencil, extended-range or YCbCr texture, a depth slot, an atomic or `ulong`
        /// access, a written storage slot over a swizzled texture, or an integer write to an
        /// `RGB9E5Float` texture. The slot binds the declared type's neutral. Every other color
        /// texture (storage slots, texture buffers, multisampled and sparse textures, writes,
        /// `short` and `ushort` results and queries included) answers Metal's words and its real
        /// size (the slot's `cross_class` variant) and announces nothing. Likewise a texture of
        /// another numeric class an argument buffer names that the command reaches
        /// (`PlanDiagnostic::CrossClassHeapNeutral`), where the function samples, gathers, writes
        /// or atomically updates its heap class, or reads it with no exact answer (a storage or
        /// multisampled texture, a `short`, `ushort` or `ulong` result, or a depth, stencil,
        /// extended-range or two-plane texture); its reads and queries of a sampled color texture
        /// of one sample answer Metal's words and announce nothing.
        CrossClassTextureNeutral = "cross-class-texture-neutral", Precision,
            "the function reads the slot's neutral texture (zero texels; zero size), where plain \
             Metal reads the texture in its format's own class and hands back its words as the \
             declared type";
        /// An execution of an indirect command buffer replayed on the host ran a command that
        /// holds a nil buffer set at an offset whose low 32 bits are not 0. Metal's function sees
        /// those bits as the buffer's address (a pointer into no buffer); the host replay binds
        /// no buffer there, so the function sees none. An execution the GPU replays alone passes
        /// the bits, as Metal does.
        IndirectNilBufferOffset = "indirect-nil-buffer-offset", Precision,
            "the function sees no buffer, a null address, where Metal's sees the nil buffer's \
             offset as its address; neither holds a buffer, so only the address's value \
             differs";
        /// A texture of a BC1 to BC5 format (`BC1_RGBA` to `BC5_RGSnorm`, the sRGB forms too)
        /// made on a device whose driver is not measured to decode their texels as Metal does
        /// (BC1 to BC3 `DriverFact::ColorBlockDecodeIsMetals`: MoltenVK 1.4.1 on Apple's GPUs
        /// and 1.4.2 on the M3 Max, RADV from Mesa 26.2.4 on the 780M; BC4 and BC5
        /// `DriverFact::ChannelBlockDecodeIsMetals`: MoltenVK 1.4.1 on Apple's GPUs and 1.4.2 on
        /// the M3 Max). Only a texture without a decoded twin
        /// (`decoded-block-twin`): one over the caller's host pages or laid out apart from a heap's
        /// bytes, a memoryless or sparse texture, one over a surface or the caller's image, one
        /// whose twin's format the device cannot sample, filter and copy into.
        CompressedTexelDecode = "compressed-texel-decode", Precision,
            "a read, sample or gather of its texels answers the device's decode of its blocks \
             (an sRGB form's code through Metal's table), where Metal interpolates with 6-bit \
             weights: NVIDIA's (615.71.09, the RTX 3090 and RTX 5080 Laptop GPU) answers a BC1 \
             to BC3 channel up to 3/255 off Metal's (BC3's alpha 6/255), BC4 and BC5 up to \
             7.3/255 (signed 0.064); ANV's up to 1.3/255 (signed 0.013), lavapipe's 2.3/255 \
             (signed 0.021), RADV's (Mesa 26.2.4, the Radeon 780M) a BC4 or BC5 signed texel \
             with an endpoint of -128 up to 1.7/255; its blocks' bytes, copies and sizes are \
             Metal's";
        /// A texture of a BC1 to BC5 format made on a device whose driver is not measured to
        /// decode its texels as Metal does (`compressed-texel-decode`'s facts), whose image is its
        /// own for good: it keeps a decoded twin, Metal's texels, which its samples read
        /// (`texture/decoded_twin.rs`), and announces no `compressed-texel-decode`.
        DecodedBlockTwin = "decoded-block-twin", Performance,
            "a second image of its decoded texels beside its blocks, 4 to 8 times their bytes \
             (R8G8B8A8_UNORM for BC1 to BC3, R32_SFLOAT and R32G32_SFLOAT for BC4 and BC5), \
             and a decode after each write into it: a pass on the GPU for a blit, a copy from a \
             texture or a CPU copy through a submission; on the CPU, on the device's decode \
             thread, where its CPU copies go on the host or the staging queue, and then once as \
             it is made, the first command buffer that may read it waiting for that at its \
             submission";
        /// A CPU copy into a texture with a decoded twin (`replace_region`) whose CPU copies go
        /// on the host or the staging queue: its region is decoded on the CPU, on the device's
        /// decode thread, and copied into the twin; the first command buffer that may read the
        /// texture waits at its submission for that, and the call itself where a command buffer
        /// that may read it is submitted and not yet complete (which then reads the copied
        /// texels, as Metal's does) or the texture is untracked (`texture/twin_decode.rs`).
        CpuBlockDecode = "cpu-block-decode", Performance,
            "the written blocks decoded on the CPU, on the device's decode thread split over up \
             to 8 threads, and their texels copied into the decoded twin on the host or the \
             staging queue: the first command buffer that may read the texture waits at its \
             submission for that, and the call waits for it where a command buffer that may read \
             the texture is submitted and not yet complete, or the texture is untracked";
        /// A texture of an ASTC LDR format (`ASTC_4x4_LDR` to `ASTC_12x12_LDR`, the sRGB forms
        /// too) made on a device without that ASTC format (`textureCompressionASTC_LDR`), of a
        /// PVRTC format on a device without it (`VK_IMG_format_pvrtc`'s RGBA formats; the RGB ones
        /// on every device), or of an ETC2 or EAC format on a device without them
        /// (`textureCompressionETC2`): its blocks lie in an image of a carrier format whose blocks
        /// are also 16 (ASTC, `EAC_RGBA8`, `RG11`) or 8 (PVRTC, the other ETC2 and EAC formats)
        /// bytes, which copies and `get_bytes` move unchanged, and its samples read a decoded
        /// twin, Metal's decode of the blocks (an ASTC error block's magenta included) in the
        /// texels [`metal2vulkan::device::DecodedTexelsPolicy`] names (a PVRTC texture's, an ASTC
        /// sRGB one's and an 8-bit ETC2 or EAC one's, 8-bit codes, exact, under either), decoded
        /// after each write into it as a BC texture's twin is (`decoded-block-twin`; a PVRTC
        /// texture's whole levels). An ASTC texture's first view in its other colour space gives
        /// it a second twin of that form's decode. Also a texture of a 4:2:2 format (`GBGR422`,
        /// `BGRG422`) on every device, whose Vulkan formats reconstruct chroma by the
        /// implementation's rules: its texels' bytes lie in an `R8G8_UNORM` carrier, and its twin
        /// holds Metal's texels as 8-bit codes, exact, decoded in whole rows.
        CarriedDecodedTwin = "carried-decoded-twin", Performance,
            "a carrier image of its blocks, at most about 1 byte a texel, and a decoded twin of \
             4 bytes a texel (R8G8B8A8, `DecodedTexelsPolicy::Compact`; an EAC 11-bit texture's \
             16-bit codes, 2 bytes a channel) or 16 (R32G32B32A32_SFLOAT, \
             `DecodedTexelsPolicy::Exact`; an EAC 11-bit texture's floats, 4 bytes a channel), \
             two for an ASTC texture viewed in both colour spaces, and a decode after each write \
             into it: a pass on the GPU for a blit, a copy from a texture or a CPU copy through a \
             submission; on the CPU, on the device's decode thread, where its CPU copies go on \
             the host or the staging queue";
        /// A texture with an ASTC LDR decoded twin (`carried-decoded-twin`) of 8-bit codes, or an
        /// EAC 11-bit one of 16-bit codes, under
        /// [`metal2vulkan::device::DecodedTexelsPolicy::Compact`]: Metal decodes an ASTC LDR channel
        /// to a 12-bit value (`C >> 4` over 4095 of the 16-bit interpolated channel `C`), which the
        /// twin's R8G8B8A8 texels hold as the nearest 8-bit code (at most 1/510 off Metal's value).
        /// It reads an EAC 11-bit channel as `c / 2047` (signed `c / 1023`), which the twin holds
        /// as the nearest 16-bit normalized code (at most half a 16-bit step off).
        /// `DecodedTexelsPolicy::Exact` holds Metal's values instead. (An ASTC sRGB texture's twin
        /// holds Metal's own 8-bit codes, which its slots decode through Metal's table: exact.)
        DecodedTexelPrecision = "decoded-texel-precision", Precision,
            "a read, sample or gather of its texels answers the nearest 8-bit code to Metal's \
             12-bit value, at most 1/510 off it; an EAC 11-bit texture's, the nearest 16-bit \
             normalized code to Metal's 11-bit value, at most half a 16-bit step off it";
        /// A command queue made for presenting
        /// ([`Device::new_command_queue_for_presenting`](crate::Device::new_command_queue_for_presenting))
        /// on a device that reserved no `VkQueue` for presenting: its queue family has no queue
        /// past the ones the round robin takes (MoltenVK's four families of one queue, RADV's and
        /// lavapipe's single graphics queue).
        SharedPresentQueue = "shared-present-queue", Performance,
            "it submits to a `VkQueue` other command queues share, so a present that blocks under \
             its lock (FIFO on a window the compositor does not show, a wait for the queue's \
             idle at a resize) holds their submissions until it returns, where Metal's present \
             holds no other queue";
        /// A heap texture sized or made (`heap_texture_size_and_align`, `Heap::new_texture`)
        /// whose image on the device takes more of a heap than Metal's answer, which the runtime
        /// models (`texture/heap_tiled.rs`): more bytes, or a coarser alignment (the image's own,
        /// at least `bufferImageGranularity`, its size in whole granularity pages). No image
        /// binds at Metal's offsets then, so the runtime answers the device's (the consumer's
        /// report: on NVIDIA 615.71.09, the RTX 3090, a 4 x 4 `Depth32Float` texture answered
        /// 1024 bytes, where Metal answers 128). Not where `optimized-depth-stencil-texture`
        /// says why.
        HeapTextureLargerThanMetal = "heap-texture-larger-than-metal", Precision,
            "heap_texture_size_and_align answers more bytes or a coarser alignment than Metal's, \
             so a heap sized by Metal's answers holds fewer such textures, and they lie at other \
             offsets; each texture keeps its own bytes";
        /// A pass renders a color attachment whose texture's image is linear (a texture over a
        /// buffer's memory, a no-copy buffer's host pages included) through an image its texture
        /// keeps, where the device renders such an image only under
        /// `VK_NV_linear_color_attachment`'s rule, that every color attachment of the render pass
        /// instance be linear, and the pass breaks it: another of its color attachments is not
        /// linear, or it resolves a multisampled attachment into the texture. The notice names
        /// which. Where the pass loads it, the words the pass changed are merged back into the
        /// texture's bytes, so no pixel its draws left as they were is written (`linear_merge.rs`).
        LinearAttachmentCopies = "linear-attachment-copies", Performance,
            "its texels are copied into an image the pass renders, and what the pass wrote is \
             merged or copied back after it";
        /// A pass clears a color attachment whose texture's image is linear and which the device
        /// renders under `VK_NV_linear_color_attachment`, where the rule allows it in place:
        /// it clears an image its texture keeps and copies it back after, as clearing a linear
        /// image of host memory in place measured slower than that copy. Only where no command of
        /// the pass may read the texture's bytes (`linear_attachment::choose`), so the two ways
        /// cannot be told apart.
        LinearAttachmentClearCopies = "linear-attachment-clear-copies", Performance,
            "the pass clears an image of the device's and copies it back after, as clearing the \
             texture's linear image in place costs more";
        /// A pass renders a color attachment that it stores with `DontCare` (or resolves alone,
        /// `MultisampleResolve`) and whose texture's image lies over the buffer the texture lies
        /// in (a linear image over an ordinary buffer's memory) through an image its texture keeps, so that the buffer keeps its bytes, as
        /// Metal's does (`kept_bytes.rs`): rendered in place, the pass would write them. Not where
        /// the pass hands the attachment to a driver that stores `DONT_CARE` as Metal's own
        /// `DontCare` (`DriverFact::DontCareStoreIsMetals`), one render pass instance the driver
        /// runs unbroken: Metal keeps the bytes itself.
        DontCareStandIn = "dont-care-stand-in", Performance,
            "an image of the attachment's size, which its texture keeps for the queue, renders the \
             pass; a `Load` copies the buffer's texels over the render area into it before the \
             pass, and nothing is copied back";
        /// A no-copy buffer or heap released with a deallocator while a command buffer that may
        /// still reach its pages holds them. Metal runs the deallocator at the release, even while
        /// a command buffer in flight reaches the buffer unnamed, and that command buffer's later
        /// accesses reach nothing (the M3 Max under macOS 27.0.1, probe `dealloc-probe4`: through a
        /// loaded pointer, and through an indirect command the GPU encoded with the buffer's
        /// address). Vulkan frees no memory that submitted commands may still use
        /// (`VUID-vkFreeMemory-memory-00677`: all submitted commands that refer to memory must
        /// have completed execution), and keeps imported pages mapped while imported, so the pages
        /// stay, and the deallocator runs once they go (`buffer.rs`):
        /// - with no keep-alive ([`KeepAlive`](crate::KeepAlive)), while the device's import of
        ///   them is held: a command buffer submitted before a texture over them was released may
        ///   still read that texture through its resource ID, and its image holds the import until
        ///   the command buffer completes (`heap_sets.rs`);
        /// - with no keep-alive, while a live buffers' table that lists them is held: a command
        ///   buffer whose commands may reach the buffer unnamed holds its first submission's table
        ///   and its host steps' until it completes, and one a rebind encoded again, or a host
        ///   step recorded, before its first submission until then (`live_buffers.rs`);
        /// - with no keep-alive, on a device whose driver fails pending work where memory a
        ///   device-address buffer lay in is freed
        ///   (`RuntimeDriverDefect::FreedAddressMemoryFailsPendingWork`: MoltenVK), while the
        ///   batches submitted before the import's memory went are in flight (`freed_memory.rs`);
        /// - with no keep-alive, a detached buffer's or heap's rebind or release while the pages
        ///   its detach let go of are still imported, as above: the deallocator its detach held
        ///   back (Metal runs it at the replace or the release) runs once they go (`detach.rs`).
        ///
        /// With a keep-alive the deallocator runs at the release, as Metal's, and the import keeps
        /// the keep-alive, which keeps the pages mapped, until it is freed: not announced.
        ///
        /// Announced once per device, on the thread that lets go of the buffer last, whatever
        /// `DeviceOptions::announce_each` says: that may be a runtime thread, as a command
        /// buffer that used the buffer completes, outside any program call or host step.
        HostDeallocatorDelayed = "host-deallocator-delayed", Precision,
            "the deallocator runs once every command buffer that may still reach the pages \
             (through a texture over them, or a table of the live buffers it holds) lets go of \
             them, where Metal's runs at the release; the pages stay mapped until then";
        /// A no-copy buffer, heap or rebind was made over host pages the process maps
        /// read-only, from the caller's dma-buf source (`HostMemory::with_dma_buf`): on Linux,
        /// where drivers pin a pointer's imported pages for writing, such pages are imported only
        /// as dma-bufs of their pages, whose memory the GPU writes. Announced once per device, by
        /// the call that makes it, or at each such call where `DeviceOptions::announce_each`
        /// asks.
        ReadOnlyPagesWritten = "read-only-pages-written", Precision,
            "the GPU's writes through it reach the pages, and the CPU then reads them there, \
             where after Metal's GPU writes into pages the process maps read-only the CPU reads \
             them as they were (an M3 Max under macOS 27.0.1, private and shared memory alike)";
        /// Batches of command buffers committed on one queue that the scheduler found ready
        /// together, or held while two or more earlier submissions of their queue ran, went to the
        /// driver in one `vkQueueSubmit`, only the last signalling its queue's timeline and their
        /// events (`CompletionPolicy::Grouped`, the default, on a driver whose queue
        /// merges a submission that signals nothing into the next one: Mesa's drivers;
        /// `command_buffer/group.rs`): one kernel submission for up to four batches. Announced
        /// once per device, on the scheduler's thread, as the first such submission is made,
        /// whatever `DeviceOptions::announce_each` says: no program call or host step meets it.
        GroupedCompletion = "grouped-completion", Precision,
            "each command buffer of the submission but the last is seen complete (its status, \
             completed handlers and waits), and the event values it signals are reached, once \
             the last completed, late by the GPU time of the batches after it, at most three, \
             never early and in commit order, and a submission held while earlier ones of its \
             queue ran late besides by any gap between the last of those and its own start on \
             the GPU, at most a completion turn's reaction and its vkQueueSubmit; and the \
             submission checks of the batches behind \
             a queue's next one run with its own, before the batch before them is submitted; \
             their commands, barriers and GPU times are unchanged; CompletionPolicy::Exact \
             submits and checks each batch alone";
        /// In a build without debug assertions, a notice announced where the runtime's rule
        /// announces none ([`DegradeNotice::misplaced`] names its kind and [`NoticePlace`]):
        /// inside a pipeline pool's piece, or, for a kind announced at every occurrence, outside
        /// the program's call and outside a host step. The notice itself reaches the sink
        /// unchanged, from the runtime's notice thread, which holds no runtime lock, and this one
        /// follows it there. A build with debug assertions ends the process instead, with the
        /// rule's line, so every lane and gate fails on it.
        MisplacedNotice = "misplaced-notice", Performance,
            "the notice reached the message sink from the runtime's notice thread after the call \
             that met it, not in the program's order on the thread that met it; Metal's answers \
             are unchanged";
        /// A write that may store a denormal in the 32-bit float depth of a depth image (a pass's
        /// clear to one, a blit's copy into a `Depth32Float` or `Depth32Float_Stencil8` texture's
        /// depth, a `replace_region` whose bytes hold one), on a device whose driver stores one a
        /// write on the GPU gives it as a zero (no `DriverFact::DepthImageKeepsDenormals`: RADV,
        /// ANV; a host copy keeps it) or whose depth test compares one otherwise than Metal's (no
        /// `DriverFact::DepthTestKeepsDenormals`: every driver measured but MoltenVK on an Apple
        /// M4). Metal's texel keeps it, and its depth test compares a positive one above zero and
        /// a negative one not below zero. Where the driver stores it and only its depth test
        /// differs, a single-sample `Depth32Float` render target's passes test the smallest normal
        /// in its place (`texture/signed_depth.rs`): Metal's answer but for a fragment at exactly
        /// 2^-126, which passes `Equal` and fails `Greater` against it, where Metal's fails and
        /// passes; the payload says so. A draw at a denormal depth stores zero on Metal too, and
        /// announces nothing.
        DepthDenormals = "depth-denormals", Precision,
            "a denormal depth may read, sample and copy out as a zero, or the depth test may \
             compare it as zero (a positive one) or below zero (a negative one), where Metal \
             keeps it and compares a positive one above zero and a negative one not below it; a \
             render target whose passes test the smallest normal in its place differs only for a \
             fragment at exactly 2^-126";
        /// A depth test that may compare a negative value or a NaN stored in the 32-bit float depth
        /// of a render target as it is, or a pass with depth writes that may store such a value
        /// with its sign. Metal's depth test compares the fragment's depth with the stored value's
        /// magnitude, as the integers of their low 31 bits: a stored -0.5 passes `Equal` for a
        /// fragment at 0.5, and a stored NaN orders above +inf (probe `depthkeep`, the M3 Max under
        /// macOS 27.0.1: 270 of 270 outcomes); and a pass that sets a depth/stencil state with
        /// depth writes stores each texel no fragment wrote without its sign (probe `depthstore`).
        /// Vulkan's compares the stored value as it is, and its pass keeps it. A single-sample
        /// `Depth32Float` render target that a write may give such a value, or whose new texture's
        /// pattern (a NaN with the sign set) a pass loads before a write covered it
        /// (`fresh_unwritten.rs`), renders the passes that load its depth over its magnitude
        /// stand-in (`texture/signed_depth.rs`), whose tests and stores are Metal's, and announces
        /// only the corners: a pass that writes its depth over the stand-in, whose merge keeps a
        /// stored NaN where a draw wrote exactly +inf, its stand-in, over it; a pass that loads it
        /// through another image (separate depth and stencil textures' scratch image, a stand-in
        /// for custom sample positions); and a write that switches it on while another command
        /// buffer is uncommitted, whose passes encoded before it test and store the value as it is.
        /// A multisampled or `Depth32Float_Stencil8` target takes no stand-in: its words are
        /// stripped of their signs after a pass with depth writes, Metal's stores, and its test,
        /// which compares the value as it is, is announced at the write (a blit's copy into its
        /// depth, or the pass that loads its new pattern), on a device that keeps such a value
        /// there (`VK_EXT_depth_range_unrestricted` enabled, or
        /// `DriverFact::DepthImageKeepsValuesOutsideUnitRange`; elsewhere `restricted-depth-upload`
        /// is announced), as is a strip the device does not make. A denormal tested over the
        /// smallest normal is `depth-denormals`'. A device whose depth test compares magnitudes as
        /// Metal's (`DriverFact::DepthTestComparesMagnitudes`: MoltenVK, whose test and store are
        /// the GPU's own Metal ones) renders every target's depth in place, strips a switched-on
        /// depth's signs after a pass that sets a state with depth writes but draws under none,
        /// which its pass leaves signed, and announces it only for a switch while another command
        /// buffer is uncommitted. A render pass's NaN clear depth, which Metal stores as
        /// `0x7fc00000` and tests above every value, is written into the depth before the pass and
        /// loaded (`nan_clear.rs`); where the device writes no such word (a memoryless depth, a
        /// multisampled one without maintenance8's colour copies or an exact `FragDepth`, a pass
        /// that does not store it on a device that keeps the earlier texels there), it is cleared
        /// to 1.0, announced here.
        SignedDepthTest = "signed-depth-test", Precision,
            "a depth test against the stored value, or the value a pass with depth writes stores, \
             can differ from Metal's in the pass or target named: the device compares a negative \
             value below zero and a NaN by its own rule and keeps the value's sign, or a pass's \
             merge keeps a stored NaN where a draw wrote +inf over it, where Metal compares the \
             value's magnitude, orders a NaN above every value, stores the value without its sign \
             and stores the draw's; or a NaN clear depth is stored and tested as 1.0, where Metal \
             stores 0x7fc00000";
        /// The device's first sparse texture made (`Heap::new_texture` in a sparse heap) on a
        /// device whose driver's first sparse resource waits for the work submitted before it on
        /// every queue (`DriverFact::FirstSparseResourceWaitsForQueues`: ANV through TR-TT, Mesa
        /// 26.2.4 on Intel's GPUs): `vkCreateImage` sets the driver's sparse translation tables
        /// up with a batch on each queue and waits for it. Later sparse textures wait for nothing.
        FirstSparseResourceWait = "first-sparse-resource-wait", Performance,
            "the call returns only once every queue finished the work submitted before it, once \
             per device, where Metal's returns at once; what the texture holds and reads is \
             Metal's";
        /// A device the runtime makes whose driver it cannot make with the driver's internal
        /// pipeline cache off (`VkDevicePipelineBinaryInternalCacheControlKHR`): its driver lists
        /// no `VK_KHR_pipeline_binary`, reports no `pipelineBinaryInternalCacheControl`, or the
        /// device enables no `VK_KHR_maintenance5`, which the extension requires. Announced once,
        /// as the device is made; the payload names the device and why. Not on MoltenVK, whose
        /// kept cache is Metal's own shader cache: that device sends the informational message
        /// `metal2vulkan-driver-pipeline-cache-kept` instead.
        DriverPipelineCacheKept = "driver-pipeline-cache-kept", Performance,
            "the driver keeps its own cache of the device's pipelines beside the shader cache's \
             records, with whatever it writes to disk and holds in memory for it; the pipelines \
             made are the same";
        /// A device made off macOS with no Metal level named: it answers as the newest measured
        /// release ([`MetalLevel::NEWEST`](crate::MetalLevel::NEWEST)), no release of the host's
        /// (`metal_level.rs`). Announced once, as the device is made; the payload names the level.
        MetalLevelNotMacOS = "metal-level-not-macos", Precision,
            "where Metal's answers differ by release (which pixel formats a texture descriptor \
             takes and Metal describes, how a heap texture's blocks lie, whether a run of compute \
             encoders' first indirect dispatch of another dispatch type ends the process), the \
             device gives the newest measured release's, whatever release the program expects";
        /// Apple's private device properties the device has no answer for
        /// (`private_properties.rs`): no document describes them and nothing the device reports
        /// maps to them. Announced once, when `Device::private_properties` is first read; the
        /// payload names the selectors.
        PrivatePropertiesUnknown = "private-properties-unknown", Precision,
            "the device gives no answer for them (`PrivateAnswer::value` is `None`); Metal's \
             measured answers on Apple's GPUs ride beside, never as this device's";
        /// A color fill (`BlitCommandEncoder::fill_texture_color`) whose color is packed in a
        /// format Metal packs no color in: compressed, 4:2:2, depth, stencil, two-plane, YCbCr,
        /// padded 10-bit, `Invalid` or a private format (`blit_fill.rs`). Metal fills bytes of the
        /// calling thread's stack there, which hold the last color fill's packed texel only where
        /// no other call used them since. The runtime fills the command buffer's last color fill's
        /// packed texel, over zeros.
        UnpackedFillColor = "unpacked-fill-color", Precision,
            "every unit of the region holds the command buffer's last packed texel over zeros, \
             whatever the color, where Metal's bytes are its stack's";
        /// A fill of a texture buffer (`BlitCommandEncoder::fill_texture`,
        /// `BlitCommandEncoder::fill_texture_color`) whose 16-byte chunks read bytes of its block
        /// past the pattern's whole copies, or past the block (`blit_fill.rs`): most patterns
        /// whose length is not a power of two, and most regions whose first byte is not at a
        /// multiple of 16 of the buffer. Metal reads bytes of the calling
        /// thread's stack there: zeros in our probes, an earlier fill's bytes in the consumer's
        /// process. The runtime writes zeros there.
        TextureBufferFillLeftover = "texture-buffer-fill-leftover", Precision,
            "those bytes of each chunk are zeros, where Metal's are its stack's";
        /// A draw of points that rasterizes, of some vertices and instances, under a render
        /// pipeline state whose vertex function writes no `[[point_size]]` and whose input
        /// primitive topology class is not the point class (`render.rs`'s `draw_settings`): a
        /// direct draw, an indirect one, or an indirect command's in an execution. Metal sizes
        /// none of those points, and what it draws is undefined (measured by the consumer on an
        /// M3 Max: most draws draw nothing, some a few of the points or a box whose size changes
        /// from run to run; probes on an M4 drew nothing). The runtime rasterizes none of them, the
        /// vertex function run for every point: one of Metal's answers. Announced at the draw's
        /// call, and for an execution's command where the runtime records it: in `commit`, on the
        /// committing thread, for the CPU's records; in the host step, on the scheduler's thread,
        /// for records the GPU may write.
        UnsizedPoints = "unsized-points", Precision,
            "the points rasterize nothing, the vertex function run for each, where what Metal \
             draws is undefined: most of its draws draw nothing, and some a few of the points \
             or a box";
        /// A draw of lines, or an emulated mesh draw of lines, under Apple's last provoking vertex
        /// (`setProvokingVertexMode:` 1), of a pipeline state that reads it (a flat input, or a
        /// slice or viewport its vertex function selects), on a device without
        /// `VK_EXT_provoking_vertex`'s last vertex (or whose pipelines of one render pass instance
        /// share one mode): each line is drawn from its last vertex to its first, so a first-vertex
        /// pipeline reads Metal's provoking vertex (`render/provoking.rs`). Announced once per kind
        /// of draw.
        ProvokingVertexLines = "provoking-vertex-lines", Precision,
            "each line's flat inputs are Metal's, but it can cover other pixels and interpolate \
             its smooth inputs otherwise than Metal's line from its first vertex to its last";
        /// A draw under Apple's last provoking vertex (`setProvokingVertexMode:` 1) the runtime
        /// draws otherwise than Metal (`render/provoking.rs`). On a device without
        /// `VK_EXT_provoking_vertex`'s last vertex (or whose pipelines of one render pass instance
        /// share one mode), what no list the runtime draws holds as Metal draws it: a draw of
        /// triangles, or an emulated mesh draw of triangles, whose fragment function reads flat
        /// inputs and `[[barycentric_coord]]`, each drawn from its last vertex, its second and
        /// third weights taken back to Metal's order from the drawn triangle's, Metal's but for
        /// rounding; patches, and mesh draws through `VK_EXT_mesh_shader`, whose pipeline state
        /// reads flat inputs: they read the first vertex. On a device with it, a strip whose vertex
        /// function computes `[[primitive_id]]` into its varying (a device without `geometryShader`
        /// and `tessellationShader`), whose IDs read each primitive's last vertex; and a slice or
        /// viewport a vertex function selects where the pipeline selects it in a synthesized
        /// geometry stage (a device without `shaderOutputLayer` or `shaderOutputViewportIndex`, an
        /// index that differs between a primitive's vertices, or weights the stage writes), from
        /// each primitive's first vertex. And on every device, a draw at a mode past 1, which
        /// Metal draws a third way and the runtime as mode 1, whose shape and pipeline state Metal
        /// draws otherwise than mode 1 (a strip's odd triangles, every line's flat inputs and
        /// `[[primitive_id]]` 0, sized points, the middle vertex of a triangle whose pipeline
        /// selects its slice or viewport, mesh draws unmeasured): announced at the draw.
        ProvokingVertexUnmatched = "provoking-vertex-unmatched", Precision,
            "flat inputs or a computed primitive ID read another vertex than Metal's, or \
             barycentric weights come in another order than Metal's or round otherwise";
        /// A draw counted by a `Counting` visibility result whose fragment variant writes more
        /// than one color attachment, on a device whose driver counts each sample once per color
        /// output ([`metal2vulkan::error::RuntimeDriverDefect::OcclusionCountPerColorOutput`]:
        /// ANV). Announced at the draw's call, or where the runtime records an execution's
        /// command; the payload names the draw and its colors.
        VisibilityCountPerColor = "visibility-count-per-color", Precision,
            "the result is Metal's count multiplied by the colors the draw writes; a `Boolean` \
             result is Metal's";
        /// A device whose policy takes native device maths (`FloatMathsPolicy::Native`, the
        /// default) where its float functions are not Metal's own (not MoltenVK,
        /// `DriverDeviation::MetalFloatFunctions`): Metal flushes a `float` product or fused
        /// multiply-add whose exact value lies below `2^-126` to the zero of its sign, deciding
        /// tininess before rounding, and the device rounds before it decides (measured on RADV,
        /// ANV, NVIDIA and lavapipe, whether they flush denormals or keep them), so the device's
        /// multiply and fused multiply-add answer `2^-126` there (`device/float_tininess.rs`).
        /// Announced once, as the device is made; the payload names the device and its policy.
        FloatTininessAfterRounding = "float-tininess-after-rounding", Precision,
            "a `float` product or fused multiply-add whose exact value lies within half a unit \
             below `2^-126` answers `2^-126` of its sign where Metal answers the zero of its sign; \
             `FloatMathsPolicy::Exact` computes Metal's";
    }
    notes {
        F16FmaDoubleRounding = "f16-fma-double-rounding";
        F16IntConvDoubleRounding = "f16-int-conv-double-rounding";
        GradFromBias = "grad-from-bias";
        DeviceTranscendentals = "device-transcendentals";
        ReductionOrder = "reduction-order";
        DeviceDenormals = "device-denormals";
        PixelSamplerScaling = "pixel-sampler-scaling";
        DeviceArithmetic = "device-arithmetic";
        DeviceSignedZeroInfNan = "device-signed-zero-inf-nan";
        DeviceRayIntersection = "device-ray-intersection";
        FastMathRewrites = "fast-math-rewrites";
        TessellationWinding = "tessellation-winding";
        TessellationFactorEdgeCases = "tessellation-factor-edge-cases";
        TessellationFactorScale = "tessellation-factor-scale";
        TessellationFactorInstance = "tessellation-factor-instance";
        Fp8Value = "fp8-value";
        MultisamplePosition = "multisample-position";
        RasterizationRateMap = "rasterization-rate-map";
        BicubicFilterArithmetic = "bicubic-filter-arithmetic";
        SparseResidency = "sparse-residency";
        DeviceNanEncoding = "device-nan-encoding";
        RestrictedDepthRange = "restricted-depth-range";
        CubeGradientFootprint = "cube-gradient-footprint";
        HalfTexelConversion = "half-texel-conversion";
        HalfSampleFilter = "half-sample-filter";
        SrgbTexelDecode = "srgb-texel-decode";
        FloatSampleFilter = "float-sample-filter";
        DeviceGatherFootprint = "device-gather-footprint";
        CubeFaceReciprocal = "cube-face-reciprocal";
        FunctionStorageAssumed = "function-storage-assumed";
        QuadsCountedBeforeDiscard = "quads-counted-before-discard";
        DeviceDepthComparison = "device-depth-comparison";
        DeviceTexelDenormals = "device-texel-denormals";
        BeyondMeasuredDeviation = "beyond-measured-deviation";
        ViewBaseTexelOffset = "view-base-texel-offset";
        DeviceColorConversion = "device-color-conversion";
        DeviceInterpolation = "device-interpolation";
        QuadsSplitByClipping = "quads-split-by-clipping";
        VertexSimdgroupGrouping = "vertex-simdgroup-grouping";
        FragmentSimdgroupGrouping = "fragment-simdgroup-grouping";
        BlendedWriteRounding = "blended-write-rounding";
        RotatedBlendRounding = "rotated-blend-rounding";
        LinkedSamplerNumbering = "linked-sampler-numbering";
        ComputedSampleDepth = "computed-sample-depth";
        ArgumentPixelSamplerScaling = "argument-pixel-sampler-scaling";
        HalfOutputPrecision = "half-output-precision";
        ExtendedRangeBlendSource = "extended-range-blend-source";
        DiscardDropsSimdgroupStores = "discard-drops-simdgroup-stores";
        HelperBranchDropsSimdgroupStores = "helper-branch-drops-simdgroup-stores";
        ExtendedRangeFloatBlendSource = "extended-range-float-blend-source";
        AnisotropicFilter = "anisotropic-filter";
        DeviceAnisotropicFilter = "device-anisotropic-filter";
        DevicePointSize = "device-point-size";
        ClippedCoverageSplit = "clipped-coverage-split";
        LoopedSampleOutputs = "looped-sample-outputs";
        DeviceUnorm8LinearFilter = "device-unorm8-linear-filter";
        HalfOutputFastMathBlend = "half-output-fast-math-blend";
        /// Class: native device sampling. The device's own sampler answers the pipeline's
        /// samples, gathers and comparisons where Metal's would be emulated
        /// (`SamplingPolicy::NativeDevice`, the default on a device whose sampler is not Metal's;
        /// `SamplingPolicy::Exact` keeps Metal's). Recorded bound, measured against an Apple M4
        /// on 2D textures of one level in seven formats under four address modes: a linear
        /// filter within 0.919 of a code (1/255) of Metal's on NVIDIA's 615.71.09, 0.548 on RADV,
        /// 0.918 on ANV and 1.0 on lavapipe (Mesa 26.2.4); an sRGB texture's within 1.832, 0.764,
        /// 0.912 and 0.781; a nearest sample's texel the neighbour of Metal's at 0.1% of the
        /// probe's coordinates on NVIDIA and ANV, 0.02% on RADV and none on lavapipe. Level
        /// selection, anisotropy, cube maps, 3D textures, gathers, comparisons and offsets are
        /// not measured.
        DeviceSampling = "device-sampling";
        /// Class: native device maths. The device's `GLSL.std.450` instruction answers a `float`
        /// or `half` math-library call where Metal's answer would be computed by a routine
        /// (`FloatMathsPolicy::Native`, the default on a device whose functions are not Metal's
        /// own; `FloatMathsPolicy::Exact` keeps Metal's), and the device's `Sqrt` answers
        /// `fast::sqrt` where Metal's `x · rsqrt(x)` would be computed
        /// (`RoundingOrderPolicy::Native`, the default). Recorded bound: Vulkan's precision table
        /// for the instruction: `Exp2` and `Exp` 3 + 2·|x| ulp; `Log2` and `Log` 3 ulp outside
        /// `[0.5, 2]`, `2^-21` absolute inside; `Pow` from `Exp2(y · Log2(x))`; `Sqrt` from `1 /
        /// InverseSqrt`; `InverseSqrt` 2 ulp; `Sin` and `Cos` `2^-11` absolute in `[-π, π]`; `Tan`
        /// from `Sin / Cos`; `Asin`, `Acos`, `Atan` and `Atan2` 4096 ulp.
        NativeTranscendentals = "native-transcendentals";
        /// Class: native device maths. The device's `OpFDiv` answers a precise or fast `float` or
        /// `half` division where Metal's correctly rounded quotient or `x · RN(1 / y)` would be
        /// computed (`FloatMathsPolicy::Native`, the default; `Exact` keeps Metal's). Recorded
        /// bound: 2.5 ulp of the quotient (a precise division's operands past a divisor of
        /// `2^126` scaled into Vulkan's range exactly); a `half` quotient within one unit in its
        /// last place.
        NativeDivision = "native-division";
        /// Class: native device maths. The device's `GLSL.std.450 Fma` answers a `half` fused
        /// multiply-add, or a `float` one on a device without `OpFmaKHR`, where the correctly
        /// rounded routine would (`FloatMathsPolicy::Native`, the default; `Exact` keeps Metal's).
        /// Recorded bound: a `half` result within one unit in its last place of Metal's; a
        /// `float` one within half a unit in the last place of the product plus half a unit in
        /// the last place of the result.
        NativeFma = "native-fma";
        /// Class: native device sampling. The device's own conversion of a texel's codes answers
        /// the pipeline's reads, samples and gathers with a `float` result where Metal's would be
        /// emulated (`TexelConversionPolicy::Device`, the default, on a device whose sampler is not
        /// Metal's; `Exact` keeps Metal's for a read, and for a sample or gather under
        /// `SamplingPolicy::Exact`). Recorded bound, of a texel whose access reads one texel of one
        /// level: an 8- or 16-bit normalized code within one `f32` ulp of Metal's `c / (2^b - 1)`
        /// where measured (lavapipe's reciprocal product); a 5- or 6-bit channel the device widens
        /// by replicating its bits within 0.18 of a code (ANV, NVIDIA; a 4-bit channel's exact); a
        /// half-float or small-float NaN in the device's encoding, where Metal answers
        /// `0x7fc00000`.
        DeviceTexelConversion = "device-texel-conversion";
        /// Class: native half rounding. The device's own conversion of a `float` to 16 bits
        /// rounds the pipeline's `half` results where Metal's `half(float)` would be computed in
        /// integer arithmetic (`HalfRoundingPolicy::Native`, the default, on a device whose
        /// conversion is not measured to be Metal's). Recorded bound: a finite result within one
        /// unit in the last place of the `half` (to nearest even on NVIDIA's RTX 5080 Laptop
        /// GPU, toward zero on its RTX 3090 and lavapipe); where the device rounds toward zero,
        /// 65504 from 65520 up where Metal answers infinity; a result below `2^-14` the zero of
        /// its sign on a device that flushes 16-bit denormals; a NaN's encoding (`0x7FFF` on the
        /// RTX 5080, where Metal answers `0x7E00`) and the sign of a zero result the device's.
        NativeHalfConversion = "native-half-conversion";
        /// Class: native half rounding. The device's `OpQuantizeToF16` rounds the pipeline's
        /// `half` results on a device without `shaderFloat16` whose quantize is not measured to
        /// be Metal's (`HalfRoundingPolicy::Native`, the default). Recorded bound: a finite
        /// result within one unit in the last place of the `half`; 65504 to 65520 possibly
        /// infinity where Metal answers 65504; a NaN's encoding the device's.
        NativeHalfQuantize = "native-half-quantize";
        /// Class: native device maths. The device's NaN answers a `float` or `half` operation's
        /// NaN result where Metal's bits would be selected (`NanMinMaxPolicy::Native`, the
        /// default on a device whose arithmetic is not measured to answer Metal's canonical NaN;
        /// `Exact` keeps Metal's). Recorded bound: where Metal answers a NaN the device answers a
        /// NaN, its sign and payload the device's own, where Metal answers `0x7fc00000`
        /// (`0x7e00` as `half`), an operand's bits, or the canonical NaN with an operand's sign;
        /// every other result unchanged.
        NativeNanEncoding = "native-nan-encoding";
        /// Class: native device maths. The device's `NMin`, `NMax` or `NClamp` answers Metal's
        /// `fmin`, `fmax`, `min3`, `max3`, `clamp` or `saturate` (its `clamp(x, +0, 1)`) on
        /// `float` or `half` where the translation would select Metal's operand
        /// (`NanMinMaxPolicy::Native`, the default on a device not measured to answer Metal's;
        /// `Exact` keeps Metal's). Recorded bound: an operand, Metal's but at three ties: of two
        /// NaNs a NaN (Metal: the second operand's bits), of `+0` and `-0` either zero (Metal: the
        /// second), and an `f32` denormal beside a zero or another denormal compared by its value
        /// where the device keeps denormals (Metal compares it as a zero and answers the second).
        NativeMinMax = "native-min-max";
        /// Class: native device maths. The device's `OpDot` answers Metal's `dot` on `float`, or
        /// on `half` where its fused multiply-add is not the measured 16-bit one (summed in
        /// `float`, rounded once to `half`), where Metal's fused chain of its products would be
        /// computed (`RoundingOrderPolicy::Native`, the default; `Exact` keeps Metal's). Recorded
        /// bound: a `float` `dot` of `n` lanes within `(3n - 1) * 2^-24 * sum |a_i * b_i|` of
        /// Metal's chain, plus `n * 2^-126` where a product is an `f32` denormal the device keeps;
        /// a `half` one within `(n + 3) * 2^-11 * sum |a_i * b_i| + (n + 2) * 2^-25`.
        NativeDot = "native-dot";
        /// Class: native device maths. The device's `FMix` answers Metal's `mix(x, y, t)` on
        /// `float`, or on `half` where its fused multiply-add is not the measured 16-bit one
        /// (computed in `float`, rounded once to `half`), where Metal's
        /// `fma(t, y, fma(-t, x, x))` would be computed (`RoundingOrderPolicy::Native`, the
        /// default; `Exact` keeps Metal's). Recorded bound: a `float` result within `2^-21 * (|x *
        /// (1 - t)| + |y * t|)` of Metal's; a `half` one within `3 * 2^-10 * (|x * (1 - t)| + |y *
        /// t|) + 2^-23`.
        NativeMix = "native-mix";
        /// A render pipeline whose color attachments the device draws in more than one group:
        /// each draw runs once per group, each group's variant over its own copy of the pass's
        /// depth and stencil. Recorded bound: groups past the first may differ at discard or
        /// depth edges if the driver compiles the variants differently; nothing broader.
        ColorAttachmentGroups = "color-attachment-groups";
        /// A tessellation pipeline drawn through the device's tessellation stages, on a device
        /// whose tessellator is not Apple's (every one but MoltenVK on an Apple GPU): the device's
        /// tessellator orders each patch's triangles, picks their first corners and joins its
        /// rings where Vulkan leaves that to the implementation, not as Metal's tessellator does.
        /// The domain points are Metal's; flat varyings, interpolation across another
        /// triangulation and the order a patch's triangles blend in may differ. On a device
        /// without `tessellationShader` the emulation draws Metal's order and records none.
        TessellationTriangleOrder = "tessellation-triangle-order";
        /// A function holds a branch whose arms hold a quad or simdgroup operation that Metal's
        /// backend makes answer over every lane of the branch's block when it speculates the arms,
        /// where no measurement says whether it speculates them (an arm of an instruction, size or
        /// shape no probe covers): the branch keeps the AIR's lanes, so that operation may answer
        /// over the lanes that take the arm where Metal's answers over the block's. Nothing else
        /// in the stage changes.
        UnmeasuredGroupSpeculation = "unmeasured-group-speculation";
        /// A sample at an implicit level or with gradients through a constexpr sampler with
        /// anisotropy, no mip filter and different minification and magnification filters, on a
        /// device that samples through Metal's own sampler (MoltenVK on Apple's GPU): no level
        /// clamp the device's sampler takes reads Metal's `NotMipmapped` footprint at
        /// `maxAnisotropy` 2 to 16, so the sample filters another anisotropic footprint.
        NotMipmappedAnisotropicFilter = "not-mipmapped-anisotropic-filter";
        /// A sample whose level the device's level arithmetic chooses (a fragment function's
        /// implicit-level sample, its bias within the device's limit; a sample with explicit
        /// gradients; a fragment function's LOD query; an explicit level Metal's level rule is not
        /// handed on, such as an integer texture's), of a texture other than a cube, without
        /// comparison, where no sample-filter note covers it: the device's level of detail and its
        /// rounding of a level may read another level or filter than Metal's near a boundary, and
        /// the query answers the device's level.
        DeviceSampleLevel = "device-sample-level";
        /// A render pipeline covers by alpha under `textureWriteRoundingMode` 2, and its fragment
        /// function writes a `float` alpha to a half-float attachment 0 that stores it as written:
        /// the function rounds the alpha to the nearest-even half Metal stores, and the coverage
        /// reads the rounded alpha where Metal's reads the alpha before rounding, so a fragment
        /// whose alpha lies within half a `half` step of a coverage threshold may cover one
        /// sample more or fewer.
        CoverageReadsRoundedAlpha = "coverage-reads-rounded-alpha";
        /// The blend of a color attachment the fragment function reads, computed by the shader in
        /// the pixel interlock's storage form, of a format whose Metal blend arithmetic is not
        /// built (a 32-bit float, 16-bit normalized, 8-bit signed normalized, 10-bit, packed
        /// float or `A8Unorm` format, among others, an 8-bit unorm one written by fewer than four
        /// lanes, or a 16-bit float or 8-bit unorm one under a blend of the second source): in
        /// `float`, each step rounded, the sources and the blend color clamped to a normalized
        /// format's range, the device's conversion storing the result; or of an 8-bit sRGB format
        /// under a blend of the second source, or a one- or two-channel one under a lerp, in a
        /// module compiled with fast math, or written by fewer than four lanes, in the 8-bit unorm
        /// blend's rounded forms over the decoded texel. Metal's blend for these is not measured.
        UnmeasuredFetchBlend = "unmeasured-fetch-blend";
    }
}

impl Degradation {
    /// The codes of retired kinds, each with the notices that replace it, or `none: ` and why no
    /// notice does where the degradation itself is gone: never announced again, and never a code
    /// of another kind (the module docs). A consumer that meets one in an older log reads its
    /// meaning here.
    pub const RETIRED_CODES: &'static [(&'static str, &'static str)] = &[
        (
            // A texture with a depth twin announced it when made, whether or not the device ever
            // compared it. Every comparison that reads the twin is a device comparison, which its
            // pipeline announces.
            "metal2vulkan-degrade-depth-twin-comparison",
            "metal2vulkan-degrade-device-depth-comparison and, on a device without \
             VK_EXT_depth_range_unrestricted, metal2vulkan-degrade-restricted-depth-range, which \
             the pipeline whose stage makes the device's depth comparison announces",
        ),
        (
            // A stage rounded a `half` through the device's `OpQuantizeToF16`, in the device's
            // rounding mode. Every `half` rounding is Metal's conversion now, on every device.
            "metal2vulkan-degrade-device-half-rounding",
            "none: every half rounding is Metal's conversion to half, on every device",
        ),
        (
            // Retired: a map's parameter data copied into a buffer announced it, its bytes this
            // crate's layout. The bytes are Metal's now, Apple's layout of 24,624 bytes, and a
            // translated decoder reads them as Apple's GPU does.
            "metal2vulkan-degrade-rate-map-parameter-data",
            "none: a rasterization-rate map's parameter data is Metal's bytes, which a translated \
             decoder reads with Apple's arithmetic",
        ),
        (
            // Retired: every linear sample of a `YCBCR8_420_2P` texture announced it at bind. The
            // function emulates Metal's filter of the format on every device, and announces a
            // sample that does not take it.
            "metal2vulkan-degrade-ycbcr-linear-filter",
            "metal2vulkan-degrade-ycbcr-filter-not-exact, which the pipeline whose stage filters \
             a two-plane YCbCr texture otherwise than Metal's exact filter of the format announces",
        ),
        (
            // A texture made after a texture reservation that followed the device's first command
            // queue or indirect command buffer passed over an ID Metal hands out a second time.
            // Plain allocation hands it out as Metal does now.
            "metal2vulkan-degrade-reissued-resource-id",
            "none: plain allocation hands out an ID a live texture holds where Metal does",
        ),
        (
            // Retired: a draw in a pass that loads a padded 10-bit color attachment the load pass
            // did not take announced it, a rotated texture's last. The load pass rewrites a rotated
            // texture's frame now, and takes every padded attachment.
            "metal2vulkan-degrade-padded-bits-kept",
            "metal2vulkan-degrade-unreached-tiles-rewritten, which a pass whose load pass \
             rewrites a padded attachment's every tile announces where that is not Metal's, and \
             metal2vulkan-degrade-extended-range-bits-kept, which names an attachment the load \
             pass does not take",
        ),
        (
            // Retired: a draw whose fragment function read an 8-bit four-channel sRGB color
            // attachment through framebuffer fetch as an input attachment (lavapipe) announced it,
            // the device's decode not Metal's table. The function snaps what the device decodes to
            // Metal's decode of the same code now, so the value is Metal's.
            "metal2vulkan-degrade-input-attachment-srgb-decode",
            "none: an 8-bit four-channel sRGB color read through an input attachment is Metal's \
             decode of its code",
        ),
        (
            // A device whose driver release was newer than the last one measured took the driver
            // facts measured on earlier releases and announced it. An unmeasured identity takes no
            // answer fact now, only the exact path, and assumes its driver's defects.
            "metal2vulkan-degrade-beyond-measured-driver-release",
            "none: a device whose driver, GPU and release were not measured takes the exact path \
             in place of an answer measured elsewhere, which degrades nothing; it says so in the \
             informational message metal2vulkan-unmeasured-driver",
        ),
        (
            // A device whose identity was not measured announced it as a performance degradation.
            // The paths it takes are exact, so nothing is degraded, and a consumer that ends on a
            // degrade notice would end on a driver update.
            "metal2vulkan-degrade-unmeasured-driver",
            "none: the paths an unmeasured identity takes are exact; the device says so in the \
             informational message metal2vulkan-unmeasured-driver",
        ),
        (
            // A stage that kept a `half` result's `float` value under the opt-in
            // `HalfRoundingPolicy::Deferred` announced it. The policy is gone, so every `half`
            // result rounds where it is computed.
            "metal2vulkan-degrade-deferred-half-rounding",
            "none: the opt-in deferred half rounding is removed; every half result rounds where it \
             is computed, under native half rounding (metal2vulkan-degrade-native-half-conversion \
             or metal2vulkan-degrade-native-half-quantize) or Metal's",
        ),
        (
            // A shadow copied back over bytes the CPU may write, under the default
            // `ShadowHostWritesPolicy::CopyBack`, announced it: a CPU write made while the command
            // buffer ran, beside the commands' writes, could be undone. The policy is gone, and
            // such a shadow is always merged back.
            "metal2vulkan-degrade-shadow-copy-back-race",
            "none: a shadow over bytes the CPU may write is always merged back, so the CPU's \
             writes beside the commands' stay, as Metal keeps them",
        ),
        (
            // A no-copy buffer's release, under the default `ImportReleasePolicy::Background`,
            // announced it: its deallocator ran on a handler thread after the release returned.
            // The policy is gone, and a deallocator waiting for the import's free runs in the
            // release.
            "metal2vulkan-degrade-deallocator-after-release",
            "none: a no-copy buffer's deallocator runs in its release, on the releasing thread, \
             as Metal runs it",
        ),
        (
            // A stage compiled under the opt-in `FramebufferFetchPolicy::AcceptUnordered` read a
            // color attachment unordered where the device has no ordered form, and announced it.
            // The policy is gone: such a fetch is refused.
            "metal2vulkan-degrade-unordered-framebuffer-fetch",
            "none: the opt-in unordered framebuffer fetch is removed; a fetch no ordered form \
             serves on the device is refused (Unsupported FramebufferFetch)",
        ),
        (
            // An aliasing image's first move over a no-copy buffer's host pages changed bytes the
            // host kept across it, on pages the process maps read-only. No image is bound to a
            // no-copy buffer's host pages: a texture over them is a shadow, which writes no byte
            // its commands do not write.
            "metal2vulkan-degrade-unrestored-first-move-bytes",
            "none: a texture over a no-copy buffer's host pages is a shadow, which takes no first \
             move over them",
        ),
        (
            // A sampler state made while a committed command buffer that listed the argument
            // samplers had not completed announced it. Each batch's submission lists the states
            // live then, so a state made before the wait that orders its ID's write is the one
            // its function reads, as Metal's.
            "metal2vulkan-degrade-resource-id-takeover",
            "none: a command reads the sampler states live as its batch is submitted, so a state \
             made before the event wait that orders its ID's write is the one its function reads, \
             as on Metal",
        ),
    ];

    /// The kind of `code`, if any.
    pub fn from_code(code: &str) -> Option<Degradation> {
        Degradation::ALL
            .iter()
            .copied()
            .find(|kind| kind.code() == code)
    }

    /// The kind's bit: its word in [`Announced`], and the bit in it.
    fn bit(self) -> (usize, u64) {
        let index = self as usize;
        (index / 64, 1 << (index % 64))
    }
}

/// A degrade notice's payload: what [`DebugMessage::degradation`] carries. No Metal counterpart:
/// Metal has no near-equivalents to announce.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DegradeNotice {
    /// What was degraded: match on it, not on the text.
    pub degradation: Degradation,
    /// The object degraded, the first the device met: its kind and what identifies it.
    pub object: String,
    /// Why, with the object's numbers.
    pub reason: String,
    /// What it costs: the kind's [`cost`](Degradation::cost).
    pub cost: &'static str,
    /// The object's ID, for a kind met once per object, whose objects
    /// [`Device::degraded_objects`] counts ([`Degradation::StaleDetachedReach`],
    /// [`Degradation::DetachedStorageKeepsWrites`], [`Degradation::HeapTextureLosslessLayout`],
    /// [`Degradation::HeapTextureBlockCompressedLayout`] and
    /// [`Degradation::HeapTextureUntiledLayout`]: the texture's, the first the device met, or
    /// each's where `DeviceOptions::announce_each` asks); `None` for every other kind.
    pub object_id: Option<ObjectId>,
    /// For [`Degradation::MisplacedNotice`], the misplaced notice's kind and where it was
    /// announced; `None` for every other kind.
    pub misplaced: Option<MisplacedNotice>,
    /// The render encoder's call whose lowering announced it, where the runtime lowered the call
    /// after it returned (`RenderLoweringPolicy::Worker`, the default): its Metal selector and
    /// its place among the encoder's calls. `None` for a notice announced inside the program's
    /// call itself, and for every other notice.
    pub call: Option<EncoderCall>,
}

/// A notice announced where the runtime's rule announces none
/// ([`Degradation::MisplacedNotice`]'s payload). No Metal counterpart.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MisplacedNotice {
    /// The misplaced notice's kind, which reached the sink unchanged just before.
    pub kind: Degradation,
    /// Where it was announced.
    pub place: NoticePlace,
}

/// Where a notice was announced against the runtime's rule ([`MisplacedNotice::place`]). No
/// Metal counterpart.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NoticePlace {
    /// Inside a pipeline pool's piece, where no kind is announced: what the program sees is
    /// decided on its own thread, in its call.
    PipelinePoolPiece,
    /// A kind announced at every occurrence (`DeviceOptions::announce_each`), on a thread
    /// outside the program's call and outside a command buffer's host step.
    OutsideProgramCall,
}

impl NoticePlace {
    /// The line a build with debug assertions ends the process with.
    pub(crate) fn line(self) -> &'static str {
        match self {
            NoticePlace::PipelinePoolPiece => {
                "metal2vulkan-runtime: a degradation was announced inside a pipeline pool piece"
            }
            NoticePlace::OutsideProgramCall => {
                "metal2vulkan-runtime: a degradation asked for at every occurrence was announced \
                 outside the program's call and outside a host step"
            }
        }
    }

    /// Where, in words.
    pub(crate) fn words(self) -> &'static str {
        match self {
            NoticePlace::PipelinePoolPiece => "inside a pipeline pool's piece",
            NoticePlace::OutsideProgramCall => {
                "outside the program's call and outside a command buffer's host step"
            }
        }
    }
}

impl DegradeNotice {
    /// The notice of `degradation` for `object`, degraded for `reason`.
    pub(super) fn new(degradation: Degradation, object: String, reason: String) -> DegradeNotice {
        DegradeNotice {
            degradation,
            object,
            reason,
            cost: degradation.cost(),
            object_id: None,
            misplaced: None,
            call: crate::render::lowering::current(),
        }
    }

    /// The notice as a message: a warning of the performance kind (a performance degradation)
    /// or the general kind (a precision or validity one), its `id_name` the code, its text
    /// `metal2vulkan: <object>: <reason>; <cost> [<code>]`.
    pub(super) fn message(self) -> DebugMessage {
        let code = self.degradation.code();
        let performance = self.degradation.category() == DegradationCategory::Performance;
        DebugMessage {
            severity: MessageSeverity::Warning,
            kind: MessageKind {
                general: !performance,
                performance,
                ..MessageKind::default()
            },
            id_name: code.to_owned(),
            id: 0,
            text: format!(
                "metal2vulkan: {}: {}; {} [{code}]",
                self.object, self.reason, self.cost
            ),
            degradation: Some(self),
        }
    }
}

/// The words of [`Announced`].
const WORDS: usize = Degradation::ALL.len().div_ceil(64);

/// The kinds a device announced, a bit each, and those it announces at every occurrence
/// (`DeviceOptions::announce_each`), a bit each.
#[derive(Debug, Default)]
pub(super) struct Announced {
    seen: [AtomicU64; WORDS],
    each: [u64; WORDS],
}

impl Announced {
    /// The record of a device that announces the kinds of `each` at every occurrence.
    pub(super) fn new(each: &BTreeSet<Degradation>) -> Announced {
        let mut words = [0; WORDS];
        for kind in each {
            let (word, bit) = kind.bit();
            words[word] |= bit;
        }
        Announced {
            seen: Default::default(),
            each: words,
        }
    }

    /// Whether `degradation` is announced now: at each occurrence of a kind the device announces
    /// at each (met at every occurrence, [`EachWatch`]), else for the first time, once per kind.
    pub(super) fn now(&self, degradation: Degradation) -> bool {
        let (word, bit) = degradation.bit();
        if self.each[word] & bit != 0 {
            met_each();
            return true;
        }
        let seen = &self.seen[word];
        seen.load(Ordering::Relaxed) & bit == 0 && seen.fetch_or(bit, Ordering::Relaxed) & bit == 0
    }

    /// Whether `degradation` is announced now for the first time, once per kind, whether or not
    /// the device announces it at each occurrence: a kind met where no program call or host step
    /// is (`Device::degrade_encoded_use_rebound`).
    fn once(&self, degradation: Degradation) -> bool {
        let (word, bit) = degradation.bit();
        let seen = &self.seen[word];
        seen.load(Ordering::Relaxed) & bit == 0 && seen.fetch_or(bit, Ordering::Relaxed) & bit == 0
    }

    /// Whether `degradation` would be announced now ([`Announced::now`]), without counting it
    /// announced: a kind the device announces at every occurrence, or one not announced yet.
    fn pending(&self, degradation: Degradation) -> bool {
        let (word, bit) = degradation.bit();
        self.each[word] & bit != 0 || self.seen[word].load(Ordering::Relaxed) & bit == 0
    }

    /// Whether the device announces `degradation` at every occurrence.
    fn each(&self, degradation: Degradation) -> bool {
        let (word, bit) = degradation.bit();
        self.each[word] & bit != 0
    }

    /// Whether the device announces any kind of `kinds` at every occurrence: an AND a word.
    fn any_each(&self, kinds: &Kinds) -> bool {
        self.each
            .iter()
            .zip(&kinds.0)
            .any(|(each, kinds)| each & kinds != 0)
    }
}

thread_local! {
    /// How many notices this thread met that their device announces, or refuses, at every
    /// occurrence ([`EachWatch`]).
    static MET_EACH: Cell<u64> = const { Cell::new(0) };
}

/// Count a notice met on this thread that its device announces, or refuses, at every occurrence.
fn met_each() {
    MET_EACH.with(|met| met.set(met.get().wrapping_add(1)));
}

/// A watch over the notices the calling thread meets, from its start, that their device announces
/// at every occurrence (`DeviceOptions::announce_each`: [`Announced::now`], a bind's notices) or
/// refuses at every occurrence while it lowers a render encoder's call
/// (`DeviceOptions::lowering_refuses`, wherever it is met). Work that met none while watched announced nothing its
/// next run from the same inputs would announce again, so a memo may keep what it derived and a
/// later use take it without deriving it again (`render/draw_shape.rs`): every other kind is
/// announced once per device, which its first derivation did. It needs no list of the kinds the
/// work may announce, so it cannot drift from the work's code.
pub(crate) struct EachWatch(u64);

impl EachWatch {
    /// A watch from now.
    pub(crate) fn start() -> EachWatch {
        EachWatch(MET_EACH.with(Cell::get))
    }

    /// Whether this thread met such a notice since the watch started.
    pub(crate) fn met(&self) -> bool {
        MET_EACH.with(Cell::get) != self.0
    }
}

/// `recorded`, the reason of a stage that records `PrecisionNote::FunctionStorageAssumed`, with
/// the private memory a thread of the stage keeps, `private_bytes`, and the Function storage the
/// device holds, `held`, measured or the floor of a device not measured: numbers only.
pub(crate) fn function_storage_reason(
    recorded: &str,
    private_bytes: u64,
    held: FunctionStorage,
) -> String {
    let evidence = match held {
        FunctionStorage::Measured(_) => "measured on this device",
        _ => "the floor of a device whose Function storage is not measured",
    };
    format!(
        "{recorded}: it keeps {private_bytes} bytes of private memory a thread, past the {} bytes \
         a thread {evidence}",
        held.bytes()
    )
}

/// Kinds of degradation, a bit each: those a variant's making may announce
/// (`VariantObjects::kinds`), which a bind of the variant tests against the kinds the device
/// announces at every occurrence before it announces any.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Kinds([u64; WORDS]);

impl Kinds {
    /// The kinds `stages` (a variant's) may announce where the variant or a `VkPipeline` of it is
    /// made (`Device::degrade_stages`, `Device::degrade_specialized`), and those of the variant's
    /// `pipeline` (`Device::degrade_pipeline`): every note's, a specialized one's under any
    /// specialization (a filter note's two-plane YCbCr kind among them), and each other kind a
    /// stage's or the pipeline's records state.
    pub(crate) fn of(stages: &[StageModule], pipeline: CompiledPipeline<'_>) -> Kinds {
        let mut kinds = Kinds::default();
        for stage in stages {
            let notes = stage.precision_notes.iter().copied();
            for note in notes.chain(stage.specialized_notes.iter().map(|s| s.note)) {
                kinds.add(Degradation::from_precision_note(note));
                if matches!(
                    note,
                    PrecisionNote::HalfSampleFilter | PrecisionNote::FloatSampleFilter
                ) {
                    kinds.add(Degradation::YCbCrFilterNotExact);
                }
            }
            if !stage.half_bit_colors.is_empty() {
                kinds.add(Degradation::ShaderHalfColorBits);
            }
            if !stage.rounded_colors.is_empty() {
                kinds.add(Degradation::ShaderColorConversion);
            }
            if stage.reconvergence == Reconvergence::ImplementationDefined {
                kinds.add(Degradation::ImplementationDefinedReconvergence);
            }
        }
        if let CompiledPipeline::Render(render) = pipeline {
            if render
                .requirements
                .properties
                .contains(&VkProperty::MetalVertexSimdgroups)
            {
                kinds.add(Degradation::UnadvertisedVertexSimdgroups);
            }
        }
        if dropped_exact_forms(pipeline.dependencies()).is_some() {
            kinds.add(Degradation::ExactFormsOverCompilerLimit);
        }
        kinds
    }

    fn add(&mut self, kind: Degradation) {
        let (word, bit) = kind.bit();
        self.0[word] |= bit;
    }
}

/// A variant's compiled pipeline, whose records beside its stages its making and binds announce
/// (`Device::degrade_pipeline`): a render state's (vertex or mesh) or a compute state's (kernel or
/// tile).
#[derive(Clone, Copy)]
pub(crate) enum CompiledPipeline<'a> {
    Render(&'a RenderPipeline),
    Kernel(&'a ComputePipeline),
}

impl<'a> CompiledPipeline<'a> {
    /// The pipeline's layout.
    fn layout(self) -> &'a metal2vulkan::pipeline::PipelineLayout {
        match self {
            CompiledPipeline::Render(p) => &p.layout,
            CompiledPipeline::Kernel(p) => &p.layout,
        }
    }

    /// The pipeline's runtime dependencies.
    fn dependencies(self) -> &'a [metal2vulkan::pipeline::RuntimeDependency] {
        match self {
            CompiledPipeline::Render(p) => &p.dependencies,
            CompiledPipeline::Kernel(p) => &p.dependencies,
        }
    }

    /// The kind of pipeline state a notice of the pipeline names, from its structure: a render
    /// pipeline with an object or mesh stage, or with the records of an emulated mesh, is a mesh
    /// render pipeline state's; a compute pipeline of a tile function
    /// (`RuntimeFact::TileFunction`) a tile render pipeline state's.
    fn object(self) -> &'static str {
        match self {
            CompiledPipeline::Render(p) => {
                let meshes = p.graphics.mesh_records.is_some()
                    || p.stages.iter().any(|s| {
                        s.stage == ShaderStage::TASK_EXT || s.stage == ShaderStage::MESH_EXT
                    });
                if meshes {
                    "a mesh render pipeline state"
                } else {
                    "a render pipeline state"
                }
            }
            CompiledPipeline::Kernel(p) => {
                let tile = p
                    .dependencies
                    .iter()
                    .any(|d| d.fact == metal2vulkan::pipeline::RuntimeFact::TileFunction);
                if tile {
                    "a tile render pipeline state"
                } else {
                    "a compute pipeline state"
                }
            }
        }
    }
}

/// The exact forms a pipeline's translation dropped to fit the device's compiler word bound
/// (`RuntimeFact::ExactFormsOverCompilerLimit` among its `dependencies`), named; `None` where it
/// dropped none.
pub(crate) fn dropped_exact_forms(
    dependencies: &[metal2vulkan::pipeline::RuntimeDependency],
) -> Option<Vec<&'static str>> {
    dependencies.iter().find_map(|d| match d.fact {
        metal2vulkan::pipeline::RuntimeFact::ExactFormsOverCompilerLimit {
            sampling,
            float_maths,
            texel_conversion,
            half_rounding,
            nan_and_min_max,
            rounding_order,
            half_blend,
        } => Some(
            [
                (sampling, "exact sampling"),
                (float_maths, "exact float maths"),
                (texel_conversion, "exact texel conversion"),
                (half_rounding, "exact half rounding"),
                (nan_and_min_max, "exact NaN and minimum and maximum"),
                (rounding_order, "exact rounding order"),
                (half_blend, "exact half blend"),
            ]
            .into_iter()
            .filter_map(|(dropped, name)| dropped.then_some(name))
            .collect(),
        ),
        _ => None,
    })
}

/// Where a variant's notices are announced: as the variant (or a `VkPipeline` of it) is made,
/// or as a dispatch or draw binds it.
#[derive(Clone, Copy)]
pub(crate) enum Occurrence<'a> {
    /// The making: every kind but those the device announces at every occurrence, once.
    Made,
    /// A bind: the kinds the device announces at every occurrence, at each, made into the
    /// bind's notices, which the bind announces (`Device::bound_notices`).
    Bound(&'a std::cell::RefCell<Vec<BoundNotice>>),
    /// A dispatch or draw that binds values its functions read as data, which no making knows:
    /// every kind, once, or at each occurrence where the device announces it so
    /// ([`Device::degrade`]).
    Drawn,
}

/// One notice a bind of a variant announces (`Device::bound_notices`): its kind, and the message
/// the device's sink hears.
pub(crate) type BoundNotice = (Degradation, DebugMessage);

/// Whether a kind announced at every occurrence (`each`) may be announced on this thread now:
/// in a host step (`exit_guard::host_step`), or in the program's call
/// (`exit_guard::in_program_call`) outside a pipeline pool's piece. Any kind otherwise.
pub(crate) fn each_in_order(each: bool) -> bool {
    !each
        || crate::exit_guard::in_host_step()
        || (crate::exit_guard::in_program_call() && !crate::pipeline::pool::in_piece())
}

/// Where an announcement on this thread now breaks the rule, `None` where it keeps it: no kind is
/// announced inside a pipeline pool's piece (whatever the program sees is decided on its own
/// thread, in its call; a piece makes its pipelines with the notices announced, or announced at the
/// first draw), and a kind asked for at every occurrence (`each`) only in the program's call or a
/// host step ([`each_in_order`]).
pub(crate) fn announcement_breach(each: bool) -> Option<NoticePlace> {
    if crate::pipeline::pool::in_piece() {
        Some(NoticePlace::PipelinePoolPiece)
    } else if !each_in_order(each) {
        Some(NoticePlace::OutsideProgramCall)
    } else {
        None
    }
}

/// `Invalid` (`Problem::UndefinedValue`) where `options` set [`ResourceOptions::PRIVATE_BIT`].
fn refuse_private_bit(call: &'static str, options: ResourceOptions) -> Result<(), Error> {
    match options.without_private_bit().1 {
        true => Err(crate::error::invalid(
            call,
            crate::error::Problem::UndefinedValue {
                ty: "MTLResourceOptions",
                raw: options.raw(),
            },
        )),
        false => Ok(()),
    }
}

/// Whether a misplaced announcement ends the process: in builds with debug assertions, which
/// every lane and gate runs.
pub(crate) const MISPLACED_ENDS: bool = cfg!(debug_assertions);

/// Where announcing `degradation` on this thread now breaks the rule ([`announcement_breach`]),
/// `None` where it keeps it. A breach ends the process, naming `degradation`, in builds with debug
/// assertions ([`MISPLACED_ENDS`]), so every lane and gate fails on it. In other builds its place
/// is answered, and the announcement goes to the notice thread (`misplaced_notices.rs`): a handler
/// may end the process at a notice, which on this thread would end it out of the program's order
/// and perhaps under a lock of the runtime's, and a sink keeping what it heard in the program's
/// thread's state would miss it.
pub(crate) fn check_announcement(degradation: Degradation, each: bool) -> Option<NoticePlace> {
    placement(degradation, each, MISPLACED_ENDS)
}

/// [`check_announcement`], ending the process at a breach where `ends`.
pub(crate) fn placement(degradation: Degradation, each: bool, ends: bool) -> Option<NoticePlace> {
    let place = announcement_breach(each)?;
    if ends {
        eprintln!("{}: {degradation:?}", place.line());
        std::process::abort();
    }
    Some(place)
}

impl Device {
    /// Announce `AttachmentReadInPlace` where `uses`, what a draw of `call` binds or reaches,
    /// share memory with an attachment of `layout`'s render pass instance
    /// (`render_target::shares_attachment`). The check runs only while the kind would be
    /// announced: once it was, a device that announces it once skips it.
    pub(super) fn degrade_attachment_reads(
        &self,
        call: &'static str,
        layout: &PassLayout,
        uses: &[Use<ResourceKey>],
    ) -> Result<(), Error> {
        let kind = Degradation::AttachmentReadInPlace;
        if !self.shared().announced.pending(kind)
            || !render_target::shares_attachment(layout, uses)?
        {
            return Ok(());
        }
        self.degrade(kind, || {
            (
                format!("a draw ({call})"),
                "a function of the draw reads or writes, other than as an attachment, memory an \
                 attachment of its render pass instance renders"
                    .to_owned(),
            )
        });
        Ok(())
    }

    /// Announce `degradation` unless the device announced it before (but a kind its options ask
    /// for at each occurrence, `DeviceOptions::announce_each`), `describe` giving the object and
    /// the reason: through the device's message sink, else on stderr, on the calling thread (the
    /// module docs).
    pub(super) fn degrade(
        &self,
        degradation: Degradation,
        describe: impl FnOnce() -> (String, String),
    ) {
        let shared = self.shared();
        // A kind the device refuses while lowering a call after it returned refuses the call,
        // at every occurrence, rather than being announced (`render/deferral.rs`): met at every
        // occurrence wherever it is met ([`EachWatch`]), as a call met in one place may be
        // lowered in the other.
        let refuses = shared.lowering_refuses(degradation);
        if refuses {
            met_each();
        }
        if refuses && crate::render::lowering::current().is_some() {
            let (object, reason) = describe();
            crate::render::lowering::refuse(DegradeNotice::new(degradation, object, reason));
            return;
        }
        // A call encoded again at a rebind met what the program's call met, which announced it:
        // each kind at most once there (`replay.rs`).
        let announce = match crate::replay::is_replaying() {
            true => shared.announced.once(degradation),
            false => shared.announced.now(degradation),
        };
        if !announce {
            return;
        }
        // No kind is announced inside a pipeline pool's piece, and a kind asked for at every
        // occurrence is announced inside the program's own call that met it, or in a command
        // buffer's host step, a serial point of its stream before the batch is submitted: a
        // consumer's handler may end the process at it, which a pipeline pool's piece, or any
        // other thread of the runtime's, would do out of the program's order (the owner's HITCH
        // rule 1), and a sink keeping what it heard in its thread's state would miss it.
        let misplaced = check_announcement(degradation, shared.announced.each(degradation));
        let (object, reason) = describe();
        let message = DegradeNotice::new(degradation, object, reason).message();
        self.deliver_placed(degradation, message, misplaced);
    }

    /// Announce `degradation` once per device, whatever `DeviceOptions::announce_each` says,
    /// `describe` giving the object and the reason: for a kind met where no program call or host
    /// step may be (a runtime thread letting go of what it held).
    pub(super) fn degrade_once(
        &self,
        degradation: Degradation,
        describe: impl FnOnce() -> (String, String),
    ) {
        if !self.shared().announced.once(degradation) {
            return;
        }
        let misplaced = check_announcement(degradation, false);
        let (object, reason) = describe();
        let message = DegradeNotice::new(degradation, object, reason).message();
        self.deliver_placed(degradation, message, misplaced);
    }

    /// Announce, once, that a command buffer's command named a buffer rebound after it was
    /// encoded and before the command buffer was first submitted
    /// (`CommandBufferError::EncodedUseRebound`), where its first submission `found` the first
    /// such use, and why no rebind encoded it again (`command_buffer/rebound_use.rs`): on the
    /// thread that submits it, with no lock of the command buffer's held.
    pub(super) fn degrade_encoded_use_rebound(
        &self,
        found: Option<crate::command_buffer::rebound_use::ReboundUse>,
    ) {
        let shared = self.shared();
        if !shared.announced.once(Degradation::EncodedUseRebound) {
            return;
        }
        let misplaced = check_announcement(Degradation::EncodedUseRebound, false);
        let mut reason = "a buffer over host pages they named was rebound after they were \
                          encoded and before the command buffer was submitted, and their use was \
                          resolved as encoded"
            .to_owned();
        if let Some(found) = found {
            reason.push_str(&format!(" ({found})"));
        }
        let message = DegradeNotice::new(
            Degradation::EncodedUseRebound,
            "a command buffer's commands".to_owned(),
            reason,
        )
        .message();
        self.deliver_placed(Degradation::EncodedUseRebound, message, misplaced);
    }

    /// Announce `kind` (`StaleDetachedReach` or `DetachedStorageKeepsWrites`) for the texture of
    /// `texture`, which a command buffer may reach without naming it while its bytes are
    /// detached (`Device::reach_stale_detached`, `Device::reach_storage_sinks`, which meet each
    /// texture once and count it): with no lock of the command buffer's held. Once per kind,
    /// naming the first texture; `DetachedStorageKeepsWrites` at each texture where the device's
    /// options ask (it is met in the program's `commit`), `StaleDetachedReach` once whatever
    /// they say (a submission may run on the scheduler's thread, outside any program call or
    /// host step).
    pub(super) fn degrade_detached_reach(&self, kind: Degradation, texture: ObjectId) {
        let shared = self.shared();
        let each = kind != Degradation::StaleDetachedReach && shared.announced.each(kind);
        let now = match each {
            true => shared.announced.now(kind),
            false => shared.announced.once(kind),
        };
        if !now {
            return;
        }
        let misplaced = check_announcement(kind, each);
        let reason = match kind {
            Degradation::StaleDetachedReach => {
                "its bytes were detached after a submitted command buffer made them resident, and \
                 a submitted command buffer bound a heap set, through which its commands may read \
                 it without naming it"
            }
            _ => {
                "its bytes are detached and its stand-in has a sink, and a committed command \
                 buffer bound a heap set, through which its commands may write and read it \
                 without naming it"
            }
        };
        let mut notice = DegradeNotice::new(
            kind,
            format!("texture of object ID {}", texture.get()),
            reason.to_owned(),
        );
        notice.object_id = Some(texture);
        self.deliver_placed(kind, notice.message(), misplaced);
    }

    /// Count the texture of `texture`, made in a heap in this program call, as laid out
    /// otherwise than Metal lays it in the heap's bytes, for `untiled`'s reason
    /// (`heap-texture-*-layout`, `texture/heap_tiled.rs`, `texture/heap.rs`), and announce it, with
    /// its ID, where it is the first of its kind (or at each texture, where the device's options
    /// ask).
    pub(super) fn degrade_heap_layout(&self, untiled: Untiled, texture: ObjectId) {
        let shared = self.shared();
        let kind = untiled.kind;
        shared.counts.heap_layout_degraded(kind);
        if !shared.announced.now(kind) {
            return;
        }
        let misplaced = check_announcement(kind, shared.announced.each(kind));
        let mut notice = DegradeNotice::new(
            kind,
            format!("texture of object ID {}", texture.get()),
            untiled.reason.to_owned(),
        );
        notice.object_id = Some(texture);
        self.deliver_placed(kind, notice.message(), misplaced);
    }

    /// How many objects this device met as `kind` so far, for a kind it meets once per object
    /// and announces once (naming the first object, [`DegradeNotice::object_id`]), so that a
    /// census counts them without a notice each: the textures made in heaps laid out otherwise
    /// than Metal lays them in the heap's bytes ([`Degradation::HeapTextureLosslessLayout`],
    /// [`Degradation::HeapTextureBlockCompressedLayout`] and
    /// [`Degradation::HeapTextureUntiledLayout`], each counted as it is made), and the textures
    /// over detached bytes a command buffer could reach unnamed
    /// ([`Degradation::StaleDetachedReach`] and [`Degradation::DetachedStorageKeepsWrites`], each
    /// counted once, where first reached). `None` for every other kind, which the device
    /// announces once without counting. Counted whether or not the device's options ask for
    /// each occurrence (`DeviceOptions::announce_each`). No Metal counterpart.
    pub fn degraded_objects(&self, kind: Degradation) -> Option<u64> {
        self.shared().counts.degraded_objects(kind)
    }

    /// `degrade` a variant's notice `at` its making or a bind: a kind the
    /// device announces at every occurrence at a bind alone, every other kind at the making
    /// alone.
    fn degrade_at(
        &self,
        at: Occurrence<'_>,
        degradation: Degradation,
        describe: impl FnOnce() -> (String, String),
    ) {
        let each = self.shared().announced.each(degradation);
        match at {
            Occurrence::Made if !each => self.degrade(degradation, describe),
            // Made into the bind's notices, which it announces as `degrade` would: a kind the
            // device announces at every occurrence is announced at each.
            Occurrence::Bound(notices) if each => {
                let (object, reason) = describe();
                let message = DegradeNotice::new(degradation, object, reason).message();
                notices.borrow_mut().push((degradation, message));
            }
            Occurrence::Made | Occurrence::Bound(_) => {}
            Occurrence::Drawn => self.degrade(degradation, describe),
        }
    }

    /// Whether a bind of a variant of `kinds` (`Kinds::of`) announces any of them: whether the
    /// device's options ask for any at every occurrence (`DeviceOptions::announce_each`), an AND
    /// a word. A bind that would take work to describe its notices (the formats it binds) asks
    /// first.
    pub(crate) fn binds_each(&self, kinds: &Kinds) -> bool {
        self.shared().announced.any_each(kinds)
    }

    /// The notices a dispatch's or draw's bind of a variant announces
    /// ([`announce_bound`](Self::announce_bound)): those its making announces whose kinds the
    /// device's options ask for at every occurrence (`DeviceOptions::announce_each`), with the
    /// making's object and reason, in the order the making announces them (a memoized variant is
    /// made once, so its making alone would announce them once). `stages` are the variant's
    /// stages, `pipeline` its compiled pipeline, and `specializations` the specializations and
    /// bound formats of the `VkPipeline`s the bind takes (`Device::degrade_specialized`).
    pub(crate) fn bound_notices(
        &self,
        (stages, pipeline): (&[StageModule], CompiledPipeline<'_>),
        specializations: &[(&SpecializationKey, Option<BoundFormats<'_>>)],
    ) -> Vec<BoundNotice> {
        let notices = std::cell::RefCell::new(Vec::new());
        let at = Occurrence::Bound(&notices);
        self.degrade_pipeline(pipeline, at);
        self.degrade_stages(stages, at);
        for (n, &(specialization, formats)) in specializations.iter().enumerate() {
            // A specialization the bind takes twice (a split dispatch's regions) announces once.
            if specializations[..n]
                .iter()
                .any(|&(earlier, _)| earlier == specialization)
            {
                continue;
            }
            self.degrade_specialized(stages, specialization, formats, at);
        }
        notices.into_inner()
    }

    /// Announce `notices`, a bind's ([`bound_notices`](Self::bound_notices)), on the calling
    /// thread, each as [`degrade`](Self::degrade) announces a kind the device announces at every
    /// occurrence: through the device's message sink, else on stderr.
    pub(crate) fn announce_bound(&self, notices: &[BoundNotice]) {
        for (degradation, message) in notices {
            met_each();
            match check_announcement(*degradation, true) {
                None => self.deliver(message),
                misplaced => self.deliver_placed(*degradation, message.clone(), misplaced),
            }
        }
    }

    /// Hand `message` to the device's message sink, else write it on stderr.
    fn deliver(&self, message: &DebugMessage) {
        match self.shared().instance.sink() {
            Some(sink) => sink.deliver(message),
            None => eprintln!("{}", message.text),
        }
    }

    /// Hand `message`, of `degradation`, to the device's message sink on this thread where its
    /// announcement keeps the rule (`misplaced` is `None`); else, in a build without debug
    /// assertions, to the notice thread, with the device's `misplaced-notice` after it
    /// ([`misplaced_notices::deliver`](crate::misplaced_notices::deliver)).
    fn deliver_placed(
        &self,
        degradation: Degradation,
        message: DebugMessage,
        misplaced: Option<NoticePlace>,
    ) {
        match misplaced {
            None => self.deliver(&message),
            Some(place) => {
                let shared = self.shared();
                crate::misplaced_notices::deliver(
                    shared.instance.sink(),
                    &shared.announced,
                    MisplacedNotice {
                        kind: degradation,
                        place,
                    },
                    message,
                );
            }
        }
    }

    /// `options` of `call`'s no-copy buffer or heap over the caller's bytes without
    /// [`ResourceOptions::PRIVATE_BIT`], and whether its memory drops the GPU's writes, as Metal's
    /// does where the options set the bit: on macOS, where the runtime imports the pages through
    /// a read-only view of them (`read_only_view.rs`). Elsewhere the GPU's writes reach the
    /// pages, announced once ([`Degradation::PrivateResourceOption`]).
    pub(super) fn no_copy_options(
        &self,
        call: &'static str,
        options: ResourceOptions,
    ) -> (ResourceOptions, bool) {
        if !crate::read_only_view::VIEWS_DROP_WRITES {
            return (self.take_options(call, options), false);
        }
        options.without_private_bit()
    }

    /// `options` as the runtime makes `call`'s resource of them where Metal makes it with
    /// [`ResourceOptions::PRIVATE_BIT`] and keeps the GPU's writes from its memory (a heap of
    /// the runtime's memory; a no-copy buffer or heap off macOS, [`Device::no_copy_options`]; an
    /// indirect command buffer, whose GPU-written records under it are not measured): without the
    /// bit, announced once ([`Degradation::PrivateResourceOption`]).
    pub(super) fn take_options(
        &self,
        call: &'static str,
        options: ResourceOptions,
    ) -> ResourceOptions {
        let (taken, private) = options.without_private_bit();
        if private {
            self.degrade(Degradation::PrivateResourceOption, || {
                (
                    format!("`{call}` with resource options {:#x}", options.raw()),
                    "the resource is made as without Metal's private option bit 24 (0x1000000), \
                     and its options read without it"
                        .to_owned(),
                )
            });
        }
        taken
    }

    /// `options` of `call`'s resource whose memory the runtime gives it (a buffer of a length or
    /// of bytes, a Vulkan buffer of the caller's): refused where they set
    /// [`ResourceOptions::PRIVATE_BIT`] (`Invalid`, `Problem::UndefinedValue`), as Metal ends the
    /// process there (the M3 Max under macOS 27.0.1: `newBufferWithLength:options:` and
    /// `newBufferWithBytes:length:options:`, `Shared` or `Private`, assert in the buffer's
    /// suballocator, "Different MTLResourceOptions used for the same heap index").
    pub(super) fn own_memory_options(
        &self,
        call: &'static str,
        options: ResourceOptions,
    ) -> Result<ResourceOptions, Error> {
        refuse_private_bit(call, options)?;
        Ok(options)
    }

    /// `descriptor`, refused where its options set [`ResourceOptions::PRIVATE_BIT`] (`Invalid`,
    /// `Problem::UndefinedValue`), as Metal's texture descriptor validation ends the process there
    /// for every texture (the M3 Max under macOS 27.0.1: "MTLResourceOptions (0x1000000) contains
    /// unknown bits", a device's texture, `Shared` or `Private`, a heap's, a buffer's and a
    /// shareable one alike).
    pub(super) fn texture_options(
        &self,
        call: &'static str,
        descriptor: &TextureDescriptor,
    ) -> Result<TextureDescriptor, Error> {
        refuse_private_bit(call, descriptor.resource_options)?;
        Ok(*descriptor)
    }

    /// Announce what `stages` (a variant's) record, `at` its making or a bind
    /// ([`Occurrence`]): each precision note no specialization removes (the specialized ones are
    /// [`Device::degrade_specialized`]'s), and reconvergence that is not Metal's.
    pub(super) fn degrade_stages(&self, stages: &[StageModule], at: Occurrence<'_>) {
        for stage in stages {
            let object = || stage_object(stage.stage).to_owned();
            for &note in &stage.precision_notes {
                if stage.specialized_notes.iter().any(|s| s.note == note) {
                    continue;
                }
                // Announced at a draw or dispatch that lists a state needing it
                // (`Device::degrade_argument_pixel_samplers`), not at the making or a bind.
                if note == PrecisionNote::ArgumentPixelSamplerScaling {
                    continue;
                }
                self.degrade_at(at, Degradation::from_precision_note(note), || {
                    (object(), self.stage_note_reason(note, stage))
                });
            }
            if !stage.half_bit_colors.is_empty() {
                self.degrade_at(at, Degradation::ShaderHalfColorBits, || {
                    (
                        object(),
                        format!(
                            "the device stores a 32-bit output's NaN to the half-float format of \
                             color attachment(s) {:?} as another NaN than the `half`'s own bits",
                            stage.half_bit_colors
                        ),
                    )
                });
            }
            if !stage.rounded_colors.is_empty() {
                self.degrade_at(at, Degradation::ShaderColorConversion, || {
                    (
                        object(),
                        format!(
                            "the device's conversion of a float to the format of color \
                             attachment(s) {:?} is not Metal's near a tie between two codes",
                            stage.rounded_colors
                        ),
                    )
                });
            }
            if stage.reconvergence == Reconvergence::ImplementationDefined {
                self.degrade_at(at, Degradation::ImplementationDefinedReconvergence, || {
                    (
                        object(),
                        "it reads subgroup state after divergence, and the device does not \
                         reconverge maximally"
                            .to_owned(),
                    )
                });
            }
        }
    }
}

impl Device {
    /// Announce, `at` its variant's making or a bind ([`Occurrence`]), what a variant's `pipeline`
    /// records beside its stages: exact forms the translation dropped to fit the device's compiler
    /// word bound ([`Degradation::ExactFormsOverCompilerLimit`]), naming the pipeline state's kind
    /// (render, mesh render, tile render or compute), and, for a render variant, requirements that
    /// hold a vertex function's subgroup operations on a driver that runs them as Metal does
    /// without advertising the vertex stage ([`Degradation::UnadvertisedVertexSimdgroups`]), and,
    /// for any, a set layout its driver refuses though it runs it
    /// ([`Degradation::SetPastLayoutSupport`], before any layout of the variant is made); for the
    /// base variant and every other alike, each pipeline by its own records.
    pub(super) fn degrade_pipeline(&self, pipeline: CompiledPipeline<'_>, at: Occurrence<'_>) {
        if let Some(dropped) = dropped_exact_forms(pipeline.dependencies()) {
            self.degrade_at(at, Degradation::ExactFormsOverCompilerLimit, || {
                (
                    pipeline.object().to_owned(),
                    format!(
                        "its largest stage module passed the device's compiler word bound, so its \
                         translation dropped {} to the device's answer",
                        dropped.join(", ")
                    ),
                )
            });
        }
        if self
            .shared()
            .announced
            .pending(Degradation::SetPastLayoutSupport)
        {
            for set in &pipeline.layout().sets {
                if set_layout_support(self, set) != SetSupport::Understated {
                    continue;
                }
                let total: u64 = set.bindings.iter().map(|b| u64::from(b.count)).sum();
                self.degrade_at(at, Degradation::SetPastLayoutSupport, || {
                    (
                        pipeline.object().to_owned(),
                        format!(
                            "its set {} holds {total} descriptors, which {} makes and runs as \
                             Metal runs the argument buffer it encodes the set in (measured, \
                             `RuntimeDriverDefect::SetLayoutSupportUnderstated`) though its \
                             driver's vkGetDescriptorSetLayoutSupport refuses the layout",
                            set.set,
                            self.name()
                        ),
                    )
                });
            }
        }
        let CompiledPipeline::Render(pipeline) = pipeline else {
            return;
        };
        if pipeline
            .requirements
            .properties
            .contains(&VkProperty::MetalVertexSimdgroups)
        {
            self.degrade_at(at, Degradation::UnadvertisedVertexSimdgroups, || {
                (
                    stage_object(ShaderStage::VERTEX).to_owned(),
                    format!(
                        "it runs simdgroup functions, which {} runs as Metal does in a vertex \
                         function (measured, `DriverDeviation::MetalVertexSimdgroups`) though \
                         its driver does not list the vertex stage among its subgroup stages",
                        self.name()
                    ),
                )
            });
        }
    }
}

impl Device {
    /// Announce, once, a pipeline state whose `dependencies` state that a stage may read an
    /// argument its function constants removed, naming that stage's function by `name`
    /// ([`Degradation::AbsentArgumentRead`]).
    pub(super) fn degrade_absent_reads(
        &self,
        dependencies: &[metal2vulkan::pipeline::RuntimeDependency],
        name: impl Fn(metal2vulkan::pipeline::Stage) -> String,
    ) {
        let Some(stage) = dependencies.iter().find_map(|d| match d.fact {
            metal2vulkan::pipeline::RuntimeFact::AbsentArgumentRead { stage } => Some(stage),
            _ => None,
        }) else {
            return;
        };
        self.degrade(Degradation::AbsentArgumentRead, || {
            (
                format!("the {stage:?} function {}", name(stage)),
                "it reads an argument its function constants removed where something observes \
                 the read"
                    .to_owned(),
            )
        });
    }

    /// Announce, once, a render pipeline state `object` whose `dependencies` state that its draws
    /// run a synthesized geometry stage ([`Degradation::SynthesizedGeometryStage`]), naming why:
    /// the barycentric weights its fragment function reads, or the slice or viewport its vertex
    /// function selects.
    pub(super) fn degrade_synthesized_geometry(
        &self,
        dependencies: &[metal2vulkan::pipeline::RuntimeDependency],
        object: impl FnOnce() -> String,
    ) {
        let Some((weights, layer, viewport)) = dependencies.iter().find_map(|d| match d.fact {
            metal2vulkan::pipeline::RuntimeFact::SynthesizedGeometry {
                weights,
                layer,
                viewport,
            } => Some((weights, layer, viewport)),
            _ => None,
        }) else {
            return;
        };
        self.degrade(Degradation::SynthesizedGeometryStage, || {
            let mut why = Vec::new();
            if weights {
                why.push(
                    "its fragment function reads [[barycentric_coord]] on a device without \
                     fragmentShaderBarycentric",
                );
            }
            if layer {
                why.push(
                    "its vertex function selects its primitive's render target slice on a device \
                     that cannot store Layer from a vertex function",
                );
            }
            if viewport {
                why.push(
                    "its vertex function selects its primitive's viewport on a device that cannot \
                     store ViewportIndex from a vertex function",
                );
            }
            (object(), why.join(", and "))
        });
    }

    /// Announce `PrecisionNote::DeviceNanEncoding` at a draw or dispatch of a pipeline of
    /// `dependencies` whose `half` texel writes to a texture slot read whether its texture's
    /// channels are 16-bit floats (`RuntimeFact::StorageTextureHalfChannels`: the slot's storage
    /// choice declares no format), where `formats` binds there a texture whose stored NaN the
    /// device's conversion decides (`metal::format::half_write_nans_converted`: 16-bit
    /// float channels, or a packed small float's), on a device without
    /// `VkProperty::MetalHalfTexelWrites`; where the command's bound formats are not known
    /// (`formats` `None`: an execution of indirect commands the GPU writes), at every such slot.
    /// A 32-bit float, normalized or integer texture there stores Metal's answer and announces
    /// nothing. The module records no note for such a write: this is its announcement.
    pub(crate) fn degrade_half_texel_writes(
        &self,
        dependencies: &[metal2vulkan::pipeline::RuntimeDependency],
        formats: Option<BoundFormats<'_>>,
    ) {
        use metal2vulkan::pipeline::RuntimeFact;
        self.degrade_heap_half_texel_writes(dependencies);
        let mut slots = dependencies.iter().filter_map(|d| match d.fact {
            RuntimeFact::StorageTextureHalfChannels { slot } => Some(slot),
            _ => None,
        });
        let Some(first) = slots.next() else {
            return;
        };
        if !self
            .shared()
            .announced
            .pending(Degradation::DeviceNanEncoding)
            || self
                .compile_device()
                .profile()
                .enabled()
                .properties
                .contains(&metal2vulkan::device::VkProperty::MetalHalfTexelWrites)
        {
            return;
        }
        let converted = |slot: SlotRef| match formats {
            Some(f) => f(slot).map(|pf| (slot, Some(pf))).filter(|(_, pf)| {
                pf.is_some_and(metal2vulkan::metal::format::half_write_nans_converted)
            }),
            None => Some((slot, None)),
        };
        let Some((slot, format)) = std::iter::once(first).chain(slots).find_map(converted) else {
            return;
        };
        self.degrade(Degradation::DeviceNanEncoding, || {
            let bound = match format {
                Some(pf) => format!("bound as {pf:?}"),
                None => "of a format the GPU's commands bind".to_owned(),
            };
            (
                format!("the {:?} function of a pipeline state", slot.stage),
                format!(
                    "it writes `half` texels to texture {} {bound}, whose NaNs the device \
                     stores as it converts them, not as Metal stores a written half (the \
                     half's own bits in a 16-bit float channel, the format's canonical NaN in a \
                     packed small float one)",
                    slot.index
                ),
            )
        });
    }

    /// Announce `PrecisionNote::DeviceNanEncoding` at a draw or dispatch of a pipeline of
    /// `dependencies` whose writes through heap handles store half-float members' bits through
    /// the unsigned integer twin (`RuntimeFact::HeapHalfTexelWrites`), where the device's heap
    /// holds a live texture such a write stores the device's conversion of a NaN in
    /// (`heap_sets::NanResidue`: a half-float texture the twin does not hold, and for a `half`
    /// write a packed small float one). Every texture the device holds is at its resource ID for
    /// every command (`heap_sets.rs`), so any of them may be one the function's handles name.
    fn degrade_heap_half_texel_writes(
        &self,
        dependencies: &[metal2vulkan::pipeline::RuntimeDependency],
    ) {
        use metal2vulkan::pipeline::RuntimeFact;
        let residue = &self.shared().heap_nan_residue;
        let Some((stage, packed)) = dependencies
            .iter()
            .filter_map(|d| match d.fact {
                RuntimeFact::HeapHalfTexelWrites { stage, packed } => Some((stage, packed)),
                _ => None,
            })
            .find(|&(_, packed)| residue.holds(packed))
        else {
            return;
        };
        if !self
            .shared()
            .announced
            .pending(Degradation::DeviceNanEncoding)
        {
            return;
        }
        let packed = match packed {
            true => " or a packed small float (`RG11B10Float`, `RGB9E5Float`) texture",
            false => "",
        };
        self.degrade(Degradation::DeviceNanEncoding, || {
            (
                format!("the {stage:?} function of a pipeline state"),
                format!(
                    "it writes texels through argument-buffer textures, and the device holds a \
                     half-float texture whose `half` bits its unsigned integer view cannot \
                     store (swizzled, or a wrapped image made without `MUTABLE_FORMAT`){packed}, \
                     whose NaNs the device stores as it converts them, not as Metal does"
                ),
            )
        });
    }

    /// Announce the specialized notes of `stages` (a variant's) that hold under `specialization`,
    /// the one a `VkPipeline` of the variant is made with, `at` its making or a bind
    /// ([`Occurrence`]): the notes the bound formats and samplers leave on an arm that is not
    /// Metal's (`StageModule::precision_notes_under`). The reason names each texture slot a holding
    /// clause tests (`StageModule::note_clauses_under`) and, where the command that makes the
    /// pipeline knows them (`formats`), the format it binds there, so the notice says which bound
    /// format is left.
    pub(super) fn degrade_specialized(
        &self,
        stages: &[StageModule],
        specialization: &SpecializationKey,
        formats: Option<BoundFormats<'_>>,
        at: Occurrence<'_>,
    ) {
        for stage in stages.iter().filter(|s| !s.specialized_notes.is_empty()) {
            for s in &stage.specialized_notes {
                // A clause of the bound formats alone is `degrade_bound_formats`', at each draw or
                // dispatch: its pipeline's key does not tell it.
                // Nor does it tell a clause of a value the function reads as data:
                // `degrade_bound_values`', at each draw or dispatch.
                let clauses: Vec<_> = stage
                    .note_clauses_under(s.note, specialization)
                    .into_iter()
                    .filter(|c| !c.tests_bound_formats() && !c.tests_bound_values())
                    .collect();
                self.degrade_clauses(
                    stage,
                    s.note,
                    &clauses,
                    formats,
                    at,
                    "under the formats and samplers its pipeline was made for",
                );
            }
        }
    }

    /// Whether a note of `stages` has a clause that tests a value its function reads as data
    /// (`SpecClause::tests_bound_values`): where none has,
    /// [`degrade_bound_values`](Self::degrade_bound_values) tests no clause, announces nothing
    /// and answers quiet, so a caller may leave it out.
    pub(crate) fn tests_bound_values(stages: &[StageModule]) -> bool {
        stages.iter().any(|stage| {
            stage
                .specialized_notes
                .iter()
                .any(|s| s.when.iter().any(|c| c.tests_bound_values()))
        })
    }

    /// Announce, at a draw or dispatch that binds a pipeline of `stages` made under
    /// `specialization`, each specialized note one of whose clauses tests a value its function
    /// reads as data (`SpecClause::tests_bound_values`: a bound texture's format class) and
    /// holds where the draw's plan pushes the values `bound` gives
    /// (`BindingPlan::specialized_value`; `StageModule::note_clauses_bound`). No pipeline's key
    /// carries those values, so no making can announce such a clause: one `VkPipeline` serves
    /// every bound value. The reason names each texture slot a holding clause tests and, where
    /// the command knows them (`formats`), the format it binds there. A kind already announced
    /// costs one atomic load.
    ///
    /// Whether it was quiet: no clause of a note it tested held. Which clauses hold is a function
    /// of `stages`, `specialization` and `bound` alone, and a note it skipped as announced stays
    /// skipped (a kind announced at every occurrence is never skipped), so another call with the
    /// same three after a quiet one announces nothing: a caller may leave it out (`render.rs`,
    /// `QuietBoundValues`).
    pub(crate) fn degrade_bound_values(
        &self,
        stages: &[StageModule],
        specialization: &SpecializationKey,
        bound: &dyn Fn(SpecKey) -> Option<u64>,
        formats: Option<BoundFormats<'_>>,
    ) -> bool {
        let mut quiet = true;
        for stage in stages {
            for s in &stage.specialized_notes {
                if !s.when.iter().any(|c| c.tests_bound_values())
                    || !self
                        .shared()
                        .announced
                        .pending(Degradation::from_precision_note(s.note))
                {
                    continue;
                }
                let clauses: Vec<_> = stage
                    .note_clauses_bound(s.note, specialization, bound)
                    .into_iter()
                    .filter(|c| c.tests_bound_values())
                    .collect();
                quiet &= clauses.is_empty();
                self.degrade_clauses(
                    stage,
                    s.note,
                    &clauses,
                    formats,
                    Occurrence::Drawn,
                    "under the formats it binds",
                );
            }
        }
        quiet
    }

    /// Announce `note` of `stage`, `at` ([`Occurrence`]), where `clauses`, the clauses of its
    /// specialized note that hold, are not empty: a filter note's slot bound to a two-plane YCbCr
    /// texture as its own kind, and the note's kind where a clause that holds tests another slot,
    /// or none, the reason naming each such slot (with its bound format, where `formats` knows it)
    /// after `under`.
    fn degrade_clauses(
        &self,
        stage: &StageModule,
        note: PrecisionNote,
        clauses: &[&metal2vulkan::pipeline::SpecClause],
        formats: Option<BoundFormats<'_>>,
        at: Occurrence<'_>,
        under: &str,
    ) {
        if clauses.is_empty() {
            return;
        }
        let all = tested_slots(clauses.iter().copied());
        // A filter note's slot bound to a two-plane YCbCr texture: its own kind, naming the slot
        // and the format; the note's own where a clause that holds tests another slot, or none.
        let filter = matches!(
            note,
            PrecisionNote::HalfSampleFilter | PrecisionNote::FloatSampleFilter
        );
        let ycbcr: Vec<(SlotRef, PixelFormat)> = all
            .iter()
            .filter(|_| filter)
            .filter_map(|&slot| {
                let format = formats.and_then(|f| f(slot))?;
                metal2vulkan::metal::format::ycbcr_format(format).map(|_| (slot, format))
            })
            .collect();
        let held = |slot: &SlotRef| ycbcr.iter().any(|(y, _)| y == slot);
        for &(slot, format) in &ycbcr {
            self.degrade_at(at, Degradation::YCbCrFilterNotExact, || {
                (
                    stage_object(stage.stage).to_owned(),
                    format!(
                        "a filtered sample of texture {} bound as {format:?} takes the device's \
                         filter, not Metal's exact filter of the format: {}",
                        slot.index,
                        self.note_reason(note)
                    ),
                )
            });
        }
        let covered = |c: &&metal2vulkan::pipeline::SpecClause| {
            let tested = tested_slots([*c]);
            !tested.is_empty() && tested.iter().all(held)
        };
        if clauses.iter().all(covered) {
            return;
        }
        self.degrade_at(at, Degradation::from_precision_note(note), || {
            let bound: Vec<String> = all
                .iter()
                .filter(|slot| !held(slot))
                .map(|&slot| match formats.map(|f| f(slot)) {
                    Some(Some(format)) => format!("texture {} bound as {format:?}", slot.index),
                    Some(None) => format!("texture {} bound to no texture", slot.index),
                    None => format!("texture {}", slot.index),
                })
                .collect();
            let named = match bound.is_empty() {
                true => String::new(),
                false => format!(" ({})", bound.join(", ")),
            };
            (
                stage_object(stage.stage).to_owned(),
                format!("{} {under}{named}", self.note_reason(note)),
            )
        });
    }

    /// Announce, at a draw or dispatch that binds a pipeline of `stages`, each specialized note
    /// one of whose clauses tests only the formats bound at texture slots
    /// (`SpecClause::tests_bound_formats`, `SpecKey::BoundTexelFormat`) and holds of the classes
    /// of the formats `formats` binds there (`metal::format::bound_texel_format`), a command
    /// whose formats are not known (`None`: an execution of commands the GPU writes) holding. A
    /// slot bound to nothing reads zero and holds no NaN, so a clause testing it does not hold.
    /// Such a clause is announced here alone: no pipeline's key tells it. A filter note's (a
    /// two-plane YCbCr texture that Metal's own sampler filters) is announced as
    /// `ycbcr-filter-not-exact`. A kind already announced costs one atomic load. Whether any
    /// note named a format (`VariantObjects::degrade_bound_formats` keeps the formats under which
    /// none did).
    pub(crate) fn degrade_bound_formats(
        &self,
        stages: &[StageModule],
        formats: Option<BoundFormats<'_>>,
    ) -> bool {
        let mut named = false;
        for stage in stages {
            for s in &stage.specialized_notes {
                // A filter note's clauses of the bound formats test a two-plane YCbCr format
                // alone (`metal2vulkan::lower`'s filter of such a texture by Metal's own sampler):
                // its own kind.
                let filter = matches!(
                    s.note,
                    PrecisionNote::HalfSampleFilter | PrecisionNote::FloatSampleFilter
                );
                let kind = match filter {
                    true => Degradation::YCbCrFilterNotExact,
                    false => Degradation::from_precision_note(s.note),
                };
                if !s.when.iter().any(|c| c.tests_bound_formats())
                    || !self.shared().announced.pending(kind)
                {
                    continue;
                }
                let class = |slot: SlotRef| {
                    formats
                        .and_then(|f| f(slot))
                        .map(metal2vulkan::metal::format::bound_texel_format)
                };
                // A slot bound to nothing holds no format, so no NaN: a clause testing one fails.
                let unbound = |slot: SlotRef| formats.is_some_and(|f| f(slot).is_none());
                let held: Vec<_> = stage
                    .bound_format_clauses(s.note, &class)
                    .into_iter()
                    .filter(|c| {
                        !c.all.iter().any(
                            |t| matches!(t.key, SpecKey::BoundTexelFormat(slot) if unbound(slot)),
                        )
                    })
                    .collect();
                if held.is_empty() {
                    continue;
                }
                let slots = tested_slots(held.iter().copied());
                named = true;
                self.degrade(kind, || {
                    let bound: Vec<String> = slots
                        .iter()
                        .map(|&slot| match formats.and_then(|f| f(slot)) {
                            Some(format) => format!("texture {} bound as {format:?}", slot.index),
                            None => format!(
                                "texture {} of a format the GPU's commands bind",
                                slot.index
                            ),
                        })
                        .collect();
                    let bound = bound.join(", ");
                    let reason = self.note_reason(s.note);
                    let message = match s.note {
                        _ if filter => format!(
                            "a filtered sample of {bound} takes the device's filter of its \
                             texels, not Metal's exact filter of the format: {reason}"
                        ),
                        PrecisionNote::DeviceNanEncoding => {
                            format!("{reason} where it binds a texture whose format may hold one ({bound})")
                        }
                        _ => format!("{reason} where it binds {bound}"),
                    };
                    (stage_object(stage.stage).to_owned(), message)
                });
            }
        }
        named
    }

    /// Announce `cross-class-texture-neutral` at a draw or dispatch whose plan binds a slot's
    /// declared neutral where a texture of another numeric class is bound
    /// (`PlanDiagnostic::CrossClassNeutral` among `diagnostics`), naming the first such slot
    /// and the format `formats` binds there. A kind already announced costs one atomic load.
    pub(crate) fn degrade_cross_class(
        &self,
        diagnostics: &[PlanDiagnostic],
        formats: BoundFormats<'_>,
    ) {
        let Some(first) = diagnostics.iter().find(|d| {
            matches!(
                d,
                PlanDiagnostic::CrossClassNeutral { .. }
                    | PlanDiagnostic::CrossClassHeapNeutral { .. }
            )
        }) else {
            return;
        };
        self.degrade(Degradation::CrossClassTextureNeutral, || match *first {
            PlanDiagnostic::CrossClassHeapNeutral { stage, element, .. } => (
                format!(
                    "the {stage:?} function's argument-buffer texture (heap element {element})"
                ),
                "a texture whose numeric class is not the one the function declares, which the \
                 function samples, gathers, writes or reads where its access has no exact answer \
                 for it"
                    .to_owned(),
            ),
            PlanDiagnostic::CrossClassNeutral { slot, .. } => {
                let bound = match formats(slot) {
                    Some(format) => format!("{format:?}"),
                    None => "a format the commands bind".to_owned(),
                };
                (
                    format!("the {:?} function's texture {}", slot.stage, slot.index),
                    format!(
                        "bound to a texture of {bound}, whose numeric class is not the declared \
                         one, where the slot's accesses have no exact answer for it"
                    ),
                )
            }
            _ => (String::new(), String::new()),
        });
    }

    /// Announce `PrecisionNote::ArgumentPixelSamplerScaling` at a draw or dispatch of a variant
    /// of `stages` whose commands list `samplers` (the device's live sampler states its sampler
    /// table lists): where a stage records the note and a listed state has pixel coordinates and
    /// a linear minification or magnification filter or a comparison, the states whose access
    /// through an argument buffer may hand the device a scaled coordinate. A listed state is any an
    /// argument buffer the commands read may name by its ID; which ones a program's argument
    /// buffers hold, the CPU or the GPU writing them, the runtime does not see. Nothing else
    /// announces the note: a pipeline that records it but never runs while such a state lives
    /// announces nothing.
    pub(crate) fn degrade_argument_pixel_samplers(
        &self,
        stages: &[StageModule],
        samplers: &[crate::SamplerState],
    ) {
        const NOTE: PrecisionNote = PrecisionNote::ArgumentPixelSamplerScaling;
        let Some(stage) = self.argument_pixel_samplers_noted(stages) else {
            return;
        };
        let Some(state) = samplers.iter().find(|s| s.scales_in_an_argument_buffer()) else {
            return;
        };
        self.degrade(Degradation::ArgumentPixelSamplerScaling, || {
            (
                stage_object(stage.stage).to_owned(),
                format!(
                    "{}: it samples through an argument buffer's sampler while sampler state {} \
                     of pixel coordinates with a linear filter or a comparison lives, which a \
                     program's argument buffer may name",
                    self.note_reason(NOTE),
                    state.gpu_resource_id().0
                ),
            )
        });
    }

    /// The stage of `stages` whose note [`degrade_argument_pixel_samplers`] may announce: one
    /// that records `PrecisionNote::ArgumentPixelSamplerScaling`, where the device has not
    /// announced it yet; `None` where it would announce nothing, whatever the states.
    ///
    /// [`degrade_argument_pixel_samplers`]: Self::degrade_argument_pixel_samplers
    pub(crate) fn argument_pixel_samplers_noted<'a>(
        &self,
        stages: &'a [StageModule],
    ) -> Option<&'a StageModule> {
        const NOTE: PrecisionNote = PrecisionNote::ArgumentPixelSamplerScaling;
        let stage = stages.iter().find(|s| s.precision_notes.contains(&NOTE))?;
        self.shared()
            .announced
            .pending(Degradation::ArgumentPixelSamplerScaling)
            .then_some(stage)
    }

    /// Why `stage`, which records `note`, degrades: [`note_reason`](Self::note_reason)'s, and for
    /// `PrecisionNote::FunctionStorageAssumed` the private memory a thread of the stage keeps and
    /// the Function storage the device's profile holds, measured or the floor of a device not
    /// measured, in bytes.
    fn stage_note_reason(&self, note: PrecisionNote, stage: &StageModule) -> String {
        let recorded = self.note_reason(note);
        if note != PrecisionNote::FunctionStorageAssumed {
            return recorded;
        }
        let held = self.compile_device().profile().function_storage();
        function_storage_reason(&recorded, stage.private_bytes, held)
    }

    /// Why a stage that records `note` degrades: the note, and for
    /// `PrecisionNote::BeyondMeasuredDeviation` each deviation the device's profile takes past
    /// its measurement, with the last release measured.
    fn note_reason(&self, note: PrecisionNote) -> String {
        let recorded = format!("its translation records `PrecisionNote::{note:?}`");
        if note == PrecisionNote::FragmentSimdgroupGrouping {
            return format!(
                "{recorded}: {}'s fragment subgroups are not Apple's simdgroups, so its simdgroup \
                 lanes and operations group other pixels than an Apple GPU's",
                self.name()
            );
        }
        if note != PrecisionNote::BeyondMeasuredDeviation {
            return recorded;
        }
        let profile = self.compile_device().profile();
        let named: Vec<String> = profile
            .driver_deviations()
            .iter()
            .map(|(deviation, last)| {
                format!("{deviation:?} (measured through driverVersion {last:#x})")
            })
            .collect();
        format!(
            "{recorded}: the driver's deviations measured on earlier releases: {}",
            named.join(", ")
        )
    }

    /// Announce what the pass `layout` (an ended encoder's, recorded) degrades: separate depth
    /// and stencil textures, a depth/stencil resolve one aspect at a time, a resolve into a
    /// scratch image, a device-rounded color resolve, and a `DontCare` store, which a pass
    /// recorded `whole` may store as Metal's own.
    pub(super) fn degrade_pass(&self, layout: &PassLayout, whole: Option<WholePass>) {
        let depth_stencil = layout.depth_stencil.as_ref();
        if let Some((ds, separate)) =
            depth_stencil.and_then(|ds| ds.separate.as_ref().map(|separate| (ds, separate)))
        {
            self.degrade(Degradation::SeparateDepthStencil, || {
                match &separate.depth {
                    Some(depth) => (
                        format!(
                            "a render pass of a {:?} depth texture and a {:?} stencil texture",
                            depth.texture.pixel_format(),
                            separate.stencil.texture.pixel_format()
                        ),
                        format!(
                        "a Vulkan subpass has one depth/stencil attachment, so the pass renders \
                         into a {:?} scratch image",
                        ds.target.format
                    ),
                    ),
                    None => (
                        format!(
                        "a render pass of a {:?} stencil texture alone at custom sample positions",
                        separate.stencil.texture.pixel_format()
                    ),
                        format!(
                            "Vulkan lets no image without a depth aspect take custom sample \
                         positions, so the pass renders into a {:?} scratch image",
                            ds.target.format
                        ),
                    ),
                }
            });
        }
        if let Some((ds, resolve)) = depth_stencil
            .and_then(|ds| ds.resolve.as_ref().map(|resolve| (ds, resolve)))
            .filter(|(_, resolve)| resolve.steps.iter().any(|step| step.one_aspect))
        {
            self.degrade(Degradation::OneAspectResolve, || {
                let reason = if resolve.split {
                    format!(
                        "its depth ({}) and its stencil ({}) resolve into different textures, \
                         each through a scratch image of its own",
                        resolve_mode(resolve.depth_mode),
                        resolve_mode(resolve.stencil_mode)
                    )
                } else {
                    format!(
                        "the device cannot resolve its depth ({}) and its stencil ({}) in one \
                         subpass (`independentResolve`, `independentResolveNone`)",
                        resolve_mode(resolve.depth_mode),
                        resolve_mode(resolve.stencil_mode)
                    )
                };
                (
                    format!(
                        "a render pass resolving its {:?} depth/stencil attachment",
                        ds.target.texture.pixel_format()
                    ),
                    reason,
                )
            });
        }
        let colors = layout.colors.iter().flatten();
        let into_scratch = colors
            .clone()
            .filter_map(|color| color.resolve.as_ref())
            .chain(
                depth_stencil_steps(layout)
                    .filter(|step| !step.one_aspect)
                    .map(|step| &step.resolve),
            )
            .chain(
                depth_stencil
                    .and_then(|ds| ds.resolve.as_ref())
                    .and_then(|resolve| resolve.drawn_into.as_ref()),
            )
            .find_map(|resolve| resolve.copy_to.as_ref().map(|to| (to, &resolve.attachment)));
        if let Some((texture, scratch)) = into_scratch {
            self.degrade(Degradation::ResolveScratch, || {
                let depth_stencil = !super::texture::image_aspects(scratch.format)
                    .contains(ImageAspectFlags::COLOR);
                let reason = if texture.layers < scratch.layers && depth_stencil {
                    format!(
                        "a layered pass of {} layers resolves its depth and stencil from the \
                         first alone, as Metal does",
                        scratch.layers
                    )
                } else if texture.layers < scratch.layers {
                    format!(
                        "a layered pass of {} layers resolves into {} slices of it, the layers that fit",
                        scratch.layers, texture.layers
                    )
                } else {
                    "its image cannot be an attachment: the texture's usage lacks `RenderTarget`"
                        .to_owned()
                };
                (
                    format!(
                        "a resolve into a {:?} texture",
                        texture.texture.pixel_format()
                    ),
                    reason,
                )
            });
        }
        let exact = colors
            .clone()
            .find(|color| color.resolve.as_ref().is_some_and(|r| r.shader.is_some()));
        if let Some(color) = exact {
            self.degrade(Degradation::ShaderResolve, || {
                // An attachment whose image takes no input attachment usage is resolved from a
                // copy of its samples (`color_resolve.rs`).
                let copied = color.planes.is_none()
                    && matches!(render_target::takes_input(&color.target), Ok(false));
                (
                    format!(
                        "a resolve of a {:?} color attachment of {} samples",
                        color.target.texture.pixel_format(),
                        color.target.samples
                    ),
                    format!(
                        "{}{}",
                        match color_resolve::extended(color.target.texture.pixel_format()) {
                            Some(_) => format!(
                                "no device's resolve averages the codes {:?} holds",
                                color.target.format
                            ),
                            None => format!(
                                "the device's `AVERAGE` resolve of {:?} at {} samples is not \
                                 measured to be Metal's",
                                color.target.format, color.target.samples
                            ),
                        },
                        match copied {
                            true => {
                                "; the attachment's image takes no input attachment usage, so its \
                                 samples are first copied into a scratch image that has it"
                            }
                            false => "",
                        }
                    ),
                )
            });
        }
        // An extended-range resolve whose rule the codes cannot give (`color_resolve.rs`).
        let near = colors.clone().find(|color| {
            color
                .resolve
                .as_ref()
                .and_then(|r| r.shader)
                .is_some_and(|rules| !rules.exact())
        });
        if let Some(color) = near {
            self.degrade(Degradation::ExtendedRangeResolve, || {
                (
                    format!(
                        "a resolve of a {:?} color attachment of {} samples",
                        color.target.texture.pixel_format(),
                        color.target.samples
                    ),
                    "Metal averages the samples' values, finer than the codes the attachment \
                     holds; the resolve averages the codes"
                        .to_owned(),
                )
            });
        }
        let unmeasured = colors.clone().find(|color| {
            color.resolve.is_some() && color.target.samples > color_resolve::METAL_SAMPLES
        });
        if let Some(color) = unmeasured {
            self.degrade(Degradation::UnmeasuredResolve, || {
                (
                    format!(
                        "a resolve of a {:?} color attachment of {} samples",
                        color.target.texture.pixel_format(),
                        color.target.samples
                    ),
                    format!(
                        "Apple's GPUs make no texture of more than {} samples",
                        color_resolve::METAL_SAMPLES
                    ),
                )
            });
        }
        // A depth the device has no mode for, drawn after the pass (`depth_stencil_resolve.rs`).
        if let Some((ds, ResolveMode::Picked(pick))) = layout
            .depth_stencil
            .as_ref()
            .and_then(|ds| ds.resolve.as_ref().map(|r| (ds, r.depth_mode)))
        {
            self.degrade(Degradation::ShaderResolve, || {
                let mode = format!("{pick:?}").to_uppercase();
                (
                    format!(
                        "a depth resolve of a {:?} attachment of {} samples by `{pick:?}`",
                        ds.target.texture.pixel_format(),
                        ds.target.samples
                    ),
                    format!(
                        "the device has no `{mode}` depth resolve mode (`supportedDepthResolveModes`)"
                    ),
                )
            });
        }
        // A stencil picked by depth, drawn after the pass (`depth_stencil_resolve.rs`).
        if let Some((ds, ResolveMode::Picked(pick))) = layout
            .depth_stencil
            .as_ref()
            .and_then(|ds| ds.resolve.as_ref().map(|r| (ds, r.stencil_mode)))
        {
            let what = || {
                format!(
                    "a stencil resolve of a {:?} attachment of {} samples by \
                     `DepthResolvedSample` after `{pick:?}`",
                    ds.target.texture.pixel_format(),
                    ds.target.samples
                )
            };
            self.degrade(Degradation::ShaderResolve, || {
                (
                    what(),
                    "no Vulkan resolve mode takes the stencil of the sample the depth filter took"
                        .to_owned(),
                )
            });
            if ds.target.samples > color_resolve::METAL_SAMPLES {
                self.degrade(Degradation::UnmeasuredResolve, || {
                    (
                        what(),
                        format!(
                            "Apple's GPUs make no texture of more than {} samples",
                            color_resolve::METAL_SAMPLES
                        ),
                    )
                });
            }
        }
        // A resolve the device makes where its average is not measured to be Metal's.
        // Whatever the format: where the device's average is Metal's (MoltenVK's is Metal's own
        // resolve), a format without a closed rule resolves as Metal's too.
        let rounded = colors.clone().find(|color| {
            color.resolve.as_ref().is_some_and(|r| r.shader.is_none())
                && !color_resolve::average_is_metals(self, color.target.samples)
        });
        if let Some(color) = rounded {
            // Under the device's resolve, a format of Metal's rule at its count resolves by that
            // rule under the exact one.
            let switch = self.compile_device().policy().resolve == ResolvePolicy::Device
                && color_resolve::from_planes(color.target.format, color.target.samples);
            self.degrade(Degradation::DeviceRoundedResolve, || {
                (
                    format!(
                        "a resolve of a {:?} color attachment of {} samples",
                        color.target.texture.pixel_format(),
                        color.target.samples
                    ),
                    format!(
                        "Vulkan's `AVERAGE` resolve of {:?} leaves the average's rounding, and \
                         what a NaN or infinite sample makes of it, to the device{}",
                        color.target.format,
                        match switch {
                            true => {
                                "; `ResolvePolicy::Exact` resolves it by Metal's rule after the \
                                 pass"
                            }
                            false => "",
                        }
                    ),
                )
            });
        }
        // A separate texture stored `DontCare` keeps its texels (`depth_stencil_scratch.rs`), and
        // a memoryless one keeps none in Metal either.
        let dont_care = |store| {
            matches!(
                store,
                StoreAction::DontCare | StoreAction::MultisampleResolve
            )
        };
        // Whether the attachment `pick` names, loaded by `load` and stored by `store`, is stored
        // as Metal's (`HandedToMetal`).
        let handed = HandedToMetal::of(self, layout, whole);
        let metals = |pick: &dyn Fn(&HandedDontCare) -> bool, load, store| {
            handed
                .as_ref()
                .is_some_and(|handed| handed.stores(pick, load, store))
        };
        // An attachment cleared and resolved into another pixel format by a pass without a draw
        // call renders into a backing of its own, which copies nothing back: its texture keeps
        // its texels, as Metal's keeps them there (`render.rs`).
        let kept = |color: &ColorTarget| {
            color.clears_and_resolves_across_formats()
                && color.stand_in.as_ref().is_some_and(|s| !s.copies())
        };
        // A texture over a buffer rendered through an image the pass copies nothing back from
        // where it does not store it (a shadow, `render/shadows.rs`; a stand-in, which copies
        // back only what Metal stores) keeps the buffer's bytes, Metal's answer (`kept_bytes.rs`).
        let keeps_bytes = |target: &Target, stand_in: Option<&StandIn>| {
            let own = stand_in.map_or(target, |s| &s.own);
            own.texture.buffer_texels().is_some()
                && (stand_in.is_some_and(StandIn::copies) || own.texture.shadow().is_some())
        };
        let color = layout.colors.iter().enumerate().find_map(|(index, color)| {
            color
                .as_ref()
                .filter(|color| !color.memoryless && dont_care(color.store) && !kept(color))
                .filter(|color| !keeps_bytes(&color.target, color.stand_in.as_ref()))
                .filter(|color| !handed.as_ref().is_some_and(|h| h.color(index, color)))
                .map(|color| {
                    (
                        StoredAttachment::Color(index),
                        color.target.texture.pixel_format(),
                        color.store,
                    )
                })
        });
        let patterned = depth_stencil.filter(|ds| {
            ds.separate.is_none()
                && ds.depth.zip(ds.stencil).is_some_and(|(depth, stencil)| {
                    super::resolve::layout_patterned(depth.store, stencil.store)
                })
        });
        // A pass's hidden depth and stencil are the runtime's own image, of no texture of the
        // program's (`DepthStencilTarget::hidden`).
        let aspect = depth_stencil
            .filter(|ds| ds.separate.is_none() && !ds.hidden)
            .filter(|ds| !keeps_bytes(&ds.target, ds.stand_in.as_ref()))
            .and_then(|ds| {
                let stored_stencil = patterned.is_some();
                let picks: [(&str, _, fn(&HandedDontCare) -> bool); 2] = [
                    ("depth", ds.depth, |h| h.depth),
                    ("stencil", ds.stencil.filter(|_| !stored_stencil), |h| {
                        h.stencil
                    }),
                ];
                picks.into_iter().find_map(|(name, aspect, pick)| {
                    aspect
                        .filter(|aspect| !aspect.memoryless && dont_care(aspect.store))
                        .filter(|aspect| !metals(&pick, aspect.load, aspect.store))
                        .map(|aspect| {
                            (
                                StoredAttachment::Aspect(name),
                                ds.target.texture.pixel_format(),
                                aspect.store,
                            )
                        })
                })
            });
        if let Some((which, format, store)) = color.or(aspect) {
            let how = match render_target::DontCare::of(self) {
                render_target::DontCare::Discarded => "which is Vulkan's `DONT_CARE`",
                render_target::DontCare::Stored => {
                    "which the runtime stores (Vulkan's `STORE`), as the device's `DONT_CARE` \
                     is not measured to keep one of Metal's answers"
                }
            };
            self.degrade(Degradation::DontCareStore, || {
                let which = match which {
                    StoredAttachment::Color(index) => format!("color attachment {index}"),
                    StoredAttachment::Aspect(name) => format!("the {name} attachment"),
                };
                (
                    format!("{which} of a render pass, a {format:?} texture"),
                    format!("it is stored with `{store:?}`, {how}"),
                )
            });
        }
        if let Some(ds) = patterned {
            self.degrade(Degradation::MixedDepthStencilResolve, || {
                (
                    format!(
                        "a render pass's {:?} depth/stencil attachment",
                        ds.target.texture.pixel_format()
                    ),
                    "its depth is stored and resolved (`StoreAndMultisampleResolve`) while its \
                     stencil is resolved alone (`MultisampleResolve`), which Metal's validation \
                     stops"
                        .to_string(),
                )
            });
        }
    }
}

/// The attachment a `DontCare` store's notice names (`Device::degrade_pass`), named only where
/// the notice is made: a color attachment by index, or the depth or the stencil aspect.
#[derive(Clone, Copy)]
enum StoredAttachment {
    Color(usize),
    Aspect(&'static str),
}

/// A pass recorded as one render pass instance in which the runtime recorded nothing a driver
/// restarts its Metal pass for (`Device::degrade_pass`, `DriverFact::DontCareStoreIsMetals`):
/// no execution of render commands, no barrier between draws, no tessellated draw, no views,
/// and the device's own sample positions.
#[derive(Clone, Copy)]
pub(super) struct WholePass<'a> {
    /// The instance's shape.
    pub(super) shape: &'a PassShape,
    /// The pass draws (the render encoder's `drew`).
    pub(super) drew: bool,
}

/// What a pass recorded whole ([`WholePass`]) hands a driver that stores `DONT_CARE` as Metal's
/// own `DontCare` on the same GPU (`DriverFact::DontCareStoreIsMetals`): the attachments it
/// stores so, as Metal's (`Device::degrade_pass`; a texture over a buffer keeps its bytes there,
/// `kept_bytes.rs`). But an attachment the pass clears, draws nothing into, and resolves alone:
/// Metal leaves the clear in it where it compresses the texture and keeps its texels where it
/// does not, and the driver's image of the texture may be compressed otherwise.
pub(super) struct HandedToMetal {
    /// The attachments the pass hands the driver as `DONT_CARE`.
    handed: HandedDontCare,
    /// The pass draws.
    drew: bool,
}

impl HandedToMetal {
    /// What `layout`'s pass, recorded `whole`, hands `device`'s driver as Metal's; `None` for a
    /// pass not recorded whole or a driver without the fact.
    pub(super) fn of(
        device: &Device,
        layout: &PassLayout,
        whole: Option<WholePass>,
    ) -> Option<Self> {
        let facts = &device.properties().driver_facts;
        whole
            .filter(|_| facts.contains(&DriverFact::DontCareStoreIsMetals))
            .map(|whole| HandedToMetal {
                handed: render_target::handed_dont_care(device, layout, whole.shape),
                drew: whole.drew,
            })
    }

    /// Whether the attachment `pick` names, loaded by `load` and stored by `store`, is stored as
    /// Metal's.
    fn stores(
        &self,
        pick: &dyn Fn(&HandedDontCare) -> bool,
        load: LoadAction,
        store: StoreAction,
    ) -> bool {
        let resolved_clear = load == LoadAction::Clear && store == StoreAction::MultisampleResolve;
        pick(&self.handed) && (self.drew || !resolved_clear)
    }

    /// Whether color attachment `index`, `color`, is stored as Metal's.
    pub(super) fn color(&self, index: usize, color: &ColorTarget) -> bool {
        let pick = |h: &HandedDontCare| index < 32 && h.colors & 1 << index != 0;
        self.stores(&pick, color.load, color.store)
    }
}

/// A depth or stencil resolve mode as a notice names it.
fn resolve_mode(mode: ResolveMode) -> String {
    if mode == ResolveMode::None {
        "not resolved".to_owned()
    } else {
        format!("by {mode:?}")
    }
}

/// The object a stage's notice names: the stage's function and its pipeline state's kind.
fn stage_object(stage: ShaderStage) -> &'static str {
    if stage == ShaderStage::COMPUTE {
        "the kernel of a compute pipeline state"
    } else if stage == ShaderStage::VERTEX {
        "the vertex function of a render pipeline state"
    } else if stage == ShaderStage::FRAGMENT {
        "the fragment function of a render pipeline state"
    } else if stage == ShaderStage::TESSELLATION_CONTROL
        || stage == ShaderStage::TESSELLATION_EVALUATION
    {
        "a tessellation stage of a render pipeline state"
    } else if stage == ShaderStage::TASK_EXT {
        "the object function of a mesh render pipeline state"
    } else if stage == ShaderStage::MESH_EXT {
        "the mesh function of a mesh render pipeline state"
    } else {
        "a stage of a pipeline state"
    }
}

/// What split an ended encoder's pass between its render pass instances, beside its
/// attachments' roles (`Device::degrade_instances`).
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct SplitCauses {
    /// Some of its draws test the stencil in the shader (`shadow_stencil.rs`).
    pub(super) shadow: bool,
    /// Some of its draws test the depth in the shader (`depth_stand_in.rs`).
    pub(super) shader_tests: bool,
    /// The tile dispatches it runs between its instances (`render/tile_dispatch.rs`).
    pub(super) dispatches: usize,
    /// The splits the program's memory barriers made (`fence.rs`), and its indirect render
    /// commands' barriers (`render/executions.rs`).
    pub(super) barriers: usize,
    /// The splits between draws that reach the pass's tile threadgroup memory
    /// (`render/tile_memory.rs`).
    pub(super) tile_memory: usize,
    /// Some of its instances render some of its layers.
    pub(super) viewed: bool,
}

impl Device {
    /// Announce what the render pass instances of an ended encoder's pass degrade, `layout` its
    /// layout with its attachments' stand-ins and `shapes` its instances' shapes in order: a split
    /// (more than one instance, for the first of `causes` that holds), a memoryless attachment's
    /// backing image, a fetched color render target's scratch image, and a depth/stencil render
    /// target's made for custom sample positions (`render.rs`, `attachment_stand_in.rs`). A split
    /// the interlock alone makes on a device that runs it no slower than whole is not announced,
    /// nor the backing it gives a memoryless attachment.
    pub(super) fn degrade_instances<S: std::borrow::Borrow<PassShape>>(
        &self,
        layout: &PassLayout,
        shapes: &[S],
        causes: SplitCauses,
    ) {
        // An instance's shape, whether the caller holds the shapes or references to them.
        fn of<S: std::borrow::Borrow<PassShape>>(shape: &S) -> &PassShape {
            shape.borrow()
        }
        let SplitCauses {
            shadow,
            shader_tests,
            dispatches,
            barriers,
            tile_memory,
            viewed,
        } = causes;
        let instances = shapes.len();
        let colors = layout.colors.iter().flatten().count();
        let pass =
            || format!("a render pass of {colors} color attachments in {instances} instances");
        // The first color attachment whose role changes between two instances: left out of the
        // draws' subpass (the interlock), or read as an input attachment.
        let changed = shapes.windows(2).find_map(|pair| {
            let pair = [of(&pair[0]), of(&pair[1])];
            let role = |shape: &PassShape, index: &u32| {
                (
                    shape.storage.contains(index),
                    shape.no_inputs.contains(index),
                )
            };
            let indices = pair
                .into_iter()
                .flat_map(|shape| shape.storage.iter().chain(&shape.no_inputs));
            indices.copied().find_map(|index| {
                let (before, after) = (role(pair[0], &index), role(pair[1], &index));
                (before != after).then_some((index, before.0 != after.0))
            })
        });
        // The pass's hidden depth and stencil, arriving after its first instance.
        let hidden = shapes.windows(2).any(|pair| {
            of(&pair[0]).depth_stencil.is_none() && of(&pair[1]).depth_stencil.is_some()
        });
        // On a device whose driver ran a pass the interlock splits no slower split than whole
        // (`DriverFact::InterlockedPassSplitNoSlower`), a split the interlock alone makes (each
        // instance's shape differs from the one before in the colors its subpass leaves out
        // alone, and no other cause holds) is the device's faster form: announced neither as a
        // split nor by the memoryless backing it gives an attachment.
        let split_form = self
            .properties()
            .driver_facts
            .contains(&DriverFact::InterlockedPassSplitNoSlower);
        let storage_alone = |pair: &[S]| {
            let (a, b) = (of(&pair[0]), of(&pair[1]));
            a.storage != b.storage
                && PassShape {
                    storage: b.storage.clone(),
                    ..a.clone()
                } == *b
        };
        let chosen = split_form
            && instances > 1
            && shapes.windows(2).all(storage_alone)
            && !shadow
            && !shader_tests
            && dispatches == 0
            && barriers == 0
            && tile_memory == 0;
        if instances > 1 && !chosen {
            let color = changed.and_then(|(index, interlock)| {
                let color = layout.colors.get(index as usize)?.as_ref()?;
                Some((
                    index,
                    interlock,
                    color.stand_in.as_ref().map_or(&color.target, |s| &s.own),
                ))
            });
            self.degrade(Degradation::RenderPassSplit, || {
                let reason = match color {
                    Some((index, true, target)) => format!(
                        "some of its draws read color attachment {index} ({}) through the pixel \
                         interlock, which keeps it out of their subpass, and other draws, or its \
                         resolve, need it in theirs, so it runs as {instances} Vulkan render \
                         pass instances",
                        described(target)
                    ),
                    Some((index, false, target)) => format!(
                        "a draw reads color attachment {index} ({}) as an input attachment \
                         after others drew it, and its image takes no input attachment usage: \
                         a stand-in that has it makes it an input attachment of the later \
                         draws' subpass alone, so it runs as {instances} Vulkan render pass \
                         instances",
                        described(target)
                    ),
                    None if hidden => format!(
                        "a draw after others uses depth or the stencil it has no attachment for, \
                         which Metal tests against a hidden depth and stencil: they are an \
                         attachment of the draws' subpass from that draw on, so it runs as \
                         {instances} Vulkan render pass instances"
                    ),
                    None if shadow => format!(
                        "some of its draws test the stencil in the fragment shader, each run of \
                         them between the stencil's copies to and from its shadow, so it runs \
                         as {instances} Vulkan render pass instances"
                    ),
                    None if shader_tests => format!(
                        "some of its draws test the depth in the fragment shader, each run of \
                         them between the depth's copies to and from its stand-in, so it runs as \
                         {instances} Vulkan render pass instances"
                    ),
                    None if dispatches > 0 => format!(
                        "it runs {dispatches} tile dispatches, each outside a render pass \
                         instance after the one before it, so it runs as {instances} Vulkan \
                         render pass instances"
                    ),
                    None if barriers > 0 => format!(
                        "memory barriers between its draws end {barriers} of its Vulkan \
                         render pass instances (whose own barriers order framebuffer-space \
                         stages alone), so it runs as {instances} Vulkan render pass instances"
                    ),
                    None if tile_memory > 0 => format!(
                        "draws that read and write its tile threadgroup memory see each other's \
                         writes in order, which Vulkan gives one draw of another's storage writes \
                         across render pass instances alone, ending {tile_memory} of them, so it \
                         runs as {instances} Vulkan render pass instances"
                    ),
                    None if viewed => format!(
                        "some of its draws render some of its layers (vertex amplification as \
                         Vulkan multiview, or a view mapping that moves a draw to one layer), \
                         each run of them in a render pass instance over those layers, so it \
                         runs as {instances} Vulkan render pass instances"
                    ),
                    None => format!(
                        "its draws keep different attachments out of their subpass, so it runs \
                         as {instances} Vulkan render pass instances"
                    ),
                };
                (pass(), reason)
            });
        }
        let fetched = |index: usize| {
            shapes
                .iter()
                .any(|shape| of(shape).storage.iter().any(|&i| i as usize == index))
        };
        let colors = layout
            .colors
            .iter()
            .enumerate()
            .filter_map(|(index, color)| {
                let color = color.as_ref()?;
                let stand_in = color.stand_in.as_ref()?;
                Some((Some(index), &color.target, stand_in, color.memoryless))
            });
        // Whether the attachment is memoryless: a backing of one that is not is the kept backing,
        // whose pass keeps the texture's texels as Metal's and announces nothing, never
        // `memoryless-backed`'s "a memoryless texture".
        let depth_stencil = layout.depth_stencil.iter().filter_map(|ds| {
            let memoryless = [ds.depth, ds.stencil]
                .iter()
                .flatten()
                .any(|aspect| aspect.memoryless);
            Some((None, &ds.target, ds.stand_in.as_ref()?, memoryless))
        });
        for (index, target, stand_in, memoryless) in colors.chain(depth_stencil) {
            let own = &stand_in.own;
            // What the pass renders into, in words: made only where a notice is announced.
            let image = || match target.texture.placed_bytes() {
                Some(bytes) => format!("a scratch image of {bytes} bytes"),
                None => "a scratch image".to_owned(),
            };
            let planes = index
                .and_then(|index| layout.colors.get(index)?.as_ref())
                .is_some_and(|color| color.planes.is_some());
            // A depth/stencil stand-in that copies is the one made for custom sample positions:
            // for an image not made for them, or for one that was on a driver that keeps the
            // depth per sample only through a copy.
            // By the stand-in's cause: a fetched texture's scratch alone is
            // `fetch-through-scratch`, naming the draw that fetched it; a texture without render
            // target usage is `undeclared-usage-shadow`; a bound linear attachment's kept image
            // was announced by `linear_attachment.rs`. Before, every stand-in that copies was
            // announced as a fetch's, a linear texture rendered through copies among them (read
            // "through the pixel interlock" where no draw read it).
            if stand_in.cause == Cause::Positions {
                self.degrade(Degradation::PositionsThroughScratch, || {
                    let image = image();
                    let made = own.texture.image_plan().is_ok_and(|plan| {
                        plan.flags.contains(
                            metal2vulkan::vk::ImageCreateFlags::SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_EXT,
                        )
                    });
                    let why = match own.texture.callers_image() {
                        true => " (the caller made it without it)",
                        false => "",
                    };
                    let reason = match made {
                        true => format!(
                            "{} renders at custom sample positions, and this device's driver \
                             answers a later read of a depth rendered there with the depth at the \
                             standard positions, and a later pass at other positions with the \
                             depth at its own (`DriverFact::PositionedDepthReadAtStandardPositions`), \
                             keeping each sample's depth only through a copy, so the pass renders \
                             into {image}",
                            pass()
                        ),
                        false => format!(
                            "{} renders at custom sample positions, at which Vulkan rasterizes a \
                             depth/stencil attachment only where its image was made with \
                             `VK_IMAGE_CREATE_SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_BIT_EXT`, and \
                             this one's was not{why}, so the pass renders into {image}",
                            pass()
                        ),
                    };
                    (
                        format!("a {} depth/stencil render target", described(own)),
                        reason,
                    )
                });
            } else if stand_in.cause == Cause::Unattachable {
                self.degrade(Degradation::UndeclaredUsageShadow, || {
                    let image = image();
                    let attachment = match index {
                        Some(index) => format!("color attachment {index}"),
                        None => "depth/stencil attachment".to_owned(),
                    };
                    (
                        format!("a {} texture", described(own)),
                        format!(
                            "{} renders it as its {attachment}, which its usage does not declare \
                             (no `RenderTarget`), so the pass renders into {image}",
                            pass()
                        ),
                    )
                });
            } else if let Cause::Fetch { by } = &stand_in.cause {
                self.degrade(Degradation::FetchThroughScratch, || {
                    let image = image();
                    let (form, missing) = if layout.fetch_inputs {
                        ("as an input attachment", "input attachment")
                    } else if planes {
                        (
                            "through the pixel interlock as sample planes",
                            "sampled usage with a view of its texel size's unsigned-integer \
                             format, which the copies to and from the planes take,",
                        )
                    } else {
                        ("through the pixel interlock", "`STORAGE`")
                    };
                    // The runtime's own placed images take input attachment usage wherever the
                    // fetch reads input attachments (`image_plan.rs`).
                    let why = if own.texture.callers_image() {
                        "the caller made it without that usage"
                    } else if own.texture.placed_bytes().is_none() {
                        "a texture over a buffer's bytes is a linear image without it"
                    } else {
                        "the device lays its image out otherwise with it"
                    };
                    (
                        format!("a {} render target", described(own)),
                        format!(
                            "{by} in {} reads it {form}, and its image has no {missing} usage \
                             ({why}), so the pass renders into {image}",
                            pass()
                        ),
                    )
                });
            } else if matches!(
                stand_in.cause,
                Cause::Linear | Cause::KeptBytes | Cause::Magnitude
            ) {
                // Announced where `linear_attachment.rs` or `kept_bytes.rs` chose the kept image; a
                // magnitude stand-in's pass tests and keeps the depth as Metal's
                // (`texture/signed_depth.rs`, which announces what it does not).
            } else if memoryless
                && (index.is_some_and(fetched) || (index.is_none() && shadow) || !chosen)
            {
                // Not for a split the device chose (`chosen`).
                self.degrade(Degradation::MemorylessBacked, || {
                    let image = image();
                    let why = if index.is_some_and(fetched) {
                        "a framebuffer fetch through the pixel interlock reads it".to_owned()
                    } else if index.is_none() && shadow {
                        "a draw tests its stencil in the fragment shader, which copies it to and \
                         from a shadow"
                            .to_owned()
                    } else {
                        format!("its render pass splits into {instances} instances")
                    };
                    (
                        format!("a memoryless {} texture", described(own)),
                        format!("{why}, so it renders into {image}"),
                    )
                });
            }
        }
    }
}

impl Device {
    /// Announce the shader-side stencil of an ended encoder's pass, `layout` its layout with its
    /// attachments' stand-ins, when some of its draws test the stencil in the fragment shader
    /// (`shadow`, `shadow_stencil.rs`).
    pub(super) fn degrade_shadow_stencil(&self, layout: &PassLayout, shadow: bool) {
        let Some(ds) = layout.depth_stencil.as_ref().filter(|_| shadow) else {
            return;
        };
        self.degrade(Degradation::ShaderSideStencil, || {
            let own = ds.stand_in.as_ref().map_or(&ds.target, |s| &s.own);
            (
                format!("a {} stencil attachment", described(own)),
                "a draw's fragment function returns `[[stencil]]` on a device without \
                 `VK_EXT_shader_stencil_export`, so it tests and writes a shadow of the stencil \
                 itself"
                    .to_owned(),
            )
        });
    }

    /// Announce the depth stand-in of an ended encoder's pass whose draws test the depth in the
    /// shader (`tests`, `depth_stand_in.rs`), once.
    pub(super) fn degrade_shader_tests(&self, layout: &PassLayout, tests: bool) {
        let Some(ds) = layout.depth_stencil.as_ref().filter(|_| tests) else {
            return;
        };
        self.degrade(Degradation::ShaderSideDepthTest, || {
            let own = ds.stand_in.as_ref().map_or(&ds.target, |s| &s.own);
            (
                format!("a {} depth attachment", described(own)),
                "a draw's fragment function stores a color through the pixel interlock before \
                 the depth and stencil tests, which may drop the fragment, so it runs them \
                 itself against stand-ins of the depth and, where the draw's stencil test acts, \
                 of the stencil"
                    .to_owned(),
            )
        });
    }

    /// Announce a tile dispatch's scratch images (`render/tile_dispatch.rs`): it stages `staged`,
    /// the pass's attachment `own` among them, through scratch images over `area`.
    pub(super) fn degrade_tile_scratch(&self, own: &Target, staged: usize, area: [u32; 2]) {
        self.degrade(Degradation::TileDispatchScratch, || {
            let [width, height] = area;
            (
                format!("a tile dispatch's {} color attachment", described(own)),
                format!(
                    "its tile function stages {staged} attachment(s) through storage views in \
                     their staging formats, of scratch images over the {width}x{height} pixels \
                     the dispatch covers, which each attachment is copied into and back from"
                ),
            )
        });
    }

    /// Announce the fragment imageblock storage of an ended encoder's pass
    /// (`fragment_imageblock.rs`).
    pub(super) fn degrade_fragment_imageblock(&self, imageblock: &PassImageblock) {
        self.degrade(Degradation::FragmentImageblockStorage, || {
            let (words, [width, height], bytes) = imageblock.described();
            let image = match bytes {
                Some(bytes) => format!("an R32Uint array image of {words} layers, {bytes} bytes"),
                None => format!("an R32Uint array image of {words} layers"),
            };
            (
                format!("a {width}x{height} render pass's fragment imageblock"),
                format!(
                    "its draws' fragment functions or its tile functions declare an explicit \
                     imageblock of {words} words a pixel, which Metal keeps in tile memory, so \
                     they read and write it in {image}"
                ),
            )
        });
    }

    /// Announce the sample planes of an ended encoder's pass, `layout` its layout with its
    /// attachments' stand-ins and planes (`sample_planes.rs`).
    pub(super) fn degrade_planes(&self, layout: &PassLayout) {
        let planes = layout.colors.iter().enumerate().find_map(|(index, color)| {
            let color = color.as_ref()?;
            Some((index, color, color.planes.as_ref()?))
        });
        if let Some((index, color, planes)) = planes {
            self.degrade(Degradation::SamplePlanes, || {
                let own = color.stand_in.as_ref().map_or(&color.target, |s| &s.own);
                let image = match planes.texture.placed_bytes() {
                    Some(bytes) => {
                        format!("an array image of {} layers, {bytes} bytes", planes.layers)
                    }
                    None => format!("an array image of {} layers", planes.layers),
                };
                (
                    format!("a multisampled {} render target", described(own)),
                    format!(
                        "a render pass reads color attachment {index} through the pixel \
                         interlock on a device without multisampled storage images, so its \
                         draws read and write its samples in {image}, one sample per layer"
                    ),
                )
            });
        }
    }
}

/// `target`'s pixel format, extent and samples, for a notice.
fn described(target: &Target) -> String {
    let [width, height] = target.extent;
    format!(
        "{:?} {width}x{height} of {} sample(s)",
        target.texture.pixel_format(),
        target.samples
    )
}
