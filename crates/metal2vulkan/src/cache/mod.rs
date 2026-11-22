//! Caching: keys, versions, and artifact encodings. The crate keeps no cache of its own; the
//! consumer owns storage, bounding, and eviction.
//!
//! Keys are SHA-256 content hashes over a canonical encoding this crate owns: equal keys mean
//! equal output under the stated [`SEMANTICS_VERSION`], and every key includes the device
//! profile's fingerprint, so an artifact never serves another device.
//!
//! - A pipeline's [`PipelineKey`] comes before compiling from `Device::compute_pipeline_key` (no
//!   decode) or `Device::render_pipeline_key` (which also gives each stage's
//!   [`FunctionCompileKey`](crate::library::FunctionCompileKey)); a compiled pipeline carries it
//!   as `key`.
//! - [`PipelineArtifact`], implemented by `ComputePipeline` and `RenderPipeline`, encodes an
//!   artifact ([`PipelineArtifact::to_bytes`]) and decodes the bytes stored under a key
//!   ([`PipelineArtifact::from_bytes_for_key`]); `CompiledFunction` has the same pair. A decode
//!   checks the format and semantics versions, the key, the device's profile fingerprint, and a
//!   checksum, then scans the modules; it does not re-run the verifier. Every [`ArtifactError`]
//!   is a cache miss: compile again and replace the entry.
//!
//! [`VariantKey`] names a variant of a pipeline, the lowering inputs each of its functions
//! compiles under ([`VariantKey::lowering_inputs`]), and
//! [`SpecializationKey`] is the data of one `VkSpecializationInfo`.
//!
//! Three versions move independently: the crate's (the API), [`SEMANTICS_VERSION`] (the output:
//! every key includes it, so a bump misses every cached artifact), and [`ARTIFACT_FORMAT_VERSION`]
//! (the encoding).

// Key derivation is `pipeline::compile::key`. `CompiledFunction::to_bytes`/`from_bytes` and the
// `PipelineArtifact` impls frame a canonical body encoding (`codec`) with magic,
// `ARTIFACT_FORMAT_VERSION`, `SEMANTICS_VERSION`, the key, the profile fingerprint, and an XXH64
// checksum (`envelope`, `checksum`). A load is trusted by that identity plus a linear sanity scan
// of every module (`scan`), not by re-running the owned verifier.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub use crate::air::container::Hash256;
use crate::device::Device;
use crate::pipeline::lower_inputs::StorageImageChoice;
use crate::pipeline::sym::SlotRef;
use crate::vk;

mod artifacts;
mod checksum;
mod codec;
// Content digests: `ComputePipeline::content_digest` and `RenderPipeline::content_digest`.
pub(crate) mod digest;
mod envelope;
mod scan;
#[cfg(test)]
mod tests;

/// The output-semantics version: the version of what the translator emits. Every pipeline and
/// function key includes it and every encoded artifact carries it, so a bump misses every cached
/// artifact and key.
///
/// Bump it whenever the output for any input may change under an unchanged key: a module's words,
/// a refusal or its class, a requirement, a reflection, interface or binding-plan fact (whether a
/// kernel observes its threadgroup partition among them, which a runtime may keep beside the
/// lowerings), or a precision note. `src/tests/golden_hashes.rs` fails when a golden output hash changes under an
/// unchanged version, and its bless records the version. Independent of the crate's version and of
/// [`ARTIFACT_FORMAT_VERSION`]. The commit that bumps it says what changed; this comment keeps no
/// history.
pub const SEMANTICS_VERSION: u32 = 2275;

/// The byte-layout version of encoded artifacts (`CompiledFunction::to_bytes` and the
/// [`PipelineArtifact`] encodings). Every encoded artifact carries it, and a decode under another
/// value is an [`ArtifactError`], a cache miss. Independent of [`SEMANTICS_VERSION`].
///
/// Bump it whenever the encoding changes: a field, variant or tag added, removed, reordered or
/// re-typed in an encoded type, its `codec` body or its `envelope`. A variant removed while its tag
/// stays unused needs no bump: an entry holding it fails to decode, which is a miss. The commit that
/// bumps it says what changed; this comment keeps no history.
pub const ARTIFACT_FORMAT_VERSION: u32 = 591;

/// The version of what this crate gives a runtime for the runs of a compiled pipeline, beside the
/// compile: the device a run opens and what it enables, the Vulkan and Metal vocabularies a runtime
/// maps, the record and table formats it writes (indirect commands, rate and tile maps, function
/// tables, argument-buffer layouts), the texel decoders, and the other run-time data `bind` holds.
/// Independent of [`SEMANTICS_VERSION`] (what is compiled) and [`ARTIFACT_FORMAT_VERSION`] (how it
/// is encoded): two runs of byte-identical artifacts under one runtime version do the same thing. A
/// consumer that keeps records of what runs of an artifact did (its own regression records, or what
/// a run refused) keys them on it beside the artifact, as it keys cached artifacts on
/// [`SEMANTICS_VERSION`].
///
/// Bump it whenever a run of an unchanged artifact may do something else: an answer, a refusal or
/// its class, an announcement, or what the run records and submits. A field added to an encoded
/// artifact whose zero value would change what existing pipelines do needs the same bump (or the
/// executor's), since its digest omits it.
///
/// This workspace's validation holds every recorded run to it beside the run's artifact content
/// digest (`cache::digest`), and its runtime guard fails when the runtime's sources no longer hash
/// to the hash it pins (`runtime-sources.sha256` in the validation crate). An edit there either
/// changes no behaviour, and then updates that hash alone with the commit saying why, or changes
/// behaviour and also bumps this version. The commit that bumps it says what changed; this comment
/// keeps no history.
pub const RUNTIME_VERSION: u32 = 2550;

/// Writes a digest as lowercase hex.
pub(crate) fn write_hex(f: &mut fmt::Formatter<'_>, name: &str, hash: &Hash256) -> fmt::Result {
    f.write_str(name)?;
    f.write_str("(")?;
    for b in hash {
        write!(f, "{b:02x}")?;
    }
    f.write_str(")")
}

/// The cache key of a pipeline: H(`SEMANTICS_VERSION`, profile fingerprint, per stage the
/// `FunctionCompileKey`, translation-relevant descriptor fields).
#[non_exhaustive]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PipelineKey(pub Hash256);

impl PipelineKey {
    /// The key with this digest (for keys the consumer stored itself).
    pub const fn new(hash: Hash256) -> PipelineKey {
        PipelineKey(hash)
    }

    /// The digest.
    pub const fn digest(&self) -> &Hash256 {
        &self.0
    }
}

impl fmt::Debug for PipelineKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_hex(f, "PipelineKey", &self.0)
    }
}

/// The per-function variant choices of a pipeline: storage-image choices for slots whose
/// `StorageTextureFormat` fact maps to `Variant`, slots that a `BufferRangeLimit` variant moves to
/// `BufferForm::Psb`, vertex buffers fetched in the shader, a kernel's emulated simdgroups, and the
/// other forms each field names. Canonical (ordered maps) and empty on the common path.
///
/// [`VariantKey::lowering_inputs`] is each function's `LoweringInputs` under the key, so the
/// choice reaches `select` and the `FunctionCompileKey`; a render pipeline's keys under one are
/// `Device::render_variant_key`'s.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VariantKey {
    #[cfg_attr(feature = "serde", serde(with = "crate::pipeline::serde_pairs"))]
    pub storage_images: BTreeMap<SlotRef, StorageImageChoice>,
    pub psb_slots: BTreeSet<SlotRef>,
    /// Metal vertex-buffer indices whose layouts the vertex function fetches in the shader
    /// (`VertexFetch::Pulled`) instead of through fixed-function vertex input: the shape binds
    /// them at an offset or with a per-bind stride Vulkan's vertex input cannot fetch.
    #[cfg_attr(feature = "serde", serde(default))]
    pub pulled_vertex_buffers: BTreeSet<u32>,
    /// The kernel's simdgroups run in the emulated tier, which is exact for every threadgroup size:
    /// a dispatch whose threadgroup has a partial simdgroup asks for it where the kernel's own tier
    /// runs simdgroups on whole subgroups.
    #[cfg_attr(feature = "serde", serde(default))]
    pub emulated_simdgroups: bool,
    /// Every threadgroup of the dispatch is one row, so each of its simdgroups lies within the row:
    /// a kernel whose lockstep simdgroup barriers order only pairs that meet where a simdgroup
    /// spans two rows (`RuntimeFact::RowLockstep`) runs without them, keeping its threadgroup ones,
    /// and without the subgroup size they pinned (`LoweringInputs::one_row`). A dispatch whose
    /// every region is one row asks for it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub one_row: bool,
    /// The axes along which every threadgroup of the dispatch has one thread (bit 0 `x`, bit 1 `y`,
    /// bit 2 `z`): a kernel whose walk is predicated, where those axes' thread positions alone make
    /// some of its synchronizations' blocks divergent (`RuntimeFact::UnitAxes`'s `relief`),
    /// compiles with them uniform (`LoweringInputs::unit_axes`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub unit_axes: u8,
    /// Texture slots bound to an 8-bit sRGB texture the device cannot sample, which the runtime
    /// holds as the format's UNORM twin (`metal::format::srgb_raw_twin`), whose views read the
    /// codes (`c / 255`): every slot of a `RuntimeFact::SrgbTexture` the shape binds in such a
    /// format (`metal::texel_format::srgb_twin_on`). The function decodes what it reads there by
    /// the MSL transfer through the device's `Pow`, as the device would through an sRGB view it
    /// lacks (`LoweringInputs::srgb_textures`); a slot bound in an sRGB format the device samples
    /// reads the device's decode and is no slot of this.
    #[cfg_attr(feature = "serde", serde(default))]
    pub srgb_textures: BTreeSet<SlotRef>,
    /// Texture-buffer slots whose function reads it through its parts, each with the parts of the
    /// texture buffer the shape binds there: every slot of a `RuntimeFact::TexelBufferParts` the
    /// shape binds a texture buffer of more texels than one view holds at. Each such slot binds an array of that many texel buffer
    /// views (`bind::texel_parts`). A texture buffer one view holds adds nothing, so its draws
    /// keep their key.
    #[cfg_attr(
        feature = "serde",
        serde(default, with = "crate::pipeline::serde_pairs")
    )]
    pub texel_buffer_parts: BTreeMap<SlotRef, u32>,
    /// Texture slots a constexpr sampler samples, each with the address-mode map of the texture the
    /// shape binds there where it is not the identity: every slot of a
    /// `RuntimeFact::ConstexprSampled` the shape binds such a texture at. The function's constexpr
    /// samplers sample the slot with their modes mapped. A texture of the identity map adds
    /// nothing, so ordinary draws keep their key.
    #[cfg_attr(
        feature = "serde",
        serde(default, with = "crate::pipeline::serde_pairs")
    )]
    pub address_maps: BTreeMap<SlotRef, crate::metal::sampler::AddressModeMap>,
    /// The kernel runs its grid flat (`LoweringInputs::flat_grid`): one variant runs every grid
    /// and threadgroup size of a kernel that does not observe its threadgroup partition
    /// (`KernelFacts::partition_oblivious`), reading the grid from the region record at its
    /// `PushField::ReplayRegionAddress`.
    #[cfg_attr(feature = "serde", serde(default))]
    pub flat_grid: bool,
    /// With [`flat_grid`](Self::flat_grid), its dispatch's threadgroups are the call's
    /// (`LoweringInputs::call_threadgroups`): a one-row grid in threadgroups of one row, so a
    /// kernel that observes its partition through its simdgroups alone
    /// (`KernelFacts::partition_simdgroups`) runs flat too.
    #[cfg_attr(feature = "serde", serde(default))]
    pub call_threadgroups: bool,
    /// The kernel, or the render pipeline's vertex and fragment functions, are a device-replay
    /// variant (`LoweringInputs::replay_bindings`): every buffer they read is at its device
    /// address in the slot table a replay's preprocess writes for its command. The runtime compiles a kernel's with `flat_grid` per pipeline state an execution of
    /// commands a kernel writes may name, to run it on the GPU alone; and a render pipeline's for
    /// the state an execution of GPU-written draws inherits, its draws' buffers the records'.
    #[cfg_attr(feature = "serde", serde(default))]
    pub replay_bindings: bool,
    /// With `replay_bindings`, the render pipeline's vertex and fragment functions run a draw
    /// replay merged into one draw (`LoweringInputs::merged_draws`): the variant the runtime
    /// compiles where an execution of GPU-written draws of one list type or points draws as one
    /// draw.
    #[cfg_attr(feature = "serde", serde(default))]
    pub merged_draws: bool,
    /// The render pipeline draws each vertex amplification as a draw of its own
    /// (`LoweringInputs::amplification_draws` for its vertex and fragment functions): the variant a
    /// shape whose amplification Vulkan multiview cannot draw as Metal renders it asks for (a `nil`
    /// mapping above a count of 1, layers out of order, apart or repeated, a viewport offset, a
    /// stage that chooses or reads its slice, or more amplifications than the device has views;
    /// `bind::graphics`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub amplification_draws: Option<crate::pipeline::AmplificationDraws>,
    /// The kernel folds Metal's group counts into one dispatch within the device's limits
    /// (`LoweringInputs::folded_groups`): the variant a threadgroup dispatch whose group counts
    /// are GPU data runs on, for a kernel that observes its threadgroup partition, reading its
    /// group counts and every grid builtin's region from the region record at its
    /// `PushField::ReplayRegionAddress`.
    #[cfg_attr(feature = "serde", serde(default))]
    pub folded_groups: bool,
    /// The kernel runs a ragged `dispatchThreads` grid in the threadgroups its threads fill, at the
    /// call's threadgroup size, the last along an axis Metal's partial threadgroup
    /// (`LoweringInputs::partial_threadgroups`): the variant such a dispatch of a kernel that
    /// observes its threadgroup partition runs on, one `VkPipeline` of the call's size in place of
    /// an edge region's of each remainder.
    #[cfg_attr(feature = "serde", serde(default))]
    pub partial_threadgroups: bool,
    /// Texture slots bound in one of Apple's extended-range formats, by the format
    /// (`LoweringInputs::extended_range`): the function reads and writes the bound texture's raw
    /// texels there, decoding and encoding Metal's values.
    #[cfg_attr(
        feature = "serde",
        serde(default, with = "crate::pipeline::serde_pairs")
    )]
    pub extended_range: BTreeMap<SlotRef, crate::metal::format::ExtendedRange>,
    /// Color attachments of the pass the render pipeline has no pixel format for, the pass's
    /// pixel format of each by color index: plain Metal stores zero
    /// bytes into each at every sample a draw covers. The variant's pipeline has each attachment
    /// in that format, written whole and unblended, and its fragment function writes zero to it
    /// (`LoweringInputs::zeroed_colors`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub zeroed_colors: BTreeMap<u32, crate::metal::PixelFormat>,
    /// Color attachments the render pipeline has a pixel format for and the pass has none, by
    /// color index: plain Metal drops the fragment function's
    /// writes there. The variant's fragment function keeps no output at each
    /// (`LoweringInputs::dropped_colors`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub dropped_colors: BTreeSet<u32>,
    /// The bytes of each cell of a tile function's explicit imageblock standing alone that the
    /// render pass keeps, fewer than the cell holds: plain
    /// Metal reads the block's other words as zero and drops their writes. The variant's
    /// function does (`LoweringInputs::kept_imageblock_bytes`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub kept_imageblock_bytes: Option<u32>,
    /// The fragment function clamps the depth it writes to the draw's depth clamp range itself
    /// (`LoweringInputs::depth_clamp_range`): the variant a draw
    /// in depth clamp mode through a range above 0 runs, where Metal stores a negative NaN as 0.
    ///
    #[cfg_attr(feature = "serde", serde(default))]
    pub depth_clamp_range: bool,
    /// The fragment function's triangles are drawn with Metal's last vertex first
    /// (`LoweringInputs::rotated_weights`): the variant a listed
    /// triangle draw under Apple's last provoking vertex runs, whose `[[barycentric_coord]]` reads
    /// take Metal's weights back from the drawn triangle's.
    #[cfg_attr(feature = "serde", serde(default))]
    pub rotated_weights: bool,
    /// The pipeline draws its lines, or its triangles' edges, as bands of triangles through
    /// synthesized tessellation stages, each patch one primitive of this kind
    /// (`pipeline::link::band`): the variant a draw of lines, or of
    /// triangles in fill mode `Lines`, at an Apple `setLineWidth:` width the device's own lines do
    /// not draw runs. Its graphics state states it (`GraphicsState::line_band`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub line_band: Option<crate::pipeline::BandPrimitive>,
    /// Storage-image texture slots bound to a texture with a swizzle (`MTLTexture.swizzle` other
    /// than the default) whose texels the function reads or writes
    /// (`LoweringInputs::swizzled_storage`): a Vulkan storage view takes no swizzle, so the
    /// function applies the bound texture's swizzle to its reads and undoes it on its writes, as
    /// plain Metal does, under the slot's `SpecKey::TextureSwizzle`, and the slot's storage view
    /// takes none. It holds too each sampled slot of
    /// `extended_range` bound to such a texture whose texels the function reads or samples: its
    /// view holds raw codes, so the function swizzles the decoded texels and the view takes none.
    /// So does, on a device without Metal's own sampler, each sampled slot bound to a swizzled
    /// packed normalized texture, whose texels the function converts by each channel's own scale
    /// before it swizzles them.
    #[cfg_attr(feature = "serde", serde(default))]
    pub swizzled_storage: BTreeSet<SlotRef>,
    /// Every function that reads visible-function tables by their resource ids reads their
    /// words through their device addresses (`LoweringInputs::addressed_function_tables`): the
    /// variant a shape whose listed tables may pass `maxStorageBufferRange` asks for
    /// (`FunctionTablesShape::bytes_class`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub addressed_function_tables: bool,
    /// The runtime sampler slots of a `RuntimeFact::CubicSampler` that the shape binds to a sampler
    /// whose min or mag filter is `Cubic`, each with the bound word masked by
    /// `SamplerWord::CUBIC_STATE` (`LoweringInputs::cubic_samplers`): the variant whose function
    /// samples through each with Metal's bicubic filter of that state.
    #[cfg_attr(
        feature = "serde",
        serde(default, with = "crate::pipeline::serde_pairs")
    )]
    pub cubic_samplers: BTreeMap<SlotRef, u32>,
    /// The de-indexed form of a pipeline whose vertex function writes the primitive ID's varying
    /// (`LoweringInputs::primitive_id_indices`): the vertex
    /// function of an indexed draw of this index type runs the draw's indices as `index count`
    /// vertices from 0, read at the draw's index address, so each primitive's stream position names
    /// it. The runtime asks for it for an indexed draw of such a pipeline.
    #[cfg_attr(feature = "serde", serde(default))]
    pub primitive_id_indices: Option<crate::metal::IndexType>,
    /// A draw of Metal's private quads or quad strip, listed as two triangles a quad, of a
    /// pipeline whose fragment function reads Vulkan's `PrimitiveId`: the variant's fragment function reads `[[primitive_id]]` as that
    /// value shifted right by one, the quad's (`LoweringInputs::quad_primitive_ids`). The runtime
    /// asks for it for such a draw (`RuntimeFact::QuadPrimitiveIds`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub quad_primitive_ids: bool,
    /// A draw of one of Metal's private primitive types drawn in the pulled list form
    ///: the variant's vertex function runs each position of the
    /// consumer's list once, without indices, reading its index and Metal's base vertex at the
    /// list's address (`LoweringInputs::private_list`). The runtime asks for it for such a draw
    /// whose vertex function writes memory or the primitive ID's varying, or whose vertex start
    /// a list's base vertex cannot carry (`RuntimeFact::PrivateList`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub private_list: bool,
    /// Sampled texture slots whose compile states `RuntimeFact::CrossClassReads` bound to a
    /// texture of a color format whose numeric class is not the slot's declared one, by the
    /// bound format's class (`LoweringInputs::cross_class`): the variant's function reads,
    /// samples and gathers the texture in its own class and hands back its words as the declared
    /// type, as plain Metal does.
    #[cfg_attr(
        feature = "serde",
        serde(default, with = "crate::pipeline::serde_pairs")
    )]
    pub cross_class: BTreeMap<SlotRef, crate::pipeline::sym::SampledType>,
    /// The runtime sampler slots of a `RuntimeFact::AnisotropicSampler` that the shape binds to a
    /// sampler whose word states anisotropy (`SamplerWord::ANISOTROPIC`,
    /// `LoweringInputs::anisotropic_samplers`): the variant whose function takes Metal's
    /// anisotropic filter, emulated, at each one's samples.
    #[cfg_attr(feature = "serde", serde(default))]
    pub anisotropic_samplers: BTreeSet<SlotRef>,
    /// The texture slots of a kernel of OpenCL mode whose `RuntimeFact::OpenClLineTexture` (a 1D
    /// image it reads as the 2D texture of height 1 Apple's OpenCL binds) the shape binds a 1D or
    /// 1D array texture at (`LoweringInputs::open_cl_lines`): the variant's kernel declares each
    /// as that 1D texture and reads it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub open_cl_lines: BTreeSet<SlotRef>,
    /// The render pipeline draws a rasterization-rate-mapped pass's regions several at a time,
    /// each one draw of a multi-draw whose index selects its viewport and scissor: its vertex
    /// function stores `ViewportIndex` as `DrawIndex`
    /// (`LoweringInputs::region_viewports`, `RuntimeFact::RegionViewports`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub region_viewports: bool,
}

impl VariantKey {
    /// The empty key: no variant is needed.
    pub fn new() -> VariantKey {
        VariantKey::default()
    }

    /// Whether nothing needs a variant (the common path).
    pub fn is_empty(&self) -> bool {
        self.storage_images.is_empty()
            && self.psb_slots.is_empty()
            && self.pulled_vertex_buffers.is_empty()
            && !self.emulated_simdgroups
            && !self.one_row
            && self.unit_axes == 0
            && self.srgb_textures.is_empty()
            && self.texel_buffer_parts.is_empty()
            && self.address_maps.is_empty()
            && !self.flat_grid
            && !self.call_threadgroups
            && !self.replay_bindings
            && !self.merged_draws
            && self.amplification_draws.is_none()
            && !self.folded_groups
            && !self.partial_threadgroups
            && self.extended_range.is_empty()
            && self.zeroed_colors.is_empty()
            && self.dropped_colors.is_empty()
            && self.kept_imageblock_bytes.is_none()
            && !self.depth_clamp_range
            && !self.rotated_weights
            && self.line_band.is_none()
            && self.swizzled_storage.is_empty()
            && !self.addressed_function_tables
            && self.cubic_samplers.is_empty()
            && self.primitive_id_indices.is_none()
            && !self.quad_primitive_ids
            && !self.private_list
            && self.cross_class.is_empty()
            && self.anisotropic_samplers.is_empty()
            && self.open_cl_lines.is_empty()
            && !self.region_viewports
    }

    /// The lowering inputs of the function of `stage` under this key: its storage-image choices,
    /// device-address slots, kept storage views, raw sRGB slots, cubic and anisotropic samplers,
    /// for the kernel the emulated simdgroups, the flat grid, the folded groups, the partial
    /// threadgroups and the replay's bindings, for the vertex and fragment functions the draws of one amplification each, for
    /// the fragment function the colors it writes zero to and those it drops, its depth's clamp
    /// range and whether its draws' tests are read-only, for a tile function the imageblock bytes
    /// its pass keeps, and for the vertex function the
    /// dropped `Layer` output and the window depth
    /// (a vertex function's projection is
    /// derived from the vertex descriptor and
    /// [`pulled_vertex_buffers`](VariantKey::pulled_vertex_buffers) when the pipeline compiles,
    /// `Device::render_variant_key`). A key carries no stage-input descriptor: a kernel of a
    /// compute pipeline with one compiles its variant under these inputs with the pipeline's
    /// `stage_input` added (`LoweringInputs::stage_input`).
    pub fn lowering_inputs(
        &self,
        stage: crate::pipeline::sym::Stage,
    ) -> crate::pipeline::LoweringInputs {
        let mut out = crate::pipeline::LoweringInputs::new();
        out.storage_images = self
            .storage_images
            .iter()
            .filter(|(s, _)| s.stage == stage)
            .map(|(s, c)| (*s, *c))
            .collect();
        out.psb_slots = self
            .psb_slots
            .iter()
            .filter(|s| s.stage == stage)
            .copied()
            .collect();
        out.emulated_simdgroups =
            self.emulated_simdgroups && stage == crate::pipeline::sym::Stage::Kernel;
        out.one_row = self.one_row && stage == crate::pipeline::sym::Stage::Kernel;
        out.unit_axes = match stage {
            crate::pipeline::sym::Stage::Kernel => self.unit_axes,
            _ => 0,
        };
        out.flat_grid = self.flat_grid && stage == crate::pipeline::sym::Stage::Kernel;
        out.call_threadgroups =
            self.call_threadgroups && stage == crate::pipeline::sym::Stage::Kernel;
        out.folded_groups = self.folded_groups && stage == crate::pipeline::sym::Stage::Kernel;
        out.partial_threadgroups =
            self.partial_threadgroups && stage == crate::pipeline::sym::Stage::Kernel;
        out.replay_bindings = self.replay_bindings
            && matches!(
                stage,
                crate::pipeline::sym::Stage::Kernel
                    | crate::pipeline::sym::Stage::Vertex
                    | crate::pipeline::sym::Stage::Fragment
            );
        out.merged_draws = self.merged_draws
            && self.replay_bindings
            && matches!(
                stage,
                crate::pipeline::sym::Stage::Vertex | crate::pipeline::sym::Stage::Fragment
            );
        out.amplification_draws = self
            .amplification_draws
            .filter(|_| stage != crate::pipeline::sym::Stage::Kernel);
        out.region_viewports =
            self.region_viewports && stage == crate::pipeline::sym::Stage::Vertex;
        out.srgb_textures = self
            .srgb_textures
            .iter()
            .filter(|s| s.stage == stage)
            .copied()
            .collect();
        out.texel_buffer_parts = self
            .texel_buffer_parts
            .iter()
            .filter(|(s, _)| s.stage == stage)
            .map(|(s, parts)| (*s, *parts))
            .collect();
        out.address_maps = self
            .address_maps
            .iter()
            .filter(|(s, _)| s.stage == stage)
            .map(|(s, m)| (*s, *m))
            .collect();
        out.cubic_samplers = self
            .cubic_samplers
            .iter()
            .filter(|(s, _)| s.stage == stage)
            .map(|(s, w)| (*s, *w))
            .collect();
        out.anisotropic_samplers = self
            .anisotropic_samplers
            .iter()
            .filter(|s| s.stage == stage)
            .copied()
            .collect();
        out.open_cl_lines = self
            .open_cl_lines
            .iter()
            .filter(|s| s.stage == stage)
            .copied()
            .collect();
        out.extended_range = self
            .extended_range
            .iter()
            .filter(|(s, _)| s.stage == stage)
            .map(|(s, x)| (*s, *x))
            .collect();
        out.swizzled_storage = self
            .swizzled_storage
            .iter()
            .filter(|s| s.stage == stage)
            .copied()
            .collect();
        out.cross_class = self
            .cross_class
            .iter()
            .filter(|(s, _)| s.stage == stage)
            .map(|(s, c)| (*s, *c))
            .collect();
        if stage == crate::pipeline::sym::Stage::Fragment {
            out.zeroed_colors = self.zeroed_colors.clone();
            out.dropped_colors = self.dropped_colors.clone();
            out.depth_clamp_range = self.depth_clamp_range;
            out.rotated_weights = self.rotated_weights;
        }
        out.primitive_id_indices = self
            .primitive_id_indices
            .filter(|_| stage == crate::pipeline::sym::Stage::Vertex);
        out.quad_primitive_ids =
            self.quad_primitive_ids && stage == crate::pipeline::sym::Stage::Fragment;
        out.private_list = self.private_list && stage == crate::pipeline::sym::Stage::Vertex;
        out.kept_imageblock_bytes = self
            .kept_imageblock_bytes
            .filter(|_| stage == crate::pipeline::sym::Stage::Kernel);
        out.addressed_function_tables = self.addressed_function_tables;
        out
    }
}

/// Specialization data for one Vulkan pipeline, ready for `VkSpecializationInfo`: the packed
/// constant bytes and the map entries into them.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SpecializationKey {
    pub data: Vec<u8>,
    pub entries: Vec<vk::SpecializationMapEntry>,
}

impl SpecializationKey {
    /// The key with no specialization constants.
    pub fn new() -> SpecializationKey {
        SpecializationKey::default()
    }
}

/// Why encoded artifact bytes were rejected. The consumer treats every variant as a cache miss.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ArtifactError {
    /// The magic, format version, or semantics version differs from this build's.
    Stale,
    /// The bytes fail the checksum or the linear sanity scan.
    Corrupt,
    /// The artifact was compiled for a different profile fingerprint or key.
    WrongDevice,
}

impl fmt::Display for ArtifactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl std::error::Error for ArtifactError {}

/// A pipeline artifact that can be cached across processes.
///
/// `from_bytes` checks the magic, format version, [`SEMANTICS_VERSION`], key, the device's profile
/// fingerprint, and a checksum, then runs a linear sanity scan; it trusts the words by identity and
/// does not re-run the owned verifier.
pub trait PipelineArtifact: Sized + Send + Sync {
    /// The artifact's key.
    fn key(&self) -> &PipelineKey;
    /// The versioned, self-checking encoding.
    fn to_bytes(&self) -> Vec<u8>;
    /// Write [`PipelineArtifact::to_bytes`] to `out` without holding the encoding whole: the
    /// pipelines write it a chunk at a time, so writing one holds little beside it. An unstable
    /// internal of the validation worker, not API.
    #[doc(hidden)]
    fn write_to(&self, out: &mut dyn std::io::Write) -> std::io::Result<()> {
        out.write_all(&self.to_bytes())
    }
    /// Decodes an artifact compiled for `device`, or says why the bytes cannot be trusted.
    fn from_bytes(bytes: &[u8], device: &Device) -> Result<Self, ArtifactError>;
    /// [`PipelineArtifact::from_bytes`] for the bytes a cache stored under `key`: bytes of any
    /// other key are [`ArtifactError::WrongDevice`], like bytes of another profile.
    fn from_bytes_for_key(
        bytes: &[u8],
        key: &PipelineKey,
        device: &Device,
    ) -> Result<Self, ArtifactError> {
        let artifact = Self::from_bytes(bytes, device)?;
        if artifact.key() == key {
            Ok(artifact)
        } else {
            Err(ArtifactError::WrongDevice)
        }
    }
}
