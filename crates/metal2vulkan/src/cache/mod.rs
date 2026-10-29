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
//! - A [`BindingPlan`](crate::bind::BindingPlan) memoizes in memory on `(PipelineKey, BindShape)`.
//! - A failed compile memoizes only as `Error::is_permanent_for` allows.
//!
//! [`VariantKey`] names the variant of a pipeline a bind shape or a dispatch asks for, and
//! [`SpecializationKey`] is the data of one `VkSpecializationInfo`.
//!
//! Three versions move independently: the crate's (the API), [`SEMANTICS_VERSION`] (the output:
//! every key includes it, so a bump misses every cached artifact), and [`ARTIFACT_FORMAT_VERSION`]
//! (the encoding).

// Key derivation is `pipeline::compile::key`. `CompiledFunction::to_bytes`/`from_bytes` and the
// `PipelineArtifact` impls frame a canonical body encoding (`codec`) with magic,
// `ARTIFACT_FORMAT_VERSION`, `SEMANTICS_VERSION`, the key, the profile fingerprint, and a SHA-256
// checksum (`envelope`). A load is trusted by that identity plus a linear sanity scan of every
// module (`scan`), not by re-running the owned verifier.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub use crate::air::container::Hash256;
use crate::device::Device;
use crate::pipeline::lower_inputs::StorageImageChoice;
use crate::pipeline::sym::{SlotRef, SnapshotForm};
use crate::vk;

mod artifacts;
mod codec;
// Content digests: `ComputePipeline::content_digest`, `RenderPipeline::content_digest` and
// `BindingPlan::content_digest`.
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
/// a refusal or its class, a requirement, a reflection, interface or binding-plan fact, or a
/// precision note. `src/tests/golden_hashes.rs` fails when a golden output hash changes under an
/// unchanged version, and its bless records the version. Independent of the crate's version and of
/// [`ARTIFACT_FORMAT_VERSION`]. The commit that bumps it says what changed; this comment keeps no
/// history.
pub const SEMANTICS_VERSION: u32 = 1929;

/// The byte-layout version of encoded artifacts (`CompiledFunction::to_bytes` and the
/// [`PipelineArtifact`] encodings). Every encoded artifact carries it, and a decode under another
/// value is an [`ArtifactError`], a cache miss. Independent of [`SEMANTICS_VERSION`].
///
/// Bump it whenever the encoding changes: a field, variant or tag added, removed, reordered or
/// re-typed in an encoded type, its `codec` body or its `envelope`. A variant removed while its tag
/// stays unused needs no bump: an entry holding it fails to decode, which is a miss. The commit that
/// bumps it says what changed; this comment keeps no history.
pub const ARTIFACT_FORMAT_VERSION: u32 = 479;

/// The version of what this crate's runtime does with a compiled pipeline: instantiating a binding
/// plan (`BindingPlan::instantiate_*`: descriptor ranges, push bytes, root, argument, and handle
/// tables, dispatch regions and draws), the device replay and its preprocess, resident sets, the
/// indirect clamp, status decoding, and the other run-time data the binder computes. Independent of
/// [`SEMANTICS_VERSION`] (what is compiled) and [`ARTIFACT_FORMAT_VERSION`] (how it is encoded):
/// two runs of byte-identical artifacts and plans under one runtime version do the same thing. A
/// consumer that keeps records of what runs of an artifact did (its own regression records, or what
/// a run refused) keys them on it beside the artifact, as it keys cached artifacts on
/// [`SEMANTICS_VERSION`].
///
/// Bump it whenever a run of an unchanged artifact and plan may do something else: an answer, a
/// refusal or its class, an announcement, or what the run records and submits. A field added to an
/// encoded artifact or a plan whose zero value would change what existing pipelines do needs the
/// same bump (or the executor's), since their digests omit it.
///
/// This workspace's validation holds every recorded run to it beside the run's artifact content
/// digest (`cache::digest`), and its runtime guard fails when the runtime's sources no longer hash
/// to the hash it pins (`runtime-sources.sha256` in the validation crate). An edit there either
/// changes no behaviour, and then updates that hash alone with the commit saying why, or changes
/// behaviour and also bumps this version. The commit that bumps it says what changed; this comment
/// keeps no history.
pub const RUNTIME_VERSION: u32 = 1703;

/// The digest of the sources this build of the crate was compiled from: SHA-256, as lowercase hex,
/// of the manifest, the build script, and exactly the files the crate roots' module tree compiles
/// or reads. A cache of translations across builds keys on it beside [`SEMANTICS_VERSION`] and
/// [`ARTIFACT_FORMAT_VERSION`], so a build of other sources never reads another build's output,
/// even when no version moved; `metal2vulkan-runtime`'s shader cache does. An edit to a test-only
/// module leaves it unchanged.
pub const SOURCE_DIGEST: &str = env!("M2V_SOURCE_DIGEST");

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

/// The per-function variant choices a bind shape or a dispatch can require: storage-image choices
/// for slots whose `StorageTextureFormat` fact maps to `Variant`, slots that a `BufferRangeLimit`
/// variant moves to `BufferForm::Psb`, vertex buffers the shape binds where fixed-function vertex
/// input cannot fetch them, a kernel's emulated simdgroups for a dispatch whose threadgroup has a
/// partial simdgroup, a vertex function's dropped `Layer` output in a pass of one layer, and the
/// read-only slots Profile F's per-stage storage-buffer fallback keeps as storage views. Canonical
/// (ordered maps) and empty on the common path.
///
/// [`VariantKey::lowering_inputs`] is each function's `LoweringInputs` under the key, so the
/// choice reaches `select` and the `FunctionCompileKey`; a render pipeline compiles under a key
/// with `Device::compile_render_variant`.
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
    /// runs simdgroups on whole subgroups (`BindingPlan::dispatch_variant`,
    /// `BindError::NeedsVariant`), and the plan of a pipeline compiled under it states it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub emulated_simdgroups: bool,
    /// The vertex function's `[[render_target_array_index]]` output is not written: in a pass of at
    /// most one layer (`BindShape::render_target_array_length` 0 or 1) every index renders slice 0,
    /// so a pipeline whose vertex function chooses its slice runs on this variant there, and only
    /// there (a pass of more layers asks for the variant that writes `Layer`). The plan of a
    /// pipeline compiled under it states it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub dropped_vertex_layer: bool,
    /// Read-only slots that Profile F's per-stage storage-buffer fallback keeps as storage views:
    /// the shape binds a buffer whose length class does not prove that a uniform view's range fits
    /// `maxUniformBufferRange` at a slot the fallback views as a uniform buffer, so every such
    /// candidate stays a storage view and the fallback views others (or the compile refuses the
    /// per-stage limit). The plan of a pipeline compiled under it states it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub storage_view_slots: BTreeSet<SlotRef>,
    /// Texture slots bound to an 8-bit sRGB texture the device cannot sample, which the runtime
    /// holds as the format's UNORM twin (`metal::format::srgb_raw_twin`), whose views read the
    /// codes (`c / 255`): every slot of a `RuntimeFact::SrgbTexture` the shape binds in such a
    /// format (`metal::texel_format::srgb_twin_on`). The function decodes what it reads there
    /// by the MSL transfer through the device's `Pow`, as the device would through an sRGB view
    /// it lacks
    /// (`LoweringInputs::srgb_textures`); a slot bound in an sRGB format the device samples reads
    /// the device's decode and is no slot of this. The plan of a pipeline compiled under it
    /// states it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub srgb_textures: BTreeSet<SlotRef>,
    /// Fragment texture slots the draw reads through a snapshot an earlier draw of the same
    /// primitives took (`LoweringInputs::snapshot_textures`), each with the form the snapshot
    /// keeps its texels in (`SnapshotForm::of` the slot's bound view): the slots
    /// `BindShape::snapshot_textures` names, each one of the fragment stage's
    /// `StageInterface::snapshot_textures`. The plan of a pipeline compiled under it states it,
    /// and binds each slot's snapshot (`WriteSource::SnapshotTexels`, an image of the form's
    /// `SnapshotForm::format`) and leaves which of its draws runs to the consumer
    /// (`PushHole::SnapshotPhase`).
    #[cfg_attr(
        feature = "serde",
        serde(default, with = "crate::pipeline::serde_pairs")
    )]
    pub snapshot_textures: BTreeMap<SlotRef, SnapshotForm>,
    /// Texture slots whose function reads the two-plane YCbCr texture the shape binds there in
    /// place: every slot of a `RuntimeFact::YCbCrInPlace` the shape binds a texture whose planes
    /// are read so (`TextureShape::Texture::ycbcr_in_place`). Each such slot binds the planes'
    /// texel buffer alone, and its accesses read the codes and convert them as Metal does
    /// (`bind::YCbCrInPlace`). The plan of a pipeline compiled under it states it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub ycbcr_in_place: BTreeSet<SlotRef>,
    /// Slots of [`ycbcr_in_place`](Self::ycbcr_in_place) whose function's every sample the
    /// device's own sampler Y′CbCr conversion takes, at which the shape binds a texture whose
    /// planes are also held as an image it samples, each with that conversion
    /// (`TextureShape::Texture::ycbcr_planes_sampler`, `RuntimeFact::YCbCrInPlace`'s `device`):
    /// each such slot binds that image through an immutable sampler of the conversion, and its
    /// samples are the device's (`bind::YCbCrPlanesSampler`). The plan of a pipeline compiled
    /// under it states it.
    #[cfg_attr(
        feature = "serde",
        serde(default, with = "crate::pipeline::serde_pairs")
    )]
    pub ycbcr_planes_sampled: BTreeMap<SlotRef, crate::bind::YCbCrPlanesSampler>,
    /// Texture slots a constexpr sampler samples, each with the address-mode map of the texture
    /// the shape binds there where it is not the identity: every slot of a
    /// `RuntimeFact::ConstexprSampled` the shape binds such a texture at. The function's constexpr
    /// samplers sample the slot with their modes mapped; the plan of a pipeline compiled under it
    /// states it. A texture of the identity map adds nothing, so ordinary draws keep their key.
    #[cfg_attr(
        feature = "serde",
        serde(default, with = "crate::pipeline::serde_pairs")
    )]
    pub address_maps: BTreeMap<SlotRef, crate::metal::sampler::AddressModeMap>,
    /// The kernel runs its grid flat (`LoweringInputs::flat_grid`): one variant runs every grid
    /// and threadgroup size of a kernel that does not observe its threadgroup partition
    /// (`KernelFacts::partition_oblivious`), reading the grid from the region record at its
    /// `PushField::ReplayRegionAddress`. The plan of a pipeline compiled under it states it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub flat_grid: bool,
    /// The kernel, or the render pipeline's vertex and fragment functions, are a device-replay
    /// variant (`LoweringInputs::replay_bindings`): every buffer they read is at its device
    /// address in the slot table a replay's preprocess writes for its command, so
    /// `BindingPlan::instantiate_replay` instantiates a compute plan, which does not state the
    /// key. The runtime compiles a kernel's with `flat_grid` per pipeline state an execution of
    /// commands a kernel writes may name, to run it on the GPU alone; and a render pipeline's for
    /// the state an execution of GPU-written draws inherits, its draws' buffers the records'.
    #[cfg_attr(feature = "serde", serde(default))]
    pub replay_bindings: bool,
    /// With `replay_bindings`, the render pipeline's vertex and fragment functions run a draw
    /// replay merged into one draw (`LoweringInputs::merged_draws`): the variant the runtime
    /// compiles where an execution of GPU-written draws of one list type or points draws as one
    /// draw. The plan of a pipeline compiled under it states it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub merged_draws: bool,
    /// The render pipeline draws each vertex amplification as a draw of its own
    /// (`LoweringInputs::amplification_draws` for its vertex and fragment functions): the
    /// variant a shape whose amplification Vulkan multiview cannot draw as Metal renders it asks
    /// for (a `nil` mapping above a count of 1, layers out of order, apart or repeated, a
    /// viewport offset, a stage that chooses or reads its slice, or more amplifications than
    /// the device has views; `bind::graphics`). The plan of a pipeline compiled under it states
    /// it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub amplification_draws: Option<crate::pipeline::AmplificationDraws>,
    /// The vertex function compensates its position for a viewport Vulkan's cannot be
    /// (`LoweringInputs::viewport_compensation`): the variant a shape whose viewport has a
    /// negative width, or a depth range past `[0, 1]` the device does not take, asks for
    /// (`BindShape::viewport_compensation`). The plan of a pipeline compiled under it states it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub viewport_compensation: Option<crate::pipeline::ViewportCompensation>,
    /// The kernel folds Metal's group counts into one dispatch within the device's limits
    /// (`LoweringInputs::folded_groups`): the variant a threadgroup dispatch whose group counts
    /// are GPU data runs on, for a kernel that observes its threadgroup partition, reading its
    /// group counts and every grid builtin's region from the region record at its
    /// `PushField::ReplayRegionAddress`. The plan of a pipeline compiled under it states it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub folded_groups: bool,
    /// Texture slots bound in one of Apple's extended-range formats, by the format
    /// (`LoweringInputs::extended_range`): the function reads and writes the bound texture's raw
    /// texels there, decoding and encoding Metal's values. The plan of a pipeline compiled under
    /// it states it.
    #[cfg_attr(
        feature = "serde",
        serde(default, with = "crate::pipeline::serde_pairs")
    )]
    pub extended_range: BTreeMap<SlotRef, crate::metal::format::ExtendedRange>,
    /// Color attachments of the pass the render pipeline has no pixel format for, the pass's
    /// pixel format of each by color index (`BindShape::zeroed_colors`): plain Metal stores zero
    /// bytes into each at every sample a draw covers. The variant's pipeline has each attachment
    /// in that format, written whole and unblended, and its fragment function writes zero to it
    /// (`LoweringInputs::zeroed_colors`). The plan of a pipeline compiled under it states it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub zeroed_colors: BTreeMap<u32, crate::metal::PixelFormat>,
    /// Color attachments the render pipeline has a pixel format for and the pass has none, by
    /// color index (`BindShape::dropped_colors`): plain Metal drops the fragment function's
    /// writes there. The variant's fragment function keeps no output at each
    /// (`LoweringInputs::dropped_colors`). The plan of a pipeline compiled under it states it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub dropped_colors: BTreeSet<u32>,
    /// The bytes of each cell of a tile function's explicit imageblock standing alone that the
    /// render pass keeps, fewer than the cell holds (`BindShape::kept_imageblock_bytes`): plain
    /// Metal reads the block's other words as zero and drops their writes. The variant's
    /// function does (`LoweringInputs::kept_imageblock_bytes`). The plan of a tile function with
    /// such an imageblock states it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub kept_imageblock_bytes: Option<u32>,
    /// The fragment function clamps the depth it writes to the draw's depth clamp range itself
    /// (`BindShape::depth_clamp_range`, `LoweringInputs::depth_clamp_range`): the variant a draw
    /// in depth clamp mode through a range above 0 runs, where Metal stores a negative NaN as 0.
    /// The plan of a pipeline compiled under it states it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub depth_clamp_range: bool,
    /// The fragment function's triangles are drawn with Metal's last vertex first
    /// (`BindShape::rotated_weights`, `LoweringInputs::rotated_weights`): the variant a listed
    /// triangle draw under Apple's last provoking vertex runs, whose `[[barycentric_coord]]` reads
    /// take Metal's weights back from the drawn triangle's. The plan of a pipeline compiled under
    /// it states it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub rotated_weights: bool,
    /// The synthesized vertex stage of an emulated mesh function's pipeline draws each line or
    /// triangle from its last index first (`BindShape::mesh_last_first`): the variant a mesh draw
    /// under Apple's last provoking vertex runs on a device that takes flat inputs from the
    /// first (`pipeline::link::mesh`). The plan of a pipeline compiled under it states it
    /// (`RuntimeFact::MeshLastFirst`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub mesh_last_first: bool,
    /// Storage-image texture slots bound to a texture with a swizzle (`MTLTexture.swizzle` other
    /// than the default) whose texels the function reads or writes
    /// (`LoweringInputs::swizzled_storage`): a Vulkan storage view takes no swizzle, so the
    /// function applies the bound texture's swizzle to its reads and undoes it on its writes, as
    /// plain Metal does, under the slot's `SpecKey::TextureSwizzle`, and the slot's storage view
    /// takes none (`ImageViewDesc::shader_swizzle`). The plan of a pipeline compiled under it
    /// states it. It holds too each sampled slot of `extended_range` bound to such a texture whose
    /// texels the function reads or samples: its view holds raw codes, so the function swizzles the
    /// decoded texels and the view takes none. So does, on a device without Metal's own sampler,
    /// each sampled slot bound to a swizzled packed normalized texture, whose texels the function
    /// converts by each channel's own scale before it swizzles them.
    #[cfg_attr(feature = "serde", serde(default))]
    pub swizzled_storage: BTreeSet<SlotRef>,
    /// Every function that reads visible-function tables by their resource ids reads their
    /// words through their device addresses (`LoweringInputs::addressed_function_tables`): the
    /// variant a shape whose listed tables may pass `maxStorageBufferRange` asks for
    /// (`FunctionTablesShape::bytes_class`). The plan of a pipeline compiled under it states it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub addressed_function_tables: bool,
    /// Color attachments the draw's render pass instance leaves out of its subpass, which its
    /// draws read and write as storage images in the pixel interlock, that the fragment function
    /// writes through its first output (`BindShape::interlocked_colors`): the variant's function
    /// writes each through the interlocked storage form, as a function that fetches the color
    /// writes it, so the draw records in that instance. Not a lowering input of its own: the render
    /// pipeline's compile adds the colors to the fragment function's
    /// `LoweringInputs::framebuffer_fetch`, whose exactness conditions refuse what the form does
    /// not keep Metal's (a blend, among others).
    #[cfg_attr(feature = "serde", serde(default))]
    pub interlocked_colors: BTreeSet<u32>,
    /// The draw's depth and stencil tests may drop fragments but write neither attachment and
    /// count no visibility result (`BindShape::fragment_tests`), and the fragment function stores
    /// a color through the interlocked storage form before the tests
    /// (`RuntimeFact::TestedStorageWrites`): the variant whose function declares early fragment
    /// tests (`LoweringInputs::read_only_tests`). The plan of a pipeline compiled under it states
    /// it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub read_only_tests: bool,
    /// Only the draw's depth test can drop fragments (`BindShape::fragment_tests`,
    /// `FragmentTests::Depth`), and the fragment function stores a color through the interlocked
    /// storage form before the tests (`RuntimeFact::TestedStorageWrites`): the variant whose
    /// function runs the depth test itself against a stand-in of the depth attachment
    /// (`LoweringInputs::shader_tests`). The plan of a pipeline compiled under it states it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub shader_tests: bool,
    /// Under `shader_tests`, the draw's stencil test can drop fragments or write too
    /// (`FragmentTests::Stencil`, or `FragmentTests::Drop` for a function without an early-tests
    /// variant), in a pass with a depth and a stencil attachment: the variant whose function also
    /// runs the stencil test itself against a stand-in of the stencil attachment
    /// (`LoweringInputs::shader_stencil`). The plan of a pipeline compiled under it states it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub shader_stencil: bool,
    /// Under `shader_tests` at more than one sample, on a device whose fragment function computes
    /// each sample's depth (`VkProperty::CenterFragCoordDepth`): the vertex function also writes
    /// its window depth for it (`LoweringInputs::window_depth_varying`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub window_depth_varying: bool,
    /// The runtime sampler slots of a `RuntimeFact::CubicSampler` that the shape binds to a
    /// sampler whose min or mag filter is `Cubic`, each with the bound word masked by
    /// `SamplerWord::CUBIC_STATE` (`LoweringInputs::cubic_samplers`): the variant whose function
    /// samples through each with Metal's bicubic filter of that state. The plan of a pipeline
    /// compiled under it states it.
    #[cfg_attr(
        feature = "serde",
        serde(default, with = "crate::pipeline::serde_pairs")
    )]
    pub cubic_samplers: BTreeMap<SlotRef, u32>,
    /// The de-indexed form of a pipeline whose vertex function writes the primitive ID's varying
    /// (`LoweringInputs::primitive_id_indices`, `BindShape::primitive_id_indices`): the vertex
    /// function of an indexed draw of this index type runs the draw's indices as `index count`
    /// vertices from 0, read at the draw's index address, so each primitive's stream position
    /// names it. The runtime asks for it for an indexed draw of such a pipeline, and the plan of
    /// a pipeline compiled under it states it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub primitive_id_indices: Option<crate::metal::IndexType>,
    /// A draw of Metal's private quads or quad strip, listed as two triangles a quad, of a
    /// pipeline whose fragment function reads Vulkan's `PrimitiveId` (`BindShape::
    /// quad_primitive_ids`): the variant's fragment function reads `[[primitive_id]]` as that
    /// value shifted right by one, the quad's (`LoweringInputs::quad_primitive_ids`). The runtime
    /// asks for it for such a draw, and the plan of a pipeline compiled under it states it
    /// (`RuntimeFact::QuadPrimitiveIds`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub quad_primitive_ids: bool,
    /// A draw of one of Metal's private primitive types drawn in the pulled list form
    /// (`BindShape::private_list`): the variant's vertex function runs each position of the
    /// consumer's list once, without indices, reading its index and Metal's base vertex at the
    /// list's address (`LoweringInputs::private_list`). The runtime asks for it for such a draw
    /// whose vertex function writes memory or the primitive ID's varying, or whose vertex start
    /// a list's base vertex cannot carry, and the plan of a pipeline compiled under it states it
    /// (`RuntimeFact::PrivateList`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub private_list: bool,
    /// Extended-range color attachments whose `half` shadow the draw's render pass instance
    /// holds (`BindShape::extended_shadows`): the variant's fragment function reads and writes
    /// each through the interlocked storage form against the shadow as Metal's tile value
    /// (`LoweringInputs::extended_shadows`), and keeps writing the codes. The runtime asks for it
    /// for every draw of an instance that holds the shadow and reads or writes the attachment.
    #[cfg_attr(feature = "serde", serde(default))]
    pub extended_shadows: BTreeSet<u32>,
    /// Sampled texture slots whose compile states `RuntimeFact::CrossClassReads` bound to a
    /// texture of a color format whose numeric class is not the slot's declared one, by the
    /// bound format's class (`LoweringInputs::cross_class`): the variant's function reads,
    /// samples and gathers the texture in its own class and hands back its words as the declared
    /// type, as plain Metal does. The plan of a pipeline compiled under it states it.
    #[cfg_attr(
        feature = "serde",
        serde(default, with = "crate::pipeline::serde_pairs")
    )]
    pub cross_class: BTreeMap<SlotRef, crate::pipeline::sym::SampledType>,
    /// The runtime sampler slots of a `RuntimeFact::AnisotropicSampler` that the shape binds to a
    /// sampler whose word states anisotropy (`SamplerWord::ANISOTROPIC`,
    /// `LoweringInputs::anisotropic_samplers`): the variant whose function takes Metal's
    /// anisotropic filter, emulated, at each one's samples. The plan of a pipeline compiled under
    /// it states it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub anisotropic_samplers: BTreeSet<SlotRef>,
    /// The group of the render pipeline's color attachments the variant draws, where the device
    /// draws them in groups (`crate::pipeline::ColorGroup`, `GraphicsState::color_group`): its
    /// fragment function writes that group's attachments alone, at locations `0..`
    /// (`LoweringInputs::color_group`), and its graphics state holds them alone. The key's other
    /// choices of colors (`zeroed_colors`, `dropped_colors`, `interlocked_colors`,
    /// `extended_shadows`) then name the group's locations, as a bind plan of the variant over its
    /// group's part of the pass asks for them; no bind plan sets the group itself. 0, the first
    /// group, by default; a group past the pipeline's last is no plan's (an invariant). Not a
    /// lowering input of its own: the render pipeline's compile derives the fragment function's
    /// from it, the descriptor and the profile.
    #[cfg_attr(feature = "serde", serde(default))]
    pub color_group: u32,
}

impl VariantKey {
    /// The empty key: no variant is needed.
    pub fn new() -> VariantKey {
        VariantKey::default()
    }

    /// This key in the large-buffer form, for a pipeline of `dependencies` bound as `shape`: with
    /// every slot a large buffer would move to its device address ([`psb_slots`](Self::psb_slots):
    /// a `RuntimeFact::BufferRangeLimit` the pipeline honours by a variant) at its device address
    /// whatever buffer is bound there, but those `shape` binds without one (inline bytes, or a
    /// buffer without a device address). A pipeline of this key runs a dispatch or draw of any
    /// buffer lengths at those slots ([`covers`](Self::covers)), so one compiled before the buffers
    /// are known (an indirect command the GPU may write) is the one that runs. Every access inside
    /// a slot's window reads and writes as in the form the bound buffers ask for; outside it, the
    /// slot has a large buffer's containment. On a device without device addresses the fact is a
    /// plan-time check, and the key is unchanged.
    pub fn large_buffer_form(
        &self,
        dependencies: &[crate::pipeline::RuntimeDependency],
        shape: &crate::bind::BindShape,
    ) -> VariantKey {
        use crate::bind::BufferShape;
        use crate::pipeline::{Mechanism, RuntimeFact};
        let addressless = |slot: &SlotRef| {
            let bound = shape
                .stages
                .get(&slot.stage)
                .and_then(|stage| stage.buffers.get(&slot.index));
            matches!(
                bound,
                Some(
                    BufferShape::Bytes { .. }
                        | BufferShape::Buffer {
                            device_address: false,
                            ..
                        }
                )
            )
        };
        let mut out = self.clone();
        out.psb_slots.extend(
            dependencies
                .iter()
                .filter_map(|dependency| match dependency.fact {
                    RuntimeFact::BufferRangeLimit { slot }
                        if dependency.mechanism == Mechanism::Variant && !addressless(&slot) =>
                    {
                        Some(slot)
                    }
                    _ => None,
                }),
        );
        out
    }

    /// Whether a pipeline compiled under this key runs a dispatch or draw whose plan settles on
    /// `settled`: the same key but for device-address slots this one adds, as
    /// [`large_buffer_form`](Self::large_buffer_form) adds them.
    pub fn covers(&self, settled: &VariantKey) -> bool {
        let mut widened = settled.clone();
        widened.psb_slots.extend(self.psb_slots.iter().copied());
        widened == *self
    }

    /// Whether nothing needs a variant (the common path).
    pub fn is_empty(&self) -> bool {
        self.storage_images.is_empty()
            && self.psb_slots.is_empty()
            && self.pulled_vertex_buffers.is_empty()
            && !self.emulated_simdgroups
            && !self.dropped_vertex_layer
            && self.storage_view_slots.is_empty()
            && self.srgb_textures.is_empty()
            && self.snapshot_textures.is_empty()
            && self.ycbcr_in_place.is_empty()
            && self.ycbcr_planes_sampled.is_empty()
            && self.address_maps.is_empty()
            && !self.flat_grid
            && !self.replay_bindings
            && !self.merged_draws
            && self.amplification_draws.is_none()
            && self.viewport_compensation.is_none()
            && !self.folded_groups
            && self.extended_range.is_empty()
            && self.zeroed_colors.is_empty()
            && self.dropped_colors.is_empty()
            && self.kept_imageblock_bytes.is_none()
            && !self.depth_clamp_range
            && !self.rotated_weights
            && !self.mesh_last_first
            && self.swizzled_storage.is_empty()
            && !self.addressed_function_tables
            && self.interlocked_colors.is_empty()
            && !self.read_only_tests
            && !self.shader_tests
            && !self.shader_stencil
            && !self.window_depth_varying
            && self.cubic_samplers.is_empty()
            && self.primitive_id_indices.is_none()
            && !self.quad_primitive_ids
            && !self.private_list
            && self.extended_shadows.is_empty()
            && self.cross_class.is_empty()
            && self.anisotropic_samplers.is_empty()
            && self.color_group == 0
    }

    /// The lowering inputs of the function of `stage` under this key: its storage-image choices,
    /// device-address slots, kept storage views, raw sRGB slots, cubic and anisotropic samplers,
    /// for the kernel the emulated simdgroups, the flat grid, the folded groups and the replay's
    /// bindings, for the vertex and fragment functions the draws of one amplification each, for
    /// the fragment function the colors it writes zero to and those it drops, its depth's clamp
    /// range and whether its draws' tests are read-only, for a tile function the imageblock bytes
    /// its pass keeps, and for the vertex function the
    /// dropped `Layer` output, the viewport compensation and the window depth
    /// (a vertex function's projection is
    /// derived from the vertex descriptor and
    /// [`pulled_vertex_buffers`](VariantKey::pulled_vertex_buffers) when the pipeline compiles,
    /// `Device::compile_render_variant`). A key carries no stage-input descriptor: a kernel of a
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
        out.flat_grid = self.flat_grid && stage == crate::pipeline::sym::Stage::Kernel;
        out.folded_groups = self.folded_groups && stage == crate::pipeline::sym::Stage::Kernel;
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
        out.dropped_vertex_layer =
            self.dropped_vertex_layer && stage == crate::pipeline::sym::Stage::Vertex;
        out.amplification_draws = self
            .amplification_draws
            .filter(|_| stage != crate::pipeline::sym::Stage::Kernel);
        out.viewport_compensation = self
            .viewport_compensation
            .filter(|_| stage == crate::pipeline::sym::Stage::Vertex);
        out.storage_view_slots = self
            .storage_view_slots
            .iter()
            .filter(|s| s.stage == stage)
            .copied()
            .collect();
        out.srgb_textures = self
            .srgb_textures
            .iter()
            .filter(|s| s.stage == stage)
            .copied()
            .collect();
        out.snapshot_textures = self
            .snapshot_textures
            .iter()
            .filter(|(s, _)| s.stage == stage)
            .map(|(s, form)| (*s, *form))
            .collect();
        out.ycbcr_in_place = self
            .ycbcr_in_place
            .iter()
            .filter(|s| s.stage == stage)
            .copied()
            .collect();
        out.ycbcr_planes_sampled = self
            .ycbcr_planes_sampled
            .iter()
            .filter(|(s, _)| s.stage == stage)
            .map(|(s, sampler)| (*s, *sampler))
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
            out.read_only_tests = self.read_only_tests;
            out.shader_tests = self.shader_tests;
            out.shader_stencil = self.shader_stencil;
            out.extended_shadows = self.extended_shadows.clone();
        }
        out.window_depth_varying =
            self.window_depth_varying && stage == crate::pipeline::sym::Stage::Vertex;
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
