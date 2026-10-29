//! `LoweringInputs`: the public, canonical projection of a function's lowering inputs.
//!
//! These are the only per-function inputs besides the function, its constant values, and the
//! profile: the vertex-fetch projection, storage-image choices, the slots a `BufferRangeLimit`
//! variant moves to device addresses, whether a kernel's simdgroups are emulated for dispatches
//! with a partial simdgroup, whether a vertex function's `Layer` output is dropped for a pass of
//! one layer, a kernel's stage-input descriptor, the read-only slots Profile F's per-stage
//! storage-buffer fallback keeps as storage views, a fragment function's signed-integer color
//! attachments, and whether a kernel compiles as a device-replay variant, the texture slots bound
//! in an sRGB format whose raw codes the function reads, and how a fragment function's color
//! outputs reach attachments whose formats convert them. Local size is never one. Every field is
//! part of the `FunctionCompileKey`.

use std::collections::{BTreeMap, BTreeSet};

use crate::metal;
use crate::pipeline::descriptor::{
    StageInputDescriptor, TessellationDescriptor, VertexAttributeDescriptor,
    VertexBufferLayoutDescriptor,
};
use crate::pipeline::sym::{SampledType, SlotRef, SnapshotForm, VaryingKey};
use crate::vk;

/// Per-function lowering inputs, public projection.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LoweringInputs {
    /// How each vertex attribute is fetched; `None` for non-vertex functions.
    pub vertex_fetch: Option<VertexFetchProjection>,
    /// The representation chosen for each storage texture slot whose format is a variant.
    #[cfg_attr(feature = "serde", serde(with = "crate::pipeline::serde_pairs"))]
    pub storage_images: BTreeMap<SlotRef, StorageImageChoice>,
    /// Slots moved to `BufferForm::Psb` by a `BufferRangeLimit` variant.
    pub psb_slots: BTreeSet<SlotRef>,
    /// A kernel's simdgroups run in the emulated tier whatever tier the profile's subgroups give
    /// them (`VariantKey::emulated_simdgroups`): the variant a dispatch with a partial simdgroup
    /// runs on. A kernel compiled under it depends on `RuntimeFact::EmulatedSimdgroups`. Only a
    /// kernel has simdgroups this emulates; a vertex or fragment function compiles as without it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub emulated_simdgroups: bool,
    /// A vertex function does not write its `[[render_target_array_index]]` output
    /// (`VariantKey::dropped_vertex_layer`): the variant a pass of at most one layer runs on, where
    /// every index renders slice 0. A vertex function that writes the output depends on
    /// `RuntimeFact::RenderTargetArrayLength`, which states whether it was dropped; any other
    /// function compiles as without it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub dropped_vertex_layer: bool,
    /// A kernel's stage-input descriptor (`MTLComputePipelineDescriptor.stageInputDescriptor`): how
    /// each `[[stage_in]]` attribute is fetched from its buffer. A kernel reading a stage-in
    /// attribute needs it (`NeedsInput(StageInputDescriptor)` without it);
    /// `Device::compile_compute_pipeline` compiles a `Function` kernel under the pipeline's, and
    /// a kernel compiled on its own for such a pipeline compiles under the same descriptor. The
    /// key takes it canonicalized ([`StageInputDescriptor::canonical`]); any other function
    /// compiles as without it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub stage_input: Option<StageInputDescriptor>,
    /// Read-only slots that Profile F's per-stage storage-buffer fallback keeps as storage views
    /// (`VariantKey::storage_view_slots`): the shape binds them where a uniform view's range may
    /// exceed `maxUniformBufferRange`. The fallback views other read-only slots as uniform buffers
    /// instead, and the function depends on `RuntimeFact::UniformViewRange` with
    /// `UniformViewChoice::Kept` for each of them; a slot the fallback cannot consider is ignored.
    #[cfg_attr(feature = "serde", serde(default))]
    pub storage_view_slots: BTreeSet<SlotRef>,
    /// The color attachments a fragment function writes that its render pipeline gives a
    /// signed-integer pixel format (`MTLRenderPipelineColorAttachmentDescriptor.pixelFormat`). A
    /// 32-bit integer color output to one is declared `int`, because Vulkan writes a signed-integer
    /// attachment only from a signed output; every other integer output is declared `uint`.
    /// `Device::compile_render_pipeline` sets it for a `Function` fragment stage, and a fragment
    /// compiled on its own for such a pipeline compiles under the same set; any other function
    /// compiles as without it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub signed_color_attachments: BTreeSet<u32>,
    /// A tile function's color attachments, the pixel format of each by color index
    /// (`MTLTileRenderPipelineDescriptor.colorAttachments[n].pixelFormat`). `Some` compiles a
    /// kernel as a tile function, whose implicit imageblock holds those attachments' pixels;
    /// `Device::compile_tile_pipeline` sets it, and a tile function compiled on its own for such a
    /// pipeline compiles under the same map. `None` for every other function: a kernel that reaches
    /// the implicit imageblock then is `NeedsInput(TileAttachments)`.
    #[cfg_attr(feature = "serde", serde(default))]
    pub tile_attachments: Option<BTreeMap<u32, metal::PixelFormat>>,
    /// The fragment function's render pipeline rasterizes more than one sample
    /// (`rasterSampleCount` above 1 with rasterization on). Metal reads `[[position]]` and
    /// center-sampled varyings at the pixel center in every invocation, so under it a fragment
    /// function reads `[[position]]`'s x and y as `floor(FragCoord.xy) + 0.5`, and each
    /// center-sampled varying through `InterpolateAtOffset` at offset 0 on a device with
    /// interpolation functions whose offset reads from the center
    /// (`VkProperty::InterpolateAtOffsetCentered`), or as it is on a device that reads it at the
    /// pixel center itself, as Metal does (`VkProperty::MetalCenterVaryings`): the pixel center
    /// whatever samples its invocation shades (`StageIo::{multisampled, varyings_at_pixel}`). A
    /// function that reads its color attachments (`framebuffer_fetch`) reads, and in the storage
    /// form writes, each one's multisampled image at its `SampleId`, running once per covered
    /// sample, as Metal does. `Device::compile_render_pipeline` sets it for a `Function` fragment
    /// stage, and a fragment compiled on its own for such a pipeline compiles under it: link
    /// refuses one compiled without it whose sample-rate invocations would read either elsewhere,
    /// or that reads its color attachments, as `NeedsInput(Multisampled)`, and one compiled under
    /// it that reads its color attachments into a pipeline of one sample likewise. Any other
    /// function compiles as without it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub multisampled: bool,
    /// A kernel compiles as a device-replay variant of an indirect-command template: every buffer
    /// slot it reaches is a device-address slot (`BufferForm::Psb`), read through its entry of the
    /// root table at `PushField::RootTableAddress`, and a slot's Metal address (a pointer compare's
    /// or a `memmove`'s) is its entry's `guest + bind_offset`, never a `SlotGuestBase` push field.
    /// So everything the function reads of its buffers is in the root table, which the device
    /// replay's preprocess writes per replayed command from the command's record (`bind::replay`).
    /// A vertex or fragment function compiles so too, for a replayed draw: its buffers are the
    /// command's, in the slot table at its own stage's `RootTableAddress`. It needs device
    /// addresses (`bufferDeviceAddress` and `shaderInt64`: `GpuEncodedIndirectCommands` names
    /// them otherwise); a tile, object or mesh function, a kernel with a stage-input descriptor,
    /// and a vertex function that fetches attributes of a vertex descriptor are
    /// `Unsupported{GpuEncodedIndirectCommands, NotImplemented}` under it (a replayed command is a
    /// compute dispatch with no stage-in region, or a draw whose buffers no fetch reads).
    #[cfg_attr(feature = "serde", serde(default))]
    pub replay_bindings: bool,
    /// With `replay_bindings`, the vertex function of a draw replay merged into one draw
    /// (`bind::replay`'s `merged.rs`, `DrawReplay::merged`): it reads its header's device address
    /// at `PushField::MergedDrawAddress` in place of `RootTableAddress`, finds the command its
    /// corner (`InstanceIndex` and `VertexIndex`) falls in, and runs that command's vertex: its
    /// `[[vertex_id]]`, `[[instance_id]]`, `[[base_vertex]]` and `[[base_instance]]` are the
    /// command's draw's, and its buffers are in the command's vertex slot table. A fragment
    /// function under it reads no buffer and no `[[primitive_id]]`, which a merged draw does not
    /// keep per command: one that does is `Unsupported{GpuEncodedIndirectCommands,
    /// NotImplemented}`, and so is a vertex function that writes the primitive ID's varying or
    /// draws amplifications as draws of their own. The vertex function
    /// states `RuntimeFact::MergedDraws`. Any other stage compiles as without it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub merged_draws: bool,
    /// The render pipeline state a fragment function's framebuffer fetch takes its form from, on
    /// a device that orders a fetch only through the ordered pixel interlock (no
    /// `rasterizationOrderColorAttachmentAccess`, with `fragmentShaderPixelInterlock`): there each
    /// color attachment the function reads is read, and written, as a storage image inside the
    /// interlock where the pipeline's state lets that form be exact, and otherwise is refused.
    /// `Device::compile_render_pipeline`
    /// sets it for a `Function` fragment stage that declares a `[[color(n)]]` input on such a
    /// device, and a fragment compiled on its own for such a pipeline compiles under the same
    /// state; without it such a function is `NeedsInput(FramebufferFetch)`. On a device that orders
    /// fetch through rasterization-order input attachments, the state of each extended-range
    /// attachment the function writes where the pipeline blends it, whose blend the function
    /// computes over its input attachment (its depth, coverage and amplification bits unset,
    /// which that blend does not read); without it such a function is
    /// `NeedsInput(FramebufferFetch)` there too. `None` on every other device and for every other
    /// function.
    #[cfg_attr(feature = "serde", serde(default))]
    pub framebuffer_fetch: Option<FramebufferFetchInputs>,
    /// Texture slots bound to an 8-bit sRGB texture the device cannot sample, which the runtime
    /// holds as the format's UNORM twin, whose views read the codes (`VariantKey::srgb_textures`,
    /// `RuntimeFact::SrgbTexture`): each `half` or `float` access of such a slot decodes what its
    /// own binding reads (one texel's codes, or the device's filter of them; a bicubic sample's
    /// taps) by the MSL transfer through the device's `Pow`, recording
    /// `PrecisionNote::SrgbTexelDecode`. A slot of no such access is ignored.
    #[cfg_attr(feature = "serde", serde(default))]
    pub srgb_textures: BTreeSet<SlotRef>,
    /// Fragment texture slots of `StageInterface::snapshot_textures` the draw reads through a
    /// snapshot (`VariantKey::snapshot_textures`), each with its snapshot's form, drawn twice
    /// (`PushField::SnapshotPhase`). The entry opens with the ordered pixel interlock's critical
    /// section, in which, in the draw that takes the snapshot, a non-helper invocation copies
    /// each slot's texel at its pixel into the slot's snapshot (`ResourceKind::SnapshotTexels`),
    /// in the slot's form; that draw then returns. In the draw that
    /// reads it, the critical section is empty, and every read of the slot reads the snapshot, or
    /// nothing in a helper invocation (`lower::fragment`'s `snapshot`). The binder names only slots
    /// of that interface fact (`bind::graphics`); for another, the reads are not Metal's.
    #[cfg_attr(
        feature = "serde",
        serde(default, with = "crate::pipeline::serde_pairs")
    )]
    pub snapshot_textures: BTreeMap<SlotRef, SnapshotForm>,
    /// Texture slots bound to a two-plane YCbCr texture whose planes are read in place
    /// (`VariantKey::ycbcr_in_place`, `RuntimeFact::YCbCrInPlace`): each is declared as the
    /// planes' uniform texel buffer, and its samples, reads and size queries read the codes and
    /// convert them as Metal does (`bind::YCbCrInPlace`). Ignored on a profile that does not read
    /// planes in place (`ProfileFields::ycbcr_in_place`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub ycbcr_in_place: BTreeSet<SlotRef>,
    /// Slots of [`ycbcr_in_place`](Self::ycbcr_in_place) bound to a texture whose planes are also
    /// held as an image the device's own sampler Y′CbCr conversion samples, each with that
    /// conversion (`VariantKey::ycbcr_planes_sampled`): each is declared as a combined image
    /// sampler whose immutable sampler is the conversion (`ResourceKind::YCbCrPlanes`), and its
    /// samples are the device's at level 0 (`bind::YCbCrPlanesSampler`); its size queries read the
    /// push field as the in-place form's do.
    #[cfg_attr(
        feature = "serde",
        serde(default, with = "crate::pipeline::serde_pairs")
    )]
    pub ycbcr_planes_sampled: BTreeMap<SlotRef, crate::bind::YCbCrPlanesSampler>,
    /// The fragment function's render pipeline has a vertex function that writes
    /// `[[viewport_array_index]]`. That function stores `ViewportIndex` as 0 for an index at or
    /// past the draw's viewport count, where Metal draws through viewport 0 and its fragment
    /// function still reads the index as written, so it also writes the index to a flat varying
    /// (`IoRole::ViewportIndex`). Under this field a fragment function reads its
    /// `[[viewport_array_index]]` from that varying; without it, from `ViewportIndex` (0 when no
    /// stage writes the index; a constant 0 on a device that cannot declare it,
    /// `StageIo::zero_indices`). `Device::compile_render_pipeline` sets it for a `Function`
    /// fragment stage that reads `[[viewport_array_index]]` under such a vertex function, and a
    /// fragment compiled on its own for such a pipeline compiles under it: link refuses a
    /// fragment function compiled for the other kind of vertex function as
    /// `NeedsInput(ViewportIndex)`. Any other function compiles as without it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub viewport_index_varying: bool,
    /// A stage's `maxCallStackDepth` when it is not Metal's default, 1 (`None`): it sizes the
    /// call stack of the stage's recursion through visible functions. A pipeline compiles each
    /// `Function` stage under its own: a compute kernel under
    /// `ComputePipelineDescriptor::max_call_stack_depth`, a tile function under
    /// `TileRenderPipelineDescriptor::max_call_stack_depth`, and a render pipeline's vertex and
    /// fragment functions under `RenderPipelineDescriptor::max_vertex_call_stack_depth` and
    /// `max_fragment_call_stack_depth`. Any other function compiles as without it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub max_call_stack_depth: Option<u32>,
    /// The kernel runs its dispatch's grid flat (`VariantKey::flat_grid`): the dispatch is any
    /// number of threadgroups of the variant's local size, whose invocations count the grid's
    /// threads in order (`x` fastest) from the `threads_per_grid` of the 32-byte region record at
    /// `PushField::ReplayRegionAddress`, each taking the thread position it counts to and returning
    /// early past the grid's last thread. So one variant runs every grid and threadgroup size of
    /// its pipeline, which only a kernel that does not observe its threadgroup partition
    /// (`KernelFacts::partition_oblivious`) allows: it reads no dispatch builtin but
    /// `thread_position_in_grid` and `threads_per_grid`. Any other function is
    /// `Unsupported{GpuEncodedIndirectCommands, NotRepresentable}` under it. Under
    /// `replay_bindings` the replay's preprocess writes the record per command
    /// (`ReplayVariant::flat`); otherwise the kernel binds as any other, and a preprocess writes
    /// the record beside its dispatch's group counts. It needs device addresses, and refuses a
    /// stage-input region, a tile function and `emulated_simdgroups`
    /// (`Unsupported{GpuEncodedIndirectCommands, NotImplemented}`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub flat_grid: bool,
    /// The fragment function's render pipeline has a vertex function that writes
    /// `[[render_target_array_index]]`. That function stores `Layer` as 0 for an index at or past
    /// the render pass's `renderTargetArrayLength`, where Metal renders the pass's first layer, and
    /// its variant for a pass of one layer writes no `Layer`, while Metal's fragment function reads
    /// the index as written in both; so in every variant it also writes the index to a flat varying
    /// (`IoRole::RenderTargetArrayIndex`). Under this field a fragment function reads its
    /// `[[render_target_array_index]]` from that varying; without it, from `Layer` (0 when no stage
    /// writes the index; a constant 0 on a device that cannot declare it,
    /// `StageIo::zero_indices`). `Device::compile_render_pipeline` sets it for a `Function`
    /// fragment stage that reads `[[render_target_array_index]]` under such a vertex function,
    /// and a fragment compiled on its own for such a pipeline compiles under it: link refuses a
    /// fragment function compiled for the other kind of vertex function as
    /// `NeedsInput(RenderTargetArrayIndex)`. Any other function compiles as without it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub render_target_array_index_varying: bool,
    /// The function's render pipeline carries Metal's `[[primitive_id]]` from its vertex function
    /// to its fragment function through a flat `uint` varying (`IoRole::PrimitiveId`), which a
    /// fragment function reads as its `[[primitive_id]]`: on a device that cannot declare a
    /// fragment `PrimitiveId` (neither `geometryShader` nor `tessellationShader`), and behind a
    /// post-tessellation vertex function emulated on a device without `tessellationShader`
    /// (`LoweringInputs::tessellation`), where Vulkan's `PrimitiveId` would count the
    /// tessellator's triangles.
    ///
    /// Under this field a vertex function writes the varying. A plain one writes it at each
    /// return: `(VertexIndex - first) / n` in wrapping `u32` arithmetic, `first` the draw's first
    /// vertex (Vulkan's base vertex of a draw without indices, read as `[[base_vertex]]` is) and
    /// `n` its vertices per primitive, `PushField::VerticesPerPrimitive` (3 for a triangle list,
    /// 2 for a line list, 1 for a strip or points). A primitive's flat inputs read its provoking
    /// vertex, its first in a pipeline of Vulkan's default provoking vertex, at stream position
    /// `n · i` of a list's primitive `i` and `i` of a strip's, so every fragment reads its
    /// primitive's index in its instance, Metal's
    /// `[[primitive_id]]` (Metal probes on the M3 Max, macOS 27.0.1). An emulated
    /// post-tessellation one writes its patch's position in the draw, `patchStart + i`, at every
    /// vertex of the patch: Metal's primitive id behind the tessellator (measured on the M3 Max,
    /// in every instance, with or without a patch index buffer). A post-tessellation vertex
    /// function drawn through the device's tessellation stages is refused under it
    /// (`NotImplemented(Tessellation)`).
    ///
    /// A fragment function reads `[[primitive_id]]` from the varying under this field, as it does
    /// without it on a device that cannot declare a fragment `PrimitiveId`.
    /// `Device::compile_render_pipeline` sets it for a `Function` vertex stage whose pipeline's
    /// fragment function reads `[[primitive_id]]`, on a device that cannot declare it or behind an
    /// emulated post-tessellation vertex function, and for such a `Function` fragment stage on a
    /// device that declares it. Link refuses a vertex function compiled without it behind a
    /// fragment function that reads the varying, and a fragment function that reads Vulkan's
    /// `PrimitiveId` behind a vertex function that writes the varying, as
    /// `NeedsInput(PrimitiveId)`. Any other function compiles as without it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub primitive_id_varying: bool,
    /// Under `primitive_id_varying`, the de-indexed form of an indexed draw of this index type
    /// (`VariantKey::primitive_id_indices`): the draw runs `index count` vertices from vertex 0
    /// without indices, so its stream position `p = VertexIndex` is the position the primitive
    /// ID divides, `(VertexIndex - 0) / n`. The vertex reads its raw index at
    /// `PushField::CaptureIndexAddress + p · size` and runs `raw + base vertex` from
    /// `PushField::CaptureLayout` `[0, index count, base instance, base vertex]`: `[[vertex_id]]`,
    /// `[[base_vertex]]` and every per-vertex attribute, all pulled, read that vertex. Refused
    /// `NotImplemented(PrimitiveId)`: a vertex function that writes device memory, which would run
    /// once per index rather than once per vertex Metal shades, and a hand-built projection
    /// that fetches an attribute in fixed function; on a device without device addresses it
    /// needs them. Any other function compiles as without it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub primitive_id_indices: Option<crate::metal::IndexType>,
    /// The draw lists Metal's private quads or quad strip as two triangles a quad, the quad's
    /// triangles adjacent (`VariantKey::quad_primitive_ids`): a fragment function's
    /// `[[primitive_id]]` read from Vulkan's `PrimitiveId` is that value shifted right by one, the
    /// quad's, as Metal counts one primitive a quad. The function's pipeline states it
    /// (`RuntimeFact::QuadPrimitiveIds`). Any other function, and one that reads the primitive
    /// ID's varying (`primitive_id_varying`), compiles as without it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub quad_primitive_ids: bool,
    /// The draw is the list a consumer writes for one of Metal's private primitive types (a line
    /// loop, a triangle fan, quads, a quad strip), drawn without indices in the pulled list form
    /// (`VariantKey::private_list`): `list length` vertices from vertex `first`, Vulkan's first
    /// vertex (read as the primitive ID's `first` is: the `BaseVertex` builtin with
    /// `shaderDrawParameters`, lane 0 of `PushField::BaseVertexInstance` without). The vertex at
    /// `VertexIndex` reads its raw index, a `u32`, at `PushField::CaptureIndexAddress +
    /// 4 · VertexIndex`, and Metal's base vertex at `PushField::CaptureIndexAddress + 4 · (first -
    /// 1)`, the word before the list's first; it runs `raw + base vertex` (wrapping as `u32`s),
    /// which `[[vertex_id]]` and every per-vertex attribute, all pulled, read, and
    /// `[[base_vertex]]` reads that word. So each list position runs the vertex function once,
    /// as Metal runs it for these types (once per vertex of its list: Metal probes on the M3 Max,
    /// macOS 27.0.1), whatever vertex reuse the device does, and any `u32` vertex start or base
    /// vertex is drawn: a vertex function that writes memory, or the primitive ID's varying
    /// (`primitive_id_varying`, its `(VertexIndex - first) / n` the list primitive's), runs as
    /// Metal's. Refused on a device without device addresses (`Feature::PrivatePrimitives`). The
    /// function's pipeline states it (`RuntimeFact::PrivateList`). Any other function compiles
    /// as without it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub private_list: bool,
    /// The function's render pipeline draws each vertex amplification as a draw of its own
    /// (`VariantKey::amplification_draws`), where Vulkan multiview cannot draw the encoder's
    /// mappings as Metal renders them. Under it a vertex or fragment function reads its
    /// `[[amplification_id]]` from `PushField::AmplificationId`, which the binder writes before
    /// each of those draws, not from `ViewIndex`. A vertex function adds the amplification's render
    /// target slice offset (`PushField::AmplificationLayerOffset`) to the slice index it writes (0
    /// where it writes none) and stores `Layer` as that sum where it is below the pass's length and
    /// 0 where it is not, only under [`AmplificationDraws::layered`]; it adds the viewport offset
    /// (`PushField::AmplificationViewportOffset`) to its viewport index and stores `ViewportIndex`
    /// as that sum below the draw's viewport count and 0 at or past it, where it writes an index of
    /// its own or under [`AmplificationDraws::viewports`]. Where the device cannot store either
    /// builtin from a vertex function, it stores the value to a flat varying the synthesized
    /// geometry stages store the builtin from (`IoRole::SelectedLayer`,
    /// `IoRole::SelectedViewport`), and on a device without `multiViewport` it stores no viewport
    /// at all. The raw index varyings it writes hold the sums. A fragment function reads its `[[render_target_array_index]]` and
    /// `[[viewport_array_index]]` from those varyings where its vertex function writes the index
    /// (`render_target_array_index_varying`, `viewport_index_varying`), and otherwise as the
    /// offsets themselves, from the push fields. The vertex function states
    /// `RuntimeFact::AmplificationDraws`. Any other function compiles as without it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub amplification_draws: Option<AmplificationDraws>,
    /// How each color attachment the fragment function writes takes each channel of its float
    /// output (red, green, blue, alpha; `None` for a channel the format lacks), where the
    /// attachment's format converts it (a normalized, sRGB or small float format): Metal's
    /// [`ChannelWrite`] rule where the attachment stores the output as written, which the device's
    /// conversion stores, the function recording `PrecisionNote::DeviceColorConversion` where the
    /// device is not measured to convert it as Metal does; [`ChannelWrite::Device`] where the
    /// device's conversion of another value stays (blending other than factors one and zero, the
    /// alpha alpha-to-coverage reads, or a format whose conversion Metal was not measured with),
    /// recorded alike; and a half-float channel's rule under
    /// `TextureWriteRoundingMode::NearestEven`, which the function rounds itself. On a device that
    /// stores Metal's codes (`VkProperty::MetalColorWrites`) only the half-float channels. A rule
    /// of Metal's the function's lowering on the device does not tell from the device's
    /// conversion (an 8-bit unorm channel's on a device not measured to store it as Metal does,
    /// say) is named [`ChannelWrite::Device`], so pipelines that differ only there, as a blend
    /// sweep over one fragment function does, compile it once.
    /// `Device::compile_render_pipeline` sets it for a `Function` fragment stage from the
    /// pipeline's color attachments, blend states and alpha-to-coverage, and a fragment compiled
    /// on its own for such a pipeline compiles under the same map: link refuses one compiled under
    /// another as `NeedsInput(ColorWrites)`. Empty for any other function.
    #[cfg_attr(feature = "serde", serde(default))]
    pub color_writes: BTreeMap<u32, [Option<ChannelWrite>; 4]>,
    /// The varyings the fragment function's mesh render pipeline's mesh function writes per
    /// primitive: its primitive data (`IoRole::PrimitiveVarying`) and its raw render target slice
    /// and viewport indices (`IoRole::RenderTargetArrayIndex`, `IoRole::ViewportIndex`). A
    /// fragment function reads an input of such a key per primitive (`PerPrimitiveEXT`, flat,
    /// whatever interpolation it states: the value is the primitive's), and lists it among
    /// `StageIo::primitive_inputs`. `Device::compile_mesh_pipeline` sets it for a `Function`
    /// fragment stage from the pipeline's mesh function, and a fragment compiled on its own for
    /// such a pipeline compiles under the same keys: link refuses one compiled for another
    /// pipeline as `NeedsInput(PrimitiveVaryings)`. Empty for a render pipeline's fragment
    /// function and for any other function.
    #[cfg_attr(feature = "serde", serde(default))]
    pub primitive_varyings: BTreeSet<VaryingKey>,
    /// The fragment function's stage inputs (their user varying keys) that its render pipeline's
    /// vertex function writes no output of and that Metal's linking drops: no instruction the
    /// linking keeps reads them, with every rule it keeps beyond Metal on, and their values,
    /// followed forward, reach no kept output, effect, branch condition or memory
    /// (`reflection`'s module `linked`). Metal makes such a pipeline; Vulkan requires every
    /// fragment input the vertex stage declares as an output, so the function declares no input for
    /// them and reads each as zero, which reaches nothing kept. `Device::compile_render_pipeline`
    /// sets it for a `Function` fragment stage from the pipeline's vertex function and color
    /// attachments; empty for any other function.
    #[cfg_attr(feature = "serde", serde(default))]
    pub absent_varyings: BTreeSet<VaryingKey>,
    /// The vertex function's position is compensated for a viewport Vulkan's cannot be
    /// (`VariantKey::viewport_compensation`): a negative width, or, under
    /// [`ViewportCompensation::depth`], a depth range past `[0, 1]` on a device that does not take
    /// one. The function reads the draw's `[sx, a, b, clip]` from `PushField::ViewportCompensation`
    /// (four `f32`s' bits) and writes `Position` as `(sx x, y, a w + b z, w)`; under `depth` it
    /// also writes two `ClipDistance` elements after its own, `z` and `w - z` where `clip` is 1
    /// (Metal's near and far planes) and 1 where it is 0. Under
    /// [`ViewportCompensation::point_guard`] it reads the draw's `[gx, gy]` from
    /// `PushField::PointGuard` (two `f32`s' bits), writes `Position` as `(gx sx x, gy y, ..)`, and
    /// writes four `ClipDistance` elements after the others: `1.5 w - x`, `1.5 w + x`, `1.5 w - y`
    /// and `1.5 w + y`, of the function's own `x` and `y`. It states
    /// `RuntimeFact::ViewportCompensation`. A post-tessellation vertex function is compensated
    /// as a vertex function is; any other function compiles as without it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub viewport_compensation: Option<ViewportCompensation>,
    /// The kernel runs a threadgroup dispatch whose group counts are GPU data
    /// (`dispatchThreadgroupsWithIndirectBuffer:`) as one Vulkan dispatch within the device's
    /// group-count limits, whatever the counts (`VariantKey::folded_groups`): Vulkan workgroup
    /// `(wx, wy, wz)` of a dispatch of `(nx, ny, nz)` workgroups is Metal's threadgroup number `(wz
    /// · ny + wy) · nx + wx`, counted over Metal's group counts `x` fastest, in 64 bits. The group
    /// counts are `threads_per_grid / threads_per_threadgroup` of the 32-byte region record at
    /// `PushField::ReplayRegionAddress` (the `DispatchRegion` value: base 0, the packed threadgroup
    /// size, and `threads_per_grid`), which a preprocess writes beside the dispatch's group counts,
    /// and every grid builtin reads its region from that record (`MultiRegion` 1). A workgroup past
    /// Metal's last threadgroup returns before the body, as a whole. So a kernel that observes its
    /// threadgroup partition runs any group counts on one variant of the call's threadgroup size.
    /// It needs device addresses, and refuses a stage-input region, a tile function, `flat_grid`
    /// and `replay_bindings` (`Unsupported{GpuEncodedIndirectCommands, NotImplemented}`). Any other
    /// function compiles as without it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub folded_groups: bool,
    /// Texture slots bound in one of Apple's extended-range formats
    /// (`VariantKey::extended_range`): each such slot is declared as an image of the format's raw
    /// unsigned texels (`metal::format::ExtendedRange::raw_format`), whose reads decode Metal's
    /// values and whose writes encode them. A sample or gather of such a slot is
    /// `NotImplemented(Format)`.
    #[cfg_attr(
        feature = "serde",
        serde(default, with = "crate::pipeline::serde_pairs")
    )]
    pub extended_range: BTreeMap<SlotRef, metal::format::ExtendedRange>,
    /// The color attachments the fragment function's render pipeline gives one of Apple's
    /// extended-range formats, by color index ([`ExtendedRangeColor`]): each such attachment
    /// holds the format's raw texels (`metal::format::ExtendedRange::raw_format`), so the
    /// function writes it through an output variable of the raw format's unsigned components and
    /// stores the texel Metal stores for its `float` or `half` output (a `float` rounded to `half`
    /// first, nearest even, then encoded as a kernel write encodes it). A dual-source second
    /// output to such an attachment is `NotImplemented(Format)`. `Device::compile_render_pipeline`
    /// sets it for a `Function` fragment stage from the pipeline's color attachments, and a
    /// fragment compiled on its own for such a pipeline compiles under the same map: link refuses
    /// one compiled under another as `NeedsInput(ColorWrites)`. Empty for any other function.
    #[cfg_attr(feature = "serde", serde(default))]
    pub extended_range_colors: BTreeMap<u32, ExtendedRangeColor>,
    /// The color attachments the device renders through a word view
    /// (`metal::format::WordAttachment`), by color index: an attachment the pipeline writes
    /// unblended (or blended with factors one and zero, added), under a write mask of all of the
    /// format's channels or none, and that is not attachment 0 under alpha-to-coverage, of a format
    /// the device samples but does not render (one written under a mask of some of its channels
    /// is stored in the interlocked storage form instead, named in
    /// [`framebuffer_fetch`](Self::framebuffer_fetch)). The function writes such an attachment
    /// through an output variable of the word's unsigned integer, the word Metal stores for its
    /// `float` or `half` output (`ChannelWrite::HalfUnorm`'s rule per channel); a dual-source
    /// second output there is stored nowhere, as Metal's unblended write reads none, and a color
    /// input of one is `NotImplemented(Format)`. `Device::compile_render_pipeline` sets it for a
    /// `Function` fragment stage from the pipeline's color attachments, and link refuses a fragment
    /// compiled under another map as `NeedsInput(ColorWrites)`. Empty for any other function.
    #[cfg_attr(feature = "serde", serde(default))]
    pub word_colors: BTreeMap<u32, metal::format::WordAttachment>,
    /// Color attachments of the pass the fragment function's render pipeline has no pixel
    /// format for, the pass's pixel format of each by color index
    /// (`VariantKey::zeroed_colors`): plain Metal stores zero bytes into each at every sample a
    /// draw covers (Metal probes). The function writes zero to each, through an output variable of
    /// the format's numeric class at that color, at every return; the pipeline compiled under the
    /// key gives each attachment that format, written whole and unblended. A function that writes
    /// or reads such a color itself is `NotImplemented` (unmeasured). Any other function compiles
    /// as without it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub zeroed_colors: BTreeMap<u32, metal::PixelFormat>,
    /// Color attachments the fragment function's render pipeline has a pixel format for and the
    /// pass has none, by color index (`VariantKey::dropped_colors`): plain Metal drops what the
    /// function writes there (a Metal probe). The function keeps no output variable at each, its
    /// writes stored nowhere, so the pipeline writes no location without a color attachment of its
    /// subpass. Any other function compiles as without it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub dropped_colors: BTreeSet<u32>,
    /// The bytes of each cell of a tile function's explicit imageblock standing alone that its
    /// render pass keeps, fewer than the cell holds (`VariantKey::kept_imageblock_bytes`): plain
    /// Metal keeps the block's words below the pass's imageblock sample length rounded up to 8,
    /// and reads every other word as zero and drops its writes, the dispatch's own included
    /// (Metal probes). The function reads each word of a cell past these bytes as zero and writes
    /// it nowhere, whatever the access (a whole number of words). A function without such an
    /// imageblock is not lowered under it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub kept_imageblock_bytes: Option<u32>,
    /// The fragment function clamps the depth it writes to the draw's depth clamp range itself
    /// (`VariantKey::depth_clamp_range`): it reads the range's `[low, high]` from
    /// `PushField::DepthClampRange` (two `f32`s' bits) and stores its depth output as Metal does in
    /// depth clamp mode through that range, whatever the device's own clamp: a value with the sign
    /// bit set as +0 and one past 1 up to +inf as 1, then clamped to the range, but a negative NaN
    /// as 0 (a Metal probe, an M3 Max and an M4 alike). It states `RuntimeFact::DepthClampRange`.
    /// The variant a draw in depth clamp mode through a range above 0 runs, whose Vulkan depth
    /// clamp holds 0; a function that writes no depth, or any other function, compiles as without
    /// it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub depth_clamp_range: bool,
    /// The fragment function's triangles are drawn with Metal's last vertex first
    /// (`VariantKey::rotated_weights`): a triangle Metal draws as `(a, b, c)` is drawn as
    /// `(c, a, b)`, so that a first-vertex device takes its flat inputs from Metal's provoking
    /// vertex under Apple's `setProvokingVertexMode:` 1. Its `[[barycentric_coord]]` reads take
    /// Metal's weights back from the drawn triangle's first two, `(x, y)`: `y` (Metal's first,
    /// bit for bit), `1 - (x + y)`, and `1 - (y + (1 - (x + y)))`, which are Metal's second and
    /// third but for rounding (Metal probes on the Apple M3 Max, macOS 27.0.1). It states
    /// `RuntimeFact::RotatedWeights`. A function that reads no weights, or any other function,
    /// compiles as without it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub rotated_weights: bool,
    /// Storage-image texture slots bound to a swizzled texture (`VariantKey::swizzled_storage`):
    /// the function applies the bound texture's swizzle, its slot's `SpecKey::TextureSwizzle`
    /// word, to each texel it reads there, out of bounds too, after converting the texel, and
    /// undoes it on each texel it writes, before converting it: channel `c` stores the value of
    /// the first lane whose source is `c`, else lane `c`'s own (plain Metal's answer on an M4,
    /// where atomics ignore the swizzle, as here). It states `RuntimeFact::TextureSwizzle`. A
    /// `ulong` texel, an imageblock slice written to such a slot, or a read of fewer than four
    /// lanes there is `NotImplemented(TextureSwizzle)` (unmeasured). A slot the function does
    /// not read or write texels of compiles as without it.
    ///
    /// It holds too each sampled slot of `extended_range` bound to a swizzled texture, whose view
    /// holds raw codes: the function applies the swizzle to each texel it reads, samples or gathers
    /// there after decoding it, a border or out-of-bounds texel included, and a gather of a
    /// component gathers the component the swizzle names (a `Zero` or `One` 0 or 1 at every tap),
    /// as plain Metal does. And on a device without Metal's own sampler each sampled slot bound to
    /// a swizzled packed normalized texture: the function swizzles the texels its class converts,
    /// likewise.
    #[cfg_attr(feature = "serde", serde(default))]
    pub swizzled_storage: BTreeSet<SlotRef>,
    /// The function reads every visible-function table it reads by resource id through the
    /// table's device address (`VariantKey::addressed_function_tables`): the directory's entry
    /// of the id holds the address of the table's words in place of an element, and a read of
    /// word `w` is checked against the table's size word, so a table whose words pass
    /// `maxStorageBufferRange` is read whole. It declares neither the function-table array nor
    /// the snapshot, and states `RuntimeFact::FunctionTableRange` with `addressed`. It needs
    /// device addresses (`Unsupported{BufferRange}` without them). A function that reads no
    /// table by resource id compiles as without it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub addressed_function_tables: bool,
    /// Colors the fragment function's render pipeline has no attachment for, by color index:
    /// plain Metal discards what the function writes there. The function keeps no output variable
    /// at each it writes, its writes stored nowhere, so the pipeline writes no location without a
    /// color attachment (`StageIo::unattached_colors`). `Device::compile_render_pipeline` sets it
    /// for a `Function` fragment stage to the colors it writes without an attachment, but color 0
    /// under alpha to coverage, whose alpha the coverage reads; a fragment compiled on its own for
    /// such a pipeline may compile under the same set, and link refuses one whose stored-nowhere
    /// colors include an attachment of the pipeline, or color 0 under alpha to coverage
    /// (`NeedsInput(ColorWrites)`). Any other function compiles as without it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub unattached_colors: BTreeSet<u32>,
    /// The render pipeline's private sample mask, its bits below the pipeline's sample count,
    /// ANDed with its private sample coverage's, by the window pixel's place in its 2x2 quad
    /// (index `(x & 1) | (y & 1) << 1`), which the fragment function applies itself: each return
    /// ANDs the pixel's into the function's `[[sample_mask]]` output, or writes it as that output
    /// where the function writes none (`StageIo::sample_mask`). Metal applies both after the
    /// function runs, so the function still runs for the masked samples and reads the unmasked
    /// coverage, and under `[[early_fragment_tests]]` they keep only the colors.
    /// `Device::compile_render_pipeline` sets it for a `Function` fragment stage of a pipeline
    /// whose masks drop a sample, where they differ between the places of a quad, or where the
    /// function writes memory, reads `[[sample_mask]]` or declares `[[early_fragment_tests]]`;
    /// the pipeline then has no Vulkan sample mask. Any other function compiles as without it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub sample_mask: Option<[u32; 4]>,
    /// The render pipeline's private alpha test (`alphaTestEnabled`, `alphaTestFunction`), which
    /// the fragment function runs itself after its body: each return kills the invocation unless
    /// its color 0's alpha passes the comparison against the encoder's reference
    /// (`PushField::AlphaTestReference`), both as Metal compares them, 8-bit unorm codes
    /// (`StageIo::alpha_test`). `Device::compile_render_pipeline` sets it for a `Function`
    /// fragment stage of a pipeline with the test on, for every function but `Always`, which
    /// tests nothing. Any other function compiles as without it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub alpha_test: Option<crate::metal::CompareFunction>,
    /// The render pipeline's private `fragmentDepthCompareClampMask`: a fragment function's depth
    /// comparison through `[[sampler(s)]]` where bit `s` is set, or through a constexpr sampler
    /// where bit 31 is, clamps its reference to `[0, 1]`, a NaN taken as 0, before comparing, as
    /// Metal's does (`StageIo::depth_compares`); one through an argument buffer's sampler is
    /// refused under a mask other than 0. `Device::compile_render_pipeline` sets it for a
    /// `Function` fragment stage. Any other function compiles as with 0.
    #[cfg_attr(feature = "serde", serde(default))]
    pub depth_compare_clamp_mask: u32,
    /// The depth and stencil tests of the draws that run the fragment function write neither
    /// attachment and count no visibility result (`VariantKey::read_only_tests`): a function that
    /// stores a color through the interlocked storage form and declares no early tests only
    /// because it discards (`RuntimeFact::TestedStorageWrites`) is lowered with early fragment
    /// tests, which then drop what Metal's later tests drop and leave the attachments as Metal's
    /// do, and states the fact with `early_tests`. Any other function compiles as without it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub read_only_tests: bool,
    /// Only the depth test of the draws that run the fragment function can drop fragments
    /// (`VariantKey::shader_tests`): a function that stores a color through the interlocked
    /// storage form before the tests (`RuntimeFact::TestedStorageWrites`) runs the depth test
    /// itself, inside the ordered pixel interlock, against a stand-in of the depth attachment, and
    /// states `RuntimeFact::ShaderTests`. Any other function compiles as without it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub shader_tests: bool,
    /// Under `shader_tests`, the draw's stencil test can drop fragments or write too
    /// (`VariantKey::shader_stencil`): the function also runs the stencil test itself, before its
    /// depth test, against a stand-in of the stencil attachment, and states
    /// `RuntimeFact::ShaderStencilTest`.
    #[cfg_attr(feature = "serde", serde(default))]
    pub shader_stencil: bool,
    /// A vertex function's fragment function runs its draw's depth test itself at more than one
    /// sample from each sample's computed depth (`VariantKey::window_depth_varying`): the function
    /// also writes its position's window depth `z / w` to a no-perspective varying
    /// (`IoRole::WindowDepth`), which the fragment function reads. Read only by a vertex function.
    #[cfg_attr(feature = "serde", serde(default))]
    pub window_depth_varying: bool,
    /// The color attachments the fragment function's render pipeline gives a format held with
    /// its channels rotated (`metal::format::rotated_channels`, `ARGB8Unorm`), by color index:
    /// each such attachment's image holds Metal's alpha, red, green and blue in its red, green,
    /// blue and alpha, so the function writes each `float` or `half` output to it, of either
    /// dual-source index, rotated (`metal::format::to_rotated_image`), after Metal's conversion
    /// of its channels (`color_writes`). An attachment whose image holds Metal's alpha in red
    /// (`metal::format::alpha_in_red`, `A8Unorm` in `R8_UNORM`) is listed too: the rotation's red
    /// is Metal's alpha (and `alpha_in_red_colors` names it). An output of fewer than four lanes
    /// there is
    /// `NotImplemented(Format)`. `Device::compile_render_pipeline` sets it for a `Function`
    /// fragment stage from the pipeline's color attachments, and a fragment compiled on its own for
    /// such a pipeline compiles under the same set: link refuses one compiled under another as
    /// `NeedsInput(ColorWrites)`. Empty for any other function.
    #[cfg_attr(feature = "serde", serde(default))]
    pub rotated_colors: BTreeSet<u32>,
    /// The color attachments of `rotated_colors` whose image holds Metal's alpha in red
    /// (`metal::format::alpha_in_red`: `A8Unorm` in `R8_UNORM`), by color index. An output to one
    /// holds Metal's alpha in the image's alpha lane as well as its red (the image keeps only the
    /// red), so alpha to coverage through attachment 0 reads Metal's alpha as Metal's does; a
    /// color input (`[[color(n)]]`) of one reads the image's red as Metal's alpha, `(0, 0, 0, r)`,
    /// as Metal reads an `A8Unorm` attachment (probe `a8`'s uses on an M3 Max under macOS
    /// 27.0.1), through the input attachment or the interlocked storage form alike.
    /// `Device::compile_render_pipeline` sets it for a `Function` fragment stage from the
    /// pipeline's color attachments, and link refuses a fragment compiled under another set as
    /// `NeedsInput(ColorWrites)`. Empty for any other function.
    #[cfg_attr(feature = "serde", serde(default))]
    pub alpha_in_red_colors: BTreeSet<u32>,
    /// The second output (`[[color(n), index(1)]]`) the fragment function writes for an
    /// attachment of `rotated_colors` whose blend reads a value no channel of the image holds where
    /// Vulkan's factors read it (Metal's source alpha in a color factor), by color index: per
    /// channel of the image, what the output holds there ([`FactorLane`]), which link's blend
    /// reads through Vulkan's `SRC1` factors (`pipeline::rotated_blend`). The function writes it
    /// beside its first output to the attachment, in place of any second output of its own there,
    /// which Metal's blend then does not read; without a first output there it writes neither.
    /// `Device::compile_render_pipeline` sets it for a `Function` fragment stage from the
    /// pipeline's color attachments, and link refuses a fragment compiled under another map as
    /// `NeedsInput(ColorWrites)`. Empty for any other function.
    #[cfg_attr(feature = "serde", serde(default))]
    pub rotated_factors: BTreeMap<u32, [FactorLane; 4]>,
    /// A tile function's pipeline sets `threadgroupSizeMatchesTileSize`
    /// (`MTLTileRenderPipelineDescriptor`). Metal then answers the pass's tile as the function's
    /// threadgroup, whatever threads per tile the dispatch runs: its threads per threadgroup,
    /// simdgroups and quadgroups per threadgroup, and their `dispatch_` forms are the tile's, its
    /// position in the grid is `threadgroup · tile + position in threadgroup`, its threads per grid
    /// the tiles times the tile, and its thread index in the threadgroup `simdgroup · 32 + lane`
    /// (its quadgroups four consecutive of those). Under it a tile function reads those from the
    /// region's nominal threadgroup, which the binder sets to the tile, and states
    /// `RuntimeFact::TileThreadgroup` with `tile_sized`. `Device::compile_tile_pipeline` sets it
    /// from the descriptor, and a tile function compiled on its own for such a pipeline compiles
    /// under it: the pipeline refuses one compiled under the other as
    /// `InvalidDescriptor(TileThreadgroupSize)`. Any other function compiles as without it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub threadgroup_size_matches_tile_size: bool,
    /// The address-mode map of the texture bound at each texture slot a constexpr sampler
    /// samples, where it is not the identity (`VariantKey::address_maps`,
    /// `RuntimeFact::ConstexprSampled`): each constexpr sampler sampling the slot samples it with
    /// its address modes mapped, as the binder maps a runtime sampler's, its immutable sampler and
    /// every emulation that reads the modes alike. A mode the map sends where a constexpr sampler
    /// cannot address (mirror-clamp-to-edge) is refused `NotImplemented(SamplerAddressMode)`. Empty
    /// for any other function.
    #[cfg_attr(
        feature = "serde",
        serde(default, with = "crate::pipeline::serde_pairs")
    )]
    pub address_maps: BTreeMap<SlotRef, metal::sampler::AddressModeMap>,
    /// The runtime sampler slots bound to a sampler whose min or mag filter is Metal's `Cubic`
    /// (`VariantKey::cubic_samplers`, `RuntimeFact::CubicSampler`), each with the bound word's
    /// `SamplerWord::CUBIC_STATE` bits. A sample through such a slot is Metal's bicubic filter
    /// of that state, which the function emulates by fetches as it does a
    /// constexpr bicubic sampler's, refusing what that emulation refuses and a mip filter
    /// (`NotImplemented(SamplerBicubicFilter)`). A slot the function samples through without
    /// being named here filters as its word says.
    #[cfg_attr(
        feature = "serde",
        serde(default, with = "crate::pipeline::serde_pairs")
    )]
    pub cubic_samplers: BTreeMap<SlotRef, u32>,
    /// The attachments of `rotated_colors` whose blend the fragment function shapes or
    /// announces, by color index (`pipeline::rotated_blend`): one blended in a measured form that
    /// reads Metal's blue premultiplied (`RotatedBlendForm::Premultiplied`), and one blended in
    /// none of the measured forms (`RotatedBlendForm::Unmeasured`). An attachment blended in
    /// another measured form, or not blended, is not listed. `Device::compile_render_pipeline` sets
    /// it for a `Function` fragment stage from the pipeline's color attachments, and link refuses a
    /// fragment compiled under another map as `NeedsInput(ColorWrites)`. Empty for any other
    /// function.
    #[cfg_attr(feature = "serde", serde(default))]
    pub rotated_blends: BTreeMap<u32, RotatedBlendForm>,
    /// The extended-range color attachments, by color index, whose `half` shadow the draw's
    /// render pass instance holds (`VariantKey::extended_shadows`): Metal's tile memory holds
    /// such an attachment as `half` through the pass and rounds it to the codes once at the
    /// store, so a function that reads one in the interlocked storage form reads the shadow's texel
    /// (`ResourceKind::ExtendedShadow`) as the tile value, and a write stores the value it computes
    /// there as well as the codes. Each is a single-sample attachment the function writes or reads
    /// in the storage form under `framebuffer_fetch`; any other is
    /// `NotImplemented(FramebufferFetch)`. Empty for any other function.
    #[cfg_attr(feature = "serde", serde(default))]
    pub extended_shadows: BTreeSet<u32>,
    /// Sampled texture slots bound to a texture whose format's numeric class (`Float`, `Sint` or
    /// `Uint`) is not the class the function declares there, by the bound format's class
    /// (`VariantKey::cross_class`): each such slot is declared as an image of the bound class,
    /// and its reads, samples and gathers take the bound format's own result, whose 32-bit words
    /// the function hands back as the declared type, a `half` result each word's low 16 bits, as
    /// plain Metal answers (measured on Apple M3 Max and M4). Only slots whose compile states
    /// `RuntimeFact::CrossClassReads` take it. Empty for any other function.
    #[cfg_attr(
        feature = "serde",
        serde(default, with = "crate::pipeline::serde_pairs")
    )]
    pub cross_class: BTreeMap<SlotRef, SampledType>,
    /// The render pipeline's `textureWriteRoundingMode` (Apple SPI), for a fragment function's
    /// texture writes that Metal's mode 2 stores otherwise than its default (a `float` texel into a
    /// half-float, `RG11B10Float` or `RGB9E5Float` texture, a `half` texel into one of the latter
    /// two). Under `NearestEven` such a write the translation has not built is refused where its
    /// slot's storage choice declares such a format, and stated where the bound texture decides
    /// (`RuntimeFact::StorageTextureWriteRounding`, `RuntimeFact::HeapWriteRounding`), which the
    /// binder refuses where one is bound. The function records the mode it compiled such writes
    /// under (`StageIo::texture_write_rounding`), and link refuses one compiled under the other
    /// mode as `NeedsInput(ColorWrites)`. `Device::compile_render_pipeline` sets it for a
    /// `Function` fragment stage from the descriptor. Any other function, and a fragment function
    /// that writes no such texel, compiles as without it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub texture_write_rounding: metal::TextureWriteRoundingMode,
    /// The runtime sampler slots bound to a sampler whose word states anisotropy
    /// (`VariantKey::anisotropic_samplers`, `RuntimeFact::AnisotropicSampler`): a sample through
    /// such a slot that Metal's anisotropic filter, emulated, reaches
    /// (`lower::sampler::anisotropic`) takes it where the slot's word states anisotropy and the
    /// plain sample elsewhere. A slot the function samples through without being named here takes
    /// the plain sample, with no emulation in the module: the binder names every slot whose bound
    /// word states anisotropy. Empty for any other function.
    #[cfg_attr(feature = "serde", serde(default))]
    pub anisotropic_samplers: BTreeSet<SlotRef>,
    /// The `float` and `half` varyings the fragment function reads interpolated (`center`,
    /// `centroid` or `sample`, with or without perspective) whose keys its render pipeline's
    /// vertex function writes the same value at every vertex of a primitive
    /// (`StageIo::uniform_outputs`). The function reads each flat, from the provoking vertex, as
    /// Metal's interpolation returns such a value: a zero or subnormal as `+0`, an infinity as the
    /// largest finite value of its sign, and a NaN as `+FLT_MAX`; every other value exactly
    /// (measured on the M3 Max in every qualifier). It lists them among
    /// `StageIo::uniform_inputs`. `Device::compile_render_pipeline` sets it for a `Function`
    /// fragment stage from the pipeline's vertex function, and a fragment compiled on its own for
    /// such a pipeline compiles under the same keys: link refuses one compiled under a key its
    /// pipeline's vertex function does not write so (`NeedsInput(UniformVaryings)`). Empty for
    /// any other function.
    #[cfg_attr(feature = "serde", serde(default))]
    pub uniform_varyings: BTreeSet<VaryingKey>,
    /// The attributes, by Metal attribute index, that the vertex function fetches in fixed
    /// function from layouts its vertex descriptor steps per instance, which every vertex of one
    /// instance (so of one primitive) reads alike; set only where one of them makes a varying the
    /// function writes the same at every vertex of a primitive (`StageIo::uniform_outputs`).
    /// A pulled attribute's step is in its `VertexFetch::Pulled` already. The function lists
    /// them as `StageIo::instanced_attributes`, and link refuses it behind a descriptor that
    /// steps one of them per vertex (`InvalidDescriptor(VertexAttribute)`, as for another
    /// projection). Empty for any other function.
    #[cfg_attr(feature = "serde", serde(default))]
    pub instanced_attributes: BTreeSet<u32>,
    /// The group of its render pipeline's color attachments a fragment function writes
    /// ([`ColorGroup`]), where the pipeline has more color attachments than the device draws to at
    /// once, or one at an index at or past that count: the function writes the group's colors at
    /// locations `0..` in the order of their indices, and every other color index it names, its
    /// `[[color(n)]]` inputs' included, is renumbered past them ([`ColorGroup::location`]), so a
    /// color of another group or without an attachment is stored nowhere
    /// (`unattached_colors`). Where the pipeline takes more than one group, the function reads no
    /// color outside its group, and its `FragDepth`, `SampleMask` and `FragStencilRefEXT` outputs
    /// are decorated `Invariant` (`PrecisionNote::ColorAttachmentGroups`).
    /// `Device::compile_render_variant` sets it for a `Function` fragment stage from the
    /// descriptor, `VariantKey::color_group` and the profile; `None`, each color at its own index,
    /// for a pipeline the device draws in one pass as it is, and for any other function.
    #[cfg_attr(feature = "serde", serde(default))]
    pub color_group: Option<ColorGroup>,
    /// The payload bytes of the mesh function's mesh render pipeline's object function: the
    /// object function's payload arena (`StageInterface::payload`), where it is larger than the
    /// mesh function's own. Metal sizes a pipeline's payload by the type the object function
    /// declares (`payloadMemoryLength` 0, the descriptor's default), and a mesh function may read
    /// its payload past the type it declares itself, within that; so the mesh function's payload
    /// arena holds these bytes, read as its own (`lower::mesh::payload_extent`, measured on the M3
    /// Max under macOS 27.0.1). `Device::compile_mesh_pipeline` sets it for a `Function` mesh stage
    /// from the pipeline's object function, and a mesh function compiled on its own for such a
    /// pipeline compiles under the same bytes: link refuses one whose arena is smaller than its
    /// object function's as `NeedsInput(ObjectPayload)`. `None` for a mesh function whose own
    /// arena holds the object function's payload, and for any other function.
    #[cfg_attr(feature = "serde", serde(default))]
    pub object_payload: Option<u32>,
    /// A post-tessellation vertex function compiled for a device without `tessellationShader`,
    /// which runs it as a plain vertex function that draws the tessellator's triangles itself
    /// (`lower::tessellation`'s emulation), bakes its render pipeline descriptor's tessellation
    /// state in (the partition, the winding and the maximum factor place its points): that state,
    /// `None` for Metal's defaults (`TessellationDescriptor::default`, which canonical inputs
    /// hold as `None`). `Device::compile_render_pipeline` sets it for such a function on such a
    /// device only, and link refuses a function compiled for another state
    /// (`NeedsInput(TessellationDescriptor)`). Any other function, and any function on a device
    /// with `tessellationShader`, compiles as without it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub tessellation: Option<TessellationDescriptor>,
}

/// One group of a render pipeline's color attachments (`LoweringInputs::color_group`,
/// `GraphicsState::color_group`). A device draws to at most [`width`](Self::width) color
/// attachments at once, the lesser of its `maxColorAttachments` and
/// `maxFragmentOutputAttachments`, at locations below it. A render pipeline whose attachments with
/// a pixel format ([`present`](Self::present)) number more draws them in groups of `width`, in the
/// order of their indices, each group a variant of the pipeline (`VariantKey::color_group`) that
/// writes its attachments at locations `0..`; one whose attachments number no more but reach an
/// index at or past `width` takes one group, renumbered below `width`.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ColorGroup {
    /// The pipeline's color attachments with a pixel format, by index, ascending.
    pub present: Vec<u32>,
    /// The attachments a group holds: the lesser of the device's `maxColorAttachments` and
    /// `maxFragmentOutputAttachments`.
    pub width: u32,
    /// The group: the attachments of [`present`](Self::present) from `group · width` on, `width`
    /// of them or the rest.
    pub group: u32,
}

impl ColorGroup {
    /// Group `group` of a pipeline whose attachments with a pixel format are `present`
    /// (ascending) on a device that draws to `width` at once; `None` where every one of them is
    /// below `width`, which the device draws in one pass at their own indices.
    pub fn of(present: Vec<u32>, width: u32, group: u32) -> Option<ColorGroup> {
        let beyond = present.last().is_some_and(|&last| last >= width);
        beyond.then_some(ColorGroup {
            present,
            width,
            group,
        })
    }

    /// The attachments a group holds on `profile`: the lesser of its `maxColorAttachments` and
    /// `maxFragmentOutputAttachments` (a subpass, or a dynamic rendering, holds at most the
    /// first, `VUID-VkSubpassDescription-colorAttachmentCount-00845`, and a fragment shader's
    /// outputs lie below the second, `VUID-RuntimeSpirv-Location-06272`).
    pub fn width_on(profile: &crate::device::DeviceProfile) -> u32 {
        let l = profile.limits();
        l.max_color_attachments
            .min(l.max_fragment_output_attachments)
    }

    /// The group's attachments, by location: attachment `colors()[l]` at location `l`.
    pub fn colors(&self) -> &[u32] {
        let width = self.width as usize;
        let start = (self.group as usize).saturating_mul(width);
        let end = start.saturating_add(width).min(self.present.len());
        self.present.get(start..end).unwrap_or(&[])
    }

    /// How many groups the pipeline's attachments take: one draw of the pipeline is one draw of
    /// each group's variant.
    pub fn groups(&self) -> u32 {
        match self.width {
            0 => 0,
            width => u32::try_from(self.present.len())
                .unwrap_or(u32::MAX)
                .div_ceil(width),
        }
    }

    /// The location of Metal's color index `color` in the group's variant: an attachment of the
    /// group at its place among [`colors`](Self::colors); every other index past them, in the order
    /// of the indices, so no two indices share a location and none outside the group reaches one
    /// below `colors().len()`.
    pub fn location(&self, color: u32) -> u32 {
        let colors = self.colors();
        match colors.iter().position(|&c| c == color) {
            Some(at) => at as u32,
            None => {
                let below = colors.iter().filter(|&&c| c < color).count() as u32;
                colors.len() as u32 + color - below
            }
        }
    }
}

/// How the blend of an attachment held with its channels rotated reads the fragment function's
/// first output there (`LoweringInputs::rotated_blends`). Apple's blend does not compute each
/// factor pair alike in its color and alpha arithmetic, and the rotation moves Metal's blue into
/// the image's alpha and Metal's alpha into its red: under Metal's color factors `SourceAlpha` over
/// `OneMinusSourceAlpha`, Metal's color is `fma(d, 1 - a, s * a)`, and the image's alpha blends a
/// factor and one minus it as `fma(f, s - d, d)`, which rounds one code apart near half-code ties.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RotatedBlendForm {
    /// The image's alpha (Metal's blue) blends under factors one and one minus the second
    /// output's alpha (Metal's source alpha), and the function writes it premultiplied: Metal's
    /// blue times Metal's alpha, each clamped to `[0, 1]` with a NaN taken as 0 as Metal's blend
    /// clamps its source, multiplied in the output's float format and not contracted. The
    /// blend then computes `fma(d, 1 - a, s * a)`, Metal's own.
    Premultiplied,
    /// None of the measured forms: the blend stores Metal's channels rotated by Vulkan's factors
    /// (`pipeline::rotated_blend`'s search), which Apple's blend may round one code apart from
    /// Metal's (up to 1,767 of 65,536 tie-heavy texels in our probe's sweep). The function
    /// records `PrecisionNote::RotatedBlendRounding`.
    Unmeasured,
}

/// One channel of the second output a fragment function writes for an attachment held with its
/// channels rotated (`LoweringInputs::rotated_factors`): a constant, or a channel of its first
/// output to the attachment as the function returns it, so the blend reads the very value
/// Metal's blend reads.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FactorLane {
    /// `0`.
    Zero,
    /// `1`.
    One,
    /// The first output's channel: Metal's red, green, blue or alpha (0 to 3).
    Output(u8),
}

/// How a fragment function writes a color attachment of one of Apple's extended-range formats
/// (`LoweringInputs::extended_range_colors`), as Metal stores its output there (measured on an
/// Apple M3 Max and an M4): a `float` output is rounded to `half` (nearest
/// even), a `half` one is taken as it is, and the value is stored as a kernel write stores it.
/// Under blending Metal first clamps the output to the format's range (each color channel to the
/// values of codes 0 and 1022 as `half`s, an `_sRGB` color channel of `BGRA10_XR_sRGB` to its
/// decoded codes 0 and 1022, a NaN to the low end; alpha to `[0, 1]`, a NaN to 0).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExtendedRangeColor {
    /// The attachment's format.
    pub form: metal::format::ExtendedRange,
    /// The pipeline enables blending at the attachment, with factors one and zero for color and
    /// alpha and the operation add (any other blend is not built): the output is clamped to the
    /// format's range first.
    pub blended: bool,
    /// A bfloat attachment of a pipeline whose `textureWriteRoundingMode` is
    /// `metal::TextureWriteRoundingMode::NearestEven`: a `float` output rounds to `half` to
    /// nearest even, where at the default mode it rounds toward zero, a finite value past 65504
    /// saturating there. False for every other form, whose store is measured at the default mode
    /// alone: an extended-range form's `float` output rounds to nearest even, a padded one's toward
    /// zero.
    pub nearest_even: bool,
}

impl ExtendedRangeColor {
    /// The write of an attachment of `form`, blended (with factors one and zero) where `blended`,
    /// at the default `textureWriteRoundingMode`.
    pub fn new(form: metal::format::ExtendedRange, blended: bool) -> ExtendedRangeColor {
        ExtendedRangeColor {
            form,
            blended,
            nearest_even: false,
        }
    }

    /// This write in a pipeline of `textureWriteRoundingMode` `rounding`: `nearest_even` where
    /// the form is a bfloat one and `rounding` is `NearestEven`.
    pub fn with_rounding(self, rounding: metal::TextureWriteRoundingMode) -> ExtendedRangeColor {
        ExtendedRangeColor {
            nearest_even: self.form.bfloat()
                && rounding == metal::TextureWriteRoundingMode::NearestEven,
            ..self
        }
    }
}

/// Metal's conversion of a fragment function's float output into one channel of a color
/// attachment (probes on an Apple M3 Max and an M4, every code's exact value and every tie with two
/// `f32` steps either side).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ChannelWrite {
    /// A unorm channel of `bits` bits: the value clamped to `[0, 1]` (a NaN 0) times
    /// `2^bits - 1`, the exact product rounded to nearest.
    Unorm {
        /// The channel's bits.
        bits: u8,
    },
    /// A snorm channel of `bits` bits: the value clamped to `[-1, 1]` times `2^(bits-1) - 1`,
    /// the exact product rounded to nearest. A NaN is 0, or -1.0 where the attachment blends
    /// with factors one and zero (`blended`).
    Snorm {
        /// The channel's bits.
        bits: u8,
        /// The attachment blends (with factors one and zero).
        blended: bool,
    },
    /// An 8-bit sRGB channel: the value clamped to `[0, 1]` (a NaN 0) times 4095, the exact
    /// product rounded to nearest, encoded by Metal's twelve-bit thresholds.
    Srgb,
    /// A floating-point channel of a five-bit exponent (bias 15) and `mantissa` fraction bits:
    /// a half (`mantissa` 10, `signed`) or one of `RG11B10Float`'s unsigned 11- and 10-bit
    /// floats (6 and 5). The value rounded toward zero, a finite value past the largest to the
    /// largest, an infinity kept, a NaN to the quiet NaN (`0x7e00`, `0x7e0`, `0x3f0`); where
    /// unsigned, a negative value (an infinity included) to 0.
    FloatTowardZero {
        /// The channel's fraction bits.
        mantissa: u8,
        /// The channel has a sign bit.
        signed: bool,
    },
    /// The device converts the channel: a near-equivalent the function records as
    /// `PrecisionNote::DeviceColorConversion`.
    Device,
    /// A floating-point channel as [`ChannelWrite::FloatTowardZero`]'s, rounded to nearest even:
    /// a finite value from the midpoint past the largest up to infinity, an infinity kept, a NaN
    /// to the quiet NaN; where unsigned, a negative value to 0. A render pipeline's half-float
    /// channel under `TextureWriteRoundingMode::NearestEven`.
    FloatNearestEven {
        /// The channel's fraction bits.
        mantissa: u8,
        /// The channel has a sign bit.
        signed: bool,
    },
    /// A floating-point channel as [`ChannelWrite::FloatNearestEven`]'s whose blend Metal rounds
    /// to nearest even after blending (a render pipeline's half-float channel under
    /// `TextureWriteRoundingMode::NearestEven` that blends with factors other than one and zero),
    /// which only the device's blend can compute here: the device converts its result by its own
    /// rounding, a near-equivalent the function records as
    /// `PrecisionNote::BlendedWriteRounding`.
    BlendedNearestEven {
        /// The channel's fraction bits.
        mantissa: u8,
        /// The channel has a sign bit.
        signed: bool,
    },
    /// A unorm channel of `bits` bits of a packed 16-bit format (`B5G6R5Unorm`, `A1BGR5Unorm`,
    /// `ABGR4Unorm`, `BGR5A1Unorm`): the value rounded to a half first (to nearest even, a
    /// finite value from 65520 up to infinity), then as [`ChannelWrite::Unorm`]'s: clamped to
    /// `[0, 1]` (a NaN 0) times `2^bits - 1`, the exact product rounded to nearest even. A Clear
    /// load action of these formats converts so too.
    HalfUnorm {
        /// The channel's bits.
        bits: u8,
    },
}

/// What a vertex function compiled for draws of one amplification each writes beyond its own
/// outputs (`LoweringInputs::amplification_draws`, `VariantKey::amplification_draws`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AmplificationDraws {
    /// The render pass has at least two layers (`renderTargetArrayLength` above 1): the vertex
    /// function writes `Layer`, its own slice index plus the amplification's offset, 0 at or past
    /// the pass's length. In a pass of one layer every amplification renders it, and no `Layer`
    /// is written.
    pub layered: bool,
    /// A view mapping offsets the viewport: a vertex function that writes no viewport index of
    /// its own writes `ViewportIndex` as the amplification's offset, 0 at or past the draw's
    /// viewport count. One that writes its own adds the offset whatever this says.
    pub viewports: bool,
}

impl AmplificationDraws {
    /// The draws of a pass of more than one layer (`layered`) under mappings that offset the
    /// viewport (`viewports`).
    pub fn new(layered: bool, viewports: bool) -> AmplificationDraws {
        AmplificationDraws { layered, viewports }
    }
}

/// How a vertex function compiled for a viewport Vulkan's cannot be compensates its position
/// (`LoweringInputs::viewport_compensation`, `VariantKey::viewport_compensation`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ViewportCompensation {
    /// The depth too: the function rewrites `z` and writes the two clip distances, for a depth
    /// range past `[0, 1]` the Vulkan viewport cannot hold. Without it only `x` is negated (a
    /// negative width), and the function declares no clip distance of its own.
    pub depth: bool,
    /// Metal's guard band for points too: the function scales its position's `x` and `y` by the
    /// draw's `PushField::PointGuard`, for a draw of points through Vulkan viewports widened by the
    /// inverse scale about their centers, and writes four clip distances after the others, which
    /// cull a point where Metal does, its `|x|` or `|y|` past `1.5 w`.
    pub point_guard: bool,
    /// Mirrored and unmirrored viewports in one draw: the function negates its position's `x` where
    /// the viewport it selects (the `ViewportIndex` it stores, or the amplification's, else the
    /// first) has its bit set in the draw's `PushField::ViewportMirrors`, on top of `sx`.
    pub mirrors: bool,
    /// Metal's clamp of each vertex's depth before the interpolation, under `depth`: the function
    /// clamps its compensated depth `a w + b z` to `[lo w, hi w]` of the draw's
    /// `PushField::VertexDepthClamp` `[lo, hi]` where `w` is positive, so the device interpolates
    /// the vertices' clamped depths, as Metal does, where it would clamp each sample. The clip
    /// distances take the unclamped `z`.
    #[cfg_attr(feature = "serde", serde(default))]
    pub vertex_clamp: bool,
}

impl ViewportCompensation {
    /// The compensation of the position's `x`, and of its depth where `depth`.
    pub fn new(depth: bool) -> ViewportCompensation {
        ViewportCompensation {
            depth,
            point_guard: false,
            mirrors: false,
            vertex_clamp: false,
        }
    }

    /// This compensation, with Metal's guard band for points where `point_guard`.
    pub fn with_point_guard(self, point_guard: bool) -> ViewportCompensation {
        ViewportCompensation {
            point_guard,
            ..self
        }
    }

    /// This compensation, negating `x` by the selected viewport's bit of the draw's mirrors where
    /// `mirrors`.
    pub fn with_mirrors(self, mirrors: bool) -> ViewportCompensation {
        ViewportCompensation { mirrors, ..self }
    }

    /// This compensation, clamping each vertex's depth by the draw's
    /// `PushField::VertexDepthClamp` where `vertex_clamp`.
    pub fn with_vertex_clamp(self, vertex_clamp: bool) -> ViewportCompensation {
        ViewportCompensation {
            vertex_clamp,
            ..self
        }
    }
}

/// The render pipeline state a fragment function's interlocked framebuffer fetch depends on
/// (`LoweringInputs::framebuffer_fetch`).
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FramebufferFetchInputs {
    /// Each color attachment index the function declares a `[[color(n)]]` input for: the
    /// descriptor's attachment there, `None` where the descriptor sets none.
    pub attachments: BTreeMap<u32, Option<FetchedAttachment>>,
    /// The pipeline has a depth or a stencil attachment, whose tests can drop a fragment after
    /// its function ran.
    pub depth_or_stencil: bool,
    /// The descriptor enables alpha to coverage.
    pub alpha_to_coverage: bool,
    /// The descriptor's `maxVertexAmplificationCount` is above 1: the pipeline may draw
    /// multiview.
    pub amplified: bool,
}

impl FramebufferFetchInputs {
    /// The state of a pipeline whose color attachments are `colors` (by Metal index), whose depth
    /// and stencil formats are `depth` and `stencil`, under `alpha_to_coverage` and
    /// `max_vertex_amplification_count`, for a function that declares color inputs at `read`.
    pub fn of(
        read: impl IntoIterator<Item = u32>,
        colors: &[Option<crate::pipeline::descriptor::ColorAttachmentDescriptor>],
        depth_or_stencil: bool,
        alpha_to_coverage: bool,
        max_vertex_amplification_count: u32,
    ) -> FramebufferFetchInputs {
        FramebufferFetchInputs {
            attachments: read
                .into_iter()
                .map(|color| {
                    let attachment =
                        colors
                            .get(color as usize)
                            .copied()
                            .flatten()
                            .map(|a| FetchedAttachment {
                                pixel_format: a.pixel_format,
                                blending_enabled: a.blending_enabled,
                                write_mask: crate::lower::fragment::compiled_write_mask(
                                    a.pixel_format,
                                    a.write_mask,
                                    a.blending_enabled,
                                ),
                                replaces: crate::pipeline::color_writes::replaces(&a),
                                equation: BlendEquation::of(&a),
                            });
                    (color, attachment)
                })
                .collect(),
            depth_or_stencil,
            alpha_to_coverage,
            amplified: max_vertex_amplification_count > 1,
        }
    }
}

/// What the interlocked framebuffer fetch reads of one color attachment's descriptor.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FetchedAttachment {
    pub pixel_format: metal::PixelFormat,
    pub blending_enabled: bool,
    /// The descriptor's write mask, as the function compiles it: for a format whose kept
    /// channels a draw names (every format but an extended-range or an sRGB one), the mask's class
    /// (none where it covers none of the format's channels, all where it covers every one, and for
    /// a mask covering some, every channel of the format but its first), so that the masks of a
    /// class share one key; any other format's mask as the descriptor sets it.
    pub write_mask: metal::ColorWriteMask,
    /// The descriptor's blend factors are one and zero, added, for color and alpha: where
    /// blending is enabled, Metal stores the source as blending disabled does (but a signed
    /// normalized NaN, and a 16-bit float attachment's blend without fast math, which keeps the
    /// destination's product by zero).
    #[cfg_attr(feature = "serde", serde(default))]
    pub replaces: bool,
    /// The descriptor's blend where it enables blending, which the function computes in the
    /// shader for an extended-range attachment (`metal::format::ExtendedRange`) it writes.
    #[cfg_attr(feature = "serde", serde(default))]
    pub equation: Option<BlendEquation>,
}

/// A color attachment's blend as its descriptor states it: the factors and operation of the color
/// channels and of alpha.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BlendEquation {
    /// `sourceRGBBlendFactor`.
    pub source_rgb: metal::BlendFactor,
    /// `destinationRGBBlendFactor`.
    pub destination_rgb: metal::BlendFactor,
    /// `rgbBlendOperation`.
    pub rgb_operation: metal::BlendOperation,
    /// `sourceAlphaBlendFactor`.
    pub source_alpha: metal::BlendFactor,
    /// `destinationAlphaBlendFactor`.
    pub destination_alpha: metal::BlendFactor,
    /// `alphaBlendOperation`.
    pub alpha_operation: metal::BlendOperation,
}

impl BlendEquation {
    /// The blend of attachment `c`, where it enables blending.
    pub fn of(c: &crate::pipeline::descriptor::ColorAttachmentDescriptor) -> Option<BlendEquation> {
        c.blending_enabled.then_some(BlendEquation {
            source_rgb: c.source_rgb_blend_factor,
            destination_rgb: c.destination_rgb_blend_factor,
            rgb_operation: c.rgb_blend_operation,
            source_alpha: c.source_alpha_blend_factor,
            destination_alpha: c.destination_alpha_blend_factor,
            alpha_operation: c.alpha_blend_operation,
        })
    }

    /// Whether a factor reads the fragment's second output (`[[color(n), index(1)]]`): dual-source
    /// blending.
    pub(crate) fn reads_second_source(&self) -> bool {
        use metal::BlendFactor as F;
        [
            self.source_rgb,
            self.destination_rgb,
            self.source_alpha,
            self.destination_alpha,
        ]
        .into_iter()
        .any(|f| {
            matches!(
                f,
                F::Source1Color
                    | F::OneMinusSource1Color
                    | F::Source1Alpha
                    | F::OneMinusSource1Alpha
            )
        })
    }
}

impl LoweringInputs {
    /// No inputs: fixed-function vertex fetch, no variants.
    pub fn new() -> LoweringInputs {
        LoweringInputs::default()
    }
}

/// How a storage image is declared and viewed for one bound pixel format.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StorageImageChoice {
    /// The format the shader declares (`Unknown` for formatless `Identity` choices, so every such
    /// format of one numeric type shares one key).
    pub vk_format: vk::Format,
    pub class: StorageImageClass,
    /// The mutable-format alias view, when the bound format is not itself storage-capable.
    pub alias: Option<AliasView>,
    /// The numeric type of the bound format's components, kept where the choice declares no
    /// format (`vk_format` `Unknown`): a function whose texels at the slot take that type and
    /// nothing else states it depends on it (`RuntimeFact::StorageTextureNumericType`). `None` when
    /// `vk_format` names the format, whose own numeric type it is.
    #[cfg_attr(feature = "serde", serde(default))]
    pub numeric: Option<SampledType>,
    /// The bound format is one of Apple's RGBX formats (`metal::format::x_channel`), held in its
    /// RGBA twin: after the class's conversion, a read's alpha is 1 (`1.0`, or the integer 1) and
    /// a write stores its green in the fourth channel, Metal's X field.
    #[cfg_attr(feature = "serde", serde(default))]
    pub x_channel: bool,
}

impl StorageImageChoice {
    /// A choice with no alias view, no separate numeric type and no RGBX conversion.
    pub fn new(vk_format: vk::Format, class: StorageImageClass) -> StorageImageChoice {
        StorageImageChoice {
            vk_format,
            class,
            alias: None,
            numeric: None,
            x_channel: false,
        }
    }
}

/// What the shader does around a storage image access.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum StorageImageClass {
    /// The bound format is storage-capable; no conversion.
    Identity,
    /// A bgra8 image viewed as rgba8, or a bgr10a2 image as rgb10a2: the shader exchanges red and
    /// blue.
    SwizzleBgra,
    /// An sRGB image viewed as its UNORM alias: the shader encodes and decodes sRGB by the MSL
    /// transfer through the device's `Pow` (`PrecisionNote::DevicePackConversion` for a write,
    /// `PrecisionNote::SrgbTexelDecode` for a read), where Metal takes its twelve-bit tables.
    SrgbEncode,
    /// A `BGRA8Unorm_sRGB` image viewed as `R8G8B8A8_UNORM`, on a device that cannot declare its
    /// `B8G8R8A8_UNORM` twin (a format without a SPIR-V image format, so it needs the device's
    /// without-format access): the shader encodes and decodes sRGB as [`SrgbEncode`] does and
    /// exchanges red and blue as [`SwizzleBgra`] does.
    ///
    /// [`SrgbEncode`]: StorageImageClass::SrgbEncode
    /// [`SwizzleBgra`]: StorageImageClass::SwizzleBgra
    SrgbEncodeBgra,
    /// An `ARGB8Unorm` image (alpha in byte 0, then red, green, and blue) viewed as
    /// `R8G8B8A8_UNORM`: the shader rotates the channels, and the device converts a written
    /// value.
    RotateArgb,
    /// An `A8Unorm` texture held in `R8_UNORM` (`metal::format::FormatEmulation::AlphaInRed`):
    /// a read is the byte as alpha, `(0, 0, 0, r)`, and a write stores the texel's alpha, as
    /// Metal's do (probe `a8`, an M3 Max under macOS 27.0.1: kernel writes of `float` and `half`
    /// texels and `read_write` access).
    AlphaInRed,
    /// A 4:4:4 two-plane YCbCr texture a slot only writes, through an `R32G32B32A32_UINT` image
    /// of the texture's own (the consumer's write shadow): the shader writes each texel as the
    /// bits of its red, green and blue `float` channels (a `half` texel's widened exactly) and a
    /// 1, the mark of a written texel, which the consumer then encodes into the planes as Metal
    /// stores it (`bind::ycbcr_write`). The slot is declared `Rgba32ui`; it has no reads.
    YCbCrWrite,
    /// An `RGB9E5Float` image a slot writes, and may read, viewed through its `R32_UINT` alias, on
    /// every device (one that stores the format rounds each mantissa half up, as Vulkan's
    /// conversion states, where Metal truncates it): the shader writes each texel as the word Metal
    /// stores, and a read decodes the word exactly (each channel `m · 2^(e - 24)`, alpha 1). Metal's
    /// word (an Apple M3 Max under macOS 27.0.1: kernel writes of `float` and `half` texels, both
    /// accesses): each red, green and blue channel clamped to `[0, 65408]` (a NaN or a negative
    /// value 0), the exponent `e = max(⌊log2 m⌋, -16) + 16` of the largest clamped channel `m` (0
    /// where all are 0), and each mantissa the clamped channel times `2^(24 - e)` truncated. The
    /// slot is declared `R32ui`; it takes no atomic.
    SharedExponentBits,
}

/// The alias view of a storage image created with `MUTABLE_FORMAT`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AliasView {
    /// The format of the bound image; the view uses the choice's `vk_format`.
    pub image_format: vk::Format,
}

impl AliasView {
    /// The alias of an image of `image_format`.
    pub fn new(image_format: vk::Format) -> AliasView {
        AliasView { image_format }
    }
}

/// How each vertex attribute of a vertex function is fetched. Everything else about vertex input is
/// link-time `VertexInputState`.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VertexFetchProjection {
    /// By Metal attribute index.
    pub attributes: BTreeMap<u32, VertexFetch>,
}

impl VertexFetchProjection {
    /// No attributes.
    pub fn new() -> VertexFetchProjection {
        VertexFetchProjection::default()
    }
}

/// How one vertex attribute is fetched.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VertexFetch {
    /// Fixed-function vertex input. The attribute's whole format is link-time state
    /// (`VertexInputState`): no instruction of the module names it, so every format the device
    /// fetches in fixed function shares one module. What the device's conversion of the format
    /// may answer otherwise than Metal's fetch (a normalized format's quotient, a narrow float's
    /// NaN) is noted where the render pipeline links, on its vertex stage
    /// (`PrecisionNote::DeviceVertexConversion`, `PrecisionNote::DeviceNanEncoding`).
    FixedFunction,
    /// Fetched in the shader from a buffer, decoding this Metal format.
    ///
    /// Everything that changes the fetch code is here: the source buffer, the attribute's
    /// offset in an element, and the step. The layout's stride is not: the shader reads it from
    /// `PushField::VertexBufferStride(buffer_index)`. Build it with [`VertexFetch::pulled`],
    /// which canonicalizes the step rate.
    Pulled {
        format: metal::VertexFormat,
        /// The Metal buffer index the attribute's layout reads.
        buffer_index: u32,
        /// The attribute's byte offset in an element.
        offset: u32,
        step_function: metal::VertexStepFunction,
        /// The instance step rate for `PerInstance`; 1 for `PerVertex` and 0 for `Constant`,
        /// whose fetch does not depend on it.
        step_rate: u32,
    },
}

impl VertexFetch {
    /// The pulled fetch of attribute `attr` from layout `layout`, with the step rate
    /// canonicalized (equal fetch code gives an equal key).
    pub fn pulled(
        attr: &VertexAttributeDescriptor,
        layout: &VertexBufferLayoutDescriptor,
    ) -> VertexFetch {
        let step_rate = match layout.step_function {
            metal::VertexStepFunction::PerVertex => 1,
            metal::VertexStepFunction::Constant => 0,
            _ => layout.step_rate,
        };
        VertexFetch::Pulled {
            format: attr.format,
            buffer_index: attr.buffer_index,
            offset: attr.offset,
            step_function: layout.step_function,
            step_rate,
        }
    }
}
