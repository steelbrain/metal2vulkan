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
    StageInputDescriptor, VertexAttributeDescriptor, VertexBufferLayoutDescriptor,
};
use crate::pipeline::sym::{SampledType, SlotRef, VaryingKey};
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
    /// A kernel's stage-input descriptor (`MTLComputePipelineDescriptor.stageInputDescriptor`): how
    /// each `[[stage_in]]` attribute is fetched from its buffer. A kernel reading a stage-in
    /// attribute needs it (`NeedsInput(StageInputDescriptor)` without it);
    /// `Device::compile_compute_pipeline` compiles a `Function` kernel under the pipeline's, and
    /// a kernel compiled on its own for such a pipeline compiles under the same descriptor. The
    /// key takes it canonicalized ([`StageInputDescriptor::canonical`]); any other function
    /// compiles as without it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub stage_input: Option<StageInputDescriptor>,
    /// The color attachments a fragment function writes that its render pipeline gives a
    /// signed-integer pixel format (`MTLRenderPipelineColorAttachmentDescriptor.pixelFormat`). A
    /// 32-bit integer color output to one is declared `int`, because Vulkan writes a signed-integer
    /// attachment only from a signed output; every other integer output is declared `uint`. Under
    /// the identity domains it also holds the signed-integer attachments the function reads
    /// (`[[color(n)]]`), whose integer
    /// read fetches the attachment's own view through an `int` image (`PushField::FetchedColor`).
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
    /// The fragment function's render pipeline rasterizes more than one sample (`rasterSampleCount`
    /// above 1 with rasterization on). Metal reads `[[position]]` and center-sampled varyings at
    /// the pixel center in every invocation, so under it a fragment function reads `[[position]]`'s
    /// x and y as `floor(FragCoord.xy) + 0.5`, and each center-sampled varying through
    /// `InterpolateAtOffset` at offset 0 on a device with interpolation functions whose offset
    /// reads from the center (`VkProperty::InterpolateAtOffsetCentered`): the pixel center whatever
    /// samples its invocation shades (`StageIo::{multisampled, varyings_at_pixel}`). A function
    /// that reads its color attachments reads each one's multisampled image at its `SampleId`,
    /// running once per covered sample, as
    /// Metal does. `Device::compile_render_pipeline` sets it for a `Function` fragment stage, and a
    /// fragment compiled on its own for such a pipeline compiles under it: link refuses one
    /// compiled without it whose sample-rate invocations would read either elsewhere, or that reads
    /// its color attachments, as `NeedsInput(Multisampled)`, and one compiled under it that reads
    /// its color attachments into a pipeline of one sample likewise. Any other function compiles as
    /// without it.
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
    /// and a vertex function that fetches an attribute in fixed function are
    /// `Unsupported{GpuEncodedIndirectCommands, NotImplemented}` under it (a replayed command is a
    /// compute dispatch with no stage-in region, or a draw whose buffers no fixed-function input
    /// reads: a pulled attribute reads its slot table).
    #[cfg_attr(feature = "serde", serde(default))]
    pub replay_bindings: bool,
    /// With `replay_bindings`, the vertex function of a draw replay merged into one draw
    /// (`bind::replay`'s `merged.rs`): it reads its header's device address
    /// at `PushField::MergedDrawAddress` in place of `RootTableAddress`, finds the command its
    /// corner (`InstanceIndex` and `VertexIndex`) falls in, and runs that command's vertex: its
    /// `[[vertex_id]]`, `[[instance_id]]`, `[[base_vertex]]` and `[[base_instance]]` are the
    /// command's draw's, its pulled attributes step by the command's vertex and instance, and its
    /// buffers are in the command's vertex slot table. A fragment
    /// function under it reads no buffer and no `[[primitive_id]]`, which a merged draw does not
    /// keep per command: one that does is `Unsupported{GpuEncodedIndirectCommands,
    /// NotImplemented}`, and so is a vertex function that writes the primitive ID's varying or
    /// draws amplifications as draws of their own. The vertex function
    /// states `RuntimeFact::MergedDraws`. Any other stage compiles as without it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub merged_draws: bool,
    /// Texture slots bound to an 8-bit sRGB texture the device cannot sample, which the runtime
    /// holds as the format's UNORM twin, whose views read the codes (`VariantKey::srgb_textures`,
    /// `RuntimeFact::SrgbTexture`): each `half` or `float` access of such a slot decodes what its
    /// own binding reads (one texel's codes, or the device's filter of them; a bicubic sample's
    /// taps) by the MSL transfer through the device's `Pow`, recording
    /// `PrecisionNote::SrgbTexelDecode`. A slot of no such access is ignored.
    #[cfg_attr(feature = "serde", serde(default))]
    pub srgb_textures: BTreeSet<SlotRef>,
    /// Storage texture slots bound to an 8-bit sRGB texture whose storage element is a view of
    /// the format's UNORM twin (`metal::format::srgb_raw_twin`), as Vulkan stores no sRGB format
    /// through a storage image on the devices it is built for: each such slot a `half` or `float`
    /// function writes or reads is declared without a format, of class `SrgbEncode`, so its
    /// writes encode red, green and blue by the device's linear-to-sRGB conversion and keep alpha
    /// linear (the MSL specification, 8.7.7), recording `PrecisionNote::DevicePackConversion`, and
    /// its reads decode. A slot of no such access, or of another numeric type, is ignored.
    #[cfg_attr(feature = "serde", serde(default))]
    pub srgb_storage: BTreeSet<SlotRef>,
    /// Texture-buffer slots bound to a texture buffer of more texels than one view holds, each
    /// with its parts (`VariantKey::texel_buffer_parts`, `RuntimeFact::TexelBufferParts`): each is
    /// declared as an array of that many texel buffers, texel `i` of its reads, writes and atomics
    /// texel `i − k · P` of part `k = min(⌊i / P⌋, parts − 1)`, `P` the device's
    /// `bind::part_texels`, and its width `(parts − 1) · P` plus the last part's
    /// (`bind::texel_parts`).
    #[cfg_attr(
        feature = "serde",
        serde(default, with = "crate::pipeline::serde_pairs")
    )]
    pub texel_buffer_parts: BTreeMap<SlotRef, u32>,
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
    /// The kernel is the kernel of a compute pipeline of OpenCL mode
    /// (`ComputePipelineDescriptor::open_cl_mode_enabled`): it takes Metal's three hidden buffers
    /// at indices 0 to 2, each of its arguments bound at a buffer index sits 3 past its own, and
    /// OpenCL C's work-item functions read the dispatch's values (`air::canon::opencl`).
    /// `Device::compile_compute_pipeline` compiles a `Function` kernel under the pipeline's, and a
    /// kernel compiled on its own for such a pipeline compiles under it too: the pipeline refuses
    /// a kernel compiled under the other mode (`InvalidDescriptor(OpenClMode)`). Any other
    /// function compiles as without it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub open_cl_mode: bool,
    /// The kernel runs its dispatch's grid flat (`VariantKey::flat_grid`): the dispatch is any
    /// number of threadgroups of the variant's local size, whose invocations count the grid's
    /// threads in order (`x` fastest) from the `threads_per_grid` of the 32-byte region record at
    /// `PushField::ReplayRegionAddress`, each taking the thread position it counts to and returning
    /// early past the grid's last thread. So one variant runs every grid and threadgroup size of
    /// its pipeline, which only a kernel that does not observe its threadgroup partition
    /// (`KernelFacts::partition_oblivious`) allows: it reads no dispatch builtin but
    /// `thread_position_in_grid` and `threads_per_grid`. Any other function is
    /// `Unsupported{GpuEncodedIndirectCommands, NotRepresentable}` under it, but a kernel that
    /// observes its partition through its simdgroups alone under
    /// [`call_threadgroups`](Self::call_threadgroups). Under
    /// `replay_bindings` the replay's preprocess writes the record per command; otherwise the
    /// kernel binds as any other, and a preprocess writes
    /// the record beside its dispatch's group counts. The identity domains refuse it
    /// (`Unsupported{IdentityDomains}`). It needs device addresses, and refuses a
    /// stage-input region, a tile function and `emulated_simdgroups`
    /// (`Unsupported{GpuEncodedIndirectCommands, NotImplemented}`). Its tangled instructions take
    /// the profile's reconvergence mode even where the plain module's are reached together, as
    /// a partial-threadgroups variant's do ([`partial_threadgroups`](Self::partial_threadgroups)):
    /// its invocations past the grid return before the body, and SPIR-V reconverges only over
    /// all invocations, those that returned included (SPIR-V 1.6 section 2.2.5, "Dynamic
    /// Instance"); so on a device without maximal reconvergence it records
    /// `Reconvergence::ImplementationDefined` and `PrecisionNote::DeviceReconvergence`.
    #[cfg_attr(feature = "serde", serde(default))]
    pub flat_grid: bool,
    /// With [`flat_grid`](Self::flat_grid), the dispatch's threadgroups are the call's
    /// (`VariantKey::call_threadgroups`): the grid is one row along `x`, in threadgroups of the
    /// variant's local size, one row along `x` too, so workgroup `w`, counted `x` fastest, is
    /// Metal's threadgroup `w`, and its invocations count its threads in the order of its
    /// simdgroups' lanes (`lin`, the flattened tiers' subgroups). A kernel that observes its
    /// partition through its simdgroups alone (`KernelFacts::partition_simdgroups`) then computes
    /// what Metal's dispatch computes, and may read `thread_index_in_simdgroup`,
    /// `thread_index_in_quadgroup` and `threads_per_simdgroup` besides the grid's builtins. The
    /// invocations past the grid's last thread return before the body, so its simdgroup
    /// operations and barriers run on the live lanes of the last simdgroup, Metal's partial one;
    /// a module that would hold a workgroup barrier (a simdgroup barrier or fence emitted at
    /// workgroup scope, in a tier whose subgroup is not the simdgroup) is
    /// `Unsupported{GpuEncodedIndirectCommands, NotImplemented}`, and so is the input without
    /// `flat_grid` or with `replay_bindings`. The caller states the dispatch's shape: nothing in
    /// the module checks it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub call_threadgroups: bool,
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
    /// fragment `PrimitiveId` (neither `geometryShader` nor `tessellationShader`).
    ///
    /// Under this field a vertex function writes the varying. A plain one writes it at each
    /// return: `(VertexIndex - first) / n` in wrapping `u32` arithmetic, `first` the draw's first
    /// vertex (Vulkan's base vertex of a draw without indices, read as `[[base_vertex]]` is) and
    /// `n` its vertices per primitive, `PushField::VerticesPerPrimitive` (3 for a triangle list,
    /// 2 for a line list, 1 for a strip or points). A primitive's flat inputs read its provoking
    /// vertex, its first in a pipeline of Vulkan's default provoking vertex, at stream position
    /// `n · i` of a list's primitive `i` and `i` of a strip's, so every fragment reads its
    /// primitive's index in its instance, Metal's
    /// `[[primitive_id]]` (Metal probes on the M3 Max, macOS 27.0.1). A post-tessellation vertex
    /// function is refused under it (`NotImplemented(Tessellation)`).
    ///
    /// A fragment function reads `[[primitive_id]]` from the varying under this field, as it does
    /// without it on a device that cannot declare a fragment `PrimitiveId`.
    /// `Device::compile_render_pipeline` sets it for a `Function` vertex stage whose pipeline's
    /// fragment function reads `[[primitive_id]]`, on a device that cannot declare it. Link refuses a vertex function compiled without it behind a
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
    /// The function's render pipeline state is made for indirect command buffers
    /// (`supportIndirectCommandBuffers`): a vertex function reads the execution range at
    /// `PushField::IndirectRange` and its command, `PushField::IndirectSequence` plus
    /// `PushField::IndirectBase`, in its first block, and an invocation of a command outside the range (`command - location >= length`,
    /// in `u32`) stores its `[[position]]` as `(2, 2, 2, 1)`, outside every clip volume, and
    /// returns before the body, so its command draws nothing and writes no memory. A direct draw
    /// of the state is pushed a range of every command. Refused on a device without device
    /// addresses (`Feature::GpuEncodedIndirectCommands`). `Device::compile_render_pipeline` sets
    /// it for a `Function` vertex stage of a descriptor that sets the flag, whose vertex function
    /// then pulls every attribute it reads. Any other function compiles as without it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub indirect_range: bool,
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
    /// its own or under [`AmplificationDraws::viewports`]. On a device without `multiViewport` it
    /// stores no viewport at all. The raw index varyings it writes hold the sums. A fragment
    /// function reads its `[[render_target_array_index]]` and `[[viewport_array_index]]` from those
    /// varyings where its vertex function writes the index (`render_target_array_index_varying`,
    /// `viewport_index_varying`), and otherwise as the offsets themselves, from the push fields.
    /// The vertex function states `RuntimeFact::AmplificationDraws`. Any other function compiles as
    /// without it.
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
    /// `TextureWriteRoundingMode::NearestEven`, which the function rounds itself. A rule
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
    /// The kernel runs a `dispatchThreads` grid that is not a whole number of threadgroups in the
    /// threadgroups its threads fill, rounded up, each at the call's threadgroup size `L`
    /// (`VariantKey::partial_threadgroups`; `lower::dispatch`'s partial threadgroups): where its
    /// `DispatchRegion` carries `lower::dispatch::NONUNIFORM`, the last threadgroup along an axis
    /// answers Metal's partial threadgroup, `min(L, threads_per_grid − position · L)` threads per
    /// axis, as `threads_per_threadgroup`, in its simdgroup and quadgroup counts and in its
    /// threads' positions, and its invocations past them run none of the body (they return before
    /// it, or, in a body that holds a workgroup barrier, walk it predicated to each barrier), so
    /// its simdgroup operations run on the live lanes, Metal's partial simdgroup. Every lane read
    /// ballots its live lanes (`simd::partly_live`). It declares `MultiRegion` and reads the
    /// region whatever builtins it reads, and states `RuntimeFact::PartialThreadgroups`. A tile
    /// function, compute derivatives, an imageblock slice write every invocation shares, emulated
    /// simdgroups (the input
    /// `emulated_simdgroups` or a kernel whose lockstep order needs them), `flat_grid`,
    /// `folded_groups` and `replay_bindings` are `Unsupported{PartialThreadgroups,
    /// NotImplemented}`, and a body whose uniform barriers' vote word its workgroup memory cannot
    /// hold is `Unsupported{PartialThreadgroups}` needing the `Limit` of
    /// `maxComputeSharedMemorySize` the word passes. Any other function compiles as without it,
    /// but for its reconvergence: its tangled instructions take the profile's mode even where
    /// the plain module's are reached together, since its invocations past the grid return before
    /// the body and SPIR-V reconverges only over all invocations, those that returned included
    /// (SPIR-V 1.6 section 2.2.5, "Dynamic Instance"); so on a device without maximal
    /// reconvergence it records `Reconvergence::ImplementationDefined` and
    /// `PrecisionNote::DeviceReconvergence` where the plain module may record `Guaranteed` and
    /// none.
    #[cfg_attr(feature = "serde", serde(default))]
    pub partial_threadgroups: bool,
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
    /// takes the format's own attachment). The function writes such an attachment
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
    /// key gives each attachment that format, written whole and unblended. What the function
    /// writes to such a color itself is stored nowhere, as plain Metal stores the zero there all
    /// the same (a Metal probe); a function that reads such a color is `NotImplemented`
    /// (unmeasured). Any other function compiles as without it.
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
    /// as plain Metal does. And each sampled slot bound to a swizzled packed normalized texture:
    /// the function swizzles the texels its class converts, likewise.
    #[cfg_attr(feature = "serde", serde(default))]
    pub swizzled_storage: BTreeSet<SlotRef>,
    /// The function reads every visible-function table it reads by resource id through the
    /// table's device address (`VariantKey::addressed_function_tables`): the directory's entry
    /// of the id holds the address of the table's words in place of an element, and a read of
    /// word `w` is checked against the table's size word, so a table whose words pass
    /// `maxStorageBufferRange` is read whole. It declares no snapshot, and states
    /// `RuntimeFact::FunctionTableRange` with `addressed`. It needs
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
    /// 27.0.1), through the input attachment.
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
    /// The attachments of `rotated_colors` whose blend is not exact by construction, by color
    /// index (`pipeline::rotated_blend`): Vulkan's factors store Metal's channels rotated, which a
    /// blend may round one code apart from Metal's (up to 1,767 of 65,536 tie-heavy texels on
    /// Apple's in our probe's sweep), and the function records
    /// `PrecisionNote::RotatedBlendRounding`. An attachment blended under a minimum or a maximum, or
    /// by Vulkan factors that are all `ZERO` or `ONE`, or not blended, is not listed. `Device::compile_render_pipeline` sets it for a `Function` fragment stage from the
    /// pipeline's color attachments, and link refuses a fragment compiled under another set as
    /// `NeedsInput(ColorWrites)`. Empty for any other function.
    #[cfg_attr(feature = "serde", serde(default))]
    pub rotated_blends: BTreeSet<u32>,
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
    /// With [`open_cl_mode`](Self::open_cl_mode), the kernel's texture slots of a 1D image
    /// (`RuntimeFact::OpenClLineTexture`) bound to a 1D or 1D array texture
    /// (`VariantKey::open_cl_lines`), where Apple's OpenCL binds a 2D texture of height 1: the
    /// kernel declares each as that 1D texture and reads it, as a Radeon Pro 5300M (macOS 26.7.1)
    /// answers (`air::SpecRequest::open_cl_lines`). Every other 1D image is read as the 2D
    /// texture. Empty for any other function.
    #[cfg_attr(feature = "serde", serde(default))]
    pub open_cl_lines: BTreeSet<SlotRef>,
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
    /// The payload bytes of the object or mesh function's mesh render pipeline: for a mesh
    /// function, its object function's payload arena (`StageInterface::payload`), where it is
    /// larger than the mesh function's own; for an object function, and a mesh function without
    /// one, the descriptor's `payloadMemoryLength`, where it is larger than the function's payload
    /// type. Metal sizes a pipeline's payload by the type the object function declares
    /// (`payloadMemoryLength` 0, the descriptor's default) or by that length, and a mesh function
    /// may read its payload past the type it declares itself, within that; so the function's
    /// payload arena holds these bytes, read as its own (`lower::mesh::payload_extent`, measured on
    /// the M3 Max under macOS 27.0.1). `Device::compile_mesh_pipeline` sets it for a `Function`
    /// object and mesh stage, and a function compiled on its own for such a pipeline compiles
    /// under the same bytes: link refuses a mesh function whose arena is smaller than its object
    /// function's as `NeedsInput(ObjectPayload)`. `None` for a function whose own arena holds the
    /// bytes, and for any other function.
    #[cfg_attr(feature = "serde", serde(default))]
    pub object_payload: Option<u32>,
    /// The function's render pipeline draws a rasterization-rate-mapped pass's regions several
    /// at a time (`VariantKey::region_viewports`): each region one draw of a
    /// `vkCmdDrawMultiEXT` or `vkCmdDrawMultiIndexedEXT`, its viewport and scissor at the draw's
    /// index among the command's. Under it a vertex function stores `ViewportIndex` as its
    /// `DrawIndex`, "the zero-based index of the draw"
    /// within such a command (the specification's `DrawIndex`), which selects "the viewport
    /// transform, scissor rectangle, and exclusive scissor rectangle" (its `ViewportIndex`); the
    /// index is the same at every vertex of a draw, so of a primitive. It needs the device to
    /// store `ViewportIndex` from a vertex function (`shaderOutputViewportIndex`), to read
    /// `DrawIndex` (`shaderDrawParameters`) and to take several viewports (`multiViewport`):
    /// `Unsupported(RasterizationRateMap)` names them otherwise. A merged draw's vertex function
    /// (`merged_draws`) takes it too, its corner from its `VertexIndex` and `InstanceIndex` as
    /// without it, and so does a draw replay's of one draw per command (`replay_bindings`). A
    /// function that writes its own `[[viewport_array_index]]` or
    /// `[[render_target_array_index]]`, one compiled under `amplification_draws`, and a
    /// post-tessellation vertex function, is
    /// `NotImplemented(RasterizationRateMap)` under it: the
    /// runtime draws those region by region. The vertex function states `RuntimeFact::RegionViewports`. A
    /// fragment function compiled under it (behind such a vertex function, where it reads
    /// `[[viewport_array_index]]`) reads 0 there, the index Metal's reads behind a vertex
    /// function that writes none. Any other stage compiles as without it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub region_viewports: bool,
    /// Where a fragment function's push fields start on `Device::from_queries`' profile, whose
    /// render pipeline pushes one range `[0, W)` for both stages, `W` the device's push window
    /// ([`DeviceProfile::push_window`](crate::device::DeviceProfile::push_window), 256 or 128):
    /// its vertex function's fields first, from 0 (from 24 for a post-tessellation function, past
    /// the control stage's range), then the fragment function's from this offset, the vertex
    /// function's push end
    /// ([`CompiledFunction::fragment_push_base`](crate::library::CompiledFunction::fragment_push_base)).
    /// `Device::compile_render_pipeline` sets it for a `Function` fragment stage from the vertex
    /// function it compiled, and `Device::render_pipeline_key` hands it out in the fragment
    /// function's inputs from the vertex function's plan; a fragment compiled on its own for such a
    /// pipeline compiles under the same offset, and link refuses a fragment whose fields start below the vertex function's push
    /// end. A fragment function whose fields do not fit in `[push_base, W)` pushes its parameter
    /// block's address at `push_base` rounded up to 8 (the address's alignment) and spills the
    /// rest into the block, as a kernel spills past its window; the vertex function's window ends
    /// at `W - 8`, so the address always fits. On any other profile, and for any other function, it is
    /// 0.
    #[cfg_attr(feature = "serde", serde(default))]
    pub push_base: u32,
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
/// A padded form's output Metal blends unclamped; a blend into one of Apple's extended-range forms
/// is not implemented (`lower::fragment::fetch::codes_round_the_tile`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExtendedRangeColor {
    /// The attachment's format.
    pub form: metal::format::ExtendedRange,
    /// The pipeline enables blending at the attachment: the fragment function computes a padded
    /// form's blend over an ordered read of the attachment (`lower::fragment`). A blend into a
    /// bfloat attachment is Metal's pipeline error (`InvalidDescriptor(ColorAttachment)`), and one
    /// into one of Apple's extended-range forms is not implemented (`Unsupported(Format)`).
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
    /// The write of an attachment of `form`, blended where `blended`, at the default
    /// `textureWriteRoundingMode`.
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

impl LoweringInputs {
    /// No inputs: fixed-function vertex fetch, no variants.
    pub fn new() -> LoweringInputs {
        LoweringInputs::default()
    }

    /// The inputs of a kernel's `variant`, where these are its plain module's: a compute
    /// pipeline's [`lowering_inputs`](crate::pipeline::ComputePipelineDescriptor::lowering_inputs)
    /// narrowed to the attributes its kernel reads
    /// ([`Device::kernel_inputs_read`](crate::device::Device::kernel_inputs_read)). These inputs
    /// as they are for [`KernelVariant::Plain`]; with
    /// [`partial_threadgroups`](Self::partial_threadgroups) for
    /// [`KernelVariant::PartialThreadgroups`], and with
    /// [`emulated_simdgroups`](Self::emulated_simdgroups) for
    /// [`KernelVariant::EmulatedSimdgroups`]. A consumer that lowers a kernel's variants itself
    /// builds each one's inputs here, so two consumers of one device lower one module per variant.
    pub fn kernel_variant(&self, variant: KernelVariant) -> LoweringInputs {
        let mut inputs = self.clone();
        match variant {
            KernelVariant::Plain => {}
            KernelVariant::PartialThreadgroups => inputs.partial_threadgroups = true,
            KernelVariant::EmulatedSimdgroups => inputs.emulated_simdgroups = true,
        }
        inputs
    }
}

/// One of the modules a compute pipeline's kernel is lowered to
/// ([`LoweringInputs::kernel_variant`]): its plain module, and the variants a dispatch that the
/// plain module does not run as Metal runs it takes instead.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum KernelVariant {
    /// The plain module: a whole number of threadgroups, each of whole SIMD groups where the
    /// profile's subgroups give them.
    Plain,
    /// The partial-threadgroups variant ([`LoweringInputs::partial_threadgroups`]): a
    /// `dispatchThreads` grid that is not a whole number of threadgroups, or a kernel that does
    /// not observe its threadgroup partition in tiles of its grid.
    PartialThreadgroups,
    /// The emulated-SIMD-group variant ([`LoweringInputs::emulated_simdgroups`]): a threadgroup
    /// that ends in a partial SIMD group.
    EmulatedSimdgroups,
}

impl KernelVariant {
    /// Every variant, the plain module first.
    pub const ALL: [KernelVariant; 3] = [
        KernelVariant::Plain,
        KernelVariant::PartialThreadgroups,
        KernelVariant::EmulatedSimdgroups,
    ];
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
    /// Fixed-function vertex input of an unnormalized signed-integer format, into a variable
    /// declared `int`: Vulkan's numeric-type match lets a `SINT` format feed a signed variable
    /// only, and the device sign-extends each component as Metal's fetch does. The module reads
    /// the variable's bits by `OpBitcast`. Taken under the identity domains, where the pulled
    /// fetch is not; elsewhere such an attribute is pulled and sign-extended in the shader.
    SignedFixedFunction,
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
