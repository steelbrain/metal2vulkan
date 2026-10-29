//! `RuntimeFact`, `Mechanism`, and `UniformViewChoice`.
//!
//! `RuntimeFact` is the single dependency vocabulary: `select` emits facts, the planner supplies
//! their values, and `device::mechanism(fact, profile)` is a total function with one [`Mechanism`]
//! per (fact, profile).

use crate::air::container::Hash256;

use crate::pipeline::sym::{ImageHandleClass, SlotRef, Stage};

/// Translation-relevant state that Metal binds at pipeline or encode time.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RuntimeFact {
    /// Metal threads per threadgroup.
    ThreadsPerThreadgroup,
    /// Length of threadgroup-memory argument `index`.
    ThreadgroupMemoryLength { index: u32 },
    /// Every dispatch region's local size must be a multiple of this.
    LocalSizeConstraint { multiple: u32 },
    /// Local sizes are checked against the device's per-axis, total, and subgroup limits.
    LocalSizeLimits,
    /// Grid size and region base of a dispatch.
    DispatchGrid,
    /// GPU-written indirect group counts.
    IndirectGroupCount,
    /// Whether the slot is bound, for slots whose null-ness the shader observes.
    ResourcePresence { slot: SlotRef },
    /// Rasterization sample count.
    SampleCount,
    /// Format of the storage texture bound to the slot.
    StorageTextureFormat { slot: SlotRef },
    /// Coordinate mode of the runtime sampler bound to the slot, and, for a sample that may read
    /// pixel coordinates, whether that pixel sampler filters nearest: the fields the function
    /// reads of the slot's `SpecKey::SamplerState`.
    SamplerCoordinates { slot: SlotRef },
    /// Reduction mode of the runtime sampler bound to the slot.
    SamplerReduction { slot: SlotRef },
    /// View shape of the texture bound to the slot (cube-as-2D-array).
    ImageViewShape { slot: SlotRef },
    /// How a vertex attribute is fetched (fixed-function or pulled, and the pulled format).
    VertexFetch { attr: u32 },
    /// Stride of a vertex buffer layout.
    VertexBufferStride { layout: u32 },
    /// Base vertex and base instance.
    BaseVertexInstance,
    /// The slot needs a device address (`BufferForm::Psb`).
    DeviceAddresses { slot: SlotRef },
    /// The slot's window may exceed the descriptor range limit.
    BufferRangeLimit { slot: SlotRef },
    /// The texel buffer bound to the slot must fit the texel-buffer limits.
    TexelBufferRange { slot: SlotRef },
    /// Framebuffer fetch of this color index.
    FramebufferFetch { color: u32 },
    /// The function writes the command buffer's `ExecutionStatus` buffer (resident-set misses,
    /// absent reads, sequentially consistent fences, an argument buffer's cubic sampler): through
    /// the buffer's device address (`PushField::ExecutionStatusAddress`), or, on a device that
    /// does not address buffers, through the stage's binding of it
    /// (`SyntheticBinding::ExecutionStatus`).
    ExecutionStatus,
    /// The set-1 heap array of `class` is declared with the default `Float` sampled type because
    /// some heap group of that class is only queried: nothing structural types it (the
    /// argument-buffer member's texture type is only its MSL type name). A resident texture that
    /// matches the class in everything but its numeric type cannot be an element of the array, so a
    /// query through such a handle answers the neutral's; the binder reports each such texture
    /// (`PlanDiagnostic::IncompatibleHeapEntry`).
    HeapNumericType { class: ImageHandleClass },
    /// The kernel's simdgroups are emulated through workgroup memory: beside its threadgroup arena
    /// it takes `bytes_per_simdgroup` for each of the `⌈N / 32⌉` simdgroups of a threadgroup of `N`
    /// threads, which every dispatch region must fit in `maxComputeSharedMemorySize`. `arena` is
    /// the static part of the threadgroup arena the module declares (`TgShape::static_bytes`),
    /// `None` when it declares none; a declared arena takes whole words and at least
    /// `MIN_ARENA_BYTES` (`lower::simd::workgroup::arena_bytes`).
    SimdgroupEmulation {
        bytes_per_simdgroup: u32,
        arena: Option<u32>,
    },
    /// The render encoder's vertex amplification count, which a stage reads as
    /// `[[amplification_count]]` (`SpecKey::AmplificationCount`).
    AmplificationCount,
    /// The kernel was compiled under `VariantKey::emulated_simdgroups`: its simdgroups run in the
    /// emulated tier whatever tier the profile's subgroups give them, so every threadgroup size
    /// runs, partial simdgroups included. Its plans state the variant.
    EmulatedSimdgroups,
    /// The compute encoder's imageblock size (`setImageblockWidth:height:`,
    /// `SpecKey::ImageblockWidth` and `SpecKey::ImageblockHeight`): the kernel's explicit
    /// imageblock takes `cell_bytes` of the threadgroup's Workgroup memory per pixel, after its
    /// threadgroup arena.
    ImageblockSize { cell_bytes: u32 },
    /// The vertex function writes `[[render_target_array_index]]`, which selects a slice only
    /// in a render pass whose `renderTargetArrayLength` is at least 2: in a pass of at most one
    /// layer every index renders slice 0, and the function runs on the variant that does not write
    /// `Layer` (`VariantKey::dropped_vertex_layer`). `layer_dropped` states that it was compiled
    /// under that variant. Its plans state the variant the shape's length asks for, and refuse
    /// amplified draws whichever it is. The variant that writes `Layer` also reads the length from
    /// `PushField::RenderTargetArrayLength`, which the binder writes per draw, and stores 0 for an
    /// index at or past it, where Metal renders the pass's first layer.
    RenderTargetArrayLength { layer_dropped: bool },
    /// The kernel reads the compute encoder's stage-in region (`setStageInRegion:`): a
    /// `[[stage_in]]` attribute stepping by thread position clamps its element to the region and
    /// adds the region's origin, and `[[grid_origin]]`/`[[grid_size]]` read the region
    /// (`PushField::StageInRegion`).
    StageInRegion,
    /// The kernel fetches `[[stage_in]]` attributes from the buffer at Metal buffer index
    /// `layout` at the stride the encoder binds it with (`setBuffer:offset:attributeStride:
    /// atIndex:` and its forms), its stage-input layout being `MTLBufferLayoutStrideDynamic`: the
    /// binder writes the bound stride into `PushField::StageInStride`, or 0 for a buffer bound
    /// without one, as Metal reads it. A layout of a fixed stride reads its own, compiled in,
    /// whatever the bind's, and records nothing.
    StageInStride { layout: u32 },
    /// A read-only slot that Profile F's per-stage storage-buffer fallback may read through a
    /// uniform view, whose window does not bound its descriptor range within
    /// `maxUniformBufferRange` (a `Whole` window, or a longer `Bounded` one): the view is exact
    /// only where the range the shape binds fits. `choice` states what this compile made of it.
    /// A plan whose shape does not prove the fit for a `Viewed` slot asks for the variant that
    /// keeps every such slot a storage view (`VariantKey::storage_view_slots`); instantiation
    /// refuses a range above the limit (`BindError::DescriptorRangeExceeded`).
    UniformViewRange {
        slot: SlotRef,
        choice: UniformViewChoice,
    },
    /// A `device` slot read through a uniform view (Profile F's per-stage storage-buffer
    /// fallback). A uniform view need not observe what the dispatch or draw itself writes, so no
    /// buffer slot the pipeline writes may be bound overlapping its window: instantiation checks
    /// the Metal addresses (`BindError::UniformViewAliased`).
    UniformViewAliasing { slot: SlotRef },
    /// The fragment function reads or writes the render pass's imageblock, whose cells take
    /// `words` 32-bit words per pixel (`air.imageblock_data_size` rounded up to a word): the
    /// consumer binds the pass's imageblock storage (`SyntheticBinding::Imageblock`), which
    /// `GraphicsState::fragment_imageblock` describes. `writes`: the function writes the
    /// imageblock, so the storage holds its write before the pipeline's coverage (alpha to
    /// coverage) could drop the fragment (the depth and stencil tests run before its write: its
    /// module declares early tests, or it states [`TestedStorageWrites`](Self::TestedStorageWrites));
    /// `master`: the fingerprint of the master
    /// layout its views view (`FragmentImageblock::master`); `data_size`: the cell's bytes
    /// (`air.imageblock_data_size`); `alignment`: the alignment Metal states for its views
    /// (`air.arg_type_align_size`, `FragmentImageblock::alignment`).
    FragmentImageblock {
        words: u32,
        writes: bool,
        master: Hash256,
        data_size: u32,
        alignment: u32,
    },
    /// The kernel is a tile function (`Device::compile_tile_pipeline`): a dispatch covers the
    /// render pass's render area (`BindState::render_area`, `SpecKey::RenderAreaWidth` and
    /// `SpecKey::RenderAreaHeight`) in tiles of the imageblock size (`BindState::imageblock_size`),
    /// one threadgroup per tile, and stages the color attachments it reaches through the
    /// threadgroup's imageblock. A layered pass is not modeled: the planner refuses a render
    /// target array length above 1.
    TileFunction,
    /// The tile function stages color attachment `color` as a storage image of pixel format
    /// `format` (`LoweringInputs::tile_attachments`): its texel conversions and declared image
    /// format are that format's. The binder views the attachment in that format's staging format
    /// (`WriteSource::TileAttachment`), and a tile pipeline whose descriptor gives the attachment
    /// another format is `InvalidDescriptor(ColorAttachment)`.
    TileAttachmentFormat {
        color: u32,
        format: crate::metal::PixelFormat,
    },
    /// The function's texels at the storage texture bound to the slot take the numeric type
    /// (signed or unsigned) of its pixel format, which nothing in the function states: an integer
    /// imageblock slice element spells no signedness, and no other operation on the slot names
    /// one. The type is the slot's `StorageImageChoice` input (`StorageImageChoice::numeric` where
    /// the choice declares no format), so the binder keys the slot's choice even where a formatless
    /// declaration would serve its format.
    StorageTextureNumericType { slot: SlotRef },
    /// The function reads the texture bound to the slot (a texture or texture buffer) under the
    /// in-shader read check and uses the texel's alpha: Metal's out-of-bounds read is the zero
    /// texel with the pixel format's missing alpha read as 1 (MSL 4.1 sections 2.9.1 and 6.13),
    /// so the shader takes that alpha from `SpecKey::OutOfBoundsAlpha`, which the binder sets from
    /// the bound texture's pixel format.
    OutOfBoundsAlpha { slot: SlotRef },
    /// The texture bound to the slot is sampled through Metal's bicubic filter, which the
    /// function emulates by fetches: how the filter treats the bound format's texels (bilinear
    /// or Catmull-Rom, the clamp or rounding of the sum, the channels a border tap reads) is the
    /// slot's `SpecKey::BicubicTexels` specialization (`metal::bicubic::texels`), and the planner
    /// refuses a bound format of no measured family.
    BicubicTexels { slot: SlotRef },
    /// The function reads the mipmap level count of the storage texture bound to the slot
    /// (`PushField::TextureLevels`): Metal's `get_num_mip_levels` on a `write` or `read_write`
    /// texture, which a storage image cannot query (`OpImageQueryLevels` needs a sampled image),
    /// and the bound of the slot's level views. The binder writes the bound texture's
    /// `mipmapLevelCount`, and 0 where it binds a neutral.
    StorageTextureLevels { slot: SlotRef },
    /// The slot's binding is an array of single-level storage views: element `i` views level `i` of
    /// the bound texture (all its layers), and every element past its last level the neutral of the
    /// declared type. A storage access reaches its view's base level only, so a read, write, or
    /// size query at a level other than the constant 0 indexes the array by the level. The array
    /// holds as many elements as a texture of the slot's type can have levels on the device; the
    /// binder refuses a texture with more.
    StorageTextureLevelViews { slot: SlotRef },
    /// The function observes the residency of its reads, samples, or gathers of the texture
    /// bound to the slot: bit 0 of their `{T, i8}` status word, MSL's
    /// `sparse_color::resident()`. Whether that texture is sparse is a fact of the bound texture
    /// (`MTLTexture.isSparse`, `bind::TextureFacts::sparse`). A texture that is not sparse is
    /// resident everywhere, and the status word Metal returns for it is compiled in, exact on
    /// every device. `sparse_arm`: the module also answers a sparse texture, with the device's
    /// sparse residency (`OpImageSparse*` and `OpImageSparseTexelsResident` under
    /// `shaderResourceResidency`), selected by `SpecKey::SparseTexture`. Without it the binder
    /// refuses a sparse texture at the slot (`BindError::Unsupported`): naming
    /// `shaderResourceResidency` where the device lacks it, `NotImplemented` where one of the
    /// slot's accesses has no sparse form (a bicubic sample), and `NotRepresentable` where the
    /// function updates the slot atomically (`SparseTextureWrites`). A slot the function writes
    /// takes either arm, its writes dropped.
    ///
    /// `tile_map` (on every device): where the slot's type has a tile map (a single-sample 2D, 2D
    /// array or 3D color texture) and each observed access names its texels, the module answers a
    /// sparse texture from the slot's tile map instead (`ResourceKind::TileMap`,
    /// `bind::SparseTileMap`), selected by the same `SpecKey::SparseTexture`: each access is the
    /// plain one, its residency the bits of the tiles it reads (for a sample or gather, the texels
    /// Metal's footprint rule counts), and a read of a texel that is not resident Metal's unmapped
    /// value, the out-of-bounds texel. At most one of `sparse_arm` and `tile_map` holds:
    /// `sparse_arm` only where the slot has no tile map. With neither, the binder refuses a sparse
    /// texture at the slot as above (an access the tile map does not answer on a device without
    /// `shaderResourceResidency`: a filtered sample or a gather outside Metal's measured rule, or a
    /// slot updated atomically).
    SparseTexture {
        slot: SlotRef,
        sparse_arm: bool,
        tile_map: bool,
    },
    /// The function reads the instance count of the instance acceleration structure bound at the
    /// buffer slot (`get_instance_count()`, `PushField::InstanceCount`): no Vulkan instruction
    /// reads a structure's instance count, and the consumer that built the Metal structure knows
    /// it, so the binder writes the count `StageTable::acceleration_structures` states for the
    /// slot, and 0 for an unbound slot (the structure of no instances the binder stands in for
    /// one). A slot read only this way needs no acceleration-structure descriptor, so no ray
    /// tracing feature.
    AccelerationStructureInstanceCount { slot: SlotRef },
    /// The function reads color attachment `color`, and writes it when `writes`, as a storage
    /// image inside the ordered pixel interlock: framebuffer fetch on a device that orders it
    /// only that way (no `rasterizationOrderColorAttachmentAccess`, with
    /// `fragmentShaderPixelInterlock`), where its render pipeline's state
    /// (`LoweringInputs::framebuffer_fetch`) lets the form be exact
    /// (`GraphicsState::storage_attachments`). `format` is the attachment's Vulkan format, which
    /// the module's storage image declaration serves. `early_tests`: the module declares early
    /// fragment tests: Metal's function declares them, or its pipeline state has a depth or
    /// stencil attachment and they keep its writes Metal's (it runs no discard, exports no depth,
    /// stencil, or coverage, and writes no other memory), or its draws' tests write nothing
    /// (`TestedStorageWrites`). Link checks the form against the pipeline it links into: one whose
    /// state would not choose it, would view another format, or needs early tests the module does
    /// not declare (a depth or stencil attachment where it was compiled without one, but for
    /// `TestedStorageWrites`) is `NeedsInput(FramebufferFetch)`. `kept`: the format's channels each
    /// write keeps from the texel it overwrites, those outside the descriptor's write mask, which
    /// the write takes from the texel it reads first (all four where the mask covers none of the
    /// format's channels, so it stores nothing; empty where it covers every one, or the function
    /// does not write the attachment); a pipeline of another mask is `NeedsInput(FramebufferFetch)`
    /// too. `blend`: the blend the function computes in the shader, from the texel it reads first,
    /// for an extended-range attachment it writes under blending, which the function then need not
    /// read, or a 16-bit float one (Metal's blend in the output's arithmetic); a pipeline of another
    /// blend there is `NeedsInput(FramebufferFetch)` too. `rounded`: that blend is computed
    /// without fast math, each product rounded before the sum
    /// (`lower::arith::extended_range::blend::BlendMath::Rounded`), whose model holds blends the
    /// fast one refuses; link checks the blend under that math.
    InterlockedFramebufferFetch {
        color: u32,
        format: crate::vk::Format,
        writes: bool,
        early_tests: bool,
        kept: crate::vk::ColorComponentFlags,
        blend: Option<crate::pipeline::lower_inputs::BlendEquation>,
        rounded: bool,
    },
    /// The function's `[[stencil]]` output is a shader-side stencil: on a device without
    /// `VK_EXT_shader_stencil_export`, with the ordered pixel interlock, the function applies
    /// Metal's stencil test and operations to a shadow of the stencil attachment inside the
    /// interlock (`SyntheticBinding::ShadowStencil`, `PushField::ShadowStencilState`) and kills a
    /// fragment that fails (`GraphicsState::shadow_stencil`). `formatless`: the module declares
    /// the shadow without a format (`Unknown`), else as `R8ui`. `multisampled`: compiled for a
    /// pipeline of more than one sample (`LoweringInputs::multisampled`), the shadow is a
    /// multisampled image and the function tests each sample of its coverage.
    ShadowStencil {
        formatless: bool,
        multisampled: bool,
    },
    /// The kernel's predicated walk votes in a workgroup word of its own
    /// (`lower::simd::divergent::VOTE_BYTES`, 4 bytes beside its threadgroup arena): it runs a
    /// workgroup barrier that part of the workgroup skips in the source in every invocation, and
    /// a loop holding one iterates while any invocation is still in it. The planner counts the word
    /// with the arena's static part and `[[threadgroup(n)]]` lengths against
    /// `maxComputeSharedMemorySize`, and the reflected static threadgroup length includes it. A
    /// kernel in the emulated tier votes in its scratch instead
    /// (`RuntimeFact::SimdgroupEmulation`).
    PredicationVote,
    /// The function reads the class of the texture (or texture buffer) bound to the slot, which
    /// the binder sets from the bound format: the slot's `SpecKey::TexelFormat` specialization
    /// (`metal::texel_format::texel_format`). A read, gather or sample with a `half` result on a
    /// device that samples through Metal's own sampler converts its texel by it, Metal's conversion
    /// depending on the format (a read of a 32-bit float texel rounds to nearest even, a sample of
    /// one toward zero; a normalized texel's value is rounded onto its format's grid first); an
    /// 8-bit sRGB slot's raw read, a fetched gather and a NaN texel's rule (the NaN switch) read
    /// it too.
    TexelFormat { slot: SlotRef },
    /// The function reads, samples, or gathers the texture bound to the slot with a `half` or
    /// `float` result, which an 8-bit sRGB format bound there would decode. Bound in such a
    /// format that the device cannot sample, which the runtime holds as the format's UNORM twin
    /// (`metal::format::srgb_raw_twin`, `metal::texel_format::srgb_twin_on`), the slot joins the
    /// plan's variant (`VariantKey::srgb_textures`, `LoweringInputs::srgb_textures`), under which
    /// the function decodes the codes its own binding reads by the MSL transfer through the
    /// device's `Pow` (`PrecisionNote::SrgbTexelDecode`).
    /// Bound in any other format, or unbound, the slot leaves the variant, and the function reads
    /// the device's decode through the texture's own view.
    SrgbTexture { slot: SlotRef },
    /// A `ulong` atomic (`atomic_{min,max}`) updates the `ulong` texture bound to the slot, so the
    /// slot is declared as an `R64ui` storage image: the binder views the bound `RG32Uint` texture
    /// as `R64_UINT`, each texel's low word in R and high word in G one little-endian 64-bit
    /// integer. A texture is that view of an image the consumer created with `MUTABLE_FORMAT`, and
    /// a texture buffer an `R64_UINT` texel view of its bytes; an unbound slot takes the `R64_UINT`
    /// neutral. A `ulong` slot without the fact is an `RG32` image.
    UlongAtomicView { slot: SlotRef },
    /// The vertex function writes `[[viewport_array_index]]`, which selects a viewport only below
    /// the render encoder's viewport count (`setViewports:count:`): Metal draws a primitive whose
    /// index is at or past the count through viewport 0, so the function stores `ViewportIndex`
    /// as 0 there, reading the count from `PushField::ViewportCount`, which the binder writes per
    /// draw (`BindState::viewport_count`).
    ViewportCount,
    /// The tile function aliases an explicit imageblock onto its implicit one
    /// (`air.alias_implicit_imageblock`), of `data_size` bytes (`air.imageblock_data_size`): a
    /// tile pipeline state's `imageblockSampleLength`, which Metal answers with that size
    /// whatever the attachments pack to (measured on an M3 Max and an M4). Nothing on the device
    /// depends on it.
    AliasedImageblock { data_size: u32 },
    /// The stage recurses through visible functions on a call stack sized for its pipeline's
    /// `maxCallStackDepth` for it, `depth` (`LoweringInputs::max_call_stack_depth`, at least 1):
    /// a compute or tile pipeline's, or a render pipeline's vertex or fragment depth. A
    /// `StageSource::Compiled` stage serves only a pipeline of that depth for it
    /// (`InvalidDescriptor(MaxCallStackDepth)` otherwise); a `Function` stage is compiled under
    /// its pipeline's. Nothing on the device reads it.
    CallStackDepth { depth: u32 },
    /// The kernel was compiled under `VariantKey::flat_grid` (`LoweringInputs::flat_grid`): it
    /// runs any grid in threadgroups of its own size, reading the grid from the 32-byte region
    /// record at its `PushField::ReplayRegionAddress`, which a preprocess writes beside the
    /// dispatch's group counts. Its plans state the variant.
    FlatGrid,
    /// The function writes texture slot `slot` (a `write` or an imageblock slice's copy), or,
    /// with `atomic`, updates it by a texture atomic. Metal drops every write a function makes to
    /// a sparse texture, its mapped tiles' included: the texture keeps its contents for the
    /// writing thread, the threadgroup's other threads, later work, and blits (Apple M4 on macOS
    /// 27.0.1 and M3 Max on macOS 26.7; Metal's API validation stops the texture's `ShaderWrite`
    /// usage instead). Under the slot's `SpecKey::SparseTexture` specialization, 1 where the bound
    /// texture is sparse, the module drops the slot's writes. A texture atomic on a sparse texture
    /// has no deterministic answer on Metal (the value it returns changes between processes, and it
    /// zeroes texels it does not address), so with `atomic` the binder refuses a sparse texture at
    /// the slot (`Unsupported { feature: SparseTexture { slot }, needs: NotRepresentable }`).
    SparseTextureWrites { slot: SlotRef, atomic: bool },
    /// The tile function reads its threadgroup argument `index` (`[[threadgroup(index)]]`):
    /// memory of its tile that the render pass keeps across its tile dispatches, at the offset
    /// and in the tile memory the encoder sets (`setThreadgroupMemoryLength:offset:atIndex:` and
    /// the pass's `threadgroupMemoryLength`, `BindState::tile_threadgroup`). The binder
    /// specializes `SpecKey::TgOffset(index)`, `SpecKey::TileThreadgroupLength` and
    /// `SpecKey::TileGridWidth`, binds the pass's `SyntheticBinding::TileMemory`, and refuses an
    /// index the encoder did not set (`BindError::MissingThreadgroupMemory`). A fragment
    /// function's threadgroup arguments are the same memory: its draws read and write it in the
    /// buffer, at its tile, which `SpecKey::ImageblockWidth` and `SpecKey::ImageblockHeight` (the
    /// pass's tile size) place.
    TileThreadgroupMemory { index: u32 },
    /// The function reads texture buffer slot `slot` (not as `ulong` texels), and applies the
    /// bound texture buffer's swizzle to each read under the slot's `SpecKey::TextureSwizzle`
    /// specialization: Metal returns each channel from the source the swizzle names, a missing
    /// channel and an out-of-bounds texel included (Apple M4 and M3 Max). The binder specializes
    /// it, and refuses a swizzled texture buffer at a slot whose stage reads texels without this
    /// fact, as `ulong` ones (`Unsupported { feature: TextureSwizzle }`); a slot only queried or
    /// written binds one. Under `LoweringInputs::swizzled_storage` the function states it too for a
    /// storage slot it reads or writes texels of: it applies the bound texture's swizzle to each
    /// read and undoes it on each write; and for an extended-range sampled slot of it, whose
    /// decoded texels its reads, samples and gathers swizzle, and a packed normalized sampled slot
    /// of it, whose converted texels they swizzle.
    TextureSwizzle { slot: SlotRef },
    /// The tile function declares an explicit imageblock that stands alone (no implicit one
    /// beside it, and not aliased onto one), of `data_size` bytes (`air.imageblock_data_size`)
    /// whose layout aligns to `alignment` (`air.arg_type_align_size`). Its cells are the render
    /// pass's imageblock storage (`SyntheticBinding::Imageblock`), which the pass's fragment
    /// functions' imageblocks are in and the consumer keeps for the pass: a dispatch whose plan
    /// writes `WriteSource::Imageblock` binds it, and the pass's draws and dispatches share it
    /// by its bytes. With the color attachments it gives the tile pipeline state's
    /// `imageblockSampleLength` as a fragment function's explicit imageblock does (measured on an
    /// M4).
    TileImageblock { data_size: u32, alignment: u32 },
    /// The vertex function was compiled under `VariantKey::amplification_draws`
    /// (`LoweringInputs::amplification_draws` of these fields): its pipeline draws each vertex
    /// amplification as a draw of its own, amplification-major as Metal renders them (a Metal probe
    /// on the M3 Max and an M4), and reads the amplification and its view mapping's offsets from
    /// `PushField::AmplificationId`, `AmplificationLayerOffset` and `AmplificationViewportOffset`,
    /// which the binder writes before each of those draws. With `layered` it writes `Layer` (the
    /// pass has at least two layers), and with `viewports` it writes `ViewportIndex` where it
    /// writes no viewport index of its own (a mapping offsets the viewport). Its plans state the
    /// variant the shape's amplification asks for.
    AmplificationDraws { layered: bool, viewports: bool },
    /// The vertex function was compiled under `VariantKey::viewport_compensation`
    /// (`LoweringInputs::viewport_compensation` of `depth`): it negates its position's `x` and,
    /// under `depth`, rewrites its depth and clips at Metal's planes by the draw's
    /// `PushField::ViewportCompensation`, which the binder writes from
    /// `BindState::viewport_compensation`. Its plans state the variant the shape asks for.
    /// Under `point_guard` it also scales its position by the draw's `PushField::PointGuard`
    /// and culls a point past Metal's guard band by four clip distances. Under `mirrors` it negates
    /// `x` by the selected viewport's bit of the draw's `PushField::ViewportMirrors`. Under
    /// `vertex_clamp` it clamps its compensated depth by the draw's `PushField::VertexDepthClamp`
    /// before the interpolation, as Metal clamps each vertex's depth.
    ViewportCompensation {
        depth: bool,
        point_guard: bool,
        mirrors: bool,
        vertex_clamp: bool,
    },
    /// The kernel was compiled under `VariantKey::folded_groups`
    /// (`LoweringInputs::folded_groups`): it runs a threadgroup dispatch of its own threadgroup
    /// size whose group counts it reads from the 32-byte region record at its
    /// `PushField::ReplayRegionAddress`, which a preprocess writes beside the dispatch's group
    /// counts, folded within the device's limits. Its plans state the variant.
    FoldedGroups,
    /// The vertex or fragment function was compiled under `VariantKey::replay_bindings`
    /// (`LoweringInputs::replay_bindings`): every buffer it reads is in the slot table at its
    /// stage's `PushField::RootTableAddress`, which a draw replay's preprocess writes per
    /// replayed command (`bind::draw_replay_preprocess`). Its plans state the variant.
    ReplayBindings,
    /// The vertex function was compiled under `VariantKey::merged_draws`
    /// (`LoweringInputs::merged_draws`): it runs a merged draw replay's corners, reading its
    /// command, vertex, instance and slot table from the header at its
    /// `PushField::MergedDrawAddress` (`bind::replay`'s `merged.rs`). Its plans state the variant.
    MergedDraws,
    /// The `stage` function may read an argument its pipeline's function constants removed (a
    /// texture's read, sample, gather, query, or atomic, or a sample through a removed sampler),
    /// a read Metal gives no value. Where one runs, the function takes zero for it and sets
    /// `StatusBits::ABSENT_ARGUMENT_READ` in the command buffer's `ExecutionStatus` buffer (through
    /// the address the stage's `ExecutionStatus` fact asks for); a consumer that reads the bit
    /// after the command buffer reports its execution as failed, since none of its results is then
    /// Metal's. A run that never reaches such a read is Metal's exactly. Nothing binds it.
    AbsentArgumentRead { stage: Stage },
    /// The fragment function was compiled under `VariantKey::zeroed_colors`
    /// (`LoweringInputs::zeroed_colors`): it writes zero to color attachment `color`, of pixel
    /// format `format`, which its Metal pipeline lacks and the pass has, as plain Metal stores
    /// zero bytes there at every sample a draw covers. Its plans state the variant the shape asks
    /// for (`BindShape::zeroed_colors`).
    ZeroedColor {
        color: u32,
        format: crate::metal::PixelFormat,
    },
    /// The fragment function was compiled under `LoweringInputs::dropped_colors` holding
    /// `color`: it keeps no output at that color attachment, which the render pass lacks and
    /// whose writes plain Metal drops. A plan of a draw whose pass has the attachment asks for the
    /// variant without it.
    DroppedColor { color: u32 },
    /// The fragment function was compiled under `VariantKey::depth_clamp_range`
    /// (`LoweringInputs::depth_clamp_range`) and writes its depth: it clamps that depth to the
    /// draw's depth clamp range itself, read from `PushField::DepthClampRange`, which the binder
    /// writes from `BindState::depth_clamp_range`, a negative NaN storing 0 as Metal's does. Its
    /// plans state the variant the shape asks for (`BindShape::depth_clamp_range`).
    DepthClampRange,
    /// The fragment function was compiled under `VariantKey::rotated_weights`
    /// (`LoweringInputs::rotated_weights`) and reads `[[barycentric_coord]]`: it takes Metal's
    /// weights back from a triangle drawn with Metal's last vertex first. Its plans state the
    /// variant the shape asks for (`BindShape::rotated_weights`).
    RotatedWeights,
    /// The emulated mesh function's pipeline was compiled under `VariantKey::mesh_last_first`:
    /// its synthesized vertex stage draws each line or triangle from its last index first. Its
    /// plans state the variant the shape asks for (`BindShape::mesh_last_first`).
    MeshLastFirst,
    /// The function reads, samples, gathers, or fetches through a sampler the color texture
    /// bound to the slot, on a profile whose sparse textures are sparse images (it enables
    /// `shaderResourceResidency`): where that texture is sparse, a block of it that holds no
    /// mapped tile reads the device's value, which may differ from Metal's in the lanes the
    /// format lacks (RADV's alpha 0), so the function sets the lanes Metal reads as constants
    /// after the access, from `SpecKey::ConstantLanes`, which the binder sets from the bound
    /// texture's `MTLTexture.pixelFormat` and `swizzle` (0 where `isSparse` is false).
    ConstantLanes { slot: SlotRef },
    /// The function reads visible-function tables by their resource ids: a listed table whose words
    /// pass `maxStorageBufferRange` cannot be the function-table array's element, nor the snapshot
    /// hold every listed table's words, so the plan of a shape whose tables may pass it
    /// (`FunctionTablesShape::bytes_class`) asks for the variant that reads every table's words
    /// through its device address (`VariantKey::addressed_function_tables`). `addressed` states
    /// that this compile is that variant.
    FunctionTableRange { addressed: bool },
    /// The fragment function stores some channels of a color attachment through the interlocked
    /// storage form (`InterlockedFramebufferFetch`) in a pipeline with a depth or stencil
    /// attachment, and discards, but declares no `[[early_fragment_tests]]`, exports no depth,
    /// stencil, or coverage, and writes no other memory: Metal tests its fragments after it runs,
    /// so whether its write stores is the draw's depth and stencil tests' to decide. Without
    /// `early_tests` its module stores before Vulkan's tests, Metal's store wherever the draw's
    /// tests drop no fragment; with it (`LoweringInputs::read_only_tests`) the module declares
    /// early fragment tests, Metal's store wherever they write neither attachment and count no
    /// visibility result. The plan of a draw whose tests drop fragments asks for the variant with
    /// `early_tests` where they write and count nothing (`VariantKey::read_only_tests`), and
    /// refuses any other (`BindShape::fragment_tests`).
    ///
    /// `early_variant`: whether that variant serves a draw whose tests drop fragments but write and
    /// count nothing, which it does for a function that only its discard keeps from early tests. A
    /// function that writes other memory, or returns depth, stencil, or a sample mask, beside a
    /// depth or stencil attachment states the fact without it: Metal runs its memory writes for the
    /// fragments its tests then drop, and tests what it returns, which early tests would not. Its
    /// stores before the tests are Metal's only where the draw's tests drop no fragment, and the
    /// plan refuses every other draw.
    TestedStorageWrites {
        early_tests: bool,
        early_variant: bool,
    },
    /// On a device that orders framebuffer fetch only through the ordered pixel interlock, the
    /// fragment function writes
    /// a color through its first output and does not declare early fragment tests, and it
    /// discards, writes other memory, or returns depth, stencil, or a sample mask. A draw may move
    /// such a color into the interlocked storage form (`BindShape::interlocked_colors`, or an
    /// extended-range color's `BindShape::extended_shadows`), whose variant then stores before the
    /// depth and stencil tests and states [`TestedStorageWrites`](Self::TestedStorageWrites) with
    /// this `early_variant`, where the write stores something. So the plan of a pipeline that
    /// writes no color so selects the tests of such a draw from this as from that fact, and the
    /// variant it names plans itself.
    ///
    /// The plan reads it only for a draw that moves a color the function writes as an ordinary
    /// output under a write mask that names some of the attachment's components (and only where
    /// the pipeline has a depth or stencil attachment, whose tests the variant then depends on). A
    /// write whose mask names none of them stores nothing: no test bears on it, its variant states
    /// no store before the tests, and the plan selects none for it, as for a draw that moves
    /// nothing. A draw whose tests neither the early-tests variant nor the function's own tests
    /// serve (tests the draw does not state) is refused, as any draw whose variant is refused: the
    /// runtime keeps the color out of the storage form there (the pass splits, or the shadow ends).
    MovableStorageWrites { early_variant: bool },
    /// The tile function reads a builtin whose answer depends on its pipeline's
    /// `threadgroupSizeMatchesTileSize` (its threads per threadgroup, simdgroups or quadgroups
    /// per threadgroup or their `dispatch_` forms, its position in the grid, the threads per
    /// grid, or its thread or quadgroup index in the threadgroup), and was compiled under
    /// `tile_sized`, that flag (`LoweringInputs::threadgroup_size_matches_tile_size`). With
    /// `tile_sized` the binder runs each dispatch with the pass's tile, its imageblock size
    /// (`BindShape::imageblock_size`), as Metal's threadgroup in every region (`MultiRegion` 1),
    /// the threads per tile its workgroups; a tile pipeline whose descriptor gives the flag the
    /// other value is `InvalidDescriptor(TileThreadgroupSize)`.
    TileThreadgroup { tile_sized: bool },
    /// The function samples the texture bound to the slot through a constexpr sampler of address
    /// modes `modes` (on `s`, `t` and `r`, as Metal names them), whose state the pipeline holds
    /// (its immutable sampler, and the modes its emulations read): a texture whose address-mode
    /// map (`TextureFacts::address_modes`, the runtime's extension) changes one of them puts the
    /// slot in the plan's variant with the map restricted to the modes the slot's constexpr
    /// samplers state (`VariantKey::address_maps`, `LoweringInputs::address_maps`), under which the
    /// constexpr samplers sample it with their modes mapped, as a runtime sampler is mapped at
    /// bind. A texture whose map keeps every such mode, every ordinary texture's among them, leaves
    /// the slot out, so it keeps every key.
    ConstexprSampled {
        slot: SlotRef,
        modes: [crate::metal::SamplerAddressMode; 3],
    },
    /// The function samples, gathers, or queries a level of a texture through the runtime sampler
    /// bound to the slot, whose min or mag filter may be Metal's `Cubic` (raw value 2): Metal then
    /// samples through the sampler as through a constexpr bicubic sampler of the same state,
    /// whatever the level. Bound to such a sampler, the slot joins the plan's
    /// variant (`VariantKey::cubic_samplers`, `LoweringInputs::cubic_samplers`) with the bound
    /// word's `SamplerWord::CUBIC_STATE` bits, under which the function emulates the bicubic
    /// filter of that state (`lower::sampler::bicubic`) or refuses what it does not measure. Bound
    /// to any other sampler, or to none, the slot leaves the variant.
    CubicSampler { slot: SlotRef },
    /// The `stage` function samples, gathers, or queries through samplers an argument buffer
    /// holds, and tests each one's table entry for a cubic filter (`SamplerWord::CUBIC`): where
    /// one runs through a sampler whose min or mag filter is `Cubic`, whose bicubic filter it does
    /// not emulate there, it sets `StatusBits::CUBIC_ARGUMENT_SAMPLER` in the command buffer's
    /// `ExecutionStatus` buffer (through the address the stage's `ExecutionStatus` fact asks
    /// for), and a consumer that reads the bit reports the command buffer's execution as failed. A
    /// run that never reaches such a sampler is Metal's exactly. The planner admits a cubic listed
    /// state (`BindState::argument_samplers`) beside a stage's sampler table only where the stage
    /// states this fact (a device that cannot write the buffer from the stage tests nothing, and
    /// its stage is refused `NotImplemented(SamplerBicubicFilter)` there). A fragment stage never
    /// states it: the mark is a store, which costs a fragment function the device's hidden-surface
    /// removal and early depth test although it never runs.
    CubicArgumentSampler { stage: Stage },
    /// The function runs the draw's depth test itself (`LoweringInputs::shader_tests`): compiled
    /// for a draw whose tests may drop fragments, a function with `TestedStorageWrites` tests its
    /// fragment's depth against a stand-in of the depth attachment inside the ordered pixel
    /// interlock (`SyntheticBinding::DepthStandIn`, `PushField::ShaderTestsState`), stores its
    /// colors and writes the stand-in only where it passes, and kills a fragment that fails
    /// (`GraphicsState::shader_tests`). `formatless`: the module declares the stand-in without a
    /// format (`Unknown`), else as `R32f`. `multisampled`: compiled for a pipeline of more than
    /// one sample, the stand-in holds each sample, `planes` as a single-sample 2D array of one
    /// layer per sample, else as a multisampled image.
    ShaderTests {
        formatless: bool,
        multisampled: bool,
        planes: bool,
    },
    /// The `stage` function samples, gathers, or queries a level of a heap texture of a sampled
    /// float class of one sample, where for a texture a heap holds by its raw texels alone
    /// (`bind::heap_raw_form`) it reads the class's neutral (zero, or the neutral's level) where
    /// Metal samples the texture. The function emulates Metal's sample or gather of such a
    /// texture from its raw texels where the access is a 2D texture's sample or gather of a
    /// `float` or `half` result (a `half` gather of a bfloat form alone) without comparison,
    /// `min_lod_clamp`, a dynamic offset, or observed residency, through a constexpr sampler
    /// with one filter for minification and magnification, no mip filter, and no anisotropy,
    /// which states no fact; or through the runtime sampler in slot `sampler` of the stage, which
    /// states the fact naming it: the emulation follows the slot's word, and reads the neutral
    /// where the word states a mip filter, anisotropy, or two filters. Every other such access
    /// states it with `sampler` `None`, an argument buffer's sampler's among them. The planner
    /// refuses a texture a heap holds by its raw texels alone that the stage reaches (a member
    /// of the plan it names, or one resident for the command) where the stage states the fact
    /// with `None`, or naming a slot whose bound sampler states what the emulation does not
    /// answer (`Unsupported(ArgumentBufferTextures)`).
    HeapRawSamples { stage: Stage, sampler: Option<u32> },
    /// Beside its depth test (`ShaderTests`), the function runs the draw's stencil test itself,
    /// before the depth test, against a stand-in of the stencil attachment inside the ordered
    /// pixel interlock (`SyntheticBinding::StencilStandIn`, `PushField::ShaderStencilState`): the
    /// face's comparison of the reference with the stand-in's value under the read mask, then
    /// its stencil-failure, depth-failure or pass operation under the write mask, as the two
    /// tests decide. A fragment that fails either test is killed (`ShaderTestsUse::stencil`).
    /// `formatless`: the module declares the stand-in without a format (`Unknown`), else as
    /// `R8ui`. The stand-in holds the pipeline's samples as a multisampled image at more than one.
    ShaderStencilTest { formatless: bool },
    /// On a device that orders framebuffer fetch through rasterization-order input attachments,
    /// the function computes Metal's blend `blend` into extended-range color attachment `color`
    /// in the shader: it reads the attachment's raw texel through its input attachment
    /// (`RuntimeFact::FramebufferFetch`), blends its output over the texel's tile value, and
    /// writes the encode of the result through its first output, the channels `kept` (the
    /// format's channels the descriptor's write mask leaves out) taken from the texel. Link
    /// gives the attachment no Vulkan blend and writes every raw component, and a pipeline of
    /// another blend or write mask there is `NeedsInput(FramebufferFetch)`.
    ExtendedRangeInputBlend {
        color: u32,
        blend: crate::pipeline::lower_inputs::BlendEquation,
        kept: crate::vk::ColorComponentFlags,
    },
    /// On a device that orders framebuffer fetch through rasterization-order input attachments,
    /// the function writes color attachment `color`, which renders through a view of its word
    /// (`metal::format::WordAttachment`, an 8-bit sRGB one), under a write mask of some of the
    /// word's channels: it reads the texel's word through the attachment's input attachment
    /// (`RuntimeFact::FramebufferFetch`) and writes Metal's word of its output with the fields of
    /// the channels `kept` (those of the word the descriptor's write mask leaves out) taken from
    /// the word read. Link writes the whole word, and a pipeline of another write mask there is
    /// `NeedsInput(FramebufferFetch)`.
    WordInputMerge {
        color: u32,
        kept: crate::vk::ColorComponentFlags,
    },
    /// The function reads, samples or gathers the sampled color texture (not a texture buffer,
    /// a multisampled or a depth texture) bound to the slot only with `float`, `half` or 32-bit
    /// integer results: bound to a texture of a color format whose numeric class is not the
    /// slot's declared one (`metal::format::sampled_type`), the slot joins the plan's variant
    /// (`VariantKey::cross_class`, `LoweringInputs::cross_class`), under which the function
    /// declares the image of the bound class, takes each access's result of that class and
    /// hands back its 32-bit words as the declared type (a `half` result each word's low 16
    /// bits), as plain Metal answers. A slot without the fact (a `short` or `ushort` result,
    /// say), or one bound to a format the variant does not take, binds the declared type's
    /// neutral, and the plan says so (`PlanDiagnostic::CrossClassNeutral`).
    CrossClassReads { slot: SlotRef },
    /// The fragment function, compiled under `TextureWriteRoundingMode::NearestEven`, writes the
    /// storage texture or texture buffer bound to the slot, whose storage choice declares no
    /// format, by a write Metal's mode 2 stores otherwise there and the translation has not built:
    /// a `half` texel into an `RG11B10Float` or `RGB9E5Float` texture, and where `halves` (a
    /// `float` texel) a half-float one too (`metal::texel_format::half_channels`). The planner
    /// refuses the slot bound to such a texture
    /// (`NotImplemented(PrivateRenderState(TextureWriteRounding))`); any other texture and a
    /// neutral draw as Metal's.
    StorageTextureWriteRounding { slot: SlotRef, halves: bool },
    /// The `stage` function, compiled under `TextureWriteRoundingMode::NearestEven`, writes heap
    /// textures by such writes (`StorageTextureWriteRounding`, `halves` likewise). The planner
    /// refuses a texture of such a format that the stage reaches in a storage class of the plan
    /// (a member it names, or one resident for the command) with the same refusal.
    HeapWriteRounding { stage: Stage, halves: bool },
    /// The `stage` function reads members of the heap array of `class`, its group's own, and a
    /// member of another numeric class than the class's reads the neutral there, where plain
    /// Metal reads it in its own class: where `every_member`, every such member (an access has
    /// no answer for any: a sample, gather or LOD query, a write or atomic, a storage or
    /// multisampled group's read, a `short`, `ushort` or `ulong` read); else only one of a format
    /// whose read in another class is not measured (`metal::texel_format::crosses_classes`: a
    /// depth, stencil, extended-range or two-plane one), since the group's reads and queries
    /// read every other one in its own class (`metal::texel_format::heap_numeric_base`). The
    /// planner reports each such member the stage reaches there
    /// (`PlanDiagnostic::CrossClassHeapNeutral`), which the runtime announces.
    HeapCrossClassNeutral {
        stage: Stage,
        class: ImageHandleClass,
        every_member: bool,
    },
    /// The function samples a texture through the runtime sampler bound to the slot where Metal's
    /// anisotropic filter, emulated (`lower::sampler::anisotropic`), reaches the sample: bound to a
    /// sampler whose word states anisotropy (`SamplerWord::ANISOTROPIC`, `maxAnisotropy` above 1),
    /// the slot joins the plan's variant (`VariantKey::anisotropic_samplers`,
    /// `LoweringInputs::anisotropic_samplers`), under which each such sample takes the emulation
    /// where the slot's word states anisotropy and the plain sample elsewhere. Bound to any other
    /// sampler, or to none, the slot leaves the variant, and the function samples there as the
    /// word's other fields say, with no emulation in its module.
    AnisotropicSampler { slot: SlotRef },
    /// The function writes integer texels to the storage texture (or texture buffer) bound to the
    /// slot, whose storage choice declares no format, and reads the width of each of its channels
    /// (`PushField::TextureIntegerChannels`): Metal saturates each lane of an integer write to its
    /// channel's range as the format's signedness reads it (`uint` 300 into `R8Uint` stores 255),
    /// where a Vulkan write of a value the channel cannot represent is undefined. The binder
    /// writes `metal::format::integer_channels_word` of the bound format, and 0 for a neutral.
    StorageTextureIntegerChannels { slot: SlotRef },
    /// The render pipeline draws through synthesized geometry stages, one per topology class, of
    /// which a draw runs its class's (`RenderPipeline::runs_stage`): its fragment function reads
    /// `[[barycentric_coord]]` on a device without `fragmentShaderBarycentric` (`weights`), or its
    /// vertex function selects its primitive's render target slice (`layer`) or viewport
    /// (`viewport`) on a device that cannot store `Layer` or `ViewportIndex` from a vertex
    /// function (`pipeline::link::geometry`). Each draw runs a stage Metal's draw does not, at the
    /// stage's cost, with Metal's answer; the consumer announces it. Nothing binds it.
    SynthesizedGeometry {
        weights: bool,
        layer: bool,
        viewport: bool,
    },
    /// An object function's `[[threadgroup(n)]]` arguments, whose lengths the render encoder sets
    /// (`setObjectThreadgroupMemoryLength:atIndex:`, the object stage's
    /// `StageTable::threadgroup_lengths`), each a `ThreadgroupMemoryLength`: the static part of
    /// its threadgroup arena (`memory::arena::TgShape::static_bytes`) and its payload's bytes
    /// (`lower::mesh::payload_extent`), beside which the binder places the lengths, each rounded
    /// up to 16, and checks the total against the object stage's workgroup memory limit
    /// (`lower::mesh::stage_workgroup_limit`).
    ObjectThreadgroupArena {
        static_bytes: u32,
        payload: Option<u32>,
    },
    /// The render pipeline was compiled under `VariantKey::quad_primitive_ids`, and its fragment
    /// function reads `[[primitive_id]]` from Vulkan's `PrimitiveId` shifted right by one
    /// (`LoweringInputs::quad_primitive_ids`): for a draw of Metal's private quads or quad strip,
    /// listed as two triangles a quad. Its plans state the variant the shape asks for
    /// (`BindShape::quad_primitive_ids`).
    QuadPrimitiveIds,
    /// The render pipeline was compiled under `VariantKey::private_list`, and its vertex function
    /// runs the consumer's list of one of Metal's private primitive types in the pulled list form
    /// (`LoweringInputs::private_list`): its index and Metal's base vertex read at
    /// `PushHole::CaptureIndexAddress`, each list position run once. Its plans state the variant
    /// the shape asks for (`BindShape::private_list`).
    PrivateList,
    /// The function reads the texture bound to the slot only as the in-place form of a two-plane
    /// YCbCr texture reads it (`bind::YCbCrInPlace`): sampled through a constexpr sampler or
    /// through the runtime sampler slots whose bits `samplers` holds (bit `n` for
    /// `[[sampler(n)]]` of the slot's stage), read, or its size queried, as a 2D `float` texture.
    /// Recorded on a profile that reads planes in place (`ProfileFields::ycbcr_in_place`). Where
    /// the binder binds a texture whose planes are read so (`TextureShape::Texture::ycbcr_in_place`)
    /// and each of those samplers states a filter the form reads (one filter for minification and
    /// magnification, no anisotropy, no cubic filter, comparison or reduction), the slot joins the
    /// plan's variant (`VariantKey::ycbcr_in_place`, `LoweringInputs::ycbcr_in_place`), under
    /// which the slot's one binding is the planes' texel buffer (`WriteSource::YCbCrInPlace`) and
    /// every access of it reads the codes and converts them as Metal does, its words in the slot's
    /// push field (`PushField::YCbCrInPlace`). Elsewhere the slot binds the texture's image as
    /// before. `device`: every sample of the slot is one the device's own sampler Y′CbCr
    /// conversion takes (none addresses a border through a constexpr sampler or places a texel
    /// offset by a parent's size): where the bound texture's planes are also held as an image it
    /// samples (`TextureShape::Texture::ycbcr_planes_sampler`) and each runtime sampler addresses
    /// no border on either axis, the slot samples that image instead
    /// (`VariantKey::ycbcr_planes_sampled`, `bind::YCbCrPlanesSampler`).
    YCbCrInPlace {
        slot: SlotRef,
        samplers: u32,
        device: bool,
    },
}

/// What one compile made of a slot of a `RuntimeFact::UniformViewRange`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum UniformViewChoice {
    /// A uniform view.
    Viewed,
    /// A storage view the fallback did not need to move.
    Spare,
    /// A storage view the compile's `LoweringInputs::storage_view_slots` keeps one.
    Kept,
}

/// How a profile honors one runtime fact.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Mechanism {
    /// Same SPIR-V; the value is a specialization constant.
    Specialization,
    /// Different SPIR-V; a separate per-function compile (`LowerInputs`).
    Variant,
    /// A per-draw push-constant field.
    PushConstant,
    /// Vulkan dynamic state.
    DynamicState,
    /// Dynamic state for fixed-function vertex layouts, a push field for pulled ones.
    DynamicStateOrPushConstant,
    /// A Vulkan builtin (e.g. base vertex with `shaderDrawParameters`).
    Builtin,
    /// Checked by the planner; a violation is a `BindError`.
    PlanTimeCheck,
    /// Realized by the binder's view choice; no shader change.
    BinderView,
    /// Part of the `LowerInputs` key projection; the rest is link-time state.
    KeyProjection,
    /// `NumWorkgroups` for single-region dispatches, push fields for multi-region ones.
    RegionSelected,
    /// The indirect-clamp prepass.
    IndirectClamp,
    /// A synthetic binding: the `ExecutionStatus` fact on a device that does not address buffers.
    SyntheticBinding,
    /// Graphics state plus the render-pass contract.
    RenderPassContract,
    /// Nothing is needed on this profile.
    None,
    /// Static pipeline state for fixed-function layouts (no dynamic stride on the profile), a
    /// push field for pulled ones.
    PipelineStateOrPushConstant,
}
