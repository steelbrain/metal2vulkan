//! Link symbols: resource slots, specialization keys, varyings, push fields.
//!
//! A `TypedModule` refers to every number that link assigns (descriptor set and binding,
//! `Location`/`Component`, `SpecId`) through these typed symbols. The serializer records one
//! relocation per [`Symbol`], and `relocate` writes only those literals. [`PushField`] is a
//! placement key chosen per function in `select`, not a link symbol.

use crate::vk;

/// A Metal shader stage.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Stage {
    Vertex,
    Fragment,
    Kernel,
    /// An object function, a mesh render pipeline's first stage (`TaskEXT`).
    Object,
    /// A mesh function, a mesh render pipeline's geometry stage (`MeshEXT`).
    Mesh,
}

impl Stage {
    /// Whether the stage's threads run in workgroups: a kernel's, an object function's or a mesh
    /// function's threadgroup, whose threads share its threadgroup memory and meet at its
    /// barriers (`Workgroup` scope). A render stage's threadgroup memory is its tile's.
    pub(crate) fn has_workgroup(self) -> bool {
        matches!(self, Stage::Kernel | Stage::Object | Stage::Mesh)
    }
}

/// A grid or threadgroup axis.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Axis {
    X,
    Y,
    Z,
}

/// The Metal argument table a slot belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SlotClass {
    /// `[[buffer(n)]]`, n < 31.
    Buffer,
    /// `[[texture(n)]]`, n < 128.
    Texture,
    /// `[[sampler(n)]]`, n < 16.
    Sampler,
}

/// A Metal argument slot of one stage: the unit the binder binds and runtime facts name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SlotRef {
    pub stage: Stage,
    pub class: SlotClass,
    pub index: u32,
}

/// Dimensionality of an image handle class.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ImageDim {
    D1,
    D2,
    D3,
    Cube,
    Buffer,
}

/// Numeric type an image returns or accepts.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SampledType {
    Float,
    Sint,
    Uint,
    /// Metal's `ulong` texel: one unsigned 64-bit integer per texel, its low word in R and its high
    /// word in G of an `RG32Uint` texture, the only format Metal binds to a `ulong` texture
    /// parameter.
    Uint64,
}

/// How a draw read through a snapshot keeps a texture's texel in the snapshot
/// (`VariantKey::snapshot_textures`): the texel's codes, from which the read gives back the
/// fetched value bit for bit, in as few bytes as hold them, or the fetched lanes' bits.
///
/// The copy takes the codes from the value the fetch returns, and the read decodes them as Metal
/// reads the format: a normalized code `c` as `c / (2^b - 1)` correctly rounded (a signed one's
/// magnitude so, `-1` at most), a half-float code widened exactly, an integer code extended. A
/// device whose own decode is exactly that (its value one Metal reads) is given back its fetched
/// value; one whose decode is not is given Metal's.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SnapshotForm {
    /// The fetched `vec4`'s 32 bits a lane (`R32G32B32A32_UINT`): any texel.
    Bits,
    /// Four 8-bit unsigned normalized codes (`R8G8B8A8_UINT`).
    Unorm8,
    /// Four 8-bit signed normalized codes (`R8G8B8A8_UINT`, two's complement).
    Snorm8,
    /// Four 16-bit unsigned normalized codes (`R16G16B16A16_UINT`).
    Unorm16,
    /// Four 16-bit signed normalized codes (`R16G16B16A16_UINT`, two's complement).
    Snorm16,
    /// Four half-float codes (`R16G16B16A16_UINT`).
    Half,
    /// Four 8-bit unsigned integer codes (`R8G8B8A8_UINT`).
    Uint8,
    /// Four 8-bit signed integer codes (`R8G8B8A8_UINT`, two's complement).
    Sint8,
    /// Four 16-bit unsigned integer codes (`R16G16B16A16_UINT`).
    Uint16,
    /// Four 16-bit signed integer codes (`R16G16B16A16_UINT`, two's complement).
    Sint16,
}

impl SnapshotForm {
    /// The form a texture slot of sampled type `ty` keeps its texels in, read through a sampled
    /// view of `format`: the codes of an 8- or 16-bit normalized, half-float or integer format
    /// whose class is the slot's, one, two or four channels; [`Bits`](Self::Bits) for any other
    /// format (an sRGB format's view decodes its codes, so its texels are no codes' values).
    pub fn of(format: vk::Format, ty: SampledType) -> SnapshotForm {
        use vk::Format as F;
        let form = match format {
            F::R8_UNORM
            | F::R8G8_UNORM
            | F::R8G8B8A8_UNORM
            | F::B8G8R8A8_UNORM
            | F::A8B8G8R8_UNORM_PACK32 => SnapshotForm::Unorm8,
            F::R8_SNORM | F::R8G8_SNORM | F::R8G8B8A8_SNORM | F::A8B8G8R8_SNORM_PACK32 => {
                SnapshotForm::Snorm8
            }
            F::R16_UNORM | F::R16G16_UNORM | F::R16G16B16A16_UNORM => SnapshotForm::Unorm16,
            F::R16_SNORM | F::R16G16_SNORM | F::R16G16B16A16_SNORM => SnapshotForm::Snorm16,
            F::R16_SFLOAT | F::R16G16_SFLOAT | F::R16G16B16A16_SFLOAT => SnapshotForm::Half,
            F::R8_UINT | F::R8G8_UINT | F::R8G8B8A8_UINT | F::A8B8G8R8_UINT_PACK32 => {
                SnapshotForm::Uint8
            }
            F::R8_SINT | F::R8G8_SINT | F::R8G8B8A8_SINT | F::A8B8G8R8_SINT_PACK32 => {
                SnapshotForm::Sint8
            }
            F::R16_UINT | F::R16G16_UINT | F::R16G16B16A16_UINT => SnapshotForm::Uint16,
            F::R16_SINT | F::R16G16_SINT | F::R16G16B16A16_SINT => SnapshotForm::Sint16,
            _ => SnapshotForm::Bits,
        };
        let class = match form {
            SnapshotForm::Bits => return form,
            SnapshotForm::Uint8 | SnapshotForm::Uint16 => SampledType::Uint,
            SnapshotForm::Sint8 | SnapshotForm::Sint16 => SampledType::Sint,
            _ => SampledType::Float,
        };
        match class == ty {
            true => form,
            false => SnapshotForm::Bits,
        }
    }

    /// The snapshot's format: an unsigned integer format of the codes' width, four channels.
    pub fn format(self) -> vk::Format {
        match self {
            SnapshotForm::Bits => vk::Format::R32G32B32A32_UINT,
            SnapshotForm::Unorm8
            | SnapshotForm::Snorm8
            | SnapshotForm::Uint8
            | SnapshotForm::Sint8 => vk::Format::R8G8B8A8_UINT,
            SnapshotForm::Unorm16
            | SnapshotForm::Snorm16
            | SnapshotForm::Half
            | SnapshotForm::Uint16
            | SnapshotForm::Sint16 => vk::Format::R16G16B16A16_UINT,
        }
    }

    /// The width in bits of each code, and whether it is signed; `None` for
    /// [`Bits`](Self::Bits).
    pub(crate) fn code(self) -> Option<(u8, bool)> {
        match self {
            SnapshotForm::Bits => None,
            SnapshotForm::Unorm8 | SnapshotForm::Uint8 => Some((8, false)),
            SnapshotForm::Snorm8 | SnapshotForm::Sint8 => Some((8, true)),
            SnapshotForm::Unorm16 | SnapshotForm::Half | SnapshotForm::Uint16 => Some((16, false)),
            SnapshotForm::Snorm16 | SnapshotForm::Sint16 => Some((16, true)),
        }
    }
}

/// Whether an image is sampled or used as a storage image.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ImageAccess {
    Sampled,
    Storage,
}

/// The storage format class of an image handle class.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum StorageFormatClass {
    /// A sampled image: no storage format.
    NotStorage,
    /// A storage image declared with format `Unknown` (formatless).
    Unknown,
    /// A storage image declared with this format.
    Format(vk::Format),
}

/// The image part of a [`HandleClass`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ImageHandleClass {
    pub dim: ImageDim,
    pub arrayed: bool,
    pub multisampled: bool,
    pub sampled_type: SampledType,
    pub depth_compare: bool,
    pub access: ImageAccess,
    pub format: StorageFormatClass,
}

/// One runtime array of the set-1 heap: a fixed enumeration of dimension × sampled type ×
/// sampled/storage × storage format class, plus samplers by their use: an argument buffer's sampler
/// made for a sample with comparison or without, of an integer texture or not, and sampling a cube
/// with explicit gradients in the isotropic form or not (for which the sampler is made isotropic),
/// as a combined binding's runtime sampler is made for its use (`CombinedSampler`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum HandleClass {
    Image(ImageHandleClass),
    Sampler {
        compare: bool,
        integer: bool,
    },
    /// The visible-function tables an argument buffer can name: a read-only storage buffer of
    /// one table's words (`pipeline::table`) per element.
    FunctionTable,
}

impl HandleClass {
    /// The sampler heap class of use `use_`.
    pub fn sampler(use_: crate::metal::sampler::SamplerUse) -> HandleClass {
        HandleClass::Sampler {
            compare: use_.compare,
            integer: use_.integer_texture,
        }
    }

    /// The sampler use of a sampler class, `None` for an image class.
    pub fn sampler_use(self) -> Option<crate::metal::sampler::SamplerUse> {
        match self {
            HandleClass::Sampler { compare, integer } => {
                Some(crate::metal::sampler::SamplerUse::new(compare, integer))
            }
            HandleClass::Image(_) | HandleClass::FunctionTable => None,
        }
    }
}

/// A binding the translator adds that has no Metal slot.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SyntheticBinding {
    /// Per-stage uniform buffer holding the push fields that did not fit the push window.
    ArgumentTable,
    /// The command buffer's `ExecutionStatus` buffer: the product's prepasses' status binding, and
    /// a stage module's on a device that does not address buffers, a storage buffer the binder
    /// writes as `WriteSource::ExecutionStatus`. A stage module of a device that addresses buffers
    /// reaches the buffer by device address ([`PushField::ExecutionStatusAddress`]) instead.
    ExecutionStatus,
    /// The resident root table.
    RootTable,
    /// The handle table.
    HandleTable,
    /// The pipeline constant blob, a read-only storage buffer.
    ConstantBlob,
    /// The render encoder's tessellation factor buffer (`setTessellationFactorBuffer:offset:
    /// instanceStride:`), a read-only storage buffer of a tessellation pipeline's synthesized
    /// control stage, declared with the vertex stage's symbols.
    TessellationFactors,
    /// The render pass's fragment imageblock: a coherent `R32Uint` 2D-array storage image
    /// whose layer `w` holds word `w` of every pixel's imageblock cell, read and written inside the
    /// fragment's ordered pixel-interlock critical section.
    Imageblock,
    /// The sampler table: the handle table's layout over the argument buffers' samplers, whose
    /// resource ids live apart from the textures' (the sampler half): 16-byte entries `{id low, id
    /// high, element, word}` sorted by id after leading null entries, where `element` indexes the
    /// set-1 sampler arrays and `word` is the sampler's
    /// [`SamplerWord`](crate::metal::sampler::SamplerWord). A sampler handle the function loads is
    /// the index of the entry its id finds (0 for an id the table lacks), and every use of it reads
    /// that entry's element and word.
    SamplerTable,
    /// The shadow of the render pass's stencil attachment a fragment function's shader-side
    /// stencil tests and writes (`GraphicsState::shadow_stencil`): a coherent `R8Uint` 2D storage
    /// image holding pixel `(x, y)`'s stencil value at texel `(x, y)`, read and written inside the
    /// fragment's ordered pixel-interlock critical section.
    ShadowStencil,
    /// The function-table directory: the handle table's layout over the visible-function tables
    /// the pipeline state made, whose resource ids live apart from textures' and samplers'.
    FunctionTableDirectory,
    /// A patch draw's patch index buffer (`DrawCall::Patches::patch_index`), a read-only storage
    /// buffer of a tessellation pipeline's synthesized control stage, declared with the vertex
    /// stage's symbols: the `uint` at a patch's position in the draw is its patch index.
    PatchIndices,
    /// An indexed patch draw's control-point index buffer (`DrawCall::IndexedPatches`), a
    /// read-only storage buffer of the control stage of a pipeline with a control-point index
    /// type, declared with the vertex stage's symbols: the index at `patch · N + k` is control
    /// point `k`'s element.
    ControlPointIndices,
    /// The render pass's tile threadgroup memory: a storage buffer of words holding each tile's
    /// `SpecKey::TileThreadgroupLength` bytes, tile `(x, y)` at tile index
    /// `x + y · SpecKey::TileGridWidth`, which the pass keeps across its draws and tile
    /// dispatches and starts at zero (`RuntimeFact::TileThreadgroupMemory`).
    TileMemory,
    /// The function-table snapshot of a device without the set-1 function-table array's
    /// features: a read-only storage buffer of words the binder writes per command, holding a
    /// null table's record at word 0 and then each listed table's record, its word count and
    /// its words ([`FunctionTableSnapshot`](crate::bind::FunctionTableSnapshot)). The stage's
    /// function-table directory gives each id the word offset of its table's record.
    FunctionTableSnapshot,
    /// An emulated mesh function's records (`StageInterface::mesh_records`): a storage buffer of
    /// words from the draw's first word (`lower::mesh::work::RECORDS_BASE`), each mesh
    /// threadgroup's zeroed by its own threads before its body runs, which the mesh function's
    /// compute entry writes and the pipeline's synthesized vertex stage reads. Declared with the
    /// mesh stage's symbols.
    MeshRecords,
    /// An emulated mesh pipeline's work buffer for one draw: its object threadgroups' grids and
    /// payloads, their offsets in the draw's mesh threadgroups, and where its records begin,
    /// which its object function writes and its mesh function and synthesized vertex stage read
    /// (`lower::mesh::work`). Declared with the mesh stage's symbols.
    MeshWork,
    /// An emulated tessellation draw's work buffer (`GraphicsState::emulated_tessellation`): its
    /// arguments, where its records begin and its chunks' draws, which the tessellation scan
    /// writes and the post-tessellation vertex function reads (`lower::tessellation::work`).
    /// Declared with the vertex stage's symbols.
    TessellationWork,
    /// An emulated tessellation draw's records: one per instance and patch, each its processed
    /// factors, its first triangle and its stream state, which the tessellation scan writes and
    /// the post-tessellation vertex function reads (`lower::tessellation::work`). Declared with
    /// the vertex stage's symbols.
    TessellationRecords,
    /// The stand-in of the render pass's depth attachment a fragment function that runs the
    /// draw's depth test itself tests and writes (`GraphicsState::shader_tests`): a coherent
    /// `R32f` storage image holding pixel `(x, y)`'s depth at texel `(x, y)` (its samples as the
    /// stand-in's view says), read and written inside the fragment's ordered pixel-interlock
    /// critical section.
    DepthStandIn,
    /// The stand-in of the render pass's stencil attachment a fragment function that runs the
    /// draw's stencil test itself, beside its depth test, tests and writes
    /// (`ShaderTestsUse::stencil`): a coherent `R8ui` storage image holding pixel `(x, y)`'s
    /// stencil at texel `(x, y)` (each sample, multisampled at more than one), read and written
    /// inside the fragment's ordered pixel-interlock critical section.
    StencilStandIn,
}

/// What kind of resource a [`ResourceSlot`] names.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ResourceKind {
    Buffer,
    Texture,
    Sampler,
    Heap(HandleClass),
    Synthetic(SyntheticBinding),
    /// A framebuffer-fetch input for this color index.
    InputAttachment(u32),
    /// A combined image sampler: the texture slot `ResourceSlot::index` of the stage, sampled
    /// through `CombinedSampler::sampler`. One binding per (texture, sampler, use), so each carries
    /// exactly the `VkSampler` its use needs.
    CombinedTextureSampler(CombinedSampler),
    /// A separate sampler that samples set-1 heap images: one binding per (sampler, use) of the
    /// stage, so each carries exactly the `VkSampler` its use needs. A constexpr sampler is the
    /// binding's immutable sampler. `ResourceSlot::index` is the use
    /// ([`ResourceSlot::heap_sampler`]): the kind carries only the sampler, so `ResourceKind`
    /// stays the size of its one `CombinedSampler` variant (a second one would widen every
    /// `ResourceSlot`, and the `BuildError` that names one, on every builder frame).
    HeapSampler(SamplerRef),
    /// The acceleration structure bound at the stage's `[[buffer(n)]]` slot
    /// `ResourceSlot::index`, which holds structures of this level.
    AccelerationStructure(AccelerationStructureLevel),
    /// A tile function's color attachment of this color index, as a storage image the
    /// function's tile staging loads and stores.
    TileAttachment(u32),
    /// The tile map of the sparse texture bound at the stage's texture slot
    /// `ResourceSlot::index` (`pipeline::tile_map`): a read-only storage buffer of words from
    /// which a function answers the texture's residency where the device has no shader
    /// residency (`RuntimeFact::SparseTexture { tile_map: true, .. }`).
    TileMap,
    /// The image half of a combined image sampler a stage takes apart: a texture slot of the stage
    /// as a sampled image, sampled through the stage's [`ResourceKind::SeparateSampler`] of this
    /// sampler and a use. One binding per (texture, sampler, use), as the combined binding it
    /// stands for. `ResourceSlot::index` holds the use above the texture slot
    /// ([`ResourceSlot::separate_texture`]): the kind carries only the sampler, as
    /// [`ResourceKind::HeapSampler`]'s does, so `ResourceKind` keeps one `CombinedSampler` variant.
    SeparateTexture(SamplerRef),
    /// The sampler half of the combined image samplers a stage takes apart: one binding per
    /// (sampler, use) of the stage, so each carries exactly the `VkSampler` its use needs. A
    /// constexpr sampler is the binding's immutable sampler. `ResourceSlot::index` is the use
    /// ([`ResourceSlot::separate_sampler`]), as a heap sampler's is.
    SeparateSampler(SamplerRef),
    /// The `half` shadow of extended-range color attachment `n` (`ResourceSlot::index` too) for
    /// the draw's render pass instance (`VariantKey::extended_shadows`): an `RGBA16_SFLOAT` 2D
    /// storage image of the attachment's extent holding Metal's tile value of each texel, which
    /// the function's interlocked storage form reads and writes beside the attachment's codes.
    ExtendedShadow(u32),
    /// The snapshot of the texture bound at fragment texture slot `n` (`ResourceSlot::index` too)
    /// for a draw that reads it through one (`LoweringInputs::snapshot_textures`): an
    /// `R32G32B32A32_UINT` 2D storage image covering the draw's render area, each texel the bits
    /// of the four values a fetch of the texture's level 0 there returned.
    SnapshotTexels(u32),
    /// The planes of the two-plane YCbCr texture bound at the stage's texture slot
    /// `ResourceSlot::index`, held as an image of the multi-planar format that the device's own
    /// sampler Y′CbCr conversion samples (`LoweringInputs::ycbcr_planes_sampled`): a combined
    /// image sampler whose immutable sampler is the device's conversion of this
    /// (`vk::SamplerDesc::ycbcr`), whose image the consumer views with the same conversion.
    YCbCrPlanes(crate::bind::YCbCrPlanesSampler),
}

/// What a bound acceleration structure holds: the binder binds a top-level structure, so a
/// primitive structure is bound through a one-instance top-level structure over it.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AccelerationStructureLevel {
    /// `primitive_acceleration_structure`: geometry.
    Primitive,
    /// `instance_acceleration_structure`: instances of acceleration structures, primitive ones
    /// or, under a multi-level intersector, instance ones. A Vulkan top-level structure holds one
    /// level of instances, so the binder builds instances of primitive structures only.
    Instance,
}

/// The sampler half of a combined image sampler.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SamplerRef {
    /// The runtime sampler bound at `[[sampler(n)]]` of the same stage.
    Slot(u32),
    /// A constexpr sampler, by its canonical AIR state words (`air::abi::SamplerState`, bit 63 of
    /// the first word clear): link bakes it into the layout as an immutable sampler.
    Constexpr([u64; 2]),
}

impl SamplerRef {
    /// A constexpr sampler's address modes on its `s`, `t` and `r` coordinates, as Metal names
    /// them (`MTLSamplerAddressMode`; `address::clamp_to_zero` is `ClampToZero`, a border color's
    /// `ClampToBorderColor`). `None` for a runtime sampler's slot, whose modes its bound state
    /// holds, and for words that state no sampler.
    pub fn constexpr_address_modes(self) -> Option<[crate::metal::SamplerAddressMode; 3]> {
        match self {
            SamplerRef::Constexpr(words) => crate::lower::sampler::address_modes(words),
            SamplerRef::Slot(_) => None,
        }
    }
}

/// How a combined image sampler samples: its sampler, and the use facts that change or check the
/// `VkSampler` (`metal::sampler::SamplerUse`): depth comparison, which is static in AIR, whether
/// the texture is an integer one (integer border colors), whether it samples a cube with explicit
/// gradients (no anisotropy).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CombinedSampler {
    pub sampler: SamplerRef,
    /// The shader samples with depth comparison through this binding.
    pub compare: bool,
    /// The texture's sampled type is an integer type.
    pub integer: bool,
}

impl CombinedSampler {
    /// Sampler `sampler` used with `compare` on a texture that is `integer` or not.
    pub fn new(sampler: SamplerRef, compare: bool, integer: bool) -> CombinedSampler {
        CombinedSampler {
            sampler,
            compare,
            integer,
        }
    }

    /// The use as a heap sampler symbol's index, or a separate sampler's: bit 0 comparison, bit 1
    /// an integer texel.
    pub fn heap_index(&self) -> u32 {
        u32::from(self.compare) | (u32::from(self.integer) << 1)
    }

    /// Sampler `sampler` with the use a heap or separate sampler symbol's index `index` states
    /// ([`CombinedSampler::heap_index`]).
    pub fn from_heap_index(sampler: SamplerRef, index: u32) -> CombinedSampler {
        CombinedSampler::new(sampler, index & 1 != 0, index & 2 != 0)
    }

    /// The use's rank in a separate texture symbol's index: comparison, then an integer texel,
    /// from the most significant bit down, so the ranks order as the uses' fields do in
    /// `CombinedSampler`'s derived order.
    fn use_rank(&self) -> u32 {
        (u32::from(self.compare) << 1) | u32::from(self.integer)
    }

    /// Sampler `sampler` with the use of rank `rank` ([`CombinedSampler::use_rank`]).
    fn from_use_rank(sampler: SamplerRef, rank: u32) -> CombinedSampler {
        CombinedSampler::new(sampler, rank & 2 != 0, rank & 1 != 0)
    }

    /// The use the binder (and link, for a constexpr sampler) builds the `VkSampler` for.
    pub fn sampler_use(&self) -> crate::metal::sampler::SamplerUse {
        crate::metal::sampler::SamplerUse::new(self.compare, self.integer)
    }
}

/// The low bits of a separate texture symbol's index that hold its texture slot; the use's rank is
/// above them ([`ResourceSlot::separate_texture`]).
pub(crate) const SEPARATE_TEXTURE_SLOT_BITS: u32 = 28;

/// A descriptor binding symbol: link assigns its set and binding number.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ResourceSlot {
    pub stage: Stage,
    pub kind: ResourceKind,
    pub index: u32,
}

impl ResourceSlot {
    /// The binding symbol of heap sampler `c` of `stage`: the sampler in the kind, its use in the
    /// index ([`CombinedSampler::heap_index`]).
    pub fn heap_sampler(stage: Stage, c: CombinedSampler) -> ResourceSlot {
        ResourceSlot {
            stage,
            kind: ResourceKind::HeapSampler(c.sampler),
            index: c.heap_index(),
        }
    }

    /// The heap sampler a `HeapSampler` binding symbol names ([`ResourceSlot::heap_sampler`]).
    pub fn heap_sampler_use(&self) -> Option<CombinedSampler> {
        match self.kind {
            ResourceKind::HeapSampler(sampler) => {
                Some(CombinedSampler::from_heap_index(sampler, self.index))
            }
            _ => None,
        }
    }

    /// The binding symbol of separate sampler `c` of `stage`: the sampler in the kind, its use in
    /// the index ([`CombinedSampler::heap_index`]).
    pub fn separate_sampler(stage: Stage, c: CombinedSampler) -> ResourceSlot {
        ResourceSlot {
            stage,
            kind: ResourceKind::SeparateSampler(c.sampler),
            index: c.heap_index(),
        }
    }

    /// The separate sampler a `SeparateSampler` binding symbol names
    /// ([`ResourceSlot::separate_sampler`]).
    pub fn separate_sampler_use(&self) -> Option<CombinedSampler> {
        match self.kind {
            ResourceKind::SeparateSampler(sampler) => {
                Some(CombinedSampler::from_heap_index(sampler, self.index))
            }
            _ => None,
        }
    }

    /// The binding symbol of the separate texture of `stage` at texture slot `texture`, sampled
    /// through `c`: the sampler in the kind, and in the index the use's rank above the slot's 28
    /// bits. Its symbols order by sampler, use, and slot, as the `CombinedTextureSampler` symbols
    /// they stand for do, so link numbers their bindings in the same order. `None` for a slot the
    /// bits cannot hold: canonicalization refuses any texture slot past Metal's 128 entries
    /// (`AbiError::SlotPastTable`), so lowering never asks for one.
    pub fn separate_texture(
        stage: Stage,
        texture: u32,
        c: CombinedSampler,
    ) -> Option<ResourceSlot> {
        (texture >> SEPARATE_TEXTURE_SLOT_BITS == 0).then(|| ResourceSlot {
            stage,
            kind: ResourceKind::SeparateTexture(c.sampler),
            index: (c.use_rank() << SEPARATE_TEXTURE_SLOT_BITS) | texture,
        })
    }

    /// The texture slot and the sampler and use of a `SeparateTexture` binding symbol
    /// ([`ResourceSlot::separate_texture`]).
    pub fn separate_texture_of(&self) -> Option<(u32, CombinedSampler)> {
        match self.kind {
            ResourceKind::SeparateTexture(sampler) => {
                Some(ResourceSlot::separate_texture_parts(sampler, self.index))
            }
            _ => None,
        }
    }

    /// The texture slot, and `sampler` with its use, that a `SeparateTexture(sampler)` symbol's
    /// index `index` states ([`ResourceSlot::separate_texture`]).
    pub(crate) fn separate_texture_parts(
        sampler: SamplerRef,
        index: u32,
    ) -> (u32, CombinedSampler) {
        (
            index & ((1 << SEPARATE_TEXTURE_SLOT_BITS) - 1),
            CombinedSampler::from_use_rank(sampler, index >> SEPARATE_TEXTURE_SLOT_BITS),
        )
    }

    /// The binding symbol of a Metal argument slot.
    pub fn metal(slot: SlotRef) -> ResourceSlot {
        let kind = match slot.class {
            SlotClass::Buffer => ResourceKind::Buffer,
            SlotClass::Texture => ResourceKind::Texture,
            SlotClass::Sampler => ResourceKind::Sampler,
        };
        ResourceSlot {
            stage: slot.stage,
            kind,
            index: slot.index,
        }
    }
}

/// The AIR user key of a stage-interface element (`air.arg_name` / `user(...)` ABI metadata).
///
/// Metal links vertex outputs to fragment inputs by this key; it is ABI data, compared only for
/// equality and ordering, never interpreted.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AirUserKey(pub Box<[u8]>);

/// A varying symbol: link assigns its `Location` and `Component`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VaryingKey(pub AirUserKey);

/// A specialization-constant symbol: link assigns its `SpecId`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SpecKey {
    /// Metal threads per threadgroup on one axis (`sx`, `sy`, `sz`).
    LocalSize(Axis),
    /// `sn = sx·sy·sz`, the flattened Vulkan workgroup width.
    FlatLocalSize,
    /// Boolean: the dispatch takes grid and region base from push fields.
    MultiRegion,
    /// Length of threadgroup-memory argument `i` (`setThreadgroupMemoryLength`).
    TgLength(u32),
    /// Whether the slot is bound (only for slots whose null-ness the shader observes).
    Presence(SlotRef),
    /// Rasterization sample count.
    SampleCount,
    /// The render encoder's vertex amplification count (`setVertexAmplificationCount:`, default
    /// 1): Metal's `[[amplification_count]]`. A Vulkan pipeline is created per view mask, so the
    /// count is fixed with it.
    AmplificationCount,
    /// The compute encoder's imageblock width (`setImageblockWidth:height:`, default 0); in a
    /// fragment function that reaches its tile's threadgroup memory, the render pass's tile
    /// width.
    ImageblockWidth,
    /// The compute encoder's imageblock height (`setImageblockWidth:height:`, default 0); in a
    /// fragment function that reaches its tile's threadgroup memory, the render pass's tile
    /// height.
    ImageblockHeight,
    /// The width of the render area a tile dispatch covers (`renderTargetWidth`,
    /// `BindState::render_area`, default 0).
    RenderAreaWidth,
    /// The height of the render area a tile dispatch covers (`renderTargetHeight`,
    /// `BindState::render_area`, default 0).
    RenderAreaHeight,
    /// Boolean: an out-of-bounds read of the texture bound to the slot reads alpha 1, because its
    /// pixel format has no alpha component (or is a depth or stencil format, whose missing
    /// components Metal leaves undefined), rather than 0 (default false). Metal reads a zero
    /// texel out of bounds, with the format's missing components at their defaults
    /// (`RuntimeFact::OutOfBoundsAlpha`).
    OutOfBoundsAlpha(SlotRef),
    /// How Metal's bicubic filter treats the texels of the format bound at the texture slot
    /// (`u32`, the bits of `metal::bicubic::BicubicTexels`).
    BicubicTexels(SlotRef),
    /// Boolean: the texture bound to the slot is sparse (`MTLTexture.isSparse`, default false),
    /// so its accesses that observe residency take the sparse form (`RuntimeFact::SparseTexture`).
    SparseTexture(SlotRef),
    /// How Metal converts the texels of the format bound at the texture slot to `half` (`u32`,
    /// the class of `metal::texel_format::TexelFormat`).
    TexelFormat(SlotRef),
    /// The state of the runtime sampler bound to the slot (`u32`, default 0): its
    /// [`SamplerWord`](crate::metal::sampler::SamplerWord) masked by `fields`, the word's bits
    /// the function reads, so samplers that differ only in other fields share a pipeline. A slot
    /// bound to no sampler takes 0.
    SamplerState { slot: SlotRef, fields: u32 },
    /// The byte offset of a tile or fragment function's threadgroup argument `i` in its tile's
    /// threadgroup memory (`setThreadgroupMemoryLength:offset:atIndex:`, default 0).
    TgOffset(u32),
    /// The bytes of each tile's threadgroup memory: the render pass's `threadgroupMemoryLength`
    /// rounded up to 16 (default 0).
    TileThreadgroupLength,
    /// The tiles across the render pass's render area, `ceil(width / tile width)` (default 0):
    /// the stride of a tile row in `SyntheticBinding::TileMemory`.
    TileGridWidth,
    /// The swizzle of the texture buffer bound to the slot (`u32`, default the identity's word):
    /// its `MTLTextureSwizzleChannels`, each channel's `MTLTextureSwizzle` raw value in three
    /// bits, red's lowest, which the function applies to its reads (`RuntimeFact::TextureSwizzle`);
    /// at a storage slot of `LoweringInputs::swizzled_storage`, the bound texture's swizzle, which
    /// the function applies to its reads and undoes on its writes, and at an extended-range sampled
    /// slot of it, which the function applies to the decoded texels it reads, samples and gathers,
    /// and at a packed normalized sampled slot of it, to the texels its class converts.
    TextureSwizzle(SlotRef),
    /// An object function's threads per threadgroup on one axis (`threadsPerObjectThreadgroup`):
    /// its `LocalSize`, a key of its own so that a mesh render pipeline's object and mesh
    /// threadgroups take separate sizes.
    ObjectLocalSize(Axis),
    /// An object function's `FlatLocalSize`.
    ObjectFlatLocalSize,
    /// An object function's `MultiRegion`.
    ObjectMultiRegion,
    /// The level, in sixty-fourths of a level, that Metal's `calculate_clamped_lod` answers
    /// outside a fragment function through the runtime sampler bound to the slot, before the
    /// texture's last level bounds it (`u32`, default 0): `metal::sampler::query_min_level` of
    /// its descriptor (0 without a mip filter, else its `lodMinClamp` on Metal's grid). A slot
    /// bound to no sampler takes 0.
    SamplerMinLevel(SlotRef),
    /// The lanes an access of the sparse texture bound to the slot reads as constants
    /// (`u32`, default 0; `RuntimeFact::ConstantLanes`): bit `lane` (0 red to 3 alpha) where the
    /// lane's source through the texture's swizzle (`MTLTexture.swizzle`) is a channel its pixel
    /// format (`MTLTexture.pixelFormat`) lacks or a `Zero` or `One` swizzle, bit `4 + lane` where
    /// that constant is 1. A slot bound to a texture that is not sparse (`MTLTexture.isSparse`),
    /// or to none, takes 0.
    ConstantLanes(SlotRef),
    /// Boolean: the texture bound to the slot is in a format whose conversion to `half` is not
    /// measured on a device that samples through Metal's own texture unit (default false; its
    /// class, as `TexelFormat` states it, has no measured family, no 8-bit sRGB bit and no packed
    /// field: `metal::texel_format::TexelFormat::unmeasured_on_metal_sampler`). A slot bound to
    /// nothing takes false: its zero texel converts alike under every class. On such a device a
    /// function notes its `half` conversion of the slot where it holds
    /// (`PrecisionNote::HalfTexelConversion`), so the `TexelFormat` bits that only that note read
    /// leave the pipeline's key.
    UnmeasuredTexels(SlotRef),
    /// Boolean: on a device that samples through Metal's own texture unit, the texture bound to
    /// the slot converts every texel an image instruction hands back to a `half` result as that
    /// `float` texel rounded to nearest even (default true): its class has no YCbCr field on the
    /// grid `8192` (`metal::texel_format::TexelFormat::plain_on_metal_sampler`). A slot bound
    /// to nothing takes true: its zero texel converts alike under every class. Where it holds, such
    /// an access converts the instruction's own texel by the device's conversion, which Metal's
    /// compiler may fold into the image instruction (`lower::texture::half`), and specialization
    /// drops the class's routine; the class itself stays push data (`SpecKey::TexelFormat`).
    PlainHalfTexels(SlotRef),
    /// The class of the format of the texture bound to the slot (`u32`, as `TexelFormat` states
    /// it: `metal::texel_format::texel_format`), read by a specialized note's test alone
    /// ([`SpecTest`](crate::pipeline::SpecTest)): no stage declares it as a constant and no
    /// pipeline's key carries it, so the bound format decides the note without a `VkPipeline` of
    /// its own. The runtime reads it from the texture a draw or dispatch binds
    /// (`StageModule::bound_format_clauses`); where the format is not known (a reading of the
    /// module alone, `StageModule::precision_notes_under`), such a test holds.
    BoundTexelFormat(SlotRef),
    /// The plane axes of the texture bound to the slot on which the base-level size Metal's
    /// texel-offset rule reads is not a power of two where the texture's own base level's is
    /// (`u32`, default 0): bit `a` (0 width, 1 height, 2 depth) for a view whose first level lies
    /// above its parent's base level (`MTLTexture.parentRelativeLevel`) where its parent's
    /// base-level size on that axis is not a power of two and its own is. Metal drops a texel
    /// offset on a repeating or mirrored axis whose level size lies in `[2, 7]` and whose base
    /// size, the parent's, is not a power of two. A slot bound to any other texture, or to none,
    /// takes 0.
    OffsetBaseAxes(SlotRef),
    /// The degree the runtime sampler bound to the slot filters with (`u32`, default 0): `log2`
    /// of its `maxAnisotropy` as Metal samples it (rounded up to a power of two; 32 for 17 to 31),
    /// which an emulated anisotropic sample reads. A slot bound to an isotropic sampler, or to
    /// none, takes 0.
    SamplerAnisotropy(SlotRef),
    /// Boolean: the descriptor range bound at the buffer slot, which the function reads through a
    /// uniform view, starts at the bind offset, so its `SlotBindOffset` residual is 0 (default
    /// false): the slot holds inline bytes, which the consumer places at an aligned offset, or a
    /// buffer at an offset that is a multiple of the view's descriptor base alignment
    /// (`max(minUniformBufferOffsetAlignment, 16)`), under a window that starts in the range's
    /// first aligned block. A slot bound to nothing takes true: its residual is 0. Where it holds,
    /// the function reads the slot at the Metal offset alone, a constant for a constant member,
    /// which a driver can load into registers ahead of the shader (Mesa's ANV does so only for a
    /// uniform load at a constant offset); elsewhere at the residual plus the offset.
    RangeAtBindOffset(SlotRef),
}

impl SpecKey {
    /// The keys of a `stage` function's threadgroup size and grid form: its `LocalSize` on each
    /// axis, its `FlatLocalSize` and its `MultiRegion`. An object function has keys of its own;
    /// every other stage that runs in threadgroups takes the kernel's.
    pub(crate) fn dispatch_keys(stage: Stage) -> DispatchKeys {
        match stage {
            Stage::Object => DispatchKeys {
                local: [Axis::X, Axis::Y, Axis::Z].map(SpecKey::ObjectLocalSize),
                flat: SpecKey::ObjectFlatLocalSize,
                multi_region: SpecKey::ObjectMultiRegion,
            },
            _ => DispatchKeys {
                local: [Axis::X, Axis::Y, Axis::Z].map(SpecKey::LocalSize),
                flat: SpecKey::FlatLocalSize,
                multi_region: SpecKey::MultiRegion,
            },
        }
    }

    /// Whether a function reads this key's value from its push field (`PushField::Specialized`)
    /// rather than from a specialization constant: the format class of the texture bound at a
    /// slot (`TexelFormat`), which a guest's draws vary most and whose data form costs a making
    /// least. Every other key stays a specialization; read as data, the bound sampler's word, say,
    /// made one NVIDIA making cost about four times a specialized one.
    pub fn read_as_data(self) -> bool {
        matches!(self, SpecKey::TexelFormat(_))
    }
}

/// The specialization keys of one stage's threadgroup size and grid form
/// ([`SpecKey::dispatch_keys`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DispatchKeys {
    /// `sx`, `sy`, `sz`.
    pub local: [SpecKey; 3],
    /// `sn = sx·sy·sz`.
    pub flat: SpecKey,
    /// Whether the dispatch takes its grid and region base from push fields.
    pub multi_region: SpecKey,
}

impl DispatchKeys {
    /// The axis `key` is the local size of, or `None` for any other key.
    pub(crate) fn axis(&self, key: SpecKey) -> Option<usize> {
        self.local.iter().position(|k| *k == key)
    }
}

/// A push-constant field kind, in the fixed placement order (the derived `Ord`).
///
/// `select` places a function's fields in this order into the stage's push window and spills the
/// rest to the stage's `ArgumentTable`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PushField {
    /// Device address of the resident root table.
    RootTableAddress,
    /// Device address of the `ExecutionStatus` buffer: a function that counts resident-set misses
    /// has dynamic roots, so it reaches memory by device address already.
    ExecutionStatusAddress,
    /// Device address (`u64`) of the first index a de-indexed draw reads
    /// (`LoweringInputs::primitive_id_indices`): the index buffer's address plus the draw's index
    /// buffer offset, a multiple of the index size; or of a private list form draw's list
    /// (`LoweringInputs::private_list`). A vertex function compiled in either form reads its
    /// stream position's raw index there: a hole the consumer fills
    /// (`PushHole::CaptureIndexAddress`), placed with the other addresses so the vertex window
    /// always holds it.
    CaptureIndexAddress,
    /// Device address (`u64`) of a merged draw replay's header (`bind::replay`'s `merged.rs`),
    /// which a vertex function compiled under `LoweringInputs::merged_draws` reads its corner's
    /// command, vertex and instance from, and its command's slot table, in place of
    /// `RootTableAddress`: a hole the consumer fills (`PushHole::MergedDrawAddress`), placed with
    /// the other addresses. The merge passes read the header there too.
    MergedDrawAddress,
    /// Which draw of a fragment function compiled under `LoweringInputs::snapshot_textures` runs
    /// (`u32`): a hole the consumer fills (`PushHole::SnapshotPhase`), placed beside the addresses
    /// so the fragment window always holds it. 1 in the draw that takes the snapshot, whose
    /// fragments copy their pixels' texels into it and run nothing else; 0 in the draw that reads
    /// it, whose fragments copy nothing and run the function.
    SnapshotPhase,
    // The three amplification words come right after the addresses in the order, so every stage
    // window holds them (at most 32 bytes before them, 12 of them): a draw of one amplification
    // writes them alone, between its draws, never through an `ArgumentTable`.
    /// The amplification (`u32`) a draw of one amplification draws, which a function compiled
    /// under `LoweringInputs::amplification_draws` reads as its `[[amplification_id]]`.
    AmplificationId,
    /// That amplification's `renderTargetArrayIndexOffset` (`u32`, its view mapping's; 0 for
    /// `nil` mappings): under `LoweringInputs::amplification_draws` a vertex function adds it to
    /// the render target slice index it writes (0 where it writes none), and a fragment
    /// function whose vertex function writes none reads it as its `[[render_target_array_index]]`.
    AmplificationLayerOffset,
    /// That amplification's `viewportArrayIndexOffset` (`u32`), added to the viewport index and
    /// read as `AmplificationLayerOffset` is.
    AmplificationViewportOffset,
    /// The chunk an emulated tessellation draw's `vkCmdDrawIndirect` draws (`u32`): an emulated
    /// post-tessellation vertex function draws the triangles of the draw's chunk this field names
    /// (`lower::tessellation::work`). Placed with the amplification fields, so the vertex stage's
    /// window holds it (at most 44 bytes before it there: `SnapshotPhase` is a fragment
    /// function's): the draw of each chunk writes it alone, between its draws.
    TessellationChunk,
    /// Region base and grid of a multi-region dispatch.
    DispatchRegion,
    /// Device address (`u64`) of the region record a device-replay variant's dispatch reads its
    /// `DispatchRegion` value from: the replay's preprocess writes one record per replayed command
    /// and region, so a ragged or split command's regions take the base, nominal size, and grid of
    /// the command. Only a replay variant (`LoweringInputs::replay_bindings`) that reads a grid
    /// builtin declares it, in place of `DispatchRegion`, and in its place in the order: a
    /// per-region field the push window always holds.
    ReplayRegionAddress,
    /// Base vertex and base instance without `shaderDrawParameters`.
    BaseVertexInstance,
    /// The draw's vertices per primitive `n`, a `u32`: 3 for a triangle list, 2 for a line list,
    /// 1 for a strip or points. A vertex function compiled under
    /// `LoweringInputs::primitive_id_varying` writes `(VertexIndex - first) / n` to the primitive
    /// ID's varying (`IoRole::PrimitiveId`), `first` the draw's base vertex (Vulkan's `BaseVertex`
    /// with `shaderDrawParameters`, lane 0 of `BaseVertexInstance` without). The binder writes it
    /// for each draw without indices, direct or indirect.
    VerticesPerPrimitive,
    /// Residual bind offset of a descriptor-form slot: the bind offset minus the descriptor base,
    /// modulo 2^32.
    SlotBindOffset(SlotRef),
    /// Guest base address of a static-root slot.
    SlotGuestBase(SlotRef),
    /// Stride of a pulled vertex buffer layout.
    VertexBufferStride(u32),
    /// The compute encoder's stage-in region (`setStageInRegion:`): its origin `xyz` and its size
    /// `xyz`, one `uvec4` each (`w` 0).
    StageInRegion,
    /// The tessellation factor buffer's residual bind offset (`u32`), as `SlotBindOffset`'s: a
    /// field of a tessellation pipeline's synthesized control stage.
    TessellationFactorOffset,
    /// The tessellation factor buffer's `instanceStride` (`u32`), a control-stage field.
    TessellationFactorInstanceStride,
    /// The encoder's tessellation factor scale rounded to `half` (`float`), a control-stage field.
    TessellationFactorScale,
    /// The mipmap level count (`u32`) of the storage texture bound to the slot, 0 where the binder
    /// binds a neutral (`RuntimeFact::StorageTextureLevels`).
    TextureLevels(SlotRef),
    /// The instance count (`u32`) of the instance acceleration structure bound at the buffer slot,
    /// 0 where none is bound (`RuntimeFact::AccelerationStructureInstanceCount`).
    InstanceCount(SlotRef),
    /// Device address (`u64`) of an indirect command buffer's record storage (its header), which
    /// the device replay's preprocess module reads (`bind::replay`). No translated function
    /// reads it.
    CommandRecordsAddress,
    /// The render encoder's stencil state for a fragment function's shader-side stencil
    /// (`RuntimeFact::ShadowStencil`), a `uvec2`: the front face's word, then the back face's.
    /// Each word holds the face's `MTLCompareFunction` in bits 0 to 2, its stencil-failure and
    /// depth-stencil-pass `MTLStencilOperation`s in bits 3 to 5 and 6 to 8, its read mask in bits
    /// 9 to 16, its write mask in bits 17 to 24, and in bit 31 whether the draw's pass has a
    /// stencil attachment for the function to test against (`bind::shadow_stencil_words`).
    ShadowStencilState,
    /// The whole indices (`u64`) from an indexed indirect draw's bound index offset to its index
    /// buffer's end, which the index-clamp prepass reads. No translated function reads it.
    IndexCapacity,
    /// The render encoder's viewport count (`u32`, `setViewports:count:`) for a vertex function
    /// that writes `[[viewport_array_index]]` (`RuntimeFact::ViewportCount`): it stores
    /// `ViewportIndex` as 0 for an index at or past it. 0 where the encoder never set its
    /// viewports, which keeps every index 0, as Metal's one default viewport does
    /// (`BindState::viewport_count`).
    ViewportCount,
    /// The render pass's `renderTargetArrayLength` (`u32`) for a vertex function that writes
    /// `[[render_target_array_index]]`, in the variant that writes `Layer`
    /// (`RuntimeFact::RenderTargetArrayLength` without `layer_dropped`): it stores `Layer` as 0
    /// for an index at or past it, as Metal renders the pass's first layer there
    /// (`BindState::render_target_array_length`).
    RenderTargetArrayLength,
    /// The patch index buffer's residual bind offset (`u32`), as `SlotBindOffset`'s: a field of
    /// a tessellation pipeline's synthesized control stage.
    PatchIndexOffset,
    /// Whether the patch draw binds a patch index buffer (`u32`, 1 or 0), a control-stage field.
    PatchIndexBound,
    /// The control-point index buffer's residual bind offset (`u32`), a control-stage field of a
    /// pipeline with a control-point index type. An emulated post-tessellation vertex function
    /// reads this field and the two before it in its own push window
    /// (`GraphicsState::emulated_tessellation`).
    ControlPointIndexOffset,
    /// The stride (`u32`) the compute encoder binds the kernel buffer at this Metal index with
    /// (`setBuffer:offset:attributeStride:atIndex:` and its forms), 0 where it binds none, which
    /// a `[[stage_in]]` layout of `MTLBufferLayoutStrideDynamic` fetches at
    /// (`RuntimeFact::StageInStride`).
    StageInStride(u32),
    /// The draw's viewport compensation for a vertex function compiled under
    /// `LoweringInputs::viewport_compensation` (`RuntimeFact::ViewportCompensation`): one
    /// `uvec4` holding the bits of four `f32`s, `[sx, a, b, clip]`, which the function applies to
    /// its position as `(sx x, y, a w + b z, w)`, clipping at `z` and `w - z` where `clip` is 1
    /// (`BindState::viewport_compensation`).
    ViewportCompensation,
    /// The draw's point guard band for a vertex function compiled under
    /// `ViewportCompensation::point_guard`: one `uvec2` holding the bits of two `f32`s,
    /// `[gx, gy]`, the inverse powers of two its Vulkan viewports are widened by, which the
    /// function scales its position's `x` and `y` by (`CompensatedViewport::guard`).
    PointGuard,
    /// The draw's depth clamp range for a fragment function compiled under
    /// `LoweringInputs::depth_clamp_range` (`RuntimeFact::DepthClampRange`): one `uvec2` holding
    /// the bits of two `f32`s, `[low, high]`, which the function clamps the depth it writes to
    /// after Metal's `[0, 1]` (`BindState::depth_clamp_range`).
    DepthClampRange,
    /// The draw's mirrored viewports for a vertex function compiled under
    /// `ViewportCompensation::mirrors`: one `u32` whose bit `i` is set where viewport `i` has a
    /// negative width, which the function negates its position's `x` by for the viewport it
    /// selects (`CompensatedViewport::mirrors`).
    ViewportMirrors,
    /// The draw's clamp of each vertex's depth for a vertex function compiled under
    /// `ViewportCompensation::vertex_clamp`: one `uvec2` holding the bits of two `f32`s,
    /// `[lo, hi]`, which the function clamps its compensated depth's `z / w`, the Vulkan
    /// viewport's normalized depth, to before the interpolation (`CompensatedViewport::vertex_clamp`).
    VertexDepthClamp,
    /// The draw's depth state for a fragment function that runs the draw's depth test itself
    /// (`RuntimeFact::ShaderTests`), a `uvec4`: word 0 holds the depth `MTLCompareFunction` in
    /// bits 0 to 2, whether depth writes are enabled in bit 3, and whether the encoder's depth
    /// clip mode is `MTLDepthClipModeClamp` in bit 4; words 1 and 2 the bits of viewport 0's near
    /// and far depth (`f32`); word 3 is 0 (`bind::shader_tests_words`).
    ShaderTestsState,
    /// The draw's layout for a vertex function compiled in the primitive ID's de-indexed form
    /// (`LoweringInputs::primitive_id_indices`), a `uvec4`, of a draw run from vertex 0: 0, the
    /// index count, the first instance, and the base vertex. Stream position `p = VertexIndex`
    /// reads its raw index at `PushField::CaptureIndexAddress`, and its vertex is that index plus
    /// the base vertex.
    CaptureLayout,
    /// The channels each storage-form color attachment's write keeps from its texel, where the
    /// fragment function takes them from the draw: one `u32` whose bits `4c` to `4c + 3` are color
    /// `c`'s `R`, `G`, `B` and `A`, set where the pipeline's write mask leaves the channel out
    /// (`StorageAttachmentUse::kept`), so every write mask that keeps some of a format's channels
    /// shares one module.
    StorageKeptChannels,
    /// The draw's stencil state for a fragment function that runs the draw's stencil test itself
    /// beside its depth test (`RuntimeFact::ShaderStencilTest`), a `uvec4`: words 0 and 1 the
    /// front face's and the back face's words, each holding the face's `MTLCompareFunction` in
    /// bits 0 to 2, its stencil-failure and depth-stencil-pass `MTLStencilOperation`s in bits 3 to
    /// 5 and 6 to 8, its read mask in bits 9 to 16, its write mask in bits 17 to 24, and its
    /// depth-failure operation in bits 25 to 27; word 2 the front reference in bits 0 to 7 and
    /// the back reference in bits 8 to 15 (`BindState::stencil_references`); word 3 is 0
    /// (`bind::shader_stencil_words`).
    ShaderStencilState,
    /// The channel widths of the storage texture bound to the slot (`u32`;
    /// `metal::format::integer_channels_word`, 0 where the binder binds a neutral): byte `c`
    /// holds `32 - bits` of channel `c` (red to alpha) of an integer format narrower than 32 bits,
    /// else 0, and an integer texel's write saturates each lane to that range in the image's
    /// signedness, as Metal stores it (`RuntimeFact::StorageTextureIntegerChannels`, a slot whose
    /// storage choice declares no format).
    TextureIntegerChannels(SlotRef),
    /// The render encoder's blend color (`setBlendColorRed:green:blue:alpha:`) for a fragment
    /// function that computes a blend in the shader under a blend that reads it (an
    /// extended-range attachment's, or a fetched attachment's in the storage form): one `uvec4`
    /// holding the four `f32`s' bits as the encoder set them (`BindState::blend_color`).
    HalfBlendColor,
    /// The render encoder's alpha test reference (`setAlphaTestReferenceValue:`) for a fragment
    /// function compiled under `LoweringInputs::alpha_test`: one `u32`, the reference's 8-bit
    /// unorm code as Metal's alpha test compares it (`bind::alpha_test_code` of
    /// `BindState::alpha_test_reference`), which the function compares its color 0's alpha's code
    /// with.
    AlphaTestReference,
    /// The planes of the two-plane YCbCr texture bound to the slot, read in place
    /// (`RuntimeFact::YCbCrInPlace`, `LoweringInputs::ycbcr_in_place`): two `uvec4`s, the
    /// texture's size, its planes' first elements and rows in the slot's texel buffer, and its
    /// conversion (`bind::YCbCrInPlace::words`), from the bound texture's facts
    /// (`TextureFacts::ycbcr_in_place`).
    YCbCrInPlace(SlotRef),
    /// The value a draw or dispatch binds for specialization key `key`, carried as data rather
    /// than specialized: one `u32` holding exactly the value the binder computes for the key's
    /// specialization constant from the bound textures and samplers (a boolean key as 1 or 0; a
    /// `SamplerState` word masked by its `fields`). A function that reads a key this way declares
    /// no specialization constant for it, so its pipelines' keys do not carry the value and one
    /// `VkPipeline` serves every value. Last in the order, so these fields follow every other kind
    /// into the window and from there into the `ArgumentTable`.
    Specialized(SpecKey),
}

/// A relocatable literal: which decoration operand the serializer reserved, and for what.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Symbol {
    /// `OpDecorate Binding` literal.
    Binding(ResourceSlot),
    /// `OpDecorate DescriptorSet` literal.
    DescriptorSet(ResourceSlot),
    /// `OpDecorate Location` literal.
    Location(VaryingKey),
    /// `OpDecorate Component` literal.
    Component(VaryingKey),
    /// `OpDecorate SpecId` literal.
    SpecId(SpecKey),
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A heap sampler's symbol round-trips its use, and `ResourceKind` stays the size of its
    /// `CombinedSampler` variant, so heap samplers widen no `ResourceSlot`.
    #[test]
    fn heap_sampler_symbols_round_trip_without_widening_the_kind() {
        for words in [[0, 0], [u64::MAX >> 1, 7]] {
            for compare in [false, true] {
                for integer in [false, true] {
                    let c = CombinedSampler::new(SamplerRef::Constexpr(words), compare, integer);
                    let s = ResourceSlot::heap_sampler(Stage::Fragment, c);
                    assert_eq!(s.heap_sampler_use(), Some(c));
                }
            }
        }
        assert_eq!(
            ResourceSlot::metal(slot(SlotClass::Sampler, 0)).heap_sampler_use(),
            None
        );
        assert_eq!(
            std::mem::size_of::<ResourceKind>(),
            std::mem::size_of::<CombinedSampler>()
        );
    }

    /// A separate texture's symbol round-trips its texture slot, sampler, and use, and the symbols
    /// order as the `CombinedTextureSampler` symbols they stand for (by sampler, use, and slot), so
    /// link numbers the bindings of both forms alike. A slot past the index's 28 bits has no
    /// symbol; canonicalization refuses every slot past Metal's 128 entries first.
    #[test]
    fn separate_texture_symbols_round_trip_in_the_combined_order() {
        let mut pairs = Vec::new();
        for sampler in [
            SamplerRef::Slot(0),
            SamplerRef::Slot(15),
            SamplerRef::Constexpr([0, 0]),
            SamplerRef::Constexpr([u64::MAX >> 1, 7]),
        ] {
            for compare in [false, true] {
                for integer in [false, true] {
                    let c = CombinedSampler::new(sampler, compare, integer);
                    assert_eq!(
                        ResourceSlot::separate_sampler(Stage::Fragment, c).separate_sampler_use(),
                        Some(c)
                    );
                    for texture in [0, 1, 127, (1 << 28) - 1] {
                        let s = ResourceSlot::separate_texture(Stage::Fragment, texture, c)
                            .expect("a slot the index holds");
                        assert_eq!(s.separate_texture_of(), Some((texture, c)));
                        let combined = ResourceSlot {
                            stage: Stage::Fragment,
                            kind: ResourceKind::CombinedTextureSampler(c),
                            index: texture,
                        };
                        assert_eq!(combined.separate_texture_of(), None);
                        pairs.push((combined, s));
                    }
                    assert_eq!(
                        ResourceSlot::separate_texture(Stage::Fragment, 1 << 28, c),
                        None
                    );
                    assert_eq!(
                        ResourceSlot::separate_texture(Stage::Fragment, u32::MAX, c),
                        None
                    );
                }
            }
        }
        let mut by_combined = pairs.clone();
        by_combined.sort_by_key(|p| p.0);
        let mut by_separate = pairs;
        by_separate.sort_by_key(|p| p.1);
        assert_eq!(by_combined, by_separate);
    }

    fn slot(class: SlotClass, index: u32) -> SlotRef {
        SlotRef {
            stage: Stage::Kernel,
            class,
            index,
        }
    }

    /// The fixed kind order is the derived `Ord`; within a kind, fields order by slot.
    #[test]
    fn push_fields_order_by_d13_kind_order() {
        let ordered = [
            PushField::RootTableAddress,
            PushField::ExecutionStatusAddress,
            PushField::CaptureIndexAddress,
            PushField::SnapshotPhase,
            PushField::DispatchRegion,
            PushField::ReplayRegionAddress,
            PushField::BaseVertexInstance,
            PushField::VerticesPerPrimitive,
            PushField::SlotBindOffset(slot(SlotClass::Buffer, 0)),
            PushField::SlotBindOffset(slot(SlotClass::Buffer, 7)),
            PushField::SlotGuestBase(slot(SlotClass::Buffer, 0)),
            PushField::VertexBufferStride(0),
            PushField::VertexBufferStride(3),
            PushField::CaptureLayout,
        ];
        let mut shuffled = ordered;
        shuffled.reverse();
        shuffled.sort();
        assert_eq!(shuffled, ordered);
    }

    #[test]
    fn metal_slots_map_to_their_binding_kind() {
        for (class, kind) in [
            (SlotClass::Buffer, ResourceKind::Buffer),
            (SlotClass::Texture, ResourceKind::Texture),
            (SlotClass::Sampler, ResourceKind::Sampler),
        ] {
            let r = ResourceSlot::metal(slot(class, 5));
            assert_eq!(
                r,
                ResourceSlot {
                    stage: Stage::Kernel,
                    kind,
                    index: 5
                }
            );
        }
    }

    #[test]
    fn symbols_are_send_and_sync() {
        fn send_sync<T: Send + Sync>() {}
        send_sync::<SlotRef>();
        send_sync::<ResourceSlot>();
        send_sync::<HandleClass>();
        send_sync::<VaryingKey>();
        send_sync::<SpecKey>();
        send_sync::<PushField>();
        send_sync::<Symbol>();
    }
}
