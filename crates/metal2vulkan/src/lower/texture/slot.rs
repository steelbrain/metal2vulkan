//! Texture slots: the one image declaration of each slot, its storage format, and the profile
//! checks of the slot and its uses.

use crate::air::abi::Access as AbiAccess;
use crate::air::intrinsics::{ImageDim, ImageShape};
use crate::device::profile::Containment;
use crate::device::requirements::{VkExtension, VkFeature, VkProperty};
use crate::error::{Capability, Error, Feature, LowerFamily, RuntimeInput, Unsupported};
use crate::lower::select::{DescriptorCounts, SelectCx};
use crate::memory::contain;
use crate::metal::TextureType;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::interface::{TextureAccess, TextureSlot};
use crate::pipeline::lower_inputs::StorageImageClass;
use crate::pipeline::sym::{
    ImageAccess, ImageDim as SymDim, ImageHandleClass, PushField, SampledType, SlotRef, Stage,
    StorageFormatClass,
};
use crate::spirv::builder::ModuleBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::TypeId;
use crate::spirv::ir::{Dim, ImageType, ImageUsage, Type as SType};
use crate::vk;

use super::refuse;

/// The numeric class of a texture's texels (Metal's `T` in `texture2d<T>`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TexelClass {
    Float,
    Sint,
    Uint,
    /// `ulong`: an `RG32Uint` texture, declared as an `RG32` unsigned image whose two words the
    /// shader joins and splits, or, where a `ulong` atomic updates it, as the `R64ui` image the
    /// binder views it as.
    Uint64,
}

impl TexelClass {
    /// The interface's numeric type of a slot of this class.
    pub fn sampled_type(self) -> SampledType {
        match self {
            TexelClass::Float => SampledType::Float,
            TexelClass::Sint => SampledType::Sint,
            TexelClass::Uint => SampledType::Uint,
            TexelClass::Uint64 => SampledType::Uint64,
        }
    }
}

/// What the function does with a slot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Uses {
    pub read: bool,
    pub write: bool,
    pub query: bool,
    /// Sampled, gathered, or LOD-queried through a combined image sampler.
    pub sample: bool,
    /// Sampled with depth comparison.
    pub compare: bool,
    /// Updated by a texture atomic: it reads and writes texels of an `R32` format.
    pub atomic: bool,
    /// Read, written, or size-queried at a level other than the constant 0: a storage slot is
    /// declared as its level views.
    pub level_views: bool,
    /// One of those levels is not a constant: the views are indexed by a `NonUniform` value.
    pub dynamic_level: bool,
    /// The largest constant level among them (0 for none). A constant at or past the views'
    /// count indexes them as a dynamic level does, clamped and `NonUniform`.
    pub top_level: u64,
    /// Its level count is queried (`get_num_mip_levels` of a storage texture).
    pub levels: bool,
    /// An access's residency is observed (bit 0 of its status word).
    pub residency: bool,
    /// One of those accesses has no sparse form (a bicubic sample's fetches): the slot takes no
    /// sparse arm, and the binder refuses a sparse texture at it.
    pub plain_residency: bool,
    /// One of those accesses has no answer from a tile map: a sample or gather whose footprint the
    /// device's filter decides (Metal's rule does not), or a bicubic sample. The slot takes no
    /// tile-map arm: the sparse arm where the device enables `shaderResourceResidency`, and
    /// elsewhere the binder refuses a sparse texture at it.
    pub untiled_residency: bool,
    /// A cube's texels are fetched by face (an emulated linear cube sample): the slot is declared
    /// as the binder's 2D-array view of its faces.
    pub faces: bool,
    /// A `half` texel's write reads whether the bound texture's channels are 16-bit floats
    /// (`HalfNans::SlotWord`): the slot's storage choice declares no format.
    pub half_channels: bool,
    /// An integer texel's write reads the widths of the bound texture's channels
    /// (`IntRange::SlotWord`): the slot's storage choice declares no format.
    pub int_channels: bool,
    /// A read, sample or gather takes a result whose answer from a texture of another numeric
    /// class is not measured (a `short`, `ushort` or `ulong` one): the slot states no
    /// `RuntimeFact::CrossClassReads`.
    pub narrow_result: bool,
}

/// The `OpTypeImage` a slot is declared as, and the conversion a storage image's class needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImageDecl {
    pub dim: Dim,
    pub arrayed: bool,
    pub multisampled: bool,
    pub depth: bool,
    pub usage: ImageUsage,
    pub format: crate::spv::ImageFormat,
    pub texel: TexelClass,
    /// A cube texture declared as the 2D array the binder views it as (layer = face + 6 · cube).
    pub cube_as_array: bool,
    /// The storage image's class; `Identity` for sampled images.
    pub class: StorageImageClass,
    /// A storage slot declared as an array of this many single-level views, element `i` the
    /// bound texture's level `i`.
    pub level_views: Option<u32>,
    /// The slot's variant binds a texture of this extended-range format
    /// (`LoweringInputs::extended_range`): the image is declared of the format's raw `uint`
    /// texels (`texel` [`TexelClass::Uint`]), which its reads decode and its writes encode.
    pub extended: Option<crate::metal::format::ExtendedRange>,
    /// The slot's variant binds a texture of an RGBX format (`StorageImageChoice::x_channel`):
    /// after the class's conversion, its reads take alpha 1 and its writes store green in the
    /// fourth channel.
    pub x_channel: bool,
}

impl ImageDecl {
    /// The class of the texels the function holds: `float` for an extended-range slot, whose
    /// image holds raw `uint` texels, else the image's own.
    pub fn held_class(&self) -> TexelClass {
        match self.extended {
            Some(_) => TexelClass::Float,
            None => self.texel,
        }
    }

    /// Intern the image type.
    pub fn declare(&self, mb: &mut ModuleBuilder) -> BResult<TypeId> {
        let sampled_type = match self.texel {
            TexelClass::Float => mb.ty(SType::Float { width: 32 })?,
            TexelClass::Sint => mb.ty(SType::Int {
                width: 32,
                signed: true,
            })?,
            // A `ulong` slot is an `RG32` unsigned image, or one of 64-bit texels.
            TexelClass::Uint64 if self.format == crate::spv::ImageFormat::R64ui => {
                mb.ty(SType::Int {
                    width: 64,
                    signed: false,
                })?
            }
            TexelClass::Uint | TexelClass::Uint64 => mb.ty(SType::Int {
                width: 32,
                signed: false,
            })?,
        };
        mb.ty(SType::Image(ImageType {
            sampled_type,
            dim: self.dim,
            depth: self.depth,
            arrayed: self.arrayed,
            multisampled: self.multisampled,
            usage: self.usage,
            format: self.format,
        }))
    }
}

/// One texture slot of the function.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlotPlan {
    pub slot: SlotRef,
    /// The parameter's declared access.
    pub access: AbiAccess,
    /// The shape its operations spell (all agree).
    pub shape: Option<ImageShape>,
    /// The texel class its reads and writes spell; `None` when only queries use it.
    pub texel: Option<TexelClass>,
    pub uses: Uses,
    /// `is_null` observes the slot's presence (`RuntimeFact::ResourcePresence`).
    pub presence: bool,
    /// A read whose out-of-bounds texel the shader selects observes its alpha, which the slot's
    /// `OutOfBoundsAlpha` specialization gives (`RuntimeFact::OutOfBoundsAlpha`): a read the shader
    /// checks, or a level-view read under any check, past the bound level count.
    pub oob_alpha: bool,
    /// The device is left both an out-of-bounds store and an out-of-bounds load of the slot, so
    /// its variable is decorated `Volatile`; set by [`finish`].
    pub volatile: bool,
    /// The slot's accesses whose residency is observed take their sparse forms under its
    /// `SparseTexture` specialization, set by [`finish`].
    pub sparse_arm: bool,
    /// The slot's accesses whose residency is observed answer it from the slot's tile map under
    /// its `SparseTexture` specialization, set by [`finish`]: on every profile, where the slot's
    /// type has a tile map and each observed access names its texels.
    pub tile_map: bool,
    /// The slot's writes are dropped under its `SparseTexture` specialization, as Metal drops a
    /// function's writes to a sparse texture (`RuntimeFact::SparseTextureWrites`), set by
    /// [`finish`].
    pub sparse_writes: bool,
    /// A read, gather, or sample converts the slot's texels to a `half` result by the bound
    /// format's class, which the slot's `TexelFormat` specialization gives
    /// (`RuntimeFact::TexelFormat`, `half`).
    pub half_format: bool,
    /// The slot declares its `TexelFormat` specialization without the conversion routine: a raw
    /// linear sample reads the bound format's channels from it (`RuntimeFact::TexelFormat`).
    /// Implied by `half_format`.
    pub texel_format: bool,
    /// An access with a `half` result converts a value of the slot that may be filtered
    /// (`half::TexelLanes::Filtered`), whose packed class takes the routine's `OTHER` arm: where
    /// none does, the arm runs only for an unmeasured class without a packed field.
    pub half_filtered: bool,
    /// A `half` conversion of the slot on a device that samples through Metal's own texture unit
    /// notes the bound format where its class is unmeasured there
    /// (`TexelFormat::unmeasured_on_metal_sampler`): the slot declares its `UnmeasuredTexels`
    /// specialization, which the note's clause tests.
    pub unmeasured: bool,
    /// A read, gather, or sample reading one texel of the slot with a `float` result converts its
    /// texel by the bound format's class (`float_texel`): the slot declares its `TexelFormat`
    /// specialization and the module the `float` routine.
    pub float_format: bool,
    /// A `float` write to the slot rounds its channels by the bound format's class
    /// (`norm_write`): the slot declares its `TexelFormat` specialization and the module the
    /// rounding routine.
    pub norm_write: bool,
    /// A read of the slot, or an emulated linear sample of its cube's faces, takes an 8-bit sRGB
    /// texel's raw code from the slot's raw view (`ResourceKind::RawTexture`), the slot being one
    /// of `LoweringInputs::srgb_textures` (`RuntimeFact::SrgbTexture`).
    pub raw: bool,
    /// A read of the slot takes a 16-bit float texel's own 16 bits from the slot's unsigned
    /// integer alias view (`ResourceKind::RawTexture` of an unsigned integer image type,
    /// `RuntimeFact::HalfBitsTexture`) where the bound format's class names such channels, and
    /// widens them exactly (`HalfTexelBitsPolicy::Exact`). Never together with
    /// [`raw`](Self::raw): a slot read through one alias view is read through the one view.
    pub half_bits: bool,
    /// A read of the slot, a texture buffer, applies the bound texture buffer's swizzle, which
    /// the slot's `TextureSwizzle` specialization gives (`RuntimeFact::TextureSwizzle`); or the
    /// slot, a storage slot of `LoweringInputs::swizzled_storage`, applies the bound texture's to
    /// its reads and undoes it on its writes, or an extended-range sampled slot of it to the
    /// decoded texels it reads, samples and gathers.
    pub swizzle: bool,
    /// The slot's color reads and accesses through a sampler set the lanes Metal reads as
    /// constants after them, which the slot's `ConstantLanes` specialization gives
    /// (`RuntimeFact::ConstantLanes`): on a profile whose sparse textures are sparse images, a
    /// block of one that holds no mapped tile reads the device's value.
    pub constant_lanes: bool,
    /// An access of the slot places a texel offset by Metal's rule on an axis that may repeat or
    /// mirror (`sampler::position::offset_dropped`): the slot declares its `OffsetBaseAxes`
    /// specialization, which states the axes on which the base-level size the rule reads, a
    /// view's parent's, is not a power of two where the bound view's is.
    pub offset_base: bool,
    /// The declaration, set by [`finish`] for a slot that is read, written, or queried, or
    /// whose faces are fetched.
    pub decl: Option<ImageDecl>,
}

impl SlotPlan {
    /// A slot the function has not used yet.
    pub fn new(slot: SlotRef, access: AbiAccess) -> SlotPlan {
        SlotPlan {
            slot,
            access,
            shape: None,
            texel: None,
            uses: Uses::default(),
            presence: false,
            oob_alpha: false,
            volatile: false,
            sparse_arm: false,
            tile_map: false,
            sparse_writes: false,
            half_format: false,
            texel_format: false,
            half_filtered: false,
            unmeasured: false,
            float_format: false,
            norm_write: false,
            raw: false,
            half_bits: false,
            swizzle: false,
            constant_lanes: false,
            offset_base: false,
            decl: None,
        }
    }

    /// Whether the slot declares its `TexelFormat` specialization: a conversion or a rounding by
    /// the bound format's class takes it.
    pub fn declares_texel_format(&self) -> bool {
        self.half_format || self.texel_format || self.float_format || self.norm_write
    }

    /// Whether the slot is a storage image or storage texel buffer.
    pub fn storage(&self) -> bool {
        matches!(self.access, AbiAccess::Write | AbiAccess::ReadWrite)
    }

    /// Record an operation of shape `shape` with texel class `texel`: every operation on a slot
    /// must agree, since the slot has one declaration.
    pub fn observe(&mut self, shape: ImageShape, texel: Option<TexelClass>) -> Result<(), Error> {
        match self.shape {
            Some(s) if s != shape => return Err(refuse()),
            _ => self.shape = Some(shape),
        }
        match (self.texel, texel) {
            (Some(a), Some(b)) if a != b => return Err(refuse()),
            (None, Some(b)) => self.texel = Some(b),
            _ => {}
        }
        Ok(())
    }

    /// The slot's interface entry, once an operation used it.
    pub fn interface(&self) -> Option<TextureSlot> {
        let shape = self.shape?;
        let u = &self.uses;
        if !(u.read || u.write || u.query || u.sample || u.atomic) {
            return None;
        }
        let texture_type = match (shape.dim, shape.array, shape.ms) {
            (ImageDim::D1, false, _) => TextureType::Type1D,
            (ImageDim::D1, true, _) => TextureType::Type1DArray,
            (ImageDim::D2, false, false) => TextureType::Type2D,
            (ImageDim::D2, true, false) => TextureType::Type2DArray,
            (ImageDim::D2, false, true) => TextureType::Type2DMultisample,
            (ImageDim::D2, true, true) => TextureType::Type2DMultisampleArray,
            (ImageDim::D3, _, _) => TextureType::Type3D,
            (ImageDim::Cube, false, _) => TextureType::Cube,
            (ImageDim::Cube, true, _) => TextureType::CubeArray,
            (ImageDim::Buffer, _, _) => TextureType::TextureBuffer,
        };
        let sampled_type = self.texel.unwrap_or(TexelClass::Float).sampled_type();
        Some(TextureSlot {
            texture_type,
            sampled_type,
            depth_compare: self.uses.compare,
            access: TextureAccess::new(u.sample, u.read || u.atomic, u.write || u.atomic),
        })
    }
}

/// One heap group of the function: the handles loaded from argument-buffer members that merges
/// join, declared as one runtime array of the set-1 heap, like a slot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeapPlan {
    /// The members' declared access: a sampled or a storage class.
    pub access: ImageAccess,
    /// The shape its operations spell (all agree).
    pub shape: Option<ImageShape>,
    /// The texel class its reads and writes spell; `None` when only queries use it (declared
    /// `Float`, with `RuntimeFact::HeapNumericType`).
    pub texel: Option<TexelClass>,
    pub uses: Uses,
    /// The heap class and its declaration, set by [`finish`] for a group that is read, written,
    /// queried, or sampled.
    pub class: Option<ImageHandleClass>,
    pub decl: Option<ImageDecl>,
    /// Some group of its heap class is left out-of-bounds stores and some out-of-bounds loads
    /// by the device, so the class's array is decorated `Volatile`; set by [`finish`].
    pub volatile: bool,
    /// The shader makes some of its comparisons, reading its members without comparison; set by
    /// `sampler::select`.
    pub compares_plain: bool,
    /// The float class of its shape and its declaration, for a compared group whose comparisons
    /// the shader makes ([`HeapPlan::compares_plain`]); set by [`finish`]. Its samples without
    /// comparison read the members there: the binder holds a `Depth32Float` member with a depth
    /// twin through its colour image in that class, and through the twin in the depth-comparison
    /// class, whose depth view ANV clamps into `[0, 1]` and RADV holds a denormal in as zero
    /// (runtime lane `depth_compares` through argument buffers).
    pub plain: Option<(ImageHandleClass, ImageDecl)>,
    /// Its float reads decode a member of a form a heap reads (`ExtendedRange::HEAP_READ`) where
    /// the handle's class index names one; set by `select_read`.
    pub reads_raw: bool,
    /// The unsigned integer class of its shape and its declaration, for a sampled float group
    /// whose reads decode such members ([`HeapPlan::reads_raw`]); set by [`finish`]. The binder
    /// holds such a member's raw texels there, and its neutral in the float class.
    pub raw: Option<(ImageHandleClass, ImageDecl)>,
    /// The other numeric classes its reads read a member of in (`Read::heap_cross`), each where
    /// the handle's class index names a member of that class; set by `select_read`.
    pub cross_reads: std::collections::BTreeSet<TexelClass>,
    /// Its queries query a member in the class the handle's class index names
    /// (`QueryOp::heap_cross`); set by `select_query`.
    pub cross_queries: bool,
    /// Some read or query of it has no answer for a member of another numeric class (a storage
    /// or multisampled group's, a `short`, `ushort` or `ulong` read); set by `select_read` and
    /// `select_query`. With its samples, writes and atomics, the function states
    /// `RuntimeFact::HeapCrossClassNeutral` for every member of its class (`every_member`), and
    /// for those of a format no crossing read takes otherwise.
    pub uncrossed: bool,
    /// The sampled classes of its shape of those other numeric types, and their declarations;
    /// set by [`finish`] for a sampled color group of one sample. The binder holds each member in
    /// the class of its own numeric type, and the neutral in the others.
    pub cross: std::collections::BTreeMap<TexelClass, (ImageHandleClass, ImageDecl)>,
    /// Its `half` or `float` writes store a half-float member's bits through the unsigned integer
    /// twin where the handle's class index names one (`Write::heap_bits`); set by `select_write`.
    pub writes_bits: bool,
    /// The twin of a written float storage group ([`HeapPlan::writes_bits`]): the formatless
    /// unsigned integer storage class of its shape that holds half-float members through their
    /// unsigned integer aliases (`StorageFormatClass::HalfBits`), and its declaration; set by
    /// [`finish`]. The binder holds each such member there too, at the same element.
    pub bits: Option<(ImageHandleClass, ImageDecl)>,
}

impl HeapPlan {
    /// A group the function has not used yet, of members with `access`.
    pub fn new(access: ImageAccess) -> HeapPlan {
        HeapPlan {
            access,
            shape: None,
            texel: None,
            uses: Uses::default(),
            class: None,
            decl: None,
            volatile: false,
            compares_plain: false,
            plain: None,
            reads_raw: false,
            raw: None,
            cross_reads: Default::default(),
            cross_queries: false,
            uncrossed: false,
            cross: Default::default(),
            writes_bits: false,
            bits: None,
        }
    }

    /// Record an operation of shape `shape` with texel class `texel` ([`SlotPlan::observe`]).
    pub fn observe(&mut self, shape: ImageShape, texel: Option<TexelClass>) -> Result<(), Error> {
        match self.shape {
            Some(s) if s != shape => return Err(refuse()),
            _ => self.shape = Some(shape),
        }
        match (self.texel, texel) {
            (Some(a), Some(b)) if a != b => return Err(refuse()),
            (None, Some(b)) => self.texel = Some(b),
            _ => {}
        }
        Ok(())
    }
}

/// The heap class and declaration of group `h` with shape `shape` (a sampled group's texture
/// sampled through a separate heap sampler), checked on the profile: the descriptor-indexing
/// features the heap array needs (a runtime array, partially bound, indexed non-uniformly, updated
/// after bind), and a storage class's formatless access. What the profile lacks is one
/// `Unsupported{ArgumentBufferTextures}` naming all of it.
fn declare_heap(
    cx: &SelectCx<'_>,
    h: &HeapPlan,
    shape: ImageShape,
) -> Result<(ImageHandleClass, ImageDecl), Error> {
    let storage = h.access == ImageAccess::Storage;
    let (dim, sym) = match shape.dim {
        ImageDim::D1 => (Dim::D1, SymDim::D1),
        ImageDim::D2 => (Dim::D2, SymDim::D2),
        ImageDim::D3 => (Dim::D3, SymDim::D3),
        // A cube group that is only sampled and queried is the sampled cube class, its members
        // held through their cube views: a size, array-size, or level-count query is the cube
        // view's own, as a slot cube's is (an arrayed cube's size counts cubes). A cube read or
        // written by face is addressed through a 2D-array view, which needs its own class; texture
        // buffers need texel-buffer heap classes.
        ImageDim::Cube if !storage && !(h.uses.read || h.uses.write || h.uses.atomic) => {
            if shape.array && !cx.profile.fields().images.cube_array {
                return Err(Unsupported::needs(
                    Feature::CubeArray,
                    Capability::Feature(VkFeature::ImageCubeArray),
                )
                .into());
            }
            (Dim::Cube, SymDim::Cube)
        }
        ImageDim::Cube | ImageDim::Buffer => {
            return Err(Unsupported::not_implemented(Feature::ArgumentBufferTextures).into())
        }
    };
    // A depth texture read without comparison is a member of the float class of its shape: the
    // binder holds a depth texture there through its depth-aspect view (`bind::heap::fit`), whose
    // texel's first component is the depth, and the class is declared without `Depth`, an operand
    // Vulkan ignores. A group compared anywhere is the depth-comparison class of its shape, as a
    // compared slot is (the depth-comparison classes, 2D and cube): the binder holds only depth
    // formats there, through the same view, and the class is declared `Depth`. Metal has no
    // writable depth texture.
    let compare = shape.depth && h.uses.compare;
    if shape.depth && storage {
        return Err(refuse());
    }
    if compare && !matches!(shape.dim, ImageDim::D2 | ImageDim::Cube) {
        return Err(cx.invariant("a compared depth heap group that is neither 2D nor cube"));
    }
    let d = &cx.profile.fields().descriptors;
    let images = &cx.profile.fields().images;
    let mut missing = Vec::new();
    let mut need = |has: bool, f: VkFeature| {
        if !has {
            missing.push(Capability::Feature(f));
        }
    };
    need(
        d.runtime_descriptor_array,
        VkFeature::RuntimeDescriptorArray,
    );
    need(
        d.partially_bound,
        VkFeature::DescriptorBindingPartiallyBound,
    );
    if storage {
        need(
            d.non_uniform_storage_image,
            VkFeature::ShaderStorageImageArrayNonUniformIndexing,
        );
        need(
            d.update_after_bind_storage_image,
            VkFeature::DescriptorBindingStorageImageUpdateAfterBind,
        );
        // A heap's storage classes are declared without a format, and the binder views its
        // members for every access the device's features grant (`bind::heap`): a device's
        // formats' own bits do not serve a heap.
        need(
            !h.uses.read || images.storage_read_without_format,
            VkFeature::ShaderStorageImageReadWithoutFormat,
        );
        need(
            !h.uses.write || images.storage_write_without_format,
            VkFeature::ShaderStorageImageWriteWithoutFormat,
        );
        need(
            !shape.ms || images.storage_multisample,
            VkFeature::ShaderStorageImageMultisample,
        );
    } else {
        need(
            d.non_uniform_sampled_image,
            VkFeature::ShaderSampledImageArrayNonUniformIndexing,
        );
        need(
            d.update_after_bind_sampled_image,
            VkFeature::DescriptorBindingSampledImageUpdateAfterBind,
        );
    }
    if !missing.is_empty() {
        return Err(
            Unsupported::needs_any_of(Feature::ArgumentBufferTextures, vec![missing]).into(),
        );
    }
    let texel = h.texel.unwrap_or(TexelClass::Float);
    // A heap `ulong` texture would need a heap class of its own (its `RG32` declaration is not
    // the formatless class's); not implemented.
    if texel == TexelClass::Uint64 {
        return Err(refuse());
    }
    let class = ImageHandleClass {
        dim: sym,
        arrayed: shape.array,
        multisampled: shape.ms,
        sampled_type: texel.sampled_type(),
        depth_compare: compare,
        access: h.access,
        format: if storage {
            StorageFormatClass::Unknown
        } else {
            StorageFormatClass::NotStorage
        },
    };
    let decl = ImageDecl {
        dim,
        arrayed: shape.array,
        multisampled: shape.ms,
        depth: compare,
        usage: if storage {
            ImageUsage::Storage
        } else {
            ImageUsage::Sampled
        },
        format: crate::spv::ImageFormat::Unknown,
        texel,
        cube_as_array: false,
        class: StorageImageClass::Identity,
        level_views: None,
        extended: None,
        x_channel: false,
    };
    Ok((class, decl))
}

/// The set-1 heap features a function's heap classes need besides the module's own (the layout's
/// binding flags): `descriptorBindingPartiallyBound` and the classes' update-after-bind features.
/// `compile` adds them to the artifact's requirements.
pub fn heap_layout_features(plan: &crate::lower::select::Plan) -> Vec<VkFeature> {
    let mut out = Vec::new();
    for h in plan.families.texture.heap.values() {
        let Some(class) = h.class else { continue };
        out.push(VkFeature::DescriptorBindingPartiallyBound);
        out.push(match class.access {
            ImageAccess::Storage => VkFeature::DescriptorBindingStorageImageUpdateAfterBind,
            _ => VkFeature::DescriptorBindingSampledImageUpdateAfterBind,
        });
    }
    out.sort();
    out.dedup();
    out
}

fn needs(feature: Feature, f: VkFeature) -> Error {
    Unsupported::needs(feature, Capability::Feature(f)).into()
}

/// How a read of an image or texel buffer returns Metal's out-of-bounds texel.
///
/// Metal reads the zero texel out of bounds, with the pixel format's missing components at their
/// defaults (alpha 1), when any operand is out of range: the coordinate, the level, the array
/// slice, the cube face, the cube index, or the sample (MSL 4.1 sections 2.9.1 and 6.13; measured).
/// That is Vulkan's out-of-bounds read under `robustImageAccess2` (images) and
/// `robustBufferAccess2` (texel buffers): zero with `(0, 0, 1)` substituted for the view format's
/// missing G, B, and A. `robustImageAccess` alone reads a missing alpha as 0 or 1 and leaves a
/// fetch at a level past the view's undefined, and `robustBufferAccess` may read any value of
/// the buffer, so neither is Metal's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReadCheck {
    /// No containment (`Containment::UnsafeOff`): the access as the function spells it.
    Off,
    /// The device's robustness returns Metal's texel; the access is left to it.
    Device,
    /// The shader clamps every operand into the bound resource, so the access stays inside it,
    /// and selects Metal's out-of-bounds texel where an operand was out of range.
    Shader,
}

/// How a read of an image (`buffer`: a texel buffer) returns Metal's out-of-bounds texel on the
/// profile ([`ReadCheck`]).
pub(super) fn read_check(cx: &SelectCx<'_>, buffer: bool) -> ReadCheck {
    if !matches!(cx.profile.fields().containment, Containment::On { .. }) {
        ReadCheck::Off
    } else if contain::device_reads_texels(cx.profile, buffer) {
        ReadCheck::Device
    } else {
        ReadCheck::Shader
    }
}

/// Whether a write needs the in-shader bounds check: Metal discards an out-of-bounds write, and
/// Vulkan does under `robustImageAccess` (images, either level) or `robustBufferAccess2` (texel
/// buffers). Under `robustBufferAccess` a write past a texel buffer may instead modify any memory
/// bound to the underlying buffer, so it is checked in the shader.
pub(super) fn write_guard(cx: &SelectCx<'_>, buffer: bool) -> bool {
    !contain::device_drops_texel_writes(cx.profile, buffer)
}

/// Whether the device's robustness is left both an out-of-bounds store and an out-of-bounds
/// load of one resource variable with `uses` (`buffer`: a texel buffer), so that variable must be
/// `Volatile`.
///
/// Vulkan's robust image and texel-buffer access discard an out-of-bounds store and read zero out
/// of bounds, but "if a value was previously written to the same out of bounds location in the
/// same shader invocation, that value may be returned instead; using the VolatileTexel image
/// operand, the Volatile memory semantic, or the Volatile decoration to load the value will
/// prevent prior stored values from being returned". Metal reads zero there (measured). A store is
/// left to the device where [`write_guard`] does not guard it, a load where [`read_check`] leaves
/// it to the device, and a texture atomic (which stores and returns the texel) is both where it is
/// not guarded. A load the shader checks reads a clamped coordinate inside the resource, and a
/// guarded store never reaches the device, so neither can meet the other out of bounds. Without
/// containment nothing out of bounds is defined.
fn forwards(cx: &SelectCx<'_>, uses: &Uses, buffer: bool) -> bool {
    if !matches!(cx.profile.fields().containment, Containment::On { .. }) {
        return false;
    }
    let atomic = uses.atomic && !write_guard(cx, buffer);
    let store = (uses.write && !write_guard(cx, buffer)) || atomic;
    let load = (uses.read && read_check(cx, buffer) == ReadCheck::Device) || atomic;
    store && load
}

/// A write from a vertex or fragment function needs the stage's store feature.
pub(super) fn stage_stores(cx: &SelectCx<'_>) -> Result<(), Error> {
    let leaf = &cx.profile.fields().leaf;
    match cx.stage {
        Stage::Vertex if !leaf.vertex_stores_and_atomics => Err(needs(
            Feature::StorageTextureInVertexStage,
            VkFeature::VertexPipelineStoresAndAtomics,
        )),
        Stage::Fragment if !leaf.fragment_stores_and_atomics => Err(needs(
            Feature::Lowering(LowerFamily::Texture),
            VkFeature::FragmentStoresAndAtomics,
        )),
        _ => Ok(()),
    }
}

/// The format a storage slot is declared with and its class: the slot's `StorageImageChoice` from
/// the lowering inputs, or, without one, a formatless `Identity` declaration where the device's
/// `shaderStorageImage{Read,Write}WithoutFormat` grant its accesses without a format; otherwise the
/// format is an input the caller must supply (`NeedsInput`). A choice whose view SPIR-V names no
/// format for is declared without one wherever the profile accesses storage so at all, by the
/// features or by the formats' own bits (`DeviceProfile::accesses_without_format`; the choice
/// checked the view's bits).
///
/// A slot that is only queried accesses no texel, so it is declared `Unknown` with class
/// `Identity` whatever the choice (queries need no format feature, and no format then pairs with
/// its `float` sampled type). A read or written slot's choice must have the numeric type of the
/// slot's texel class (a formatless choice's kept type, when it keeps one) and a 32-bit component
/// (VUID-StandaloneSpirv-Image-04965); a choice that disagrees is `NotRepresentable(Format)`,
/// never a mismatched declaration.
fn storage_format(
    cx: &SelectCx<'_>,
    s: &SlotPlan,
    texel: TexelClass,
) -> Result<(crate::spv::ImageFormat, StorageImageClass), Error> {
    if s.uses.atomic {
        return atomic_format(cx, s, texel);
    }
    if !(s.uses.read || s.uses.write) {
        return Ok((
            crate::spv::ImageFormat::Unknown,
            StorageImageClass::Identity,
        ));
    }
    if texel == TexelClass::Uint64 {
        return ulong_format(cx, s);
    }
    let images = &cx.profile.fields().images;
    let formatless_ok = (!s.uses.read || images.storage_read_without_format)
        && (!s.uses.write || images.storage_write_without_format);
    let Some(choice) = cx.inputs.storage_images.get(&s.slot) else {
        if formatless_ok {
            return Ok((
                crate::spv::ImageFormat::Unknown,
                StorageImageClass::Identity,
            ));
        }
        return Err(Error::NeedsInput(RuntimeInput::StorageTextureFormat {
            slot: s.slot,
        }));
    };
    // The view format's numeric type must be the declared sampled type's, whether the format is
    // declared or (without a SPIR-V image format) formatless; a formatless choice states the type
    // it keeps, if any.
    let numeric = if choice.vk_format == vk::Format::UNDEFINED {
        choice.numeric.map(Some)
    } else {
        Some(crate::bind::neutral::numeric_type(choice.vk_format))
    };
    if let Some(numeric) = numeric {
        let class = match numeric {
            Some(SampledType::Float) => TexelClass::Float,
            Some(SampledType::Sint) => TexelClass::Sint,
            Some(SampledType::Uint) => TexelClass::Uint,
            Some(SampledType::Uint64) => TexelClass::Uint64,
            None => return Err(Unsupported::not_representable(Feature::Format).into()),
        };
        if class != texel {
            return Err(Unsupported::not_representable(Feature::Format).into());
        }
    }
    // A view format without a SPIR-V image format is declared `Unknown`.
    let format = crate::metal::format::spirv_image_format(choice.vk_format)
        .unwrap_or(crate::spv::ImageFormat::Unknown);
    // The sampled type is 32-bit; the 64-bit formats need a 64-bit one (`Int64ImageEXT`).
    if matches!(
        format,
        crate::spv::ImageFormat::R64ui | crate::spv::ImageFormat::R64i
    ) {
        return Err(Unsupported::not_representable(Feature::Format).into());
    }
    if format == crate::spv::ImageFormat::Unknown
        && !cx
            .profile
            .accesses_without_format(s.uses.read, s.uses.write)
    {
        let mut missing = Vec::new();
        if s.uses.read && !images.storage_read_without_format {
            missing.push(Capability::Feature(
                VkFeature::ShaderStorageImageReadWithoutFormat,
            ));
        }
        if s.uses.write && !images.storage_write_without_format {
            missing.push(Capability::Feature(
                VkFeature::ShaderStorageImageWriteWithoutFormat,
            ));
        }
        return Err(Unsupported::needs_any_of(Feature::Format, vec![missing]).into());
    }
    if crate::spirv::requirements::extended_format(format) && !images.storage_extended_formats {
        return Err(needs(
            Feature::Format,
            VkFeature::ShaderStorageImageExtendedFormats,
        ));
    }
    // The converting classes exist for 8-bit UNORM color formats, half and small float ones and
    // 1-bit alpha ones only. A YCbCr write slot's `float` texels are stored as their bits in an
    // unsigned integer image, whose class `declare_slot` gives it., as a half-float slot only
    // written stores its half bits in its unsigned integer alias.
    let converting = !matches!(
        choice.class,
        StorageImageClass::Identity
            | StorageImageClass::YCbCrWrite
            | StorageImageClass::HalfBits
            | StorageImageClass::SharedExponentBits
    );
    if converting && texel != TexelClass::Float {
        return Err(Unsupported::not_representable(Feature::Format).into());
    }
    Ok((format, choice.class))
}

/// The format of a `ulong` storage slot that is read or written: Metal binds only an `RG32Uint`
/// texture to a `ulong` texture parameter (its API validation rejects every other format as
/// incompatible with `MTLDataTypeULong`), viewed as `RG32_UINT`, which the profile must support for
/// the slot's storage usage. The slot is declared `Rg32ui` where the profile has
/// `shaderStorageImageExtendedFormats`, or where the slot's choice names that view (the binder's,
/// for a device that cannot access it without a format); otherwise without a format where the
/// profile reads and writes so (`Unknown`); otherwise it is `Unsupported{Format}` naming either
/// route. A choice of any other view is `NotRepresentable(Format)`.
fn ulong_format(
    cx: &SelectCx<'_>,
    s: &SlotPlan,
) -> Result<(crate::spv::ImageFormat, StorageImageClass), Error> {
    let mut declared = false;
    if let Some(choice) = cx.inputs.storage_images.get(&s.slot) {
        let view = matches!(
            choice.vk_format,
            vk::Format::R32G32_UINT | vk::Format::UNDEFINED
        );
        if !view || choice.class != StorageImageClass::Identity || choice.alias.is_some() {
            return Err(Unsupported::not_representable(Feature::Format).into());
        }
        declared = choice.vk_format == vk::Format::R32G32_UINT;
    }
    let buffer = s.shape.is_some_and(|shape| shape.dim == ImageDim::Buffer);
    let usage = if buffer {
        crate::device::requirements::FormatUsage::StorageTexelBuffer
    } else {
        crate::device::requirements::FormatUsage::StorageImage
    };
    if !cx.profile.supports_format(vk::Format::R32G32_UINT, usage) {
        return Err(Unsupported::needs(
            Feature::Format,
            Capability::Format {
                format: vk::Format::R32G32_UINT,
                usage,
            },
        )
        .into());
    }
    let images = &cx.profile.fields().images;
    if images.storage_extended_formats {
        return Ok((crate::spv::ImageFormat::Rg32ui, StorageImageClass::Identity));
    }
    // Without a format by the device's features, or by `R32G32_UINT`'s own bits.
    let use_ = match buffer {
        true => crate::metal::format::StorageUse::texel_buffer(s.uses.read, s.uses.write),
        false => crate::metal::format::StorageUse::texture(s.uses.read, s.uses.write),
    };
    let formatless = ((!s.uses.read || images.storage_read_without_format)
        && (!s.uses.write || images.storage_write_without_format))
        || crate::metal::format::without_format_gaps(vk::Format::R32G32_UINT, use_, cx.profile)
            .is_empty();
    if formatless && !declared {
        return Ok((
            crate::spv::ImageFormat::Unknown,
            StorageImageClass::Identity,
        ));
    }
    let extended = vec![Capability::Feature(
        VkFeature::ShaderStorageImageExtendedFormats,
    )];
    let mut without = Vec::new();
    if s.uses.read {
        without.push(Capability::Feature(
            VkFeature::ShaderStorageImageReadWithoutFormat,
        ));
    }
    if s.uses.write {
        without.push(Capability::Feature(
            VkFeature::ShaderStorageImageWriteWithoutFormat,
        ));
    }
    let routes = if declared {
        vec![extended]
    } else {
        vec![extended, without]
    };
    Err(Unsupported::needs_any_of(Feature::Format, routes).into())
}

/// The format of a slot a texture atomic updates: Metal's texture atomics need an `R32Uint` or
/// `R32Sint` texture of the texel's numeric type, and a Vulkan image atomic needs the declared
/// `R32ui` / `R32i` format (VUID-StandaloneSpirv-OpImageTexelPointer-04658) with the format's
/// storage-atomic feature (a texel buffer's storage-texel-buffer-atomic one). The slot's choice,
/// when there is one, must be that format; without one, Metal's rule gives it (the binder's view of
/// an `R32` texture, and the `R32` neutral of an unbound storage slot of that numeric type, have
/// it).
///
/// A `ulong` slot is declared `R64ui`, which the binder views its `RG32Uint` texture as
/// (`RuntimeFact::UlongAtomicView`), where the profile has what that needs
/// (`DeviceProfile::ulong_atomic_gaps`); elsewhere it is `Unsupported{ImageAtomics}` naming every
/// missing capability, never an `RG32` declaration, which no image atomic takes.
fn atomic_format(
    cx: &SelectCx<'_>,
    s: &SlotPlan,
    texel: TexelClass,
) -> Result<(crate::spv::ImageFormat, StorageImageClass), Error> {
    let buffer = s.shape.is_some_and(|shape| shape.dim == ImageDim::Buffer);
    let (format, vk_format) = match texel {
        TexelClass::Uint => (crate::spv::ImageFormat::R32ui, vk::Format::R32_UINT),
        TexelClass::Sint => (crate::spv::ImageFormat::R32i, vk::Format::R32_SINT),
        TexelClass::Uint64 => {
            let gaps = cx.profile.ulong_atomic_gaps(buffer);
            if !gaps.is_empty() {
                return Err(Unsupported::needs_any_of(Feature::ImageAtomics, vec![gaps]).into());
            }
            (crate::spv::ImageFormat::R64ui, vk::Format::R64_UINT)
        }
        TexelClass::Float => return Err(refuse()),
    };
    if let Some(choice) = cx.inputs.storage_images.get(&s.slot) {
        if choice.vk_format != vk_format || choice.class != StorageImageClass::Identity {
            return Err(Unsupported::not_representable(Feature::Format).into());
        }
    }
    let usage = if buffer {
        crate::device::requirements::FormatUsage::StorageTexelBufferAtomic
    } else {
        crate::device::requirements::FormatUsage::StorageImageAtomic
    };
    if !cx.profile.supports_format(vk_format, usage) {
        return Err(Unsupported::needs(
            Feature::Format,
            Capability::Format {
                format: vk_format,
                usage,
            },
        )
        .into());
    }
    Ok((format, StorageImageClass::Identity))
}

/// The most mipmap levels a texture of `dim` can have on `profile`: `⌊log2 d⌋ + 1` for its largest
/// extent `d`, the profile's `maxImageDimension` of that type (a cube's, a 2D array's 2D's), at
/// least 1. A storage slot's level views hold this many elements, so every level of any texture the
/// device can create has its element.
pub fn max_levels(profile: &crate::device::DeviceProfile, dim: ImageDim) -> u32 {
    let l = &profile.fields().limits;
    let d = match dim {
        ImageDim::D1 => l.max_image_dimension_1d,
        ImageDim::D2 => l.max_image_dimension_2d,
        ImageDim::D3 => l.max_image_dimension_3d,
        ImageDim::Cube => l.max_image_dimension_cube,
        ImageDim::Buffer => 1,
    };
    d.max(1).ilog2() + 1
}

/// The level views of storage slot `s`, checked on the profile, with their facts and push field:
/// `None` for a slot without them.
fn level_views(
    cx: &SelectCx<'_>,
    s: &SlotPlan,
    shape: ImageShape,
    facts: &mut Vec<RuntimeFact>,
    push: &mut Vec<PushField>,
) -> Result<Option<u32>, Error> {
    let u = &s.uses;
    if u.levels || u.level_views {
        facts.push(RuntimeFact::StorageTextureLevels { slot: s.slot });
        push.push(PushField::TextureLevels(s.slot));
    }
    if !u.level_views {
        return Ok(None);
    }
    // An image texel pointer into an element is not modelled (the atomics take no level).
    if u.atomic {
        return Err(refuse());
    }
    let n = max_levels(cx.profile, shape.dim);
    // A constant at or past `n` is no level of any texture; `emit` indexes it as a dynamic one.
    let non_uniform = u.dynamic_level || u.top_level >= u64::from(n);
    if non_uniform && !cx.profile.fields().descriptors.non_uniform_storage_image {
        return Err(needs(
            Feature::Lowering(LowerFamily::Texture),
            VkFeature::ShaderStorageImageArrayNonUniformIndexing,
        ));
    }
    facts.push(RuntimeFact::StorageTextureLevelViews { slot: s.slot });
    Ok(Some(n))
}

/// The class of the image slot `s` is declared as, a combined binding's too, where neither an
/// extended-range nor a YCbCr write variant declares it otherwise: the class of the texture the
/// slot's variant binds (`LoweringInputs::cross_class`), whose texels its accesses read in that
/// class, else the class the function declares there (`Float` for a slot it only queries).
pub(crate) fn image_class(inputs: &crate::pipeline::LoweringInputs, s: &SlotPlan) -> TexelClass {
    match inputs.cross_class.get(&s.slot) {
        Some(SampledType::Sint) => TexelClass::Sint,
        Some(SampledType::Uint) => TexelClass::Uint,
        Some(SampledType::Float) => TexelClass::Float,
        _ => s.texel.unwrap_or(TexelClass::Float),
    }
}

/// Whether slot `s`'s accesses answer from a texture of any numeric class
/// (`RuntimeFact::CrossClassReads`): a color texture of a `float`, signed or unsigned integer
/// class that the function reads, samples or gathers with `float`, `half`, 32-bit or 16-bit integer
/// results, or writes, or only queries; a storage texture, a texture buffer and a multisampled
/// texture too. Not a depth texture, an atomic's or a `ulong` access's.
fn cross_reads(s: &SlotPlan, shape: ImageShape) -> bool {
    !shape.depth
        && !s.uses.atomic
        && (s.uses.read || s.uses.sample || s.uses.write || s.uses.query)
        && !s.uses.narrow_result
        && matches!(
            s.texel,
            None | Some(TexelClass::Float | TexelClass::Sint | TexelClass::Uint)
        )
}

/// The declaration of slot `s`, checked on the profile, with the facts, push fields, and
/// descriptors it contributes.
fn declare_slot(
    cx: &SelectCx<'_>,
    s: &SlotPlan,
    shape: ImageShape,
    facts: &mut Vec<RuntimeFact>,
    push: &mut Vec<PushField>,
    counts: &mut DescriptorCounts,
) -> Result<ImageDecl, Error> {
    let images = &cx.profile.fields().images;
    let storage = s.storage();
    // A slot bound in an extended-range format holds its raw texels in an unsigned integer
    // image.
    let extended = cx.inputs.extended_range.get(&s.slot).copied();
    // A slot that only writes a 4:4:4 two-plane YCbCr texture holds each written texel's bits
    // and mark in an unsigned integer image (`StorageImageClass::YCbCrWrite`): a `float` slot's
    // writes alone, as the binder chooses it (`bind::plan`); any other use is the format's
    // refusal.
    let ycbcr_write = storage
        && cx
            .inputs
            .storage_images
            .get(&s.slot)
            .is_some_and(|c| c.class == StorageImageClass::YCbCrWrite);
    let float = image_class(cx.inputs, s) == TexelClass::Float;
    if ycbcr_write && (s.uses.read || s.uses.atomic || !float) {
        return Err(Unsupported::not_implemented(Feature::Format).into());
    }
    // A slot that reads and writes a half-float texture through its unsigned integer alias
    // holds each texel's half bits there (`StorageImageClass::HalfBits`): a `float` slot's reads
    // and writes, as the binder chooses it (`metal::format::storage_choice`); no atomic.
    let half_bits = storage
        && cx.inputs.storage_images.get(&s.slot).is_some_and(|c| {
            matches!(
                c.class,
                StorageImageClass::HalfBits | StorageImageClass::SharedExponentBits
            )
        });
    if half_bits && (s.uses.atomic || !float) {
        return Err(Unsupported::not_representable(Feature::Format).into());
    }
    // A texture of another class, read in its own (`LoweringInputs::cross_class`), only where
    // the slot's accesses answer so; the binder asks for no other.
    let cross = cx.inputs.cross_class.contains_key(&s.slot);
    if cross && !cross_reads(s, shape) {
        return Err(Unsupported::not_implemented(Feature::Format).into());
    }
    let texel = match extended {
        Some(_) => TexelClass::Uint,
        None if ycbcr_write || half_bits => TexelClass::Uint,
        None => image_class(cx.inputs, s),
    };
    if shape.depth && storage {
        // Metal has no writable depth textures.
        return Err(refuse());
    }
    if storage && shape.ms && !images.storage_multisample {
        return Err(needs(
            Feature::StorageImageMultisample,
            VkFeature::ShaderStorageImageMultisample,
        ));
    }
    let cube = shape.dim == ImageDim::Cube;
    // A cube that is read or written, or whose faces are fetched, and every storage cube, is the
    // binder's 2D-array view.
    let cube_as_array = cube && (storage || s.uses.read || s.uses.write || s.uses.faces);
    if cube && shape.array && !cube_as_array && !images.cube_array {
        return Err(needs(Feature::CubeArray, VkFeature::ImageCubeArray));
    }
    let dim = match shape.dim {
        ImageDim::D1 => Dim::D1,
        ImageDim::D2 => Dim::D2,
        ImageDim::D3 => Dim::D3,
        ImageDim::Cube if cube_as_array => Dim::D2,
        ImageDim::Cube => Dim::Cube,
        ImageDim::Buffer => Dim::Buffer,
    };
    // Before the format: a refusal of the views holds whatever format the binder would supply.
    let views = if storage {
        level_views(cx, s, shape, facts, push)?
    } else {
        None
    };
    let (format, class) = if storage {
        facts.push(RuntimeFact::StorageTextureFormat { slot: s.slot });
        if s.uses.half_channels {
            facts.push(RuntimeFact::StorageTextureHalfChannels { slot: s.slot });
            push.push(PushField::TextureHalfChannels(s.slot));
        }
        if s.uses.int_channels {
            facts.push(RuntimeFact::StorageTextureIntegerChannels { slot: s.slot });
            push.push(PushField::TextureIntegerChannels(s.slot));
        }
        storage_format(cx, s, texel)?
    } else {
        (
            crate::spv::ImageFormat::Unknown,
            StorageImageClass::Identity,
        )
    };
    // An RGBX format's texels, converted around the class's conversion where the slot reads or
    // writes them (an atomic or `ulong` slot's choice is never one: `storage_format` refuses it).
    let x_channel = storage
        && (s.uses.read || s.uses.write)
        && cx
            .inputs
            .storage_images
            .get(&s.slot)
            .is_some_and(|c| c.x_channel);
    if texel == TexelClass::Uint64 && format == crate::spv::ImageFormat::R64ui {
        // The binder views the slot's `RG32Uint` texture as `R64_UINT`.
        facts.push(RuntimeFact::UlongAtomicView { slot: s.slot });
    }
    if cube_as_array {
        facts.push(RuntimeFact::ImageViewShape { slot: s.slot });
    }
    if shape.dim == ImageDim::Buffer {
        facts.push(RuntimeFact::TexelBufferRange { slot: s.slot });
    }
    if storage {
        let n = views.unwrap_or(1);
        counts.storage_images = counts.storage_images.saturating_add(n);
    } else {
        // The raw view's binding beside the texture's own: an 8-bit sRGB texture's codes, or a
        // 16-bit float texture's own bits.
        counts.sampled_images = counts
            .sampled_images
            .saturating_add(if s.raw || s.half_bits { 2 } else { 1 });
    }
    Ok(ImageDecl {
        dim,
        arrayed: shape.array || cube_as_array,
        multisampled: shape.ms,
        depth: shape.depth,
        usage: if storage {
            ImageUsage::Storage
        } else {
            ImageUsage::Sampled
        },
        format,
        texel,
        cube_as_array,
        class,
        level_views: views,
        extended,
        x_channel,
    })
}

/// Whether a slot of `shape` can have a tile map: a single-sample 2D, 2D array, or 3D color texture
/// (`bind::SparseTileMap`), the sparse types the case model measures.
fn tiled_shape(shape: ImageShape) -> bool {
    !shape.ms
        && !shape.depth
        && match shape.dim {
            ImageDim::D2 => true,
            ImageDim::D3 => !shape.array,
            _ => false,
        }
}

/// The device compares a depth texture's texels (a sample or gather with comparison the shader
/// does not make itself). Metal's `Depth32Float` texels hold any `float`, which a depth image keeps
/// outside `[0, 1]` where the consumer enabled `VK_EXT_depth_range_unrestricted`, or where its
/// driver is measured to keep them without it
/// (`VkProperty::DepthImageKeepsValuesOutsideUnitRange`, MoltenVK's), which the module then
/// requires; the device's comparison reads a depth image (the texture's own, or its depth twin):
/// on a device with neither the function records `PrecisionNote::RestrictedDepthRange`. Its reads,
/// samples and gathers, and the comparisons the shader makes of them, read the texels the consumer
/// holds them in, which a colour image keeps, so they record nothing: the interface's
/// `TextureSlot::access` states them, and the consumer announces an upload its own depth image may
/// clamp (`restricted-depth-upload`). No module requires the extension, since every value within
/// `[0, 1]` is exact without it.
pub(crate) fn depth_comparison(cx: &mut SelectCx<'_>) {
    let enabled = &cx.profile.fields().enabled;
    if enabled
        .extensions
        .contains(&VkExtension::ExtDepthRangeUnrestricted)
    {
        return;
    }
    match enabled
        .properties
        .contains(&VkProperty::DepthImageKeepsValuesOutsideUnitRange)
    {
        true => cx.families.texture.kept_depth_range = true,
        false => {
            cx.precision.insert(PrecisionNote::RestrictedDepthRange);
        }
    }
}

/// Complete the texture family's plan: declare every slot the function reads, writes, or
/// queries, or whose faces it fetches, and contribute its descriptors and runtime facts.
pub fn finish(cx: &mut SelectCx<'_>) -> Result<(), Error> {
    let mut facts = Vec::new();
    let mut push = Vec::new();
    let mut counts = DescriptorCounts::default();
    let mut decls = Vec::new();
    // A slot's format input (`NeedsInput(StorageTextureFormat)`) waits for the heap's classes: a
    // class the device refuses refuses the function whatever format the caller gives, so that
    // refusal is the answer, not an input that would not let the function compile.
    let mut input = None;
    for (index, s) in &cx.families.texture.slots {
        let used = s.uses.read || s.uses.write || s.uses.query || s.uses.atomic || s.uses.faces;
        let Some(shape) = s.shape.filter(|_| used) else {
            continue;
        };
        let d = match declare_slot(cx, s, shape, &mut facts, &mut push, &mut counts) {
            Ok(d) => d,
            Err(needs @ Error::NeedsInput(_)) => {
                input = Some(needs);
                break;
            }
            Err(error) => return Err(error),
        };
        let volatile = forwards(cx, &s.uses, shape.dim == ImageDim::Buffer);
        decls.push((*index, d, volatile));
    }
    // The slots read, sampled or gathered in any class (a slot only sampled is declared by its
    // combined bindings, `sampler::finish`).
    for s in cx.families.texture.slots.values() {
        if s.shape.is_some_and(|shape| cross_reads(s, shape)) {
            facts.push(RuntimeFact::CrossClassReads { slot: s.slot });
        }
    }
    let mut heap = Vec::new();
    for (group, h) in &cx.families.texture.heap {
        let used = h.uses.read || h.uses.write || h.uses.query || h.uses.sample;
        let Some(shape) = h.shape.filter(|_| used) else {
            continue;
        };
        let (class, decl) = declare_heap(cx, h, shape)?;
        if h.texel.is_none() {
            // Only queried: `Float` is a default, not the members' type.
            facts.push(RuntimeFact::HeapNumericType { class });
        }
        heap.push((*group, (class, decl)));
    }
    if let Some(needs) = input {
        return Err(needs);
    }
    // The groups of one heap class index one array: a load through any of them may meet a store
    // through any other at the same element.
    let mut class_uses: std::collections::BTreeMap<ImageHandleClass, Uses> = Default::default();
    for (group, (class, _)) in &heap {
        if let Some(h) = cx.families.texture.heap.get(group) {
            let u = class_uses.entry(*class).or_default();
            u.read |= h.uses.read;
            u.write |= h.uses.write;
            u.atomic |= h.uses.atomic;
        }
    }
    let volatile_classes: Vec<ImageHandleClass> = class_uses
        .iter()
        .filter(|(class, u)| forwards(cx, u, class.dim == SymDim::Buffer))
        .map(|(class, _)| *class)
        .collect();
    for (group, (class, decl)) in heap {
        if let Some(h) = cx.families.texture.heap.get_mut(&group) {
            h.class = Some(class);
            h.decl = Some(decl);
            h.volatile = volatile_classes.contains(&class);
            // The float class of the shape, which the binder holds every depth member in too
            // (`declare_heap`), declared without `Depth`.
            h.plain = (class.depth_compare && h.compares_plain).then_some((
                ImageHandleClass {
                    depth_compare: false,
                    ..class
                },
                ImageDecl {
                    depth: false,
                    ..decl
                },
            ));
            // The unsigned integer class of a sampled float group whose reads decode raw members.
            let sampled_float = class.access == ImageAccess::Sampled
                && class.sampled_type == SampledType::Float
                && !class.depth_compare;
            // The unsigned integer twin of a float storage group whose writes store half bits.
            let storage_float = class.access == ImageAccess::Storage
                && class.sampled_type == SampledType::Float
                && !class.multisampled;
            h.bits = (h.writes_bits && storage_float).then_some((
                ImageHandleClass {
                    sampled_type: SampledType::Uint,
                    format: StorageFormatClass::HalfBits,
                    ..class
                },
                ImageDecl {
                    texel: TexelClass::Uint,
                    ..decl
                },
            ));
            h.raw = (h.reads_raw && sampled_float).then_some((
                ImageHandleClass {
                    sampled_type: SampledType::Uint,
                    ..class
                },
                ImageDecl {
                    texel: TexelClass::Uint,
                    ..decl
                },
            ));
            // The classes of the other numeric types its reads and queries read a member in.
            let color = class.access == ImageAccess::Sampled
                && !class.depth_compare
                && !class.multisampled
                && class.dim != SymDim::Buffer;
            let own = h.texel;
            let mut wanted = h.cross_reads.clone();
            if h.cross_queries {
                wanted.extend([TexelClass::Float, TexelClass::Sint, TexelClass::Uint]);
            }
            h.cross = wanted
                .into_iter()
                .filter(|&k| color && own.is_some_and(|own| own != k))
                .filter_map(|k| {
                    let sampled_type = match k {
                        TexelClass::Float => SampledType::Float,
                        TexelClass::Sint => SampledType::Sint,
                        TexelClass::Uint => SampledType::Uint,
                        TexelClass::Uint64 => return None,
                    };
                    let class = ImageHandleClass {
                        sampled_type,
                        ..class
                    };
                    Some((k, (class, ImageDecl { texel: k, ..decl })))
                })
                .collect();
            // A member of another numeric class reads the class's neutral where an access has no
            // answer for it (any such member), and where a crossing read does not take its format
            // (a depth, stencil, extended-range or two-plane one): the planner reports each such
            // member the stage reaches.
            let every_member =
                h.uncrossed || h.uses.sample || h.uses.write || h.uses.atomic || h.uses.faces;
            if own.is_some() {
                facts.push(RuntimeFact::HeapCrossClassNeutral {
                    stage: cx.stage,
                    class,
                    every_member,
                });
            }
        }
    }
    // The slots whose accesses observe residency, and the binder states whether its bound texture
    // is sparse. On every device, a slot of a type with a tile map whose observed accesses each
    // name their texels (reads, fetches, a pixel sampler's fetch, a sample or gather with Metal's
    // footprint) has a tile-map arm: the residency of those texels' tiles, Metal's whatever the
    // device's sparse blocks. Any other slot has a sparse arm where the profile enables
    // `shaderResourceResidency` and every such access has a sparse form (a texel buffer is never
    // sparse): the residency of the device's blocks. A slot the function updates atomically has
    // neither, and the binder refuses a sparse texture there: Metal's answer to a texture atomic on
    // one is not deterministic.
    let residency = cx
        .profile
        .fields()
        .enabled
        .features
        .contains(&VkFeature::ShaderResourceResidency);
    let mut sparse = Vec::new();
    for (index, s) in &cx.families.texture.slots {
        if !s.uses.residency {
            continue;
        }
        let buffer = s.shape.is_some_and(|x| x.dim == ImageDim::Buffer);
        let atomic = s.uses.atomic;
        let tile_map = !s.uses.untiled_residency && !atomic && s.shape.is_some_and(tiled_shape);
        let sparse_arm = residency && !tile_map && !s.uses.plain_residency && !buffer && !atomic;
        if tile_map {
            counts.storage_buffers = counts.storage_buffers.saturating_add(1);
        }
        facts.push(RuntimeFact::SparseTexture {
            slot: s.slot,
            sparse_arm,
            tile_map,
        });
        sparse.push((*index, sparse_arm, tile_map));
    }
    for (index, sparse_arm, tile_map) in sparse {
        if let Some(s) = cx.families.texture.slots.get_mut(&index) {
            s.sparse_arm = sparse_arm;
            s.tile_map = tile_map;
        }
    }
    // The slots the function writes or updates atomically of a type Metal makes sparse (not a
    // texel buffer or a 1D texture): Metal drops every write to a sparse texture, which the slot's
    // `SparseTexture` specialization drops, and the binder refuses a sparse texture at a slot
    // updated atomically.
    let mut written = Vec::new();
    for (index, s) in &cx.families.texture.slots {
        let sparse_type = s
            .shape
            .is_some_and(|x| !matches!(x.dim, ImageDim::Buffer | ImageDim::D1));
        if !(s.uses.write || s.uses.atomic) || !sparse_type {
            continue;
        }
        facts.push(RuntimeFact::SparseTextureWrites {
            slot: s.slot,
            atomic: s.uses.atomic,
        });
        written.push((*index, !s.uses.atomic));
    }
    for (index, sparse_writes) in written {
        if let Some(s) = cx.families.texture.slots.get_mut(&index) {
            s.sparse_writes = sparse_writes;
        }
    }
    for (index, d, volatile) in decls {
        // An 8-bit sRGB format's codes convert through Metal's twelve-bit tables (`emit::
        // srgb_codes`, `emit::srgb_raw_texel`), which the module declares once.
        if matches!(
            d.class,
            StorageImageClass::SrgbEncode | StorageImageClass::SrgbEncodeBgra
        ) {
            let written = cx
                .families
                .texture
                .slots
                .get(&index)
                .is_some_and(|s| s.uses.write);
            let dir = match written {
                true => crate::air::intrinsics::PackDir::Pack,
                false => crate::air::intrinsics::PackDir::Unpack,
            };
            cx.families.arith.srgb.get_or_insert(dir);
        }
        if let Some(s) = cx.families.texture.slots.get_mut(&index) {
            s.decl = Some(d);
            s.volatile = volatile;
        }
    }
    cx.facts.extend(facts);
    cx.push.extend(push);
    cx.descriptors = cx.descriptors.plus(&counts);
    Ok(())
}
