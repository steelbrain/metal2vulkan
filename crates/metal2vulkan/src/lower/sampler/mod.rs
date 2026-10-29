//! `lower::sampler`: samplers and sampling.
//!
//! **Samplers.** A sampler operand is a runtime `[[sampler(n)]]` parameter (a static slot), an
//! element of a sampler array, or a constexpr sampler's address (`air::abi::StaticSampler`), or a
//! handle selected or merged from those (`memory::handles`), or a sampler loaded from an argument
//! buffer (the heap).
//!
//! **Heap images**. A texture loaded from an argument buffer (a heap element) is sampled through a
//! separate sampler binding, `ResourceKind::HeapSampler`, one per distinct (sampler, comparison,
//! integer texel) of the stage: `OpSampledImage` of the heap element (its access chain
//! `NonUniform`) and that sampler. A constexpr sampler is the binding's immutable sampler, made for
//! its use as a combined binding's is; the sample of the null element (the null or a missed handle)
//! is zero, as a heap read's. A comparison samples a heap depth texture of the depth-comparison
//! class through a comparison heap sampler. A runtime `[[sampler(n)]]` beside a heap image is a
//! heap sampler binding without an immutable sampler, which the binder writes with the sampler
//! bound at slot `n` made for the use, as a combined binding's runtime sampler. Several sampler
//! candidates beside a heap image are not implemented (`ArgumentBufferTextures`).
//!
//! **Argument buffers' samplers** (the sampler half). A heap image sampled through a sampler loaded
//! from an argument buffer reads `OpSampledImage` of its heap element and of the heap sampler's
//! element in the set-1 sampler array of the use (`HandleClass::Sampler`: with comparison or
//! without, of an integer texture or not, for a cube's explicit gradients or not), both
//! `NonUniform`. The binder writes element `i + 1` of each such array with the argument buffers'
//! sampler `i` made for the use, and element 0 with the neutral sampler (a missed or null sampler
//! handle). The handle is the index of the sampler's table entry: the sampling reads the entry's
//! element, and its state word (`metal::sampler::SamplerWord`) where it tests the sampler's pixel
//! coordinates ([`Pixel::Heap`]). A static texture sampled through a heap sampler is not
//! implemented (`ArgumentBufferTextures`).
//!
//! **The read sampler** (`air.get_read_sampler`) is what Metal's front end passes as the sampler
//! operand of a texture `read`, which ignores it: it denotes no sampler state and carries no
//! runtime value, so its strategy yields none ([`SamplerStrategy::ReadSampler`]) and it is never
//! carried between blocks. A `read` in any block may use it; every other use (a sample, gather, or
//! LOD query through it, a merge, a store) is refused where it is selected, so no emitted
//! instruction reads it.
//!
//! **Several candidates**. A sampling operation whose texture or sampler has several static
//! candidates is a switch-at-use over every (texture, sampler) combination, each arm the static
//! sampling of its combined binding. A switch-at-use holds no tangled instruction: a fragment's
//! implicit-LOD sample takes the coordinate's derivatives before the switch, scaled by `2^bias`
//! (the whole bias, as Metal applies it), and samples with those gradients in every arm ([`Hoist`],
//! `PrecisionNote::GradFromBias`) unless every arm fetches. An LOD query has no gradient form:
//! every candidate's query runs before the switch, at the call's own position with the call's
//! coordinate, and the switch selects the handles' one, up to [`LOD_QUERY_CANDIDATES`] combinations
//! (past them `DivergentImplicitLod`, not implemented). Runtime and constexpr samplers mix freely
//! among the candidates. The binding-array rule (rule 1) is not implemented.
//!
//! **Bindings**. Every sampling operation (sample, gather, LOD query) reads one combined image
//! sampler, `ResourceKind::CombinedTextureSampler`, per distinct (texture slot, sampler,
//! comparison) of the function; its image is the texture's own shape as a sampled image, and a
//! constexpr sampler is baked into the layout for its use ([`constexpr_desc`]). Selecting a
//! sampling operation records the texture's use (sampled, with comparison) in the texture
//! family's slot, whose plain binding exists only for reads, writes, and queries.
//!
//! **Separate bindings**. A combined binding counts as one sampler and one sampled image, so a
//! stage whose combined bindings (with its heap samplers) are more samplers than
//! `maxPerStageDescriptorSamplers` takes every one of them apart: the same sampled image as a
//! separate texture binding (`ResourceKind::SeparateTexture`), sampled through the stage's
//! separate sampler binding of its use (`ResourceKind::SeparateSampler`, one per distinct
//! (sampler, comparison, integer texel, cube gradients), a constexpr one its immutable sampler),
//! combined by `OpSampledImage` in the block that samples. Each pair keeps its image and its
//! `VkSampler`, so it samples exactly as its combined binding does; only the sampler count
//! changes. Every binding is a static slot (an array element is its own slot, `memory::handles`),
//! so neither half is indexed and neither takes `NonUniform`. Within the limit the stage keeps
//! its combined bindings.
//!
//! **Level of detail**. In a fragment function an implicit-LOD sample (with its bias) and
//! `calculate_*_lod` use the fragment's implicit derivatives and are tangled. Metal reads a bias as
//! `B = clamp(floor(half_rne(bias) * 64) / 64, -32, 31.984375)` and applies it whole, where Vulkan
//! clamps a `Bias` operand to `maxSamplerLodBias`. Through a sampler without a bias of its own the
//! `Bias` operand is the bias as given, which Metal's own sampler (MoltenVK) reads as Metal does and
//! another device's sampler reads its own way. Through a constexpr sampler with a bias of its own
//! the shader forms `B` (`emit::metal_bias`): the `Bias` operand within the limit, and a bias that
//! may reach past it also samples with the derivatives scaled by `2^B` ([`BiasReach`],
//! `bias_gradients`). Through an
//! anisotropic sampler of a 2D texture Metal's `B` shifts the footprint's minor axis alone, which
//! the axis gradients give (`emit::axis_gradients`): a constexpr sampler's always, a runtime or
//! argument buffer's sampler's where its word states anisotropy, chosen per invocation. Outside a
//! fragment function, Metal's sample without a level or gradients reads the
//! default level 0, shifted by its bias and bounded below by its `min_lod_clamp`: an explicit level
//! ([`Lod::Base`]), and `calculate_*_lod` is undefined in Metal (`NotRepresentable`). Gradients,
//! explicit levels, and gathers (which read the base level) are not tangled.
//!
//! **Pixel coordinates**. Vulkan unnormalized samplers, which take no gather, offset, comparison,
//! array, 3D image or level of detail, are never used: the plane coordinate (and any gradient) is
//! divided by the level-0 size in the shader, always for a constexpr pixel sampler, and selected by
//! the word of the slot's `SamplerState` specialization for a runtime sampler, so the one sampling
//! instruction reads either coordinate. The device's sampler reads its texels from that scaled
//! coordinate in its own arithmetic, which near a texel edge or a filter step's edge may read
//! another texel or weight than Metal's at the coordinate as given
//! (`PrecisionNote::PixelSamplerScaling`, `PrecisionNote::ArgumentPixelSamplerScaling`). A
//! constexpr pixel sampler with nearest filtering and clamp-to-edge addressing instead fetches its
//! texel, `OpImageFetch(clamp(floor(coord) + offset))` ([`SampleOp::PixelFetch`]), which also
//! takes a dynamic offset on any profile. An argument buffer's sampler is known only at run time:
//! a sample through one without comparison emits both the scaled sample and that fetch, and
//! selects the fetch where its table entry's word states the state
//! (`metal::sampler::SamplerWord::exact_fetch`); gathers and comparisons stay scaled.
//!
//! **Texel offsets**. Metal reads each component of an offset as the two's-complement value of its
//! low four bits ([`metal_offset`]), which is the identity on its documented range
//! [`METAL_TEXEL_OFFSET`]. A constant offset is that value as `ConstOffset` within the device's
//! range. A dynamic offset is the `Offset` operand of that value (`OpBitFieldSExtract(v, 0, 4)`):
//! on a gather it needs `shaderImageGatherExtended`, on a sample also `maintenance8`
//! (`VK_KHR_maintenance8`); a nearest pixel fetch adds it to the texel coordinate on any profile.
//! Without those features the access takes the offset's arms ([`OffsetArms`]): a switch over the
//! values Metal reads on its dynamic lanes, each arm the access with its value as `ConstOffset`,
//! which reads what the `Offset` operand of that value reads; a fragment's implicit-level sample
//! takes its derivatives before the switch (`PrecisionNote::GradFromBias`), and a function past
//! [`OFFSET_ARMS_PER_FUNCTION`] arms is refused as without them.
//! Both operands are signed `int` vectors, as Metal's are (the builder admits no other: SPIRV-Cross
//! would pass an unsigned one to MSL, which rejects it). A cube takes no offset in Metal or SPIR-V:
//! a nonzero one there is `NotRepresentable(TexelOffset)`.
//!
//! **Bicubic filtering**. A sample through a constexpr sampler with Metal's bicubic filter is
//! [`SampleOp::Bicubic`]: Metal's measured filter emulated by fetches of the base level
//! ([`bicubic`]), whose treatment of the bound format is the texture slot's `BicubicTexels`
//! specialization; a gather through one is an ordinary gather.
//!
//! **The device's sampler**. Every other sample, gather and comparison of a format the device
//! samples is the device's: Metal's own on a device that samples through Metal's own sampler
//! (`VkProperty::MetalSampler`, MoltenVK), and the device's own elsewhere
//! (`VkProperty::DeviceSampling`, recorded as `PrecisionNote::DeviceSampling`); the module
//! requires the property. A cube's direction takes Metal's face at a tie (`face::metal_ties`).
//! Every footprint Metal's rule decides in the shader (the emulated filters the device has no form
//! for) reads Metal's position ([`position`]): the coordinate wrapped into one period first on a
//! repeating or mirrored axis, then scaled with ties away from zero. Metal reads a cube's
//! coordinate as a direction through a pixel sampler too, so a cube's pixel sampler is a normalized
//! one here.
//!
//! Not lowered yet: a static texture sampled through an argument buffer's sampler, a heap image
//! sampled through several sampler candidates, the binding-array rule, and the bicubic forms
//! [`bicubic`] refuses.

pub mod anisotropic;
pub mod bicubic;
mod constexpr;
mod emit;
pub mod extended;
pub mod face;
pub mod footprint;
pub mod heap_raw;
pub mod in_place;
pub mod level;
pub mod linear;
pub mod position;
pub mod routine;
mod select;
mod shared;

pub use constexpr::{address_modes, constexpr_desc, state_of, state_words};
pub use emit::{hoisted_gradients, sampling_value};
pub use select::{select_heap_sampling, select_sampling};
pub(crate) use shared::{shared_value, Candidates};

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};

use crate::air::abi::{AbiInt, ParamBinding, ParamRole};
use crate::air::intrinsics::{AirIntrinsic, SamplerOp};
use crate::air::ir::{Callee, Inst, InstId, InstKind, ValueId};
use crate::error::{Error, Feature, InternalError, InternalKind, LowerFamily, Unsupported};
use crate::lower::emit::EmitCx;
use crate::lower::select::{DescriptorCounts, FamilyPlans, Plan, SelectCx};
use crate::lower::strategy::FamilyStrategy;
use crate::lower::texture::{ImageDecl, TexelClass, TexelForm};
use crate::memory::handles::ENTRY_FIELD_WORDS;
use crate::metal::sampler::{SamplerUse, SamplerWord};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::interface::SamplerSlot;
use crate::pipeline::lower_inputs::StorageImageClass;
use crate::pipeline::sym::{
    CombinedSampler, HandleClass, ResourceKind, ResourceSlot, SamplerRef, SlotRef, SpecKey,
};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{ConstId, DemoteTy, GlobalId, Value as SValue};
use crate::spirv::ir::{ConstBits, Dim, ExecMode, ImageUsage, StorageClass, Type as SType};

fn refuse() -> Error {
    Unsupported::not_implemented(Feature::Lowering(LowerFamily::Sampler)).into()
}

fn invariant(detail: &str) -> Error {
    Error::Internal(InternalError::new(InternalKind::Invariant, detail))
}

/// The sampler strategies of the family's own instructions.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SamplerStrategy {
    /// `air.get_read_sampler`: no value.
    ReadSampler,
}

impl FamilyStrategy for SamplerStrategy {
    fn name(&self, _: &FamilyPlans) -> Cow<'static, str> {
        Cow::Borrowed(match self {
            SamplerStrategy::ReadSampler => "sampler.read_sampler",
        })
    }

    fn words(&self) -> u32 {
        match self {
            SamplerStrategy::ReadSampler => 0,
        }
    }

    /// The read sampler has no value.
    fn yields_value(&self) -> bool {
        match self {
            SamplerStrategy::ReadSampler => false,
        }
    }
}

/// One combined image-sampler binding of the function, before its texture's numeric type is
/// final: the texture slot index, the sampler, and whether it samples with comparison.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BindingKey {
    pub texture: u32,
    pub sampler: SamplerRef,
    pub compare: bool,
}

/// One separate heap sampler use: heap group `group` sampled through `sampler`, with comparison or
/// without. Its binding is final once the group's numeric type is ([`finish`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HeapKey {
    pub group: u32,
    pub sampler: SamplerRef,
    pub compare: bool,
}

/// One use of an argument buffer's sampler: heap group `group` sampled through a heap sampler
/// handle, with comparison or without. Its sampler array is final once the group's numeric type
/// is ([`finish`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HeapSamplerKey {
    pub group: u32,
    pub compare: bool,
}

/// An upper estimate of the SPIR-V words of an argument-buffer sampler's cubic test past its
/// entry's word: the mask, the test, the selection construct, and the status mark (the push field's
/// load, the address, the pointer, the bit, and the atomic `OR`).
pub const CUBIC_REACH_WORDS: u32 = 64;

/// The sampled image a sampling operation reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Source {
    /// A combined image-sampler binding.
    Combined(BindingKey),
    /// Element `handle` (a heap handle's value) of the heap array of `key.group`, through the
    /// separate heap sampler of `key`.
    Heap { handle: ValueId, key: HeapKey },
    /// Element `handle` of the heap array of `key.group`, through element `sampler` (a heap
    /// sampler handle's value) of the sampler array of `key`'s use.
    HeapSampler {
        handle: ValueId,
        sampler: ValueId,
        key: HeapSamplerKey,
    },
}

/// How a fragment's nonzero bias reaches the level. Metal shifts the level by its own reading of
/// the bias `b`, `B = clamp(floor(half(b) · 64) / 64, -32, 31.984375)` (`emit::metal_bias`; an
/// Apple M4, every quad of five samplers alike), the whole of it: Vulkan clamps the `Bias` operand,
/// with the sampler's own `mipLodBias`, into `±maxSamplerLodBias` (2 to 16 on the committed
/// profiles), where Metal does not (an Apple M3 Max magnifies with a bias of -17.5, -100, and -1000
/// at a level of 17 before it, authored case `a2_fragment_bias_clamp`). Through an isotropic
/// sampler Metal's level is the derivatives' level plus `B`: within the limit the `Bias` operand
/// `B`; past it only explicit gradients reach the level, the coordinate's derivatives scaled by
/// `2^B` (`bias_gradients`). Through an anisotropic sampler `B` shifts only the footprint's minor
/// axis: Metal's level is `max(log2 minor + B, log2 major - log2 maxAnisotropy)`, which neither the
/// `Bias` operand nor `2^B` derivatives give, but the axis gradients do (`emit::axis_gradients`).
/// Every route but `Metal`'s takes `B` from the module and the device's level arithmetic,
/// `PrecisionNote::GradFromBias` where gradients carry it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BiasReach {
    /// A sampler without a bias of its own: the `Bias` operand as given, which Metal's own sampler
    /// (`VkProperty::MetalSampler`, MoltenVK) reads as Metal reads a bias, and another device's
    /// sampler its own way.
    Metal,
    /// No bias, or a constant whose `B`, with the sampler's bias, is within the limit, through an
    /// isotropic constexpr sampler: the `Bias` operand `B` alone.
    Within,
    /// A constant whose `B`, with the sampler's bias, is past the limit, through an isotropic
    /// constexpr sampler: the derivatives scaled by `2^B` alone.
    Beyond,
    /// Through an anisotropic constexpr sampler of a 2D plane: the axis gradients alone.
    Axes,
    /// Decided per invocation: the `Bias` operand `B` where `|B + sampler| <= maxSamplerLodBias`,
    /// and the derivatives scaled by `2^B` past the limit (both from the sample's own position, so
    /// the implicit sample and the derivatives stay in the fragment's uniform control flow). A
    /// dynamic bias through a constexpr sampler with a bias of its own.
    Either {
        /// The profile's `maxSamplerLodBias` (`f32` bits).
        max_bias_bits: u32,
        /// The sampler's own bias (`f32` bits): the constexpr sampler's `mipLodBias`.
        sampler_bias_bits: u32,
    },
}

/// How a sample chooses its level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Lod {
    /// The fragment's implicit derivatives, shifted by a nonzero bias that reaches `reach`.
    Implicit {
        bias: Option<ValueId>,
        reach: BiasReach,
    },
    /// An explicit level; `None` is level 0.
    Level(Option<ValueId>),
    /// Outside a fragment function, a sample without a level or gradients reads Metal's default
    /// level 0, shifted by its whole bias and clamped from below by its `min_lod_clamp`: the
    /// explicit level `max(bias, min)`.
    Base {
        bias: Option<ValueId>,
        min: Option<ValueId>,
    },
    /// Explicit gradients `(dPdx, dPdy)`.
    Grad(ValueId, ValueId),
    /// Explicit gradients `(dPdx, dPdy)` of a cube direction, sampled with Metal's footprint
    /// in the form [`CubeFootprint`] names: emission gives the device gradients from which its cube
    /// derivative rule reads it (`emit::cube_footprint`), and the module records
    /// `PrecisionNote::CubeGradientFootprint`.
    CubeGrad(ValueId, ValueId, CubeFootprint),
    /// The fragment's implicit derivatives of the coordinate, hoisted before a switch-at-use by
    /// the enclosing [`Hoist`]: an explicit-gradient sample.
    Hoisted,
}

/// The footprint a cube's explicit gradients take, and the device rule it is handed for
/// (`emit::cube_footprint`): Metal's under macOS 27 (the oracle, the Apple M3 Max), but where the
/// stage's route is MoltenVK's graphics fixup on macOS 26.7's identity, which hands that GPU its
/// own earlier footprint.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CubeFootprint {
    /// Metal's isotropic footprint, through an isotropic constexpr sampler: the Metal length of
    /// each gradient's two tangent components, on one tangent, for Vulkan's cube derivative rule
    /// (`CubeGradientRoute::Vulkan`).
    Length,
    /// Metal's anisotropic face gradients through an anisotropic constexpr sampler: each gradient
    /// without its normal component, mirrored on the faces Metal mirrors, for Vulkan's rule.
    Face,
    /// The gradients for MoltenVK's graphics fixup under macOS 27
    /// (`CubeGradientRoute::AppleProjected`), through any sampler: an `x` or `z` face's without its
    /// normal component, which Metal reads as its own; a `y` face's Metal length on `x`, which the
    /// fixup hands Metal on both tangents (an isotropic sample's footprint kept, an anisotropic
    /// one's ellipse on the diagonal).
    Projected,
    /// Metal's gradients as the call states them, through any sampler: for a driver that hands
    /// the stage's cube gradients to Metal's own `gradientcube` unchanged
    /// (`CubeGradientRoute::Metal`), whose GPU reads them by Metal's own rule. Exact: the module
    /// records no `PrecisionNote::CubeGradientFootprint` for it.
    Metal,
    /// A runtime or argument buffer's sampler's, by its word: [`Face`] where the word states
    /// `SamplerWord::ANISOTROPIC`, else [`Length`] (`emit::word_footprint`).
    ///
    /// [`Face`]: CubeFootprint::Face
    /// [`Length`]: CubeFootprint::Length
    Word,
}

/// For an implicit-LOD sample through a multi-candidate texture or sampler: the coordinate's
/// derivatives, taken before the switch-at-use in the sample's own position, scaled by `2^bias`
/// with the whole bias, as Metal applies it (`bias_gradients`; a constexpr arm's sampler adds its
/// own `mipLodBias`), are the gradients of every arm's sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Hoist {
    /// The plane coordinate: `lanes` floats.
    pub coord: ValueId,
    pub lanes: u8,
    /// A nonzero bias.
    pub bias: Option<ValueId>,
}

/// The most (texture, sampler) combinations whose LOD queries a fragment `calculate_*_lod` through
/// a switch-at-use evaluates before the switch. Every fragment runs each candidate's query, so this
/// bounds the call's cost at sixteen image instructions; past it the call is
/// `DivergentImplicitLod`, not implemented (the binding-array rule would make one query).
pub const LOD_QUERY_CANDIDATES: usize = 16;

/// The census name of the sampling operations that read separate bindings: each one's uses and the
/// words the separate form adds to it, [`SEPARATE_WORDS`].
pub const SEPARATE_BINDING: &str = "sampler.separate_binding";

/// The estimated words the separate form adds to a sampling operation: the loads of the texture and
/// the sampler, and `OpSampledImage`.
pub const SEPARATE_WORDS: u32 = 16;

/// The estimated words of `bias_gradients` with the two derivatives it scales: the derivatives, the
/// largest magnitude, its exponent, the two normalizations, the clamped exponent sum, `Exp2`, and
/// the two products, with their constants.
pub const BIAS_GRADIENT_WORDS: u32 = 120;

/// The estimated words of `emit::metal_bias`: the clamp into `±64`, the bits' rounding to a
/// `half`'s precision, the small-magnitude select, the floor at `1/64` and the clamp into Metal's
/// range, and the NaN select, with their constants.
pub const METAL_BIAS_WORDS: u32 = 110;

/// The estimated words of `emit::axis_gradients` beyond the derivatives: the level-0 size, the two
/// texel-space lengths, the minor axis's scale, its floor at a sixteenth of the major (the lift and
/// the perpendicular) and the major axis's raise, and the selects that place them.
pub const AXIS_GRADIENT_WORDS: u32 = 250;

/// The estimated words a switch-at-use that emulates once adds besides its arms (SEMANTICS 1012,
/// `sampler::shared_value`): per dispatch (the facts, the taps' reads, a linear sample's device
/// sample) its selection, default arm, and merge, the facts' null values, and the selection of the
/// switch's null value.
pub const SHARED_WORDS: u32 = 160;

/// Metal's texel offset range, inclusive: "This integer value must be in the range -8 to +7"
/// (MSL 4.1 §6.13, for `sample`, `sample_compare`, `gather`, and `gather_compare`). Every Vulkan
/// device's offset and gather-offset ranges must contain it (the spec's required limits), and
/// selection checks that the profile's do.
pub const METAL_TEXEL_OFFSET: (i32, i32) = (-8, 7);

/// The bits of an offset component Metal reads: the low four, as a two's-complement value.
pub const METAL_TEXEL_OFFSET_BITS: u32 = 4;

/// The texel offset Metal applies for offset component `x`: the two's-complement value of its
/// low [`METAL_TEXEL_OFFSET_BITS`] bits, always in [`METAL_TEXEL_OFFSET`] and the identity there.
/// An Apple M3 Max applies exactly this to every dynamic `sample` and `gather` offset of authored
/// case `a2_dynamic_texel_offset` (8 as -8, -9 as 7, 16 as 0, `i32::MAX` as -1, 128 of 128 words),
/// where the clamp answered 7 for 8.
pub fn metal_offset(x: i32) -> i32 {
    let unused = 32 - METAL_TEXEL_OFFSET_BITS;
    x.wrapping_shl(unused).wrapping_shr(unused)
}

/// A texel offset.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Offset {
    /// A constant within the device's range (`ConstOffset`).
    Const(Vec<i32>),
    /// A dynamic offset, read as [`metal_offset`] reads it: the `Offset` operand of a gather
    /// (`ImageGatherExtended`) or a sample (also `maintenance8`), or where the profile lacks that
    /// operand's features its [`OffsetArms`]; a pixel fetch adds the value to the texel
    /// coordinate.
    Dynamic(ValueId, Option<OffsetArms>),
}

/// A dynamic texel offset on a profile without the `Offset` operand's features: the access is a
/// switch over the values Metal reads on the offset's dynamic lanes, each arm the access with that
/// value as `ConstOffset` (`spirv::builder::ops::image::ArmedOffset`), which reads the texels the
/// `Offset` operand of the value reads. A lane the IR states as a constant takes its constant in
/// every arm, so an offset of one dynamic lane has 16 arms and one of two 256.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct OffsetArms {
    /// Each plane lane's offset as Metal reads it ([`metal_offset`]) where the IR states it
    /// constant, else `None`.
    pub lanes: Vec<Option<i32>>,
}

impl OffsetArms {
    /// The number of arms: 16 to the power of the dynamic lanes.
    pub fn arms(&self) -> u32 {
        self.lanes
            .iter()
            .filter(|l| l.is_none())
            .fold(1u32, |n, _| n.saturating_mul(OFFSET_ARM_VALUES))
    }
}

/// The values one dynamic offset lane takes, Metal's range [`METAL_TEXEL_OFFSET`].
pub const OFFSET_ARM_VALUES: u32 = 1 << METAL_TEXEL_OFFSET_BITS;

/// The most arms of armed offsets ([`OffsetArms`]) one function's accesses take together: two
/// accesses of two dynamic lanes, or 32 of one; past it the access is refused as without the arms
/// (`docs/FAST-PATHS.md`, known gaps). A driver's pipeline making grows faster than the arms: ANV
/// (Mesa 26.2.4, Iris Xe) made a kernel of 512 arms in 0.63 s (0.02 s with the `Offset` operand),
/// of 1056 in 3.2 s and of 1536 in 5.9 s, past the 5-s making ceiling.
pub const OFFSET_ARMS_PER_FUNCTION: u32 = 512;

/// The estimated words of one arm of an armed offset: its case literal and label, its sampled
/// image, the access with its `ConstOffset`, its branch, and its merge operands.
pub const OFFSET_ARM_WORDS: u32 = 24;

/// The level, in sixty-fourths, a clamped LOD query outside a fragment function answers before
/// the texture's last level bounds it ([`SampleOp::QueryLevel`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MinLevel {
    /// A constexpr sampler's, known when the function is compiled.
    Const(u32),
    /// The runtime sampler of slot `n`'s, its `SpecKey::SamplerMinLevel` specialization.
    Slot(u32),
}

/// The sampling operation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SampleOp {
    /// `OpImageSample*`: with a comparison reference (`dref`), a `MinLod` clamp, and an offset.
    Sample {
        lod: Lod,
        dref: Option<ValueId>,
        min_lod: Option<ValueId>,
        offset: Option<Offset>,
    },
    /// `OpImageGather` of `component` (0 for a depth gather), or `OpImageDrefGather` with a
    /// comparison reference (`dref`).
    Gather {
        component: u32,
        dref: Option<ValueId>,
        offset: Option<Offset>,
    },
    /// `OpImageQueryLod`: the clamped (`x`) or unclamped (`y`) level.
    Lod { clamped: bool },
    /// Metal's `calculate_clamped_lod` or `calculate_unclamped_lod` outside a fragment function,
    /// measured on the M3 Max and an M4: the level of detail is 0, so the unclamped level (`min`
    /// `None`) is 0, and the clamped one is the sampler's `metal::sampler::query_min_level` bounded
    /// by the texture's last level (`OpImageQueryLevels` less 1).
    QueryLevel { min: Option<MinLevel> },
    /// A sample through a nearest, clamp-to-edge constexpr pixel sampler: the exact texel
    /// `OpImageFetch(clamp(floor(coord) + offset, 0, size - 1))` of the base level, whose offset
    /// may be dynamic.
    PixelFetch { offset: Option<Offset> },
    /// A sample through a constexpr sampler with Metal's bicubic filter, emulated by fetches of
    /// the base level.
    Bicubic(bicubic::Bicubic),
}

/// Whether the coordinate is in pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Pixel {
    /// Normalized.
    None,
    /// A constexpr pixel-coordinate sampler: always scaled.
    Always,
    /// A runtime sampler in slot `n`: scaled when its `SamplerState` specialization's word states
    /// pixel coordinates.
    Spec(u32),
    /// An argument buffer's sampler of heap handle `handle`, the index of its sampler table
    /// entry: scaled when the entry's word states pixel coordinates; a sample without comparison is
    /// fetched exactly instead where the word states what
    /// `metal::sampler::SamplerWord::exact_fetch` tests for its image.
    Heap(ValueId),
}

/// A sampling operation on a combined image sampler, or on a heap element through a heap sampler.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Sampling {
    pub source: Source,
    pub op: SampleOp,
    /// The plane coordinate: `lanes` floats.
    pub coord: ValueId,
    pub lanes: u8,
    /// The array layer (an integer).
    pub layer: Option<ValueId>,
    /// The held texel (`None` for an LOD query).
    pub texel: Option<TexelForm>,
    /// The texture is a depth texture (its sample reads one component).
    pub depth: bool,
    /// The lowered type of the result (`{T, i8}` or the LOD query's `float`).
    pub result: DemoteTy,
    pub pixel: Pixel,
    /// The function observes bit 0 of the result's status word: on a slot with a sparse arm a
    /// sample, gather, or fetch takes its sparse form under the slot's `SparseTexture`
    /// specialization; otherwise (and for a bicubic sample, which has none) the word is 0.
    pub residency: bool,
    /// Metal's footprint of an observed sample or gather where the rule decides it: its word is the
    /// residency of exactly those texels, through the slot's sparse arm or tile map.
    pub footprint: Option<footprint::Footprint>,
    /// It reads a cube by a direction, whose ties between major axes the device is handed broken
    /// as Metal breaks them ([`face`]).
    pub cube: bool,
    /// Whether the access reads one texel of one level: its `half` texel's lanes for the class's
    /// conversion.
    pub one_texel: crate::lower::texture::OneTexel,
    /// A `half` texel of the slot, or of the heap color texture, converts by the bound format's
    /// class (`texture::half`): not a raw, bicubic or depth one's, nor where the sample's texel is
    /// the device's (`texture::device_texels`: a device that does not sample through Metal's own
    /// sampler), rounded exactly. The emission follows it.
    pub half_texel: bool,
    /// Metal's value of the texels the access reads in a block of a sparse image that holds no
    /// mapped tile.
    pub unmapped: Unmapped,
    /// A heap sample or gather that answers, where its handle names a texture the heap holds by
    /// its raw texels alone, Metal's sample or gather of that texture ([`heap_raw`]).
    pub heap_raw: Option<heap_raw::HeapRaw>,
    /// The slot's variant binds a texture of another numeric class
    /// (`LoweringInputs::cross_class`): [`texel`](Self::texel) and [`result`](Self::result) are
    /// that class's own (`texture::native_form`), whose words the sampling hands back as this
    /// declared result ([`sampling_value`]).
    pub declared: Option<DeclaredResult>,
    /// The slot is a packed normalized sampled slot of `LoweringInputs::swizzled_storage`, whose
    /// view takes no swizzle: the sampling swizzles the texel its class converts under the slot's
    /// `TextureSwizzle` specialization, a gather of a component gathering the component the
    /// swizzle names, and sets the lanes Metal reads as constants after it ([`sampling_value`]).
    pub swizzle: bool,
    /// A slot's access that places a texel offset by Metal's rule on an axis that may repeat or
    /// mirror (a bicubic filter's footprint): the rule reads the slot's `OffsetBaseAxes`
    /// specialization's tests (`texture::OffsetBase`), which state where a view's parent's
    /// base-level size, which Metal reads, is not a power of two where the bound view's is
    /// (`position::offset_dropped`).
    pub offset_base: bool,
    /// A sample through an anisotropic sampler that Metal's anisotropic filter, emulated by the
    /// device's samples at explicit levels, answers ([`anisotropic`]): through a constexpr
    /// sampler always, through a runtime sampler slot where its word states anisotropy.
    pub anisotropic: Option<anisotropic::Anisotropic>,
    /// The slot's binding views the codes of an 8-bit sRGB texture held as its UNORM twin
    /// (`LoweringInputs::srgb_textures`): what the device reads there (one texel's codes, or its
    /// filter of them) is decoded by the device's transfer (`texture::srgb_texel`), or a bicubic
    /// sample's taps are (`texture::srgb_raw_tap`).
    pub raw: bool,
    /// A fragment function's depth comparison whose reference the pipeline's private
    /// `fragmentDepthCompareClampMask` clamps (`LoweringInputs::depth_compare_clamp_mask`): the
    /// reference is clamped to `[0, 1]`, a NaN taken as 0, before every comparison
    /// ([`emit::reference`]).
    pub clamp_reference: bool,
}

/// The declared result of a sampling whose slot's variant binds a texture of another numeric
/// class ([`Sampling::declared`]).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DeclaredResult {
    /// The declared texel's conversion.
    pub conv: crate::lower::texture::TexelConv,
    /// The declared result's lowered type (`{T, i8}` or `T`).
    pub result: DemoteTy,
}

/// `result`, the lowered type of a sampling's result (`{T, i8}` or `T`), with its texel's
/// elements those of a texel of conversion `conv` (`F32` or `I32`; `F16`, a `short` or `ushort`
/// sampling's of a float texture): the result a sampling of a texture's own class takes
/// (`texture::native_form`). `None` where it holds no texel.
pub(crate) fn native_result(
    result: &DemoteTy,
    conv: crate::lower::texture::TexelConv,
) -> Option<DemoteTy> {
    use crate::lower::texture::TexelConv;
    use crate::spirv::ids::ScalarTy;
    let elem = match conv {
        TexelConv::F32 => ScalarTy::Float { bits: 32 },
        TexelConv::F16 => ScalarTy::Float { bits: 32 },
        TexelConv::I32 => ScalarTy::Int { bits: 32 },
        _ => return None,
    };
    let texel = |t: &DemoteTy| match t {
        DemoteTy::Vector { len, .. } => Some(DemoteTy::Vector { elem, len: *len }),
        DemoteTy::Scalar(_) => Some(DemoteTy::Scalar(elem)),
        _ => None,
    };
    match result {
        DemoteTy::Composite(parts) => {
            let (first, rest) = parts.split_first()?;
            let mut out = vec![texel(first)?];
            out.extend(rest.iter().cloned());
            Some(DemoteTy::Composite(out.into()))
        }
        other => texel(other),
    }
}

/// Metal's value of an access's texels in a block of a sparse image that holds no mapped tile,
/// which reads the device's value (RADV: zero in every lane, where Metal reads a format's missing
/// alpha as 1; a device without `residencyNonResidentStrict`: any value), where the module sets
/// it: on a profile whose sparse textures are sparse images (it enables `shaderResourceResidency`),
/// for each access of a slot's color texture but a bicubic sample's fetches and a level query,
/// whether the function reads its residency or not. Elsewhere the bound image holds Metal's value,
/// or the binder refuses a sparse texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Unmapped {
    /// The device's value: the profile's sparse textures are no sparse images, or the access
    /// reads depth, a heap element, or a bicubic sample's taps.
    Device,
    /// The lanes Metal reads as constants (a channel the format lacks, through the swizzle), the
    /// slot's `ConstantLanes` specialization's, are set after the access's conversion; the
    /// others are the device's: an access whose residency the function does not read, a
    /// filtered sample, or an access whose texels the device's footprint decides.
    Constants,
    /// As [`Unmapped::Constants`], after the texel the access reads reads zero where its residency
    /// says it is unmapped, Metal's unmapped value (zero memory, its missing channels the
    /// constants): a pixel fetch, whose texel and residency the function names.
    Named,
}

impl Sampling {
    /// The shape the arms of a switch-at-use share where it emulates once (SEMANTICS 1012,
    /// `sampler::shared_value`): this sampling with its texture slot and sampler taken out, where
    /// it is an emulation on a combined binding whose residency the function does not observe: a
    /// bicubic sample. Arms of one shape differ only in the binding they read: each emulation
    /// states its sampler's addressing in its own fields.
    pub(crate) fn shared_shape(&self) -> Option<Sampling> {
        let Source::Combined(key) = self.source else {
            return None;
        };
        // A texture of another class's words are handed back as declared in each arm's own access. A swizzled
        // slot's sampling swizzles under its own slot's swizzle; an offset placed by a slot's
        // `OffsetBaseAxes` is placed by its own slot's.
        if self.residency
            || self.footprint.is_some()
            || self.declared.is_some()
            || self.swizzle
            || self.offset_base
        {
            return None;
        }
        let mut k = self.clone();
        k.source = Source::Combined(BindingKey {
            texture: 0,
            sampler: SamplerRef::Slot(0),
            ..key
        });
        let SampleOp::Bicubic(b) = &mut k.op else {
            return None;
        };
        b.texture = 0;
        Some(k)
    }

    /// The estimated words each arm of a switch-at-use that emulates once adds (SEMANTICS 1012):
    /// per dispatch its case, label, branch, and merge operands, and its binding's load, and in it
    /// the reads of the candidate's own image: the facts' size query and the bicubic filter's tap
    /// fetches. [`Sampling::words`] counts the emulation once.
    pub(crate) fn shared_arm_words(&self) -> u32 {
        // A dispatch's arm: its case, label, branch, and merge operands, the binding's loads,
        // `OpSampledImage` of a separate binding, and `OpImage`.
        const ARM: u32 = 48;
        // A read in an arm: the fetch or access with its operands, and its merge operands.
        const READ: u32 = 16;
        let facts = ARM + 3 * READ;
        let SampleOp::Bicubic(b) = &self.op else {
            return facts;
        };
        // A 1D texture's one row of two taps, else each row of four and the half sum's row its
        // loop reads.
        let (rows, taps) = if b.address.len() == 1 { (2, 2) } else { (5, 4) };
        facts + rows * (ARM + taps * READ)
    }

    /// The census name; a heap element's names its operation, and one through an argument buffer's
    /// sampler its operation too.
    pub fn name(&self) -> &'static str {
        let compares = matches!(self.op, SampleOp::Gather { dref: Some(_), .. });
        if let Source::HeapSampler { .. } = self.source {
            return match &self.op {
                SampleOp::Sample { .. } => "sampler.argument_sampler.sample",
                SampleOp::PixelFetch { .. } => "sampler.argument_sampler.pixel_fetch",
                SampleOp::Bicubic(_) => "sampler.argument_sampler.bicubic",
                SampleOp::Gather { .. } => "sampler.argument_sampler.gather",
                SampleOp::Lod { .. } => "sampler.argument_sampler.query_lod",
                SampleOp::QueryLevel { .. } => "sampler.argument_sampler.query_level",
            };
        }
        if let Source::Heap { .. } = self.source {
            return match &self.op {
                SampleOp::Sample { .. } => "sampler.heap.sample",
                SampleOp::PixelFetch { .. } => "sampler.heap.pixel_fetch",
                SampleOp::Bicubic(_) => "sampler.heap.bicubic",
                SampleOp::Gather { .. } => "sampler.heap.gather",
                SampleOp::Lod { .. } => "sampler.heap.query_lod",
                SampleOp::QueryLevel { .. } => "sampler.heap.query_level",
            };
        }
        match &self.op {
            SampleOp::Sample {
                lod:
                    Lod::Implicit {
                        reach: BiasReach::Within | BiasReach::Metal,
                        ..
                    },
                ..
            } => "sampler.sample_implicit",
            // The `Bias` operand within the limit, the scaled derivatives past it, the axis
            // gradients through an anisotropic word.
            SampleOp::Sample {
                lod:
                    Lod::Implicit {
                        reach: BiasReach::Either { .. },
                        ..
                    },
                ..
            } => "sampler.sample_implicit.bias_gradients",
            SampleOp::Sample {
                lod:
                    Lod::Implicit {
                        reach: BiasReach::Beyond | BiasReach::Axes,
                        ..
                    },
                ..
            } => "sampler.sample_bias_gradients",
            SampleOp::Sample {
                lod: Lod::Grad(..), ..
            } => "sampler.sample_grad",
            SampleOp::Sample {
                lod: Lod::CubeGrad(..),
                ..
            } => "sampler.sample_cube_grad",
            SampleOp::Sample {
                lod: Lod::Hoisted, ..
            } => "sampler.sample_hoisted_grad",
            SampleOp::Sample {
                lod: Lod::Base { .. },
                ..
            } => "sampler.sample_base_level",
            SampleOp::Sample { .. } => "sampler.sample",
            SampleOp::PixelFetch { .. } => "sampler.pixel_fetch",
            SampleOp::Bicubic(_) => "sampler.bicubic",
            SampleOp::Gather { .. } if compares => "sampler.gather_compare",
            SampleOp::Gather { .. } => "sampler.gather",
            SampleOp::Lod { .. } => "sampler.query_lod",
            SampleOp::QueryLevel { .. } => "sampler.query_level",
        }
    }

    /// The estimated words: the load, the coordinate, the operation, and the texel conversion, plus
    /// the pixel scaling, the reading of a dynamic `Offset` operand, a heap element's access chain,
    /// sampled image, and null select, the sparse arm, a bias past the limit's derivatives,
    /// `bias_gradients`, and (for a dynamic bias) the test, the selection, and the second sample, a
    /// `half` texel's call of its conversion routine or its exact rounding outside the bicubic
    /// filter, an sRGB texel's raw decode, a cube gradient's footprint, an argument buffer's pixel
    /// sampler's flagged fetch, the positions of an observed footprint, the lanes Metal reads as
    /// constants, a cube direction's broken ties, and a raw heap sample's decode.
    pub fn words(&self) -> u32 {
        // A swizzled slot's sampling: its swizzle, and a gather's of each component.
        match (self.swizzle, &self.op) {
            (false, _) => self.plain_words(),
            (true, SampleOp::Gather { .. }) => 4 * self.plain_words() + SWIZZLE_WORDS,
            (true, _) => self.plain_words() + SWIZZLE_WORDS,
        }
    }

    /// [`words`](Self::words) of the sampling without its slot's swizzle.
    fn plain_words(&self) -> u32 {
        let (base, offset) = match &self.op {
            SampleOp::Sample {
                lod: Lod::Implicit { reach, .. },
                offset,
                ..
            } => (
                match reach {
                    BiasReach::Metal => 64,
                    BiasReach::Within => 64 + METAL_BIAS_WORDS,
                    BiasReach::Beyond => 64 + METAL_BIAS_WORDS + BIAS_GRADIENT_WORDS,
                    BiasReach::Axes => 64 + METAL_BIAS_WORDS + AXIS_GRADIENT_WORDS,
                    BiasReach::Either { .. } => 64 + METAL_BIAS_WORDS + BIAS_GRADIENT_WORDS + 96,
                },
                offset.as_ref(),
            ),
            // The major axis and, per gradient, its two counted components, their length, and
            // the gradient on one tangent; Metal's own gradients are handed as they are. A
            // footprint by the word computes both forms, the word's bit, and their selects.
            SampleOp::Sample {
                lod: Lod::CubeGrad(_, _, CubeFootprint::Word),
                offset,
                ..
            } => (64 + 200 + 200 + 48, offset.as_ref()),
            SampleOp::Sample {
                lod: Lod::CubeGrad(_, _, form),
                offset,
                ..
            } if *form != CubeFootprint::Metal => (64 + 200, offset.as_ref()),
            SampleOp::Sample { offset, .. } => (64, offset.as_ref()),
            // The size query, the floor, the clamp, and the fetch.
            SampleOp::PixelFetch { offset } => (96, offset.as_ref()),
            SampleOp::Gather { offset, .. } => (56, offset.as_ref()),
            SampleOp::Lod { .. } => (24, None),
            // The image, its level count, the subtraction, two conversions, the scale, and the
            // minimum.
            SampleOp::QueryLevel { .. } => (40, None),
            SampleOp::Bicubic(b) => (b.words(), None),
        };
        // The bitcast, the `OpBitFieldSExtract`, and its two constants; an armed offset's
        // selector and arms (each swizzled gather's own, `words`).
        let wrap = match offset {
            Some(Offset::Dynamic(_, None)) => 24,
            Some(Offset::Dynamic(_, Some(a))) => 24 + 16 + a.arms() * OFFSET_ARM_WORDS,
            _ => 0,
        };
        // An argument buffer's sampler adds its entry's element, the access chain, and the load;
        // and its word's cubic test, the selection, and the status mark.
        let heap = match self.source {
            Source::Combined(_) => 0,
            Source::Heap { .. } => 32,
            Source::HeapSampler { .. } => 56 + 2 * ENTRY_FIELD_WORDS + CUBIC_REACH_WORDS,
        };
        let residency = if self.residency {
            crate::lower::texture::RESIDENCY_WORDS
        } else {
            0
        };
        // A `half` texel's call of the conversion routine or its exact rounding, or an sRGB texel's
        // raw decode; the bicubic routines round their `half` result themselves.
        let half = match (&self.op, self.texel) {
            (SampleOp::Bicubic(_), _) | (_, None) => 0,
            _ if self.raw => crate::lower::texture::SRGB_DECODE_WORDS,
            // A heap color texture's call reads the class its handle carries first.
            (_, Some(f)) if f.conv == crate::lower::texture::TexelConv::F16 => {
                let class = match self.source {
                    Source::Heap { .. } | Source::HeapSampler { .. } if !self.depth => {
                        crate::lower::texture::HEAP_CLASS_WORDS
                    }
                    _ => 0,
                };
                crate::lower::texture::half::CALL_WORDS.max(f.read_words()) + class
            }
            (_, Some(f)) => f.read_words(),
        };
        // An argument buffer's sampler also fetches the texel the rule names: the size query, the
        // floor, the clamp, the fetch (with its own dynamic offset), and the flags' test and
        // selection.
        let fetch = match (&self.pixel, &self.op) {
            (Pixel::Heap(_), SampleOp::Sample { dref: None, .. }) => 96 + 32 + wrap,
            _ => 0,
        };
        // An argument buffer's sampler reads its entry's word for each test of it: its pixel
        // coordinates and its exact fetch.
        let word = match self.pixel {
            Pixel::Heap(_) => (1 + u32::from(fetch != 0)) * ENTRY_FIELD_WORDS,
            _ => 0,
        };
        // The position of each axis of Metal's footprint of an observed sample or gather.
        let footprint = self
            .footprint
            .map_or(0, |fp| u32::from(fp.axes) * position::POSITION_WORDS);
        // The slot's parent tests in Metal's offset rule, on each plane axis of the bicubic
        // footprint that places the offset.
        let parent = match self.offset_base {
            true => u32::from(self.lanes) * position::PARENT_RULE_WORDS,
            false => 0,
        };
        // Metal's unmapped value: the lanes Metal reads as constants, and a named texel's
        // selection of zero (a gather's per tap).
        let unmapped = match self.unmapped {
            Unmapped::Device => 0,
            Unmapped::Constants => CONSTANT_LANES_WORDS,
            Unmapped::Named => CONSTANT_LANES_WORDS + 64,
        };
        base + wrap
            + heap
            + residency
            + half
            + fetch
            + word
            + footprint
            + parent
            + unmapped
            + if self.cube { face::TIE_WORDS } else { 0 }
            + if self.pixel == Pixel::None { 0 } else { 48 }
            + self.heap_raw.as_ref().map_or(0, heap_raw::HeapRaw::words)
    }
}

/// The estimated words of a swizzle of a sampled texel: the word's fields, their tests, and the
/// selects of each lane's source (`extended`'s).
const SWIZZLE_WORDS: u32 = extended::SWIZZLE_WORDS;

/// The estimated words of setting an access's lanes to the constants Metal reads
/// (`texture::constant_lanes`): per lane its extraction and two selections (the slot's tests are
/// specialization constants), the construction, and their constants, at most 8 words each.
pub const CONSTANT_LANES_WORDS: u32 = 4 * 3 * 8 + 16;

/// A runtime sampler slot the function samples with.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RuntimeSlot {
    pub slot: SlotRef,
    /// It samples with depth comparison.
    pub compare: bool,
    /// A coordinate-sensitive operation uses it (`SamplerCoordinates`): the function reads its
    /// word's pixel coordinates.
    pub coordinates: bool,
    /// An extended-range or heap raw texture's emulated filter reads it: the function reads its
    /// word's filter, coordinates, address modes, and border color.
    pub linear: bool,
    /// An extended-range sample reads its levels by its word (`extended::Level::Word`): the function
    /// reads its word's mip filter and the levels its clamp leaves.
    pub fetch_levels: bool,
    /// A cube's explicit gradients take their footprint by its word (`CubeFootprint::Word`): the
    /// function reads whether its word is anisotropic.
    pub anisotropy: bool,
    /// A sample's `half` texel converts by the bound format's class where the sampler reads one
    /// texel of one level (`texture::OneTexel::Word`), on a device that samples through Metal's
    /// own sampler (`VkProperty::MetalSampler`): the function reads its word's filter fields.
    pub texels: bool,
    /// A clamped LOD query outside a fragment function reads it ([`SampleOp::QueryLevel`]), or an
    /// extended-range sample whose linear mip filter blends two levels (`extended::Level::Word`):
    /// the function reads its `SpecKey::SamplerMinLevel` specialization.
    pub min_level: bool,
    /// An emulated anisotropic sample reads it ([`anisotropic`]): the function reads its word's
    /// anisotropy and mip filter and its `SpecKey::SamplerAnisotropy` specialization.
    pub emulated: bool,
}

impl RuntimeSlot {
    /// The fields of the slot's sampler word the function reads: the mask of its
    /// `SpecKey::SamplerState` specialization.
    pub fn fields(&self) -> u32 {
        let mut fields = 0;
        if self.coordinates {
            fields |= SamplerWord::PIXEL_COORDINATES;
        }
        if self.linear {
            fields |= SamplerWord::LINEAR_STATE;
        }
        if self.fetch_levels {
            fields |= SamplerWord::MIP_FILTER | SamplerWord::LEVEL_MIN | SamplerWord::LEVEL_MAX;
        }
        if self.anisotropy {
            fields |= SamplerWord::ANISOTROPIC;
        }
        if self.emulated {
            fields |= SamplerWord::ANISOTROPIC | SamplerWord::MIP_FILTER;
        }
        if self.texels {
            fields |= SamplerWord::LINEAR_FILTER;
        }
        fields
    }

    /// The slot's interface entry.
    pub fn interface(&self) -> SamplerSlot {
        SamplerSlot {
            compare: self.compare,
            coordinate_sensitive: self.coordinates,
        }
    }
}

/// The declaration of one combined binding, set by [`finish`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CombinedDecl {
    /// The texture's image as a sampled image of its own shape.
    pub image: ImageDecl,
    pub sampler: CombinedSampler,
    /// The binding's symbol: its `CombinedTextureSampler`, or its `SeparateTexture` where the
    /// stage takes its combined bindings apart.
    pub symbol: ResourceSlot,
}

/// The sampler family's part of the per-function plan (`select::FamilyPlans`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SamplerPlan {
    /// Every runtime sampler slot the function uses, by slot index.
    pub slots: BTreeMap<u32, RuntimeSlot>,
    /// The arms of the entry's armed offsets together ([`OffsetArms`]), at most
    /// [`OFFSET_ARMS_PER_FUNCTION`].
    pub offset_arms: u32,
    /// Every combined binding the sampling operations read.
    pub combined: BTreeSet<BindingKey>,
    /// Their declarations.
    pub decls: BTreeMap<BindingKey, CombinedDecl>,
    /// The stage takes its combined bindings apart (set by [`finish`]): each is a separate texture
    /// binding and the stage's separate sampler binding of its use, because one combined binding
    /// per pair needs more samplers than the stage may have.
    pub separate: bool,
    /// Every separate heap sampler use.
    pub heap: BTreeSet<HeapKey>,
    /// The binding each reads, set by [`finish`].
    pub heap_decls: BTreeMap<HeapKey, CombinedSampler>,
    /// Every use of an argument buffer's sampler.
    pub heap_samplers: BTreeSet<HeapSamplerKey>,
    /// The sampler array each reads, by its use, set by [`finish`].
    pub heap_sampler_decls: BTreeMap<HeapSamplerKey, SamplerUse>,
    /// The texture slots a bicubic sample reads, each with its `BicubicTexels` specialization.
    pub bicubic: BTreeSet<u32>,
    /// The bicubic routines the samples call, by dimension, an extended-range sample's among them.
    pub bicubic_routines: BTreeSet<bicubic::routines::Routine>,
    /// A sample relies on the device's sampler: the module requires `VkProperty::MetalSampler`
    /// on a device that samples through Metal's own sampler, and `VkProperty::DeviceSampling` on
    /// one that samples natively.
    pub metal_sampler: bool,
    /// A sample, gather or comparison goes to the device's sampler: on a device that samples
    /// natively the module requires `VkProperty::DeviceSampling` and records
    /// `PrecisionNote::DeviceSampling`; on one that samples through Metal's own it requires
    /// `VkProperty::MetalSampler`.
    pub device_sampling: bool,
    /// Every operation through an argument buffer's sampler tests its table entry's word for a
    /// cubic filter (`SamplerWord::CUBIC`) and marks the command buffer's `ExecutionStatus` where
    /// it holds one (`RuntimeFact::CubicArgumentSampler`): set where the stage can write the
    /// buffer.
    pub cubic_reach: bool,
    /// The sampling routines the samples call, each shape once (`routine::note`): the census counts
    /// each one's words once.
    pub routines: Vec<routine::Shape>,
}

/// The image facts emission reads for a combined binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CombinedImage {
    /// Plane size components (a cube's size has 2).
    pub plane: u8,
    pub arrayed: bool,
    pub texel: TexelClass,
}

/// The plane size components of an image of dimension `dim` (a cube's size has 2).
fn plane_of(dim: Dim) -> u8 {
    match dim {
        Dim::D1 | Dim::Buffer => 1,
        Dim::D2 | Dim::Cube | Dim::SubpassData => 2,
        Dim::D3 => 3,
    }
}

/// The variables a combined binding's sampling reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Binding {
    /// The combined image sampler.
    Combined(GlobalId),
    /// The separate texture and the stage's separate sampler of the binding's use.
    Separate { image: GlobalId, sampler: GlobalId },
}

/// The sampler family's module-scope objects.
#[derive(Clone, Debug, Default)]
pub struct Declared {
    /// The variables of each combined binding, with its image facts.
    pub combined: BTreeMap<BindingKey, (Binding, CombinedImage)>,
    /// The sampler variable of each heap sampler use.
    pub heap: BTreeMap<HeapKey, GlobalId>,
    /// The set-1 sampler array of each use of an argument buffer's sampler.
    pub heap_samplers: BTreeMap<HeapSamplerKey, GlobalId>,
    /// The `SamplerState` specialization constant (`u32`) of each runtime sampler slot whose
    /// word the function reads, masked by [`RuntimeSlot::fields`].
    pub words: BTreeMap<u32, SValue>,
    /// The `SamplerMinLevel` specialization constant (`u32`) of each runtime sampler slot a
    /// clamped LOD query outside a fragment function reads.
    pub min_levels: BTreeMap<u32, SValue>,
    /// The `SamplerAnisotropy` specialization constant (`u32`) of each runtime sampler slot an
    /// emulated anisotropic sample reads ([`anisotropic`]).
    pub degrees: BTreeMap<u32, SValue>,
    /// The `BicubicTexels` specialization constant (`u32`) of each texture slot a bicubic sample
    /// reads.
    pub bicubic: BTreeMap<u32, ConstId>,
    /// The 2D bicubic samples' routine, when the module has one.
    pub bicubic_plane: Option<bicubic::routines::Plane>,
    /// The 1D bicubic samples' routine, when the module has one.
    pub bicubic_line: Option<bicubic::routines::Line>,
    /// The in-place two-plane YCbCr linear samples' routines, when the module has one.
    pub linear_quad: Option<bicubic::routines::Quad>,
}

/// Complete the sampler family's plan: declare each combined binding (its image from the
/// texture slot's final shape and numeric type) and each heap sampler (its use from the heap
/// group's final numeric type), run the constexpr feature checks, choose the form of the combined
/// bindings against the stage's sampler limit, and contribute the descriptors and the
/// runtime-sampler facts.
pub fn finish(cx: &mut SelectCx<'_>) -> Result<(), Error> {
    let mut counts = DescriptorCounts::default();
    let mut decls = BTreeMap::new();
    for key in &cx.families.sampler.combined {
        let s = cx
            .families
            .texture
            .slots
            .get(&key.texture)
            .ok_or_else(|| invariant("a combined binding of an unplanned texture slot"))?;
        let shape = s
            .shape
            .ok_or_else(|| invariant("a sampled texture slot without a shape"))?;
        let texel = crate::lower::texture::image_class(cx.inputs, s);
        let sampler = CombinedSampler::new(key.sampler, key.compare, texel != TexelClass::Float);
        if let SamplerRef::Constexpr(words) = key.sampler {
            constexpr_desc(words, sampler.sampler_use(), cx.profile)?;
        }
        use crate::air::intrinsics::ImageDim as D;
        let dim = match shape.dim {
            D::D1 => Dim::D1,
            D::D2 => Dim::D2,
            D::D3 => Dim::D3,
            D::Cube => Dim::Cube,
            D::Buffer => return Err(invariant("a sampled texture buffer")),
        };
        let image = ImageDecl {
            dim,
            arrayed: shape.array,
            multisampled: false,
            depth: shape.depth,
            usage: ImageUsage::Sampled,
            format: crate::spv::ImageFormat::Unknown,
            texel,
            cube_as_array: false,
            class: StorageImageClass::Identity,
            level_views: None,
            extended: None,
            x_channel: false,
            in_place: false,
            planes_sampler: None,
        };
        let symbol = ResourceSlot {
            stage: cx.stage,
            kind: ResourceKind::CombinedTextureSampler(sampler),
            index: key.texture,
        };
        decls.insert(
            *key,
            CombinedDecl {
                image,
                sampler,
                symbol,
            },
        );
    }
    let mut heap_decls = BTreeMap::new();
    for key in &cx.families.sampler.heap {
        let h = cx
            .families
            .texture
            .heap
            .get(&key.group)
            .filter(|h| h.decl.is_some())
            .ok_or_else(|| invariant("a heap sampler of an undeclared heap group"))?;
        let texel = h.texel.unwrap_or(TexelClass::Float);
        let sampler = CombinedSampler::new(key.sampler, key.compare, texel != TexelClass::Float);
        // A runtime heap sampler is the sampler bound at its slot, which the binder writes for
        // this use.
        if let SamplerRef::Constexpr(words) = key.sampler {
            constexpr_desc(words, sampler.sampler_use(), cx.profile)?;
        }
        heap_decls.insert(*key, sampler);
    }
    // One heap sampler binding per distinct sampler use (`declare`).
    let heap_samplers: BTreeSet<CombinedSampler> = heap_decls.values().copied().collect();
    let count = |n: usize| u32::try_from(n).unwrap_or(u32::MAX);
    // Each combined binding is one sampled image and one sampler. Where they are more samplers than
    // the stage may have, the stage takes them apart: the same sampled image per binding, and one
    // separate sampler per distinct use, which counts as one sampler however many textures it
    // samples. The sampled images, and every other count, are the same in both forms, so this is
    // the one limit the separate form relieves; one it still exceeds is the stage's refusal
    // (`select`).
    let combined = count(decls.len()).saturating_add(count(heap_samplers.len()));
    let limit = cx.profile.fields().limits.max_per_stage_descriptor_samplers;
    let separate = combined > limit;
    let uses: BTreeSet<CombinedSampler> = decls.values().map(|d| d.sampler).collect();
    counts.samplers = if separate {
        count(uses.len()).saturating_add(count(heap_samplers.len()))
    } else {
        combined
    };
    counts.sampled_images = count(decls.len());
    if separate {
        // A separate texture's symbol holds its use above its texture slot. Canonicalization
        // refuses any texture slot past Metal's 128 entries (`AbiError::SlotPastTable`), so every
        // slot here fits.
        for (key, d) in decls.iter_mut() {
            d.symbol = ResourceSlot::separate_texture(cx.stage, key.texture, d.sampler)
                .ok_or_else(|| invariant("a separate texture's slot past Metal's table"))?;
        }
    }
    // An argument buffer's sampler reads the set-1 sampler array of its use: its image's heap array
    // needs the same descriptor-indexing features (runtime array, partially bound,
    // `shaderSampledImageArrayNonUniformIndexing`, which covers arrays of samplers, and
    // `descriptorBindingSampledImageUpdateAfterBind`, which covers `SAMPLER` bindings), which the
    // group's declaration already required. Set 1 is counted by link.
    let mut heap_sampler_decls = BTreeMap::new();
    for key in &cx.families.sampler.heap_samplers {
        let h = cx
            .families
            .texture
            .heap
            .get(&key.group)
            .filter(|h| h.decl.is_some())
            .ok_or_else(|| invariant("an argument sampler of an undeclared heap group"))?;
        let texel = h.texel.unwrap_or(TexelClass::Float);
        heap_sampler_decls.insert(
            *key,
            SamplerUse::new(key.compare, texel != TexelClass::Float),
        );
    }
    // The bound format's treatment under the bicubic filter.
    for index in &cx.families.sampler.bicubic {
        let s = cx
            .families
            .texture
            .slots
            .get(index)
            .ok_or_else(|| invariant("a bicubic sample of an unplanned texture slot"))?;
        cx.facts.insert(RuntimeFact::BicubicTexels { slot: s.slot });
    }
    for r in cx.families.sampler.slots.values() {
        cx.facts
            .insert(RuntimeFact::SamplerReduction { slot: r.slot });
        if r.coordinates {
            cx.facts
                .insert(RuntimeFact::SamplerCoordinates { slot: r.slot });
        }
    }
    cx.families.sampler.decls = decls;
    cx.families.sampler.separate = separate;
    cx.families.sampler.heap_decls = heap_decls;
    cx.families.sampler.heap_sampler_decls = heap_sampler_decls;
    cx.descriptors = cx.descriptors.plus(&counts);
    Ok(())
}

/// The set-1 heap features a function's argument-buffer samplers need besides the module's own
/// (the layout's binding flags): `descriptorBindingPartiallyBound` and
/// `descriptorBindingSampledImageUpdateAfterBind`. `compile` adds them to the artifact's
/// requirements.
pub fn heap_layout_features(plan: &Plan) -> Vec<crate::device::requirements::VkFeature> {
    use crate::device::requirements::VkFeature;
    if plan.families.sampler.heap_sampler_decls.is_empty() {
        return Vec::new();
    }
    vec![
        VkFeature::DescriptorBindingPartiallyBound,
        VkFeature::DescriptorBindingSampledImageUpdateAfterBind,
    ]
}

/// Declare the combined bindings, or their separate textures and samplers, and the runtime
/// samplers' coordinate constants.
pub fn declare(
    mb: &mut ModuleBuilder,
    plan: &Plan,
    _modes: &mut Vec<ExecMode>,
) -> BResult<Declared> {
    let mut d = Declared::default();
    // One separate sampler variable per distinct use.
    let mut separate: BTreeMap<CombinedSampler, GlobalId> = BTreeMap::new();
    for (key, c) in &plan.families.sampler.decls {
        let image = c.image.declare(mb)?;
        let binding = if plan.families.sampler.separate {
            let g = mb.resource(c.symbol, StorageClass::UniformConstant, image)?;
            let use_ = c.sampler;
            let sampler = match separate.get(&use_) {
                Some(s) => *s,
                None => {
                    let ty = mb.ty(SType::Sampler)?;
                    let s = mb.resource(
                        ResourceSlot::separate_sampler(plan.stage, use_),
                        StorageClass::UniformConstant,
                        ty,
                    )?;
                    separate.insert(use_, s);
                    s
                }
            };
            Binding::Separate { image: g, sampler }
        } else {
            let ty = mb.ty(SType::SampledImage { image })?;
            Binding::Combined(mb.resource(c.symbol, StorageClass::UniformConstant, ty)?)
        };
        d.combined.insert(
            *key,
            (
                binding,
                CombinedImage {
                    plane: plane_of(c.image.dim),
                    arrayed: c.image.arrayed,
                    texel: c.image.texel,
                },
            ),
        );
    }
    // One sampler variable per distinct heap sampler use.
    let mut samplers: BTreeMap<CombinedSampler, GlobalId> = BTreeMap::new();
    for (key, c) in &plan.families.sampler.heap_decls {
        let g = match samplers.get(c) {
            Some(g) => *g,
            None => {
                let ty = mb.ty(SType::Sampler)?;
                let g = mb.resource(
                    ResourceSlot::heap_sampler(plan.stage, *c),
                    StorageClass::UniformConstant,
                    ty,
                )?;
                samplers.insert(*c, g);
                g
            }
        };
        d.heap.insert(*key, g);
    }
    // One sampler array per distinct use of an argument buffer's sampler.
    let mut arrays: BTreeMap<SamplerUse, GlobalId> = BTreeMap::new();
    for (key, use_) in &plan.families.sampler.heap_sampler_decls {
        let g = match arrays.get(use_) {
            Some(g) => *g,
            None => {
                let sampler = mb.ty(SType::Sampler)?;
                let array = mb.ty(SType::RuntimeArray {
                    elem: sampler,
                    stride: None,
                })?;
                let g = mb.resource(
                    ResourceSlot {
                        stage: plan.stage,
                        kind: ResourceKind::Heap(HandleClass::sampler(*use_)),
                        index: 0,
                    },
                    StorageClass::UniformConstant,
                    array,
                )?;
                arrays.insert(*use_, g);
                g
            }
        };
        d.heap_samplers.insert(*key, g);
    }
    for (n, r) in &plan.families.sampler.slots {
        let fields = r.fields();
        if fields != 0 {
            let u = mb.ty(SType::Int {
                width: 32,
                signed: false,
            })?;
            let key = SpecKey::SamplerState {
                slot: r.slot,
                fields,
            };
            let c = mb.spec(key, u, ConstBits(0))?;
            d.words.insert(*n, mb.const_value(c)?);
        }
        if r.min_level {
            let u = mb.ty(SType::Int {
                width: 32,
                signed: false,
            })?;
            let c = mb.spec(SpecKey::SamplerMinLevel(r.slot), u, ConstBits(0))?;
            d.min_levels.insert(*n, mb.const_value(c)?);
        }
        if r.emulated {
            let u = mb.ty(SType::Int {
                width: 32,
                signed: false,
            })?;
            let c = mb.spec(SpecKey::SamplerAnisotropy(r.slot), u, ConstBits(0))?;
            d.degrees.insert(*n, mb.const_value(c)?);
        }
    }
    for index in &plan.families.sampler.bicubic {
        let Some(s) = plan.families.texture.slots.get(index) else {
            continue;
        };
        let u = mb.ty(SType::Int {
            width: 32,
            signed: false,
        })?;
        let neutral = crate::metal::bicubic::BicubicTexels::NEUTRAL.0;
        let c = mb.spec(
            SpecKey::BicubicTexels(s.slot),
            u,
            ConstBits(u64::from(neutral)),
        )?;
        d.bicubic.insert(*index, c);
    }
    for r in &plan.families.sampler.bicubic_routines {
        match r {
            bicubic::routines::Routine::Plane => {
                d.bicubic_plane = Some(bicubic::routines::define_plane(mb)?);
            }
            bicubic::routines::Routine::Line => {
                d.bicubic_line = Some(bicubic::routines::define_line(mb)?);
            }
            bicubic::routines::Routine::Quad => {
                d.linear_quad = Some(bicubic::routines::define_quad(mb)?);
            }
        }
    }
    Ok(d)
}

/// The words of the bicubic routines `plan` defines, by census name: each module's once, and of
/// each sampling routine its samples call.
pub fn routine_words(plan: &SamplerPlan) -> impl Iterator<Item = (&'static str, u64)> + '_ {
    plan.bicubic_routines
        .iter()
        .map(|r| (r.census_name(), u64::from(r.words())))
        // And each sampling routine's, once.
        .chain(
            plan.routines
                .iter()
                .map(|r| (r.census_name(), u64::from(r.words()))),
        )
}

/// Plan a used sampler parameter: a runtime sampler slot, or an array of `N` samplers in slots
/// `n .. n + N`. Its uses decide each slot's facts.
pub fn select_param(cx: &mut SelectCx<'_>, b: &ParamBinding) -> Result<(), Error> {
    let ParamRole::Sampler { location } = &b.role else {
        return Err(refuse());
    };
    let (AbiInt::Const(first), AbiInt::Const(count)) = (location.index, location.count) else {
        return Err(refuse());
    };
    for k in 0..count {
        let index = first.checked_add(k).ok_or_else(refuse)?;
        let slot = select::sampler_slot(cx.stage, index);
        cx.families
            .sampler
            .slots
            .entry(index)
            .or_insert(RuntimeSlot {
                slot,
                compare: false,
                coordinates: false,
                linear: false,
                fetch_levels: false,
                anisotropy: false,
                texels: false,
                min_level: false,
                emulated: false,
            });
    }
    Ok(())
}

/// The value of a sampler parameter: its handle index, or offset 0 into a sampler array
/// (`memory::handles`). Sampling operations name bindings; this value feeds only handle selects,
/// phis, and array element loads.
pub fn param_value(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    b: &ParamBinding,
) -> BResult<SValue> {
    let v = match crate::memory::handles::binding_slots(b) {
        Some((_, first, 1))
            if crate::memory::provenance::handle_array_of(cx.entry, b).is_none() =>
        {
            crate::memory::handles::Candidate::Sampler(first).handle()
        }
        Some(_) => 0,
        None => return Err(cx.fail(format!("parameter {} is no sampler slot", b.param))),
    };
    let c = fb.u32_const(v)?;
    fb.module().const_value(c)
}

/// Select the strategy of instruction `inst` of this family.
pub fn select(_cx: &mut SelectCx<'_>, _at: InstId, inst: &Inst) -> Result<SamplerStrategy, Error> {
    match &inst.kind {
        InstKind::Call(c)
            if matches!(
                c.callee,
                Callee::Air {
                    op: AirIntrinsic::Sampler(SamplerOp::GetReadSampler),
                    ..
                }
            ) =>
        {
            Ok(SamplerStrategy::ReadSampler)
        }
        _ => Err(refuse()),
    }
}

/// Emit strategy `s` of instruction `at` into the open block.
pub fn emit(
    _cx: &mut EmitCx<'_>,
    _fb: &mut FunctionBuilder<'_>,
    _at: InstId,
    s: &SamplerStrategy,
) -> BResult<()> {
    match s {
        SamplerStrategy::ReadSampler => Ok(()),
    }
}
