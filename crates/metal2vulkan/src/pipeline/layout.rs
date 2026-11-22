//! `PipelineLayout`: the complete, dense, collision-free layout of a pipeline.
//!
//! Link assigns binding numbers; the layout lists exactly the statically used descriptors of every
//! stage, so no hole sits under a used binding and no Metal argument table collides with another.
//! Set 0 holds every per-slot and synthetic binding; set 1 exists only for the handle heap.

use crate::pipeline::sym::{
    AccelerationStructureLevel, CombinedSampler, HandleClass, SlotRef, Stage, SyntheticBinding,
};
use crate::vk;

/// A pipeline layout as plain data.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PipelineLayout {
    /// Set 0 (always) and set 1 (only for the handle heap), in set order.
    pub sets: Vec<DescriptorSetLayout>,
    /// One range per stage that has push fields, over the bytes its fields take.
    pub push_constants: Vec<PushConstantRange>,
}

/// One descriptor set of a [`PipelineLayout`].
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DescriptorSetLayout {
    pub set: u32,
    /// Dense bindings `0..n`, in binding order.
    pub bindings: Vec<DescriptorBinding>,
    /// `UPDATE_AFTER_BIND_POOL` for the heap set; empty for set 0.
    pub flags: vk::DescriptorSetLayoutCreateFlags,
}

/// One binding of a [`DescriptorSetLayout`].
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DescriptorBinding {
    pub binding: u32,
    pub ty: vk::DescriptorType,
    /// Array size; for a heap array, the profile's update-after-bind maximum.
    pub count: u32,
    /// Exactly the stages that statically use the binding.
    pub stages: vk::ShaderStageFlags,
    /// What feeds the binding.
    pub source: BindingSource,
    /// A constexpr sampler baked into the layout.
    pub immutable_sampler: Option<vk::SamplerDesc>,
    /// Binding flags (`PARTIALLY_BOUND`, `UPDATE_AFTER_BIND`, `VARIABLE_DESCRIPTOR_COUNT`).
    pub flags: vk::DescriptorBindingFlags,
    /// The most variables one stage's entry point uses at the binding (those its `OpEntryPoint`
    /// lists): a function that reaches a buffer through several types uses a variable of each
    /// type at the buffer's binding (each `Aliased` where it is writable). 0 where only a stage
    /// the pipeline adds at its creation uses it (a tessellation pipeline's control stage), whose
    /// module the layout does not see, or no stage uses it. A
    /// translator from SPIR-V to another language may handle a binding of several variables
    /// apart: MoltenVK 1.4.1's SPIRV-Cross writes Metal that does not compile for a dynamic buffer
    /// binding of more than one.
    #[cfg_attr(feature = "serde", serde(default))]
    pub variables: u32,
    /// The constexpr samplers baked into a sampler array's binding
    /// (`BindingSource::SamplerArray`), one per element (`count` of them); empty for every other
    /// binding.
    #[cfg_attr(feature = "serde", serde(default))]
    pub immutable_samplers: Vec<vk::SamplerDesc>,
}

/// What feeds a [`DescriptorBinding`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BindingSource {
    /// A Metal `[[buffer(n)]]` slot.
    Buffer(SlotRef),
    /// A Metal `[[texture(n)]]` slot (image or texel buffer).
    Texture(SlotRef),
    /// A Metal `[[sampler(n)]]` slot.
    Sampler(SlotRef),
    /// A texture slot and a sampler fed through one combined descriptor. A constexpr sampler is the
    /// binding's `immutable_sampler`; a runtime one is the `[[sampler(n)]]` slot of the texture's
    /// stage, made for `sampler.sampler_use()`.
    CombinedTextureSampler {
        texture: SlotRef,
        sampler: CombinedSampler,
    },
    /// The planes of the two-plane YCbCr texture bound at a Metal `[[texture(n)]]` slot, held as
    /// an image the device's own sampler Y′CbCr conversion samples
    /// (`ResourceKind::YCbCrPlanes`): one combined descriptor, its image the planes' image viewed
    /// with the conversion, its sampler the binding's `immutable_sampler`
    /// (`vk::SamplerDesc::ycbcr`).
    YCbCrPlanes(SlotRef),
    /// A binding the translator adds for one stage (argument table, root or handle table,
    /// constant blob).
    Synthetic {
        stage: Stage,
        binding: SyntheticBinding,
    },
    /// One runtime array of the set-1 heap.
    Heap(HandleClass),
    /// An array of the identity domains' set 1: binding 0 the sampled images, binding 1 the
    /// storage images, element `i` the texture whose resource ID holds index `i`
    /// (`docs/RESOURCE-ID.md`). The consumer writes an element as it makes a texture, for every
    /// pipeline at once.
    IdentityArray,
    /// A separate sampler of `stage` that samples set-1 heap images. A constexpr sampler is the
    /// binding's `immutable_sampler`, which takes no descriptor write.
    HeapSampler {
        stage: Stage,
        sampler: CombinedSampler,
    },
    /// The framebuffer-fetch input of this color index.
    InputAttachment(u32),
    /// The acceleration structure bound at a Metal `[[buffer(n)]]` slot, bound as a top-level
    /// structure: a primitive-level one through a one-instance top-level structure with the
    /// identity transform, mask `0xFF`, and instance index and user id 0.
    AccelerationStructure {
        slot: SlotRef,
        level: AccelerationStructureLevel,
    },
    /// A tile function's color attachment of this color index, bound as a storage image.
    TileAttachment(u32),
    /// The tile map of the sparse texture bound at a Metal `[[texture(n)]]` slot, a read-only
    /// storage buffer (`pipeline::tile_map`, `bind::SparseTileMap`).
    TileMap(SlotRef),
    /// The image half of a texture slot and a sampler a stage samples through separate
    /// descriptors: the texture as a sampled image of its own type (a cube is a cube view), with
    /// the comparison of `sampler`, which the stage's [`BindingSource::SeparateSampler`] of
    /// `sampler` samples. One binding per (texture, sampler, use).
    SeparateTexture {
        texture: SlotRef,
        sampler: CombinedSampler,
    },
    /// The sampler half: one binding per (sampler, use) of `stage`. A constexpr sampler is the
    /// binding's `immutable_sampler`, which takes no descriptor write; a runtime one is the
    /// `[[sampler(n)]]` slot of `stage`, made for `sampler.sampler_use()`.
    SeparateSampler {
        stage: Stage,
        sampler: CombinedSampler,
    },
    /// A constexpr sampler array of `stage` (`ResourceKind::SamplerArray`) that samples set-1
    /// heap images, made for the use with comparison or without (`compare`) of an integer texture
    /// or not (`integer`): its samplers are the binding's `immutable_samplers`, which take no
    /// descriptor write.
    SamplerArray {
        stage: Stage,
        compare: bool,
        integer: bool,
    },
}

/// One push-constant range of a [`PipelineLayout`]: a stage's fixed window (vertex `[0, 64)`,
/// fragment `[64, 128)`, compute `[0, 128)`), trimmed to the fields the stage uses.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PushConstantRange {
    pub stages: vk::ShaderStageFlags,
    pub offset: u32,
    pub size: u32,
}
