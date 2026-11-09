//! The levels a draw's slot reads of a texture a color attachment of its render pass instance
//! renders (`texture/attached_reads.rs`), counted (`WorkCounts::attached_read_slots` and kin).
//!
//! - **What is counted.** Each slot of a draw that takes every step and reads through copies
//!   (`render.rs`'s `draw_resident`) whose sampled binding (a sampled image or combined image
//!   sampler write, not a storage one) presents a texture that an attachment of the instance
//!   renders (`Texture::attached_in`): one count per slot of each such draw, by the first of
//!   `AttachedLevels` that holds. It weighs the texture's level count, the lowest attached level
//!   counted from the texture's own first level (`Texture::lowest_attached_level`), and the
//!   slot's reads (`StageModule::level_reads` of the module of the slot's stage).
//! - **The samplers bound at the draw.** A sample's levels follow the sampler the draw binds with
//!   the texture (`SampledLevels::level`). For a runtime sampler that is the descriptor the bind
//!   makes of the sampler bound at its slot with its bound LOD clamps (`binding::bound_sampler`,
//!   then `bind::sampler_desc` for the write's use and view format), whose address-mode map
//!   (`address_map.rs`) changes no level; for a constexpr sampler, the immutable sampler of the
//!   layout binding the write fills. Where several writes pair the slot with one sampler, the
//!   highest level among them counts. A sampler that no combined write pairs with the slot (a
//!   separate sampler), an unbound one, or one whose descriptor the bind refuses, counts as
//!   reaching the attached level.
//! - **Nothing else changes.** The counts read the draw's state; the draw binds and copies as
//!   before.

use super::RenderLowering;
use crate::argument_table::ArgumentTable;
use crate::binding;
use crate::device::Device;
use crate::work_counts::AttachedLevels;
use metal2vulkan::bind::{BindingPlan, DescriptorWrite, ImageSource, SamplerSource, WriteSource};
use metal2vulkan::metal::sampler::SamplerUse;
use metal2vulkan::pipeline::{
    BindingSource, RenderPipeline, SampledLevels, SamplerLevels, SamplerRef, SlotRef, Stage,
};
use metal2vulkan::vk;

impl RenderLowering {
    /// Count each slot of a draw of `plan` and pipeline `compiled` that samples a texture a color
    /// attachment of the pass renders, by how the levels its reads reach meet the attached one
    /// (the module docs).
    pub(super) fn count_attached_levels(
        &self,
        device: &Device,
        plan: &BindingPlan,
        compiled: &RenderPipeline,
    ) {
        let counts = self.core.counts();
        for (at, write) in plan.descriptors.iter().enumerate() {
            let Some(slot) = sampled_slot(write) else {
                continue;
            };
            // Each slot once, at its first sampled write.
            if (plan.descriptors[..at].iter()).any(|w| sampled_slot(w) == Some(slot)) {
                continue;
            }
            let texture = self
                .slot_table(slot.stage)
                .and_then(|t| t.textures.get(&slot.index));
            let Some((attached, level_count)) =
                texture.and_then(|texture| texture.lowest_attached_level(&self.layout))
            else {
                continue;
            };
            let reads = (compiled.stages.iter())
                .find(|m| Some(m.stage) == shader_stage(slot.stage))
                .and_then(|m| m.level_reads.iter().find(|r| r.slot == slot.index));
            let levels = match reads {
                _ if level_count == 1 => AttachedLevels::OneLevel,
                _ if attached == 0 => AttachedLevels::FirstLevel,
                None => AttachedLevels::AnyLevel,
                Some(reads) if reads.any => AttachedLevels::AnyLevel,
                Some(reads) => {
                    let highest = reads.highest_level(|sampled| {
                        self.sampled_level(device, (plan, compiled), slot, sampled)
                    });
                    match highest {
                        Some(highest) if highest < attached => AttachedLevels::Below,
                        _ => AttachedLevels::Reaching,
                    }
                }
            };
            counts.attached_read_slot(levels);
        }
    }

    /// The highest level, counted from the view's first, that `sampled`, samples of the texture at
    /// `slot`, reach through the samplers a draw of `plan` and `compiled` binds with it (the
    /// module docs): the highest among the writes that pair them; `None` where none does, or
    /// where one may reach any level.
    fn sampled_level(
        &self,
        device: &Device,
        (plan, compiled): (&BindingPlan, &RenderPipeline),
        slot: SlotRef,
        sampled: &SampledLevels,
    ) -> Option<u32> {
        let mut highest = None;
        for write in &plan.descriptors {
            let WriteSource::CombinedImageSampler {
                image: ImageSource::Texture { slot: of, view },
                sampler,
            } = &write.source
            else {
                continue;
            };
            if *of != slot {
                continue;
            }
            let desc = match (*sampler, sampled.sampler) {
                (
                    SamplerSource::Slot {
                        slot: bound,
                        compare,
                        integer,
                        opaque_black,
                    },
                    SamplerRef::Slot(index),
                ) if bound.index == index => {
                    let (state, clamps) = self.slot_table(bound.stage)?.samplers.get(&index)?;
                    let descriptor = binding::bound_sampler(state, *clamps);
                    let use_ = SamplerUse::new(compare, integer);
                    let format = Some(view.format);
                    let compile = device.compile_device();
                    metal2vulkan::bind::sampler_desc(
                        compile,
                        &descriptor,
                        use_,
                        opaque_black,
                        format,
                    )
                    .ok()?
                }
                (SamplerSource::Immutable, SamplerRef::Constexpr(_)) => {
                    match immutable_sampler(compiled, write) {
                        Some((paired, desc)) if paired == sampled.sampler => desc,
                        _ => continue,
                    }
                }
                _ => continue,
            };
            let finite = SamplerLevels::finite_texels(view.format);
            let level = sampled.level(SamplerLevels::of(&desc, finite))?;
            highest = highest.max(Some(level));
        }
        highest
    }

    /// The encoder's table of `stage`'s slots.
    fn slot_table(&self, stage: Stage) -> Option<&ArgumentTable> {
        match stage {
            Stage::Vertex => Some(&self.vertex),
            Stage::Fragment => Some(&self.fragment),
            Stage::Object => Some(&self.object),
            Stage::Mesh => Some(&self.mesh),
            _ => None,
        }
    }
}

/// The texture slot a sampled write of a plan binds: a sampled image's or a combined image
/// sampler's, not a storage image's.
fn sampled_slot(write: &DescriptorWrite) -> Option<SlotRef> {
    match &write.source {
        WriteSource::Texture { slot, .. } if write.ty != vk::DescriptorType::STORAGE_IMAGE => {
            Some(*slot)
        }
        WriteSource::CombinedImageSampler {
            image: ImageSource::Texture { slot, .. },
            ..
        } => Some(*slot),
        _ => None,
    }
}

/// The shader stage of the module that reads `stage`'s slots.
fn shader_stage(stage: Stage) -> Option<vk::ShaderStage> {
    match stage {
        Stage::Vertex => Some(vk::ShaderStage::VERTEX),
        Stage::Fragment => Some(vk::ShaderStage::FRAGMENT),
        Stage::Object => Some(vk::ShaderStage::TASK_EXT),
        Stage::Mesh => Some(vk::ShaderStage::MESH_EXT),
        _ => None,
    }
}

/// The constexpr sampler of the combined image sampler binding of `compiled`'s layout that
/// `write` fills, and the binding's immutable sampler.
fn immutable_sampler(
    compiled: &RenderPipeline,
    write: &DescriptorWrite,
) -> Option<(SamplerRef, vk::SamplerDesc)> {
    let set = (compiled.layout.sets.iter()).find(|set| set.set == write.set)?;
    let binding = (set.bindings.iter()).find(|b| b.binding == write.binding)?;
    match binding.source {
        BindingSource::CombinedTextureSampler { sampler, .. } => {
            Some((sampler.sampler, binding.immutable_sampler?))
        }
        _ => None,
    }
}
