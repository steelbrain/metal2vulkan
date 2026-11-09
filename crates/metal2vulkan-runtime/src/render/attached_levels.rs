//! The levels a draw's slot reads of a texture a color attachment of its render pass instance
//! renders (`texture/attached_reads.rs`): those before the attached one are read in place, the
//! rest through the texture's copy, and each slot is counted (`WorkCounts::attached_read_slots`
//! and kin).
//!
//! - **The slots.** Each slot of a draw that reads through copies (`render.rs`'s `draw_resident`)
//!   whose sampled binding (a sampled image or combined image sampler write, not a storage one)
//!   presents a texture that an attachment of the instance renders (`Texture::attached_in`), by
//!   the first of `AttachedLevels` that holds. It weighs the texture's level count, the lowest
//!   attached level counted from the texture's own first level
//!   (`Texture::lowest_attached_level`), and the slot's reads (`StageModule::level_reads` of the
//!   module of the slot's stage).
//! - **The samplers bound at the draw.** A sample's levels follow the sampler the draw binds with
//!   the texture (`SampledLevels::level`). For a runtime sampler that is the descriptor the bind
//!   makes of the sampler bound at its slot with its bound LOD clamps (`binding::bound_sampler`,
//!   then `bind::sampler_desc` for the write's use and view format), whose address-mode map
//!   (`address_map.rs`) changes no level; for a constexpr sampler, the immutable sampler of the
//!   layout binding the write fills. Where several writes pair the slot with one sampler, the
//!   highest level among them counts. A sampler that no combined write pairs with the slot (a
//!   separate sampler), an unbound one, or one whose descriptor the bind refuses, counts as
//!   reaching the attached level.
//! - **Narrowed.** A slot every read of which reaches only levels before the lowest attached one
//!   `k` binds its texture's own image through a view of its levels `0..k` (the same first level,
//!   so an explicit level of detail selects the same level as through the whole view), and names
//!   those levels in its uses, in place of the copy (`binding::NarrowedSlots`). The copy holds
//!   them as they were before the instance; nothing the instance does writes them where none of
//!   these meets them, so the image holds the same: the texture is an image of its own that the
//!   bind can read in place (`Texture::reads_levels_in_place`: not sparse, no decoded twin, the
//!   sampled usage), no attachment of the instance (a resolve's, a depth or stencil one's) is at
//!   those levels (`render_target::shares_attachment`), no earlier draw wrote them since the
//!   instance's last barrier (`UnorderedWrites::written`; a draw after a barrier records into a
//!   new instance, `render.rs`), and no write among the draw's own uses does (else every slot of
//!   the draw is withdrawn and takes its copy). Vulkan reads one subresource of an image while
//!   another is an attachment: the rule against it is of a subresource "simultaneously being used
//!   as an attachment" (the Render Pass chapter), and a view of other levels is no such one.
//! - **Counted** at each draw that takes every step; a draw again or rebound of one
//!   (`render/draw_again.rs`) binds as it did and counts nothing here. A draw rebound works its
//!   narrowed slots out again and is rebound only where they are its kept draw's.

use super::RenderLowering;
use crate::argument_table::ArgumentTable;
use crate::binding::{self, NarrowedSlots};
use crate::device::Device;
use crate::render_target;
use crate::resource::ResourceKey;
use crate::texture::Texture;
use crate::tracker::{Access, Use};
use crate::work_counts::AttachedLevels;
use metal2vulkan::bind::{BindingPlan, DescriptorWrite, ImageSource, SamplerSource, WriteSource};
use metal2vulkan::metal::sampler::SamplerUse;
use metal2vulkan::pipeline::{
    BindingSource, RenderPipeline, SampledLevels, SamplerLevels, SamplerRef, SlotRef, Stage,
};
use metal2vulkan::vk;

/// The stages' tables a draw binds.
type Stages<'a> = [(Stage, &'a ArgumentTable)];

impl RenderLowering {
    /// The slots of a draw of `plan` and pipeline `compiled`, over the tables `stages`, that
    /// sample a texture a color attachment of the pass renders, those narrowed listed (the
    /// module docs). Each of the others is counted where `count`; the caller counts those
    /// narrowed once the draw's uses are known ([`count_narrowed`](Self::count_narrowed)).
    pub(super) fn attached_slots(
        &self,
        device: &Device,
        (plan, compiled): (&BindingPlan, &RenderPipeline),
        stages: &Stages<'_>,
        count: bool,
    ) -> NarrowedSlots {
        let counts = self.core.counts();
        let mut narrowed = NarrowedSlots::default();
        for (at, write) in plan.descriptors.iter().enumerate() {
            let Some(slot) = sampled_slot(write) else {
                continue;
            };
            // Each slot once, at its first sampled write.
            if (plan.descriptors[..at].iter()).any(|w| sampled_slot(w) == Some(slot)) {
                continue;
            }
            let texture = stage_table(stages, slot.stage).and_then(|t| t.textures.get(&slot.index));
            let Some(texture) = texture else {
                continue;
            };
            let Some((attached, level_count)) = texture.lowest_attached_level(&self.layout) else {
                continue;
            };
            let reads = (compiled.stages.iter())
                .find(|m| Some(m.stage) == shader_stage(slot.stage))
                .and_then(|m| m.level_reads.iter().find(|r| r.slot == slot.index));
            let levels = match reads {
                _ if level_count == 1 => Err(AttachedLevels::OneLevel),
                _ if attached == 0 => Err(AttachedLevels::FirstLevel),
                None => Err(AttachedLevels::AnyLevel),
                Some(reads) if reads.any => Err(AttachedLevels::AnyLevel),
                Some(reads) => {
                    let highest = reads.highest_level(|sampled| {
                        self.sampled_level(device, (plan, compiled), stages, slot, sampled)
                    });
                    match highest {
                        Some(highest) if highest < attached => self.levels_read(texture, attached),
                        _ => Err(AttachedLevels::Reaching),
                    }
                }
            };
            match levels {
                Ok(used) => narrowed.narrow(slot, attached, used),
                Err(levels) if count => counts.attached_read_slot(levels),
                Err(_) => {}
            }
        }
        narrowed
    }

    /// The use of `texture`'s levels before `attached` a slot narrowed to them makes, where
    /// nothing the instance does writes them (the module docs); else why the slot keeps its copy.
    fn levels_read(
        &self,
        texture: &Texture,
        attached: u32,
    ) -> Result<Use<ResourceKey>, AttachedLevels> {
        if !texture.reads_levels_in_place() {
            return Err(AttachedLevels::Apart);
        }
        let read = Access::new(
            ash::vk::PipelineStageFlags2::ALL_GRAPHICS,
            ash::vk::AccessFlags2::SHADER_SAMPLED_READ,
        );
        let used = texture.levels_use(attached, (read, Access::NONE));
        let uses = std::slice::from_ref(&used);
        let attachment = render_target::shares_attachment(&self.layout, uses).unwrap_or(true);
        if attachment || self.unordered.written(uses) {
            return Err(AttachedLevels::Written);
        }
        Ok(used)
    }

    /// Count `narrowed` slots of a draw that takes every step, each narrowed, or written where a
    /// write of the draw's own met one's levels (`written`) and they took their copies.
    pub(super) fn count_narrowed(&self, narrowed: usize, written: bool) {
        let levels = match written {
            true => AttachedLevels::Written,
            false => AttachedLevels::Narrowed,
        };
        let counts = self.core.counts();
        for _ in 0..narrowed {
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
        stages: &Stages<'_>,
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
                    let table = stage_table(stages, bound.stage)?;
                    let (state, clamps) = table.samplers.get(&index)?;
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
}

/// The table of `stage`'s slots among `stages`.
fn stage_table<'a>(stages: &Stages<'a>, stage: Stage) -> Option<&'a ArgumentTable> {
    (stages.iter())
        .find(|(of, _)| *of == stage)
        .map(|&(_, table)| table)
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
