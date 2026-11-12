//! The levels a draw's slot reads of a texture a color attachment of its render pass instance
//! renders (`texture/attached_reads.rs`): those before the lowest attached one, or past the
//! highest, are read in place, the rest through the texture's copy, and each slot is counted
//! (`WorkCounts::attached_read_slots` and kin).
//!
//! - **The slots.** Each slot of a draw that reads through copies (`render.rs`'s `draw_resident`)
//!   whose sampled binding (a sampled image or combined image sampler write, not a storage one)
//!   presents a texture that an attachment of the instance renders (`Texture::attached_in`), by
//!   the first of `AttachedLevels` that holds. It weighs the texture's level count, the lowest
//!   and the highest attached level counted from the texture's own first level
//!   (`Texture::attached_span`), and the slot's reads (`StageModule::level_reads` of the module of
//!   the slot's stage): the highest level they reach (`LevelReads::highest_level`) and the lowest
//!   they may (`LevelReads::lowest_level`).
//! - **The samplers bound at the draw.** A sample's levels follow the sampler the draw binds with
//!   the texture (`SampledLevels::level`, `SampledLevels::lowest`). For a runtime sampler that is
//!   the descriptor the bind makes of the sampler bound at its slot with its bound LOD clamps
//!   (`binding::bound_sampler`, then `bind::sampler_desc` for the write's use and view format),
//!   whose address-mode map (`address_map.rs`) changes no level; for a constexpr sampler, the
//!   immutable sampler of the layout binding the write fills. Where several writes pair the slot
//!   with one sampler, the highest level among them counts, and the lowest. A sampler that no
//!   combined write pairs with the slot (a separate sampler), an unbound one, or one whose
//!   descriptor the bind refuses, counts as reaching every level.
//! - **Narrowed before.** A slot every read of which reaches only levels before the lowest
//!   attached one `k` binds its texture's own image through a view of its levels `0..k` (the same
//!   first level, so an explicit level of detail selects the same level as through the whole
//!   view), and names those levels in its uses, in place of the copy (`binding::NarrowedSlots`).
//! - **Narrowed past.** A slot every read of which reaches only levels past the highest attached
//!   one binds its texture's own image through a view of its levels from its first up to the
//!   highest it reads (all of them where its sampler's `maxLod` reaches past the last), the same
//!   first level again, and names in its uses only the levels from the lowest it may read. The
//!   view holds the attached levels too, unread: Vulkan's rules against reading an attachment are
//!   of accesses ("must not be accessed in any way other than as an attachment",
//!   `VUID-vkCmdDraw-None-10795`; the Render Pass chapter's of a subresource "simultaneously being
//!   used as an attachment"), and the layout the descriptor names must match the subresources
//!   accessible from the view (`VUID-vkCmdDraw-imageLayout-00344`): `GENERAL`, where every
//!   attachment rests (`render_target.rs`). So such a slot keeps its copy in an instance that holds
//!   one of the texture's attached colors in the feedback loop layout, and a later draw that would
//!   read one through a snapshot records into another instance (`Instance::general_colors`,
//!   `render/snapshot.rs`).
//! - **Either way.** The copy holds the levels read as they were before the instance; nothing the
//!   instance does writes them where none of these meets them, so the image holds the same: the
//!   texture is an image of its own that the bind can read in place
//!   (`Texture::reads_levels_in_place`: not sparse, no decoded twin, the sampled usage), no
//!   attachment of the instance (a resolve's, a depth or stencil one's) is at those levels
//!   (`render_target::shares_attachment`), no earlier draw wrote them since the instance's last
//!   barrier (`UnorderedWrites::written`; a draw after a barrier records into a new instance,
//!   `render.rs`), and no write among the draw's own uses does (else every slot of the draw is
//!   withdrawn and takes its copy). Vulkan reads one subresource of an image while another is an
//!   attachment: the rule against it is of a subresource "simultaneously being used as an
//!   attachment" (the Render Pass chapter), and other levels are no such one.
//! - **Counted** at each draw that takes every step; a draw again or rebound of one
//!   (`render/draw_again.rs`) binds as it did and counts nothing here. A draw rebound works its
//!   narrowed slots out again and is rebound only where they are its kept draw's.

use super::RenderLowering;
use crate::argument_table::ArgumentTable;
use crate::binding::{self, NarrowedLevels, NarrowedSlots};
use crate::device::Device;
use crate::render_target;
use crate::render_target::Feedback;
use crate::resource::ResourceKey;
use crate::texture::{AttachedSpan, Texture};
use crate::tracker::{Access, Use};
use crate::work_counts::AttachedLevels;
use metal2vulkan::bind::{BindingPlan, DescriptorWrite, ImageSource, SamplerSource, WriteSource};
use metal2vulkan::metal::sampler::SamplerUse;
use metal2vulkan::pipeline::{
    BindingSource, LevelReads, RenderPipeline, SampledLevels, SamplerLevels, SamplerRef, SlotRef,
    Stage,
};
use metal2vulkan::vk;

/// The stages' tables a draw binds.
type Stages<'a> = [(Stage, &'a ArgumentTable)];

impl RenderLowering {
    /// The slots of a draw of `plan` and pipeline `compiled`, over the tables `stages`, that
    /// sample a texture a color attachment of the pass renders, those narrowed listed (the
    /// module docs), in an instance that holds the colors `feedback` in the feedback loop layout.
    /// Each of the others is counted where `count`; the caller counts those narrowed once the
    /// draw's uses are known ([`count_narrowed`](Self::count_narrowed)).
    pub(super) fn attached_slots(
        &self,
        device: &Device,
        (plan, compiled): (&BindingPlan, &RenderPipeline),
        stages: &Stages<'_>,
        (feedback, count): (Feedback, bool),
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
            let Some(attached) = texture.attached_span(&self.layout) else {
                continue;
            };
            let reads = (compiled.stages.iter())
                .find(|m| Some(m.stage) == shader_stage(slot.stage))
                .and_then(|m| m.level_reads.iter().find(|r| r.slot == slot.index));
            let parts = (device, (plan, compiled), stages, slot);
            let levels = match reads {
                _ if attached.level_count == 1 => Err(AttachedLevels::OneLevel),
                None => Err(AttachedLevels::AnyLevel),
                Some(reads) if reads.any => Err(AttachedLevels::AnyLevel),
                Some(reads) => (self.levels_apart(parts, reads, (&attached, feedback)))
                    .and_then(|levels| Ok((levels, self.levels_read(texture, levels)?))),
            };
            match levels {
                Ok((levels, used)) => {
                    // A view of the levels past the attached ones holds the attached ones too.
                    let spanned = match levels.first {
                        0 => 0,
                        _ => attached.colors,
                    };
                    narrowed.narrow(slot, levels, used, spanned)
                }
                Err(levels) if count => counts.attached_read_slot(levels),
                Err(_) => {}
            }
        }
        narrowed
    }

    /// The levels apart from those `attached` names that every read of `reads` (a slot's, which
    /// may not reach any level) reaches through the samplers the draw binds, in an instance that
    /// holds the colors `feedback` in the feedback loop layout (the module docs): the levels
    /// before the lowest attached one, or past the highest; else why the slot keeps its copy.
    fn levels_apart(
        &self,
        parts: Parts<'_>,
        reads: &LevelReads,
        (attached, feedback): (&AttachedSpan, Feedback),
    ) -> Result<NarrowedLevels, AttachedLevels> {
        let highest = reads.highest_level(|sampled| self.sampled_highest(parts, sampled));
        if highest.is_some_and(|highest| highest < attached.lowest) {
            let view = attached.lowest;
            return Ok(NarrowedLevels { first: 0, view });
        }
        let reaching = match attached.lowest {
            0 => AttachedLevels::FirstLevel,
            _ => AttachedLevels::Reaching,
        };
        // The view holds the attached levels, in `GENERAL` as its descriptor names them.
        if feedback.0 & attached.colors != 0 {
            return Err(reaching);
        }
        let count = attached.level_count;
        let view = highest.map_or(count, |highest| highest.saturating_add(1).min(count));
        let lowest = reads.lowest_level(|sampled| self.sampled_lowest(parts, sampled));
        match lowest.map(|lowest| lowest.min(view - 1)) {
            Some(first) if first > attached.highest => Ok(NarrowedLevels { first, view }),
            _ => Err(reaching),
        }
    }

    /// The use of `texture`'s levels `levels` a slot narrowed to them makes, where nothing the
    /// instance does writes those its reads reach (the module docs); else why the slot keeps its
    /// copy.
    fn levels_read(
        &self,
        texture: &Texture,
        levels: NarrowedLevels,
    ) -> Result<Use<ResourceKey>, AttachedLevels> {
        if !texture.reads_levels_in_place() {
            return Err(AttachedLevels::Apart);
        }
        let read = Access::new(
            ash::vk::PipelineStageFlags2::ALL_GRAPHICS,
            ash::vk::AccessFlags2::SHADER_SAMPLED_READ,
        );
        let used = texture.levels_use(levels.first..levels.view, (read, Access::NONE));
        let uses = std::slice::from_ref(&used);
        let attachment = render_target::shares_attachment(&self.layout, uses).unwrap_or(true);
        if attachment || self.unordered.written(uses) {
            return Err(AttachedLevels::Written);
        }
        Ok(used)
    }

    /// Count the narrowed slots of a draw that takes every step, `(before, past)` those before the
    /// attached levels and those past them, each narrowed, or written where a write of the draw's
    /// own met one's levels (`written`) and they took their copies.
    pub(super) fn count_narrowed(&self, (before, past): (usize, usize), written: bool) {
        let counts = self.core.counts();
        let kinds = [
            (AttachedLevels::Narrowed, before),
            (AttachedLevels::Past, past),
        ];
        for (levels, narrowed) in kinds {
            let levels = match written {
                true => AttachedLevels::Written,
                false => levels,
            };
            for _ in 0..narrowed {
                counts.attached_read_slot(levels);
            }
        }
    }

    /// The highest level, counted from the view's first, that `sampled`, samples of the texture at
    /// the slot of `parts`, reach through the samplers its draw binds with it (the module docs):
    /// the highest among the writes that pair them; `None` where none does, or where one may
    /// reach any level.
    fn sampled_highest(&self, parts: Parts<'_>, sampled: &SampledLevels) -> Option<u32> {
        let mut highest = None;
        let paired = self.bound_samplers(parts, sampled, |levels| match sampled.level(levels) {
            Some(level) => {
                highest = highest.max(Some(level));
                true
            }
            None => false,
        });
        highest.filter(|_| paired)
    }

    /// The lowest level, counted from the view's first, that `sampled`, samples of the texture at
    /// the slot of `parts`, may reach through the samplers its draw binds with it (the module
    /// docs): the lowest among the writes that pair them; 0 where none does.
    fn sampled_lowest(&self, parts: Parts<'_>, sampled: &SampledLevels) -> u32 {
        let mut lowest = None;
        let paired = self.bound_samplers(parts, sampled, |levels| {
            let level = sampled.lowest(levels);
            lowest = Some(lowest.map_or(level, |lowest: u32| lowest.min(level)));
            true
        });
        lowest.filter(|_| paired).unwrap_or(0)
    }

    /// Hand `each` the levels of every sampler a write of the draw of `parts` pairs with its slot
    /// for `sampled`'s samples (the module docs), until it answers `false`: whether every such
    /// write's sampler was known and `each` answered `true` for each.
    fn bound_samplers(
        &self,
        (device, (plan, compiled), stages, slot): Parts<'_>,
        sampled: &SampledLevels,
        mut each: impl FnMut(SamplerLevels) -> bool,
    ) -> bool {
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
                    let state = stage_table(stages, bound.stage)
                        .and_then(|table| table.samplers.get(&index));
                    let desc = state.and_then(|(state, clamps)| {
                        let descriptor = binding::bound_sampler(state, *clamps);
                        let use_ = state.bound_use(SamplerUse::new(compare, integer));
                        let format = Some(view.format);
                        let compile = device.compile_device();
                        metal2vulkan::bind::sampler_desc(
                            compile,
                            &descriptor,
                            use_,
                            opaque_black,
                            format,
                        )
                        .ok()
                    });
                    match desc {
                        Some(desc) => desc,
                        None => return false,
                    }
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
            if !each(SamplerLevels::of(&desc, finite)) {
                return false;
            }
        }
        true
    }
}

/// What a slot's levels are worked out over: the device, the draw's plan and pipeline, the tables
/// it binds, and the slot.
type Parts<'a> = (
    &'a Device,
    (&'a BindingPlan, &'a RenderPipeline),
    &'a Stages<'a>,
    SlotRef,
);

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
