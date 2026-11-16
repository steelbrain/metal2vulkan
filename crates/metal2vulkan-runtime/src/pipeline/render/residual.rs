//! A draw's key's general form: a draw whose own `VkPipeline` is not made draws at once with its
//! general form's where that one is made, and its own is made on the device's pool meanwhile,
//! nothing waiting for it. The general form is the key with each constant that has an arm valid
//! for every binding at that arm, at that arm ([`general_form`]): the uniform views' (the residual
//! form, below) and the unmeasured texel class's (the end of these docs).
//!
//! A buffer slot a function reads through a uniform view declares the boolean specialization
//! `SpecKey::RangeAtBindOffset`. A draw's plan sets it to 1 where the slot's descriptor range starts
//! at its bind offset (a bind offset the device's view base alignment divides, or inline bytes;
//! `instantiate::starts_at_bind`), and each read then takes the Metal offset alone, a constant for
//! a constant member, which a driver can load ahead of the shader (Mesa's ANV pushes such loads);
//! elsewhere to 0, and each read takes the slot's pushed `SlotBindOffset` residual plus the offset.
//! A uniform view bound to nothing takes 0, and a storage view bound to nothing what a bind at
//! offset 0 takes, the commonest bind (`bind::plan::slots::range_at_bind_offset`): its residual is
//! 0, so either form reads the same offset of the same descriptor, and a plan of nothing bound (a
//! creation's prediction) names the key of the command that binds its storage views at offset 0.
//!
//! - **The residual form reads what the key reads, at every bind offset.** A key's residual form
//!   is the key with each of its `RangeAtBindOffset` constants at 0 ([`residual_form`]). The binder
//!   writes a slot's descriptor range and its residual from the bind offset alone, whichever form
//!   the pipeline takes (`instantiate::descriptor_range`), and where the range starts at the bind
//!   offset the residual is 0: the residual form reads byte `0 + m` of the very range the key's
//!   form reads byte `m` of. Only at a bind offset at or past the end of the bound buffer is the
//!   range widened down into it, the residual then not 0, and there either form reads only bytes
//!   outside the bound buffer.
//! - **Made at creation, both forms.** A state's creation makes its predicted key, each
//!   uniform-view slot read at its residual and each storage view as a bind at offset 0 reads it,
//!   and, where that key is not its own general form, the general form beside it, the two made
//!   together (`predicted.rs`, `RenderVariant::make_at_creation`); a compute state's creation its
//!   predicted region and that region's residual form alike (`dispatch_making.rs`). The predicted
//!   key is what the commonest draw or dispatch takes, every storage view bound at offset 0, so it
//!   makes nothing at its first use; the general form is what one binding a storage view
//!   elsewhere takes at once while its own is made. Off a profile whose storage views declare the
//!   key the two are one key, and the creation makes one pipeline.
//! - **Drawn while the key's own is made.** A draw whose key's `VkPipeline` is neither made nor
//!   linked at once, and whose general form's is made ([`general_made`]), draws with the general
//!   form's at once ([`RenderVariant::begin_graphics_pipeline`], and [`general_drawn`] for the
//!   draws that make their pipelines at once). The key's own making goes to the device's pool,
//!   where no caller began it, at the rank of a making nothing waits for (`budget::Urgency::Ahead`),
//!   and a later draw of the key takes it once made: such a draw keeps nothing to draw again
//!   (`render/draw_again.rs`), whose draws again would record through the general form's still.
//!   Only that direction: a draw whose key has 0
//!   where a made pipeline has 1 waits for its own making, as any draw does. A dispatch or a tile
//!   dispatch whose region's `VkPipeline` is not made takes its residual form's alike
//!   (`Variant::begin_region_pipeline`).
//! - **One form where the bind offset buys nothing** ([`reads_views_at_bind`]). Where the driver
//!   is not measured to load a uniform read at a constant offset ahead of the shader
//!   (`DriverFact::ConstantOffsetUniformReadsPreloaded`, which no driver carries today), every
//!   key's uniform views take their residual form ([`uniform_residual_form`]), because the other
//!   form costs each state a second `VkPipeline` at its first draw and buys nothing measurable: on
//!   MoltenVK a uniform view is a Metal constant buffer reached through an argument buffer and the
//!   two forms were measured alike, and on llvmpipe, RADV and NVIDIA the second form was counted
//!   and then timed no faster (the fact's own documentation holds the figures). On ANV, Mesa
//!   26.2.6 on the Iris Xe of RPL-P, it ran slower: `dispatch_chain --binds 4` (a uniform view of
//!   inline bytes beside three storage views bound at offset 0) took 161.78 us a dispatch with
//!   the uniform view read at its bind offset and 142.77 us with it read at its residual, and the
//!   `fixed`, `moved` and `tex` shapes 161.77, 165.23 and 128.29 us against 143.17, 143.04 and
//!   119.40 us (job m3-max-34275, medians of 7 interleaved rounds). There, where the profile takes
//!   `DriverWorkaround::PushReadsLengthenThreadPayload`, a storage view declares the key too,
//!   which a bind at offset 0 sets (`bind::ZERO_OFFSET_CLASS`) and the plan keeps, so a key whose
//!   views are all storage views bound at offset 0 reads no residual word. The plans'
//!   specializations and the dispatches' regions take that form before any key is looked up
//!   (`RenderVariant::plan`, `Variant::keyed_regions`): off that profile a state's creation makes
//!   the one pipeline its draws at every bind offset take, and on it the two the module's
//!   **Made at creation, both forms** names. A kernel's uniform view there has a storage twin
//!   (`DescriptorView::UniformWithStorageTwin`), whose key the plan keeps where no other
//!   `RangeAtBindOffset` constant of the key is 0 and the kernel reads no other word the binder
//!   writes for it ([`reads_other_words`]: a root table's address, a multi-region or flat grid's
//!   region): at 1 it reads the twin at its bind offset alone, not the uniform view, which pays
//!   only where no such word is left to read ([`uniform_residual_form_of`]); beside a storage
//!   view bound elsewhere, or such a word, it takes 0, and the uniform view at its residual. The
//!   shapes and plans stay every device's (`argument_table::AtBindViews`): a bind offset the view
//!   base alignment divides still makes a shape and a plan of its own, whose key's uniform views
//!   alone take their residual form.
//! - **Only GPU time moves.** Until the key's making lands, its draws' uniform reads take the
//!   residual: on ANV, loads where the key's form has pushed constants. `WorkCounts` counts the
//!   draws and dispatches recorded so (`residual_form_draws`) and the makings begun so
//!   (`residual_form_makings`), whichever of the general form's constants moved.
//!   Such a making may outlast its draw's command buffer, which does not wait for it
//!   (`WorkCounts::pipeline_pool_busy` says when the pool holds none).
//!
//! **The texel routes.** On a device that samples through Metal's own texture unit (MoltenVK), a
//! texture slot a function converts to `half` declares two booleans more:
//!
//! - `SpecKey::UnmeasuredTexels` names the bound class to the slot's precision note alone: no
//!   instruction reads it, so both values run alike, and 0 is its general arm. A slot bound to
//!   nothing takes 0, and a draw binding a class whose `half` conversion is not measured
//!   (`TexelFormat::unmeasured_on_metal_sampler`: BC4 to BC7 and the linear ASTC formats among
//!   them) 1. A state's creation predicts no texture bound, so such a draw keys apart from the
//!   predicted key, and its general form is the predicted key itself: the draw takes the pipeline
//!   the creation made at once while its own is made.
//! - `SpecKey::PlainHalfTexels` keeps the value the draw's own binding gives it, and is no
//!   constant of the general form. At 0 each texel converts by the class routine, the bound class
//!   read from push data, which serves every class; at 1 by the plain arm, which lets Metal's
//!   compiler fold the conversion into the image instruction (`lower::texture::half`) and serves
//!   every plain class (`TexelFormat::plain_on_metal_sampler`: every class but a YCbCr field on
//!   the grid `8192`), rounding the `float` texel to nearest even as the routine does for those.
//!   A slot bound to nothing, or to a plain class, takes 1, and a draw binding a YCbCr field on
//!   the grid `8192` takes 0. A creation predicts the plain arm, which every draw of a plain class
//!   takes, and makes that one pipeline: were 0 the general form, it would make a pipeline of the
//!   routine's arm too, which a draw of a plain class takes only while its own is still made. A
//!   draw binding a YCbCr field on the grid `8192` makes its own key, as a draw of any key its
//!   creation did not predict; on a device whose memory is the host's such a slot reads its planes
//!   in place, in a variant of its own (`VariantKey::ycbcr_in_place`), which no creation predicts
//!   either.
//!
//! A draw that takes its key's general form runs the instructions its own key runs but for the
//! residual's reads in place of the bind offset's: an unmeasured class's draw runs its own key's.
//!
//! Constants with no arm that serves every binding stay as before, each value a pipeline of its
//! own: `SpecKey::PlainHalfTexels` (above), `SpecKey::SamplerState` (each sampler word a sampling
//! emulation of its own), `SpecKey::NearestPixelSampler` (a nearest pixel sampler's sample at its
//! texel's centre, which the other arm's quotient does not read exactly) and
//! `SpecKey::BicubicTexels` (how Metal's bicubic filter treats each format's texels).
//!
//! [`general_form`]: RenderVariant::general_form
//! [`reads_views_at_bind`]: reads_views_at_bind
//! [`general_made`]: RenderVariant::general_made
//! [`general_drawn`]: RenderVariant::general_drawn

use super::{DrawPipeline, GraphicsState, RenderVariant};
use crate::degradation::BoundFormats;
use crate::pipeline::graphics::GraphicsKey;
use crate::pipeline::making::Begun;
use crate::pipeline::objects::RawPipeline;
use crate::render_target::RenderPassObject;
use metal2vulkan::cache::SpecializationKey;
use metal2vulkan::device::DriverFact;
use metal2vulkan::pipeline::{
    PipelineInterface, PushField, SlotInterface, SlotRef, SpecConstant, SpecKey, Stage,
};
use metal2vulkan::vk::SpecializationMapEntry;
use std::collections::BTreeSet;
use std::sync::Arc;

impl RenderVariant {
    /// `specialization` with each of the variant's constants that has an arm serving every binding
    /// at that arm, its general form (the module docs): `SpecKey::RangeAtBindOffset` and
    /// `SpecKey::UnmeasuredTexels` at 0; `None` where every one of them is there already.
    pub(crate) fn general_form(
        &self,
        specialization: &SpecializationKey,
    ) -> Option<SpecializationKey> {
        let constants = self
            .pipeline
            .stages
            .iter()
            .flat_map(|module| &module.specialization.constants)
            .map(|constant| (constant.id, constant.key));
        general_form(constants, specialization)
    }

    /// `specialization` with each of the variant's `SpecKey::RangeAtBindOffset` constants of a slot
    /// it reads through a uniform view at 0, the key a plan takes where the device reads its
    /// uniform views at their residuals ([`uniform_residual_form`]); `None` where every one of
    /// them is 0 there already.
    pub(crate) fn uniform_residual_form(
        &self,
        specialization: &SpecializationKey,
    ) -> Option<SpecializationKey> {
        let constants = self
            .pipeline
            .stages
            .iter()
            .flat_map(|module| &module.specialization.constants);
        uniform_residual_form(constants, &self.pipeline.interface, false, specialization)
    }

    /// `key` with its specialization's general form ([`general_form`](Self::general_form)), where
    /// `key` is not its own general form.
    pub(crate) fn general_key(&self, key: &GraphicsKey) -> Option<GraphicsKey> {
        Some(GraphicsKey {
            specialization: self.general_form(&key.specialization)?,
            topology: key.topology,
            primitive_restart: key.primitive_restart,
            pass: key.pass.clone(),
            strides: key.strides.clone(),
            fixed: key.fixed,
            sample_positions: key.sample_positions.clone(),
            feedback_loop: key.feedback_loop,
        })
    }

    /// The made `VkPipeline` of `key`'s general form (the module docs), where `key` is not its
    /// own general form.
    pub(super) fn general_made(&self, key: &GraphicsKey) -> Option<Arc<RawPipeline>> {
        self.pipelines.made(&self.general_key(key)?)
    }

    /// For a draw of `state` that binds `formats` and makes its pipeline at once
    /// ([`graphics_pipeline`](Self::graphics_pipeline)): where `key`'s `VkPipeline` is not made and
    /// its general form's is, the general form's, `key`'s making handed to the device's pool
    /// against `compatible` where no caller began it (the module docs); `key`'s own where it was
    /// made meanwhile or is linked at once; with whether it is `key`'s own. `None` where the draw
    /// makes or waits for `key`'s own.
    pub(super) fn general_drawn(
        self: &Arc<Self>,
        key: &GraphicsKey,
        compatible: &Arc<RenderPassObject>,
        (state, formats): (&GraphicsState, BoundFormats<'_>),
    ) -> Option<(Arc<RawPipeline>, bool)> {
        if self.pipelines.made(key).is_some() {
            return None;
        }
        let drawn = self.general_made(key)?;
        let device = &self.objects.device;
        match self.pipelines.begin(key) {
            Begun::Made(made) => return Some((made, true)),
            Begun::Pending(_) => {}
            Begun::Yours(ticket) => {
                match self.make_drawn(device, key, compatible, (state, formats), ticket) {
                    DrawPipeline::Made(made) => return Some((made, true)),
                    DrawPipeline::Making(_) => {
                        device.shared().counts.thread().residual_form_making()
                    }
                }
            }
        }
        device.shared().counts.thread().residual_form_drawn();
        Some((drawn, false))
    }
}

/// `specialization` with each `SpecKey::RangeAtBindOffset` constant of `constants` (a pipeline's
/// stages') at 0, its residual form (the module docs); `None` where every one of them is 0 there
/// already. A constant `specialization` gives no entry takes the module's default, 0.
pub(crate) fn residual_form<'a>(
    constants: impl Iterator<Item = &'a SpecConstant> + Clone,
    specialization: &SpecializationKey,
) -> Option<SpecializationKey> {
    let constants = constants.map(|constant| (constant.id, constant.key));
    residual_form_of(constants, |_| true, specialization)
}

/// `specialization` with each `SpecKey::RangeAtBindOffset` constant of `constants` (a pipeline's
/// stages') whose slot `interface` (the pipeline's) reads through a uniform view at 0, the key a
/// plan takes where the device reads its uniform views at their residuals (the module docs);
/// `None` where every one of them is 0 there already. A storage view's constant, which only a
/// profile taking `DriverWorkaround::PushReadsLengthenThreadPayload` declares, keeps its value:
/// at its bind offset alone it reads no residual word there. A uniform view's with a storage twin
/// (`DescriptorView::UniformWithStorageTwin`), declared there alone too, keeps its value where no
/// other `RangeAtBindOffset` constant of the key is 0 and its stage reads no other word the binder
/// writes for it ([`reads_other_words`], of the interface's push placements; `flat_grid`: the
/// pipeline is a kernel's flat-grid variant): at 1 it reads the twin, not the uniform view, which
/// pays only where it leaves no such word to read ([`uniform_residual_form_of`]). Where neither
/// declares the key this is [`residual_form`].
pub(crate) fn uniform_residual_form<'a>(
    constants: impl Iterator<Item = &'a SpecConstant> + Clone,
    interface: &PipelineInterface,
    flat_grid: bool,
    specialization: &SpecializationKey,
) -> Option<SpecializationKey> {
    let form = |slot| match interface.slots.get(&slot) {
        Some(SlotInterface::Buffer(b)) => Some(b.form),
        _ => None,
    };
    let uniform = |slot| form(slot).is_some_and(|f| f.uniform_view() && !f.storage_twin());
    let twin = |slot| form(slot).is_some_and(|f| f.storage_twin());
    let constants = constants.map(|constant| (constant.id, constant.key));
    let placed = || {
        (interface.push.iter())
            .flat_map(|placement| (placement.fields.iter()).map(|f| (placement.stage, f.field)))
    };
    let other_words = |stage, key: &SpecializationKey| {
        reads_other_words(placed(), constants.clone(), (stage, flat_grid), key)
    };
    uniform_residual_form_of(
        constants.clone(),
        uniform,
        twin,
        other_words,
        specialization,
    )
}

/// `specialization` with each `SpecKey::RangeAtBindOffset` constant of `constants` (a pipeline's
/// stages', each its ID and its meaning) whose slot `uniform` holds at 0, and each whose slot
/// `twin` holds (a uniform view with a storage twin) at 0 too where a `RangeAtBindOffset`
/// constant of that key is 0 (a view that reads its pushed residual word), or where `other_words`
/// holds of a twin's stage at 1 under that key (the stage reads another word the binder writes
/// for it, [`reads_other_words`]); `None` where every one of them is 0 there already. A constant
/// `specialization` gives no entry takes the module's default, 0.
///
/// A twin's key at 1 reads the twin, a storage view, at its bind offset, and pays where it leaves
/// no residual word to read: on ANV, Mesa 26.2.6 on the Iris Xe of RPL-P
/// (`DriverWorkaround::PushReadsLengthenThreadPayload`), `dispatch_chain`'s kernel of three
/// storage views and the word of 1 (4480 threadgroups of 256, 200 dispatches a command buffer,
/// medians of 7 interleaved rounds, job m3-max-34864) ran in 102.1 us a dispatch with its storage
/// views at offset 0 and the word inline (`--binds 4`) through the twin, against 142.8 us through
/// the uniform view at its residual; with two storage views and the word at offsets that move
/// every dispatch (`--binds moved`), which keeps the storage views' residual words, it ran in
/// 154.0 us through the twin against 143.1 us through the uniform view. The payload is the
/// stage's push reads' (job m3-max-34063: the runtime's module of a storage view ran in 120.24 us
/// a dispatch, and 55.69 us with its one push read replaced by a constant), so any other word the
/// stage reads keeps it long as a residual word does.
pub(crate) fn uniform_residual_form_of(
    constants: impl Iterator<Item = (u32, SpecKey)> + Clone,
    uniform: impl Fn(SlotRef) -> bool,
    twin: impl Fn(SlotRef) -> bool,
    other_words: impl Fn(Stage, &SpecializationKey) -> bool,
    specialization: &SpecializationKey,
) -> Option<SpecializationKey> {
    let plan = residual_form_of(constants.clone(), &uniform, specialization);
    let key = plan.as_ref().unwrap_or(specialization);
    let reads_residual = constants
        .clone()
        .any(|(id, key_of)| matches!(key_of, SpecKey::RangeAtBindOffset(_)) && !is_set(key, id));
    let reads_other = constants.clone().any(|(id, key_of)| {
        matches!(key_of, SpecKey::RangeAtBindOffset(slot)
            if twin(slot) && is_set(key, id) && other_words(slot.stage, key))
    });
    match reads_residual || reads_other {
        true => residual_form_of(
            constants,
            |slot| uniform(slot) || twin(slot),
            specialization,
        ),
        false => plan,
    }
}

/// Whether `stage`, of a pipeline whose placed push fields are `fields` (each with its stage, the
/// interface's `PushPlacement`s) and whose stages' constants are `constants`, reads under
/// `specialization` a word the binder writes for it other than a residual the key's
/// `RangeAtBindOffset` constants leave unread (`flat_grid`: the pipeline is a kernel's flat-grid
/// variant). Each field placed for the stage, in its push window or its `ArgumentTable` (a
/// uniform buffer read at constant offsets, which ANV pushes too, the module docs), is read but:
/// a slot's `SlotBindOffset` residual where the key sets the slot's `RangeAtBindOffset`; and the
/// `DispatchRegion` words where the key leaves the stage's
/// `MultiRegion` 0 (a single-region threadgroup dispatch, `lower::dispatch`'s regions), but on a
/// flat-grid variant, whose one region reads them whatever that constant says
/// (`bind::dispatch`'s ragged grids). A root table's or the execution status's address, a replay's
/// region record, a slot's guest base, a texture's levels and every other field are read wherever
/// placed.
pub(crate) fn reads_other_words(
    mut fields: impl Iterator<Item = (Stage, PushField)>,
    constants: impl Iterator<Item = (u32, SpecKey)> + Clone,
    (stage, flat_grid): (Stage, bool),
    specialization: &SpecializationKey,
) -> bool {
    let set = |wanted: SpecKey| {
        (constants.clone()).any(|(id, key)| key == wanted && is_set(specialization, id))
    };
    let multi_region = match stage {
        Stage::Object => SpecKey::ObjectMultiRegion,
        _ => SpecKey::MultiRegion,
    };
    fields.any(|(of, field)| {
        of == stage
            && match field {
                PushField::SlotBindOffset(slot) => !set(SpecKey::RangeAtBindOffset(slot)),
                PushField::DispatchRegion => flat_grid || set(multi_region),
                _ => true,
            }
    })
}

/// Whether constant `id` of `specialization` is not 0; one it gives no entry takes the module's
/// default, 0.
fn is_set(specialization: &SpecializationKey, id: u32) -> bool {
    (specialization.entries.iter())
        .any(|entry| entry.constant_id == id && entry_set(specialization, entry))
}

/// `specialization` with each `SpecKey::RangeAtBindOffset` constant of `constants` (a pipeline's
/// stages', each its ID and its meaning) whose slot `views` holds at 0; `None` where every one of
/// them is 0 there already. A constant `specialization` gives no entry takes the module's
/// default, 0.
pub(crate) fn residual_form_of(
    constants: impl Iterator<Item = (u32, SpecKey)> + Clone,
    views: impl Fn(SlotRef) -> bool,
    specialization: &SpecializationKey,
) -> Option<SpecializationKey> {
    let at_bind = |id: u32| {
        constants.clone().any(|(at, key)| {
            at == id && matches!(key, SpecKey::RangeAtBindOffset(slot) if views(slot))
        })
    };
    zeroed(specialization, at_bind)
}

/// `specialization` with each constant of `constants` (a pipeline's stages', each its ID and its
/// meaning) that has an arm serving every binding at that arm (the module docs):
/// `SpecKey::RangeAtBindOffset` (the residual form's) and `SpecKey::UnmeasuredTexels`, each at 0;
/// `None` where every one of them is 0 there already. A constant `specialization` gives no entry
/// takes the module's default, 0. `SpecKey::PlainHalfTexels` keeps its value: its plain arm
/// serves only the plain classes.
pub(crate) fn general_form(
    constants: impl Iterator<Item = (u32, SpecKey)> + Clone,
    specialization: &SpecializationKey,
) -> Option<SpecializationKey> {
    let general = |id: u32| {
        constants.clone().any(|(at, key)| {
            at == id
                && matches!(
                    key,
                    SpecKey::RangeAtBindOffset(_) | SpecKey::UnmeasuredTexels(_)
                )
        })
    };
    zeroed(specialization, general)
}

/// `specialization` with the value of each constant `chosen` names zeroed, where one of them is
/// not 0 there; `None` where none is.
fn zeroed(
    specialization: &SpecializationKey,
    chosen: impl Fn(u32) -> bool,
) -> Option<SpecializationKey> {
    let mut zeroed: Option<SpecializationKey> = None;
    for entry in &specialization.entries {
        let at = entry.offset as usize;
        let range = at..at + entry.size as usize;
        if entry_set(specialization, entry) && chosen(entry.constant_id) {
            let zeroed = zeroed.get_or_insert_with(|| specialization.clone());
            zeroed.data[range].fill(0);
        }
    }
    zeroed
}

/// Whether `entry` of `specialization` holds a value other than 0.
fn entry_set(specialization: &SpecializationKey, entry: &SpecializationMapEntry) -> bool {
    let at = entry.offset as usize;
    (specialization.data.get(at..at + entry.size as usize))
        .is_some_and(|bytes| bytes.iter().any(|&byte| byte != 0))
}

/// Whether `device` reads a uniform view at its bind offset alone where the view's range starts
/// there (`SpecKey::RangeAtBindOffset` 1), a `VkPipeline` of its own beside the residual form's
/// (the module docs); where not, every key's uniform views take their residual form
/// ([`uniform_residual_form`]).
///
/// The question is the driver's, and it is the driver fact that answers it, never the driver's
/// name: a driver that loads a uniform read at a constant offset ahead of the shader
/// ([`DriverFact::ConstantOffsetUniformReadsPreloaded`]) is one where the second form earns the
/// second pipeline. No driver is measured with the fact, so every device reads its uniform views
/// at their residuals today, ANV on the Iris Xe of RPL-P too, where the other form was measured
/// slower (the module docs); the fact's own documentation holds what was measured where, and a
/// driver measured with it takes the other form with no change here. A storage view's key, which
/// only a profile taking `DriverWorkaround::PushReadsLengthenThreadPayload` declares, is not this
/// question's: its plan keeps it on every device.
pub(crate) fn reads_views_at_bind(device: &crate::Device) -> bool {
    views_read_at_bind(&device.properties().driver_facts)
}

/// [`reads_views_at_bind`] of a device's driver facts alone, which a test can hand the facts of
/// each committed profile: the fact, and nothing of the driver's identity.
pub(crate) fn views_read_at_bind(facts: &BTreeSet<DriverFact>) -> bool {
    facts.contains(&DriverFact::ConstantOffsetUniformReadsPreloaded)
}
