//! A draw's key's residual form: a draw whose own `VkPipeline` is not made draws at once with its
//! residual form's where that one is made, and its own is made on the device's pool meanwhile,
//! nothing waiting for it.
//!
//! A buffer slot a function reads through a uniform view declares the boolean specialization
//! `SpecKey::RangeAtBindOffset`. A draw's plan sets it to 1 where the slot's descriptor range starts
//! at its bind offset (a bind offset the device's view base alignment divides, or inline bytes;
//! `instantiate::starts_at_bind`), and each read then takes the Metal offset alone, a constant for
//! a constant member, which a driver can load ahead of the shader (Mesa's ANV pushes such loads);
//! elsewhere to 0, and each read takes the slot's pushed `SlotBindOffset` residual plus the offset.
//! A slot bound to nothing takes 0: its residual is 0, so either form reads the same offset of the
//! same descriptor, and a draw that leaves such slots unbound takes the key the creation made
//! (`bind::plan::slots::range_at_bind_offset`).
//!
//! - **The residual form reads what the key reads, at every bind offset.** A key's residual form
//!   is the key with each of its `RangeAtBindOffset` constants at 0 ([`residual_form`]). The binder
//!   writes a slot's descriptor range and its residual from the bind offset alone, whichever form
//!   the pipeline takes (`instantiate::descriptor_range`), and where the range starts at the bind
//!   offset the residual is 0: the residual form reads byte `0 + m` of the very range the key's
//!   form reads byte `m` of. Only at a bind offset at or past the end of the bound buffer is the
//!   range widened down into it, the residual then not 0, and there either form reads only bytes
//!   outside the bound buffer.
//! - **Made at creation.** A state's creation makes its predicted key's residual form
//!   (`predicted.rs`), which a draw at any bind offset can take; a compute state's creation, its
//!   predicted region's (`dispatch_making.rs`), which a dispatch at any bind offset can take.
//! - **Drawn while the key's own is made.** A draw whose key's `VkPipeline` is neither made nor
//!   linked at once, and whose residual form's is made ([`residual_made`]), draws with the residual
//!   form's at once ([`RenderVariant::begin_graphics_pipeline`], and [`residual_drawn`] for the
//!   draws that make their pipelines at once). The key's own making goes to the device's pool,
//!   where no caller began it, at the rank of a making nothing waits for (`budget::Urgency::Ahead`),
//!   and a later draw of the key takes it once made. Only that direction: a draw whose key has 0
//!   where a made pipeline has 1 waits for its own making, as any draw does. A dispatch or a tile
//!   dispatch whose region's `VkPipeline` is not made takes its residual form's alike
//!   (`Variant::begin_region_pipeline`).
//! - **One form where the bind offset buys nothing** ([`reads_views_at_bind`]). Where the driver
//!   is not measured to load a uniform read at a constant offset ahead of the shader
//!   (`DriverFact::ConstantOffsetUniformReadsPreloaded`, which no driver carries today) every
//!   key is its residual form, because the other form costs each state a second `VkPipeline` at
//!   its first draw and buys nothing measurable: on MoltenVK a uniform view is a Metal constant
//!   buffer reached through an argument buffer and the two forms were measured alike, and on
//!   llvmpipe, RADV and NVIDIA the second form was counted and then timed no faster (the fact's
//!   own documentation holds the figures). The plans' specializations and the dispatches'
//!   regions take the residual form before any key is looked up (`RenderVariant::plan`,
//!   `Variant::keyed_regions`): a state's creation makes the one pipeline its draws at every bind
//!   offset take. The shapes and plans stay every device's (`argument_table::AtBindViews`): a bind
//!   offset the view base alignment divides still makes a shape and a plan of its own, whose key
//!   alone takes the residual form.
//! - **Only GPU time moves.** Until the key's making lands, its draws' uniform reads take the
//!   residual: on ANV, loads where the key's form has pushed constants. `WorkCounts` counts the
//!   draws and dispatches recorded so (`residual_form_draws`) and the makings begun so
//!   (`residual_form_makings`).
//!   Such a making may outlast its draw's command buffer, which does not wait for it
//!   (`WorkCounts::pipeline_pool_busy` says when the pool holds none).
//!
//! [`residual_form`]: RenderVariant::residual_form
//! [`reads_views_at_bind`]: reads_views_at_bind
//! [`residual_made`]: RenderVariant::residual_made
//! [`residual_drawn`]: RenderVariant::residual_drawn

use super::{DrawPipeline, GraphicsState, RenderVariant};
use crate::degradation::BoundFormats;
use crate::pipeline::graphics::GraphicsKey;
use crate::pipeline::making::Begun;
use crate::pipeline::objects::RawPipeline;
use crate::render_target::RenderPassObject;
use metal2vulkan::cache::SpecializationKey;
use metal2vulkan::device::DriverFact;
use metal2vulkan::pipeline::{SpecConstant, SpecKey};
use std::collections::BTreeSet;
use std::sync::Arc;

impl RenderVariant {
    /// `specialization` with each of the variant's `SpecKey::RangeAtBindOffset` constants at 0,
    /// its residual form (the module docs); `None` where every one of them is 0 there already. A
    /// constant `specialization` gives no entry takes the module's default, 0.
    pub(crate) fn residual_form(
        &self,
        specialization: &SpecializationKey,
    ) -> Option<SpecializationKey> {
        let constants = self
            .pipeline
            .stages
            .iter()
            .flat_map(|module| &module.specialization.constants);
        residual_form(constants, specialization)
    }

    /// The made `VkPipeline` of `key`'s residual form (the module docs), where `key` is not its
    /// own residual form.
    pub(super) fn residual_made(&self, key: &GraphicsKey) -> Option<Arc<RawPipeline>> {
        let residual = GraphicsKey {
            specialization: self.residual_form(&key.specialization)?,
            topology: key.topology,
            primitive_restart: key.primitive_restart,
            pass: key.pass.clone(),
            strides: key.strides.clone(),
            fixed: key.fixed,
            sample_positions: key.sample_positions.clone(),
        };
        self.pipelines.made(&residual)
    }

    /// For a draw of `state` that binds `formats` and makes its pipeline at once
    /// ([`graphics_pipeline`](Self::graphics_pipeline)): where `key`'s `VkPipeline` is not made and
    /// its residual form's is, the residual form's, `key`'s making handed to the device's pool
    /// against `compatible` where no caller began it (the module docs); `key`'s own where it was
    /// made meanwhile or is linked at once. `None` where the draw makes or waits for `key`'s own.
    pub(super) fn residual_drawn(
        self: &Arc<Self>,
        key: &GraphicsKey,
        compatible: &Arc<RenderPassObject>,
        (state, formats): (&GraphicsState, BoundFormats<'_>),
    ) -> Option<Arc<RawPipeline>> {
        if self.pipelines.made(key).is_some() {
            return None;
        }
        let drawn = self.residual_made(key)?;
        let device = &self.objects.device;
        match self.pipelines.begin(key) {
            Begun::Made(made) => return Some(made),
            Begun::Pending(_) => {}
            Begun::Yours(ticket) => {
                match self.make_drawn(device, key, compatible, (state, formats), ticket) {
                    DrawPipeline::Made(made) => return Some(made),
                    DrawPipeline::Making(_) => device.shared().counts.residual_form_making(),
                }
            }
        }
        device.shared().counts.residual_form_drawn();
        Some(drawn)
    }
}

/// `specialization` with each `SpecKey::RangeAtBindOffset` constant of `constants` (a pipeline's
/// stages') at 0, its residual form (the module docs); `None` where every one of them is 0 there
/// already. A constant `specialization` gives no entry takes the module's default, 0.
pub(crate) fn residual_form<'a>(
    constants: impl Iterator<Item = &'a SpecConstant> + Clone,
    specialization: &SpecializationKey,
) -> Option<SpecializationKey> {
    let at_bind = |id: u32| {
        constants.clone().any(|constant| {
            constant.id == id && matches!(constant.key, SpecKey::RangeAtBindOffset(_))
        })
    };
    let mut residual: Option<SpecializationKey> = None;
    for entry in &specialization.entries {
        let at = entry.offset as usize;
        let range = at..at + entry.size as usize;
        let set = specialization
            .data
            .get(range.clone())
            .is_some_and(|bytes| bytes.iter().any(|&byte| byte != 0));
        if set && at_bind(entry.constant_id) {
            let residual = residual.get_or_insert_with(|| specialization.clone());
            residual.data[range].fill(0);
        }
    }
    residual
}

/// Whether `device` reads a uniform view at its bind offset alone where the view's range starts
/// there (`SpecKey::RangeAtBindOffset` 1), a `VkPipeline` of its own beside the residual form's
/// (the module docs); where not, every key is its residual form, and a state's pipelines are the
/// residual form's alone.
///
/// The question is the driver's, and it is the driver fact that answers it, never the driver's
/// name: a driver that loads a uniform read at a constant offset ahead of the shader
/// ([`DriverFact::ConstantOffsetUniformReadsPreloaded`]) is one where the second form earns the
/// second pipeline. No driver is measured with the fact, so every device reads its views at their
/// residuals today; the fact's own documentation holds what was measured where, and a driver
/// measured with it takes the other form with no change here.
pub(crate) fn reads_views_at_bind(device: &crate::Device) -> bool {
    views_read_at_bind(&device.properties().driver_facts)
}

/// [`reads_views_at_bind`] of a device's driver facts alone, which a test can hand the facts of
/// each committed profile: the fact, and nothing of the driver's identity.
pub(crate) fn views_read_at_bind(facts: &BTreeSet<DriverFact>) -> bool {
    facts.contains(&DriverFact::ConstantOffsetUniformReadsPreloaded)
}
