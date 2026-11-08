//! What a host step's recording of a [`DeferredPass`] needs that the commit makes ahead of it.
//!
//! A host step records the pass on the steps' thread (`scheduler/steps.rs`), a serial point of the
//! command buffers'
//! stream: the work before its batch completed and the batch waits for the step. So the commit of
//! a pass its host step records makes, on the committing thread, what the step may need that the
//! pass alone decides, whatever its commands turn out to be, and the step makes only what the
//! commit could not know:
//!
//! - **The runs' passes of an instance** that holds an execution and that the encoder gave none
//!   (`RunPasses`, made as the encoder ends only where an execution may split its instance): a
//!   run of another shape splits the instance too, and the instance's own runs around it take
//!   these passes.
//! - **The memoryless backings.** Where the pass has a memoryless attachment whose image is
//!   lazily allocated (MoltenVK), an execution whose commands run as several renders it into a
//!   backing from the pass's first instance on (`Rendered::back_memoryless`). The commit takes
//!   the backing its texture keeps for the queue (`attachment_stand_in::kept_backing`, made the
//!   first time) and makes, for each instance that holds an execution, the runs' passes and the
//!   framebuffer over the backed layout, which the step takes where it backs the pass with the
//!   same backings.
//!
//! The commit makes them whether or not the commands split, so a pass whose commands never split
//! keeps them unused until its command buffer completes: render passes and framebuffers, and of
//! memory only the backings, one per memoryless texture, subresource and queue however many
//! passes there are (Metal's memoryless texture takes none). A part the commit fails to make is
//! left unmade, and the step makes it as it meets it.

use super::{back, backing, Backing, DeferredPass, PassInstance, Piece, RunPasses};
use crate::command_buffer::CommandBuffer;
use crate::error::Error;
use crate::render_pass::PassLayout;
use crate::render_target::{self, Feedback, FramebufferObject, Position, RenderPassObject};
use crate::Device;
use ash::vk;
use metal2vulkan::bind::ViewSetup;
use std::time::Instant;

/// What the commit made ahead for a pass its host step records (the module docs).
pub(super) struct Ahead {
    /// Each instance's runs' passes, where it holds an execution and the encoder gave it none.
    pub(super) runs: Vec<Option<RunPasses>>,
    /// The pass's memoryless backings and each instance's objects over them, where it has such
    /// an attachment.
    pub(super) backed: Option<Backed>,
    /// The passes and framebuffers, held while the pass lives and by the command buffer that
    /// records it until it completes.
    _held: (Vec<RenderPassObject>, Vec<FramebufferObject>),
}

/// The pass's memoryless backings, and for each instance that holds an execution, its runs'
/// passes and its framebuffer over them.
pub(super) struct Backed {
    backings: Vec<Backing>,
    pub(super) instances: Vec<Option<(RunPasses, vk::Framebuffer)>>,
}

impl Backed {
    /// Whether `backed`, the backings a step's recording rendered the pass into, are these.
    pub(super) fn backs(&self, backed: &[Backing]) -> Result<bool, Error> {
        if backed.len() != self.backings.len() {
            return Ok(false);
        }
        for (made, rendered) in self.backings.iter().zip(backed) {
            if made.index != rendered.index
                || !super::same_subresource(&made.target, &rendered.target)?
            {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

impl Ahead {
    /// What `pass`'s host step may need, made now for `command_buffer`'s commit (the module
    /// docs), counted in the device's work counts.
    pub(super) fn make(pass: &DeferredPass, command_buffer: &CommandBuffer) -> Ahead {
        let started = Instant::now();
        let device = command_buffer.device();
        let mut held = (Vec::new(), Vec::new());
        let runs = pass
            .instances
            .iter()
            .map(|instance| {
                if instance.runs.is_some() || !executes(instance) {
                    return None;
                }
                let own = crate::render::instance_layout(
                    &pass.layout,
                    pass.bare.as_ref(),
                    &instance.shape,
                );
                let (runs, objects) = run_passes(device, own, instance).ok()?;
                held.0.extend(objects);
                Some(runs)
            })
            .collect();
        let queue = command_buffer.queue_place().id();
        let backed = backed(command_buffer, pass, queue, &mut held)
            .ok()
            .flatten();
        let nanos = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let (passes, framebuffers) = (held.0.len() as u64, held.1.len() as u64);
        device
            .shared()
            .counts
            .made_ahead(passes, framebuffers, nanos);
        Ahead {
            runs,
            backed,
            _held: held,
        }
    }
}

/// Whether `instance` runs an execution.
fn executes(instance: &PassInstance) -> bool {
    instance
        .pieces
        .iter()
        .any(|piece| matches!(piece, Piece::Execution(_)))
}

/// The passes of `instance`'s runs over `own`, its layout: the first run's at the instance's
/// first position, one between two others, and the last run's at its last (as the encoder makes
/// them, `render.rs`), with the objects that hold them. Every attachment is in `GENERAL`: an
/// instance an execution joins holds none out of it (`render_target::depth_layout`).
fn run_passes(
    device: &Device,
    own: &PassLayout,
    instance: &PassInstance,
) -> Result<(RunPasses, [RenderPassObject; 3]), Error> {
    let at = instance.position;
    let pass = |first, last| {
        render_target::final_pass(
            device,
            own,
            &instance.shape,
            Position { first, last },
            Feedback::NONE,
            vk::ImageLayout::GENERAL,
        )
    };
    let objects = [
        pass(at.first, false)?,
        pass(false, false)?,
        pass(false, at.last)?,
    ];
    let runs = RunPasses {
        first: objects[0].raw,
        between: objects[1].raw,
        last: objects[2].raw,
    };
    Ok((runs, objects))
}

/// The memoryless backings of `pass`, a pass of `queue`, and each instance's objects over them
/// (the module docs): `None` where no instance holds an execution, the pass has no memoryless
/// attachment whose image is lazily allocated, or the step would refuse to back one (its sample
/// planes, a color resolve drawn after the pass, a depth/stencil resolve:
/// `Rendered::back_memoryless`).
fn backed(
    maker: &CommandBuffer,
    pass: &DeferredPass,
    queue: u64,
    held: &mut (Vec<RenderPassObject>, Vec<FramebufferObject>),
) -> Result<Option<Backed>, Error> {
    let device = maker.device();
    if !pass.instances.iter().any(executes) {
        return Ok(None);
    }
    let mut backings = Vec::new();
    for (index, color) in pass.layout.colors.iter().enumerate() {
        let Some(color) = color.as_ref().filter(|color| color.stand_in.is_none()) else {
            continue;
        };
        let Some(made) = backing(maker, Some(index), &color.target, None, queue)? else {
            continue;
        };
        let drawn_after = color.resolve.as_ref().is_some_and(|r| r.shader.is_some());
        if color.planes.is_some() || drawn_after {
            return Ok(None);
        }
        backings.push(made);
    }
    if let Some(ds) = pass.layout.depth_stencil.as_ref() {
        if ds.stand_in.is_none() {
            if let Some(made) = backing(maker, None, &ds.target, None, queue)? {
                if ds.resolve.is_some() {
                    return Ok(None);
                }
                backings.push(made);
            }
        }
    }
    if backings.is_empty() {
        return Ok(None);
    }
    let mut layout = pass.layout.clone();
    back(&mut layout, &backings)?;
    let mut bare = pass.bare.clone();
    if let Some(bare) = &mut bare {
        back(bare, &backings)?;
    }
    let mut instances = Vec::with_capacity(pass.instances.len());
    for instance in &pass.instances {
        if !executes(instance) {
            instances.push(None);
            continue;
        }
        let own = crate::render::instance_layout(&layout, bare.as_ref(), &instance.shape);
        let (runs, objects) = run_passes(device, own, instance)?;
        // One framebuffer serves every run: their passes differ only in their operations.
        let framebuffer = render_target::framebuffer(
            device,
            &objects[0],
            own,
            &instance.shape,
            ViewSetup::default(),
            Feedback::NONE,
        )?;
        instances.push(Some((runs, framebuffer.raw)));
        held.0.extend(objects);
        held.1.push(framebuffer);
    }
    Ok(Some(Backed {
        backings,
        instances,
    }))
}
