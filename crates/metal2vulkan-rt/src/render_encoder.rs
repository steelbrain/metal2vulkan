//! Render command encoders: one dynamic rendering a Metal render encoder, begun as the encoder is
//! made and ended as it ends.
//!
//! The pass orders itself as a blit's copy does. Its first barrier runs it after every earlier
//! command, makes their writes visible to it, and moves each attachment whose home is
//! shader-read-only to the attachment layout; an attachment whose home is general or the
//! attachment layout renders in it. It pays the barrier an earlier encoder owed, and the hazard
//! sets start again. Its second, after `vkCmdEndRendering`, moves those attachments home and makes
//! the pass's writes visible to every later command. A pass is two barriers, whatever it renders.

use crate::command::{Building, CommandBuffer};
use crate::counters::Metric;
use crate::error::Error;
use crate::hazard::Access;
use crate::pass::{Pass, RenderPassDescriptor};
use crate::record::{memory_barrier, Reach};
use ash::vk;
use std::sync::Arc;

/// `MTLRenderCommandEncoder`: one dynamic rendering into its pass's attachments, begun as the
/// encoder is made, with each attachment's load action, and ended by
/// [`end_encoding`](Self::end_encoding), with each one's store action.
///
/// It orders itself against every command before and after it with a barrier of its own on each
/// side, as a blit's copy does: it needs no hazard tracking of its own and leaves none for the
/// encoders after it.
///
/// Not built yet: every call but [`end_encoding`](Self::end_encoding). A pass renders its load
/// and store actions alone.
///
/// Owned: one thread encodes at a time. Dropping it ends it, as
/// [`end_encoding`](Self::end_encoding) does.
pub struct RenderCommandEncoder {
    command_buffer: CommandBuffer,
    /// What the command buffer records, held while the encoder is open.
    building: Option<Building>,
    /// The checked pass, for its second barrier.
    pass: Pass,
    /// The command buffer the pass records into.
    command: vk::CommandBuffer,
}

impl CommandBuffer {
    /// `renderCommandEncoderWithDescriptor:`: begins `descriptor`'s pass, one
    /// `vkCmdBeginRendering` between the pass's first barrier and its commands. The command
    /// buffer keeps each attachment's texture until it retires.
    ///
    /// Err: [`Refusal::Committed`](crate::Refusal::Committed);
    /// [`Refusal::EncoderOpen`](crate::Refusal::EncoderOpen) while another encoder of the command
    /// buffer is open; [`Refusal::OtherDevice`](crate::Refusal::OtherDevice) for another device's
    /// texture; [`Refusal::AttachmentUsage`](crate::Refusal::AttachmentUsage);
    /// [`Refusal::AttachmentPlace`](crate::Refusal::AttachmentPlace);
    /// [`Refusal::AttachmentFormat`](crate::Refusal::AttachmentFormat);
    /// [`Gap::RenderPass`](crate::Gap::RenderPass);
    /// [`Limit::ColorAttachments`](crate::Limit::ColorAttachments); [`Error::Vulkan`] where an
    /// attachment's view cannot be made or the command buffer cannot record. A refused
    /// descriptor leaves the command buffer as it was, open for another encoder.
    pub fn render_command_encoder(
        &self,
        descriptor: &RenderPassDescriptor,
    ) -> Result<RenderCommandEncoder, Error> {
        let mut building = self.open_encoder()?;
        match begin(&mut building, descriptor) {
            Ok((pass, command)) => Ok(RenderCommandEncoder {
                command_buffer: self.clone(),
                building: Some(building),
                pass,
                command,
            }),
            Err(error) => {
                self.close_encoder(building);
                Err(error)
            }
        }
    }
}

/// Checks `descriptor` and records its pass's first barrier and its `vkCmdBeginRendering`.
fn begin(
    building: &mut Building,
    descriptor: &RenderPassDescriptor,
) -> Result<(Pass, vk::CommandBuffer), Error> {
    let core = Arc::clone(building.device.core());
    let pass = Pass::check(&core, descriptor)?;
    let overs: Vec<(u32, Access)> = (pass.targets())
        .map(|t| (t.texture.hazard_id(), Access::ReadWrite))
        .collect();
    for t in pass.targets() {
        building.keep_texture(&t.texture);
        building.wrote_texture_managed(&t.texture);
    }
    building.overs_before(&overs, false)?;
    let command = building.ordered_command()?;
    // Marked after `ordered_command`, as a blit's copy marks it: the host reads a texture over a
    // Shared buffer's bytes in the buffer.
    if pass.targets().any(|t| t.texture.host_reads()) {
        building.wrote_host();
    }
    building.overs_after(&overs, false);
    core.counters.count(Metric::Passes);
    let earlier = [memory_barrier(Reach::Commands)];
    let away: Vec<vk::ImageMemoryBarrier2> = (pass.targets())
        .filter(|t| t.moves())
        .map(|t| t.barrier(t.attachment.home, t.layout))
        .collect();
    let colors: Vec<vk::RenderingAttachmentInfo> = (pass.colors.iter())
        .map(|c| c.as_ref().map_or_else(Default::default, |t| t.info()))
        .collect();
    let stencil = pass.stencil.as_ref().map(|t| t.info());
    let [width, height] = pass.area;
    let mut info = vk::RenderingInfo::default()
        .render_area(vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D { width, height },
        })
        .layer_count(1)
        .color_attachments(&colors);
    if let Some(stencil) = &stencil {
        info = info.stencil_attachment(stencil);
    }
    let device = &core.device;
    // SAFETY: the command buffer records outside any render pass and keeps each attachment's
    // texture until it retires; each attachment is a view of one level and layer of its image,
    // in its home layout between commands, which the barrier moves to the layout the pass names;
    // the render area lies inside every attachment.
    unsafe {
        device.cmd_pipeline_barrier2(
            command,
            &vk::DependencyInfo::default()
                .memory_barriers(&earlier)
                .image_memory_barriers(&away),
        );
        device.cmd_begin_rendering(command, &info);
    }
    core.counters.count(Metric::Barriers);
    Ok((pass, command))
}

impl RenderCommandEncoder {
    /// `endEncoding`: ends the pass, one `vkCmdEndRendering` and the pass's second barrier.
    pub fn end_encoding(mut self) {
        self.end();
    }

    fn end(&mut self) {
        let Some(building) = self.building.take() else {
            return;
        };
        let core = Arc::clone(building.device.core());
        let later = [memory_barrier(Reach::Commands)];
        let back: Vec<vk::ImageMemoryBarrier2> = (self.pass.targets())
            .filter(|t| t.moves())
            .map(|t| t.barrier(t.layout, t.attachment.home))
            .collect();
        let device = &core.device;
        // SAFETY: the command buffer is inside the pass `begin` recorded, which nothing else
        // records into while the encoder holds it; each attachment the barrier moves is in the
        // layout the pass rendered it in.
        unsafe {
            device.cmd_end_rendering(self.command);
            device.cmd_pipeline_barrier2(
                self.command,
                &vk::DependencyInfo::default()
                    .memory_barriers(&later)
                    .image_memory_barriers(&back),
            );
        }
        core.counters.count(Metric::Barriers);
        self.command_buffer.close_encoder(building);
    }
}

impl Drop for RenderCommandEncoder {
    fn drop(&mut self) {
        self.end();
    }
}
