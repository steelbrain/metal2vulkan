//! `generate_mipmaps` of the integer formats, as plain Metal makes them: each level's texels picked
//! from the level above, which copies give exactly on every device ([`PickPlan`] lays the copies
//! out).
//!
//! **The command.** One blit command, between barriers from the commands before it and to the
//! ones after it ([`BlitCommandEncoder::command`]), over a scratch buffer of the runtime's own:
//! the view's first level, over its layers, copied into the scratch, then every later level copied
//! out of it, a column (or a row's texel) a region. The scratch is retained until the command
//! buffer completes. Where the scratch would pass the device's `max_buffer_length`, the call is
//! refused (`NotImplemented(MipmapGeneration)`): a blit's nearest filter puts its sample on the
//! edge between two texels, which the device's precision decides.
//!
//! **A tiled texture's take** (`Buffer::new_tiled_texture`, `shadow.rs`'s **Copies taken at
//! use**) records the same run ([`PickRun`]) of the texture's levels right after its copy into
//! the first level, inside the copy.

use super::blit::BlitCommandEncoder;
use super::buffer::Buffer;
use super::error::Error;
use super::mipmap_plan::PickPlan;
use super::sparse::emulated::Written;
use super::texture::{texture_aspects, Texture};
use super::tracker::{Access, Barrier};
use super::values::ResourceOptions;
use super::view_plan::ViewRecord;
use super::Device;
use crate::device::functions::Vk;
use ash::vk;
use metal2vulkan::error::{Feature, Unsupported};
use metal2vulkan::metal::TextureType;

impl BlitCommandEncoder {
    /// Record `texture`'s mipmaps as `plan`'s picks. The caller checked Metal's rules, and
    /// `plan_mipmaps` the image's transfers and texel.
    pub(super) fn pick_mipmaps(
        &mut self,
        texture: &Texture,
        plan: &PickPlan,
        read: Access,
        write: Access,
    ) -> Result<(), Error> {
        let record = texture.record();
        let device = self.core().device().clone();
        let aspects = texture_aspects(texture.image_plan()?);
        let run = PickRun::new(&device, plan, record, aspects)?;
        let (base_level, base_layer, layers) =
            (record.base_level, record.base_layer, record.layer_count);
        let all_layers = base_layer..base_layer + layers;
        let uses = [
            texture.image_use(base_level..base_level + 1, all_layers.clone(), read, false)?,
            texture.image_use(
                base_level + 1..base_level + record.level_count,
                all_layers.clone(),
                write,
                true,
            )?,
        ];
        let image = texture.raw_image()?;
        self.command(&uses, |device, command_buffer| {
            // SAFETY: the command buffer is recording, outside a render pass; the run and the
            // texture are retained until it completes; the image is in the `GENERAL` layout with
            // transfer usages and features (`plan_mipmaps` checked them).
            unsafe { run.record(device, command_buffer, image) };
        })?;
        self.core().retain(run);
        self.core().retain(texture.clone());
        let first = first_extent(record);
        let written = (1..record.level_count)
            .map(|n| {
                let written = Written {
                    level: base_level + n,
                    layers: all_layers.clone(),
                    origin: [0; 3],
                    size: first.map(|axis| (axis >> n).max(1)),
                };
                (aspects, written)
            })
            .collect();
        self.scrub(texture, written)
    }
}

/// The first level's extent of the levels `record` presents: its depth 1 but a 3D texture's.
fn first_extent(record: &ViewRecord) -> [u32; 3] {
    let mut first = record.extent();
    if record.texture_type != TextureType::Type3D {
        first[2] = 1;
    }
    first
}

/// Whether `plan`'s scratch fits `device`'s `max_buffer_length`, which [`PickRun::new`] refuses
/// past.
pub(super) fn picks_fit(device: &Device, plan: &PickPlan) -> bool {
    plan.bytes <= device.properties().max_buffer_length
}

/// One recording of a [`PickPlan`] (the module's **The command**): its scratch buffer, the copy of
/// the first level into it and the copies of the later levels' columns out of it. The command
/// buffer that records it holds it until it completes.
pub(super) struct PickRun {
    scratch: Buffer,
    copy_in: vk::BufferImageCopy,
    picks: Vec<vk::BufferImageCopy>,
}

impl PickRun {
    /// The run of `plan`'s picks over the levels and layers `record` presents of an image of
    /// `aspects` on `device`. Refused where the scratch would pass the device's
    /// `max_buffer_length` (`NotImplemented(MipmapGeneration)`, [`picks_fit`]), or as a buffer of
    /// the runtime's own is.
    pub(super) fn new(
        device: &Device,
        plan: &PickPlan,
        record: &ViewRecord,
        aspects: vk::ImageAspectFlags,
    ) -> Result<PickRun, Error> {
        if !picks_fit(device, plan) {
            return Err(Error::Unsupported(Unsupported::not_implemented(
                Feature::MipmapGeneration,
            )));
        }
        let scratch = device.own_buffer(plan.bytes, ResourceOptions::private())?;
        let at = scratch.at(0).1;
        let (base_layer, layers) = (record.base_layer, record.layer_count);
        let subresource = |level: u32| vk::ImageSubresourceLayers {
            aspect_mask: aspects,
            mip_level: level,
            base_array_layer: base_layer,
            layer_count: layers,
        };
        let extent = |[width, height, depth]: [u32; 3]| vk::Extent3D {
            width,
            height,
            depth,
        };
        let copy_in = vk::BufferImageCopy {
            buffer_offset: at,
            buffer_row_length: 0,
            buffer_image_height: plan.image_height,
            image_subresource: subresource(record.base_level),
            image_offset: vk::Offset3D::default(),
            image_extent: extent(first_extent(record)),
        };
        let picks = plan
            .picks
            .iter()
            .map(|pick| vk::BufferImageCopy {
                buffer_offset: at + pick.buffer_offset,
                buffer_row_length: pick.row_length,
                buffer_image_height: pick.image_height,
                image_subresource: subresource(pick.level),
                image_offset: vk::Offset3D {
                    x: pick.origin[0] as i32,
                    y: pick.origin[1] as i32,
                    z: pick.origin[2] as i32,
                },
                image_extent: extent(pick.extent),
            })
            .collect();
        Ok(PickRun {
            scratch,
            copy_in,
            picks,
        })
    }

    /// Record the run over `image`.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording, outside a render pass, after a barrier that orders the run
    /// after every earlier write of the first level; the run and the image are live until it
    /// completes; the image is in the `GENERAL` layout with transfer usages and features. Every
    /// region lies inside the scratch and its level (`plan_picks`), and the picks read what the
    /// copy in wrote, past a barrier.
    pub(super) unsafe fn record(
        &self,
        device: &Vk,
        command_buffer: vk::CommandBuffer,
        image: vk::Image,
    ) {
        let raw = self.scratch.at(0).0;
        // SAFETY: the caller's contract.
        unsafe {
            device.cmd_copy_image_to_buffer(
                command_buffer,
                image,
                vk::ImageLayout::GENERAL,
                raw,
                &[self.copy_in],
            );
            Barrier::everything().record(device, command_buffer);
            if !self.picks.is_empty() {
                device.cmd_copy_buffer_to_image(
                    command_buffer,
                    raw,
                    image,
                    vk::ImageLayout::GENERAL,
                    &self.picks,
                );
            }
        }
    }
}
