//! `generate_mipmaps` of a format the device's blits cannot filter (`mipmap_plan.rs`'s
//! [`MipmapPlan::Filter`](super::mipmap_plan::MipmapPlan::Filter): an extended-range format, an
//! 8-bit sRGB texture held as its UNORM twin, a format without blits or linear filtering), by the
//! compute filter of `bind::mipmap_filter`: each level a bilinear (trilinear in 3D) filter of the
//! one above at the scaled texel centres, in `float`, in a 2D, 2D array, cube, cube array or 3D
//! texture whatever its levels' sizes. It is not Apple's datapath, and it is announced as the
//! blit chain is (`device-filtered-mipmaps`).
//!
//! **The command.** One blit command, between barriers from the commands before it and to the
//! ones after it ([`BlitCommandEncoder::command`]), over a scratch block the device keeps between
//! calls (`mipmap_scratch.rs`; `bind::MipmapFilterChain` lays it out): the first level of every
//! layer copied in, unless it is read in place (below), every later level zeroed where its
//! texels are narrower than a word (each dispatch ORs them in; a wider texel's writes every
//! word), each level's tap words uploaded, one dispatch per later level of its kernel
//! (`owned_kernel.rs`: one per `MipmapFilterKernel`), each reading the level the one before
//! wrote, and every later level copied back into the texture.
//!
//! **The first level read in place**: the first dispatch fetches its taps from a sampled view of
//! the first level (a 2D array of every layer, 3D for a 3D texture) and recovers each texel's bytes
//! (`bind::MipmapFilterChain::plan`'s `fetch`), where the image has sampled usage, the view's
//! format (the image's own, or the UNORM twin of an 8-bit sRGB one) is one the image views as and
//! samples, and the fetch is exact: the extended-range formats' integer carriers on every device,
//! the normalized and 16-bit float formats on a driver measured to return values whose recovery
//! reads every code back (`DriverFact::SampledFetchRecoversCodes`). Vulkan states no precision for
//! those conversions, so on any other driver the level is copied in. The
//! command buffer holds the block's lease and the dispatches' descriptor sets until it completes.
//! Where the device's limits do not fit the chain (a level past `maxStorageBufferRange`, the
//! scratch past `max_buffer_length`, a dispatch past `maxComputeWorkGroupCount`) or the scratch's
//! memory runs out, nothing is recorded and the call is refused.
//!
//! **A tiled texture's take** (`Buffer::new_tiled_texture`, `shadow.rs`'s **Copies taken at
//! use**) records the same run ([`FilterRun`]) of the texture's levels right after its copy into
//! the first level, inside the copy, its first level copied in.

use super::binding_view::{views_as, VkView};
use super::blit::BlitCommandEncoder;
use super::error::Error;
use super::image_plan::ImageFacts;
use super::image_plan::ImagePlan;
use super::mipmap_scratch::Lease;
use super::owned_kernel::{OwnedKernel, OwnedSet};
use super::owned_kernel::{OwnedModule, OwnedSets};
use super::sparse::emulated::Written;
use super::texture::{texture_aspects, Texture};
use super::tracker::{Access, Barrier};
use super::view_plan::ViewRecord;
use super::Device;
use crate::device::functions::Vk;
use ash::vk;
use metal2vulkan::bind::MipmapFilterChain;
use metal2vulkan::device::DriverFact;
use metal2vulkan::metal::TextureType;
use metal2vulkan::vk::{
    ComponentMapping, Format, FormatFeatureFlags2, ImageAspectFlags, ImageUsageFlags, ImageViewType,
};

/// The most bytes one `vkCmdUpdateBuffer` writes.
const UPDATE_BYTES: usize = 65536;

impl BlitCommandEncoder {
    /// Record `texture`'s mipmaps by the compute filter (the module docs): `false`, recording
    /// nothing, where the device's limits do not fit its chain. The caller planned it
    /// (`mipmap_plan::plan_mipmaps`: Metal's rules, the format, the image's own format and its
    /// transfers).
    pub(super) fn filter_mipmaps(
        &mut self,
        texture: &Texture,
        facts: &ImageFacts<'_>,
        read: Access,
        write: Access,
    ) -> Result<bool, Error> {
        let record = texture.record();
        let plan = texture.image_plan()?;
        let device = self.core().device().clone();
        let fetch = fetch_view(plan, facts, &device);
        let Some(chain) = filter_chain(&device, record, fetch) else {
            return Ok(false);
        };
        let (base_level, base_layer, layers) =
            (record.base_level, record.base_layer, record.layer_count);
        let volume = record.texture_type == TextureType::Type3D;
        // The first level's sampled view, where the chain reads it in place.
        let first_view = match chain.fetch {
            Some(format) => Some(texture.image_view(&VkView {
                view_type: if volume {
                    ImageViewType::TYPE_3D
                } else {
                    ImageViewType::TYPE_2D_ARRAY
                },
                format,
                components: ComponentMapping::IDENTITY,
                aspect: ImageAspectFlags::COLOR,
                base_level,
                level_count: 1,
                base_layer,
                layer_count: layers,
                usage: ImageUsageFlags::SAMPLED,
            })?),
            None => None,
        };
        let aspects = texture_aspects(plan);
        let Some(run) = FilterRun::new(&device, &chain, record, aspects, first_view)? else {
            return Ok(false);
        };
        let first = base_level..base_level + 1;
        let later = base_level + 1..base_level + record.level_count;
        let all_layers = base_layer..base_layer + layers;
        // The first level is read by the copy in, or by the first dispatch's fetches.
        let first_read = match first_view {
            Some(_) => Access::new(
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_SAMPLED_READ,
            ),
            None => read,
        };
        let uses = [
            texture.image_use(first, all_layers.clone(), first_read, false)?,
            texture.image_use(later, all_layers, write, true)?,
        ];
        let image = texture.raw_image()?;
        self.command(&uses, |device, command_buffer| {
            // SAFETY: the command buffer is recording, outside a render pass; the run and the
            // texture (whose image keeps the first level's view) are retained until it completes;
            // the image is in the `GENERAL` layout with transfer usages and features (checked
            // above), and sampled usage and features where its view is bound (`fetch_view`).
            unsafe { run.record(device, command_buffer, image) };
        })?;
        if first_view.is_some() {
            device.shared().counts.mipmap_first_level_fetched();
        }
        self.core().retain(run);
        self.core().retain(texture.clone());
        let written = (1u32..)
            .zip(&chain.levels[1..])
            .map(|(level, l)| {
                let written = Written {
                    level: base_level + level,
                    layers: base_layer..base_layer + layers,
                    origin: [0; 3],
                    size: l.extent,
                };
                (aspects, written)
            })
            .collect();
        self.scrub(texture, written)?;
        device.shared().counts.mipmaps_filtered();
        Ok(true)
    }
}

/// The compute filter's chain of the levels and layers `record` presents (`bind::MipmapFilterChain`),
/// its first level read in place through a sampled view of `fetch`'s format where given and the
/// format's fetch recovers its texels, on `device`: `None` where the device's limits do not fit it
/// (a level past `maxStorageBufferRange`, a dispatch past `maxComputeWorkGroupCount`, the scratch
/// past `max_buffer_length`).
pub(super) fn filter_chain(
    device: &Device,
    record: &ViewRecord,
    fetch: Option<Format>,
) -> Option<MipmapFilterChain> {
    let volume = record.texture_type == TextureType::Type3D;
    let mut extent = record.extent();
    if !volume {
        extent[2] = 1;
    }
    let limits = &device.properties().description.limits;
    let chain = MipmapFilterChain::plan(
        record.pixel_format,
        extent,
        record.layer_count,
        record.level_count,
        volume,
        limits.min_storage_buffer_offset_alignment,
        limits.max_storage_buffer_range,
        [
            limits.max_compute_work_group_count[0],
            limits.max_compute_work_group_count[1],
        ],
        fetch,
    )?;
    (chain.bytes <= device.properties().max_buffer_length).then_some(chain)
}

/// One recording of a compute filter's chain (the module's **The command**): its kernels, its
/// lease of the device's scratch block, its descriptor sets written, and the uploads, fills,
/// dispatches and copies it records. The command buffer that records it holds it until it
/// completes.
pub(super) struct FilterRun {
    /// The device whose fills it records (`word_fill::record`).
    owner: Device,
    scratch: Lease,
    sets: OwnedSets,
    /// Each later level's kernel, in order.
    kernels: Vec<OwnedKernel>,
    /// Each upload: where in the scratch, and the bytes it writes there (a level's tap words).
    uploads: Vec<(u64, Vec<u8>)>,
    /// The scratch's ranges filled with zeros: each later level's, where its texels are
    /// narrower than a word.
    zeroed: Vec<(u64, u64)>,
    /// Each dispatch's workgroups along x and y.
    groups: Vec<[u32; 2]>,
    /// The copy of the first level into the scratch, where no dispatch reads it in place.
    copy_in: Option<vk::BufferImageCopy>,
    /// The copies of every later level out of the scratch into the image.
    copies_back: Vec<vk::BufferImageCopy>,
}

impl FilterRun {
    /// The run of `chain` ([`filter_chain`]) over the levels and layers `record` presents of an
    /// image of `aspects` on `device`: its first dispatch reads `first_view`, a view of the first
    /// level given where the chain reads it in place (`MipmapFilterChain::fetch`), else the first
    /// level is copied in. `None`, making nothing, where the device's memory has no room for the
    /// scratch.
    pub(super) fn new(
        device: &Device,
        chain: &MipmapFilterChain,
        record: &ViewRecord,
        aspects: vk::ImageAspectFlags,
        first_view: Option<vk::ImageView>,
    ) -> Result<Option<FilterRun>, Error> {
        // Each later level's kernel (levels of one kernel share it).
        let mut kernels = Vec::with_capacity(chain.levels.len() - 1);
        for level in &chain.levels[1..] {
            let Some(key) = level.kernel else {
                return Err(super::error::internal(
                    "a mipmap filter level without its kernel",
                ));
            };
            kernels.push(device.owned_kernel(OwnedModule::MipmapFilter(key))?);
        }
        // The device's scratch block, kept between calls (`mipmap_scratch.rs`).
        let Some(scratch) = device.mipmap_scratch(chain.bytes)? else {
            return Ok(None);
        };
        let (raw, at) = (scratch.raw(), 0);
        let words = |bytes: u64| bytes.next_multiple_of(4);
        let (base_level, base_layer, layers) =
            (record.base_level, record.base_layer, record.layer_count);
        let mut ranges = Vec::with_capacity(chain.levels.len() - 1);
        let mut uploads = Vec::new();
        let mut zeroed = Vec::with_capacity(chain.levels.len() - 1);
        let mut groups = Vec::with_capacity(chain.levels.len() - 1);
        for pair in chain.levels.windows(2) {
            let (above, level) = (&pair[0], &pair[1]);
            let Some((offset, params)) = &level.params else {
                return Err(super::error::internal(
                    "a mipmap filter level without its words",
                ));
            };
            let bytes: Vec<u8> = params.iter().flat_map(|w| w.to_le_bytes()).collect();
            let params_range = (raw, at + offset, bytes.len() as u64);
            let destination = (raw, at + level.offset, words(level.bytes));
            let rest = [destination, params_range];
            // The first dispatch's source: the first level's view where it is read in place.
            ranges.push(match first_view.filter(|_| ranges.is_empty()) {
                Some(view) => OwnedSet {
                    images: vec![view],
                    storage_images: Vec::new(),
                    buffers: rest.to_vec(),
                },
                None => OwnedSet {
                    images: Vec::new(),
                    storage_images: Vec::new(),
                    buffers: [(raw, at + above.offset, words(above.bytes))]
                        .into_iter()
                        .chain(rest)
                        .collect(),
                },
            });
            for (k, chunk) in bytes.chunks(UPDATE_BYTES).enumerate() {
                uploads.push((at + offset + (k * UPDATE_BYTES) as u64, chunk.to_vec()));
            }
            if chain.zeroed {
                zeroed.push((at + level.offset, words(level.bytes)));
            }
            groups.push(level.groups);
        }
        let sets = OwnedSets::write_each(device, &kernels, &ranges)?;
        let region =
            |level: u32, offset: u64, [width, height, depth]: [u32; 3]| vk::BufferImageCopy {
                buffer_offset: at + offset,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: aspects,
                    mip_level: base_level + level,
                    base_array_layer: base_layer,
                    layer_count: layers,
                },
                image_offset: vk::Offset3D::default(),
                image_extent: vk::Extent3D {
                    width,
                    height,
                    depth,
                },
            };
        let copy_in = first_view
            .is_none()
            .then(|| region(0, chain.levels[0].offset, chain.levels[0].extent));
        let copies_back = (1u32..)
            .zip(&chain.levels[1..])
            .map(|(level, l)| region(level, l.offset, l.extent))
            .collect();
        Ok(Some(FilterRun {
            owner: device.clone(),
            scratch,
            sets,
            kernels,
            uploads,
            zeroed,
            groups,
            copy_in,
            copies_back,
        }))
    }

    /// Record the run over `image`.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording, outside a render pass, after a barrier that orders the run
    /// after every earlier write of the first level; the run and the image (and the first level's
    /// view, where the run reads it) are live until it completes; the image is in the `GENERAL`
    /// layout with transfer usages and features, and sampled usage and features where its view is
    /// bound. Every copy, fill, upload (each at most 65536 bytes, a multiple of 4 at a multiple of
    /// 4) and binding lies inside the scratch (`MipmapFilterChain::plan`), and each step reads
    /// what the one before it wrote, past a barrier.
    pub(super) unsafe fn record(
        &self,
        device: &Vk,
        command_buffer: vk::CommandBuffer,
        image: vk::Image,
    ) {
        let raw = self.scratch.raw();
        // SAFETY: the caller's contract.
        unsafe {
            for (offset, bytes) in &self.uploads {
                device.cmd_update_buffer(command_buffer, raw, *offset, bytes);
            }
            for &(offset, size) in &self.zeroed {
                crate::word_fill::record(
                    &self.owner,
                    command_buffer,
                    raw,
                    offset..offset + size,
                    0,
                );
            }
            if let Some(copy_in) = self.copy_in {
                device.cmd_copy_image_to_buffer(
                    command_buffer,
                    image,
                    vk::ImageLayout::GENERAL,
                    raw,
                    &[copy_in],
                );
            }
            Barrier::everything().record(device, command_buffer);
            let mut bound = vk::Pipeline::null();
            for ((set, [x, y]), kernel) in
                self.sets.sets.iter().zip(&self.groups).zip(&self.kernels)
            {
                if kernel.pipeline != bound {
                    device.cmd_bind_pipeline(
                        command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        kernel.pipeline,
                    );
                    bound = kernel.pipeline;
                }
                device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    kernel.layout,
                    0,
                    std::slice::from_ref(set),
                    &[],
                );
                device.cmd_dispatch(command_buffer, *x, *y, 1);
                Barrier::everything().record(device, command_buffer);
            }
            device.cmd_copy_buffer_to_image(
                command_buffer,
                raw,
                image,
                vk::ImageLayout::GENERAL,
                &self.copies_back,
            );
        }
    }
}

/// The format of the sampled view through which the first level of a texture whose image is
/// `plan` is read in place, where it may be (the module docs): the image's format or the UNORM
/// twin of its 8-bit sRGB one, which the image views as and samples with sampled usage, and
/// whose fetch is exact on `device`. `None` where the level is copied in. The chain takes the
/// view only for a format whose fetch recovers its texels (`MipmapFilterChain::plan`).
fn fetch_view(plan: &ImagePlan, facts: &ImageFacts<'_>, device: &super::Device) -> Option<Format> {
    let view = MipmapFilterChain::fetch_format(plan.format)?;
    let sampled = plan.usage.contains(ImageUsageFlags::SAMPLED)
        && views_as(plan, view, false)
        && facts
            .optimal(view)
            .contains(FormatFeatureFlags2::SAMPLED_IMAGE);
    let exact = !MipmapFilterChain::fetch_converts(view)
        || device
            .properties()
            .driver_facts
            .contains(&DriverFact::SampledFetchRecoversCodes);
    (sampled && exact).then_some(view)
}
