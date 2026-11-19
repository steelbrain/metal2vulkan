//! Exported surfaces: a picture's planes as textures whose memory the consumer shares with
//! another API through one handle ([`Device::new_exported_surface`]), such as a video decoder
//! that writes decoded pictures into it, or an encoder that reads the pictures rt renders. Metal
//! has no counterpart, as it has none of exports: the consumer's own path.
//!
//! The surface takes one of two forms, decided at its make by the device's queries alone:
//!
//! - **In place.** Each plane's texture is a linear image in the shared memory, laid where the
//!   answer says. A dma-buf's planes are images of the DRM format modifier
//!   `DRM_FORMAT_MOD_LINEAR` (`VK_EXT_image_drm_format_modifier`), each on its own dedicated
//!   import of the surface's memory, which is freed once the imports and the handle exist, as the
//!   imports and the handle hold its payload. An opaque handle's planes are `LINEAR` images bound
//!   in one exportable allocation that holds no buffer.
//! - **The copy form.** The surface is an exported block ([`Device::new_exported_block`]) and
//!   each plane's texture is a texture over a buffer in it ([`Buffer::new_texture`]), an optimal
//!   image the submissions copy the bytes into and back (`crate::over`). The answer names why
//!   ([`SurfaceCopy`]).
//!
//! Both forms lay the planes alike: one pitch for every plane, and each plane at an exact start.

use crate::block::Block;
use crate::buffer::Buffer;
use crate::counters::{Gauge, Metric};
use crate::device::{Core, Device};
use crate::error::{Error, Gap, Limit, Refusal};
use crate::export::{Export, ExportHandle};
use crate::memory;
use crate::submit::Held;
use crate::texture::{Backing, SurfaceImage, Texture, TextureDescriptor};
use ash::vk;
use std::sync::Arc;

/// The most planes a surface has: a DRM format's memory planes.
const MAX_PLANES: usize = 4;

/// The image usage a plane's image in place takes in every case: sampled, rendered, copied from
/// and into. Storage joins it where the query offers it.
const IN_PLACE_USAGE: vk::ImageUsageFlags = vk::ImageUsageFlags::from_raw(
    vk::ImageUsageFlags::SAMPLED.as_raw()
        | vk::ImageUsageFlags::COLOR_ATTACHMENT.as_raw()
        | vk::ImageUsageFlags::TRANSFER_SRC.as_raw()
        | vk::ImageUsageFlags::TRANSFER_DST.as_raw(),
);

/// `DRM_FORMAT_MOD_LINEAR`.
const MOD_LINEAR: u64 = 0;

/// No Metal counterpart: what [`Device::new_exported_surface`] makes.
#[derive(Clone, Copy, Debug)]
pub struct SurfaceSpec<'a> {
    /// The handle the surface's memory is shared through.
    pub handle: ExportHandle,
    /// What the planes' one pitch is a multiple of: the other API's pitch rule (256 for a VA-API
    /// import, 4 for an NVENC input, for instance). At least 1.
    pub pitch_multiple: u64,
    /// What each plane's start after the first is a multiple of: the other API's rule. At least 1.
    pub start_multiple: u64,
    /// The planes, in their order in memory: at least one, at most four.
    pub planes: &'a [PlaneSpec],
}

/// No Metal counterpart: one plane of a [`SurfaceSpec`].
#[derive(Clone, Copy, Debug)]
pub struct PlaneSpec {
    /// The plane's texture, as [`Buffer::new_texture`] takes it: 2D, of one level, sample and
    /// slice, of a color format of one texel an element.
    pub descriptor: TextureDescriptor,
    /// The texture's index, as for [`Device::new_texture`].
    pub index: Option<u32>,
    /// Rows of room below the plane's own rows, before the next plane's start.
    pub rows_below: u32,
}

/// No Metal counterpart: where a plane lies in the surface's memory, as the other API imports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaneLayout {
    /// The plane's first byte's offset in the memory.
    pub offset: u64,
    /// The bytes from a row's first byte to the next row's: the surface's one pitch.
    pub row_pitch: u64,
}

/// No Metal counterpart: the form a surface took ([`Device::new_exported_surface`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceForm {
    /// Each plane's texture is a linear image in the shared memory.
    InPlace,
    /// Each plane's texture is a texture over a buffer in an exported block, copied, and why.
    Copy(SurfaceCopy),
}

/// No Metal counterpart: why a surface took the copy form, decided by the device's queries and
/// the images it made.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceCopy {
    /// A dma-buf on a device without `VK_EXT_image_drm_format_modifier`.
    NoModifier,
    /// The plane's format in the handle's tiling (`DRM_FORMAT_MOD_LINEAR` for a dma-buf, `LINEAR`
    /// for an opaque handle) lacks sampling, rendering or either copy, or is not exportable and
    /// importable as the handle (`vkGetPhysicalDeviceImageFormatProperties2`).
    Tiling {
        /// The plane's index.
        plane: usize,
    },
    /// The plane's image is exported only in an allocation of its own
    /// (`VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT`), or, bound in the opaque handle's one
    /// allocation, requires one of its own.
    Dedicated {
        /// The plane's index.
        plane: usize,
    },
    /// The plane's image takes no row pitch that is the surface's one pitch: the driver's fixed
    /// pitch of an opaque handle's image differs from the first plane's or is not a multiple of
    /// the consumer's, or the driver refuses the pitch stated for a dma-buf's image.
    Pitch {
        /// The plane's index.
        plane: usize,
    },
    /// The plane's image does not lie at its exact start: off its memory alignment or inside the
    /// plane before it in the opaque handle's allocation, or an offset the driver refuses for a
    /// dma-buf's image.
    Start {
        /// The plane's index.
        plane: usize,
    },
    /// No memory type holds every plane's image.
    MemoryType,
}

/// No Metal counterpart: what [`Device::new_exported_surface`] gives the consumer to pass to the
/// other API.
#[non_exhaustive]
#[derive(Debug)]
pub struct SurfaceExport {
    /// The memory's handle, its size and its type, as for an exported block.
    pub export: Export,
    /// Each plane's offset and pitch, in the spec's order. Plane 0 is at offset 0.
    pub planes: Vec<PlaneLayout>,
    /// The form the surface took.
    pub form: SurfaceForm,
}

/// No Metal counterpart: a picture's planes as textures over memory shared with another API
/// ([`Device::new_exported_surface`]).
pub struct Surface {
    device: Device,
    handle: ExportHandle,
    planes: Vec<Texture>,
    /// The copy form's block and its buffer; `None` in place.
    copy: Option<(Block, Buffer)>,
}

impl std::fmt::Debug for Surface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Surface")
            .field("handle", &self.handle)
            .field("planes", &self.planes.len())
            .field("in_place", &self.copy.is_none())
            .finish()
    }
}

/// A plane's checked texture: its image format, the image usage its uses ask, the formats its
/// views may take and its texel's bytes.
pub(crate) struct PlanePlan {
    pub(crate) format: vk::Format,
    pub(crate) usage: vk::ImageUsageFlags,
    pub(crate) view_formats: Vec<vk::Format>,
    pub(crate) texel: u64,
}

/// The greatest common divisor.
fn gcd(a: u64, b: u64) -> u64 {
    match b {
        0 => a,
        _ => gcd(b, a % b),
    }
}

/// The least common multiple of `values`, none of them 0; `None` past `u64`.
fn lcm(values: impl IntoIterator<Item = u64>) -> Option<u64> {
    values
        .into_iter()
        .try_fold(1u64, |all, v| (all / gcd(all, v)).checked_mul(v))
}

/// The planes laid at `pitch`: each start, and the end of the last plane's rows and room. Plane 0
/// at 0; each next one at the first multiple of `start_multiple` and of its texel at or past the
/// previous plane's start plus its rows and room.
fn lay(spec: &SurfaceSpec<'_>, plans: &[PlanePlan], pitch: u64) -> Option<(Vec<u64>, u64)> {
    let mut starts = Vec::with_capacity(spec.planes.len());
    let mut end = 0u64;
    for (plane, plan) in spec.planes.iter().zip(plans) {
        let start = match starts.is_empty() {
            true => 0,
            false => end.checked_next_multiple_of(lcm([spec.start_multiple, plan.texel])?)?,
        };
        let rows = u64::from(plane.descriptor.height).checked_add(u64::from(plane.rows_below))?;
        end = start.checked_add(pitch.checked_mul(rows)?)?;
        starts.push(start);
    }
    Some((starts, end))
}

/// The device's image format queries for a plane in place.
struct Query<'a> {
    instance: &'a ash::Instance,
    physical: vk::PhysicalDevice,
    handle: ExportHandle,
}

impl Query<'_> {
    /// The tiling the handle's planes take in place.
    fn tiling(&self) -> vk::ImageTiling {
        match self.handle {
            ExportHandle::DmaBuf => vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT,
            _ => vk::ImageTiling::LINEAR,
        }
    }

    /// The external memory features of an image of `format`, `usage` and view-format `list`
    /// (none where it has one entry or none), in the handle's tiling; `None` where the device makes
    /// no such image.
    fn external(
        &self,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        list: &[vk::Format],
    ) -> Option<vk::ExternalMemoryFeatureFlags> {
        let mut external =
            vk::PhysicalDeviceExternalImageFormatInfo::default().handle_type(self.handle.vk());
        let mut formats = vk::ImageFormatListCreateInfo::default().view_formats(list);
        let mut modifier = vk::PhysicalDeviceImageDrmFormatModifierInfoEXT::default()
            .drm_format_modifier(MOD_LINEAR)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let mut info = vk::PhysicalDeviceImageFormatInfo2::default()
            .format(format)
            .ty(vk::ImageType::TYPE_2D)
            .tiling(self.tiling())
            .usage(usage)
            .push_next(&mut external);
        if list.len() > 1 {
            info = info
                .flags(vk::ImageCreateFlags::MUTABLE_FORMAT)
                .push_next(&mut formats);
        }
        if self.handle == ExportHandle::DmaBuf {
            info = info.push_next(&mut modifier);
        }
        let mut external_out = vk::ExternalImageFormatProperties::default();
        let mut out = vk::ImageFormatProperties2::default().push_next(&mut external_out);
        // SAFETY: `physical` is `instance`'s, the handle's extensions are enabled, and a dma-buf's
        // query runs where the device enabled the modifier extension.
        let answer = unsafe {
            (self.instance).get_physical_device_image_format_properties2(
                self.physical,
                &info,
                &mut out,
            )
        };
        answer.ok().map(|()| {
            external_out
                .external_memory_properties
                .external_memory_features
        })
    }

    /// The plane's image in place: its usage, every bit of `IN_PLACE_USAGE`, storage and the
    /// plan's usage that the query offers, and its view-format list, the plan's formats the
    /// tiling samples (none where the query refuses it), and the export features; `Err` with the
    /// copy form's reason, or the gap of a usage the plan asks that the tiling lacks.
    fn plane(
        &self,
        at: usize,
        plan: &PlanePlan,
        pixel_format: crate::PixelFormat,
    ) -> Result<Result<(vk::ImageUsageFlags, Vec<vk::Format>), SurfaceCopy>, Error> {
        let wanted = IN_PLACE_USAGE | vk::ImageUsageFlags::STORAGE | plan.usage;
        let offered = (0..32)
            .map(|bit| vk::ImageUsageFlags::from_raw(1u32 << bit))
            .filter(|&bit| wanted.contains(bit) && !bit.is_empty())
            .filter(|&bit| self.external(plan.format, bit, &[]).is_some())
            .fold(vk::ImageUsageFlags::empty(), |all, bit| all | bit);
        if !offered.contains(IN_PLACE_USAGE) {
            return Ok(Err(SurfaceCopy::Tiling { plane: at }));
        }
        let missing = plan.usage & !offered;
        if !missing.is_empty() {
            return Err(Error::Unsupported(Gap::SurfaceUsage {
                pixel_format,
                missing: missing.as_raw(),
            }));
        }
        let samples = |format: vk::Format| {
            let mut props = vk::FormatProperties2::default();
            // SAFETY: `physical` is `instance`'s.
            unsafe {
                (self.instance).get_physical_device_format_properties2(
                    self.physical,
                    format,
                    &mut props,
                )
            };
            (props.format_properties.linear_tiling_features)
                .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE)
        };
        let mut list: Vec<vk::Format> = (plan.view_formats.iter().copied())
            .filter(|&format| samples(format))
            .collect();
        if list.len() < 2 {
            list.clear();
        }
        let features = match self.external(plan.format, offered, &list) {
            Some(features) => features,
            None => {
                list.clear();
                match self.external(plan.format, offered, &list) {
                    Some(features) => features,
                    None => return Ok(Err(SurfaceCopy::Tiling { plane: at })),
                }
            }
        };
        let both =
            vk::ExternalMemoryFeatureFlags::EXPORTABLE | vk::ExternalMemoryFeatureFlags::IMPORTABLE;
        if !features.contains(both) {
            return Ok(Err(SurfaceCopy::Tiling { plane: at }));
        }
        if features.contains(vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY) {
            return Ok(Err(SurfaceCopy::Dedicated { plane: at }));
        }
        Ok(Ok((offered, list)))
    }
}

/// A plane image made for the surface, not yet bound.
struct Made {
    image: vk::Image,
    usage: vk::ImageUsageFlags,
    list: Vec<vk::Format>,
    needs: vk::MemoryRequirements,
    layout: vk::SubresourceLayout,
    requires: bool,
}

/// Makes a plane's image of `format`, `usage` and view-format `list`, external as `handle`, in
/// `tiling`: for a dma-buf, `DRM_FORMAT_MOD_LINEAR` with the implicit list where `at` is `None`,
/// else the explicit layout at `at` (offset, pitch). Its requirements and its subresource layout
/// come with it.
///
/// # Safety
///
/// The device makes such an image ([`Query::plane`] accepted it), its extensions enabled.
unsafe fn make_image(
    device: &ash::Device,
    descriptor: &TextureDescriptor,
    (format, usage, list): (vk::Format, vk::ImageUsageFlags, &[vk::Format]),
    handle: ExportHandle,
    at: Option<(u64, u64)>,
) -> Result<Made, vk::Result> {
    let dma = handle == ExportHandle::DmaBuf;
    let modifiers = [MOD_LINEAR];
    let mut implicit =
        vk::ImageDrmFormatModifierListCreateInfoEXT::default().drm_format_modifiers(&modifiers);
    let (offset, row_pitch) = at.unwrap_or((0, 0));
    let layouts = [vk::SubresourceLayout {
        offset,
        size: 0,
        row_pitch,
        array_pitch: 0,
        depth_pitch: 0,
    }];
    let mut explicit = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
        .drm_format_modifier(MOD_LINEAR)
        .plane_layouts(&layouts);
    let mut external = vk::ExternalMemoryImageCreateInfo::default().handle_types(handle.vk());
    let mut formats = vk::ImageFormatListCreateInfo::default().view_formats(list);
    let (tiling, aspect) = match dma {
        true => (
            vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT,
            vk::ImageAspectFlags::MEMORY_PLANE_0_EXT,
        ),
        false => (vk::ImageTiling::LINEAR, vk::ImageAspectFlags::COLOR),
    };
    let mut info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width: descriptor.width,
            height: descriptor.height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(tiling)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        // An external image is made undefined (`VUID-VkImageCreateInfo-pNext-01443`).
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .push_next(&mut external);
    if !list.is_empty() {
        info = info
            .flags(vk::ImageCreateFlags::MUTABLE_FORMAT)
            .push_next(&mut formats);
    }
    if dma {
        info = match at {
            Some(_) => info.push_next(&mut explicit),
            None => info.push_next(&mut implicit),
        };
    }
    // SAFETY: the caller's contract.
    let image = unsafe { device.create_image(&info, None) }?;
    let mut dedicated = vk::MemoryDedicatedRequirements::default();
    let mut needs = vk::MemoryRequirements2::default().push_next(&mut dedicated);
    // SAFETY: the image is the device's.
    unsafe {
        device.get_image_memory_requirements2(
            &vk::ImageMemoryRequirementsInfo2::default().image(image),
            &mut needs,
        )
    };
    let sub = vk::ImageSubresource {
        aspect_mask: aspect,
        mip_level: 0,
        array_layer: 0,
    };
    // SAFETY: the image is linear, or of the modifier with one memory plane.
    let layout = unsafe { device.get_image_subresource_layout(image, sub) };
    Ok(Made {
        image,
        usage,
        list: list.to_vec(),
        needs: needs.memory_requirements,
        layout,
        requires: dedicated.requires_dedicated_allocation == vk::TRUE,
    })
}

/// The memory type every plane's image takes: the Private type where they all take it, else the
/// lowest device-local one, else the lowest; `None` where none.
fn common_type(core: &Core, made: &[Made]) -> Option<u32> {
    let bits = (made.iter()).fold(u32::MAX, |all, m| all & m.needs.memory_type_bits);
    let private = core.memory_type(crate::MemoryClass::Private);
    match (bits, bits & core.device_local) {
        (0, _) => None,
        (bits, _) if bits & (1 << private) != 0 => Some(private),
        (_, 0) => Some(bits.trailing_zeros()),
        (_, local) => Some(local.trailing_zeros()),
    }
}

/// What the in-place forms make: each plane's image, bound, and the export.
struct InPlace {
    images: Vec<SurfaceImage>,
    export: Export,
    planes: Vec<PlaneLayout>,
}

/// Destroys images not yet bound or wrapped.
fn destroy(device: &ash::Device, made: Vec<Made>) {
    for m in made {
        // SAFETY: nothing uses the image, bound to nothing that outlives it.
        unsafe { device.destroy_image(m.image, None) };
    }
}

impl Device {
    /// No Metal counterpart: a surface of `spec`'s planes, whose memory the consumer shares with
    /// another API through `spec.handle`: a video decoder that writes pictures into it, or an
    /// encoder that reads the pictures rt renders into it. The answer is the surface, whose
    /// [`planes`](Surface::planes) are textures as the spec's descriptors make them, and the
    /// [`SurfaceExport`]: the memory's handle, size and type, each plane's offset and pitch, and
    /// the form the surface took. The memory's bytes are undefined, as a taken block's are.
    ///
    /// **Layout.** The planes share one pitch, a multiple of `spec.pitch_multiple` and of every
    /// plane's texel. Plane 0 is at offset 0; each next plane starts at the first multiple of
    /// `spec.start_multiple` and of its texel at or past the previous plane's start plus its
    /// rows and its `rows_below` at the pitch, so a start multiple of 1 and no rows below put a
    /// plane directly below the one before. The layout is linear (`DRM_FORMAT_MOD_LINEAR` for a
    /// dma-buf's import).
    ///
    /// **Form.** In place where the device's queries and the images made hold the layout exactly
    /// (the crate's guide says what each handle asks); else the copy form, whose reason the
    /// answer names, laid alike at the tightest pitch the multiples allow. Decided at the make:
    /// each plane's image in place moves to the general layout in a submission the call waits
    /// for, before the handle leaves. Each form counts in
    /// [`Metrics::surfaces_in_place`](crate::Metrics::surfaces_in_place) or
    /// [`Metrics::surfaces_copied`](crate::Metrics::surfaces_copied).
    ///
    /// **Views.** A plane's image in place lists, for views of another format, the formats of the
    /// texture's own list that its tiling samples, and none where the device refuses that list:
    /// a view of another format the image does not list is [`Gap::TextureView`] of
    /// [`TextureViewKind::SurfacePixelFormat`](crate::TextureViewKind::SurfacePixelFormat).
    ///
    /// Err: [`Refusal::SurfacePlanes`] for no plane, more than four, or a multiple of 0;
    /// [`Limit::NoExport`] and [`Limit::ExportType`] as for [`Device::new_exported_block`];
    /// what [`Buffer::new_texture`] refuses of a plane's descriptor and layout; [`Gap::SurfaceUsage`]
    /// for a usage a plane asks that its image in place lacks; [`Limit::BlockLen`] past the
    /// device's longest allocation; [`Limit::MemoryAllocations`]; [`Error::Vulkan`];
    /// [`Error::Quiesced`].
    pub fn new_exported_surface(
        &self,
        spec: &SurfaceSpec<'_>,
    ) -> Result<(Surface, SurfaceExport), Error> {
        let core = self.core();
        if core.closed() {
            return Err(Error::Quiesced);
        }
        let count = spec.planes.len();
        if count == 0 || count > MAX_PLANES || spec.pitch_multiple == 0 || spec.start_multiple == 0
        {
            return Err(Error::Refused(Refusal::SurfacePlanes {
                planes: count,
                pitch_multiple: spec.pitch_multiple,
                start_multiple: spec.start_multiple,
            }));
        }
        let handle = spec.handle;
        if !core.exports.offers(handle) {
            return Err(Error::Limit(Limit::NoExport { handle }));
        }
        if handle.foreign() && !core.exports.foreign {
            return Err(Error::Limit(Limit::ExportType {
                handle,
                listed: None,
            }));
        }
        let plans = (spec.planes.iter())
            .map(|plane| self.plane_plan(&plane.descriptor))
            .collect::<Result<Vec<_>, _>>()?;
        let multiple = lcm(
            std::iter::once(spec.pitch_multiple).chain(plans.iter().map(|p| p.texel))
        )
        .ok_or(Error::Refused(Refusal::SurfacePlanes {
            planes: count,
            pitch_multiple: spec.pitch_multiple,
            start_multiple: spec.start_multiple,
        }))?;
        let in_place = match handle {
            ExportHandle::DmaBuf if !core.exports.modifiers => Ok(Err(SurfaceCopy::NoModifier)),
            #[cfg(unix)]
            ExportHandle::DmaBuf => self.dma_buf_in_place(spec, &plans, multiple),
            _ => self.opaque_in_place(spec, &plans, multiple),
        }?;
        let made = match in_place {
            Ok(made) => made,
            Err(reason) => return self.copy_form(spec, &plans, multiple, reason),
        };
        let mut planes = Vec::with_capacity(count);
        let mut images = made.images.into_iter();
        for (plane, image) in spec.planes.iter().zip(images.by_ref()) {
            match self.surface_texture(&plane.descriptor, plane.index, image) {
                Ok(texture) => planes.push(texture),
                Err(error) => {
                    for image in images {
                        // SAFETY: nothing uses the image or its memory.
                        unsafe { crate::texture::discard_surface_image(core, image) };
                    }
                    return Err(error);
                }
            }
        }
        // The images move to the general layout before the handle leaves.
        let value = core.work.flush();
        core.done.wait(core, value)?;
        core.counters.count(Metric::SurfacesInPlace);
        let surface = Surface {
            device: self.clone(),
            handle,
            planes,
            copy: None,
        };
        let export = SurfaceExport {
            export: made.export,
            planes: made.planes,
            form: SurfaceForm::InPlace,
        };
        Ok((surface, export))
    }

    /// The copy form: an exported block laid at the tightest pitch the multiples allow, a buffer
    /// over it, and each plane's texture over the buffer at its start.
    fn copy_form(
        &self,
        spec: &SurfaceSpec<'_>,
        plans: &[PlanePlan],
        multiple: u64,
        reason: SurfaceCopy,
    ) -> Result<(Surface, SurfaceExport), Error> {
        let row = (spec.planes.iter().zip(plans))
            .map(|(plane, plan)| u64::from(plane.descriptor.width) * plan.texel)
            .max()
            .unwrap_or(0);
        let too_long = || Error::Limit(Limit::BlockLen { len: u64::MAX });
        let pitch = row
            .checked_next_multiple_of(multiple)
            .ok_or_else(too_long)?;
        let (starts, end) = lay(spec, plans, pitch).ok_or_else(too_long)?;
        let (block, export) = self.new_exported_block(spec.handle, end)?;
        let buffer = self.new_buffer(block.id, 0, block.len, None)?;
        let planes = (spec.planes.iter().zip(&starts))
            .map(|(plane, &start)| buffer.new_texture(&plane.descriptor, plane.index, start, pitch))
            .collect::<Result<Vec<_>, _>>()?;
        self.core().counters.count(Metric::SurfacesCopied);
        let layouts = (starts.iter())
            .map(|&offset| PlaneLayout {
                offset,
                row_pitch: pitch,
            })
            .collect();
        let surface = Surface {
            device: self.clone(),
            handle: spec.handle,
            planes,
            copy: Some((block, buffer)),
        };
        let export = SurfaceExport {
            export,
            planes: layouts,
            form: SurfaceForm::Copy(reason),
        };
        Ok((surface, export))
    }

    /// An opaque handle's planes in place: `LINEAR` images, each at its start in one exportable
    /// allocation that holds no buffer, of the driver's one pitch.
    fn opaque_in_place(
        &self,
        spec: &SurfaceSpec<'_>,
        plans: &[PlanePlan],
        multiple: u64,
    ) -> Result<Result<InPlace, SurfaceCopy>, Error> {
        let core = self.core();
        let device = &core.device;
        let query = Query {
            instance: core.instance.vk(),
            physical: core.physical,
            handle: spec.handle,
        };
        let mut made: Vec<Made> = Vec::with_capacity(plans.len());
        let copy =
            |made: Vec<Made>, reason: SurfaceCopy| -> Result<Result<InPlace, SurfaceCopy>, Error> {
                destroy(device, made);
                Ok(Err(reason))
            };
        for (at, (plane, plan)) in spec.planes.iter().zip(plans).enumerate() {
            let (usage, list) = match query.plane(at, plan, plane.descriptor.pixel_format) {
                Ok(Ok(found)) => found,
                Ok(Err(reason)) => return copy(made, reason),
                Err(error) => {
                    destroy(device, made);
                    return Err(error);
                }
            };
            // SAFETY: the query accepted the image; the handle's extensions are enabled.
            let image = unsafe {
                make_image(
                    device,
                    &plane.descriptor,
                    (plan.format, usage, &list),
                    spec.handle,
                    None,
                )
            };
            match image {
                Ok(image) => made.push(image),
                Err(result) => {
                    destroy(device, made);
                    return Err(Error::Vulkan {
                        call: "vkCreateImage",
                        result,
                    });
                }
            }
        }
        let pitch = made[0].layout.row_pitch;
        for at in 0..made.len() {
            if made[at].requires {
                return copy(made, SurfaceCopy::Dedicated { plane: at });
            }
            if made[at].layout.row_pitch != pitch || !pitch.is_multiple_of(multiple) {
                return copy(made, SurfaceCopy::Pitch { plane: at });
            }
        }
        let Some((starts, end)) = lay(spec, plans, pitch) else {
            return copy(made, SurfaceCopy::Start { plane: 0 });
        };
        for at in 0..made.len() {
            let start = starts[at];
            let clear = at == 0 || starts[at - 1] + made[at - 1].needs.size <= start;
            let aligned = start.is_multiple_of(made[at].needs.alignment);
            if made[at].layout.offset != 0 || !aligned || !clear {
                return copy(made, SurfaceCopy::Start { plane: at });
            }
        }
        let Some(index) = common_type(core, &made) else {
            return copy(made, SurfaceCopy::MemoryType);
        };
        let reach = (made.iter().zip(&starts)).map(|(m, &start)| start + m.needs.size);
        let len = reach.fold(end, u64::max);
        // SAFETY: each image's handle is the allocation's, and none requires an allocation of
        // its own.
        let memory = match unsafe { memory::export_surface(core, len, index, spec.handle) } {
            Ok(memory) => memory,
            Err(error) => {
                destroy(device, made);
                return Err(error);
            }
        };
        let raw = memory.device_memory();
        let block = Arc::new(memory::Block::new(memory, Arc::clone(core)));
        for at in 0..made.len() {
            let (image, start) = (made[at].image, starts[at]);
            // SAFETY: the allocation holds the image's size from its start, at its alignment, in a
            // type its requirements list, exportable as a handle the image was made for.
            if let Err(result) = unsafe { device.bind_image_memory(image, raw, start) } {
                destroy(device, made);
                return Err(Error::Vulkan {
                    call: "vkBindImageMemory",
                    result,
                });
            }
        }
        let fd = match core.exports.handle(raw, spec.handle) {
            Ok(fd) => fd,
            Err(error) => {
                destroy(device, made);
                return Err(error);
            }
        };
        let export = Export {
            handle: fd,
            size: block.memory.len,
            memory_type: index,
        };
        let images = (made.into_iter())
            .map(|m| SurfaceImage {
                image: m.image,
                backing: Backing::InBlock,
                bytes: m.needs.size,
                usage: m.usage,
                view_formats: m.list,
                block: Some(Arc::clone(&block)),
            })
            .collect();
        let planes = (starts.iter())
            .map(|&offset| PlaneLayout {
                offset,
                row_pitch: pitch,
            })
            .collect();
        Ok(Ok(InPlace {
            images,
            export,
            planes,
        }))
    }

    /// A dma-buf's planes in place: `DRM_FORMAT_MOD_LINEAR` images at the driver's widest pitch
    /// of the implicit list, rounded up to the multiples, each stated at its start and on its own
    /// dedicated import of one exportable allocation, which is freed once the imports and the
    /// handle exist.
    #[cfg(unix)]
    fn dma_buf_in_place(
        &self,
        spec: &SurfaceSpec<'_>,
        plans: &[PlanePlan],
        multiple: u64,
    ) -> Result<Result<InPlace, SurfaceCopy>, Error> {
        use std::os::fd::{AsRawFd, IntoRawFd};
        let core = self.core();
        let device = &core.device;
        let query = Query {
            instance: core.instance.vk(),
            physical: core.physical,
            handle: spec.handle,
        };
        let vulkan = |call: &'static str| move |result: vk::Result| Error::Vulkan { call, result };
        // Each plane's image with the implicit list: the driver's own pitch.
        let mut chosen = Vec::with_capacity(plans.len());
        let mut pitch = 0u64;
        for (at, (plane, plan)) in spec.planes.iter().zip(plans).enumerate() {
            let (usage, list) = match query.plane(at, plan, plane.descriptor.pixel_format)? {
                Ok(found) => found,
                Err(reason) => return Ok(Err(reason)),
            };
            // SAFETY: the query accepted the image; the extensions are enabled.
            let implicit = unsafe {
                make_image(
                    device,
                    &plane.descriptor,
                    (plan.format, usage, &list),
                    spec.handle,
                    None,
                )
            }
            .map_err(vulkan("vkCreateImage"))?;
            pitch = pitch.max(implicit.layout.row_pitch);
            destroy(device, vec![implicit]);
            chosen.push((usage, list));
        }
        let too_long = || Error::Limit(Limit::BlockLen { len: u64::MAX });
        let pitch = pitch
            .checked_next_multiple_of(multiple)
            .ok_or_else(too_long)?;
        let (starts, end) = lay(spec, plans, pitch).ok_or_else(too_long)?;
        // Each plane's image stated at its start and the pitch.
        let mut made: Vec<Made> = Vec::with_capacity(plans.len());
        let copy =
            |made: Vec<Made>, reason: SurfaceCopy| -> Result<Result<InPlace, SurfaceCopy>, Error> {
                destroy(device, made);
                Ok(Err(reason))
            };
        for (at, ((plane, plan), (usage, list))) in
            spec.planes.iter().zip(plans).zip(&chosen).enumerate()
        {
            let parts = (plan.format, *usage, list.as_slice());
            // SAFETY: as above; the layout is the driver's to refuse.
            let stated = unsafe {
                make_image(
                    device,
                    &plane.descriptor,
                    parts,
                    spec.handle,
                    Some((starts[at], pitch)),
                )
            };
            let refused = vk::Result::ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT;
            match stated {
                Ok(image) => made.push(image),
                Err(result) if result == refused => {
                    // The pitch alone, at offset 0, tells which the driver refuses.
                    // SAFETY: as above.
                    let at_zero = unsafe {
                        make_image(
                            device,
                            &plane.descriptor,
                            parts,
                            spec.handle,
                            Some((0, pitch)),
                        )
                    };
                    let reason = match at_zero {
                        Ok(image) => {
                            destroy(device, vec![image]);
                            SurfaceCopy::Start { plane: at }
                        }
                        Err(_) => SurfaceCopy::Pitch { plane: at },
                    };
                    return copy(made, reason);
                }
                Err(result) => {
                    destroy(device, made);
                    return Err(Error::Vulkan {
                        call: "vkCreateImage",
                        result,
                    });
                }
            }
        }
        for at in 0..made.len() {
            if made[at].layout.row_pitch != pitch {
                return copy(made, SurfaceCopy::Pitch { plane: at });
            }
            if made[at].layout.offset != starts[at] {
                return copy(made, SurfaceCopy::Start { plane: at });
            }
        }
        let Some(index) = common_type(core, &made) else {
            return copy(made, SurfaceCopy::MemoryType);
        };
        // Each import holds its plane's rows from the stated offset, whatever its requirement
        // counts of them.
        let sizes: Vec<u64> = (made.iter())
            .map(|m| m.needs.size.max(m.layout.offset + m.layout.size))
            .collect();
        let len = sizes.iter().copied().fold(end, u64::max);
        let len = match memory::round(len, memory::GRANULE, core.max_block_len) {
            Ok(len) => len,
            Err(error) => {
                destroy(device, made);
                return Err(error);
            }
        };
        let mut export = vk::ExportMemoryAllocateInfo::default().handle_types(spec.handle.vk());
        let info = vk::MemoryAllocateInfo::default()
            .allocation_size(len)
            .memory_type_index(index)
            .push_next(&mut export);
        // SAFETY: an exportable allocation of a type every image takes, bound to nothing.
        let block = match unsafe {
            memory::allocate(device, &core.counters, core.max_allocations, &info)
        } {
            Ok(block) => block,
            Err(error) => {
                destroy(device, made);
                return Err(error);
            }
        };
        let free_block = || {
            // SAFETY: nothing is bound to the allocation; the imports and the handles hold the
            // payload ("freeing ... releases that memory object's reference" alone).
            unsafe { device.free_memory(block, None) };
            core.counters.fall(Gauge::Blocks);
        };
        let Some(fds) = core.exports.fds() else {
            free_block();
            destroy(device, made);
            return Err(Error::Limit(Limit::NoExport {
                handle: spec.handle,
            }));
        };
        let mut images = Vec::with_capacity(made.len());
        let mut imported: Vec<vk::DeviceMemory> = Vec::with_capacity(made.len());
        let undo = |imported: Vec<vk::DeviceMemory>,
                    made: Vec<Made>,
                    error: Error|
         -> Result<Result<InPlace, SurfaceCopy>, Error> {
            destroy(device, made);
            for memory in imported {
                // SAFETY: the image bound to it is destroyed.
                unsafe { device.free_memory(memory, None) };
                core.counters.fall(Gauge::Blocks);
                core.counters.fall(Gauge::SurfaceImports);
            }
            free_block();
            Err(error)
        };
        for at in 0..made.len() {
            let (image, image_bits, size) =
                (made[at].image, made[at].needs.memory_type_bits, sizes[at]);
            let fd = match core.exports.handle(block, spec.handle) {
                Ok(fd) => fd,
                Err(error) => return undo(imported, made, error),
            };
            let mut properties = vk::MemoryFdPropertiesKHR::default();
            // SAFETY: the fd is a dma-buf the device exported.
            let types = unsafe {
                fds.get_memory_fd_properties(spec.handle.vk(), fd.as_raw_fd(), &mut properties)
            };
            if let Err(result) = types {
                return undo(imported, made, vulkan("vkGetMemoryFdPropertiesKHR")(result));
            }
            let bits = properties.memory_type_bits & image_bits;
            let import_type = match bits & (1 << index) {
                0 if bits == 0 => {
                    return undo(
                        imported,
                        made,
                        Error::Limit(Limit::ExportType {
                            handle: spec.handle,
                            listed: Some(properties.memory_type_bits),
                        }),
                    );
                }
                0 => bits.trailing_zeros(),
                _ => index,
            };
            let mut import = vk::ImportMemoryFdInfoKHR::default()
                .handle_type(spec.handle.vk())
                .fd(fd.as_raw_fd());
            let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image);
            let info = vk::MemoryAllocateInfo::default()
                .allocation_size(size)
                .memory_type_index(import_type)
                .push_next(&mut import)
                .push_next(&mut dedicated);
            // SAFETY: an import of the dma-buf, dedicated to the image, of a type both take, at
            // least the image's requirement and its stated rows.
            let memory = match unsafe {
                memory::allocate(device, &core.counters, core.max_allocations, &info)
            } {
                Ok(memory) => memory,
                Err(error) => return undo(imported, made, error),
            };
            // A successful import owns the fd.
            let _ = fd.into_raw_fd();
            core.counters.rise(Gauge::SurfaceImports);
            imported.push(memory);
            // SAFETY: the memory is the image's dedicated import, of its size and a type it lists.
            if let Err(result) = unsafe { device.bind_image_memory(image, memory, 0) } {
                return undo(imported, made, vulkan("vkBindImageMemory")(result));
            }
        }
        let handle = match core.exports.handle(block, spec.handle) {
            Ok(fd) => fd,
            Err(error) => return undo(imported, made, error),
        };
        free_block();
        for (m, (memory, size)) in made.into_iter().zip(imported.into_iter().zip(sizes)) {
            images.push(SurfaceImage {
                image: m.image,
                backing: Backing::Import(memory),
                bytes: size,
                usage: m.usage,
                view_formats: m.list,
                block: None,
            });
        }
        let export = Export {
            handle,
            size: len,
            memory_type: index,
        };
        let planes = (starts.iter())
            .map(|&offset| PlaneLayout {
                offset,
                row_pitch: pitch,
            })
            .collect();
        Ok(Ok(InPlace {
            images,
            export,
            planes,
        }))
    }
}

impl Surface {
    /// The planes' textures, in the spec's order.
    pub fn planes(&self) -> &[Texture] {
        &self.planes
    }

    /// The handle the surface's memory is shared through.
    pub fn handle(&self) -> ExportHandle {
        self.handle
    }

    /// The copy form's buffer, over the whole block; `None` in place.
    pub(crate) fn copied(&self) -> Option<&Buffer> {
        self.copy.as_ref().map(|(_, buffer)| buffer)
    }

    /// Whether the surface is `core`'s.
    pub(crate) fn is_of(&self, core: &Arc<Core>) -> bool {
        Arc::ptr_eq(self.device.core(), core)
    }

    /// No Metal counterpart: the other API's writes into this surface are done, and the
    /// runtime's commands may now read and write its planes, as
    /// [`Block::external_writes_done`] is for a block. The consumer calls it once the other API's
    /// work that wrote the surface has completed, and before it commits the command buffers that
    /// use what was written.
    ///
    /// In place, the call records each plane's image's acquire from the other API's queue
    /// family (`VK_QUEUE_FAMILY_FOREIGN_EXT` for a dma-buf, `VK_QUEUE_FAMILY_EXTERNAL` for an
    /// opaque handle), the general layout kept, into the upload that goes ahead of every command
    /// buffer committed after it: one barrier of the planes, counted in
    /// [`Metrics::external_acquires`](crate::Metrics::external_acquires). In the copy form it is
    /// the block's acquire, and the submissions copy the bytes into the planes' images. The
    /// contract is the block's.
    ///
    /// Err: [`Refusal::InScheduledHandler`] in a scheduled handler; [`Error::Quiesced`]; and
    /// [`Error::Vulkan`] where the upload's command buffer cannot be made.
    pub fn external_writes_done(&self) -> Result<(), Error> {
        if let Some((block, _)) = &self.copy {
            return block.external_writes_done();
        }
        crate::command::not_in_scheduled_handler()?;
        let core = self.device.core();
        if core.closed() {
            return Err(Error::Quiesced);
        }
        let images: Vec<_> = self.planes.iter().map(Texture::held_image).collect();
        let family = self.handle.family();
        let acquires: Vec<vk::ImageMemoryBarrier2<'static>> = (self.planes.iter())
            .map(|texture| {
                vk::ImageMemoryBarrier2::default()
                    .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_queue_family_index(family)
                    .dst_queue_family_index(core.family)
                    .image(texture.image())
                    .subresource_range(texture.whole())
            })
            .collect();
        let held = std::array::from_fn::<_, MAX_PLANES, _>(|at| {
            Held::Texture(Arc::clone(&images[at % images.len()]))
        });
        core.work.upload(core, None, held, |device, command| {
            let dependency = vk::DependencyInfo::default().image_memory_barriers(&acquires);
            // SAFETY: the upload's command buffer is recording, under the work lock; it holds the
            // images, which are in the general layout, and a dma-buf's foreign family is enabled.
            unsafe { device.cmd_pipeline_barrier2(command, &dependency) };
        })?;
        core.counters.count(Metric::Barriers);
        core.counters.count(Metric::ExternalAcquires);
        Ok(())
    }
}
