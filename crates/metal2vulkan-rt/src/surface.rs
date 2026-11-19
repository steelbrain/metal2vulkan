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
use crate::buffer::{Buffer, BufferObject};
#[cfg(unix)]
use crate::counters::Gauge;
use crate::counters::Metric;
use crate::device::{lock, Core, Device};
use crate::error::{Error, Gap, Limit, Refusal};
use crate::export::{Export, ExportHandle};
use crate::hazard::HeapId;
use crate::memory;
use crate::submit::Held;
use crate::texture::{Backing, SurfaceImage, Texture, TextureDescriptor, TextureObject};
use ash::vk;
use std::sync::{Arc, Mutex};

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
    /// import, 4 for NVIDIA's opaque fd route, for instance). At least 1.
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
    /// The texture's index, as for [`Device::new_texture`]; with none, a plane shaders read or
    /// write is reached through its views alone, each with an index of its own, as
    /// [`Device::new_texture_for_views`] makes a texture: the image under the textures a consumer
    /// makes over the plane as its views.
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
    kind: Kind,
}

/// The surface's form, and what its buffer is ([`Surface::buffer`]).
enum Kind {
    /// In place: the buffer's memory, and the buffer once it is made.
    InPlace(View),
    /// The copy form's block and its buffer, the surface's buffer.
    Copy(Block, Buffer),
}

/// An in-place surface's buffer: made at its first call, on the surface's own memory.
struct View {
    /// The surface's id in the hazard sets, which its planes and its buffer take as one
    /// ([`HeapId::unit`]).
    unit: Arc<HeapId>,
    /// The memory's length, the export's size.
    len: u64,
    state: Mutex<ViewState>,
}

/// What an in-place surface's buffer is made on, until it is made.
enum ViewState {
    Source(Source),
    Made(Buffer),
}

/// The memory an in-place surface's buffer is made on.
enum Source {
    /// The opaque handle's one allocation, which the planes are bound in, and its memory type.
    Opaque(Arc<memory::Block>, u32),
    /// The runtime's own fd of the dma-buf, and the memory type the planes' imports took.
    #[cfg(unix)]
    DmaBuf(std::os::fd::OwnedFd, u32),
}

impl std::fmt::Debug for Surface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Surface")
            .field("handle", &self.handle)
            .field("planes", &self.planes.len())
            .field("in_place", &matches!(self.kind, Kind::InPlace(_)))
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
        Ok(tiling_verdict(at, offered, features).map(|()| (offered, list)))
    }
}

/// Whether plane `at`'s tiling holds it in place, from the query's answers: the usage it offers,
/// and the external memory features of an image of that usage. `Tiling` where it lacks one of
/// [`IN_PLACE_USAGE`] or is not both exportable and importable; `Dedicated` where it is exported
/// only in an allocation of its own.
pub(crate) fn tiling_verdict(
    at: usize,
    offered: vk::ImageUsageFlags,
    features: vk::ExternalMemoryFeatureFlags,
) -> Result<(), SurfaceCopy> {
    let both =
        vk::ExternalMemoryFeatureFlags::EXPORTABLE | vk::ExternalMemoryFeatureFlags::IMPORTABLE;
    if !offered.contains(IN_PLACE_USAGE) || !features.contains(both) {
        return Err(SurfaceCopy::Tiling { plane: at });
    }
    if features.contains(vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY) {
        return Err(SurfaceCopy::Dedicated { plane: at });
    }
    Ok(())
}

/// What an opaque handle's layout reads of a plane's `LINEAR` image: its pitch and offset, its
/// requirement's size and alignment, and whether it requires an allocation of its own.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PlaneShape {
    pub(crate) row_pitch: u64,
    pub(crate) offset: u64,
    pub(crate) size: u64,
    pub(crate) alignment: u64,
    pub(crate) requires: bool,
}

/// An opaque handle's one pitch: the first image's, where every image takes it, it is a multiple
/// of `multiple`, and none requires an allocation of its own (`Dedicated`); else `Pitch`.
pub(crate) fn opaque_pitch(shapes: &[PlaneShape], multiple: u64) -> Result<u64, SurfaceCopy> {
    let pitch = shapes.first().map_or(0, |shape| shape.row_pitch);
    for (at, shape) in shapes.iter().enumerate() {
        if shape.requires {
            return Err(SurfaceCopy::Dedicated { plane: at });
        }
        if shape.row_pitch != pitch || !pitch.is_multiple_of(multiple) {
            return Err(SurfaceCopy::Pitch { plane: at });
        }
    }
    Ok(pitch)
}

/// Whether an opaque handle's images lie at `starts`: each at its image's offset 0, on its
/// alignment, past the plane before's requirement; else `Start`.
pub(crate) fn opaque_starts(shapes: &[PlaneShape], starts: &[u64]) -> Result<(), SurfaceCopy> {
    for (at, (shape, &start)) in shapes.iter().zip(starts).enumerate() {
        let clear = at == 0 || starts[at - 1] + shapes[at - 1].size <= start;
        if shape.offset != 0 || !start.is_multiple_of(shape.alignment.max(1)) || !clear {
            return Err(SurfaceCopy::Start { plane: at });
        }
    }
    Ok(())
}

/// The memory type every plane's image takes, from each image's type `bits`: `private` where they
/// all take it, else the lowest of `device_local`'s, else the lowest; `None` where none.
pub(crate) fn type_of(bits: &[u32], private: u32, device_local: u32) -> Option<u32> {
    let bits = bits.iter().fold(u32::MAX, |all, b| all & b);
    match (bits, bits & device_local) {
        (0, _) => None,
        (bits, _) if bits & (1 << private) != 0 => Some(private),
        (_, 0) => Some(bits.trailing_zeros()),
        (_, local) => Some(local.trailing_zeros()),
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

/// The memory type every plane's image takes ([`type_of`]).
fn common_type(core: &Core, made: &[Made]) -> Option<u32> {
    let bits: Vec<u32> = made.iter().map(|m| m.needs.memory_type_bits).collect();
    let private = core.memory_type(crate::MemoryClass::Private);
    type_of(&bits, private, core.device_local)
}

/// What the in-place forms make: each plane's image, bound, the export, and what the surface's
/// buffer is made on.
struct InPlace {
    images: Vec<SurfaceImage>,
    export: Export,
    planes: Vec<PlaneLayout>,
    source: Source,
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
        let unit = HeapId::unit(core);
        let mut planes = Vec::with_capacity(count);
        let mut images = made.images.into_iter();
        for (plane, image) in spec.planes.iter().zip(images.by_ref()) {
            match self.surface_texture(&plane.descriptor, plane.index, image, Arc::clone(&unit)) {
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
        let view = View {
            unit,
            len: made.export.size,
            state: Mutex::new(ViewState::Source(made.source)),
        };
        let surface = Surface {
            device: self.clone(),
            handle,
            planes,
            kind: Kind::InPlace(view),
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
        let max = self.core().max_block_len;
        let too_long = || Error::Limit(Limit::BlockLen { len: u64::MAX, max });
        let pitch = row
            .checked_next_multiple_of(multiple)
            .ok_or_else(too_long)?;
        let (starts, end) = lay(spec, plans, pitch).ok_or_else(too_long)?;
        let (block, export) = self.new_exported_block(spec.handle, end)?;
        let buffer = self.new_buffer(block.id, 0, block.len, None)?;
        let planes = (spec.planes.iter().zip(&starts))
            .map(|(plane, &start)| {
                buffer.new_plane_texture(&plane.descriptor, plane.index, start, pitch)
            })
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
            kind: Kind::Copy(block, buffer),
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
        let shapes: Vec<PlaneShape> = (made.iter())
            .map(|m| PlaneShape {
                row_pitch: m.layout.row_pitch,
                offset: m.layout.offset,
                size: m.needs.size,
                alignment: m.needs.alignment,
                requires: m.requires,
            })
            .collect();
        let pitch = match opaque_pitch(&shapes, multiple) {
            Ok(pitch) => pitch,
            Err(reason) => return copy(made, reason),
        };
        let Some((starts, end)) = lay(spec, plans, pitch) else {
            return copy(made, SurfaceCopy::Start { plane: 0 });
        };
        if let Err(reason) = opaque_starts(&shapes, &starts) {
            return copy(made, reason);
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
        let source = Source::Opaque(Arc::clone(&block), index);
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
            source,
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
        let max = self.core().max_block_len;
        let too_long = || Error::Limit(Limit::BlockLen { len: u64::MAX, max });
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
            memory::allocate(device, &core.counters, core.max_allocations, &info, None)
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
        let mut first_type = None;
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
            first_type = first_type.or(Some(import_type));
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
                memory::allocate(device, &core.counters, core.max_allocations, &info, None)
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
        // The runtime's own fd, which the surface's buffer's import duplicates.
        let spare = match core.exports.handle(block, spec.handle) {
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
            source: Source::DmaBuf(spare, first_type.unwrap_or(index)),
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
        match &self.kind {
            Kind::Copy(_, buffer) => Some(buffer),
            Kind::InPlace(_) => None,
        }
    }

    /// The buffer in place, where [`buffer`](Self::buffer) has made it.
    pub(crate) fn made_buffer(&self) -> Option<Buffer> {
        match &self.kind {
            Kind::InPlace(view) => match &*lock(&view.state) {
                ViewState::Made(buffer) => Some(buffer.clone()),
                ViewState::Source(_) => None,
            },
            Kind::Copy(..) => None,
        }
    }

    /// Whether the surface is `core`'s.
    pub(crate) fn is_of(&self, core: &Arc<Core>) -> bool {
        Arc::ptr_eq(self.device.core(), core)
    }

    /// An IOSurface-backed buffer's counterpart (`MTLBuffer.h` speaks of a buffer "backed by an
    /// IOSurface"; `MTLDevice.h` makes a plane's texture from one,
    /// `newTextureWithDescriptor:iosurface:plane:`, and no buffer): a buffer over the surface's
    /// own memory from its first byte, whose bytes lie at the layout the [`SurfaceExport`]
    /// states: plane `i`'s row `y` at `planes[i].offset + y * planes[i].row_pitch`. In place it
    /// spans the export's size ([`Export::size`]); in the copy form, its block's length, which
    /// holds every plane and which the export's size passes where the driver's allocation is
    /// larger. Private: its `contents` is `None`, and the host reads its bytes
    /// through a copy into a Shared buffer, where Metal's host reads an IOSurface's bytes through
    /// the surface's own mapping. Every call answers the same buffer.
    ///
    /// **Form.** In place, the buffer is made at the first call, never at the surface's make: of
    /// an opaque handle, a buffer bound at the first byte of the one allocation the planes are
    /// bound in; of a dma-buf, a buffer on one more import of the surface's dma-buf, into the
    /// memory type the planes' imports took, which the call attempts only where the fd's query
    /// and the buffer's memory requirements list that type. Either counts in
    /// [`Gauges::surface_buffers`](crate::Gauges::surface_buffers) while it lives, a dma-buf's
    /// also in [`Gauges::surface_imports`](crate::Gauges::surface_imports). The first call also
    /// records the buffer's acquire from the other API's queue family, as
    /// [`external_writes_done`](Self::external_writes_done) records the planes', into the upload
    /// that goes ahead of every command buffer committed after it, counted in
    /// [`Metrics::external_acquires`](crate::Metrics::external_acquires); from then each
    /// [`external_writes_done`](Self::external_writes_done) acquires the buffer with the planes,
    /// and each [`release_surface_to_external`](crate::BlitCommandEncoder::release_surface_to_external)
    /// releases it with them. In the copy form the buffer is the block's, which the planes are
    /// textures over, so the planes' copies carry the bytes between them.
    ///
    /// **Order.** The surface, its planes (and their views) and its buffer are one resource in
    /// the runtime's ordering of reads and writes, as a heap's resources are: a write through the
    /// buffer is ordered before a later read through a plane, and the reverse, as two uses of one
    /// texture are.
    ///
    /// **Basis.** Vulkan makes the buffer and the planes aliases of the surface's memory
    /// (section 12.11, Memory Aliasing): "A range of a VkDeviceMemory allocation is aliased if it
    /// is bound to multiple resources simultaneously, as described below, via vkBindImageMemory,
    /// vkBindBufferMemory, vkBindAccelerationStructureMemoryNV, vkBindTensorMemoryARM, via sparse
    /// memory bindings, or by binding the memory to resources in multiple Vulkan instances or
    /// external APIs using external memory handle export and import mechanisms." The planes in
    /// place are linear images in the general layout, and "Buffers, linear tensors, and linear
    /// image subresources are considered host-accessible subresources when they are in any of
    /// these layouts: VK_IMAGE_LAYOUT_PREINITIALIZED, VK_IMAGE_LAYOUT_GENERAL,
    /// VK_IMAGE_LAYOUT_ZERO_INITIALIZED_EXT", and "If two aliases are both host-accessible, then
    /// they interpret the contents of the memory in consistent ways, and data written to one alias
    /// can be read by the other alias." A dma-buf's buffer and planes are distinct imports of one
    /// payload in one instance, which the fd import allows ("multiple times into a given Vulkan
    /// instance. In all cases, each import operation must create a distinct VkDeviceMemory
    /// object"); the section names aliases through export and import "in multiple Vulkan
    /// instances or external APIs" ("Aliases created by binding the same memory to resources in
    /// multiple Vulkan instances or external APIs using external memory handle export and import
    /// mechanisms interpret the contents of the memory in consistent ways, and data written to one
    /// alias can be read by the other alias"), and the runtime takes imports in one instance as
    /// those aliases, as its planes, each on an import of its own, already do. The order is the
    /// section's: "Use of an overlapping range by two aliases must be separated by a memory
    /// dependency using the appropriate access types if at least one of those uses performs
    /// writes"; the runtime's are global memory barriers, which reach every alias.
    ///
    /// Err: in place, at the call that makes the buffer: [`Refusal::InScheduledHandler`] in a
    /// scheduled handler; [`Error::Quiesced`]; [`Limit::ExportType`] where the buffer's memory
    /// requirements (and for a dma-buf the fd's query) do not list the surface memory's type, ask
    /// more than its length, or require an allocation of its own;
    /// [`Limit::NoImport`](crate::Limit::NoImport) for a dma-buf on a device whose buffers import
    /// none; [`Error::Fd`]; [`Limit::MemoryAllocations`]; [`Limit::BlockIds`]; [`Error::Vulkan`]
    /// where the buffer, its import, its binding or the upload's command buffer cannot be made.
    /// A refused call makes nothing, and a later call tries again.
    pub fn buffer(&self) -> Result<Buffer, Error> {
        let view = match &self.kind {
            Kind::Copy(_, buffer) => return Ok(buffer.clone()),
            Kind::InPlace(view) => view,
        };
        let mut state = lock(&view.state);
        let source = match &*state {
            ViewState::Made(buffer) => return Ok(buffer.clone()),
            ViewState::Source(source) => source,
        };
        crate::command::not_in_scheduled_handler()?;
        let core = self.device.core();
        if core.closed() {
            return Err(Error::Quiesced);
        }
        let block = match source {
            Source::Opaque(under, index) => {
                // SAFETY: the allocation is the opaque handle's surface allocation, of type
                // `index`, exportable as the surface's handle.
                unsafe { memory::surface_view(core, under, self.handle, *index) }?
            }
            #[cfg(unix)]
            Source::DmaBuf(fd, index) => crate::import::surface_import(core, fd, view.len, *index)?,
        };
        let block = Arc::new(block);
        let (raw, _) = block.memory.range(0);
        let acquire = [acquire(self.handle.family(), core.family, raw)];
        let held = [Held::Block(Arc::clone(&block))];
        core.work.upload(core, None, held, |device, command| {
            let dependency = vk::DependencyInfo::default().buffer_memory_barriers(&acquire);
            // SAFETY: the upload's command buffer is recording, under the work lock; the buffer
            // spans the surface's memory, which the upload holds, and a dma-buf's foreign family
            // is enabled.
            unsafe { device.cmd_pipeline_barrier2(command, &dependency) };
        })?;
        core.counters.count(Metric::Barriers);
        core.counters.count(Metric::ExternalAcquires);
        let unit = Some(Arc::clone(&view.unit));
        let buffer = self.device.buffer(block, 0, view.len, None, None, unit);
        *state = ViewState::Made(buffer.clone());
        Ok(buffer)
    }

    /// No Metal counterpart: the other API's writes into this surface are done, and the
    /// runtime's commands may now read and write its planes, as
    /// [`Block::external_writes_done`] is for a block. The consumer calls it once the other API's
    /// work that wrote the surface has completed, and before it commits the command buffers that
    /// use what was written.
    ///
    /// In place, the call records each plane's image's acquire from the other API's queue
    /// family (`VK_QUEUE_FAMILY_FOREIGN_EXT` for a dma-buf, `VK_QUEUE_FAMILY_EXTERNAL` for an
    /// opaque handle), the general layout kept, and the surface's buffer's where
    /// [`buffer`](Self::buffer) has made it, into the upload that goes ahead of every command
    /// buffer committed after it: one barrier of the planes and the buffer, counted in
    /// [`Metrics::external_acquires`](crate::Metrics::external_acquires). In the copy form it is
    /// the block's acquire, and the submissions copy the bytes into the planes' images. The
    /// contract is the block's.
    ///
    /// Err: [`Refusal::InScheduledHandler`] in a scheduled handler; [`Error::Quiesced`]; and
    /// [`Error::Vulkan`] where the upload's command buffer cannot be made.
    pub fn external_writes_done(&self) -> Result<(), Error> {
        if let Kind::Copy(block, _) = &self.kind {
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
        let buffer = self.made_buffer();
        let buffers: Vec<vk::BufferMemoryBarrier2<'static>> = (buffer.iter())
            .map(|buffer| acquire(family, core.family, buffer.device_range().0))
            .collect();
        let held = std::array::from_fn::<_, { MAX_PLANES + 1 }, _>(|at| match &buffer {
            Some(buffer) if at == MAX_PLANES => Held::Block(Arc::clone(buffer.held().0)),
            _ => Held::Texture(Arc::clone(&images[at % images.len()])),
        });
        core.work.upload(core, None, held, |device, command| {
            let dependency = vk::DependencyInfo::default()
                .image_memory_barriers(&acquires)
                .buffer_memory_barriers(&buffers);
            // SAFETY: the upload's command buffer is recording, under the work lock; it holds the
            // images, which are in the general layout, and the buffer's memory, and a dma-buf's
            // foreign family is enabled.
            unsafe { device.cmd_pipeline_barrier2(command, &dependency) };
        })?;
        core.counters.count(Metric::Barriers);
        core.counters.count(Metric::ExternalAcquires);
        Ok(())
    }
}

/// The acquire of a surface's buffer `buffer` from the other API's queue family `from` to the
/// runtime's `to`, made visible to every later command.
fn acquire(from: u32, to: u32, buffer: vk::Buffer) -> vk::BufferMemoryBarrier2<'static> {
    vk::BufferMemoryBarrier2::default()
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
        .src_queue_family_index(from)
        .dst_queue_family_index(to)
        .buffer(buffer)
        .offset(0)
        .size(vk::WHOLE_SIZE)
}
