//! A picture's Direct3D 11 textures as a surface ([`Device::import_d3d11_surface`]), a Windows
//! host's: the planes another API decodes or draws into, shared by their NT handles, each
//! imported as an optimal image of its own and read by rt's shaders in place.
//!
//! The import makes no submission and waits for nothing: Vulkan takes such an image as already
//! in the general layout once Direct3D 11 has accessed it (section 12.5.2, below), so no move out
//! of the undefined layout comes before its first use. Each submission that uses a plane takes it
//! from `VK_QUEUE_FAMILY_EXTERNAL` at its start and gives it back at its end (`crate::submit`).

use crate::counters::Gauge;
use crate::device::{Core, Device};
use crate::error::{Error, Gap, Limit, Refusal};
use crate::hazard::HeapId;
use crate::import::ImportHandle;
use crate::memory;
use crate::surface::{Surface, MAX_PLANES};
use crate::texture::{discard_surface_image, Backing, SurfaceImage, TextureDescriptor};
use crate::PixelFormat;
use ash::vk;
use metal2vulkan::metal::format as formats;
use std::os::windows::io::{AsRawHandle, BorrowedHandle};
use std::sync::Arc;

/// No Metal counterpart: one plane of a picture another API made as Direct3D 11 textures
/// ([`Device::import_d3d11_surface`]): the shared texture's handle and its own format, and the
/// texture rt makes over it. Metal's counterpart of the whole is an IOSurface's plane under a
/// texture (`newTextureWithDescriptor:iosurface:plane:`).
#[derive(Clone, Copy, Debug)]
pub struct D3d11Plane<'a> {
    /// The shared texture's NT handle (`IDXGIResource1::CreateSharedHandle`), as Vulkan imports
    /// a Direct3D 11 texture (`VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT`). The consumer
    /// keeps it and closes it: an import takes no ownership of an NT handle.
    pub handle: BorrowedHandle<'a>,
    /// The Direct3D 11 texture's own format: `R8Unorm` or `R16Unorm` for a luma plane,
    /// `R16Uint` or `R32Uint` for a chroma plane, which the texture views as two channels. Its
    /// texel has the bytes of the descriptor's, in Metal's same view class.
    pub image_format: PixelFormat,
    /// The texture rt makes over the plane, as [`PlaneSpec::descriptor`](crate::PlaneSpec):
    /// 2D, of one level, sample and slice, its width and height the Direct3D 11 texture's, its
    /// pixel format the view's (`R8Unorm`, `RG8Unorm`, `R16Unorm`, `RG16Unorm`).
    pub descriptor: TextureDescriptor,
    /// The texture's index, as a [`PlaneSpec`](crate::PlaneSpec)'s: with none, the plane is
    /// reached through its views alone.
    pub index: Option<u32>,
}

/// The usage a plane's image takes where the import offers it and the view's format holds it:
/// sampled, rendered, copied from and into, and storage.
const WANTED: vk::ImageUsageFlags = vk::ImageUsageFlags::from_raw(
    vk::ImageUsageFlags::SAMPLED.as_raw()
        | vk::ImageUsageFlags::COLOR_ATTACHMENT.as_raw()
        | vk::ImageUsageFlags::TRANSFER_SRC.as_raw()
        | vk::ImageUsageFlags::TRANSFER_DST.as_raw()
        | vk::ImageUsageFlags::STORAGE.as_raw(),
);

/// The handle type every plane imports as.
const HANDLE: vk::ExternalMemoryHandleTypeFlags = vk::ExternalMemoryHandleTypeFlags::D3D11_TEXTURE;

impl Device {
    /// No Metal counterpart of the call: a picture's planes, Direct3D 11 textures another API
    /// made (a video decoder's, or a draw's), as a surface whose planes are textures rt's shaders
    /// read and write in place, as Metal reads an IOSurface's plane through a texture over it
    /// (`newTextureWithDescriptor:iosurface:plane:`). Each plane is its own texture and its own
    /// handle: an optimal image of the plane's [`image_format`](D3d11Plane::image_format), on a
    /// dedicated import of the handle (`VkImportMemoryWin32HandleInfoKHR`,
    /// `VkMemoryDedicatedAllocateInfo`) into the first device-local type that the handle's
    /// properties (`vkGetMemoryWin32HandlePropertiesKHR`) and the image's requirements both list,
    /// else the first both list. The plane's texture is in the descriptor's format, a view of the
    /// image's. Each import counts in [`Gauges::surface_imports`](crate::Gauges::surface_imports)
    /// while its texture lives.
    ///
    /// **Views of another format.** The image is mutable where it lists more than its own
    /// format. It lists its own format, the descriptor's and those the descriptor's texture
    /// lists for views of another format
    /// ([`Texture::new_texture_view`](crate::Texture::new_texture_view): its sRGB sibling, and
    /// with `PixelFormatView` usage the formats Metal views it as) where the import query
    /// (`vkGetPhysicalDeviceImageFormatProperties2`) takes that list with the image's usage.
    /// Where it does not, the image lists its own format and the descriptor's where they differ,
    /// and a view of a format it does not list is [`Gap::TextureView`] of
    /// [`TextureViewKind::SurfacePixelFormat`](crate::TextureViewKind::SurfacePixelFormat).
    ///
    /// **Layout.** The call makes no submission and waits for nothing: no move out of the
    /// undefined layout comes before a plane's first use. Vulkan 1.4, section 12.5.2 (Image
    /// Layout Rules with External Memory): "When images created with external memory handle
    /// types listed in the external image implied layouts table are accessed outside of Vulkan,
    /// they must be in the layout specified in the table, and can be assumed to be in that layout
    /// when they are accessed in Vulkan after external access", and its Table 17 gives
    /// `VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT` the layout `VK_IMAGE_LAYOUT_GENERAL`.
    ///
    /// **Ownership.** Each submission whose command buffers or uploads use a plane acquires it
    /// from `VK_QUEUE_FAMILY_EXTERNAL` in a command buffer of rt's own ahead of everything, from
    /// and to the general layout (`VUID-VkImageMemoryBarrier-srcQueueFamilyIndex-12332`), and
    /// releases it to that family in another after everything, in the general layout
    /// (`...-dstQueueFamilyIndex-12331`), which takes the submission's last signals, so it adds
    /// no submit info: Direct3D 11 owns the planes' memory between rt's submissions, each counted
    /// in [`Metrics::d3d11_plane_submissions`](crate::Metrics::d3d11_plane_submissions). The
    /// acquire follows a global barrier from and to every stage, so an earlier submission's
    /// release on rt's one queue happens before it (Vulkan 1.4, section 7.7.6).
    /// [`Surface::external_writes_done`] and
    /// [`release_surface_to_external`](crate::BlitCommandEncoder::release_surface_to_external)
    /// record nothing for such a surface. No keyed mutex is used, by rt or on the textures: each
    /// is made with `D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE`, not
    /// `D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX`, whose mutex the device that makes the texture and
    /// each device that opens it must acquire before rendering to it; the consumer orders the two
    /// APIs' work by its waits.
    ///
    /// **The consumer's precondition.** Direct3D 11's work that accesses a plane has completed
    /// before the consumer commits the command buffers that use it, and rt's command buffers that
    /// use a plane have completed before Direct3D 11 accesses it again. A plane rt uses before
    /// any Direct3D 11 access has undefined contents, as a Metal texture nothing wrote does.
    ///
    /// The surface has no buffer: [`Surface::buffer`] is [`Gap::D3d11SurfaceBuffer`], as its
    /// planes lie in memories of their own.
    ///
    /// Err: [`Refusal::SurfacePlanes`] for no plane or more than four; what
    /// [`Buffer::new_texture`](crate::Buffer::new_texture) refuses of a plane's descriptor, and of
    /// it with the image's format; [`Refusal::TextureViewFormat`] where the image's format and
    /// the descriptor's differ in their texel's bytes or Metal's view class; [`Limit::NoImport`]
    /// of [`ImportHandle::D3d11Texture`] where the device does not enable
    /// `VK_KHR_external_memory_win32` or its optimal images of the plane's formats and usage do
    /// not import the handle (`vkGetPhysicalDeviceImageFormatProperties2`); [`Gap::SurfaceUsage`]
    /// for a usage the descriptor asks that the import does not offer; [`Limit::ImportType`]
    /// where the handle's properties and the image's requirements share no memory type;
    /// [`Limit::MemoryAllocations`]; [`Error::Vulkan`] where the handle's properties, the image,
    /// its import or its binding fail; what the texture's index takes; [`Error::Quiesced`].
    pub fn import_d3d11_surface(&self, planes: &[D3d11Plane<'_>]) -> Result<Surface, Error> {
        let core = self.core();
        if core.closed() {
            return Err(Error::Quiesced);
        }
        if planes.is_empty() || planes.len() > MAX_PLANES {
            return Err(Error::Refused(Refusal::SurfacePlanes {
                planes: planes.len(),
                pitch_multiple: 1,
                start_multiple: 1,
            }));
        }
        let handle = ImportHandle::D3d11Texture;
        let Some(win32) = core.exports.win32() else {
            return Err(Error::Limit(Limit::NoImport { handle }));
        };
        let mut images: Vec<SurfaceImage> = Vec::with_capacity(planes.len());
        let discard = |images: Vec<SurfaceImage>| {
            for image in images {
                // SAFETY: nothing uses the image or its memory.
                unsafe { discard_surface_image(core, image) };
            }
        };
        for plane in planes {
            match self.d3d11_plane(core, win32, plane) {
                Ok(image) => images.push(image),
                Err(error) => {
                    discard(images);
                    return Err(error);
                }
            }
        }
        let unit = HeapId::unit(core);
        let mut textures = Vec::with_capacity(planes.len());
        let mut images = images.into_iter();
        for (plane, image) in planes.iter().zip(images.by_ref()) {
            match self.surface_texture(&plane.descriptor, plane.index, image, Arc::clone(&unit)) {
                Ok(texture) => textures.push(texture),
                Err(error) => {
                    discard(images.collect());
                    return Err(error);
                }
            }
        }
        Ok(Surface::imported(self.clone(), textures))
    }

    /// One plane's image, imported from its handle and bound, in the general layout.
    fn d3d11_plane(
        &self,
        core: &Arc<Core>,
        win32: &ash::khr::external_memory_win32::Device,
        plane: &D3d11Plane<'_>,
    ) -> Result<SurfaceImage, Error> {
        let handle = ImportHandle::D3d11Texture;
        let d = &plane.descriptor;
        let plan = self.plane_plan(d)?;
        let mut own = *d;
        own.pixel_format = plane.image_format;
        let image_plan = self.plane_plan(&own)?;
        let same_class = (plane.image_format == d.pixel_format)
            || formats::views_with_usage(plane.image_format, d.pixel_format);
        if image_plan.texel != plan.texel || !same_class {
            return Err(Error::Refused(Refusal::TextureViewFormat {
                texture: plane.image_format,
                view: d.pixel_format,
            }));
        }
        let (format, view) = (image_plan.format, plan.format);
        // The image's own format and the descriptor's, where they differ: the list the usage is
        // decided with.
        let pair: Vec<vk::Format> = match format == view {
            true => Vec::new(),
            false => vec![format, view],
        };
        let full = plane_list(format, view, &plan.view_formats);
        let (instance, physical) = (core.instance.vk(), core.physical);
        let mut view_features = vk::FormatProperties2::default();
        // SAFETY: `physical` is `instance`'s.
        unsafe {
            instance.get_physical_device_format_properties2(physical, view, &mut view_features)
        };
        let held = view_features.format_properties.optimal_tiling_features;
        let holds = |bit: vk::ImageUsageFlags| {
            let feature = match bit {
                vk::ImageUsageFlags::SAMPLED => vk::FormatFeatureFlags::SAMPLED_IMAGE,
                vk::ImageUsageFlags::COLOR_ATTACHMENT => vk::FormatFeatureFlags::COLOR_ATTACHMENT,
                vk::ImageUsageFlags::TRANSFER_SRC => vk::FormatFeatureFlags::TRANSFER_SRC,
                vk::ImageUsageFlags::TRANSFER_DST => vk::FormatFeatureFlags::TRANSFER_DST,
                vk::ImageUsageFlags::STORAGE => vk::FormatFeatureFlags::STORAGE_IMAGE,
                _ => return false,
            };
            held.contains(feature)
        };
        let imports = |usage: vk::ImageUsageFlags, list: &[vk::Format]| {
            (import_query(core, format, usage, list))
                .is_ok_and(|features| features.contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE))
        };
        let wanted = WANTED | plan.usage;
        let usage = (0..32)
            .map(|bit| vk::ImageUsageFlags::from_raw(1u32 << bit))
            .filter(|&bit| wanted.contains(bit) && holds(bit) && imports(bit, &pair))
            .fold(vk::ImageUsageFlags::empty(), |all, bit| all | bit);
        if usage.is_empty() || !imports(usage, &pair) {
            return Err(Error::Limit(Limit::NoImport { handle }));
        }
        // With the feedback usage where the plane asks it and the import offers it with the
        // rest, so a draw reads it as an attachment of its own pass through the loop; else
        // through its shadow.
        let looped = usage | vk::ImageUsageFlags::ATTACHMENT_FEEDBACK_LOOP_EXT;
        let usage = match plan.looped && imports(looped, &pair) {
            true => looped,
            false => usage,
        };
        let missing = plan.usage & !usage;
        if !missing.is_empty() {
            return Err(Error::Unsupported(Gap::SurfaceUsage {
                pixel_format: d.pixel_format,
                missing: missing.as_raw(),
            }));
        }
        // Every format the plane's texture views as, where the import takes that list with the
        // usage; else the pair, and a view of a format the pair does not list is refused at
        // the view (`TextureViewKind::SurfacePixelFormat`).
        let list = match full.len() > pair.len() && imports(usage, &full) {
            true => full,
            false => pair,
        };
        let device = &core.device;
        let mut external = vk::ExternalMemoryImageCreateInfo::default().handle_types(HANDLE);
        let mut listed = vk::ImageFormatListCreateInfo::default().view_formats(&list);
        let mut info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: d.width,
                height: d.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            // An external image is made undefined (`VUID-VkImageCreateInfo-pNext-01443`); after
            // Direct3D 11's access it is in the general layout (section 12.5.2).
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut external);
        if !list.is_empty() {
            info = info
                .flags(vk::ImageCreateFlags::MUTABLE_FORMAT)
                .push_next(&mut listed);
        }
        // SAFETY: the query accepted such an image for the handle; the extension is enabled.
        let image =
            unsafe { device.create_image(&info, None) }.map_err(Error::vulkan("vkCreateImage"))?;
        let destroy = || {
            // SAFETY: nothing uses the image, bound to nothing that outlives it.
            unsafe { device.destroy_image(image, None) };
        };
        // SAFETY: the image is the device's.
        let needs = unsafe { device.get_image_memory_requirements(image) };
        let raw = plane.handle.as_raw_handle() as vk::HANDLE;
        let mut properties = vk::MemoryWin32HandlePropertiesKHR::default();
        // SAFETY: the extension is enabled; the handle is the consumer's live NT handle of a
        // Direct3D 11 texture, borrowed for the call.
        let types =
            unsafe { win32.get_memory_win32_handle_properties(HANDLE, raw, &mut properties) };
        if let Err(result) = types {
            destroy();
            return Err(Error::vulkan("vkGetMemoryWin32HandlePropertiesKHR")(result));
        }
        let bits = properties.memory_type_bits & needs.memory_type_bits;
        let index = match (bits, bits & core.device_local) {
            (0, _) => {
                destroy();
                return Err(Error::Limit(Limit::ImportType {
                    handle,
                    offered: properties.memory_type_bits,
                    listed: needs.memory_type_bits,
                }));
            }
            (bits, 0) => bits.trailing_zeros(),
            (_, local) => local.trailing_zeros(),
        };
        let mut import = vk::ImportMemoryWin32HandleInfoKHR::default()
            .handle_type(HANDLE)
            .handle(raw);
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image);
        // The size of a Direct3D 11 texture's import is the system's (`allocationSize is
        // ignored`); the requirement's is named.
        let allocate = vk::MemoryAllocateInfo::default()
            .allocation_size(needs.size)
            .memory_type_index(index)
            .push_next(&mut import)
            .push_next(&mut dedicated);
        // SAFETY: an import of the handle, dedicated to the image, of a type both list.
        let memory = match unsafe {
            memory::allocate(
                device,
                &core.counters,
                core.max_allocations,
                &allocate,
                None,
            )
        } {
            Ok(memory) => memory,
            Err(error) => {
                destroy();
                return Err(error);
            }
        };
        core.counters.surface_imported(needs.size);
        // SAFETY: the memory is the image's dedicated import, of a type its requirements list.
        if let Err(result) = unsafe { device.bind_image_memory(image, memory, 0) } {
            destroy();
            // SAFETY: the image bound to nothing is destroyed.
            unsafe { device.free_memory(memory, None) };
            core.counters.fall(Gauge::Blocks);
            core.counters.surface_import_freed(needs.size);
            return Err(Error::vulkan("vkBindImageMemory")(result));
        }
        Ok(SurfaceImage {
            image,
            backing: Backing::Import(memory, needs.size),
            bytes: needs.size,
            usage,
            format,
            view_formats: list,
            block: None,
            d3d11: true,
            linear_attachment: false,
        })
    }
}

/// The formats a plane's image lists where the import takes them: its own `format`, the
/// descriptor's `view`, and `view_formats`, which the descriptor's texture lists for views of
/// another format (its own first), each once. Empty where that is the image's own format alone:
/// the image is then not mutable.
pub(crate) fn plane_list(
    format: vk::Format,
    view: vk::Format,
    view_formats: &[vk::Format],
) -> Vec<vk::Format> {
    let mut list = vec![format];
    for &listed in [view].iter().chain(view_formats) {
        if !list.contains(&listed) {
            list.push(listed);
        }
    }
    if list.len() == 1 {
        list.clear();
    }
    list
}

/// What `vkGetPhysicalDeviceImageFormatProperties2` answers of an optimal 2D image of `format`
/// and `usage` imported from a Direct3D 11 texture, mutable and listing `list` where it is not
/// empty: the handle's external memory features, or the query's error.
pub(crate) fn import_query(
    core: &Core,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    list: &[vk::Format],
) -> Result<vk::ExternalMemoryFeatureFlags, vk::Result> {
    let (instance, physical) = (core.instance.vk(), core.physical);
    let mut external = vk::PhysicalDeviceExternalImageFormatInfo::default().handle_type(HANDLE);
    let mut listed = vk::ImageFormatListCreateInfo::default().view_formats(list);
    let mut info = vk::PhysicalDeviceImageFormatInfo2::default()
        .format(format)
        .ty(vk::ImageType::TYPE_2D)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .push_next(&mut external);
    if !list.is_empty() {
        info = info
            .flags(vk::ImageCreateFlags::MUTABLE_FORMAT)
            .push_next(&mut listed);
    }
    let mut external_out = vk::ExternalImageFormatProperties::default();
    let mut out = vk::ImageFormatProperties2::default().push_next(&mut external_out);
    // SAFETY: `physical` is `instance`'s, and the handle's extension is enabled (the caller holds
    // the device's `VK_KHR_external_memory_win32`).
    unsafe { instance.get_physical_device_image_format_properties2(physical, &info, &mut out) }?;
    Ok(external_out
        .external_memory_properties
        .external_memory_features)
}
