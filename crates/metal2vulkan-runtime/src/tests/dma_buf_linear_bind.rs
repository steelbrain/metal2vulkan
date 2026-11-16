//! Whether a device binds a texture's linear image in the dma-buf import its pages already have,
//! and what that spares against a dma-buf of the texture's own pages: a measurement, not a check.
//! Ignored, printed as `LINBIND` lines.
//!
//! The pages are the probe's own: a sealed `memfd` mapped shared, one `udmabuf` of it imported
//! once, dedicated to nothing, with a `VkBuffer` made for dma-bufs bound over it whole, as a
//! no-copy buffer's pages or a guest RAM's chunk are imported (`dma_buf.rs`, `host_ram.rs`). The
//! device is the probe's own, made through the loader, with no runtime in the way:
//!
//! - **The driver's answers** (`LINBIND-QUERY`, `LINBIND-LAYOUT`, `LINBIND-SHARE`), per format of
//!   [`FORMATS`] and usage of [`usages`]: the image format query of a `VK_IMAGE_TILING_LINEAR` 2D
//!   image with `VkPhysicalDeviceExternalImageFormatInfo` of dma-bufs (an image made for a handle
//!   type the answer lists none of is invalid, `VUID-VkImageCreateInfo-pNext-00990`). Where it
//!   takes one, for each width of [`WIDTHS`], such an image made `UNDEFINED`
//!   (`VUID-VkImageCreateInfo-pNext-01443`): whether it requires or prefers a dedicated allocation
//!   (one that requires it binds at offset 0 of its own allocation alone,
//!   `VUID-vkBindImageMemory-image-01445`, `-memory-02628`), its memory's alignment, size and
//!   types, and its subresource's offset and row pitch beside Metal's: the width's bytes rounded
//!   up to 16, 64 and 256, and to the runtime's `minimumLinearTextureAlignmentForPixelFormat`
//!   answer on the same GPU.
//! - **The binding** (`LINBIND-BIND`), per format, at each width of [`BIND_WIDTHS`] by
//!   [`BIND_ROWS`] rows: the image bound in the import at an offset that meets its alignment, no
//!   multiple of the page where the alignment allows (`VUID-vkBindImageMemory-memory-02989`: it
//!   names the import's handle type; `-None-10735`, `-None-10737`), its rows' padding and the bytes
//!   around it set apart; first moved as the runtime moves an image over dma-buf pages (an acquire
//!   from `VK_QUEUE_FAMILY_FOREIGN_EXT`, `GENERAL` to `GENERAL`, `texture/first_move.rs`); then
//!   copied out (the texels the CPU wrote, read back), written by a copy, cleared, and rendered (a
//!   render pass that clears it, where its usage renders), each checked byte for byte through the
//!   pages' mapping: the texels as the driver's layout places them, the padding and the bytes
//!   around left alone.
//! - **First use** (`LINBIND-TIME`), per shape of [`TIMED`], [`ROUNDS`] rounds, the two ways in
//!   turn: from nothing to a first copy out of the texture waited for, today's way (a
//!   `DRM_FORMAT_MOD_LINEAR` image at the same pitch, a `udmabuf` of its pages, a dedicated
//!   import of it bound at 0, `dma_buf.rs`'s `dma_buf_image`) against the linear image bound in
//!   the import; each step's microseconds, and the release's.
//!
//! Run it in release under `--cfg m2v_gpu` (or `m2v_cpu_icd`) where `/dev/udmabuf` opens:
//! `cargo nextest run --release -p metal2vulkan-runtime --lib --run-ignored all
//! -E 'test(dma_buf_linear_bind)' --no-capture`.

use super::host_residency::{device, host_page, udmabuf, udmabuf_openable, Ram, Scratch};
use ash::vk;
use metal2vulkan::metal::PixelFormat;
use std::ffi::CStr;
use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd};
use std::time::Instant;

/// The formats asked: Vulkan's, its texel's bytes, and Metal's.
const FORMATS: [(vk::Format, u64, PixelFormat); 5] = [
    (vk::Format::B8G8R8A8_UNORM, 4, PixelFormat::BGRA8Unorm),
    (vk::Format::R8G8B8A8_UNORM, 4, PixelFormat::RGBA8Unorm),
    (vk::Format::R8_UNORM, 1, PixelFormat::R8Unorm),
    (vk::Format::R16G16B16A16_SFLOAT, 8, PixelFormat::RGBA16Float),
    (vk::Format::R32_SFLOAT, 4, PixelFormat::R32Float),
];

/// The widths each layout is asked at, in texels: no power of two, powers of two, screens'.
const WIDTHS: [u32; 16] = [
    1, 3, 17, 64, 65, 100, 129, 256, 640, 1000, 1024, 1280, 1366, 1920, 2560, 3840,
];

/// The rows of each image whose layout is asked.
const LAYOUT_ROWS: u32 = 4;

/// Metal's pitch alignments the driver's row pitch is set beside.
const METAL_ALIGNMENTS: [u64; 3] = [16, 64, 256];

/// The widths each format's image is bound and used at.
const BIND_WIDTHS: [u32; 3] = [100, 1366, 1920];

/// The rows of each bound image.
const BIND_ROWS: u32 = 33;

/// The shapes timed, `BGRA8Unorm`: a screen's, and about the 84 MiB of the in-place lane's case.
const TIMED: [[u32; 2]; 2] = [[1920, 1080], [5120, 4300]];

/// The rounds of each shape's two ways; the first is a warm-up, printed apart.
const ROUNDS: usize = 4;

/// The bytes of the probe's pages.
const RAM_BYTES: usize = 128 << 20;

/// Where the bound images lie (before their offset in the page).
const BIND_AT: u64 = 4 << 20;

/// Where the timed images lie.
const TIMED_AT: u64 = 16 << 20;

/// Where the copies out land, tightly packed.
const OUT_AT: u64 = 112 << 20;

/// Where the copies in come from, tightly packed.
const SOURCE_AT: u64 = 116 << 20;

/// The bytes before and after each bound image checked as left alone.
const GUARD: u64 = 512;

/// The byte of the rows' padding.
const PADDING: u8 = 0xa5;

/// The byte around the image.
const AROUND: u8 = 0x5a;

/// The texels of the timed first use's copy out, a square's side.
const FIRST_COPY: u32 = 64;

/// `DRM_FORMAT_MOD_LINEAR`.
const MOD_LINEAR: u64 = 0;

/// The dma-buf handle type.
const DMA_BUF: vk::ExternalMemoryHandleTypeFlags = vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT;

/// The usages asked: reading, reading and rendering, reading and a shader's writes.
fn usages() -> [(&'static str, vk::ImageUsageFlags); 3] {
    use vk::ImageUsageFlags as U;
    let read = U::SAMPLED | U::TRANSFER_SRC | U::TRANSFER_DST;
    [
        ("read", read),
        ("render", read | U::COLOR_ATTACHMENT),
        ("write", read | U::STORAGE),
    ]
}

/// `format`'s name, as the lines print it.
fn name(format: vk::Format) -> &'static str {
    match format {
        vk::Format::B8G8R8A8_UNORM => "BGRA8Unorm",
        vk::Format::R8G8B8A8_UNORM => "RGBA8Unorm",
        vk::Format::R8_UNORM => "R8Unorm",
        vk::Format::R16G16B16A16_SFLOAT => "RGBA16Float",
        vk::Format::R32_SFLOAT => "R32Float",
        _ => "another format",
    }
}

/// The bytes of a texel of `format` whose every channel is 1.0.
fn ones(format: vk::Format) -> Vec<u8> {
    match format {
        vk::Format::R16G16B16A16_SFLOAT => [0x00, 0x3c].repeat(4),
        vk::Format::R32_SFLOAT => vec![0x00, 0x00, 0x80, 0x3f],
        vk::Format::R8_UNORM => vec![0xff],
        _ => vec![0xff; 4],
    }
}

/// The byte `b` of row `r` of a pattern of `seed`: from 0x10 to 0x3f, so no float of the
/// formats' is a NaN or subnormal whatever byte of it the byte is.
fn pattern(seed: u64, r: u64, b: u64) -> u8 {
    let hash = (r * 131 + b * 7 + seed * 977).wrapping_mul(0x9e37_79b9) >> 7;
    0x10 + (hash % 0x30) as u8
}

/// `value` rounded up to a multiple of `to`.
fn up(value: u64, to: u64) -> u64 {
    value.div_ceil(to) * to
}

/// The median of `values`.
fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values.get(values.len() / 2).copied().unwrap_or(f64::NAN)
}

/// Microseconds since `from`.
fn us(from: Instant) -> f64 {
    from.elapsed().as_secs_f64() * 1e6
}

/// What the driver answers of an image: its memory requirements, whether it requires and prefers
/// a dedicated allocation, and its first subresource's layout.
struct Facts {
    requirements: vk::MemoryRequirements,
    requires: bool,
    prefers: bool,
    layout: vk::SubresourceLayout,
}

/// The probe's Vulkan device and its one queue's command buffer.
struct Probe {
    _entry: ash::Entry,
    instance: ash::Instance,
    physical: vk::PhysicalDevice,
    device: ash::Device,
    fd: ash::khr::external_memory_fd::Device,
    family: u32,
    queue: vk::Queue,
    pool: vk::CommandPool,
    commands: vk::CommandBuffer,
    fence: vk::Fence,
    /// Whether the device enables `VK_EXT_queue_family_foreign`.
    foreign: bool,
    /// Whether it enables `VK_EXT_image_drm_format_modifier`.
    modifiers: bool,
    memory: vk::PhysicalDeviceMemoryProperties,
}

/// The pages' import: its memory, its type, and the buffer bound over it whole.
struct Import {
    memory: vk::DeviceMemory,
    type_index: u32,
    buffer: vk::Buffer,
}

impl Probe {
    /// The device the cfg selects (a CPU device under `m2v_cpu_icd`, else the first other),
    /// with the dma-buf extensions; `None`, said, where it lists them not.
    fn new() -> Option<Probe> {
        let entry = crate::device::create::open_loader(None).unwrap();
        let application = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_2);
        let info = vk::InstanceCreateInfo::default().application_info(&application);
        // SAFETY: `info` and what it points at live for the call.
        let instance = unsafe { entry.create_instance(&info, None) }.unwrap();
        // SAFETY: the instance is live.
        let physicals = unsafe { instance.enumerate_physical_devices() }.unwrap();
        let cpu = cfg!(m2v_cpu_icd);
        let physical = physicals
            .into_iter()
            .find(|&physical| {
                // SAFETY: the instance's.
                let kind = unsafe { instance.get_physical_device_properties(physical) }.device_type;
                (kind == vk::PhysicalDeviceType::CPU) == cpu
            })
            .expect("a device of the kind the cfg selects");
        let mut driver = vk::PhysicalDeviceDriverProperties::default();
        let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut driver);
        // SAFETY: the instance's; Vulkan 1.1's command, 1.2's structure.
        unsafe { instance.get_physical_device_properties2(physical, &mut properties) };
        let name = (properties.properties.device_name_as_c_str())
            .map_or("?".to_owned(), |name| name.to_string_lossy().into_owned());
        let words = |name: Result<&CStr, _>| {
            name.map_or("?".to_owned(), |name: &CStr| {
                name.to_string_lossy().into_owned()
            })
        };
        let (driver_name, driver_info) = (
            words(driver.driver_name_as_c_str()),
            words(driver.driver_info_as_c_str()),
        );
        // SAFETY: the instance's.
        let listed = unsafe { instance.enumerate_device_extension_properties(physical) }.unwrap();
        let lists = |wanted: &CStr| {
            (listed.iter()).any(|extension| extension.extension_name_as_c_str() == Ok(wanted))
        };
        let needed = [
            ash::khr::external_memory_fd::NAME,
            ash::ext::external_memory_dma_buf::NAME,
        ];
        if !needed.iter().all(|&name| lists(name)) {
            println!("LINBIND-DEVICE {name}: lists no dma-buf imports: not run");
            return None;
        }
        let foreign = lists(ash::ext::queue_family_foreign::NAME);
        let modifiers = lists(ash::ext::image_drm_format_modifier::NAME);
        let mut enabled: Vec<*const std::ffi::c_char> =
            needed.iter().map(|name| name.as_ptr()).collect();
        if foreign {
            enabled.push(ash::ext::queue_family_foreign::NAME.as_ptr());
        }
        if modifiers {
            enabled.push(ash::ext::image_drm_format_modifier::NAME.as_ptr());
        }
        // SAFETY: the instance's.
        let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
        let family = (families.iter())
            .position(|family| family.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .expect("a graphics queue family") as u32;
        let priorities = [1.0];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        let info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&enabled);
        // SAFETY: `info` and what it points at live for the call.
        let device = unsafe { instance.create_device(physical, &info, None) }.unwrap();
        let fd = ash::khr::external_memory_fd::Device::new(&instance, &device);
        // SAFETY: the instance's.
        let memory = unsafe { instance.get_physical_device_memory_properties(physical) };
        // SAFETY: the device is live, the family and index were made; every info is valid.
        let (queue, pool, commands, fence) = unsafe {
            let queue = device.get_device_queue(family, 0);
            let pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(family)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .unwrap();
            let commands = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .command_buffer_count(1),
                )
                .unwrap()[0];
            let fence = (device.create_fence(&vk::FenceCreateInfo::default(), None)).unwrap();
            (queue, pool, commands, fence)
        };
        println!(
            "LINBIND-DEVICE {name}: driver {driver_name} ({driver_info}), queue family foreign \
             {foreign}, DRM format modifiers {modifiers}, page {}",
            host_page()
        );
        Some(Probe {
            _entry: entry,
            instance,
            physical,
            device,
            fd,
            family,
            queue,
            pool,
            commands,
            fence,
            foreign,
            modifiers,
            memory,
        })
    }

    /// The type of `bits` an import takes: host-visible and coherent first, as the runtime ranks
    /// a buffer's (`memory.rs`).
    fn type_of(&self, bits: u32) -> Option<u32> {
        let coherent =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let types = &self.memory.memory_types[..self.memory.memory_type_count as usize];
        let indices: Vec<u32> = (0..types.len() as u32)
            .filter(|index| bits & (1 << index) != 0)
            .collect();
        (indices.iter().copied())
            .find(|&index| types[index as usize].property_flags.contains(coherent))
            .or(indices.first().copied())
    }

    /// An import of the dma-buf `fd`, `size` bytes, of a type of `wanted` it takes, dedicated to
    /// `dedicated` where it is an image, with its type; why not.
    fn import(
        &self,
        fd: OwnedFd,
        size: u64,
        wanted: u32,
        dedicated: Option<vk::Image>,
    ) -> Result<(vk::DeviceMemory, u32), String> {
        let mut properties = vk::MemoryFdPropertiesKHR::default();
        // SAFETY: the device's; the descriptor is a dma-buf.
        unsafe { (self.fd).get_memory_fd_properties(DMA_BUF, fd.as_raw_fd(), &mut properties) }
            .map_err(|error| format!("the driver refuses the dma-buf ({error:?})"))?;
        let bits = properties.memory_type_bits & wanted;
        let index = (self.type_of(bits)).ok_or_else(|| {
            format!(
                "it imports as types {:#x}, none of {wanted:#x}",
                properties.memory_type_bits
            )
        })?;
        let raw = fd.into_raw_fd();
        let mut import = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(DMA_BUF)
            .fd(raw);
        let mut alone = vk::MemoryDedicatedAllocateInfo::default();
        let mut info = vk::MemoryAllocateInfo::default()
            .allocation_size(size)
            .memory_type_index(index)
            .push_next(&mut import);
        if let Some(image) = dedicated {
            alone = alone.image(image);
            info = info.push_next(&mut alone);
        }
        // SAFETY: `info` and its chain live for the call; the import takes the descriptor where
        // it succeeds.
        let memory = unsafe { self.device.allocate_memory(&info, None) }.map_err(|error| {
            // SAFETY: the import failed, so the descriptor is still ours.
            drop(unsafe { OwnedFd::from_raw_fd(raw) });
            format!("the driver refuses the import ({error:?})")
        })?;
        Ok((memory, index))
    }

    /// The pages' import: `fd`, a dma-buf of them `size` bytes long, imported dedicated to
    /// nothing, with a buffer made for dma-bufs bound over it whole.
    fn buffer_import(&self, fd: OwnedFd, size: u64) -> Import {
        let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(DMA_BUF);
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
            .push_next(&mut external);
        // SAFETY: `info` and its chain live for the call.
        let buffer = unsafe { self.device.create_buffer(&info, None) }.unwrap();
        // SAFETY: the buffer is live.
        let requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let (memory, type_index) =
            (self.import(fd, size, requirements.memory_type_bits, None)).unwrap();
        // SAFETY: the buffer is unbound; the memory takes it whole, of a type it takes.
        unsafe { self.device.bind_buffer_memory(buffer, memory, 0) }.unwrap();
        Import {
            memory,
            type_index,
            buffer,
        }
    }

    /// The image format query of a 2D image of `format`, `usage` and `tiling` (of
    /// `DRM_FORMAT_MOD_LINEAR` where that is the modifier's) for dma-bufs: its external memory
    /// properties and limits, or the driver's error.
    fn query(
        &self,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        tiling: vk::ImageTiling,
    ) -> Result<(vk::ExternalMemoryProperties, vk::ImageFormatProperties), vk::Result> {
        let mut external_info =
            vk::PhysicalDeviceExternalImageFormatInfo::default().handle_type(DMA_BUF);
        let mut modifier = vk::PhysicalDeviceImageDrmFormatModifierInfoEXT::default()
            .drm_format_modifier(MOD_LINEAR)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let mut info = vk::PhysicalDeviceImageFormatInfo2::default()
            .format(format)
            .ty(vk::ImageType::TYPE_2D)
            .tiling(tiling)
            .usage(usage)
            .push_next(&mut external_info);
        if tiling == vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT {
            info = info.push_next(&mut modifier);
        }
        let mut external = vk::ExternalImageFormatProperties::default();
        let mut properties = vk::ImageFormatProperties2::default().push_next(&mut external);
        // SAFETY: the instance's; `info` and both chains live for the call.
        unsafe {
            (self.instance).get_physical_device_image_format_properties2(
                self.physical,
                &info,
                &mut properties,
            )
        }?;
        let limits = properties.image_format_properties;
        Ok((external.external_memory_properties, limits))
    }

    /// Whether the query takes dma-bufs for such an image, and its words.
    fn takes(
        &self,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        tiling: vk::ImageTiling,
    ) -> (bool, String) {
        match self.query(format, usage, tiling) {
            Ok((external, _)) => {
                let features = external.external_memory_features;
                let takes = features.contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE)
                    && external.compatible_handle_types.contains(DMA_BUF);
                let words = format!(
                    "features {:#x} compatible {:#x}",
                    features.as_raw(),
                    external.compatible_handle_types.as_raw()
                );
                (takes, words)
            }
            Err(error) => (false, format!("{error:?}")),
        }
    }

    /// A linear image of `format`, `usage` and `extent` made for dma-bufs, `UNDEFINED`.
    fn linear_image(
        &self,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        [width, height]: [u32; 2],
    ) -> vk::Image {
        let mut external = vk::ExternalMemoryImageCreateInfo::default().handle_types(DMA_BUF);
        let info = image_info(format, usage, [width, height])
            .tiling(vk::ImageTiling::LINEAR)
            .push_next(&mut external);
        // SAFETY: `info` and its chain live for the call; the query took the image's handle type.
        unsafe { self.device.create_image(&info, None) }.unwrap()
    }

    /// A `DRM_FORMAT_MOD_LINEAR` image of `format`, `usage` and `extent` made for dma-bufs, its
    /// one plane at `offset` with rows `pitch` apart, `UNDEFINED`; or the driver's error.
    fn modifier_image(
        &self,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        extent: [u32; 2],
        (offset, pitch): (u64, u64),
    ) -> Result<vk::Image, vk::Result> {
        let planes = [vk::SubresourceLayout {
            offset,
            size: 0,
            row_pitch: pitch,
            array_pitch: 0,
            depth_pitch: 0,
        }];
        let mut explicit = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
            .drm_format_modifier(MOD_LINEAR)
            .plane_layouts(&planes);
        let mut external = vk::ExternalMemoryImageCreateInfo::default().handle_types(DMA_BUF);
        let info = image_info(format, usage, extent)
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .push_next(&mut external)
            .push_next(&mut explicit);
        // SAFETY: `info` and its chain live for the call; the query took such an image.
        unsafe { self.device.create_image(&info, None) }
    }

    /// The driver's answers of `image`, its layout asked of `aspect`.
    fn facts(&self, image: vk::Image, aspect: vk::ImageAspectFlags) -> Facts {
        let mut dedicated = vk::MemoryDedicatedRequirements::default();
        let mut requirements = vk::MemoryRequirements2::default().push_next(&mut dedicated);
        let query = vk::ImageMemoryRequirementsInfo2::default().image(image);
        let subresource = vk::ImageSubresource {
            aspect_mask: aspect,
            mip_level: 0,
            array_layer: 0,
        };
        // SAFETY: the image is live; the query is core in Vulkan 1.1; the aspect is its
        // tiling's (color of a linear image, the memory plane of a modifier's).
        let layout = unsafe {
            (self.device).get_image_memory_requirements2(&query, &mut requirements);
            self.device.get_image_subresource_layout(image, subresource)
        };
        Facts {
            requirements: requirements.memory_requirements,
            requires: dedicated.requires_dedicated_allocation != vk::FALSE,
            prefers: dedicated.prefers_dedicated_allocation != vk::FALSE,
            layout,
        }
    }

    /// Record `record` into the command buffer, submit it, and wait: the microseconds of the
    /// submission call and from its return to the fence.
    fn submit(&self, record: impl FnOnce(vk::CommandBuffer)) -> (f64, f64) {
        let device = &self.device;
        let commands = self.commands;
        // SAFETY: the command buffer is the pool's and idle (each submission is waited for);
        // `record` records valid commands of live objects.
        unsafe {
            device
                .reset_command_buffer(commands, vk::CommandBufferResetFlags::empty())
                .unwrap();
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device.begin_command_buffer(commands, &begin).unwrap();
        }
        record(commands);
        let buffers = [commands];
        let submit = [vk::SubmitInfo::default().command_buffers(&buffers)];
        // SAFETY: the command buffer is recording; the fence is unsignaled; it ends and is
        // submitted, waited for and the fence reset.
        unsafe {
            device.end_command_buffer(commands).unwrap();
            let called = Instant::now();
            device
                .queue_submit(self.queue, &submit, self.fence)
                .unwrap();
            let submitted = us(called);
            let returned = Instant::now();
            device
                .wait_for_fences(&[self.fence], true, 10_000_000_000)
                .unwrap();
            let waited = us(returned);
            device.reset_fences(&[self.fence]).unwrap();
            (submitted, waited)
        }
    }

    /// Record the first move of `image`: an acquire from `VK_QUEUE_FAMILY_FOREIGN_EXT` (from
    /// `VK_QUEUE_FAMILY_EXTERNAL` where the device enables none), `GENERAL` to `GENERAL`.
    fn acquire(&self, commands: vk::CommandBuffer, image: vk::Image) {
        let source = match self.foreign {
            true => vk::QUEUE_FAMILY_FOREIGN_EXT,
            false => vk::QUEUE_FAMILY_EXTERNAL,
        };
        let barrier = vk::ImageMemoryBarrier::default()
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(source)
            .dst_queue_family_index(self.family)
            .image(image)
            .subresource_range(color_range());
        // SAFETY: the command buffer is recording, outside a render pass; the image is live.
        unsafe {
            self.device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            )
        };
    }

    /// Record a barrier from every command's writes to the host's reads.
    fn to_host(&self, commands: vk::CommandBuffer) {
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ);
        // SAFETY: the command buffer is recording, outside a render pass.
        unsafe {
            self.device.cmd_pipeline_barrier(
                commands,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            )
        };
    }

    /// Record a copy of `image`'s `extent` texels from its origin into `buffer` at `at`, tightly.
    fn copy_out(
        &self,
        commands: vk::CommandBuffer,
        (image, extent): (vk::Image, [u32; 2]),
        (buffer, at): (vk::Buffer, u64),
    ) {
        let region = region(at, extent);
        // SAFETY: the command buffer is recording, outside a render pass; the image is live in
        // `GENERAL` with transfer-source usage, the buffer live with transfer-destination usage
        // and the region inside it.
        unsafe {
            self.device.cmd_copy_image_to_buffer(
                commands,
                image,
                vk::ImageLayout::GENERAL,
                buffer,
                &[region],
            )
        };
    }

    /// Clear `image` (of `format` and `extent`) to 1.0 in a render pass of it alone, waited for.
    fn render_ones(&self, image: vk::Image, format: vk::Format, [width, height]: [u32; 2]) {
        let device = &self.device;
        let attachment = [vk::AttachmentDescription::default()
            .format(format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::GENERAL)
            .final_layout(vk::ImageLayout::GENERAL)];
        let reference = [vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::GENERAL,
        }];
        let subpass = [vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&reference)];
        let pass_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachment)
            .subpasses(&subpass);
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(color_range());
        // SAFETY: every info is valid and lives for its call; the image is live, with color
        // attachment usage.
        let (pass, view) = unsafe {
            (
                device.create_render_pass(&pass_info, None).unwrap(),
                device.create_image_view(&view_info, None).unwrap(),
            )
        };
        let views = [view];
        let framebuffer_info = vk::FramebufferCreateInfo::default()
            .render_pass(pass)
            .attachments(&views)
            .width(width)
            .height(height)
            .layers(1);
        // SAFETY: the pass and view are live and match.
        let framebuffer = unsafe { device.create_framebuffer(&framebuffer_info, None) }.unwrap();
        let clear = [vk::ClearValue {
            color: vk::ClearColorValue { float32: [1.0; 4] },
        }];
        let begin = vk::RenderPassBeginInfo::default()
            .render_pass(pass)
            .framebuffer(framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: vk::Extent2D { width, height },
            })
            .clear_values(&clear);
        self.submit(|commands| {
            // SAFETY: the command buffer is recording, outside a render pass; the pass,
            // framebuffer and image are live, the image in `GENERAL`.
            unsafe {
                device.cmd_begin_render_pass(commands, &begin, vk::SubpassContents::INLINE);
                device.cmd_end_render_pass(commands);
            }
            self.to_host(commands);
        });
        // SAFETY: the submission that used them completed.
        unsafe {
            device.destroy_framebuffer(framebuffer, None);
            device.destroy_image_view(view, None);
            device.destroy_render_pass(pass, None);
        }
    }
}

impl Drop for Probe {
    fn drop(&mut self) {
        // SAFETY: every submission was waited for; the objects are the device's and go once.
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// A 2D image of one level and layer of `format`, `usage` and `extent`, `UNDEFINED`, exclusive.
fn image_info<'a>(
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    [width, height]: [u32; 2],
) -> vk::ImageCreateInfo<'a> {
    vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
}

/// The first level and layer of a color image.
fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

/// A copy region of an image's `extent` texels from its origin, the buffer's from `at`, tight.
fn region(at: u64, [width, height]: [u32; 2]) -> vk::BufferImageCopy {
    vk::BufferImageCopy {
        buffer_offset: at,
        buffer_row_length: 0,
        buffer_image_height: 0,
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        image_offset: vk::Offset3D::default(),
        image_extent: vk::Extent3D {
            width,
            height,
            depth: 1,
        },
    }
}

/// The runtime's `minimumLinearTextureAlignmentForPixelFormat` answer of each of [`FORMATS`] on
/// the device the cfg selects.
fn runtime_alignments() -> Vec<u64> {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    let alignments = FORMATS
        .iter()
        .map(|&(_, _, metal)| {
            (device.minimum_linear_texture_alignment_for_pixel_format(metal)).unwrap()
        })
        .collect();
    drop(device);
    alignments
}

/// The pages' bytes: the probe's view of them, which no command reads or writes while the probe
/// does (each submission is waited for).
struct Pages<'a>(&'a Ram);

impl Pages<'_> {
    /// The address `at` bytes in, of `len` bytes inside the view.
    fn at(&self, at: u64, len: u64) -> *mut u8 {
        assert!((at + len) as usize <= RAM_BYTES);
        self.0.pointer(at as usize).as_ptr()
    }

    /// A copy of the `len` bytes from `at`.
    fn read(&self, at: u64, len: u64) -> Vec<u8> {
        let mut bytes = vec![0; len as usize];
        // SAFETY: inside the view, which lives while `self.0` does; no command writes them now.
        unsafe { std::ptr::copy_nonoverlapping(self.at(at, len), bytes.as_mut_ptr(), bytes.len()) };
        bytes
    }

    /// Write `bytes` from `at`.
    fn write(&self, at: u64, bytes: &[u8]) {
        let to = self.at(at, bytes.len() as u64);
        // SAFETY: inside the view, which lives while `self.0` does; no command reads them now.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), to, bytes.len()) };
    }

    /// Write `len` bytes of `value` from `at`.
    fn fill(&self, at: u64, len: u64, value: u8) {
        // SAFETY: as `write`.
        unsafe { std::ptr::write_bytes(self.at(at, len), value, len as usize) };
    }

    /// Lay out an image's bytes at `at`: `rows` rows `pitch` apart, of `row` bytes of texels of
    /// the pattern of `seed`, their padding [`PADDING`], and [`AROUND`] from [`GUARD`] before
    /// them to `size` bytes and [`GUARD`] after.
    fn lay(&self, at: u64, (row, pitch, rows): (u64, u64, u64), size: u64, seed: u64) {
        self.fill(at - GUARD, GUARD, AROUND);
        let end = rows * pitch;
        self.fill(at + end, size.max(end) - end + GUARD, AROUND);
        for r in 0..rows {
            let line: Vec<u8> = (0..pitch)
                .map(|b| {
                    if b < row {
                        pattern(seed, r, b)
                    } else {
                        PADDING
                    }
                })
                .collect();
            self.write(at + r * pitch, &line);
        }
    }

    /// The first byte that differs from the image's bytes laid out by [`lay`](Self::lay) with
    /// texels of `texel(r, b)`: where from `at`, what it is and what it should be.
    fn differs(
        &self,
        at: u64,
        (row, pitch, rows): (u64, u64, u64),
        size: u64,
        texel: impl Fn(u64, u64) -> u8,
    ) -> Option<(i64, u8, u8)> {
        let before = self.read(at - GUARD, GUARD);
        if let Some(i) = before.iter().position(|&byte| byte != AROUND) {
            return Some((i as i64 - GUARD as i64, before[i], AROUND));
        }
        let end = rows * pitch;
        let after = self.read(at + end, size.max(end) - end + GUARD);
        if let Some(i) = after.iter().position(|&byte| byte != AROUND) {
            return Some(((end + i as u64) as i64, after[i], AROUND));
        }
        let lines = self.read(at, end);
        for r in 0..rows {
            for b in 0..pitch {
                let byte = lines[(r * pitch + b) as usize];
                let should = if b < row { texel(r, b) } else { PADDING };
                if byte != should {
                    return Some(((r * pitch + b) as i64, byte, should));
                }
            }
        }
        None
    }
}

/// What a check found, in words.
fn verdict(differs: Option<(i64, u8, u8)>) -> String {
    match differs {
        None => "ok".to_owned(),
        Some((at, byte, should)) => format!("FAIL(byte {at}: {byte:#04x} not {should:#04x})"),
    }
}

/// The driver's answers of each format's linear images for dma-bufs (`LINBIND-QUERY`,
/// `LINBIND-LAYOUT`, `LINBIND-SHARE`).
fn answers(probe: &Probe, alignments: &[u64]) {
    for (&(format, texel, _), &runtime) in FORMATS.iter().zip(alignments) {
        let named = name(format);
        // SAFETY: the instance's.
        let features = unsafe {
            (probe.instance).get_physical_device_format_properties(probe.physical, format)
        };
        for (usage_name, usage) in usages() {
            let linear = vk::ImageTiling::LINEAR;
            let (takes, words) = probe.takes(format, usage, linear);
            let modifier = match probe.modifiers {
                true => {
                    probe
                        .takes(format, usage, vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
                        .1
                }
                false => "not enabled".to_owned(),
            };
            println!(
                "LINBIND-QUERY {named} {usage_name}: linear tiling features {:#x}; linear for \
                 dma-bufs: {words}; DRM_FORMAT_MOD_LINEAR for dma-bufs: {modifier}",
                features.linear_tiling_features.as_raw()
            );
            if probe.modifiers {
                modifier_answers(probe, (format, texel), (usage_name, usage));
            }
            if !takes {
                continue;
            }
            let mut matched = [0; METAL_ALIGNMENTS.len() + 1];
            for width in WIDTHS {
                let image = probe.linear_image(format, usage, [width, LAYOUT_ROWS]);
                let facts = probe.facts(image, vk::ImageAspectFlags::COLOR);
                // SAFETY: the image is live and unbound, and no command uses it.
                unsafe { probe.device.destroy_image(image, None) };
                let tight = u64::from(width) * texel;
                let pitch = facts.layout.row_pitch;
                let rule = METAL_ALIGNMENTS.iter().chain([&runtime]);
                let equal: Vec<String> = (rule.enumerate())
                    .map(|(i, &alignment)| {
                        let metal = up(tight, alignment);
                        let same = metal == pitch && facts.layout.offset == 0;
                        matched[i] += usize::from(same);
                        format!("{alignment}:{}", if same { "=" } else { "x" })
                    })
                    .collect();
                println!(
                    "LINBIND-LAYOUT {named} {usage_name} width {width}: row pitch {pitch} \
                     (texels {tight}), offset {}, size {} (rows x pitch {}), alignment {}, types \
                     {:#x}, requires dedicated {}, prefers dedicated {}; Metal's pitch at {} \
                     (runtime {runtime})",
                    facts.layout.offset,
                    facts.requirements.size,
                    u64::from(LAYOUT_ROWS) * pitch,
                    facts.requirements.alignment,
                    facts.requirements.memory_type_bits,
                    facts.requires,
                    facts.prefers,
                    equal.join(" ")
                );
            }
            println!(
                "LINBIND-SHARE {named} {usage_name}: widths whose Metal pitch is the driver's, \
                 of {}: at 16 {}, at 64 {}, at 256 {}, at the runtime's {runtime} {}",
                WIDTHS.len(),
                matched[0],
                matched[1],
                matched[2],
                matched[3]
            );
        }
    }
}

/// Whether the driver binds a `DRM_FORMAT_MOD_LINEAR` image of `format` (texels of `texel`
/// bytes) and `usage` for dma-bufs at an offset of memory it shares, as an image made for the
/// buffer's import would be (`LINBIND-MODIFIER`): per width of [`BIND_WIDTHS`], its plane at
/// offset 0 with rows the texels' bytes rounded up to 256 apart, whether it requires or prefers a
/// dedicated allocation, its alignment and size, and its plane as laid out.
fn modifier_answers(
    probe: &Probe,
    (format, texel): (vk::Format, u64),
    (usage_name, usage): (&str, vk::ImageUsageFlags),
) {
    if !probe
        .takes(format, usage, vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
        .0
    {
        return;
    }
    let named = name(format);
    for width in BIND_WIDTHS {
        let pitch = up(u64::from(width) * texel, 256);
        let what = format!("LINBIND-MODIFIER {named} {usage_name} width {width} pitch {pitch}");
        match probe.modifier_image(format, usage, [width, LAYOUT_ROWS], (0, pitch)) {
            Ok(image) => {
                let facts = probe.facts(image, vk::ImageAspectFlags::MEMORY_PLANE_0_EXT);
                // SAFETY: the image is live and unbound, and no command uses it.
                unsafe { probe.device.destroy_image(image, None) };
                println!(
                    "{what}: requires dedicated {}, prefers dedicated {}, alignment {}, size {}, \
                     plane laid at {} with rows {} apart",
                    facts.requires,
                    facts.prefers,
                    facts.requirements.alignment,
                    facts.requirements.size,
                    facts.layout.offset,
                    facts.layout.row_pitch
                );
            }
            Err(error) => println!("{what}: refused ({error:?})"),
        }
    }
}

/// Each format's linear image bound in the import and used (`LINBIND-BIND`).
fn binds(probe: &Probe, import: &Import, pages: &Pages<'_>) {
    let page = host_page() as u64;
    for &(format, texel, _) in &FORMATS {
        let named = name(format);
        let [(_, read), (_, render), _] = usages();
        let renders = probe.takes(format, render, vk::ImageTiling::LINEAR).0;
        let usage = if renders { render } else { read };
        if !probe.takes(format, usage, vk::ImageTiling::LINEAR).0 {
            println!("LINBIND-BIND {named}: the query takes no dma-buf for its linear image");
            continue;
        }
        for width in BIND_WIDTHS {
            let extent = [width, BIND_ROWS];
            let image = probe.linear_image(format, usage, extent);
            let facts = probe.facts(image, vk::ImageAspectFlags::COLOR);
            let alignment = facts.requirements.alignment.max(1);
            // An offset that meets the alignment, and lies inside a page where it can.
            let at = match alignment < page {
                true => BIND_AT + alignment.max(GUARD),
                false => up(BIND_AT, alignment),
            };
            let at = up(at, alignment);
            let size = facts.requirements.size;
            let shape = (
                u64::from(width) * texel,
                facts.layout.row_pitch,
                u64::from(BIND_ROWS),
            );
            let what = format!(
                "LINBIND-BIND {named} {width}x{BIND_ROWS} {}: at {at} (page offset {}), pitch \
                 {}, size {size}, requires dedicated {}",
                if renders { "render" } else { "read" },
                at % page,
                facts.layout.row_pitch,
                facts.requires
            );
            let typed = facts.requirements.memory_type_bits & (1 << import.type_index) != 0;
            assert!(
                at + size <= OUT_AT,
                "the bound image fits before the copies"
            );
            if facts.requires || facts.layout.offset != 0 || !typed {
                // SAFETY: the image is live and unbound, and no command uses it.
                unsafe { probe.device.destroy_image(image, None) };
                println!(
                    "{what}: not bound (dedicated, its subresource at offset {}, or types {:#x} \
                     without the import's {})",
                    facts.layout.offset, facts.requirements.memory_type_bits, import.type_index
                );
                continue;
            }
            // SAFETY: the image is unbound and made for dma-bufs, the import's handle type; the
            // memory is no dedicated allocation, of a type its bits take, the offset a multiple
            // of its alignment and its size inside the memory.
            let bound = unsafe { probe.device.bind_image_memory(image, import.memory, at) };
            if let Err(error) = bound {
                // SAFETY: as above.
                unsafe { probe.device.destroy_image(image, None) };
                println!("{what}: the driver refuses the binding ({error:?})");
                continue;
            }
            // The texels the CPU wrote, read back through a copy (with the first move).
            pages.lay(at, shape, size, 1);
            probe.submit(|commands| {
                probe.acquire(commands, image);
                probe.copy_out(commands, (image, extent), (import.buffer, OUT_AT));
                probe.to_host(commands);
            });
            let row = shape.0;
            let out = pages.read(OUT_AT, row * u64::from(BIND_ROWS));
            let read_back = (0..u64::from(BIND_ROWS))
                .flat_map(|r| (0..row).map(move |b| (r, b)))
                .find(|&(r, b)| out[(r * row + b) as usize] != pattern(1, r, b));
            let read_words = match read_back {
                None => "ok".to_owned(),
                Some((r, b)) => format!("FAIL(row {r} byte {b})"),
            };
            let around = verdict(pages.differs(at, shape, size, |r, b| pattern(1, r, b)));
            // Written by a copy of a pattern of its own.
            pages.lay(at, shape, size, 1);
            let source: Vec<u8> = (0..u64::from(BIND_ROWS))
                .flat_map(|r| (0..row).map(move |b| pattern(2, r, b)))
                .collect();
            pages.write(SOURCE_AT, &source);
            probe.submit(|commands| {
                let region = region(SOURCE_AT, extent);
                // SAFETY: the command buffer is recording, outside a render pass; the image is
                // live in `GENERAL` with transfer-destination usage, the region inside the buffer.
                unsafe {
                    probe.device.cmd_copy_buffer_to_image(
                        commands,
                        import.buffer,
                        image,
                        vk::ImageLayout::GENERAL,
                        &[region],
                    )
                };
                probe.to_host(commands);
            });
            let written = verdict(pages.differs(at, shape, size, |r, b| pattern(2, r, b)));
            // Cleared, and rendered.
            let one = ones(format);
            let is_one = |_: u64, b: u64| one[(b % texel) as usize];
            pages.lay(at, shape, size, 1);
            probe.submit(|commands| {
                let value = vk::ClearColorValue { float32: [1.0; 4] };
                // SAFETY: the command buffer is recording, outside a render pass; the image is
                // live in `GENERAL` with transfer-destination usage.
                unsafe {
                    probe.device.cmd_clear_color_image(
                        commands,
                        image,
                        vk::ImageLayout::GENERAL,
                        &value,
                        &[color_range()],
                    )
                };
                probe.to_host(commands);
            });
            let cleared = verdict(pages.differs(at, shape, size, is_one));
            let rendered = match renders {
                true => {
                    pages.lay(at, shape, size, 1);
                    probe.render_ones(image, format, extent);
                    verdict(pages.differs(at, shape, size, is_one))
                }
                false => "no render usage".to_owned(),
            };
            // SAFETY: every submission that used it completed.
            unsafe { probe.device.destroy_image(image, None) };
            println!(
                "{what}: read {read_words} (bytes around {around}), written {written}, cleared \
                 {cleared}, rendered {rendered}"
            );
        }
    }
}

/// One first use's microseconds: the image made and checked, its pages' dma-buf, their import,
/// the binding, the submission call, the wait, and the release after.
#[derive(Default)]
struct Steps {
    image: f64,
    dma_buf: f64,
    import: f64,
    bind: f64,
    submit: f64,
    wait: f64,
    release: f64,
}

impl Steps {
    /// Its steps to the first use's end.
    fn total(&self) -> f64 {
        self.image + self.dma_buf + self.import + self.bind + self.submit + self.wait
    }
}

/// A first use of a `BGRA8Unorm` texture of `extent` whose texels lie at `at` rows `pitch`
/// apart, today's way: a `DRM_FORMAT_MOD_LINEAR` image at that pitch, a `udmabuf` of its pages,
/// a dedicated import of it bound at 0; why not.
fn today(
    probe: &Probe,
    (ram, import): (&Ram, &Import),
    extent: [u32; 2],
    (at, pitch): (u64, u64),
) -> Result<Steps, String> {
    let page = host_page() as u64;
    let usage = usages()[0].1;
    let format = vk::Format::B8G8R8A8_UNORM;
    if !probe
        .takes(format, usage, vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
        .0
    {
        return Err("the query takes no dma-buf for its DRM_FORMAT_MOD_LINEAR image".to_owned());
    }
    let mut steps = Steps::default();
    let started = Instant::now();
    let offset = at % page;
    let image = (probe.modifier_image(format, usage, extent, (offset, pitch)))
        .map_err(|error| format!("the modifier image is refused ({error:?})"))?;
    let facts = probe.facts(image, vk::ImageAspectFlags::MEMORY_PLANE_0_EXT);
    steps.image = us(started);
    let release = |memory: Option<vk::DeviceMemory>| {
        // SAFETY: no command uses them any more.
        unsafe {
            probe.device.destroy_image(image, None);
            if let Some(memory) = memory {
                probe.device.free_memory(memory, None);
            }
        }
    };
    if facts.layout.offset != offset || facts.layout.row_pitch != pitch {
        release(None);
        return Err(format!(
            "the driver lays the plane at {} with rows {} apart",
            facts.layout.offset, facts.layout.row_pitch
        ));
    }
    let started = Instant::now();
    let length = up(offset + facts.requirements.size, page);
    let fd = udmabuf(ram.file(), at - offset, length).map_err(|error| {
        release(None);
        format!("udmabuf: {error}")
    })?;
    steps.dma_buf = us(started);
    let started = Instant::now();
    let bits = facts.requirements.memory_type_bits;
    let size = facts.requirements.size;
    let (memory, _) = (probe.import(fd, size, bits, Some(image))).inspect_err(|_| release(None))?;
    steps.import = us(started);
    let started = Instant::now();
    // SAFETY: the image is unbound; the memory is its dedicated import, of its size.
    unsafe { probe.device.bind_image_memory(image, memory, 0) }.unwrap();
    steps.bind = us(started);
    (steps.submit, steps.wait) = probe.submit(|commands| {
        probe.acquire(commands, image);
        let first = [FIRST_COPY, FIRST_COPY];
        probe.copy_out(commands, (image, first), (import.buffer, OUT_AT));
        probe.to_host(commands);
    });
    let started = Instant::now();
    release(Some(memory));
    steps.release = us(started);
    Ok(steps)
}

/// The same first use, the driver's linear image for dma-bufs bound at `at` of the pages' import;
/// why not.
fn linear(
    probe: &Probe,
    import: &Import,
    extent: [u32; 2],
    (at, pitch): (u64, u64),
) -> Result<Steps, String> {
    let usage = usages()[0].1;
    let format = vk::Format::B8G8R8A8_UNORM;
    let mut steps = Steps::default();
    let started = Instant::now();
    let image = probe.linear_image(format, usage, extent);
    let facts = probe.facts(image, vk::ImageAspectFlags::COLOR);
    steps.image = us(started);
    let release = || {
        // SAFETY: no command uses it any more.
        unsafe { probe.device.destroy_image(image, None) };
    };
    let alignment = facts.requirements.alignment.max(1);
    let typed = facts.requirements.memory_type_bits & (1 << import.type_index) != 0;
    let fits = at + facts.requirements.size <= OUT_AT;
    if facts.requires || facts.layout.row_pitch != pitch || !at.is_multiple_of(alignment) {
        release();
        return Err(format!(
            "requires dedicated {}, pitch {}, alignment {alignment}",
            facts.requires, facts.layout.row_pitch
        ));
    }
    if !typed || !fits {
        release();
        return Err(format!(
            "types {:#x} without the import's {}, or {} bytes past the pages",
            facts.requirements.memory_type_bits, import.type_index, facts.requirements.size
        ));
    }
    let started = Instant::now();
    // SAFETY: the image is unbound and made for dma-bufs; the memory is no dedicated allocation,
    // of a type it takes, the offset a multiple of its alignment and its size inside it.
    unsafe { probe.device.bind_image_memory(image, import.memory, at) }.unwrap();
    steps.bind = us(started);
    (steps.submit, steps.wait) = probe.submit(|commands| {
        probe.acquire(commands, image);
        let first = [FIRST_COPY, FIRST_COPY];
        probe.copy_out(commands, (image, first), (import.buffer, OUT_AT));
        probe.to_host(commands);
    });
    let started = Instant::now();
    release();
    steps.release = us(started);
    Ok(steps)
}

/// The same first use, a `DRM_FORMAT_MOD_LINEAR` image whose plane lies at `at` of the pages'
/// whole dma-buf (`kept`, a descriptor of the one the import was made of), bound to a dedicated
/// import of that dma-buf again (Vulkan imports a payload any number of times into an instance,
/// each a memory of its own: Memory Allocation, "Importing Memory Using a File Descriptor"): no
/// new dma-buf, though an import of its own; why not.
fn reimport(
    probe: &Probe,
    (kept, import): (&OwnedFd, &Import),
    extent: [u32; 2],
    (at, pitch): (u64, u64),
) -> Result<Steps, String> {
    let usage = usages()[0].1;
    let format = vk::Format::B8G8R8A8_UNORM;
    if !probe
        .takes(format, usage, vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
        .0
    {
        return Err("the query takes no dma-buf for its DRM_FORMAT_MOD_LINEAR image".to_owned());
    }
    let mut steps = Steps::default();
    let started = Instant::now();
    let image = (probe.modifier_image(format, usage, extent, (at, pitch)))
        .map_err(|error| format!("the modifier image at {at} is refused ({error:?})"))?;
    let facts = probe.facts(image, vk::ImageAspectFlags::MEMORY_PLANE_0_EXT);
    steps.image = us(started);
    let release = |memory: Option<vk::DeviceMemory>| {
        // SAFETY: no command uses them any more.
        unsafe {
            probe.device.destroy_image(image, None);
            if let Some(memory) = memory {
                probe.device.free_memory(memory, None);
            }
        }
    };
    let size = facts.requirements.size;
    if facts.layout.offset != at || facts.layout.row_pitch != pitch || size > RAM_BYTES as u64 {
        release(None);
        return Err(format!(
            "the driver lays the plane at {} with rows {} apart, its memory {size} bytes",
            facts.layout.offset, facts.layout.row_pitch
        ));
    }
    let started = Instant::now();
    let fd = kept.try_clone().map_err(|error| {
        release(None);
        format!("dup: {error}")
    })?;
    steps.dma_buf = us(started);
    let started = Instant::now();
    let bits = facts.requirements.memory_type_bits;
    let (memory, _) = (probe.import(fd, size, bits, Some(image))).inspect_err(|_| release(None))?;
    steps.import = us(started);
    let started = Instant::now();
    // SAFETY: the image is unbound; the memory is its dedicated import, at least its size.
    unsafe { probe.device.bind_image_memory(image, memory, 0) }.unwrap();
    steps.bind = us(started);
    (steps.submit, steps.wait) = probe.submit(|commands| {
        probe.acquire(commands, image);
        let first = [FIRST_COPY, FIRST_COPY];
        probe.copy_out(commands, (image, first), (import.buffer, OUT_AT));
        probe.to_host(commands);
    });
    let started = Instant::now();
    release(Some(memory));
    steps.release = us(started);
    Ok(steps)
}

/// Each shape's first use each way (`LINBIND-TIME`).
fn timed(probe: &Probe, (ram, import): (&Ram, &Import), pages: &Pages<'_>, kept: &OwnedFd) {
    let format = vk::Format::B8G8R8A8_UNORM;
    let usage = usages()[0].1;
    let linear_taken = probe.takes(format, usage, vk::ImageTiling::LINEAR).0;
    for extent in TIMED {
        // The driver's pitch where it makes the linear image, else the texels' at 256.
        let pitch = match linear_taken {
            true => {
                let image = probe.linear_image(format, usage, extent);
                let pitch = probe
                    .facts(image, vk::ImageAspectFlags::COLOR)
                    .layout
                    .row_pitch;
                // SAFETY: the image is live and unbound, and no command uses it.
                unsafe { probe.device.destroy_image(image, None) };
                pitch
            }
            false => up(u64::from(extent[0]) * 4, 256),
        };
        let bytes = pitch * u64::from(extent[1]);
        assert!(
            TIMED_AT + bytes <= OUT_AT,
            "the timed texture fits before the copies"
        );
        pages.fill(TIMED_AT, bytes, 0x33);
        let mut ways: [(&str, Vec<Steps>); 3] = [
            ("today", Vec::new()),
            ("linear", Vec::new()),
            ("reimport", Vec::new()),
        ];
        for round in 0..ROUNDS {
            for (way, runs) in &mut ways {
                let place = (TIMED_AT, pitch);
                let steps = match *way {
                    "today" | "reimport" if !probe.modifiers => {
                        Err("no DRM format modifiers".to_owned())
                    }
                    "today" => today(probe, (ram, import), extent, place),
                    "reimport" => reimport(probe, (kept, import), extent, place),
                    _ if linear_taken => linear(probe, import, extent, place),
                    _ => Err("the query takes no dma-buf for the linear image".to_owned()),
                };
                let [width, height] = extent;
                match steps {
                    Ok(steps) => {
                        println!(
                            "LINBIND-TIME {width}x{height} pitch {pitch} {way} round {round}{}: \
                             image {:.0} dma-buf {:.0} import {:.0} bind {:.0} submit {:.0} wait \
                             {:.0} total {:.0} release {:.0} us",
                            if round == 0 { " (warm-up)" } else { "" },
                            steps.image,
                            steps.dma_buf,
                            steps.import,
                            steps.bind,
                            steps.submit,
                            steps.wait,
                            steps.total(),
                            steps.release
                        );
                        if round > 0 {
                            runs.push(steps);
                        }
                    }
                    Err(why) => println!("LINBIND-TIME {width}x{height} {way}: not run: {why}"),
                }
            }
        }
        for (way, runs) in &ways {
            if runs.is_empty() {
                continue;
            }
            let of = |step: fn(&Steps) -> f64| median(runs.iter().map(step).collect());
            println!(
                "LINBIND-TIME {}x{} {way} median of {}: image {:.0} dma-buf {:.0} import {:.0} \
                 bind {:.0} submit {:.0} wait {:.0} total {:.0} release {:.0} us",
                extent[0],
                extent[1],
                runs.len(),
                of(|s| s.image),
                of(|s| s.dma_buf),
                of(|s| s.import),
                of(|s| s.bind),
                of(|s| s.submit),
                of(|s| s.wait),
                of(Steps::total),
                of(|s| s.release)
            );
        }
    }
}

#[test]
#[ignore = "a measurement: run it by name in release, with --run-ignored all"]
fn dma_buf_linear_bind() {
    if !udmabuf_openable("dma_buf_linear_bind") {
        return;
    }
    let alignments = runtime_alignments();
    let Some(probe) = Probe::new() else {
        return;
    };
    let ram = Ram::new(RAM_BYTES, 0, RAM_BYTES);
    let pages = Pages(&ram);
    // Every page written once, so each dma-buf pins pages that are there.
    pages.fill(0, RAM_BYTES as u64, 0);
    let whole = udmabuf(ram.file(), 0, RAM_BYTES as u64).unwrap();
    let kept = whole.try_clone().unwrap();
    let import = probe.buffer_import(whole, RAM_BYTES as u64);
    // The import used once, so no timed first use pays its first submission.
    probe.submit(|commands| {
        // SAFETY: the command buffer is recording, outside a render pass; the buffer is live
        // with transfer-destination usage and the range inside it.
        unsafe { (probe.device).cmd_fill_buffer(commands, import.buffer, OUT_AT, 4096, 0) };
        probe.to_host(commands);
    });
    answers(&probe, &alignments);
    binds(&probe, &import, &pages);
    timed(&probe, (&ram, &import), &pages, &kept);
    // SAFETY: every submission that used them completed.
    unsafe {
        probe.device.destroy_buffer(import.buffer, None);
        probe.device.free_memory(import.memory, None);
    }
}
