//! The layouts the device's commands name each image in, recorded for a test that reads them: a
//! device made while [`RECORDING`] is set has its table of Vulkan commands loaded with this
//! module's own for each core command rt records that names an image layout (its barriers, its
//! copies between images and buffers, its colour clears, dynamic rendering's attachments and its
//! descriptor writes), each noting the layouts it names, then calling the driver's. The core
//! commands that name a layout and that rt does not record get no entry, so that a call of one
//! panics (ash's stub for a command it did not load) and the test learns of it.
//!
//! Under a device cfg, on every device of the cfg's kind: one command buffer runs every blit
//! entry that takes a texture over a texture over a Shared buffer in place (a copy from and to
//! another texture, from and to a buffer, and a fill of its buffer), then a pass that clears it as
//! its colour attachment where the device's linear tiling takes the render usage in place; then a
//! read of its image, and every layout recorded for the image, its descriptor's included, is the
//! general layout but the move home's old one, preinitialized. The image is host-accessible at
//! every moment, so it and the buffer are aliases that interpret their bytes alike (Vulkan's
//! memory aliasing): a copy that moved it to a transfer layout fails this.

use ash::vk;
use std::collections::HashMap;
use std::ffi::{c_void, CStr};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{LazyLock, Mutex, MutexGuard, PoisonError};

/// Whether a device made now has its commands' layouts recorded.
pub(crate) static RECORDING: AtomicBool = AtomicBool::new(false);

/// A layout a command of `device` named `image` in, `what` saying where.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct Named {
    pub(crate) what: &'static str,
    pub(crate) device: vk::Device,
    pub(crate) image: vk::Image,
    pub(crate) layout: vk::ImageLayout,
}

impl Named {
    /// Where, and the layout's number (`VkImageLayout`).
    fn shown(&self) -> String {
        format!("{}: layout {}", self.what, self.layout.as_raw())
    }
}

/// The driver's commands this module records, of one device.
#[derive(Clone, Copy)]
struct Driver {
    allocate_command_buffers: vk::PFN_vkAllocateCommandBuffers,
    create_image_view: vk::PFN_vkCreateImageView,
    update_descriptor_sets: vk::PFN_vkUpdateDescriptorSets,
    cmd_pipeline_barrier2: vk::PFN_vkCmdPipelineBarrier2,
    cmd_copy_image: vk::PFN_vkCmdCopyImage,
    cmd_copy_buffer_to_image: vk::PFN_vkCmdCopyBufferToImage,
    cmd_copy_image_to_buffer: vk::PFN_vkCmdCopyImageToBuffer,
    cmd_clear_color_image: vk::PFN_vkCmdClearColorImage,
    cmd_begin_rendering: vk::PFN_vkCmdBeginRendering,
}

#[derive(Default)]
struct Record {
    named: Vec<Named>,
    /// Each recorded device's driver commands.
    drivers: HashMap<vk::Device, Driver>,
    /// Each command buffer's device.
    devices: HashMap<vk::CommandBuffer, vk::Device>,
    /// Each image view's image.
    views: HashMap<(vk::Device, vk::ImageView), vk::Image>,
}

static RECORD: LazyLock<Mutex<Record>> = LazyLock::new(Mutex::default);

fn record() -> MutexGuard<'static, Record> {
    RECORD.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Every layout recorded for `image` of `device`, in the order its commands named them.
pub(crate) fn named(device: vk::Device, image: vk::Image) -> Vec<Named> {
    (record().named.iter())
        .filter(|named| named.device == device && named.image == image)
        .copied()
        .collect()
}

/// Forgets the layouts recorded so far: a handle of a destroyed image may name another since.
pub(crate) fn forget() {
    record().named.clear();
}

/// The core commands that name an image layout and that rt does not record: none loaded.
const UNRECORDED: [&CStr; 13] = [
    c"vkCmdPipelineBarrier",
    c"vkCmdWaitEvents",
    c"vkCmdWaitEvents2",
    c"vkCmdCopyImage2",
    c"vkCmdCopyBufferToImage2",
    c"vkCmdCopyImageToBuffer2",
    c"vkCmdBlitImage",
    c"vkCmdBlitImage2",
    c"vkCmdResolveImage",
    c"vkCmdResolveImage2",
    c"vkCmdClearDepthStencilImage",
    c"vkCmdBeginRenderPass",
    c"vkCmdBeginRenderPass2",
];

/// `device`, made from `instance` just now, with its commands' layouts recorded where
/// [`RECORDING`] is set; else `device` itself.
///
/// # Safety
///
/// `device` was made from `instance`, and nothing has used it yet.
pub(crate) unsafe fn recorded(instance: &ash::Instance, device: ash::Device) -> ash::Device {
    if !RECORDING.load(Ordering::Relaxed) {
        return device;
    }
    let handle = device.handle();
    // SAFETY: the caller's contract: `handle` is a device of `instance`.
    let driver_fn = |name: &CStr| unsafe { instance.get_device_proc_addr(handle, name.as_ptr()) };
    let load = |name: &CStr| {
        let function = driver_fn(name).unwrap_or_else(|| panic!("the driver's {name:?}"));
        function as *const c_void
    };
    // SAFETY: each pointer is the driver's command of its name, whose type its field names.
    let driver = unsafe {
        Driver {
            allocate_command_buffers: std::mem::transmute::<
                *const c_void,
                vk::PFN_vkAllocateCommandBuffers,
            >(load(c"vkAllocateCommandBuffers")),
            create_image_view: std::mem::transmute::<*const c_void, vk::PFN_vkCreateImageView>(
                load(c"vkCreateImageView"),
            ),
            update_descriptor_sets: std::mem::transmute::<
                *const c_void,
                vk::PFN_vkUpdateDescriptorSets,
            >(load(c"vkUpdateDescriptorSets")),
            cmd_pipeline_barrier2: std::mem::transmute::<
                *const c_void,
                vk::PFN_vkCmdPipelineBarrier2,
            >(load(c"vkCmdPipelineBarrier2")),
            cmd_copy_image: std::mem::transmute::<*const c_void, vk::PFN_vkCmdCopyImage>(load(
                c"vkCmdCopyImage",
            )),
            cmd_copy_buffer_to_image: std::mem::transmute::<
                *const c_void,
                vk::PFN_vkCmdCopyBufferToImage,
            >(load(c"vkCmdCopyBufferToImage")),
            cmd_copy_image_to_buffer: std::mem::transmute::<
                *const c_void,
                vk::PFN_vkCmdCopyImageToBuffer,
            >(load(c"vkCmdCopyImageToBuffer")),
            cmd_clear_color_image: std::mem::transmute::<*const c_void, vk::PFN_vkCmdClearColorImage>(
                load(c"vkCmdClearColorImage"),
            ),
            cmd_begin_rendering: std::mem::transmute::<*const c_void, vk::PFN_vkCmdBeginRendering>(
                load(c"vkCmdBeginRendering"),
            ),
        }
    };
    record().drivers.insert(handle, driver);
    let ours = |name: &CStr| -> Option<*const c_void> {
        let function: *const c_void = match name.to_bytes() {
            b"vkAllocateCommandBuffers" => allocate_command_buffers as *const c_void,
            b"vkCreateImageView" => create_image_view as *const c_void,
            b"vkUpdateDescriptorSets" => update_descriptor_sets as *const c_void,
            b"vkCmdPipelineBarrier2" => cmd_pipeline_barrier2 as *const c_void,
            b"vkCmdCopyImage" => cmd_copy_image as *const c_void,
            b"vkCmdCopyBufferToImage" => cmd_copy_buffer_to_image as *const c_void,
            b"vkCmdCopyImageToBuffer" => cmd_copy_image_to_buffer as *const c_void,
            b"vkCmdClearColorImage" => cmd_clear_color_image as *const c_void,
            b"vkCmdBeginRendering" => cmd_begin_rendering as *const c_void,
            _ => return None,
        };
        Some(function)
    };
    // SAFETY: each entry is the driver's command of its name, or this module's of the same type,
    // which calls the driver's; an unrecorded one is null, which ash loads as a stub that panics.
    unsafe {
        ash::Device::load_with(
            |name| match (ours(name), UNRECORDED.contains(&name)) {
                (Some(function), _) => function,
                (None, true) => std::ptr::null(),
                (None, false) => driver_fn(name).map_or(std::ptr::null(), |f| f as *const c_void),
            },
            handle,
        )
    }
}

/// The driver's commands of `command_buffer`'s device, and the device.
fn driver_of(command_buffer: vk::CommandBuffer) -> (vk::Device, Driver) {
    let record = record();
    let device = *(record.devices.get(&command_buffer)).expect("a recorded command buffer");
    (device, record.drivers[&device])
}

/// Notes each of `named`'s layouts of `device`.
fn note(
    device: vk::Device,
    named: impl IntoIterator<Item = (&'static str, vk::Image, vk::ImageLayout)>,
) {
    let mut record = record();
    for (what, image, layout) in named {
        record.named.push(Named {
            what,
            device,
            image,
            layout,
        });
    }
}

/// The image of `view` of `device`, where the record knows it.
fn image_of(device: vk::Device, view: vk::ImageView) -> Option<vk::Image> {
    record().views.get(&(device, view)).copied()
}

/// A slice of `count` elements at `pointer`, empty where it is null.
///
/// # Safety
///
/// A non-null `pointer` points to `count` elements, alive through the returned borrow.
unsafe fn slice<'a, T>(pointer: *const T, count: u32) -> &'a [T] {
    match pointer.is_null() {
        true => &[],
        // SAFETY: the caller's contract.
        false => unsafe { std::slice::from_raw_parts(pointer, count as usize) },
    }
}

unsafe extern "system" fn allocate_command_buffers(
    device: vk::Device,
    info: *const vk::CommandBufferAllocateInfo<'_>,
    command_buffers: *mut vk::CommandBuffer,
) -> vk::Result {
    let driver = record().drivers[&device];
    // SAFETY: the caller's arguments, passed on.
    let result = unsafe { (driver.allocate_command_buffers)(device, info, command_buffers) };
    if result == vk::Result::SUCCESS {
        // SAFETY: the call wrote `command_buffer_count` handles there.
        let made = unsafe { slice(command_buffers, (*info).command_buffer_count) };
        let mut record = record();
        for &command_buffer in made {
            record.devices.insert(command_buffer, device);
        }
    }
    result
}

unsafe extern "system" fn create_image_view(
    device: vk::Device,
    info: *const vk::ImageViewCreateInfo<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    view: *mut vk::ImageView,
) -> vk::Result {
    let driver = record().drivers[&device];
    // SAFETY: the caller's arguments, passed on.
    let result = unsafe { (driver.create_image_view)(device, info, allocator, view) };
    if result == vk::Result::SUCCESS {
        // SAFETY: the call wrote the view, of the info's image.
        let (view, image) = unsafe { (*view, (*info).image) };
        record().views.insert((device, view), image);
    }
    result
}

unsafe extern "system" fn update_descriptor_sets(
    device: vk::Device,
    write_count: u32,
    writes: *const vk::WriteDescriptorSet<'_>,
    copy_count: u32,
    copies: *const vk::CopyDescriptorSet<'_>,
) {
    use vk::DescriptorType as T;
    let mut named = Vec::new();
    // SAFETY: the caller's arguments: `write_count` writes, each with its image infos where its
    // type takes them.
    for write in unsafe { slice(writes, write_count) } {
        if ![
            T::SAMPLED_IMAGE,
            T::STORAGE_IMAGE,
            T::COMBINED_IMAGE_SAMPLER,
            T::INPUT_ATTACHMENT,
        ]
        .contains(&write.descriptor_type)
        {
            continue;
        }
        // SAFETY: as above.
        for info in unsafe { slice(write.p_image_info, write.descriptor_count) } {
            if let Some(image) = image_of(device, info.image_view) {
                named.push(("descriptor", image, info.image_layout));
            }
        }
    }
    note(device, named);
    let driver = record().drivers[&device];
    // SAFETY: the caller's arguments, passed on.
    unsafe { (driver.update_descriptor_sets)(device, write_count, writes, copy_count, copies) }
}

unsafe extern "system" fn cmd_pipeline_barrier2(
    command_buffer: vk::CommandBuffer,
    info: *const vk::DependencyInfo<'_>,
) {
    let (device, driver) = driver_of(command_buffer);
    // SAFETY: the caller's dependency info and its image barriers.
    let barriers = unsafe {
        let info = &*info;
        slice(
            info.p_image_memory_barriers,
            info.image_memory_barrier_count,
        )
    };
    note(
        device,
        barriers.iter().flat_map(|barrier| {
            [
                ("a barrier's old layout", barrier.image, barrier.old_layout),
                ("a barrier's new layout", barrier.image, barrier.new_layout),
            ]
        }),
    );
    // SAFETY: the caller's arguments, passed on.
    unsafe { (driver.cmd_pipeline_barrier2)(command_buffer, info) }
}

unsafe extern "system" fn cmd_copy_image(
    command_buffer: vk::CommandBuffer,
    source: vk::Image,
    source_layout: vk::ImageLayout,
    destination: vk::Image,
    destination_layout: vk::ImageLayout,
    count: u32,
    regions: *const vk::ImageCopy,
) {
    let (device, driver) = driver_of(command_buffer);
    note(
        device,
        [
            ("an image copy's source", source, source_layout),
            (
                "an image copy's destination",
                destination,
                destination_layout,
            ),
        ],
    );
    // SAFETY: the caller's arguments, passed on.
    unsafe {
        (driver.cmd_copy_image)(
            command_buffer,
            source,
            source_layout,
            destination,
            destination_layout,
            count,
            regions,
        )
    }
}

unsafe extern "system" fn cmd_copy_buffer_to_image(
    command_buffer: vk::CommandBuffer,
    buffer: vk::Buffer,
    image: vk::Image,
    layout: vk::ImageLayout,
    count: u32,
    regions: *const vk::BufferImageCopy,
) {
    let (device, driver) = driver_of(command_buffer);
    note(device, [("a copy from a buffer", image, layout)]);
    // SAFETY: the caller's arguments, passed on.
    unsafe {
        (driver.cmd_copy_buffer_to_image)(command_buffer, buffer, image, layout, count, regions)
    }
}

unsafe extern "system" fn cmd_copy_image_to_buffer(
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    layout: vk::ImageLayout,
    buffer: vk::Buffer,
    count: u32,
    regions: *const vk::BufferImageCopy,
) {
    let (device, driver) = driver_of(command_buffer);
    note(device, [("a copy to a buffer", image, layout)]);
    // SAFETY: the caller's arguments, passed on.
    unsafe {
        (driver.cmd_copy_image_to_buffer)(command_buffer, image, layout, buffer, count, regions)
    }
}

unsafe extern "system" fn cmd_clear_color_image(
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    layout: vk::ImageLayout,
    color: *const vk::ClearColorValue,
    count: u32,
    ranges: *const vk::ImageSubresourceRange,
) {
    let (device, driver) = driver_of(command_buffer);
    note(device, [("a colour clear", image, layout)]);
    // SAFETY: the caller's arguments, passed on.
    unsafe { (driver.cmd_clear_color_image)(command_buffer, image, layout, color, count, ranges) }
}

unsafe extern "system" fn cmd_begin_rendering(
    command_buffer: vk::CommandBuffer,
    info: *const vk::RenderingInfo<'_>,
) {
    let (device, driver) = driver_of(command_buffer);
    let mut named = Vec::new();
    // SAFETY: the caller's rendering info, its attachments and its optional depth and stencil.
    let (colors, depth, stencil) = unsafe {
        let info = &*info;
        (
            slice(info.p_color_attachments, info.color_attachment_count),
            info.p_depth_attachment.as_ref(),
            info.p_stencil_attachment.as_ref(),
        )
    };
    for attachment in colors.iter().chain(depth).chain(stencil) {
        if let Some(image) = image_of(device, attachment.image_view) {
            named.push(("an attachment", image, attachment.image_layout));
        }
        if let Some(image) = image_of(device, attachment.resolve_image_view) {
            named.push((
                "a resolve attachment",
                image,
                attachment.resolve_image_layout,
            ));
        }
    }
    note(device, named);
    // SAFETY: the caller's arguments, passed on.
    unsafe { (driver.cmd_begin_rendering)(command_buffer, info) }
}

mod on_device {
    use super::{forget, named, RECORDING};
    use crate::tests::command::on_device::{each_device, finished};
    use crate::{
        BufferLayout, ClearColor, IndexClass, LoadAction, MemoryClass, PixelFormat, Region,
        RenderPassDescriptor, StorageMode, StoreAction, TextureDescriptor, TexturePlace,
        TextureType, TextureUsage,
    };
    use ash::vk;
    use std::sync::atomic::Ordering;

    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 8;
    const ROW: u64 = WIDTH as u64 * 4;
    /// The buffer's offset in its block: a multiple of every answer and memory alignment.
    const AT: u64 = 65_536;
    const LEN: u64 = 65_536;

    /// A texture over a Shared buffer in place keeps its image in the general layout through
    /// every command: in every barrier, copy and attachment of one command buffer that runs every
    /// blit entry that takes a texture over it, and a pass that renders to it where the device
    /// takes it in place as a render target, and in its descriptor and a read of its image; the
    /// one other layout named is preinitialized, the old layout of its move home.
    #[test]
    fn a_texture_over_a_buffer_in_place_is_in_the_general_layout_in_every_command() {
        RECORDING.store(true, Ordering::Relaxed);
        each_device("layouts", |name, _, device| {
            let handle = device.core().device.handle();
            let answer = (device.minimum_linear_texture_alignment(PixelFormat::RGBA8Unorm))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let pitch = ROW.next_multiple_of(answer);
            let block = (device.take_block(MemoryClass::Shared, AT + LEN))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let buffer = (device.new_buffer(block.id, AT, LEN, Some(0)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let first = IndexClass::Texture.range().start;
            (device.grant_indices(IndexClass::Texture, first, 2)).expect("a grant");
            let mut d =
                TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, WIDTH, HEIGHT);
            forget();
            // A render target where the device's linear tiling takes one in place.
            d.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
            let mut texture = (buffer.new_texture(&d, Some(first), 0, pitch))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let renders = texture.copied_over().is_none();
            if !renders {
                drop(texture);
                forget();
                d.usage = TextureUsage::SHADER_READ;
                texture = (buffer.new_texture(&d, Some(first), 0, pitch))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                println!(
                    "layouts: {name}: no render target over a buffer in place (the linear \
                     tiling's answer): the pass is not run"
                );
            }
            assert!(texture.copied_over().is_none(), "{name}: in place");
            let mut plain_d = d;
            plain_d.usage = TextureUsage::SHADER_READ;
            plain_d.storage_mode = StorageMode::Shared;
            let plain = (device.new_texture(&plain_d, Some(first + 1)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let other = (device.new_buffer(block.id, 0, ROW * u64::from(HEIGHT), None))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let size = [WIDTH, HEIGHT, 1];
            let packed = BufferLayout {
                offset: 0,
                bytes_per_row: ROW,
                bytes_per_image: 0,
            };
            let at = TexturePlace::default();
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut blit = command_buffer.blit_command_encoder().expect("an encoder");
            let done = |result: Result<(), crate::Error>| {
                result.unwrap_or_else(|error| panic!("{name}: {error}"))
            };
            done(blit.copy_from_texture(&plain, at, size, &texture, at));
            done(blit.copy_from_texture(&texture, at, size, &plain, at));
            done(blit.copy_from_buffer(&other, packed, size, &texture, at));
            done(blit.copy_from_texture_to_buffer(&texture, at, size, &other, packed));
            done(blit.fill_buffer(&buffer, 0..pitch * u64::from(HEIGHT), 0x5A));
            blit.end_encoding();
            if renders {
                let mut pass = RenderPassDescriptor::default();
                let color = &mut pass.color_attachments[0];
                color.texture = Some(texture.clone());
                color.load_action = LoadAction::Clear;
                color.store_action = StoreAction::Store;
                color.clear_color = ClearColor {
                    red: 1.0,
                    green: 0.0,
                    blue: 0.0,
                    alpha: 1.0,
                };
                let encoder = (command_buffer.render_command_encoder(&pass))
                    .unwrap_or_else(|error| panic!("{name}: {error}"));
                encoder.end_encoding();
            }
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            let mut bytes = vec![0; (ROW * u64::from(HEIGHT)) as usize];
            let whole = Region {
                origin: [0; 3],
                size,
            };
            (texture.image_bytes(&mut bytes, ROW, whole, 0, 0))
                .unwrap_or_else(|error| panic!("{name}: {error}"));

            let named = named(handle, texture.image());
            let general = vk::ImageLayout::GENERAL;
            let moved_home = |n: &&super::Named| {
                n.what == "a barrier's old layout" && n.layout == vk::ImageLayout::PREINITIALIZED
            };
            let others: Vec<String> = (named.iter())
                .filter(|n| n.layout != general && !moved_home(n))
                .map(super::Named::shown)
                .collect();
            assert!(
                others.is_empty(),
                "{name}: layouts other than general named for the image in place: {others:?}"
            );
            assert_eq!(
                named.iter().filter(moved_home).count(),
                1,
                "{name}: one move home from preinitialized: {:?}",
                named.iter().map(super::Named::shown).collect::<Vec<_>>()
            );
            let mut kinds: Vec<&str> = named.iter().map(|n| n.what).collect();
            kinds.sort_unstable();
            kinds.dedup();
            let mut expected = vec![
                "a barrier's new layout",
                "a barrier's old layout",
                "a copy from a buffer",
                "a copy to a buffer",
                "an image copy's destination",
                "an image copy's source",
                "descriptor",
            ];
            if renders {
                expected.push("an attachment");
            }
            expected.sort_unstable();
            assert_eq!(
                kinds, expected,
                "{name}: every kind of command named the image"
            );
            println!(
                "layouts: {name}: {} layouts named for the image in place, every one general but \
                 the move home's old one; kinds {kinds:?}",
                named.len()
            );
            drop((texture, plain, other, buffer, command_buffer, queue));
            (device.release_block(block)).unwrap_or_else(|error| panic!("{name}: {error}"));
        });
    }
}
