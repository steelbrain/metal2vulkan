//! The layouts the device's commands name each image in, recorded for a test that reads them: a
//! device made while [`RECORDING`] is set has its table of Vulkan commands loaded with this
//! module's own for each core command rt records that names an image layout (its barriers, its
//! copies between images and buffers, its blits, its colour clears, dynamic rendering's
//! attachments and its descriptor writes), each noting the layouts it names, then calling the
//! driver's. Its copies between images keep each region's aspects too ([`image_copies`]), and its
//! renderings each colour attachment's view and image as made ([`rendered`]). Its barrier commands are counted too. The core commands that name a layout and that rt does not record get no entry, so
//! that a call of one panics (ash's stub for a command it did not load) and the test learns of it.
//!
//! Each command buffer's barriers, copies to a buffer and timestamp writes are kept in the order
//! recorded, for every command buffer submitted ([`submitted`]).
//!
//! The same commands keep each image's moves in the order the queue runs them ([`moves`]): each
//! command buffer's steps, as recorded, are taken in the order `vkQueueSubmit2` hands the command
//! buffers to the queue. An image made while recording moves home once: a barrier of its first
//! level from the undefined or the preinitialized layout, which keeps no bytes (the undefined
//! one) or the host's (the preinitialized one). A second such move of an image that moved home
//! is a step that drops its bytes, and any other command that names it before its move home (a
//! copy into it among them) reads or writes it in a layout it is not in; either is noted
//! against the device.
//!
//! Under a device cfg, on every device of the cfg's kind: one command buffer runs every blit
//! entry that takes a texture over a texture over a Shared buffer in place (a copy from and to
//! another texture, from and to a buffer, and a fill of its buffer), then a pass that clears it as
//! its colour attachment where the device takes the render target in place; then a
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

/// Sets [`RECORDING`] while it lives, so that no device a later test makes in the same process
/// (libtest runs every test in one) is recorded.
pub(crate) struct Recording;

impl Recording {
    pub(crate) fn start() -> Recording {
        RECORDING.store(true, Ordering::Relaxed);
        Recording
    }
}

impl Drop for Recording {
    fn drop(&mut self) {
        RECORDING.store(false, Ordering::Relaxed);
    }
}

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

/// What a command does to an image, as the queue's order of commands takes it.
#[derive(Clone, Copy)]
enum Step {
    /// A barrier's move of its first level from `layout`, undefined or preinitialized.
    Home(vk::ImageLayout),
    /// Any other command that names it, `what` saying which.
    Use(&'static str),
}

/// A copy from a buffer: [`Moves::copies_in`] counts its steps.
const COPY_IN: &str = "a copy from a buffer";

/// A command of a command buffer, as [`submitted`] keeps it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Ordered {
    Barrier,
    /// `vkCmdCopyImageToBuffer`.
    CopyToBuffer,
    /// `vkCmdWriteTimestamp2` of this query.
    Stamp(u32),
}

/// One device's images' moves, in the order its queue ran its submitted command buffers.
#[derive(Clone, Default, Debug)]
pub(crate) struct Moves {
    /// The moves home taken.
    pub(crate) homes: u64,
    /// Every other step taken.
    pub(crate) uses: u64,
    /// Of `uses`, the copies from a buffer into an image.
    pub(crate) copies_in: u64,
    /// The steps out of order: a second move home of an image, or a step before its move home.
    pub(crate) broken: Vec<String>,
}

/// The driver's commands this module records, of one device.
#[derive(Clone, Copy)]
struct Driver {
    begin_command_buffer: vk::PFN_vkBeginCommandBuffer,
    create_image: vk::PFN_vkCreateImage,
    get_device_queue: vk::PFN_vkGetDeviceQueue,
    queue_submit2: vk::PFN_vkQueueSubmit2,
    allocate_command_buffers: vk::PFN_vkAllocateCommandBuffers,
    create_image_view: vk::PFN_vkCreateImageView,
    update_descriptor_sets: vk::PFN_vkUpdateDescriptorSets,
    cmd_pipeline_barrier2: vk::PFN_vkCmdPipelineBarrier2,
    cmd_copy_image: vk::PFN_vkCmdCopyImage,
    cmd_copy_buffer_to_image: vk::PFN_vkCmdCopyBufferToImage,
    cmd_copy_image_to_buffer: vk::PFN_vkCmdCopyImageToBuffer,
    cmd_blit_image: vk::PFN_vkCmdBlitImage,
    cmd_clear_color_image: vk::PFN_vkCmdClearColorImage,
    cmd_clear_depth_stencil_image: vk::PFN_vkCmdClearDepthStencilImage,
    cmd_begin_rendering: vk::PFN_vkCmdBeginRendering,
    cmd_write_timestamp2: vk::PFN_vkCmdWriteTimestamp2,
}

#[derive(Default)]
struct Record {
    named: Vec<Named>,
    /// Each recorded device's `vkCmdPipelineBarrier2` calls.
    barriers: HashMap<vk::Device, u64>,
    /// Each recorded device's driver commands.
    drivers: HashMap<vk::Device, Driver>,
    /// Each command buffer's device.
    devices: HashMap<vk::CommandBuffer, vk::Device>,
    /// Each image view's image, type and range.
    views: HashMap<
        (vk::Device, vk::ImageView),
        (vk::Image, vk::ImageViewType, vk::ImageSubresourceRange),
    >,
    /// Each image's type and create flags, as made while recording.
    made: HashMap<(vk::Device, vk::Image), (vk::ImageType, vk::ImageCreateFlags)>,
    /// Each queue's device.
    queues: HashMap<vk::Queue, vk::Device>,
    /// Each command buffer's steps since its begin, in the order recorded.
    steps: HashMap<vk::CommandBuffer, Vec<(vk::Image, Step)>>,
    /// Each image made while recording, and whether it moved home: a handle made again starts
    /// again.
    homed: HashMap<(vk::Device, vk::Image), bool>,
    /// Each device's moves.
    moves: HashMap<vk::Device, Moves>,
    /// Each command buffer's commands since its begin, in the order recorded.
    order: HashMap<vk::CommandBuffer, Vec<Ordered>>,
    /// Each device's submitted command buffers' commands, in the order submitted.
    submitted: HashMap<vk::Device, Vec<Vec<Ordered>>>,
    /// Every copy between images recorded, in the order recorded.
    copies: Vec<ImageCopied>,
    /// Every colour attachment of a rendering recorded whose view and image were made while
    /// recording, in the order recorded.
    rendered: Vec<Rendered>,
}

/// A colour attachment a `vkCmdBeginRendering` of `device` named: its view's type and range, and
/// its image's type and create flags.
#[derive(Clone, Copy)]
pub(crate) struct Rendered {
    pub(crate) device: vk::Device,
    pub(crate) image: vk::Image,
    pub(crate) image_type: vk::ImageType,
    pub(crate) flags: vk::ImageCreateFlags,
    pub(crate) view_type: vk::ImageViewType,
    pub(crate) range: vk::ImageSubresourceRange,
}

/// A `vkCmdCopyImage` of `device`: its images, the layouts it names them in and each region's
/// source and destination aspects.
#[derive(Clone)]
pub(crate) struct ImageCopied {
    pub(crate) device: vk::Device,
    pub(crate) source: vk::Image,
    pub(crate) destination: vk::Image,
    pub(crate) layouts: [vk::ImageLayout; 2],
    pub(crate) aspects: Vec<[vk::ImageAspectFlags; 2]>,
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
/// The `vkCmdPipelineBarrier2` calls recorded for `device` so far.
pub(crate) fn barriers(device: vk::Device) -> u64 {
    record().barriers.get(&device).copied().unwrap_or(0)
}

pub(crate) fn forget() {
    record().named.clear();
}

/// Every copy between images `device` recorded so far, in the order recorded.
pub(crate) fn image_copies(device: vk::Device) -> Vec<ImageCopied> {
    (record().copies.iter())
        .filter(|copy| copy.device == device)
        .cloned()
        .collect()
}

/// Every colour attachment `device`'s renderings named so far, in the order recorded.
pub(crate) fn rendered(device: vk::Device) -> Vec<Rendered> {
    (record().rendered.iter())
        .filter(|rendered| rendered.device == device)
        .copied()
        .collect()
}

/// `device`'s images' moves so far, in the order its queue ran its command buffers.
pub(crate) fn moves(device: vk::Device) -> Moves {
    record().moves.get(&device).cloned().unwrap_or_default()
}

/// The commands of each command buffer `device` submitted so far, in the order submitted.
pub(crate) fn submitted(device: vk::Device) -> Vec<Vec<Ordered>> {
    record().submitted.get(&device).cloned().unwrap_or_default()
}

/// Notes `command` of `command_buffer`, in the order recorded.
fn ordered(command_buffer: vk::CommandBuffer, command: Ordered) {
    (record().order.entry(command_buffer).or_default()).push(command);
}

/// Notes `steps` of `command_buffer`, in the order recorded.
fn step(command_buffer: vk::CommandBuffer, steps: impl IntoIterator<Item = (vk::Image, Step)>) {
    (record().steps.entry(command_buffer).or_default()).extend(steps);
}

/// Takes `command_buffer`'s steps, of `device`, as its queue runs it: after every command buffer
/// submitted before it.
fn take(record: &mut Record, device: vk::Device, command_buffer: vk::CommandBuffer) {
    let steps = record.steps.remove(&command_buffer).unwrap_or_default();
    let moves = record.moves.entry(device).or_default();
    for (image, step) in steps {
        // An image not made while recording (a swapchain's) has no record.
        let Some(homed) = record.homed.get_mut(&(device, image)) else {
            continue;
        };
        let shown = format!("image {:#x}", ash::vk::Handle::as_raw(image));
        match step {
            Step::Home(layout) => {
                if *homed {
                    let layout = layout.as_raw();
                    let broken = format!("{shown}: a second move from layout {layout}");
                    moves.broken.push(broken);
                }
                *homed = true;
                moves.homes += 1;
            }
            Step::Use(what) => {
                if !*homed {
                    moves
                        .broken
                        .push(format!("{shown}: {what} before its move home"));
                }
                moves.uses += 1;
                moves.copies_in += u64::from(what == COPY_IN);
            }
        }
    }
}

/// The core commands that name an image layout and that rt does not record: none loaded.
const UNRECORDED: [&CStr; 11] = [
    c"vkCmdPipelineBarrier",
    c"vkCmdWaitEvents",
    c"vkCmdWaitEvents2",
    c"vkCmdCopyImage2",
    c"vkCmdCopyBufferToImage2",
    c"vkCmdCopyImageToBuffer2",
    c"vkCmdBlitImage2",
    c"vkCmdResolveImage",
    c"vkCmdResolveImage2",
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
            begin_command_buffer: std::mem::transmute::<*const c_void, vk::PFN_vkBeginCommandBuffer>(
                load(c"vkBeginCommandBuffer"),
            ),
            create_image: std::mem::transmute::<*const c_void, vk::PFN_vkCreateImage>(load(
                c"vkCreateImage",
            )),
            get_device_queue: std::mem::transmute::<*const c_void, vk::PFN_vkGetDeviceQueue>(load(
                c"vkGetDeviceQueue",
            )),
            queue_submit2: std::mem::transmute::<*const c_void, vk::PFN_vkQueueSubmit2>(load(
                c"vkQueueSubmit2",
            )),
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
            cmd_blit_image: std::mem::transmute::<*const c_void, vk::PFN_vkCmdBlitImage>(load(
                c"vkCmdBlitImage",
            )),
            cmd_clear_color_image: std::mem::transmute::<*const c_void, vk::PFN_vkCmdClearColorImage>(
                load(c"vkCmdClearColorImage"),
            ),
            cmd_clear_depth_stencil_image: std::mem::transmute::<
                *const c_void,
                vk::PFN_vkCmdClearDepthStencilImage,
            >(load(c"vkCmdClearDepthStencilImage")),
            cmd_begin_rendering: std::mem::transmute::<*const c_void, vk::PFN_vkCmdBeginRendering>(
                load(c"vkCmdBeginRendering"),
            ),
            cmd_write_timestamp2: std::mem::transmute::<*const c_void, vk::PFN_vkCmdWriteTimestamp2>(
                load(c"vkCmdWriteTimestamp2"),
            ),
        }
    };
    record().drivers.insert(handle, driver);
    let ours = |name: &CStr| -> Option<*const c_void> {
        let function: *const c_void = match name.to_bytes() {
            b"vkBeginCommandBuffer" => begin_command_buffer as *const c_void,
            b"vkCreateImage" => create_image as *const c_void,
            b"vkGetDeviceQueue" => get_device_queue as *const c_void,
            b"vkQueueSubmit2" => queue_submit2 as *const c_void,
            b"vkAllocateCommandBuffers" => allocate_command_buffers as *const c_void,
            b"vkCreateImageView" => create_image_view as *const c_void,
            b"vkUpdateDescriptorSets" => update_descriptor_sets as *const c_void,
            b"vkCmdPipelineBarrier2" => cmd_pipeline_barrier2 as *const c_void,
            b"vkCmdCopyImage" => cmd_copy_image as *const c_void,
            b"vkCmdCopyBufferToImage" => cmd_copy_buffer_to_image as *const c_void,
            b"vkCmdCopyImageToBuffer" => cmd_copy_image_to_buffer as *const c_void,
            b"vkCmdBlitImage" => cmd_blit_image as *const c_void,
            b"vkCmdClearColorImage" => cmd_clear_color_image as *const c_void,
            b"vkCmdClearDepthStencilImage" => cmd_clear_depth_stencil_image as *const c_void,
            b"vkCmdBeginRendering" => cmd_begin_rendering as *const c_void,
            b"vkCmdWriteTimestamp2" => cmd_write_timestamp2 as *const c_void,
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
    record()
        .views
        .get(&(device, view))
        .map(|&(image, _, _)| image)
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

unsafe extern "system" fn begin_command_buffer(
    command_buffer: vk::CommandBuffer,
    info: *const vk::CommandBufferBeginInfo<'_>,
) -> vk::Result {
    let (_, driver) = driver_of(command_buffer);
    // Steps of a recording never submitted are not the queue's.
    let mut record = record();
    record.steps.remove(&command_buffer);
    record.order.remove(&command_buffer);
    drop(record);
    // SAFETY: the caller's arguments, passed on.
    unsafe { (driver.begin_command_buffer)(command_buffer, info) }
}

unsafe extern "system" fn create_image(
    device: vk::Device,
    info: *const vk::ImageCreateInfo<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    image: *mut vk::Image,
) -> vk::Result {
    let driver = record().drivers[&device];
    // SAFETY: the caller's arguments, passed on.
    let result = unsafe { (driver.create_image)(device, info, allocator, image) };
    if result == vk::Result::SUCCESS {
        // SAFETY: the call wrote the image, of the caller's info.
        let (image, info) = unsafe { (*image, &*info) };
        let mut record = record();
        record.homed.insert((device, image), false);
        (record.made).insert((device, image), (info.image_type, info.flags));
    }
    result
}

unsafe extern "system" fn get_device_queue(
    device: vk::Device,
    family: u32,
    index: u32,
    queue: *mut vk::Queue,
) {
    let driver = record().drivers[&device];
    // SAFETY: the caller's arguments, passed on.
    unsafe { (driver.get_device_queue)(device, family, index, queue) };
    // SAFETY: the call wrote the queue.
    let queue = unsafe { *queue };
    record().queues.insert(queue, device);
}

unsafe extern "system" fn queue_submit2(
    queue: vk::Queue,
    count: u32,
    submits: *const vk::SubmitInfo2<'_>,
    fence: vk::Fence,
) -> vk::Result {
    let driver = {
        let mut record = record();
        let device = record.queues[&queue];
        // SAFETY: the caller's `count` submissions, each with its command buffer infos.
        for submit in unsafe { slice(submits, count) } {
            // SAFETY: as above.
            let infos = unsafe {
                slice(
                    submit.p_command_buffer_infos,
                    submit.command_buffer_info_count,
                )
            };
            for info in infos {
                take(&mut record, device, info.command_buffer);
                let order = (record.order.remove(&info.command_buffer)).unwrap_or_default();
                record.submitted.entry(device).or_default().push(order);
            }
        }
        record.drivers[&device]
    };
    // SAFETY: the caller's arguments, passed on.
    unsafe { (driver.queue_submit2)(queue, count, submits, fence) }
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
        // SAFETY: the call wrote the view, of the caller's info.
        let (view, info) = unsafe { (*view, &*info) };
        let viewed = (info.image, info.view_type, info.subresource_range);
        record().views.insert((device, view), viewed);
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
    *record().barriers.entry(device).or_default() += 1;
    ordered(command_buffer, Ordered::Barrier);
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
    let first = [vk::ImageLayout::UNDEFINED, vk::ImageLayout::PREINITIALIZED];
    step(
        command_buffer,
        barriers.iter().map(|barrier| {
            let home = first.contains(&barrier.old_layout)
                && barrier.subresource_range.base_mip_level == 0;
            match home {
                true => (barrier.image, Step::Home(barrier.old_layout)),
                false => (barrier.image, Step::Use("a barrier")),
            }
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
    // SAFETY: the caller's `count` regions.
    let aspects = unsafe { slice(regions, count) }
        .iter()
        .map(|r| [r.src_subresource.aspect_mask, r.dst_subresource.aspect_mask])
        .collect();
    record().copies.push(ImageCopied {
        device,
        source,
        destination,
        layouts: [source_layout, destination_layout],
        aspects,
    });
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
    step(
        command_buffer,
        [
            (source, Step::Use("an image copy's source")),
            (destination, Step::Use("an image copy's destination")),
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
    note(device, [(COPY_IN, image, layout)]);
    step(command_buffer, [(image, Step::Use(COPY_IN))]);
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
    step(command_buffer, [(image, Step::Use("a copy to a buffer"))]);
    ordered(command_buffer, Ordered::CopyToBuffer);
    // SAFETY: the caller's arguments, passed on.
    unsafe {
        (driver.cmd_copy_image_to_buffer)(command_buffer, image, layout, buffer, count, regions)
    }
}

unsafe extern "system" fn cmd_write_timestamp2(
    command_buffer: vk::CommandBuffer,
    stage: vk::PipelineStageFlags2,
    pool: vk::QueryPool,
    query: u32,
) {
    let (_, driver) = driver_of(command_buffer);
    ordered(command_buffer, Ordered::Stamp(query));
    // SAFETY: the caller's arguments, passed on.
    unsafe { (driver.cmd_write_timestamp2)(command_buffer, stage, pool, query) }
}

unsafe extern "system" fn cmd_blit_image(
    command_buffer: vk::CommandBuffer,
    source: vk::Image,
    source_layout: vk::ImageLayout,
    destination: vk::Image,
    destination_layout: vk::ImageLayout,
    count: u32,
    regions: *const vk::ImageBlit,
    filter: vk::Filter,
) {
    let (device, driver) = driver_of(command_buffer);
    note(
        device,
        [
            ("a blit's source", source, source_layout),
            ("a blit's destination", destination, destination_layout),
        ],
    );
    step(
        command_buffer,
        [
            (source, Step::Use("a blit's source")),
            (destination, Step::Use("a blit's destination")),
        ],
    );
    // SAFETY: the caller's arguments, passed on.
    unsafe {
        (driver.cmd_blit_image)(
            command_buffer,
            source,
            source_layout,
            destination,
            destination_layout,
            count,
            regions,
            filter,
        )
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
    step(command_buffer, [(image, Step::Use("a colour clear"))]);
    // SAFETY: the caller's arguments, passed on.
    unsafe { (driver.cmd_clear_color_image)(command_buffer, image, layout, color, count, ranges) }
}

unsafe extern "system" fn cmd_clear_depth_stencil_image(
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    layout: vk::ImageLayout,
    value: *const vk::ClearDepthStencilValue,
    count: u32,
    ranges: *const vk::ImageSubresourceRange,
) {
    let (device, driver) = driver_of(command_buffer);
    note(device, [("a depth-stencil clear", image, layout)]);
    step(
        command_buffer,
        [(image, Step::Use("a depth-stencil clear"))],
    );
    // SAFETY: the caller's arguments, passed on.
    unsafe {
        (driver.cmd_clear_depth_stencil_image)(command_buffer, image, layout, value, count, ranges)
    }
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
    let used = named
        .iter()
        .map(|&(what, image, _)| (image, Step::Use(what)));
    step(command_buffer, used.collect::<Vec<_>>());
    note(device, named);
    {
        let mut record = record();
        for attachment in colors {
            let Some(&(image, view_type, range)) =
                record.views.get(&(device, attachment.image_view))
            else {
                continue;
            };
            if let Some(&(image_type, flags)) = record.made.get(&(device, image)) {
                record.rendered.push(Rendered {
                    device,
                    image,
                    image_type,
                    flags,
                    view_type,
                    range,
                });
            }
        }
    }
    // SAFETY: the caller's arguments, passed on.
    unsafe { (driver.cmd_begin_rendering)(command_buffer, info) }
}

mod on_device {
    use super::{barriers, forget, named, submitted, Ordered, Recording};
    use crate::tests::command::on_device::{each_device, finished};
    use crate::texture::TextureObject;
    use crate::{
        BufferLayout, ClearColor, IndexClass, LoadAction, MemoryClass, PixelFormat, Region,
        RenderPassDescriptor, StorageMode, StoreAction, TextureDescriptor, TexturePlace,
        TextureType, TextureUsage,
    };
    use ash::vk;

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
        let _recording = Recording::start();
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
            // A render target where the device takes one in place.
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

    /// Every `vkCmdPipelineBarrier2` rt records is counted in [`crate::Metrics::barriers`]: a
    /// texture's move home in an upload, a replace and a read of a texture's image through the
    /// staging region (each a barrier away from home and one back), and a command buffer's blit
    /// between two textures, counted against the barrier commands the device's table saw.
    #[test]
    fn every_barrier_rt_records_is_counted() {
        let _recording = Recording::start();
        each_device("barrier count", |name, _, device| {
            let handle = device.core().device.handle();
            let counted = || device.counters().metrics().barriers;
            let before = (counted(), barriers(handle));
            let d =
                TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, WIDTH, HEIGHT);
            let first = IndexClass::Texture.range().start;
            (device.grant_indices(IndexClass::Texture, first, 2)).expect("a grant");
            let new = |index| {
                (device.new_texture(&d, Some(index))).unwrap_or_else(|e| panic!("{name}: {e}"))
            };
            let (a, b) = (new(first), new(first + 1));
            let size = [WIDTH, HEIGHT, 1];
            let whole = Region {
                origin: [0; 3],
                size,
            };
            let written = vec![0x5Au8; (ROW * u64::from(HEIGHT)) as usize];
            (a.replace_region(whole, 0, 0, crate::HostBytes::from(&written[..]), ROW, 0))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut blit = command_buffer.blit_command_encoder().expect("an encoder");
            let at = TexturePlace::default();
            (blit.copy_from_texture(&a, at, size, &b, at))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            blit.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            let mut read = vec![0u8; written.len()];
            (b.get_bytes(&mut read, ROW, 0, whole, 0, 0))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            assert_eq!(read, written, "{name}: the bytes through the copy");
            let after = (counted(), barriers(handle));
            let recorded = after.1 - before.1;
            // At least: a move home, two for the replace, two for the blit, two for the read.
            assert!(
                recorded >= 7,
                "{name}: {recorded} barrier commands recorded"
            );
            assert_eq!(
                after.0 - before.0,
                recorded,
                "{name}: barriers counted, against the barrier commands recorded"
            );
        });
    }

    /// A command buffer's end timestamp is its last command, after its tail: a blit writes a
    /// texture over a Shared buffer in an image of its own (a texel off the linear image's memory
    /// alignment), so the command buffer's last segment ends with the copy of that image back to
    /// the buffer and its end barrier, and only then the end timestamp (the pair's second query,
    /// an odd one), which `vkCmdWriteTimestamp2` orders after every earlier command. In every
    /// command buffer the device submits, the uploads among them, an end timestamp is the last
    /// command seen. lavapipe makes the texture an image of its own; a device that takes it in
    /// place runs no copy back, and the test says so.
    #[test]
    fn a_command_buffer_s_end_timestamp_follows_its_tail() {
        let _recording = Recording::start();
        each_device("end stamp", |name, _, device| {
            let handle = device.core().device.handle();
            let answer = (device.minimum_linear_texture_alignment(PixelFormat::RGBA8Unorm))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let pitch = ROW.next_multiple_of(answer);
            let block = (device.take_block(MemoryClass::Shared, AT + LEN))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let buffer = (device.new_buffer(block.id, AT, LEN, Some(0)))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let source = (device.new_buffer(block.id, 0, ROW * u64::from(HEIGHT), None))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let first = IndexClass::Texture.range().start;
            (device.grant_indices(IndexClass::Texture, first, 1)).expect("a grant");
            let mut d =
                TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, WIDTH, HEIGHT);
            d.usage = TextureUsage::SHADER_READ;
            let texture = (buffer.new_texture(&d, Some(first), 4, pitch))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let copied = texture.copied_over().is_some();
            if cfg!(m2v_cpu_icd) {
                assert!(copied, "{name}: an image of its own");
            }
            let queue = device.new_command_queue(4).expect("a queue");
            let command_buffer = queue.command_buffer().expect("a command buffer");
            let mut blit = command_buffer.blit_command_encoder().expect("an encoder");
            let packed = BufferLayout {
                offset: 0,
                bytes_per_row: ROW,
                bytes_per_image: 0,
            };
            let (size, at) = ([WIDTH, HEIGHT, 1], TexturePlace::default());
            (blit.copy_from_buffer(&source, packed, size, &texture, at))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            blit.end_encoding();
            command_buffer.commit().expect("a commit");
            finished(name, &command_buffer);
            assert_eq!(command_buffer.error(), None, "{name}");
            let submitted = submitted(handle);
            for order in &submitted {
                let end = order
                    .iter()
                    .position(|c| matches!(c, Ordered::Stamp(q) if q % 2 == 1));
                if let Some(at) = end {
                    assert_eq!(
                        at + 1,
                        order.len(),
                        "{name}: an end stamp not last: {order:?}"
                    );
                }
            }
            // The tail: the copy back between its two barriers, the end barrier, the end stamp.
            let tail = |order: &Vec<Ordered>| match order.as_slice() {
                [.., Ordered::CopyToBuffer, Ordered::Barrier, Ordered::Barrier, last] => {
                    matches!(last, Ordered::Stamp(q) if q % 2 == 1)
                }
                _ => false,
            };
            let tails = submitted.iter().filter(|order| tail(order)).count();
            println!("end stamp: {name}: image of its own {copied}; {tails} tails: {submitted:?}");
            assert_eq!(tails, usize::from(copied), "{name}: {submitted:?}");
            drop((command_buffer, queue, texture, source, buffer));
            (device.release_block(block)).unwrap_or_else(|error| panic!("{name}: {error}"));
        });
    }
}
