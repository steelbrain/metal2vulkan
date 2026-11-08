//! The caller's own GPU work on the runtime's resources through their handed-out handles
//! (`Buffer::vk_buffer`, `Texture::vk_image`), on a queue of the caller's (a reserved one where
//! the device spares one, else a command queue's `VkQueue` lent through
//! `CommandQueue::with_vk_queue`), as a consumer's device cache of decoded frames writes them:
//!
//! - **A `Shared` frame laying byte `0x20` on a multiple of 256** (`BufferDescriptor::aligned_byte`),
//!   a 1080p NV12 frame's 3,110,400 bytes, filled through `contents`, then written by the caller's
//!   `vkCmdCopyImageToBuffer` of both planes of an image the caller filled from a runtime
//!   buffer's `VkBuffer`. Each round's frame reads what the copy wrote, through `contents` and
//!   through a blit of the runtime's, and the device is not lost.
//! - **A new `Private` buffer handed out before any submission** reads what the caller's queue
//!   wrote into it, not the zeros it waited for as it was made (`Buffer::clear_first`): the
//!   hand-out makes them first. The test checks that the fill still waited as the buffer was
//!   handed out, so the path ran.
//! - **A texture over such a buffer handed out before any submission** (its plane in place, the
//!   linear image aliasing the buffer's memory) reads what the caller's copy into its `VkImage`
//!   wrote: the image's hand-out makes its move and the buffer's zeros first.
//!
//! Device tests: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects. Printed (`CALLER-QUEUE-WRITES`): the queue each ran on, the images'
//! shape and what each round read.

use super::blit_holds::Scratch;
use crate::{
    BlitOption, Buffer, BufferContents, BufferDescriptor, CommandBufferStatus, CommandQueue,
    Device, DeviceKind, DeviceOptions, IoSurfacePlane, Origin, ReservedQueue, ResourceOptions,
    ShaderCacheOptions, Size, TextureDescriptor,
};
use ash::vk;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::path::Path;
use std::time::Duration;

/// The byte the frames lay on a multiple of 256: CoreVideo's first plane's.
const K: u64 = 0x20;

/// The frame's luma plane: 1920x1080 bytes.
const W: u32 = 1920;
const H: u32 = 1080;
const LUMA: u64 = W as u64 * H as u64;

/// The NV12 frame's bytes: the luma plane and the chroma plane after it (960x540, 2 bytes a
/// texel).
const FRAME: u64 = LUMA * 3 / 2;

/// The rounds of the `Shared` frame's test, each a new frame.
const ROUNDS: u32 = 6;

/// The longest the caller's submission or a command buffer may take.
const WAIT: Duration = Duration::from_secs(20);

/// The device the cfg selects, its shader cache in `cache`, one queue reserved for the caller.
fn device(cache: &Path) -> Device {
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.to_owned()),
            ..ShaderCacheOptions::default()
        },
        reserved_queues: 1,
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// The queue the caller submits to.
enum Caller {
    /// The queue the device reserved for the caller.
    Reserved(ReservedQueue),
    /// A command queue's `VkQueue`, lent (`CommandQueue::with_vk_queue`): where none was spared.
    Lent(CommandQueue, u32),
}

impl Caller {
    /// The device's reserved queue, else a new command queue's.
    fn of(device: &Device) -> Caller {
        // SAFETY: the handles are read while the device lives.
        if let Some(&reserved) = unsafe { device.raw() }.reserved_queues.first() {
            return Caller::Reserved(reserved);
        }
        let queue = device.new_command_queue().unwrap();
        // SAFETY: the family is read; nothing is submitted.
        let family = unsafe { queue.with_vk_queue(|_, _, family| family) };
        Caller::Lent(queue, family)
    }

    fn family(&self) -> u32 {
        match self {
            Caller::Reserved(reserved) => reserved.family,
            Caller::Lent(_, family) => *family,
        }
    }

    fn describe(&self) -> String {
        match self {
            Caller::Reserved(reserved) => format!("reserved queue of family {}", reserved.family),
            Caller::Lent(_, family) => format!("lent command queue of family {family}"),
        }
    }

    /// Record a command buffer of the caller's own pool with `record`, submit it on the caller's
    /// queue and wait for it: the first failure of a Vulkan call (a device loss among them).
    ///
    /// # Safety
    ///
    /// `record` records valid commands on the objects it names, which live until this returns.
    unsafe fn run(
        &self,
        device: &Device,
        record: impl FnOnce(&ash::Device, vk::CommandBuffer),
    ) -> Result<(), vk::Result> {
        // SAFETY: the handles are used while the device lives; the objects made here are
        // destroyed here once the work completed (where a call failed, they are left to the
        // device's end), and the command buffer's allocation is made with the loader's word in
        // place.
        unsafe {
            let raw = device.raw();
            let vk_device = raw.device;
            let pool = vk_device.create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(self.family()),
                None,
            )?;
            let fence = vk_device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let ran = (|| {
                let info = vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);
                let buffers =
                    device.with_loader_word(|| vk_device.allocate_command_buffers(&info))?;
                vk_device
                    .begin_command_buffer(buffers[0], &vk::CommandBufferBeginInfo::default())?;
                record(vk_device, buffers[0]);
                vk_device.end_command_buffer(buffers[0])?;
                let submit = vk::SubmitInfo::default().command_buffers(&buffers);
                match self {
                    Caller::Reserved(reserved) => {
                        vk_device.queue_submit(reserved.queue, &[submit], fence)?
                    }
                    Caller::Lent(queue, _) => queue.with_vk_queue(|lent, vk_queue, _| {
                        lent.queue_submit(vk_queue, &[submit], fence)
                    })?,
                }
                vk_device.wait_for_fences(&[fence], true, WAIT.as_nanos() as u64)
            })();
            if ran.is_ok() {
                vk_device.destroy_fence(fence, None);
                vk_device.destroy_command_pool(pool, None);
            }
            ran
        }
    }
}

/// A plane's copy between the caller's image and a frame: the image, the aspect, its extent and
/// its offset in the frame.
struct PlaneCopy {
    image: usize,
    aspect: vk::ImageAspectFlags,
    extent: vk::Extent3D,
    at: u64,
}

/// The caller's images of a frame and their memory: NV12's two planes in one image where the
/// device copies such images, else an image a plane.
struct FrameImages {
    images: Vec<(vk::Image, vk::DeviceMemory)>,
    copies: Vec<PlaneCopy>,
    planar: bool,
}

impl FrameImages {
    fn new(device: &Device, family: u32) -> FrameImages {
        // SAFETY: the handles are used while the device lives.
        let raw = unsafe { device.raw() };
        let wanted = vk::FormatFeatureFlags::TRANSFER_SRC | vk::FormatFeatureFlags::TRANSFER_DST;
        let nv12 = vk::Format::G8_B8R8_2PLANE_420_UNORM;
        // SAFETY: the physical device is the instance's.
        let features = unsafe {
            raw.instance
                .get_physical_device_format_properties(raw.physical_device, nv12)
        };
        let planar = features.optimal_tiling_features.contains(wanted);
        let extent = |w: u32, h: u32| vk::Extent3D {
            width: w,
            height: h,
            depth: 1,
        };
        let (shapes, copies) = match planar {
            true => (
                vec![(nv12, extent(W, H))],
                vec![
                    (0, vk::ImageAspectFlags::PLANE_0, extent(W, H), 0),
                    (0, vk::ImageAspectFlags::PLANE_1, extent(W / 2, H / 2), LUMA),
                ],
            ),
            false => (
                vec![
                    (vk::Format::R8_UNORM, extent(W, H)),
                    (vk::Format::R8G8_UNORM, extent(W / 2, H / 2)),
                ],
                vec![
                    (0, vk::ImageAspectFlags::COLOR, extent(W, H), 0),
                    (1, vk::ImageAspectFlags::COLOR, extent(W / 2, H / 2), LUMA),
                ],
            ),
        };
        // SAFETY: the physical device is the instance's.
        let types = unsafe {
            raw.instance
                .get_physical_device_memory_properties(raw.physical_device)
        };
        let images = (shapes.into_iter())
            .map(|(format, extent)| {
                let families = [family];
                let info = vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(format)
                    .extent(extent)
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .queue_family_indices(&families)
                    .initial_layout(vk::ImageLayout::UNDEFINED);
                // SAFETY: the device is live and `info` valid; the image is bound once, to new
                // memory of a type it takes, and destroyed with it in `destroy`.
                unsafe {
                    let image = raw.device.create_image(&info, None).unwrap();
                    let needs = raw.device.get_image_memory_requirements(image);
                    let fits = |index: u32| needs.memory_type_bits & (1 << index) != 0;
                    let local = vk::MemoryPropertyFlags::DEVICE_LOCAL;
                    let type_index = (0..types.memory_type_count)
                        .find(|&index| {
                            fits(index)
                                && types.memory_types[index as usize]
                                    .property_flags
                                    .contains(local)
                        })
                        .or_else(|| (0..types.memory_type_count).find(|&index| fits(index)))
                        .unwrap();
                    let memory = raw
                        .device
                        .allocate_memory(
                            &vk::MemoryAllocateInfo::default()
                                .allocation_size(needs.size)
                                .memory_type_index(type_index),
                            None,
                        )
                        .unwrap();
                    raw.device.bind_image_memory(image, memory, 0).unwrap();
                    (image, memory)
                }
            })
            .collect();
        let copies = (copies.into_iter())
            .map(|(image, aspect, extent, at)| PlaneCopy {
                image,
                aspect,
                extent,
                at,
            })
            .collect();
        FrameImages {
            images,
            copies,
            planar,
        }
    }

    /// Record: the planes from `source` (a `VkBuffer` and the frame's byte 0 in it) into the
    /// images, then from the images onto `frame` likewise, visible to the host after.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording on the images' family; both buffers hold [`FRAME`] bytes
    /// from their offsets and live until the commands completed.
    unsafe fn record_copies(
        &self,
        vk_device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        (source, source_at): (vk::Buffer, u64),
        (frame, frame_at): (vk::Buffer, u64),
    ) {
        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let barriers = |from: vk::ImageLayout,
                        to: vk::ImageLayout,
                        src_access: vk::AccessFlags,
                        dst_access: vk::AccessFlags|
         -> Vec<vk::ImageMemoryBarrier<'static>> {
            (self.images.iter())
                .map(|&(image, _)| {
                    vk::ImageMemoryBarrier::default()
                        .src_access_mask(src_access)
                        .dst_access_mask(dst_access)
                        .old_layout(from)
                        .new_layout(to)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(image)
                        .subresource_range(range)
                })
                .collect()
        };
        let region = |copy: &PlaneCopy, at: u64| vk::BufferImageCopy {
            buffer_offset: at + copy.at,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: copy.aspect,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: vk::Offset3D::default(),
            image_extent: copy.extent,
        };
        let transfer = vk::PipelineStageFlags::TRANSFER;
        // SAFETY: the caller's contract.
        unsafe {
            vk_device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                transfer,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers(
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::TRANSFER_WRITE,
                ),
            );
            for copy in &self.copies {
                vk_device.cmd_copy_buffer_to_image(
                    command_buffer,
                    source,
                    self.images[copy.image].0,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[region(copy, source_at)],
                );
            }
            vk_device.cmd_pipeline_barrier(
                command_buffer,
                transfer,
                transfer,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers(
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::TRANSFER_READ,
                ),
            );
            for copy in &self.copies {
                vk_device.cmd_copy_image_to_buffer(
                    command_buffer,
                    self.images[copy.image].0,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    frame,
                    &[region(copy, frame_at)],
                );
            }
            visible_after_transfer(vk_device, command_buffer);
        }
    }

    /// Destroy the images and free their memory.
    ///
    /// # Safety
    ///
    /// No command that uses them is pending.
    unsafe fn destroy(self, device: &Device) {
        // SAFETY: the caller's contract; the handles are used while the device lives.
        unsafe {
            let raw = device.raw();
            for (image, memory) in self.images {
                raw.device.destroy_image(image, None);
                raw.device.free_memory(memory, None);
            }
        }
    }
}

/// Record a barrier making the transfers before it visible to the host and every later command.
///
/// # Safety
///
/// `command_buffer` is recording.
unsafe fn visible_after_transfer(vk_device: &ash::Device, command_buffer: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(
            vk::AccessFlags::HOST_READ
                | vk::AccessFlags::MEMORY_READ
                | vk::AccessFlags::MEMORY_WRITE,
        );
    // SAFETY: the caller's contract.
    unsafe {
        vk_device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::HOST | vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );
    }
}

/// The byte the caller's source holds at `offset` in round `round`.
fn source_byte(round: u32, offset: u64) -> u8 {
    (offset * 13 + offset / 2011 + u64::from(round) * 29 + 7) as u8
}

/// The byte the CPU writes into a frame at `offset` before the caller's copy.
fn cpu_byte(offset: u64) -> u8 {
    (offset * 5 + offset / 997 + 3) as u8
}

/// A `Shared` buffer holding `bytes`.
fn shared_with(device: &Device, bytes: &[u8]) -> Buffer {
    device
        .new_buffer_with_bytes(bytes, ResourceOptions::shared())
        .unwrap()
}

/// `length` bytes of `buffer` from byte `at`, as a blit of the runtime's copies them into a
/// `Shared` buffer the host reads.
fn blitted(device: &Device, buffer: &Buffer, at: u64, length: u64) -> Vec<u8> {
    let read = device
        .new_buffer(length, ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_buffer(buffer, at, &read, 0, length)
        .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    host_bytes(&read, length)
}

/// The first `length` bytes of a `Shared` buffer, read through `contents`.
fn host_bytes(buffer: &Buffer, length: u64) -> Vec<u8> {
    let contents = buffer.contents().unwrap();
    // SAFETY: the buffer holds `length` bytes from `contents`, which no command writes now.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), length as usize) }.to_vec()
}

/// The first offset where `read` differs from `want`, and what each holds there.
fn first_difference(read: &[u8], want: &[u8]) -> Option<(usize, u8, u8)> {
    (read.iter().zip(want).enumerate())
        .find(|(_, (read, want))| read != want)
        .map(|(at, (&read, &want))| (at, read, want))
}

/// A `Shared` frame laying byte `0x20` on a multiple of 256, filled through `contents`, then
/// written by the caller's copy of both planes from its images through the frame's `VkBuffer`:
/// each of [`ROUNDS`] new frames reads the copy through `contents` and through a blit, and the
/// caller's submissions and the runtime's command buffers all complete (no device loss).
#[test]
fn a_shared_aligned_frame_written_on_the_caller_s_queue_reads_the_writes() {
    let scratch = Scratch::new("caller-shared-frame");
    let device = device(&scratch.0);
    let caller = Caller::of(&device);
    let images = FrameImages::new(&device, caller.family());
    for round in 0..ROUNDS {
        let want: Vec<u8> = (0..FRAME).map(|at| source_byte(round, at)).collect();
        let source = shared_with(&device, &want);
        let descriptor =
            BufferDescriptor::new(FRAME, ResourceOptions::shared(), BufferContents::Zero)
                .with_aligned_byte(K);
        let frame = device.new_buffer_with_descriptor(descriptor).unwrap();
        let contents = frame.contents().unwrap();
        // SAFETY: the frame's `FRAME` bytes from `contents`, which no command uses yet.
        let bytes = unsafe { std::slice::from_raw_parts_mut(contents.as_ptr(), FRAME as usize) };
        (bytes.iter_mut().zip(0..)).for_each(|(byte, at)| *byte = cpu_byte(at));
        // SAFETY: the handles are used while both buffers live, past the caller's submission,
        // which `run` waits for; nothing of the runtime's uses either meanwhile.
        let ran = unsafe {
            let from = source.vk_buffer();
            let to = frame.vk_buffer();
            if round == 0 {
                eprintln!(
                    "CALLER-QUEUE-WRITES: {}: {}, {} images, the frame's byte 0 at {} of its \
                     VkBuffer, contents at {:#x}",
                    device.name(),
                    caller.describe(),
                    if images.planar { "NV12" } else { "plane" },
                    to.1,
                    contents.as_ptr() as usize,
                );
            }
            caller.run(&device, |vk_device, command_buffer| {
                images.record_copies(vk_device, command_buffer, from, to)
            })
        };
        assert_eq!(
            ran,
            Ok(()),
            "round {round}: the caller's copy onto the frame failed (lost: {})",
            device.is_lost()
        );
        let read = host_bytes(&frame, FRAME);
        assert_eq!(
            first_difference(&read, &want),
            None,
            "round {round}: the frame's contents after the caller's copy"
        );
        let read = blitted(&device, &frame, 0, FRAME);
        assert_eq!(
            first_difference(&read, &want),
            None,
            "round {round}: a blit of the frame after the caller's copy"
        );
        drop((frame, source));
    }
    assert!(!device.is_lost());
    eprintln!(
        "CALLER-QUEUE-WRITES: {}: {ROUNDS} Shared frames read the caller's copies",
        device.name()
    );
    // SAFETY: every command that used them completed.
    unsafe { images.destroy(&device) };
}

/// A new `Private` buffer laying byte `0x20` on a multiple of 256, handed out before any
/// submission while its fill with zeros waits, then written by the caller's copy through its
/// `VkBuffer`: a blit of the runtime's after it reads the copy, not zeros.
#[test]
fn a_new_private_buffer_handed_out_reads_the_caller_s_writes_not_its_zeros() {
    let scratch = Scratch::new("caller-private-buffer");
    let device = device(&scratch.0);
    let caller = Caller::of(&device);
    let want: Vec<u8> = (0..FRAME).map(|at| source_byte(1, at)).collect();
    let source = shared_with(&device, &want);
    let descriptor = BufferDescriptor::new(FRAME, ResourceOptions::private(), BufferContents::Zero)
        .with_aligned_byte(K);
    let frame = device.new_buffer_with_descriptor(descriptor).unwrap();
    let waited = (device.shared().textures.first_moves).fills_wait(frame.at(0).0);
    // SAFETY: the handles are used while both buffers live, past the caller's submission, which
    // `run` waits for; nothing of the runtime's uses either meanwhile.
    let ran = unsafe {
        let (from, from_at) = source.vk_buffer();
        let (to, to_at) = frame.vk_buffer();
        caller.run(&device, |vk_device, command_buffer| {
            let region = vk::BufferCopy {
                src_offset: from_at,
                dst_offset: to_at,
                size: FRAME,
            };
            vk_device.cmd_copy_buffer(command_buffer, from, to, &[region]);
            visible_after_transfer(vk_device, command_buffer);
        })
    };
    assert_eq!(ran, Ok(()), "the caller's copy onto the buffer failed");
    let read = blitted(&device, &frame, 0, FRAME);
    let difference = first_difference(&read, &want);
    eprintln!(
        "CALLER-QUEUE-WRITES: {}: {}, a new Private buffer's fill waited as it was handed out: \
         {waited}; the blit after the caller's copy read {}",
        device.name(),
        caller.describe(),
        match difference {
            None => "the copy".to_owned(),
            Some((at, read, want)) => format!("{read:#x} at {at}, not {want:#x}"),
        }
    );
    assert!(
        waited,
        "the buffer's fill no longer waited as it was handed out: the test proves nothing"
    );
    assert_eq!(
        difference, None,
        "the buffer after the caller's copy (its zeros made after it?)"
    );
}

/// A texture over a new `Private` buffer laying byte `0x20` on a multiple of 256, its plane at
/// that byte in place (the linear image aliasing the buffer's memory), handed out before any
/// submission, then written by the caller's copy into its `VkImage`: a blit of the texture, and
/// one of the buffer's bytes under it, read the copy. Where the plane is a shadow, which hands
/// out no image, nothing is checked.
#[test]
fn a_texture_over_a_new_private_buffer_handed_out_reads_the_caller_s_writes() {
    const WIDE: u32 = 3840;
    const HIGH: u32 = 2160;
    const PLANE: u64 = WIDE as u64 * HIGH as u64;
    let scratch = Scratch::new("caller-private-plane");
    let device = device(&scratch.0);
    let caller = Caller::of(&device);
    let want: Vec<u8> = (0..PLANE).map(|at| source_byte(2, at)).collect();
    let source = shared_with(&device, &want);
    let options = ResourceOptions::private();
    let descriptor =
        BufferDescriptor::new(K + PLANE, options, BufferContents::Zero).with_aligned_byte(K);
    let frame = device.new_buffer_with_descriptor(descriptor).unwrap();
    let mut texture_descriptor =
        TextureDescriptor::texture_2d(PixelFormat::R8Unorm, WIDE.into(), HIGH.into(), false);
    texture_descriptor.usage = TextureUsage::SHADER_READ;
    texture_descriptor.resource_options = options;
    let shadows = device.work_counts().shadowed_textures;
    let texture = frame
        .new_texture_with_iosurface_plane(
            &texture_descriptor,
            IoSurfacePlane::new(K, WIDE.into(), true),
        )
        .unwrap();
    let waited = (device.shared().textures.first_moves).fills_wait(frame.at(0).0);
    // SAFETY: the image is used while the texture lives, past the caller's submission, which
    // `run` waits for, in `GENERAL` as its contract says.
    let image = unsafe { texture.vk_image() };
    let Some(image) = image.filter(|_| device.work_counts().shadowed_textures == shadows) else {
        eprintln!(
            "CALLER-QUEUE-WRITES: {}: the plane is a shadow, no image handed out",
            device.name()
        );
        return;
    };
    // SAFETY: as above, and the source lives past the submission.
    let ran = unsafe {
        let (from, from_at) = source.vk_buffer();
        caller.run(&device, |vk_device, command_buffer| {
            let region = vk::BufferImageCopy {
                buffer_offset: from_at,
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
                    width: WIDE,
                    height: HIGH,
                    depth: 1,
                },
            };
            vk_device.cmd_copy_buffer_to_image(
                command_buffer,
                from,
                image,
                vk::ImageLayout::GENERAL,
                &[region],
            );
            visible_after_transfer(vk_device, command_buffer);
        })
    };
    assert_eq!(ran, Ok(()), "the caller's copy into the image failed");
    let read = device.new_buffer(PLANE, ResourceOptions::shared()).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        &texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(WIDE.into(), HIGH.into(), 1),
        &read,
        0,
        WIDE.into(),
        PLANE,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let texels = first_difference(&host_bytes(&read, PLANE), &want);
    let under = first_difference(&blitted(&device, &frame, K, PLANE), &want);
    eprintln!(
        "CALLER-QUEUE-WRITES: {}: {}, a texture over a new Private buffer (its fill waited as \
         the texture was made: {waited}) handed out: its texels {texels:?}, the bytes under it \
         {under:?} (None: the caller's copy)",
        device.name(),
        caller.describe(),
    );
    assert_eq!(texels, None, "the texture after the caller's copy");
    assert_eq!(
        under, None,
        "the buffer's bytes under the texture after the caller's copy"
    );
}
