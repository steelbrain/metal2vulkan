//! Buffers over `VkBuffer`s the caller made and bound to its own memory
//! (`Device::new_buffer_with_vk_buffer`, `runtime/wrapped_buffer.rs`) on the device the cfg
//! selects, under the validation layer. The test makes each `VkBuffer` and `VkDeviceMemory`
//! itself through the device's raw handles, as a consumer that imports memory with plain Vulkan
//! does, and destroys them after the runtime's buffer and every command buffer that used it.
//!
//! - A `Shared` buffer over host-visible coherent memory the test mapped: its `contents` is the
//!   test's mapping, and a draw of the owned fixture `ob3_vertex_input` that reads its vertices
//!   from the buffer and writes them back into it, blits that copy out of it and fill it, and
//!   textures over it that a kernel samples and writes (`host_bytes.rs`'s cases, through images
//!   their texels are copied into and back, `shadowed-texture`, the notice naming the caller's
//!   buffer) all reach the test's memory, as the runtime's runs of the cases over textures of
//!   their own do.
//! - A `Private` buffer in device memory takes a copy in and gives it back.
//! - A buffer the runtime could not have made is refused, naming why: too small for the length
//!   padded to 16 bytes, made without the usage `Device::vk_buffer_usage` names, `Shared` without
//!   a host address; `Managed` storage is refused as Metal-shaped options are.

use super::host_bytes::{buffer_output, dispatch, over, SAMPLER, TEXTURE_WRITER};
use super::vertex_strides::pipeline;
use super::{
    at_plan, buffer_with, layer_reported_nothing_at_all, public_case, read_buffer, run_case,
    validated_device, validated_device_with, Messages, ScratchCache, WAIT,
};
use crate::gpu::TEST_DEVICE;
use ash::vk;
use metal2vulkan::metal::PrimitiveType;
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, CommandBufferStatus, CpuCacheMode, Degradation, Device, Error,
    HazardTrackingMode, Invalid, Problem, RenderPassDescriptor, ResourceOptions, StorageMode,
    VkBufferMemory, WrappedBufferProblem,
};
use std::ptr::NonNull;

/// The caller's buffers' size.
const SIZE: u64 = 1024;

/// A `VkBuffer` of `size` bytes and `usage` the test made, bound at 0 to a dedicated allocation
/// of the first memory type with `flags`, mapped when host-visible.
struct Made {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    type_index: u32,
    mapped: Option<NonNull<u8>>,
    usage: vk::BufferUsageFlags,
}

impl Made {
    /// The buffer on `device`; `None` where no memory type the buffer takes has `flags`.
    ///
    /// # Safety
    ///
    /// The buffer and memory are destroyed ([`Made::destroy`]) after every runtime object and
    /// command buffer that uses them, while the device lives.
    unsafe fn new(
        device: &Device,
        size: u64,
        usage: vk::BufferUsageFlags,
        flags: vk::MemoryPropertyFlags,
    ) -> Option<Made> {
        // SAFETY: the handles are used only while the device lives.
        let raw = unsafe { device.raw() };
        let info = Made::info(size, usage, raw.queue_families);
        // SAFETY: the device is live and `info` valid.
        let buffer = unsafe { raw.device.create_buffer(&info, None) }.unwrap();
        // SAFETY: a live buffer of the device, and a physical device of the instance.
        let (requirements, properties) = unsafe {
            (
                raw.device.get_buffer_memory_requirements(buffer),
                raw.instance
                    .get_physical_device_memory_properties(raw.physical_device),
            )
        };
        let types = &properties.memory_types[..properties.memory_type_count as usize];
        let Some(type_index) = (0..types.len()).find(|&index| {
            requirements.memory_type_bits & (1 << index) != 0
                && types[index].property_flags.contains(flags)
        }) else {
            // SAFETY: unused.
            unsafe { raw.device.destroy_buffer(buffer, None) };
            return None;
        };
        let mut address =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let mut allocate = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(type_index as u32);
        if usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS) {
            allocate = allocate.push_next(&mut address);
        }
        // SAFETY: `allocate` and its chain live for the call; the type is the device's.
        let memory = unsafe { raw.device.allocate_memory(&allocate, None) }.unwrap();
        // SAFETY: the memory meets the buffer's requirements from offset 0, and is unbound.
        unsafe { raw.device.bind_buffer_memory(buffer, memory, 0) }.unwrap();
        let mapped = types[type_index]
            .property_flags
            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
            .then(|| {
                // SAFETY: host-visible memory, mapped once, whole.
                let pointer = unsafe {
                    raw.device
                        .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                }
                .unwrap();
                NonNull::new(pointer.cast::<u8>()).unwrap()
            });
        Some(Made {
            buffer,
            memory,
            type_index: type_index as u32,
            mapped,
            usage,
        })
    }

    /// The creation of a buffer of `size` bytes and `usage` on a device of queue families
    /// `families` (`DeviceHandles::queue_families`): concurrent among them where there are
    /// several, as the runtime wraps a buffer there.
    fn info(size: u64, usage: vk::BufferUsageFlags, families: &[u32]) -> vk::BufferCreateInfo<'_> {
        let info = vk::BufferCreateInfo::default().size(size).usage(usage);
        match families.len() > 1 {
            true => info
                .sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(families),
            false => info.sharing_mode(vk::SharingMode::EXCLUSIVE),
        }
    }

    /// Where the buffer is bound, with `contents` as its host address.
    fn memory(&self, contents: Option<NonNull<u8>>) -> VkBufferMemory {
        VkBufferMemory::new(self.memory, 0, self.type_index, contents)
    }

    /// The runtime's buffer of `length` bytes and `options` over this one, or its refusal.
    ///
    /// # Safety
    ///
    /// For `Shared` storage `contents` is this buffer's mapping.
    unsafe fn wrap(
        &self,
        device: &Device,
        length: u64,
        options: ResourceOptions,
        contents: Option<NonNull<u8>>,
    ) -> Result<Buffer, Error> {
        // SAFETY: the handles are used only for the call, while the device lives.
        let raw = unsafe { device.raw() };
        let info = Made::info(SIZE, self.usage, raw.queue_families);
        // SAFETY: the buffer was made with `info` and bound whole at 0 to the memory, allocated
        // with the device-address flag where the usage names device addresses; the caller's
        // contract for `contents`; `destroy` runs after the runtime's buffer is gone.
        unsafe {
            device.new_buffer_with_vk_buffer(
                self.buffer,
                &info,
                self.memory(contents),
                length,
                options,
            )
        }
    }

    /// The mapped bytes.
    fn bytes(&self) -> Vec<u8> {
        let mapped = self.mapped.expect("a mapped buffer");
        // SAFETY: `SIZE` mapped bytes, which no command buffer is using.
        unsafe { std::slice::from_raw_parts(mapped.as_ptr(), SIZE as usize) }.to_vec()
    }

    /// The mapped bytes, as 32-bit words.
    fn words(&self) -> Vec<u32> {
        self.bytes()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect()
    }

    /// Write `bytes` at the start of the mapped bytes.
    fn write(&self, bytes: &[u8]) {
        assert!(bytes.len() as u64 <= SIZE);
        let mapped = self.mapped.expect("a mapped buffer");
        // SAFETY: at most `SIZE` mapped bytes, which no command buffer is using.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped.as_ptr(), bytes.len()) };
    }

    /// Destroy the buffer and free its memory.
    ///
    /// # Safety
    ///
    /// Nothing uses them any more.
    unsafe fn destroy(self, device: &Device) {
        // SAFETY: the caller's contract.
        unsafe {
            let raw = device.raw();
            raw.device.destroy_buffer(self.buffer, None);
            raw.device.free_memory(self.memory, None);
        }
    }
}

/// The refusal of a buffer for `problem`.
fn wrapped(problem: WrappedBufferProblem) -> Error {
    Error::Invalid(Invalid::new(
        "new_buffer_with_vk_buffer",
        Problem::WrappedBuffer(problem),
    ))
}

/// Commit `command_buffer` and wait for it to complete.
fn run(command_buffer: &CommandBuffer) {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// The `Shared` buffer over the test's mapped memory: what the runtime reads and writes of it is
/// the test's memory, and the test's writes are what the runtime reads.
#[test]
fn a_callers_shared_buffer_is_a_metal_buffer() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), at_plan).unwrap();
        let usage = vk::BufferUsageFlags::from(device.vk_buffer_usage());
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        // SAFETY: destroyed below, after the buffer and every command buffer.
        let made = unsafe { Made::new(&device, SIZE, usage, host) }
            .expect("host-visible coherent memory for a buffer");
        let mapped = made.mapped.unwrap();
        let words: Vec<u8> = (0..SIZE / 4)
            .flat_map(|i| (100 + i as u32).to_le_bytes())
            .collect();
        made.write(&words);
        // SAFETY: `mapped` is the buffer's mapping.
        let buffer =
            unsafe { made.wrap(&device, SIZE, ResourceOptions::shared(), Some(mapped)) }.unwrap();
        assert_eq!(buffer.contents(), Some(mapped));
        assert_eq!(buffer.length(), SIZE);
        assert_eq!(buffer.storage_mode(), StorageMode::Shared);
        // SAFETY: only compared.
        assert_eq!(unsafe { buffer.vk_buffer() }, (made.buffer, 0));
        let queue = device.new_command_queue().unwrap();
        // A draw: each of 8 vertices reads its word from byte 0 and writes it at byte 512.
        let state = pipeline(&device, Some(4));
        let mut pass = RenderPassDescriptor::new();
        pass.render_target_width = 1;
        pass.render_target_height = 1;
        pass.default_raster_sample_count = 1;
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&state).unwrap();
        encoder.set_vertex_buffer(Some(&buffer), 0, 0).unwrap();
        encoder.set_vertex_buffer(Some(&buffer), 512, 1).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Point, 0, 8, 1, 0)
            .unwrap();
        encoder.end_encoding().unwrap();
        // Blits: a copy out of the buffer, and a fill of its last 64 bytes.
        let out = buffer_with(&device, &[0u8; 16]).unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_buffer(&buffer, 16, &out, 0, 16)
            .unwrap();
        blit.fill_buffer(&buffer, SIZE - 64..SIZE, 0x5a).unwrap();
        blit.end_encoding().unwrap();
        run(&command_buffer);
        let after = made.words();
        assert_eq!(&after[128..136], &(100..108).collect::<Vec<u32>>());
        assert_eq!(&after[240..], &[0x5a5a_5a5a; 16]);
        let copied: Vec<u32> = read_buffer(&out, 0, 16)
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        assert_eq!(copied, [104, 105, 106, 107]);
        // Textures over the buffer, from byte 0: a kernel samples one, and another writes one.
        let case = public_case(SAMPLER).unwrap();
        let expected = run_case(&device, &case).unwrap().output;
        made.write(&case.compute.textures[&0].bytes);
        let (buffers, resources) = over(&device, &case, &[(0, &buffer)]);
        dispatch(&queue, &case, &buffers, &resources);
        assert_eq!(buffer_output(&case, &buffers), expected, "{SAMPLER}");
        drop((buffers, resources));
        let case = public_case(TEXTURE_WRITER).unwrap();
        let expected = run_case(&device, &case).unwrap().output;
        made.write(&case.compute.textures[&0].bytes);
        let (buffers, resources) = over(&device, &case, &[(0, &buffer)]);
        dispatch(&queue, &case, &buffers, &resources);
        drop((buffers, resources));
        assert_eq!(made.bytes()[..expected.len()], expected, "{TEXTURE_WRITER}");
        let shadowed = messages
            .take_notices()
            .into_iter()
            .filter_map(|message| message.degradation)
            .find(|notice| notice.degradation == Degradation::ShadowedTexture)
            .expect("the textures' copies are announced");
        assert!(
            shadowed.reason.contains("the caller's VkBuffer"),
            "{}",
            shadowed.reason
        );
        // SAFETY: refused before it wraps anything.
        unsafe {
            assert_eq!(
                made.wrap(&device, SIZE, ResourceOptions::shared(), None),
                Err(wrapped(WrappedBufferProblem::Contents))
            );
            assert_eq!(
                made.wrap(&device, SIZE, ResourceOptions::private(), Some(mapped)),
                Err(wrapped(WrappedBufferProblem::Contents))
            );
        }
        drop(buffer);
        // SAFETY: the runtime's buffer is gone, and every command buffer completed.
        unsafe { made.destroy(&device) };
    }
    layer_reported_nothing_at_all(&messages);
}

/// A `Private` buffer in device memory takes a copy in and gives it back; a buffer the runtime
/// could not have made is refused, naming why.
#[test]
fn a_callers_private_buffer_copies_and_contradictions_are_refused() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let usage = vk::BufferUsageFlags::from(device.vk_buffer_usage());
        // SAFETY: destroyed below, after the buffer and every command buffer.
        let made =
            unsafe { Made::new(&device, SIZE, usage, vk::MemoryPropertyFlags::DEVICE_LOCAL) }
                .expect("device-local memory for a buffer");
        // SAFETY: `Private` storage takes no host address.
        let buffer = unsafe { made.wrap(&device, SIZE, ResourceOptions::private(), None) }.unwrap();
        assert_eq!(buffer.contents(), None);
        let bytes: Vec<u8> = (0..SIZE).map(|i| (i * 7) as u8).collect();
        let source = buffer_with(&device, &bytes).unwrap();
        let back = buffer_with(&device, &vec![0u8; SIZE as usize]).unwrap();
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_buffer(&source, 0, &buffer, 0, SIZE)
            .unwrap();
        blit.copy_from_buffer_to_buffer(&buffer, 0, &back, 0, SIZE)
            .unwrap();
        blit.end_encoding().unwrap();
        run(&command_buffer);
        assert_eq!(read_buffer(&back, 0, SIZE).unwrap(), bytes);
        drop(buffer);
        // Refusals.
        // SAFETY: each is refused before it wraps anything.
        unsafe {
            assert_eq!(
                made.wrap(&device, SIZE + 1, ResourceOptions::private(), None),
                Err(wrapped(WrappedBufferProblem::Size {
                    buffer: SIZE,
                    needed: SIZE + 16
                }))
            );
            let managed = ResourceOptions::new(
                StorageMode::Managed,
                CpuCacheMode::DefaultCache,
                HazardTrackingMode::Default,
            );
            assert_eq!(
                made.wrap(&device, SIZE, managed, None),
                Err(Error::Invalid(Invalid::new(
                    "new_buffer_with_vk_buffer",
                    Problem::StorageMode
                )))
            );
        }
        let narrow_usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        // SAFETY: destroyed below; nothing wraps it.
        let narrow = unsafe { Made::new(&device, SIZE, narrow_usage, host) }.unwrap();
        // SAFETY: refused before it wraps anything.
        unsafe {
            assert_eq!(
                narrow.wrap(&device, SIZE, ResourceOptions::shared(), narrow.mapped),
                Err(wrapped(WrappedBufferProblem::Usage {
                    missing: (usage & !narrow_usage).into()
                }))
            );
            assert_eq!(
                narrow.wrap(&device, SIZE, ResourceOptions::private(), narrow.mapped),
                Err(wrapped(WrappedBufferProblem::Usage {
                    missing: (usage & !narrow_usage).into()
                }))
            );
        }
        // SAFETY: nothing uses them: every command buffer completed and every buffer is gone.
        unsafe {
            narrow.destroy(&device);
            made.destroy(&device);
        }
    }
    layer_reported_nothing_at_all(&messages);
}
