//! The runtime's device measurements (`corpus-vulkan --probe-runtime`): facts the Metal-like
//! runtime's design takes from each device, read on the device a venue run selects, in a validated
//! context of its own
//! ([`super::device::with_runtime_probe_context`]).
//!
//! - **Host memory without a copy** ([`host_memory`]): whether the device has
//!   `VK_EXT_external_memory_host`, its `minImportedHostPointerAlignment`, the memory types a real
//!   page-aligned host allocation can be imported as, and whether a storage buffer bound to the
//!   imported memory works through a descriptor and through its device address (a dispatch reads
//!   and writes it, and the host sees the result in its own allocation).
//! - **Textures over buffers** ([`linear_images`]): per format Metal allows on a
//!   buffer, whether a linear image with sampled and storage usage exists, the row pitch the
//!   driver chooses for several widths, whether such an image aliases memory a buffer already
//!   uses at a chosen offset (both ways, by copies), and, where
//!   `VK_EXT_image_drm_format_modifier` is present, whether the caller can choose the row pitch
//!   through `DRM_FORMAT_MOD_LINEAR` with an explicit layout.
//! - **A fetch through an input attachment** ([`input_fetch`]): at 4 samples beside a depth/stencil
//!   attachment, with and without early fragment tests, the bytes a fragment shader stores that
//!   reads its own sample of the color attachment through an input attachment aliasing it and
//!   writes the color back, over one draw of overlapping triangles.
//!
//! Each step records what the device answered, or the error that stopped it, with the
//! validation-layer messages it drew; one step's failure does not stop the next. Every submission
//! waits through the hang protocol ([`crate::gpu::wait`]), and the probe's two shaders are
//! loop-free. A measurement drawn with a validation message is not evidence: the report says so.

use super::device::{with_runtime_probe_context, VulkanContext};
use super::pipeline::create_shader_module;
use crate::gpu::wait;
use crate::gpu::DeviceType;
use ash::vk;
use serde::Serialize;

mod host_memory;
mod input_fetch;
mod linear_images;
#[cfg(test)]
mod tests;

/// Measure the runtime's device facts on the device `device_type` selects
/// (`corpus-vulkan --probe-runtime`): the JSON report ([`Report`]).
pub fn probe_runtime(device_type: DeviceType) -> Result<serde_json::Value, String> {
    let report = with_runtime_probe_context(device_type, |context| Ok(measure(context)))?;
    serde_json::to_value(&report).map_err(|error| format!("serialize the runtime probe: {error}"))
}

/// The probe's report.
#[derive(Debug, Serialize)]
pub(super) struct Report {
    /// The command that wrote it.
    probe: &'static str,
    device: Identity,
    /// What the validation layer reported while the probe's context opened.
    opening_messages: Vec<String>,
    host_memory: host_memory::HostMemory,
    linear_textures: linear_images::LinearTextures,
    input_fetch: input_fetch::InputFetch,
    /// Every validation message of the run (the steps' own, and the opening's): a report with one
    /// is not evidence.
    validation_messages: usize,
}

/// Which device answered.
#[derive(Debug, Serialize)]
struct Identity {
    device: String,
    vendor_id: u32,
    device_id: u32,
    /// `VkPhysicalDeviceType`, by name.
    device_type: &'static str,
    /// `major.minor.patch`.
    api_version: String,
    driver_version: u32,
    /// `VkDriverId`, raw.
    driver_id: i32,
    driver_name: String,
    driver_info: String,
    /// The host's page size (`sysconf(_SC_PAGESIZE)`): Metal's alignment for no-copy buffers.
    host_page_size: u64,
    /// `bufferImageGranularity`.
    buffer_image_granularity: u64,
    /// Whether the context enabled `bufferDeviceAddress`.
    buffer_device_address: bool,
}

fn measure(context: &VulkanContext) -> Report {
    let opening_messages = context.take_messages();
    let device = identity(context);
    let host_memory = host_memory::measure(context);
    let linear_textures = linear_images::measure(context);
    let input_fetch = input_fetch::measure(context);
    let validation_messages = opening_messages.len()
        + host_memory.validation_messages.len()
        + linear_textures.validation_messages.len()
        + input_fetch.validation_messages.len();
    Report {
        probe: "corpus-vulkan --probe-runtime",
        device,
        opening_messages,
        host_memory,
        linear_textures,
        input_fetch,
        validation_messages,
    }
}

fn identity(context: &VulkanContext) -> Identity {
    // SAFETY: the context's physical device belongs to its live instance.
    let properties = unsafe {
        context
            .instance
            .get_physical_device_properties(context.physical)
    };
    let text = |raw: &[std::ffi::c_char]| {
        // SAFETY: the implementation NUL-terminates the strings of its properties.
        unsafe { std::ffi::CStr::from_ptr(raw.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    };
    let api = properties.api_version;
    Identity {
        device: text(&properties.device_name),
        vendor_id: properties.vendor_id,
        device_id: properties.device_id,
        device_type: device_type_name(properties.device_type),
        api_version: format!(
            "{}.{}.{}",
            vk::api_version_major(api),
            vk::api_version_minor(api),
            vk::api_version_patch(api)
        ),
        driver_version: properties.driver_version,
        driver_id: context.driver.driver_id,
        driver_name: context.driver.driver_name.clone(),
        driver_info: context.driver.driver_info.clone(),
        host_page_size: host_page_size(),
        buffer_image_granularity: properties.limits.buffer_image_granularity,
        buffer_device_address: buffer_device_address(context),
    }
}

/// Whether the context enabled `bufferDeviceAddress`.
fn buffer_device_address(context: &VulkanContext) -> bool {
    context
        .enabled
        .features
        .contains(&metal2vulkan::device::VkFeature::BufferDeviceAddress)
}

fn device_type_name(device_type: vk::PhysicalDeviceType) -> &'static str {
    match device_type {
        vk::PhysicalDeviceType::INTEGRATED_GPU => "integrated_gpu",
        vk::PhysicalDeviceType::DISCRETE_GPU => "discrete_gpu",
        vk::PhysicalDeviceType::VIRTUAL_GPU => "virtual_gpu",
        vk::PhysicalDeviceType::CPU => "cpu",
        _ => "other",
    }
}

/// The host's page size.
fn host_page_size() -> u64 {
    // SAFETY: `sysconf` reads a configuration value.
    let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    u64::try_from(size).unwrap_or(0)
}

/// Whether the device lists the extension `name`.
fn device_lists(context: &VulkanContext, name: &std::ffi::CStr) -> Result<bool, String> {
    // SAFETY: the context's physical device belongs to its live instance.
    let listed = unsafe {
        context
            .instance
            .enumerate_device_extension_properties(context.physical)
    }
    .map_err(|error| format!("enumerate Vulkan device extensions: {error}"))?;
    Ok(listed.iter().any(|extension| {
        // SAFETY: the implementation NUL-terminates each extension name.
        let listed = unsafe { std::ffi::CStr::from_ptr(extension.extension_name.as_ptr()) };
        listed == name
    }))
}

/// `value` rounded up to a multiple of `alignment` (a power of two, or any nonzero value).
fn align_up(value: u64, alignment: u64) -> u64 {
    if alignment <= 1 {
        value
    } else {
        value.div_ceil(alignment) * alignment
    }
}

/// The names of the bits of `raw` that `table` names, in table order, then each other set bit
/// in hexadecimal.
fn flag_names(raw: u64, table: &[(u64, &str)]) -> Vec<String> {
    let mut names = table
        .iter()
        .filter(|(bit, _)| raw & bit != 0)
        .map(|(_, name)| (*name).to_string())
        .collect::<Vec<_>>();
    let known = table.iter().fold(0, |known, (bit, _)| known | bit);
    let mut rest = raw & !known;
    while rest != 0 {
        let bit = rest.isolate_lowest_one();
        names.push(format!("{bit:#x}"));
        rest &= !bit;
    }
    names
}

/// A `VkResult` by its registry name where the probe can meet it, else its raw value.
fn result_name(result: vk::Result) -> String {
    let name = match result {
        vk::Result::ERROR_OUT_OF_HOST_MEMORY => "VK_ERROR_OUT_OF_HOST_MEMORY",
        vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => "VK_ERROR_OUT_OF_DEVICE_MEMORY",
        vk::Result::ERROR_INITIALIZATION_FAILED => "VK_ERROR_INITIALIZATION_FAILED",
        vk::Result::ERROR_DEVICE_LOST => "VK_ERROR_DEVICE_LOST",
        vk::Result::ERROR_FORMAT_NOT_SUPPORTED => "VK_ERROR_FORMAT_NOT_SUPPORTED",
        vk::Result::ERROR_INVALID_EXTERNAL_HANDLE => "VK_ERROR_INVALID_EXTERNAL_HANDLE",
        vk::Result::ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT => {
            "VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT"
        }
        vk::Result::ERROR_UNKNOWN => "VK_ERROR_UNKNOWN",
        other => return format!("VkResult({})", other.as_raw()),
    };
    name.to_string()
}

const MEMORY_PROPERTIES: &[(u64, &str)] = &[
    (0x1, "DEVICE_LOCAL"),
    (0x2, "HOST_VISIBLE"),
    (0x4, "HOST_COHERENT"),
    (0x8, "HOST_CACHED"),
    (0x10, "LAZILY_ALLOCATED"),
    (0x20, "PROTECTED"),
];

const HEAP_FLAGS: &[(u64, &str)] = &[(0x1, "DEVICE_LOCAL"), (0x2, "MULTI_INSTANCE")];

/// One memory type of the device.
#[derive(Debug, Serialize)]
struct MemoryType {
    index: u32,
    flags: Vec<String>,
    heap: u32,
    heap_flags: Vec<String>,
    heap_bytes: u64,
}

/// The device's memory types among `bits`.
fn memory_types(context: &VulkanContext, bits: u32) -> Vec<MemoryType> {
    // SAFETY: the context's physical device belongs to its live instance.
    let properties = unsafe {
        context
            .instance
            .get_physical_device_memory_properties(context.physical)
    };
    properties
        .memory_types_as_slice()
        .iter()
        .enumerate()
        .filter(|(index, _)| bits & (1 << index) != 0)
        .map(|(index, memory_type)| {
            let heap = properties.memory_heaps[memory_type.heap_index as usize];
            MemoryType {
                index: index as u32,
                flags: flag_names(
                    u64::from(memory_type.property_flags.as_raw()),
                    MEMORY_PROPERTIES,
                ),
                heap: memory_type.heap_index,
                heap_flags: flag_names(u64::from(heap.flags.as_raw()), HEAP_FLAGS),
                heap_bytes: heap.size,
            }
        })
        .collect()
}

/// The first memory type among `bits` whose flags hold `wanted`.
fn memory_type_with(
    flags: &[vk::MemoryPropertyFlags],
    bits: u32,
    wanted: vk::MemoryPropertyFlags,
) -> Option<u32> {
    flags
        .iter()
        .enumerate()
        .find(|(index, flags)| bits & (1 << index) != 0 && flags.contains(wanted))
        .map(|(index, _)| index as u32)
}

/// [`memory_type_with`] over the context's device.
fn host_coherent_type(context: &VulkanContext, bits: u32) -> Option<u32> {
    // SAFETY: the context's physical device belongs to its live instance.
    let properties = unsafe {
        context
            .instance
            .get_physical_device_memory_properties(context.physical)
    };
    let flags = properties
        .memory_types_as_slice()
        .iter()
        .map(|memory_type| memory_type.property_flags)
        .collect::<Vec<_>>();
    memory_type_with(
        &flags,
        bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
}

/// A resource's memory requirements, with whether it requires or prefers a dedicated allocation
/// (`VkMemoryDedicatedRequirements`, core from Vulkan 1.1).
struct Requirements {
    memory: vk::MemoryRequirements,
    requires_dedicated: bool,
    prefers_dedicated: bool,
}

/// `image`'s [`Requirements`].
fn image_requirements(context: &VulkanContext, image: vk::Image) -> Requirements {
    let mut dedicated = vk::MemoryDedicatedRequirements::default();
    let mut requirements = vk::MemoryRequirements2::default().push_next(&mut dedicated);
    // SAFETY: the image is a live, single-plane image of this device.
    unsafe {
        context.device.get_image_memory_requirements2(
            &vk::ImageMemoryRequirementsInfo2::default().image(image),
            &mut requirements,
        )
    };
    Requirements {
        memory: requirements.memory_requirements,
        requires_dedicated: dedicated.requires_dedicated_allocation != vk::FALSE,
        prefers_dedicated: dedicated.prefers_dedicated_allocation != vk::FALSE,
    }
}

/// `buffer`'s [`Requirements`].
fn buffer_requirements(context: &VulkanContext, buffer: vk::Buffer) -> Requirements {
    let mut dedicated = vk::MemoryDedicatedRequirements::default();
    let mut requirements = vk::MemoryRequirements2::default().push_next(&mut dedicated);
    // SAFETY: the buffer is a live buffer of this device.
    unsafe {
        context.device.get_buffer_memory_requirements2(
            &vk::BufferMemoryRequirementsInfo2::default().buffer(buffer),
            &mut requirements,
        )
    };
    Requirements {
        memory: requirements.memory_requirements,
        requires_dedicated: dedicated.requires_dedicated_allocation != vk::FALSE,
        prefers_dedicated: dedicated.prefers_dedicated_allocation != vk::FALSE,
    }
}

/// One object the probe created on the context's device.
enum Object {
    Buffer(vk::Buffer),
    Image(vk::Image),
    Memory(vk::DeviceMemory),
    CommandPool(vk::CommandPool),
    Fence(vk::Fence),
    ShaderModule(vk::ShaderModule),
    DescriptorSetLayout(vk::DescriptorSetLayout),
    PipelineLayout(vk::PipelineLayout),
    Pipeline(vk::Pipeline),
    DescriptorPool(vk::DescriptorPool),
    ImageView(vk::ImageView),
    RenderPass(vk::RenderPass),
    Framebuffer(vk::Framebuffer),
}

/// The objects one step created, destroyed in reverse order of creation when it drops. A step
/// drops them only once nothing it submitted is in flight: [`Owned::submit`] returns only then
/// (a wait that times out leaves the process through the hang protocol instead).
struct Owned<'a> {
    context: &'a VulkanContext,
    objects: Vec<Object>,
}

impl Drop for Owned<'_> {
    fn drop(&mut self) {
        let device = &self.context.device;
        for object in self.objects.iter().rev() {
            // SAFETY: each object was created on this device by this step, nothing in flight
            // uses it ([`Owned`]), and objects created later (which may use it) went first.
            unsafe {
                match *object {
                    Object::Buffer(buffer) => device.destroy_buffer(buffer, None),
                    Object::Image(image) => device.destroy_image(image, None),
                    Object::Memory(memory) => device.free_memory(memory, None),
                    Object::CommandPool(pool) => device.destroy_command_pool(pool, None),
                    Object::Fence(fence) => device.destroy_fence(fence, None),
                    Object::ShaderModule(module) => device.destroy_shader_module(module, None),
                    Object::DescriptorSetLayout(layout) => {
                        device.destroy_descriptor_set_layout(layout, None)
                    }
                    Object::PipelineLayout(layout) => device.destroy_pipeline_layout(layout, None),
                    Object::Pipeline(pipeline) => device.destroy_pipeline(pipeline, None),
                    Object::DescriptorPool(pool) => device.destroy_descriptor_pool(pool, None),
                    Object::ImageView(view) => device.destroy_image_view(view, None),
                    Object::RenderPass(pass) => device.destroy_render_pass(pass, None),
                    Object::Framebuffer(framebuffer) => {
                        device.destroy_framebuffer(framebuffer, None)
                    }
                }
            }
        }
    }
}

impl<'a> Owned<'a> {
    fn new(context: &'a VulkanContext) -> Self {
        Owned {
            context,
            objects: Vec::new(),
        }
    }

    fn device(&self) -> &'a ash::Device {
        &self.context.device
    }

    fn buffer(&mut self, info: &vk::BufferCreateInfo<'_>) -> Result<vk::Buffer, String> {
        // SAFETY: the device is live, and `info` and its chain are valid for the call.
        let buffer = unsafe { self.device().create_buffer(info, None) }
            .map_err(|error| format!("create a buffer: {}", result_name(error)))?;
        self.objects.push(Object::Buffer(buffer));
        Ok(buffer)
    }

    /// An image, or the result that refused it (the explicit-layout attempts expect refusals).
    fn image(&mut self, info: &vk::ImageCreateInfo<'_>) -> Result<vk::Image, vk::Result> {
        // SAFETY: the device is live, and `info` and its chain are valid for the call.
        let image = unsafe { self.device().create_image(info, None) }?;
        self.objects.push(Object::Image(image));
        Ok(image)
    }

    fn memory(&mut self, info: &vk::MemoryAllocateInfo<'_>) -> Result<vk::DeviceMemory, String> {
        // SAFETY: the device is live, and `info` and its chain are valid for the call.
        let memory = unsafe { self.device().allocate_memory(info, None) }
            .map_err(|error| format!("allocate memory: {}", result_name(error)))?;
        self.objects.push(Object::Memory(memory));
        Ok(memory)
    }

    /// A host-visible, host-coherent buffer of `size` bytes with `usage`, bound to memory of its
    /// own and mapped: `(buffer, mapped bytes)`.
    fn host_buffer(
        &mut self,
        size: u64,
        usage: vk::BufferUsageFlags,
    ) -> Result<(vk::Buffer, *mut u8), String> {
        let buffer = self.buffer(&vk::BufferCreateInfo::default().size(size).usage(usage))?;
        // SAFETY: the buffer was just created on this device.
        let requirements = unsafe { self.device().get_buffer_memory_requirements(buffer) };
        let memory_type = host_coherent_type(self.context, requirements.memory_type_bits)
            .ok_or("no host-visible, host-coherent memory type for a buffer")?;
        let memory = self.memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type),
        )?;
        // SAFETY: the buffer and memory are this device's, unbound, and the type fits.
        unsafe { self.device().bind_buffer_memory(buffer, memory, 0) }
            .map_err(|error| format!("bind a host buffer: {}", result_name(error)))?;
        let mapped = self.map(memory)?;
        Ok((buffer, mapped))
    }

    /// Map all of host-visible `memory`.
    fn map(&mut self, memory: vk::DeviceMemory) -> Result<*mut u8, String> {
        // SAFETY: the memory is this device's, host-visible, and not mapped yet; freeing it
        // unmaps it.
        unsafe {
            self.device()
                .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        }
        .map(|pointer| pointer.cast::<u8>())
        .map_err(|error| format!("map memory: {}", result_name(error)))
    }

    /// A compute pipeline for `spv` over `layout`.
    fn compute_pipeline(
        &mut self,
        spv: &[u8],
        label: &str,
        layout: vk::PipelineLayout,
    ) -> Result<vk::Pipeline, String> {
        let module = create_shader_module(self.context, spv, label)?;
        self.objects.push(Object::ShaderModule(module));
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"main");
        let info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout);
        // SAFETY: the module and layout are live objects of this device.
        let pipelines = unsafe {
            self.device()
                .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
        }
        .map_err(|(_, error)| format!("create the {label} pipeline: {}", result_name(error)))?;
        let pipeline = pipelines[0];
        self.objects.push(Object::Pipeline(pipeline));
        Ok(pipeline)
    }

    fn descriptor_set_layout(
        &mut self,
        bindings: &[vk::DescriptorSetLayoutBinding<'_>],
    ) -> Result<vk::DescriptorSetLayout, String> {
        let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(bindings);
        // SAFETY: the device is live, and `info` is valid for the call.
        let layout = unsafe { self.device().create_descriptor_set_layout(&info, None) }
            .map_err(|error| format!("create a descriptor set layout: {}", result_name(error)))?;
        self.objects.push(Object::DescriptorSetLayout(layout));
        Ok(layout)
    }

    fn pipeline_layout(
        &mut self,
        set_layout: vk::DescriptorSetLayout,
        push_bytes: u32,
    ) -> Result<vk::PipelineLayout, String> {
        let set_layouts = [set_layout];
        let ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(push_bytes)];
        let info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&ranges);
        // SAFETY: the set layout is a live object of this device.
        let layout = unsafe { self.device().create_pipeline_layout(&info, None) }
            .map_err(|error| format!("create a pipeline layout: {}", result_name(error)))?;
        self.objects.push(Object::PipelineLayout(layout));
        Ok(layout)
    }

    /// One descriptor set of `layout`, from a pool of its own that holds `storage_buffers`.
    fn descriptor_set(
        &mut self,
        layout: vk::DescriptorSetLayout,
        storage_buffers: u32,
    ) -> Result<vk::DescriptorSet, String> {
        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(storage_buffers)];
        let info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&sizes);
        // SAFETY: the device is live, and `info` is valid for the call.
        let pool = unsafe { self.device().create_descriptor_pool(&info, None) }
            .map_err(|error| format!("create a descriptor pool: {}", result_name(error)))?;
        self.objects.push(Object::DescriptorPool(pool));
        let layouts = [layout];
        let allocate = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);
        // SAFETY: the pool and layout are live objects of this device; the pool has room.
        let sets = unsafe { self.device().allocate_descriptor_sets(&allocate) }
            .map_err(|error| format!("allocate a descriptor set: {}", result_name(error)))?;
        Ok(sets[0])
    }

    /// Record one primary command buffer with `record`, submit it to the context's queue, and
    /// wait for it under the hang protocol. Returns only once nothing it submitted is in flight.
    fn submit(
        &mut self,
        operation: &str,
        record: impl FnOnce(&ash::Device, vk::CommandBuffer),
    ) -> Result<(), String> {
        let device = self.device();
        let pool_info =
            vk::CommandPoolCreateInfo::default().queue_family_index(self.context.queue_family);
        // SAFETY: the device is live, and `pool_info` is valid for the call.
        let pool = unsafe { device.create_command_pool(&pool_info, None) }
            .map_err(|error| format!("create a command pool: {}", result_name(error)))?;
        self.objects.push(Object::CommandPool(pool));
        let allocate = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command = self
            .context
            .allocate_command_buffers(&allocate)
            .map_err(|error| format!("allocate a command buffer: {}", result_name(error)))?[0];
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: the command buffer was just allocated from this thread's pool.
        unsafe { device.begin_command_buffer(command, &begin) }
            .map_err(|error| format!("begin a command buffer: {}", result_name(error)))?;
        record(device, command);
        // SAFETY: as above; recording is complete.
        unsafe { device.end_command_buffer(command) }
            .map_err(|error| format!("end a command buffer: {}", result_name(error)))?;
        // SAFETY: the device is live.
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(|error| format!("create a fence: {}", result_name(error)))?;
        self.objects.push(Object::Fence(fence));
        let commands = [command];
        let submit = vk::SubmitInfo::default().command_buffers(&commands);
        // SAFETY: the queue is the context's, used by this thread only; the command buffer is
        // recorded and its objects live until this step drops, after the wait below.
        unsafe { device.queue_submit(self.context.queue, &[submit], fence) }
            .map_err(|error| format!("submit {operation}: {}", result_name(error)))?;
        wait::vulkan::wait_for_fence(device, fence, self.context.backend, operation)
    }
}

/// A buffer barrier from `src` to `dst` access over all of `buffer`.
fn buffer_barrier(
    buffer: vk::Buffer,
    src: vk::AccessFlags,
    dst: vk::AccessFlags,
) -> vk::BufferMemoryBarrier<'static> {
    vk::BufferMemoryBarrier::default()
        .src_access_mask(src)
        .dst_access_mask(dst)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .buffer(buffer)
        .offset(0)
        .size(vk::WHOLE_SIZE)
}

/// The byte the probe writes at (`row`, `column`) of a pattern, distinct per `salt`.
fn pattern_byte(salt: u8, row: u64, column: u64) -> u8 {
    let mixed = row
        .wrapping_mul(0x9e37_79b9)
        .wrapping_add(column.wrapping_mul(0x85eb_ca6b))
        .wrapping_add(u64::from(salt).wrapping_mul(0xc2b2_ae35));
    (mixed ^ (mixed >> 13) ^ (mixed >> 29)) as u8
}

/// The `index`th word the host-memory dispatch reads, and what it writes for it.
fn dispatch_word(index: u32) -> (u32, u32) {
    let input = index.wrapping_mul(0x9e37_79b9) ^ 0x5bd1_e995;
    (input, input.wrapping_mul(3).wrapping_add(1))
}
