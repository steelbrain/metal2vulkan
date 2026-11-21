//! `VK_EXT_device_generated_commands`, which ash 0.38 does not bind: the structures, constants and
//! entry points of it the runtime's indirect command buffers use, declared from the
//! specification, and the device's entry points, loaded where device creation enabled it. Each
//! structure's size and its members' offsets are checked against the specification's layout as
//! the crate compiles. The file goes once ash binds the extension.

use ash::vk;
use std::ffi::{c_void, CStr};

/// The extension's name.
pub(crate) const NAME: &CStr = c"VK_EXT_device_generated_commands";

const fn structure(offset: i32) -> vk::StructureType {
    vk::StructureType::from_raw(1_000_572_000 + offset)
}

/// `VK_BUFFER_USAGE_2_PREPROCESS_BUFFER_BIT_EXT`.
pub(crate) const PREPROCESS_USAGE: vk::BufferUsageFlags2KHR =
    vk::BufferUsageFlags2KHR::from_raw(1 << 31);
/// `VK_INDIRECT_COMMANDS_TOKEN_TYPE_*_EXT`.
pub(crate) const TOKEN_PUSH_CONSTANT: i32 = 1;
pub(crate) const TOKEN_SEQUENCE_INDEX: i32 = 2;
pub(crate) const TOKEN_INDEX_BUFFER: i32 = 3;
pub(crate) const TOKEN_DRAW_INDEXED: i32 = 5;
/// `VK_INDIRECT_COMMANDS_INPUT_MODE_VULKAN_INDEX_BUFFER_EXT`.
pub(crate) const INPUT_MODE_VULKAN: u32 = 1;
/// The stages an execution's commands run: a render pipeline's vertex and fragment stages.
pub(crate) const STAGES: vk::ShaderStageFlags = vk::ShaderStageFlags::from_raw(
    vk::ShaderStageFlags::VERTEX.as_raw() | vk::ShaderStageFlags::FRAGMENT.as_raw(),
);

/// No Metal counterpart: `VkPhysicalDeviceDeviceGeneratedCommandsFeaturesEXT`, which ash 0.38
/// does not declare, as the floor reads and enables it ([`Features`](crate::floor::Features)).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct DgcFeatures {
    s_type: vk::StructureType,
    pub(crate) p_next: *mut c_void,
    /// `deviceGeneratedCommands`.
    pub device_generated_commands: vk::Bool32,
    /// `dynamicGeneratedPipelineLayout`, never enabled.
    pub dynamic_generated_pipeline_layout: vk::Bool32,
}

/// `VkPhysicalDeviceDeviceGeneratedCommandsPropertiesEXT`.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct DgcProperties {
    s_type: vk::StructureType,
    pub(crate) p_next: *mut c_void,
    pub(crate) max_indirect_pipeline_count: u32,
    pub(crate) max_indirect_shader_object_count: u32,
    pub(crate) max_indirect_sequence_count: u32,
    pub(crate) max_indirect_commands_token_count: u32,
    pub(crate) max_indirect_commands_token_offset: u32,
    pub(crate) max_indirect_commands_indirect_stride: u32,
    pub(crate) supported_input_modes: u32,
    pub(crate) supported_shader_stages: vk::ShaderStageFlags,
    pub(crate) stages_pipeline_binding: vk::ShaderStageFlags,
    pub(crate) stages_shader_binding: vk::ShaderStageFlags,
    pub(crate) transform_feedback: vk::Bool32,
    pub(crate) multi_draw_indirect_count: vk::Bool32,
}

// SAFETY: plain data; the chain pointer is cleared once the structure is read, as ash's
// structures' are.
unsafe impl Send for DgcFeatures {}
// SAFETY: as above.
unsafe impl Sync for DgcFeatures {}
// SAFETY: each starts with `sType` and `pNext`, as the chain requires.
unsafe impl vk::ExtendsPhysicalDeviceFeatures2 for DgcFeatures {}
// SAFETY: as above.
unsafe impl vk::ExtendsDeviceCreateInfo for DgcFeatures {}
// SAFETY: as above.
unsafe impl vk::ExtendsPhysicalDeviceProperties2 for DgcProperties {}

impl Default for DgcFeatures {
    fn default() -> DgcFeatures {
        DgcFeatures {
            s_type: structure(0),
            p_next: std::ptr::null_mut(),
            device_generated_commands: vk::FALSE,
            dynamic_generated_pipeline_layout: vk::FALSE,
        }
    }
}

impl Default for DgcProperties {
    fn default() -> DgcProperties {
        DgcProperties {
            s_type: structure(1),
            p_next: std::ptr::null_mut(),
            max_indirect_pipeline_count: 0,
            max_indirect_shader_object_count: 0,
            max_indirect_sequence_count: 0,
            max_indirect_commands_token_count: 0,
            max_indirect_commands_token_offset: 0,
            max_indirect_commands_indirect_stride: 0,
            supported_input_modes: 0,
            supported_shader_stages: vk::ShaderStageFlags::empty(),
            stages_pipeline_binding: vk::ShaderStageFlags::empty(),
            stages_shader_binding: vk::ShaderStageFlags::empty(),
            transform_feedback: vk::FALSE,
            multi_draw_indirect_count: vk::FALSE,
        }
    }
}

/// `VkIndirectCommandsPushConstantTokenEXT`.
#[repr(C)]
pub(crate) struct PushConstantToken {
    pub(crate) update_range: vk::PushConstantRange,
}

/// `VkIndirectCommandsIndexBufferTokenEXT`.
#[repr(C)]
pub(crate) struct IndexBufferToken {
    pub(crate) mode: u32,
}

/// `VkIndirectCommandsLayoutTokenEXT`; `data` points at the token's own structure, or is null.
#[repr(C)]
pub(crate) struct LayoutToken {
    s_type: vk::StructureType,
    p_next: *const c_void,
    pub(crate) token_type: i32,
    pub(crate) data: *const c_void,
    pub(crate) offset: u32,
}

impl LayoutToken {
    pub(crate) fn new(token_type: i32, data: *const c_void, offset: u32) -> LayoutToken {
        LayoutToken {
            s_type: structure(7),
            p_next: std::ptr::null(),
            token_type,
            data,
            offset,
        }
    }
}

/// `VkIndirectCommandsLayoutCreateInfoEXT`.
#[repr(C)]
pub(crate) struct LayoutCreateInfo {
    s_type: vk::StructureType,
    p_next: *const c_void,
    flags: u32,
    shader_stages: vk::ShaderStageFlags,
    indirect_stride: u32,
    pipeline_layout: vk::PipelineLayout,
    token_count: u32,
    p_tokens: *const LayoutToken,
}

impl LayoutCreateInfo {
    /// Ordered sequences (no flag) of `tokens` at `stride`, for vertex and fragment stages,
    /// through `layout`.
    pub(crate) fn new(
        stride: u32,
        layout: vk::PipelineLayout,
        tokens: &[LayoutToken],
    ) -> LayoutCreateInfo {
        LayoutCreateInfo {
            s_type: structure(6),
            p_next: std::ptr::null(),
            flags: 0,
            shader_stages: STAGES,
            indirect_stride: stride,
            pipeline_layout: layout,
            token_count: tokens.len() as u32,
            p_tokens: tokens.as_ptr(),
        }
    }
}

/// `VkGeneratedCommandsPipelineInfoEXT`.
#[repr(C)]
pub(crate) struct PipelineInfo {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    pipeline: vk::Pipeline,
}

impl PipelineInfo {
    pub(crate) fn new(pipeline: vk::Pipeline) -> PipelineInfo {
        PipelineInfo {
            s_type: structure(13),
            p_next: std::ptr::null_mut(),
            pipeline,
        }
    }
}

/// `VkGeneratedCommandsMemoryRequirementsInfoEXT`, with no execution set.
#[repr(C)]
pub(crate) struct MemoryInfo {
    s_type: vk::StructureType,
    p_next: *const c_void,
    execution_set: u64,
    layout: u64,
    max_sequence_count: u32,
    max_draw_count: u32,
}

impl MemoryInfo {
    pub(crate) fn new(pipeline: &PipelineInfo, layout: u64, sequences: u32) -> MemoryInfo {
        MemoryInfo {
            s_type: structure(2),
            p_next: <*const PipelineInfo>::cast(pipeline),
            execution_set: 0,
            layout,
            max_sequence_count: sequences,
            max_draw_count: 0,
        }
    }
}

/// `VkGeneratedCommandsInfoEXT`, with no execution set.
#[repr(C)]
pub(crate) struct CommandsInfo {
    s_type: vk::StructureType,
    p_next: *const c_void,
    shader_stages: vk::ShaderStageFlags,
    execution_set: u64,
    layout: u64,
    indirect_address: u64,
    indirect_address_size: u64,
    preprocess_address: u64,
    preprocess_size: u64,
    max_sequence_count: u32,
    sequence_count_address: u64,
    max_draw_count: u32,
}

impl CommandsInfo {
    /// At most `sequences` sequences of `layout` at `indirect` (`size` bytes), preprocessed into
    /// `preprocess` (`preprocess_size` bytes), with the inherited `pipeline`: all of them where
    /// `count` is 0, else as many as the `u32` at device address `count` says, up to `sequences`.
    pub(crate) fn new(
        pipeline: &PipelineInfo,
        layout: u64,
        (indirect, size): (u64, u64),
        (preprocess, preprocess_size): (u64, u64),
        (sequences, count): (u32, u64),
    ) -> CommandsInfo {
        CommandsInfo {
            s_type: structure(4),
            p_next: <*const PipelineInfo>::cast(pipeline),
            shader_stages: STAGES,
            execution_set: 0,
            layout,
            indirect_address: indirect,
            indirect_address_size: size,
            preprocess_address: preprocess,
            preprocess_size,
            max_sequence_count: sequences,
            sequence_count_address: count,
            max_draw_count: 0,
        }
    }
}

pub(crate) type CreateLayout = unsafe extern "system" fn(
    vk::Device,
    *const LayoutCreateInfo,
    *const c_void,
    *mut u64,
) -> vk::Result;
pub(crate) type DestroyLayout = unsafe extern "system" fn(vk::Device, u64, *const c_void);
pub(crate) type MemoryRequirements =
    unsafe extern "system" fn(vk::Device, *const MemoryInfo, *mut vk::MemoryRequirements2<'static>);
pub(crate) type Execute =
    unsafe extern "system" fn(vk::CommandBuffer, vk::Bool32, *const CommandsInfo);

/// The extension's device entry points.
pub(crate) struct Dgc {
    pub(crate) create_layout: CreateLayout,
    pub(crate) destroy_layout: DestroyLayout,
    pub(crate) memory_requirements: MemoryRequirements,
    pub(crate) execute: Execute,
}

impl Dgc {
    /// The entry points of `device`; `None` where one is missing.
    ///
    /// # Safety
    ///
    /// `device` was made from `instance` with the extension enabled.
    pub(crate) unsafe fn load(instance: &ash::Instance, device: vk::Device) -> Option<Dgc> {
        let get = |name: &CStr| {
            // SAFETY: the caller's contract; the name is the extension's own entry point.
            unsafe { (instance.fp_v1_0().get_device_proc_addr)(device, name.as_ptr()) }
        };
        // SAFETY: each pointer is the entry point of its name, whose signature the type declares
        // from the specification.
        unsafe {
            Some(Dgc {
                create_layout: std::mem::transmute::<vk::PFN_vkVoidFunction, Option<CreateLayout>>(
                    get(c"vkCreateIndirectCommandsLayoutEXT"),
                )?,
                destroy_layout: std::mem::transmute::<vk::PFN_vkVoidFunction, Option<DestroyLayout>>(
                    get(c"vkDestroyIndirectCommandsLayoutEXT"),
                )?,
                memory_requirements: std::mem::transmute::<
                    vk::PFN_vkVoidFunction,
                    Option<MemoryRequirements>,
                >(get(
                    c"vkGetGeneratedCommandsMemoryRequirementsEXT",
                ))?,
                execute: std::mem::transmute::<vk::PFN_vkVoidFunction, Option<Execute>>(get(
                    c"vkCmdExecuteGeneratedCommandsEXT",
                ))?,
            })
        }
    }
}

/// Each structure's size and members' offsets, as the specification's C declarations lay them out
/// on a 64-bit target: a wrong declaration fails the build, not a driver call.
#[cfg(target_pointer_width = "64")]
const _: () = {
    use std::mem::{offset_of, size_of};
    assert!(size_of::<DgcFeatures>() == 24);
    assert!(offset_of!(DgcFeatures, p_next) == 8);
    assert!(offset_of!(DgcFeatures, device_generated_commands) == 16);
    assert!(offset_of!(DgcFeatures, dynamic_generated_pipeline_layout) == 20);
    assert!(size_of::<DgcProperties>() == 64);
    assert!(offset_of!(DgcProperties, max_indirect_pipeline_count) == 16);
    assert!(offset_of!(DgcProperties, max_indirect_sequence_count) == 24);
    assert!(offset_of!(DgcProperties, max_indirect_commands_token_count) == 28);
    assert!(offset_of!(DgcProperties, max_indirect_commands_token_offset) == 32);
    assert!(offset_of!(DgcProperties, max_indirect_commands_indirect_stride) == 36);
    assert!(offset_of!(DgcProperties, supported_input_modes) == 40);
    assert!(offset_of!(DgcProperties, supported_shader_stages) == 44);
    assert!(offset_of!(DgcProperties, multi_draw_indirect_count) == 60);
    assert!(size_of::<PushConstantToken>() == 12);
    assert!(size_of::<IndexBufferToken>() == 4);
    assert!(size_of::<LayoutToken>() == 40);
    assert!(offset_of!(LayoutToken, token_type) == 16);
    assert!(offset_of!(LayoutToken, data) == 24);
    assert!(offset_of!(LayoutToken, offset) == 32);
    assert!(size_of::<LayoutCreateInfo>() == 56);
    assert!(offset_of!(LayoutCreateInfo, flags) == 16);
    assert!(offset_of!(LayoutCreateInfo, shader_stages) == 20);
    assert!(offset_of!(LayoutCreateInfo, indirect_stride) == 24);
    assert!(offset_of!(LayoutCreateInfo, pipeline_layout) == 32);
    assert!(offset_of!(LayoutCreateInfo, token_count) == 40);
    assert!(offset_of!(LayoutCreateInfo, p_tokens) == 48);
    assert!(size_of::<PipelineInfo>() == 24);
    assert!(offset_of!(PipelineInfo, pipeline) == 16);
    assert!(size_of::<MemoryInfo>() == 40);
    assert!(offset_of!(MemoryInfo, execution_set) == 16);
    assert!(offset_of!(MemoryInfo, layout) == 24);
    assert!(offset_of!(MemoryInfo, max_sequence_count) == 32);
    assert!(offset_of!(MemoryInfo, max_draw_count) == 36);
    assert!(size_of::<CommandsInfo>() == 96);
    assert!(offset_of!(CommandsInfo, shader_stages) == 16);
    assert!(offset_of!(CommandsInfo, execution_set) == 24);
    assert!(offset_of!(CommandsInfo, layout) == 32);
    assert!(offset_of!(CommandsInfo, indirect_address) == 40);
    assert!(offset_of!(CommandsInfo, indirect_address_size) == 48);
    assert!(offset_of!(CommandsInfo, preprocess_address) == 56);
    assert!(offset_of!(CommandsInfo, preprocess_size) == 64);
    assert!(offset_of!(CommandsInfo, max_sequence_count) == 72);
    assert!(offset_of!(CommandsInfo, sequence_count_address) == 80);
    assert!(offset_of!(CommandsInfo, max_draw_count) == 88);
};
