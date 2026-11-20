//! `VK_EXT_device_generated_commands`, which `ash 0.38` (Vulkan 1.3.281) predates: the structures
//! and commands the device replay's generated-commands path uses (`compute::device_replay`), laid
//! out as the registry defines them (`vk.xml`, extension number 573: structure types
//! `1000572000 + n`).
//!
//! The executor's context loads the commands only where it enabled the extension
//! ([`Commands::load`]). Every commands layout is made for explicit preprocessing: each execution
//! is preprocessed apart ([`Commands::preprocess`]) and runs preprocessed
//! ([`Commands::execute_preprocessed`]), a form that runs right on every driver.

use ash::vk;
use std::ffi::{c_void, CStr};

/// `VkIndirectExecutionSetEXT` and `VkIndirectCommandsLayoutEXT`: non-dispatchable handles.
pub(crate) type Handle = u64;

const fn structure(n: i32) -> vk::StructureType {
    vk::StructureType::from_raw(1_000_572_000 + n)
}

/// `VK_INDIRECT_EXECUTION_SET_INFO_TYPE_PIPELINES_EXT`.
pub(crate) const EXECUTION_SET_PIPELINES: i32 = 0;
/// `VK_INDIRECT_COMMANDS_TOKEN_TYPE_EXECUTION_SET_EXT`, `..._PUSH_CONSTANT_EXT`,
/// `..._DISPATCH_EXT`.
pub(crate) const TOKEN_EXECUTION_SET: i32 = 0;
pub(crate) const TOKEN_PUSH_CONSTANT: i32 = 1;
pub(crate) const TOKEN_DISPATCH: i32 = 9;
/// `VK_PIPELINE_CREATE_2_INDIRECT_BINDABLE_BIT_EXT`.
pub(crate) const INDIRECT_BINDABLE: vk::PipelineCreateFlags2KHR =
    vk::PipelineCreateFlags2KHR::from_raw(0x40_0000_0000);
/// `VK_BUFFER_USAGE_2_PREPROCESS_BUFFER_BIT_EXT`.
pub(crate) const PREPROCESS_BUFFER: vk::BufferUsageFlags2KHR =
    vk::BufferUsageFlags2KHR::from_raw(0x8000_0000);
/// `VK_INDIRECT_COMMANDS_LAYOUT_USAGE_EXPLICIT_PREPROCESS_BIT_EXT`: an execution of the layout
/// is preprocessed apart (`vkCmdPreprocessGeneratedCommandsEXT`) before it runs.
const LAYOUT_EXPLICIT_PREPROCESS: u32 = 0x1;
/// `VK_PIPELINE_STAGE_COMMAND_PREPROCESS_BIT_EXT`, `VK_ACCESS_COMMAND_PREPROCESS_READ_BIT_EXT` and
/// `VK_ACCESS_COMMAND_PREPROCESS_WRITE_BIT_EXT` (the values of the `NV` names `ash` has).
pub(crate) const PREPROCESS_STAGE: vk::PipelineStageFlags =
    vk::PipelineStageFlags::COMMAND_PREPROCESS_NV;
pub(crate) const PREPROCESS_READ: vk::AccessFlags = vk::AccessFlags::COMMAND_PREPROCESS_READ_NV;
pub(crate) const PREPROCESS_WRITE: vk::AccessFlags = vk::AccessFlags::COMMAND_PREPROCESS_WRITE_NV;

/// `VkIndirectExecutionSetPipelineInfoEXT`.
#[repr(C)]
pub(crate) struct ExecutionSetPipelineInfo {
    s_type: vk::StructureType,
    p_next: *const c_void,
    initial_pipeline: vk::Pipeline,
    max_pipeline_count: u32,
}

impl ExecutionSetPipelineInfo {
    pub(crate) fn new(initial_pipeline: vk::Pipeline, max_pipeline_count: u32) -> Self {
        ExecutionSetPipelineInfo {
            s_type: structure(10),
            p_next: std::ptr::null(),
            initial_pipeline,
            max_pipeline_count,
        }
    }
}

/// `VkIndirectExecutionSetCreateInfoEXT` of the pipelines type (its `info` union holds
/// `pPipelineInfo`).
#[repr(C)]
pub(crate) struct ExecutionSetCreateInfo {
    s_type: vk::StructureType,
    p_next: *const c_void,
    kind: i32,
    pipeline_info: *const ExecutionSetPipelineInfo,
}

impl ExecutionSetCreateInfo {
    pub(crate) fn new(pipeline_info: &ExecutionSetPipelineInfo) -> Self {
        ExecutionSetCreateInfo {
            s_type: structure(3),
            p_next: std::ptr::null(),
            kind: EXECUTION_SET_PIPELINES,
            pipeline_info,
        }
    }
}

/// `VkWriteIndirectExecutionSetPipelineEXT`.
#[repr(C)]
pub(crate) struct WriteExecutionSetPipeline {
    s_type: vk::StructureType,
    p_next: *const c_void,
    index: u32,
    pipeline: vk::Pipeline,
}

impl WriteExecutionSetPipeline {
    pub(crate) fn new(index: u32, pipeline: vk::Pipeline) -> Self {
        WriteExecutionSetPipeline {
            s_type: structure(8),
            p_next: std::ptr::null(),
            index,
            pipeline,
        }
    }
}

/// `VkIndirectCommandsPushConstantTokenEXT`.
#[repr(C)]
pub(crate) struct PushConstantToken {
    pub(crate) update_range: vk::PushConstantRange,
}

/// `VkIndirectCommandsExecutionSetTokenEXT`.
#[repr(C)]
pub(crate) struct ExecutionSetToken {
    pub(crate) kind: i32,
    pub(crate) shader_stages: vk::ShaderStageFlags,
}

/// `VkIndirectCommandsLayoutTokenEXT` (its `data` union is one pointer).
#[repr(C)]
pub(crate) struct LayoutToken {
    s_type: vk::StructureType,
    p_next: *const c_void,
    kind: i32,
    data: *const c_void,
    offset: u32,
}

impl LayoutToken {
    pub(crate) fn new(kind: i32, data: *const c_void, offset: u32) -> Self {
        LayoutToken {
            s_type: structure(7),
            p_next: std::ptr::null(),
            kind,
            data,
            offset,
        }
    }
}

/// `VkIndirectCommandsLayoutCreateInfoEXT`, made for explicit preprocessing
/// (`VK_INDIRECT_COMMANDS_LAYOUT_USAGE_EXPLICIT_PREPROCESS_BIT_EXT`): each of its executions is
/// preprocessed apart ([`Commands::preprocess`]) and runs preprocessed
/// ([`Commands::execute_preprocessed`]).
#[repr(C)]
pub(crate) struct LayoutCreateInfo {
    s_type: vk::StructureType,
    p_next: *const c_void,
    flags: u32,
    shader_stages: vk::ShaderStageFlags,
    indirect_stride: u32,
    pipeline_layout: vk::PipelineLayout,
    token_count: u32,
    tokens: *const LayoutToken,
}

impl LayoutCreateInfo {
    pub(crate) fn new(
        indirect_stride: u32,
        pipeline_layout: vk::PipelineLayout,
        tokens: &[LayoutToken],
    ) -> Self {
        LayoutCreateInfo {
            s_type: structure(6),
            p_next: std::ptr::null(),
            flags: LAYOUT_EXPLICIT_PREPROCESS,
            shader_stages: vk::ShaderStageFlags::COMPUTE,
            indirect_stride,
            pipeline_layout,
            token_count: tokens.len() as u32,
            tokens: tokens.as_ptr(),
        }
    }
}

/// `VkGeneratedCommandsMemoryRequirementsInfoEXT`.
#[repr(C)]
pub(crate) struct MemoryRequirementsInfo {
    s_type: vk::StructureType,
    p_next: *const c_void,
    execution_set: Handle,
    commands_layout: Handle,
    max_sequence_count: u32,
    max_draw_count: u32,
}

impl MemoryRequirementsInfo {
    pub(crate) fn new(execution_set: Handle, commands_layout: Handle, sequences: u32) -> Self {
        MemoryRequirementsInfo {
            s_type: structure(2),
            p_next: std::ptr::null(),
            execution_set,
            commands_layout,
            max_sequence_count: sequences,
            max_draw_count: 0,
        }
    }
}

/// `VkGeneratedCommandsInfoEXT`.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct GeneratedCommandsInfo {
    s_type: vk::StructureType,
    p_next: *const c_void,
    shader_stages: vk::ShaderStageFlags,
    execution_set: Handle,
    commands_layout: Handle,
    indirect_address: vk::DeviceAddress,
    indirect_address_size: vk::DeviceSize,
    preprocess_address: vk::DeviceAddress,
    preprocess_size: vk::DeviceSize,
    max_sequence_count: u32,
    sequence_count_address: vk::DeviceAddress,
    max_draw_count: u32,
}

/// One execution's addresses and sizes.
pub(crate) struct Execution {
    pub(crate) indirect: (vk::DeviceAddress, vk::DeviceSize),
    pub(crate) preprocess: (vk::DeviceAddress, vk::DeviceSize),
    pub(crate) max_sequences: u32,
    pub(crate) count_address: vk::DeviceAddress,
}

impl GeneratedCommandsInfo {
    pub(crate) fn new(execution_set: Handle, commands_layout: Handle, e: &Execution) -> Self {
        GeneratedCommandsInfo {
            s_type: structure(4),
            p_next: std::ptr::null(),
            shader_stages: vk::ShaderStageFlags::COMPUTE,
            execution_set,
            commands_layout,
            indirect_address: e.indirect.0,
            indirect_address_size: e.indirect.1,
            preprocess_address: e.preprocess.0,
            preprocess_size: e.preprocess.1,
            max_sequence_count: e.max_sequences,
            sequence_count_address: e.count_address,
            max_draw_count: 0,
        }
    }
}

type CreateExecutionSet = unsafe extern "system" fn(
    vk::Device,
    *const ExecutionSetCreateInfo,
    *const vk::AllocationCallbacks<'static>,
    *mut Handle,
) -> vk::Result;
type DestroyHandle =
    unsafe extern "system" fn(vk::Device, Handle, *const vk::AllocationCallbacks<'static>);
type UpdateExecutionSetPipeline =
    unsafe extern "system" fn(vk::Device, Handle, u32, *const WriteExecutionSetPipeline);
type CreateCommandsLayout = unsafe extern "system" fn(
    vk::Device,
    *const LayoutCreateInfo,
    *const vk::AllocationCallbacks<'static>,
    *mut Handle,
) -> vk::Result;
type MemoryRequirements = unsafe extern "system" fn(
    vk::Device,
    *const MemoryRequirementsInfo,
    *mut vk::MemoryRequirements2<'static>,
);
type Execute =
    unsafe extern "system" fn(vk::CommandBuffer, vk::Bool32, *const GeneratedCommandsInfo);
type Preprocess =
    unsafe extern "system" fn(vk::CommandBuffer, *const GeneratedCommandsInfo, vk::CommandBuffer);

/// The extension's commands, loaded for one device.
#[derive(Clone, Copy)]
pub(crate) struct Commands {
    device: vk::Device,
    create_execution_set: CreateExecutionSet,
    destroy_execution_set: DestroyHandle,
    update_execution_set_pipeline: UpdateExecutionSetPipeline,
    create_commands_layout: CreateCommandsLayout,
    destroy_commands_layout: DestroyHandle,
    memory_requirements: MemoryRequirements,
    execute: Execute,
    preprocess: Preprocess,
}

impl Commands {
    /// The commands of `device`, each from `load` (`vkGetDeviceProcAddr`); `None` when one is
    /// missing.
    ///
    /// # Safety
    /// `load` returns each name's command of `device`, which enabled the extension.
    pub(crate) unsafe fn load(
        device: vk::Device,
        load: impl Fn(&CStr) -> vk::PFN_vkVoidFunction,
    ) -> Option<Commands> {
        macro_rules! command {
            ($name:literal, $ty:ty) => {{
                let f = load($name)?;
                // SAFETY: the registry gives the command this signature.
                unsafe { std::mem::transmute::<unsafe extern "system" fn(), $ty>(f) }
            }};
        }
        Some(Commands {
            device,
            create_execution_set: command!(c"vkCreateIndirectExecutionSetEXT", CreateExecutionSet),
            destroy_execution_set: command!(c"vkDestroyIndirectExecutionSetEXT", DestroyHandle),
            update_execution_set_pipeline: command!(
                c"vkUpdateIndirectExecutionSetPipelineEXT",
                UpdateExecutionSetPipeline
            ),
            create_commands_layout: command!(
                c"vkCreateIndirectCommandsLayoutEXT",
                CreateCommandsLayout
            ),
            destroy_commands_layout: command!(c"vkDestroyIndirectCommandsLayoutEXT", DestroyHandle),
            memory_requirements: command!(
                c"vkGetGeneratedCommandsMemoryRequirementsEXT",
                MemoryRequirements
            ),
            execute: command!(c"vkCmdExecuteGeneratedCommandsEXT", Execute),
            preprocess: command!(c"vkCmdPreprocessGeneratedCommandsEXT", Preprocess),
        })
    }

    /// `vkCreateIndirectExecutionSetEXT`.
    ///
    /// # Safety
    /// `info` and what it points at are valid for the call.
    pub(crate) unsafe fn create_execution_set(
        &self,
        info: &ExecutionSetCreateInfo,
    ) -> Result<Handle, vk::Result> {
        let mut out = 0;
        // SAFETY: the caller's contract.
        match unsafe { (self.create_execution_set)(self.device, info, std::ptr::null(), &mut out) }
        {
            vk::Result::SUCCESS => Ok(out),
            error => Err(error),
        }
    }

    /// `vkDestroyIndirectExecutionSetEXT`.
    ///
    /// # Safety
    /// No pending work uses `set`.
    pub(crate) unsafe fn destroy_execution_set(&self, set: Handle) {
        // SAFETY: the caller's contract.
        unsafe { (self.destroy_execution_set)(self.device, set, std::ptr::null()) }
    }

    /// `vkUpdateIndirectExecutionSetPipelineEXT`.
    ///
    /// # Safety
    /// No submitted work references the written indices.
    pub(crate) unsafe fn update_execution_set(
        &self,
        set: Handle,
        writes: &[WriteExecutionSetPipeline],
    ) {
        // SAFETY: the caller's contract.
        unsafe {
            (self.update_execution_set_pipeline)(
                self.device,
                set,
                writes.len() as u32,
                writes.as_ptr(),
            )
        }
    }

    /// `vkCreateIndirectCommandsLayoutEXT`.
    ///
    /// # Safety
    /// `info` and what it points at are valid for the call.
    pub(crate) unsafe fn create_commands_layout(
        &self,
        info: &LayoutCreateInfo,
    ) -> Result<Handle, vk::Result> {
        let mut out = 0;
        // SAFETY: the caller's contract.
        match unsafe {
            (self.create_commands_layout)(self.device, info, std::ptr::null(), &mut out)
        } {
            vk::Result::SUCCESS => Ok(out),
            error => Err(error),
        }
    }

    /// `vkDestroyIndirectCommandsLayoutEXT`.
    ///
    /// # Safety
    /// No pending work uses `layout`.
    pub(crate) unsafe fn destroy_commands_layout(&self, layout: Handle) {
        // SAFETY: the caller's contract.
        unsafe { (self.destroy_commands_layout)(self.device, layout, std::ptr::null()) }
    }

    /// `vkGetGeneratedCommandsMemoryRequirementsEXT`.
    ///
    /// # Safety
    /// `info`'s handles are live.
    pub(crate) unsafe fn memory_requirements(
        &self,
        info: &MemoryRequirementsInfo,
    ) -> vk::MemoryRequirements {
        let mut out = vk::MemoryRequirements2::default();
        // SAFETY: the caller's contract.
        unsafe { (self.memory_requirements)(self.device, info, &mut out) };
        out.memory_requirements
    }

    /// `vkCmdPreprocessGeneratedCommandsEXT` of `info` into `command`, the state taken from
    /// `command` itself.
    ///
    /// # Safety
    /// `command` is recording, with the execution set's initial pipeline bound; `info`'s handles
    /// and addresses are live until the work completes.
    pub(crate) unsafe fn preprocess(
        &self,
        command: vk::CommandBuffer,
        info: &GeneratedCommandsInfo,
    ) {
        // SAFETY: the caller's contract.
        unsafe { (self.preprocess)(command, info, command) }
    }

    /// `vkCmdExecuteGeneratedCommandsEXT` of commands [`preprocess`](Self::preprocess) wrote.
    ///
    /// # Safety
    /// `command` is recording, with the execution set's initial pipeline bound; `info` was
    /// preprocessed in `command`, with the same contents and the same state bound, a barrier
    /// ordering that preprocessing's writes before this execution, and its handles and addresses
    /// are live until the work completes.
    pub(crate) unsafe fn execute_preprocessed(
        &self,
        command: vk::CommandBuffer,
        info: &GeneratedCommandsInfo,
    ) {
        // SAFETY: the caller's contract.
        unsafe { (self.execute)(command, vk::TRUE, info) }
    }
}
