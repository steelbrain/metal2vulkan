//! Priming a new device for the exit guard: before the exit hook is registered for it, the creating
//! thread makes the first call of each Vulkan entry point the runtime calls in its sections, the
//! exit guard's table ([`Call`]), shaped as the runtime makes it. A driver or layer that makes a
//! static at such a call's first use (the validation layer's table of stages by queue flags, which
//! its synchronization checks read in barriers, submissions and waits) registers the static's
//! destructor then, before the hook, which therefore runs first. So neither a section in progress
//! as the process exits, nor the program's own calls before it, is the first to make such a static.
//!
//! Measured: without it, at 175dd511 on lavapipe x86_64, an exit right after committing 256
//! command buffers ended by `SIGSEGV` in one of 10 rounds. The core's stack: the scheduler's
//! `vkCmdPipelineBarrier` (its first submission's barriers) faulting inside the validation
//! layer, a static it reads made after the hook's last registration and destroyed first.
//!
//! **The set is the table's, checked.** Inside a section the device's functions are the table's
//! alone (`functions.rs`), and each priming checks, in builds with debug assertions, that it made
//! every entry point of the table but [`UNPRIMED`]: an entry point added to the table and not made
//! here fails every lane at its first device. The calls: a batch's (a command pool and a command
//! buffer of it, begun, with the barrier every batch ends with and, where the queue family writes
//! timestamps, the reset and the two timestamps of a command buffer's GPU times
//! ([`TIMESTAMP_CALLS`], made only there; their reset on the host before, where the device enables
//! `hostQueryReset`, [`HOST_QUERY_RESET_CALLS`]), ended, submitted after each queue's timeline at 0
//! and signalling a timeline of its own and a new fence; the wait for the timeline, a read of every
//! timeline, the wait for the fence and its reset, a wait for the queue to be idle (its drain after
//! a failed wait), and the read of the two timestamps and, where the device calibrates them, a
//! calibration reading ([`CALIBRATION_CALLS`]), as a completion turn makes them); a sampler, an
//! occlusion query pool, a buffer in host-visible memory, mapped (and, where the device has device
//! addresses, [`DEVICE_ADDRESS_CALLS`], its address read), and an image with their memory and
//! views, a depth image, a render pass of each form (`vkCreateRenderPass`, `vkCreateRenderPass2`)
//! and a framebuffer, a descriptor set layout, a pipeline
//! layout with a push constant range and a descriptor pool with a set written, and a compute
//! pipeline made with a pipeline cache from a shader module, its stage specialized as the runtime's
//! makings are ([`SPECIALIZED`]), whose data is read; a command buffer, recorded and never
//! submitted, with what a host step records: the pipeline and the set bound (and, where the device
//! has the push command, the set's write pushed first, [`PUSH_DESCRIPTOR_CALLS`]), a push constant,
//! a dispatch and one from the buffer's arguments, a copy within the buffer, a copy from the buffer
//! to the image and back with a barrier between them, a copy between the image's texels and the
//! image's clear (a render host step's pass steps) and the depth image's (a batch's first work's),
//! then a render host step's: a graphics pipeline that rasterizes nothing, its stage specialized
//! too (and named, where the instance reports messages, [`NAMING_CALLS`]), drawn in a secondary
//! command buffer, directly and from the buffer's arguments, with a
//! vertex buffer and the states a draw sets, inside a query of the occlusion pool, which the render
//! pass's subpass executes; where the queues bind sparse memory, an empty `vkQueueBindSparse`
//! ([`SPARSE_CALLS`]); a wait for the device to be idle; and the release of each object made.
//! Nothing of it outlives the call.
//!
//! **Specialized, with an extended instruction, as the runtime's makings are.** A stage with
//! specialization constants takes paths of its own: in a driver's compiler, and in the Khronos
//! validation layer's checks of the stage, which run `spirv-opt`'s passes and the SPIR-V validator
//! over the module specialized at each pipeline's creation. A module the layer validated before, in
//! any process, it skips at its creation (its validation cache, kept on disk), and priming's are
//! the same in every process: so without a specialized stage, priming ran no validator at all, and
//! the process's first making of a specialized pipeline made the validator's statics. And the
//! validator makes one of them only at an operand naming an untyped definition made before it,
//! such as an extended instruction's set: so priming's modules take `GLSL.std.450`'s `UMin` of
//! their constant, as translated modules take that set's instructions. Measured on the M3 Max
//! (macOS 27.0.1, MoltenVK 1.4.1, the layer and `spirv-tools` 1.4.357.0 from Homebrew), in a lane
//! process that exited while a pool's making of a host step's variant was in progress: `SIGSEGV` at
//! `0x50` inside `spvtools::val::InstructionCanHaveTypeOperand`, its function-local
//! `std::unordered_set` already destroyed (`CoreChecks::ValidateShaderStage`, from
//! `vkCreateComputePipelines` on the pool's thread, in its making section). A diagnostic interposer
//! of the C library's exit handlers showed the validator's two such sets (that one and
//! `InstructionRequiresTypeOperand`'s) made after the hook's first registration in each of 10 runs,
//! the second after its last, by a making on the program's thread or the pool's; with priming
//! specialized and taking the extended instruction, both made in priming, before the hook's first
//! registration, in each of 10.
//!
//! Priming makes each entry point once, in the runtime's shapes; it cannot promise that a layer
//! makes no new static on a code path a later call takes with other parameters. A section's
//! first call of each entry point, and every section whose count reaches a power of two,
//! register the hook again for those (`exit_guard`).

use crate::draw_source::SizedIndexBind;
use crate::dynamic_state::{ColorWriteMasks, ExtendedDynamicState};
use crate::error::{vulkan, Error};
use crate::exit_guard::Call;
use crate::generated_commands::{
    Commands, ExecutionSetCreateInfo, ExecutionSetPipelineInfo, ExecutionSetToken,
    LayoutCreateInfo, LayoutToken, EXECUTION_SET_PIPELINES, INDIRECT_BINDABLE, TOKEN_DISPATCH,
    TOKEN_EXECUTION_SET,
};
use crate::gpu_time::Calibration;
use crate::loader_word::LoaderWord;
use crate::memory;
use crate::release::{self, Generated, Released};
use crate::sharing::Sharing;
use crate::tracker::Barrier;
use ash::vk;
use std::time::{Duration, Instant};

use super::functions::Vk;
use super::QueueSlot;

/// The table's entry points priming cannot make before the hook is registered for the device it
/// primes: the release of a device, an instance and a messenger, the commands of a mesh draw
/// and of an execution of commands the GPU generates, which take pipelines of their own kinds, and
/// the layout of a linear image, of which a device need make none of any format. The first
/// release of each in the process, the first call of each command in a section, and the first
/// layout read in a section, registers the hook again as its section ends
/// (`exit_guard::called`).
pub(crate) const UNPRIMED: [Call; 9] = [
    Call::DestroyDevice,
    Call::DestroyDebugUtilsMessenger,
    Call::DestroyInstance,
    Call::CmdDrawMeshTasks,
    Call::CmdDrawMeshTasksIndirect,
    Call::CmdDrawMeshTasksIndirectCount,
    Call::CmdExecuteGeneratedCommands,
    Call::CmdPreprocessGeneratedCommands,
    Call::GetImageSubresourceLayout,
];

/// The table's entry points a queue family that writes no timestamps (`timestampValidBits` 0)
/// never calls, nor primes: a command buffer's GPU times (`gpu_time.rs`).
pub(crate) const TIMESTAMP_CALLS: [Call; 3] = [
    Call::CmdResetQueryPool,
    Call::CmdWriteTimestamp,
    Call::GetQueryPoolResults,
];

/// The table's entry points a device that writes no timestamps, or does not enable
/// `hostQueryReset` (an adopted one), never calls, nor primes: a timestamp slot's reset on the
/// host (`gpu_time.rs`).
pub(crate) const HOST_QUERY_RESET_CALLS: [Call; 1] = [Call::ResetQueryPool];

/// The table's entry points a device without device addresses (`bufferDeviceAddress`) never
/// calls, nor primes: a buffer's address (`buffer.rs`).
pub(crate) const DEVICE_ADDRESS_CALLS: [Call; 1] = [Call::GetBufferDeviceAddress];

/// The table's entry points a device without `VK_EXT_device_generated_commands` never calls,
/// nor primes: the release of the extension's objects (`generated_commands.rs`).
pub(crate) const GENERATED_COMMANDS_CALLS: [Call; 2] = [
    Call::DestroyIndirectExecutionSet,
    Call::DestroyIndirectCommandsLayout,
];

/// The table's entry points a device that binds no sparse memory never calls, nor primes: a
/// batch's sparse binding (`sparse/bind.rs`).
pub(crate) const SPARSE_CALLS: [Call; 1] = [Call::QueueBindSparse];

/// The table's entry points a device without the push command (Vulkan 1.4, or
/// `VK_KHR_push_descriptor`) never calls, nor primes: a host step's set 0 pushed
/// (`pipeline::objects::pushes_set_zero`).
pub(crate) const PUSH_DESCRIPTOR_CALLS: [Call; 1] = [Call::CmdPushDescriptorSet];

/// The table's entry points a device that calibrates no timestamps never calls, nor primes: a
/// completion turn's calibration reading (`gpu_time.rs`).
pub(crate) const CALIBRATION_CALLS: [Call; 1] = [Call::GetCalibratedTimestamps];

/// The table's entry points a device whose instance reports no messages never calls, nor primes:
/// a graphics pipeline's debug name (`pipeline/graphics.rs`).
pub(crate) const NAMING_CALLS: [Call; 1] = [Call::SetDebugUtilsObjectName];

/// The table's entry points a device without extended dynamic state (Vulkan 1.3, or
/// `VK_EXT_extended_dynamic_state`) never calls, nor primes: a render encoder's dynamic states
/// (`dynamic_state.rs`).
pub(crate) const EXTENDED_DYNAMIC_STATE_CALLS: [Call; 9] = [
    Call::CmdSetCullMode,
    Call::CmdSetFrontFace,
    Call::CmdSetPrimitiveTopology,
    Call::CmdBindVertexBuffers2,
    Call::CmdSetDepthTestEnable,
    Call::CmdSetDepthWriteEnable,
    Call::CmdSetDepthCompareOp,
    Call::CmdSetStencilTestEnable,
    Call::CmdSetStencilOp,
];

/// The table's entry points a device without a dynamic primitive restart (Vulkan 1.3, or
/// `VK_EXT_extended_dynamic_state2`) never calls, nor primes (`dynamic_state.rs`).
pub(crate) const PRIMITIVE_RESTART_CALLS: [Call; 1] = [Call::CmdSetPrimitiveRestartEnable];

/// The table's entry points a device without the dynamic color write mask never calls, nor
/// primes: a snapshot draw's masks (`dynamic_state.rs`, `render/snapshot.rs`).
pub(crate) const COLOR_WRITE_MASK_CALLS: [Call; 1] = [Call::CmdSetColorWriteMask];

/// The table's entry points a device without `maintenance5` never calls, nor primes: an index
/// buffer bound with its size (`draw_source.rs`).
pub(crate) const SIZED_INDEX_BIND_CALLS: [Call; 1] = [Call::CmdBindIndexBuffer2];

/// The table's entry points a device that converts no YCbCr format itself never calls, nor
/// primes: the release of a YCbCr sample kernel's conversion (`ycbcr_device.rs`).
pub(crate) const YCBCR_CONVERSION_CALLS: [Call; 1] = [Call::DestroySamplerYcbcrConversion];

/// The debug name of a render host step's pipeline that priming makes, where the instance reports
/// messages ([`NAMING_CALLS`]).
const PRIMING_PIPELINE: &std::ffi::CStr =
    c"metal2vulkan: a priming pipeline that rasterizes nothing";

/// The longest the priming submission is waited for: it holds two barriers.
const LIMIT: Duration = Duration::from_secs(10);

/// How long one wait lasts before it is repeated, as a queue waiter's.
const WAIT_SLICE: Duration = Duration::from_millis(100);

/// The smallest compute shader with a specialization constant ([`SPECIALIZED`]) and an extended
/// instruction: `main`, of one invocation, takes `GLSL.std.450`'s `UMin` of the constant and
/// returns (SPIR-V 1.0). Also the words of a host step's own module in
/// `tests/teardown_threads.rs`.
pub(crate) const SHADER: [u32; 60] = [
    0x0723_0203,
    0x0001_0000,
    0,
    10,
    0,
    // OpCapability Shader
    0x0002_0011,
    1,
    // %8 = OpExtInstImport "GLSL.std.450"
    0x0006_000b,
    8,
    0x4c53_4c47,
    0x6474_732e,
    0x3035_342e,
    0,
    // OpMemoryModel Logical GLSL450
    0x0003_000e,
    0,
    1,
    // OpEntryPoint GLCompute %1 "main"
    0x0005_000f,
    5,
    1,
    0x6e69_616d,
    0,
    // OpExecutionMode %1 LocalSize 1 1 1
    0x0006_0010,
    1,
    17,
    1,
    1,
    1,
    // OpDecorate %5 SpecId 0
    0x0004_0047,
    5,
    1,
    0,
    // %2 = OpTypeVoid; %3 = OpTypeFunction %2
    0x0002_0013,
    2,
    0x0003_0021,
    3,
    2,
    // %6 = OpTypeInt 32 0; %5 = OpSpecConstant %6 0
    0x0004_0015,
    6,
    32,
    0,
    0x0004_0032,
    6,
    5,
    0,
    // %1 = OpFunction %2 None %3; %4 = OpLabel; %9 = OpExtInst %6 %8 UMin %5 %5; OpReturn;
    // OpFunctionEnd
    0x0005_0036,
    2,
    1,
    0,
    3,
    0x0002_00f8,
    4,
    0x0007_000c,
    6,
    9,
    8,
    38,
    5,
    5,
    0x0001_00fd,
    0x0001_0038,
];

/// The smallest vertex shader with a specialization constant ([`SPECIALIZED`]) and an extended
/// instruction: `main` takes `GLSL.std.450`'s `UMin` of the constant and returns (SPIR-V 1.0), for
/// a pipeline that rasterizes nothing.
const VERTEX_SHADER: [u32; 54] = [
    0x0723_0203,
    0x0001_0000,
    0,
    10,
    0,
    // OpCapability Shader
    0x0002_0011,
    1,
    // %8 = OpExtInstImport "GLSL.std.450"
    0x0006_000b,
    8,
    0x4c53_4c47,
    0x6474_732e,
    0x3035_342e,
    0,
    // OpMemoryModel Logical GLSL450
    0x0003_000e,
    0,
    1,
    // OpEntryPoint Vertex %1 "main"
    0x0005_000f,
    0,
    1,
    0x6e69_616d,
    0,
    // OpDecorate %5 SpecId 0
    0x0004_0047,
    5,
    1,
    0,
    // %2 = OpTypeVoid; %3 = OpTypeFunction %2
    0x0002_0013,
    2,
    0x0003_0021,
    3,
    2,
    // %6 = OpTypeInt 32 0; %5 = OpSpecConstant %6 0
    0x0004_0015,
    6,
    32,
    0,
    0x0004_0032,
    6,
    5,
    0,
    // %1 = OpFunction %2 None %3; %4 = OpLabel; %9 = OpExtInst %6 %8 UMin %5 %5; OpReturn;
    // OpFunctionEnd
    0x0005_0036,
    2,
    1,
    0,
    3,
    0x0002_00f8,
    4,
    0x0007_000c,
    6,
    9,
    8,
    38,
    5,
    5,
    0x0001_00fd,
    0x0001_0038,
];

/// The constant [`SHADER`] and the vertex shader declare, specialized as the runtime's makings
/// specialize their modules' (`pipeline/objects.rs`, `pipeline/graphics.rs`): a specialized stage
/// takes paths of its own, in a driver's compiler and in a layer's checks of the stage, which a
/// making then reaches first (the module docs).
const SPECIALIZED: [vk::SpecializationMapEntry; 1] = [vk::SpecializationMapEntry {
    constant_id: 0,
    offset: 0,
    size: 4,
}];

/// The value of [`SPECIALIZED`]'s constant.
const SPECIALIZED_VALUE: [u8; 4] = 1u32.to_le_bytes();

/// The specialization of each priming stage: [`SPECIALIZED`] at [`SPECIALIZED_VALUE`].
fn specialization() -> vk::SpecializationInfo<'static> {
    vk::SpecializationInfo::default()
        .map_entries(&SPECIALIZED)
        .data(&SPECIALIZED_VALUE)
}

/// What the device primed has: a queue family that writes timestamps, `hostQueryReset`, device
/// addresses, `VK_EXT_device_generated_commands`, sparse binding on its queues, the push command,
/// calibrated timestamps, and the render encoder's commands of [`PrimedDraws`].
#[derive(Clone, Copy)]
pub(crate) struct Has {
    pub(crate) timestamps: bool,
    pub(crate) host_query_reset: bool,
    pub(crate) device_addresses: bool,
    pub(crate) generated_commands: bool,
    pub(crate) sparse_binding: bool,
    pub(crate) push_descriptor: bool,
    pub(crate) calibration: bool,
    pub(crate) object_names: bool,
    pub(crate) extended_dynamic_state: bool,
    pub(crate) primitive_restart: bool,
    pub(crate) color_write_masks: bool,
    pub(crate) sized_index_bind: bool,
    pub(crate) ycbcr_conversion: bool,
}

/// The commands a render encoder records that the device has where it has their feature or
/// extension, which the priming's render host step makes (a host step records them inside a
/// section).
#[derive(Clone, Copy, Default)]
pub(crate) struct PrimedDraws {
    pub(crate) dynamic: Option<ExtendedDynamicState>,
    pub(crate) color_write_masks: Option<ColorWriteMasks>,
    pub(crate) sized_index_bind: Option<SizedIndexBind>,
}

impl PrimedDraws {
    /// `has`, with what it says of these commands set from them.
    pub(crate) fn has(&self, has: Has) -> Has {
        Has {
            extended_dynamic_state: self.dynamic.is_some(),
            primitive_restart: self
                .dynamic
                .is_some_and(|dynamic| dynamic.primitive_restart_enable.is_some()),
            color_write_masks: self.color_write_masks.is_some(),
            sized_index_bind: self.sized_index_bind.is_some(),
            ..has
        }
    }
}

/// Prime `device` of `physical`, made from `instance`, whose queues are `queues`, the first of
/// its primary family `queue_family`, its buffers and images shared as `sharing` says, on this
/// thread; `has` says what the device has, `generated` holds the commands of
/// `VK_EXT_device_generated_commands` where it has them, `drawn` the render encoder's commands
/// it has, and `ycbcr` a Y′CbCr conversion of a format it converts, where it converts one.
///
/// # Safety
///
/// The device is live and enables timeline semaphores, and `bufferDeviceAddress` where `has`
/// says it has device addresses; `queues` holds at least one of its queues, which nothing else
/// uses yet; `generated` is the device's, which enables the extension and `maintenance5`;
/// `loader_word` is the device's loader word; the device enables `samplerYcbcrConversion` and
/// converts `ycbcr`'s format with its chroma location where `ycbcr` is given.
pub(super) unsafe fn prime(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    device: &Vk,
    loader_word: &LoaderWord,
    queue_family: u32,
    queues: &[QueueSlot],
    sharing: Sharing<'_>,
    has: Has,
    generated: Option<&Commands>,
    push: Option<vk::PFN_vkCmdPushDescriptorSetKHR>,
    calibration: Option<Calibration>,
    drawn: PrimedDraws,
    ycbcr: Option<vk::SamplerYcbcrConversionCreateInfo<'static>>,
) -> Result<(), Error> {
    let mut primer = Primer {
        device,
        loader_word,
        sharing,
        push,
        calibration,
        drawn,
        // SAFETY: the caller's.
        memory: unsafe { instance.get_physical_device_memory_properties(physical) },
        calls: 0,
        held: Vec::new(),
    };
    // SAFETY: the caller's.
    let primed = unsafe {
        primer
            .batch(queue_family, queues, has.timestamps, has.host_query_reset)
            .and_then(|()| primer.objects(queue_family, has.device_addresses))
            .and_then(|()| match generated {
                Some(commands) => primer.generated(commands),
                None => Ok(()),
            })
            .and_then(|()| primer.sparse(queues, has.sparse_binding))
            .and_then(|()| primer.ycbcr(ycbcr))
    };
    // SAFETY: the submission completed, or was never made, or the device is lost; nothing else
    // uses the objects held.
    unsafe {
        let _ = device.device_wait_idle();
        primer.made(Call::DeviceWaitIdle);
        while let Some(release) = primer.held.pop() {
            let call = release(device);
            primer.made(call);
        }
    }
    if primed.is_ok() && cfg!(debug_assertions) {
        let missing = missing(primer.calls, has);
        assert!(
            missing.is_empty(),
            "the exit guard's entry points {missing:?} are not primed (device/prime.rs)"
        );
    }
    primed
}

/// The table's entry points, but [`UNPRIMED`], those of [`TIMESTAMP_CALLS`] where the device
/// has no timestamps, those of [`HOST_QUERY_RESET_CALLS`] where it has no timestamps or does not
/// enable `hostQueryReset`, those of [`DEVICE_ADDRESS_CALLS`] where it has no device addresses,
/// those of [`GENERATED_COMMANDS_CALLS`] where it has no generated commands, those of
/// [`SPARSE_CALLS`] where it binds no sparse memory, those of [`PUSH_DESCRIPTOR_CALLS`] where it
/// has no push command, those of [`CALIBRATION_CALLS`] where it calibrates no timestamps, those
/// of [`NAMING_CALLS`] where its instance reports no messages, and those of
/// [`EXTENDED_DYNAMIC_STATE_CALLS`], [`PRIMITIVE_RESTART_CALLS`],
/// [`COLOR_WRITE_MASK_CALLS`] and [`SIZED_INDEX_BIND_CALLS`] where it has
/// not their commands, and those of [`YCBCR_CONVERSION_CALLS`] where it converts no YCbCr format,
/// that `calls` (one bit each) does not hold.
pub(crate) fn missing(calls: u128, has: Has) -> Vec<Call> {
    Call::ALL
        .iter()
        .copied()
        .filter(|&call| {
            calls & call.bit() == 0
                && !UNPRIMED.contains(&call)
                && (has.timestamps || !TIMESTAMP_CALLS.contains(&call))
                && ((has.timestamps && has.host_query_reset)
                    || !HOST_QUERY_RESET_CALLS.contains(&call))
                && (has.device_addresses || !DEVICE_ADDRESS_CALLS.contains(&call))
                && (has.generated_commands || !GENERATED_COMMANDS_CALLS.contains(&call))
                && (has.sparse_binding || !SPARSE_CALLS.contains(&call))
                && (has.push_descriptor || !PUSH_DESCRIPTOR_CALLS.contains(&call))
                && (has.calibration || !CALIBRATION_CALLS.contains(&call))
                && (has.object_names || !NAMING_CALLS.contains(&call))
                && (has.extended_dynamic_state || !EXTENDED_DYNAMIC_STATE_CALLS.contains(&call))
                && (has.primitive_restart || !PRIMITIVE_RESTART_CALLS.contains(&call))
                && (has.color_write_masks || !COLOR_WRITE_MASK_CALLS.contains(&call))
                && (has.sized_index_bind || !SIZED_INDEX_BIND_CALLS.contains(&call))
                && (has.ycbcr_conversion || !YCBCR_CONVERSION_CALLS.contains(&call))
        })
        .collect()
}

/// What a render host step's priming records with ([`Primer::render_step`]): the never-submitted
/// command buffer, its pool, the render pass and framebuffer, the pipeline layout and cache, and
/// the buffer it binds as a vertex buffer.
struct RenderStep {
    recorded: vk::CommandBuffer,
    commands: vk::CommandPool,
    render_pass: vk::RenderPass,
    framebuffer: vk::Framebuffer,
    layout: vk::PipelineLayout,
    cache: vk::PipelineCache,
    buffer: vk::Buffer,
    /// The occlusion query pool, whose query the secondary begins and ends around its draw.
    queries: vk::QueryPool,
}

/// A release of an object a priming made, which answers its entry point.
type Release = Box<dyn FnOnce(&ash::Device) -> Call>;

/// A priming in progress.
struct Primer<'a> {
    device: &'a Vk,
    /// The device's loader word, put back before each call the loader serves.
    loader_word: &'a LoaderWord,
    /// How the device's buffers and images are shared among its queue families.
    sharing: Sharing<'a>,
    /// The push command, where the device has it (`binding::push_descriptor_command`).
    push: Option<vk::PFN_vkCmdPushDescriptorSetKHR>,
    /// The device's calibrated timestamps, where its queue family writes timestamps.
    calibration: Option<Calibration>,
    /// The render encoder's commands the device has.
    drawn: PrimedDraws,
    memory: vk::PhysicalDeviceMemoryProperties,
    /// The table's entry points called, one bit each.
    calls: u128,
    /// What was made, released last first.
    held: Vec<Release>,
}

impl Primer<'_> {
    /// `call` was made.
    fn made(&mut self, call: Call) {
        self.calls |= call.bit();
    }

    /// The command buffer `vkAllocateCommandBuffers` allocates of `info`, the device's loader
    /// word put back first: the loader serves the call and reads the word, which MoltenVK 1.4.1
    /// overwrites whenever it names the device in a message, as this priming's own submission makes
    /// it do where its performance tracking logs each activity at once (a
    /// `SIGSEGV` at `0x1cdc0de` in the loader's `vkAllocateCommandBuffers`).
    ///
    /// # Safety
    ///
    /// `info` names a command pool of the device's, which this priming holds.
    unsafe fn allocate_command_buffer(
        &self,
        info: &vk::CommandBufferAllocateInfo<'_>,
    ) -> Result<vk::CommandBuffer, Error> {
        // SAFETY: the caller's; the device is live while it is primed.
        let buffers = unsafe {
            self.loader_word.restore();
            self.device.allocate_command_buffers(info)
        };
        Ok(buffers.map_err(vulkan)?[0])
    }

    /// Release `object` with the others, last first.
    fn hold<T: Released + 'static>(&mut self, object: T) {
        self.held.push(Box::new(move |device: &ash::Device| {
            // SAFETY: the object is the device's, and the device is idle when the priming
            // releases what it holds.
            unsafe { release::object(device, object) };
            T::CALL
        }));
    }

    /// A batch's sparse binding, where the device binds sparse memory: an empty bind on the first
    /// queue, which the device's idle wait after the priming waits for.
    ///
    /// # Safety
    ///
    /// As [`prime`]'s; with `sparse_binding`, the queue family binds sparse memory.
    unsafe fn sparse(&mut self, queues: &[QueueSlot], sparse_binding: bool) -> Result<(), Error> {
        if !sparse_binding {
            return Ok(());
        }
        let device = self.device;
        let info = [vk::BindSparseInfo::default()];
        // SAFETY: the caller's; the bind binds nothing, waits for nothing and signals nothing,
        // and the queue is used under its lock.
        queues[0]
            .with(|queue| unsafe { device.queue_bind_sparse(queue, &info, vk::Fence::null()) })
            .map_err(vulkan)?;
        self.made(Call::QueueBindSparse);
        Ok(())
    }

    /// A batch's calls: a command buffer of a new pool with the barriers every batch begins and
    /// ends with (and, with `timestamps`, the reset of a timestamp query pool of two and a
    /// timestamp before and after them, as a command buffer's GPU times are written), submitted
    /// to the first queue after every queue's timeline at 0, signalling a timeline semaphore to 1
    /// and a new fence; the wait for the timeline, a read of every timeline, the wait for the
    /// fence and its reset, and, with `timestamps`, the read of the two timestamps; with
    /// `host_reset` too, their reset on the host first, as a timestamp slot's (`gpu_time.rs`).
    ///
    /// # Safety
    ///
    /// As [`prime`]'s; with `host_reset`, the device enables `hostQueryReset`.
    unsafe fn batch(
        &mut self,
        queue_family: u32,
        queues: &[QueueSlot],
        timestamps: bool,
        host_reset: bool,
    ) -> Result<(), Error> {
        let device = self.device;
        let mut kind = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let info = vk::SemaphoreCreateInfo::default().push_next(&mut kind);
        // SAFETY: the caller's.
        let signal = unsafe { device.create_semaphore(&info, None) }.map_err(vulkan)?;
        self.made(Call::CreateSemaphore);
        self.hold(signal);
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(queue_family);
        // SAFETY: the caller's.
        let pool = unsafe { device.create_command_pool(&pool_info, None) }.map_err(vulkan)?;
        self.made(Call::CreateCommandPool);
        self.hold(pool);
        let allocate = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: the pool is this priming's.
        let buffer = unsafe { self.allocate_command_buffer(&allocate) }?;
        self.made(Call::AllocateCommandBuffers);
        let queries = if timestamps {
            let info = vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count(2);
            // SAFETY: the caller's.
            let queries = unsafe { device.create_query_pool(&info, None) }.map_err(vulkan)?;
            self.hold(queries);
            if host_reset {
                // SAFETY: the device enables `hostQueryReset` (the caller's); nothing uses the
                // new pool's queries.
                unsafe { device.reset_query_pool(queries, 0, 2) };
                self.made(Call::ResetQueryPool);
            }
            Some(queries)
        } else {
            None
        };
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: the command buffer was just allocated and only this call records it.
        unsafe {
            device
                .begin_command_buffer(buffer, &begin)
                .map_err(vulkan)?;
            self.made(Call::BeginCommandBuffer);
            if let Some(queries) = queries {
                device.cmd_reset_query_pool(buffer, queries, 0, 2);
                self.made(Call::CmdResetQueryPool);
                device.cmd_write_timestamp(buffer, vk::PipelineStageFlags::TOP_OF_PIPE, queries, 0);
            }
            Barrier::to_host().record(device, buffer);
            self.made(Call::CmdPipelineBarrier);
            if let Some(queries) = queries {
                device.cmd_write_timestamp(
                    buffer,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    queries,
                    1,
                );
                self.made(Call::CmdWriteTimestamp);
            }
            device.end_command_buffer(buffer).map_err(vulkan)?;
            self.made(Call::EndCommandBuffer);
        }
        let waits: Vec<vk::Semaphore> = queues.iter().map(|queue| queue.timeline).collect();
        let wait_values = vec![0; waits.len()];
        let wait_stages = vec![vk::PipelineStageFlags::ALL_COMMANDS; waits.len()];
        let signals = [signal];
        let signal_values = [1];
        let buffers = [buffer];
        let mut timeline = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&wait_values)
            .signal_semaphore_values(&signal_values);
        let submit = vk::SubmitInfo::default()
            .wait_semaphores(&waits)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&buffers)
            .signal_semaphores(&signals)
            .push_next(&mut timeline);
        // SAFETY: the caller's.
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(vulkan)?;
        self.made(Call::CreateFence);
        self.hold(fence);
        // SAFETY: the command buffer is executable; every wait is for a value the timeline holds
        // already; the signal raises the semaphore; the fence is new; the queue is used under its
        // lock.
        queues[0]
            .with(|queue| unsafe { device.queue_submit(queue, &[submit], fence) })
            .map_err(vulkan)?;
        self.made(Call::QueueSubmit);
        let start = Instant::now();
        let semaphores = [signal];
        let values = [1];
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        loop {
            // SAFETY: the semaphore is live; its signal was submitted.
            match unsafe { device.wait_semaphores(&info, WAIT_SLICE.as_nanos() as u64) } {
                Ok(()) => break,
                Err(vk::Result::TIMEOUT) if start.elapsed() < LIMIT => {}
                Err(error) => return Err(vulkan(error)),
            }
        }
        self.made(Call::WaitSemaphores);
        // A waker's nudge of the scheduler's thread (`scheduler/mod.rs`): the semaphore, at 1,
        // raised from the host, with no signal of it pending.
        let raised = vk::SemaphoreSignalInfo::default()
            .semaphore(signal)
            .value(2);
        // SAFETY: the semaphore is live, its one submitted signal landed, and 2 is above it.
        unsafe { device.signal_semaphore(&raised) }.map_err(vulkan)?;
        self.made(Call::SignalSemaphore);
        for queue in queues {
            // SAFETY: the timeline is live.
            unsafe { device.get_semaphore_counter_value(queue.timeline) }.map_err(vulkan)?;
        }
        self.made(Call::GetSemaphoreCounterValue);
        let fences = [fence];
        loop {
            // SAFETY: the fence is live; its submission was made.
            match unsafe { device.wait_for_fences(&fences, true, WAIT_SLICE.as_nanos() as u64) } {
                Ok(()) => break,
                Err(vk::Result::TIMEOUT) if start.elapsed() < LIMIT => {}
                Err(error) => return Err(vulkan(error)),
            }
        }
        self.made(Call::WaitForFences);
        // SAFETY: the fence is signaled and its submission completed.
        unsafe { device.reset_fences(&fences) }.map_err(vulkan)?;
        self.made(Call::ResetFences);
        // A completion turn's drain of the queues after a failed wait
        // (`Device::drain_queues`).
        // SAFETY: the queue is used under its lock; its one submission completed.
        queues[0]
            .with(|queue| unsafe { device.queue_wait_idle(queue) })
            .map_err(vulkan)?;
        self.made(Call::QueueWaitIdle);
        if let Some(queries) = queries {
            let mut results = [[0u64; 2]; 2];
            // SAFETY: the pool's two queries were written by the submission, which completed;
            // 64-bit results with availability, two words a query.
            let read = unsafe {
                device.get_query_pool_results(
                    queries,
                    0,
                    &mut results,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WITH_AVAILABILITY,
                )
            };
            match read {
                Ok(()) | Err(vk::Result::NOT_READY) => {}
                Err(error) => return Err(vulkan(error)),
            }
            self.made(Call::GetQueryPoolResults);
        }
        if let Some(calibration) = self.calibration {
            let _ = calibration.reading();
            self.made(Call::GetCalibratedTimestamps);
        }
        Ok(())
    }

    /// Release `object`, made through `commands`, with the others, last first.
    fn hold_generated(&mut self, commands: Commands, object: Generated) {
        let call = match object {
            Generated::ExecutionSet(_) => Call::DestroyIndirectExecutionSet,
            Generated::CommandsLayout(_) => Call::DestroyIndirectCommandsLayout,
        };
        self.held.push(Box::new(move |_: &ash::Device| {
            // SAFETY: the object is this priming's, and the device is idle when the priming
            // releases what it holds.
            unsafe { release::generated(&commands, object) };
            call
        }));
    }

    /// An execution set and an indirect commands layout of `VK_EXT_device_generated_commands`,
    /// over an indirect-bindable pipeline of the smallest shader, which the runtime releases
    /// through `commands` (`generated_commands.rs`).
    ///
    /// # Safety
    ///
    /// As [`prime`]'s, `commands` the device's.
    unsafe fn generated(&mut self, commands: &Commands) -> Result<(), Error> {
        let device = self.device;
        // SAFETY: the caller's, for every call below; each object is this priming's, and each
        // create info and what it points at live for its call.
        unsafe {
            let module_info = vk::ShaderModuleCreateInfo::default().code(&SHADER);
            let module = device
                .create_shader_module(&module_info, None)
                .map_err(vulkan)?;
            self.hold(module);
            let layout = device
                .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default(), None)
                .map_err(vulkan)?;
            self.hold(layout);
            let mut flags =
                vk::PipelineCreateFlags2CreateInfoKHR::default().flags(INDIRECT_BINDABLE);
            let specialization = specialization();
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main")
                .specialization_info(&specialization);
            let info = vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(layout)
                .push_next(&mut flags);
            let pipeline = device
                .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
                .map_err(|(_, error)| vulkan(error))?[0];
            self.hold(pipeline);
            let pipelines = ExecutionSetPipelineInfo::new(pipeline, 1);
            let set = commands
                .create_execution_set(&ExecutionSetCreateInfo::new(&pipelines))
                .map_err(vulkan)?;
            self.hold_generated(*commands, Generated::ExecutionSet(set));
            let set_token = ExecutionSetToken {
                kind: EXECUTION_SET_PIPELINES,
                shader_stages: vk::ShaderStageFlags::COMPUTE,
            };
            let tokens = [
                LayoutToken::new(
                    TOKEN_EXECUTION_SET,
                    std::ptr::from_ref(&set_token).cast(),
                    0,
                ),
                LayoutToken::new(TOKEN_DISPATCH, std::ptr::null(), 4),
            ];
            let layout_info = LayoutCreateInfo::new(16, layout, &tokens);
            let commands_layout = commands
                .create_commands_layout(&layout_info)
                .map_err(vulkan)?;
            self.hold_generated(*commands, Generated::CommandsLayout(commands_layout));
        }
        Ok(())
    }

    /// A Y′CbCr conversion of `info`, a format the device converts, where it converts one
    /// ([`YCBCR_CONVERSION_CALLS`]).
    ///
    /// # Safety
    ///
    /// As [`prime`]'s; the device enables `samplerYcbcrConversion` where `info` is given, and
    /// converts its format with its chroma location.
    unsafe fn ycbcr(
        &mut self,
        info: Option<vk::SamplerYcbcrConversionCreateInfo<'static>>,
    ) -> Result<(), Error> {
        let Some(info) = info else {
            return Ok(());
        };
        // SAFETY: the caller's.
        let conversion =
            unsafe { self.device.create_sampler_ycbcr_conversion(&info, None) }.map_err(vulkan)?;
        self.hold(conversion);
        Ok(())
    }

    /// An object of every other kind the runtime releases, the read of a pipeline cache's data,
    /// and a command buffer of a pool of `queue_family` recorded as a host step records one;
    /// `device_addresses` where the device has them.
    ///
    /// # Safety
    ///
    /// As [`prime`]'s.
    unsafe fn objects(&mut self, queue_family: u32, device_addresses: bool) -> Result<(), Error> {
        let device = self.device;
        // SAFETY: the caller's, for every call below; each object is this priming's.
        unsafe {
            let sampler = device
                .create_sampler(&vk::SamplerCreateInfo::default(), None)
                .map_err(vulkan)?;
            self.made(Call::CreateSampler);
            self.hold(sampler);

            let info = vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::OCCLUSION)
                .query_count(1);
            let queries = device.create_query_pool(&info, None).map_err(vulkan)?;
            self.made(Call::CreateQueryPool);
            self.hold(queries);

            // A storage buffer, and a texel buffer of `R32_UINT`, whose uniform texel buffer use
            // every device has, in host-visible memory, which every buffer may take, mapped; the
            // indirect dispatch's and draw's arguments too.
            let mut usage = vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER
                | vk::BufferUsageFlags::VERTEX_BUFFER
                | vk::BufferUsageFlags::INDEX_BUFFER
                | vk::BufferUsageFlags::INDIRECT_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST;
            if device_addresses {
                usage |= vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
            }
            let info = self
                .sharing
                .buffer(vk::BufferCreateInfo::default().size(256).usage(usage));
            let buffer = device.create_buffer(&info, None).map_err(vulkan)?;
            self.made(Call::CreateBuffer);
            let requirements = device.get_buffer_memory_requirements(buffer);
            self.made(Call::GetBufferMemoryRequirements);
            let host =
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let memory = match self.allocate(requirements, host, device_addresses) {
                Ok(memory) => memory,
                Err(error) => {
                    release::object(device, buffer);
                    return Err(error);
                }
            };
            self.hold(buffer);
            device
                .bind_buffer_memory(buffer, memory, 0)
                .map_err(vulkan)?;
            self.made(Call::BindBufferMemory);
            // Unmapped as the memory is freed.
            device
                .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .map_err(vulkan)?;
            self.made(Call::MapMemory);
            if device_addresses {
                let info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
                device.get_buffer_device_address(&info);
                self.made(Call::GetBufferDeviceAddress);
            }
            let info = vk::BufferViewCreateInfo::default()
                .buffer(buffer)
                .format(vk::Format::R32_UINT)
                .range(vk::WHOLE_SIZE);
            let view = device.create_buffer_view(&info, None).map_err(vulkan)?;
            self.made(Call::CreateBufferView);
            self.hold(view);

            // A color attachment of `R8G8B8A8_UNORM`, which every device can sample, draw to and
            // copy, two texels wide for a copy between its texels.
            let format = vk::Format::R8G8B8A8_UNORM;
            let info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D {
                    width: 2,
                    height: 1,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(
                    vk::ImageUsageFlags::COLOR_ATTACHMENT
                        | vk::ImageUsageFlags::SAMPLED
                        | vk::ImageUsageFlags::TRANSFER_SRC
                        | vk::ImageUsageFlags::TRANSFER_DST,
                )
                .initial_layout(vk::ImageLayout::UNDEFINED);
            let info = self.sharing.image(info);
            let image = device.create_image(&info, None).map_err(vulkan)?;
            self.made(Call::CreateImage);
            let requirements = device.get_image_memory_requirements(image);
            self.made(Call::GetImageMemoryRequirements);
            let mut requirements2 = vk::MemoryRequirements2::default();
            let query = vk::ImageMemoryRequirementsInfo2::default().image(image);
            device.get_image_memory_requirements2(&query, &mut requirements2);
            self.made(Call::GetImageMemoryRequirements2);
            let memory = match self.allocate(requirements, vk::MemoryPropertyFlags::empty(), false)
            {
                Ok(memory) => memory,
                Err(error) => {
                    release::object(device, image);
                    return Err(error);
                }
            };
            self.hold(image);
            device.bind_image_memory(image, memory, 0).map_err(vulkan)?;
            self.made(Call::BindImageMemory);
            let info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            let view = device.create_image_view(&info, None).map_err(vulkan)?;
            self.made(Call::CreateImageView);
            self.hold(view);
            // A depth image of `D16_UNORM`, which every device can draw to and clear, for a
            // batch's first work's clear of a neutral depth image.
            let info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::D16_UNORM)
                .extent(vk::Extent3D {
                    width: 1,
                    height: 1,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(
                    vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                        | vk::ImageUsageFlags::TRANSFER_DST,
                )
                .initial_layout(vk::ImageLayout::UNDEFINED);
            let info = self.sharing.image(info);
            let depth = device.create_image(&info, None).map_err(vulkan)?;
            let requirements = device.get_image_memory_requirements(depth);
            let memory = match self.allocate(requirements, vk::MemoryPropertyFlags::empty(), false)
            {
                Ok(memory) => memory,
                Err(error) => {
                    release::object(device, depth);
                    return Err(error);
                }
            };
            self.hold(depth);
            device.bind_image_memory(depth, memory, 0).map_err(vulkan)?;
            let attachments = [vk::AttachmentDescription::default()
                .format(format)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::DONT_CARE)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
            let colors = [vk::AttachmentReference {
                attachment: 0,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            }];
            let subpasses = [vk::SubpassDescription::default()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(&colors)];
            let info = vk::RenderPassCreateInfo::default()
                .attachments(&attachments)
                .subpasses(&subpasses);
            let render_pass = device.create_render_pass(&info, None).map_err(vulkan)?;
            self.made(Call::CreateRenderPass);
            self.hold(render_pass);
            // The same in `vkCreateRenderPass2`'s form, a render encoder's pass's
            // (`render_target.rs`).
            let attachments = [vk::AttachmentDescription2::default()
                .format(format)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::DONT_CARE)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
            let colors = [vk::AttachmentReference2::default()
                .attachment(0)
                .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .aspect_mask(vk::ImageAspectFlags::COLOR)];
            let subpasses = [vk::SubpassDescription2::default()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(&colors)];
            let info = vk::RenderPassCreateInfo2::default()
                .attachments(&attachments)
                .subpasses(&subpasses);
            let render_pass2 = device.create_render_pass2(&info, None).map_err(vulkan)?;
            self.made(Call::CreateRenderPass2);
            self.hold(render_pass2);
            let views = [view];
            let info = vk::FramebufferCreateInfo::default()
                .render_pass(render_pass)
                .attachments(&views)
                .width(1)
                .height(1)
                .layers(1);
            let framebuffer = device.create_framebuffer(&info, None).map_err(vulkan)?;
            self.made(Call::CreateFramebuffer);
            self.hold(framebuffer);

            // A compute pipeline with one storage buffer binding and a push constant range,
            // made with a pipeline cache.
            let bindings = [vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)];
            let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
            let mut support = vk::DescriptorSetLayoutSupport::default();
            device.get_descriptor_set_layout_support(&info, &mut support);
            self.made(Call::GetDescriptorSetLayoutSupport);
            let set_layout = device
                .create_descriptor_set_layout(&info, None)
                .map_err(vulkan)?;
            self.made(Call::CreateDescriptorSetLayout);
            self.hold(set_layout);
            let set_layouts = [set_layout];
            let push = [vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                offset: 0,
                size: 4,
            }];
            let info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&set_layouts)
                .push_constant_ranges(&push);
            let layout = device.create_pipeline_layout(&info, None).map_err(vulkan)?;
            self.made(Call::CreatePipelineLayout);
            self.hold(layout);
            let sizes = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
            }];
            let info = vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&sizes);
            let pool = device.create_descriptor_pool(&info, None).map_err(vulkan)?;
            self.made(Call::CreateDescriptorPool);
            self.hold(pool);
            // A set of it, written: the runtime releases its pools with their sets.
            let info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&set_layouts);
            let set = device.allocate_descriptor_sets(&info).map_err(vulkan)?[0];
            self.made(Call::AllocateDescriptorSets);
            let buffers = [vk::DescriptorBufferInfo {
                buffer,
                offset: 0,
                range: vk::WHOLE_SIZE,
            }];
            let writes = [vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&buffers)];
            device.update_descriptor_sets(&writes, &[]);
            self.made(Call::UpdateDescriptorSets);
            let info = vk::ShaderModuleCreateInfo::default().code(&SHADER);
            let module = device.create_shader_module(&info, None).map_err(vulkan)?;
            self.made(Call::CreateShaderModule);
            self.hold(module);
            let cache = device
                .create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None)
                .map_err(vulkan)?;
            self.made(Call::CreatePipelineCache);
            self.hold(cache);
            let specialization = specialization();
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main")
                .specialization_info(&specialization);
            let info = [vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(layout)];
            let pipeline = device
                .create_compute_pipelines(cache, &info, None)
                .map_err(|(_, error)| vulkan(error))?[0];
            self.made(Call::CreateComputePipelines);
            self.hold(pipeline);
            device.get_pipeline_cache_data(cache).map_err(vulkan)?;
            self.made(Call::GetPipelineCacheData);

            // A host step's recording, in a command buffer never submitted, which its pool
            // releases.
            let info = vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::TRANSIENT)
                .queue_family_index(queue_family);
            let commands = device.create_command_pool(&info, None).map_err(vulkan)?;
            self.hold(commands);
            let allocate = vk::CommandBufferAllocateInfo::default()
                .command_pool(commands)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let recorded = self.allocate_command_buffer(&allocate)?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device
                .begin_command_buffer(recorded, &begin)
                .map_err(vulkan)?;
            device.cmd_bind_pipeline(recorded, vk::PipelineBindPoint::COMPUTE, pipeline);
            self.made(Call::CmdBindPipeline);
            // Where the device has the push command, the set's write pushed with a layout whose
            // set 0 takes push descriptors, before the set is bound with the pipeline's layout
            // (which disturbs it). The push ignores the write's `dstSet`.
            if let Some(push) = self.push {
                let info = vk::DescriptorSetLayoutCreateInfo::default()
                    .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
                    .bindings(&bindings);
                let pushed_set = device
                    .create_descriptor_set_layout(&info, None)
                    .map_err(vulkan)?;
                self.hold(pushed_set);
                let pushed_sets = [pushed_set];
                let info = vk::PipelineLayoutCreateInfo::default().set_layouts(&pushed_sets);
                let pushed = device.create_pipeline_layout(&info, None).map_err(vulkan)?;
                self.hold(pushed);
                push(
                    recorded,
                    vk::PipelineBindPoint::COMPUTE,
                    pushed,
                    0,
                    writes.len() as u32,
                    writes.as_ptr(),
                );
                self.made(Call::CmdPushDescriptorSet);
            }
            device.cmd_bind_descriptor_sets(
                recorded,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                &[set],
                &[],
            );
            self.made(Call::CmdBindDescriptorSets);
            device.cmd_push_constants(recorded, layout, vk::ShaderStageFlags::COMPUTE, 0, &[0; 4]);
            self.made(Call::CmdPushConstants);
            device.cmd_dispatch(recorded, 1, 1, 1);
            self.made(Call::CmdDispatch);
            // An emulated mesh command's dispatch from its scan's arguments.
            device.cmd_dispatch_indirect(recorded, buffer, 0);
            self.made(Call::CmdDispatchIndirect);
            let region = [vk::BufferCopy {
                src_offset: 0,
                dst_offset: 128,
                size: 4,
            }];
            device.cmd_copy_buffer(recorded, buffer, buffer, &region);
            self.made(Call::CmdCopyBuffer);
            let region = [vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width: 1,
                    height: 1,
                    depth: 1,
                })];
            device.cmd_copy_buffer_to_image(
                recorded,
                buffer,
                image,
                vk::ImageLayout::GENERAL,
                &region,
            );
            self.made(Call::CmdCopyBufferToImage);
            // The image's write before its read, and the buffer's reads before its write: the
            // validation layer checks a recording's order without its submission.
            Barrier::everything().record(device, recorded);
            device.cmd_copy_image_to_buffer(
                recorded,
                image,
                vk::ImageLayout::GENERAL,
                buffer,
                &region,
            );
            self.made(Call::CmdCopyImageToBuffer);
            // A copy of its first texel to its second, after the read.
            Barrier::everything().record(device, recorded);
            let texel = vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            };
            let region = [vk::ImageCopy {
                src_subresource: texel,
                src_offset: vk::Offset3D::default(),
                dst_subresource: texel,
                dst_offset: vk::Offset3D { x: 1, y: 0, z: 0 },
                extent: vk::Extent3D {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
            }];
            let general = vk::ImageLayout::GENERAL;
            device.cmd_copy_image(recorded, image, general, image, general, &region);
            self.made(Call::CmdCopyImage);
            // A clear of the image after the copy, as a render host step's pass steps clear a
            // sample plane or a stencil shadow (`render/pass_steps.rs`).
            Barrier::everything().record(device, recorded);
            let whole = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };
            let black = vk::ClearColorValue::default();
            device.cmd_clear_color_image(recorded, image, general, &black, &[whole]);
            self.made(Call::CmdClearColorImage);
            // A clear of the depth image, as a batch's first work clears a neutral depth image.
            let depth_whole = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                ..whole
            };
            let zero = vk::ClearDepthStencilValue::default();
            device.cmd_clear_depth_stencil_image(recorded, depth, general, &zero, &[depth_whole]);
            self.made(Call::CmdClearDepthStencilImage);
            // What an encoder records that no host step does: a buffer's fill and its inline
            // update, each at bytes
            // of its own, a copy of the query's result, and a blit of the image's first texel to
            // its second after the clear.
            device.cmd_fill_buffer(recorded, buffer, 192, 4, 0);
            self.made(Call::CmdFillBuffer);
            device.cmd_update_buffer(recorded, buffer, 208, &[0; 4]);
            self.made(Call::CmdUpdateBuffer);
            let results = vk::QueryResultFlags::empty();
            device.cmd_copy_query_pool_results(recorded, queries, 0, 1, buffer, 224, 4, results);
            self.made(Call::CmdCopyQueryPoolResults);
            Barrier::everything().record(device, recorded);
            let corner = |x| vk::Offset3D { x, y: 0, z: 0 };
            let far = |x| vk::Offset3D { x, y: 1, z: 1 };
            let blit = [vk::ImageBlit {
                src_subresource: texel,
                src_offsets: [corner(0), far(1)],
                dst_subresource: texel,
                dst_offsets: [corner(1), far(2)],
            }];
            device.cmd_blit_image(
                recorded,
                image,
                general,
                image,
                general,
                &blit,
                vk::Filter::NEAREST,
            );
            self.made(Call::CmdBlitImage);
            // The image's read before the pass's layout transition of it.
            Barrier::everything().record(device, recorded);
            self.render_step(RenderStep {
                recorded,
                commands,
                render_pass,
                framebuffer,
                layout,
                cache,
                buffer,
                queries,
            })?;
            device.end_command_buffer(recorded).map_err(vulkan)?;
        }
        Ok(())
    }

    /// A render host step's recording into `step.recorded` (`render/executions.rs`): a graphics
    /// pipeline that rasterizes nothing, made with the pipeline cache and drawn in a secondary
    /// command buffer after each state a draw records is set, and the pass's begin, the
    /// secondary's execution and its end.
    ///
    /// # Safety
    ///
    /// As [`prime`]'s; `step.recorded` is recording outside a render pass, and the objects are
    /// this priming's.
    unsafe fn render_step(&mut self, step: RenderStep) -> Result<(), Error> {
        let device = self.device;
        let RenderStep {
            recorded,
            commands,
            render_pass,
            framebuffer,
            layout,
            cache,
            buffer,
            queries,
        } = step;
        // SAFETY: the caller's, for every call below.
        unsafe {
            let info = vk::ShaderModuleCreateInfo::default().code(&VERTEX_SHADER);
            let module = device.create_shader_module(&info, None).map_err(vulkan)?;
            self.hold(module);
            let specialization = specialization();
            let stages = [vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(module)
                .name(c"main")
                .specialization_info(&specialization)];
            let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
            // Triangles: a point list's vertex shader must write the point size where the device
            // does not enable `maintenance5`.
            let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
            let raster = vk::PipelineRasterizationStateCreateInfo::default()
                .rasterizer_discard_enable(true)
                .polygon_mode(vk::PolygonMode::FILL)
                .line_width(1.0);
            let info = [vk::GraphicsPipelineCreateInfo::default()
                .stages(&stages)
                .vertex_input_state(&vertex_input)
                .input_assembly_state(&assembly)
                .rasterization_state(&raster)
                .layout(layout)
                .render_pass(render_pass)
                .subpass(0)];
            let pipeline = device
                .create_graphics_pipelines(cache, &info, None)
                .map_err(|(_, error)| vulkan(error))?[0];
            self.made(Call::CreateGraphicsPipelines);
            self.hold(pipeline);
            // Its debug name, where the instance reports messages, as a making names its pipeline
            // (`pipeline/graphics.rs`): the loader serves the call.
            if let Some(name) = device.object_name() {
                self.loader_word.restore();
                let info = vk::DebugUtilsObjectNameInfoEXT::default()
                    .object_handle(pipeline)
                    .object_name(PRIMING_PIPELINE);
                let _ = name(device.handle(), &info);
                self.made(Call::SetDebugUtilsObjectName);
            }

            let allocate = vk::CommandBufferAllocateInfo::default()
                .command_pool(commands)
                .level(vk::CommandBufferLevel::SECONDARY)
                .command_buffer_count(1);
            let draws = self.allocate_command_buffer(&allocate)?;
            let inheritance = vk::CommandBufferInheritanceInfo::default()
                .render_pass(render_pass)
                .subpass(0);
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(
                    vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT
                        | vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE,
                )
                .inheritance_info(&inheritance);
            device.begin_command_buffer(draws, &begin).map_err(vulkan)?;
            // The states a draw records, each set before the pipeline is bound: one that
            // rasterizes nothing keeps them all static, and a state set before a bind may be
            // static in the pipeline bound.
            let extent = vk::Extent2D {
                width: 1,
                height: 1,
            };
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: 1.0,
                height: 1.0,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            device.cmd_set_viewport(draws, 0, &[viewport]);
            self.made(Call::CmdSetViewport);
            let area = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            };
            device.cmd_set_scissor(draws, 0, &[area]);
            self.made(Call::CmdSetScissor);
            device.cmd_set_depth_bias(draws, 0.0, 0.0, 0.0);
            self.made(Call::CmdSetDepthBias);
            device.cmd_set_blend_constants(draws, &[0.0; 4]);
            self.made(Call::CmdSetBlendConstants);
            let faces = vk::StencilFaceFlags::FRONT_AND_BACK;
            device.cmd_set_stencil_compare_mask(draws, faces, 0xff);
            self.made(Call::CmdSetStencilCompareMask);
            device.cmd_set_stencil_write_mask(draws, faces, 0xff);
            self.made(Call::CmdSetStencilWriteMask);
            device.cmd_set_stencil_reference(draws, faces, 0);
            self.made(Call::CmdSetStencilReference);
            // The extended dynamic states and the extension states the device has, set before
            // the bind too.
            let drawn = self.drawn;
            if let Some(dynamic) = drawn.dynamic {
                (dynamic.cull_mode)(draws, vk::CullModeFlags::NONE);
                self.made(Call::CmdSetCullMode);
                (dynamic.front_face)(draws, vk::FrontFace::COUNTER_CLOCKWISE);
                self.made(Call::CmdSetFrontFace);
                (dynamic.primitive_topology)(draws, vk::PrimitiveTopology::TRIANGLE_LIST);
                self.made(Call::CmdSetPrimitiveTopology);
                (dynamic.depth_test_enable)(draws, vk::FALSE);
                self.made(Call::CmdSetDepthTestEnable);
                (dynamic.depth_write_enable)(draws, vk::FALSE);
                self.made(Call::CmdSetDepthWriteEnable);
                (dynamic.depth_compare_op)(draws, vk::CompareOp::ALWAYS);
                self.made(Call::CmdSetDepthCompareOp);
                (dynamic.stencil_test_enable)(draws, vk::FALSE);
                self.made(Call::CmdSetStencilTestEnable);
                let keep = vk::StencilOp::KEEP;
                (dynamic.stencil_op)(draws, faces, keep, keep, keep, vk::CompareOp::ALWAYS);
                self.made(Call::CmdSetStencilOp);
                if let Some(restart) = dynamic.primitive_restart_enable {
                    restart(draws, vk::FALSE);
                    self.made(Call::CmdSetPrimitiveRestartEnable);
                }
            }
            if let Some(ColorWriteMasks(masks)) = drawn.color_write_masks {
                let all = [vk::ColorComponentFlags::RGBA];
                masks(draws, 0, all.len() as u32, all.as_ptr());
                self.made(Call::CmdSetColorWriteMask);
            }
            // A clear of the attachment inside the pass, as a render encoder's clear of a stand-in
            // records it.
            let clear = [vk::ClearAttachment {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                color_attachment: 0,
                clear_value: vk::ClearValue::default(),
            }];
            let rect = [vk::ClearRect {
                rect: area,
                base_array_layer: 0,
                layer_count: 1,
            }];
            device.cmd_clear_attachments(draws, &clear, &rect);
            self.made(Call::CmdClearAttachments);
            device.cmd_bind_pipeline(draws, vk::PipelineBindPoint::GRAPHICS, pipeline);
            device.cmd_bind_vertex_buffers(draws, 0, &[buffer], &[0]);
            self.made(Call::CmdBindVertexBuffers);
            if let Some(dynamic) = drawn.dynamic {
                let (buffers, offsets) = ([buffer], [0]);
                let none = std::ptr::null();
                (dynamic.bind_vertex_buffers2)(
                    draws,
                    0,
                    1,
                    buffers.as_ptr(),
                    offsets.as_ptr(),
                    none,
                    none,
                );
                self.made(Call::CmdBindVertexBuffers2);
            }
            // An execution under a visibility result mode counts its draws in a query begun and
            // ended in its secondary.
            device.cmd_begin_query(draws, queries, 0, vk::QueryControlFlags::empty());
            self.made(Call::CmdBeginQuery);
            device.cmd_draw(draws, 3, 1, 0, 0);
            self.made(Call::CmdDraw);
            // An emulated mesh command's draw from its scan's arguments.
            device.cmd_draw_indirect(draws, buffer, 0, 1, 16);
            self.made(Call::CmdDrawIndirect);
            // A draw recorded once its pipeline is made (`render/pending.rs`): indexed, direct
            // and from its arguments.
            device.cmd_bind_index_buffer(draws, buffer, 0, vk::IndexType::UINT32);
            self.made(Call::CmdBindIndexBuffer);
            if let Some(bind) = drawn.sized_index_bind {
                bind(draws, buffer, 0, 256, vk::IndexType::UINT32);
                self.made(Call::CmdBindIndexBuffer2);
            }
            device.cmd_draw_indexed(draws, 3, 1, 0, 0, 0);
            self.made(Call::CmdDrawIndexed);
            device.cmd_draw_indexed_indirect(draws, buffer, 0, 1, 20);
            self.made(Call::CmdDrawIndexedIndirect);
            device.cmd_end_query(draws, queries, 0);
            self.made(Call::CmdEndQuery);
            device.end_command_buffer(draws).map_err(vulkan)?;

            let begin = vk::RenderPassBeginInfo::default()
                .render_pass(render_pass)
                .framebuffer(framebuffer)
                .render_area(area);
            device.cmd_begin_render_pass(
                recorded,
                &begin,
                vk::SubpassContents::SECONDARY_COMMAND_BUFFERS,
            );
            self.made(Call::CmdBeginRenderPass);
            device.cmd_execute_commands(recorded, &[draws]);
            self.made(Call::CmdExecuteCommands);
            device.cmd_end_render_pass(recorded);
            self.made(Call::CmdEndRenderPass);
        }
        Ok(())
    }

    /// Memory of the first type `requirements` allow with `properties`, with the device-address
    /// flag where `device_addresses`, held (released after what is bound to it).
    ///
    /// # Safety
    ///
    /// As [`prime`]'s.
    unsafe fn allocate(
        &mut self,
        requirements: vk::MemoryRequirements,
        properties: vk::MemoryPropertyFlags,
        device_addresses: bool,
    ) -> Result<vk::DeviceMemory, Error> {
        let types = &self.memory.memory_types[..self.memory.memory_type_count as usize];
        let index = (0_u32..)
            .zip(types)
            .find(|&(index, kind)| {
                requirements.memory_type_bits & (1 << index) != 0
                    && memory::allocatable(kind.property_flags)
                    && kind.property_flags.contains(properties)
            })
            .map(|(index, _)| index)
            .ok_or_else(|| vulkan(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY))?;
        let mut flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let mut info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(index);
        if device_addresses {
            info = info.push_next(&mut flags);
        }
        // SAFETY: the caller's; the type is one the requirements allow.
        let memory = unsafe { self.device.allocate_memory(&info, None) }.map_err(vulkan)?;
        self.made(Call::AllocateMemory);
        self.hold(memory);
        Ok(memory)
    }
}
