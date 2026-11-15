//! A degrade notice names the command buffer whose work announced it
//! (`DegradeNotice::command_buffer`, `CommandBuffer::object_id`), on whichever thread the work
//! runs: an indirect command buffer's execution recorded at a host step on the scheduler's steps
//! thread announces `released-command-state` there naming the executing command buffer, its commit
//! announces `host-replayed-indirect-commands` on the program's thread naming it too, and a notice
//! of the device's (`private-resource-option`, a buffer of commands made while another command
//! buffer's encoder is open) names none.
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects. The kernels are `irw_encode_dispatch` and `irw_mark` of the owned
//! fixture `validation/fixtures/public/engine/icb_routed_writes.ll` (`llvm-as`).

use super::blit_holds::{Scratch, WAIT};
use super::encoder_holds::library;
use crate::{
    CommandBufferStatus, ComputePipelineState, Degradation, DegradeNotice, Device, DeviceKind,
    DeviceOptions, IndirectCommandBufferDescriptor, MessageSink, PipelineOptions, ResourceOptions,
    ShaderCacheOptions, Size,
};
use metal2vulkan::metal::{IndirectCommandType, ResourceUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread::ThreadId;

/// What the sink heard: each notice, with the thread that announced it and whether that thread
/// was recording a host step then.
type Heard = Arc<Mutex<Vec<(DegradeNotice, ThreadId, bool)>>>;

/// A device of the kind the cfg selects, its shader cache in `cache`, its notices into `heard`.
fn device(cache: &Path, heard: &Heard) -> Device {
    let heard = Arc::clone(heard);
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
        messages: Some(MessageSink::new(move |message| {
            if let Some(notice) = &message.degradation {
                let step = crate::exit_guard::in_host_step();
                let thread = std::thread::current().id();
                heard.lock().unwrap().push((notice.clone(), thread, step));
            }
        })),
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// A state of the fixture's `kernel`, made with `supportIndirectCommandBuffers` where `indirect`.
fn state(device: &Device, kernel: &str, indirect: bool) -> ComputePipelineState {
    let function = library(device, "engine/icb_routed_writes.ll")
        .new_function_with_name(kernel)
        .unwrap();
    let mut descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    descriptor.support_indirect_command_buffers = indirect;
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// A command buffer's kernel writes a command naming the ID of a state the program released while
/// no indirect command buffer of the device lived, whose program the device keeps for none
/// (`released_programs.rs`), and a later encoder executes it: the commit leaves the execution to a
/// host step, whose recording runs nothing for it and announces `released-command-state`. Both
/// that notice, heard at the host step on a thread other than the program's, and the commit's
/// `host-replayed-indirect-commands`, heard on the program's thread outside any host step, name
/// the executing command buffer, not the one made before it; the device's
/// `private-resource-option`, of a buffer of commands made while the executing command buffer's
/// encoder is open, names none.
#[test]
fn notices_name_the_command_buffer_whose_work_announced_them() {
    let scratch = Scratch::new("notice-command-buffers");
    let heard = Heard::default();
    let device = device(&scratch.0, &heard);
    let shared = ResourceOptions::shared();
    // The program releases the state before it makes any buffer of commands.
    let released = state(&device, "irw_mark", true).gpu_resource_id();
    let writer = state(&device, "irw_encode_dispatch", false);
    let mut descriptor = IndirectCommandBufferDescriptor::new();
    descriptor.command_types = IndirectCommandType::CONCURRENT_DISPATCH;
    descriptor.max_kernel_buffer_bind_count = 1;
    let icb = device
        .new_indirect_command_buffer(&descriptor, 1, shared)
        .unwrap();
    let table = [icb.gpu_resource_id().0, released.0].map(u64::to_le_bytes);
    let table = device
        .new_buffer_with_bytes(&table.concat(), shared)
        .unwrap();
    let output = device.new_buffer(16, shared).unwrap();

    let queue = device.new_command_queue().unwrap();
    let before = queue.command_buffer().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let id = command_buffer.object_id();
    assert_ne!(before.object_id(), id);
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder
        .use_resource(icb.as_resource(), ResourceUsage::WRITE)
        .unwrap();
    encoder.set_compute_pipeline_state(&writer).unwrap();
    encoder.set_buffer(Some(&table), 0, 0).unwrap();
    encoder.set_buffer(Some(&output), 0, 1).unwrap();
    let one = Size::new(1, 1, 1);
    encoder.dispatch_threads(one, one).unwrap();
    encoder.end_encoding().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    // The device's notice, made between the encoder's calls.
    let private = ResourceOptions::from_raw(shared.raw() | ResourceOptions::PRIVATE_BIT).unwrap();
    let _private = device
        .new_indirect_command_buffer(&descriptor, 1, private)
        .unwrap();
    encoder
        .use_resource(output.as_resource(), ResourceUsage::WRITE)
        .unwrap();
    encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);

    let program = std::thread::current().id();
    let heard = heard.lock().unwrap();
    let find = |kind: Degradation| {
        (heard.iter())
            .find(|(notice, ..)| notice.degradation == kind)
            .unwrap_or_else(|| panic!("no {}: {heard:?}", kind.code()))
    };
    let (notice, thread, step) = find(Degradation::ReleasedCommandState);
    assert!(*step, "released-command-state outside a host step");
    assert_ne!(*thread, program, "a host step on the program's thread");
    assert_eq!(notice.command_buffer, Some(id), "{notice:?}");
    let (notice, thread, step) = find(Degradation::HostReplayedIndirectCommands);
    assert!(
        !*step && *thread == program,
        "the commit's notice elsewhere"
    );
    assert_eq!(notice.command_buffer, Some(id), "{notice:?}");
    let (notice, ..) = find(Degradation::PrivateResourceOption);
    assert_eq!(notice.command_buffer, None, "{notice:?}");
    drop(before);
}
