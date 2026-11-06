//! A predicated loop's runs again (`command_buffer/rerun.rs`) record the work its laying left
//! apart with descriptor sets and uploads of the check's own arenas, let go once each run
//! completed: the command buffer's own upload arena holds no more after forty runs than after
//! three, and a descriptor arena lent so takes every set, its pools reset and taken again run
//! after run.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The kernel is `scale_and_offset` of the
//! owned fixture `validation/fixtures/public/examples/runtime_examples.ll` (`llvm-as`):
//! `output[i] = input[i] * scale + i`.

use crate::descriptor_arena::{DescriptorArena, SetNeeds};
use crate::device::lock;
use crate::{
    CommandBufferStatus, ComputePipelineState, Device, DeviceKind, DeviceOptions,
    IndirectCommandBufferDescriptor, PipelineOptions, ResourceOptions, Size,
};
use ash::vk;
use metal2vulkan::metal::{IndirectCommandType, ResourceUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The longest the test waits for the loop's next run again, and for its command buffer: the
/// runs again take as long as the GPU's other work lets them (the forty took 4 s on an RTX 5080
/// Laptop GPU alone and up to 59 s in a suite sharing it), so the wait bounds a stall, not the
/// forty.
const WAIT: Duration = Duration::from_secs(60);

/// The threads each dispatch runs.
const COUNT: u32 = 1000;

/// The device the cfg selects: a CPU device under `m2v_cpu_icd`, else a GPU.
fn device() -> Device {
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// A state of the fixture's `scale_and_offset` that an indirect command buffer's commands may
/// set.
fn state(device: &Device) -> ComputePipelineState {
    let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/examples/runtime_examples.ll");
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    let library = metal2vulkan::library::Library::from_air(&bitcode).unwrap();
    let library = device.new_library_with_compile_library(library);
    let function = library.new_function_with_name("scale_and_offset").unwrap();
    let mut descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    descriptor.support_indirect_command_buffers = true;
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// The little-endian bytes of `words`.
fn bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

/// A DoWhile whose body is work the runtime records apart, which each run again records again (a
/// dispatch from GPU arguments, which a host step reads, and an indirect command buffer's
/// execution), and whose condition holds until the host writes its word, runs again and again;
/// meanwhile the command buffer's upload arena stays as it was after its third run again. Once
/// the word is written the loop ends and the command buffer completes, the output written.
#[test]
fn a_loop_run_again_holds_no_more_uploads_each_run() {
    let device = device();
    let state = state(&device);
    let shared = ResourceOptions::shared();
    let input: Vec<u8> = (0..COUNT).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let input = device.new_buffer_with_bytes(&input, shared).unwrap();
    let output = device.new_buffer(u64::from(COUNT) * 4, shared).unwrap();
    let scale = device
        .new_buffer_with_bytes(&2.0f32.to_le_bytes(), shared)
        .unwrap();
    let arguments = bytes(&[COUNT, 1, 1, 64, 1, 1]);
    let arguments = device.new_buffer_with_bytes(&arguments, shared).unwrap();
    let word = device.new_buffer(16, shared).unwrap();
    let mut descriptor = IndirectCommandBufferDescriptor::new();
    descriptor.command_types = IndirectCommandType::CONCURRENT_DISPATCH_THREADS;
    descriptor.max_kernel_buffer_bind_count = 3;
    let icb = device
        .new_indirect_command_buffer(&descriptor, 1, shared)
        .unwrap();
    let command = icb.indirect_compute_command_at(0).unwrap();
    command.set_compute_pipeline_state(&state).unwrap();
    for (index, buffer) in (0..).zip([&input, &output, &scale]) {
        command.set_kernel_buffer(Some(buffer), 0, index).unwrap();
    }
    let (grid, group) = (Size::new(u64::from(COUNT), 1, 1), Size::new(64, 1, 1));
    command.concurrent_dispatch_threads(grid, group).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&state).unwrap();
    for (index, buffer) in (0..).zip([&input, &output, &scale]) {
        encoder.set_buffer(Some(buffer), 0, index).unwrap();
    }
    let resources = [
        input.as_resource(),
        output.as_resource(),
        scale.as_resource(),
    ];
    let usage = ResourceUsage::READ | ResourceUsage::WRITE;
    encoder.use_resources(&resources, usage).unwrap();
    encoder.encode_start_do_while().unwrap();
    encoder.dispatch_threads_indirect(&arguments, 0).unwrap();
    encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
    // Again while word 0 is 0 (Metal's code 0, `==`).
    let end = encoder.encode_end_do_while(Some(&word), 0, 0, 0);
    assert_eq!(end, Ok(true));
    encoder.end_encoding().unwrap();
    let before = device.work_counts().predicated_loop_runs_again;
    let started = Instant::now();
    command_buffer.commit().unwrap();
    // The longest wait for one run again.
    let mut longest = Duration::ZERO;
    let mut ran_again = |runs: u64| {
        let mut seen = device.work_counts().predicated_loop_runs_again;
        let mut since = Instant::now();
        while seen < before + runs {
            std::thread::sleep(Duration::from_millis(1));
            let now = device.work_counts().predicated_loop_runs_again;
            if now > seen {
                longest = longest.max(since.elapsed());
                (seen, since) = (now, Instant::now());
            }
            assert!(
                since.elapsed() < WAIT,
                "no run again in {WAIT:?} after {} of {runs}",
                seen - before
            );
        }
    };
    ran_again(3);
    let early = command_buffer.upload_bytes();
    ran_again(40);
    let late = command_buffer.upload_bytes();
    eprintln!(
        "RERUN-ARENAS: 40 runs again in {:?}, the longest wait for one {longest:?}",
        started.elapsed()
    );
    let contents = word.contents().unwrap();
    // SAFETY: the word's first 4 bytes; the GPU only reads them.
    unsafe { contents.as_ptr().cast::<u32>().write_volatile(1) };
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(
        late,
        early,
        "the upload arena grew by {} bytes over 37 runs again",
        late.saturating_sub(early)
    );
    let contents = output.contents().unwrap();
    // SAFETY: the output's first words, which no command buffer writes now.
    let words = unsafe { std::slice::from_raw_parts(contents.as_ptr().cast::<f32>(), 4) };
    assert_eq!(words, [0.0, 3.0, 6.0, 9.0]);
}

/// A descriptor arena lent another (`DescriptorArena::lend`, as a run again's work apart records)
/// allocates every set from the lent one and opens no pool of its own; the lent one's pools,
/// reset inside a section of the exit guard after each of twenty runs of forty sets
/// (`DescriptorArena::reset`, as each run again completed), take every later run's sets: the pools
/// the first run opened are all it ever opens.
#[test]
fn a_lent_descriptor_arena_takes_every_set_and_its_reset_opens_no_pool() {
    let device = device();
    let raw = device.shared().vk();
    let bindings = [vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)];
    let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    // SAFETY: `info` is live for the call.
    let layout = unsafe { raw.create_descriptor_set_layout(&info, None) }.unwrap();
    let needs = SetNeeds {
        counts: vec![(vk::DescriptorType::STORAGE_BUFFER.as_raw(), 1)],
        flags: vk::DescriptorPoolCreateFlags::empty(),
    };
    let mut own = DescriptorArena::default();
    let lent = Arc::new(Mutex::new(DescriptorArena::default()));
    own.lend(Some(Arc::clone(&lent)));
    let mut opened = None;
    for run in 0..20 {
        for _ in 0..40 {
            own.allocate(&device, layout, &needs).unwrap();
        }
        assert_eq!(
            own.pools(),
            0,
            "run {run}: a set from the arena's own pools"
        );
        let made = device.work_counts().arena_pools_made;
        assert_eq!(*opened.get_or_insert(made), made, "run {run} opened a pool");
        // Inside a section, as the run again's settle resets it: there the device's functions
        // are the exit guard's table alone (`device/functions.rs`).
        let section = crate::exit_guard::enter().expect("a section");
        lock(&lent).reset().unwrap();
        drop(section);
    }
    own.lend(None);
    assert!(lock(&lent).pools() > 0);
    drop(lent);
    // SAFETY: no set of the layout is in use, and its arenas' pools are gone.
    unsafe { raw.destroy_descriptor_set_layout(layout, None) };
}
