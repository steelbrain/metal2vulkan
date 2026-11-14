//! Executions of compute commands a kernel writes into a buffer that inherits the encoder's
//! pipeline state and buffers (the runtime's `gpu_execute.rs`, **Inherited buffers**;
//! `bind::ReplayExecution::parent_buffers`), on the device the cfg selects, under the validation
//! layer. `icb_inherited_dispatches`' `cid_encode` writes command `tid` as a dispatch of
//! `1 + tid % 4` threadgroups of 64 threads, or a reset where `tid % 5` is 4, binding nothing; the
//! encoder's state `cid_count` adds `input[gid] * multiplier + bias` to `sums[0]` and 1 to
//! `sums[1 + gid]` by relaxed atomics, and does not observe its threadgroup partition.
//!
//! The execution runs on the GPU alone, its flat variant's record kept and the host reading no
//! record, and once the states' pipelines are made recording no step, with the GPU form's
//! dispatches and with its generated commands where the device has them, and
//! computes what Metal computes: every command that is not reset runs its grid once, with the
//! encoder's buffers as they stand at the call: `input` in a heap's buffer the encoder alone names
//! (no `useResource:`, no `useHeap:`) at a non-zero offset, which a blit of the same command
//! buffer writes before the execution; `sums` at a non-zero offset; `multiplier` bound at one
//! offset and moved by `setBufferOffset:` before the call; and `bias` inline bytes
//! (`setBytes:`).
//!
//! The host step runs an execution whose range a buffer holds, and orders its commands by their
//! records' barriers alone, in a serial encoder as in a concurrent one (the runtime's
//! `indirect_execute.rs`, **Order**): Metal's serial encoder adds a barrier before and after the
//! range of commands alone.

use super::degrade_notices::{counts, on_device_with};
use super::dispatch_executions::{barriers, own_barriers, ran_flat};
use super::indirect_host_steps::{complete, HOST_STEPS};
use super::render_executions::icb;
use super::render_inherited_buffers::reasons;
use super::{at_plan, buffer_with};
use crate::air_library::AirLibrary;
use crate::candidate::runtime_case::read_buffer;
use metal2vulkan::metal::{DispatchType, IndirectCommandType, ResourceUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    CommandQueue, ComputePipelineState, Device, DeviceOptions, HeapDescriptor, HeapType,
    IndirectCommandBufferDescriptor, Library, PipelineOptions, ResourceOptions, Size,
};

/// The buffer's commands.
const COMMANDS: u64 = 16;
/// The threads of a command's threadgroup, and the most threads a command runs.
const LOCAL: u64 = 64;
const MOST: u64 = 4 * LOCAL;

/// Where the encoder binds what the commands inherit: `input` in the heap's buffer, `sums` (slot
/// 2 of 256-byte slots, 64 bytes in), and `multiplier` once its offset moved (first bound at 0).
const INPUT_AT: u64 = 256;
const SUMS_AT: u64 = 2 * 256 + 64;
const MULTIPLIER_AT: u64 = 256;
/// The multiplier, and the inline bytes' bias.
const MULTIPLIER: u32 = 3;
const BIAS: u32 = 5;

/// `input[g]`.
fn input(g: u32) -> u32 {
    7 * g + 1
}

fn words(values: impl IntoIterator<Item = u32>) -> Vec<u8> {
    values.into_iter().flat_map(u32::to_le_bytes).collect()
}

/// `bytes` after `at` filler bytes.
fn at(at: u64, bytes: &[u8]) -> Vec<u8> {
    let mut out = vec![0xEE; at as usize];
    out.extend_from_slice(bytes);
    out
}

/// The threads command `tid` runs: none where it is reset.
fn threads(tid: u64) -> u64 {
    match tid % 5 {
        4 => 0,
        _ => LOCAL * (1 + tid % 4),
    }
}

/// What Metal computes into `sums`: the sum, then each thread's count of the commands whose grid
/// holds it.
fn expected() -> Vec<u32> {
    let mut sums = vec![0u32; 1 + MOST as usize];
    for tid in 0..COMMANDS {
        for g in 0..threads(tid) as u32 {
            sums[0] = sums[0].wrapping_add(input(g).wrapping_mul(MULTIPLIER) + BIAS);
            sums[1 + g as usize] += 1;
        }
    }
    sums
}

/// The state `name` of the fixture's library, made with `supportIndirectCommandBuffers`.
fn state(device: &Device, library: &Library, name: &str) -> ComputePipelineState {
    let function = library.new_function_with_name(name).unwrap();
    let mut descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    descriptor.support_indirect_command_buffers = true;
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// How [`run`] executes the commands: in an encoder of `dispatch_type`, the range given in the
/// call, or, where `ranged`, read from a buffer (`execute_commands_in_buffer_indirect`), which the
/// host step runs.
#[derive(Clone, Copy, Debug)]
struct Encoding {
    dispatch_type: DispatchType,
    ranged: bool,
}

/// A serial encoder's execution of the range in the call.
const GIVEN: Encoding = Encoding {
    dispatch_type: DispatchType::Serial,
    ranged: false,
};

/// `writer` (`cid_encode`) writes [`COMMANDS`] commands into a buffer that inherits the encoder's
/// state and buffers, then an encoder executes them as `encoding` says with `count` (`cid_count`)
/// and its buffers, in a command buffer of `queue`: what `sums` holds, the records the host read
/// across the run, whether the command buffer ran a flat variant (the GPU alone's execution
/// record), the host steps recorded, and the pipeline barriers recorded. A test's runs share its
/// one queue, whose kept barriers its first run records (`dispatch_executions::own_barriers`): a
/// command queue of each run's own, at the device's queue plan, may take a `VkQueue` of its own,
/// whose kept barriers each run would record again.
fn run(
    device: &Device,
    queue: &CommandQueue,
    (writer, count): (&ComputePipelineState, &ComputePipelineState),
    encoding: Encoding,
) -> (Vec<u32>, (u64, bool, u64, u64)) {
    let mut descriptor = IndirectCommandBufferDescriptor::new();
    descriptor.command_types = IndirectCommandType::CONCURRENT_DISPATCH;
    descriptor.inherit_pipeline_state = true;
    descriptor.inherit_buffers = true;
    let icb = icb(device, &descriptor, COMMANDS);
    let encoded = buffer_with(device, &icb.gpu_resource_id().0.to_le_bytes()).unwrap();
    // The input in a heap's buffer, filler until the blit copies it in.
    let heap = device
        .new_heap(&HeapDescriptor::new(
            HeapType::Automatic,
            1 << 16,
            ResourceOptions::shared(),
        ))
        .unwrap();
    let staged = at(INPUT_AT, &words((0..MOST as u32).map(input)));
    let input = heap
        .new_buffer(staged.len() as u64, ResourceOptions::shared())
        .unwrap();
    let contents = input.contents().unwrap();
    // SAFETY: the buffer is `Shared`, its `staged.len()` bytes are the CPU's, and no command
    // reads them yet.
    unsafe { std::ptr::write_bytes(contents.as_ptr(), 0xEE, staged.len()) };
    let staged = buffer_with(device, &staged).unwrap();
    let sums_bytes = 4 * (1 + MOST);
    let sums = buffer_with(device, &at(SUMS_AT, &vec![0; sums_bytes as usize])).unwrap();
    let multiplier = buffer_with(device, &at(MULTIPLIER_AT, &words([MULTIPLIER]))).unwrap();
    // The execution range (location 0, length `COMMANDS`) a ranged execution reads.
    let range = buffer_with(device, &words([0, COMMANDS as u32])).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let before = device.work_counts();
    let barriers_before = barriers(device);
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    let length = input.length() - INPUT_AT;
    blit.copy_from_buffer_to_buffer(&staged, INPUT_AT, &input, INPUT_AT, length)
        .unwrap();
    blit.end_encoding().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder
        .use_resource(icb.as_resource(), ResourceUsage::WRITE)
        .unwrap();
    encoder.set_compute_pipeline_state(writer).unwrap();
    encoder.set_buffer(Some(&encoded), 0, 0).unwrap();
    let grid = Size::new(COMMANDS, 1, 1);
    encoder.dispatch_threads(grid, grid).unwrap();
    encoder.end_encoding().unwrap();
    let mut encoder = command_buffer
        .compute_command_encoder_with_dispatch_type(encoding.dispatch_type)
        .unwrap();
    encoder.set_compute_pipeline_state(count).unwrap();
    encoder.set_buffer(Some(&input), INPUT_AT, 0).unwrap();
    encoder.set_buffer(Some(&sums), SUMS_AT, 1).unwrap();
    encoder.set_buffer(Some(&multiplier), 0, 2).unwrap();
    encoder.set_buffer_offset(MULTIPLIER_AT, 2).unwrap();
    encoder.set_bytes(&BIAS.to_le_bytes(), 3).unwrap();
    match encoding.ranged {
        false => encoder.execute_commands_in_buffer(&icb, 0..COMMANDS),
        true => encoder.execute_commands_in_buffer_indirect(&icb, &range, 0),
    }
    .unwrap();
    encoder.end_encoding().unwrap();
    complete(&command_buffer);
    let after = device.work_counts();
    let counted = (
        after.indirect_records_read - before.indirect_records_read,
        ran_flat(&command_buffer),
        after.host_steps_recorded - before.host_steps_recorded,
        barriers(device) - barriers_before,
    );
    let bytes = read_buffer(&sums, SUMS_AT, sums_bytes).unwrap();
    let words = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .collect();
    (words, counted)
}

/// A kernel's commands that inherit the encoder's state and buffers run on the GPU alone, a flat
/// variant's run and no record read by the host, and, once the states' pipelines are made (a
/// second run: the first may wait at a step for what the states' creations left making,
/// `WorkCounts::creations_left_making`), no step recorded, and compute what Metal computes with
/// the encoder's
/// buffers as they stood at the call (the module docs): on the device's own path, and with the
/// device's generated commands withheld, by the indirect dispatches. Nothing is announced.
#[test]
fn inherited_buffers_dispatch_on_the_gpu_alone() {
    for withheld in [false, true] {
        let adjust = |options: &mut DeviceOptions| {
            at_plan(options);
            options.withhold_generated_commands = withheld;
        };
        on_device_with(adjust, |device, messages| {
            let bitcode = llvm_assemble(&engine_dir().join("icb_inherited_dispatches.ll")).unwrap();
            let library = device.new_library_with_air(&bitcode).unwrap();
            let states = (
                &state(device, &library, "cid_encode"),
                &state(device, &library, "cid_count"),
            );
            let queue = device.new_command_queue().unwrap();
            let (cold, (read, flat, _, _)) = run(device, &queue, states, GIVEN);
            let (sums, (read_warm, flat_warm, steps, _)) = run(device, &queue, states, GIVEN);
            let counted = (read_warm, flat_warm, steps);
            let notices = messages.take_notices();
            assert_eq!(
                (read, flat),
                (0, true),
                "withheld {withheld}: the first run on the GPU alone; announced {:?}",
                reasons(&notices)
            );
            assert_eq!(
                counted,
                (0, true, 0),
                "withheld {withheld}: on the GPU alone; announced {:?}",
                reasons(&notices)
            );
            assert_eq!(cold, expected(), "withheld {withheld}: the first run");
            assert_eq!(sums, expected(), "withheld {withheld}");
            let notices = counts(&notices);
            assert!(!notices.contains_key(&HOST_STEPS[0]), "{notices:?}");
        });
    }
}

/// The host step orders the commands by their records' barriers alone, in a serial encoder as in a
/// concurrent one (the module docs): an execution whose range a buffer holds, of commands that
/// each add to the inherited `sums` (their uses meet, as a serial encoder's own dispatches' would
/// need a barrier between every two), runs in the host step and computes what Metal computes in
/// both, and, once the states' pipelines and the queue's kept barriers are made (a first run of
/// each), records as many barriers in the serial encoder as in the concurrent one: none between
/// its commands, none of whose records sets one.
#[test]
fn a_host_step_orders_the_commands_by_their_barriers_alone() {
    on_device_with(at_plan, |device, messages| {
        let bitcode = llvm_assemble(&engine_dir().join("icb_inherited_dispatches.ll")).unwrap();
        let library = device.new_library_with_air(&bitcode).unwrap();
        let states = (
            &state(device, &library, "cid_encode"),
            &state(device, &library, "cid_count"),
        );
        let queue = device.new_command_queue().unwrap();
        let mut taken = Vec::new();
        for dispatch_type in [DispatchType::Serial, DispatchType::Concurrent] {
            let encoding = Encoding {
                dispatch_type,
                ranged: true,
            };
            let (cold, _) = run(device, &queue, states, encoding);
            assert_eq!(cold, expected(), "{dispatch_type:?}: the first run");
            let at = format!("{dispatch_type:?}");
            taken.push(own_barriers(device, &at, || {
                let (sums, (_, flat, steps, barriers)) = run(device, &queue, states, encoding);
                assert_eq!(sums, expected(), "{at}");
                assert_eq!((flat, steps), (false, 1), "{at}: in the host step");
                barriers
            }));
        }
        assert_eq!(
            taken[0], taken[1],
            "no barrier between the commands in the serial encoder, as in the concurrent one: \
             {taken:?}"
        );
        let notices = counts(&messages.take_notices());
        assert!(notices.contains_key(&HOST_STEPS[0]), "{notices:?}");
    });
}
