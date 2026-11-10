//! Executions of render commands a kernel writes, in their GPU form (the runtime's
//! `render/executions/gpu.rs`), on the device the cfg selects, under the validation layer. The
//! commands draw `render_executions`' template (rasterization disabled: vertex `vid` of instance
//! `iid` writes `params[0] * (vid + 1) + params[1] + 1000 * iid` to word `16 * iid + vid` of
//! `out`) under the encoder's state, which the buffer inherits. The point writer
//! (`icb_render_replay_encode_draw_only`) draws 4 points from vertex `4 * tid` in command `tid`;
//! the triangle writer (`kernel_icb_encode_render_commands`) binds `params` as vertex buffer 0
//! and fragment buffer 1 and draws one triangle from vertex `3 * tid`; the CPU writes the rest.
//! They draw as the host step draws them, as Metal's:
//!
//! - on the GPU, the host reading no record, each with its own vertex start, instance count and
//!   base instance; a reset command draws nothing;
//! - a check that looks for every kind keeps the host step, which calls it and draws the same;
//! - a check that looks for what the records cannot hold (points, where only triangles are
//!   written) is never called, and the commands draw on the GPU; where the CPU writes a point
//!   draw after the commit, before the pass's batch is ready, the host step records the pass
//!   instead and calls the check;
//! - a check of the CPU's records, which `commit` records, is called there with each command,
//!   before `commit` returns, where the records may hold draws of points and it looks for them;
//!   never where they hold triangles alone, or where it looks for nothing;
//! - the CPU's records, which `commit` records, written again after the commit, before the pass's
//!   batch is submitted (it waits for an event), draw as written then, the host step recording
//!   the pass again and announcing nothing;
//! - two executions of one buffer in one pass draw each its own range;
//! - a buffer that may hold a barrier takes the host step, announced once;
//! - an address outside the resident resources misses, counted, while its neighbours draw;
//! - a record past the descriptor's bind count sets the sticky encoding fault: nothing draws, and
//!   the execution report says why.

use super::degrade_notices::on_device_with;
use super::degrade_notices::{assert_made_once, on_device};
use super::indirect_host_steps::{self, complete, HOST_STEPS};
use super::render_executions::{expected, forced_with, icb, pass, Drawn, Setup, WINDOW};
use super::{at_plan, buffer_with, read_buffer, Messages, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::bind::ReplayFaults;
use metal2vulkan::metal::{IndirectCommandType, PrimitiveType, ResourceUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::{fixtures_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, CommandBufferStatus, ComputePipelineState, Degradation, Device,
    EventRef, IndirectCommandBuffer, IndirectCommandBufferDescriptor, IndirectCommandCheckNeeds,
    PipelineOptions, Size,
};
use std::sync::{Arc, Mutex};

/// Draw commands of two vertex buffers and `fragment` fragment buffers that inherit the encoder's
/// pipeline state.
fn inheriting(fragment: u32) -> IndirectCommandBufferDescriptor {
    let mut descriptor = IndirectCommandBufferDescriptor::new();
    descriptor.command_types = IndirectCommandType::DRAW;
    descriptor.inherit_pipeline_state = true;
    descriptor.max_vertex_buffer_bind_count = 2;
    descriptor.max_fragment_buffer_bind_count = fragment;
    descriptor
}

/// Bind window `window` of `setup`'s `params` and `out` as command `index`'s vertex buffers 0
/// and 1.
fn buffers(setup: &Setup, icb: &IndirectCommandBuffer, index: u64, window: usize) {
    let command = icb.indirect_render_command_at(index).unwrap();
    let offset = (window * WINDOW * 4) as u64;
    command
        .set_vertex_buffer(Some(&setup.params), offset, 0)
        .unwrap();
    command
        .set_vertex_buffer(Some(&setup.out), offset, 1)
        .unwrap();
}

/// How an execution is encoded.
enum Call {
    Plain,
    /// With a caller's check (`execute_commands_in_buffer_checked`) that looks for what the
    /// needs say, noting each command it is called with.
    Checked(IndirectCommandCheckNeeds, Arc<Mutex<Vec<u64>>>),
}

impl Call {
    /// A check that looks for `needs`, and the commands it will be called with.
    fn checked(needs: IndirectCommandCheckNeeds) -> (Call, Arc<Mutex<Vec<u64>>>) {
        let seen = Arc::default();
        (Call::Checked(needs, Arc::clone(&seen)), seen)
    }
}

/// What a test runs: the template's `setup`, a writer's state, and the buffer its third argument
/// is.
struct Writing<'a> {
    setup: &'a Setup,
    writer: &'a ComputePipelineState,
    out: &'a Buffer,
}

/// Encode, in one command buffer of a new queue, what [`encode_into`] does.
fn encode(
    device: &Device,
    writing: &Writing<'_>,
    (icb, threads): (&IndirectCommandBuffer, u64),
    (ranges, call): (&[(u64, u64)], &Call),
) -> CommandBuffer {
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    encode_into(&command_buffer, writing, (icb, threads), (ranges, call));
    command_buffer
}

/// Encode into `command_buffer`: the writer over `threads` threads writing `icb` (its arguments
/// the buffer's ID, `params`, and `writing.out`), then one render encoder that sets the template's
/// state, makes `params` and `out` resident and executes `ranges` of `icb` (each its start and
/// end) as `call` says.
fn encode_into(
    command_buffer: &CommandBuffer,
    writing: &Writing<'_>,
    (icb, threads): (&IndirectCommandBuffer, u64),
    (ranges, call): (&[(u64, u64)], &Call),
) {
    let setup = writing.setup;
    let device = command_buffer.device();
    let arguments = buffer_with(device, &icb.gpu_resource_id().0.to_le_bytes()).unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(writing.writer).unwrap();
    encoder.set_buffer(Some(&arguments), 0, 0).unwrap();
    encoder.set_buffer(Some(&setup.params), 0, 1).unwrap();
    encoder.set_buffer(Some(writing.out), 0, 2).unwrap();
    encoder
        .use_resource(icb.as_resource(), ResourceUsage::WRITE)
        .unwrap();
    let grid = Size::new(threads, 1, 1);
    encoder.dispatch_threads(grid, grid).unwrap();
    encoder.end_encoding().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass()).unwrap();
    encoder.set_render_pipeline_state(&setup.state).unwrap();
    setup.resident(&mut encoder);
    for &(start, end) in ranges {
        match call {
            Call::Plain => encoder.execute_commands_in_buffer(icb, start..end),
            Call::Checked(needs, seen) => {
                let seen = Arc::clone(seen);
                encoder.execute_commands_in_buffer_checked(
                    icb,
                    start..end,
                    Box::new(move |index, _| seen.lock().unwrap().push(index)),
                    *needs,
                )
            }
        }
        .unwrap();
    }
    encoder.end_encoding().unwrap();
}

/// The point writer's state.
fn points(device: &Device) -> ComputePipelineState {
    indirect_host_steps::state(device, "icb_render_replay_encode_draw_only", false)
}

/// The triangle writer's state.
fn triangles(device: &Device) -> ComputePipelineState {
    let bitcode =
        llvm_assemble(&fixtures_dir().join("kernel_icb_encode_render_commands.ll")).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = library
        .new_function_with_name("kernel_icb_encode_render_commands")
        .unwrap();
    let descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// A buffer of 4 commands, each over its own window: the point writer's draws in commands 0 and
/// 1 (dispatched over 2 threads), the CPU's draw of 2 vertices from 1, 2 instances from 1, in
/// command 2, and command 3 reset after a draw.
fn written(device: &Device, setup: &Setup) -> IndirectCommandBuffer {
    let icb = icb(device, &inheriting(1), 4);
    for index in 0..4 {
        buffers(setup, &icb, index, index as usize);
    }
    let command = icb.indirect_render_command_at(2).unwrap();
    command
        .draw_primitives(PrimitiveType::Point, 1, 2, 2, 1)
        .unwrap();
    let reset = icb.indirect_render_command_at(3).unwrap();
    reset
        .draw_primitives(PrimitiveType::Point, 0, 4, 1, 0)
        .unwrap();
    reset.reset().unwrap();
    icb
}

/// What [`written`]'s commands draw.
fn written_draws() -> Vec<Drawn> {
    vec![
        drawn(0, 0, 4),
        drawn(1, 4, 4),
        Drawn {
            window: 2,
            start: 1,
            count: 2,
            instances: 2,
            base: 1,
        },
    ]
}

/// A draw of `count` vertices from `start` in window `window`, one instance.
fn drawn(window: usize, start: usize, count: usize) -> Drawn {
    Drawn {
        window,
        start,
        count,
        instances: 1,
        base: 0,
    }
}

/// A buffer of 4 commands for the triangle writer, each of whose vertex buffer 1 is window 0 of
/// `out` (the writer binds vertex buffer 0, window 0 of `params`), with room for its fragment
/// buffer 1.
fn for_triangles(device: &Device, setup: &Setup) -> IndirectCommandBuffer {
    let icb = icb(device, &inheriting(2), 4);
    for index in 0..4 {
        buffers(setup, &icb, index, 0);
    }
    icb
}

/// A kernel's draw of a primitive type past Metal's five draws points, as Metal's encoding
/// writes it (the M3 Max, macOS 27.0.1: a kernel's draws of values 5 to 12 and past drew what
/// value 0 draws, direct and indexed): the owned `icb_render_typed_encode`'s draws of quads (7),
/// of type 9 and of 300 each draw all six vertices, direct (window 0) and over six indices that
/// permute them (window 1), where quads would draw four and type 9 fail the command buffer.
#[test]
fn a_kernel_s_draw_of_a_type_past_metal_s_five_draws_points() {
    on_device_with(at_plan, |device, _| {
        let writer = indirect_host_steps::state(device, "icb_render_typed_encode", false);
        let mut descriptor = inheriting(1);
        descriptor.command_types =
            IndirectCommandType::DRAW.union(IndirectCommandType::DRAW_INDEXED);
        for value in [7u32, 9, 300] {
            let setup = Setup::new(device);
            let icb = icb(device, &descriptor, 2);
            for index in 0..2 {
                buffers(&setup, &icb, index, index as usize);
            }
            let indices: [u16; 6] = [5, 3, 0, 1, 4, 2];
            let mut typed: Vec<u8> = [value, 0, 0, 0]
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect();
            typed.extend(indices.iter().flat_map(|h| h.to_le_bytes()));
            let typed = buffer_with(device, &typed).unwrap();
            let writing = Writing {
                setup: &setup,
                writer: &writer,
                out: &typed,
            };
            let command_buffer = encode(device, &writing, (&icb, 2), (&[(0, 2)], &Call::Plain));
            complete(&command_buffer);
            assert_eq!(
                setup.output(),
                expected(&[drawn(0, 0, 6), drawn(1, 0, 6)]),
                "{value} on {}",
                device.name()
            );
        }
    });
}

/// The reason the host-step notice delivered so far gives (the rule that kept the pass from the
/// GPU alone), the notices put back for [`assert_made_once`]; empty where none was delivered.
fn host_step_reason(messages: &Messages) -> String {
    let notices = messages.take_notices();
    let reason = (notices.iter())
        .filter_map(|message| message.degradation.as_ref())
        .find(|notice| notice.degradation == Degradation::HostReplayedIndirectCommands)
        .map(|notice| notice.reason.clone());
    messages.restore_notices(notices);
    reason.unwrap_or_default()
}

/// The commands' GPU form and host reads across `run`: commands recorded in the GPU form, and
/// records the host read.
fn counted(device: &Device, run: impl FnOnce()) -> (u64, u64) {
    let before = device.work_counts();
    run();
    let after = device.work_counts();
    (
        after.gpu_form_commands - before.gpu_form_commands,
        after.indirect_records_read - before.indirect_records_read,
    )
}

/// Commands a kernel and the CPU write draw on the GPU, the host reading no record, each with its
/// own vertex start, instance count and base instance (`vertex_id`, `instance_id`); a reset
/// command draws nothing. The template's vertex function writes memory, so they draw one draw
/// each, not merged. No host step is announced.
#[test]
fn written_commands_draw_on_the_gpu_with_their_own_arguments() {
    on_device(|device, messages| {
        let setup = Setup::new(device);
        let writer = points(device);
        let writing = Writing {
            setup: &setup,
            writer: &writer,
            out: &setup.out,
        };
        let icb = written(device, &setup);
        let merged = device.work_counts().merged_draw_commands;
        let (gpu, read) = counted(device, || {
            complete(&encode(
                device,
                &writing,
                (&icb, 2),
                (&[(0, 4)], &Call::Plain),
            ));
        });
        assert_eq!(setup.output(), expected(&written_draws()));
        assert_eq!((gpu, read), (4, 0), "drawn in the GPU form");
        assert_eq!(
            device.work_counts().merged_draw_commands,
            merged,
            "one draw each"
        );
        assert_made_once(messages, &[], &forced_with(device, &[]));
    });
}

/// A check that looks for every kind keeps the host step, which reads every record and calls it
/// with each command before any runs; the same commands in their GPU form draw what it drew.
#[test]
fn the_gpu_form_draws_what_the_host_step_draws() {
    on_device_with(at_plan, |device, messages| {
        let writer = points(device);
        let (checked, seen) = Call::checked(IndirectCommandCheckNeeds::ALL);
        let mut outputs = Vec::new();
        for call in [checked, Call::Plain] {
            let setup = Setup::new(device);
            let writing = Writing {
                setup: &setup,
                writer: &writer,
                out: &setup.out,
            };
            let icb = written(device, &setup);
            let (gpu, read) = counted(device, || {
                complete(&encode(device, &writing, (&icb, 2), (&[(0, 4)], &call)));
            });
            match call {
                Call::Checked(..) => assert!(gpu == 0 && read >= 4, "{gpu} {read}"),
                Call::Plain => assert_eq!((gpu, read), (4, 0)),
            }
            outputs.push(setup.output());
        }
        assert_eq!(*seen.lock().unwrap(), [0, 1, 2, 3], "each command checked");
        assert_eq!(outputs[0], expected(&written_draws()), "the host step");
        assert_eq!(outputs[1], outputs[0], "the GPU form");
        let reason = host_step_reason(messages);
        assert!(
            reason.contains("check looks for draws of points, which"),
            "{reason}"
        );
        assert_made_once(messages, &HOST_STEPS, &forced_with(device, &[]));
    });
}

/// A check that looks for draws of points and for states that may encode, of commands the
/// triangle writer writes under the template's state, which encodes none, has nothing to see: it
/// is never called, and the commands draw on the GPU, the host reading no record.
#[test]
fn a_check_with_nothing_to_see_is_never_called() {
    on_device(|device, messages| {
        let setup = Setup::new(device);
        let writer = triangles(device);
        let scratch = buffer_with(device, &[0; 64]).unwrap();
        let writing = Writing {
            setup: &setup,
            writer: &writer,
            out: &scratch,
        };
        let icb = for_triangles(device, &setup);
        let (call, seen) = Call::checked(IndirectCommandCheckNeeds::ALL);
        let (gpu, read) = counted(device, || {
            complete(&encode(device, &writing, (&icb, 4), (&[(0, 4)], &call)));
        });
        assert_eq!((gpu, read), (4, 0), "drawn in the GPU form");
        assert!(seen.lock().unwrap().is_empty(), "the check was called");
        assert_eq!(setup.output(), expected(&[drawn(0, 0, 12)]));
        assert_made_once(messages, &[], &forced_with(device, &[]));
    });
}

/// Where the CPU writes a draw of points after the commit, before the pass's batch is ready (its
/// command buffer waits for an event the test signals after the write), the check, which looks
/// for points, has something to see: the host step records the pass instead of the GPU form the
/// commit recorded, calls the check with each command, and draws the point draw too.
#[test]
fn facts_that_grow_before_the_batch_is_ready_take_the_host_step() {
    on_device(|device, messages| {
        let setup = Setup::new(device);
        let writer = triangles(device);
        let scratch = buffer_with(device, &[0; 64]).unwrap();
        let writing = Writing {
            setup: &setup,
            writer: &writer,
            out: &scratch,
        };
        let icb = for_triangles(device, &setup);
        let (call, seen) = Call::checked(IndirectCommandCheckNeeds::new(true, false));
        let event = device.new_shared_event().unwrap();
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        command_buffer
            .encode_wait_for_event(EventRef::from(&event), 1)
            .unwrap();
        encode_into(&command_buffer, &writing, (&icb, 2), (&[(0, 4)], &call));
        let (gpu, _) = counted(device, || command_buffer.commit().unwrap());
        assert_eq!(gpu, 4, "the commit recorded the GPU form");
        buffers(&setup, &icb, 2, 2);
        icb.indirect_render_command_at(2)
            .unwrap()
            .draw_primitives(PrimitiveType::Point, 0, 4, 1, 0)
            .unwrap();
        let (_, read) = counted(device, || {
            event.set_signaled_value(1);
            indirect_host_steps::wait(&command_buffer);
        });
        assert!(read >= 4, "the host step read the records: {read}");
        assert_eq!(*seen.lock().unwrap(), [0, 1, 2, 3], "each command checked");
        assert_eq!(setup.output(), expected(&[drawn(0, 0, 6), drawn(2, 0, 4)]));
        let reason = host_step_reason(messages);
        assert!(reason.contains("grew after the commit"), "{reason}");
        assert_made_once(messages, &HOST_STEPS, &forced_with(device, &[]));
    });
}

/// A check of commands the CPU wrote, in a buffer no encoder made resident for writing, which
/// `commit` records from the CPU's records: `commit` calls it with each command of the range
/// before it returns, where the CPU drew points and the check looks for them; where the CPU drew
/// triangles alone, or the check looks for nothing, never. The commands draw the same each time,
/// and no host step runs.
#[test]
fn commit_checks_the_cpu_s_records_where_the_check_may_see_something() {
    on_device_with(at_plan, |device, messages| {
        let points = IndirectCommandCheckNeeds::new(true, false);
        let nothing = IndirectCommandCheckNeeds::new(false, false);
        for (primitive_type, needs) in [
            (PrimitiveType::Point, points),
            (PrimitiveType::Triangle, points),
            (PrimitiveType::Point, nothing),
        ] {
            let setup = Setup::new(device);
            let icb = icb(device, &inheriting(1), 4);
            for index in 0..4 {
                buffers(&setup, &icb, index, index as usize);
                icb.indirect_render_command_at(index)
                    .unwrap()
                    .draw_primitives(primitive_type, 0, 3, 1, 0)
                    .unwrap();
            }
            let seen: Arc<Mutex<Vec<u64>>> = Arc::default();
            let log = Arc::clone(&seen);
            let queue = device.new_command_queue().unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.render_command_encoder(&pass()).unwrap();
            encoder.set_render_pipeline_state(&setup.state).unwrap();
            setup.resident(&mut encoder);
            encoder
                .execute_commands_in_buffer_checked(
                    &icb,
                    0..4,
                    Box::new(move |index, _| log.lock().unwrap().push(index)),
                    needs,
                )
                .unwrap();
            encoder.end_encoding().unwrap();
            let (_, read) = counted(device, || command_buffer.commit().unwrap());
            let at_commit = seen.lock().unwrap().clone();
            indirect_host_steps::wait(&command_buffer);
            let at = format!("{primitive_type:?}, {needs:?}");
            assert!(!icb.gpu_written(), "{at}");
            assert_eq!(read, 4, "{at}: the commit read each record");
            let every: Vec<u64> = match primitive_type == PrimitiveType::Point && needs == points {
                true => vec![0, 1, 2, 3],
                false => Vec::new(),
            };
            assert_eq!(at_commit, every, "{at}: checked in `commit`");
            assert_eq!(*seen.lock().unwrap(), every, "{at}: and only there");
            let draws: Vec<Drawn> = (0..4).map(|window| drawn(window, 0, 3)).collect();
            assert_eq!(setup.output(), expected(&draws), "{at}");
        }
        assert_made_once(messages, &[], &forced_with(device, &[]));
    });
}

/// Commands the CPU wrote, in a buffer no encoder made resident for writing, draw as their records
/// stand when the pass's batch is submitted, as Metal's execution reads them as it runs: where the
/// CPU draws 2 more vertices in command 1 after the commit, before the batch is submitted (it waits
/// for an event the test signals after the write), the host step records the pass again from the
/// records then (each record read again) and command 1 draws 5 vertices; without the write, the
/// commit's recording draws 3. Nothing is announced.
#[test]
fn the_cpu_s_writes_after_the_commit_draw_where_the_batch_waits() {
    on_device_with(at_plan, |device, messages| {
        for write in [false, true] {
            let setup = Setup::new(device);
            let icb = icb(device, &inheriting(1), 4);
            for index in 0..4 {
                buffers(&setup, &icb, index, index as usize);
                icb.indirect_render_command_at(index)
                    .unwrap()
                    .draw_primitives(PrimitiveType::Point, 0, 3, 1, 0)
                    .unwrap();
            }
            let event = device.new_shared_event().unwrap();
            let queue = device.new_command_queue().unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            command_buffer
                .encode_wait_for_event(EventRef::from(&event), 1)
                .unwrap();
            let mut encoder = command_buffer.render_command_encoder(&pass()).unwrap();
            encoder.set_render_pipeline_state(&setup.state).unwrap();
            setup.resident(&mut encoder);
            encoder.execute_commands_in_buffer(&icb, 0..4).unwrap();
            encoder.end_encoding().unwrap();
            let (_, read) = counted(device, || {
                command_buffer.commit().unwrap();
                if write {
                    icb.indirect_render_command_at(1)
                        .unwrap()
                        .draw_primitives(PrimitiveType::Point, 0, 5, 1, 0)
                        .unwrap();
                }
                event.set_signaled_value(1);
                indirect_host_steps::wait(&command_buffer);
            });
            assert!(!icb.gpu_written(), "write {write}");
            let reads = if write { 8 } else { 4 };
            assert_eq!(read, reads, "write {write}: the records read");
            let count = |window: usize| match (write, window) {
                (true, 1) => 5,
                _ => 3,
            };
            let draws: Vec<Drawn> = (0..4)
                .map(|window| drawn(window, 0, count(window)))
                .collect();
            assert_eq!(setup.output(), expected(&draws), "write {write}");
        }
        assert_made_once(messages, &[], &forced_with(device, &[]));
    });
}

/// Two executions of one buffer in one pass, of ranges that overlap, draw each its own range in
/// the GPU form.
#[test]
fn two_ranges_of_one_buffer_draw_on_the_gpu() {
    on_device(|device, _| {
        let setup = Setup::new(device);
        let writer = points(device);
        let writing = Writing {
            setup: &setup,
            writer: &writer,
            out: &setup.out,
        };
        let icb = written(device, &setup);
        let (gpu, read) = counted(device, || {
            let ranges = [(0, 2), (1, 4)];
            complete(&encode(
                device,
                &writing,
                (&icb, 2),
                (&ranges, &Call::Plain),
            ));
        });
        assert_eq!(setup.output(), expected(&written_draws()));
        assert_eq!((gpu, read), (5, 0));
    });
}

/// A buffer the CPU set a barrier in, even one cleared since (its records may hold one), takes
/// the host step, announced once, and draws as before.
#[test]
fn a_buffer_that_may_hold_a_barrier_takes_the_host_step() {
    on_device(|device, messages| {
        let setup = Setup::new(device);
        let writer = points(device);
        let writing = Writing {
            setup: &setup,
            writer: &writer,
            out: &setup.out,
        };
        let icb = written(device, &setup);
        let command = icb.indirect_render_command_at(2).unwrap();
        command.set_barrier().unwrap();
        command.clear_barrier().unwrap();
        let (gpu, read) = counted(device, || {
            complete(&encode(
                device,
                &writing,
                (&icb, 2),
                (&[(0, 4)], &Call::Plain),
            ));
        });
        assert_eq!(setup.output(), expected(&written_draws()));
        assert_eq!(gpu, 0);
        assert!(read >= 4, "{read}");
        let reason = host_step_reason(messages);
        assert!(reason.contains("records may hold barriers"), "{reason}");
        assert_made_once(messages, &HOST_STEPS, &forced_with(device, &[]));
    });
}

/// A command's buffers in a heap no encoder of the command buffer used miss: unbound, counted in
/// the execution report, as the host step counts them; its neighbours draw.
#[test]
fn addresses_outside_the_resident_resources_miss_on_the_gpu() {
    on_device(|device, _| {
        let setup = Setup::new(device);
        let (away, _heap) = Setup::in_heap(device);
        let writer = points(device);
        let writing = Writing {
            setup: &setup,
            writer: &writer,
            out: &setup.out,
        };
        let icb = icb(device, &inheriting(1), 2);
        buffers(&setup, &icb, 0, 0);
        buffers(&away, &icb, 1, 1);
        let command_buffer = encode(device, &writing, (&icb, 2), (&[(0, 2)], &Call::Plain));
        let (gpu, read) = counted(device, || {
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
        });
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        assert_eq!((gpu, read), (2, 0));
        assert_eq!(setup.output(), expected(&[drawn(0, 0, 4)]));
        assert_eq!(away.output(), expected(&[]));
        let report = command_buffer.execution_report().unwrap();
        assert!(report.resident_miss);
        assert_eq!(report.resident_misses, 2);
    });
}

/// A record the triangle writer encodes with a fragment buffer past the descriptor's count sets
/// the header's sticky encoding fault: in the GPU form no command draws, and the execution report
/// says why.
#[test]
fn an_encoding_fault_draws_nothing_on_the_gpu_and_is_reported() {
    on_device(|device, _| {
        let setup = Setup::new(device);
        let writer = triangles(device);
        let scratch = buffer_with(device, &[0; 64]).unwrap();
        let writing = Writing {
            setup: &setup,
            writer: &writer,
            out: &scratch,
        };
        let icb = icb(device, &inheriting(1), 2);
        let command_buffer = encode(device, &writing, (&icb, 2), (&[(0, 2)], &Call::Plain));
        let (gpu, read) = counted(device, || {
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
        });
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let report = command_buffer.execution_report().unwrap();
        assert!(report.replay_fault);
        assert_eq!(report.replay_faults, ReplayFaults::ENCODING);
        assert_eq!((gpu, read), (2, 0));
        assert_eq!(setup.output(), expected(&[]));
        let written = read_buffer(&scratch, 0, 8).unwrap();
        assert_eq!(written, [2, 0, 0, 0, 7, 0, 0, 0], "the writer ran");
    });
}
