//! Frames whose command buffer executes many render commands a kernel writes, on the device the
//! cfg selects, without the validation layer: report only, to see how long the host step holds
//! such a command buffer between its first submission and its last, frame after frame.
//!
//! Each frame is one command buffer: the indexed writer (`icb_render_indexed_encode`) over
//! [`COMMANDS`] threads writes command `tid`'s indexed draw of points into a buffer it made
//! resident for writing, then a render encoder under the template's state executes all of them.
//! The commands inherit the encoder's state; the CPU set each one's two vertex buffers once. The
//! buffer's command types are indexed draws, so the pass is recorded by a host step once the work
//! before it completed (`Degradation::HostReplayedIndirectCommands`). Each frame is committed and
//! waited for before the next, as a page that renders one frame at a time does.
//!
//! - **`alone`**: the frames on a device that runs nothing else.
//! - **`beside_queues`**: a second queue's thread commits a small compute command buffer every
//!   millisecond meanwhile, each waited for.
//! - **`bundle`**: as a browser's render bundle: the commands inherit nothing, the CPU set the
//!   template's state on each, and each command's vertex buffer 0 is a buffer of its own (a copy
//!   of the first window of `params`), every one of which the render encoder makes resident.
//! - **`bundle_no_copy`**: as `bundle`, each command's own buffer a no-copy buffer over host
//!   pages of its own, as a guest's buffers reach the runtime.
//!
//! Printed as `HOST-STEP-FRAMES` lines: per frame, its commit to its kernel start (the scheduler's
//! first submission), the kernel start to the kernel end (its last batch submitted), the end to its
//! completion, and the work counts that moved over the frame.

use super::host_memory::{no_copy, Mapping, GUEST_PAGE};
use super::indirect_host_steps;
use super::render_executions::{icb, pass, Setup, WINDOW};
use super::{buffer_with, read_buffer, validated_device_with, Messages, ScratchCache, WAIT};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{IndirectCommandType, RenderStages, ResourceUsage};
use metal2vulkan_runtime::{
    host_time, Buffer, CommandBufferStatus, CommandQueue, ComputePipelineState, Device,
    IndirectCommandBuffer, IndirectCommandBufferDescriptor, ResourceOptions, Size, WorkCounts,
};
use std::collections::BTreeMap;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// The commands each frame writes and executes: a render bundle of 5001 draws, as the consumer's
/// guest's page executes.
const COMMANDS: u64 = 5001;

/// The frames each stream runs.
const FRAMES: usize = 6;

/// How the commands are set up (the module docs).
#[derive(Clone, Copy, PartialEq)]
enum Shape {
    /// The encoder's state inherited, `params`' windows shared.
    Inherited,
    /// Nothing inherited, a buffer of each command's own, a no-copy one where `no_copy`.
    Bundle { no_copy: bool },
}

/// What every frame uses: the template's setup, the writer, the buffer of commands, the writer's
/// argument (the buffer's ID), the two index buffers its draws read, and each command's own
/// buffer (with the host pages under them, for the no-copy ones).
struct Frames {
    device: Device,
    setup: Setup,
    writer: ComputePipelineState,
    icb: IndirectCommandBuffer,
    arguments: Buffer,
    short_indices: Buffer,
    wide_indices: Buffer,
    own: Vec<Buffer>,
    _pages: Option<Mapping>,
}

impl Frames {
    fn new(device: Device, shape: Shape) -> Frames {
        let setup = Setup::new(&device);
        let writer = indirect_host_steps::state(&device, "icb_render_indexed_encode", false);
        let mut descriptor = IndirectCommandBufferDescriptor::new();
        descriptor.command_types = IndirectCommandType::DRAW_INDEXED;
        descriptor.inherit_pipeline_state = shape == Shape::Inherited;
        descriptor.max_vertex_buffer_bind_count = 2;
        descriptor.max_fragment_buffer_bind_count = 1;
        let icb = icb(&device, &descriptor, COMMANDS);
        let window = read_buffer(&setup.params, 0, (WINDOW * 4) as u64).unwrap();
        let (own, pages) = match shape {
            Shape::Inherited => (Vec::new(), None),
            Shape::Bundle { no_copy: false } => (
                (0..COMMANDS)
                    .map(|_| buffer_with(&device, &window).unwrap())
                    .collect(),
                None,
            ),
            Shape::Bundle { no_copy: true } => {
                let pages = Mapping::anonymous(COMMANDS as usize * GUEST_PAGE);
                let own = (0..COMMANDS as usize)
                    .map(|index| {
                        let at = index * GUEST_PAGE;
                        pages.write(at, &window);
                        // SAFETY: a page of the mapping of its own, which outlives the buffer
                        // (`Frames` drops the buffers first).
                        unsafe {
                            let pointer = NonNull::new_unchecked(pages.pointer().as_ptr().add(at));
                            no_copy(
                                &device,
                                pointer,
                                GUEST_PAGE,
                                ResourceOptions::shared(),
                                None,
                            )
                        }
                        .unwrap()
                    })
                    .collect();
                (own, Some(pages))
            }
        };
        for index in 0..COMMANDS {
            let command = icb.indirect_render_command_at(index).unwrap();
            let offset = ((index as usize % 4) * WINDOW * 4) as u64;
            match own.get(index as usize) {
                Some(buffer) => {
                    command.set_render_pipeline_state(&setup.state).unwrap();
                    command.set_vertex_buffer(Some(buffer), 0, 0).unwrap();
                }
                None => command
                    .set_vertex_buffer(Some(&setup.params), offset, 0)
                    .unwrap(),
            }
            command
                .set_vertex_buffer(Some(&setup.out), offset, 1)
                .unwrap();
        }
        let arguments = buffer_with(&device, &icb.gpu_resource_id().0.to_le_bytes()).unwrap();
        // Command 0 draws indices 1..6 less 2, the others indices 0..3 plus 4: within a window.
        let shorts: Vec<u8> = [0u16, 2, 3, 4, 5, 6, 7, 8]
            .iter()
            .flat_map(|i| i.to_le_bytes())
            .collect();
        let wides: Vec<u8> = [0u32, 1, 2, 3]
            .iter()
            .flat_map(|i| i.to_le_bytes())
            .collect();
        Frames {
            short_indices: buffer_with(&device, &shorts).unwrap(),
            wide_indices: buffer_with(&device, &wides).unwrap(),
            device,
            setup,
            writer,
            icb,
            arguments,
            own,
            _pages: pages,
        }
    }

    /// One frame on `queue`, committed and waited for: its commit to its kernel start, the kernel
    /// start to its end, and the end to its completion, in ms.
    fn frame(&self, queue: &CommandQueue) -> (f64, f64, f64) {
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&self.writer).unwrap();
        encoder.set_buffer(Some(&self.arguments), 0, 0).unwrap();
        encoder.set_buffer(Some(&self.short_indices), 0, 1).unwrap();
        encoder.set_buffer(Some(&self.wide_indices), 0, 2).unwrap();
        encoder.set_buffer(Some(&self.setup.params), 0, 3).unwrap();
        encoder.set_buffer(Some(&self.setup.out), 0, 4).unwrap();
        encoder
            .use_resource(self.icb.as_resource(), ResourceUsage::WRITE)
            .unwrap();
        encoder
            .dispatch_threads(Size::new(COMMANDS, 1, 1), Size::new(64, 1, 1))
            .unwrap();
        encoder.end_encoding().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass()).unwrap();
        encoder
            .set_render_pipeline_state(&self.setup.state)
            .unwrap();
        self.setup.resident(&mut encoder);
        for indices in [&self.short_indices, &self.wide_indices] {
            encoder
                .use_resource(
                    indices.as_resource(),
                    ResourceUsage::READ,
                    RenderStages::VERTEX,
                )
                .unwrap();
        }
        if !self.own.is_empty() {
            let own: Vec<_> = self.own.iter().map(Buffer::as_resource).collect();
            encoder
                .use_resources(&own, ResourceUsage::READ, RenderStages::VERTEX)
                .unwrap();
        }
        encoder
            .execute_commands_in_buffer(&self.icb, 0..COMMANDS)
            .unwrap();
        encoder.end_encoding().unwrap();
        let committed = host_time();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        let completed = host_time();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let (start, end) = (
            command_buffer.kernel_start_time(),
            command_buffer.kernel_end_time(),
        );
        let ms = |seconds: f64| seconds * 1000.0;
        (ms(start - committed), ms(end - start), ms(completed - end))
    }
}

/// The stream's frames, printed (the module docs).
fn measure(stream: &str, shape: Shape, beside_queues: bool) {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.layers.clear();
        // A consumer's device keeps no execution record.
        options.record_executions = false;
    })
    .unwrap();
    let name = device.name().to_owned();
    let frames = Frames::new(device, shape);
    let stop = Arc::new(AtomicBool::new(false));
    let beside = beside_queues.then(|| {
        let device = frames.device.clone();
        let stop = Arc::clone(&stop);
        let state = indirect_host_steps::state(&device, "a4_index_word", false);
        std::thread::spawn(move || {
            let queue = device.new_command_queue().unwrap();
            let input = buffer_with(&device, &[0; 4 * 3 * 64 + 8]).unwrap();
            let output = buffer_with(&device, &[0; 4 * 64]).unwrap();
            let mut committed = 0u64;
            while !stop.load(Ordering::Relaxed) {
                let command_buffer = queue.command_buffer().unwrap();
                let mut encoder = command_buffer.compute_command_encoder().unwrap();
                encoder.set_compute_pipeline_state(&state).unwrap();
                encoder.set_buffer(Some(&input), 0, 0).unwrap();
                encoder.set_buffer(Some(&output), 0, 1).unwrap();
                encoder
                    .dispatch_threads(Size::new(64, 1, 1), Size::new(64, 1, 1))
                    .unwrap();
                encoder.end_encoding().unwrap();
                command_buffer.commit().unwrap();
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                committed += 1;
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            committed
        })
    });
    let queue = frames.device.new_command_queue().unwrap();
    for frame in 0..FRAMES {
        let before = fields(&frames.device.work_counts());
        let (to_start, submitting, to_completion) = frames.frame(&queue);
        let after = fields(&frames.device.work_counts());
        let moved: Vec<String> = after
            .iter()
            .filter_map(|(field, &value)| {
                let was = before.get(field).copied().unwrap_or(0);
                (value != was).then(|| format!("{field} {}", value.wrapping_sub(was)))
            })
            .collect();
        eprintln!(
            "HOST-STEP-FRAMES {name}: {stream}: frame {frame}: commit to kernel start {to_start:.2} \
             ms, kernel start to end {submitting:.2} ms, end to completion {to_completion:.2} ms"
        );
        eprintln!(
            "HOST-STEP-FRAMES {name}: {stream}: frame {frame}: moved: {}",
            moved.join(", ")
        );
    }
    stop.store(true, Ordering::Relaxed);
    if let Some(thread) = beside {
        let committed = thread.join().unwrap();
        eprintln!("HOST-STEP-FRAMES {name}: {stream}: the other queue committed {committed}");
    }
}

/// A count's fields by name, from its `Debug` form: the top-level integers.
fn fields(counts: &WorkCounts) -> BTreeMap<String, u64> {
    let text = format!("{counts:?}");
    let body = text
        .find('{')
        .map(|at| &text[at + 1..text.len() - 1])
        .unwrap_or("");
    let mut fields = BTreeMap::new();
    let (mut depth, mut start) = (0usize, 0usize);
    let bytes = body.as_bytes();
    let mut take = |entry: &str| {
        if let Some((name, value)) = entry.split_once(':') {
            if let Ok(value) = value.trim().parse::<u64>() {
                fields.insert(name.trim().to_owned(), value);
            }
        }
    };
    for (at, &byte) in bytes.iter().enumerate() {
        match byte {
            b'{' | b'[' | b'(' => depth += 1,
            b'}' | b']' | b')' => depth -= 1,
            b',' if depth == 0 => {
                take(&body[start..at]);
                start = at + 1;
            }
            _ => {}
        }
    }
    take(&body[start..]);
    fields
}

#[test]
#[ignore = "a measurement: run it alone, in release"]
fn host_step_frames_alone() {
    measure("alone", Shape::Inherited, false);
}

#[test]
#[ignore = "a measurement: run it alone, in release"]
fn host_step_frames_beside_queues() {
    measure("beside_queues", Shape::Inherited, true);
}

#[test]
#[ignore = "a measurement: run it alone, in release"]
fn host_step_frames_bundle() {
    measure("bundle", Shape::Bundle { no_copy: false }, false);
}

#[test]
#[ignore = "a measurement: run it alone, in release"]
fn host_step_frames_bundle_no_copy() {
    measure("bundle_no_copy", Shape::Bundle { no_copy: true }, false);
}
