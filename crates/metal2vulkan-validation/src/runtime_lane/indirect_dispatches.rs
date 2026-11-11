//! Indirect dispatches (`dispatch_threadgroups_indirect`, `dispatch_threads_indirect`) on the
//! device the cfg selects, under the validation layer. The owned `icb_replay_grid_template`
//! fixture's kernel adds what each thread observes of its dispatch (its threadgroup's size, the
//! nominal size, its threadgroup's position, the grid's threadgroups and threads, its position in
//! its threadgroup, and its neighbour's index through threadgroup memory) into an output slot of
//! its position in the grid. An indirect dispatch writes what the direct dispatch of the sizes its
//! arguments state writes, which the runtime's direct path gives as Metal's:
//!
//! - ragged and uniform thread grids in one, two and three dimensions, and group counts, the
//!   arguments at an offset of their buffer;
//! - arguments a blit encoder of the same command buffer wrote before the dispatch;
//! - a group count past the device's `maxComputeWorkGroupCount` on y (65536 groups), which the
//!   direct dispatch splits;
//! - nothing for a zero axis of the grid or the group counts, a threadgroup with a zero axis, or
//!   one past the state's `max_total_threads_per_threadgroup`, as our probes measured Metal's on an
//!   M3 Max and an M4;
//! - arguments at an offset that is not a multiple of 4 read at the multiple below, as plain
//!   Metal reads them;
//! - refused as Metal's API validation refuses arguments past the buffer, and with no pipeline
//!   state set.
//!
//! Each test's device announces the host read once ([`INDIRECT_DISPATCHES`]): the fixture
//! observes its threadgroup partition, so its thread dispatches take the host step.
//!
//! The GPU path runs a kernel that does not observe its partition on its flat variant
//! ([`FLAT_SUM`], each thread adding into the output slot of its position, so a thread run twice or
//! not at all shows): both forms over ragged and uniform grids write what the direct dispatch
//! writes, on the flat variant (its execution's pipeline states `FlatGrid`), with the host read not
//! announced; arguments an earlier dispatch of the same encoder wrote are read; item 37's cases of
//! nothing write nothing.
//!
//! The folded path runs the fixture's threadgroup dispatches, which observe the partition, on its
//! folded variant of the call's threadgroup size: group counts within the device's limits and past
//! them (65536 groups along `x` and along `y`, past lavapipe's `x` and every committed profile's
//! `y`) write what the direct dispatch writes, on the folded variant (its execution's pipeline
//! states `FoldedGroups`), with the host read not announced; arguments an earlier dispatch of the
//! same encoder wrote are read; a zero group count writes nothing.

use super::degrade_notices::on_device_with;
use super::degrade_notices::{
    assert_made_once, counts, forced, on_device, shared_in_system_memory,
};
use super::indirect_host_steps::state;
use super::{at_plan, buffer_with, layer_reported_nothing_at_all, read_buffer, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::bind::StatusReport;
use metal2vulkan::pipeline::{ComputePipelineDescriptor, RuntimeFact, StageSource};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, CommandBufferStatus, ComputeCommandEncoder, ComputePipelineState,
    Degradation, Device, Error, ExecutedPipeline, Invalid, PipelineOptions, Problem, Size,
};

/// The kind every test here takes: an indirect dispatch's arguments are read on the host,
/// announced once per device however many dispatches take it.
pub(super) const INDIRECT_DISPATCHES: [Degradation; 1] = [Degradation::HostReadDispatchArguments];

/// The kinds a test's steps force beside [`INDIRECT_DISPATCHES`], where the device has not
/// announced them: `split-dispatch` where a dispatch of threads leaves a partial threadgroup
/// (`ragged`: Vulkan dispatches whole workgroups), and the `Shared` buffers' system memory.
fn forced_beside(device: &Device, ragged: bool) -> std::collections::BTreeSet<Degradation> {
    forced(&[
        (Degradation::SplitDispatch, ragged),
        (
            Degradation::SharedInSystemMemory,
            shared_in_system_memory(device),
        ),
    ])
}

/// The output's slots, and the fixture's `params[1]`.
const SLOTS: u32 = 64;

/// The most threads a threadgroup here holds: the threadgroup memory stages a word for each.
const MAX_GROUP: u32 = 64;

/// The arguments' offset in their buffer.
const OFFSET: u64 = 16;

/// The sizes of a dispatch.
#[derive(Clone, Copy, Debug)]
enum Sizes {
    /// `dispatchThreads`: the grid's threads, in threadgroups of `group`.
    Threads { grid: [u32; 3], group: [u32; 3] },
    /// `dispatchThreadgroups`: `groups` threadgroups of `group`.
    Threadgroups { groups: [u32; 3], group: [u32; 3] },
}

impl Sizes {
    /// The indirect arguments' words.
    fn words(self) -> Vec<u32> {
        match self {
            Sizes::Threads { grid, group } => grid.into_iter().chain(group).collect(),
            Sizes::Threadgroups { groups, .. } => groups.to_vec(),
        }
    }
}

/// How a dispatch is encoded.
#[derive(Clone, Copy, Debug)]
enum Encoded {
    Direct,
    Indirect,
}

/// `words` as bytes.
fn bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// A Metal size of `axes`.
fn size([x, y, z]: [u32; 3]) -> Size {
    Size::new(x.into(), y.into(), z.into())
}

/// The fixture's state, and the buffers one dispatch uses.
struct Fixture {
    state: ComputePipelineState,
    params: Buffer,
    out: Buffer,
    /// The arguments, at [`OFFSET`].
    arguments: Buffer,
}

impl Fixture {
    /// The fixture on `device`, the arguments of `sizes` written at [`OFFSET`].
    fn new(device: &Device, sizes: Sizes) -> Fixture {
        let mut words = vec![0; (OFFSET / 4) as usize];
        words.extend(sizes.words());
        Fixture {
            state: state(device, "icb_replay_grid_template", false),
            params: buffer_with(device, &bytes(&[7, SLOTS])).unwrap(),
            out: buffer_with(device, &bytes(&[0; SLOTS as usize])).unwrap(),
            arguments: buffer_with(device, &bytes(&words)).unwrap(),
        }
    }

    /// Set the state and its buffers on `encoder`.
    fn bind(&self, encoder: &mut ComputeCommandEncoder) {
        encoder.set_compute_pipeline_state(&self.state).unwrap();
        encoder.set_buffer(Some(&self.params), 0, 0).unwrap();
        encoder.set_buffer(Some(&self.out), 0, 1).unwrap();
        encoder
            .set_threadgroup_memory_length(4 * MAX_GROUP, 0)
            .unwrap();
    }

    /// Encode a dispatch of `sizes` on `encoder` as `how`.
    fn dispatch(&self, encoder: &mut ComputeCommandEncoder, sizes: Sizes, how: Encoded) {
        match (sizes, how) {
            (Sizes::Threads { grid, group }, Encoded::Direct) => {
                encoder.dispatch_threads(size(grid), size(group))
            }
            (Sizes::Threadgroups { groups, group }, Encoded::Direct) => {
                encoder.dispatch_threadgroups(size(groups), size(group))
            }
            (Sizes::Threads { .. }, Encoded::Indirect) => {
                encoder.dispatch_threads_indirect(&self.arguments, OFFSET)
            }
            (Sizes::Threadgroups { group, .. }, Encoded::Indirect) => {
                encoder.dispatch_threadgroups_indirect(&self.arguments, OFFSET, size(group))
            }
        }
        .unwrap();
    }

    /// The output's slots.
    fn out(&self) -> Vec<u32> {
        read_buffer(&self.out, 0, u64::from(SLOTS) * 4)
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect()
    }
}

/// Commit `command_buffer`, wait for it, and check it completed.
fn complete(command_buffer: &CommandBuffer, what: &str) {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(
        command_buffer.status(),
        CommandBufferStatus::Completed,
        "{what}: {:?}",
        command_buffer.error()
    );
}

/// What one dispatch of `sizes`, encoded as `how`, writes on `device`.
fn run(device: &Device, sizes: Sizes, how: Encoded) -> Vec<u32> {
    let fixture = Fixture::new(device, sizes);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    fixture.bind(&mut encoder);
    fixture.dispatch(&mut encoder, sizes, how);
    encoder.end_encoding().unwrap();
    complete(&command_buffer, &format!("{sizes:?} {how:?}"));
    fixture.out()
}

/// Every indirect dispatch here writes what the direct dispatch of its arguments writes, and
/// that is not nothing.
#[test]
fn an_indirect_dispatch_runs_as_the_direct_dispatch_of_its_arguments() {
    on_device_with(at_plan, |device, messages| {
        let threads = |grid, group| Sizes::Threads { grid, group };
        let groups = |groups, group| Sizes::Threadgroups { groups, group };
        for sizes in [
            threads([10, 3, 1], [4, 2, 1]),
            threads([8, 4, 1], [4, 2, 1]),
            threads([5, 3, 3], [2, 2, 2]),
            threads([100, 1, 1], [32, 1, 1]),
            groups([3, 2, 1], [4, 2, 1]),
            groups([2, 1, 3], [2, 2, 2]),
            groups([1, 65536, 1], [1, 1, 1]),
        ] {
            let direct = run(device, sizes, Encoded::Direct);
            assert_ne!(direct, vec![0; SLOTS as usize], "{sizes:?}");
            assert_eq!(run(device, sizes, Encoded::Indirect), direct, "{sizes:?}");
        }
        // A grid of 10 x 3 threads in threadgroups of 4 x 2 leaves partial ones.
        assert_made_once(messages, &INDIRECT_DISPATCHES, &forced_beside(device, true));
    });
}

/// The arguments a blit encoder of the same command buffer copies over zeros run, as Metal's
/// dispatch reads them when it runs.
#[test]
fn an_indirect_dispatch_reads_the_arguments_the_work_before_it_wrote() {
    on_device_with(at_plan, |device, messages| {
        let sizes = Sizes::Threads {
            grid: [10, 3, 1],
            group: [4, 2, 1],
        };
        let direct = run(device, sizes, Encoded::Direct);
        let fixture = Fixture::new(device, sizes);
        let written = buffer_with(device, &bytes(&sizes.words())).unwrap();
        let zeros = buffer_with(device, &[0; 24]).unwrap();
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_buffer(&zeros, 0, &fixture.arguments, OFFSET, 24)
            .unwrap();
        blit.copy_from_buffer_to_buffer(&written, 0, &fixture.arguments, OFFSET, 24)
            .unwrap();
        blit.end_encoding().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        fixture.bind(&mut encoder);
        fixture.dispatch(&mut encoder, sizes, Encoded::Indirect);
        encoder.end_encoding().unwrap();
        complete(&command_buffer, "written arguments");
        assert_eq!(fixture.out(), direct);
        // A grid of 10 x 3 threads in threadgroups of 4 x 2 leaves partial ones.
        assert_made_once(messages, &INDIRECT_DISPATCHES, &forced_beside(device, true));
    });
}

/// A zero axis of the grid or the group counts, a threadgroup with a zero axis, and one past the
/// state's maximum run nothing, without an error (probes 17b and 18).
#[test]
fn an_indirect_dispatch_metal_runs_nothing_for_runs_nothing() {
    on_device_with(at_plan, |device, messages| {
        let max =
            state(device, "icb_replay_grid_template", false).max_total_threads_per_threadgroup();
        let past = [MAX_GROUP, max.div_ceil(MAX_GROUP) + 1, 1];
        for sizes in [
            Sizes::Threads {
                grid: [0, 3, 1],
                group: [4, 2, 1],
            },
            Sizes::Threads {
                grid: [10, 3, 1],
                group: [0, 2, 1],
            },
            Sizes::Threads {
                grid: [64, 64, 1],
                group: past,
            },
            Sizes::Threadgroups {
                groups: [3, 0, 1],
                group: [4, 2, 1],
            },
        ] {
            assert_eq!(
                run(device, sizes, Encoded::Indirect),
                vec![0; SLOTS as usize],
                "{sizes:?}"
            );
        }
        assert_made_once(
            messages,
            &INDIRECT_DISPATCHES,
            &forced_beside(device, false),
        );
    });
}

/// Refused at the call as Metal's API validation refuses it: arguments past the buffer's length,
/// and no pipeline state set.
#[test]
fn indirect_dispatches_are_refused_where_metal_refuses_them() {
    let messages = super::Messages::default();
    let cache = super::ScratchCache::new().unwrap();
    {
        let device =
            super::validated_device(crate::gpu::TEST_DEVICE, &messages, cache.path()).unwrap();
        let sizes = Sizes::Threads {
            grid: [10, 3, 1],
            group: [4, 2, 1],
        };
        let fixture = Fixture::new(&device, sizes);
        let length = fixture.arguments.length();
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        let problem = |result: Result<(), Error>| match result {
            Err(Error::Invalid(Invalid { problem, .. })) => problem,
            other => panic!("an indirect dispatch Metal refuses gave {other:?}"),
        };
        assert_eq!(
            problem(encoder.dispatch_threads_indirect(&fixture.arguments, OFFSET)),
            Problem::NoPipelineState
        );
        fixture.bind(&mut encoder);
        assert_eq!(
            problem(encoder.dispatch_threads_indirect(&fixture.arguments, length - 20)),
            Problem::IndirectDispatchRange {
                offset: length - 20,
                bytes: 24,
                length,
            }
        );
        assert_eq!(
            problem(encoder.dispatch_threadgroups_indirect(
                &fixture.arguments,
                length - 8,
                size([4, 2, 1]),
            )),
            Problem::IndirectDispatchRange {
                offset: length - 8,
                bytes: 12,
                length,
            }
        );
        encoder.end_encoding().unwrap();
        complete(&command_buffer, "refused dispatches");
        assert_eq!(fixture.out(), vec![0; SLOTS as usize]);
    }
    layer_reported_nothing_at_all(&messages);
}

/// An offset that is not a multiple of 4 reads the arguments at the multiple below, as plain
/// Metal reads them where its API validation asserts: the dispatch writes what
/// the one at [`OFFSET`] writes, its arguments running past the end of the buffer from the offset
/// given but not from there.
#[test]
fn an_unaligned_offset_reads_at_the_multiple_below() {
    on_device_with(at_plan, |device, messages| {
        let threads = Sizes::Threads {
            grid: [10, 3, 1],
            group: [4, 2, 1],
        };
        let groups = Sizes::Threadgroups {
            groups: [3, 2, 1],
            group: [4, 2, 1],
        };
        for (sizes, offset) in [
            (threads, OFFSET + 2),
            (threads, OFFSET + 3),
            (groups, OFFSET + 1),
        ] {
            let direct = run(device, sizes, Encoded::Direct);
            assert_ne!(direct, vec![0; SLOTS as usize], "{sizes:?}");
            let fixture = Fixture::new(device, sizes);
            assert!(fixture.arguments.length() < offset + 4 * sizes.words().len() as u64);
            let queue = device.new_command_queue().unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            fixture.bind(&mut encoder);
            match sizes {
                Sizes::Threads { .. } => {
                    encoder.dispatch_threads_indirect(&fixture.arguments, offset)
                }
                Sizes::Threadgroups { group, .. } => {
                    encoder.dispatch_threadgroups_indirect(&fixture.arguments, offset, size(group))
                }
            }
            .unwrap();
            encoder.end_encoding().unwrap();
            complete(&command_buffer, &format!("{sizes:?} at {offset}"));
            assert_eq!(fixture.out(), direct, "{sizes:?} at {offset}");
        }
        // A grid of 10 x 3 threads in threadgroups of 4 x 2 leaves partial ones.
        assert_made_once(messages, &INDIRECT_DISPATCHES, &forced_beside(device, true));
    });
}

/// `flat_sum(params, out, p, g)`: each thread adds `params[0] + 3 p.x + 5 p.y + 7 p.z + 11 g.x +
/// 13 g.y + 17 g.z` to `out[p.x + p.y g.x + p.z g.x g.y]` by a relaxed atomic whose result it does
/// not read, reading only `thread_position_in_grid` and `threads_per_grid` (the kernel of the
/// device replay's flat tests): it does not observe its threadgroup partition, and a thread run
/// twice adds twice. The same addition as a plain load and store takes a lockstep barrier, which
/// observes the partition, so it has no flat variant (a known gap).
pub(super) const FLAT_SUM: &str = r#"define void @flat_sum(ptr addrspace(1) noundef "air-buffer-no-alias" %b0, ptr addrspace(1) noundef "air-buffer-no-alias" %b1, <3 x i32> noundef %p, <3 x i32> noundef %g) {
entry:
  %base = load i32, ptr addrspace(1) %b0, align 4
  %px = extractelement <3 x i32> %p, i64 0
  %py = extractelement <3 x i32> %p, i64 1
  %pz = extractelement <3 x i32> %p, i64 2
  %gx = extractelement <3 x i32> %g, i64 0
  %gy = extractelement <3 x i32> %g, i64 1
  %gz = extractelement <3 x i32> %g, i64 2
  %a0 = mul i32 %px, 3
  %a1 = mul i32 %py, 5
  %a2 = mul i32 %pz, 7
  %a3 = mul i32 %gx, 11
  %a4 = mul i32 %gy, 13
  %a5 = mul i32 %gz, 17
  %s0 = add i32 %base, %a0
  %s1 = add i32 %s0, %a1
  %s2 = add i32 %s1, %a2
  %s3 = add i32 %s2, %a3
  %s4 = add i32 %s3, %a4
  %v = add i32 %s4, %a5
  %row = mul i32 %py, %gx
  %plane = mul i32 %gx, %gy
  %layer = mul i32 %pz, %plane
  %i0 = add i32 %px, %row
  %i = add i32 %i0, %layer
  %i64 = zext i32 %i to i64
  %o = getelementptr inbounds i32, ptr addrspace(1) %b1, i64 %i64
  %old = atomicrmw add ptr addrspace(1) %o, i32 %v monotonic, align 4
  ret void
}

!air.kernel = !{!0}
!air.version = !{!20}
!air.language_version = !{!21}
!0 = !{ptr @flat_sum, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"b0"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"b1"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"p"}
!6 = !{i32 3, !"air.threads_per_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"g"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// The flat kernel's output slots: the largest grid here, 100 threads.
const FLAT_SLOTS: u64 = 128;

/// The flat kernel's state on `device`.
fn flat_state(device: &Device) -> ComputePipelineState {
    let library = device
        .new_library_with_air(&llvm_assemble_text(FLAT_SUM).unwrap())
        .unwrap();
    let function = library.new_function_with_name("flat_sum").unwrap();
    device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap()
}

/// What one dispatch of the flat kernel of `sizes`, encoded as `how`, wrote on `device`, and
/// whether it ran on the flat variant. With a `writer` (the owned `a4_index_word`, `out[t] =
/// in[3 t + 1]`) the arguments are zero until a dispatch of it before the flat kernel's, in the
/// same encoder, writes them.
fn flat_run(
    device: &Device,
    state: &ComputePipelineState,
    sizes: Sizes,
    how: Encoded,
    writer: Option<&ComputePipelineState>,
) -> (Vec<u32>, bool) {
    let stated = sizes.words();
    let mut words = vec![0; (OFFSET / 4) as usize];
    match writer {
        Some(_) => words.extend(vec![0; stated.len()]),
        None => words.extend(&stated),
    }
    let arguments = buffer_with(device, &bytes(&words)).unwrap();
    let params = buffer_with(device, &bytes(&[1000])).unwrap();
    let out = buffer_with(device, &vec![0; FLAT_SLOTS as usize * 4]).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    if let Some(writer) = writer {
        let mut sources = vec![0; 3 * stated.len() + 1];
        for (t, word) in stated.iter().enumerate() {
            sources[3 * t + 1] = *word;
        }
        let sources = buffer_with(device, &bytes(&sources)).unwrap();
        let count = stated.len() as u64;
        encoder.set_compute_pipeline_state(writer).unwrap();
        encoder.set_buffer(Some(&sources), 0, 0).unwrap();
        encoder.set_buffer(Some(&arguments), OFFSET, 1).unwrap();
        encoder
            .dispatch_threads(Size::new(count, 1, 1), Size::new(count, 1, 1))
            .unwrap();
    }
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_buffer(Some(&params), 0, 0).unwrap();
    encoder.set_buffer(Some(&out), 0, 1).unwrap();
    match (sizes, how) {
        (Sizes::Threads { grid, group }, Encoded::Direct) => {
            encoder.dispatch_threads(size(grid), size(group))
        }
        (Sizes::Threadgroups { groups, group }, Encoded::Direct) => {
            encoder.dispatch_threadgroups(size(groups), size(group))
        }
        (Sizes::Threads { .. }, Encoded::Indirect) => {
            encoder.dispatch_threads_indirect(&arguments, OFFSET)
        }
        (Sizes::Threadgroups { group, .. }, Encoded::Indirect) => {
            encoder.dispatch_threadgroups_indirect(&arguments, OFFSET, size(group))
        }
    }
    .unwrap();
    encoder.end_encoding().unwrap();
    complete(&command_buffer, &format!("flat {sizes:?} {how:?}"));
    assert_eq!(
        command_buffer.execution_report(),
        Some(StatusReport::default()),
        "flat {sizes:?} {how:?}"
    );
    let flat = command_buffer
        .executions()
        .iter()
        .any(|execution| match execution.pipeline() {
            ExecutedPipeline::Compute(pipeline) => pipeline
                .dependencies
                .iter()
                .any(|dependency| dependency.fact == RuntimeFact::FlatGrid),
            ExecutedPipeline::Render(_) => false,
        });
    let out = read_buffer(&out, 0, FLAT_SLOTS * 4)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect();
    (out, flat)
}

/// An indirect dispatch of a kernel that does not observe its partition runs on its flat variant
/// and writes what the direct dispatch of its arguments writes, each thread once; arguments an
/// earlier dispatch of the encoder wrote are read; Metal's cases of nothing write nothing; the host
/// read is not announced.
#[test]
fn a_partition_oblivious_indirect_dispatch_runs_on_the_gpu() {
    on_device_with(at_plan, |device, messages| {
        let kernel = flat_state(device);
        let threads = |grid, group| Sizes::Threads { grid, group };
        let groups = |groups, group| Sizes::Threadgroups { groups, group };
        for sizes in [
            threads([10, 3, 1], [4, 2, 1]),
            threads([8, 4, 1], [4, 2, 1]),
            threads([5, 3, 3], [2, 2, 2]),
            threads([100, 1, 1], [32, 1, 1]),
            threads([7, 2, 3], [1, 1, 1]),
            groups([3, 2, 1], [4, 2, 1]),
            groups([2, 1, 3], [2, 2, 2]),
        ] {
            let (direct, _) = flat_run(device, &kernel, sizes, Encoded::Direct, None);
            assert_ne!(direct, vec![0; FLAT_SLOTS as usize], "{sizes:?}");
            let (indirect, flat) = flat_run(device, &kernel, sizes, Encoded::Indirect, None);
            assert_eq!(indirect, direct, "{sizes:?}");
            assert!(flat, "{sizes:?}: not on the flat variant");
        }

        // Arguments a dispatch before it in the encoder wrote, over zeros, in both forms.
        let writer = state(device, "a4_index_word", false);
        for sizes in [threads([10, 3, 1], [4, 2, 1]), groups([3, 2, 1], [4, 2, 1])] {
            let (direct, _) = flat_run(device, &kernel, sizes, Encoded::Direct, None);
            let written = flat_run(device, &kernel, sizes, Encoded::Indirect, Some(&writer));
            assert_eq!(written, (direct, true), "{sizes:?} written");
        }

        let max = kernel.max_total_threads_per_threadgroup();
        let past = [64, max.div_ceil(64) + 1, 1];
        for sizes in [
            threads([0, 3, 1], [4, 2, 1]),
            threads([10, 3, 1], [0, 2, 1]),
            threads([64, 64, 1], past),
            groups([3, 0, 1], [4, 2, 1]),
        ] {
            let (out, flat) = flat_run(device, &kernel, sizes, Encoded::Indirect, None);
            assert_eq!(out, vec![0; FLAT_SLOTS as usize], "{sizes:?}");
            assert!(flat, "{sizes:?}: not on the flat variant");
        }
        let announced = counts(&messages.take_notices());
        assert!(
            !announced.contains_key(&Degradation::HostReadDispatchArguments),
            "the host read was announced: {announced:?}"
        );
    });
}

/// What one threadgroup dispatch of the fixture of `sizes`, encoded as `how`, wrote on `device`,
/// and whether it ran on the folded variant. With a `writer` (the owned `a4_index_word`, `out[t]
/// = in[3 t + 1]`) the arguments state one threadgroup until a dispatch of it before the
/// fixture's, in the same encoder, writes the counts of `sizes`.
fn folded_run(
    device: &Device,
    sizes: Sizes,
    how: Encoded,
    writer: Option<&ComputePipelineState>,
) -> (Vec<u32>, bool) {
    let stated = match (sizes, writer) {
        (Sizes::Threadgroups { group, .. }, Some(_)) => Sizes::Threadgroups {
            groups: [1, 1, 1],
            group,
        },
        _ => sizes,
    };
    let fixture = Fixture::new(device, stated);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    if let Some(writer) = writer {
        let words = sizes.words();
        let mut spread = vec![0; 3 * words.len() + 1];
        for (t, word) in words.iter().enumerate() {
            spread[3 * t + 1] = *word;
        }
        let sources = buffer_with(device, &bytes(&spread)).unwrap();
        let count = words.len() as u64;
        encoder.set_compute_pipeline_state(writer).unwrap();
        encoder.set_buffer(Some(&sources), 0, 0).unwrap();
        encoder
            .set_buffer(Some(&fixture.arguments), OFFSET, 1)
            .unwrap();
        encoder
            .dispatch_threads(Size::new(count, 1, 1), Size::new(count, 1, 1))
            .unwrap();
    }
    fixture.bind(&mut encoder);
    fixture.dispatch(&mut encoder, sizes, how);
    encoder.end_encoding().unwrap();
    complete(&command_buffer, &format!("folded {sizes:?} {how:?}"));
    assert_eq!(
        command_buffer.execution_report(),
        Some(StatusReport::default()),
        "folded {sizes:?} {how:?}"
    );
    let folded = command_buffer
        .executions()
        .iter()
        .any(|execution| match execution.pipeline() {
            ExecutedPipeline::Compute(pipeline) => pipeline
                .dependencies
                .iter()
                .any(|dependency| dependency.fact == RuntimeFact::FoldedGroups),
            ExecutedPipeline::Render(_) => false,
        });
    (fixture.out(), folded)
}

/// An indirect threadgroup dispatch of a kernel that observes its partition runs on its folded
/// variant and writes what the direct dispatch of its arguments writes, each threadgroup once at
/// Metal's position, past the device's group-count limits too; arguments an earlier dispatch of the
/// encoder wrote are read; a zero group count writes nothing; the host read is not announced.
#[test]
fn an_indirect_threadgroup_dispatch_runs_on_the_gpu() {
    on_device_with(at_plan, |device, messages| {
        let groups = |groups, group| Sizes::Threadgroups { groups, group };
        for sizes in [
            groups([3, 2, 1], [4, 2, 1]),
            groups([2, 1, 3], [2, 2, 2]),
            groups([5, 3, 2], [1, 1, 1]),
            groups([1, 65536, 1], [1, 1, 1]),
            groups([65536, 1, 1], [1, 1, 1]),
        ] {
            let (direct, _) = folded_run(device, sizes, Encoded::Direct, None);
            assert_ne!(direct, vec![0; SLOTS as usize], "{sizes:?}");
            let (indirect, folded) = folded_run(device, sizes, Encoded::Indirect, None);
            assert_eq!(indirect, direct, "{sizes:?}");
            assert!(folded, "{sizes:?}: not on the folded variant");
        }

        // Arguments dispatches before it in the encoder wrote: zeros, then the counts.
        let writer = state(device, "a4_index_word", false);
        for sizes in [
            groups([3, 2, 1], [4, 2, 1]),
            groups([1, 65536, 1], [1, 1, 1]),
        ] {
            let (direct, _) = folded_run(device, sizes, Encoded::Direct, None);
            let written = folded_run(device, sizes, Encoded::Indirect, Some(&writer));
            assert_eq!(written, (direct, true), "{sizes:?} written");
        }

        for sizes in [groups([3, 0, 1], [4, 2, 1]), groups([0, 1, 1], [1, 1, 1])] {
            let (out, folded) = folded_run(device, sizes, Encoded::Indirect, None);
            assert_eq!(out, vec![0; SLOTS as usize], "{sizes:?}");
            assert!(folded, "{sizes:?}: not on the folded variant");
        }
        let announced = counts(&messages.take_notices());
        assert!(
            !announced.contains_key(&Degradation::HostReadDispatchArguments),
            "the host read was announced: {announced:?}"
        );
    });
}

/// A direct `dispatchThreads:` of a ragged grid runs on the flat variant (one dispatch, no split)
/// and each of Metal's threads runs once, at its own position: a 1206 x 2622 grid in 32 x 32
/// and 32 x 8 threadgroups (3.16 M threads, the flat count's 32-bit form), and ragged 3-D grids.
/// The expectation is computed here, not by another dispatch.
#[test]
fn a_ragged_direct_dispatch_runs_each_thread_once_on_the_flat_variant() {
    on_device(|device, messages| {
        let kernel = flat_state(device);
        for (grid, group) in [
            ([1206u32, 2622, 1], [32u32, 32, 1]),
            ([1206, 2622, 1], [32, 8, 1]),
            ([37, 19, 5], [4, 4, 2]),
            ([1000, 3, 2], [64, 1, 1]),
        ] {
            let slots = u64::from(grid[0] * grid[1] * grid[2]);
            let params = buffer_with(device, &bytes(&[1000])).unwrap();
            let out = buffer_with(device, &vec![0; slots as usize * 4]).unwrap();
            let queue = device.new_command_queue().unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encoder.set_compute_pipeline_state(&kernel).unwrap();
            encoder.set_buffer(Some(&params), 0, 0).unwrap();
            encoder.set_buffer(Some(&out), 0, 1).unwrap();
            encoder.dispatch_threads(size(grid), size(group)).unwrap();
            encoder.end_encoding().unwrap();
            complete(&command_buffer, &format!("ragged {grid:?} {group:?}"));
            let flat =
                command_buffer
                    .executions()
                    .iter()
                    .any(|execution| match execution.pipeline() {
                        ExecutedPipeline::Compute(pipeline) => pipeline
                            .dependencies
                            .iter()
                            .any(|dependency| dependency.fact == RuntimeFact::FlatGrid),
                        ExecutedPipeline::Render(_) => false,
                    });
            assert!(flat, "{grid:?} {group:?}: not on the flat variant");
            let words = read_buffer(&out, 0, slots * 4).unwrap();
            let [gx, gy, gz] = grid;
            let common = 1000 + 11 * gx + 13 * gy + 17 * gz;
            let mut wrong = 0u64;
            let mut first = None;
            for (i, w) in words.as_chunks::<4>().0.iter().enumerate() {
                let i = i as u32;
                let (x, y, z) = (i % gx, (i / gx) % gy, i / (gx * gy));
                let want = common + 3 * x + 5 * y + 7 * z;
                let got = u32::from_le_bytes(*w);
                if got != want {
                    wrong += 1;
                    first.get_or_insert((x, y, z, got, want));
                }
            }
            assert_eq!(wrong, 0, "{grid:?} {group:?}: first wrong {first:?}");
        }
        let _ = messages.take_notices();
    });
}

/// A kernel that marks the grid's last column and last row: the thread at `x = gx - 1` adds 1
/// into `out[y]`, the thread at `y = gy - 1` into `out[gy + x]`. Over a grid of more than 2^32
/// threads only these edges are stored, and a position counted in 32 bits would wrap the rows past
/// 2^32 threads onto the first ones (those counted twice, the last never).
const FLAT_EDGES: &str = r#"define void @flat_edges(ptr addrspace(1) noundef "air-buffer-no-alias" %b0, <3 x i32> noundef %p, <3 x i32> noundef %g) {
entry:
  %px = extractelement <3 x i32> %p, i64 0
  %py = extractelement <3 x i32> %p, i64 1
  %gx = extractelement <3 x i32> %g, i64 0
  %gy = extractelement <3 x i32> %g, i64 1
  %lx = add i32 %gx, -1
  %ly = add i32 %gy, -1
  %ex = icmp eq i32 %px, %lx
  br i1 %ex, label %col, label %next
col:
  %yi = zext i32 %py to i64
  %o0 = getelementptr inbounds i32, ptr addrspace(1) %b0, i64 %yi
  %a0 = atomicrmw add ptr addrspace(1) %o0, i32 1 monotonic, align 4
  br label %next
next:
  %ey = icmp eq i32 %py, %ly
  br i1 %ey, label %row, label %done
row:
  %xi = add i32 %gy, %px
  %xw = zext i32 %xi to i64
  %o1 = getelementptr inbounds i32, ptr addrspace(1) %b0, i64 %xw
  %a1 = atomicrmw add ptr addrspace(1) %o1, i32 1 monotonic, align 4
  br label %done
done:
  ret void
}

!air.kernel = !{!0}
!air.version = !{!20}
!air.language_version = !{!21}
!0 = !{ptr @flat_edges, !1, !2}
!1 = !{}
!2 = !{!3, !5, !6}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"b0"}
!5 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"p"}
!6 = !{i32 2, !"air.threads_per_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"g"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// A direct dispatch of more than 2^32 threads (65,536 x 65,601: ragged, so not the exact split, and 65 rows past 2^32) runs on the
/// flat variant's 64-bit arm, and every thread of the last column and the last row runs once at
/// its own position, the rows past 2^32 included: the 32-bit count is taken only when the whole
/// dispatch fits it.
#[test]
fn a_direct_dispatch_past_2_32_threads_keeps_the_flat_variant_s_64_bit_count() {
    edges_counted_once([65_536, 65_601, 1], [32, 32, 1]);
}

/// The flat variant chooses its arm per workgroup: a workgroup whose last index `wx·N + N - 1`
/// is below 2^32 counts in 32 bits, a later one of the same dispatch in 64 bits. With a
/// threadgroup of 105 threads (7 x 5 x 3, not a power of two) the first 64-bit workgroup starts
/// inside a row, so the last column and the last row are each counted once only where both arms
/// meet exactly there.
#[test]
fn a_flat_dispatch_switches_to_its_64_bit_count_at_the_workgroup_that_passes_2_32() {
    edges_counted_once([65_536, 65_601, 1], [7, 5, 3]);
}

/// Dispatch [`FLAT_EDGES`] over `grid` (more than 2^32 threads) in threadgroups of `group`: it
/// runs on the flat variant and marks every thread of the last column and the last row once.
fn edges_counted_once(grid: [u32; 3], group: [u32; 3]) {
    on_device(|device, messages| {
        let library = device
            .new_library_with_air(&llvm_assemble_text(FLAT_EDGES).unwrap())
            .unwrap();
        let function = library.new_function_with_name("flat_edges").unwrap();
        let kernel = device
            .new_compute_pipeline_state(
                &ComputePipelineDescriptor::new(StageSource::Function(
                    function.function_descriptor(),
                )),
                &PipelineOptions::default(),
            )
            .unwrap();
        assert!(u64::from(grid[0]) * u64::from(grid[1]) > 1 << 32);
        let [gx, gy, _] = grid;
        let slots = u64::from(gy + gx);
        let out = buffer_with(device, &vec![0; slots as usize * 4]).unwrap();
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&kernel).unwrap();
        encoder.set_buffer(Some(&out), 0, 0).unwrap();
        encoder.dispatch_threads(size(grid), size(group)).unwrap();
        encoder.end_encoding().unwrap();
        complete(&command_buffer, &format!("past 2^32 {grid:?} {group:?}"));
        let flat = command_buffer
            .executions()
            .iter()
            .any(|execution| match execution.pipeline() {
                ExecutedPipeline::Compute(pipeline) => pipeline
                    .dependencies
                    .iter()
                    .any(|dependency| dependency.fact == RuntimeFact::FlatGrid),
                ExecutedPipeline::Render(_) => false,
            });
        assert!(flat, "{grid:?} {group:?}: not on the flat variant");
        let words = read_buffer(&out, 0, slots * 4).unwrap();
        let counts: Vec<u32> = words
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes(*w))
            .collect();
        let wrong: Vec<(usize, u32)> = counts
            .iter()
            .enumerate()
            .filter(|(_, &c)| c != 1)
            .map(|(i, &c)| (i, c))
            .take(8)
            .collect();
        assert!(
            wrong.is_empty(),
            "slots not counted once (first 8, rows then columns at {gy}): {wrong:?}"
        );
        let _ = messages.take_notices();
    });
}
