//! Executions of indirect command buffers' CPU-encoded compute commands
//! (`execute_commands_in_buffer`) on the device the cfg selects, under the validation layer:
//!
//! - Every committed public kernel case of buffers alone (with threadgroup lengths, an explicit
//!   imageblock or stage inputs) runs from an indirect command as the encoder's own dispatch of
//!   it runs: the same bytes (but those of the case's declared out-of-bounds reads, which Metal
//!   leaves undefined, zeroed in both as the Metal oracle records them), the same execution
//!   report, and the same compiled variant planned for the same shape.
//! - Commands run as their records stand at commit; a command Metal runs nothing for (no
//!   dispatch, an empty or oversized threadgroup) is left out and its neighbours run; an inheriting
//!   buffer runs the encoder's pipeline state and buffers.
//! - A buffer address resolves against what any encoder of the command buffer made resident,
//!   before the execution or after it; one in no resident resource misses, counted in the execution
//!   report; the segment runs between the encoder's commands, after what they wrote and before what
//!   they read, and its dispatches keep their place in the execution record.
//! - A kernel that calls through a visible-function table its argument buffer holds calls the
//!   table's functions from a command as from its dispatch: the live tables of the command's
//!   pipeline state are its heap's.
//! - What Metal's validation stops is refused at the call, changing nothing.

use super::degrade_notices::counts;
use super::host_memory::{host_page, no_copy, not_importable, Mapping};
use super::{
    bound_buffers, compute_descriptor, encode_case, in_shard, layer_reported_nothing,
    layer_reported_nothing_at_all, public_case, public_cases, read_buffer, validated_device,
    validated_device_with, wait_for_device, CaseResources, DeviceWait, Messages, PublicCase,
    ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::candidate::engine::{ComputeCase, ComputeOutput};
use crate::candidate::runtime_case::{CaseError, Committed};
use crate::case::{AuthoredCase, Stage};
use crate::executor_contract::out_of_bounds::metal_recorded;
use crate::executor_contract::stage_inputs;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::{Dispatch, StatusReport};
use metal2vulkan::metal::{
    BarrierScope, DispatchType, IndirectCommandType, RenderStages, ResourceUsage,
};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, LinkedFunctions, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, CommandBufferError, CommandBufferStatus, ComputeCommandEncoder,
    ComputePipelineState, DebugMessage, Degradation, Device, DeviceOptions, Error,
    ExecutedPipeline, Execution, Heap, HeapDescriptor, HeapType, HostMemory, IndirectCommand,
    IndirectCommandBuffer, IndirectCommandBufferDescriptor, IndirectCommandBufferProblem, Invalid,
    PipelineCreationCause, PipelineOptions, Problem, Region, RenderPassDescriptor, ResourceOptions,
    RuntimeGap, Size, VisibleFunctionTableDescriptor, VkResultCode,
};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;

/// A kernel whose threads each write their threadgroup's size and position, over a ragged grid.
const RAGGED: &str = "a-ragged-thread-grid-observes-each-threadgroups-size-and-position";

/// A kernel of dynamic threadgroup memory.
const WORD: &str = "threadgroup-word";

/// A kernel that reads buffer 0 and writes buffer 1.
const CHAIN: &str = "saturating-halving-and-wide-product-integers-at-their-edges";

/// A kernel whose `[[stage_in]]` layout's stride is `MTLBufferLayoutStrideDynamic`.
const DYNAMIC: &str = "stage-in-of-a-dynamic-stride-layout-of-an-executed-command-template";

/// The kind [`a_record_the_gpu_wrote_runs_the_state_its_id_named_as_it_was_written`] takes where
/// a host step cannot run what Metal runs.
pub(super) const RELEASED_STATES: [Degradation; 1] = [Degradation::ReleasedCommandState];

/// Run `test` on a validated device of the lane, then fail on any validation message.
fn on_device(test: impl FnOnce(&Device, &Messages)) {
    on_device_with(|_| {}, test);
}

/// [`on_device`] with the options `adjust` changes.
fn on_device_with(adjust: impl FnOnce(&mut DeviceOptions), test: impl FnOnce(&Device, &Messages)) {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), adjust).unwrap();
        test(&device, &messages);
    }
    layer_reported_nothing_at_all(&messages);
}

/// `case`'s pipeline state, made with `supportIndirectCommandBuffers` (as the oracle's
/// otherwise, `compute_descriptor`).
fn icb_state(device: &Device, case: &PublicCase) -> Result<ComputePipelineState, String> {
    let name = &case.case.name;
    let library = device
        .new_library_with_air(&case.bitcode)
        .map_err(|error| format!("{name}: library: {error}"))?;
    let function = library
        .new_function_with_name_constant_values(&case.compute.entry, &case.compute.constants)
        .map_err(|error| format!("{name}: function: {error}"))?;
    let mut descriptor = compute_descriptor(&function, LinkedFunctions::new(), &case.compute);
    descriptor.support_indirect_command_buffers = true;
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .map_err(|error| format!("{name}: pipeline state: {error}"))
}

/// A buffer of `count` commands of both dispatch kinds, any kernel buffer, nothing inherited;
/// with dynamic attribute strides where `strides`, for a case whose stage inputs of a dynamic
/// layout a command binds at their strides (`encode_command`).
fn icb(device: &Device, count: u64, strides: bool) -> IndirectCommandBuffer {
    let mut descriptor = IndirectCommandBufferDescriptor::new();
    descriptor.command_types = IndirectCommandType::CONCURRENT_DISPATCH
        .union(IndirectCommandType::CONCURRENT_DISPATCH_THREADS);
    descriptor.max_kernel_buffer_bind_count = 31;
    descriptor.support_dynamic_attribute_stride = strides;
    device
        .new_indirect_command_buffer(&descriptor, count, ResourceOptions::shared())
        .unwrap()
}

fn size([x, y, z]: [u32; 3]) -> Size {
    Size::new(x.into(), y.into(), z.into())
}

/// Encode `case`'s dispatch as command `index` of `icb` with `pipeline` (none where `icb`
/// inherits it) over `buffers` (none where it inherits them): what `encode_case` sets on an
/// encoder, set on the command. A buffer a stage input of a dynamic layout reads is bound at its
/// stride, as the encoder binds it: Metal's command reads a buffer set without one at stride
/// 2^32 - 1 there, which is no dispatch's answer.
fn encode_command(
    icb: &IndirectCommandBuffer,
    index: u64,
    pipeline: Option<&ComputePipelineState>,
    case: &ComputeCase,
    buffers: &BTreeMap<u32, Buffer>,
) -> Result<(), Error> {
    let command = icb.indirect_compute_command_at(index)?;
    if let Some(pipeline) = pipeline {
        command.set_compute_pipeline_state(pipeline)?;
    }
    for (&slot, buffer) in buffers {
        match case.attribute_strides.get(&slot) {
            Some(&stride) => {
                command.set_kernel_buffer_with_stride(Some(buffer), 0, stride.into(), slot)?
            }
            None => command.set_kernel_buffer(Some(buffer), 0, slot)?,
        }
    }
    for (&slot, &length) in &case.threadgroup_lengths {
        command.set_threadgroup_memory_length(length, slot)?;
    }
    if let Some([width, height]) = case.imageblock {
        command.set_imageblock_size(width.into(), height.into())?;
    }
    if let (Some(_), Dispatch::Threads { threads, .. }) = (&case.stage_input, case.dispatch) {
        let region = stage_inputs::region(case.stage_in_region, threads);
        let [x, y, z] = region.origin.map(u64::from);
        let [width, height, depth] = region.size.map(u64::from);
        command.set_stage_in_region(Region::new_3d(x, y, z, width, height, depth))?;
    }
    match case.dispatch {
        Dispatch::Threads {
            threads,
            threads_per_group,
        } => command.concurrent_dispatch_threads(size(threads), size(threads_per_group)),
        Dispatch::Threadgroups {
            groups,
            threads_per_group,
        } => command.concurrent_dispatch_threadgroups(size(groups), size(threads_per_group)),
        other => panic!("{other:?} is no dispatch of a command"),
    }
}

/// Make `buffers` and `resources`' argument-buffer buffers resident on `encoder`, for reading and
/// writing, as an indirect command's buffers must be.
fn make_resident(
    encoder: &mut ComputeCommandEncoder,
    buffers: &BTreeMap<u32, Buffer>,
    resources: &CaseResources,
) {
    let resident: Vec<_> = buffers
        .values()
        .chain(resources.argument_buffers.values())
        .map(Buffer::as_resource)
        .collect();
    encoder
        .use_resources(&resident, ResourceUsage::READ | ResourceUsage::WRITE)
        .unwrap();
}

/// Commit `command_buffer`, wait for it, and require it completed.
fn complete(command_buffer: &CommandBuffer) {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// The bytes `case`'s output selects of `buffers`.
fn output(case: &ComputeCase, buffers: &BTreeMap<u32, Buffer>) -> Vec<u8> {
    let ComputeOutput::Buffer {
        binding,
        offset,
        length,
    } = case.output
    else {
        panic!("not a buffer output: {:?}", case.output);
    };
    read_buffer(&buffers[&binding], offset, length).unwrap()
}

/// What one run observed: the output, the execution report, the execution record.
struct Run {
    output: Vec<u8>,
    report: StatusReport,
    executions: Vec<Execution>,
}

/// Why a run did not complete: the runtime's error at the dispatch or the commit, the command
/// buffer's error once it completed (with the driver's Vulkan error for the command whose
/// pipeline it did not make, `CommandBuffer::failed_pipeline`), its not completing within its
/// bound of the device's own time (`wait_for_device`), or anything else.
enum Failed {
    Runtime(Error),
    Command(Option<CommandBufferError>, Option<VkResultCode>),
    /// The command buffer did not complete within `bound` of the device's own time; `drained`,
    /// whether it then completed within as long again, which the run waited for so that the next
    /// run's work never queues behind its own (`run`).
    TimedOut {
        bound: Duration,
        drained: bool,
    },
    Other(String),
}

impl std::fmt::Display for Failed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Failed::Runtime(error) => write!(f, "{error}"),
            Failed::Command(error, _) => write!(f, "the command buffer failed: {error:?}"),
            Failed::TimedOut { bound, drained } => write!(
                f,
                "timed out: the command buffer did not complete within {} ms of the device's \
                 own time{}",
                bound.as_millis(),
                match drained {
                    true => "",
                    false => ", nor within as long again",
                }
            ),
            Failed::Other(why) => f.write_str(why),
        }
    }
}

impl From<String> for Failed {
    fn from(why: String) -> Failed {
        Failed::Other(why)
    }
}

/// The Vulkan error of a run that failed making the case's pipeline, in either of the runtime's
/// forms: the commit's `Error::Vulkan` where the commit made it (an indirect command's variants,
/// which the commit settles, `indirect_execute.rs`), or the command buffer's `PipelineCreation`
/// where its dispatch's making went on after the dispatch and failed, then failed again when the
/// recording made it once more (`pending_dispatch.rs`), naming that dispatch's pipeline with the
/// same error (`CommandBuffer::failed_pipeline`).
fn making_failed(failed: &Failed) -> Option<VkResultCode> {
    match failed {
        Failed::Runtime(Error::Vulkan(code)) => Some(*code),
        Failed::Command(
            Some(CommandBufferError::PipelineCreation(PipelineCreationCause::Vulkan(code))),
            Some(made),
        ) if made == code => Some(*code),
        _ => None,
    }
}

/// Whether the dispatch (`direct`, the device reporting `direct_reported`) and the command
/// (`indirect`, `indirect_reported`) failed alike making the case's pipeline: the same Vulkan
/// error from the runtime ([`making_failed`]), and the same messages from the device for each,
/// every one a pipeline compile's failure. The device's failure, which an indirect command neither
/// causes nor hides. The messages compare as sets: the dispatch's making, which failed once
/// after the dispatch, is made again by its recording (`pending_dispatch.rs`), so the device
/// reports the same failure once more for it.
fn compile_failed_alike(
    direct: &Failed,
    indirect: &Failed,
    direct_reported: &[DebugMessage],
    indirect_reported: &[DebugMessage],
) -> bool {
    let (Some(a), Some(b)) = (making_failed(direct), making_failed(indirect)) else {
        return false;
    };
    let within = |some: &[DebugMessage], all: &[DebugMessage]| {
        some.iter().all(|message| all.contains(message))
    };
    a == b
        && !direct_reported.is_empty()
        && !indirect_reported.is_empty()
        && within(direct_reported, indirect_reported)
        && within(indirect_reported, direct_reported)
        && direct_reported
            .iter()
            .all(|message| message.text.contains("pipeline compile failed"))
}

/// `case` run on `device` with `pipeline` (the case's state, `icb_state`): its buffers made
/// resident, then its dispatch from an indirect command when `indirect`, else as the encoder's
/// own dispatch; with what the wait for its command buffer saw, where the run got that far
/// ([`wait_for_device`]).
///
/// A command buffer that does not complete within `bound` of the device's own time is waited
/// for as long again before the run returns, so that the next run's work does not queue behind
/// it on the device (lavapipe runs one queue): its timeout fails its own run alone.
fn run(
    device: &Device,
    case: &PublicCase,
    pipeline: &ComputePipelineState,
    indirect: bool,
    bound: Duration,
) -> (Result<Run, Failed>, Option<DeviceWait>) {
    let committed = match commit(device, case, pipeline, indirect) {
        Ok(committed) => committed,
        Err(failed) => return (Err(failed), None),
    };
    let waited = wait_for_device(&committed.command_buffer, bound);
    if !waited.completed {
        let drained = wait_for_device(&committed.command_buffer, bound).completed;
        return (Err(Failed::TimedOut { bound, drained }), Some(waited));
    }
    (finish(committed, &case.compute), Some(waited))
}

/// The bound of the dispatch form's run of a census case ([`run`]), of the device's own time: a
/// guard against a hang alone. Generous, since how long a case's work takes is the device's (on
/// lavapipe one core runs it, of the kind and at the clock the host's scheduler gives it); with
/// the wait as long again after a timeout, within the test's own end (`.config/nextest.toml`), so
/// the census names a hung case itself.
const HANG: Duration = Duration::from_secs(300);

/// The bound of the command form's run of a census case ([`run`]), of the device's own time,
/// after the dispatch form's run waited as `direct`: the same work on the same device, so twice
/// the dispatch's own time, for the device's speed changing between the runs, and [`WAIT`] more;
/// [`WAIT`] where the dispatch reached no wait.
fn command_bound(direct: Option<DeviceWait>) -> Duration {
    let own = direct.map_or(Duration::ZERO, |waited| {
        waited.wall.saturating_sub(waited.queued)
    });
    own * 2 + WAIT
}

/// [`run`]'s command buffer of `case`, encoded and committed.
fn commit(
    device: &Device,
    case: &PublicCase,
    pipeline: &ComputePipelineState,
    indirect: bool,
) -> Result<Committed, Failed> {
    let compute = &case.compute;
    let resources = CaseResources::new(device, case)?;
    let buffers = bound_buffers(device, compute, &resources).map_err(|e| format!("{e:?}"))?;
    let queue = device.new_command_queue().map_err(|e| e.to_string())?;
    let command_buffer = queue.command_buffer().map_err(|e| e.to_string())?;
    let mut encoder = command_buffer
        .compute_command_encoder()
        .map_err(|e| e.to_string())?;
    make_resident(&mut encoder, &buffers, &resources);
    if indirect {
        let icb = icb(device, 1, !compute.attribute_strides.is_empty());
        encode_command(&icb, 0, Some(pipeline), compute, &buffers)
            .map_err(|e| format!("encode_command: {e}"))?;
        encoder
            .execute_commands_in_buffer(&icb, 0..1)
            .map_err(|e| format!("execute_commands_in_buffer: {e}"))?;
    } else {
        encode_case(&mut encoder, pipeline, compute, &buffers, &resources).map_err(
            |e| match e {
                CaseError::Runtime { error, .. } => Failed::Runtime(*error),
                other => Failed::Other(format!("{other:?}")),
            },
        )?;
    }
    encoder.end_encoding().map_err(|e| e.to_string())?;
    command_buffer.commit().map_err(Failed::Runtime)?;
    Ok(Committed {
        command_buffer,
        buffers,
        resources,
        attachments: Default::default(),
        executions: Vec::new(),
    })
}

/// What [`run`]'s `committed` command buffer, which completed, observed of `compute`.
fn finish(committed: Committed, compute: &ComputeCase) -> Result<Run, Failed> {
    let command_buffer = &committed.command_buffer;
    if command_buffer.status() != CommandBufferStatus::Completed {
        return Err(Failed::Command(
            command_buffer.error(),
            command_buffer
                .failed_pipeline()
                .and_then(|pipeline| match pipeline.error {
                    Error::Vulkan(code) => Some(code),
                    _ => None,
                }),
        ));
    }
    let report = command_buffer
        .execution_report()
        .ok_or_else(|| "no report".to_owned())?;
    let executions = command_buffer.executions();
    let output = committed.output(compute).map_err(|e| format!("{e:?}"))?;
    Ok(Run {
        output,
        report,
        executions,
    })
}

/// Whether `compute` binds buffers alone, which an indirect command's kernel buffers can hold
/// (their pointers' targets resident), and dispatches as a command does.
fn buffers_alone(compute: &ComputeCase) -> bool {
    compute.textures.is_empty()
        && compute.samplers.is_empty()
        && compute.argument_buffer_textures.is_empty()
        && compute.argument_buffer_samplers.is_empty()
        && compute.objects.is_empty()
        && compute.command_buffers.is_empty()
        && compute.function_tables.is_empty()
        && compute.acceleration_structures.is_empty()
        && compute.tile.is_none()
        && compute.buffers.keys().all(|&slot| slot < 31)
        && matches!(compute.output, ComputeOutput::Buffer { .. })
        && matches!(
            compute.dispatch,
            Dispatch::Threads { .. } | Dispatch::Threadgroups { .. }
        )
}

/// Whether two executions ran the same compiled variant, planned for the same shape. Only
/// executions over the same buffers compare so: a buffer's shape states where the device's
/// sub-allocator placed it (its length and offset in its `VkBuffer`'s space).
fn same_execution(a: &Execution, b: &Execution) -> bool {
    same_variant(a, b) && a.shape() == b.shape()
}

/// Whether two executions ran the same compiled variant.
fn same_variant(a: &Execution, b: &Execution) -> bool {
    match (a.pipeline(), b.pipeline()) {
        (ExecutedPipeline::Compute(a), ExecutedPipeline::Compute(b)) => Arc::ptr_eq(a, b),
        _ => false,
    }
}

/// Every committed public kernel case of buffers alone runs from an indirect command as the
/// encoder's own dispatch of it runs, on one pipeline state of the case: the same output (the
/// bytes of its declared out-of-bounds reads, undefined, zeroed in both, `metal_recorded`), the
/// same execution report, and one execution of the same compiled variant planned for the same
/// shape. A case the runtime refuses directly is refused the same from a command. A run whose
/// command buffer does not complete within its bound of the device's own time ([`run`]) fails as
/// timed out: the dispatch's, [`HANG`], against a hang alone; the command's, held to the
/// dispatch's own time ([`command_bound`]). A case whose dispatch timed out runs no command;
/// where the device still runs its work after as long again, the shard stops there and says how
/// many of its cases did not run. Each case's waits print with their wall time and the part of it
/// a thread of the process waited for a CPU.
///
/// The cases are split into `SHARDS` tests by their place in the cases' order, every
/// `SHARDS`th case to one, so nextest runs the pieces in parallel (one test of all of them took
/// about 325 s on lavapipe). A shard maps only its own cases: assembling and mapping each case
/// (`llvm-as`, the AIR interface, the Metal observation) took most of that time, and one shard that
/// mapped every case took 224 s alone.
fn every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch(shard: usize) {
    let cases = public_cases(in_shard(
        |case: &AuthoredCase| case.stage == Stage::Kernel && case.public_source.is_some(),
        shard,
        SHARDS,
    ))
    .unwrap();
    let messages = Messages::default();
    let (mut alike, mut refused, mut failed) = (Vec::new(), Vec::new(), Vec::new());
    let (mut compile_failed, mut timed_out) = (Vec::new(), Vec::new());
    let mut skipped = 0;
    // The case whose work the device still ran after its timeout, that timeout's bound, and how
    // many cases followed it.
    let mut stopped = None;
    let device_name;
    {
        let cache = ScratchCache::new().unwrap();
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        device_name = device.name().to_owned();
        for (at, (name, case)) in cases.iter().enumerate() {
            let Ok(case) = case else {
                skipped += 1;
                continue;
            };
            if !buffers_alone(&case.compute) {
                skipped += 1;
                continue;
            }
            let pipeline = match icb_state(&device, case) {
                Ok(pipeline) => pipeline,
                Err(why) => {
                    refused.push(why);
                    continue;
                }
            };
            let (direct, direct_waited) = run(&device, case, &pipeline, false, HANG);
            let direct_reported = messages.take();
            let (indirect, indirect_waited) = match timed(&direct) {
                true => (Err(Failed::Other("not run".to_owned())), None),
                false => run(&device, case, &pipeline, true, command_bound(direct_waited)),
            };
            let indirect_reported = messages.take();
            eprintln!(
                "{name}: {}; {}",
                clocks("the dispatch", direct_waited),
                clocks("the command", indirect_waited)
            );
            if let Some(bound) = still_running(&direct).or(still_running(&indirect)) {
                stopped = Some((name, bound, cases.len() - at - 1));
            }
            // The device's own failure making the pipeline, reported for both runs alike.
            let mut device_failure = false;
            match (direct, indirect) {
                (Ok(direct), Ok(indirect)) => {
                    let mut differs = Vec::new();
                    let defined = |output: &[u8]| metal_recorded(&case.case, output.to_vec());
                    if defined(&direct.output) != defined(&indirect.output) {
                        differs.push(format!(
                            "output {:02x?} from the command, {:02x?} from the dispatch",
                            indirect.output, direct.output
                        ));
                    }
                    if direct.report != indirect.report {
                        differs.push(format!(
                            "report {:?} from the command, {:?} from the dispatch",
                            indirect.report, direct.report
                        ));
                    }
                    let same = direct.executions.len() == 1
                        && indirect.executions.len() == 1
                        && same_execution(&direct.executions[0], &indirect.executions[0]);
                    if !same {
                        differs.push("another execution".to_owned());
                    }
                    if differs.is_empty() {
                        alike.push(name);
                    } else {
                        failed.push(format!("{name}: {}", differs.join("; ")));
                    }
                }
                (direct, indirect) if timed(&direct) || timed(&indirect) => {
                    timed_out.push(format!(
                        "{name}: the dispatch gave {:?}, the command {:?}",
                        direct.err().map(|e| e.to_string()),
                        indirect.err().map(|e| e.to_string())
                    ))
                }
                (Err(direct), Err(indirect))
                    if compile_failed_alike(
                        &direct,
                        &indirect,
                        &direct_reported,
                        &indirect_reported,
                    ) =>
                {
                    let texts: BTreeSet<&str> =
                        direct_reported.iter().map(|m| m.text.as_str()).collect();
                    let (dispatch, command) = (direct_reported.len(), indirect_reported.len());
                    compile_failed.push(format!(
                        "{name}: {direct} (from the command: {indirect}); the device reported \
                         {dispatch} times for the dispatch and {command} for the command \
                         {texts:?}"
                    ));
                    device_failure = true;
                }
                (Err(direct), Err(indirect)) => {
                    refused.push(format!("{name}: {direct} (from the command: {indirect})"))
                }
                (direct, indirect) => failed.push(format!(
                    "{name}: the dispatch gave {:?}, the command {:?}",
                    direct.err().map(|e| e.to_string()),
                    indirect.err().map(|e| e.to_string())
                )),
            }
            let reported = [direct_reported, indirect_reported].concat();
            if !device_failure && !reported.is_empty() {
                failed.push(format!(
                    "{name}: the validation layer reported {reported:#?}"
                ));
            }
            if stopped.is_some() {
                break;
            }
        }
    }
    let leaked = messages.take();
    eprintln!(
        "runtime indirect commands on {device_name}, shard {shard} of {SHARDS}: {} match their \
         own dispatch, {} refused alike, {} failed to compile alike on the device, {} timed out, \
         {} failed, of {} cases of buffers alone ({skipped} others)",
        alike.len(),
        refused.len(),
        compile_failed.len(),
        timed_out.len(),
        failed.len(),
        cases.len() - skipped - stopped.map_or(0, |(_, _, after)| after)
    );
    if let Some((name, bound, after)) = stopped {
        eprintln!(
            "stopped at {name}: the device still ran its work after twice {} ms of its own \
             time; the {after} cases after it in the shard did not run",
            bound.as_millis()
        );
    }
    for why in &timed_out {
        eprintln!("timed out: {why}");
    }
    for why in &refused {
        eprintln!("refused: {why}");
    }
    if !compile_failed.is_empty() {
        eprintln!(
            "compile failed alike: the device's own failure making the pipeline; \
             docs/FAST-PATHS.md's known gaps record authored-four-iteration-loop's (3cf0a3f0; \
             Apple's compiler, on MoltenVK)"
        );
    }
    for why in &compile_failed {
        eprintln!("compile failed alike: {why}");
    }
    assert!(
        failed.is_empty() && timed_out.is_empty() && leaked.is_empty(),
        "failed:\n{}\ntimed out:\n{}\nat device destruction: {leaked:#?}",
        failed.join("\n"),
        timed_out.join("\n")
    );
    layer_reported_nothing(
        &messages,
        "every case's commands were valid and left nothing behind",
    );
    assert!(!alike.is_empty(), "no case ran alike");
}

/// Whether `run` timed out ([`Failed::TimedOut`]).
fn timed(run: &Result<Run, Failed>) -> bool {
    matches!(run, Err(Failed::TimedOut { .. }))
}

/// The bound `run` timed out at, where the device still ran its work after as long again.
fn still_running(run: &Result<Run, Failed>) -> Option<Duration> {
    match run {
        Err(Failed::TimedOut {
            bound,
            drained: false,
        }) => Some(*bound),
        _ => None,
    }
}

/// What `form`'s wait for its command buffer saw ([`wait_for_device`]): its wall time and the
/// part of it that a thread of the process waited for a CPU.
fn clocks(form: &str, waited: Option<DeviceWait>) -> String {
    match waited {
        Some(waited) => format!(
            "{form} waited {} ms, {} ms of it for a CPU{}",
            waited.wall.as_millis(),
            waited.queued.as_millis(),
            match waited.completed {
                true => "",
                false => ", and timed out",
            }
        ),
        None => format!("{form} reached no wait"),
    }
}

/// The pieces `every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch` is split into. At
/// 8 a piece took 31-93 s on the M4 under the validation layer, past the Mac lane's 60 s a test; at
/// 32, mostly 9-17 s, but five pieces took 22-54 s in a whole lane's run on a busy mini. 64
/// halves each piece.
const SHARDS: usize = 64;

census_shards!(
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch, SHARDS;
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_0 = 0,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_1 = 1,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_2 = 2,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_3 = 3,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_4 = 4,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_5 = 5,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_6 = 6,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_7 = 7,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_8 = 8,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_9 = 9,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_10 = 10,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_11 = 11,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_12 = 12,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_13 = 13,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_14 = 14,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_15 = 15,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_16 = 16,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_17 = 17,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_18 = 18,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_19 = 19,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_20 = 20,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_21 = 21,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_22 = 22,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_23 = 23,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_24 = 24,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_25 = 25,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_26 = 26,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_27 = 27,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_28 = 28,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_29 = 29,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_30 = 30,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_31 = 31,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_32 = 32,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_33 = 33,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_34 = 34,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_35 = 35,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_36 = 36,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_37 = 37,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_38 = 38,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_39 = 39,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_40 = 40,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_41 = 41,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_42 = 42,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_43 = 43,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_44 = 44,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_45 = 45,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_46 = 46,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_47 = 47,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_48 = 48,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_49 = 49,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_50 = 50,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_51 = 51,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_52 = 52,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_53 = 53,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_54 = 54,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_55 = 55,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_56 = 56,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_57 = 57,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_58 = 58,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_59 = 59,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_60 = 60,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_61 = 61,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_62 = 62,
    every_case_of_buffers_alone_runs_from_a_command_as_its_dispatch_63 = 63,
);

/// A command runs as its record stands when the command buffer is committed: one written after
/// the execution was encoded and the encoder ended runs, and the reset command beside it runs
/// nothing (`exec_cpu_written_after_encode`).
#[test]
fn commands_run_as_their_records_stand_at_commit() {
    on_device(|device, _| {
        let case = public_case(RAGGED).unwrap();
        let pipeline = icb_state(device, &case).unwrap();
        let resources = CaseResources::new(device, &case).unwrap();
        let buffers = bound_buffers(device, &case.compute, &resources).unwrap();
        let initial = output(&case.compute, &buffers);
        let icb = icb(device, 2, false);
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        make_resident(&mut encoder, &buffers, &resources);
        encoder.execute_commands_in_buffer(&icb, 0..2).unwrap();
        encoder.end_encoding().unwrap();
        assert!(command_buffer.executions().is_empty());
        assert_eq!(output(&case.compute, &buffers), initial);
        encode_command(&icb, 1, Some(&pipeline), &case.compute, &buffers).unwrap();
        complete(&command_buffer);
        assert_eq!(output(&case.compute, &buffers), case.metal_output);
        assert_eq!(command_buffer.executions().len(), 1);
        assert_eq!(
            command_buffer.execution_report(),
            Some(StatusReport::default())
        );
    });
}

/// The kind a host execution of a command holding a nil buffer set at an offset takes
/// ([`a_nil_buffer_at_an_offset_binds_none_and_is_announced`]).
pub(super) const NIL_OFFSETS: [Degradation; 1] = [Degradation::IndirectNilBufferOffset];

/// A command holding a nil buffer set at an offset whose low 32 bits are not 0 runs, binding no
/// buffer there, and announces `indirect-nil-buffer-offset` once a device: Metal's kernel sees
/// the bits as the buffer's address, the host execution's sees none. A nil
/// buffer at offset 0 announces nothing.
#[test]
fn a_nil_buffer_at_an_offset_binds_none_and_is_announced() {
    on_device(|device, messages| {
        let nil_offsets = |messages: &Messages| -> Vec<Degradation> {
            let notices = messages.take_notices();
            counts(&notices);
            notices
                .into_iter()
                .filter_map(|message| message.degradation)
                .map(|notice| notice.degradation)
                .filter(|kind| NIL_OFFSETS.contains(kind))
                .collect()
        };
        messages.take_notices();
        let case = public_case(RAGGED).unwrap();
        let pipeline = icb_state(device, &case).unwrap();
        let icb = icb(device, 2, false);
        for (index, offset) in [(0, 0), (1, (1 << 32) + 0x30)] {
            // Fresh buffers for each run: the kernel adds into its output.
            let resources = CaseResources::new(device, &case).unwrap();
            let buffers = bound_buffers(device, &case.compute, &resources).unwrap();
            encode_command(&icb, index, Some(&pipeline), &case.compute, &buffers).unwrap();
            let command = icb.indirect_compute_command_at(index).unwrap();
            command.set_kernel_buffer(None, offset, 30).unwrap();
            let queue = device.new_command_queue().unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            make_resident(&mut encoder, &buffers, &resources);
            encoder
                .execute_commands_in_buffer(&icb, index..index + 1)
                .unwrap();
            encoder.end_encoding().unwrap();
            complete(&command_buffer);
            assert_eq!(output(&case.compute, &buffers), case.metal_output);
            let announced = if offset == 0 {
                vec![]
            } else {
                NIL_OFFSETS.to_vec()
            };
            assert_eq!(nil_offsets(messages), announced, "offset {offset:#x}");
        }
    });
}

/// A `Private` buffer's commands are the CPU's to encode as a `Shared` one's: a command encoded
/// on it reads back as set and runs as the encoder's own dispatch, as plain Metal encodes, reads
/// back and runs it where its validation stops the call (`encode_private` and
/// `execute_private_cpu_encoded`, an M3 Max and an M4).
#[test]
fn a_private_buffer_s_commands_are_encoded_and_run() {
    on_device(|device, _| {
        let case = public_case(RAGGED).unwrap();
        let pipeline = icb_state(device, &case).unwrap();
        let resources = CaseResources::new(device, &case).unwrap();
        let buffers = bound_buffers(device, &case.compute, &resources).unwrap();
        let mut descriptor = IndirectCommandBufferDescriptor::new();
        descriptor.command_types = IndirectCommandType::CONCURRENT_DISPATCH
            .union(IndirectCommandType::CONCURRENT_DISPATCH_THREADS);
        descriptor.max_kernel_buffer_bind_count = 31;
        let icb = device
            .new_indirect_command_buffer(&descriptor, 1, ResourceOptions::private())
            .unwrap();
        encode_command(&icb, 0, Some(&pipeline), &case.compute, &buffers).unwrap();
        let read = icb.commands(0..1).unwrap();
        let [IndirectCommand::Compute(read)] = read.as_slice() else {
            panic!("one compute command: {read:?}");
        };
        assert!(read.pipeline.is_some() && read.dispatch.is_some());
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        make_resident(&mut encoder, &buffers, &resources);
        encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
        encoder.end_encoding().unwrap();
        complete(&command_buffer);
        assert_eq!(output(&case.compute, &buffers), case.metal_output);
    });
}

/// A command's reset keeps the pipeline state it executes with: encoded in full, reset, and its
/// buffers and dispatch encoded again with no pipeline state set, it runs the state it had, as
/// Metal's does (`host_reset_then_dispatch`, on the M3 Max and the M4),
/// while `commands` reads none, as Metal's getter reports none. A buffer reset with
/// `reset_with_range` is unmeasured and keeps nothing: the same command there runs nothing.
#[test]
fn a_reset_command_keeps_the_pipeline_it_runs() {
    on_device(|device, _| {
        let case = public_case(RAGGED).unwrap();
        let pipeline = icb_state(device, &case).unwrap();
        let resources = CaseResources::new(device, &case).unwrap();
        let sets: Vec<_> = (0..2)
            .map(|_| bound_buffers(device, &case.compute, &resources).unwrap())
            .collect();
        let initial = output(&case.compute, &sets[0]);
        let icb = icb(device, 2, false);
        for (index, buffers) in sets.iter().enumerate() {
            encode_command(&icb, index as u64, Some(&pipeline), &case.compute, buffers).unwrap();
        }
        icb.indirect_compute_command_at(0).unwrap().reset().unwrap();
        icb.reset_with_range(1..2).unwrap();
        for (index, buffers) in sets.iter().enumerate() {
            encode_command(&icb, index as u64, None, &case.compute, buffers).unwrap();
        }
        let read = icb.commands(0..1).unwrap();
        let [IndirectCommand::Compute(read)] = read.as_slice() else {
            panic!("one compute command: {read:?}");
        };
        assert!(read.pipeline.is_none() && read.dispatch.is_some());
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        for buffers in &sets {
            make_resident(&mut encoder, buffers, &resources);
        }
        encoder.execute_commands_in_buffer(&icb, 0..2).unwrap();
        encoder.end_encoding().unwrap();
        complete(&command_buffer);
        assert_eq!(output(&case.compute, &sets[0]), case.metal_output);
        assert_eq!(output(&case.compute, &sets[1]), initial);
    });
}

/// A command names its state without holding it, as Metal's does, so once the program releases
/// the state a state made after takes its ID; but the command holds what the state runs, and
/// runs it, as Metal's command the CPU encoded does: whether no state
/// takes the ID, a compute state of another interface ([`CHAIN`]'s kernel) does, or a render
/// state does. Nothing is announced, reading the command (`IndirectCommandBuffer::commands`)
/// included.
#[test]
fn a_command_naming_a_released_state_runs_its_program() {
    // What takes the released state's ID: nothing, a compute state, a render state.
    for taker in ["nothing", "compute", "render"] {
        on_device(|device, messages| {
            let case = public_case(RAGGED).unwrap();
            let resources = CaseResources::new(device, &case).unwrap();
            let buffers = bound_buffers(device, &case.compute, &resources).unwrap();
            let icb = icb(device, 1, false);
            {
                let pipeline = icb_state(device, &case).unwrap();
                encode_command(&icb, 0, Some(&pipeline), &case.compute, &buffers).unwrap();
            }
            let _compute = (taker == "compute")
                .then(|| icb_state(device, &public_case(CHAIN).unwrap()).unwrap());
            let _render = (taker == "render")
                .then(|| super::indirect_command_buffers::render_template_state(device, true));
            let read = icb.commands(0..1).unwrap();
            assert_eq!(read.len(), 1);
            let queue = device.new_command_queue().unwrap();
            // One execution: the case's kernel adds to its output, as its Metal output is one
            // run's.
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            make_resident(&mut encoder, &buffers, &resources);
            encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
            encoder.end_encoding().unwrap();
            complete(&command_buffer);
            assert_eq!(
                output(&case.compute, &buffers),
                case.metal_output,
                "taker {taker}: the released state's program ran"
            );
            let counted = counts(&messages.take_notices());
            assert!(
                !counted.contains_key(&Degradation::ReleasedCommandState),
                "taker {taker}: announced {counted:?}"
            );
        });
    }
}

/// A record the GPU wrote runs the state that held its ID as the GPU wrote it, released since or
/// not, as Metal's runs it (measured on an M3 Max: `released_programs.rs`); a host step, which runs
/// records the GPU may write, cannot tell the GPU's write of an ID from the CPU's. Command 0 is
/// encoded with a state, and copied to command 1 by a blit (the GPU writes it, so the buffer's
/// executions take the host step or the GPU alone), which then gets buffers of its own; the
/// program releases the state.
///
/// - No state takes the ID: both commands run the released state's program, unannounced.
/// - A compute state of another interface ([`CHAIN`]'s kernel) takes the ID: both still run the
///   released state's program, as Metal runs the command the CPU encoded and the record its copy
///   wrote before the new state took the ID, unannounced.
/// - The new state takes the ID and an encoder then makes the buffer resident for writing, so
///   the GPU may write command 0's record while the new state holds the ID: command 0 runs the
///   released state's program, as Metal runs the command the CPU encoded, announced once for two
///   executions (`Degradation::ReleasedCommandState`), as Metal runs the new state for a record
///   the GPU wrote then; and at each on a device whose options ask for the kind at each
///   occurrence.
#[test]
fn a_record_the_gpu_wrote_runs_the_state_its_id_named_as_it_was_written() {
    // `rewritable`: an encoder makes the buffer resident for writing after the new state took the
    // ID, and command 0 alone runs.
    let run = |device: &Device, messages: &Messages, taken: bool, rewritable: bool| {
        let case = public_case(RAGGED).unwrap();
        let resources = CaseResources::new(device, &case).unwrap();
        let sets: Vec<_> = (0..2)
            .map(|_| bound_buffers(device, &case.compute, &resources).unwrap())
            .collect();
        let icb = icb(device, 2, false);
        let queue = device.new_command_queue().unwrap();
        let id = {
            let pipeline = icb_state(device, &case).unwrap();
            encode_command(&icb, 0, Some(&pipeline), &case.compute, &sets[0]).unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.copy_indirect_command_buffer(&icb, 0..1, &icb, 1)
                .unwrap();
            blit.end_encoding().unwrap();
            complete(&command_buffer);
            encode_command(&icb, 1, None, &case.compute, &sets[1]).unwrap();
            pipeline.gpu_resource_id().0
        };
        let holder = taken.then(|| icb_state(device, &public_case(CHAIN).unwrap()).unwrap());
        if let Some(holder) = &holder {
            assert_eq!(
                holder.gpu_resource_id().0,
                id,
                "the state made after takes the ID"
            );
        }
        if rewritable {
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encoder
                .use_resource(icb.as_resource(), ResourceUsage::WRITE)
                .unwrap();
            encoder.end_encoding().unwrap();
            complete(&command_buffer);
        }
        // The case's kernel adds to its output, as its Metal output is one run's: each command's
        // output is read after its first execution.
        let execute = |command: usize| {
            for run in 0..2 {
                let command_buffer = queue.command_buffer().unwrap();
                let mut encoder = command_buffer.compute_command_encoder().unwrap();
                for buffers in &sets {
                    make_resident(&mut encoder, buffers, &resources);
                }
                let at = command as u64;
                encoder
                    .execute_commands_in_buffer(&icb, at..at + 1)
                    .unwrap();
                encoder.end_encoding().unwrap();
                complete(&command_buffer);
                if run == 0 {
                    assert_eq!(
                        output(&case.compute, &sets[command]),
                        case.metal_output,
                        "taken {taken}, rewritable {rewritable}: command {command} ran the \
                         released state's program"
                    );
                }
            }
            counts(&messages.take_notices())
                .get(&Degradation::ReleasedCommandState)
                .copied()
                .unwrap_or(0)
        };
        let commands = if rewritable { 0..1 } else { 0..2 };
        commands.map(execute).collect::<Vec<usize>>()
    };
    on_device(|device, messages| {
        assert_eq!(run(device, messages, false, false), [0, 0]);
    });
    on_device(|device, messages| {
        assert_eq!(run(device, messages, true, false), [0, 0]);
    });
    on_device(|device, messages| {
        assert_eq!(run(device, messages, true, true), [1]);
    });
    // Asked for at each occurrence: each execution announces it.
    let each = |options: &mut DeviceOptions| {
        options.announce_each = RELEASED_STATES.into_iter().collect();
    };
    on_device_with(each, |device, messages| {
        assert_eq!(run(device, messages, true, true), [2]);
    });
}

/// Commands Metal runs nothing for are left out, and their neighbours run (ruling a): a command
/// with no dispatch, one with no pipeline state, one with a threadgroup of no threads, and one
/// with more threads than the state's `max_total_threads_per_threadgroup`, each over buffers of
/// its own, which keep their bytes.
#[test]
fn commands_that_run_nothing_leave_their_neighbours_running() {
    on_device(|device, _| {
        let case = public_case(RAGGED).unwrap();
        let pipeline = icb_state(device, &case).unwrap();
        let resources = CaseResources::new(device, &case).unwrap();
        let sets: Vec<_> = (0..6)
            .map(|_| bound_buffers(device, &case.compute, &resources).unwrap())
            .collect();
        let initial = output(&case.compute, &sets[0]);
        let icb = icb(device, 6, false);
        for (index, buffers) in sets.iter().enumerate() {
            encode_command(&icb, index as u64, Some(&pipeline), &case.compute, buffers).unwrap();
        }
        let groups = Size::new(1, 1, 1);
        // 1: no dispatch (reset, then its buffers bound again).
        icb.reset_with_range(1..2).unwrap();
        let unset = icb.indirect_compute_command_at(1).unwrap();
        unset.set_compute_pipeline_state(&pipeline).unwrap();
        for (&slot, buffer) in &sets[1] {
            unset.set_kernel_buffer(Some(buffer), 0, slot).unwrap();
        }
        // 2: no pipeline state.
        icb.reset_with_range(2..3).unwrap();
        let stateless = icb.indirect_compute_command_at(2).unwrap();
        for (&slot, buffer) in &sets[2] {
            stateless.set_kernel_buffer(Some(buffer), 0, slot).unwrap();
        }
        stateless
            .concurrent_dispatch_threadgroups(groups, Size::new(4, 1, 1))
            .unwrap();
        // 3 and 4: no threads, and one past the state's limit.
        let limit = u64::from(pipeline.max_total_threads_per_threadgroup());
        for (index, threads) in [(3, Size::new(0, 1, 1)), (4, Size::new(limit + 1, 1, 1))] {
            icb.indirect_compute_command_at(index)
                .unwrap()
                .concurrent_dispatch_threadgroups(groups, threads)
                .unwrap();
        }
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        for buffers in &sets {
            make_resident(&mut encoder, buffers, &resources);
        }
        encoder.execute_commands_in_buffer(&icb, 0..6).unwrap();
        encoder.end_encoding().unwrap();
        complete(&command_buffer);
        for (index, buffers) in sets.iter().enumerate() {
            let expected = match index {
                0 | 5 => &case.metal_output,
                _ => &initial,
            };
            assert_eq!(&output(&case.compute, buffers), expected, "command {index}");
        }
        assert_eq!(command_buffer.executions().len(), 2);
    });
}

/// A buffer that inherits its pipeline state and buffers runs the encoder's, as they are at the
/// call (`exec_cpu_inherited`), with the command's own threadgroup lengths. With no pipeline state
/// set, a range that is not empty is refused, and an empty one, at the start or at the size, runs
/// nothing, as plain Metal's. Buffers the command sets anyway are kept but not bound, as plain
/// Metal's: theirs stay as they were.
#[test]
fn inheriting_commands_run_the_encoder_s_state() {
    on_device(|device, _| {
        let case = public_case(WORD).unwrap();
        let pipeline = icb_state(device, &case).unwrap();
        let resources = CaseResources::new(device, &case).unwrap();
        let buffers = bound_buffers(device, &case.compute, &resources).unwrap();
        let own = bound_buffers(device, &case.compute, &resources).unwrap();
        let initial = output(&case.compute, &own);
        let mut descriptor = IndirectCommandBufferDescriptor::new();
        descriptor.command_types = IndirectCommandType::CONCURRENT_DISPATCH
            .union(IndirectCommandType::CONCURRENT_DISPATCH_THREADS);
        descriptor.inherit_pipeline_state = true;
        descriptor.inherit_buffers = true;
        descriptor.max_kernel_buffer_bind_count = 31;
        let icb = device
            .new_indirect_command_buffer(&descriptor, 1, ResourceOptions::shared())
            .unwrap();
        encode_command(&icb, 0, None, &case.compute, &own).unwrap();
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.execute_commands_in_buffer(&icb, 0..0).unwrap();
        encoder.execute_commands_in_buffer(&icb, 1..1).unwrap();
        assert_eq!(
            encoder.execute_commands_in_buffer(&icb, 0..1).err(),
            Some(Error::Invalid(Invalid::new(
                "execute_commands_in_buffer",
                Problem::NoPipelineState
            )))
        );
        encoder.set_compute_pipeline_state(&pipeline).unwrap();
        for (&slot, buffer) in &buffers {
            encoder.set_buffer(Some(buffer), 0, slot).unwrap();
        }
        encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
        // Later bindings are the encoder's own: the execution keeps those of the call.
        for &slot in buffers.keys() {
            encoder.set_buffer(None, 0, slot).unwrap();
        }
        encoder.end_encoding().unwrap();
        complete(&command_buffer);
        assert_eq!(output(&case.compute, &buffers), case.metal_output);
        assert_eq!(output(&case.compute, &own), initial);
        assert_eq!(
            command_buffer.execution_report(),
            Some(StatusReport::default())
        );
    });
}

/// A command's kernel buffer set without a stride, where the pipeline's stage-in layout of
/// `MTLBufferLayoutStrideDynamic` reads it, on a buffer that supports dynamic attribute strides,
/// is read at stride `2^32 - 1`, as plain Metal's fetch reads it (the record's 32-bit stride word;
/// the stride form at `attributeStride:0xFFFFFFFF` read the same on an M3 Max under macOS 27.0.1):
/// the plain setter's command and the stride form's at `2^32 - 1` commit, complete and leave the
/// same output, which the case's own stride's does not.
#[test]
fn a_command_s_unstrided_stage_in_buffer_reads_at_the_record_s_stride() {
    on_device(|device, _| {
        let case = public_case(DYNAMIC).unwrap();
        let index = *case
            .compute
            .attribute_strides
            .keys()
            .next()
            .expect("a dynamic layout");
        let pipeline = icb_state(device, &case).unwrap();
        let run = |bind: &dyn Fn(&metal2vulkan_runtime::IndirectComputeCommand, &Buffer)| {
            let resources = CaseResources::new(device, &case).unwrap();
            let buffers = bound_buffers(device, &case.compute, &resources).unwrap();
            let icb = icb(device, 1, true);
            encode_command(&icb, 0, Some(&pipeline), &case.compute, &buffers).unwrap();
            bind(
                &icb.indirect_compute_command_at(0).unwrap(),
                &buffers[&index],
            );
            let queue = device.new_command_queue().unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            make_resident(&mut encoder, &buffers, &resources);
            encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
            encoder.end_encoding().unwrap();
            complete(&command_buffer);
            output(&case.compute, &buffers)
        };
        let plain = run(&|command, buffer| {
            command.set_kernel_buffer(Some(buffer), 0, index).unwrap();
        });
        let wide = run(&|command, buffer| {
            command
                .set_kernel_buffer_with_stride(Some(buffer), 0, 0xFFFF_FFFF, index)
                .unwrap();
        });
        let own = run(&|_, _| {});
        assert_eq!(plain, wide);
        assert_eq!(own, case.metal_output);
        assert_ne!(plain, own);
    });
}

/// A command's buffer address in no resource the command buffer's encoders made resident, and
/// in no buffer of its own (a buffer placed in a heap no encoder of the command buffer used, which
/// Metal's M4 did not reach either), misses: the slot reads zero and drops writes, so the buffer
/// keeps its bytes, and the execution report counts one miss per slot.
#[test]
fn addresses_outside_the_resident_resources_miss() {
    on_device(|device, _| {
        let case = public_case(RAGGED).unwrap();
        let pipeline = icb_state(device, &case).unwrap();
        let resources = CaseResources::new(device, &case).unwrap();
        let own = bound_buffers(device, &case.compute, &resources).unwrap();
        let (_heap, buffers) = placed(device, &own);
        let initial = output(&case.compute, &buffers);
        let icb = icb(device, 1, false);
        encode_command(&icb, 0, Some(&pipeline), &case.compute, &buffers).unwrap();
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
        encoder.end_encoding().unwrap();
        complete(&command_buffer);
        assert_eq!(output(&case.compute, &buffers), initial);
        let report = command_buffer.execution_report().unwrap();
        assert!(report.resident_miss);
        assert_eq!(report.resident_misses, buffers.len() as u32);
    });
}

/// Residency is the command buffer's (the consumer's report on an M3 Max, and our probe on
/// an M4, every run): a command's buffers that only an earlier encoder, or a later compute or
/// render encoder, made resident run as those its own encoder made resident, with no miss; and
/// so do buffers of their own that no encoder made resident.
#[test]
fn commands_reach_what_any_encoder_of_their_command_buffer_made_resident() {
    on_device(|device, _| {
        let case = public_case(RAGGED).unwrap();
        let pipeline = icb_state(device, &case).unwrap();
        let queue = device.new_command_queue().unwrap();
        let run = |when: &str| {
            let resources = CaseResources::new(device, &case).unwrap();
            let buffers = bound_buffers(device, &case.compute, &resources).unwrap();
            let icb = icb(device, 1, false);
            encode_command(&icb, 0, Some(&pipeline), &case.compute, &buffers).unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let compute = || {
                let mut encoder = command_buffer.compute_command_encoder().unwrap();
                make_resident(&mut encoder, &buffers, &resources);
                encoder.end_encoding().unwrap();
            };
            if when == "earlier compute" {
                compute();
            }
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            if when == "same" {
                make_resident(&mut encoder, &buffers, &resources);
            }
            encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
            encoder.end_encoding().unwrap();
            match when {
                "later compute" => compute(),
                "later render" => {
                    let mut pass = RenderPassDescriptor::new();
                    pass.render_target_width = 1;
                    pass.render_target_height = 1;
                    pass.default_raster_sample_count = 1;
                    let mut later = command_buffer.render_command_encoder(&pass).unwrap();
                    let resident: Vec<_> = buffers
                        .values()
                        .chain(resources.argument_buffers.values())
                        .map(Buffer::as_resource)
                        .collect();
                    later
                        .use_resources(
                            &resident,
                            ResourceUsage::READ | ResourceUsage::WRITE,
                            RenderStages::VERTEX,
                        )
                        .unwrap();
                    later.end_encoding().unwrap();
                }
                _ => {}
            }
            complete(&command_buffer);
            let report = command_buffer.execution_report().unwrap();
            (output(&case.compute, &buffers), report.resident_misses)
        };
        let same = run("same");
        assert_eq!(same.1, 0);
        // A buffer of its own is reached though no encoder made it resident ("none").
        for when in ["earlier compute", "later compute", "later render", "none"] {
            assert_eq!(run(when), same, "{when}");
        }
    });
}

/// `buffers`, each copied into a buffer placed in a `Shared` automatic heap of their own: the
/// heap, and the placed buffers by the same slots.
fn placed(device: &Device, buffers: &BTreeMap<u32, Buffer>) -> (Heap, BTreeMap<u32, Buffer>) {
    let size = buffers
        .values()
        .map(|buffer| buffer.length().next_multiple_of(65536))
        .sum::<u64>()
        + 65536;
    let heap = device
        .new_heap(&HeapDescriptor::new(
            HeapType::Automatic,
            size,
            ResourceOptions::shared(),
        ))
        .unwrap();
    let placed = buffers
        .iter()
        .map(|(&index, buffer)| {
            let length = buffer.length();
            let bytes = read_buffer(buffer, 0, length).unwrap();
            let into = heap.new_buffer(length, ResourceOptions::shared()).unwrap();
            let contents = into.contents().expect("a Shared buffer");
            // SAFETY: the buffer holds `length` bytes at `contents`, and no command uses them.
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len())
            };
            (index, into)
        })
        .collect();
    (heap, placed)
}

/// A command's buffers placed in a heap reach where an encoder of the command buffer used the
/// heap, earlier or later (an M4: reached, every run), and the execution report
/// counts no miss.
#[test]
fn commands_reach_a_heap_any_encoder_of_their_command_buffer_used() {
    on_device(|device, _| {
        let case = public_case(RAGGED).unwrap();
        let pipeline = icb_state(device, &case).unwrap();
        let queue = device.new_command_queue().unwrap();
        let resources = CaseResources::new(device, &case).unwrap();
        let own = bound_buffers(device, &case.compute, &resources).unwrap();
        let direct = {
            let icb = icb(device, 1, false);
            encode_command(&icb, 0, Some(&pipeline), &case.compute, &own).unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            make_resident(&mut encoder, &own, &resources);
            encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
            encoder.end_encoding().unwrap();
            complete(&command_buffer);
            output(&case.compute, &own)
        };
        for when in ["earlier", "later"] {
            let fresh = bound_buffers(device, &case.compute, &resources).unwrap();
            let (heap, buffers) = placed(device, &fresh);
            let icb = icb(device, 1, false);
            encode_command(&icb, 0, Some(&pipeline), &case.compute, &buffers).unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let use_heap = || {
                let mut encoder = command_buffer.compute_command_encoder().unwrap();
                encoder.use_heap(&heap).unwrap();
                encoder.end_encoding().unwrap();
            };
            if when == "earlier" {
                use_heap();
            }
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
            encoder.end_encoding().unwrap();
            if when == "later" {
                use_heap();
            }
            complete(&command_buffer);
            assert_eq!(output(&case.compute, &buffers), direct, "{when}");
            let report = command_buffer.execution_report().unwrap();
            assert_eq!(report.resident_misses, 0, "{when}");
        }
    });
}

/// `buffers`, each copied into a no-copy buffer over host pages of its own: the pages, which must
/// outlive the buffers, and the buffers by the same slots; `None` where the venue refuses the
/// import.
fn no_copies(
    device: &Device,
    buffers: &BTreeMap<u32, Buffer>,
) -> Option<(Vec<Mapping>, BTreeMap<u32, Buffer>)> {
    let mut pages = Vec::new();
    let mut copies = BTreeMap::new();
    for (&index, buffer) in buffers {
        let bytes = read_buffer(buffer, 0, buffer.length()).unwrap();
        let len = bytes.len().max(1).next_multiple_of(host_page());
        let memory = Mapping::anonymous(len);
        memory.write(0, &bytes);
        // SAFETY: the caller keeps the mapping until the buffer is gone.
        let copy = unsafe {
            no_copy(
                device,
                memory.pointer(),
                len,
                ResourceOptions::shared(),
                None,
            )
        };
        let copy = match copy {
            Err(error) if not_importable(&error) => {
                eprintln!("NO-COPY-REFUSED {error:?}: nothing to run");
                return None;
            }
            copy => copy.unwrap(),
        };
        pages.push(memory);
        copies.insert(index, copy);
    }
    Some((pages, copies))
}

/// A command's buffers that are no-copy buffers no command used are not reached, as on Metal's
/// M3 Max and M4 (every run; the consumer's `compute_indirect_command_buffer`): each slot misses,
/// as the execution report counts. Made resident by the command's encoder they are reached, and
/// from then on though no encoder names them.
#[test]
fn commands_reach_a_no_copy_buffer_once_a_command_used_it() {
    on_device(|device, _| {
        let case = public_case(RAGGED).unwrap();
        let pipeline = icb_state(device, &case).unwrap();
        let queue = device.new_command_queue().unwrap();
        let resources = CaseResources::new(device, &case).unwrap();
        let own = bound_buffers(device, &case.compute, &resources).unwrap();
        let direct = {
            let icb = icb(device, 1, false);
            encode_command(&icb, 0, Some(&pipeline), &case.compute, &own).unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            make_resident(&mut encoder, &own, &resources);
            encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
            encoder.end_encoding().unwrap();
            complete(&command_buffer);
            output(&case.compute, &own)
        };
        let fresh = bound_buffers(device, &case.compute, &resources).unwrap();
        let Some((pages, buffers)) = no_copies(device, &fresh) else {
            return;
        };
        let icb = icb(device, 1, false);
        encode_command(&icb, 0, Some(&pipeline), &case.compute, &buffers).unwrap();
        let run = |of: &BTreeMap<u32, Buffer>, resident: bool| {
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            if resident {
                make_resident(&mut encoder, of, &resources);
            }
            encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
            encoder.end_encoding().unwrap();
            complete(&command_buffer);
            command_buffer.execution_report().unwrap().resident_misses
        };
        let before = output(&case.compute, &buffers);
        assert_ne!(before, direct);
        assert_eq!(
            run(&buffers, false),
            buffers.len() as u32,
            "no command used them"
        );
        assert_eq!(
            output(&case.compute, &buffers),
            before,
            "no command used them"
        );
        assert_eq!(run(&buffers, true), 0, "made resident");
        assert_eq!(output(&case.compute, &buffers), direct, "made resident");
        // Their bytes as they were, then the command again, no encoder naming them.
        for ((buffer, memory), original) in buffers.values().zip(&pages).zip(fresh.values()) {
            memory.write(
                0,
                &read_buffer(original, 0, buffer.length().min(original.length())).unwrap(),
            );
        }
        assert_eq!(output(&case.compute, &buffers), before);
        assert_eq!(
            run(&buffers, false),
            0,
            "an earlier command buffer used them"
        );
        assert_eq!(
            output(&case.compute, &buffers),
            direct,
            "an earlier command buffer used them"
        );
        drop(icb);
        drop(buffers);
        drop(pages);
    });
}

/// A command's no-copy buffers that a command used and that were then rebound to other pages are
/// not reached until a command uses them again, as on Metal's M3 Max (the consumer's
/// measurement, three runs alike: an indirect command's kernel buffer used, then detached and
/// replaced, reads 0 and drops writes): each slot misses, as the execution report counts, and
/// made resident again they are reached over the new pages.
#[test]
fn commands_reach_a_rebound_no_copy_buffer_once_a_command_used_it_again() {
    on_device(|device, _| {
        let case = public_case(RAGGED).unwrap();
        let pipeline = icb_state(device, &case).unwrap();
        let queue = device.new_command_queue().unwrap();
        let resources = CaseResources::new(device, &case).unwrap();
        let own = bound_buffers(device, &case.compute, &resources).unwrap();
        let direct = {
            let icb = icb(device, 1, false);
            encode_command(&icb, 0, Some(&pipeline), &case.compute, &own).unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            make_resident(&mut encoder, &own, &resources);
            encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
            encoder.end_encoding().unwrap();
            complete(&command_buffer);
            output(&case.compute, &own)
        };
        let fresh = bound_buffers(device, &case.compute, &resources).unwrap();
        let Some((pages, buffers)) = no_copies(device, &fresh) else {
            return;
        };
        let icb = icb(device, 1, false);
        encode_command(&icb, 0, Some(&pipeline), &case.compute, &buffers).unwrap();
        let run = |resident: bool| {
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            if resident {
                make_resident(&mut encoder, &buffers, &resources);
            }
            encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
            encoder.end_encoding().unwrap();
            complete(&command_buffer);
            command_buffer.execution_report().unwrap().resident_misses
        };
        // Each buffer used, then given new pages holding its bytes as they were.
        let before = output(&case.compute, &buffers);
        assert_eq!(run(true), 0, "made resident");
        let mut new_pages = Vec::new();
        for (buffer, original) in buffers.values().zip(fresh.values()) {
            let memory = Mapping::anonymous(buffer.length() as usize);
            memory.write(
                0,
                &read_buffer(original, 0, buffer.length().min(original.length())).unwrap(),
            );
            // SAFETY: the mapping outlives the buffer, dropped first, and no other thread uses
            // the buffer.
            unsafe { buffer.rebind_bytes_no_copy(HostMemory::new(memory.pointer()), None, None) }
                .unwrap();
            new_pages.push(memory);
        }
        assert_eq!(output(&case.compute, &buffers), before);
        assert_eq!(
            run(false),
            buffers.len() as u32,
            "rebound since a command used them"
        );
        assert_eq!(
            output(&case.compute, &buffers),
            before,
            "rebound since a command used them"
        );
        assert_eq!(run(true), 0, "made resident again");
        assert_eq!(
            output(&case.compute, &buffers),
            direct,
            "the new pages, made resident again"
        );
        drop(icb);
        drop(buffers);
        drop((pages, new_pages));
    });
}

/// What Metal's validation stops is refused at the call, changing nothing: a range that ends
/// past the size. An empty range runs nothing, and so does a buffer of no command types, as
/// plain Metal's, directly or through a range in a buffer.
#[test]
fn ranges_past_the_size_are_refused() {
    on_device(|device, _| {
        let icb = icb(device, 2, false);
        let none = device
            .new_indirect_command_buffer(
                &IndirectCommandBufferDescriptor::new(),
                2,
                ResourceOptions::shared(),
            )
            .unwrap();
        let range = super::buffer_with(device, &[0, 0, 0, 0, 2, 0, 0, 0]).unwrap();
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        assert_eq!(
            encoder.execute_commands_in_buffer(&icb, 1..3).err(),
            Some(Error::Invalid(Invalid::new(
                "execute_commands_in_buffer",
                Problem::IndirectCommandBuffer(IndirectCommandBufferProblem::Range {
                    location: 1,
                    length: 2,
                    size: 2,
                })
            )))
        );
        encoder.execute_commands_in_buffer(&icb, 2..2).unwrap();
        encoder.execute_commands_in_buffer(&none, 0..2).unwrap();
        encoder
            .execute_commands_in_buffer_indirect(&none, &range, 0)
            .unwrap();
        encoder.end_encoding().unwrap();
        complete(&command_buffer);
        assert!(command_buffer.executions().is_empty());
    });
}

/// `case`'s buffers on `device`, holding the case's bytes, with `first` and `second` at slots 0
/// and 1.
fn chain_set(
    device: &Device,
    case: &PublicCase,
    first: &Buffer,
    second: &Buffer,
) -> BTreeMap<u32, Buffer> {
    let mut set: BTreeMap<u32, Buffer> = case
        .compute
        .buffers
        .iter()
        .map(|(&slot, bytes)| (slot, super::buffer_with(device, bytes).unwrap()))
        .collect();
    set.insert(0, first.clone());
    set.insert(1, second.clone());
    set
}

/// The segment runs where it was encoded: after the encoder's earlier dispatch and before its
/// later one. A chain of three dispatches each reading what the one before wrote (the middle one
/// from a command) gives what three command buffers of one dispatch each give, in a serial
/// encoder, and in a concurrent one with the program's barriers between them. The execution
/// record holds the command's dispatch between the encoder's two, each of the one variant (their
/// buffers differ, so their shapes may: [`same_execution`]).
#[test]
fn a_segment_runs_between_the_encoder_s_commands() {
    on_device(|device, messages| {
        let case = public_case(CHAIN).unwrap();
        let pipeline = icb_state(device, &case).unwrap();
        let queue = device.new_command_queue().unwrap();
        let length = case.compute.buffers[&1].len() as u64;
        let none = CaseResources::default();
        // Three runs alone, each over the last one's output.
        let mut expected = Vec::new();
        let mut input = super::buffer_with(device, &case.compute.buffers[&0]).unwrap();
        for _ in 0..3 {
            let output = super::buffer_with(device, &case.compute.buffers[&1]).unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            let set = chain_set(device, &case, &input, &output);
            encode_case(&mut encoder, &pipeline, &case.compute, &set, &none).unwrap();
            encoder.end_encoding().unwrap();
            complete(&command_buffer);
            expected.push(read_buffer(&output, 0, length).unwrap());
            input = output;
        }
        for dispatch_type in [DispatchType::Serial, DispatchType::Concurrent] {
            // in -> a -> b -> c.
            let mut chain = vec![super::buffer_with(device, &case.compute.buffers[&0]).unwrap()];
            for _ in 0..3 {
                chain.push(super::buffer_with(device, &case.compute.buffers[&1]).unwrap());
            }
            let sets: Vec<_> = (0..3)
                .map(|step| chain_set(device, &case, &chain[step], &chain[step + 1]))
                .collect();
            let icb = icb(device, 1, false);
            encode_command(&icb, 0, Some(&pipeline), &case.compute, &sets[1]).unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer
                .compute_command_encoder_with_dispatch_type(dispatch_type)
                .unwrap();
            // Resident before the first dispatch, so all three plan for one shape.
            make_resident(&mut encoder, &sets[1], &none);
            let barrier = |encoder: &mut ComputeCommandEncoder| {
                if dispatch_type == DispatchType::Concurrent {
                    encoder.memory_barrier_with_scope(BarrierScope::BUFFERS);
                }
            };
            encode_case(&mut encoder, &pipeline, &case.compute, &sets[0], &none).unwrap();
            barrier(&mut encoder);
            encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
            barrier(&mut encoder);
            encode_case(&mut encoder, &pipeline, &case.compute, &sets[2], &none).unwrap();
            encoder.end_encoding().unwrap();
            assert_eq!(command_buffer.executions().len(), 2);
            complete(&command_buffer);
            layer_reported_nothing(messages, &format!("{dispatch_type:?}"));
            for (step, expected) in expected.iter().enumerate() {
                let written = read_buffer(&chain[step + 1], 0, length).unwrap();
                assert_eq!(&written, expected, "{dispatch_type:?} step {step}");
            }
            let executions = command_buffer.executions();
            assert_eq!(executions.len(), 3);
            assert!(
                same_variant(&executions[0], &executions[1]),
                "{dispatch_type:?}: the command's execution is not the encoder's first variant"
            );
            assert!(
                same_variant(&executions[0], &executions[2]),
                "{dispatch_type:?}: the encoder's last execution is not its first variant"
            );
        }
        // The record keeps the encoding order: another kernel's command between two dispatches.
        let other = public_case(WORD).unwrap();
        let other_pipeline = icb_state(device, &other).unwrap();
        let other_resources = CaseResources::new(device, &other).unwrap();
        let other_buffers = bound_buffers(device, &other.compute, &other_resources).unwrap();
        let icb = icb(device, 1, false);
        encode_command(
            &icb,
            0,
            Some(&other_pipeline),
            &other.compute,
            &other_buffers,
        )
        .unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        make_resident(&mut encoder, &other_buffers, &other_resources);
        for step in 0..2 {
            if step == 1 {
                encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
            }
            let input = super::buffer_with(device, &case.compute.buffers[&0]).unwrap();
            let output = super::buffer_with(device, &case.compute.buffers[&1]).unwrap();
            let set = chain_set(device, &case, &input, &output);
            encode_case(&mut encoder, &pipeline, &case.compute, &set, &none).unwrap();
        }
        encoder.end_encoding().unwrap();
        complete(&command_buffer);
        let executions = command_buffer.executions();
        assert_eq!(executions.len(), 3);
        assert!(same_variant(&executions[0], &executions[2]));
        assert!(!same_variant(&executions[0], &executions[1]));
        assert_eq!(output(&other.compute, &other_buffers), other.metal_output);
    });
}

/// A kernel calling through a visible-function table its argument buffer holds calls the table's
/// functions from a command as from its own dispatch: the owned `uni_argument_tables` fixture's
/// `argument_table_dispatch` (`function_tables`' kernel) calls entry `tid % 3` of the table whose
/// ID is the argument buffer's first word, `table_add`, `table_mul` and `table_sub` of
/// `visible_table_functions`, with `(tid, 3)`, and writes the result to buffer 0.
#[test]
fn a_command_calls_through_the_tables_its_argument_buffer_holds() {
    on_device(|device, _| {
        let fixture = |stem: &str| llvm_assemble(&engine_dir().join(format!("{stem}.ll"))).unwrap();
        let caller = device
            .new_library_with_air(&fixture("uni_argument_tables"))
            .unwrap();
        let functions = device
            .new_library_with_air(&fixture("visible_table_functions"))
            .unwrap();
        let names = ["table_add", "table_mul", "table_sub"];
        let linked = names.iter().fold(LinkedFunctions::new(), |linked, name| {
            linked.with_function(
                functions
                    .new_function_with_name(name)
                    .unwrap()
                    .linked_function(),
            )
        });
        let kernel = caller
            .new_function_with_name("argument_table_dispatch")
            .unwrap();
        let mut descriptor = ComputePipelineDescriptor::new(StageSource::Function(
            kernel.function_descriptor().with_linked(linked),
        ));
        descriptor.support_indirect_command_buffers = true;
        let pipeline = device
            .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap();
        let table = pipeline
            .new_visible_function_table(&VisibleFunctionTableDescriptor::new(names.len() as u64))
            .unwrap();
        for (index, name) in names.iter().enumerate() {
            let function = functions.new_function_with_name(name).unwrap();
            let handle = pipeline.function_handle(&function).unwrap().unwrap();
            table.set_function(Some(&handle), index as u32).unwrap();
        }
        const THREADS: u32 = 40;
        let arguments = device
            .new_buffer_with_bytes(
                &table.gpu_resource_id().0.to_le_bytes(),
                ResourceOptions::shared(),
            )
            .unwrap();
        let out = |device: &Device| {
            device
                .new_buffer(u64::from(THREADS) * 4, ResourceOptions::shared())
                .unwrap()
        };
        let (dispatched, commanded) = (out(device), out(device));
        let grid = Size::new(THREADS.into(), 1, 1);
        let icb = icb(device, 1, false);
        let command = icb.indirect_compute_command_at(0).unwrap();
        command.set_compute_pipeline_state(&pipeline).unwrap();
        command.set_kernel_buffer(Some(&commanded), 0, 0).unwrap();
        command.set_kernel_buffer(Some(&arguments), 0, 1).unwrap();
        command.concurrent_dispatch_threads(grid, grid).unwrap();
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&pipeline).unwrap();
        encoder.set_buffer(Some(&dispatched), 0, 0).unwrap();
        encoder.set_buffer(Some(&arguments), 0, 1).unwrap();
        encoder.dispatch_threads(grid, grid).unwrap();
        let resident = [commanded.as_resource(), arguments.as_resource()];
        encoder
            .use_resources(&resident, ResourceUsage::READ | ResourceUsage::WRITE)
            .unwrap();
        encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
        encoder.end_encoding().unwrap();
        complete(&command_buffer);
        let expected: Vec<u8> = (0..THREADS)
            .map(|tid| match tid % 3 {
                0 => tid.wrapping_add(3),
                1 => tid.wrapping_mul(3),
                _ => tid.wrapping_sub(3),
            })
            .flat_map(u32::to_le_bytes)
            .collect();
        let length = u64::from(THREADS) * 4;
        assert_eq!(read_buffer(&dispatched, 0, length).unwrap(), expected);
        assert_eq!(read_buffer(&commanded, 0, length).unwrap(), expected);
        assert_eq!(
            command_buffer.execution_report(),
            Some(StatusReport::default())
        );
        drop(table);
    });
}
