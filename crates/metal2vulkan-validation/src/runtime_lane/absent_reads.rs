//! Reads of an argument a function constant removes, on the device the cfg selects, under the
//! validation layer.
//!
//! The owned fixture `engine/rfs_absent_reads`, with `has_source` false: `rfs_absent_query`
//! writes `7 + i`, or the width of the removed texture `source` where its run-time flag is
//! nonzero; `rfs_absent_read` writes `(i, 0.5, 0.25, 1)`, or texel `(i, 0)` of `source`;
//! `rfs_absent_sampler` writes texel `(i, 0)` of the present `tex`, or a sample of it through the
//! removed sampler `smp`. Metal gives each such read no value. Making the three pipeline states
//! announces [`Degradation::AbsentArgumentRead`] once, beside what their stages record
//! (`degrade_notices::made_compute_kinds`: off Metal's sampler, the device's conversion of the
//! texel `rfs_absent_read` and `rfs_absent_sampler` read); a command buffer whose flag is 0 completes
//! with Metal's answer (its authored cases' Metal rows), and one whose flag is 1 runs the read,
//! which takes zero (Metal's answer under its debug layer), and completes too, its execution
//! report stating the read (it failed before). The kernel marks the command buffer's status by the
//! status buffer's device address where the device addresses buffers, and through its binding of
//! the buffer (`SyntheticBinding::ExecutionStatus`) where it does not: the device as it is, the
//! device without `shaderInt64`, and the device restricted to `vk12-floor` (where the profiles
//! layer is installed) run the same kernels.

use super::degrade_notices::{assert_once, counts, made_compute_kinds, on_device_with};
use super::shared_kinds;
use super::Messages;
use super::{at_plan, buffer_with, read_buffer};
use crate::air_library::AirLibrary;
use crate::candidate::LaneRestriction;
use metal2vulkan::device::VkFeature;
use metal2vulkan::library::{FunctionConstantValue, FunctionConstantValues};
use metal2vulkan::metal::{DataType, PixelFormat, TextureUsage};
use metal2vulkan::pipeline::{
    BindingSource, ComputePipelineDescriptor, PushField, StageSource, SyntheticBinding,
};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    CommandBufferError, CommandBufferStatus, ComputePipelineState, Degradation, Device,
    DeviceOptions, ExecutedPipeline, PipelineOptions, Region, ResourceOptions, Size, Texture,
    TextureDescriptor,
};

/// What one run of a kernel left: the command buffer's status and error, whether its execution
/// report states a removed argument's read, the words of `out`, and how the dispatch's pipeline
/// reached the command buffer's status.
type Ran = (
    CommandBufferStatus,
    Option<CommandBufferError>,
    Option<bool>,
    Vec<u32>,
    Reach,
);
use std::collections::BTreeSet;
use std::time::Duration;

/// The kind making the states takes.
pub(super) const ABSENT_READS: [Degradation; 1] = [Degradation::AbsentArgumentRead];

/// The longest a command buffer may take.
const WAIT: Duration = Duration::from_secs(60);

/// Threads per dispatch, one threadgroup.
const THREADS: u64 = 4;

/// How a dispatch's pipeline reached the command buffer's `ExecutionStatus` buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reach {
    /// By its device address, a push field.
    Address,
    /// Through the kernel's binding of it.
    Binding,
}

/// The state of `name` in the fixture with `has_source` false, and the kinds its making announces
/// by its stage ([`made_compute_kinds`]).
fn state(device: &Device, name: &str) -> (ComputePipelineState, BTreeSet<Degradation>) {
    let bitcode = llvm_assemble(&engine_dir().join("rfs_absent_reads.ll")).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let constants = FunctionConstantValues::defaults()
        .with_index(0, FunctionConstantValue::new(DataType::Bool, [0]));
    let function = library
        .new_function_with_name_constant_values(name, &constants)
        .unwrap();
    let state = device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap_or_else(|error| panic!("{name} on {}: {error}", device.name()));
    (
        state,
        made_compute_kinds(device, function.function_descriptor()),
    )
}

/// `tex`: 4x1 `RGBA32Float`, texel `i` = `(i + 1) * (1, 2, 3, 4)`.
fn texels() -> Vec<f32> {
    (0..THREADS)
        .flat_map(|i| (1..=4).map(move |c| ((i + 1) * c) as f32))
        .collect()
}

fn texture(device: &Device) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA32Float, THREADS, 1, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    let texture = device.new_texture(&descriptor).unwrap();
    let bytes: Vec<u8> = texels().iter().flat_map(|f| f.to_le_bytes()).collect();
    texture
        .replace_region(
            Region::new_2d(0, 0, THREADS, 1),
            0,
            0,
            &bytes,
            16 * THREADS,
            0,
        )
        .unwrap();
    texture
}

/// One dispatch of `state` over [`THREADS`] threads with `flag` at buffer 1 and `tex` at texture
/// 0: the command buffer's status and error, whether its execution report states a removed
/// argument's read, and the words of the 16-byte-per-thread `out` at buffer 0, which starts
/// `0xab` in every byte.
fn run(device: &Device, state: &ComputePipelineState, flag: u32, tex: Option<&Texture>) -> Ran {
    let bytes = 16 * THREADS;
    let out = buffer_with(device, &vec![0xab; bytes as usize]).unwrap();
    let flag = buffer_with(device, &flag.to_le_bytes()).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    if let Some(tex) = tex {
        encoder.set_texture(Some(tex), 0).unwrap();
    }
    encoder.set_buffer(Some(&out), 0, 0).unwrap();
    encoder.set_buffer(Some(&flag), 0, 1).unwrap();
    let grid = Size::new(THREADS, 1, 1);
    encoder.dispatch_threads(grid, grid).unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let words = read_buffer(&out, 0, bytes)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect();
    let absent = command_buffer
        .execution_report()
        .map(|report| report.absent_argument_read);
    let executions = command_buffer.executions();
    let [execution] = executions.as_slice() else {
        panic!("{} executions of one dispatch", executions.len());
    };
    let ExecutedPipeline::Compute(pipeline) = execution.pipeline() else {
        panic!("a dispatch of a render pipeline");
    };
    let bound = pipeline.layout.sets.iter().any(|set| {
        set.bindings.iter().any(|b| {
            b.source
                == BindingSource::Synthetic {
                    stage: metal2vulkan::pipeline::Stage::Kernel,
                    binding: SyntheticBinding::ExecutionStatus,
                }
        })
    });
    let pushed = pipeline
        .interface
        .push
        .iter()
        .flat_map(|placement| &placement.fields)
        .any(|f| f.field == PushField::ExecutionStatusAddress);
    let reach = match (bound, pushed) {
        (true, false) => Reach::Binding,
        (false, true) => Reach::Address,
        other => panic!("the status reached as (binding, address) {other:?}"),
    };
    (
        command_buffer.status(),
        command_buffer.error(),
        absent,
        words,
        reach,
    )
}

/// The three kernels' states are announced once; flag 0 completes with what each writes without
/// the read, its report stating no such read, and flag 1 completes too, each read zero (the
/// width 0, the texel and the sample `(0, 0, 0, 0)`), its report stating the read: the path ran,
/// through the status as `reach`. Over the whole test every kind is announced at most once: the
/// runs may announce a kind of their own for the first time (on a device without unified memory,
/// the `Shared` buffers and texture they make in system memory, `SharedInSystemMemory`), never one
/// announced before.
fn reads_zero_and_completes(device: &Device, messages: &Messages, reach: Reach) {
    let names = ["rfs_absent_query", "rfs_absent_read", "rfs_absent_sampler"];
    let (states, made): (Vec<_>, Vec<_>) = names.iter().map(|name| state(device, name)).unzip();
    let mut kinds = ABSENT_READS.to_vec();
    kinds.extend(made.into_iter().flatten());
    let mut notices = messages.take_notices();
    assert_once(&notices, &kinds);
    let tex = texture(device);
    let as_words = |floats: &[f32]| floats.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    let query: Vec<u32> = (0..THREADS as u32).map(|i| 7 + i).collect();
    let read: Vec<f32> = (0..THREADS)
        .flat_map(|i| [i as f32, 0.5, 0.25, 1.0])
        .collect();
    let expected = [
        (None, query),
        (None, as_words(&read)),
        (Some(&tex), as_words(&texels())),
    ];
    for ((name, state), (tex, want)) in names.iter().zip(&states).zip(expected) {
        let (status, error, absent, words, ran) = run(device, state, 0, tex);
        assert_eq!(status, CommandBufferStatus::Completed, "{name}: {error:?}");
        assert_eq!(error, None, "{name}");
        assert_eq!(absent, Some(false), "{name}: flag 0");
        assert_eq!(&words[..want.len()], &want[..], "{name}: flag 0");
        assert_eq!(ran, reach, "{name}: flag 0");
        let (status, error, absent, words, ran) = run(device, state, 1, tex);
        assert_eq!(
            status,
            CommandBufferStatus::Completed,
            "{name}: flag 1: {error:?}"
        );
        assert_eq!(error, None, "{name}: flag 1");
        assert_eq!(absent, Some(true), "{name}: flag 1");
        assert_eq!(
            &words[..want.len()],
            &vec![0; want.len()][..],
            "{name}: flag 1"
        );
        assert_eq!(ran, reach, "{name}: flag 1");
        println!(
            "ABSENT-READ-COMPLETED {}: {name}, the status reached {reach:?}",
            device.name()
        );
    }
    notices.extend(messages.take_notices());
    println!("ANNOUNCED {}: {:?}", device.name(), counts(&notices));
    kinds.extend(shared_kinds(device));
    assert_once(&notices, &kinds);
}

/// [`reads_zero_and_completes`] on the device as it is, which marks the status by address where
/// it addresses buffers (`bufferDeviceAddress` and `shaderInt64`) and through its binding
/// elsewhere.
#[test]
fn a_read_of_a_removed_argument_reads_zero_and_its_command_buffer_completes() {
    on_device_with(at_plan, |device, messages| {
        let addressed = device
            .compile_device()
            .profile()
            .addressed_needs()
            .is_empty();
        let reach = match addressed {
            true => Reach::Address,
            false => Reach::Binding,
        };
        reads_zero_and_completes(device, messages, reach);
    });
}

/// [`reads_zero_and_completes`] on the device without `shaderInt64`, which then addresses no
/// buffer: the kernels mark the status through their binding of it.
#[test]
fn without_buffer_addresses_a_removed_arguments_read_marks_through_the_status_binding() {
    on_device_with(
        |options| {
            options.withheld_features.insert(VkFeature::ShaderInt64);
            at_plan(options)
        },
        |device, messages| reads_zero_and_completes(device, messages, Reach::Binding),
    );
}

/// [`reads_zero_and_completes`] on the device restricted to `vk12-floor` through the profiles
/// layer (Vulkan 1.2 and nothing optional: no buffer device address, no 64-bit integers, 4 storage
/// buffers a stage): the kernels mark the status through their binding of it. Skipped, visibly,
/// where the profiles layer is not installed or the device lacks a capability the profile states
/// (the restricted context's first check refuses it); every other failure fails.
#[test]
fn on_vk12_floor_a_removed_arguments_read_marks_through_the_status_binding() {
    let restriction =
        match crate::candidate::lane_restriction("vk12-floor", crate::gpu::TEST_DEVICE).unwrap() {
            LaneRestriction::Takes(restriction) => restriction,
            LaneRestriction::Lacks { device, gaps } => {
                println!(
                    "SKIPPED {device}: cannot take vk12-floor: lacks {}",
                    gaps.join(", ")
                );
                return;
            }
            LaneRestriction::NoLayer => {
                println!("SKIPPED: VK_LAYER_KHRONOS_profiles is not installed");
                return;
            }
        };
    on_device_with(
        |options: &mut DeviceOptions| {
            restriction.restrict(options).unwrap();
            at_plan(options)
        },
        |device, messages| {
            let restricted = restriction.check(device).unwrap();
            println!(
                "RESTRICTED {}: {}, kept {:?}",
                device.name(),
                restricted.record,
                restricted.kept
            );
            reads_zero_and_completes(device, messages, Reach::Binding);
        },
    );
}
