//! Device tests of the runtime's parts validation's executor reads on the device the cfg selects,
//! under the validation layer: a command buffer's execution record, the layer settings a device's
//! instance is made with, and a device that withholds recommended features.

use super::renders::{is_public_render, render_cases_where, run_render};
use super::{
    at_plan, bound_buffers, encode_case, in_shard, layer_reported_nothing_at_all, pipeline_state,
    public_case, run_case, validated_device_with, CaseResources, Messages, ScratchCache, WAIT,
};
use crate::budgets::creation_within_ceiling;
use crate::candidate::engine::compute_shape;
use crate::candidate::engine::variants::compile_compute_for_shape;
use crate::candidate::runtime_case::executor_layer_settings;
use crate::coverage::ROBUSTNESS_FEATURES;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::{Dispatch, StatusReport};
use metal2vulkan::library::Library;
use metal2vulkan::pipeline::FunctionDescriptor;
use metal2vulkan::Budget;
use metal2vulkan_runtime::{CommandBufferStatus, Device, DeviceOptions, ExecutedPipeline};

/// A public kernel case of two buffers, one threadgroup.
pub(super) const KERNEL: &str = "saturating-halving-and-wide-product-integers-at-their-edges";

/// Run `test` on a lane device made with `adjust`'s options, then fail on any validation message.
pub(super) fn on_device_with(adjust: impl FnOnce(&mut DeviceOptions), test: impl FnOnce(&Device)) {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), adjust).unwrap();
        test(&device);
    }
    layer_reported_nothing_at_all(&messages);
}

/// The kernel case's run records one execution: the compiled compute pipeline its dispatch ran,
/// which is the case's compile for the recorded bind shape and dispatch (the compile the
/// executor's artifact digest covers), planned for the recorded bind shape as for the one the
/// executor computes for the case (the two tell different view base alignments apart:
/// `engine::VIEW_BASE_ALIGNMENT`), and each `VkPipeline` the dispatch records, created within the
/// ceiling. A dispatch of no threads records an execution with no `VkPipeline`.
#[test]
fn the_execution_record_names_each_dispatchs_compiled_pipeline() {
    on_device_with(at_plan, |device| {
        let case = public_case(KERNEL).unwrap();
        let run = run_case(device, &case).unwrap();
        assert_eq!(run.output, case.metal_output);
        let [execution] = run.executions.as_slice() else {
            panic!("{} executions, not one", run.executions.len());
        };
        let ExecutedPipeline::Compute(pipeline) = execution.pipeline() else {
            panic!("a dispatch executed a render pipeline");
        };
        assert!(!execution.pipeline_creation().is_empty());
        for &created in execution.pipeline_creation() {
            creation_within_ceiling("the case's pipeline", created).unwrap();
        }
        let compile = device.compile_device();
        let addressed = compile.profile().descriptors().buffer_device_address;
        let planned = |shape| {
            let plan = pipeline.plan(compile, shape).unwrap();
            plan.content_digest().unwrap()
        };
        assert_eq!(
            planned(execution.shape()),
            planned(&compute_shape(&case.compute, addressed).unwrap()),
            "the runtime planned other than the executor plans"
        );
        let library = Library::from_air(&case.bitcode).unwrap();
        let function = FunctionDescriptor::new(library.function(&case.compute.entry).unwrap());
        let (own, _) = compile_compute_for_shape(
            compile,
            &function,
            execution.shape(),
            Some(&case.compute.dispatch),
            None,
            &Budget::standard(),
        )
        .unwrap();
        assert_eq!(pipeline.content_digest(), own.content_digest());

        // The same case dispatched over no threads.
        let pipeline = pipeline_state(device, &case).unwrap();
        let mut empty = case.compute.clone();
        empty.dispatch = Dispatch::Threads {
            threads: [0, 1, 1],
            threads_per_group: [1, 1, 1],
        };
        let resources = CaseResources::new(device, &case).unwrap();
        let buffers = bound_buffers(device, &empty, &resources).unwrap();
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encode_case(&mut encoder, &pipeline, &empty, &buffers, &resources).unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let executions = command_buffer.executions();
        let [execution] = executions.as_slice() else {
            panic!(
                "{} executions of the empty dispatch, not one",
                executions.len()
            );
        };
        assert!(matches!(execution.pipeline(), ExecutedPipeline::Compute(_)));
        assert!(execution.pipeline_creation().is_empty());
    });
}

/// Each committed public render case the runtime draws records one execution: the compiled
/// render pipeline its draw ran, and the one `VkPipeline` it records, created within the ceiling.
/// In `EXECUTION_SHARDS` tests (`in_shard`): as one, it passed the Mac lane's 60 s a test on the
/// M4 and took 52 s on the 780M.
fn the_execution_record_names_each_draws_compiled_pipeline(shard: usize) {
    let cases = render_cases_where(in_shard(is_public_render, shard, EXECUTION_SHARDS));
    let mut drawn = 0usize;
    on_device_with(at_plan, |device| {
        for (name, case) in &cases {
            let Ok(case) = case else { continue };
            let Ok(run) = run_render(device, case) else {
                continue;
            };
            let [execution] = run.executions.as_slice() else {
                panic!("{name}: {} executions, not one", run.executions.len());
            };
            assert!(
                matches!(execution.pipeline(), ExecutedPipeline::Render(_)),
                "{name}: a draw executed a compute pipeline"
            );
            let [created] = execution.pipeline_creation() else {
                panic!("{name}: a draw recorded other than one VkPipeline");
            };
            creation_within_ceiling("the case's pipeline", *created).unwrap();
            drawn += 1;
        }
    });
    eprintln!("{drawn} public render cases drawn, shard {shard} of {EXECUTION_SHARDS}");
    assert!(drawn > 0, "no public render case of shard {shard} drew");
}

/// The pieces `the_execution_record_names_each_draws_compiled_pipeline` is split into.
const EXECUTION_SHARDS: usize = 8;

census_shards!(
    the_execution_record_names_each_draws_compiled_pipeline, EXECUTION_SHARDS;
    the_execution_record_names_each_draws_compiled_pipeline_0 = 0,
    the_execution_record_names_each_draws_compiled_pipeline_1 = 1,
    the_execution_record_names_each_draws_compiled_pipeline_2 = 2,
    the_execution_record_names_each_draws_compiled_pipeline_3 = 3,
    the_execution_record_names_each_draws_compiled_pipeline_4 = 4,
    the_execution_record_names_each_draws_compiled_pipeline_5 = 5,
    the_execution_record_names_each_draws_compiled_pipeline_6 = 6,
    the_execution_record_names_each_draws_compiled_pipeline_7 = 7,
);

/// A device that withholds the robustness features (the executor's robustness-off leg) enables
/// none of them, its compile device states no robust access, and the kernel case, compiled for
/// the in-shader containment encoding, matches Metal with a clean execution report.
#[test]
fn a_device_that_withholds_robustness_runs_the_in_shader_encoding() {
    on_device_with(
        |options| options.withheld_features = ROBUSTNESS_FEATURES.into_iter().collect(),
        |device| {
            for feature in ROBUSTNESS_FEATURES {
                assert!(
                    !device.enabled().features.contains(&feature),
                    "{feature:?} enabled"
                );
            }
            let images = device.compile_device().profile().images();
            assert!(!images.robust_image_access && !images.robust_image_access2);
            let case = public_case(KERNEL).unwrap();
            let run = run_case(device, &case).unwrap();
            assert_eq!(run.report, StatusReport::default());
            assert_eq!(run.output, case.metal_output);
        },
    );
}

/// A device whose instance takes the validation layer's settings through `VK_EXT_layer_settings`
/// (every occurrence of a message reported, synchronization validation on: the executor's
/// context's settings) is made, and runs the kernel case cleanly.
#[test]
fn a_device_takes_the_validation_layers_settings() {
    on_device_with(
        |options| options.layer_settings = executor_layer_settings(),
        |device| {
            let case = public_case(KERNEL).unwrap();
            let run = run_case(device, &case).unwrap();
            assert_eq!(run.report, StatusReport::default());
            assert_eq!(run.output, case.metal_output);
        },
    );
}
