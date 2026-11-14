//! The runtime path's routing, device-free, over the committed public cases.

use super::*;
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
use crate::candidate::runtime_case::layer_reported_nothing_at_all;
use crate::check::admission::{admit, ExecutionRequest};
use crate::gpu::DeviceType;
use crate::store::CorpusStore;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::path::Path;

/// The route of each committed public case `names` names, prepared for the CPU device.
fn routes(names: &[&str]) -> BTreeMap<String, Route> {
    prepared(names)
        .into_iter()
        .map(|(name, prepared)| (name, route(&prepared)))
        .collect()
}

/// Each committed public case `names` names, prepared for the CPU device.
fn prepared(names: &[&str]) -> BTreeMap<String, PreparedCase> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus");
    let device = DeviceType::Cpu;
    let cases = CorpusStore::new(&root).read_all_cases().unwrap();
    names
        .iter()
        .map(|&name| {
            let case = cases
                .iter()
                .find(|case| case.name == name)
                .unwrap_or_else(|| panic!("the committed public case {name}"))
                .clone();
            let admitted = admit(&root, case, ExecutionRequest::EngineCandidate { device })
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let prepared = super::super::prepare_case(device, admitted)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            (name.to_owned(), prepared)
        })
        .collect()
}

/// Under a restriction whose kept members the device holds otherwise, a patch draw of a profile
/// without `tessellationShader` (which compiles it emulated) runs on the runtime, the only path
/// that runs the emulation; one of a profile with the feature, and a draw of primitives, go to
/// the executor's context as before.
#[test]
fn a_patch_draw_a_profile_emulates_runs_on_the_runtime() {
    let patches = "back-culled-clockwise-quad-output-of-a-patch-and-its-mirror";
    let primitives = "authored-fragment-constant-color";
    let cases = prepared(&[patches, primitives]);
    let floor = crate::engine::load_profile("vk12-floor").unwrap();
    let lavapipe = crate::engine::load_profile("lavapipe").unwrap();
    assert!(emulated_patches(&cases[patches].mapped, &floor));
    assert!(!emulated_patches(&cases[patches].mapped, &lavapipe));
    assert!(!emulated_patches(&cases[primitives].mapped, &floor));
}

/// The runtime runs a kernel case of buffers, of textures (sparse and multisampled ones included)
/// and samplers, of threadgroup memory, of an explicit imageblock, of a tile pass, of
/// argument-buffer buffers, textures and samplers, of indirect command buffers of compute commands
/// and the compute pipeline states its argument buffers hold, of rasterization-rate maps, of
/// function constant values, of stage inputs, of visible-function tables, and of out-of-bounds
/// targets (buffers and stage inputs a guard holds, textures, thread and threadgroup memory, a
/// handle array), and a render case of one direct or indexed draw (single-sampled or multisampled,
/// with its vertex inputs and attachments, its argument buffers, a fragment case's explicit
/// imageblock, and a rasterization-rate map); every other case waits on the executor's context for the steps that
/// add each prerequisite it needs. (No committed public case links functions without a
/// visible-function table: the linked kernels the runtime runs are private, and `runtime_lane`'s
/// `a_kernel_runs_the_functions_its_descriptor_links` runs the owned linking fixtures.)
#[test]
fn committed_cases_route_by_their_own_inputs() {
    use Prerequisite as P;
    let expected = [
        (
            "saturating-halving-and-wide-product-integers-at-their-edges",
            Route::Runtime,
        ),
        (
            "sample-each-of-three-textures-through-one-dynamic-array-index",
            Route::Runtime,
        ),
        ("threadgroup-scratch-of-eighty-byte-elements", Route::Runtime),
        ("imageblock-word", Route::Runtime),
        (
            "map-a-physical-coordinate-through-a-uniform-rate-map",
            Route::Runtime,
        ),
        (
            "map-physical-coordinates-through-rate-maps-an-argument-buffer-holds",
            Route::Runtime,
        ),
        ("walk-a-device-pointer-chain-through-a-loop-phi", Route::Runtime),
        ("gated-bias-on", Route::Runtime),
        ("fold-a-zero-function-constant-gate", Route::Runtime),
        ("fragment-custom-imageblock-depth", Route::Runtime),
        (
            "primitive-acceleration-structure-intersection",
            Route::Executor(BTreeSet::from([P::RayTracing])),
        ),
        ("pipeline-states-beside-the-data-of-two-stages", Route::Runtime),
        ("pipeline-states-beside-the-data-a-vertex-reads", Route::Runtime),
        ("gpu-encoded-draws-run-the-inherited-vertex-buffers", Route::Runtime),
        ("gpu-encoded-draws-run-the-host-encoded-vertex-buffers", Route::Runtime),
        ("execute-two-gpu-encoded-template-dispatches", Route::Runtime),
        ("encode-compute-commands-beside-output", Route::Runtime),
        (
            "host-encoded-over-limit-and-ragged-dispatches-in-two-command-buffers",
            Route::Runtime,
        ),
        ("encode-render-commands-beside-output", Route::Runtime),
        ("imageblock-three-widths-load", Route::Runtime),
        ("tile-grid-float-attachment-doubles-each-pixel-once", Route::Runtime),
        (
            "two-bound-visible-tables-answer-null-and-size",
            Route::Runtime,
        ),
        ("kernel-stage-input-word", Route::Runtime),
        ("kernel-stage-input-fetches-past-its-buffer", Route::Runtime),
        (
            "sample-a-cube-with-explicit-gradients-through-a-sampler-held-beside-it-in-an-argument-buffer",
            Route::Runtime,
        ),
        ("loads-and-stores-past-a-device-buffer", Route::Runtime),
        ("reads-and-writes-past-a-texture", Route::Runtime),
        ("loads-and-stores-past-a-thread-local-array", Route::Runtime),
        ("loads-and-stores-past-a-threadgroup-allocation", Route::Runtime),
        ("reads-through-an-index-past-a-top-level-texture-array", Route::Runtime),
        (
            "a-write-to-an-unmapped-texel-read-back-after-a-texture-barrier",
            Route::Runtime,
        ),
        (
            "four-multisample-reads-at-two-samples",
            Route::Runtime,
        ),
        ("authored-fragment-constant-color", Route::Runtime),
        ("three-color-inputs-color-attachment", Route::Runtime),
        (
            "color-of-a-sample-rate-fragment-only-where-its-stencil-passes",
            Route::Runtime,
        ),
        ("pass-on-a-vertex-attribute-the-pipeline-declares", Route::Runtime),
        (
            "zero-negative-and-nan-factors-cull-or-clamp-triangle-patches",
            Route::Runtime,
        ),
        (
            "control-point-and-patch-input-fetches-past-their-buffers",
            Route::Runtime,
        ),
        (
            "fragment-positions-in-a-half-quality-rate-mapped-pass",
            Route::Runtime,
        ),
        (
            "sample-in-a-fragment-through-a-sampler-held-beside-its-texture-in-an-argument-buffer",
            Route::Runtime,
        ),
        (
            "sixteen-overlapping-triangles-update-a-raster-ordered-argument-buffer-member",
            Route::Runtime,
        ),
        ("indexed-draw-reads-past-its-uint32-index-buffer", Route::Runtime),
        (
            "indexed-draw-reads-past-its-uint16-index-buffer-from-an-offset",
            Route::Runtime,
        ),
        ("vertex-attribute-fetches-past-its-vertex-buffer", Route::Runtime),
        ("gpu-encoded-vertex-buffers-run-the-host-encoded-draws", Route::Runtime),
    ];
    let names: Vec<&str> = expected.iter().map(|(name, _)| *name).collect();
    let routes = routes(&names);
    for (name, route) in expected {
        assert_eq!(routes[name], route, "{name}");
    }
}

/// A case the runtime runs replays its templates through the runtime's own pipeline states, so the
/// executor's context's refusal of them (a compute template with a stage input or an imageblock, a
/// render template with vertex inputs or a post-tessellation stage) is not that run's: the pipelines
/// the authored compile census lists for it are its own, and each command buffer it executes is
/// uncovered, never the context's refusal.
#[test]
fn a_runtime_case_compiles_its_own_pipeline_where_the_context_refuses_its_templates() {
    use super::super::{case_compiles, CompileSubject};
    use crate::worker::compile::WorkerDevice;
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus");
    let device = DeviceType::Cpu;
    let cases = CorpusStore::new(&root).read_all_cases().unwrap();
    let profile = WorkerDevice::new(
        crate::engine::load_description("lavapipe").unwrap(),
        metal2vulkan::device::DevicePolicy::default(),
    )
    .unwrap();
    for name in [
        "a-kernel-encodes-the-stage-in-regions-of-its-dispatches",
        "host-encoded-stage-in-regions-and-attribute-strides-of-executed-commands",
        "a-kernel-encodes-the-imageblock-sizes-of-its-dispatches",
        "host-encoded-vertex-attribute-strides-of-executed-draws",
        "patch-draws-a-kernel-encodes-at-factor-scales-and-the-host-encodes",
        "indexed-patch-draws-a-kernel-encodes-on-a-template-without-control-point-indices",
    ] {
        let case = cases
            .iter()
            .find(|case| case.name == name)
            .unwrap_or_else(|| panic!("the committed public case {name}"))
            .clone();
        let admitted = admit(&root, case, ExecutionRequest::EngineCandidate { device })
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let prepared = super::super::prepare_case(device, admitted)
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(route(&prepared), Route::Runtime, "{name}");
        assert!(
            matches!(
                prepared.templates(),
                Err(ExecutorError::NotSupported(
                    crate::check::admission::NotSupported::NotImplemented(_)
                ))
            ),
            "{name}: the executor's context replays its templates"
        );
        let planned =
            case_compiles(&prepared, &profile).unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(
            planned
                .compiles
                .iter()
                .map(|compile| compile.subject)
                .collect::<Vec<_>>(),
            [CompileSubject::Case],
            "{name}"
        );
        assert_eq!(planned.host_replayed, 1, "{name}");
    }
}

/// An object or mesh case compiles device-free as the runtime compiles its draw: the authored
/// compile census plans one mesh draw request (`PipelineRequest::MeshDraw`) of its functions, its
/// generated stages from its companion's source, for its own stage's bind shape, and that request
/// compiles on a profile with mesh shaders and on one that emulates them, with no table handles.
#[test]
fn a_mesh_case_compiles_its_draw_device_free() {
    use super::super::{case_compiles, CompileSubject, PlannedPipeline};
    use crate::acceptance::{compile_pipeline, AcceptancePipeline, PipelineRequest};
    use crate::worker::compile::WorkerDevice;
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus");
    let device = DeviceType::Cpu;
    let profiles = ["lavapipe", "moltenvk-m3"].map(|stem| {
        let worker = WorkerDevice::new(
            crate::engine::load_description(stem).unwrap(),
            metal2vulkan::device::DevicePolicy::default(),
        )
        .unwrap();
        (stem, worker)
    });
    for fixture in [
        crate::mesh_cases::grid(),
        crate::mesh_cases::object(),
        crate::mesh_cases::device_write(),
        crate::mesh_cases::observed(),
        crate::mesh_cases::launched(),
        crate::mesh_cases::payload(),
        crate::mesh_cases::payload_barrier(),
        crate::mesh_cases::payload_outputs(),
    ] {
        let name = fixture.case.name.clone();
        let admitted = admit(
            &root,
            fixture.case,
            ExecutionRequest::EngineCandidate { device },
        )
        .unwrap_or_else(|error| panic!("{name}: {error}"));
        let prepared = super::super::prepare_case(device, admitted)
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        for (stem, worker) in &profiles {
            let planned = case_compiles(&prepared, worker)
                .unwrap_or_else(|error| panic!("{name} {stem}: {error}"));
            assert_eq!(planned.host_replayed, 0, "{name} {stem}");
            let [compile] = planned.compiles.as_slice() else {
                panic!("{name} {stem}: {} compiles", planned.compiles.len());
            };
            assert_eq!(compile.subject, CompileSubject::Case, "{name} {stem}");
            let PlannedPipeline::Mesh {
                request, libraries, ..
            } = &compile.pipeline
            else {
                panic!("{name} {stem}: not a mesh draw's compile");
            };
            assert!(
                matches!(request, PipelineRequest::MeshDraw { .. }),
                "{name} {stem}"
            );
            let bitcode = libraries
                .iter()
                .map(|library| library.bitcode().unwrap())
                .collect::<Vec<_>>();
            let bitcode = bitcode.iter().map(Vec::as_slice).collect::<Vec<_>>();
            match compile_pipeline(&bitcode, request, worker.device(), None) {
                Ok(compiled) => {
                    assert!(
                        matches!(compiled.pipeline, AcceptancePipeline::Render(_)),
                        "{name} {stem}"
                    );
                    request.check_handles(&compiled.handles).unwrap();
                }
                Err(error) => panic!("{name} {stem}: {error:?}"),
            }
        }
    }
}

/// A row or refusal of the executor's context names the executor and why the runtime did not run
/// the case: the prerequisite its family waits for, or the gap the runtime raised.
#[test]
fn executor_rows_say_why_the_runtime_did_not_run_them() {
    assert_eq!(
        executor_members(&ExecutorRun::WaitsFor(BTreeSet::from([
            Prerequisite::DrawOutOfBoundsTargets,
            Prerequisite::RayTracing,
        ]))),
        [
            ("executed_by", serde_json::json!("executor")),
            (
                "runtime_waits_for",
                serde_json::json!(["ray-tracing", "draw-out-of-bounds-targets"])
            ),
        ]
    );
    assert_eq!(
        executor_members(&ExecutorRun::Gap("dispatch: Textures".into())),
        [
            ("executed_by", serde_json::json!("executor")),
            ("runtime_gap", serde_json::json!("dispatch: Textures")),
        ]
    );
}

/// Before the command buffer commits, a refusal of a runtime mechanism the driver runs wrong
/// (`Needs::RuntimeDriverDefect`) sends the case to the executor, as a typed gap does, keyed on
/// the refusal's type; after it commits, or for any other refusal, the run reports the refusal.
#[test]
fn a_runtime_driver_defect_sends_the_case_back_before_it_commits() {
    use metal2vulkan::error::{Capability, Feature, RuntimeDriverDefect};
    let refusal = |unsupported: Unsupported| CaseError::Runtime {
        call: "make_render_command_encoder".into(),
        error: Box::new(Error::Unsupported(unsupported)),
    };
    let defect = || {
        refusal(Unsupported::runtime_driver_defect(
            Feature::SeparateMultisampledDepthStencil,
            RuntimeDriverDefect::MultisampledDepthStencilColourCopies,
        ))
    };
    match stop(defect(), false) {
        Stop::Gap(gap) => assert_eq!(
            gap,
            "make_render_command_encoder: driver defect MultisampledDepthStencilColourCopies \
             (SeparateMultisampledDepthStencil)"
        ),
        Stop::Refused(refused) => panic!("not sent back: {}", refused.error),
        Stop::Error(error) => panic!("not sent back: {error}"),
    }
    assert!(matches!(
        stop(defect(), true),
        Stop::Error(ExecutorError::NotSupported(_))
    ));
    let lacking = refusal(Unsupported::needs(
        Feature::SeparateMultisampledDepthStencil,
        Capability::Feature(VkFeature::Maintenance8),
    ));
    assert!(matches!(
        stop(lacking, false),
        Stop::Error(ExecutorError::NotSupported(_))
    ));
}

/// Before the command buffer commits, the runtime's refusal of a helper's pipeline (a sample
/// fill's, a sample readback's: the runner's own work around the case's pass) sends the case to
/// the executor, keyed on the error's origin and type; the same refusal of the case's own
/// pipeline, or one after the commit, is the run's refusal.
#[test]
fn a_refused_helper_pipeline_sends_the_case_back_before_it_commits() {
    use metal2vulkan::device::VkProperty;
    use metal2vulkan::error::{Capability, Feature};
    let unsupported = || {
        Unsupported::needs(
            Feature::RasterSampleCount,
            Capability::Property(VkProperty::FramebufferIntegerColorSampleCount(4)),
        )
    };
    let helper = |error: Error| CaseError::Helper {
        call: "rt6_fill_uint4: new_render_pipeline_state".into(),
        error: Box::new(error),
    };
    let sent_back = "rt6_fill_uint4: new_render_pipeline_state: helper refused RasterSampleCount; \
                     needs AnyOf([[Property(FramebufferIntegerColorSampleCount(4))]])";
    for error in [
        Error::Unsupported(unsupported()),
        Error::Compile(metal2vulkan::error::Error::Unsupported(unsupported())),
    ] {
        match stop(helper(error), false) {
            Stop::Gap(gap) => assert_eq!(gap, sent_back),
            Stop::Refused(refused) => panic!("not sent back: {}", refused.error),
            Stop::Error(error) => panic!("not sent back: {error}"),
        }
    }
    assert!(matches!(
        stop(helper(Error::Unsupported(unsupported())), true),
        Stop::Error(ExecutorError::NotSupported(_))
    ));
    let own = CaseError::Runtime {
        call: "new_render_pipeline_state".into(),
        error: Box::new(Error::Unsupported(unsupported())),
    };
    assert!(matches!(
        stop(own, false),
        Stop::Error(ExecutorError::NotSupported(_))
    ));
}

/// A public kernel case of the runtime's families runs through the runtime path on the device the
/// cfg selects, under the validation layer with the executor's settings: its output is Metal's,
/// its row records the runtime and the device's identity, and what it executed is what the
/// executor's own context compiles and runs for it: the same artifact digest, the same SPIR-V,
/// census, and comparison class, and the same output.
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
#[test]
fn a_public_kernel_case_runs_through_the_runtime_as_the_executor_compiles_it() {
    runs_as_the_executor_compiles_it("saturating-halving-and-wide-product-integers-at-their-edges");
}

/// A public kernel case of function constant values runs through the runtime path, its function
/// specialized with the case's values, as
/// [`a_public_kernel_case_runs_through_the_runtime_as_the_executor_compiles_it`] runs its case.
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
#[test]
fn a_kernel_of_function_constants_runs_through_the_runtime_as_the_executor_compiles_it() {
    runs_as_the_executor_compiles_it("fold-a-zero-function-constant-gate");
}

/// A public kernel case of visible-function tables runs through the runtime path, two tables of
/// its authored sizes bound at their buffer indices, each entry null, as
/// [`a_public_kernel_case_runs_through_the_runtime_as_the_executor_compiles_it`] runs its case.
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
#[test]
fn a_kernel_of_function_tables_runs_through_the_runtime_as_the_executor_compiles_it() {
    runs_as_the_executor_compiles_it("two-bound-visible-tables-answer-null-and-size");
}

/// A public kernel case of stage inputs runs through the runtime path, its pipeline state of the
/// oracle's stage-input descriptor and its encoder's stage-in region the oracle's, as
/// [`a_public_kernel_case_runs_through_the_runtime_as_the_executor_compiles_it`] runs its case.
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
#[test]
fn a_kernel_of_stage_inputs_runs_through_the_runtime_as_the_executor_compiles_it() {
    runs_as_the_executor_compiles_it("kernel-stage-input-word");
}

/// Public kernel cases of out-of-bounds targets no guard holds run through the runtime path: reads
/// and writes past a texture, and loads and stores past a threadgroup allocation, as
/// [`a_public_kernel_case_runs_through_the_runtime_as_the_executor_compiles_it`] runs its case,
/// each target's containment decided before the commit as the executor's own context decides it.
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
#[test]
fn unguarded_out_of_bounds_kernels_run_as_the_executor_compiles_them() {
    runs_as_the_executor_compiles_it("reads-and-writes-past-a-texture");
    runs_as_the_executor_compiles_it("loads-and-stores-past-a-threadgroup-allocation");
}

/// Public kernel cases of guarded out-of-bounds targets run through the runtime path, each target
/// placed in a heap of its own between guards: a storage buffer, a buffer reached by its address, a
/// loaded pointer's buffer with a reach before it, a stage input's buffer, and a texture buffer, as
/// [`a_public_kernel_case_runs_through_the_runtime_as_the_executor_compiles_it`] runs its case: the
/// guard counts and their verdict are the executor's. A device that leaves a target uncontained
/// (MoltenVK's robust buffer forms) refuses the case on both paths, each refusal recording the same
/// compile.
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
#[test]
fn guarded_out_of_bounds_kernels_run_as_the_executor_compiles_them() {
    for name in [
        "loads-and-stores-past-a-device-buffer",
        "loads-and-stores-past-a-device-buffer-by-its-address",
        "loads-and-stores-past-a-loaded-pointers-buffer-with-a-reach-before-it",
        "kernel-stage-input-fetches-past-its-buffer",
        "reads-and-writes-past-a-texture-buffer",
    ] {
        runs_as_the_executor_compiles_it(name);
    }
}

/// A public tile kernel runs through the runtime path, dispatched on a render encoder of its pass
/// as the case runner dispatches it (`runtime_case::encoded_tile`), its output an attachment over
/// two rows of 16x16 tiles whose last column and row are partial, as
/// [`a_public_kernel_case_runs_through_the_runtime_as_the_executor_compiles_it`] runs its case.
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
#[test]
fn a_tile_kernel_runs_through_the_runtime_as_the_executor_compiles_it() {
    runs_as_the_executor_compiles_it("tile-grid-float-attachment-doubles-each-pixel-once");
}

/// A public vertex case fetching an attribute past its vertex buffer runs through the runtime
/// path, the buffer placed in a heap of its own between guards, as
/// [`a_public_kernel_case_runs_through_the_runtime_as_the_executor_compiles_it`] runs its case: the
/// draw's containment is decided before the commit as the executor decides it, its guard counts are
/// the executor's, and its reads past the buffer, which reach the heap's bytes, match Metal's under
/// the cause the row records. A device that leaves the fetch uncontained (MoltenVK's vertex input)
/// refuses the case on both paths, each refusal recording the same compile.
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
#[test]
fn a_vertex_buffer_target_runs_as_the_executor_compiles_it() {
    runs_as_the_executor_compiles_it("vertex-attribute-fetches-past-its-vertex-buffer");
}

/// The public indexed draws whose indices run past their index buffer run through the runtime
/// path, each buffer placed in a heap of its own between guards, as
/// [`a_public_kernel_case_runs_through_the_runtime_as_the_executor_compiles_it`] runs its case: the
/// draw contains its positions past the buffer as the executor's containment states (its execution
/// record's `index_padding`: a copy of its indices padded with zeros, on every device), its facts
/// and guard counts are the executor's, and its output is the executor's, which matches Metal's.
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
#[test]
fn index_buffer_targets_run_as_the_executor_compiles_them() {
    for name in [
        "indexed-draw-reads-past-its-uint32-index-buffer",
        "indexed-draw-reads-past-its-uint16-index-buffer-from-an-offset",
    ] {
        runs_as_the_executor_compiles_it(name);
    }
}

/// Public cases of indirect render commands run through the runtime path, as
/// [`a_public_kernel_case_runs_through_the_runtime_as_the_executor_compiles_it`] runs its case: a
/// kernel that encodes render commands into a command buffer of the oracle's render descriptor
/// (`DRAW`), read back as the runtime decodes them; a kernel's and a vertex function's argument
/// buffers holding a render template's state beside a compute template's, each made with
/// `support_indirect_command_buffers` (`template_states`); and render commands a kernel or a
/// vertex function wrote, with the host's, executed after it in a render encoder of the oracle's
/// pass, their buffers the host's or inherited.
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
#[test]
fn render_commands_and_render_states_run_as_the_executor_compiles_them() {
    for name in [
        "encode-render-commands-beside-output",
        "pipeline-states-beside-the-data-of-two-stages",
        "pipeline-states-beside-the-data-a-vertex-reads",
        "gpu-encoded-draws-run-the-host-encoded-vertex-buffers",
        "gpu-encoded-draws-run-the-inherited-vertex-buffers",
        "gpu-encoded-vertex-buffers-run-the-host-encoded-draws",
    ] {
        runs_as_the_executor_compiles_it(name);
    }
}

/// The executor's own context runs an execution of render commands the GPU may write where the
/// runtime does, by the draw replay rules both share: on the device, one indirect draw per command
/// of the parent template's draw replay variant (`compute::draw_replay`; its vertex function
/// writes memory, so the commands' points do not merge), where a kernel or a vertex function writes
/// the draws or their buffers, the commands' own or the encoder's where they inherit them (the
/// preprocess reads the parent's buffers). The environment names the path each execution took
/// (`command_replay`), each output Metal's; the census of each is held to the runtime path's by
/// `render_commands_and_render_states_run_as_the_executor_compiles_them`.
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
#[test]
fn the_executor_draws_gpu_written_render_commands_where_the_runtime_does() {
    use crate::candidate::tests::compile_worker;
    use crate::candidate::tests::engine_compute::BACKEND;
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus");
    let device = crate::gpu::TEST_DEVICE;
    let store = CorpusStore::new(&root);
    let cases = store.read_all_cases().unwrap();
    for (name, path) in [
        (
            "gpu-encoded-draws-run-the-host-encoded-vertex-buffers",
            "device-draws",
        ),
        (
            "gpu-encoded-vertex-buffers-run-the-host-encoded-draws",
            "device-draws",
        ),
        (
            "gpu-encoded-draws-run-the-inherited-vertex-buffers",
            "device-draws",
        ),
    ] {
        let case = cases
            .iter()
            .find(|case| case.name == name)
            .expect("the committed public case")
            .clone();
        let metal = crate::runtime_lane::metal_output(&store, &case).unwrap();
        let admitted = admit(&root, case, ExecutionRequest::EngineCandidate { device }).unwrap();
        let prepared = super::super::prepare_case(device, admitted).unwrap();
        let on_context = super::super::run_on_context(&prepared, BACKEND, compile_worker())
            .result
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(on_context.output, metal, "{name}: the executor's output");
        assert_eq!(
            on_context.environment["command_replay"],
            serde_json::json!([path]),
            "{name}"
        );
    }
}

/// Public cases whose command buffers execute ranges of two or more compute commands, a kernel's
/// and the host's, run through the runtime path and on the executor's own context (its device
/// replay), each output Metal's. Where the context enables `VK_EXT_device_generated_commands`, the
/// executor runs each range as one command buffer's executions, one after another, as the runtime
/// does: on RADV before 26.2 the driver's own preprocessing of every execution after the first ran
/// the shader the one before bound last, and the device faulted
/// (`RuntimeDriverDefect::ImplicitPreprocessAfterExecutionSet`), so both preprocess the
/// executions apart there. The rows' artifacts are not compared: the executor's digests what it
/// compiled for the commands, and the runtime path's records no digest (as
/// [`runs_as_the_executor_compiles_it`] states).
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
#[test]
fn compute_command_ranges_run_on_both_paths_as_metal_ran_them() {
    use crate::candidate::tests::compile_worker;
    use crate::candidate::tests::engine_compute::BACKEND;
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus");
    let device = crate::gpu::TEST_DEVICE;
    let store = CorpusStore::new(&root);
    let cases = store.read_all_cases().unwrap();
    for name in [
        "execute-two-gpu-encoded-template-dispatches",
        "gpu-encoded-buffers-keep-the-host-encoded-dispatches",
        "gpu-encoded-ragged-grids-replay-with-the-inherited-pipeline-and-buffers",
        "host-encoded-over-limit-and-ragged-dispatches-in-two-command-buffers",
    ] {
        let case = cases
            .iter()
            .find(|case| case.name == name)
            .expect("the committed public case")
            .clone();
        assert!(case.out_of_bounds.is_none(), "{name}");
        let metal = crate::runtime_lane::metal_output(&store, &case).unwrap();
        let admitted = admit(&root, case, ExecutionRequest::EngineCandidate { device }).unwrap();
        let prepared = super::super::prepare_case(device, admitted).unwrap();
        assert_eq!(route(&prepared), Route::Runtime, "{name}");
        let RuntimeOutcome::Ran(attempt) = run(&prepared, BACKEND) else {
            panic!("{name}: the runtime raised a gap");
        };
        let ran = attempt
            .result
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(ran.output, metal, "{name}: the runtime path's output");
        let on_context = super::super::run_on_context(&prepared, BACKEND, compile_worker())
            .result
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        // Which replay ran each command buffer the case executes, for the lane's record.
        eprintln!(
            "{name}: the executor's command replay {}",
            on_context.environment["command_replay"]
        );
        assert_eq!(on_context.output, metal, "{name}: the executor's output");
    }
}

/// A public render case runs through the runtime path, drawn as the case runner draws it
/// (`runtime_case::encoded_render`), as
/// [`a_public_kernel_case_runs_through_the_runtime_as_the_executor_compiles_it`] runs its case:
/// the artifact digest and the SPIR-V of every stage are the executor's.
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
#[test]
fn a_render_case_runs_through_the_runtime_as_the_executor_compiles_it() {
    runs_as_the_executor_compiles_it("authored-fragment-constant-color");
}

/// A public multisampled render case runs through the runtime path, its target filled and its
/// output read sample by sample, in a pass every venue runs on the runtime (`RGBA8Unorm` at 4x
/// without stencil: an integer target's 4x is one RADV lacks, and the runtime sends a
/// stencil-tested pass back to the executor on NVIDIA), as
/// [`a_public_kernel_case_runs_through_the_runtime_as_the_executor_compiles_it`] runs its case.
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
#[test]
fn a_multisampled_render_case_runs_through_the_runtime_as_the_executor_compiles_it() {
    runs_as_the_executor_compiles_it("sample-mask-out-4x");
}

/// The committed public case `name` through the runtime path and through the executor's own
/// context, compared (the tests above).
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
fn runs_as_the_executor_compiles_it(name: &str) {
    use crate::candidate::engine::RefusedOn;
    use crate::candidate::tests::compile_worker;
    use crate::candidate::tests::engine_compute::BACKEND;
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus");
    let device = crate::gpu::TEST_DEVICE;
    let store = CorpusStore::new(&root);
    let case = store
        .read_all_cases()
        .unwrap()
        .into_iter()
        .find(|case| case.name == name)
        .expect("the committed public case");
    let metal = crate::runtime_lane::metal_output(&store, &case).unwrap();
    let authored = case.clone();
    let executor = match case.stage {
        crate::case::Stage::Kernel => "engine-runtime-compute",
        // A mesh draw is a render encoder's draw (`runtime_case::encoded_mesh`).
        crate::case::Stage::Vertex
        | crate::case::Stage::Fragment
        | crate::case::Stage::Object
        | crate::case::Stage::Mesh => "engine-runtime-render",
    };
    let admitted = admit(&root, case, ExecutionRequest::EngineCandidate { device }).unwrap();
    let prepared = super::super::prepare_case(device, admitted).unwrap();
    assert_eq!(route(&prepared), Route::Runtime);
    let RuntimeOutcome::Ran(attempt) = run(&prepared, BACKEND) else {
        panic!("{name}: the runtime raised a gap");
    };
    let on_context = super::super::run_on_context(&prepared, BACKEND, compile_worker());
    let run = match attempt.result {
        Ok(run) => run,
        // A refusal after the dispatch took its pipeline (a target the device leaves
        // uncontained) is the executor's, and records the same compile.
        Err(
            error @ ExecutorError::NotSupported(crate::check::admission::NotSupported::Uncontained(
                _,
            )),
        ) => {
            assert_eq!(on_context.result.err(), Some(error), "{name}");
            let artifact = |refused_on: Option<RefusedOn>| {
                refused_on
                    .and_then(|opened| opened.artifact)
                    .unwrap_or_else(|| panic!("{name}: the refusal records no artifact"))
            };
            let recorded = artifact(attempt.refused_on);
            assert!(recorded.sha256.is_some(), "{name}: the refusal's digest");
            assert_eq!(recorded, artifact(on_context.refused_on), "{name}");
            return;
        }
        Err(error) => panic!("{name}: {error}"),
    };
    match &authored.out_of_bounds {
        None => assert_eq!(run.output, metal),
        // A case of out-of-bounds targets is judged by their containment.
        Some(_) => assert!(
            crate::executor_contract::out_of_bounds::matches(
                &authored,
                &run.comparison,
                &metal,
                &run.output,
                Some(
                    run.census
                        .out_of_bounds
                        .as_ref()
                        .expect("the run's containment facts")
                ),
            ),
            "the run's output and containment match Metal's"
        ),
    }
    assert_eq!(run.environment["executed_by"], "runtime");
    assert_eq!(run.environment["executor"], executor);
    assert!(
        run.environment["runtime_degrades"].is_array(),
        "the row records the run's degrade notices"
    );
    assert!(crate::c3::DeviceIdentity::recorded(&run.environment).is_some());
    assert_eq!(
        run.census.leg,
        Some(crate::candidate::requested_leg()),
        "the census records the process's leg"
    );
    // A case whose command buffers execute has no digest: the authored compile census cannot
    // compile what the runtime ran for their commands (`record_uncovered`).
    let artifact = run.census.artifact.clone().expect("the run's artifact");
    let executes = authored
        .command_buffers
        .iter()
        .any(|resource| resource.execute.is_some());
    assert_eq!(
        artifact.sha256.is_some(),
        !executes,
        "{name}: the run's artifact has a digest exactly when no command buffer executes"
    );

    let on_context = on_context
        .result
        .unwrap_or_else(|error| panic!("{name}: {error}"));
    assert_eq!(run.census.artifact, on_context.census.artifact);
    assert_eq!(run.spv_sha256, on_context.spv_sha256);
    assert_eq!(run.census.modules, on_context.census.modules);
    assert_eq!(run.census.device, on_context.census.device);
    assert_eq!(run.comparison, on_context.comparison);
    match (&run.census.out_of_bounds, &on_context.census.out_of_bounds) {
        // A fixed-function fetch past a vertex buffer in a heap of its own reads the heap's
        // bytes where the executor's allocation of its own reads zero: the facts are the executor's
        // but for the cause the runtime path records, and each output matches Metal's under its own
        // facts.
        (Some(ran), Some(executor)) if ran.targets.iter().any(|t| t.read_cause.is_some()) => {
            use crate::executor_contract::out_of_bounds::{Medium, ReadCause, TargetRun};
            for target in &ran.targets {
                let fetched = target.mechanism.medium == Medium::VertexInput;
                assert_eq!(
                    target.read_cause,
                    fetched.then_some(ReadCause::HeapBoundRange),
                    "{name}: {target:?}"
                );
            }
            let uncaused = OutOfBoundsRun {
                targets: ran
                    .targets
                    .iter()
                    .map(|target| TargetRun {
                        read_cause: None,
                        ..target.clone()
                    })
                    .collect(),
            };
            assert_eq!(&uncaused, executor, "{name}");
            // What the fetch past the buffer read on this device, for the lane's record.
            eprintln!(
                "{name}: the runtime path's output {:02x?}, the executor's {:02x?}",
                run.output, on_context.output
            );
            assert!(
                crate::executor_contract::out_of_bounds::matches(
                    &authored,
                    &on_context.comparison,
                    &metal,
                    &on_context.output,
                    Some(executor),
                ),
                "{name}: the executor's output matches Metal's"
            );
        }
        (ran, executor) => {
            assert_eq!(ran, executor, "{name}");
            assert_eq!(run.output, on_context.output, "{name}");
        }
    }
}

/// The runtime path's case runner calls through a kernel case's visible-function tables: each
/// table of its authored size, each authored entry the handle the pipeline state gives the
/// function it holds among the case's linked modules, every other entry null, with the kernel,
/// table and words of the executor context's own check (`engine_compute`'s
/// `a_kernel_calls_through_its_visible_function_table`): no committed public case holds a function
/// in a table. The tables' bytes the row's artifact records are the executor's: its compile's
/// handles of the same linked functions, encoded. An entry whose function no linked module holds
/// cannot run.
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
#[test]
fn the_runtime_path_calls_the_functions_its_tables_hold() {
    use crate::acceptance::compile_pipeline;
    use crate::acceptance::LinkedLibrary;
    use crate::candidate::engine::tables::{FunctionTable, TableFunction};
    use crate::candidate::engine::{compute_request, compute_tables, LinkedModule};
    use crate::candidate::runtime_case::{
        commit_case, compute_descriptor, linked_functions, open_device, validated_options,
        visible_function_tables, Messages, ScratchCache,
    };
    use crate::candidate::tests::engine_compute::case;
    use metal2vulkan::bind::Dispatch;
    use metal2vulkan_devtools::{engine_dir, llvm_assemble};
    use metal2vulkan_runtime::{CommandBufferStatus, PipelineOptions};
    use std::time::Duration;
    let caller = llvm_assemble(&engine_dir().join("visible_table_caller.ll")).unwrap();
    let functions = llvm_assemble(&engine_dir().join("visible_table_functions.ll")).unwrap();
    let modules = [LinkedModule {
        module_sha256: "functions".into(),
        bitcode: functions,
        library: LinkedLibrary {
            functions: ["table_add", "table_mul", "table_sub"]
                .map(str::to_owned)
                .to_vec(),
        },
    }];
    const THREADS: u32 = 12;
    let mut kernel = case(
        "visible_table_dispatch",
        vec![0; 4 * THREADS as usize],
        Dispatch::Threads {
            threads: [THREADS, 1, 1],
            threads_per_group: [THREADS, 1, 1],
        },
    );
    let entry = |function: &str| TableFunction {
        module_sha256: "functions".into(),
        function: function.into(),
    };
    kernel.function_tables = BTreeMap::from([(
        1,
        FunctionTable {
            size: 4,
            entries: BTreeMap::from([
                (0, entry("table_add")),
                (1, entry("table_mul")),
                (3, entry("table_sub")),
            ]),
        },
    )]);
    let expected = (0..THREADS)
        .flat_map(|tid| {
            match tid % 4 {
                0 => tid + 3,
                1 => tid * 3,
                2 => 0xffff_fffd,
                _ => tid.wrapping_sub(3),
            }
            .to_le_bytes()
        })
        .collect::<Vec<_>>();
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let options = validated_options(&messages, cache.path());
        let device = open_device(crate::gpu::TEST_DEVICE, options).unwrap();
        let function = device
            .new_library_with_air(&caller)
            .unwrap()
            .new_function_with_name_constant_values("visible_table_dispatch", &kernel.constants)
            .unwrap();
        let linked = linked_functions(&device, &modules, &kernel.constants).unwrap();
        let descriptor = compute_descriptor(&function, linked.functions.clone(), &kernel);
        let pipeline = device
            .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap();
        let tables = visible_function_tables(&pipeline, &kernel, &linked).unwrap();
        let words = tables.words;
        let none = BTreeMap::new();
        let committed = commit_case(
            &device,
            &kernel,
            &none,
            &BTreeMap::new(),
            &pipeline,
            tables.tables,
        )
        .unwrap();
        committed
            .command_buffer
            .wait_until_completed_for(Duration::from_secs(60))
            .unwrap();
        assert_eq!(
            committed.command_buffer.status(),
            CommandBufferStatus::Completed
        );
        assert_eq!(committed.output(&kernel).unwrap(), expected);

        // The words the artifact records are the executor's: its compile's handles, encoded.
        let [execution] = committed.executions.as_slice() else {
            panic!("one dispatch");
        };
        let (request, libraries) = compute_request(
            &crate::acceptance::retained(&caller),
            &modules,
            &kernel,
            execution.shape(),
        );
        let libraries = libraries
            .iter()
            .map(|library| library.bitcode().unwrap())
            .collect::<Vec<_>>();
        let libraries = libraries.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let compiled = compile_pipeline(&libraries, &request, device.compile_device(), None)
            .unwrap_or_else(|error| panic!("{error:?}"));
        assert_eq!(
            words,
            compute_tables(&kernel, &modules, &compiled.handles).unwrap()
        );

        let table = kernel.function_tables.get_mut(&1).unwrap();
        table.entries.insert(2, entry("table_half"));
        let unlinked = match visible_function_tables(&pipeline, &kernel, &linked) {
            Ok(_) => panic!("a table of an entry no linked module holds"),
            Err(error) => error,
        };
        assert!(matches!(unlinked, CaseError::Case(_)), "{unlinked}");
    }
    layer_reported_nothing_at_all(&messages);
}

/// The engine's candidate draws each owned object and mesh case through the runtime's mesh path
/// (`runtime_case::encoded_mesh`) and reads what Metal drew ([`crate::mesh_cases`]): a mesh grid
/// through its fragment partner, an object function's payload and grid through its mesh and
/// fragment partners, a mesh function's device write with rasterization off, a mesh grid through
/// the generated observer fragment in its fragment partner's place, an object function's launch
/// through the generated launch observer and its observer fragment, and a mesh function's payload
/// through the generated object function that writes it (the four payload cases the Metal
/// executor's test draws, one of them on a GPU only, for the reason the list states), one read past
/// the mesh function's own payload type within the object function's. A device without
/// `VK_EXT_mesh_shader` draws through the runtime's emulation, whose notice the row records, and
/// a device with it records none.
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
#[test]
fn mesh_cases_draw_through_the_runtime_as_metal_drew_them() {
    use crate::candidate::tests::engine_compute::BACKEND;
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus");
    let device = crate::gpu::TEST_DEVICE;
    let mut emulated = BTreeSet::new();
    for fixture in [
        crate::mesh_cases::grid(),
        crate::mesh_cases::object(),
        crate::mesh_cases::device_write(),
        crate::mesh_cases::observed(),
        crate::mesh_cases::launched(),
        crate::mesh_cases::payload(),
        crate::mesh_cases::payload_barrier(),
        crate::mesh_cases::payload_outputs(),
        // On a GPU only: its mesh function (`mesh_order_mesh`, and `mesh_order_solo` without an
        // object stage alike) aborts the CPU ICD's compiler ("Cannot emit physreg copy
        // instruction"), docs/FAST-PATHS.md's known gaps.
        #[cfg(m2v_gpu)]
        crate::mesh_cases::payload_order(),
    ] {
        let name = fixture.case.name.clone();
        let admitted = admit(
            &root,
            fixture.case,
            ExecutionRequest::EngineCandidate { device },
        )
        .unwrap_or_else(|error| panic!("{name}: {error}"));
        let prepared = super::super::prepare_case(device, admitted)
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(route(&prepared), Route::Runtime, "{name}");
        let RuntimeOutcome::Ran(attempt) = run(&prepared, BACKEND) else {
            panic!("{name}: the runtime raised a gap");
        };
        let run = attempt
            .result
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(run.output, fixture.metal, "{name}");
        assert_eq!(run.environment["executed_by"], "runtime", "{name}");
        assert_eq!(
            run.environment["executor"], "engine-runtime-render",
            "{name}"
        );
        let notices = run.environment["runtime_degrades"]
            .as_array()
            .unwrap_or_else(|| panic!("{name}: the row records the run's degrade notices"));
        eprintln!("{name}: degrade notices {notices:?}");
        let emulation = metal2vulkan_runtime::Degradation::EmulatedMeshDraws.code();
        emulated.insert(notices.iter().any(|code| code == emulation));
    }
    // One device draws every case alike: natively, or through the emulation each time.
    assert_eq!(emulated.len(), 1, "{emulated:?}");
}
