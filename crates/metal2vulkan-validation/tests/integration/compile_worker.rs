//! The engine compile worker across a real process boundary: every validation binary that
//! compiles engine pipelines for a device serves the worker, and what comes back through the
//! pipeline's cache artifact bytes is the pipeline the public API compiles in process. A pipeline
//! holding a constant blob crosses it without the worker copying the blob.

use metal2vulkan::device::DevicePolicy;
use metal2vulkan::library::FunctionConstantValues;
use metal2vulkan::pipeline::ComputePipeline;
use metal2vulkan_devtools::{engine_dir, llvm_assemble, AssembleError};
use metal2vulkan_validation::acceptance::{
    compile_draw_in_worker, compile_pipeline, compile_row_in_worker, retained, row,
    AcceptancePipeline, PipelineRequest, Projection, RequestFunction, RowCompile, RowPipeline,
    RowStage,
};
use metal2vulkan_validation::engine::{load_description, RowSource};
use metal2vulkan_validation::worker::compile::{
    compile_compute, CompileWorker, NotCompiled, WorkerCompile, WorkerDevice,
};
use metal2vulkan_validation::worker::{WorkerFailure, DEADLINE, MEMORY_LIMIT_BYTES};
use std::path::{Path, PathBuf};

fn fixture(name: &str) -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join(name)) {
        Ok(bytes) => Some(bytes),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}: {why}"),
    }
}

/// `program`, executed once before any compile: the OS may scan a freshly linked binary on its
/// first exec (seconds on macOS), which the 20-s worker deadline would count. A tool's worker is
/// the tool's own executable, which has run by the time it spawns one.
fn worker(program: &Path) -> CompileWorker {
    let _ = std::process::Command::new(program)
        .arg("--help")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    CompileWorker::program(program)
}

fn device(stem: &str) -> WorkerDevice {
    WorkerDevice::new(load_description(stem).unwrap(), DevicePolicy::default()).unwrap()
}

/// Every binary that compiles through the worker serves it (and installs the allocator it
/// refuses to serve without).
fn serving() -> [PathBuf; 2] {
    [bin_exe!("corpus-pipeline"), bin_exe!("corpus-vulkan")]
}

#[test]
fn every_device_tool_serves_the_compile_worker_with_the_public_apis_pipeline() {
    let Some(bitcode) = fixture("store_const.ll") else {
        return;
    };
    let device = device("moltenvk-m3");
    let request = PipelineRequest::Compute {
        entry: "store_const".into(),
        constants: FunctionConstantValues::defaults(),
    };
    let AcceptancePipeline::Compute(expected) =
        compile_pipeline(&[&bitcode], &request, device.device(), None)
            .unwrap()
            .pipeline
    else {
        panic!("a kernel compiles to a compute pipeline");
    };
    for path in serving() {
        let program = path.display();
        let worker = worker(&path);
        let compiled = compile_compute(
            &worker,
            &device,
            &retained(&bitcode),
            "store_const",
            &FunctionConstantValues::defaults(),
        )
        .unwrap_or_else(|failure| panic!("{program}: {}", failure.message()));
        let compiled_pipeline = compiled.result.unwrap();
        assert_eq!(compiled_pipeline.pipeline, *expected, "{program}");
        assert!(compiled_pipeline.handles.is_empty(), "{program}");
        assert!(
            compiled.allocator_peak.is_some_and(|peak| peak > 0),
            "{program}"
        );
        assert!(compiled.measured.elapsed < DEADLINE, "{program}");
        if cfg!(any(target_os = "macos", target_os = "linux")) {
            assert!(compiled.measured.peak_resident_bytes.is_some(), "{program}");
        }
        // The worker's report carries its counting allocator's process peak beside the sampled
        // RSS: at least the compile's own peak, which it contains.
        let process_peak = compiled.measured.allocator_peak_bytes;
        assert!(
            process_peak.is_some_and(|peak| peak >= compiled.allocator_peak.unwrap()),
            "{program}: {process_peak:?}"
        );
        assert!(compiled.measured.peak_bytes() >= process_peak, "{program}");
    }
}

/// A logged draw (`corpus-pipeline --draws`) crosses the worker boundary: the vertex and
/// fragment functions of two sources, drawn together under the logged template, come back as the
/// render pipeline the in-process compile builds, a fragment row's.
#[test]
fn a_logged_draw_compiles_in_the_worker_as_in_process() {
    let Some(bitcode) = fixture("stage_io.ll") else {
        return;
    };
    let worker = worker(&bin_exe!("corpus-pipeline"));
    let device = device("moltenvk-m3");
    let undefined = FunctionConstantValues::defaults();
    let mut template = row::fragment_row(&bitcode, "stage_io_fragment", &undefined)
        .unwrap()
        .template;
    template.vertex_descriptor = row::vertex_row(&bitcode, "stage_io_vertex", &undefined)
        .unwrap()
        .template
        .vertex_descriptor;
    assert!(template.vertex_descriptor.is_some());
    let request = PipelineRequest::Draw {
        vertex: RequestFunction::new("stage_io_vertex", undefined.clone()),
        fragment: RequestFunction::new("stage_io_fragment", undefined),
        template: template.clone(),
        shape: None,
    };
    let AcceptancePipeline::Render(expected) =
        compile_pipeline(&[&bitcode, &bitcode], &request, device.device(), None)
            .unwrap()
            .pipeline
    else {
        panic!("a draw compiles to a render pipeline");
    };
    let (compiled, measured) = compile_draw_in_worker(
        &worker,
        &device,
        &[retained(&bitcode), retained(&bitcode)],
        "stage_io_vertex",
        "stage_io_fragment",
        &template,
    );
    let RowCompile::Compiled(pipeline) = compiled else {
        panic!("{compiled:?}");
    };
    let RowPipeline::Render {
        pipeline,
        row: RowStage::Fragment,
    } = *pipeline
    else {
        panic!("a draw came back as another kind");
    };
    assert_eq!(pipeline.stages.len(), 2);
    assert_eq!(*pipeline, *expected);
    assert!(measured.is_some());
}

#[test]
fn a_vertex_function_and_a_refusal_cross_the_worker_boundary() {
    let Some(bitcode) = fixture("stage_io.ll") else {
        return;
    };
    let worker = worker(&bin_exe!("corpus-pipeline"));
    let device = device("moltenvk-m3");
    // A vertex function that reads attributes and a fragment function, as L4 rows: the
    // descriptor and the companion are derived (and the companion assembled) in the worker, and
    // the render pipeline itself (not a refusal) crosses the boundary.
    for (entry, stage, stages) in [
        ("stage_io_vertex", RowStage::Vertex, 1),
        ("stage_io_fragment", RowStage::Fragment, 2),
    ] {
        let request = PipelineRequest::Row {
            entry: entry.into(),
            stage,
            projection: Projection::Derived,
            constants: FunctionConstantValues::defaults(),
        };
        let AcceptancePipeline::Render(expected) =
            compile_pipeline(&[&bitcode], &request, device.device(), None)
                .unwrap()
                .pipeline
        else {
            panic!("{entry} compiles to a render pipeline");
        };
        let (compiled, measured) = compile_row_in_worker(
            &worker,
            &device,
            &retained(&bitcode),
            entry,
            stage,
            Projection::Derived,
            &FunctionConstantValues::defaults(),
        );
        let RowCompile::Compiled(pipeline) = compiled else {
            panic!("{entry}: {compiled:?}");
        };
        let RowPipeline::Render { pipeline, .. } = *pipeline else {
            panic!("{entry} came back as another kind");
        };
        assert_eq!(*pipeline, *expected);
        assert_eq!(pipeline.stages.len(), stages);
        assert!(measured.unwrap().elapsed < DEADLINE);
    }

    // The product's typed refusal comes back as its outcome, not as a worker failure.
    let refused = compile_compute(
        &worker,
        &device,
        &retained(&bitcode),
        "no_such_entry",
        &Default::default(),
    )
    .unwrap();
    let Err(NotCompiled::Product(outcome, detail)) = refused.result else {
        panic!("no_such_entry compiled");
    };
    assert_eq!(outcome.label(), "invalid_descriptor", "{detail}");
}

#[test]
fn a_program_that_does_not_serve_the_worker_is_a_worker_failure() {
    let device = device("runtime-floor");
    // `true` exits successfully without a result: the result cannot be read.
    match compile_compute(
        &CompileWorker::program("true"),
        &device,
        &retained(b"bitcode"),
        "main",
        &Default::default(),
    ) {
        Err(failure @ WorkerFailure::Output(..)) => {
            assert_eq!(failure.outcome().key(), "worker_exit")
        }
        other => panic!("{other:?}"),
    }
    // A program that cannot start is a setup failure (transient tooling).
    match compile_compute(
        &CompileWorker::program("/nonexistent/compile-worker"),
        &device,
        &retained(b"bitcode"),
        "main",
        &Default::default(),
    ) {
        Err(failure @ WorkerFailure::Setup(_)) => assert!(failure.outcome().is_transient()),
        other => panic!("{other:?}"),
    }
}

/// A kernel reading two words of `@g`, a closed constant struct DAG `depth` levels deep
/// (`%e1 = { i32, i32 }`, `%e{k} = { %e{k-1}, %e{k-1} }`), all zero: past 14 levels `@g` is in
/// the constant blob, `2^(depth+2)` bytes (the product's `tests/constant_blob_footprint.rs`).
fn blob_kernel(depth: u32) -> String {
    let mut types = String::from("%e1 = type { i32, i32 }\n");
    for k in 2..=depth {
        types += &format!("%e{k} = type {{ %e{p}, %e{p} }}\n", p = k - 1);
    }
    format!(
        r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

{types}@g = internal addrspace(2) constant %e{depth} zeroinitializer, align 4

define void @blob(ptr addrspace(1) %out, i32 %t) {{
  %v0 = load i32, ptr addrspace(2) @g, align 4
  %q = getelementptr inbounds i8, ptr addrspace(2) @g, i32 4
  %v1 = load i32, ptr addrspace(2) %q, align 4
  %i0 = shl i32 %t, 1
  %o0 = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %i0
  store i32 %v0, ptr addrspace(1) %o0, align 4
  %i1 = add i32 %i0, 1
  %o1 = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %i1
  store i32 %v1, ptr addrspace(1) %o1, align 4
  ret void
}}

!air.kernel = !{{!0}}
!0 = !{{ptr @blob, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4}}
!3 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}}
!4 = !{{i32 1, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}}
"#
    )
}

/// Compile [`blob_kernel`] `depth` levels deep for `lavapipe` through `corpus-pipeline`'s compile
/// worker (the child assembles the text): the compile, and its blob's length, `2^(depth+2)`.
/// Under the resident-table domains: the identity domains refuse constant data over 64 KiB.
fn compile_blob(depth: u32) -> (WorkerCompile<ComputePipeline>, u64) {
    let worker = worker(&bin_exe!("corpus-pipeline"));
    let text = blob_kernel(depth);
    let compiled = compile_compute(
        &worker,
        &WorkerDevice::new(
            load_description("lavapipe").unwrap(),
            DevicePolicy::RESIDENT_TABLES,
        )
        .unwrap(),
        &RowSource::Text(text.as_str().into()),
        "blob",
        &FunctionConstantValues::defaults(),
    )
    .unwrap_or_else(|failure| panic!("{depth} levels: {}", failure.message()));
    let blob = match &compiled.result {
        Ok(compiled) => compiled
            .pipeline
            .stage
            .constant_blob
            .as_ref()
            .map_or(0, |blob| blob.len() as u64),
        Err(refused) => panic!("{depth} levels: {refused:?}"),
    };
    assert_eq!(blob, 4 << depth, "{depth} levels");
    (compiled, blob)
}

/// The worker reports a pipeline holding a constant blob without copying the blob: it writes the
/// artifact to the parent's file a chunk at a time, so its process's allocator peak passes its
/// compile's by under a quarter of the blob (4 MiB at 20 levels). Encoding the artifact whole and
/// then as base64 for the report held over twice the blob beside the pipeline.
#[test]
fn a_constant_blob_crosses_the_worker_without_a_copy() {
    if fixture("store_const.ll").is_none() {
        return;
    }
    let (compiled, blob) = compile_blob(20);
    let process = compiled.measured.allocator_peak_bytes.unwrap();
    let compile = compiled.allocator_peak.unwrap();
    eprintln!("20 levels: process peak {process} bytes, compile peak {compile} bytes, blob {blob}");
    assert!(
        process < compile + blob / 4,
        "process peak {process} bytes, compile peak {compile} bytes, blob {blob}"
    );
}

/// The largest closed constant the ledger admits, 26 levels (a 256-MiB blob), compiles through
/// the worker boundary within its ceilings: under 500 MiB, the sampled resident set and the
/// allocator's peak alike, and within the 20-second deadline. The ceilings are release-mode
/// contracts, so this runs in release, ignored otherwise.
#[test]
#[ignore = "a release-mode measurement of the worker's ceilings"]
fn the_largest_admitted_constant_blob_compiles_within_the_worker_ceilings() {
    if fixture("store_const.ll").is_none() {
        return;
    }
    let (compiled, blob) = compile_blob(26);
    let measured = compiled.measured;
    eprintln!(
        "26 levels: blob {blob} bytes, {:?}, resident peak {:?} bytes, allocator peak {:?} \
         bytes, compile peak {:?} bytes",
        measured.elapsed,
        measured.peak_resident_bytes,
        measured.allocator_peak_bytes,
        compiled.allocator_peak
    );
    assert!(measured.elapsed < DEADLINE, "{:?}", measured.elapsed);
    let peak = measured.peak_bytes().unwrap();
    assert!(peak < MEMORY_LIMIT_BYTES, "{peak} bytes");
}
