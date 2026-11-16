//! The engine candidate takes a case that links visible functions up to the device: in an owned
//! scratch corpus, a case over the public linked caller (`engine/linked_visible_caller`) whose
//! visible-function references name the harvested modules the Metal oracle links passes admission
//! and the executor's device-free checks, and its kernel, linked with those modules' functions,
//! compiles through a real compile worker into a pipeline that validates on every committed profile
//! it compiles for. No device is opened: the GPU run (Metal and the candidate) is the operator's.

use base64::Engine as _;
use metal2vulkan::bind::BufferRanges;
use metal2vulkan::device::DevicePolicy;
use metal2vulkan_devtools::{engine_dir, llvm_assemble, AssembleError};
use metal2vulkan_validation::acceptance::{retained, LinkedLibrary};
use metal2vulkan_validation::candidate::engine::{
    compile_case, compute_case, compute_shape, linked_modules, prepare_case, LinkedModule,
};
use metal2vulkan_validation::case::AuthoredCase;
use metal2vulkan_validation::check::admission::{admit, ExecutionRequest};
use metal2vulkan_validation::engine::load_description;
use metal2vulkan_validation::gpu::DeviceType;
use metal2vulkan_validation::hash::sha256_bytes;
use metal2vulkan_validation::l3::{check_words_l3, L3Failure};
use metal2vulkan_validation::library_module::{merge_library_module_shards, LibraryModuleRow};
use metal2vulkan_validation::worker::compile::{CompileWorker, WorkerDevice};
use metal2vulkan_validation::ScratchDir;

/// The fixture `stem`'s text and its bitcode, or `None` without `llvm-as`.
fn fixture(stem: &str) -> Option<(String, Vec<u8>)> {
    let path = engine_dir().join(format!("{stem}.ll"));
    let text = std::fs::read_to_string(&path).unwrap();
    match llvm_assemble(&path) {
        Ok(bitcode) => Some((text, bitcode)),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {stem}: {why}"),
    }
}

fn b64(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// `corpus-vulkan`, a tool that runs the engine candidate, as the compile worker, executed
/// once first: the OS may scan a freshly linked binary on its first exec, which the 20-s worker
/// deadline would count.
fn worker() -> CompileWorker {
    let program = bin_exe!("corpus-vulkan");
    let _ = std::process::Command::new(&program)
        .arg("--help")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    CompileWorker::program(program)
}

#[test]
fn a_case_with_visible_function_references_compiles_linked_through_the_worker() {
    let (Some((caller_ll, caller)), Some((functions_ll, functions))) = (
        fixture("linked_visible_caller"),
        fixture("linked_visible_functions"),
    ) else {
        return;
    };
    let scratch = ScratchDir::new("engine-linked-case").unwrap();
    let root = scratch.path();
    let library = "44".repeat(32);
    // The caller declares two kernels, so the case is over its public entry row, which the checkout
    // supplies: the scratch corpus holds only the harvested modules it links.
    let caller_sha256 = sha256_bytes(caller_ll.as_bytes());
    // The functions module defines `linked_scale` and `linked_blend`; `linked_scale` references
    // `linked_offset`, which the caller's own library defines and so also supplies.
    let module = |ll: &str, bitcode: &[u8], label: &str| LibraryModuleRow {
        module_sha256: sha256_bytes(ll.as_bytes()),
        air_ll: ll.into(),
        blob_b64: b64(bitcode),
        lib_sha256s: vec![library.clone()],
        label: label.into(),
    };
    let functions_module = module(
        &functions_ll,
        &functions,
        "test/linked_visible_functions.air",
    );
    let caller_module = module(&caller_ll, &caller, "test/linked_visible_caller.air");
    merge_library_module_shards(root, [functions_module.clone(), caller_module.clone()]).unwrap();
    let threads = 8u32;
    let bytes = vec![0u8; 4 * threads as usize];
    let mut case: AuthoredCase = serde_json::from_value(serde_json::json!({
        "air_sha256": caller_sha256,
        "case_id": "pending",
        "name": "linked-visible-caller",
        "entry": "linked_visible_caller",
        "stage": "kernel",
        "buffers": [
            {"binding": 0, "role": "output", "initial_bytes_b64": b64(&bytes)},
            {"binding": 1, "role": "output", "initial_bytes_b64": b64(&bytes)},
        ],
        "visible_function_references": [
            {"module_sha256": functions_module.module_sha256, "function": "linked_blend"},
            {"module_sha256": caller_module.module_sha256, "function": "linked_offset"},
            {"module_sha256": functions_module.module_sha256, "function": "linked_sample"},
            {"module_sha256": functions_module.module_sha256, "function": "linked_scale"},
        ],
        "dispatch": {"grid": [threads, 1, 1], "threads_per_threadgroup": [threads, 1, 1]},
        "output": {"kind": "buffer", "binding": 0, "offset": 0, "length": bytes.len()},
        "compare": {"kind": "exact"},
        "execution_safety": "loop_free",
    }))
    .unwrap();
    case.case_id = case.computed_case_id().unwrap();

    let admitted = admit(
        root,
        case,
        ExecutionRequest::EngineCandidate {
            device: DeviceType::Gpu,
        },
    )
    .unwrap_or_else(|error| panic!("admission: {error}"));
    let checked = admitted.checked();
    let compute = compute_case(&checked.primary.case, &checked.primary.interface).unwrap();
    // Each module once, with the functions the Metal oracle links from it, in reference order.
    let linked = linked_modules(&checked.primary.linked_functions).unwrap();
    let names = |names: &[&str]| LinkedLibrary {
        functions: names.iter().map(|name| (*name).to_owned()).collect(),
    };
    assert_eq!(
        linked,
        [
            LinkedModule {
                module_sha256: functions_module.module_sha256.clone(),
                bitcode: functions.clone(),
                library: names(&["linked_blend", "linked_sample", "linked_scale"]),
            },
            LinkedModule {
                module_sha256: caller_module.module_sha256.clone(),
                bitcode: caller.clone(),
                library: names(&["linked_offset"]),
            },
        ]
    );
    // The executor's device-free checks no longer refuse a case for linking.
    if let Err(error) = prepare_case(DeviceType::Gpu, admitted) {
        panic!("the executor's device-free checks refused the linked case: {error}");
    }

    let worker = worker();
    for stem in ["moltenvk-m3", "vk12-floor", "lavapipe"] {
        let device =
            WorkerDevice::new(load_description(stem).unwrap(), DevicePolicy::default()).unwrap();
        let profile = device.device().profile();
        let addressed = profile.descriptors().buffer_device_address;
        let shape = compute_shape(&compute, addressed, BufferRanges::of(profile.limits())).unwrap();
        let pipeline = compile_case(
            &worker,
            &device,
            &retained(&caller),
            &linked,
            &compute,
            &shape,
        )
        .unwrap_or_else(|error| panic!("{stem}: {error}"))
        .pipeline;
        match check_words_l3(&pipeline.stage.spirv, &pipeline.requirements, profile) {
            Ok(()) => {}
            Err(L3Failure::Tooling(why)) => eprintln!("SKIPPED spirv-val on {stem}: {why}"),
            Err(failure) => panic!("{stem}: {failure}"),
        }
        // Without the linked modules the same compile is the product's typed refusal of the
        // unresolved reference, never a candidate-side one.
        match compile_case(&worker, &device, &retained(&caller), &[], &compute, &shape) {
            Err(error) => assert!(
                error.to_string().contains("LinkedFunction"),
                "{stem}: {error}"
            ),
            Ok(_) => panic!("{stem}: the caller compiled without its linked functions"),
        }
    }
}

/// A case with a visible-function table over the public table caller
/// (`engine/visible_table_caller`, entry `tid % size` of the table at buffer 1) passes the
/// executor's device-free checks, and its kernel, linked with the table's functions, compiles
/// through a real compile worker, which returns the product's handles of those functions with
/// the pipeline: the table's buffer is the product's table words, bound at the table's index in
/// the shape the pipeline was compiled for. No device is opened.
#[test]
fn a_case_with_a_visible_function_table_binds_the_products_table_words() {
    use metal2vulkan::bind::BoundBuffer;
    use metal2vulkan::pipeline::Stage;
    use metal2vulkan_validation::candidate::engine::bind_state;
    let (Some((caller_ll, caller)), Some((functions_ll, functions))) = (
        fixture("visible_table_caller"),
        fixture("visible_table_functions"),
    ) else {
        return;
    };
    let scratch = ScratchDir::new("engine-table-case").unwrap();
    let root = scratch.path();
    let library = "55".repeat(32);
    // The caller declares two kernels, so the case is over its public entry row, which the checkout
    // supplies: the scratch corpus holds only the harvested module it links.
    let caller_sha256 = sha256_bytes(caller_ll.as_bytes());
    let functions_module = LibraryModuleRow {
        module_sha256: sha256_bytes(functions_ll.as_bytes()),
        air_ll: functions_ll.clone(),
        blob_b64: b64(&functions),
        lib_sha256s: vec![library.clone()],
        label: "test/visible_table_functions.air".into(),
    };
    merge_library_module_shards(root, [functions_module.clone()]).unwrap();
    let module = functions_module.module_sha256.clone();
    let threads = 10u32;
    let bytes = vec![0u8; 4 * threads as usize];
    // `table_add`, `table_mul`, a null entry, `table_sub`, and `table_half` (another type).
    let entry = |index: u32, function: &str| serde_json::json!({"index": index, "module_sha256": module, "function": function});
    let mut case: AuthoredCase = serde_json::from_value(serde_json::json!({
        "air_sha256": caller_sha256,
        "case_id": "pending",
        "name": "visible-table-dispatch",
        "entry": "visible_table_dispatch",
        "stage": "kernel",
        "buffers": [
            {"binding": 0, "role": "output", "initial_bytes_b64": b64(&bytes)},
        ],
        // The module's other kernel references these two directly; this entry calls them
        // through the table. Linking takes each once.
        "visible_function_references": [
            {"module_sha256": module, "function": "table_add"},
            {"module_sha256": module, "function": "table_mul"},
        ],
        "visible_function_tables": [{
            "binding": 1,
            "size": 5,
            "entries": [
                entry(0, "table_add"),
                entry(1, "table_mul"),
                entry(3, "table_sub"),
                entry(4, "table_half"),
            ],
        }],
        "dispatch": {"grid": [threads, 1, 1], "threads_per_threadgroup": [threads, 1, 1]},
        "output": {"kind": "buffer", "binding": 0, "offset": 0, "length": bytes.len()},
        "compare": {"kind": "exact"},
        "execution_safety": "loop_free",
    }))
    .unwrap();
    case.case_id = case.computed_case_id().unwrap();

    let admitted = admit(
        root,
        case,
        ExecutionRequest::EngineCandidate {
            device: DeviceType::Gpu,
        },
    )
    .unwrap_or_else(|error| panic!("admission: {error}"));
    let checked = admitted.checked();
    let compute = compute_case(&checked.primary.case, &checked.primary.interface).unwrap();
    assert_eq!(compute.function_tables[&1].size, 5);
    // The referenced functions, then the table's others in entry order, from their one module.
    let linked = linked_modules(&checked.primary.linked_functions).unwrap();
    assert_eq!(
        linked,
        [LinkedModule {
            module_sha256: module.clone(),
            bitcode: functions.clone(),
            library: LinkedLibrary {
                functions: ["table_add", "table_mul", "table_sub", "table_half"]
                    .map(str::to_owned)
                    .to_vec(),
            },
        }]
    );
    if let Err(error) = prepare_case(DeviceType::Gpu, admitted) {
        panic!("the executor's device-free checks refused the table case: {error}");
    }

    let worker = worker();
    for stem in ["moltenvk-m3", "vk12-floor", "lavapipe"] {
        let device =
            WorkerDevice::new(load_description(stem).unwrap(), DevicePolicy::default()).unwrap();
        let profile = device.device().profile();
        let addressed = profile.descriptors().buffer_device_address;
        let shape = compute_shape(&compute, addressed, BufferRanges::of(profile.limits())).unwrap();
        let compiled = compile_case(
            &worker,
            &device,
            &retained(&caller),
            &linked,
            &compute,
            &shape,
        )
        .unwrap_or_else(|error| panic!("{stem}: {error}"));
        // Bound, size 5, then each entry's handle: the linked `functions` from 2, null at 2.
        let words = compiled.tables[&(Stage::Kernel, 1)]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect::<Vec<_>>();
        assert_eq!(words, [1, 5, 2, 3, 0, 4, 5], "{stem}");
        assert_eq!(compiled.tables.len(), 1, "{stem}");
        // The table is a buffer of its words' length at its index, in the compiled-for shape.
        let state = bind_state(
            &compute,
            &Default::default(),
            &Default::default(),
            BufferRanges::default(),
        )
        .unwrap();
        match state.stages[&Stage::Kernel].buffers.get(&1) {
            Some(BoundBuffer::Buffer { length, .. }) => assert_eq!(*length, 28, "{stem}"),
            other => panic!("{stem}: {other:?}"),
        }
        let pipeline = compiled.pipeline;
        match check_words_l3(&pipeline.stage.spirv, &pipeline.requirements, profile) {
            Ok(()) => {}
            Err(L3Failure::Tooling(why)) => eprintln!("SKIPPED spirv-val on {stem}: {why}"),
            Err(failure) => panic!("{stem}: {failure}"),
        }
        // The plan of the executor's shape binds a buffer at the table's index.
        let plan = pipeline.plan(device.device(), &shape).unwrap();
        assert!(
            plan.diagnostics.is_empty(),
            "{stem}: {:?}",
            plan.diagnostics
        );
    }
}
