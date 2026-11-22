//! The engine candidate takes a case that links visible functions up to the device: in an owned
//! scratch corpus, a case over the public linked caller (`engine/linked_visible_caller`) whose
//! visible-function references name the harvested modules the Metal oracle links passes admission
//! and the executor's device-free checks, and its kernel, linked with those modules' functions,
//! compiles through a real compile worker into a pipeline that validates on every committed profile
//! it compiles for. No device is opened: the GPU run (Metal and the candidate) is the operator's.

use base64::Engine as _;
use metal2vulkan::device::DevicePolicy;
use metal2vulkan_devtools::{engine_dir, llvm_assemble, AssembleError};
use metal2vulkan_validation::acceptance::{retained, LinkedLibrary};
use metal2vulkan_validation::candidate::engine::{
    compute_case, kernel_request, linked_modules, prepare_case, LinkedModule,
};
use metal2vulkan_validation::case::AuthoredCase;
use metal2vulkan_validation::check::admission::{admit, ExecutionRequest};
use metal2vulkan_validation::engine::load_description;
use metal2vulkan_validation::gpu::DeviceType;
use metal2vulkan_validation::hash::sha256_bytes;
use metal2vulkan_validation::l3::{check_words_l3, L3Failure};
use metal2vulkan_validation::library_module::{merge_library_module_shards, LibraryModuleRow};
use metal2vulkan_validation::worker::compile::{
    compile_compute_request, CompileWorker, WorkerDevice,
};
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
    let compute = compute_case(&checked.primary.case).unwrap();
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
    for stem in ["moltenvk-m3", "runtime-floor", "lavapipe"] {
        let device =
            WorkerDevice::new(load_description(stem).unwrap(), DevicePolicy::default()).unwrap();
        let profile = device.device().profile();
        let compile = |linked: &[LinkedModule]| {
            let (request, libraries) = kernel_request(&retained(&caller), linked, &compute);
            compile_compute_request(&worker, &device, &libraries, &request)
                .unwrap_or_else(|error| panic!("{stem}: {error:?}"))
                .result
        };
        let pipeline = compile(&linked).unwrap_or_else(|error| panic!("{stem}: {error:?}"));
        match check_words_l3(&pipeline.stage.spirv, &pipeline.requirements, profile) {
            Ok(()) => {}
            Err(L3Failure::Tooling(why)) => eprintln!("SKIPPED spirv-val on {stem}: {why}"),
            Err(failure) => panic!("{stem}: {failure}"),
        }
        // Without the linked modules the same compile is the product's typed refusal of the
        // unresolved reference, never a candidate-side one.
        match compile(&[]) {
            Err(error) => assert!(
                format!("{error:?}").contains("LinkedFunction"),
                "{stem}: {error:?}"
            ),
            Ok(_) => panic!("{stem}: the caller compiled without its linked functions"),
        }
    }
}
