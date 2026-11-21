//! The engine compute candidate on a device: the binder's plan executed through Vulkan under
//! `VK_LAYER_KHRONOS_validation`, with the status read before the output. `G-linux`'s `m2v_cpu_icd`
//! step runs these on lavapipe. The modules carry no execution bound: the looping kernels run as
//! compiled on both lanes, GPU and CPU ICD alike.

use super::*;
use crate::acceptance::LinkedLibrary;
use crate::candidate::engine::tests::{assemble, buffer_table_kernel, counting_kernel};
use crate::candidate::engine::{
    compile_case, ComputeCase, ComputeOutput, ComputeRun, LinkedModule,
};
use crate::check::admission::ExecutorError;
use metal2vulkan::bind::Dispatch;
use metal2vulkan::library::FunctionConstantValues;
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use std::collections::BTreeMap;

pub(crate) fn case(entry: &str, buffer: Vec<u8>, dispatch: Dispatch) -> ComputeCase {
    let length = buffer.len() as u64;
    ComputeCase {
        out_of_bounds: Vec::new(),
        creation_bound: crate::budgets::PIPELINE_CREATION_CEILING,
        entry: entry.into(),
        constants: FunctionConstantValues::defaults(),
        buffers: BTreeMap::from([(0, buffer)]),
        argument_buffer_buffers: BTreeMap::new(),
        argument_buffer_textures: BTreeMap::new(),
        argument_buffer_samplers: BTreeMap::new(),
        textures: BTreeMap::new(),
        samplers: BTreeMap::new(),
        threadgroup_lengths: BTreeMap::new(),
        dispatch,
        output: ComputeOutput::Buffer {
            binding: 0,
            offset: 0,
            length,
        },
    }
}

fn one_thread() -> Dispatch {
    Dispatch::Threads {
        threads: [1, 1, 1],
        threads_per_group: [1, 1, 1],
    }
}

pub(crate) fn run(case: &ComputeCase, bitcode: &[u8]) -> Result<ComputeRun, ExecutorError> {
    run_linked(case, bitcode, &[])
}

/// [`run`] of a kernel that links the functions of `linked`, compiled in the device tests' compile
/// worker ([`compile_worker`]) by the request the tools send it, with its visible-function tables
/// encoded from the handles the compile gave.
fn run_linked(
    case: &ComputeCase,
    bitcode: &[u8],
    linked: &[LinkedModule],
) -> Result<ComputeRun, ExecutorError> {
    platform::execute_engine(
        crate::gpu::TEST_DEVICE,
        "engine-test",
        case,
        |device, shape| {
            compile_case(
                compile_worker(),
                device,
                &crate::acceptance::retained(bitcode),
                linked,
                case,
                shape,
            )
        },
    )
    .result
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn a_store_runs_through_the_binder_under_the_validation_layer() {
    let bitcode = llvm_assemble(&engine_dir().join("store_const.ll")).unwrap();
    let run = run(&case("store_const", vec![0xab; 8], one_thread()), &bitcode).unwrap();
    let mut expected = vec![0xab; 8];
    expected[..4].copy_from_slice(&42u32.to_le_bytes());
    assert_eq!(run.output, expected);
    assert_eq!(
        run.status,
        Some(metal2vulkan::bind::StatusReport::default())
    );
    assert_eq!(run.environment["executor"], "engine-compute");
    assert_eq!(run.translator_fingerprint.len(), 64);
}

/// A kernel that calls visible functions it does not define runs with the functions its case
/// links: `linked_scale` and `linked_blend` of the public functions library and `linked_offset` of
/// the kernel's own, linked statically by the compile, compute the values the MSL states (`3 *
/// tid + (tid & 7)` and `tid * 0.5 + 1`).
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn a_kernel_runs_with_its_linked_visible_functions() {
    let caller = llvm_assemble(&engine_dir().join("linked_visible_caller.ll")).unwrap();
    let functions = llvm_assemble(&engine_dir().join("linked_visible_functions.ll")).unwrap();
    let names = |names: &[&str]| LinkedLibrary {
        functions: names.iter().map(|name| (*name).to_owned()).collect(),
    };
    let linked = [
        LinkedModule {
            module_sha256: "functions".into(),
            bitcode: functions,
            library: names(&["linked_blend", "linked_scale"]),
        },
        LinkedModule {
            module_sha256: "caller".into(),
            bitcode: caller.clone(),
            library: names(&["linked_offset"]),
        },
    ];
    const THREADS: u32 = 16;
    let bytes = vec![0; 4 * THREADS as usize];
    let mut kernel = case(
        "linked_visible_caller",
        bytes.clone(),
        Dispatch::Threads {
            threads: [THREADS, 1, 1],
            threads_per_group: [THREADS, 1, 1],
        },
    );
    kernel.buffers.insert(1, bytes.clone());
    let words = run_linked(&kernel, &caller, &linked).unwrap().output;
    let expected = (0..THREADS)
        .flat_map(|tid| (3 * tid + (tid & 7)).to_le_bytes())
        .collect::<Vec<_>>();
    assert_eq!(words, expected);
    kernel.output = ComputeOutput::Buffer {
        binding: 1,
        offset: 0,
        length: bytes.len() as u64,
    };
    let floats = run_linked(&kernel, &caller, &linked).unwrap().output;
    let expected = (0..THREADS)
        .flat_map(|tid| (tid as f32 * 0.5 + 1.0).to_le_bytes())
        .collect::<Vec<_>>();
    assert_eq!(floats, expected);
}

/// A kernel runs with the public linked modules, each found by its `.ll` text's hash and linked as
/// the executor links a case's (`linked_modules`): its text assembled. `linked_mps_callbacks` reads
/// element `tid` of its second and third buffers through `prefixSecondary_f` and `prefixTertiary_f`
/// and stores `(a, b, c, a + b + c + popcount(tid))` through `postfixPrimary_4xf` at element `4
/// tid` of its last buffer, which the callback loads from the kernel's table of buffer addresses.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn a_kernel_runs_with_the_public_linked_modules() {
    use crate::candidate::engine::linked_modules;
    use crate::library_module::{
        defined_function_names, public_library_modules, ResolvedFunctionReference,
        ResolvedLinkedFunctions,
    };
    let caller = llvm_assemble(&engine_dir().join("linked_mps_callbacks.ll")).unwrap();
    let modules = public_library_modules().unwrap();
    let reference = |function: &str| ResolvedFunctionReference {
        function: function.into(),
        module: modules
            .iter()
            .find(|module| {
                defined_function_names(&module.air_ll)
                    .iter()
                    .any(|f| f == function)
            })
            .unwrap()
            .clone(),
    };
    let linked = linked_modules(&ResolvedLinkedFunctions {
        references: vec![
            reference("postfixPrimary_4xf"),
            reference("prefixSecondary_f"),
            reference("prefixTertiary_f"),
        ],
        visible: vec![],
        intersection: vec![],
    })
    .unwrap();
    assert_eq!(linked.len(), 2);
    const THREADS: u32 = 16;
    let floats = |value: &dyn Fn(f32) -> f32| {
        (0..THREADS)
            .flat_map(|tid| value(tid as f32).to_le_bytes())
            .collect::<Vec<_>>()
    };
    let mut kernel = case(
        "linked_mps_callbacks",
        floats(&|tid| tid),
        Dispatch::Threads {
            threads: [THREADS, 1, 1],
            threads_per_group: [THREADS, 1, 1],
        },
    );
    kernel.buffers.insert(1, floats(&|tid| 100.0 + tid));
    kernel.buffers.insert(2, floats(&|tid| 1000.0 + 2.0 * tid));
    kernel.buffers.insert(3, vec![0xab; 16 * THREADS as usize]);
    kernel.buffers.insert(4, vec![0; 4]);
    kernel.output = ComputeOutput::Buffer {
        binding: 3,
        offset: 0,
        length: 16 * u64::from(THREADS),
    };
    let words = run_linked(&kernel, &caller, &linked).unwrap().output;
    let expected = (0..THREADS)
        .flat_map(|tid| {
            let (a, b, c) = (tid as f32, 100.0 + tid as f32, 1000.0 + 2.0 * tid as f32);
            [a, b, c, a + b + c + tid.count_ones() as f32]
        })
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    assert_eq!(words, expected);
}

/// A loop runs to completion as compiled, with no execution bound, on both lanes.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn a_loop_runs_to_completion_as_compiled() {
    let bitcode = assemble(&counting_kernel()).expect("llvm-as on the device lane");
    let ok = run(&case("count", vec![0; 4], one_thread()), &bitcode).unwrap();
    assert_eq!(ok.output, 4u32.to_le_bytes());
}

/// `dispatchThreads` with a partial edge group: the binder splits it into regions, and every
/// thread sees Metal's grid builtins.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn a_partial_threadgroup_edge_computes_metals_grid_builtins() {
    let text = format!(
        "{}define void @grid_ids(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0, i32 noundef %tid, i32 noundef %grp, i32 noundef %tpg) {{\n\
entry:\n\
  %s = shl i32 %tpg, 16\n\
  %v = or i32 %s, %grp\n\
  %i = zext i32 %tid to i64\n\
  %p = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %i\n\
  store i32 %v, ptr addrspace(1) %p, align 4\n\
  ret void\n\
}}\n\n\
!air.kernel = !{{!0}}\n!air.version = !{{!6}}\n!air.language_version = !{{!7}}\n\
!0 = !{{ptr @grid_ids, !1, !2}}\n!1 = !{{}}\n!2 = !{{!3, !4, !5, !8}}\n\
!3 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"out\"}}\n\
!4 = !{{i32 1, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"tid\"}}\n\
!5 = !{{i32 2, !\"air.threadgroup_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"grp\"}}\n\
!8 = !{{i32 3, !\"air.threads_per_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"tpg\"}}\n\
!6 = !{{i32 2, i32 8, i32 0}}\n!7 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n",
        counting_kernel()
            .split("define")
            .next()
            .expect("the shared header")
    );
    let bitcode = assemble(&text).expect("llvm-as on the device lane");
    let threads = 5u32;
    let dispatch = Dispatch::Threads {
        threads: [threads, 1, 1],
        threads_per_group: [2, 1, 1],
    };
    let run = run(
        &case("grid_ids", vec![0xff; 4 * threads as usize], dispatch),
        &bitcode,
    )
    .unwrap();
    let expected: Vec<u8> = (0..threads)
        .flat_map(|tid| ((threads << 16) | (tid / 2)).to_le_bytes())
        .collect();
    assert_eq!(run.output, expected);
}

/// A device buffer array: its elements are the buffers at consecutive slots, and the function reads
/// them as its buffer table: `out[1] = tab[0][0] + tab[1][1]`.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn a_device_buffer_array_binds_its_elements_at_consecutive_slots() {
    let bitcode = assemble(&buffer_table_kernel()).expect("llvm-as on the device lane");
    let mut table = case("table", vec![0xff; 8], one_thread());
    let word = |value: u32| value.to_le_bytes().to_vec();
    table.buffers.insert(1, [word(40), word(1)].concat());
    table.buffers.insert(2, [word(3), word(2)].concat());
    let run = run(&table, &bitcode).unwrap();
    assert_eq!(run.output, [vec![0xff; 4], word(42)].concat());
}

/// An argument-buffer buffer: the root node's embedded pointer is the second node's guest address,
/// written into the owner at its field and reached through the resident set. The public fixture
/// walks `root -> next` with `steps = 2` (7 + 11 + 11 = 29 in every lane). Its loop runs as
/// compiled on both lanes.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn an_argument_buffer_pointer_reaches_its_pointee() {
    let fixture = metal2vulkan_devtools::fixtures_dir().join("kernel_device_pointer_chain_phi.ll");
    let bitcode = llvm_assemble(&fixture).unwrap();
    // `struct Node { device uint *next; uint value; }`: the pointer at 0, the value at 8.
    let node = |value: u32| {
        let mut bytes = vec![0; 16];
        bytes[8..12].copy_from_slice(&value.to_le_bytes());
        bytes
    };
    let mut walk = case(
        "kernel_device_pointer_chain_phi",
        vec![0xab; 16],
        Dispatch::Threads {
            threads: [4, 1, 1],
            threads_per_group: [4, 1, 1],
        },
    );
    let out = walk.buffers.remove(&0).unwrap();
    walk.buffers = BTreeMap::from([(0, node(7)), (1, out), (2, 2u32.to_le_bytes().to_vec())]);
    walk.argument_buffer_buffers = BTreeMap::from([((0, 0), node(11))]);
    walk.output = ComputeOutput::Buffer {
        binding: 1,
        offset: 0,
        length: 16,
    };
    let run = run(&walk, &bitcode).unwrap();
    assert_eq!(run.output, [29u32.to_le_bytes(); 4].concat());
}

/// The public loaded-pointer case's buffer (`ob2_loaded_pointer`): an argument buffer holds the
/// address of a 16-byte buffer, which the kernel stores past (words 4 to 63) through the pointer it
/// loads. The candidate allocates that buffer inside a guard of the case's reach; the dynamic
/// root's in-shader check skips every store, so the buffer keeps its words and the census counts no
/// changed guard or padding byte.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn a_loaded_pointers_buffer_written_past_its_end_is_guarded() {
    use crate::case::out_of_bounds::{ContainmentForm, OutOfBoundsResource, OutOfBoundsTarget};
    use crate::executor_contract::out_of_bounds::{Check, Medium};
    let bitcode = llvm_assemble(&engine_dir().join("ob2_loaded_pointer.ll")).unwrap();
    let words = |words: &[u32]| -> Vec<u8> { words.iter().flat_map(|w| w.to_le_bytes()).collect() };
    let idx = [0u32, 1, 2, 3, 4, 5, 17, 63, 4, 6, 9, 12, 30, 44, 50, 63];
    let own = [0xD100_0001u32, 0xD100_0002, 0xD100_0003, 0xD100_0004];
    let mut stores = case(
        "ob2_loaded_pointer",
        words(&idx),
        Dispatch::Threads {
            threads: [8, 1, 1],
            threads_per_group: [8, 1, 1],
        },
    );
    stores.buffers.insert(1, vec![0; 8]);
    stores.buffers.insert(2, vec![0xab; 32]);
    stores.argument_buffer_buffers = BTreeMap::from([((1, 0), words(&own))]);
    let target = OutOfBoundsResource::ArgumentBufferBuffer {
        buffer_binding: 1,
        field_offset: 0,
    };
    stores.out_of_bounds = vec![OutOfBoundsTarget {
        resource: target,
        form: ContainmentForm::LoadedPointer,
        reach_before: 0,
        reach_after: 256,
    }];
    stores.output = ComputeOutput::ArgumentBufferBuffer {
        owner: 1,
        field_offset: 0,
        offset: 0,
        length: 16,
    };
    let run = run(&stores, &bitcode).unwrap();
    assert_eq!(run.output, words(&own));
    let targets = run
        .census
        .out_of_bounds
        .expect("an out-of-bounds case's census records its targets")
        .targets;
    let [ran] = targets.as_slice() else {
        panic!("{targets:?}");
    };
    assert_eq!(ran.resource, target);
    assert_eq!(ran.form, ContainmentForm::LoadedPointer);
    assert_eq!(
        (ran.mechanism.medium, ran.mechanism.check),
        (Medium::DeviceAddress, Check::Shader)
    );
    assert_eq!((ran.escaped, ran.past_end), (Some(0), Some(0)), "{ran:?}");
}

/// The public storage-buffer case (`ob1_storage_buffer`) through the runtime: eight threads load
/// past a 16-byte source and store past a 64-byte output. Each buffer is a guarded target the
/// kernel reaches by its device address, every access at a computed index checked against the
/// buffer in the shader in each variant the translator lowers (`platform::runtime_bounds`): a load
/// outside reads zero and a store outside does nothing, so the output holds the source's four
/// words, four zeros, the four stores inside it and its initial bytes, and no guard byte changes.
#[cfg(target_os = "linux")]
#[test]
fn a_device_buffer_accessed_past_its_end_is_contained_through_the_runtime() {
    use crate::case::out_of_bounds::{ContainmentForm, OutOfBoundsResource, OutOfBoundsTarget};
    use crate::executor_contract::out_of_bounds::{Check, Medium};
    let bitcode = llvm_assemble(&engine_dir().join("ob1_storage_buffer.ll")).unwrap();
    let words = |words: &[u32]| -> Vec<u8> { words.iter().flat_map(|w| w.to_le_bytes()).collect() };
    let idx = [0u32, 1, 2, 3, 4, 5, 17, 63, 8, 9, 10, 11, 16, 17, 40, 79];
    let src = [0xA100_0001u32, 0xA100_0002, 0xA100_0003, 0xA100_0004];
    let mut past = case(
        "ob1_storage_buffer",
        words(&idx),
        Dispatch::Threads {
            threads: [8, 1, 1],
            threads_per_group: [8, 1, 1],
        },
    );
    past.buffers.insert(1, words(&src));
    past.buffers.insert(2, vec![0xab; 64]);
    let target = |binding| OutOfBoundsTarget {
        resource: OutOfBoundsResource::Buffer { binding },
        form: ContainmentForm::StorageBuffer,
        reach_before: 0,
        reach_after: 256,
    };
    past.out_of_bounds = vec![target(1), target(2)];
    past.output = ComputeOutput::Buffer {
        binding: 2,
        offset: 0,
        length: 64,
    };
    let run = run_through_runtime(&past, &bitcode).unwrap();
    let stored = [0xA000_0000u32, 0xA000_0001, 0xA000_0002, 0xA000_0003];
    let want = [&src[..], &[0; 4], &stored, &[0xABAB_ABAB; 4]].concat();
    assert_eq!(run.output, words(&want));
    let targets = run
        .census
        .out_of_bounds
        .expect("an out-of-bounds case's census records its targets")
        .targets;
    assert_eq!(targets.len(), 2, "{targets:?}");
    for ran in &targets {
        assert_eq!(
            (ran.form, ran.mechanism.medium, ran.mechanism.check),
            (
                ContainmentForm::DeviceAddress,
                Medium::DeviceAddress,
                Check::Shader
            ),
            "{ran:?}"
        );
        assert_eq!((ran.escaped, ran.past_end), (Some(0), Some(0)), "{ran:?}");
    }
}

/// [`run`] through the runtime ([`run_on_runtime`] with no linked module).
#[cfg(target_os = "linux")]
pub(crate) fn run_through_runtime(
    case: &ComputeCase,
    bitcode: &[u8],
) -> Result<ComputeRun, ExecutorError> {
    run_on_runtime(case, bitcode, &[])
}

/// [`run_linked`] through the runtime (`platform::execute_runtime`): the kernel and the modules it
/// links as the engine hands them over, its buffers as the case holds them, each texture made
/// with its role's usage, and no float reduction.
#[cfg(target_os = "linux")]
fn run_on_runtime(
    case: &ComputeCase,
    bitcode: &[u8],
    linked: &[LinkedModule],
) -> Result<ComputeRun, ExecutorError> {
    platform::execute_runtime(
        crate::gpu::TEST_DEVICE,
        "engine-test",
        case,
        crate::candidate::engine::RuntimeKernel {
            bitcode,
            linked,
            buffers: &case.buffers,
            texture_usages: &case
                .textures
                .iter()
                .map(|(&slot, texture)| {
                    let usage = crate::check::interface::texture_shader_usage(texture.role, None);
                    (slot, usage)
                })
                .collect(),
            member_usages: &case
                .argument_buffer_textures
                .iter()
                .map(|(&key, texture)| {
                    let usage = crate::check::interface::texture_shader_usage(texture.role, None);
                    (key, usage)
                })
                .collect(),
            reductions: &std::collections::BTreeSet::new(),
        },
    )
    .result
}

/// A store runs through the runtime under the validation layer, and its row records what the
/// runtime answers: the state's precision notes and the runtime's fingerprint, and no SPIR-V,
/// status, subgroup shape, containment or artifact.
#[cfg(target_os = "linux")]
#[test]
fn a_store_runs_through_the_runtime_under_the_validation_layer() {
    let bitcode = llvm_assemble(&engine_dir().join("store_const.ll")).unwrap();
    let run = run_on_runtime(
        &case("store_const", vec![0xab; 8], one_thread()),
        &bitcode,
        &[],
    )
    .unwrap();
    let mut expected = vec![0xab; 8];
    expected[..4].copy_from_slice(&42u32.to_le_bytes());
    assert_eq!(run.output, expected);
    assert_eq!(run.environment["executor"], "runtime-compute");
    assert_eq!(
        run.translator_fingerprint,
        crate::candidate::engine::runtime_fingerprint()
    );
    assert_eq!((run.spv_sha256, run.status), (None, None));
    let [module] = &run.census.modules[..] else {
        panic!("one module: {:?}", run.census.modules);
    };
    assert_eq!((module.subgroup, module.containment), (None, None));
    assert_eq!(run.census.artifact, None);
}

/// A kernel that calls visible functions it does not define runs through the runtime with the
/// functions its case links, as [`a_kernel_runs_with_its_linked_visible_functions`] runs it on the
/// executor's own context.
#[cfg(target_os = "linux")]
#[test]
fn a_kernel_runs_through_the_runtime_with_its_linked_visible_functions() {
    let caller = llvm_assemble(&engine_dir().join("linked_visible_caller.ll")).unwrap();
    let functions = llvm_assemble(&engine_dir().join("linked_visible_functions.ll")).unwrap();
    let names = |names: &[&str]| LinkedLibrary {
        functions: names.iter().map(|name| (*name).to_owned()).collect(),
    };
    let linked = [
        LinkedModule {
            module_sha256: "functions".into(),
            bitcode: functions,
            library: names(&["linked_blend", "linked_scale"]),
        },
        LinkedModule {
            module_sha256: "caller".into(),
            bitcode: caller.clone(),
            library: names(&["linked_offset"]),
        },
    ];
    const THREADS: u32 = 16;
    let bytes = vec![0; 4 * THREADS as usize];
    let mut kernel = case(
        "linked_visible_caller",
        bytes.clone(),
        Dispatch::Threads {
            threads: [THREADS, 1, 1],
            threads_per_group: [THREADS, 1, 1],
        },
    );
    kernel.buffers.insert(1, bytes.clone());
    let words = run_on_runtime(&kernel, &caller, &linked).unwrap().output;
    let expected = (0..THREADS)
        .flat_map(|tid| (3 * tid + (tid & 7)).to_le_bytes())
        .collect::<Vec<_>>();
    assert_eq!(words, expected);
    kernel.output = ComputeOutput::Buffer {
        binding: 1,
        offset: 0,
        length: bytes.len() as u64,
    };
    let floats = run_on_runtime(&kernel, &caller, &linked).unwrap().output;
    let expected = (0..THREADS)
        .flat_map(|tid| (tid as f32 * 0.5 + 1.0).to_le_bytes())
        .collect::<Vec<_>>();
    assert_eq!(floats, expected);
}

/// A sequentially consistent fence is refused through the runtime, typed, as
/// `ToolingRequirement::SeqCstFence` says of the case (`executor_contract::seq_cst_fence`): the
/// owned fixture `a4_device_fence` stores, fences at the device's scope in `seq_cst` order, and
/// stores again. The day the runtime runs it, this fails, and the requirement goes.
#[cfg(target_os = "linux")]
#[test]
fn a_seq_cst_fence_is_refused_through_the_runtime_as_its_requirement_says() {
    let bitcode = llvm_assemble(&engine_dir().join("a4_device_fence.ll")).unwrap();
    let threads = Dispatch::Threads {
        threads: [8, 1, 1],
        threads_per_group: [8, 1, 1],
    };
    let mut fence = case("a4_device_fence", vec![0; 64], threads);
    fence.buffers.insert(1, vec![0; 64]);
    fence.output = ComputeOutput::Buffer {
        binding: 1,
        offset: 0,
        length: 64,
    };
    let error = run_on_runtime(&fence, &bitcode, &[]).unwrap_err();
    assert!(matches!(error, ExecutorError::NotSupported(_)), "{error}");
    assert!(
        error
            .to_string()
            .contains("a sequentially consistent fence"),
        "{error}"
    );
}

/// The other way of [`a_seq_cst_fence_is_refused_through_the_runtime_as_its_requirement_says`]:
/// the owned fixture `rlm_runtime_limits`'s `rlm_fence_acq_rel` holds the same two stores around a
/// device-scope fence of `acq_rel` order, which carries no `ToolingRequirement::SeqCstFence`, and
/// the runtime makes and runs it: its words are the stores'.
#[cfg(target_os = "linux")]
#[test]
fn a_fence_of_another_order_is_made_through_the_runtime() {
    let bitcode = llvm_assemble(&engine_dir().join("rlm_runtime_limits.ll")).unwrap();
    let threads = Dispatch::Threads {
        threads: [8, 1, 1],
        threads_per_group: [8, 1, 1],
    };
    let input = (0..8u32).map(|t| t * 3 + 1).collect::<Vec<_>>();
    let mut fence = case(
        "rlm_fence_acq_rel",
        input.iter().flat_map(|word| word.to_le_bytes()).collect(),
        threads,
    );
    fence.buffers.insert(1, vec![0; 64]);
    fence.output = ComputeOutput::Buffer {
        binding: 1,
        offset: 0,
        length: 64,
    };
    let output = run_on_runtime(&fence, &bitcode, &[]).unwrap().output;
    let expected = input
        .iter()
        .map(|word| word + 1)
        .chain(input.iter().map(|word| word * 2))
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    assert_eq!(output, expected);
}

/// `ToolingRequirement::TextureBuffer`'s limit, and `ToolingRequirement::TextureBufferMember`'s at
/// the texture, both ways: the runtime makes no texture buffer, so `new_texture` refuses one, typed
/// (`Gap::TextureBuffer`), where it makes a 2D texture of the same format and width. The day the
/// runtime makes one, this fails.
#[cfg(target_os = "linux")]
#[test]
fn a_texture_buffer_is_refused_where_a_2d_texture_is_made() {
    use metal2vulkan_runtime as rt;
    platform::on_runtime_device(crate::gpu::TEST_DEVICE, |device| {
        let index = rt::IndexClass::Texture.range().start;
        let _grant = device
            .grant_indices(rt::IndexClass::Texture, index, 1)
            .unwrap();
        let made = |texture_type| {
            let descriptor =
                rt::TextureDescriptor::new(texture_type, rt::PixelFormat::RGBA32Float, 16, 1);
            device.new_texture(&descriptor, Some(index)).map(drop)
        };
        let refused = made(rt::TextureType::TextureBuffer);
        assert!(
            matches!(
                refused,
                Err(rt::Error::Refused(rt::Refusal::TextureBufferNotOverBuffer))
            ),
            "{refused:?}"
        );
        made(rt::TextureType::Type2D).unwrap();
    })
    .unwrap();
}

/// `ToolingRequirement::TextureBufferMember`'s limit at the pipeline, both ways: the make of
/// `rlm_runtime_limits`'s `rlm_member_texture_buffer`, which reads a texture buffer an argument
/// buffer holds, is refused, typed, where the same read of a 2D texture the argument buffer holds,
/// `rlm_member_texture`, is made. The refusal is the translator's: the runtime's identity domains
/// hold no texture buffer among an argument buffer's textures (`Unsupported{IdentityDomains}`),
/// and a device whose answer lacks the features an argument buffer's texel buffers are read
/// through is refused at that query first (`Feature::ArgumentBufferTextures`), as lavapipe's is
/// under the runtime. The day the runtime makes the first, this fails.
#[cfg(target_os = "linux")]
#[test]
fn a_texture_buffer_member_is_refused_at_the_make_where_a_2d_member_is_made() {
    use metal2vulkan::error::Feature;
    let bitcode = llvm_assemble(&engine_dir().join("rlm_runtime_limits.ll")).unwrap();
    platform::on_runtime_device(crate::gpu::TEST_DEVICE, |device| {
        match platform::make_kernel(device, &bitcode, "rlm_member_texture_buffer") {
            Err(metal2vulkan_runtime::Error::Translation {
                error: metal2vulkan::error::Error::Unsupported(refused),
                ..
            }) => assert!(
                matches!(
                    refused.feature,
                    Feature::IdentityDomains | Feature::ArgumentBufferTextures
                ),
                "{refused:?}"
            ),
            Err(error) => panic!("not the translator's typed refusal: {error}"),
            Ok(_) => panic!("the runtime made a texture buffer member's pipeline"),
        }
        platform::make_kernel(device, &bitcode, "rlm_member_texture").unwrap();
    })
    .unwrap();
}
