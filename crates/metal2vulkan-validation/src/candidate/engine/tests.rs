//! Device-free tests of the engine candidate: the status and output rules, the case mapping,
//! and the binder accepting the executor's bind state for a looping kernel that reaches no status
//! buffer. No device is opened.

use super::*;
use crate::case::ScalarType;
use crate::check::admission::NotSupported;
use crate::requirement::ToolingRequirement;
use metal2vulkan::bind::{PushHole, ResourceClass, WriteSource};
use metal2vulkan::device::{Device, DeviceDescription, DevicePolicy};
use metal2vulkan::library::Library;
use metal2vulkan::metal::{DataType, ResourceUsage};
use metal2vulkan::pipeline::{
    ComputePipeline, ComputePipelineDescriptor, FunctionDescriptor, StageSource,
};
use metal2vulkan::Budget;
use metal2vulkan_devtools::{llvm_assemble, AssembleError};

mod argument_buffer_members;

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// A kernel with one loop: `out[0] = 4` after four iterations.
pub(crate) fn counting_kernel() -> String {
    format!(
        "{HEADER}define void @count(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0) {{\n\
entry:\n\
  br label %loop\n\
loop:\n\
  %i = phi i32 [ 0, %entry ], [ %n, %loop ]\n\
  %n = add i32 %i, 1\n\
  %again = icmp ult i32 %n, 4\n\
  br i1 %again, label %loop, label %done\n\
done:\n\
  store i32 %n, ptr addrspace(1) %0, align 4\n\
  ret void\n\
}}\n\n\
!air.kernel = !{{!0}}\n!air.version = !{{!6}}\n!air.language_version = !{{!7}}\n\
!0 = !{{ptr @count, !1, !2}}\n!1 = !{{}}\n!2 = !{{!3}}\n\
!3 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"out\"}}\n\
!6 = !{{i32 2, i32 8, i32 0}}\n!7 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n"
    )
}

/// `llvm-as` of `text`, or `None` (skipped visibly) without the tool.
pub(crate) fn assemble(text: &str) -> Option<Vec<u8>> {
    let scratch = crate::ScratchDir::new("engine-test").unwrap();
    let ll = scratch.path().join("k.ll");
    std::fs::write(&ll, text).unwrap();
    match llvm_assemble(&ll) {
        Ok(bitcode) => Some(bitcode),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the kernel: {why}"),
    }
}

fn device(stem: &str, policy: DevicePolicy) -> Device {
    let path = crate::engine::profiles_dir().join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).unwrap();
    let description = DeviceDescription::from_khronos_json(&text, None).unwrap();
    Device::new(description, policy).unwrap()
}

fn compile(device: &Device, bitcode: &[u8], entry: &str) -> ComputePipeline {
    let function = Library::from_air(bitcode)
        .and_then(|library| library.function(entry))
        .unwrap();
    device
        .compile_compute_pipeline(
            &ComputePipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(
                function,
            ))),
            &Budget::standard(),
        )
        .unwrap()
}

/// The status buffer starts zero (no fuel words), and an indirect overflow or a resident-set miss
/// stops the run's output from being evidence.
#[test]
fn the_status_starts_zero_and_stops_non_evidence() {
    let bytes = ExecutionStatus::initial();
    assert!(bytes.iter().all(|&b| b == 0));
    assert!(status_verdict(&bytes).is_ok());
    let with_status = |bits: u32| {
        let mut bytes = bytes;
        bytes[..4].copy_from_slice(&bits.to_le_bytes());
        bytes
    };
    for bits in [
        metal2vulkan::bind::StatusBits::OVERFLOW,
        metal2vulkan::bind::StatusBits::RESIDENT_MISS,
    ] {
        assert!(status_verdict(&with_status(bits.0)).is_err(), "{bits:?}");
    }
    let mut missed = bytes;
    let at = metal2vulkan::bind::StatusWord::ResidentMisses.byte_offset() as usize;
    missed[at..at + 4].copy_from_slice(&3u32.to_le_bytes());
    assert!(status_verdict(&missed).is_err());
    // A read of an argument the function constants removed is the product's typed refusal of the
    // case, not a failed run.
    let absent = with_status(metal2vulkan::bind::StatusBits::ABSENT_ARGUMENT_READ.0);
    match status_verdict(&absent) {
        Err(crate::check::admission::ExecutorError::NotSupported(NotSupported::Product {
            outcome,
            at_bind: false,
            ..
        })) => assert!(
            outcome.key().contains("FunctionConstantArgument"),
            "{}",
            outcome.key()
        ),
        other => panic!("an absent argument read: {other:?}"),
    }
}

/// [`compute_case`] of a case whose entry declares no stage input: every case here but the
/// stage-input ones, which map against their own interface.
fn map_kernel(case: &AuthoredCase) -> Result<ComputeCase, NotSupported> {
    compute_case(case, &kernel_interface(vec![]))
}

/// The AIR-layer interface of a kernel with `params`.
fn kernel_interface(params: Vec<crate::check::interface::Param>) -> AirInterface {
    AirInterface::synthetic(
        metal2vulkan_air::abi::Stage::Kernel {
            max_total_threads: None,
            work_group_size: None,
        },
        params,
        vec![],
        vec![],
    )
}

fn kernel_case() -> AuthoredCase {
    let mut case = crate::case::vector_function_constant_test_case("11".repeat(32), "main".into());
    case.function_constants.clear();
    case
}

#[test]
fn the_case_maps_onto_the_binder_or_names_the_gap() {
    let mut case = kernel_case();
    case.threadgroup_memory
        .push(crate::case::ThreadgroupMemoryResource {
            binding: 2,
            length: 100,
        });
    let mapped = map_kernel(&case).unwrap();
    assert_eq!(mapped.threadgroup_lengths, BTreeMap::from([(2, 100)]));
    let dispatch = case.dispatch.clone().unwrap();
    assert_eq!(
        mapped.dispatch,
        Dispatch::Threads {
            threads: dispatch.grid,
            threads_per_group: dispatch.threads_per_threadgroup,
        }
    );
    assert_eq!(
        mapped.buffers.keys().copied().collect::<Vec<_>>(),
        case.buffers.iter().map(|b| b.binding).collect::<Vec<_>>()
    );
    // A declared fuel is the Metal oracle's alone: the candidate runs the same case.
    case.fuel = Some(9);
    assert_eq!(map_kernel(&case).unwrap(), mapped);

    let state = bind_state(
        &mapped,
        &BTreeMap::from([(
            0,
            BufferAddress {
                guest: 0x1000,
                host: Some(0x1000),
            },
        )]),
        &BTreeMap::new(),
    )
    .unwrap();
    assert!(state.used_resources.is_empty());
    let table = &state.stages[&ShaderStage::Kernel];
    assert_eq!(table.threadgroup_lengths, mapped.threadgroup_lengths);
    for (index, bytes) in &mapped.buffers {
        let expected_host = (*index == 0).then_some(0x1000);
        match table.buffers[index] {
            BoundBuffer::Buffer {
                length,
                offset,
                device_address,
                ..
            } => {
                assert_eq!((length, offset), (bytes.len() as u64, 0));
                assert_eq!(device_address, expected_host);
            }
            other => panic!("{other:?}"),
        }
    }

    // An explicit imageblock runs, its dimensions the dispatch's imageblock size.
    assert_eq!(mapped.imageblock, None);
    assert_eq!(mapped.tile, None);
    assert_eq!(state.imageblock_size, Default::default());
    let mut imageblock = kernel_case();
    imageblock.imageblock = Some(crate::case::ImageblockResource {
        dimensions: [16, 8],
        implicit_coverage: None,
    });
    let explicit = map_kernel(&imageblock).unwrap();
    assert_eq!(explicit.imageblock, Some([16, 8]));
    assert_eq!(explicit.tile, None);
    let state = bind_state(&explicit, &BTreeMap::new(), &BTreeMap::new()).unwrap();
    assert_eq!(
        state.imageblock_size,
        metal2vulkan::bind::ImageblockSize::new(16, 8)
    );
    assert_eq!(state.shape().imageblock_size, state.imageblock_size);
    let mut vertex = kernel_case();
    vertex.stage = Stage::Vertex;
    assert!(matches!(
        map_kernel(&vertex),
        Err(NotSupported::NotImplemented(what)) if what.contains("not a compute case")
    ));
}

/// A kernel case's stage inputs run as the Metal oracle fetches them: each input's data is its own
/// buffer at the shared slot rule's slot (the lowest one no AIR argument occupies), the kernel
/// compiles under the descriptor `executor_contract::stage_inputs` derives, and the dispatch's
/// stage-in region is the grid. A location the entry does not declare, or a slot the case binds
/// itself, is a typed gap.
#[test]
fn stage_inputs_are_bound_where_the_oracle_binds_them() {
    use crate::check::interface::tests::param;
    use metal2vulkan_air::abi::{AbiInt, ParamRole, Slot};
    let at = |index| Slot {
        index: AbiInt::Const(index),
        count: AbiInt::Const(1),
    };
    let buffer = param(
        1,
        ParamRole::Buffer(metal2vulkan_air::abi::BufferArg {
            location: at(0),
            access: metal2vulkan_air::abi::Access::ReadWrite,
            space: metal2vulkan_air::ir::AddrSpace::Device,
            buffer_size: None,
            layout: None,
            raster_order_group: None,
        }),
    );
    let interface = kernel_interface(vec![
        param(0, ParamRole::StageIn { location: at(4) }),
        buffer,
    ]);
    let mut case = kernel_case();
    case.kernel_stage_inputs
        .push(crate::case::KernelStageInput {
            location: 4,
            format: crate::case::AttributeFormat::Float3,
            stride: 16,
            dynamic_stride: false,
            bytes_b64: base64::engine::general_purpose::STANDARD.encode([7u8; 16]),
        });
    let mapped = compute_case(&case, &interface).unwrap();
    assert_eq!(mapped.buffers.get(&1), Some(&vec![7u8; 16]));
    let d = mapped
        .stage_input
        .clone()
        .expect("a stage-input descriptor");
    assert_eq!(
        d.attributes[&4],
        metal2vulkan::pipeline::StageInputAttributeDescriptor::new(
            metal2vulkan::metal::AttributeFormat::Float3,
            0,
            1
        )
    );
    assert_eq!(d.layouts[&1].stride, Some(16));
    assert_eq!(
        d.layouts[&1].step_function,
        metal2vulkan::metal::StepFunction::ThreadPositionInGridX
    );
    let state = bind_state(&mapped, &BTreeMap::new(), &BTreeMap::new()).unwrap();
    let grid = case.dispatch.as_ref().unwrap().grid;
    assert_eq!(
        state.stage_in_region,
        Some(metal2vulkan::bind::StageInRegion::new([0; 3], grid))
    );
    assert!(matches!(
        state.stages[&ShaderStage::Kernel].buffers[&1],
        BoundBuffer::Buffer {
            length: 16,
            attribute_stride: None,
            ..
        }
    ));
    assert!(mapped.attribute_strides.is_empty());
    // A dynamic stride: the layout's is `MTLBufferLayoutStrideDynamic`, and the data's buffer is
    // bound at the case's stride, as the oracle binds it.
    let mut dynamic = case.clone();
    dynamic.kernel_stage_inputs[0].dynamic_stride = true;
    let mapped = compute_case(&dynamic, &interface).unwrap();
    assert_eq!(
        mapped.stage_input.as_ref().unwrap().layouts[&1].stride,
        None
    );
    assert_eq!(mapped.attribute_strides, BTreeMap::from([(1, 16)]));
    let state = bind_state(&mapped, &BTreeMap::new(), &BTreeMap::new()).unwrap();
    assert!(matches!(
        state.stages[&ShaderStage::Kernel].buffers[&1],
        BoundBuffer::Buffer {
            attribute_stride: Some(16),
            ..
        }
    ));
    // Without stage inputs: no descriptor, and no region, as the oracle sets none.
    let plain = map_kernel(&kernel_case()).unwrap();
    assert_eq!(plain.stage_input, None);
    let state = bind_state(&plain, &BTreeMap::new(), &BTreeMap::new()).unwrap();
    assert_eq!(state.stage_in_region, None);

    let gap = |case: &AuthoredCase, interface: &AirInterface, what: &str| match compute_case(
        case, interface,
    ) {
        Err(NotSupported::NotImplemented(detail)) => assert!(detail.contains(what), "{detail}"),
        other => panic!("{what}: {other:?}"),
    };
    gap(&case, &kernel_interface(vec![]), "no AIR stage input");
    let mut taken = case.clone();
    taken.buffers.push(crate::case::BufferResource {
        binding: 1,
        role: crate::case::ResourceRole::Input,
        bytes_b64: Some("AAAAAA==".into()),
        initial_bytes_b64: None,
        rasterization_rate_map: None,
    });
    gap(&taken, &interface, "bound twice");
}

/// A case's visible-function references are no resource gap: its function links them, each
/// module once with the functions it supplies and its retained bitcode, as the Metal oracle links
/// them. Its visible-function tables map to tables the executor binds; an intersection-function
/// table of opaque triangle entries is a buffer of the product's table words, and a linked
/// intersection function and an argument-buffer intersection-function table are requirements that
/// stop this executor ([`require_case_families`]). The linking takes the tables' entries' functions
/// as the oracle's does.
#[test]
fn references_link_visible_tables_map_and_intersection_tables_are_words_or_typed_gaps() {
    use crate::case::{
        ArgumentBufferIntersectionFunctionTableResource, FunctionTableEntry, FunctionTableResource,
        IntersectionFunctionSignature, IntersectionFunctionTableEntry,
        IntersectionFunctionTableResource, LinkedFunctionResource,
    };
    use crate::library_module::{
        LibraryModuleRow, ResolvedFunctionEntry, ResolvedFunctionReference, ResolvedFunctionTable,
        ResolvedIntersectionFunctionEntry, ResolvedIntersectionFunctionTable,
        ResolvedIntersectionFunctionTableLocation,
    };
    use metal2vulkan::pipeline::FunctionHandle;
    let module = LibraryModuleRow {
        module_sha256: "22".repeat(32),
        air_ll: "define i32 @a(i32 %v) { ret i32 %v }".into(),
        blob_b64: b64(b"linked bitcode"),
        lib_sha256s: vec!["33".repeat(32)],
        label: "test/linked.air".into(),
    };
    let mut case = kernel_case();
    case.visible_function_references = vec![LinkedFunctionResource {
        module_sha256: module.module_sha256.clone(),
        function: "a".into(),
    }];
    assert!(map_kernel(&case).is_ok());
    let reference = |function: &str| ResolvedFunctionReference {
        function: function.into(),
        module: module.clone(),
    };
    let references = ResolvedLinkedFunctions {
        references: vec![reference("a"), reference("b")],
        visible: vec![],
        intersection: vec![],
    };
    assert_eq!(
        linked_modules(&references).unwrap(),
        [LinkedModule {
            module_sha256: module.module_sha256.clone(),
            bitcode: b"linked bitcode".to_vec(),
            library: LinkedLibrary {
                functions: vec!["a".into(), "b".into()],
            },
        }]
    );
    assert!(linked_modules(&ResolvedLinkedFunctions::default())
        .unwrap()
        .is_empty());

    let entry = |function: &str| ResolvedFunctionEntry {
        index: 0,
        function: function.into(),
        module: module.clone(),
    };
    let tables = ResolvedLinkedFunctions {
        visible: vec![ResolvedFunctionTable {
            binding: 1,
            size: 1,
            entries: vec![entry("c")],
        }],
        intersection: vec![ResolvedIntersectionFunctionTable {
            location: ResolvedIntersectionFunctionTableLocation::Direct { binding: 2 },
            size: 1,
            entries: vec![ResolvedIntersectionFunctionEntry::Linked(entry("a"))],
        }],
        ..references
    };
    assert_eq!(
        linked_modules(&tables).unwrap(),
        [LinkedModule {
            module_sha256: module.module_sha256.clone(),
            bitcode: b"linked bitcode".to_vec(),
            library: LinkedLibrary {
                functions: vec!["a".into(), "b".into(), "c".into()],
            },
        }]
    );

    let mut visible = case.clone();
    visible.visible_function_tables = vec![FunctionTableResource {
        binding: 1,
        size: 3,
        entries: vec![FunctionTableEntry {
            index: 2,
            module_sha256: module.module_sha256.clone(),
            function: "c".into(),
        }],
    }];
    assert_eq!(require_case_families(&visible), Ok(()));
    let mapped = map_kernel(&visible).unwrap();
    assert_eq!(
        mapped.function_tables,
        BTreeMap::from([(
            1,
            FunctionTable {
                size: 3,
                entries: BTreeMap::from([(
                    2,
                    tables::TableFunction {
                        module_sha256: module.module_sha256.clone(),
                        function: "c".into(),
                    }
                )]),
            }
        )])
    );
    // A table at a buffer's index is refused (the checker refuses it first).
    let mut shared = visible.clone();
    shared.visible_function_tables[0].binding = 0;
    match map_kernel(&shared) {
        Err(NotSupported::NotImplemented(detail)) => {
            assert!(detail.contains("also a visible function table"), "{detail}")
        }
        other => panic!("{other:?}"),
    }

    let mut opaque = case.clone();
    opaque.intersection_function_tables = vec![IntersectionFunctionTableResource {
        binding: 2,
        size: 3,
        entries: vec![IntersectionFunctionTableEntry::OpaqueTriangle {
            index: 1,
            signature: vec![IntersectionFunctionSignature::TriangleData],
        }],
    }];
    assert_eq!(require_case_families(&opaque), Ok(()));
    let words: Vec<u8> = [1u32, 3, 0, FunctionHandle::RESERVED_WORD, 0]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect();
    assert_eq!(map_kernel(&opaque).unwrap().buffers.get(&2), Some(&words));
    let mut past = opaque.clone();
    past.intersection_function_tables[0].entries =
        vec![IntersectionFunctionTableEntry::OpaqueTriangle {
            index: 3,
            signature: vec![],
        }];
    assert!(matches!(
        map_kernel(&past),
        Err(NotSupported::NotImplemented(detail)) if detail.contains("past its size")
    ));
    let mut shared = opaque.clone();
    shared.intersection_function_tables[0].binding = 0;
    assert!(matches!(
        map_kernel(&shared),
        Err(NotSupported::NotImplemented(detail))
            if detail.contains("also an intersection function table")
    ));

    let mut intersection = case.clone();
    intersection.intersection_function_tables = vec![IntersectionFunctionTableResource {
        binding: 2,
        size: 1,
        entries: vec![IntersectionFunctionTableEntry::Linked {
            index: 0,
            module_sha256: module.module_sha256.clone(),
            function: "a".into(),
        }],
    }];
    let linked = Err(NotSupported::Requirements(
        [ToolingRequirement::LinkedIntersectionFunction].into(),
    ));
    assert_eq!(require_case_families(&intersection), linked);
    assert_eq!(map_kernel(&intersection).map(|_| ()), linked);

    let mut encoded = case;
    encoded.argument_buffer_intersection_function_tables =
        vec![ArgumentBufferIntersectionFunctionTableResource {
            buffer_binding: 0,
            field_offset: 0,
            size: 1,
            entries: vec![],
        }];
    let encoded_table = Err(NotSupported::Requirements(
        [ToolingRequirement::ArgumentBufferIntersectionFunctionTable].into(),
    ));
    assert_eq!(require_case_families(&encoded), encoded_table);
    assert_eq!(map_kernel(&encoded).map(|_| ()), encoded_table);
}

/// A case's acceleration structures map to the structures the executor builds as the Metal
/// oracle does: a primitive structure its authored vertices, an instance structure its instance
/// count, each instance the oracle's (opaque, mask `0xFFFFFFFF`, which Vulkan's 8-bit instance mask
/// cannot hold: the binder refuses it). A render case refuses them. The bind state states each
/// instance structure's count, which `get_instance_count()` reads; a primitive structure is not
/// listed.
#[test]
fn acceleration_structures_map_to_what_the_oracle_builds() {
    use crate::case::{AccelerationStructureKind, AccelerationStructureResource};
    let vertices: Vec<u8> = AccelerationStructureKind::INSTANCED_TRIANGLE
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let mut case = kernel_case();
    case.acceleration_structures = vec![
        AccelerationStructureResource {
            binding: 5,
            kind: AccelerationStructureKind::Primitive,
            primitive_triangles_b64: Some(b64(&vertices)),
            child_references: vec![],
        },
        AccelerationStructureResource {
            binding: 6,
            kind: AccelerationStructureKind::Instance,
            primitive_triangles_b64: None,
            child_references: vec![0, 0],
        },
    ];
    assert_eq!(require_case_families(&case), Ok(()));
    assert_eq!(
        map_kernel(&case).unwrap().acceleration_structures,
        BTreeMap::from([
            (5, CaseStructure::Primitive { vertices }),
            (6, CaseStructure::Instances { instances: 2 }),
        ])
    );
    let state = bind_state(
        &map_kernel(&case).unwrap(),
        &BTreeMap::new(),
        &BTreeMap::new(),
    )
    .unwrap();
    assert_eq!(
        state.stages[&ShaderStage::Kernel].acceleration_structures,
        BTreeMap::from([(6, metal2vulkan::bind::BoundAccelerationStructure::new(2))])
    );
    assert_eq!(
        metal2vulkan::bind::instance_record(&CaseStructure::instance(), 0, 0),
        Err(metal2vulkan::error::Unsupported::not_representable(
            metal2vulkan::error::Feature::RayTracing
        ))
    );
    let mut shared = case.clone();
    shared.acceleration_structures[0].binding = 0;
    assert!(matches!(
        map_kernel(&shared),
        Err(NotSupported::NotImplemented(detail)) if detail.contains("bound twice")
    ));
}

/// A table's buffer is the product's table words (`FunctionTableWords`): bound, its size, and each
/// authored entry the handle the compile gave the function it holds, found by module and name among
/// the function's linked libraries; every other entry is null. It is bound as a buffer of its
/// words' length at its index, in the shape the compile is asked for. A function that is not
/// linked, or that has no handle, fails the run.
#[test]
fn a_tables_words_are_its_size_and_the_handles_of_its_functions() {
    use crate::acceptance::LinkedHandles;
    use metal2vulkan::pipeline::FunctionHandle;
    // The handles a compile of a kernel linking `[a, b]` of one module and `[c]` of another
    // gives (`functions` numbered from 2), as they cross the worker exchange.
    let handle = |word: u32| {
        Some(serde_json::from_value::<FunctionHandle>(serde_json::json!(word)).unwrap())
    };
    let handles: LinkedHandles = vec![vec![handle(2), handle(3)], vec![handle(4)]];
    let module = |sha: &str, names: &[&str]| LinkedModule {
        module_sha256: sha.repeat(32),
        bitcode: Vec::new(),
        library: LinkedLibrary {
            functions: names.iter().map(|name| (*name).to_owned()).collect(),
        },
    };
    let linked = [module("aa", &["a", "b"]), module("bb", &["c"])];
    let entry = |sha: &str, function: &str| tables::TableFunction {
        module_sha256: sha.repeat(32),
        function: function.into(),
    };
    let table = FunctionTable {
        size: 5,
        entries: BTreeMap::from([
            (0, entry("bb", "c")),
            (1, entry("aa", "a")),
            (3, entry("aa", "b")),
            (4, entry("bb", "c")),
        ]),
    };
    let tables_at = BTreeMap::from([(7, table.clone())]);
    let buffers = table_buffers(ShaderStage::Kernel, &tables_at, &linked, &handles).unwrap();
    let words = |bytes: &[u8]| {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect::<Vec<_>>()
    };
    assert_eq!(buffers.len(), 1);
    let bytes = &buffers[&(ShaderStage::Kernel, 7)];
    assert_eq!(words(bytes), [1, 5, 4, 2, 0, 3, 4]);
    assert_eq!(bytes.len() as u64, table.byte_len());

    // The kernel's bind state binds the table as a buffer of that length, in the shape.
    let mut case = map_kernel(&kernel_case()).unwrap();
    case.function_tables = tables_at.clone();
    let state = bind_state(
        &case,
        &BTreeMap::from([(7, BufferAddress::synthetic(7))]),
        &BTreeMap::new(),
    )
    .unwrap();
    match state.stages[&ShaderStage::Kernel].buffers.get(&7) {
        Some(BoundBuffer::Buffer { length, .. }) => assert_eq!(*length, table.byte_len()),
        other => panic!("{other:?}"),
    }
    let shape = compute_shape(&case, false).unwrap();
    assert_eq!(state.shape(), shape);
    assert_eq!(
        compute_tables(&case, &linked, std::slice::from_ref(&handles)).unwrap(),
        buffers
    );
    assert!(compute_tables(&case, &linked, &[]).is_err());

    // An entry whose function is not linked, or which has no handle, fails the run.
    let mut stray = tables_at.clone();
    stray
        .get_mut(&7)
        .unwrap()
        .entries
        .insert(2, entry("cc", "c"));
    match table_buffers(ShaderStage::Kernel, &stray, &linked, &handles) {
        Err(ExecutorError::Failed(why)) => assert!(why.contains("not linked"), "{why}"),
        other => panic!("{other:?}"),
    }
    let mut unhandled: LinkedHandles = handles.clone();
    unhandled[1][0] = None;
    match table_buffers(ShaderStage::Kernel, &tables_at, &linked, &unhandled) {
        Err(ExecutorError::Failed(why)) => assert!(why.contains("no table handle"), "{why}"),
        other => panic!("{other:?}"),
    }
    // Without entries the table is bound, sized, and all null.
    let empty = BTreeMap::from([(
        3,
        FunctionTable {
            size: 2,
            entries: BTreeMap::new(),
        },
    )]);
    let buffers = table_buffers(ShaderStage::Vertex, &empty, &[], &Vec::new()).unwrap();
    assert_eq!(words(&buffers[&(ShaderStage::Vertex, 3)]), [1, 2, 0, 0]);
}

/// The case's own function links the functions of the case's linked modules, and its libraries
/// follow the stage libraries in stage order: the kernel of a compute case, the vertex function of
/// a rasterization-disabled vertex case, the fragment function a passthrough vertex companion draws
/// (the companion's library first), and the vertex function an observer fragment companion observes
/// (the case's library first). The generated companion links nothing, so the linked libraries are
/// the second function's in a fragment case's draw and the first's in an observed vertex case's.
#[test]
fn the_cases_own_function_links_and_its_libraries_follow_the_stages() {
    use super::graphics::tests::two_stage_case;
    use crate::companion::{ObservedValue, ObserverFragment, ObserverOutput, PassthroughVertex};
    let linked = [
        LinkedModule {
            module_sha256: "11".repeat(32),
            bitcode: b"first linked".to_vec(),
            library: LinkedLibrary {
                functions: vec!["a".into()],
            },
        },
        LinkedModule {
            module_sha256: "22".repeat(32),
            bitcode: b"second linked".to_vec(),
            library: LinkedLibrary {
                functions: vec!["b".into(), "c".into()],
            },
        },
    ];
    let libraries = linked
        .iter()
        .map(|module| module.library.clone())
        .collect::<Vec<_>>();
    let constants = FunctionConstantValues::defaults;
    let linking = |entry: &str| RequestFunction {
        entry: entry.into(),
        constants: constants(),
        linked: libraries.clone(),
    };
    let alone = |entry: &str| RequestFunction::new(entry, constants());
    let retained = crate::acceptance::retained;
    let (case, companion) = (retained(b"case"), RowSource::Text("companion".into()));
    let expected_libraries = |stages: &[&RowSource<'static>]| {
        stages
            .iter()
            .map(|&stage| stage.clone())
            .chain([retained(b"first linked"), retained(b"second linked")])
            .collect::<Vec<_>>()
    };

    let kernel = map_kernel(&kernel_case()).unwrap();
    let shape = BindShape::default();
    let (request, given) = compute_request(&case, &linked, &kernel, &shape);
    assert_eq!(
        request,
        PipelineRequest::BoundCompute {
            function: linking(&kernel.entry),
            shape: shape.clone(),
            dispatch: Some(kernel.dispatch),
            stage_input: None,
            tile: None,
            replay: false,
        }
    );
    assert_eq!(given, expected_libraries(&[&case]));
    // A plain request's serialization names no replay, so it reads back as before.
    let json = serde_json::to_value(&request).unwrap();
    assert!(json.to_string().find("\"replay\"").is_none(), "{json}");
    assert_eq!(
        serde_json::from_value::<PipelineRequest>(json).unwrap(),
        request
    );
    // The device-replay variant's request: the same function and libraries, the variant's indirect
    // dispatch, and the replay input, which its serialization carries.
    let variant = ComputeCase {
        dispatch: replay_dispatch([8, 1, 1]),
        ..kernel.clone()
    };
    let (replayed, given) = replay_request(&case, &linked, &variant, &shape);
    assert_eq!(
        replayed,
        PipelineRequest::BoundCompute {
            function: linking(&kernel.entry),
            shape: shape.clone(),
            dispatch: Some(replay_dispatch([8, 1, 1])),
            stage_input: None,
            tile: None,
            replay: true,
        }
    );
    assert_eq!(given, expected_libraries(&[&case]));
    let json = serde_json::to_value(&replayed).unwrap();
    assert!(json.to_string().contains("\"replay\":true"), "{json}");
    assert_eq!(
        serde_json::from_value::<PipelineRequest>(json).unwrap(),
        replayed
    );
    assert_eq!(
        request.linked_ranges(),
        [std::ops::Range { start: 1, end: 3 }]
    );

    let mut vertex_case = two_stage_case("shade", "unused", vec![], BTreeMap::new(), 3);
    vertex_case.fragment = None;
    let (request, given) = render_request(&case, None, &linked, &vertex_case, &shape).unwrap();
    assert_eq!(
        request,
        PipelineRequest::VertexDraw {
            function: linking("shade"),
            template: graphics::render_template(&vertex_case),
            shape: Some(shape.clone()),
            form: crate::candidate::engine::variants::DrawForm::Bound,
        }
    );
    assert_eq!(given, expected_libraries(&[&case]));
    assert_eq!(
        request.linked_ranges(),
        [std::ops::Range { start: 1, end: 3 }]
    );

    let passthrough = Companion::PassthroughVertex(PassthroughVertex {
        varyings: vec![],
        viewport_array_index: false,
        render_target_array_index: false,
    });
    let mut fragment_case =
        two_stage_case(passthrough.entry(), "shade", vec![], BTreeMap::new(), 3);
    fragment_case.companion = Some(passthrough.clone());
    let (request, given) =
        render_request(&case, Some(&companion), &linked, &fragment_case, &shape).unwrap();
    assert_eq!(
        request,
        PipelineRequest::Draw {
            vertex: alone(passthrough.entry()),
            fragment: linking("shade"),
            template: graphics::render_template(&fragment_case),
            shape: Some(shape.clone()),
        }
    );
    assert_eq!(given, expected_libraries(&[&companion, &case]));
    assert_eq!(request.linked_ranges(), [2..2, 2..4]);

    let observer = Companion::ObserverFragment(ObserverFragment {
        input: ObservedValue::Position,
        output: ObserverOutput::Float,
    });
    let mut observed_case = two_stage_case("shade", observer.entry(), vec![], BTreeMap::new(), 3);
    observed_case.companion = Some(observer.clone());
    let (request, given) =
        render_request(&case, Some(&companion), &linked, &observed_case, &shape).unwrap();
    assert_eq!(
        request,
        PipelineRequest::Draw {
            vertex: linking("shade"),
            fragment: alone(observer.entry()),
            template: graphics::render_template(&observed_case),
            shape: Some(shape.clone()),
        }
    );
    assert_eq!(given, expected_libraries(&[&case, &companion]));
    assert_eq!(request.linked_ranges(), [2..4, 4..4]);

    // The companion's source is its opaque AIR text, which the worker assembles.
    assert_eq!(
        companion_source(&observed_case),
        Some(RowSource::Text(
            observer.air_text(crate::companion::Spelling::Opaque).into()
        ))
    );
    assert_eq!(companion_source(&vertex_case), None);

    // A fragment function that is no companion's, and a companion without its source.
    let owned = two_stage_case("shade", "colour", vec![], BTreeMap::new(), 3);
    assert!(matches!(
        render_request(&case, Some(&companion), &linked, &owned, &shape),
        Err(ExecutorError::Failed(_))
    ));
    assert!(matches!(
        render_request(&case, None, &linked, &observed_case, &shape),
        Err(ExecutorError::Failed(_))
    ));

    // The tables of the case's own stage take its function's handles, in stage order; the
    // generated companion links nothing.
    assert_eq!(observed_case.companion_stage(), Some(ShaderStage::Fragment));
    assert_eq!(fragment_case.companion_stage(), Some(ShaderStage::Vertex));
    assert_eq!(vertex_case.companion_stage(), None);
    let handle = |word: u32| {
        Some(
            serde_json::from_value::<metal2vulkan::pipeline::FunctionHandle>(serde_json::json!(
                word
            ))
            .unwrap(),
        )
    };
    let own = vec![vec![handle(2)], vec![handle(3), handle(4)]];
    observed_case.vertex.resources.function_tables = BTreeMap::from([(
        5,
        FunctionTable {
            size: 1,
            entries: BTreeMap::from([(
                0,
                tables::TableFunction {
                    module_sha256: "22".repeat(32),
                    function: "c".into(),
                },
            )]),
        },
    )]);
    let buffers = render_tables(&observed_case, &linked, &[own.clone(), Vec::new()]).unwrap();
    assert_eq!(
        buffers,
        BTreeMap::from([(
            (ShaderStage::Vertex, 5),
            [1u32, 1, 4]
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect::<Vec<_>>()
        )])
    );
    assert!(render_tables(&observed_case, &linked, &[own]).is_err());
}

#[test]
fn function_constant_types_map_to_metal_data_types() {
    assert_eq!(ScalarType::F32.metal_data_type(1), Some(DataType::Float));
    assert_eq!(ScalarType::I16.metal_data_type(3), Some(DataType::Short3));
    assert_eq!(ScalarType::Bool.metal_data_type(4), Some(DataType::Bool4));
    assert_eq!(ScalarType::U64.metal_data_type(2), Some(DataType::ULong2));
    assert_eq!(ScalarType::F64.metal_data_type(1), None);
    assert_eq!(ScalarType::U32.metal_data_type(0), None);
    assert_eq!(ScalarType::U32.metal_data_type(5), None);
}

#[test]
fn outputs_are_exact_ranges_of_the_bound_buffer() {
    let buffers = BTreeMap::from([(1, vec![1, 2, 3, 4, 5])]);
    let none = BTreeMap::new();
    let select = |output, textures: &BTreeMap<u32, crate::literal::LiteralTexture>| {
        select_output(
            output,
            &buffers,
            &BTreeMap::new(),
            textures,
            &BTreeMap::new(),
            &BTreeMap::new(),
        )
    };
    assert_eq!(select(ComputeOutput::None, &none), Ok(vec![]));
    let range = |binding, offset, length| ComputeOutput::Buffer {
        binding,
        offset,
        length,
    };
    assert_eq!(select(range(1, 1, 3), &none), Ok(vec![2, 3, 4]));
    assert!(select(range(1, 3, 3), &none).is_err());
    assert!(select(range(0, 0, 1), &none).is_err());
    assert!(select(range(1, u64::MAX, 2), &none).is_err());
    let region = ComputeOutput::Texture {
        binding: 2,
        origin: [1, 0, 0],
        dimensions: [2, 1, 1],
    };
    assert!(select(region, &none).is_err());
    let texture = crate::literal::LiteralTexture {
        binding: 2,
        role: crate::case::ResourceRole::Output,
        texture_type: crate::case::TextureType::D2,
        format: crate::case::TextureFormat::R8Unorm,
        dimensions: [4, 1, 1],
        sample_count: 1,
        bytes: vec![1, 2, 3, 4],
        sparse: None,
        mip_levels: 1,
    };
    let textures = BTreeMap::from([(2, texture)]);
    assert_eq!(select(region, &textures), Ok(vec![2, 3]));

    // An argument-buffer buffer's range, by its owner and field offset.
    let embedded = BTreeMap::from([((1, 8), vec![9, 8, 7, 6])]);
    let pointee = |owner, field_offset, offset, length| ComputeOutput::ArgumentBufferBuffer {
        owner,
        field_offset,
        offset,
        length,
    };
    let selected = |output| {
        select_output(
            output,
            &buffers,
            &embedded,
            &none,
            &BTreeMap::new(),
            &BTreeMap::new(),
        )
    };
    assert_eq!(selected(pointee(1, 8, 1, 2)), Ok(vec![8, 7]));
    assert!(selected(pointee(1, 0, 0, 1)).is_err());
    assert!(selected(pointee(1, 8, 3, 2)).is_err());

    // An argument-buffer texture's region, by its owner and field offset.
    let texture = crate::literal::LiteralTexture {
        binding: 1,
        role: crate::case::ResourceRole::Output,
        texture_type: crate::case::TextureType::D2,
        format: crate::case::TextureFormat::R8Unorm,
        dimensions: [2, 2, 1],
        sample_count: 1,
        bytes: vec![1, 2, 3, 4],
        sparse: None,
        mip_levels: 1,
    };
    let member = BTreeMap::from([((1, 16), texture)]);
    let region = |field_offset, origin| ComputeOutput::ArgumentBufferTexture {
        owner: 1,
        field_offset,
        origin,
        dimensions: [1, 2, 1],
    };
    let selected = |output| {
        select_output(
            output,
            &buffers,
            &embedded,
            &none,
            &member,
            &BTreeMap::new(),
        )
    };
    assert_eq!(selected(region(16, [1, 0, 0])), Ok(vec![2, 4]));
    assert!(selected(region(8, [0, 0, 0])).is_err());
    assert!(selected(region(16, [2, 0, 0])).is_err());

    // A tile pass's render target's region, rows tightly packed as the Metal oracle reads them.
    let attachment = crate::literal::LiteralTexture {
        binding: 2,
        role: crate::case::ResourceRole::InOut,
        texture_type: crate::case::TextureType::D2,
        format: crate::case::TextureFormat::R16Uint,
        dimensions: [3, 2, 1],
        sample_count: 1,
        bytes: (0u8..12).collect(),
        sparse: None,
        mip_levels: 1,
    };
    let attachments = BTreeMap::from([(2, attachment)]);
    let target = |index, origin, dimensions| ComputeOutput::RenderTarget {
        index,
        origin,
        dimensions,
    };
    let selected = |output| {
        select_output(
            output,
            &buffers,
            &embedded,
            &none,
            &BTreeMap::new(),
            &attachments,
        )
    };
    assert_eq!(
        selected(target(2, [1, 0], [2, 2])),
        Ok(vec![2, 3, 4, 5, 8, 9, 10, 11])
    );
    assert!(selected(target(1, [0, 0], [1, 1])).is_err());
    assert!(selected(target(2, [2, 0], [2, 1])).is_err());
}

/// A tile kernel over render targets runs in the Metal oracle's tile pass: its render area the
/// dispatch grid's width and height, its tile size the imageblock's dimensions, one threadgroup of
/// the case's threads per tile (a partial tile included), each render target a 2D attachment of its
/// format and contents, which the executor binds over the whole tiles (past its texels the
/// nearest), bound with the whole tiles as the render area, as the runtime's dispatch binds it, and
/// compiled as a tile pipeline under the attachments' formats; a render-target output selects an
/// attachment.
#[test]
fn a_tile_kernel_runs_in_its_tile_pass() {
    use crate::case::{
        Dispatch as CaseDispatch, ImageblockResource, ImplicitImageblockCoverage,
        RenderTargetResource, TextureFormat,
    };
    let mut case = kernel_case();
    case.imageblock = Some(ImageblockResource {
        dimensions: [16, 16],
        implicit_coverage: Some(ImplicitImageblockCoverage::FullSingleSample),
    });
    case.dispatch = Some(CaseDispatch {
        grid: [40, 24, 1],
        threads_per_threadgroup: [8, 8, 1],
        stage_in_region: None,
    });
    let pixels: Vec<u8> = (0..40 * 24 * 4).map(|i| i as u8).collect();
    case.render_targets = vec![RenderTargetResource {
        index: 1,
        format: TextureFormat::R32Uint,
        dimensions: [40, 24],
        initial_bytes_b64: b64(&pixels),
    }];
    case.output = OutputSelection::RenderTarget {
        index: 1,
        origin: [8, 2],
        dimensions: [4, 3],
    };
    let mapped = map_kernel(&case).unwrap();
    assert_eq!(
        mapped.dispatch,
        Dispatch::Threadgroups {
            groups: [3, 2, 1],
            threads_per_group: [8, 8, 1],
        }
    );
    assert_eq!(mapped.imageblock, Some([16, 16]));
    let pass = mapped.tile.as_ref().unwrap();
    assert_eq!(pass.area, [40, 24]);
    assert_eq!(pass.tiles, [48, 32]);
    assert_eq!(
        pass.formats(),
        BTreeMap::from([(1, metal2vulkan::metal::PixelFormat::R32Uint)])
    );
    let attachment = &pass.attachments[&1];
    assert_eq!(
        (
            attachment.texture_type,
            attachment.dimensions,
            attachment.sample_count
        ),
        (crate::case::TextureType::D2, [40, 24, 1], 1)
    );
    assert_eq!(attachment.bytes, pixels);
    // The executor's attachment over the whole tiles: the authored texels, and past them the
    // nearest, clamped to their edge.
    let tiled = &pass.tiled[&1];
    assert_eq!(
        (tiled.format, tiled.dimensions, tiled.sample_count),
        (attachment.format, [48, 32, 1], 1)
    );
    let padded: Vec<u8> = (0..32usize)
        .flat_map(|y| (0..48usize).map(move |x| (x.min(39), y.min(23))))
        .flat_map(|(x, y)| {
            let at = (y * 40 + x) * 4;
            pixels[at..at + 4].to_vec()
        })
        .collect();
    assert_eq!(tiled.bytes, padded);
    assert_eq!(
        mapped.output,
        ComputeOutput::RenderTarget {
            index: 1,
            origin: [8, 2],
            dimensions: [4, 3],
        }
    );
    let state = bind_state(&mapped, &BTreeMap::new(), &BTreeMap::new()).unwrap();
    assert_eq!(state.render_area, [48, 32]);
    assert_eq!(
        state.imageblock_size,
        metal2vulkan::bind::ImageblockSize::new(16, 16)
    );
    let shape = state.shape();
    assert_eq!(shape.render_area, [48, 32]);
    let library = RowSource::Text(Cow::Borrowed(""));
    let (request, _) = compute_request(&library, &[], &mapped, &shape);
    match request {
        PipelineRequest::BoundCompute {
            tile, stage_input, ..
        } => {
            assert_eq!(tile, Some(pass.formats()));
            assert_eq!(stage_input, None);
        }
        other => panic!("{other:?}"),
    }
    // The selected region, rows tightly packed (the oracle's `LiteralRenderTarget::select`).
    let finals = BTreeMap::from([(1, attachment.clone())]);
    let expected = crate::literal::LiteralRenderTarget {
        index: 1,
        format: TextureFormat::R32Uint,
        dimensions: [40, 24],
        samples: 1,
        bytes: pixels.clone(),
    }
    .select([8, 2], [4, 3])
    .unwrap();
    assert_eq!(
        select_output(
            mapped.output,
            &BTreeMap::new(),
            &BTreeMap::new(),
            &BTreeMap::new(),
            &BTreeMap::new(),
            &finals
        ),
        Ok(expected)
    );

    // Render targets of a kernel with no implicit imageblock, and a tile grid of more than one
    // plane, are not implemented.
    let mut plain = case.clone();
    plain.imageblock = None;
    assert!(matches!(
        map_kernel(&plain),
        Err(NotSupported::NotImplemented(what)) if what.contains("no tile function")
    ));
    let mut planes = case.clone();
    planes.dispatch = Some(CaseDispatch {
        grid: [40, 24, 2],
        threads_per_threadgroup: [8, 8, 1],
        stage_in_region: None,
    });
    assert!(matches!(
        map_kernel(&planes),
        Err(NotSupported::NotImplemented(what)) if what.contains("more than one plane")
    ));
    // A render-target output of a kernel in no tile pass is not implemented.
    let mut output = kernel_case();
    output.output = case.output.clone();
    assert!(matches!(
        map_kernel(&output),
        Err(NotSupported::NotImplemented(what)) if what.contains("no tile pass")
    ));
}

/// `bytes` as the base64 an authored case spells.
fn b64(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// Device buffer array elements bind at consecutive Metal slots (the oracle's and the buffer
/// table's `first + k`); an argument-buffer buffer is a used resource whose guest address the
/// owner holds at its field, the owner covering that field; each maps its output, and a
/// collision, an owner that is not bound, or an element slot that overflows is a gap.
#[test]
fn arrays_and_argument_buffers_map_onto_the_binder() {
    use crate::case::{
        ArgumentBufferBufferResource, DeviceBufferArrayElementResource, DeviceBufferArrayResource,
        ResourceRole,
    };
    let element = |index, bytes: &[u8]| DeviceBufferArrayElementResource {
        index,
        role: ResourceRole::Input,
        bytes_b64: Some(b64(bytes)),
        initial_bytes_b64: None,
    };
    let mut case = kernel_case();
    case.device_buffer_arrays.push(DeviceBufferArrayResource {
        binding: 4,
        length: 2,
        elements: vec![element(1, &[2; 8]), element(0, &[1; 4])],
    });
    case.argument_buffer_buffers
        .push(ArgumentBufferBufferResource {
            buffer_binding: 0,
            field_offset: 8,
            role: ResourceRole::Output,
            bytes_b64: None,
            initial_bytes_b64: Some(b64(&[5; 12])),
            rasterization_rate_map: None,
        });
    case.output = OutputSelection::DeviceBufferArrayElement {
        binding: 4,
        element: 1,
        offset: 2,
        length: 4,
    };
    let mapped = map_kernel(&case).unwrap();
    assert_eq!(mapped.buffers[&4], vec![1; 4]);
    assert_eq!(mapped.buffers[&5], vec![2; 8]);
    // The owner (4 authored bytes) now covers its pointer field at 8.
    assert_eq!(mapped.buffers[&0].len(), 16);
    assert_eq!(&mapped.buffers[&0][..4], &[0xab; 4]);
    assert_eq!(
        mapped.argument_buffer_buffers,
        BTreeMap::from([((0, 8), vec![5; 12])])
    );
    assert_eq!(
        mapped.output,
        ComputeOutput::Buffer {
            binding: 5,
            offset: 2,
            length: 4
        }
    );

    let host = |guest| BufferAddress {
        guest,
        host: Some(guest),
    };
    let addresses = mapped
        .buffers
        .keys()
        .map(|&index| (index, host(0x1000 * u64::from(index + 1))))
        .collect();
    let embedded = BTreeMap::from([((0, 8), host(0xdead_0000))]);
    let state = bind_state(&mapped, &addresses, &embedded).unwrap();
    assert_eq!(
        state.stages[&ShaderStage::Kernel]
            .buffers
            .keys()
            .copied()
            .collect::<Vec<_>>(),
        [0, 4, 5]
    );
    let [used] = state.used_resources.as_slice() else {
        panic!("{:?}", state.used_resources);
    };
    assert_eq!(used.class, ResourceClass::Buffer);
    assert_eq!(used.token, 0);
    assert_eq!(
        (used.gpu_address, used.device_address, used.length),
        (Some(0xdead_0000), Some(0xdead_0000), 12)
    );
    assert_eq!(used.usage, ResourceUsage::READ | ResourceUsage::WRITE);
    // The executor's shape is the one it compiles for, with and without device addresses.
    for addressed in [false, true] {
        let shape = compute_shape(&mapped, addressed).unwrap();
        assert_eq!(shape.used.len(), 1);
        let addresses = mapped
            .buffers
            .keys()
            .map(|&index| {
                let mut address = BufferAddress::synthetic(index);
                address.host = addressed.then_some(0x9000);
                (index, address)
            })
            .collect();
        let mut used = BufferAddress::synthetic_used(0);
        used.host = addressed.then_some(0xa000);
        let embedded = BTreeMap::from([((0, 8), used)]);
        assert_eq!(
            bind_state(&mapped, &addresses, &embedded).unwrap().shape(),
            shape
        );
    }
    assert_ne!(
        BufferAddress::synthetic_used(0).guest,
        BufferAddress::synthetic(0).guest
    );

    // The owner is created holding the pointer, little endian, at its field.
    let fields = argument_buffers::field_words(
        &embedded,
        &BTreeMap::new(),
        &BTreeMap::new(),
        &BTreeMap::new(),
        std::iter::empty(),
    )
    .unwrap();
    let owner_bytes = |index| argument_buffers::owner_bytes(&mapped.buffers, index, &fields);
    let owner = owner_bytes(0).unwrap();
    assert_eq!(&owner[..4], &[0xab; 4]);
    assert_eq!(&owner[4..8], &[0; 4]);
    assert_eq!(&owner[8..], &0xdead_0000u64.to_le_bytes());
    assert_eq!(owner_bytes(5).unwrap(), vec![2; 8]);
    assert!(owner_bytes(1).is_err());

    let mut pointee = case.clone();
    pointee.output = OutputSelection::ArgumentBufferBuffer {
        buffer_binding: 0,
        field_offset: 8,
        offset: 0,
        length: 12,
    };
    assert_eq!(
        map_kernel(&pointee).unwrap().output,
        ComputeOutput::ArgumentBufferBuffer {
            owner: 0,
            field_offset: 8,
            offset: 0,
            length: 12
        }
    );

    let gap = |case: &AuthoredCase, expected: &str| match map_kernel(case) {
        Err(NotSupported::NotImplemented(what)) => assert!(what.contains(expected), "{what}"),
        other => panic!("{other:?}"),
    };
    let mut collision = case.clone();
    collision.device_buffer_arrays[0].binding = 0;
    gap(&collision, "buffer 0 bound twice");
    let mut orphan = case.clone();
    orphan.argument_buffer_buffers[0].buffer_binding = 3;
    gap(&orphan, "without its owner buffer");
    let mut overflow = case.clone();
    overflow.device_buffer_arrays[0].binding = u32::MAX;
    gap(&overflow, "overflows");
}

/// A texture array at `binding` of `r32_uint` 1×1 elements holding `values`, authored `role`.
fn texture_array(
    binding: u32,
    role: crate::case::ResourceRole,
    values: &[u32],
) -> crate::case::TextureArrayResource {
    use crate::case::{TextureArrayElement, TextureArrayResource, TextureFormat, TextureType};
    let element = |value: u32| TextureArrayElement {
        dimensions: [1, 1, 1],
        bytes_b64: (role == crate::case::ResourceRole::Input).then(|| b64(&value.to_le_bytes())),
        initial_bytes_b64: (role != crate::case::ResourceRole::Input)
            .then(|| b64(&value.to_le_bytes())),
    };
    TextureArrayResource {
        binding,
        role,
        texture_type: TextureType::D2,
        format: TextureFormat::R32Uint,
        sample_count: 1,
        overrides_texture_at_base: false,
        elements: values.iter().copied().map(element).collect(),
    }
}

/// Element `k` of a texture array at `[[texture(n)]]` is the texture bound at slot `n + k`, as
/// the Metal oracle binds it and the product reads it: the case maps each element there beside its
/// textures, its bind state binds each as a texture of the array's type and format, and an element
/// output is that texture's region. Element 0 of an array that overrides the texture at its base
/// slot replaces it (Metal binds arrays last); any other collision and a slot that overflows are
/// gaps.
#[test]
fn texture_arrays_bind_each_element_at_its_slot() {
    use crate::case::{ResourceRole, TextureFormat, TextureResource, TextureType};
    let mut case = kernel_case();
    case.texture_arrays
        .push(texture_array(2, ResourceRole::InOut, &[7, 11, 13]));
    case.textures.push(TextureResource {
        binding: 0,
        role: ResourceRole::Input,
        texture_type: TextureType::D2,
        format: TextureFormat::R32Uint,
        dimensions: [1, 1, 1],
        sample_count: 1,
        bytes_b64: Some(b64(&5u32.to_le_bytes())),
        initial_bytes_b64: None,
        sparse: None,
        mip_levels: 1,
    });
    case.output = OutputSelection::TextureArrayElement {
        binding: 2,
        element: 1,
        origin: [0, 0, 0],
        dimensions: [1, 1, 1],
    };
    assert_eq!(require_case_families(&case), Ok(()));
    let mapped = map_kernel(&case).unwrap();
    assert_eq!(
        mapped
            .textures
            .iter()
            .map(|(&slot, texture)| (slot, texture.binding, texture.bytes.clone()))
            .collect::<Vec<_>>(),
        [
            (0, 0, 5u32.to_le_bytes().to_vec()),
            (2, 2, 7u32.to_le_bytes().to_vec()),
            (3, 3, 11u32.to_le_bytes().to_vec()),
            (4, 4, 13u32.to_le_bytes().to_vec()),
        ]
    );
    assert!(mapped.textures[&3].format == TextureFormat::R32Uint);
    assert_eq!(
        mapped.output,
        ComputeOutput::Texture {
            binding: 3,
            origin: [0, 0, 0],
            dimensions: [1, 1, 1],
        }
    );
    let state = bind_state(&mapped, &BTreeMap::new(), &BTreeMap::new()).unwrap();
    let table = &state.stages[&ShaderStage::Kernel];
    assert_eq!(
        table.textures.keys().copied().collect::<Vec<_>>(),
        [0, 2, 3, 4]
    );
    for slot in [2, 3, 4] {
        assert_eq!(
            table.textures[&slot],
            resources::bound_texture(&mapped.textures[&slot]).unwrap()
        );
    }
    // The final element texture is the observation.
    let mut finals = mapped.textures.clone();
    finals.get_mut(&3).unwrap().bytes = 99u32.to_le_bytes().to_vec();
    assert_eq!(
        select_output(
            mapped.output,
            &BTreeMap::new(),
            &BTreeMap::new(),
            &finals,
            &BTreeMap::new(),
            &BTreeMap::new()
        ),
        Ok(99u32.to_le_bytes().to_vec())
    );

    // The function-constant alternative: element 0 replaces the layered texture at the base.
    let mut alternative = kernel_case();
    alternative.textures.push(TextureResource {
        binding: 0,
        role: ResourceRole::Input,
        texture_type: TextureType::D2Array,
        format: TextureFormat::R32Uint,
        dimensions: [1, 1, 1],
        sample_count: 1,
        bytes_b64: Some(b64(&5u32.to_le_bytes())),
        initial_bytes_b64: None,
        sparse: None,
        mip_levels: 1,
    });
    let mut overriding = texture_array(0, ResourceRole::Input, &[7, 11]);
    overriding.overrides_texture_at_base = true;
    alternative.texture_arrays.push(overriding);
    let mapped = map_kernel(&alternative).unwrap();
    assert_eq!(mapped.textures.len(), 2);
    assert_eq!(mapped.textures[&0].texture_type, TextureType::D2);
    assert_eq!(mapped.textures[&0].bytes, 7u32.to_le_bytes());
    assert_eq!(mapped.textures[&1].bytes, 11u32.to_le_bytes());

    let gap = |case: &AuthoredCase, expected: &str| match map_kernel(case) {
        Err(NotSupported::NotImplemented(what)) => assert!(what.contains(expected), "{what}"),
        other => panic!("{other:?}"),
    };
    // Without the declaration, an element on a texture's slot is bound twice.
    alternative.texture_arrays[0].overrides_texture_at_base = false;
    gap(&alternative, "texture slot 0 is bound twice");
    // An element past the first never replaces a texture.
    let mut shifted = case.clone();
    shifted.textures[0].binding = 3;
    shifted.texture_arrays[0].overrides_texture_at_base = true;
    gap(&shifted, "texture slot 3 is bound twice");
    let mut overflow = case.clone();
    overflow.texture_arrays[0].binding = u32::MAX;
    gap(&overflow, "overflows");
}

/// A kernel over `out` (`[[buffer(0)]]`) and a device buffer array of two `device uint *`
/// (`[[buffer(1..3)]]`, a buffer table): `out[1] = tab[0][0] + tab[1][1]`.
pub(crate) fn buffer_table_kernel() -> String {
    format!(
        "{HEADER}define void @table(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0, ptr noundef %1) {{\n\
entry:\n\
  %e0 = load ptr addrspace(1), ptr %1, align 8\n\
  %t1 = getelementptr inbounds ptr addrspace(1), ptr %1, i64 1\n\
  %e1 = load ptr addrspace(1), ptr %t1, align 8\n\
  %a = load i32, ptr addrspace(1) %e0, align 4\n\
  %q = getelementptr inbounds i32, ptr addrspace(1) %e1, i64 1\n\
  %b = load i32, ptr addrspace(1) %q, align 4\n\
  %s = add i32 %a, %b\n\
  %o = getelementptr inbounds i32, ptr addrspace(1) %0, i64 1\n\
  store i32 %s, ptr addrspace(1) %o, align 4\n\
  ret void\n\
}}\n\n\
!air.kernel = !{{!0}}\n!air.version = !{{!6}}\n!air.language_version = !{{!7}}\n\
!0 = !{{ptr @table, !1, !2}}\n!1 = !{{}}\n!2 = !{{!3, !4}}\n\
!3 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"out\"}}\n\
!4 = !{{i32 1, !\"air.buffer\", !\"air.location_index\", i32 1, i32 2, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"tab\"}}\n\
!6 = !{{i32 2, i32 8, i32 0}}\n!7 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n"
    )
}

/// A case with a device buffer array maps its elements onto consecutive slots, and the binder
/// plans and instantiates the executor's state for a buffer-table kernel compiled for that
/// shape, with and without device addresses, with no diagnostic.
#[test]
fn a_device_buffer_array_plans_as_the_functions_buffer_table() {
    use crate::case::{DeviceBufferArrayElementResource, DeviceBufferArrayResource, ResourceRole};
    let Some(bitcode) = assemble(&buffer_table_kernel()) else {
        return;
    };
    let mut case = kernel_case();
    case.entry = "table".into();
    case.buffers[0].initial_bytes_b64 = Some(b64(&[0; 8]));
    case.device_buffer_arrays.push(DeviceBufferArrayResource {
        binding: 1,
        length: 2,
        elements: (0..2)
            .map(|index| DeviceBufferArrayElementResource {
                index,
                role: ResourceRole::Input,
                bytes_b64: Some(b64(&[index as u8 + 1; 8])),
                initial_bytes_b64: None,
            })
            .collect(),
    });
    let compute = map_kernel(&case).unwrap();
    assert_eq!(
        compute.buffers.keys().copied().collect::<Vec<_>>(),
        [0, 1, 2]
    );
    let library = Library::from_air(&bitcode).unwrap();
    let function = FunctionDescriptor::new(library.function("table").unwrap());
    for (stem, addressed) in [
        ("lavapipe", true),
        ("moltenvk-m3", true),
        ("vk12-floor", false),
    ] {
        let device = device(stem, DevicePolicy::default());
        let shape = compute_shape(&compute, addressed).unwrap();
        let (pipeline, inputs) = variants::compile_compute_for_shape(
            &device,
            &function,
            &shape,
            Some(&compute.dispatch),
            None,
            &Budget::standard(),
        )
        .unwrap_or_else(|error| panic!("{stem}: compile: {error:?}"));
        let plan = pipeline.plan(&device, &shape).unwrap();
        assert!(variants::compiled_for(&inputs, &plan.variant), "{stem}");
        assert!(
            plan.diagnostics.is_empty(),
            "{stem}: {:?}",
            plan.diagnostics
        );
        let addresses = compute
            .buffers
            .keys()
            .map(|&index| {
                let mut address = BufferAddress::synthetic(index);
                address.host = addressed.then_some(address.guest);
                (index, address)
            })
            .collect();
        let state = bind_state(&compute, &addresses, &BTreeMap::new()).unwrap();
        assert_eq!(state.shape(), shape, "{stem}");
        let commands = plan
            .instantiate_dispatch(&state, &compute.dispatch)
            .unwrap_or_else(|error| panic!("{stem}: instantiate: {error}"));
        assert!(!commands.regions.is_empty(), "{stem}");
    }
}

/// The binder accepts the executor's bind state for a compiled looping kernel, with and without
/// device addresses, and plans every buffer write the executor handles; the loop reaches no
/// status buffer.
#[test]
fn the_binder_plans_the_executors_state() {
    let Some(bitcode) = assemble(&counting_kernel()) else {
        return;
    };
    let case = ComputeCase {
        out_of_bounds: Vec::new(),
        creation_bound: crate::budgets::PIPELINE_CREATION_CEILING,
        entry: "count".into(),
        constants: FunctionConstantValues::defaults(),
        buffers: BTreeMap::from([(0, vec![0; 4])]),
        rate_maps: BTreeMap::new(),
        argument_buffer_buffers: BTreeMap::new(),
        argument_buffer_rate_maps: BTreeMap::new(),
        argument_buffer_textures: BTreeMap::new(),
        argument_buffer_samplers: BTreeMap::new(),
        objects: BTreeMap::new(),
        command_buffers: BTreeMap::new(),
        textures: BTreeMap::new(),
        samplers: BTreeMap::new(),
        function_tables: BTreeMap::new(),
        acceleration_structures: BTreeMap::new(),
        threadgroup_lengths: BTreeMap::new(),
        imageblock: None,
        tile: None,
        stage_input: None,
        stage_in_region: None,
        attribute_strides: Default::default(),
        dispatch: Dispatch::Threads {
            threads: [5, 1, 1],
            threads_per_group: [2, 1, 1],
        },
        output: ComputeOutput::Buffer {
            binding: 0,
            offset: 0,
            length: 4,
        },
    };
    for stem in ["lavapipe", "moltenvk-m3", "vk12-floor"] {
        let device = device(stem, DevicePolicy::default());
        let pipeline = compile(&device, &bitcode, "count");
        for address in [
            BufferAddress::synthetic(0),
            BufferAddress {
                guest: 0x10_0000,
                host: Some(0x10_0000),
            },
        ] {
            let state =
                bind_state(&case, &BTreeMap::from([(0, address)]), &BTreeMap::new()).unwrap();
            let plan = pipeline.plan(&device, &state.shape()).unwrap();
            let commands = plan.instantiate_dispatch(&state, &case.dispatch).unwrap();
            assert!(!commands.regions.is_empty(), "{stem}");
            assert!(commands.indirect_clamp.is_none());
            assert!(plan.descriptors.iter().all(|write| matches!(
                write.source,
                WriteSource::Buffer { .. }
                    | WriteSource::Neutral(_)
                    | WriteSource::ArgumentTable { .. }
                    | WriteSource::ConstantBlob { .. }
            )));
            assert!(
                !commands
                    .push_holes
                    .iter()
                    .any(|hole| matches!(hole, PushHole::ExecutionStatusAddress { .. })),
                "{stem}: the loop reaches no status buffer"
            );
        }
    }
}

#[test]
fn identity_is_per_profile_and_per_engine_slot() {
    let a = engine_fingerprint(device("lavapipe", DevicePolicy::default()).profile());
    let b = engine_fingerprint(device("moltenvk-m3", DevicePolicy::default()).profile());
    assert_ne!(a, b);
    assert_eq!(a.len(), 64);
    assert_eq!(
        engine_environment_id("ubuntu-26.04-lavapipe"),
        "ubuntu-26.04-lavapipe@engine"
    );
}

/// A row run under the profiles layer has its own slot per profile, and a row of the robustness-off
/// leg its own beside each, every one an engine slot; an operator ID naming any engine slot
/// is not one the executors accept.
#[test]
fn a_restricted_row_has_its_own_engine_slot() {
    use crate::coverage::Leg;
    let native = engine_slot("ubuntu-26.04-lavapipe", None, Leg::Stated);
    let floor = engine_slot("ubuntu-26.04-lavapipe", Some("vk12-floor"), Leg::Stated);
    let baseline = engine_slot(
        "ubuntu-26.04-lavapipe",
        Some("lunarg-desktop-baseline-2023"),
        Leg::Stated,
    );
    let off = engine_slot("ubuntu-26.04-lavapipe", None, Leg::RobustnessOff);
    let floor_off = engine_slot(
        "ubuntu-26.04-lavapipe",
        Some("vk12-floor"),
        Leg::RobustnessOff,
    );
    assert_eq!(native, engine_environment_id("ubuntu-26.04-lavapipe"));
    assert_eq!(floor, "ubuntu-26.04-lavapipe@engine@vk12-floor");
    assert_eq!(off, "ubuntu-26.04-lavapipe@engine+robustness-off");
    assert_eq!(
        floor_off,
        "ubuntu-26.04-lavapipe@engine@vk12-floor+robustness-off"
    );
    let slots = [&native, &floor, &baseline, &off, &floor_off];
    for (index, slot) in slots.iter().enumerate() {
        assert!(is_engine_slot(slot), "{slot}");
        assert!(!slots[..index].contains(slot), "{slot} twice");
    }
    for operator in [
        "ubuntu-26.04-lavapipe",
        "ubuntu-26.04-lavapipe@engine-x",
        "x@engine@",
        "x@engine@a@b",
        "x@engine+robustness-on",
        "x@engine@+robustness-off",
        "x@engine@a+b",
    ] {
        assert!(!is_engine_slot(operator), "{operator}");
    }
}

/// The checker accepts a committed public fragment case over an explicit imageblock with no
/// requirement that stops this executor, which prepares it device-free: the case's plane lands at
/// its master member's offset in the pass's one cell (`fragment_custom_imageblock`'s master holds
/// four halves, `user(depth)` the second), every other byte zero as the Metal oracle's initializer
/// leaves an unauthored member, and the output is that member's plane.
#[test]
fn a_checked_fragment_imageblock_case_maps_its_plane_onto_the_cells() {
    use super::graphics::RenderOutput;
    use super::imageblock::Member;
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus");
    let case = CorpusStore::new(&root)
        .read_all_cases()
        .unwrap()
        .into_iter()
        .find(|case| case.name == "fragment-custom-imageblock-depth")
        .expect("the committed public case");
    let device = DeviceType::Cpu;
    let admitted = admit(&root, case, ExecutionRequest::EngineCandidate { device })
        .unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(
        crate::executor_contract::require_executor(
            &admitted.checked().requirements(),
            Executor::EngineCandidate
        ),
        Ok(())
    );
    let prepared = prepare_case(device, admitted).unwrap_or_else(|error| panic!("{error}"));
    let Mapped::Render(render) = &prepared.mapped else {
        panic!("a fragment case maps onto a draw");
    };
    let cells = render.imageblock.as_ref().expect("the case's imageblock");
    assert_eq!(cells.dimensions, [1, 1]);
    assert_eq!(cells.cell_bytes, 8);
    // The authored depth, half 1.099609375 (0x3c66), little-endian at byte 2.
    assert_eq!(cells.bytes, [0, 0, 0x66, 0x3c, 0, 0, 0, 0]);
    assert_eq!(
        render.output,
        RenderOutput::Imageblock {
            member: Member { offset: 2, size: 2 },
            origin: [0, 0],
            dimensions: [1, 1],
        }
    );
}

/// The checker accepts the committed public cases over pipeline-state members (a kernel's and a
/// rasterization-disabled vertex function's), resolving their two templates, with no requirement
/// that stops this executor, which prepares each device-free: every authored pipeline state is its
/// authored token in its owner at its field, the word a command's pipeline field records, and the
/// owner is created holding it there beside the authored data the function reads.
#[test]
fn checked_pipeline_state_cases_prepare_with_their_tokens_in_their_owners() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus");
    let cases = CorpusStore::new(&root).read_all_cases().unwrap();
    for name in [
        "pipeline-states-beside-the-data-of-two-stages",
        "pipeline-states-beside-the-data-a-vertex-reads",
    ] {
        let case = cases
            .iter()
            .find(|case| case.name == name)
            .unwrap_or_else(|| panic!("the committed public case {name}"))
            .clone();
        let expected = case
            .pipeline_states
            .iter()
            .map(|pipeline| {
                (
                    (pipeline.buffer_binding, pipeline.field_offset),
                    pipeline.token,
                )
            })
            .collect::<BTreeMap<_, _>>();
        assert_eq!(
            expected.keys().copied().collect::<Vec<_>>(),
            [(0, 0), (0, 8)],
            "{name}"
        );
        let device = DeviceType::Cpu;
        let admitted = admit(&root, case, ExecutionRequest::EngineCandidate { device })
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(admitted.checked().pipeline_templates.len(), 2, "{name}");
        assert_eq!(
            crate::executor_contract::require_executor(
                &admitted.checked().requirements(),
                Executor::EngineCandidate
            ),
            Ok(()),
            "{name}"
        );
        let prepared = prepare_case(device, admitted).unwrap_or_else(|error| panic!("{error}"));
        let (objects, buffers) = match &prepared.mapped {
            Mapped::Compute(compute) => (&compute.objects, &compute.buffers),
            Mapped::Render(render) => (
                &render.vertex.resources.objects,
                &render.vertex.resources.buffers,
            ),
            Mapped::Mesh(_) => panic!("{name}: a kernel or vertex case maps onto no mesh draw"),
        };
        assert_eq!(objects, &expected, "{name}");
        let fields = argument_buffers::field_words(
            &BTreeMap::new(),
            &BTreeMap::new(),
            &BTreeMap::new(),
            &BTreeMap::new(),
            objects.iter().map(|(&key, &word)| (key, word)),
        )
        .unwrap();
        let owner = argument_buffers::owner_bytes(buffers, 0, &fields).unwrap();
        let word = |at: usize| u64::from_le_bytes(owner[at..at + 8].try_into().unwrap());
        assert_eq!(
            [word(0), word(8)],
            [expected[&(0, 0)], expected[&(0, 8)]],
            "{name}"
        );
        assert_eq!(
            owner[16..20],
            buffers[&0][16..20],
            "{name}: the first scale"
        );
    }
}

/// The committed render replay cases (a rasterization-disabled vertex producer, the row's shape,
/// and a kernel whose draws run on host-encoded or inherited vertex buffers) carry no requirement,
/// and prepare device-free: the parent pipeline's token maps to the render template, a
/// rasterization-disabled vertex function bound by its buffers alone, beside the producer's own
/// executor case.
#[test]
fn checked_render_replay_cases_prepare_their_render_template() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus");
    let cases = CorpusStore::new(&root).read_all_cases().unwrap();
    for (name, render) in [
        (
            "gpu-encoded-vertex-buffers-run-the-host-encoded-draws",
            true,
        ),
        (
            "gpu-encoded-draws-run-the-host-encoded-vertex-buffers",
            false,
        ),
        ("gpu-encoded-draws-run-the-inherited-vertex-buffers", false),
    ] {
        let case = cases
            .iter()
            .find(|case| case.name == name)
            .unwrap_or_else(|| panic!("the committed public case {name}"))
            .clone();
        let device = DeviceType::Cpu;
        let admitted = admit(&root, case, ExecutionRequest::EngineCandidate { device })
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        assert!(admitted.checked().requirements().is_empty(), "{name}");
        let prepared = prepare_case(device, admitted).unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(
            matches!(prepared.mapped, Mapped::Render(_)),
            render,
            "{name}"
        );
        assert_eq!(
            prepared
                .templates()
                .unwrap()
                .keys()
                .copied()
                .collect::<Vec<_>>(),
            [1],
            "{name}"
        );
        let TemplateCase::Render(template) = &prepared.templates().unwrap()[&1].mapped else {
            panic!("{name}: the parent pipeline is a render template");
        };
        assert!(template.fragment.is_none(), "{name}");
        assert_eq!(
            template
                .vertex
                .resources
                .buffers
                .keys()
                .copied()
                .collect::<Vec<_>>(),
            [0, 1],
            "{name}"
        );
    }
}

/// A replayed draw runs only in its template's topology class (Metal compiled the template for it),
/// whatever primitive of that class the record names; another class fails the run, as a record the
/// GPU encoded with a draw its template does not define.
#[test]
fn a_replayed_draw_stays_in_its_templates_topology_class() {
    use crate::case::Primitive;
    use graphics::{replayed_draw_in_class, Assembly};
    let points = Assembly::Primitives(Primitive::Point);
    assert_eq!(replayed_draw_in_class(&points, Primitive::Point), Ok(()));
    let error = replayed_draw_in_class(&points, Primitive::Triangle).unwrap_err();
    assert!(error.contains("outside the topology class"), "{error}");
    let strips = Assembly::Primitives(Primitive::TriangleStrip);
    assert_eq!(replayed_draw_in_class(&strips, Primitive::Triangle), Ok(()));
    assert!(replayed_draw_in_class(&strips, Primitive::LineStrip).is_err());
    let lines = Assembly::Primitives(Primitive::Line);
    assert_eq!(replayed_draw_in_class(&lines, Primitive::LineStrip), Ok(()));
}

/// The device replay registers each compute template with its own threadgroup size and lengths, and
/// each dispatch the host encodes into an executed compute command buffer with its pipeline (the
/// parent's where the commands inherit it) and its regions' sizes, deduplicated in order; where the
/// producer encodes an action, each template also registers every edge size of its own. K14b's
/// committed case whose producer encodes the dispatches registers `icb_replay_template`'s eight
/// sizes of 1 to 8 threads (threadgroup length 32), the one whose host encodes them only its own
/// (groups of 8). A host command of another threadgroup size registers a second variant; one with
/// no dispatch makes the producer's, and the template's edges, registered; a ragged host-encoded
/// thread grid registers its edges.
#[test]
fn committed_compute_replay_cases_register_their_templates_variant() {
    use crate::case::indirect_commands::HostAction;
    use std::num::NonZeroU64;
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus");
    let cases = CorpusStore::new(&root).read_all_cases().unwrap();
    let mut lengths = [0u32; 31];
    lengths[0] = 32;
    let variant = |token: u64, threads: [u32; 3]| {
        ReplayVariant::new(NonZeroU64::new(token).unwrap(), threads, lengths)
    };
    let prepared = |name: &str| {
        let case = cases
            .iter()
            .find(|case| case.name == name)
            .unwrap_or_else(|| panic!("the committed public case {name}"))
            .clone();
        let device = DeviceType::Cpu;
        let admitted = admit(&root, case, ExecutionRequest::EngineCandidate { device })
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        prepare_case(device, admitted).unwrap_or_else(|error| panic!("{name}: {error}"))
    };
    // The producer encodes the first case's actions, so its template registers every edge size
    // of its own; the host encodes the second's.
    let edges =
        |token: u64| -> Vec<ReplayVariant> { (1..=8).map(|x| variant(token, [x, 1, 1])).collect() };
    let prepared_gpu = prepared("execute-two-gpu-encoded-template-dispatches");
    let Mapped::Compute(compute) = &prepared_gpu.mapped else {
        panic!("a kernel case");
    };
    assert_eq!(
        replay_variants(compute, prepared_gpu.templates().unwrap()).unwrap(),
        edges(1)
    );
    let name = "gpu-encoded-buffers-keep-the-host-encoded-dispatches";
    let host = prepared(name);
    let Mapped::Compute(compute) = &host.mapped else {
        panic!("{name}: a kernel case");
    };
    assert_eq!(
        replay_variants(compute, host.templates().unwrap()).unwrap(),
        [variant(2, [8, 1, 1])],
        "{name}"
    );
    let prepared = prepared("gpu-encoded-buffers-keep-the-host-encoded-dispatches");
    let Mapped::Compute(mut compute) = prepared.mapped else {
        panic!("a kernel case");
    };
    let commands = &mut compute
        .command_buffers
        .values_mut()
        .next()
        .unwrap()
        .commands;
    let Some(HostAction::DispatchThreadgroups {
        threads_per_threadgroup,
        ..
    }) = &mut commands[1].action
    else {
        panic!("command 1 dispatches threadgroups");
    };
    *threads_per_threadgroup = [4, 2, 1];
    let first = commands[0].action.take();
    let mut expected = vec![variant(2, [4, 2, 1])];
    expected.extend(edges(2));
    expected.sort_by_key(|v| v.threads_per_threadgroup);
    assert_eq!(
        replay_variants(&compute, prepared.templates.as_ref().unwrap()).unwrap(),
        expected,
        "a command whose action the producer encodes registers the template's edges"
    );
    // A host-encoded thread grid registers its regions' sizes: 10 x 3 threads in groups of 4 x 2
    // run groups of 4 x 2, 2 x 2, 4 x 1, and 2 x 1.
    let commands = &mut compute
        .command_buffers
        .values_mut()
        .next()
        .unwrap()
        .commands;
    commands[0].action = first;
    commands[1].action = Some(HostAction::DispatchThreads {
        grid: [10, 3, 1],
        threads_per_threadgroup: [4, 2, 1],
    });
    let mut expected = vec![
        variant(2, [2, 1, 1]),
        variant(2, [2, 2, 1]),
        variant(2, [4, 1, 1]),
        variant(2, [4, 2, 1]),
        variant(2, [8, 1, 1]),
    ];
    expected.sort_by_key(|v| v.threads_per_threadgroup);
    assert_eq!(
        replay_variants(&compute, prepared.templates.as_ref().unwrap()).unwrap(),
        expected
    );
}

/// A template the executor's own context cannot replay refuses a case only on that context. The
/// committed kernel case whose patch draws run a post-tessellation template prepares, routes to the
/// runtime, which replays every template through its own pipeline states, and keeps the context's
/// refusal for a run the runtime hands back.
#[test]
fn a_template_only_the_runtime_replays_refuses_only_on_the_executors_context() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus");
    let name = "patch-draws-a-kernel-encodes-at-factor-scales-and-the-host-encodes";
    let case = CorpusStore::new(&root)
        .read_all_cases()
        .unwrap()
        .into_iter()
        .find(|case| case.name == name)
        .unwrap_or_else(|| panic!("the committed public case {name}"));
    let device = DeviceType::Cpu;
    let admitted = admit(&root, case, ExecutionRequest::EngineCandidate { device })
        .unwrap_or_else(|error| panic!("{name}: {error}"));
    let prepared = prepare_case(device, admitted).unwrap_or_else(|error| panic!("{name}: {error}"));
    assert_eq!(runtime_run::route(&prepared), runtime_run::Route::Runtime);
    let refused = prepared.templates().err().expect("the context's refusal");
    assert!(
        matches!(&refused, ExecutorError::NotSupported(NotSupported::NotImplemented(why))
            if why.contains("is not a rasterization-disabled vertex function")),
        "{refused}"
    );
}

/// K14d's committed replay cases register the variants their regions select. The producer of
/// `gpu-encoded-ragged-grids-...` encodes every action, so its template (groups of 8 x 4, a
/// 128-byte length) registers all 32 sizes no larger; the host of `host-encoded-over-limit-...`
/// encodes every action, so it registers the template's own size and each of its dispatches'
/// regions (one-thread groups, groups of 2 and their one-thread edge, and groups of 4 x 2 with
/// their 1 x 2, 4 x 1, and 1 x 1 edges); and the replay-fault case's template (groups of 4 x 4, 64
/// bytes) registers its 16 sizes, none with the commands' 128-byte length.
/// A device replay records as many regions per command as keep its dispatches within the
/// product's bound, so a range executor v92 ran is never refused for its length: 3,971 commands of
/// 33 variants take all eight regions, 3,972 take seven (eight would be `TooManyDispatches`), and
/// Metal's most, 16,384 commands of 64 variants, one.
#[test]
fn a_long_replay_records_the_regions_its_dispatch_bound_holds() {
    use metal2vulkan::bind::BindState;
    use metal2vulkan::bind::{replay_preprocess, ReplayExecution, ReplayRefusal};
    use metal2vulkan::bind::{MAX_REPLAY_DISPATCHES, MAX_REPLAY_REGIONS};
    use metal2vulkan::error::BindError;
    use std::num::NonZeroU64;
    let path = crate::engine::profiles_dir().join("gate-lavapipe.json");
    let text = std::fs::read_to_string(path).unwrap();
    let description = DeviceDescription::from_khronos_json(&text, None).unwrap();
    let device = Device::new(description, DevicePolicy::default()).unwrap();
    let variants = |n: u32| -> Vec<ReplayVariant> {
        (1..=n)
            .map(|x| ReplayVariant::new(NonZeroU64::new(7).unwrap(), [x, 1, 1], [0; 31]))
            .collect()
    };
    let run = |count: u32, registered: &[ReplayVariant], regions: u32| {
        let mut execution = ReplayExecution::new(count, 0, count);
        execution.regions = regions;
        replay_preprocess(&device, &BindState::default(), &execution, registered).map(|_| ())
    };
    for (count, n, regions) in [
        (1, 1, MAX_REPLAY_REGIONS),
        (3_971, 33, 8),
        (3_972, 33, 7),
        (16_384, 64, 1),
    ] {
        let registered = variants(n);
        assert_eq!(
            replay_regions(count, registered.len()),
            regions,
            "{count} x {n}"
        );
        let dispatches = u64::from(count) * u64::from(n) * u64::from(regions);
        assert!(dispatches <= MAX_REPLAY_DISPATCHES, "{count} x {n}");
        run(count, &registered, regions).unwrap_or_else(|e| panic!("{count} x {n}: {e:?}"));
    }
    // Eight regions at the boundary are the refusal the chosen count avoids.
    let registered = variants(33);
    assert!(matches!(
        run(3_972, &registered, MAX_REPLAY_REGIONS),
        Err(BindError::InvalidReplay {
            reason: ReplayRefusal::TooManyDispatches { .. }
        })
    ));
}

#[test]
fn k14d_replay_cases_register_their_regions_variants() {
    use std::num::NonZeroU64;
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus");
    let cases = CorpusStore::new(&root).read_all_cases().unwrap();
    let registered = |name: &str| {
        let case = cases
            .iter()
            .find(|case| case.name == name)
            .unwrap_or_else(|| panic!("the committed public case {name}"))
            .clone();
        let device = DeviceType::Cpu;
        let admitted = admit(&root, case, ExecutionRequest::EngineCandidate { device })
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let prepared =
            prepare_case(device, admitted).unwrap_or_else(|error| panic!("{name}: {error}"));
        let Mapped::Compute(compute) = &prepared.mapped else {
            panic!("{name}: a kernel case");
        };
        replay_variants(compute, prepared.templates().unwrap()).unwrap()
    };
    let variants = |length: u32, sizes: &[[u32; 3]]| -> Vec<ReplayVariant> {
        let mut lengths = [0u32; 31];
        lengths[0] = length;
        let mut out: Vec<ReplayVariant> = sizes
            .iter()
            .map(|&size| ReplayVariant::new(NonZeroU64::new(1).unwrap(), size, lengths))
            .collect();
        out.sort_by_key(|v| v.threads_per_threadgroup);
        out
    };
    let edges = |local: [u32; 3]| -> Vec<[u32; 3]> {
        let mut out = Vec::new();
        for z in 1..=local[2] {
            for y in 1..=local[1] {
                for x in 1..=local[0] {
                    out.push([x, y, z]);
                }
            }
        }
        out
    };
    assert_eq!(
        registered("gpu-encoded-ragged-grids-replay-with-the-inherited-pipeline-and-buffers"),
        variants(128, &edges([8, 4, 1]))
    );
    assert_eq!(
        registered("host-encoded-over-limit-and-ragged-dispatches-in-two-command-buffers"),
        variants(
            128,
            &[
                [1, 1, 1],
                [1, 2, 1],
                [2, 1, 1],
                [4, 1, 1],
                [4, 2, 1],
                [8, 4, 1]
            ]
        )
    );
    assert_eq!(
        registered("gpu-encoded-commands-of-a-length-no-variant-registers"),
        variants(64, &edges([4, 4, 1]))
    );
}

/// A device replay's fault refuses the row (`not_supported{bind}`) only when every reason is one of
/// the product's typed limits (no registered variant, a grid past `2^32 - 1` threads, more regions
/// than the execution records); any other reason, alone or beside them, fails the row, as the host
/// replay fails a dynamic stride, and as a preprocess that
/// disagrees with the host decode must. The verdict names the reasons.
#[test]
fn a_replay_fault_refuses_the_row_only_for_the_products_typed_limits() {
    use metal2vulkan::bind::ReplayFaults as F;
    let limits = [F::VARIANT, F::GROUP_COUNT, F::REGIONS];
    let others = [
        F::ENCODING,
        F::ACTION,
        F::PIPELINE,
        F::BUFFERS,
        F::RECORD,
        F::RANGE,
    ];
    let refused = |faults: u32| match replay_fault_verdict("command buffer (0, 0)", F(faults)) {
        ExecutorError::NotSupported(NotSupported::Bind(why)) => Ok(why),
        ExecutorError::Failed(why) => Err(why),
        other => panic!("{other:?}"),
    };
    for bit in limits {
        let why = refused(bit.0).unwrap_or_else(|why| panic!("{bit:?} failed: {why}"));
        assert!(why.starts_with("command buffer (0, 0): "), "{why}");
    }
    let all_limits = limits.iter().fold(0, |all, bit| all | bit.0);
    let why = refused(all_limits).unwrap_or_else(|why| panic!("{why}"));
    assert!(
        why.contains(
            "no registered variant, a grid past 2^32 - 1 threads, more regions than the \
             execution records"
        ),
        "{why}"
    );
    for bit in others {
        let why = refused(bit.0).expect_err("fails the row");
        assert!(why.contains(&format!("{:#x}", bit.0)), "{why}");
        refused(bit.0 | F::VARIANT.0).expect_err("fails the row beside a limit");
    }
    // A fault bit with no reason is no typed limit.
    refused(0).expect_err("fails the row");
}

/// An executed command buffer of `descriptor`'s family with no commands, no parent pipeline, and
/// no parent buffers: what `replay_templates` reads of it is that it executes, and its family.
fn executed_command_buffer(
    descriptor: crate::case::indirect_commands::CommandDescriptor,
) -> crate::case::indirect_commands::CommandBufferResource {
    crate::case::indirect_commands::CommandBufferResource {
        buffer_binding: 0,
        field_offset: 16,
        capacity: 1,
        execution_range: [0, 1],
        inherit_pipeline_state: false,
        inherit_buffers: false,
        descriptor,
        commands: vec![],
        execute: Some(crate::case::indirect_commands::Execution {
            pipeline: None,
            buffers: vec![],
            vertex_buffers: vec![],
            fragment_buffers: vec![],
            object_buffers: vec![],
            mesh_buffers: vec![],
        }),
    }
}

/// A compute command buffer's descriptor, for [`executed_command_buffer`].
fn compute_commands() -> crate::case::indirect_commands::CommandDescriptor {
    crate::case::indirect_commands::CommandDescriptor::Compute {
        dispatch: crate::case::indirect_commands::ComputeDispatch::Both,
        max_buffer_bind_count: 1,
        max_threadgroup_memory_bind_count: 0,
        support_dynamic_attribute_stride: false,
    }
}

/// The templates the replay maps are of the families the case's executed command buffers run. The
/// committed case over a fragment template's and a kernel template's pipeline states
/// (`pipeline-states-beside-the-data-of-two-stages`), given an executed compute command buffer,
/// maps only the kernel template: no compute command selects the fragment one, which the checker
/// and the oracle accept beside compute commands. Given an executed render command buffer instead,
/// the fragment template is the one mapped, and refused: it is no rasterization-disabled vertex
/// function.
#[test]
fn replay_templates_map_only_the_families_executed_commands_run() {
    use crate::case::indirect_commands::CommandDescriptor;
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus");
    let case = CorpusStore::new(&root)
        .read_all_cases()
        .unwrap()
        .into_iter()
        .find(|case| case.name == "pipeline-states-beside-the-data-of-two-stages")
        .expect("the committed public case");
    let request = ExecutionRequest::EngineCandidate {
        device: DeviceType::Cpu,
    };
    let mut checked = admit(&root, case, request)
        .unwrap_or_else(|error| panic!("{error}"))
        .into_checked_for(request)
        .unwrap();
    let stages = checked
        .pipeline_templates
        .values()
        .map(|template| template.case.stage)
        .collect::<Vec<_>>();
    assert!(stages.contains(&Stage::Fragment) && stages.contains(&Stage::Kernel));
    // Not executing commands, the case selects no template.
    assert!(replay_templates(&checked).unwrap().is_empty());
    checked
        .primary
        .case
        .command_buffers
        .push(executed_command_buffer(compute_commands()));
    let templates = replay_templates(&checked).unwrap_or_else(|error| panic!("{error}"));
    let kernel_token = checked
        .primary
        .case
        .pipeline_states
        .iter()
        .find(|state| checked.pipeline_templates[&state.case_id].case.stage == Stage::Kernel)
        .unwrap()
        .token;
    assert_eq!(
        templates.keys().copied().collect::<Vec<_>>(),
        [kernel_token]
    );
    assert!(matches!(
        templates[&kernel_token].mapped,
        TemplateCase::Compute(_)
    ));
    checked.primary.case.command_buffers[0] = executed_command_buffer(CommandDescriptor::Render {
        draws: crate::case::indirect_commands::RenderDraws::default(),
        max_vertex_buffer_bind_count: 1,
        max_fragment_buffer_bind_count: 0,
        max_object_buffer_bind_count: 0,
        max_mesh_buffer_bind_count: 0,
        support_dynamic_attribute_stride: false,
    });
    let error = replay_templates(&checked).err().unwrap().to_string();
    assert!(
        error.contains("is not a rasterization-disabled vertex function"),
        "{error}"
    );
}

/// The mirror and the mixed run of
/// [`replay_templates_map_only_the_families_executed_commands_run`]. Beside render commands only
/// (`gpu-encoded-draws-run-the-inherited-vertex-buffers`), a kernel pipeline-state literal that
/// binds beyond buffers (`pipeline-states-beside-the-data-of-two-stages`'s kernel template with a
/// sampler added) is no template: no render command selects it, so it is skipped, not refused.
/// Once compute commands execute too, it is a template, and refused. With its buffers only, the
/// mixed run maps both families, each token to its own.
#[test]
fn replay_templates_skip_another_familys_literal_and_map_both_families_of_a_mixed_run() {
    use crate::case::{SamplerAddressMode, SamplerFilter, SamplerMipFilter, SamplerResource};
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus");
    let cases = CorpusStore::new(&root).read_all_cases().unwrap();
    let request = ExecutionRequest::EngineCandidate {
        device: DeviceType::Cpu,
    };
    let checked_case = |name: &str| {
        let case = cases
            .iter()
            .find(|case| case.name == name)
            .unwrap_or_else(|| panic!("the committed public case {name}"))
            .clone();
        admit(&root, case, request)
            .unwrap_or_else(|error| panic!("{name}: {error}"))
            .into_checked_for(request)
            .unwrap()
    };
    let mut checked = checked_case("gpu-encoded-draws-run-the-inherited-vertex-buffers");
    assert!(checked
        .primary
        .case
        .command_buffers
        .iter()
        .all(|command| command.execute.is_none() || command.is_render()));
    let render = replay_templates(&checked).unwrap_or_else(|error| panic!("{error}"));
    assert!(!render.is_empty());
    assert!(render
        .values()
        .all(|template| matches!(template.mapped, TemplateCase::Render(_))));

    let mut beside = checked_case("pipeline-states-beside-the-data-of-two-stages");
    let mut literal = beside
        .primary
        .case
        .pipeline_states
        .iter()
        .find(|state| beside.pipeline_templates[&state.case_id].case.stage == Stage::Kernel)
        .cloned()
        .unwrap();
    let mut kernel = beside.pipeline_templates.remove(&literal.case_id).unwrap();
    assert!(kernel.case.samplers.is_empty());
    kernel.case.samplers.push(SamplerResource {
        binding: 0,
        address_mode: SamplerAddressMode::ClampToEdge,
        min_filter: SamplerFilter::Nearest,
        mag_filter: SamplerFilter::Nearest,
        mip_filter: SamplerMipFilter::NotMipmapped,
        normalized_coordinates: true,
    });
    let token = render.keys().max().unwrap() + 1;
    literal.token = token;
    let case_id = literal.case_id.clone();
    checked.pipeline_templates.insert(case_id.clone(), kernel);
    checked.primary.case.pipeline_states.push(literal);

    // Render commands only: the kernel literal is skipped, not refused.
    let skipped = replay_templates(&checked).unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(
        skipped.keys().collect::<Vec<_>>(),
        render.keys().collect::<Vec<_>>()
    );

    // Compute commands execute too: the literal is a template, and it binds beyond buffers.
    checked
        .primary
        .case
        .command_buffers
        .push(executed_command_buffer(compute_commands()));
    let error = replay_templates(&checked).err().unwrap().to_string();
    assert!(
        error.contains("binds more than buffers and threadgroup memory"),
        "{error}"
    );

    // With its buffers only, the mixed run maps both families.
    checked
        .pipeline_templates
        .get_mut(&case_id)
        .unwrap()
        .case
        .samplers
        .pop();
    let both = replay_templates(&checked).unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(
        both.keys().copied().collect::<Vec<_>>(),
        render.keys().copied().chain([token]).collect::<Vec<_>>()
    );
    assert!(matches!(both[&token].mapped, TemplateCase::Compute(_)));
    for token in render.keys() {
        assert!(matches!(both[token].mapped, TemplateCase::Render(_)));
    }
}

/// An allocation's guest address names a buffer whatever device address it has: NVIDIA's lie
/// below 2^32 (a vertex buffer at 0x5020000 on an RTX 3090), where a guest address equal to it
/// was a nil buffer's offset to a record, and RADV's sign-extended upper half
/// (0xffff800100040000 on a Radeon 780M) lies among the records, from 2^63. One offset for all,
/// on the address read as signed, keeps them apart and in [2^32, 2^63); an address it cannot move
/// there is refused.
#[test]
fn addressed_allocations_take_metal_addresses() {
    use metal2vulkan::bind::indirect_commands::{records_address, BufferBinding};
    let metal = |address: BufferAddress| {
        let names = BufferBinding {
            address: address.guest,
            attribute_stride: None,
        }
        .names_buffer();
        names && address.guest < records_address(0)
    };
    let low = BufferAddress::addressed(0x502_0000).unwrap();
    assert_eq!(low.host, Some(0x502_0000));
    assert!(metal(low), "{low:x?}");
    let next = BufferAddress::addressed(0x502_0000 + 256).unwrap();
    assert_eq!(next.guest - low.guest, 256);
    let radv = BufferAddress::addressed(0xffff_8001_0004_0000).unwrap();
    assert!(metal(radv) && radv.guest < low.guest, "{radv:x?}");
    // The top of the device's space and its bottom stay next to each other.
    let top = BufferAddress::addressed(u64::MAX).unwrap();
    let bottom = BufferAddress::addressed(0).unwrap();
    assert!(metal(top) && metal(bottom), "{top:x?} {bottom:x?}");
    assert_eq!(bottom.guest - top.guest, 1);
    let lowest = BufferAddress::addressed(0xffff_8000_0000_0000).unwrap();
    assert_eq!(lowest.guest, BufferBinding::MIN_BUFFER_ADDRESS);
    let highest = BufferAddress::addressed(records_address(0) - BufferAddress::GUEST_OFFSET - 1);
    assert_eq!(highest.unwrap().guest, records_address(0) - 1);
    assert!(BufferAddress::addressed(0xffff_7fff_ffff_ffff).is_err());
    assert!(BufferAddress::addressed(records_address(0) - BufferAddress::GUEST_OFFSET).is_err());
    assert!(BufferAddress::addressed(records_address(1)).is_err());
}
