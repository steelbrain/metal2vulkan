//! Device-free tests of the engine candidate: the status and output rules and the case mapping.
//! No device is opened.

use super::*;
use crate::case::ScalarType;
use crate::check::admission::NotSupported;
use crate::requirement::ToolingRequirement;
use metal2vulkan::device::{Device, DeviceDescription, DevicePolicy};
use metal2vulkan::metal::DataType;
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
    Device::from_queries(description, policy).unwrap()
}

/// A difference under notes is a noted difference unless the C3 reviewed list holds its case under
/// an answer gap the case's structure meets: then it stays a mismatch naming the gap. Judged over
/// the committed list's every answer gap over a public fixture and its committed case, and a
/// committed public case it does not hold; a difference no note can explain is a mismatch for
/// either.
#[test]
fn a_difference_under_notes_is_noted_unless_an_answer_gap_holds_its_case_open() {
    let root = crate::source::corpus_root();
    let store = CorpusStore::new(&root);
    let list = ReviewedList::load(&ReviewedList::committed_path()).unwrap();
    let noted = ComparisonClass::Unbounded(
        [metal2vulkan::pipeline::PrecisionNote::ClippedCoverageSplit].into(),
    );
    let status = |case: &AuthoredCase, comparison: &ComparisonClass| {
        let checked = crate::check::check_case(&root, case.clone())
            .unwrap_or_else(|errors| panic!("{}: {errors:?}", case.name));
        difference_status(&checked, true, comparison, None, || Ok(list.clone())).unwrap()
    };
    let mut public = 0;
    for entry in &list.answer_gaps {
        let case = store.find_case(&entry.case_id).unwrap().unwrap();
        if case.public_source.is_none() {
            continue;
        }
        let held = (CandidateStatus::Mismatch, Some(entry.gap));
        assert_eq!(status(&case, &noted), held, "{}", case.name);
        let unexplained = (CandidateStatus::Mismatch, None);
        assert_eq!(status(&case, &ComparisonClass::Exact), unexplained);
        public += 1;
    }
    assert!(public >= 5, "{public}");
    let unheld = store
        .read_all_cases()
        .unwrap()
        .into_iter()
        .find(|case| {
            case.public_source.is_some() && list.holding_open(&case.case_id, |_| true).is_none()
        })
        .unwrap();
    let difference = (CandidateStatus::NotedDifference, None);
    assert_eq!(status(&unheld, &noted), difference);
    let unexplained = (CandidateStatus::Mismatch, None);
    assert_eq!(status(&unheld, &ComparisonClass::Exact), unexplained);
}

/// [`compute_case`] of a case whose entry declares no stage input: every case here but the
/// stage-input ones, which map against their own interface.
fn map_kernel(case: &AuthoredCase) -> Result<ComputeCase, NotSupported> {
    compute_case(case)
}

fn kernel_case() -> AuthoredCase {
    let mut case = crate::case::vector_function_constant_test_case("11".repeat(32), "main".into());
    case.function_constants.clear();
    case
}

#[test]
fn the_case_maps_or_names_the_gap() {
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

    // An explicit imageblock is the runtime's limit.
    let mut imageblock = kernel_case();
    imageblock.imageblock = Some(crate::case::ImageblockResource {
        dimensions: [16, 8],
        implicit_coverage: None,
    });
    assert_eq!(
        map_kernel(&imageblock),
        Err(NotSupported::Requirements(
            [ToolingRequirement::KernelImageblock].into()
        ))
    );
    let mut vertex = kernel_case();
    vertex.stage = Stage::Vertex;
    assert!(matches!(
        map_kernel(&vertex),
        Err(NotSupported::NotImplemented(what)) if what.contains("not a compute case")
    ));
}

/// A case's visible-function references are no resource gap: its function links them, each
/// module once with the functions it supplies and its retained bitcode, as the Metal oracle links
/// them. A kernel's visible-function table and intersection-function table, a linked intersection
/// function and an argument-buffer intersection-function table are requirements that stop this
/// executor ([`require_case_families`]). The linking takes the tables' entries' functions as the
/// oracle's does.
#[test]
fn references_link_and_function_tables_are_typed_gaps() {
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

    // A kernel's function table and intersection-function table are the runtime's limits.
    let refused = |requirements: &[ToolingRequirement]| {
        Err(NotSupported::Requirements(
            requirements.iter().copied().collect(),
        ))
    };
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
    assert_eq!(
        require_case_families(&visible),
        refused(&[ToolingRequirement::KernelFunctionTable])
    );
    let mut opaque = case.clone();
    opaque.intersection_function_tables = vec![IntersectionFunctionTableResource {
        binding: 2,
        size: 3,
        entries: vec![IntersectionFunctionTableEntry::OpaqueTriangle {
            index: 1,
            signature: vec![IntersectionFunctionSignature::TriangleData],
        }],
    }];
    assert_eq!(
        require_case_families(&opaque),
        refused(&[ToolingRequirement::KernelAccelerationStructure])
    );
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
    let linked = refused(&[
        ToolingRequirement::LinkedIntersectionFunction,
        ToolingRequirement::KernelAccelerationStructure,
    ]);
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
    // The authored compile census's: the kernel as the runtime compiles it for the dispatch.
    let (request, given) = kernel_request(&case, &linked, &kernel);
    assert_eq!(
        request,
        PipelineRequest::Kernel {
            function: linking(&kernel.entry),
            threads_per_threadgroup: kernel.dispatch.threads_per_group(),
        }
    );
    assert_eq!(given, expected_libraries(&[&case]));
    let json = serde_json::to_value(&request).unwrap();
    assert_eq!(
        serde_json::from_value::<PipelineRequest>(json).unwrap(),
        request
    );

    let mut vertex_case = two_stage_case("shade", "unused", vec![], BTreeMap::new(), 3);
    vertex_case.fragment = None;
    let (request, given) = render_request(&case, None, &linked, &vertex_case).unwrap();
    assert_eq!(
        request,
        PipelineRequest::VertexDraw {
            function: linking("shade"),
            template: graphics::render_template(&vertex_case),
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
        render_request(&case, Some(&companion), &linked, &fragment_case).unwrap();
    assert_eq!(
        request,
        PipelineRequest::Draw {
            vertex: alone(passthrough.entry()),
            fragment: linking("shade"),
            template: graphics::render_template(&fragment_case),
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
        render_request(&case, Some(&companion), &linked, &observed_case).unwrap();
    assert_eq!(
        request,
        PipelineRequest::Draw {
            vertex: linking("shade"),
            fragment: alone(observer.entry()),
            template: graphics::render_template(&observed_case),
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
        render_request(&case, Some(&companion), &linked, &owned),
        Err(ExecutorError::Failed(_))
    ));
    assert!(matches!(
        render_request(&case, None, &linked, &observed_case),
        Err(ExecutorError::Failed(_))
    ));

    // The generated companion is the stage the case's own function is not.
    assert_eq!(observed_case.companion_stage(), Some(ShaderStage::Fragment));
    assert_eq!(fragment_case.companion_stage(), Some(ShaderStage::Vertex));
    assert_eq!(vertex_case.companion_stage(), None);
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
    let selected = |output| select_output(output, &buffers, &embedded, &none, &BTreeMap::new());
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
    let selected = |output| select_output(output, &buffers, &embedded, &none, &member);
    assert_eq!(selected(region(16, [1, 0, 0])), Ok(vec![2, 4]));
    assert!(selected(region(8, [0, 0, 0])).is_err());
    assert!(selected(region(16, [2, 0, 0])).is_err());
}

/// `bytes` as the base64 an authored case spells.
fn b64(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// Device buffer array elements bind at consecutive Metal slots (the oracle's `first + k`); an
/// argument-buffer buffer's guest address is held by its owner at its field, the owner covering
/// that field; each maps its output, and a collision, an owner that is not bound, or an element
/// slot that overflows is a gap.
#[test]
fn arrays_and_argument_buffers_map() {
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

    // The owner is created holding the pointer, little endian, at its field.
    let fields = BTreeMap::from([((0, 8), 0xdead_0000)]);
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
/// textures, and an element output is that texture's region. Element 0 of an array that overrides the texture at its base
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
    // The final element texture is the observation.
    let mut finals = mapped.textures.clone();
    finals.get_mut(&3).unwrap().bytes = 99u32.to_le_bytes().to_vec();
    assert_eq!(
        select_output(
            mapped.output,
            &BTreeMap::new(),
            &BTreeMap::new(),
            &finals,
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

#[test]
fn identity_is_per_profile_and_per_engine_slot() {
    let a = engine_fingerprint(device("lavapipe", DevicePolicy::RESIDENT_TABLES).profile());
    let b = engine_fingerprint(device("moltenvk-m3", DevicePolicy::RESIDENT_TABLES).profile());
    assert_ne!(a, b);
    assert_eq!(a.len(), 64);
    assert_eq!(
        engine_environment_id("ubuntu-26.04-lavapipe"),
        "ubuntu-26.04-lavapipe@engine"
    );
}

/// A row run under the profiles layer has its own slot per profile, every one an engine slot; an
/// operator ID naming any engine slot is not one the executors accept, nor is a slot with a suffix
/// after its profile.
#[test]
fn a_restricted_row_has_its_own_engine_slot() {
    let native = engine_slot("ubuntu-26.04-lavapipe", None);
    let floor = engine_slot("ubuntu-26.04-lavapipe", Some("runtime-floor"));
    let baseline = engine_slot("ubuntu-26.04-lavapipe", Some("gate-lavapipe"));
    assert_eq!(native, engine_environment_id("ubuntu-26.04-lavapipe"));
    assert_eq!(floor, "ubuntu-26.04-lavapipe@engine@runtime-floor");
    let slots = [&native, &floor, &baseline];
    for (index, slot) in slots.iter().enumerate() {
        assert!(is_engine_slot(slot), "{slot}");
        assert!(!slots[..index].contains(slot), "{slot} twice");
    }
    for operator in [
        "ubuntu-26.04-lavapipe",
        "ubuntu-26.04-lavapipe@engine-x",
        "x@engine@",
        "x@engine@a@b",
        "x@engine+robustness-off",
        "x@engine@+robustness-off",
        "x@engine@a+b",
    ] {
        assert!(!is_engine_slot(operator), "{operator}");
    }
}

/// The committed public cases over an indirect command buffer or a pipeline state an argument
/// buffer holds stop this executor as they are prepared, device-free, by the runtime's limits the
/// vocabulary names: compute commands, draws other than indexed ones, inheritance other than the
/// pipeline state's alone, commands the host encodes, and pipeline states in an argument buffer.
#[test]
fn committed_command_buffer_and_pipeline_state_cases_refuse_by_their_requirements() {
    use ToolingRequirement::{
        ArgumentBufferPipelineState, HostEncodedIndirectCommands, IndirectCommandInheritance,
        IndirectComputeCommands, IndirectDrawTypes,
    };
    let root = metal2vulkan_devtools::validation_dir().join("corpus");
    let cases = CorpusStore::new(&root).read_all_cases().unwrap();
    for (name, expected) in [
        (
            "pipeline-states-beside-the-data-of-two-stages",
            vec![ArgumentBufferPipelineState],
        ),
        (
            "pipeline-states-beside-the-data-a-vertex-reads",
            vec![ArgumentBufferPipelineState],
        ),
        (
            "execute-two-gpu-encoded-template-dispatches",
            vec![IndirectComputeCommands, ArgumentBufferPipelineState],
        ),
        (
            "gpu-encoded-buffers-keep-the-host-encoded-dispatches",
            vec![IndirectComputeCommands, HostEncodedIndirectCommands],
        ),
        (
            "gpu-encoded-vertex-buffers-run-the-host-encoded-draws",
            vec![IndirectDrawTypes, HostEncodedIndirectCommands],
        ),
        (
            "gpu-encoded-draws-run-the-host-encoded-vertex-buffers",
            vec![IndirectDrawTypes, HostEncodedIndirectCommands],
        ),
        (
            "gpu-encoded-draws-run-the-inherited-vertex-buffers",
            vec![IndirectDrawTypes, IndirectCommandInheritance],
        ),
        (
            "patch-draws-a-kernel-encodes-at-factor-scales-and-the-host-encodes",
            vec![
                IndirectDrawTypes,
                IndirectCommandInheritance,
                HostEncodedIndirectCommands,
            ],
        ),
    ] {
        let case = cases
            .iter()
            .find(|case| case.name == name)
            .unwrap_or_else(|| panic!("the committed public case {name}"))
            .clone();
        let device = DeviceType::Cpu;
        let admitted = admit(&root, case, ExecutionRequest::EngineCandidate { device })
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let refused = prepare_case(device, admitted).err();
        assert!(
            matches!(&refused, Some(ExecutorError::NotSupported(NotSupported::Requirements(stops)))
                if stops.iter().eq(&expected)),
            "{name}: {refused:?}"
        );
    }
}

/// The runtime draws a vertex or fragment case of any sample count, its own stage's buffers,
/// textures and samplers, argument-buffer members, vertex buffers, color targets, and one direct
/// or indexed draw of primitives or one patch draw with no index buffer, observing a buffer, a
/// texture, an argument-buffer member, a color target or nothing; and a rasterization-disabled
/// vertex case, which renders into the oracle's sink; a depth/stencil attachment of one aspect, of
/// one sample or more, observed or not; a fragment pass with no attachment, of one sample or more;
/// a layered pass; and an attribute buffer's out-of-bounds target. A case with anything else (a
/// depth/stencil attachment of both aspects, a patch draw through a patch index buffer, or an index
/// buffer's out-of-bounds target) is refused before it runs.
#[test]
fn a_plain_draw_and_only_one_runs_through_the_runtime() {
    use super::graphics::tests::two_stage_case;
    use super::graphics::{ColorTarget, RenderCase, RenderOutput};
    use crate::case::TextureFormat;
    use crate::literal::LiteralDepthStencil;
    let target = ColorTarget {
        format: TextureFormat::Rgba8Unorm,
        dimensions: [2, 2],
        samples: 1,
        bytes: vec![0; 16],
    };
    let plain = two_stage_case(
        "v",
        "f",
        Vec::new(),
        BTreeMap::from([(0, target.clone())]),
        3,
    );
    assert!(runs_render_through_runtime(&plain));
    let indexed = RenderCase {
        draw: DrawCall::Indexed {
            index_type: metal2vulkan::vk::IndexType::UINT16,
            index_buffer: metal2vulkan::bind::IndexBufferRef::new(graphics::INDEX_BUFFER_TOKEN, 6),
            index_count: 3,
            index_offset: 0,
            instance_count: 1,
            base_vertex: 0,
            base_instance: 0,
        },
        index_buffer: Some(vec![0; 6]),
        output: RenderOutput::None,
        ..plain.clone()
    };
    assert!(runs_render_through_runtime(&indexed));
    let unrasterized = RenderCase {
        fragment: None,
        color_targets: BTreeMap::new(),
        output: RenderOutput::Buffer {
            stage: ShaderStage::Vertex,
            binding: 0,
            offset: 0,
            length: 4,
        },
        ..plain.clone()
    };
    assert!(runs_render_through_runtime(&unrasterized));
    let patches = |patch_index| RenderCase {
        assembly: graphics::Assembly::Patches {
            tessellation: metal2vulkan::pipeline::TessellationDescriptor::default(),
            factors: vec![0; 12],
        },
        draw: DrawCall::Patches {
            control_points: 3,
            patch_start: 0,
            patch_count: 1,
            patch_index,
            instance_count: 1,
            base_instance: 0,
            control_point_index: None,
        },
        ..plain.clone()
    };
    assert!(runs_render_through_runtime(&patches(None)));
    // An attribute buffer's out-of-bounds target draws through the runtime, whose facts refuse or
    // run it (`runtime_bounds::drawn`); no other target does (an index buffer's case carries the
    // runtime's limit, `index_buffer_past_end`, which stops it before either context).
    use crate::case::out_of_bounds::ContainmentForm;
    let bounded = |resource, form| RenderCase {
        out_of_bounds: vec![OutOfBoundsTarget {
            resource,
            form,
            reach_before: 0,
            reach_after: 16,
        }],
        ..plain.clone()
    };
    assert!(runs_render_through_runtime(&bounded(
        OutOfBoundsResource::AttributeBuffer { location: 0 },
        ContainmentForm::VertexInput
    )));
    // An argument buffer's buffer, texture and sampler, observed through the buffer member or the
    // texture member.
    let mut member = plain.clone();
    if let Some(fragment) = &mut member.fragment {
        let resources = &mut fragment.resources;
        resources
            .argument_buffer_buffers
            .insert((0, 0), vec![0; 16]);
        resources.argument_buffer_textures.insert(
            (0, 8),
            crate::literal::LiteralTexture {
                binding: 0,
                role: crate::case::ResourceRole::Input,
                texture_type: crate::case::TextureType::D2,
                format: TextureFormat::Rgba8Unorm,
                dimensions: [1, 1, 1],
                sample_count: 1,
                bytes: vec![0; 4],
                sparse: None,
                mip_levels: 1,
            },
        );
    }
    assert!(runs_render_through_runtime(&member));
    for output in [
        RenderOutput::ArgumentBufferBuffer {
            stage: metal2vulkan::pipeline::Stage::Fragment,
            owner: 0,
            field_offset: 0,
            offset: 0,
            length: 16,
        },
        RenderOutput::ArgumentBufferTexture {
            stage: metal2vulkan::pipeline::Stage::Fragment,
            owner: 0,
            field_offset: 8,
            origin: [0, 0, 0],
            dimensions: [1, 1, 1],
        },
    ] {
        let observed = RenderCase {
            output,
            ..member.clone()
        };
        assert!(runs_render_through_runtime(&observed));
    }
    // A depth/stencil attachment of one aspect, of one sample or more, observed or not, with or
    // without a color target; one of both aspects is refused before it runs.
    let depth = LiteralDepthStencil {
        dimensions: [2, 2],
        samples: 1,
        depth: Some(vec![0; 16]),
        stencil: None,
    };
    let stencil = LiteralDepthStencil {
        depth: None,
        stencil: Some(vec![0; 4]),
        ..depth.clone()
    };
    let multisampled_depth = LiteralDepthStencil {
        samples: 4,
        depth: Some(vec![0; 64]),
        ..depth.clone()
    };
    let multisampled_stencil = LiteralDepthStencil {
        samples: 4,
        stencil: Some(vec![0; 16]),
        ..stencil.clone()
    };
    for attachment in [&depth, &stencil, &multisampled_depth, &multisampled_stencil] {
        let attached = RenderCase {
            depth_stencil: Some(attachment.clone()),
            ..plain.clone()
        };
        assert!(runs_render_through_runtime(&attached));
        let observed = RenderCase {
            color_targets: BTreeMap::new(),
            output: match attachment.depth {
                Some(_) => RenderOutput::Depth {
                    origin: [0, 0],
                    dimensions: [1, 1],
                },
                None => RenderOutput::Stencil {
                    origin: [0, 0],
                    dimensions: [1, 1],
                },
            },
            ..attached
        };
        assert!(runs_render_through_runtime(&observed));
    }
    // A pass of more samples, its color target filled and read back sample by sample.
    let multisampled = RenderCase {
        raster_sample_count: Some(4),
        color_targets: BTreeMap::from([(
            0,
            ColorTarget {
                samples: 4,
                bytes: vec![0; 64],
                ..target.clone()
            },
        )]),
        ..plain.clone()
    };
    assert!(runs_render_through_runtime(&multisampled));
    // A fragment pass of no attachment, of one sample or more.
    for raster_sample_count in [None, Some(4)] {
        let attachmentless = RenderCase {
            color_targets: BTreeMap::new(),
            output: RenderOutput::None,
            raster_sample_count,
            ..plain.clone()
        };
        assert!(runs_render_through_runtime(&attachmentless));
    }
    // A layered pass, of one slice.
    let layered = RenderCase {
        render_target_array_length: 1,
        ..plain.clone()
    };
    assert!(runs_render_through_runtime(&layered));
    let others = [
        RenderCase {
            depth_stencil: Some(LiteralDepthStencil {
                stencil: Some(vec![0; 4]),
                ..depth.clone()
            }),
            ..plain.clone()
        },
        bounded(
            OutOfBoundsResource::IndexBuffer,
            ContainmentForm::IndexBuffer,
        ),
        patches(Some((
            metal2vulkan::bind::IndexBufferRef::new(graphics::INDEX_BUFFER_TOKEN, 4),
            0,
        ))),
        RenderCase {
            draw: patches(None).draw,
            ..plain.clone()
        },
    ];
    for other in &others {
        assert!(!runs_render_through_runtime(other), "{other:?}");
    }
}
