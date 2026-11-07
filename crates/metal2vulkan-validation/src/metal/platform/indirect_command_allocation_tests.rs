//! Native indirect-command allocation. Every test here except the pure allocation-order check
//! creates Metal objects on the system device, so those compile only under `--cfg m2v_gpu`.

use super::indirect_commands::allocate_command_resources;
#[cfg(m2v_gpu)]
use super::libraries::{load_library, shader_air};
use super::*;

/// The case's own module, run as compiled.
#[cfg(m2v_gpu)]
fn loop_free(row: &crate::source::SourceRow) -> crate::metal::OracleSource<'_> {
    crate::metal::OracleSource {
        row,
        entry_ll: &row.air_ll,
        air: shader_air(row).unwrap().bitcode.into(),
        bound: crate::oracle_fuel::OracleBound::LoopFree,
        companion: None,
    }
}
use crate::case::indirect_commands::{CommandBufferResource, CommandDescriptor, ComputeDispatch};
#[cfg(m2v_gpu)]
use crate::case::ScalarType;
#[cfg(m2v_gpu)]
use crate::check::interface::tests::{argument_buffer, object, param};
#[cfg(m2v_gpu)]
use crate::check::interface::AirInterface;
#[cfg(m2v_gpu)]
use crate::ScratchDir;
#[cfg(m2v_gpu)]
use metal2vulkan_air::abi::ParamRole;
#[cfg(m2v_gpu)]
use objc2_metal::MTLArgumentEncoder;
#[cfg(m2v_gpu)]
use objc2_metal::MTLIndirectCommandBuffer;
#[cfg(m2v_gpu)]
use std::fs;

/// An entry that declares `params` and nothing else.
#[cfg(m2v_gpu)]
fn declaring(params: Vec<crate::check::interface::Param>) -> AirInterface {
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

fn case(descriptor: CommandDescriptor, capacity: u32) -> AuthoredCase {
    let mut case = crate::case::attachmentless_fragment_test_case("11".repeat(32), "unused".into());
    case.command_buffers.push(CommandBufferResource {
        buffer_binding: 0,
        field_offset: 0,
        capacity,
        execution_range: [0, capacity],
        inherit_pipeline_state: false,
        inherit_buffers: false,
        descriptor,
        commands: vec![],
        execute: None,
    });
    case
}

fn compute_descriptor() -> CommandDescriptor {
    CommandDescriptor::Compute {
        dispatch: ComputeDispatch::Both,
        max_buffer_bind_count: 0,
        max_threadgroup_memory_bind_count: 0,
        support_dynamic_attribute_stride: false,
    }
}

#[cfg(m2v_gpu)]
#[test]
fn typed_public_fixtures_create_compute_pipelines_without_submission() {
    let device = system_device().expect("Metal device");
    for (fixture, entry) in [
        (
            "kernel_source_order_product_is_not_regroupable",
            "kernel_source_order_product_is_not_regroupable",
        ),
        ("kernel_agx_emask_edge_lanes", "agx_emask_edge_lanes"),
        ("kernel_copy_word", "copy_word"),
    ] {
        let source = SourceRow {
            air_sha256: String::new(),
            stage: "Kernel".into(),
            entry: entry.into(),
            air_ll: "not native AIR".into(),
            blob_b64: Some("invalid base64: public fixture takes precedence".into()),
            lib_sha256s: vec![],
            label: format!("public/{fixture}"),
        };
        let commands = make_indirect_commands(&device, &case(compute_descriptor(), 1)).unwrap();
        let mut buffers = Vec::new();
        let (library, _) = load_producer_library(
            &device,
            &loop_free(&source),
            &commands,
            &declaring(vec![]),
            &mut buffers,
            &[],
        )
        .unwrap();
        assert_eq!(buffers.len(), 1);
        let ordinary = load_library(&device, &shader_air(&source).unwrap().bitcode, None).unwrap();
        for library in [ordinary, library] {
            let function = library
                .newFunctionWithName(&NSString::from_str(entry))
                .unwrap();
            let pipeline = compile::bounded("compute pipeline creation", || {
                device.newComputePipelineStateWithFunction_error(&function)
            })
            .unwrap();
            assert!(pipeline.maxTotalThreadsPerThreadgroup() > 0);
        }
    }
}

#[cfg(m2v_gpu)]
#[test]
fn native_command_policy_storage_compiles_and_encodes_without_submission() {
    let device = system_device().expect("Metal device");
    let source = r#"
                target datalayout = "e-p:64:64-i64:64-n8:16:32"
                target triple = "air64_v28-apple-macosx26.0.0"
                %Command = type opaque
                %Value = type { %Command addrspace(1)* }
                %Owner = type { %Value }
                declare void @air.set_barrier_compute_command(%Command addrspace(1)*, i32)
                define void @producer(%Owner addrspace(1)* %owner) {
                    %field = getelementptr %Owner, %Owner addrspace(1)* %owner, i64 0, i32 0, i32 0
                    %command = load %Command addrspace(1)*, %Command addrspace(1)* addrspace(1)* %field, align 8
                    call void @air.set_barrier_compute_command(%Command addrspace(1)* %command, i32 0)
                    ret void
                }
                !air.kernel = !{!0}
                !air.version = !{!8}
                !air.language_version = !{!9}
                !0 = !{void (%Owner addrspace(1)*)* @producer, !1, !2}
                !1 = !{}
                !2 = !{!3}
                !3 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.struct_type_info", !4, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Owner", !"air.arg_name", !"owner"}
                !4 = !{i32 0, i32 8, i32 0, !"command_buffer", !"command", !"air.indirect_argument", !5}
                !5 = !{i32 0, !"air.command_buffer", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"command_buffer", !"air.arg_name", !"command"}
                !8 = !{i32 2, i32 8, i32 0}
                !9 = !{!"Metal", i32 3, i32 1, i32 0}
                !llvm.module.flags = !{!10, !11, !12, !13, !14, !15}
                !10 = !{i32 7, !"air.max_device_buffers", i32 31}
                !11 = !{i32 7, !"air.max_constant_buffers", i32 31}
                !12 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
                !13 = !{i32 7, !"air.max_textures", i32 128}
                !14 = !{i32 7, !"air.max_read_write_textures", i32 8}
                !15 = !{i32 7, !"air.max_samplers", i32 16}
            "#;
    let metadata_only = source
        .lines()
        .filter(|line| !line.contains("@air.set_barrier_compute_command"))
        .collect::<Vec<_>>()
        .join("\n");
    let wrapped = source
                .replace("%Owner = type { %Value }", "%Inner = type { %Value }\n%Owner = type { %Inner }")
                .replace("%owner, i64 0, i32 0, i32 0", "%owner, i64 0, i32 0, i32 0, i32 0")
                .replace("!4 = !{", "!4 = !{!\"air.struct_type_info\", !16, i32 0, i32 8, i32 0, !\"Inner\", !\"inner\", !\"air.indirect_argument\", i32 0}\n!16 = !{");
    let wrapped_metadata_only = wrapped
        .lines()
        .filter(|line| !line.contains("@air.set_barrier_compute_command"))
        .collect::<Vec<_>>()
        .join("\n");
    let scalar = source
                .replace("%Owner = type { %Value }", "%Owner = type { %Value, i64 }")
                .replace("!\"air.buffer_size\", i32 8", "!\"air.buffer_size\", i32 16")
                .replace("!\"air.arg_type_size\", i32 8", "!\"air.arg_type_size\", i32 16")
                .replace("!\"air.indirect_argument\", !5}", "!\"air.indirect_argument\", !5, i32 8, i32 8, i32 0, !\"ulong\", !\"data\"}")
                .replace("                    ret void", "                    %raw = bitcast %Owner addrspace(1)* %owner to i64 addrspace(1)*\n                    %data = getelementptr i64, i64 addrspace(1)* %raw, i64 1\n                    %value = load i64, i64 addrspace(1)* %data\n                    store i64 %value, i64 addrspace(1)* %data\n                    ret void");
    let setter =
        "call void @air.set_barrier_compute_command(%Command addrspace(1)* %command, i32 0)";
    let branches = source.replace(setter, &format!(
                "br i1 true, label %left, label %right\nleft:\n  {setter}\n  br label %exit\nright:\n  {setter}\n  br label %exit\nexit:"
            ));
    let forwarded = format!(
                "{}\ndefine internal void @forward(%Command addrspace(1)* %command) {{\n  {setter}\n  ret void\n}}\n",
                source.replace(setter, "call void @forward(%Command addrspace(1)* %command)")
            );
    for (variant, source) in [
        ("direct", source),
        ("metadata_only", metadata_only.as_str()),
        ("nested", wrapped.as_str()),
        ("nested_metadata_only", wrapped_metadata_only.as_str()),
        ("scalar", scalar.as_str()),
        ("branches", branches.as_str()),
        ("forwarded", forwarded.as_str()),
    ] {
        let scratch = ScratchDir::new("native-guard-helper").unwrap();
        let original = scratch.path().join("original.ll");
        let object = scratch.path().join("original.air");
        let recovered = scratch.path().join("recovered.ll");
        fs::write(&original, source).unwrap();
        for (tool, options, input, output) in [
            ("metal-as", &[][..], &original, &object),
            (
                "metal-opt",
                &["-S", "-passes=verify"][..],
                &object,
                &recovered,
            ),
        ] {
            let args = options.iter().map(std::ffi::OsStr::new).chain([
                input.as_os_str(),
                "-o".as_ref(),
                output.as_os_str(),
            ]);
            let result = compile::xcrun(tool, args).unwrap();
            assert!(result.success, "{tool}: {}", result.stderr);
        }
        let recovered = fs::read_to_string(&recovered).unwrap();
        assert!(recovered.contains("target triple = \"air64_v28-apple-macosx26.0.0\""));
        assert!(recovered.contains("%Command = type opaque"));
        assert!(recovered.contains("%Command addrspace(1)*"));
        let case = case(compute_descriptor(), 1);
        let commands = make_indirect_commands(&device, &case).unwrap();
        let source = SourceRow {
            air_sha256: String::new(),
            stage: "Kernel".into(),
            entry: "producer".into(),
            // The native loader must use original bytes, not translation text.
            air_ll: "not native AIR".into(),
            blob_b64: Some(
                base64::engine::general_purpose::STANDARD.encode(fs::read(&object).unwrap()),
            ),
            lib_sha256s: vec![],
            label: "synthetic native producer".into(),
        };
        let mut bound = Vec::new();
        let (library, _) = load_producer_library(
            &device,
            &loop_free(&source),
            &commands,
            &declaring(vec![]),
            &mut bound,
            &[30],
        )
        .unwrap();
        assert_eq!(bound.len(), 1);
        assert_eq!(bound[0].0, 29);
        assert_eq!(
            bound[0].1.gpuAddress(),
            commands.policy.as_ref().unwrap().1.gpuAddress()
        );
        if variant == "direct" {
            // The entry's own argument at slot 28 is not free for the policy buffer.
            let owner = param(
                0,
                ParamRole::Buffer(metal2vulkan_air::abi::BufferArg {
                    location: metal2vulkan_air::abi::Slot {
                        index: metal2vulkan_air::abi::AbiInt::Const(28),
                        count: metal2vulkan_air::abi::AbiInt::Const(1),
                    },
                    access: metal2vulkan_air::abi::Access::ReadWrite,
                    space: metal2vulkan_air::ir::AddrSpace::Device,
                    buffer_size: None,
                    layout: None,
                    raster_order_group: None,
                }),
            );
            load_producer_library(
                &device,
                &loop_free(&source),
                &commands,
                &declaring(vec![owner]),
                &mut bound,
                &[30],
            )
            .unwrap();
            assert_eq!(bound[1].0, 27);
            let mut empty = case.clone();
            empty.command_buffers.clear();
            let empty = make_indirect_commands(&device, &empty).unwrap();
            assert!(load_producer_library(
                &device,
                &loop_free(&source),
                &empty,
                &declaring(vec![]),
                &mut bound,
                &[]
            )
            .err()
            .unwrap()
            .contains("policy binding"));
            assert_eq!(bound.len(), 2);
            let all_slots = (0..31).collect::<Vec<_>>();
            assert!(load_producer_library(
                &device,
                &loop_free(&source),
                &commands,
                &declaring(vec![]),
                &mut bound,
                &all_slots
            )
            .err()
            .unwrap()
            .contains("exhausted"));
            assert_eq!(bound.len(), 2);
            let mut missing = source.clone();
            missing.blob_b64 = None;
            assert!(shader_air(&missing)
                .err()
                .unwrap()
                .contains("original typed AIR"));
            assert_eq!(bound.len(), 2);
        }
        let function = library
            .newFunctionWithName(&NSString::from_str("producer"))
            .unwrap();
        let pipeline = compile::bounded("compute pipeline creation", || {
            device.newComputePipelineStateWithFunction_error(&function)
        })
        .unwrap_or_else(|error| panic!("{variant} pipeline: {error:?}"));
        assert!(pipeline.maxTotalThreadsPerThreadgroup() > 0);
        let owner_encoder = unsafe { function.newArgumentEncoderWithBufferIndex(0) };
        let owner = device
            .newBufferWithLength_options(
                owner_encoder.encodedLength(),
                MTLResourceOptions::StorageModeShared,
            )
            .unwrap();
        let (policies, policy) = commands.policy.as_ref().unwrap();
        let initial = policies.bytes();
        // No command queue or command buffer is created. The buffers remain host-owned.
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(policy.contents().as_ptr().cast::<u8>(), initial.len())
        };
        bytes.copy_from_slice(initial);
        unsafe {
            owner_encoder.setArgumentBuffer_offset(Some(&owner), 0);
            owner_encoder.setIndirectCommandBuffer_atIndex(Some(&commands.commands[0].1), 0);
        }
        policies.validate_readback(bytes).unwrap();
        bytes[20..24].copy_from_slice(&1u32.to_le_bytes());
        assert_eq!(
            policies.validate_readback(bytes),
            Err(metal2vulkan::bind::indirect_commands::CommandBufferError::EncodingFault(1))
        );
        assert_eq!(
            commands.validate_readback(),
            Err(metal2vulkan::bind::indirect_commands::CommandBufferError::EncodingFault(1))
        );
    }
}

#[cfg(m2v_gpu)]
#[test]
fn specialized_compute_pipeline_supports_indirect_commands_without_submission() {
    let device = system_device().expect("Metal device");
    let source = NSString::from_str(
        r#"
                #include <metal_stdlib>
                using namespace metal;
                constant uint value [[function_constant(0)]];
                kernel void child(device uint* output [[buffer(0)]]) {
                    output[0] = value;
                }
            "#,
    );
    let library = compile::bounded("test library compilation", || {
        device.newLibraryWithSource_options_error(&source, None)
    })
    .unwrap();
    let name = NSString::from_str("child");
    let constants = [crate::literal::LiteralFunctionConstant {
        index: 0,
        scalar_type: ScalarType::U32,
        lanes: 1,
        bytes: 7u32.to_le_bytes().to_vec(),
    }];
    let function = make_function(&library, &name, &constants).unwrap();
    let linked = load_linked_functions(
        &device,
        &crate::library_module::ResolvedLinkedFunctions::default(),
        &constants,
    )
    .unwrap();
    let none = declaring(vec![]);
    let direct = make_compute_pipeline(&device, &function, &[], &none, &linked, false).unwrap();
    let indirect = make_compute_pipeline(&device, &function, &[], &none, &linked, true).unwrap();
    assert!(direct.maxTotalThreadsPerThreadgroup() > 0);
    assert!(indirect.supportIndirectCommandBuffers());
    assert!(indirect.maxTotalThreadsPerThreadgroup() > 0);
}

#[cfg(m2v_gpu)]
#[test]
fn checked_compute_template_binds_native_pipeline_without_submission() {
    let device = system_device().expect("Metal device");
    let scratch = crate::ScratchDir::new("native-pipeline-template").unwrap();
    let ll = include_str!("../../../fixtures/public/kernel_vector_function_constant.ll");
    let source = Arc::new(SourceRow {
        air_sha256: crate::hash::sha256_bytes(ll.as_bytes()),
        air_ll: ll.into(),
        stage: "Kernel".into(),
        entry: "kernel_vector_function_constant".into(),
        blob_b64: None,
        lib_sha256s: vec![],
        label: "public/kernel_vector_function_constant.ll".into(),
    });
    let mut case = crate::case::vector_function_constant_test_case(
        source.air_sha256.clone(),
        source.entry.clone(),
    );
    case.case_id = case.computed_case_id().unwrap();
    let checked = crate::check::check_case_against_source(scratch.path(), case, &source)
        .unwrap()
        .primary;
    let handle = crate::case::indirect_commands::PipelineResource {
        buffer_binding: 0,
        field_offset: 0,
        token: 1,
        air_sha256: source.air_sha256.clone(),
        case_id: checked.case.case_id.clone(),
    };
    let template = CheckedShader {
        case: checked.case,
        source: source.clone(),
        interface: checked.interface,
        reflection: checked.reflection,
        companion: checked.companion,
        requirements: checked.requirements,
        linked_functions: checked.linked_functions,
        input_sha256: checked.input_sha256,
        loop_class: checked.loop_class,
        fuel_scope: checked.fuel_scope,
        entry_projection: checked.entry_projection,
    };
    let templates = BTreeMap::from([(
        handle.case_id.clone(),
        crate::metal::OracleTemplate {
            air: shader_air(&template.source).unwrap().bitcode,
            shader: &template,
        },
    )]);
    let pipelines = make_pipeline_templates(&device, &templates).unwrap();
    assert_eq!(pipelines.len(), 1);
    assert!(pipelines[&handle.case_id].supports_indirect_commands());
    check_native_pipeline_encoding(&device, &handle, &pipelines);
}

#[cfg(m2v_gpu)]
fn check_native_pipeline_encoding(
    device: &ProtocolObject<dyn MTLDevice>,
    handle: &crate::case::indirect_commands::PipelineResource,
    pipelines: &BTreeMap<String, NativePipeline>,
) {
    // The pipeline member is `[[id(7)]]`.
    let (stage, role) = match pipelines[&handle.case_id] {
        NativePipeline::Compute(_) => ("compute", ParamRole::ComputePipelineState(object(7))),
        NativePipeline::Render(_) => ("render", ParamRole::RenderPipelineState(object(7))),
    };
    let source = NSString::from_str(&format!(
        r#"
                #include <metal_stdlib>
                using namespace metal;
                struct Handles {{
                    {stage}_pipeline_state pipeline [[id(7)]];
                    command_buffer commands [[id(8)]];
                }};
                kernel void producer(device Handles& handles [[buffer(0)]]) {{
                    {stage}_command command(handles.commands, 0);
                    command.set_{stage}_pipeline_state(handles.pipeline);
                }}
            "#
    ));
    let library = compile::bounded("test library compilation", || {
        device.newLibraryWithSource_options_error(&source, None)
    })
    .unwrap();
    let producer = library
        .newFunctionWithName(&NSString::from_str("producer"))
        .unwrap();
    let length = unsafe { producer.newArgumentEncoderWithBufferIndex(0) }.encodedLength();
    let owner = device
        .newBufferWithLength_options(length, MTLResourceOptions::StorageModeShared)
        .unwrap();
    let owners = [(0, owner)];
    let handles = [handle.clone()];
    let fields = argument_buffer(0, &[(0, role)]);
    encode_pipeline_templates(&producer, &owners, &handles, &fields, pipelines).unwrap();
    assert!(
        encode_pipeline_templates(&producer, &owners, &handles, &fields, &BTreeMap::new())
            .unwrap_err()
            .contains("missing compiled")
    );
    assert!(
        encode_pipeline_templates(&producer, &[], &handles, &fields, pipelines)
            .unwrap_err()
            .contains("not allocated")
    );
    let not_a_pipeline = argument_buffer(0, &[(0, ParamRole::CommandBuffer(object(7)))]);
    assert!(
        encode_pipeline_templates(&producer, &owners, &handles, &not_a_pipeline, pipelines)
            .unwrap_err()
            .contains("disagrees with its AIR member role")
    );
}

#[cfg(m2v_gpu)]
#[test]
fn checked_render_templates_compile_and_bind_without_submission() {
    let device = system_device().expect("Metal device");
    let scratch = crate::ScratchDir::new("render-pipeline-templates").unwrap();
    for (name, ll, mut case) in [
        (
            "vertex_narrow_attributes",
            include_str!("../../../fixtures/public/vertex_narrow_attributes.ll"),
            crate::case::narrow_vertex_test_case(String::new(), String::new()),
        ),
        (
            "fragment_depth_stencil",
            include_str!("../../../fixtures/public/fragment_depth_stencil.ll"),
            crate::case::combined_depth_stencil_test_case(String::new(), String::new()),
        ),
    ] {
        case.air_sha256 = crate::hash::sha256_bytes(ll.as_bytes());
        case.entry = name.into();
        case.case_id = case.computed_case_id().unwrap();
        let source = Arc::new(SourceRow {
            air_sha256: case.air_sha256.clone(),
            stage: format!("{:?}", case.stage),
            entry: name.into(),
            air_ll: ll.into(),
            blob_b64: None,
            lib_sha256s: vec![],
            label: format!("public/{name}.ll"),
        });
        let checked = crate::check::check_case_against_source(scratch.path(), case, &source)
            .unwrap()
            .primary;
        if checked.case.stage == crate::case::Stage::Vertex {
            let layout = make_vertex_layout(&checked.case, &checked.interface).unwrap();
            assert_eq!(layout.slots, [(30, 0), (29, 1)]);
            let resources = LiteralResources::prepare(&checked.case).unwrap();
            let buffers = make_vertex_buffers(&device, &checked.case, &resources, &layout).unwrap();
            for ((slot, buffer, stride), literal) in buffers.iter().zip(&resources.vertex_inputs) {
                assert_eq!(*slot, 30 - literal.location as usize);
                assert_eq!(*stride, None);
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        buffer.contents().as_ptr().cast::<u8>(),
                        buffer.length(),
                    )
                };
                assert_eq!(bytes, literal.bytes);
            }
        }
        let handle = crate::case::indirect_commands::PipelineResource {
            buffer_binding: 0,
            field_offset: 0,
            token: 1,
            air_sha256: source.air_sha256.clone(),
            case_id: checked.case.case_id.clone(),
        };
        let template = CheckedShader {
            case: checked.case,
            source: source.clone(),
            interface: checked.interface,
            reflection: checked.reflection,
            companion: checked.companion,
            requirements: checked.requirements,
            linked_functions: checked.linked_functions,
            input_sha256: checked.input_sha256,
            loop_class: checked.loop_class,
            fuel_scope: checked.fuel_scope,
            entry_projection: checked.entry_projection,
        };
        let templates = BTreeMap::from([(
            handle.case_id.clone(),
            crate::metal::OracleTemplate {
                air: shader_air(&template.source).unwrap().bitcode,
                shader: &template,
            },
        )]);
        let pipelines = make_pipeline_templates(&device, &templates).unwrap();
        assert!(pipelines[&handle.case_id].fits(&ParamRole::RenderPipelineState(object(0))));
        assert!(pipelines[&handle.case_id].supports_indirect_commands());
        check_native_pipeline_encoding(&device, &handle, &pipelines);
    }
}

#[test]
fn malformed_later_descriptor_prevents_all_resource_allocations() {
    let mut resources = case(compute_descriptor(), u32::MAX).command_buffers;
    resources[0].execution_range = [0, 0];
    let mut invalid = resources[0].clone();
    invalid.field_offset = 8;
    invalid.capacity = 1;
    invalid.execution_range = [0, 2];
    resources.push(invalid);
    let mut allocations = 0;
    let result = allocate_command_resources(&resources, |resource, _| {
        allocations += 1;
        Ok(resource.capacity)
    });
    assert!(result.unwrap_err().contains("range"));
    assert_eq!(allocations, 0);
    resources[1].execution_range = [0, 1];
    let capacities =
        allocate_command_resources(&resources, |resource, _| Ok(resource.capacity)).unwrap();
    assert_eq!(capacities, [u32::MAX, 1]);
}

#[cfg(m2v_gpu)]
#[test]
fn tiny_native_allocations_preserve_authored_capacity_without_submission() {
    let device = system_device().expect("Metal device");
    for descriptor in [
        compute_descriptor(),
        CommandDescriptor::Render {
            draws: crate::case::indirect_commands::RenderDraws::default(),
            max_vertex_buffer_bind_count: 0,
            max_fragment_buffer_bind_count: 0,
            max_object_buffer_bind_count: 0,
            max_mesh_buffer_bind_count: 0,
            support_dynamic_attribute_stride: false,
        },
    ] {
        for capacity in [0, 1, 2] {
            let case = case(descriptor.clone(), capacity);
            let buffers = make_indirect_commands(&device, &case).unwrap();
            assert_eq!(buffers.commands.len(), 1);
            assert_eq!(buffers.commands[0].0, (0, 0));
            assert_eq!(buffers.commands[0].1.size(), capacity as usize);
            buffers.validate_readback().unwrap();
        }
    }
    let mut empty = case(compute_descriptor(), 1);
    empty.command_buffers.clear();
    let resources = make_indirect_commands(&device, &empty).unwrap();
    assert!(resources.commands.is_empty());
    assert!(resources.policy.is_none());
    resources.validate_readback().unwrap();
    let mut invalid = case(compute_descriptor(), 1);
    invalid.command_buffers[0].execution_range = [0, 2];
    assert!(make_indirect_commands(&device, &invalid)
        .err()
        .unwrap()
        .contains("range"));
}

#[cfg(m2v_gpu)]
#[test]
fn native_argument_encoding_checks_owner_and_uses_the_air_member_id_without_submission() {
    let device = system_device().expect("Metal device");
    let source = NSString::from_str(
        r#"
                #include <metal_stdlib>
                using namespace metal;
                struct Handles { command_buffer commands [[id(7)]]; };
                kernel void encode_handles(device Handles& handles [[buffer(0)]]) {
                    compute_command command(handles.commands, 0);
                    command.set_barrier();
                }
            "#,
    );
    let library = compile::bounded("test library compilation", || {
        device.newLibraryWithSource_options_error(&source, None)
    })
    .unwrap();
    let function = library
        .newFunctionWithName(&NSString::from_str("encode_handles"))
        .unwrap();
    let commands = make_indirect_commands(&device, &case(compute_descriptor(), 1)).unwrap();
    let fields = argument_buffer(0, &[(0, ParamRole::CommandBuffer(object(7)))]);
    let length = unsafe { function.newArgumentEncoderWithBufferIndex(0) }.encodedLength();
    assert!(length > 1);
    let owner = device
        .newBufferWithLength_options(length, MTLResourceOptions::StorageModeShared)
        .unwrap();
    let owners = [(0, owner)];
    encode_indirect_commands(&function, &owners, &commands.commands, &fields).unwrap();
    assert!(
        encode_indirect_commands(&function, &[], &commands.commands, &fields)
            .unwrap_err()
            .contains("not allocated")
    );
    let short = device
        .newBufferWithLength_options(length - 1, MTLResourceOptions::StorageModeShared)
        .unwrap();
    assert!(
        encode_indirect_commands(&function, &[(0, short)], &commands.commands, &fields)
            .unwrap_err()
            .contains("Metal requires")
    );
    let a_pipeline = argument_buffer(0, &[(0, ParamRole::RenderPipelineState(object(7)))]);
    assert!(
        encode_indirect_commands(&function, &owners, &commands.commands, &a_pipeline)
            .unwrap_err()
            .contains("disagrees with its AIR member role")
    );
}
