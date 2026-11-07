use super::*;
use crate::case::{
    BufferResource, Comparison, Dispatch, ExecutionSafety, FunctionTableEntry,
    FunctionTableResource, OutputSelection, ResourceRole, ThreadgroupMemoryResource,
};

#[test]
fn authored_entry_selects_one_root_for_inspection_and_translation() {
    let scratch = crate::ScratchDir::new("authored-entry-selection").unwrap();
    let ll = r#"
            target datalayout = "e-p:64:64:64"
            target triple = "air64-apple-macosx14.0.0"
            define void @selected() {
                ret void
            }
            define void @other() {
                ret void
            }
            !air.kernel = !{!0, !1}
            !0 = !{ptr @selected, !2, !2}
            !1 = !{ptr @other, !2, !2}
            !2 = !{}
        "#;
    assert!(crate::air::reflect::inspect_stage(ll, crate::case::Stage::Kernel).is_err());
    let source = SourceRow {
        air_sha256: crate::hash::sha256_bytes(ll.as_bytes()),
        air_ll: ll.into(),
        stage: "Kernel".into(),
        entry: "selected".into(),
        blob_b64: None,
        lib_sha256s: vec![],
        label: "synthetic entry selection".into(),
    };
    let mut case = crate::case::attachmentless_fragment_test_case(
        source.air_sha256.clone(),
        source.entry.clone(),
    );
    case.stage = Stage::Kernel;
    case.draw = None;
    case.dispatch = Some(Dispatch {
        grid: [1; 3],
        threads_per_threadgroup: [1; 3],
        stage_in_region: None,
    });
    case.case_id = case.computed_case_id().unwrap();
    let checked = check_case_against_source(scratch.path(), case, &source).unwrap();
    assert_eq!(
        checked.primary.reflection.entry_point.as_deref(),
        Some("selected")
    );
    // The AIR interface is the selected entry's: `other` is a second declaration it ignores.
    let interface =
        interface::tests::of_text(ll, "selected", &BTreeMap::new()).expect("selected entry");
    assert!(interface.params.is_empty());
    assert!(interface::tests::of_text(ll, "missing", &BTreeMap::new())
        .unwrap_err()
        .contains("no entry point"));
}

#[test]
fn omitted_function_constants_require_proof_that_no_use_survives() {
    let scratch = crate::ScratchDir::new("omitted-function-constant").unwrap();
    // The public fixture declares its constant in `!air.function_constants`, as Apple's compiler
    // emits it; the checker binds only declared constants.
    let ll = include_str!("../../fixtures/public/kernel_vector_function_constant.ll");
    let mut source = SourceRow {
        air_sha256: crate::hash::sha256_bytes(ll.as_bytes()),
        air_ll: ll.into(),
        stage: "Kernel".into(),
        entry: "kernel_vector_function_constant".into(),
        blob_b64: None,
        lib_sha256s: vec![],
        label: "synthetic constant check".into(),
    };
    let mut case = crate::case::vector_function_constant_test_case(
        source.air_sha256.clone(),
        source.entry.clone(),
    );
    case.case_id = case.computed_case_id().unwrap();
    check_case_against_source(scratch.path(), case.clone(), &source).unwrap();
    case.function_constants.clear();
    case.case_id = case.computed_case_id().unwrap();
    let errors = check_case_against_source(scratch.path(), case.clone(), &source)
        .err()
        .unwrap();
    assert!(
        errors.iter().any(|error| error
            == "function constant 0 is omitted, but a value use of it survives specialization"),
        "{errors:?}"
    );

    // Keep the declaration but replace its load with an explicit value. The absent
    // constant now has no surviving use and must not become a mandatory authored input.
    source.air_ll = source.air_ll.replace(
        "load <4 x i32>, ptr addrspace(2) @vector_value.MTL_FC_INIT_0_Dv4_j, align 16",
        "add <4 x i32> zeroinitializer, <i32 1, i32 2, i32 3, i32 4>",
    );
    source.air_sha256 = crate::hash::sha256_bytes(source.air_ll.as_bytes());
    case.air_sha256 = source.air_sha256.clone();
    case.case_id = case.computed_case_id().unwrap();
    check_case_against_source(scratch.path(), case, &source).unwrap();
}

#[test]
fn pipeline_templates_resolve_exact_identity_and_match_reflected_stage() {
    use crate::case::indirect_commands::PipelineResource;
    use interface::tests::{argument_buffer, object};
    use metal2vulkan_air::abi::ParamRole;

    let scratch = crate::ScratchDir::new("pipeline-template-check").unwrap();
    let store = crate::store::CorpusStore::new(scratch.path());
    let source = SourceRow {
        air_sha256: crate::hash::sha256_bytes(crate::case::ATTACHMENTLESS_FRAGMENT_AIR.as_bytes()),
        air_ll: crate::case::ATTACHMENTLESS_FRAGMENT_AIR.into(),
        stage: "Fragment".into(),
        entry: "fragment_no_writes".into(),
        blob_b64: None,
        lib_sha256s: vec!["aa".repeat(32)],
        label: "synthetic pipeline template".into(),
    };
    crate::source::write_source_shards(scratch.path(), [source.clone()]).unwrap();
    let mut template = crate::case::attachmentless_fragment_test_case(
        source.air_sha256.clone(),
        "fragment_no_writes".into(),
    );
    template.case_id = template.computed_case_id().unwrap();
    store.put_case(template.clone()).unwrap();
    let render = || ParamRole::RenderPipelineState(object(0));
    let compute = || ParamRole::ComputePipelineState(object(0));
    let mut producer = template.clone();
    producer.pipeline_states.push(PipelineResource {
        buffer_binding: 0,
        field_offset: 8,
        token: 1,
        air_sha256: template.air_sha256.clone(),
        case_id: template.case_id.clone(),
    });
    let check = |case: &AuthoredCase, interface: &interface::AirInterface| {
        resolve_pipeline_templates(scratch.path(), case, interface)
    };
    let checked = check(&producer, &argument_buffer(0, &[(8, render())])).unwrap();
    assert_eq!(checked.len(), 1);
    assert_eq!(*checked.values().next().unwrap().source, source);
    // A host-encoded compute command sets a compute pipeline, never a render pipeline state's
    // token.
    let mut encoding = producer.clone();
    encoding
        .command_buffers
        .push(crate::case::indirect_commands::CommandBufferResource {
            buffer_binding: 0,
            field_offset: 0,
            capacity: 1,
            execution_range: [0, 1],
            inherit_pipeline_state: false,
            inherit_buffers: false,
            descriptor: crate::case::indirect_commands::CommandDescriptor::Compute {
                dispatch: crate::case::indirect_commands::ComputeDispatch::Both,
                max_buffer_bind_count: 0,
                max_threadgroup_memory_bind_count: 0,
                support_dynamic_attribute_stride: false,
            },
            commands: vec![crate::case::indirect_commands::HostCommand {
                index: 0,
                action: None,
                barrier: false,
                pipeline: Some(1),
                buffers: vec![],
                threadgroup_memory_lengths: vec![],
                stage_in_region: None,
                vertex_buffers: vec![],
                fragment_buffers: vec![],
            }],
            execute: None,
        });
    let rendering = argument_buffer(
        0,
        &[(0, ParamRole::CommandBuffer(object(0))), (8, render())],
    );
    assert!(check(&encoding, &rendering).err().unwrap()[0]
        .contains("sets pipeline 1, a template of the other command family"));
    encoding.command_buffers[0].commands[0].pipeline = None;
    assert_eq!(check(&encoding, &rendering).unwrap().len(), 1);
    let mut duplicate = producer.pipeline_states[0].clone();
    duplicate.field_offset = 16;
    duplicate.token = 2;
    producer.pipeline_states.push(duplicate);
    let two = argument_buffer(0, &[(8, render()), (16, render())]);
    assert_eq!(check(&producer, &two).unwrap().len(), 1);
    let conflicting = argument_buffer(0, &[(8, render()), (16, compute())]);
    assert!(check(&producer, &conflicting).err().unwrap()[0].contains("conflicting"));
    let mut variant = template.clone();
    variant.name.push_str("-bounded");
    variant.execution_safety = ExecutionSafety::AuthoredBounded;
    variant.rationale = Some("The straight-line shader has no backedges.".into());
    variant.case_id = variant.computed_case_id().unwrap();
    store.put_case(variant.clone()).unwrap();
    producer.pipeline_states[1].case_id = variant.case_id;
    let shared = check(&producer, &two).unwrap();
    let sources = shared
        .values()
        .map(|shader| &shader.source)
        .collect::<Vec<_>>();
    assert_eq!(sources.len(), 2);
    assert!(Arc::ptr_eq(sources[0], sources[1]));
    producer.pipeline_states.pop();
    let as_compute = argument_buffer(0, &[(8, compute())]);
    assert!(check(&producer, &as_compute).err().unwrap()[0].contains("disagrees"));
    let not_a_pipeline = argument_buffer(0, &[(8, ParamRole::CommandBuffer(object(0)))]);
    assert!(check(&producer, &not_a_pipeline).err().unwrap()[0].contains("no pipeline-state"));
    producer.pipeline_states[0].case_id = "22".repeat(32);
    let one = argument_buffer(0, &[(8, render())]);
    assert!(check(&producer, &one).err().unwrap()[0].contains("does not exist"));

    let borrowed = check_case_against_source(scratch.path(), template.clone(), &source).unwrap();
    assert!(std::ptr::eq(borrowed.primary.source, &source));
    assert!(borrowed.pipeline_templates.is_empty());
    template.entry = "wrong_entry".into();
    template.case_id = template.computed_case_id().unwrap();
    store.put_case(template.clone()).unwrap();
    producer.pipeline_states[0].case_id = template.case_id;
    assert!(check(&producer, &one)
        .err()
        .unwrap()
        .iter()
        .any(|error| error.contains("entry mismatch")));
}

/// An executed render command runs a rasterization-disabled vertex template (the owned
/// `icb_render_replay_template`) in its own topology class, whether the command sets the template
/// or inherits it from the execution's parent; a fragment template, a draw of another class, or a
/// patch draw of a vertex template is refused, and a render command buffer the case never executes
/// constrains nothing. A mesh template (the owned `icb_mesh_template`'s `icb_mesh_marks`) that
/// does not rasterize runs on a buffer that allows mesh draws, and only there; a vertex template
/// runs on no buffer of mesh draws alone.
#[test]
fn executed_render_commands_run_rasterization_disabled_vertex_templates_of_their_class() {
    use crate::case::indirect_commands::{
        CommandBufferResource, Execution, HostAction, HostCommand, PipelineResource,
        PipelineTemplate,
    };
    use crate::case::{BufferResource, Draw, Primitive};
    use interface::tests::{argument_buffer, object};
    use metal2vulkan_air::abi::ParamRole;

    let scratch = crate::ScratchDir::new("render-replay-template-check").unwrap();
    let store = crate::store::CorpusStore::new(scratch.path());
    let row = |ll: &str, stage: &str, entry: &str| SourceRow {
        air_sha256: crate::hash::sha256_bytes(ll.as_bytes()),
        air_ll: ll.into(),
        stage: stage.into(),
        entry: entry.into(),
        blob_b64: None,
        lib_sha256s: vec!["aa".repeat(32)],
        label: "synthetic render replay template".into(),
    };
    let vertex = row(
        include_str!("../../fixtures/public/engine/icb_render_replay_template.ll"),
        "Vertex",
        "icb_render_replay_template",
    );
    let fragment = row(
        crate::case::ATTACHMENTLESS_FRAGMENT_AIR,
        "Fragment",
        "fragment_no_writes",
    );
    let tessellation = row(
        include_str!("../../fixtures/public/engine/icb_render_patch_template.ll"),
        "Vertex",
        "icb_render_patch_template",
    );
    // A module of several entries is a public fixture's row per entry, which the checker finds
    // among the public fixtures, not the scratch store's source shards.
    let mesh = row(
        include_str!("../../fixtures/public/engine/icb_mesh_template.ll"),
        "Mesh",
        "icb_mesh_marks",
    );
    crate::source::write_source_shards(
        scratch.path(),
        [vertex.clone(), fragment.clone(), tessellation.clone()],
    )
    .unwrap();
    let mut template =
        crate::case::narrow_vertex_test_case(vertex.air_sha256.clone(), vertex.entry.clone());
    template.vertex_inputs.clear();
    template.buffers = vec![
        BufferResource {
            binding: 0,
            role: ResourceRole::Input,
            bytes_b64: Some("AwAAAAUAAAA=".into()),
            initial_bytes_b64: None,
            rasterization_rate_map: None,
        },
        BufferResource {
            binding: 1,
            role: ResourceRole::Output,
            bytes_b64: None,
            initial_bytes_b64: Some(base64::engine::general_purpose::STANDARD.encode([0u8; 64])),
            rasterization_rate_map: None,
        },
    ];
    template.draw = Some(Draw {
        primitive: Primitive::Point,
        vertex_start: 0,
        vertex_count: 4,
        instance_count: 1,
        index_buffer: None,
        base_vertex: 0,
        base_instance: 0,
    });
    template.output = OutputSelection::Buffer {
        binding: 1,
        offset: 0,
        length: 64,
    };
    template.case_id = template.computed_case_id().unwrap();
    store.put_case(template.clone()).unwrap();
    let mut painting =
        crate::case::attachmentless_fragment_test_case(fragment.air_sha256.clone(), fragment.entry);
    painting.case_id = painting.computed_case_id().unwrap();
    store.put_case(painting.clone()).unwrap();

    let draw = |primitive| HostAction::DrawPrimitives {
        primitive,
        vertex_start: 0,
        vertex_count: 4,
        instance_count: 1,
        base_instance: 0,
    };
    let mut producer = painting.clone();
    producer.pipeline_states.push(PipelineResource {
        buffer_binding: 0,
        field_offset: 8,
        token: 1,
        air_sha256: template.air_sha256.clone(),
        case_id: template.case_id.clone(),
    });
    producer.command_buffers.push(CommandBufferResource {
        buffer_binding: 0,
        field_offset: 0,
        capacity: 1,
        execution_range: [0, 1],
        inherit_pipeline_state: false,
        inherit_buffers: false,
        descriptor: crate::case::indirect_commands::CommandDescriptor::Render {
            draws: crate::case::indirect_commands::RenderDraws::default(),
            max_vertex_buffer_bind_count: 2,
            max_fragment_buffer_bind_count: 0,
            max_object_buffer_bind_count: 0,
            max_mesh_buffer_bind_count: 0,
            support_dynamic_attribute_stride: false,
        },
        commands: vec![HostCommand {
            index: 0,
            action: Some(draw(Primitive::Point)),
            barrier: false,
            pipeline: Some(1),
            buffers: vec![],
            threadgroup_memory_lengths: vec![],
            stage_in_region: None,
            vertex_buffers: vec![],
            fragment_buffers: vec![],
        }],
        execute: Some(Execution {
            pipeline: None,
            buffers: vec![],
            vertex_buffers: vec![],
            fragment_buffers: vec![],
            object_buffers: vec![],
            mesh_buffers: vec![],
        }),
    });
    let interface = argument_buffer(
        0,
        &[
            (0, ParamRole::CommandBuffer(object(0))),
            (8, ParamRole::RenderPipelineState(object(0))),
        ],
    );
    let check = |case: &AuthoredCase| {
        let resolved = resolve_pipeline_templates(scratch.path(), case, &interface)?;
        check_render_replay_templates(case, &interface, &resolved).map(|()| resolved.len())
    };
    assert_eq!(check(&producer), Ok(1));
    // A point template draws points only: a triangle is no draw its pipeline defines.
    let mut other_class = producer.clone();
    other_class.command_buffers[0].commands[0].action = Some(draw(Primitive::Triangle));
    assert!(check(&other_class).unwrap_err()[0].contains("does not draw"));
    // Nor does a vertex template draw patches.
    let mut patches = producer.clone();
    patches.command_buffers[0].descriptor =
        crate::case::indirect_commands::CommandDescriptor::Render {
            draws: crate::case::indirect_commands::RenderDraws {
                primitives: false,
                indexed: false,
                patches: true,
                indexed_patches: false,
                mesh_threadgroups: false,
                mesh_threads: false,
            },
            max_vertex_buffer_bind_count: 2,
            max_fragment_buffer_bind_count: 0,
            max_object_buffer_bind_count: 0,
            max_mesh_buffer_bind_count: 0,
            support_dynamic_attribute_stride: false,
        };
    patches.command_buffers[0].commands[0].action = Some(HostAction::DrawPatches {
        control_points: 1,
        patch_start: 0,
        patch_count: 1,
        patch_index_buffer: None,
        patch_index_buffer_offset: 0,
        instance_count: 1,
        base_instance: 0,
        factor_buffer: 1,
        factor_buffer_offset: 0,
        factor_instance_stride: 0,
    });
    assert!(check(&patches).unwrap_err()[0].contains("does not draw"));
    // An inherited parent template is the class the draw runs in.
    let mut inheriting = other_class.clone();
    inheriting.command_buffers[0].inherit_pipeline_state = true;
    inheriting.command_buffers[0].commands[0].pipeline = None;
    inheriting.command_buffers[0]
        .execute
        .as_mut()
        .unwrap()
        .pipeline = Some(PipelineTemplate {
        token: 2,
        air_sha256: template.air_sha256.clone(),
        case_id: template.case_id.clone(),
    });
    assert!(check(&inheriting).unwrap_err()[0].contains("does not draw"));
    inheriting.command_buffers[0].commands[0].action = Some(draw(Primitive::Point));
    assert_eq!(check(&inheriting), Ok(1));
    // A fragment template draws into attachments the execution's pass does not author.
    let mut drawing_fragments = producer.clone();
    drawing_fragments.pipeline_states[0].air_sha256 = painting.air_sha256.clone();
    drawing_fragments.pipeline_states[0].case_id = painting.case_id.clone();
    assert!(check(&drawing_fragments).unwrap_err()[0]
        .contains("is not a rasterization-disabled vertex case"));
    // Unexecuted, the same commands select nothing a replay runs.
    drawing_fragments.command_buffers[0].execute = None;
    assert_eq!(check(&drawing_fragments), Ok(1));

    // A rasterization-disabled post-tessellation template (the owned `icb_render_patch_template`,
    // triangle patches of one control point) runs a patch draw that states its control points and
    // whose factors, 8 bytes a patch at each patch's position, lie inside the factor buffer; it
    // draws no primitives.
    let mut patch_template = crate::case::narrow_vertex_test_case(
        tessellation.air_sha256.clone(),
        tessellation.entry.clone(),
    );
    patch_template.vertex_inputs.clear();
    patch_template.draw = None;
    patch_template.buffers = vec![BufferResource {
        binding: 0,
        role: ResourceRole::Output,
        bytes_b64: None,
        initial_bytes_b64: Some(base64::engine::general_purpose::STANDARD.encode([0u8; 6400])),
        rasterization_rate_map: None,
    }];
    patch_template.output = OutputSelection::Buffer {
        binding: 0,
        offset: 0,
        length: 6400,
    };
    patch_template.tessellation = Some(crate::case::TessellationDraw {
        factors: vec![crate::case::TessellationFactors {
            edge_f16: vec![0x3c00; 3],
            inside_f16: vec![0x3c00],
        }],
        instance_count: 1,
        amplification_count: 1,
        control_points: vec![],
        patch_inputs: vec![],
        partition_mode: Default::default(),
        output_winding: Default::default(),
        cull_mode: Default::default(),
    });
    patch_template.case_id = patch_template.computed_case_id().unwrap();
    store.put_case(patch_template.clone()).unwrap();
    let patch_draw = |control_points, patch_count| HostAction::DrawPatches {
        control_points,
        patch_start: 1,
        patch_count,
        patch_index_buffer: None,
        patch_index_buffer_offset: 0,
        instance_count: 1,
        base_instance: 0,
        factor_buffer: 5,
        factor_buffer_offset: 8,
        factor_instance_stride: 0,
    };
    let mut patching = producer.clone();
    patching.pipeline_states[0].air_sha256 = patch_template.air_sha256.clone();
    patching.pipeline_states[0].case_id = patch_template.case_id.clone();
    patching.buffers.push(BufferResource {
        binding: 5,
        role: ResourceRole::Input,
        bytes_b64: Some(base64::engine::general_purpose::STANDARD.encode([0u8; 32])),
        initial_bytes_b64: None,
        rasterization_rate_map: None,
    });
    patching.command_buffers[0].descriptor =
        crate::case::indirect_commands::CommandDescriptor::Render {
            draws: crate::case::indirect_commands::RenderDraws {
                primitives: true,
                indexed: false,
                patches: true,
                indexed_patches: false,
                mesh_threadgroups: false,
                mesh_threads: false,
            },
            max_vertex_buffer_bind_count: 2,
            max_fragment_buffer_bind_count: 0,
            max_object_buffer_bind_count: 0,
            max_mesh_buffer_bind_count: 0,
            support_dynamic_attribute_stride: false,
        };
    // Patches at positions 1 and 2: their factors end at byte 8 + 8 * 3, the buffer's end.
    patching.command_buffers[0].commands[0].action = Some(patch_draw(1, 2));
    assert_eq!(check(&patching), Ok(1));
    let mut past = patching.clone();
    past.command_buffers[0].commands[0].action = Some(patch_draw(1, 3));
    assert!(check(&past).unwrap_err()[0].contains("past factor buffer 5"));
    let mut other_points = patching.clone();
    other_points.command_buffers[0].commands[0].action = Some(patch_draw(3, 2));
    assert!(check(&other_points).unwrap_err()[0].contains("does not draw"));
    let mut primitives = patching.clone();
    primitives.command_buffers[0].commands[0].action = Some(draw(Primitive::Point));
    assert!(check(&primitives).unwrap_err()[0].contains("does not draw"));

    // A mesh template without rasterization: two threadgroups of four threads, each marking its
    // word of mesh buffer 0.
    let mut mesh_template = painting.clone();
    mesh_template.name = "marks-two-threadgroups".into();
    mesh_template.air_sha256 = mesh.air_sha256.clone();
    mesh_template.entry = mesh.entry.clone();
    mesh_template.stage = Stage::Mesh;
    mesh_template.render_targets.clear();
    mesh_template.pipeline_states.clear();
    mesh_template.command_buffers.clear();
    mesh_template.draw = None;
    mesh_template.buffers = vec![BufferResource {
        binding: 0,
        role: ResourceRole::Output,
        bytes_b64: None,
        initial_bytes_b64: Some(base64::engine::general_purpose::STANDARD.encode([0xab; 48])),
        rasterization_rate_map: None,
    }];
    mesh_template.mesh_draw = Some(crate::case::MeshDraw {
        object: None,
        mesh: None,
        fragment: None,
        grid: crate::case::MeshGrid::Threadgroups([2, 1, 1]),
        threads_per_object_threadgroup: [1, 1, 1],
        threads_per_mesh_threadgroup: [4, 1, 1],
        rasterization: false,
        observation: None,
        object_launch: None,
    });
    mesh_template.output = OutputSelection::Buffer {
        binding: 0,
        offset: 0,
        length: 48,
    };
    mesh_template.case_id = mesh_template.computed_case_id().unwrap();
    store.put_case(mesh_template.clone()).unwrap();
    let mesh_draws = crate::case::indirect_commands::RenderDraws {
        primitives: false,
        indexed: false,
        patches: false,
        indexed_patches: false,
        mesh_threadgroups: true,
        mesh_threads: false,
    };
    // A kernel's mesh draws under the mesh template the execution's parent sets.
    let mut meshing = producer.clone();
    meshing.pipeline_states.clear();
    meshing.command_buffers[0].commands.clear();
    meshing.command_buffers[0].inherit_pipeline_state = true;
    meshing.command_buffers[0].descriptor =
        crate::case::indirect_commands::CommandDescriptor::Render {
            draws: mesh_draws,
            max_vertex_buffer_bind_count: 0,
            max_fragment_buffer_bind_count: 0,
            max_object_buffer_bind_count: 0,
            max_mesh_buffer_bind_count: 1,
            support_dynamic_attribute_stride: false,
        };
    meshing.command_buffers[0]
        .execute
        .as_mut()
        .unwrap()
        .pipeline = Some(PipelineTemplate {
        token: 3,
        air_sha256: mesh_template.air_sha256.clone(),
        case_id: mesh_template.case_id.clone(),
    });
    assert_eq!(check(&meshing), Ok(1));
    // A vertex template draws none of a buffer of mesh draws alone.
    let mut vertex_parent = meshing.clone();
    vertex_parent.command_buffers[0]
        .execute
        .as_mut()
        .unwrap()
        .pipeline = Some(PipelineTemplate {
        token: 3,
        air_sha256: template.air_sha256.clone(),
        case_id: template.case_id.clone(),
    });
    assert!(check(&vertex_parent).unwrap_err()[0].contains("hold none of"));
    // Nor does a mesh template draw any of a buffer of draws of primitives alone.
    let mut primitive_buffer = producer.clone();
    primitive_buffer.pipeline_states[0].air_sha256 = mesh_template.air_sha256.clone();
    primitive_buffer.pipeline_states[0].case_id = mesh_template.case_id.clone();
    primitive_buffer.command_buffers[0].commands[0].action = None;
    assert!(check(&primitive_buffer).unwrap_err()[0].contains("hold none of"));
}

#[test]
fn pipeline_template_admission_bounds_pending_dependencies_before_lookup() {
    use crate::case::indirect_commands::PipelineResource;
    use metal2vulkan_air::abi::ParamRole;

    let scratch = crate::ScratchDir::new("pipeline-template-bound").unwrap();
    let mut case = crate::case::attachmentless_fragment_test_case(
        "11".repeat(32),
        "fragment_no_writes".into(),
    );
    let mut members = Vec::new();
    for index in 0..=MAX_PIPELINE_TEMPLATES {
        let offset = index as u32 * 8;
        case.pipeline_states.push(PipelineResource {
            buffer_binding: 0,
            field_offset: offset,
            token: index as u64 + 1,
            air_sha256: "11".repeat(32),
            case_id: format!("{index:064x}"),
        });
        members.push((
            offset,
            ParamRole::RenderPipelineState(interface::tests::object(0)),
        ));
    }
    let interface = interface::tests::argument_buffer(0, &members);
    assert!(
        resolve_pipeline_templates(scratch.path(), &case, &interface)
            .err()
            .unwrap()[0]
            .contains("exceeds 256")
    );
    case.pipeline_states.pop();
    assert!(
        resolve_pipeline_templates(scratch.path(), &case, &interface)
            .err()
            .unwrap()[0]
            .contains("does not exist")
    );
}

/// A kernel case over `source` with a single 4-byte output buffer at `binding` and no
/// other resources, so a test only spells the resource it is about.
fn bare_kernel_case(source: &crate::source::SourceRow, binding: u32) -> AuthoredCase {
    AuthoredCase {
        air_sha256: source.air_sha256.clone(),
        case_id: String::new(),
        name: String::new(),
        entry: source.entry.clone(),
        stage: Stage::Kernel,
        buffers: vec![BufferResource {
            binding,
            role: ResourceRole::Output,
            bytes_b64: None,
            initial_bytes_b64: Some("q6urqw==".into()),
            rasterization_rate_map: None,
        }],
        argument_buffer_buffers: vec![],
        device_buffer_arrays: vec![],
        command_buffers: vec![],
        pipeline_states: vec![],
        threadgroup_memory: vec![],
        imageblock: None,
        fragment_imageblock: None,
        acceleration_structures: vec![],
        visible_function_references: vec![],
        visible_function_tables: vec![],
        intersection_function_tables: vec![],
        argument_buffer_intersection_function_tables: vec![],
        textures: vec![],
        texture_arrays: vec![],
        argument_buffer_textures: vec![],
        argument_buffer_samplers: vec![],
        samplers: vec![],
        render_targets: vec![],
        depth_stencil: None,
        vertex_inputs: vec![],
        vertex_observation: None,
        varying_values: Vec::new(),
        kernel_stage_inputs: vec![],
        function_constants: vec![],
        dispatch: Some(Dispatch {
            grid: [1, 1, 1],
            threads_per_threadgroup: [1, 1, 1],
            stage_in_region: None,
        }),
        draw: None,
        tessellation: None,
        mesh_draw: None,
        output: OutputSelection::Buffer {
            binding,
            offset: 0,
            length: 4,
        },
        compare: Comparison::Exact,
        execution_safety: ExecutionSafety::LoopFree,
        fuel: None,
        out_of_bounds: None,
        rationale: None,
        authored_by: Some("test".into()),
        render_pass_rate_map: None,
        raster_sample_count: None,
        public_source: None,
        no_effect: None,
        creation_bound: None,
    }
}

#[test]
fn command_buffer_fields_require_exact_coordinates_and_resource_kind() {
    use crate::case::indirect_commands::{
        CommandBufferResource, CommandDescriptor, ComputeDispatch,
    };
    use metal2vulkan_air::abi::ParamRole;
    let source = crate::source::SourceRow {
        air_sha256: "11".repeat(32),
        stage: "Kernel".into(),
        entry: "k".into(),
        air_ll: String::new(),
        blob_b64: None,
        lib_sha256s: vec![],
        label: "synthetic".into(),
    };
    let mut case = bare_kernel_case(&source, 0);
    case.buffers.clear();
    let command = || ParamRole::CommandBuffer(interface::tests::object(0));
    let command_errors = |case: &AuthoredCase, interface: &interface::AirInterface| {
        let mut errors = Vec::new();
        conformance::validate(case, interface, &mut errors);
        errors
            .into_iter()
            .filter(|error| error.contains("indirect command-buffer field"))
            .collect::<Vec<_>>()
    };
    let interface = interface::tests::argument_buffer(0, &[(8, command())]);
    assert_eq!(command_errors(&case, &interface).len(), 1);
    case.command_buffers.push(CommandBufferResource {
        buffer_binding: 0,
        field_offset: 8,
        capacity: 1,
        execution_range: [0, 1],
        inherit_pipeline_state: false,
        inherit_buffers: false,
        descriptor: CommandDescriptor::Compute {
            dispatch: ComputeDispatch::Both,
            max_buffer_bind_count: 0,
            max_threadgroup_memory_bind_count: 0,
            support_dynamic_attribute_stride: false,
        },
        commands: vec![],
        execute: None,
    });
    assert!(command_errors(&case, &interface).is_empty());
    case.command_buffers[0].field_offset = 16;
    assert_eq!(command_errors(&case, &interface).len(), 2);
    case.command_buffers[0].field_offset = 8;
    let buffer_member = interface::tests::argument_buffer(
        0,
        &[(
            8,
            ParamRole::Buffer(metal2vulkan_air::abi::BufferArg {
                location: metal2vulkan_air::abi::Slot {
                    index: metal2vulkan_air::abi::AbiInt::Const(0),
                    count: metal2vulkan_air::abi::AbiInt::Const(1),
                },
                access: metal2vulkan_air::abi::Access::Read,
                space: metal2vulkan_air::ir::AddrSpace::Device,
                buffer_size: None,
                layout: None,
                raster_order_group: None,
            }),
        )],
    );
    assert_eq!(command_errors(&case, &buffer_member).len(), 1);
}

#[test]
fn the_live_argument_buffer_owns_the_members_at_a_shared_slot() {
    use crate::case::ArgumentBufferBufferResource;
    use metal2vulkan_air::abi::{AbiInt, Access, BufferArg, ParamRole, Slot, StructRef};
    let source = crate::source::SourceRow {
        air_sha256: "11".repeat(32),
        stage: "Kernel".into(),
        entry: "k".into(),
        air_ll: String::new(),
        blob_b64: None,
        lib_sha256s: vec![],
        label: "synthetic".into(),
    };
    let mut case = bare_kernel_case(&source, 0);
    case.buffers.clear();
    case.argument_buffer_buffers
        .push(ArgumentBufferBufferResource {
            buffer_binding: 0,
            field_offset: 8,
            role: ResourceRole::Input,
            bytes_b64: None,
            initial_bytes_b64: None,
            rasterization_rate_map: None,
        });
    let buffer = ParamRole::Buffer(BufferArg {
        location: Slot {
            index: AbiInt::Const(0),
            count: AbiInt::Const(1),
        },
        access: Access::Read,
        space: metal2vulkan_air::ir::AddrSpace::Device,
        buffer_size: None,
        layout: None,
        raster_order_group: None,
    });
    // Two argument buffers at slot 0 whose layouts disagree at offset 8: the parameter the
    // constants compile in decides, whichever order the entry declares them in.
    let command = interface::tests::argument_buffer(
        0,
        &[(8, ParamRole::CommandBuffer(interface::tests::object(0)))],
    );
    let live = interface::tests::argument_buffer(0, &[(8, buffer)]);
    let shared = |live_first: bool| {
        let mut dead = command.params[0].clone();
        dead.live = false;
        let mut present = live.params[0].clone();
        present.binding.param = 1;
        let ParamRole::ArgumentBuffer(argument) = &mut present.binding.role else {
            unreachable!("an argument buffer");
        };
        argument.layout = Some(StructRef(1));
        let params = if live_first {
            vec![present, dead]
        } else {
            vec![dead, present]
        };
        interface::AirInterface::synthetic(
            command.stage,
            params,
            Vec::new(),
            vec![command.structs[0].clone(), live.structs[0].clone()],
        )
    };
    for live_first in [true, false] {
        let mut errors = Vec::new();
        conformance::validate(&case, &shared(live_first), &mut errors);
        assert!(
            !errors
                .iter()
                .any(|error| error.contains("argument-buffer buffer")
                    || error.contains("indirect command-buffer field")),
            "{errors:?}"
        );
    }
}

#[test]
fn a_gated_off_argument_buffer_binds_nothing_beside_the_live_one() {
    use crate::case::ArgumentBufferBufferResource;
    use metal2vulkan_air::abi::{AbiInt, Access, BufferArg, ParamRole, Slot, StructRef};
    let source = crate::source::SourceRow {
        air_sha256: "11".repeat(32),
        stage: "Kernel".into(),
        entry: "k".into(),
        air_ll: String::new(),
        blob_b64: None,
        lib_sha256s: vec![],
        label: "synthetic".into(),
    };
    let buffer = ParamRole::Buffer(BufferArg {
        location: Slot {
            index: AbiInt::Const(0),
            count: AbiInt::Const(1),
        },
        access: Access::Read,
        space: metal2vulkan_air::ir::AddrSpace::Device,
        buffer_size: None,
        layout: None,
        raster_order_group: None,
    });
    // Both alternatives at slot 0 hold a buffer with [[id(0)]]: the live one at offset 0, the
    // gated-off one at offset 8.
    let at_zero = interface::tests::argument_buffer(0, &[(0, buffer.clone())]);
    let at_eight = interface::tests::argument_buffer(0, &[(8, buffer)]);
    let alternatives = |live: bool| {
        let first = at_zero.params[0].clone();
        let mut second = at_eight.params[0].clone();
        second.binding.param = 1;
        second.live = false;
        let ParamRole::ArgumentBuffer(argument) = &mut second.binding.role else {
            unreachable!("an argument buffer");
        };
        argument.layout = Some(StructRef(1));
        let mut first = first;
        first.live = live;
        interface::AirInterface::synthetic(
            at_zero.stage,
            vec![first, second],
            Vec::new(),
            vec![at_zero.structs[0].clone(), at_eight.structs[0].clone()],
        )
    };
    let mut case = bare_kernel_case(&source, 0);
    case.buffers.clear();
    case.argument_buffer_buffers
        .push(ArgumentBufferBufferResource {
            buffer_binding: 0,
            field_offset: 8,
            role: ResourceRole::Input,
            bytes_b64: None,
            initial_bytes_b64: None,
            rasterization_rate_map: None,
        });
    let member_errors = |interface: &interface::AirInterface| {
        let mut errors = Vec::new();
        conformance::validate(&case, interface, &mut errors);
        errors
            .into_iter()
            .filter(|error| error.contains("argument-buffer buffer"))
            .collect::<Vec<_>>()
    };

    // Beside the live layout, offset 8 is nothing the encoder writes: authoring it is refused,
    // and the oracle finds no member to encode there.
    let live = alternatives(true);
    assert!(live.member_at(0, 8).is_none());
    let (param, member) = live.member_at(0, 0).expect("the live member");
    assert_eq!((param.binding.param, member.id), (0, Some(0)));
    let errors = member_errors(&live);
    assert_eq!(errors.len(), 2, "{errors:?}");
    assert!(errors
        .iter()
        .any(|error| error.contains("manifest declares argument-buffer buffer (0, 8)")));

    // With neither alternative live, either layout may be authored.
    let neither = alternatives(false);
    let (param, _) = neither.member_at(0, 8).expect("the gated-off member");
    assert_eq!(param.binding.param, 1);
    assert!(member_errors(&neither).is_empty());
}

/// An argument buffer holding `array<texture2d<float>, 2>`, whose element `index` a kernel reads
/// (a constant, or the dynamic `%i`).
fn indexed_argument_buffer_texture(index: &str) -> String {
    format!(
        r#"
target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"
define void @k(ptr addrspace(2) %args, ptr addrspace(1) %out, i32 %i) {{
  %slot = getelementptr [2 x ptr addrspace(1)], ptr addrspace(2) %args, i64 0, i32 {index}
  %tex = load ptr addrspace(1), ptr addrspace(2) %slot, align 8
  %s = tail call ptr addrspace(2) @air.get_read_sampler()
  %r = tail call {{ <4 x float>, i8 }} @air.read_texture_2d.v4f32(ptr addrspace(1) %tex, ptr addrspace(2) %s, <2 x i32> zeroinitializer, <2 x i32> zeroinitializer, i32 0, i32 1)
  %v = extractvalue {{ <4 x float>, i8 }} %r, 0
  store <4 x float> %v, ptr addrspace(1) %out, align 16
  ret void
}}
declare ptr addrspace(2) @air.get_read_sampler()
declare {{ <4 x float>, i8 }} @air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)
!air.kernel = !{{!0}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !5, !6}}
!3 = !{{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 16, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Args", !"air.arg_name", !"args"}}
!4 = !{{i32 0, i32 16, i32 0, !"array<texture2d<float>, 2>", !"t", !"air.indirect_argument", !7}}
!7 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 2, !"air.sample", !"air.arg_type_name", !"array<texture2d<float>, 2>", !"air.arg_name", !"t"}}
!5 = !{{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}}
!6 = !{{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"i"}}
"#
    )
}

#[test]
fn an_argument_buffer_texture_is_checked_against_its_dynamically_indexed_uses() {
    use crate::case::{ArgumentBufferTextureResource, TextureFormat, TextureType};
    let source = crate::source::SourceRow {
        air_sha256: "22".repeat(32),
        stage: "Kernel".into(),
        entry: "k".into(),
        air_ll: String::new(),
        blob_b64: None,
        lib_sha256s: vec![],
        label: "synthetic".into(),
    };
    let authored = |texture_type: TextureType| {
        let mut case = bare_kernel_case(&source, 1);
        for offset in [0, 8] {
            case.argument_buffer_textures
                .push(ArgumentBufferTextureResource {
                    buffer_binding: 0,
                    field_offset: offset,
                    role: ResourceRole::Input,
                    texture_type,
                    format: TextureFormat::Rgba32Float,
                    dimensions: [1, 1, 1],
                    sample_count: 1,
                    bytes_b64: None,
                    initial_bytes_b64: None,
                });
        }
        case
    };
    let texture_errors = |case: &AuthoredCase, index: &str| {
        let facts = interface::tests::of_text(
            &indexed_argument_buffer_texture(index),
            "k",
            &BTreeMap::new(),
        )
        .unwrap();
        let mut errors = Vec::new();
        conformance::validate(case, &facts, &mut errors);
        errors
            .into_iter()
            .filter(|error| error.contains("argument-buffer texture"))
            .collect::<Vec<_>>()
    };
    let three_d = authored(TextureType::D3);
    // A constant index checks the element it reads (offset 8).
    assert_eq!(texture_errors(&three_d, "1").len(), 1);
    // A dynamic index may read either element, so both are checked against the 2D read.
    let errors = texture_errors(&three_d, "%i");
    assert_eq!(errors.len(), 2, "{errors:?}");
    assert!(errors
        .iter()
        .all(|error| error.contains("requires texture type D2, got D3")));
    let two_d = authored(TextureType::D2);
    assert!(texture_errors(&two_d, "1").is_empty());
    assert!(texture_errors(&two_d, "%i").is_empty());
}

/// A kernel sampling the texture of an argument buffer `{ texture2d<float> t; sampler s; }`
/// through the sampler beside it.
const ARGUMENT_BUFFER_SAMPLER: &str = r#"
target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"
%struct.Material = type { %t, %s }
%t = type { ptr addrspace(1) }
%s = type { ptr addrspace(2) }
define void @k(ptr addrspace(2) %m, ptr addrspace(1) %out) {
  %sf = getelementptr inbounds %struct.Material, ptr addrspace(2) %m, i64 0, i32 1, i32 0
  %smp = load ptr addrspace(2), ptr addrspace(2) %sf, align 8
  %tf = getelementptr inbounds %struct.Material, ptr addrspace(2) %m, i64 0, i32 0, i32 0
  %tex = load ptr addrspace(1), ptr addrspace(2) %tf, align 8
  %r = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %tex, ptr addrspace(2) %smp, <2 x float> zeroinitializer, i1 true, <2 x i32> zeroinitializer, i1 true, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v = extractvalue { <4 x float>, i8 } %r, 0
  store <4 x float> %v, ptr addrspace(1) %out, align 16
  ret void
}
declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !7}
!3 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 16, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Material", !"air.arg_name", !"m"}
!4 = !{i32 0, i32 8, i32 0, !"texture2d<float, sample>", !"t", !"air.indirect_argument", !5, i32 8, i32 8, i32 0, !"sampler", !"s", !"air.indirect_argument", !6}
!5 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}
!6 = !{i32 1, !"air.sampler", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}
!7 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
"#;

/// A live argument buffer's sampler member is authored like its other members. Metal reads whatever
/// the field holds, so leaving it out (a null sampler) is refused, and a sampler authored where no
/// sampler member is (here the texture's field) is refused too.
#[test]
fn an_argument_buffer_sampler_member_is_authored() {
    use crate::case::{
        ArgumentBufferSamplerResource, ArgumentBufferTextureResource, SamplerAddressMode,
        SamplerFilter, SamplerMipFilter, TextureFormat, TextureType,
    };
    let source = crate::source::SourceRow {
        air_sha256: "33".repeat(32),
        stage: "Kernel".into(),
        entry: "k".into(),
        air_ll: String::new(),
        blob_b64: None,
        lib_sha256s: vec![],
        label: "synthetic".into(),
    };
    let facts = interface::tests::of_text(ARGUMENT_BUFFER_SAMPLER, "k", &BTreeMap::new()).unwrap();
    let (_, member) = facts.member_at(0, 8).expect("the sampler member");
    assert!(matches!(
        member.role,
        metal2vulkan_air::abi::ParamRole::Sampler { .. }
    ));
    let mut case = bare_kernel_case(&source, 1);
    case.argument_buffer_textures
        .push(ArgumentBufferTextureResource {
            buffer_binding: 0,
            field_offset: 0,
            role: ResourceRole::Input,
            texture_type: TextureType::D2,
            format: TextureFormat::Rgba32Float,
            dimensions: [1, 1, 1],
            sample_count: 1,
            bytes_b64: None,
            initial_bytes_b64: None,
        });
    let sampler_errors = |case: &AuthoredCase| {
        let mut errors = Vec::new();
        conformance::validate(case, &facts, &mut errors);
        errors
            .into_iter()
            .filter(|error| error.contains("argument-buffer sampler"))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        sampler_errors(&case),
        ["AIR requires argument-buffer sampler (0, 8), but the manifest does not declare it"]
    );
    let sampler = |field_offset| ArgumentBufferSamplerResource {
        buffer_binding: 0,
        field_offset,
        address_mode: SamplerAddressMode::ClampToEdge,
        min_filter: SamplerFilter::Nearest,
        mag_filter: SamplerFilter::Nearest,
        mip_filter: SamplerMipFilter::NotMipmapped,
        normalized_coordinates: true,
        compare_function: None,
    };
    let mut authored = case.clone();
    authored.argument_buffer_samplers.push(sampler(8));
    assert!(sampler_errors(&authored).is_empty());
    authored.argument_buffer_samplers.push(sampler(0));
    assert_eq!(
        sampler_errors(&authored),
        ["manifest declares argument-buffer sampler (0, 0), but the AIR entry binds none there"]
    );
}

/// Whether `case`'s selected output is writable in `ll`, by the AIR interface and the imageblock
/// families the checker reads from the declared reflection.
fn output_is_writable(ll: &str, case: &AuthoredCase) -> bool {
    let interface = interface::tests::of_text(ll, &case.entry, &BTreeMap::new()).unwrap();
    let reflection = reflect(ll, case).unwrap();
    let imageblock = validate_reflected_families(case, &reflection, &mut Vec::new());
    conformance::selected_output_is_writable(case, &interface, &imageblock)
}

#[test]
fn selected_identity_output_requires_a_reflected_write() {
    let ll = include_str!("../../fixtures/public/kernel_implicit_imageblock_half2.ll");
    let case = crate::case::narrow_implicit_imageblock_test_case(
        crate::hash::sha256_bytes(ll.as_bytes()),
        "kernel_implicit_imageblock_half2".into(),
    );
    assert!(output_is_writable(ll, &case));

    let read_only_ll = ll.replace(
            "  call void @air.store.implicit_imageblock.v2f16(<2 x half> %value, i32 0, <2 x i16> %position, i32 0, i16 0)\n",
            "",
        );
    assert!(!output_is_writable(&read_only_ll, &case));
}

/// A tile case's render targets are its pass's render area, the dispatch grid's width and height in
/// one plane, whatever its tile size: a grid of several tiles is authorable, and an attachment of
/// another size, or a grid of more than one plane, is not.
#[test]
fn implicit_imageblock_attachments_are_the_render_area() {
    let ll = include_str!("../../fixtures/public/kernel_implicit_imageblock_half2.ll");
    let tile = crate::case::narrow_implicit_imageblock_test_case(
        crate::hash::sha256_bytes(ll.as_bytes()),
        "kernel_implicit_imageblock_half2".into(),
    );
    let reflection = reflect(ll, &tile).unwrap();
    let area_errors = |case: &AuthoredCase| {
        let mut errors = Vec::new();
        validate_reflected_families(case, &reflection, &mut errors);
        errors
            .into_iter()
            .filter(|error| error.contains("the render area"))
            .collect::<Vec<_>>()
    };
    assert_eq!(area_errors(&tile), Vec::<String>::new());
    let with = |grid: [u32; 3], target: [u32; 2]| {
        let mut case = tile.clone();
        case.dispatch.as_mut().unwrap().grid = grid;
        case.render_targets[0].dimensions = target;
        case
    };
    // Three by two tiles of 16x16, the last ones partial.
    assert_eq!(
        area_errors(&with([40, 24, 1], [40, 24])),
        Vec::<String>::new()
    );
    assert_eq!(area_errors(&with([40, 24, 1], [16, 16])).len(), 1);
    assert_eq!(area_errors(&with([16, 16, 2], [16, 16])).len(), 1);
}

#[test]
fn exact_empty_output_requires_no_write() {
    let ll = r#"
define void @no_output() {
entry:
  ret void
}
!air.kernel = !{!0}
!0 = !{ptr @no_output, !1, !1}
!1 = !{}
"#;
    let mut case: AuthoredCase = serde_json::from_value(serde_json::json!({
        "air_sha256": crate::hash::sha256_bytes(ll.as_bytes()),
        "case_id": "test-no-output",
        "name": "no-output",
        "entry": "no_output",
        "stage": "kernel",
        "dispatch": {"grid": [1, 1, 1], "threads_per_threadgroup": [1, 1, 1]},
        "output": {"kind": "none"},
        "compare": {"kind": "exact"},
        "execution_safety": "loop_free"
    }))
    .unwrap();
    case.case_id = case.computed_case_id().unwrap();
    case.validate_literal_resources().unwrap();
    let no_writes = |ll: &str, case: &AuthoredCase| {
        let interface = interface::tests::of_text(ll, &case.entry, &BTreeMap::new()).unwrap();
        conformance::selected_output_is_writable(case, &interface, &Default::default())
    };
    assert!(no_writes(ll, &case));

    // A vertex entry's live stage outputs are writes the empty selection would leave unobserved.
    let vertex_ll = include_str!("../../fixtures/public/vertex_varying_triangle.ll");
    let mut vertex_case = case.clone();
    vertex_case.stage = Stage::Vertex;
    vertex_case.entry = "vertex_varying_triangle".into();
    assert!(!no_writes(vertex_ll, &vertex_case));

    // A buffer the kernel writes is a write, whatever the selection.
    let writable_ll = include_str!("../../fixtures/public/kernel_vector_function_constant.ll");
    let mut writable_case = crate::case::vector_function_constant_test_case(
        crate::hash::sha256_bytes(writable_ll.as_bytes()),
        "kernel_vector_function_constant".into(),
    );
    writable_case.output = OutputSelection::None;
    assert!(!no_writes(writable_ll, &writable_case));
}

#[test]
fn threadgroup_memory_is_authored_and_reflected_without_a_descriptor() {
    let source = crate::source::public_sources()
        .unwrap()
        .into_iter()
        .find(|source| source.entry == "threadgroup_word")
        .unwrap();
    let build = |length: u32| {
        let mut case = bare_kernel_case(&source, 1);
        case.name = "threadgroup-word".into();
        case.threadgroup_memory = vec![ThreadgroupMemoryResource { binding: 0, length }];
        case.case_id = case.computed_case_id().unwrap();
        case
    };
    let checked = check_case(&crate::source::corpus_root(), build(16)).unwrap();
    let (param, size) = checked
        .primary
        .interface
        .params_of(|role| match role {
            metal2vulkan_air::abi::ParamRole::Threadgroup(buffer) => Some(buffer),
            _ => None,
        })
        .map(|(param, buffer)| (param, conformance::threadgroup_size(param, buffer)))
        .next()
        .unwrap();
    assert_eq!(param.slots(), Some((0, 1)));
    assert_eq!(size, Some(4));

    // The AIR element size is four bytes, but Metal allocates threadgroup memory in
    // sixteen-byte units and asserts on anything else, so four is not a length this case can
    // ask for -- see `METAL_THREADGROUP_MEMORY_GRANULARITY`.
    let Err(errors) = check_case(&crate::source::corpus_root(), build(4)) else {
        panic!("Metal cannot allocate four bytes of threadgroup memory");
    };
    assert!(
        errors
            .iter()
            .any(|error| error.contains("must be a multiple of 16 bytes")),
        "{errors:?}"
    );

    // Declaring nothing at all would satisfy the reflected requirement without allocating.
    let Err(errors) = check_case(&crate::source::corpus_root(), build(0)) else {
        panic!("a zero-byte declaration allocates nothing");
    };
    assert!(
        errors.iter().any(|error| error.contains("zero bytes")),
        "{errors:?}"
    );
}

#[test]
fn a_fixed_texture_array_is_checked_at_the_length_it_declares() {
    let source = crate::source::public_sources()
        .unwrap()
        .into_iter()
        .find(|source| source.entry == "texture_array_width")
        .unwrap();
    let build = |elements: usize| {
        let mut case = bare_kernel_case(&source, 0);
        case.name = "fixed-texture-array-width".into();
        case.texture_arrays = vec![crate::case::TextureArrayResource {
            binding: 0,
            role: ResourceRole::Input,
            texture_type: crate::case::TextureType::D2,
            format: crate::case::TextureFormat::R32Float,
            sample_count: 1,
            overrides_texture_at_base: false,
            elements: (0..elements)
                .map(|_| crate::case::TextureArrayElement {
                    dimensions: [1, 1, 1],
                    bytes_b64: Some("AAAAAA==".into()),
                    initial_bytes_b64: None,
                })
                .collect(),
        }];
        case.case_id = case.computed_case_id().unwrap();
        case
    };

    // `array<texture2d<float, sample>, 2>` occupies exactly the two descriptors it
    // declares, so a case authoring both elements is complete.
    let checked = check_case(&crate::source::corpus_root(), build(2)).unwrap();
    let table = checked
        .primary
        .interface
        .params
        .iter()
        .find(|param| param.table)
        .unwrap();
    assert_eq!(table.slots(), Some((0, 2)));

    // Authoring past the declared length has nowhere to bind the third element.
    let Err(errors) = check_case(&crate::source::corpus_root(), build(3)) else {
        panic!("a third element has nowhere to bind");
    };
    assert!(
        errors.iter().any(|error| error
            == "texture-array binding 0 has 3 authored elements, but AIR alternatives declare [2]"),
        "{errors:?}"
    );
}

#[test]
fn function_table_entry_resolves_exact_cross_library_module_and_symbol() {
    let scratch = crate::ScratchDir::new("function-table-check").unwrap();
    let air_ll = r#"
define void @main(ptr addrspace(1) %output, ptr addrspace(1) %table) {
entry:
  store i32 42, ptr addrspace(1) %output, align 4
  ret void
}
!air.kernel = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint"}
!4 = !{i32 1, !"air.visible_function_table", !"air.location_index", i32 3, i32 1, !"air.read", !"air.arg_type_name", !"visible_function_table"}
"#;
    let entry_library = "11".repeat(32);
    let callback_library = "22".repeat(32);
    let source = SourceRow {
        air_sha256: crate::hash::sha256_bytes(air_ll.as_bytes()),
        stage: "Kernel".into(),
        entry: "main".into(),
        air_ll: air_ll.into(),
        blob_b64: None,
        lib_sha256s: vec![entry_library],
        label: "local/function-table-entry.ll".into(),
    };
    crate::source::write_source_shards(scratch.path(), [source.clone()]).unwrap();
    let module_ll = "define <4 x float> @shade(ptr %payload) { ret <4 x float> zeroinitializer }";
    let module_sha256 = crate::hash::sha256_bytes(module_ll.as_bytes());
    // The loop-free classifier decodes the retained blob, so it must be the module's bitcode.
    let module_bitcode = match crate::check::loop_free::assemble(module_ll) {
        Ok(bitcode) => bitcode,
        Err(why) => {
            eprintln!(
                "SKIP function_table_entry_resolves_exact_cross_library_module_and_symbol: {why}"
            );
            return;
        }
    };
    crate::library_module::merge_library_module_shards(
        scratch.path(),
        [crate::library_module::LibraryModuleRow {
            module_sha256: module_sha256.clone(),
            air_ll: module_ll.into(),
            blob_b64: base64::engine::general_purpose::STANDARD.encode(module_bitcode),
            lib_sha256s: vec![callback_library.clone()],
            label: "local/library-module/shade.ll".into(),
        }],
    )
    .unwrap();
    let mut case = AuthoredCase {
        air_sha256: source.air_sha256,
        case_id: String::new(),
        name: "function-table".into(),
        entry: source.entry,
        stage: Stage::Kernel,
        buffers: vec![BufferResource {
            binding: 0,
            role: ResourceRole::Output,
            bytes_b64: None,
            initial_bytes_b64: Some("q6urqw==".into()),
            rasterization_rate_map: None,
        }],
        argument_buffer_buffers: vec![],
        device_buffer_arrays: vec![],
        command_buffers: vec![],
        pipeline_states: vec![],
        threadgroup_memory: vec![],
        imageblock: None,
        fragment_imageblock: None,
        acceleration_structures: vec![],
        visible_function_references: vec![],
        visible_function_tables: vec![FunctionTableResource {
            binding: 3,
            size: 1,
            entries: vec![FunctionTableEntry {
                index: 0,
                module_sha256,
                function: "shade".into(),
            }],
        }],
        intersection_function_tables: vec![],
        argument_buffer_intersection_function_tables: vec![],
        textures: vec![],
        texture_arrays: vec![],
        argument_buffer_textures: vec![],
        argument_buffer_samplers: vec![],
        samplers: vec![],
        render_targets: vec![],
        depth_stencil: None,
        vertex_inputs: vec![],
        vertex_observation: None,
        varying_values: Vec::new(),
        kernel_stage_inputs: vec![],
        function_constants: vec![],
        dispatch: Some(Dispatch {
            grid: [1, 1, 1],
            threads_per_threadgroup: [1, 1, 1],
            stage_in_region: None,
        }),
        draw: None,
        tessellation: None,
        mesh_draw: None,
        output: OutputSelection::Buffer {
            binding: 0,
            offset: 0,
            length: 4,
        },
        compare: Comparison::Exact,
        execution_safety: ExecutionSafety::LoopFree,
        fuel: None,
        out_of_bounds: None,
        rationale: None,
        authored_by: Some("test".into()),
        render_pass_rate_map: None,
        raster_sample_count: None,
        public_source: None,
        no_effect: None,
        creation_bound: None,
    };
    case.case_id = case.computed_case_id().unwrap();
    let checked = check_case(scratch.path(), case).unwrap();
    assert_eq!(checked.primary.linked_functions.visible.len(), 1);
    assert_eq!(checked.primary.linked_functions.visible[0].binding, 3);
    assert_eq!(
        checked.primary.linked_functions.visible[0].entries[0].function,
        "shade"
    );
    assert_eq!(
        checked.primary.linked_functions.visible[0].entries[0]
            .module
            .lib_sha256s,
        vec![callback_library]
    );
    assert!(checked.primary.reflection.bindings.iter().any(|binding| {
        binding.kind == crate::air::reflect::ResourceKind::VisibleFunctionTable
            && binding.metal_index == 3
            && binding.descriptor.is_none()
    }));
}

#[test]
fn every_required_slot_is_authored_and_every_authored_slot_is_declared() {
    // `threadgroup_word` binds threadgroup memory at 0 and a written device buffer at 1.
    let source = crate::source::public_sources()
        .unwrap()
        .into_iter()
        .find(|source| source.entry == "threadgroup_word")
        .unwrap();
    let interface =
        interface::tests::of_text(&source.air_ll, &source.entry, &BTreeMap::new()).unwrap();
    let errors = |case: &AuthoredCase| {
        let mut errors = Vec::new();
        conformance::validate(case, &interface, &mut errors);
        errors
    };
    let mut case = bare_kernel_case(&source, 1);
    case.threadgroup_memory = vec![ThreadgroupMemoryResource {
        binding: 0,
        length: 16,
    }];
    assert_eq!(errors(&case), Vec::<String>::new());

    let mut missing = case.clone();
    missing.threadgroup_memory.clear();
    assert_eq!(
        errors(&missing),
        ["AIR requires threadgroup memory 0, but the manifest does not declare it"]
    );
    let mut undeclared = case.clone();
    undeclared.buffers[0].binding = 5;
    assert_eq!(
        errors(&undeclared),
        [
            "AIR requires buffer 1, but the manifest does not declare it",
            "manifest declares buffer 5, but the AIR entry binds none there",
        ]
    );
    let mut sampler = case.clone();
    sampler.samplers.push(crate::case::SamplerResource {
        binding: 0,
        address_mode: crate::case::SamplerAddressMode::ClampToEdge,
        min_filter: crate::case::SamplerFilter::Nearest,
        mag_filter: crate::case::SamplerFilter::Nearest,
        mip_filter: crate::case::SamplerMipFilter::NotMipmapped,
        normalized_coordinates: true,
    });
    assert_eq!(
        errors(&sampler),
        ["manifest declares sampler 0, but the AIR entry binds none there"]
    );
}
