use super::*;

#[test]
fn command_buffer_literals_participate_in_identity_and_validate_resources() {
    use crate::case::indirect_commands::{
        CommandBufferResource, CommandDescriptor, ComputeDispatch,
    };
    let mut case = example();
    let original = case.computed_case_id().unwrap();
    let resource = CommandBufferResource {
        buffer_binding: case.buffers[0].binding,
        field_offset: 0,
        capacity: 2,
        execution_range: [0, 2],
        inherit_pipeline_state: false,
        inherit_buffers: false,
        descriptor: CommandDescriptor::Compute {
            dispatch: ComputeDispatch::Both,
            max_buffer_bind_count: 1,
            max_threadgroup_memory_bind_count: 0,
            support_dynamic_attribute_stride: false,
        },
        commands: vec![],
        execute: None,
    };
    assert!(serde_json::to_value(case.semantic())
        .unwrap()
        .get("command_buffers")
        .is_none());
    case.command_buffers.push(resource.clone());
    assert_ne!(case.computed_case_id().unwrap(), original);
    let mut other = resource.clone();
    other.field_offset = 8;
    case.command_buffers.push(other);
    let ordered_id = case.computed_case_id().unwrap();
    let ordered_input = case.computed_input_sha256().unwrap();
    case.command_buffers.reverse();
    assert_eq!(case.computed_case_id().unwrap(), ordered_id);
    assert_eq!(case.computed_input_sha256().unwrap(), ordered_input);
    case.command_buffers.reverse();
    case.command_buffers.pop();
    for changed in 0..5 {
        let mut variant = case.clone();
        let command = &mut variant.command_buffers[0];
        match changed {
            0 => command.capacity += 1,
            1 => command.execution_range = [1, 2],
            2 => command.inherit_pipeline_state = true,
            3 => command.inherit_buffers = true,
            _ => {
                if let CommandDescriptor::Compute { dispatch, .. } = &mut command.descriptor {
                    *dispatch = ComputeDispatch::Threads;
                }
            }
        }
        assert_ne!(
            variant.computed_case_id().unwrap(),
            case.computed_case_id().unwrap()
        );
        assert_ne!(
            variant.computed_input_sha256().unwrap(),
            case.computed_input_sha256().unwrap()
        );
    }
    let input = case.computed_input_sha256().unwrap();
    case.command_buffers[0].execution_range = [1, 2];
    assert_ne!(case.computed_input_sha256().unwrap(), input);
    case.command_buffers.push(resource);
    assert!(case
        .validate_literal_resources()
        .unwrap_err()
        .iter()
        .any(|error| error.contains("duplicate indirect command-buffer")));
    case.command_buffers.pop();
    case.argument_buffer_buffers
        .push(ArgumentBufferBufferResource {
            buffer_binding: case.command_buffers[0].buffer_binding,
            field_offset: case.command_buffers[0].field_offset,
            role: ResourceRole::Input,
            bytes_b64: Some("AAAAAA==".into()),
            initial_bytes_b64: None,
            rasterization_rate_map: None,
        });
    assert!(case
        .validate_literal_resources()
        .unwrap_err()
        .iter()
        .any(|error| error.contains("duplicate indirect command-buffer")));
    case.argument_buffer_buffers.pop();
    case.command_buffers[0].buffer_binding = u32::MAX;
    assert!(case
        .validate_literal_resources()
        .unwrap_err()
        .iter()
        .any(|error| error.contains("owns an undeclared buffer")));
    // The Metal oracle and the old-engine candidate supply command-buffer handles, so the shared
    // gate admits the literal. The engine candidate maps a command buffer as its record
    // storage, an argument-buffer buffer whose address its owner holds, and refuses one whose owner
    // is not bound with a typed executor reason.
    assert!(crate::executor_contract::require_case(&case, "test").is_ok());
    assert_engine_refuses(&case, "without its owner buffer");
    case.command_buffers[0].buffer_binding = case.buffers[0].binding;
    let mapped = crate::candidate::engine::compute_case(&case, &kernel_interface()).unwrap();
    let key = (case.buffers[0].binding, 0);
    assert!(mapped.objects.is_empty());
    assert_eq!(
        mapped.command_buffers,
        std::collections::BTreeMap::from([(key, case.command_buffers[0].clone())])
    );
    let mut records = vec![0; case.command_buffers[0].record_storage_bytes().unwrap()];
    case.command_buffers[0]
        .initialize_records(&mut records)
        .unwrap();
    assert_eq!(
        mapped.argument_buffer_buffers,
        std::collections::BTreeMap::from([(key, records)])
    );
}

/// The AIR-layer interface of a kernel without parameters.
fn kernel_interface() -> crate::check::interface::AirInterface {
    crate::check::interface::AirInterface::synthetic(
        metal2vulkan_air::abi::Stage::Kernel {
            max_total_threads: None,
            work_group_size: None,
        },
        vec![],
        vec![],
        vec![],
    )
}

fn assert_engine_refuses(case: &AuthoredCase, what: &str) {
    match crate::candidate::engine::compute_case(case, &kernel_interface()) {
        Err(crate::check::admission::NotSupported::NotImplemented(detail)) => {
            assert!(detail.contains(what), "{detail}")
        }
        other => panic!("expected a typed not-implemented refusal, got {other:?}"),
    }
}

#[test]
fn pipeline_handles_require_unique_tokens_and_hash_the_template() {
    use crate::case::indirect_commands::PipelineResource;
    let mut case = example();
    case.pipeline_states.push(PipelineResource {
        buffer_binding: case.buffers[0].binding,
        field_offset: 0,
        token: 1,
        air_sha256: "33".repeat(32),
        case_id: "11".repeat(32),
    });
    let id = case.computed_case_id().unwrap();
    let input = case.computed_input_sha256().unwrap();
    case.pipeline_states[0].case_id = "22".repeat(32);
    assert_ne!(case.computed_case_id().unwrap(), id);
    assert_ne!(case.computed_input_sha256().unwrap(), input);
    let id = case.computed_case_id().unwrap();
    let input = case.computed_input_sha256().unwrap();
    case.pipeline_states[0].air_sha256 = "44".repeat(32);
    assert_ne!(case.computed_case_id().unwrap(), id);
    assert_ne!(case.computed_input_sha256().unwrap(), input);
    let mut other = case.pipeline_states[0].clone();
    other.field_offset = 8;
    case.pipeline_states.push(other);
    assert!(case
        .validate_literal_resources()
        .unwrap_err()
        .iter()
        .any(|error| error.contains("tokens must be nonzero and unique")));
    case.pipeline_states.pop();
    case.pipeline_states[0].token = 0;
    assert!(case
        .validate_literal_resources()
        .unwrap_err()
        .iter()
        .any(|error| error.contains("tokens must be nonzero and unique")));
    case.pipeline_states[0].token = 1;
    assert!(crate::executor_contract::require_case(&case, "test").is_ok());
    // The engine candidate maps a pipeline state as its authored token in its owner, the
    // word a command's pipeline field records, and refuses one whose owner is not bound.
    let mapped = crate::candidate::engine::compute_case(&case, &kernel_interface()).unwrap();
    assert_eq!(
        mapped.objects.into_iter().collect::<Vec<_>>(),
        [((case.buffers[0].binding, 0), 1)]
    );
    case.pipeline_states[0].buffer_binding = u32::MAX;
    assert_engine_refuses(&case, "without its owner buffer");
}

/// What the host encodes and executes: a case without it keeps its identity, one with it changes
/// it; each rule of the host state is checked against the descriptor, the case's buffers and
/// tokens, and the stage; and the host's records and the parent state are what the literal states,
/// decoded back through the host contract after the producer.
#[test]
fn host_commands_and_executions_validate_and_encode_their_records() {
    use crate::case::indirect_commands::{
        CommandBinding, CommandBufferResource, CommandDescriptor, ComputeDispatch, Execution,
        HostAction, HostCommand, HostStageInRegion, PipelineResource, PipelineTemplate,
        ThreadgroupLength,
    };
    use metal2vulkan::bind::indirect_commands::{Action, BufferStage};
    use std::num::NonZeroU64;
    let mut case = example();
    case.case_id = "22".repeat(32);
    // A 512-byte buffer a command binds at offset 256.
    case.buffers.push(BufferResource {
        binding: 1,
        role: ResourceRole::Input,
        bytes_b64: Some(base64::engine::general_purpose::STANDARD.encode([0u8; 512])),
        initial_bytes_b64: None,
        rasterization_rate_map: None,
    });
    case.pipeline_states.push(PipelineResource {
        buffer_binding: 0,
        field_offset: 8,
        token: 7,
        air_sha256: "33".repeat(32),
        case_id: "44".repeat(32),
    });
    let produced = CommandBufferResource {
        buffer_binding: 0,
        field_offset: 0,
        capacity: 4,
        execution_range: [0, 4],
        inherit_pipeline_state: false,
        inherit_buffers: false,
        descriptor: CommandDescriptor::Compute {
            dispatch: ComputeDispatch::Both,
            max_buffer_bind_count: 2,
            max_threadgroup_memory_bind_count: 1,
            support_dynamic_attribute_stride: false,
        },
        commands: vec![],
        execute: None,
    };
    case.command_buffers.push(produced.clone());
    case.validate_literal_resources().unwrap();
    // Neither field serializes while absent, so an existing case keeps its identity.
    let value = serde_json::to_value(&produced).unwrap();
    assert!(value.get("commands").is_none() && value.get("execute").is_none());
    let produced_id = case.computed_case_id().unwrap();
    let command = HostCommand {
        index: 1,
        action: Some(HostAction::DispatchThreads {
            grid: [4, 1, 1],
            threads_per_threadgroup: [4, 1, 1],
        }),
        barrier: true,
        pipeline: Some(7),
        buffers: vec![CommandBinding {
            index: 1,
            buffer: 1,
            offset: 256,
            attribute_stride: None,
        }],
        threadgroup_memory_lengths: vec![ThreadgroupLength {
            index: 0,
            length: 32,
        }],
        stage_in_region: None,
        vertex_buffers: vec![],
        fragment_buffers: vec![],
    };
    case.command_buffers[0].commands.push(command.clone());
    case.command_buffers[0].execute = Some(Execution {
        pipeline: None,
        buffers: vec![],
        vertex_buffers: vec![],
        fragment_buffers: vec![],
        object_buffers: vec![],
        mesh_buffers: vec![],
    });
    case.validate_literal_resources().unwrap();
    let executed_id = case.computed_case_id().unwrap();
    assert_ne!(executed_id, produced_id);
    case.command_buffers[0].execute = None;
    assert_ne!(case.computed_case_id().unwrap(), executed_id);
    assert_ne!(case.computed_case_id().unwrap(), produced_id);
    case.command_buffers[0].execute = Some(Execution {
        pipeline: None,
        buffers: vec![],
        vertex_buffers: vec![],
        fragment_buffers: vec![],
        object_buffers: vec![],
        mesh_buffers: vec![],
    });

    let rejects = |case: &AuthoredCase, what: &str| {
        let errors = case.validate_literal_resources().unwrap_err();
        assert!(
            errors.iter().any(|error| error.contains(what)),
            "{what}: {errors:?}"
        );
    };
    let edit = |change: &dyn Fn(&mut CommandBufferResource)| {
        let mut variant = case.clone();
        change(&mut variant.command_buffers[0]);
        variant
    };
    let command_edit =
        |change: &dyn Fn(&mut HostCommand)| edit(&|resource| change(&mut resource.commands[0]));
    rejects(
        &command_edit(&|command| command.pipeline = Some(9)),
        "pipeline token 9 is not the case's",
    );
    rejects(
        &command_edit(&|command| command.index = 4),
        "is not below the capacity",
    );
    rejects(
        &edit(&|resource| {
            let first = resource.commands[0].clone();
            resource.commands.push(first);
        }),
        "sorted by a unique index",
    );
    for (binding, what) in [
        ((1, 1, 16), "not a multiple of 256"),
        ((1, 1, 512), "not a multiple of 256"),
        ((1, 5, 0), "buffer 5 is not a case buffer"),
        ((2, 1, 0), "buffer index 2 is not below 2"),
    ] {
        rejects(
            &command_edit(&|command| {
                command.buffers = vec![CommandBinding {
                    index: binding.0,
                    buffer: binding.1,
                    offset: binding.2,
                    attribute_stride: None,
                }]
            }),
            what,
        );
    }
    rejects(
        &command_edit(&|command| {
            command.buffers.insert(
                0,
                CommandBinding {
                    index: 1,
                    buffer: 0,
                    offset: 0,
                    attribute_stride: None,
                },
            )
        }),
        "sorted and unique",
    );
    // An attribute stride needs a descriptor that supports dynamic strides; a parent's inherited
    // buffers carry none, and a compute command's stage-in region is its own.
    let dynamic = |resource: &mut CommandBufferResource| {
        if let CommandDescriptor::Compute {
            support_dynamic_attribute_stride,
            ..
        } = &mut resource.descriptor
        {
            *support_dynamic_attribute_stride = true;
        }
    };
    let stride = |resource: &mut CommandBufferResource| {
        resource.commands[0].buffers[0].attribute_stride = Some(12)
    };
    rejects(
        &edit(&stride),
        "buffer index 1 binds an attribute stride, which this stage or descriptor does not admit",
    );
    let strided = edit(&|resource| {
        dynamic(resource);
        stride(resource);
        resource.commands[0].stage_in_region = Some(HostStageInRegion {
            origin: [1, 2, 3],
            size: [4, 5, 6],
        });
    });
    strided.validate_literal_resources().unwrap();
    rejects(
        &edit(&|resource| {
            dynamic(resource);
            resource.inherit_buffers = true;
            resource.commands[0].buffers.clear();
            resource.execute.as_mut().unwrap().buffers = vec![CommandBinding {
                index: 0,
                buffer: 1,
                offset: 0,
                attribute_stride: Some(12),
            }];
        }),
        "binds an attribute stride",
    );
    // Neither field serializes while absent, so an existing case keeps its identity.
    let value = serde_json::to_value(&case.command_buffers[0].commands[0]).unwrap();
    assert!(value.get("stage_in_region").is_none());
    assert!(value["buffers"][0].get("attribute_stride").is_none());
    assert_ne!(
        strided.computed_case_id().unwrap(),
        edit(&dynamic).computed_case_id().unwrap()
    );
    for (index, length) in [(0, 20), (1, 16)] {
        rejects(
            &command_edit(&|command| {
                command.threadgroup_memory_lengths = vec![ThreadgroupLength { index, length }]
            }),
            "threadgroup lengths must be sorted",
        );
    }
    rejects(
        &edit(&|resource| {
            if let CommandDescriptor::Compute { dispatch, .. } = &mut resource.descriptor {
                *dispatch = ComputeDispatch::Threadgroups;
            }
        }),
        "admits no such dispatch",
    );
    rejects(
        &edit(&|resource| resource.inherit_buffers = true),
        "sets the buffers it inherits",
    );
    // An inherited pipeline: a command may not set it, and the execution names its parent.
    rejects(
        &edit(&|resource| resource.inherit_pipeline_state = true),
        "sets the pipeline it inherits",
    );
    let parent = PipelineTemplate {
        token: 9,
        air_sha256: "55".repeat(32),
        case_id: "66".repeat(32),
    };
    let inherit = |parent: Option<PipelineTemplate>| {
        edit(&|resource| {
            resource.inherit_pipeline_state = true;
            resource.commands[0].pipeline = None;
            resource.execute.as_mut().unwrap().pipeline = parent.clone();
        })
    };
    inherit(Some(parent.clone()))
        .validate_literal_resources()
        .unwrap();
    rejects(&inherit(None), "exactly when its commands inherit it");
    rejects(
        &edit(&|resource| resource.execute.as_mut().unwrap().pipeline = Some(parent.clone())),
        "exactly when its commands inherit it",
    );
    // A parent token is a pipeline token like a pipeline state's: nonzero and unique.
    for token in [0, 7] {
        rejects(
            &inherit(Some(PipelineTemplate {
                token,
                ..parent.clone()
            })),
            "tokens must be nonzero and unique",
        );
    }
    rejects(
        &inherit(Some(PipelineTemplate {
            case_id: "not-a-hash".into(),
            ..parent.clone()
        })),
        "parent pipeline case_id",
    );
    rejects(
        &edit(&|resource| {
            resource.execute.as_mut().unwrap().buffers = vec![CommandBinding {
                index: 0,
                buffer: 0,
                offset: 0,
                attribute_stride: None,
            }]
        }),
        "parent buffers need inherited buffers",
    );
    rejects(
        &edit(&|resource| {
            resource.capacity = 65;
            resource.execution_range = [0, 65];
        }),
        "at most 64 commands",
    );
    // A reversed range is refused by the execution's own check, never subtracted.
    rejects(
        &edit(&|resource| resource.execution_range = [2, 1]),
        "an execution's range is reversed",
    );
    // A compute command binds no render stage's buffer and draws nothing.
    let render_binding = CommandBinding {
        index: 0,
        buffer: 1,
        offset: 0,
        attribute_stride: None,
    };
    rejects(
        &command_edit(&|command| command.vertex_buffers = vec![render_binding.clone()]),
        "binds no vertex or fragment buffer",
    );
    rejects(
        &command_edit(&|command| command.fragment_buffers = vec![render_binding.clone()]),
        "binds no vertex or fragment buffer",
    );
    rejects(
        &command_edit(&|command| {
            command.action = Some(HostAction::DrawPrimitives {
                primitive: Primitive::Point,
                vertex_start: 0,
                vertex_count: 1,
                instance_count: 1,
                base_instance: 0,
            })
        }),
        "admits no such dispatch",
    );
    rejects(
        &edit(&|resource| {
            resource.execute.as_mut().unwrap().vertex_buffers = vec![render_binding.clone()];
            resource.inherit_buffers = true;
            resource.commands[0].buffers.clear();
        }),
        "vertex buffers are no stage of the commands the execution runs",
    );
    // A vertex or fragment case's draw is a producer too.
    for stage in [Stage::Vertex, Stage::Fragment] {
        let mut render = case.clone();
        render.stage = stage;
        render.dispatch = None;
        render.draw = Some(Draw {
            primitive: Primitive::Point,
            vertex_start: 0,
            vertex_count: 1,
            instance_count: 1,
            index_buffer: None,
            base_vertex: 0,
            base_instance: 0,
        });
        render.validate_literal_resources().unwrap();
    }
    // A tile function's dispatch is a render pass's: no executor models its commands.
    let mut tile = case.clone();
    tile.imageblock = Some(ImageblockResource {
        dimensions: [1, 1],
        implicit_coverage: Some(ImplicitImageblockCoverage::FullSingleSample),
    });
    rejects(&tile, "a tile function's commands");

    // The host's records, written into initialized storage and decoded as the producer left them.
    let mut inherited = inherit(Some(parent.clone()));
    let resource = &mut inherited.command_buffers[0];
    resource.inherit_buffers = true;
    resource.commands[0].buffers.clear();
    resource.execute.as_mut().unwrap().buffers = vec![CommandBinding {
        index: 3,
        buffer: 1,
        offset: 256,
        attribute_stride: None,
    }];
    inherited.validate_literal_resources().unwrap();
    let address = |buffer: u32| (buffer == 1).then_some(0x1_0001_0000);
    for (resource, buffers) in [
        (&case.command_buffers[0], Some(0x1_0001_0100)),
        (&inherited.command_buffers[0], None),
    ] {
        let mut bytes = vec![0; resource.record_storage_bytes().unwrap()];
        resource.initialize_records(&mut bytes).unwrap();
        resource.encode_host_records(&mut bytes, address).unwrap();
        let records = resource
            .validated_records(&bytes)
            .unwrap()
            .collect::<Vec<_>>();
        assert_eq!(records.len(), 4);
        for (index, record) in records.iter().enumerate() {
            if index != 1 {
                assert_eq!(record.action, Action::Empty, "{index}");
                continue;
            }
            assert_eq!(
                record.action,
                Action::DispatchThreads {
                    grid: [4, 1, 1],
                    local_size: [4, 1, 1],
                }
            );
            assert!(record.barrier);
            assert_eq!(record.threadgroup_memory_lengths[0], 32);
            let kernel = record.bindings.buffers(BufferStage::Kernel);
            match buffers {
                Some(at) => {
                    assert_eq!(record.bindings.pipeline, NonZeroU64::new(7));
                    assert_eq!(kernel[1].map(|b| b.address), Some(at));
                }
                None => {
                    assert_eq!(record.bindings.pipeline, None);
                    assert!(kernel.iter().all(Option::is_none));
                }
            }
        }
    }
    // The stride form's stride and the stage-in region, as the record holds them.
    let resource = &strided.command_buffers[0];
    let mut bytes = vec![0; resource.record_storage_bytes().unwrap()];
    resource.initialize_records(&mut bytes).unwrap();
    resource.encode_host_records(&mut bytes, address).unwrap();
    let record = resource.validated_records(&bytes).unwrap().nth(1).unwrap();
    let kernel = record.bindings.buffers(BufferStage::Kernel)[1].expect("a buffer set");
    assert_eq!(
        (kernel.address, kernel.attribute_stride),
        (0x1_0001_0100, Some(12))
    );
    assert_eq!(record.stage_in_region, [1, 2, 3, 4, 5, 6]);
    // The parent state: absent when nothing executes, else the parent's token and buffers.
    assert_eq!(produced.parent_bindings(address), Ok(None));
    let parent_state = inherited.command_buffers[0]
        .parent_bindings(address)
        .unwrap()
        .unwrap();
    assert_eq!(parent_state.pipeline, NonZeroU64::new(9));
    assert_eq!(
        parent_state.buffers(BufferStage::Kernel)[3].map(|b| b.address),
        Some(0x1_0001_0100)
    );
    assert!(inherited.command_buffers[0]
        .parent_bindings(|_| None)
        .is_err());
    // A buffer without an address, or storage shorter than the capacity, writes nothing.
    let mut bytes = vec![0; case.command_buffers[0].record_storage_bytes().unwrap()];
    case.command_buffers[0]
        .initialize_records(&mut bytes)
        .unwrap();
    assert!(case.command_buffers[0]
        .encode_host_records(&mut bytes, |_| None)
        .is_err());
    let short = bytes.len() - 1;
    assert!(case.command_buffers[0]
        .encode_host_records(&mut bytes[..short], address)
        .is_err());
}

/// Host-encoded render commands: a draw, a render pipeline token, and vertex and fragment buffers,
/// each checked against the render descriptor's counts and family, encoded into the records the
/// host contract decodes, with the parent's vertex and fragment buffers.
#[test]
fn host_render_commands_validate_and_encode_their_records() {
    use crate::case::indirect_commands::{
        authored_primitive, metal_primitive_type, CommandBinding, CommandBufferResource,
        CommandDescriptor, Execution, HostAction, HostCommand, HostStageInRegion, PipelineTemplate,
        ThreadgroupLength,
    };
    use metal2vulkan::bind::indirect_commands::{Action, BufferBinding, BufferStage};
    use std::num::NonZeroU64;
    let mut case = example();
    case.case_id = "22".repeat(32);
    case.stage = Stage::Vertex;
    case.dispatch = None;
    case.draw = Some(Draw {
        primitive: Primitive::Point,
        vertex_start: 0,
        vertex_count: 2,
        instance_count: 1,
        index_buffer: None,
        base_vertex: 0,
        base_instance: 0,
    });
    case.buffers.push(BufferResource {
        binding: 1,
        role: ResourceRole::Input,
        bytes_b64: Some(base64::engine::general_purpose::STANDARD.encode([0u8; 512])),
        initial_bytes_b64: None,
        rasterization_rate_map: None,
    });
    let draw = HostAction::DrawPrimitives {
        primitive: Primitive::TriangleStrip,
        vertex_start: 3,
        vertex_count: 4,
        instance_count: 2,
        base_instance: 1,
    };
    let binding = |index, offset| CommandBinding {
        index,
        buffer: 1,
        offset,
        attribute_stride: None,
    };
    case.command_buffers.push(CommandBufferResource {
        buffer_binding: 0,
        field_offset: 0,
        capacity: 2,
        execution_range: [0, 2],
        inherit_pipeline_state: true,
        inherit_buffers: false,
        descriptor: CommandDescriptor::Render {
            draws: crate::case::indirect_commands::RenderDraws::default(),
            max_vertex_buffer_bind_count: 2,
            max_fragment_buffer_bind_count: 1,
            max_object_buffer_bind_count: 0,
            max_mesh_buffer_bind_count: 0,
            support_dynamic_attribute_stride: false,
        },
        commands: vec![HostCommand {
            index: 1,
            action: Some(draw),
            barrier: false,
            pipeline: None,
            buffers: vec![],
            threadgroup_memory_lengths: vec![],
            stage_in_region: None,
            vertex_buffers: vec![binding(1, 256)],
            fragment_buffers: vec![binding(0, 0)],
        }],
        execute: Some(Execution {
            pipeline: Some(PipelineTemplate {
                token: 9,
                air_sha256: "55".repeat(32),
                case_id: "66".repeat(32),
            }),
            buffers: vec![],
            vertex_buffers: vec![],
            fragment_buffers: vec![],
            object_buffers: vec![],
            mesh_buffers: vec![],
        }),
    });
    case.validate_literal_resources().unwrap();
    // The new fields serialize only when present.
    let value = serde_json::to_value(&case.command_buffers[0].commands[0]).unwrap();
    assert!(value.get("buffers").is_none() && value.get("vertex_buffers").is_some());
    assert_eq!(value["action"]["kind"], "draw_primitives");

    let rejects = |case: &AuthoredCase, what: &str| {
        let errors = case.validate_literal_resources().unwrap_err();
        assert!(
            errors.iter().any(|error| error.contains(what)),
            "{what}: {errors:?}"
        );
    };
    let command_edit = |change: &dyn Fn(&mut HostCommand)| {
        let mut variant = case.clone();
        change(&mut variant.command_buffers[0].commands[0]);
        variant
    };
    rejects(
        &command_edit(&|command| {
            command.action = Some(HostAction::DispatchThreads {
                grid: [1; 3],
                threads_per_threadgroup: [1; 3],
            })
        }),
        "draws, never dispatches",
    );
    rejects(
        &command_edit(&|command| command.barrier = true),
        "render command's barrier needs mesh draws",
    );
    rejects(
        &command_edit(&|command| {
            command.stage_in_region = Some(HostStageInRegion {
                origin: [0; 3],
                size: [1; 3],
            })
        }),
        "a render command has no stage-in region",
    );
    // A vertex buffer's attribute stride needs a descriptor supporting dynamic strides; a
    // fragment buffer takes none.
    let strided = |command: &mut HostCommand| command.vertex_buffers[0].attribute_stride = Some(8);
    rejects(
        &command_edit(&strided),
        "vertex buffers: buffer index 1 binds an attribute stride",
    );
    let mut dynamic = case.clone();
    if let CommandDescriptor::Render {
        support_dynamic_attribute_stride,
        ..
    } = &mut dynamic.command_buffers[0].descriptor
    {
        *support_dynamic_attribute_stride = true;
    }
    let mut vertex_stride = dynamic.clone();
    strided(&mut vertex_stride.command_buffers[0].commands[0]);
    vertex_stride.validate_literal_resources().unwrap();
    let mut fragment_stride = dynamic.clone();
    fragment_stride.command_buffers[0].commands[0].fragment_buffers[0].attribute_stride = Some(8);
    rejects(
        &fragment_stride,
        "fragment buffers: buffer index 0 binds an attribute stride",
    );
    rejects(
        &command_edit(&|command| command.buffers = vec![binding(0, 0)]),
        "binds no kernel buffer or threadgroup length",
    );
    rejects(
        &command_edit(&|command| {
            command.threadgroup_memory_lengths = vec![ThreadgroupLength {
                index: 0,
                length: 16,
            }]
        }),
        "binds no kernel buffer or threadgroup length",
    );
    rejects(
        &command_edit(&|command| command.vertex_buffers = vec![binding(2, 0)]),
        "vertex buffers: buffer index 2 is not below 2",
    );
    rejects(
        &command_edit(&|command| command.fragment_buffers = vec![binding(1, 0)]),
        "fragment buffers: buffer index 1 is not below 1",
    );
    rejects(
        &command_edit(&|command| command.vertex_buffers = vec![binding(0, 16)]),
        "not a multiple of 256",
    );
    rejects(
        &command_edit(&|command| command.pipeline = Some(9)),
        "sets the pipeline it inherits",
    );
    let mut inherited = case.clone();
    let resource = &mut inherited.command_buffers[0];
    resource.inherit_buffers = true;
    resource.commands[0].vertex_buffers.clear();
    resource.commands[0].fragment_buffers.clear();
    let execution = resource.execute.as_mut().unwrap();
    execution.vertex_buffers = vec![binding(4, 256)];
    execution.fragment_buffers = vec![binding(3, 0)];
    inherited.validate_literal_resources().unwrap();
    let mut kernel_parent = inherited.clone();
    kernel_parent.command_buffers[0]
        .execute
        .as_mut()
        .unwrap()
        .buffers = vec![binding(0, 0)];
    rejects(
        &kernel_parent,
        "kernel buffers are no stage of the commands the execution runs",
    );

    // The records: the draw at its slot with its bindings, the parent's vertex and fragment
    // buffers, each decoded back through the host contract.
    let address = |buffer: u32| (buffer == 1).then_some(0x1_0001_0000);
    let resource = &case.command_buffers[0];
    let mut bytes = vec![0; resource.record_storage_bytes().unwrap()];
    resource.initialize_records(&mut bytes).unwrap();
    resource.encode_host_records(&mut bytes, address).unwrap();
    let records = resource
        .validated_records(&bytes)
        .unwrap()
        .collect::<Vec<_>>();
    assert_eq!(records[0].action, Action::Empty);
    assert_eq!(
        records[1].action,
        Action::DrawPrimitives {
            primitive_type: metal_primitive_type(Primitive::TriangleStrip).as_raw(),
            vertex_start: 3,
            vertex_count: 4,
            instance_count: 2,
            base_instance: 1,
        }
    );
    assert_eq!(records[1].bindings.pipeline, None);
    assert_eq!(
        records[1].bindings.buffers(BufferStage::Vertex)[1],
        Some(BufferBinding {
            address: 0x1_0001_0100,
            attribute_stride: None,
        })
    );
    assert_eq!(
        records[1].bindings.buffers(BufferStage::Fragment)[0].map(|b| b.address),
        Some(0x1_0001_0000)
    );
    let resource = &vertex_stride.command_buffers[0];
    resource.initialize_records(&mut bytes).unwrap();
    resource.encode_host_records(&mut bytes, address).unwrap();
    let record = resource.validated_records(&bytes).unwrap().nth(1).unwrap();
    let vertex = record.bindings.buffers(BufferStage::Vertex)[1].expect("a buffer set");
    assert_eq!(
        (vertex.address, vertex.attribute_stride),
        (0x1_0001_0100, Some(8))
    );
    let parent = inherited.command_buffers[0]
        .parent_bindings(address)
        .unwrap()
        .unwrap();
    assert_eq!(parent.pipeline, NonZeroU64::new(9));
    assert_eq!(
        parent.buffers(BufferStage::Vertex)[4].map(|b| b.address),
        Some(0x1_0001_0100)
    );
    assert_eq!(
        parent.buffers(BufferStage::Fragment)[3].map(|b| b.address),
        Some(0x1_0001_0000)
    );
    // Every authored primitive is the Metal primitive type its record holds, and back.
    for primitive in [
        Primitive::Point,
        Primitive::Line,
        Primitive::LineStrip,
        Primitive::Triangle,
        Primitive::TriangleStrip,
    ] {
        assert_eq!(
            authored_primitive(metal_primitive_type(primitive).as_raw()),
            Some(primitive)
        );
    }
    assert_eq!(authored_primitive(5), None);
}

/// Host-encoded indexed draws: the descriptor names its draw kinds (absent, draws of primitives
/// alone, so earlier cases keep their identities), a draw needs its kind, and an indexed draw's
/// index buffer is a case buffer whose indices lie aligned inside it, as Metal's validation and the
/// runtime's setter require; its record holds the buffer's address plus the offset, the Metal index
/// type, and a signed base vertex.
#[test]
fn host_indexed_draws_validate_and_encode_their_records() {
    use crate::case::indirect_commands::{
        metal_primitive_type, CommandBufferResource, CommandDescriptor, Execution, HostAction,
        HostCommand, PipelineTemplate, RenderDraws,
    };
    use crate::case::IndexType;
    use metal2vulkan::bind::indirect_commands::Action;
    let mut case = example();
    case.case_id = "22".repeat(32);
    case.stage = Stage::Vertex;
    case.dispatch = None;
    case.draw = Some(Draw {
        primitive: Primitive::Point,
        vertex_start: 0,
        vertex_count: 2,
        instance_count: 1,
        index_buffer: None,
        base_vertex: 0,
        base_instance: 0,
    });
    for (binding, length) in [(1u32, 512usize), (2, 64)] {
        case.buffers.push(BufferResource {
            binding,
            role: ResourceRole::Input,
            bytes_b64: Some(base64::engine::general_purpose::STANDARD.encode(vec![0u8; length])),
            initial_bytes_b64: None,
            rasterization_rate_map: None,
        });
    }
    let indexed = HostAction::DrawIndexedPrimitives {
        primitive: Primitive::Point,
        index_count: 6,
        index_type: IndexType::Uint16,
        index_buffer: 2,
        index_buffer_offset: 4,
        instance_count: 2,
        base_vertex: -3,
        base_instance: 1,
    };
    let draws = RenderDraws {
        primitives: false,
        indexed: true,
        patches: false,
        indexed_patches: false,
        mesh_threadgroups: false,
        mesh_threads: false,
    };
    case.command_buffers.push(CommandBufferResource {
        buffer_binding: 0,
        field_offset: 0,
        capacity: 1,
        execution_range: [0, 1],
        inherit_pipeline_state: true,
        inherit_buffers: false,
        descriptor: CommandDescriptor::Render {
            draws,
            max_vertex_buffer_bind_count: 2,
            max_fragment_buffer_bind_count: 0,
            max_object_buffer_bind_count: 0,
            max_mesh_buffer_bind_count: 0,
            support_dynamic_attribute_stride: false,
        },
        commands: vec![HostCommand {
            index: 0,
            action: Some(indexed),
            barrier: false,
            pipeline: None,
            buffers: vec![],
            threadgroup_memory_lengths: vec![],
            stage_in_region: None,
            vertex_buffers: vec![],
            fragment_buffers: vec![],
        }],
        execute: Some(Execution {
            pipeline: Some(PipelineTemplate {
                token: 9,
                air_sha256: "55".repeat(32),
                case_id: "66".repeat(32),
            }),
            buffers: vec![],
            vertex_buffers: vec![],
            fragment_buffers: vec![],
            object_buffers: vec![],
            mesh_buffers: vec![],
        }),
    });
    case.validate_literal_resources().unwrap();
    // The draw kinds serialize only where they are not draws of primitives alone.
    let descriptor = serde_json::to_value(&case.command_buffers[0].descriptor).unwrap();
    assert_eq!(
        descriptor["draws"],
        serde_json::json!({"primitives": false, "indexed": true, "patches": false})
    );
    let primitives = CommandDescriptor::Render {
        draws: RenderDraws::default(),
        max_vertex_buffer_bind_count: 2,
        max_fragment_buffer_bind_count: 0,
        max_object_buffer_bind_count: 0,
        max_mesh_buffer_bind_count: 0,
        support_dynamic_attribute_stride: false,
    };
    let value = serde_json::to_value(&primitives).unwrap();
    assert!(value.get("draws").is_none());
    assert_eq!(
        serde_json::from_value::<CommandDescriptor>(value).unwrap(),
        primitives
    );
    let action = serde_json::to_value(indexed).unwrap();
    assert_eq!(action["kind"], "draw_indexed_primitives");
    assert_eq!(action["index_type"], "uint16");

    let rejects = |change: &dyn Fn(&mut CommandBufferResource), what: &str| {
        let mut variant = case.clone();
        change(&mut variant.command_buffers[0]);
        let errors = variant.validate_literal_resources().unwrap_err();
        assert!(
            errors.iter().any(|error| error.contains(what)),
            "{what}: {errors:?}"
        );
    };
    let field = |change: fn(&mut HostAction)| {
        move |resource: &mut CommandBufferResource| {
            let mut edited = indexed;
            change(&mut edited);
            resource.commands[0].action = Some(edited);
        }
    };
    rejects(
        &|resource| {
            resource.commands[0].action = Some(HostAction::DrawPrimitives {
                primitive: Primitive::Point,
                vertex_start: 0,
                vertex_count: 1,
                instance_count: 1,
                base_instance: 0,
            })
        },
        "admits no such draw",
    );
    rejects(
        &|resource| {
            resource.descriptor = CommandDescriptor::Render {
                draws: RenderDraws::default(),
                max_vertex_buffer_bind_count: 2,
                max_fragment_buffer_bind_count: 0,
                max_object_buffer_bind_count: 0,
                max_mesh_buffer_bind_count: 0,
                support_dynamic_attribute_stride: false,
            }
        },
        "admits no such draw",
    );
    rejects(
        &field(|draw| {
            if let HostAction::DrawIndexedPrimitives { index_count, .. } = draw {
                *index_count = 0;
            }
        }),
        "no indices or instances",
    );
    rejects(
        &field(|draw| {
            if let HostAction::DrawIndexedPrimitives { instance_count, .. } = draw {
                *instance_count = 0;
            }
        }),
        "no indices or instances",
    );
    rejects(
        &field(|draw| {
            if let HostAction::DrawIndexedPrimitives {
                index_buffer_offset,
                ..
            } = draw
            {
                *index_buffer_offset = 3;
            }
        }),
        "not aligned",
    );
    rejects(
        &field(|draw| {
            if let HostAction::DrawIndexedPrimitives { index_count, .. } = draw {
                *index_count = 31;
            }
        }),
        "not aligned",
    );
    rejects(
        &field(|draw| {
            if let HostAction::DrawIndexedPrimitives { index_buffer, .. } = draw {
                *index_buffer = 7;
            }
        }),
        "not a case buffer",
    );
    // Thirty indices end at the buffer's last byte.
    let mut last = case.clone();
    field(|draw| {
        if let HostAction::DrawIndexedPrimitives { index_count, .. } = draw {
            *index_count = 30;
        }
    })(&mut last.command_buffers[0]);
    last.validate_literal_resources().unwrap();

    // The record: the index buffer's address plus the offset, the Metal index type, a signed base
    // vertex, under a header whose policy allows indexed draws alone.
    let address = |buffer: u32| (buffer == 2).then_some(0x2_0000);
    let resource = &case.command_buffers[0];
    let mut bytes = vec![0; resource.record_storage_bytes().unwrap()];
    resource.initialize_records(&mut bytes).unwrap();
    resource.encode_host_records(&mut bytes, address).unwrap();
    let records = resource
        .validated_records(&bytes)
        .unwrap()
        .collect::<Vec<_>>();
    assert_eq!(
        records[0].action,
        Action::DrawIndexedPrimitives {
            primitive_type: metal_primitive_type(Primitive::Point).as_raw(),
            index_count: 6,
            index_type: 0,
            index_buffer: 0x2_0004,
            instance_count: 2,
            base_vertex: -3,
            base_instance: 1,
        }
    );
    assert_eq!(
        resource.descriptor.product(),
        metal2vulkan::bind::indirect_commands::CommandDescriptor::Render {
            draws: metal2vulkan::bind::indirect_commands::RenderDraws {
                primitives: false,
                indexed: true,
                patches: false,
                indexed_patches: false,
                mesh_threadgroups: false,
                mesh_threads: false,
            },
            max_vertex_buffer_bind_count: 2,
            max_fragment_buffer_bind_count: 0,
            max_object_buffer_bind_count: 0,
            max_mesh_buffer_bind_count: 0,
            support_dynamic_attribute_stride: false,
        }
    );
    // An index buffer without an address writes nothing.
    assert!(resource.encode_host_records(&mut bytes, |_| None).is_err());
}

/// Host-encoded patch draws: a draw needs its kind; its patch index buffer, where it names one, is
/// a case buffer whose indices lie aligned inside it, and its factor buffer a case buffer at an
/// offset and an instance stride that are multiples of 4, as Metal's validation and the runtime's
/// setter require; its record holds each buffer's address plus its offset (0 for no patch index
/// buffer) and the factor scale 1.
#[test]
fn host_patch_draws_validate_and_encode_their_records() {
    use crate::case::indirect_commands::{
        CommandBufferResource, CommandDescriptor, Execution, HostAction, HostCommand,
        PipelineTemplate, RenderDraws,
    };
    use metal2vulkan::bind::indirect_commands::Action;
    let mut case = example();
    case.case_id = "23".repeat(32);
    case.stage = Stage::Vertex;
    case.dispatch = None;
    case.draw = Some(Draw {
        primitive: Primitive::Point,
        vertex_start: 0,
        vertex_count: 2,
        instance_count: 1,
        index_buffer: None,
        base_vertex: 0,
        base_instance: 0,
    });
    for (binding, length) in [(1u32, 512usize), (2, 24), (3, 48)] {
        case.buffers.push(BufferResource {
            binding,
            role: ResourceRole::Input,
            bytes_b64: Some(base64::engine::general_purpose::STANDARD.encode(vec![0u8; length])),
            initial_bytes_b64: None,
            rasterization_rate_map: None,
        });
    }
    let patches = HostAction::DrawPatches {
        control_points: 1,
        patch_start: 1,
        patch_count: 2,
        patch_index_buffer: Some(2),
        patch_index_buffer_offset: 12,
        instance_count: 2,
        base_instance: 5,
        factor_buffer: 3,
        factor_buffer_offset: 8,
        factor_instance_stride: 16,
    };
    case.command_buffers.push(CommandBufferResource {
        buffer_binding: 0,
        field_offset: 0,
        capacity: 1,
        execution_range: [0, 1],
        inherit_pipeline_state: true,
        inherit_buffers: false,
        descriptor: CommandDescriptor::Render {
            draws: RenderDraws {
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
        },
        commands: vec![HostCommand {
            index: 0,
            action: Some(patches),
            barrier: false,
            pipeline: None,
            buffers: vec![],
            threadgroup_memory_lengths: vec![],
            stage_in_region: None,
            vertex_buffers: vec![],
            fragment_buffers: vec![],
        }],
        execute: Some(Execution {
            pipeline: Some(PipelineTemplate {
                token: 9,
                air_sha256: "55".repeat(32),
                case_id: "66".repeat(32),
            }),
            buffers: vec![],
            vertex_buffers: vec![],
            fragment_buffers: vec![],
            object_buffers: vec![],
            mesh_buffers: vec![],
        }),
    });
    case.validate_literal_resources().unwrap();
    let action = serde_json::to_value(patches).unwrap();
    assert_eq!(action["kind"], "draw_patches");
    // No patch index buffer is an absent field.
    let mut unindexed = patches;
    if let HostAction::DrawPatches {
        patch_index_buffer, ..
    } = &mut unindexed
    {
        *patch_index_buffer = None;
    }
    let value = serde_json::to_value(unindexed).unwrap();
    assert!(value.get("patch_index_buffer").is_none());
    assert_eq!(
        serde_json::from_value::<HostAction>(value).unwrap(),
        unindexed
    );

    let rejects = |change: &dyn Fn(&mut CommandBufferResource), what: &str| {
        let mut variant = case.clone();
        change(&mut variant.command_buffers[0]);
        let errors = variant.validate_literal_resources().unwrap_err();
        assert!(
            errors.iter().any(|error| error.contains(what)),
            "{what}: {errors:?}"
        );
    };
    let field = |change: fn(&mut HostAction)| {
        move |resource: &mut CommandBufferResource| {
            let mut edited = patches;
            change(&mut edited);
            resource.commands[0].action = Some(edited);
        }
    };
    rejects(
        &|resource| {
            resource.descriptor = CommandDescriptor::Render {
                draws: RenderDraws::default(),
                max_vertex_buffer_bind_count: 2,
                max_fragment_buffer_bind_count: 0,
                max_object_buffer_bind_count: 0,
                max_mesh_buffer_bind_count: 0,
                support_dynamic_attribute_stride: false,
            }
        },
        "admits no such draw",
    );
    rejects(
        &field(|draw| {
            if let HostAction::DrawPatches { instance_count, .. } = draw {
                *instance_count = 0;
            }
        }),
        "no instances",
    );
    rejects(
        &field(|draw| {
            if let HostAction::DrawPatches {
                patch_index_buffer_offset,
                ..
            } = draw
            {
                *patch_index_buffer_offset = 10;
            }
        }),
        "not aligned",
    );
    // Patch indices at positions 1 and 2 from byte 16 end past the buffer's 24 bytes.
    rejects(
        &field(|draw| {
            if let HostAction::DrawPatches {
                patch_index_buffer_offset,
                ..
            } = draw
            {
                *patch_index_buffer_offset = 16;
            }
        }),
        "not aligned",
    );
    rejects(
        &field(|draw| {
            if let HostAction::DrawPatches {
                patch_index_buffer, ..
            } = draw
            {
                *patch_index_buffer = Some(7);
            }
        }),
        "not a case buffer",
    );
    rejects(
        &field(|draw| {
            if let HostAction::DrawPatches { factor_buffer, .. } = draw {
                *factor_buffer = 7;
            }
        }),
        "not a case buffer",
    );
    rejects(
        &field(|draw| {
            if let HostAction::DrawPatches {
                factor_buffer_offset,
                ..
            } = draw
            {
                *factor_buffer_offset = 6;
            }
        }),
        "multiple of 4",
    );
    rejects(
        &field(|draw| {
            if let HostAction::DrawPatches {
                factor_instance_stride,
                ..
            } = draw
            {
                *factor_instance_stride = 2;
            }
        }),
        "multiple of 4",
    );

    // The record: each buffer's address plus its offset, the scale 1, under a header whose policy
    // allows patch draws alone.
    let address = |buffer: u32| match buffer {
        2 => Some(0x2_0000),
        3 => Some(0x3_0000),
        _ => None,
    };
    let resource = &case.command_buffers[0];
    let mut bytes = vec![0; resource.record_storage_bytes().unwrap()];
    resource.initialize_records(&mut bytes).unwrap();
    resource.encode_host_records(&mut bytes, address).unwrap();
    let records = resource
        .validated_records(&bytes)
        .unwrap()
        .collect::<Vec<_>>();
    assert_eq!(
        records[0].action,
        Action::DrawPatches {
            control_points: 1,
            patch_start: 1,
            patch_count: 2,
            patch_index_buffer: 0x2_000c,
            instance_count: 2,
            base_instance: 5,
            factor_buffer: 0x3_0008,
            factor_instance_stride: 16,
            factor_scale: 1.0f32.to_bits(),
        }
    );
    let mut unindexed_case = case.clone();
    unindexed_case.command_buffers[0].commands[0].action = Some(unindexed);
    let resource = &unindexed_case.command_buffers[0];
    resource.initialize_records(&mut bytes).unwrap();
    resource.encode_host_records(&mut bytes, address).unwrap();
    let record = resource.validated_records(&bytes).unwrap().next().unwrap();
    assert!(matches!(
        record.action,
        Action::DrawPatches {
            patch_index_buffer: 0,
            ..
        }
    ));
    // A factor buffer without an address writes nothing.
    assert!(resource
        .encode_host_records(&mut bytes, |buffer| (buffer == 2).then_some(0x2_0000))
        .is_err());
}

/// Executed command buffers run in key order on both executors: a case's identity sorts its command
/// buffers by key, so two listings of the same buffers are one case and must be one execution.
/// Listed `(1, 0)` then `(0, 0)`, they run `(0, 0)` first on the oracle (`executed_in_key_order`)
/// and in the candidate's keyed map alike.
#[test]
fn executed_command_buffers_run_in_identity_order() {
    use crate::case::indirect_commands::{
        executed_in_key_order, CommandBufferResource, CommandDescriptor, ComputeDispatch, Execution,
    };
    let mut case = example();
    case.case_id = "22".repeat(32);
    case.buffers.push(BufferResource {
        binding: 1,
        role: ResourceRole::Input,
        bytes_b64: Some(base64::engine::general_purpose::STANDARD.encode([0u8; 16])),
        initial_bytes_b64: None,
        rasterization_rate_map: None,
    });
    let executed = |buffer_binding| CommandBufferResource {
        buffer_binding,
        field_offset: 0,
        capacity: 1,
        execution_range: [0, 1],
        inherit_pipeline_state: false,
        inherit_buffers: false,
        descriptor: CommandDescriptor::Compute {
            dispatch: ComputeDispatch::Both,
            max_buffer_bind_count: 1,
            max_threadgroup_memory_bind_count: 0,
            support_dynamic_attribute_stride: false,
        },
        commands: vec![],
        execute: Some(Execution {
            pipeline: None,
            buffers: vec![],
            vertex_buffers: vec![],
            fragment_buffers: vec![],
            object_buffers: vec![],
            mesh_buffers: vec![],
        }),
    };
    let mut produced = executed(2);
    produced.execute = None;
    case.buffers.push(BufferResource {
        binding: 2,
        role: ResourceRole::Input,
        bytes_b64: Some(base64::engine::general_purpose::STANDARD.encode([0u8; 16])),
        initial_bytes_b64: None,
        rasterization_rate_map: None,
    });
    case.command_buffers = vec![executed(1), produced, executed(0)];
    case.validate_literal_resources().unwrap();
    let keys = |case: &AuthoredCase| {
        executed_in_key_order(&case.command_buffers)
            .into_iter()
            .map(|resource| (resource.buffer_binding, resource.field_offset))
            .collect::<Vec<_>>()
    };
    assert_eq!(keys(&case), [(0, 0), (1, 0)]);
    let mut reversed = case.clone();
    reversed.command_buffers.reverse();
    assert_eq!(
        reversed.computed_case_id().unwrap(),
        case.computed_case_id().unwrap()
    );
    assert_eq!(keys(&reversed), keys(&case));
    // The candidate's map iterates the same keys in the same order.
    for listing in [&case, &reversed] {
        let mapped = crate::candidate::engine::compute_case(listing, &kernel_interface()).unwrap();
        assert_eq!(
            mapped
                .command_buffers
                .iter()
                .filter(|(_, resource)| resource.execute.is_some())
                .map(|(&key, _)| key)
                .collect::<Vec<_>>(),
            keys(&case)
        );
    }
}

/// A render descriptor of mesh draws: its draw kinds and object and mesh counts serialize only
/// where set, so the cases written before them keep their identities, and change a case's
/// identity where set; an execution's parent object and mesh buffers are admitted only on a
/// buffer that allows mesh draws and inherits buffers, and bind at their own stages; a
/// host-encoded barrier needs mesh draws, and on a buffer of them is not modelled.
#[test]
fn mesh_draw_descriptors_keep_identities_and_admit_object_and_mesh_parents() {
    use crate::case::indirect_commands::{
        CommandBinding, CommandBufferResource, CommandDescriptor, Execution, HostCommand,
        PipelineTemplate, RenderDraws,
    };
    use metal2vulkan::bind::indirect_commands::BufferStage;
    use std::num::NonZeroU64;
    let mut case = example();
    case.case_id = "22".repeat(32);
    case.buffers = vec![
        BufferResource {
            binding: 0,
            role: ResourceRole::Input,
            bytes_b64: Some(base64::engine::general_purpose::STANDARD.encode([0u8; 8])),
            initial_bytes_b64: None,
            rasterization_rate_map: None,
        },
        BufferResource {
            binding: 1,
            role: ResourceRole::Output,
            bytes_b64: None,
            initial_bytes_b64: Some(base64::engine::general_purpose::STANDARD.encode([0u8; 512])),
            rasterization_rate_map: None,
        },
    ];
    case.output = OutputSelection::Buffer {
        binding: 1,
        offset: 0,
        length: 4,
    };
    let primitives = CommandDescriptor::Render {
        draws: RenderDraws::default(),
        max_vertex_buffer_bind_count: 1,
        max_fragment_buffer_bind_count: 0,
        max_object_buffer_bind_count: 0,
        max_mesh_buffer_bind_count: 0,
        support_dynamic_attribute_stride: false,
    };
    let value = serde_json::to_value(&primitives).unwrap();
    for key in ["max_object_buffer_bind_count", "max_mesh_buffer_bind_count"] {
        assert!(value.get(key).is_none(), "{key}");
    }
    let binding = |index| CommandBinding {
        index,
        buffer: 1,
        offset: 256,
        attribute_stride: None,
    };
    let parent = Execution {
        pipeline: Some(PipelineTemplate {
            token: 4,
            air_sha256: "55".repeat(32),
            case_id: "66".repeat(32),
        }),
        buffers: vec![],
        vertex_buffers: vec![],
        fragment_buffers: vec![],
        object_buffers: vec![],
        mesh_buffers: vec![],
    };
    let value = serde_json::to_value(&parent).unwrap();
    for key in ["object_buffers", "mesh_buffers"] {
        assert!(value.get(key).is_none(), "{key}");
    }
    let mesh_draws = RenderDraws {
        primitives: false,
        indexed: false,
        patches: false,
        indexed_patches: false,
        mesh_threadgroups: true,
        mesh_threads: false,
    };
    case.command_buffers.push(CommandBufferResource {
        buffer_binding: 0,
        field_offset: 0,
        capacity: 1,
        execution_range: [0, 1],
        inherit_pipeline_state: true,
        inherit_buffers: true,
        descriptor: CommandDescriptor::Render {
            draws: mesh_draws,
            max_vertex_buffer_bind_count: 0,
            max_fragment_buffer_bind_count: 0,
            max_object_buffer_bind_count: 1,
            max_mesh_buffer_bind_count: 2,
            support_dynamic_attribute_stride: false,
        },
        commands: vec![],
        execute: Some(Execution {
            object_buffers: vec![binding(0)],
            mesh_buffers: vec![binding(1)],
            ..parent
        }),
    });
    case.validate_literal_resources().unwrap();
    let descriptor = serde_json::to_value(&case.command_buffers[0].descriptor).unwrap();
    assert_eq!(
        descriptor["draws"],
        serde_json::json!({"primitives": false, "indexed": false, "patches": false,
            "mesh_threadgroups": true})
    );
    assert_eq!(descriptor["max_mesh_buffer_bind_count"], 2);
    let identity = case.computed_case_id().unwrap();
    let mut counted = case.clone();
    if let CommandDescriptor::Render {
        max_object_buffer_bind_count,
        ..
    } = &mut counted.command_buffers[0].descriptor
    {
        *max_object_buffer_bind_count = 2;
    }
    assert_ne!(counted.computed_case_id().unwrap(), identity);

    // The parent's object and mesh buffers bind at their own stages.
    let address = |buffer: u32| (buffer == 1).then_some(0x2_0000_0000);
    let state = case.command_buffers[0]
        .parent_bindings(address)
        .unwrap()
        .unwrap();
    assert_eq!(state.pipeline, NonZeroU64::new(4));
    assert_eq!(
        state.buffers(BufferStage::Object)[0].map(|b| b.address),
        Some(0x2_0000_0100)
    );
    assert_eq!(
        state.buffers(BufferStage::Mesh)[1].map(|b| b.address),
        Some(0x2_0000_0100)
    );

    let rejects = |change: &dyn Fn(&mut CommandBufferResource), what: &str| {
        let mut variant = case.clone();
        change(&mut variant.command_buffers[0]);
        let errors = variant.validate_literal_resources().unwrap_err();
        assert!(
            errors.iter().any(|error| error.contains(what)),
            "{what}: {errors:?}"
        );
    };
    // On a buffer of no mesh draws, the parent's object and mesh buffers bind no stage its
    // commands run.
    let without_mesh = |resource: &mut CommandBufferResource| {
        if let CommandDescriptor::Render { draws, .. } = &mut resource.descriptor {
            *draws = RenderDraws::default();
        }
    };
    rejects(
        &without_mesh,
        "the parent encoder's object buffers are no stage",
    );
    rejects(
        &|resource| {
            without_mesh(resource);
            if let Some(execution) = &mut resource.execute {
                execution.object_buffers.clear();
            }
        },
        "the parent encoder's mesh buffers are no stage",
    );
    rejects(
        &|resource| resource.inherit_buffers = false,
        "an execution's parent buffers need inherited buffers",
    );
    let barrier = HostCommand {
        index: 0,
        action: None,
        barrier: true,
        pipeline: None,
        buffers: vec![],
        threadgroup_memory_lengths: vec![],
        stage_in_region: None,
        vertex_buffers: vec![],
        fragment_buffers: vec![],
    };
    rejects(
        &|resource| resource.commands = vec![barrier.clone()],
        "barrier on a buffer of mesh draws is not modelled",
    );
    rejects(
        &|resource| {
            without_mesh(resource);
            if let Some(execution) = &mut resource.execute {
                execution.object_buffers.clear();
                execution.mesh_buffers.clear();
            }
            resource.commands = vec![barrier.clone()];
        },
        "barrier needs mesh draws",
    );
}
