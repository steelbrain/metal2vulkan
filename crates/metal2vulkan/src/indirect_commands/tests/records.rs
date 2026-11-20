use super::*;

/// A reset record is zero throughout, its buffer slots' strides too, as Metal's reset leaves a
/// slot's getters answering 0 on a buffer of dynamic attribute strides.
#[test]
fn reset_clears_action_and_bindings_to_zero() {
    let mut bytes = vec![0xa5; COMMAND_BYTES];
    CommandRecord::default().encode_into(&mut bytes).unwrap();
    assert_eq!(
        CommandRecord::decode(&bytes).unwrap(),
        CommandRecord::default()
    );
    assert!(bytes.iter().all(|b| *b == 0));
}

/// A compute command's reset keeps the pipeline it executes with, reported cleared: the pipeline
/// word holds it and the state word flags it, `decode` reports it as `retained_pipeline` beside no
/// set pipeline, and `resolve_bindings` executes it. A retained pipeline beside a set one does
/// not encode; a state word with a bit other than `RETAINED` and `KERNEL_PIPELINE`, or
/// `RETAINED` without a pipeline, does not decode.
#[test]
fn a_retained_pipeline_runs_and_reads_back_cleared() {
    let token = NonZeroU64::new(0x0123_4567_89ab_cdef);
    let record = CommandRecord {
        action: Action::DispatchThreads {
            grid: [8, 1, 1],
            local_size: [8, 1, 1],
        },
        retained_pipeline: token,
        ..Default::default()
    };
    let mut bytes = vec![0; COMMAND_BYTES];
    record.encode_into(&mut bytes).unwrap();
    assert_eq!(&bytes[8..16], &token.unwrap().get().to_le_bytes());
    let state = layout::STATE as usize;
    assert_eq!(&bytes[state..state + 4], &layout::RETAINED.to_le_bytes());
    let decoded = CommandRecord::decode(&bytes).unwrap();
    assert_eq!(decoded, record);
    assert_eq!(decoded.bindings.pipeline, None);
    let owned = Inheritance {
        pipeline: StateSource::Command,
        buffers: StateSource::Command,
    };
    let parent = Bindings::default();
    assert_eq!(
        decoded.resolve_bindings(owned, &parent).unwrap().pipeline,
        token
    );
    // A set pipeline is reported, and its state word is 0.
    let set = CommandRecord {
        bindings: Bindings {
            pipeline: token,
            ..Default::default()
        },
        ..Default::default()
    };
    set.encode_into(&mut bytes).unwrap();
    assert_eq!(&bytes[state..state + 4], &[0; 4]);
    assert_eq!(CommandRecord::decode(&bytes).unwrap(), set);
    let both = CommandRecord {
        retained_pipeline: token,
        ..set
    };
    let before = bytes.clone();
    assert_eq!(
        both.encode_into(&mut bytes),
        Err(RecordError::PipelineState)
    );
    assert_eq!(bytes, before);
    put_word(&mut bytes, state / 4, 4);
    assert_eq!(
        CommandRecord::decode(&bytes),
        Err(RecordError::PipelineState)
    );
    CommandRecord::default().encode_into(&mut bytes).unwrap();
    put_word(&mut bytes, state / 4, layout::RETAINED);
    assert_eq!(
        CommandRecord::decode(&bytes),
        Err(RecordError::PipelineState)
    );
}

/// A pipeline a kernel wrote (`CommandRecord::kernel_pipeline`) is flagged in the state word
/// (`layout::KERNEL_PIPELINE`), set or retained, and decodes as written. Flagged without a
/// pipeline, it neither encodes nor decodes.
#[test]
fn a_kernel_s_pipeline_is_told_from_the_cpu_s() {
    let token = NonZeroU64::new(0x0123_4567_89ab_cdef);
    let state = layout::STATE as usize;
    let written = CommandRecord {
        bindings: Bindings {
            pipeline: token,
            ..Default::default()
        },
        kernel_pipeline: true,
        ..Default::default()
    };
    let retained = CommandRecord {
        bindings: Bindings::default(),
        retained_pipeline: token,
        ..written.clone()
    };
    let mut bytes = vec![0; COMMAND_BYTES];
    for (record, word) in [
        (&written, layout::KERNEL_PIPELINE),
        (&retained, layout::RETAINED | layout::KERNEL_PIPELINE),
    ] {
        record.encode_into(&mut bytes).unwrap();
        assert_eq!(&bytes[state..state + 4], &word.to_le_bytes());
        assert_eq!(&CommandRecord::decode(&bytes).unwrap(), record);
        assert!(RecordView::new(&bytes).unwrap().kernel_pipeline());
    }
    // Flagged beside no pipeline.
    let unnamed = CommandRecord {
        kernel_pipeline: true,
        ..Default::default()
    };
    let before = bytes.clone();
    assert_eq!(
        unnamed.encode_into(&mut bytes),
        Err(RecordError::PipelineState)
    );
    assert_eq!(bytes, before);
    CommandRecord::default().encode_into(&mut bytes).unwrap();
    put_word(&mut bytes, state / 4, layout::KERNEL_PIPELINE);
    assert_eq!(
        CommandRecord::decode(&bytes),
        Err(RecordError::PipelineState)
    );
}

#[test]
fn stage_namespaces_and_full_width_addresses_survive_round_trip() {
    for action in [
        Action::Empty,
        Action::DispatchThreadgroups {
            grid: [2, 3, 4],
            local_size: [5, 6, 7],
        },
        Action::DispatchThreads {
            grid: [11, 13, 17],
            local_size: [2, 3, 4],
        },
        Action::DrawPrimitives {
            primitive_type: 3,
            vertex_start: 7,
            vertex_count: 19,
            instance_count: 2,
            base_instance: 23,
        },
    ] {
        let mut record = CommandRecord {
            action,
            barrier: true,
            bindings: Bindings {
                pipeline: NonZeroU64::new(0x1234_5678_9876_5432),
                ..Default::default()
            },
            ..Default::default()
        };
        for (stage, address) in [
            (BufferStage::Kernel, 0x1111_2222_3333_4444),
            (BufferStage::Vertex, 0x5555_6666_7777_8888),
            (BufferStage::Fragment, 0x9999_aaaa_bbbb_cccc),
            (BufferStage::Object, 0xdddd_eeee_ffff_1234),
        ] {
            record.bindings.buffers_mut(stage)[31] = Some(BufferBinding {
                address,
                attribute_stride: Some(128),
            });
            assert_eq!(
                record.bindings.buffers(stage)[31].map(|b| b.address),
                Some(address)
            );
        }
        record.threadgroup_memory_lengths[THREADGROUP_COUNT - 1] = 4096;
        record.stage_in_region = [1, 2, 3, 4, 5, u32::MAX];
        record.imageblock_size = [32, u16::MAX];
        let mut bytes = vec![0; COMMAND_BYTES];
        record.encode_into(&mut bytes).unwrap();
        assert_eq!(CommandRecord::decode(&bytes).unwrap(), record);
    }
}

/// A record keeps any threadgroup-memory length, as Metal's commands do (the probe 1: 20 and past
/// the device's maximum read back as set).
#[test]
fn threadgroup_lengths_round_trip_as_set() {
    for length in [0, 1, 15, 16, 17, 20, u32::MAX - 15, u32::MAX] {
        let record = CommandRecord {
            threadgroup_memory_lengths: [length; THREADGROUP_COUNT],
            ..Default::default()
        };
        let mut bytes = vec![0; COMMAND_BYTES];
        record.encode_into(&mut bytes).unwrap();
        assert_eq!(CommandRecord::decode(&bytes).unwrap(), record);
    }
}

#[test]
fn malformed_records_are_rejected_before_host_replay() {
    let mut bytes = vec![0; COMMAND_BYTES];
    assert!(matches!(
        CommandRecord::decode(&bytes[..COMMAND_BYTES - 1]),
        Err(RecordError::Length { .. })
    ));
    // Tags 0 to 8 are actions (5 a patch draw; 6 an indexed patch draw; 7 and 8 mesh draws); 9 is
    // none.
    put_word(&mut bytes, 0, 9);
    assert_eq!(
        CommandRecord::decode(&bytes),
        Err(RecordError::Action { value: 9 })
    );
    put_word(&mut bytes, 0, 0);
    put_word(&mut bytes, 1, 2);
    assert_eq!(
        CommandRecord::decode(&bytes),
        Err(RecordError::Barrier { value: 2 })
    );
    put_word(&mut bytes, 1, 0);
    let mut record = CommandRecord::default();
    record.bindings.buffers_mut(BufferStage::Kernel)[0] = Some(BufferBinding {
        address: 0,
        attribute_stride: Some(u64::MAX),
    });
    let before = bytes.clone();
    assert_eq!(
        record.encode_into(&mut bytes),
        Err(RecordError::ReservedStride)
    );
    assert_eq!(bytes, before);
}

/// A render descriptor's header carries the render and draw bits and the vertex and fragment
/// binding counts, and its records may draw but neither dispatch nor bind kernel buffers.
#[test]
fn a_render_descriptor_allows_draws_and_refuses_dispatches() {
    let render = CommandDescriptor::Render {
        draws: RenderDraws::PRIMITIVES,
        max_vertex_buffer_bind_count: 2,
        max_fragment_buffer_bind_count: 3,
        max_object_buffer_bind_count: 0,
        max_mesh_buffer_bind_count: 0,
        support_dynamic_attribute_stride: false,
    };
    let header = render.encode_header(4, OWNED_STATE).unwrap();
    let word = |index: usize| u32::from_le_bytes(header[index * 4..][..4].try_into().unwrap());
    assert_eq!(word(0), 4);
    assert_eq!(word(2), RENDER | DRAW);
    assert_eq!(word(3), (2 << 8) | (3 << 16));
    let draw = CommandRecord {
        action: Action::DrawPrimitives {
            primitive_type: 3,
            vertex_start: 0,
            vertex_count: 3,
            instance_count: 1,
            base_instance: 0,
        },
        ..Default::default()
    };
    assert_eq!(render.validate_record(&draw), Ok(()));
    let dispatch = CommandRecord {
        action: Action::DispatchThreads {
            grid: [1; 3],
            local_size: [1; 3],
        },
        ..Default::default()
    };
    assert!(render.validate_record(&dispatch).is_err());
    let mut kernel = CommandRecord::default();
    kernel.bindings.buffers_mut(BufferStage::Kernel)[0] = Some(BufferBinding {
        address: 8,
        attribute_stride: None,
    });
    assert!(render.validate_record(&kernel).is_err());
}

/// An indexed draw and a patch draw round-trip with full-width addresses and a negative base
/// vertex, their arguments in the stage-in region's words, which they leave unset.
#[test]
fn indexed_and_patch_draws_round_trip_in_the_stage_in_words() {
    let indexed = Action::DrawIndexedPrimitives {
        primitive_type: 3,
        index_count: 6,
        index_type: 1,
        index_buffer: 0x8765_4321_0fed_cba8,
        instance_count: 2,
        base_vertex: -5,
        base_instance: 9,
    };
    let patches = Action::DrawPatches {
        control_points: 4,
        patch_start: 1,
        patch_count: 7,
        patch_index_buffer: 0x1234_5678_9abc_def0,
        instance_count: 3,
        base_instance: 2,
        factor_buffer: 0xfedc_ba98_7654_3210,
        factor_instance_stride: 48,
        factor_scale: 0.5f32.to_bits(),
    };
    for (action, tag) in [(indexed, 4), (patches, 5)] {
        let record = CommandRecord {
            action,
            bindings: Bindings {
                pipeline: NonZeroU64::new(0x42),
                ..Default::default()
            },
            ..Default::default()
        };
        let mut bytes = vec![0; COMMAND_BYTES];
        record.encode_into(&mut bytes).unwrap();
        assert_eq!(host::get_word(&bytes, 0), tag);
        assert_eq!(CommandRecord::decode(&bytes).unwrap(), record);
        // A stage-in region beside it is refused, writing nothing.
        let staged = CommandRecord {
            stage_in_region: [1, 0, 0, 0, 0, 0],
            ..record
        };
        let before = bytes.clone();
        assert_eq!(
            staged.encode_into(&mut bytes),
            Err(RecordError::StageInRegion)
        );
        assert_eq!(bytes, before);
    }
    let mut bytes = vec![0; COMMAND_BYTES];
    let record = CommandRecord {
        action: indexed,
        ..Default::default()
    };
    record.encode_into(&mut bytes).unwrap();
    // Primitive type, count, index type, instances, the address's low and high words, the base
    // vertex and the base instance, from word 4.
    let words: Vec<u32> = (4..12).map(|word| host::get_word(&bytes, word)).collect();
    assert_eq!(
        words,
        [3, 6, 1, 2, 0x0fed_cba8, 0x8765_4321, -5i32 as u32, 9]
    );
}

/// Mesh draws of threadgroups (tag 7) and of threads (tag 8) round-trip, their grid and the
/// threads of an object and of a mesh threadgroup in the stage-in region's words, which they leave
/// unset; a stage-in region beside one is refused, writing nothing. A render record's mesh
/// buffers take the words of a compute record's kernel buffers.
#[test]
fn mesh_draws_round_trip_in_the_stage_in_words() {
    assert_eq!(
        layout::buffers(BufferStage::Mesh),
        layout::buffers(BufferStage::Kernel)
    );
    assert_eq!(BufferStage::of(true)[0], BufferStage::Mesh);
    let groups = Action::DrawMeshThreadgroups {
        threadgroups: [3, 2, u32::MAX],
        object_threads: [4, 1, 1],
        mesh_threads: [32, 2, 1],
    };
    let threads = Action::DrawMeshThreads {
        threads: [12, 4, 1],
        object_threads: [1, 1, 1],
        mesh_threads: [4, 2, 1],
    };
    for (action, tag, words) in [
        (groups, 7, [3, 2, u32::MAX, 4, 1, 1, 32, 2, 1]),
        (threads, 8, [12, 4, 1, 1, 1, 1, 4, 2, 1]),
    ] {
        let mut record = CommandRecord {
            action,
            barrier: true,
            ..Default::default()
        };
        record.bindings.buffers_mut(BufferStage::Object)[0] = Some(BufferBinding {
            address: 0x1_0010_0000,
            attribute_stride: None,
        });
        record.bindings.buffers_mut(BufferStage::Mesh)[2] = Some(BufferBinding {
            address: 0x1_0020_0040,
            attribute_stride: None,
        });
        let mut bytes = vec![0; COMMAND_BYTES];
        record.encode_into(&mut bytes).unwrap();
        assert_eq!(host::get_word(&bytes, 0), tag);
        let at: Vec<u32> = (4..13).map(|word| host::get_word(&bytes, word)).collect();
        assert_eq!(at, words);
        assert_eq!(CommandRecord::decode(&bytes).unwrap(), record);
        let staged = CommandRecord {
            stage_in_region: [0, 0, 0, 1, 1, 1],
            ..record
        };
        let before = bytes.clone();
        assert_eq!(
            staged.encode_into(&mut bytes),
            Err(RecordError::StageInRegion)
        );
        assert_eq!(bytes, before);
    }
}

/// An indexed patch draw round-trips: a patch draw's arguments in its words, the control-point
/// index buffer's full-width address in the first two threadgroup lengths' words, which it leaves
/// unset. A stage-in region or one of those two lengths beside it is refused, writing nothing; a
/// later threadgroup length is kept apart from the address.
#[test]
fn an_indexed_patch_draw_round_trips_its_control_point_index_buffer() {
    let record = CommandRecord {
        action: Action::DrawIndexedPatches {
            control_points: 5,
            patch_start: 7,
            patch_count: 11,
            patch_index_buffer: 0x1234_5678_9abc_def0,
            control_point_index_buffer: 0x0bad_cafe_dead_beef,
            instance_count: 13,
            base_instance: 17,
            factor_buffer: 0xfedc_ba98_7654_3210,
            factor_instance_stride: 44,
            factor_scale: 1.0f32.to_bits(),
        },
        bindings: Bindings {
            pipeline: NonZeroU64::new(0x42),
            ..Default::default()
        },
        ..Default::default()
    };
    let mut bytes = vec![0; COMMAND_BYTES];
    record.encode_into(&mut bytes).unwrap();
    assert_eq!(host::get_word(&bytes, 0), 6);
    let words: Vec<u32> = (4..15).map(|word| host::get_word(&bytes, word)).collect();
    assert_eq!(
        words,
        [
            5,
            7,
            11,
            13,
            0x9abc_def0,
            0x1234_5678,
            0x7654_3210,
            0xfedc_ba98,
            17,
            44,
            0x3f80_0000
        ]
    );
    let at = layout::CONTROL_POINT_INDICES as usize / 4;
    assert_eq!(
        [host::get_word(&bytes, at), host::get_word(&bytes, at + 1)],
        [0xdead_beef, 0x0bad_cafe]
    );
    assert_eq!(CommandRecord::decode(&bytes).unwrap(), record);
    let before = bytes.clone();
    let staged = CommandRecord {
        stage_in_region: [0, 0, 0, 0, 0, 1],
        ..record.clone()
    };
    assert_eq!(
        staged.encode_into(&mut bytes),
        Err(RecordError::StageInRegion)
    );
    for index in 0..2 {
        let mut lengths = [0; THREADGROUP_BUFFER_ARGUMENT_COUNT as usize];
        lengths[index] = 16;
        let set = CommandRecord {
            threadgroup_memory_lengths: lengths,
            ..record.clone()
        };
        assert_eq!(
            set.encode_into(&mut bytes),
            Err(RecordError::ThreadgroupMemoryLength)
        );
    }
    assert_eq!(bytes, before);
    let mut lengths = [0; THREADGROUP_BUFFER_ARGUMENT_COUNT as usize];
    lengths[2] = 32;
    let later = CommandRecord {
        threadgroup_memory_lengths: lengths,
        ..record
    };
    later.encode_into(&mut bytes).unwrap();
    assert_eq!(CommandRecord::decode(&bytes).unwrap(), later);
}

/// A render descriptor's header carries a bit per draw kind it allows, and its records may draw
/// those kinds and the kinds their records hold; one that allows none is refused.
#[test]
fn a_render_descriptor_allows_the_draw_kinds_it_names() {
    let indexed = CommandRecord {
        action: Action::DrawIndexedPrimitives {
            primitive_type: 3,
            index_count: 3,
            index_type: 0,
            index_buffer: 64,
            instance_count: 1,
            base_vertex: 0,
            base_instance: 0,
        },
        ..Default::default()
    };
    let patches = CommandRecord {
        action: Action::DrawPatches {
            control_points: 3,
            patch_start: 0,
            patch_count: 1,
            patch_index_buffer: 0,
            instance_count: 1,
            base_instance: 0,
            factor_buffer: 64,
            factor_instance_stride: 0,
            factor_scale: 1.0f32.to_bits(),
        },
        ..Default::default()
    };
    let primitives = CommandRecord {
        action: Action::DrawPrimitives {
            primitive_type: 3,
            vertex_start: 0,
            vertex_count: 3,
            instance_count: 1,
            base_instance: 0,
        },
        ..Default::default()
    };
    let indexed_patches = CommandRecord {
        action: Action::DrawIndexedPatches {
            control_points: 3,
            patch_start: 0,
            patch_count: 1,
            patch_index_buffer: 0,
            control_point_index_buffer: 128,
            instance_count: 1,
            base_instance: 0,
            factor_buffer: 64,
            factor_instance_stride: 0,
            factor_scale: 1.0f32.to_bits(),
        },
        ..Default::default()
    };
    let descriptor = |draws| CommandDescriptor::Render {
        draws,
        max_vertex_buffer_bind_count: 1,
        max_fragment_buffer_bind_count: 1,
        max_object_buffer_bind_count: 0,
        max_mesh_buffer_bind_count: 0,
        support_dynamic_attribute_stride: false,
    };
    for (draws, bits) in [
        (
            RenderDraws {
                primitives: false,
                indexed: true,
                patches: false,
                indexed_patches: false,
                mesh_threadgroups: false,
                mesh_threads: false,
            },
            DRAW_INDEXED,
        ),
        (
            RenderDraws {
                primitives: false,
                indexed: false,
                patches: true,
                indexed_patches: false,
                mesh_threadgroups: false,
                mesh_threads: false,
            },
            DRAW_PATCHES,
        ),
        (
            RenderDraws {
                primitives: true,
                indexed: true,
                patches: true,
                indexed_patches: false,
                mesh_threadgroups: false,
                mesh_threads: false,
            },
            DRAW | DRAW_INDEXED | DRAW_PATCHES,
        ),
        (
            RenderDraws {
                primitives: false,
                indexed: false,
                patches: false,
                indexed_patches: true,
                mesh_threadgroups: false,
                mesh_threads: false,
            },
            DRAW_INDEXED_PATCHES,
        ),
    ] {
        let render = descriptor(draws);
        let header = render.encode_header(1, OWNED_STATE).unwrap();
        assert_eq!(host::get_word(&header, 2), RENDER | bits);
        // A buffer's records hold a draw of a kind its draws allow, or of a kind whose record
        // one they allow holds: primitives in every one, indexed draws in an indexed or a patch
        // buffer's, patch and indexed patch draws in a patch buffer's.
        let patch_records = draws.patches || draws.indexed_patches;
        for (record, allowed) in [
            (&primitives, true),
            (&indexed, draws.indexed || patch_records),
            (&patches, patch_records),
            (&indexed_patches, patch_records),
        ] {
            assert_eq!(render.validate_record(record).is_ok(), allowed);
        }
    }
    let none = descriptor(RenderDraws {
        primitives: false,
        indexed: false,
        patches: false,
        indexed_patches: false,
        mesh_threadgroups: false,
        mesh_threads: false,
    });
    assert!(none.encode_header(1, OWNED_STATE).is_err());
}

/// A slot's words hold what Metal's getters answer on a buffer of dynamic attribute strides: an
/// address and a stride of 0 where no setter set it, a nil buffer's offset as its address, all ones
/// for the pipeline's stride and the stride of the stride form; each decodes back, and the words of
/// 0 decode as no setter's.
#[test]
fn buffer_slots_hold_the_answers_of_metal_s_getters() {
    let slot = |address, attribute_stride| {
        Some(BufferBinding {
            address,
            attribute_stride,
        })
    };
    let mut record = CommandRecord::default();
    let kernel = record.bindings.buffers_mut(BufferStage::Kernel);
    kernel[1] = slot(0x30, None);
    kernel[2] = slot(0, None);
    kernel[3] = slot(0x30, Some(48));
    kernel[4] = slot(0x1_0000_0030, Some(0));
    kernel[5] = slot(0x1_0000_0030, None);
    let mut bytes = vec![0xa5; COMMAND_BYTES];
    record.encode_into(&mut bytes).unwrap();
    let double = |bytes: &[u8], word: usize| {
        u64::from(host::get_word(bytes, word)) | (u64::from(host::get_word(bytes, word + 1)) << 32)
    };
    let words = |index: usize| {
        let word = HEADER_WORDS + index * BUFFER_WORDS;
        (double(&bytes, word), double(&bytes, word + 2))
    };
    assert_eq!(words(0), (0, 0), "no setter's");
    assert_eq!(words(1), (0x30, u64::MAX), "a nil buffer at 0x30");
    assert_eq!(words(2), (0, u64::MAX), "a nil buffer at 0");
    assert_eq!(words(3), (0x30, 48), "a nil buffer at 0x30, stride 48");
    assert_eq!(words(4), (0x1_0000_0030, 0), "a buffer at stride 0");
    assert_eq!(words(5), (0x1_0000_0030, u64::MAX), "a buffer");
    assert_eq!(CommandRecord::decode(&bytes).unwrap(), record);
    let names: Vec<bool> = record.bindings.buffers(BufferStage::Kernel)[..6]
        .iter()
        .map(|b| b.is_some_and(|b| b.names_buffer()))
        .collect();
    assert_eq!(names, [false, false, false, false, true, true]);
    // A nil buffer set at offset 0 at stride 0 holds a fresh slot's words, and reads as one, as
    // Metal's getters answer it.
    record.bindings.buffers_mut(BufferStage::Kernel)[3] = slot(0, Some(0));
    record.encode_into(&mut bytes).unwrap();
    assert_eq!(
        CommandRecord::decode(&bytes)
            .unwrap()
            .bindings
            .buffers(BufferStage::Kernel)[3],
        None
    );
}
