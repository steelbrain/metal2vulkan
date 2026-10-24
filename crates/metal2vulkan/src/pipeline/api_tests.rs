//! Unit tests of the public API types: constructors and serde round trips under the `serde`
//! feature. `Send + Sync`, `#[non_exhaustive]`, and the frozen names are pinned from outside the
//! crate by `tests/api_surface.rs`.

use std::sync::Arc;

use super::*;
use crate::cache::{PipelineKey, VariantKey};
use crate::pipeline::sym::{SlotClass, SlotRef, SpecKey, Stage};
use crate::{metal, vk};

fn slot(stage: Stage, class: SlotClass, index: u32) -> SlotRef {
    SlotRef {
        stage,
        class,
        index,
    }
}

#[test]
fn keys_and_inputs_start_empty() {
    assert!(VariantKey::new().is_empty());
    let mut v = VariantKey::new();
    v.psb_slots
        .insert(slot(Stage::Kernel, SlotClass::Buffer, 3));
    assert!(!v.is_empty());
    assert_eq!(LoweringInputs::new(), LoweringInputs::default());
    let key = PipelineKey::new([0xab; 32]);
    assert_eq!(key.digest(), &[0xab; 32]);
    assert!(format!("{key:?}").starts_with("PipelineKey(abab"));
}

#[test]
fn stage_module_bytes_are_little_endian_words() {
    let m = StageModule {
        stage: vk::ShaderStage::COMPUTE,
        entry_point: "main".into(),
        spirv: Arc::from(vec![0x0723_0203u32, 0x0001_0500]),
        specialization: SpecializationLayout::default(),
        required_subgroup_size: None,
        stage_flags: vk::PipelineShaderStageCreateFlags::empty(),
        reconvergence: Reconvergence::Guaranteed,
        precision_notes: Vec::new(),
        strategies: StrategyCensus::default(),
        constant_blob: None,
        rounded_colors: Vec::new(),
        half_bit_colors: Vec::new(),
        own_pixel_textures: Vec::new(),
        snapshot_textures: Vec::new(),
        private_bytes: 0,
        specialized_notes: Vec::new(),
    };
    assert_eq!(
        m.spirv_bytes(),
        [0x03, 0x02, 0x23, 0x07, 0x00, 0x05, 0x01, 0x00]
    );
}

/// A stage's notes under a specialization: every note no specialized entry names, and each
/// specialized note one of whose clauses holds, a constant the key leaves out (or gives under
/// another id) reading its default.
#[test]
fn a_stage_s_notes_under_a_specialization_are_its_held_clauses() {
    use crate::cache::SpecializationKey;
    let texture = slot(Stage::Kernel, SlotClass::Texture, 0);
    let sampler = slot(Stage::Kernel, SlotClass::Sampler, 0);
    let texel = SpecKey::TexelFormat(texture);
    let word = SpecKey::SamplerState {
        slot: sampler,
        fields: 3,
    };
    let constant = |id, key| SpecConstant {
        id,
        key,
        size: 4,
        default: 0,
        unread_bits: 0,
    };
    let m = StageModule {
        stage: vk::ShaderStage::COMPUTE,
        entry_point: "main".into(),
        spirv: Arc::from(Vec::new()),
        specialization: SpecializationLayout {
            constants: vec![constant(3, texel), constant(4, word)],
        },
        required_subgroup_size: None,
        stage_flags: vk::PipelineShaderStageCreateFlags::empty(),
        reconvergence: Reconvergence::Guaranteed,
        precision_notes: vec![
            PrecisionNote::GradFromBias,
            PrecisionNote::PixelSamplerScaling,
            PrecisionNote::HalfSampleFilter,
        ],
        strategies: StrategyCensus::default(),
        constant_blob: None,
        rounded_colors: Vec::new(),
        half_bit_colors: Vec::new(),
        own_pixel_textures: Vec::new(),
        snapshot_textures: Vec::new(),
        private_bytes: 0,
        specialized_notes: vec![
            SpecializedNote {
                note: PrecisionNote::PixelSamplerScaling,
                when: vec![
                    SpecClause {
                        all: vec![SpecTest::differ(word, 1, 0)],
                    },
                    SpecClause {
                        all: vec![SpecTest::equal(texel, 1, 1), SpecTest::equal(word, 2, 2)],
                    },
                ],
            },
            SpecializedNote {
                note: PrecisionNote::HalfSampleFilter,
                when: vec![SpecClause {
                    all: vec![SpecTest::equal(texel, 0x109, 0)],
                }],
            },
        ],
    };
    let key = |values: &[(u32, u32)]| {
        let mut k = SpecializationKey::new();
        for &(id, v) in values {
            k.entries.push(vk::SpecializationMapEntry {
                constant_id: id,
                offset: k.data.len() as u32,
                size: 4,
            });
            k.data.extend(v.to_le_bytes());
        }
        k
    };
    use PrecisionNote::{GradFromBias as G, HalfSampleFilter as H, PixelSamplerScaling as P};
    for (values, want) in [
        // Every default: the class 0 takes the filter note, the word 0 no scaling.
        (vec![], vec![G, H]),
        (vec![(3, 0x100)], vec![G]),
        (vec![(3, 0x100), (4, 1)], vec![G, P]),
        (vec![(3, 1), (4, 2)], vec![G, P]),
        (vec![(3, 1), (4, 0)], vec![G]),
        // A constant under another id leaves the stage's at its default.
        (vec![(9, 0x100), (4, 0)], vec![G, H]),
    ] {
        let mut sorted = m.precision_notes.clone();
        sorted.retain(|n| want.contains(n));
        assert_eq!(m.precision_notes_under(&key(&values)), sorted, "{values:?}");
    }
}

/// One populated value of each serde-enabled pipeline and cache type.
#[cfg(feature = "serde")]
mod serde_round_trips {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};

    use crate::cache::{ArtifactError, SpecializationKey};
    use crate::pipeline::facts::{Mechanism, RuntimeFact};
    use crate::pipeline::sym::{SampledType, SpecKey, SyntheticBinding};

    fn round_trip<T>(value: &T)
    where
        T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
    {
        let json = serde_json::to_string(value).expect("serializes");
        let back: T = serde_json::from_str(&json).expect("deserializes");
        assert_eq!(&back, value, "{json}");
    }

    fn buffer(index: u32) -> SlotRef {
        slot(Stage::Fragment, SlotClass::Buffer, index)
    }

    #[test]
    fn cache_keys() {
        round_trip(&PipelineKey::new([7; 32]));
        let mut storage_images = BTreeMap::new();
        storage_images.insert(
            slot(Stage::Kernel, SlotClass::Texture, 2),
            StorageImageChoice {
                vk_format: vk::Format::R8G8B8A8_UNORM,
                class: StorageImageClass::SwizzleBgra,
                alias: Some(AliasView::new(vk::Format::B8G8R8A8_UNORM)),
                numeric: None,
                x_channel: false,
            },
        );
        let mut formatless =
            StorageImageChoice::new(vk::Format::UNDEFINED, StorageImageClass::Identity);
        formatless.numeric = Some(SampledType::Sint);
        storage_images.insert(slot(Stage::Kernel, SlotClass::Texture, 3), formatless);
        round_trip(&VariantKey {
            storage_images,
            psb_slots: BTreeSet::from([buffer(1)]),
            pulled_vertex_buffers: BTreeSet::from([2, 5]),
            emulated_simdgroups: true,
            dropped_vertex_layer: true,
            storage_view_slots: BTreeSet::from([buffer(3)]),
            srgb_textures: BTreeSet::from([slot(Stage::Kernel, SlotClass::Texture, 4)]),
            srgb_twins: BTreeSet::from([slot(Stage::Kernel, SlotClass::Texture, 4)]),
            ycbcr_textures: BTreeSet::from([slot(Stage::Kernel, SlotClass::Texture, 6)]),
            flat_grid: true,
            replay_bindings: true,
            merged_draws: true,
            amplification_draws: Some(AmplificationDraws::new(true, false)),
            viewport_compensation: Some(ViewportCompensation::new(true)),
            folded_groups: true,
            extended_range: BTreeMap::from([(
                slot(Stage::Kernel, SlotClass::Texture, 5),
                crate::metal::format::ExtendedRange::Bgra10 { srgb: true },
            )]),
            zeroed_colors: BTreeMap::from([(1, metal::PixelFormat::RGBA8Unorm)]),
            dropped_colors: BTreeSet::from([2]),
            kept_imageblock_bytes: Some(8),
            depth_clamp_range: true,
            rotated_weights: true,
            mesh_last_first: true,
            swizzled_storage: BTreeSet::from([slot(Stage::Kernel, SlotClass::Texture, 6)]),
            addressed_function_tables: true,
            interlocked_colors: BTreeSet::from([1]),
            read_only_tests: true,
            shader_tests: true,
            shader_stencil: true,
            window_depth_varying: true,
            address_maps: BTreeMap::from([(
                slot(Stage::Kernel, SlotClass::Texture, 7),
                metal::sampler::AddressModeMap::IDENTITY.with(
                    metal::SamplerAddressMode::Repeat,
                    metal::SamplerAddressMode::ClampToEdge,
                ),
            )]),
            cubic_samplers: BTreeMap::from([(slot(Stage::Fragment, SlotClass::Sampler, 0), 0x2d0)]),
            captured_positions: Some(crate::pipeline::CaptureForm::Vertices),
            primitive_id_indices: Some(metal::IndexType::UInt16),
            quad_primitive_ids: true,
            private_list: true,
            extended_shadows: BTreeSet::from([0]),
            snapshot_textures: BTreeMap::from([(
                slot(Stage::Fragment, SlotClass::Texture, 2),
                crate::pipeline::SnapshotForm::Unorm8,
            )]),
            cross_class: BTreeMap::from([(
                slot(Stage::Kernel, SlotClass::Texture, 8),
                SampledType::Uint,
            )]),
            anisotropic_samplers: BTreeSet::from([slot(Stage::Fragment, SlotClass::Sampler, 1)]),
            depth16_pass: Some(false),
            depth16_biased: true,
            depth16_bias_raises: Some(true),
            color_group: 1,
        });
        round_trip(&SpecializationKey {
            data: vec![1, 0, 0, 0],
            entries: vec![vk::SpecializationMapEntry {
                constant_id: 3,
                offset: 0,
                size: 4,
            }],
        });
        round_trip(&ArtifactError::WrongDevice);
    }

    #[test]
    fn lower_inputs() {
        let mut attributes = BTreeMap::new();
        attributes.insert(0, VertexFetch::FixedFunction { narrow_float: true });
        attributes.insert(
            1,
            VertexFetch::Pulled {
                format: metal::VertexFormat::UChar3,
                buffer_index: 29,
                offset: 12,
                step_function: metal::VertexStepFunction::PerInstance,
                step_rate: 2,
            },
        );
        round_trip(&LoweringInputs {
            vertex_fetch: Some(VertexFetchProjection { attributes }),
            storage_images: BTreeMap::new(),
            psb_slots: BTreeSet::from([buffer(0), buffer(30)]),
            emulated_simdgroups: true,
            dropped_vertex_layer: true,
            stage_input: Some({
                let mut d = StageInputDescriptor::new();
                d.attributes.insert(
                    3,
                    StageInputAttributeDescriptor::new(metal::AttributeFormat::Half2, 8, 5),
                );
                let mut l = StageInputLayoutDescriptor::new(Some(12));
                l.step_function = metal::StepFunction::ThreadPositionInGridYIndexed;
                d.layouts.insert(5, l);
                d.index_type = metal::IndexType::UInt32;
                d.index_buffer_index = 6;
                d
            }),
            storage_view_slots: BTreeSet::from([buffer(4)]),
            signed_color_attachments: BTreeSet::from([0, 6]),
            tile_attachments: Some(BTreeMap::from([
                (0, metal::PixelFormat::RGBA16Float),
                (3, metal::PixelFormat::R32Uint),
            ])),
            multisampled: true,
            replay_bindings: true,
            merged_draws: true,
            framebuffer_fetch: Some(FramebufferFetchInputs {
                attachments: BTreeMap::from([
                    (
                        0,
                        Some(FetchedAttachment {
                            pixel_format: metal::PixelFormat::RGBA32Float,
                            blending_enabled: true,
                            write_mask: metal::ColorWriteMask::RED,
                            replaces: true,
                            equation: Some(crate::pipeline::BlendEquation {
                                source_rgb: metal::BlendFactor::One,
                                destination_rgb: metal::BlendFactor::Zero,
                                rgb_operation: metal::BlendOperation::Add,
                                source_alpha: metal::BlendFactor::One,
                                destination_alpha: metal::BlendFactor::Zero,
                                alpha_operation: metal::BlendOperation::Add,
                            }),
                        }),
                    ),
                    (2, None),
                ]),
                depth_or_stencil: true,
                alpha_to_coverage: false,
                amplified: true,
                half_blends: BTreeSet::from([0]),
            }),
            srgb_textures: BTreeSet::from([slot(Stage::Fragment, SlotClass::Texture, 7)]),
            srgb_twins: BTreeSet::from([slot(Stage::Fragment, SlotClass::Texture, 7)]),
            snapshot_textures: BTreeMap::from([(
                slot(Stage::Fragment, SlotClass::Texture, 3),
                crate::pipeline::SnapshotForm::Half,
            )]),
            ycbcr_textures: BTreeSet::from([slot(Stage::Fragment, SlotClass::Texture, 8)]),
            viewport_index_varying: true,
            // An emulated post-tessellation vertex function's state.
            tessellation: Some(crate::pipeline::TessellationDescriptor {
                max_tessellation_factor: 16,
                partition_mode: metal::TessellationPartitionMode::FractionalOdd,
                ..Default::default()
            }),
            max_call_stack_depth: Some(8),
            flat_grid: true,
            render_target_array_index_varying: true,
            primitive_id_varying: true,
            primitive_id_indices: Some(crate::metal::IndexType::UInt16),
            quad_primitive_ids: true,
            private_list: true,
            color_writes: BTreeMap::from([
                (
                    0,
                    [
                        Some(ChannelWrite::Srgb),
                        Some(ChannelWrite::Srgb),
                        Some(ChannelWrite::Srgb),
                        Some(ChannelWrite::Device),
                    ],
                ),
                (
                    1,
                    [Some(ChannelWrite::Snorm {
                        bits: 16,
                        blended: true,
                    }); 4],
                ),
                (
                    2,
                    [
                        Some(ChannelWrite::FloatTowardZero {
                            mantissa: 10,
                            signed: true,
                        }),
                        None,
                        None,
                        None,
                    ],
                ),
            ]),
            amplification_draws: Some(AmplificationDraws::new(false, true)),
            primitive_varyings: BTreeSet::from([
                VaryingKey::user(b"weight"),
                VaryingKey::viewport_index(),
            ]),
            absent_varyings: BTreeSet::from([VaryingKey::user(b"generated(5colorDv4_f)")]),
            viewport_compensation: Some(ViewportCompensation::new(false)),
            folded_groups: true,
            extended_range: BTreeMap::from([(
                slot(Stage::Fragment, SlotClass::Texture, 2),
                crate::metal::format::ExtendedRange::Bgr10 { srgb: false },
            )]),
            extended_range_colors: BTreeMap::from([(
                4,
                ExtendedRangeColor::new(
                    crate::metal::format::ExtendedRange::Bgra10 { srgb: true },
                    true,
                ),
            )]),
            word_colors: BTreeMap::from([(
                5,
                crate::metal::format::WordAttachment::layout(metal::PixelFormat::ABGR4Unorm)
                    .expect("a word layout"),
            )]),
            srgb_fetches: BTreeSet::from([0, 2]),
            zeroed_colors: BTreeMap::from([(2, metal::PixelFormat::RGBA32Sint)]),
            dropped_colors: BTreeSet::from([3]),
            kept_imageblock_bytes: Some(12),
            depth_clamp_range: true,
            rotated_weights: true,
            swizzled_storage: BTreeSet::from([slot(Stage::Fragment, SlotClass::Texture, 3)]),
            addressed_function_tables: true,
            unattached_colors: BTreeSet::from([5, 7]),
            sample_mask: Some([0b1010, 0b0010, 0b1000, 0]),
            alpha_test: Some(metal::CompareFunction::GreaterEqual),
            depth_compare_clamp_mask: 0x8000_0005,
            read_only_tests: true,
            shader_tests: true,
            shader_stencil: true,
            window_depth_varying: true,
            rotated_colors: BTreeSet::from([4]),
            alpha_in_red_colors: BTreeSet::from([4]),
            rotated_factors: BTreeMap::from([(
                4,
                [
                    crate::pipeline::FactorLane::One,
                    crate::pipeline::FactorLane::Output(3),
                    crate::pipeline::FactorLane::Output(3),
                    crate::pipeline::FactorLane::Zero,
                ],
            )]),
            threadgroup_size_matches_tile_size: true,
            address_maps: BTreeMap::from([(
                slot(Stage::Fragment, SlotClass::Texture, 9),
                metal::sampler::AddressModeMap::IDENTITY.with(
                    metal::SamplerAddressMode::MirrorRepeat,
                    metal::SamplerAddressMode::ClampToEdge,
                ),
            )]),
            cubic_samplers: BTreeMap::from([(slot(Stage::Fragment, SlotClass::Sampler, 2), 0x2d0)]),
            rotated_blends: BTreeMap::from([
                (4, crate::pipeline::RotatedBlendForm::Premultiplied),
                (6, crate::pipeline::RotatedBlendForm::Unmeasured),
            ]),
            captured_positions: Some(crate::pipeline::CaptureForm::Indexed(
                metal::IndexType::UInt16,
            )),
            half_output_lanes: BTreeMap::from([(
                1,
                [
                    Some(crate::pipeline::HalfOutputLane::Float32),
                    Some(crate::pipeline::HalfOutputLane::Float32Added),
                    None,
                    None,
                ],
            )]),
            extended_shadows: BTreeSet::from([0]),
            cross_class: BTreeMap::from([(
                slot(Stage::Fragment, SlotClass::Texture, 10),
                SampledType::Float,
            )]),
            texture_write_rounding: metal::TextureWriteRoundingMode::NearestEven,
            anisotropic_samplers: BTreeSet::from([slot(Stage::Fragment, SlotClass::Sampler, 3)]),
            uniform_varyings: BTreeSet::from([VaryingKey::user(b"user(tint)")]),
            instanced_attributes: BTreeSet::from([1]),
            depth16_codes: true,
            depth16_bias_raises: Some(false),
            color_group: Some(crate::pipeline::ColorGroup {
                present: vec![0, 1, 2, 3, 5, 7],
                width: 4,
                group: 1,
            }),
            object_payload: Some(64),
        });
    }

    #[test]
    fn layout_and_vertex_input() {
        let layout = PipelineLayout {
            sets: vec![DescriptorSetLayout {
                set: 0,
                bindings: vec![
                    DescriptorBinding {
                        binding: 0,
                        ty: vk::DescriptorType::STORAGE_BUFFER,
                        count: 1,
                        stages: vk::ShaderStageFlags::FRAGMENT,
                        source: BindingSource::Buffer(buffer(0)),
                        immutable_sampler: None,
                        flags: vk::DescriptorBindingFlags::empty(),
                        variables: 1,
                    },
                    DescriptorBinding {
                        binding: 1,
                        ty: vk::DescriptorType::STORAGE_BUFFER,
                        count: 1,
                        stages: vk::ShaderStageFlags::FRAGMENT,
                        source: BindingSource::Synthetic {
                            stage: Stage::Fragment,
                            binding: SyntheticBinding::ExecutionStatus,
                        },
                        immutable_sampler: None,
                        flags: vk::DescriptorBindingFlags::empty(),
                        variables: 1,
                    },
                ],
                flags: vk::DescriptorSetLayoutCreateFlags::empty(),
            }],
            push_constants: vec![PushConstantRange {
                stages: vk::ShaderStageFlags::FRAGMENT,
                offset: 64,
                size: 16,
            }],
            push_descriptor_eligible: true,
        };
        round_trip(&layout);
        round_trip(&VertexInputState {
            bindings: vec![VertexBinding {
                binding: 0,
                buffer_index: 30,
                input_rate: vk::VertexInputRate::VERTEX,
                divisor: 1,
                stride: VertexStride::PerBind,
            }],
            attributes: vec![VertexAttribute {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 0,
            }],
            pulled: vec![PulledAttribute {
                attribute: 1,
                buffer_index: 29,
                format: metal::VertexFormat::UChar3,
                offset: 12,
                step_function: metal::VertexStepFunction::PerInstance,
                step_rate: 2,
                stride: VertexStride::Static(20),
            }],
            variant_pulled: BTreeSet::from([3]),
        });
    }

    #[test]
    fn graphics_interface_and_artifact_facts() {
        round_trip(&GraphicsState {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            color_attachments: vec![
                Some(ColorAttachmentState {
                    format: vk::Format::B8G8R8A8_UNORM,
                    write_mask: vk::ColorComponentFlags::R,
                    blend: Some(BlendState {
                        src_color: vk::BlendFactor::SRC_ALPHA,
                        dst_color: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                        color_op: vk::BlendOp::ADD,
                        src_alpha: vk::BlendFactor::ONE,
                        dst_alpha: vk::BlendFactor::ZERO,
                        alpha_op: vk::BlendOp::ADD,
                    }),
                }),
                None,
            ],
            depth_format: Some(vk::Format::D32_SFLOAT),
            stencil_format: None,
            rasterization_samples: vk::SampleCountFlags::COUNT_4,
            alpha_to_coverage: true,
            alpha_to_one: false,
            rasterizer_discard: false,
            dynamic_states: vec![vk::DynamicState::VIEWPORT],
            input_attachments: vec![InputAttachmentUse {
                color_index: 0,
                input_attachment_index: 0,
                format_class: SampledType::Float,
            }],
            rasterization_order_attachment_access: true,
            max_vertex_amplification_count: 2,
            declares_layer: true,
            tessellation: Some(TessellationState {
                patch_control_points: 1,
                domain: PatchDomain::Quad,
                control_points: 16,
                factor_step_function: metal::TessellationFactorStepFunction::PerPatchAndPerInstance,
                push_offset: 20,
                control_point_index_type: metal::TessellationControlPointIndexType::UInt16,
            }),
            fragment_imageblock: Some(FragmentImageblock {
                mechanism: ImageblockMechanism::StorageImageInterlock,
                words: 3,
                master: [9; 32],
                data_size: 12,
                alignment: 4,
            }),
            sample_shading: true,
            vertex_writes_memory: true,
            writes_viewport_index: true,
            storage_attachments: vec![StorageAttachmentUse {
                color_index: 1,
                format: vk::Format::R16G16B16A16_SFLOAT,
                writes: true,
                formatless: false,
                view: StorageAttachmentView::SamplePlanes { layers: 4 },
                view_alias: Some(vk::Format::R32_UINT),
                kept: vk::ColorComponentFlags::G | vk::ColorComponentFlags::A,
            }],
            shadow_stencil: Some(ShadowStencilUse {
                format: vk::Format::R8_UINT,
                formatless: true,
                samples: 4,
            }),
            early_fragment_tests: true,
            // An emulated mesh function's records.
            mesh_records: Some(crate::pipeline::MeshRecords {
                vertices: 3,
                primitives: 1,
                corners: 3,
                vertex_words: 4,
                primitive_words: 5,
                fields: vec![
                    crate::pipeline::RecordField {
                        value: crate::pipeline::RecordValue::Position,
                        per_primitive: false,
                        offset: 0,
                        words: 4,
                        numeric: crate::pipeline::stage_io::IoNumeric::Float,
                    },
                    crate::pipeline::RecordField {
                        value: crate::pipeline::RecordValue::Culled,
                        per_primitive: true,
                        offset: 3,
                        words: 1,
                        numeric: crate::pipeline::stage_io::IoNumeric::Uint,
                    },
                    crate::pipeline::RecordField {
                        value: crate::pipeline::RecordValue::Varying(
                            crate::pipeline::sym::VaryingKey::user(b"c"),
                        ),
                        per_primitive: true,
                        offset: 4,
                        words: 1,
                        numeric: crate::pipeline::stage_io::IoNumeric::Sint,
                    },
                ],
                slot_words: 12,
            }),
            // An emulated tessellation pipeline's state.
            emulated_tessellation: Some(crate::pipeline::EmulatedTessellation {
                domain: crate::pipeline::interface::PatchDomain::Quad,
                control_points: 4,
                descriptor: crate::pipeline::TessellationDescriptor::default(),
            }),
            writes_point_size: true,
            flat_fragment_inputs: true,
            clip_distances: 2,
            writes_depth: true,
            reads_barycentric_coord: true,
            reads_primitive_id: true,
            reads_position_depth: true,
            fragment_writes_memory: true,
            color_outputs: 0b101,
            sample_mask: Some(0b1001),
            shader_tests: Some(crate::pipeline::graphics::ShaderTestsUse {
                depth: crate::pipeline::graphics::StandInUse {
                    format: vk::Format::R32_SFLOAT,
                    formatless: false,
                    samples: 4,
                    view: crate::pipeline::graphics::StandInView::SamplePlanes { layers: 4 },
                },
                stencil: Some(crate::pipeline::graphics::StandInUse {
                    format: vk::Format::R8_UINT,
                    formatless: true,
                    samples: 4,
                    view: crate::pipeline::graphics::StandInView::Image,
                }),
            }),
            color_write_rules: vec![(0, 0x3322), (2, 0x0001)],
            snapshot_reads: true,
            color_group: Some(crate::pipeline::ColorGroup {
                present: vec![0, 5],
                width: 4,
                group: 0,
            }),
            logic_op: Some(vk::LogicOp::EQUIVALENT),
        });
        let mut slots = BTreeMap::new();
        slots.insert(
            buffer(0),
            SlotInterface::Buffer(BufferSlot {
                access: AccessSet::READ_WRITE,
                window: Window::Bounded { start: -8, len: 24 },
                form: BufferForm::Descriptor(DescriptorView::UniformVec4),
                kind: BufferKind::VisibleFunctionTable,
            }),
        );
        slots.insert(
            slot(Stage::Fragment, SlotClass::Texture, 1),
            SlotInterface::Texture(TextureSlot {
                texture_type: metal::TextureType::Type2DArray,
                sampled_type: SampledType::Uint,
                depth_compare: false,
                access: TextureAccess::new(false, true, true),
            }),
        );
        slots.insert(
            slot(Stage::Fragment, SlotClass::Sampler, 0),
            SlotInterface::Sampler(SamplerSlot {
                compare: true,
                coordinate_sensitive: true,
            }),
        );
        round_trip(&PipelineInterface {
            slots,
            threadgroup: vec![ThreadgroupArgument {
                stage: Stage::Kernel,
                index: 0,
                alignment: 16,
            }],
            needs_pointer_translation: true,
            needs_handle_translation: false,
            push: Vec::new(),
            reaches_command_buffers: true,
            command_writes: CommandWrites {
                operations: CommandWrites::DRAW | CommandWrites::RESET,
                primitive_types: 0b1000,
            },
            texture_writing_stages: vec![Stage::Vertex, Stage::Fragment],
            numbered_samplers: vec![
                (
                    Stage::Fragment,
                    vec![NumberedSampler::Constexpr([7, 0]), NumberedSampler::Read],
                ),
                (Stage::Vertex, vec![NumberedSampler::Write]),
            ],
        });
        round_trip(&RuntimeDependency {
            fact: RuntimeFact::StorageTextureFormat {
                slot: slot(Stage::Kernel, SlotClass::Texture, 0),
            },
            mechanism: Mechanism::Variant,
        });
        round_trip(&RuntimeDependency {
            fact: RuntimeFact::StorageTextureNumericType {
                slot: slot(Stage::Kernel, SlotClass::Texture, 0),
            },
            mechanism: Mechanism::Variant,
        });
        round_trip(&SpecializationLayout {
            constants: vec![SpecConstant {
                id: 0,
                key: SpecKey::SamplerState {
                    slot: slot(Stage::Fragment, SlotClass::Sampler, 1),
                    fields: crate::metal::sampler::SamplerWord::PIXEL_NEAREST,
                },
                size: 4,
                default: 0,
                unread_bits: 0,
            }],
        });
        round_trip(&ComputePipelineMetalInfo {
            max_total_threads_per_threadgroup: 1024,
            thread_execution_width: 32,
            static_threadgroup_memory_length: 4096,
            support_indirect_command_buffers: true,
            metal_max_total_threads_per_threadgroup: 960,
        });
        let mut census = StrategyCensus::default();
        census.entries.insert(
            "Access::SsboWord".into(),
            StrategyUse { uses: 3, words: 90 },
        );
        round_trip(&census);
        round_trip(&PrecisionNote::GradFromBias);
        round_trip(&Reconvergence::ImplementationDefined);
    }

    /// Slot-keyed maps are pair sequences; a repeated key is rejected so the encoding stays
    /// canonical.
    #[test]
    fn slot_keyed_maps_reject_duplicate_keys() {
        let pair = serde_json::to_string(&buffer(1)).expect("serializes");
        let choice = serde_json::to_string(&StorageImageChoice::new(
            vk::Format::R32_UINT,
            StorageImageClass::Identity,
        ))
        .expect("serializes");
        let once = format!(r#"{{"storage_images":[[{pair},{choice}]],"psb_slots":[]}}"#);
        assert!(serde_json::from_str::<VariantKey>(&once).is_ok());
        let twice =
            format!(r#"{{"storage_images":[[{pair},{choice}],[{pair},{choice}]],"psb_slots":[]}}"#);
        let err = serde_json::from_str::<VariantKey>(&twice).unwrap_err();
        assert!(err.to_string().contains("duplicate map key"), "{err}");
    }
}

#[test]
fn descriptor_constructors_use_metal_defaults() {
    let attachment = ColorAttachmentDescriptor::new(metal::PixelFormat::BGRA8Unorm);
    assert!(!attachment.blending_enabled);
    assert_eq!(attachment.source_rgb_blend_factor, metal::BlendFactor::One);
    assert_eq!(
        attachment.destination_alpha_blend_factor,
        metal::BlendFactor::Zero
    );
    assert_eq!(attachment.write_mask, metal::ColorWriteMask::ALL);
    let layout = VertexBufferLayoutDescriptor::new(Some(16));
    assert_eq!(layout.step_function, metal::VertexStepFunction::PerVertex);
    assert_eq!(layout.step_rate, 1);
    let tess = TessellationDescriptor::default();
    assert_eq!(tess.max_tessellation_factor, 16);
    assert_eq!(tess.partition_mode, metal::TessellationPartitionMode::Pow2);
    let values = crate::library::FunctionConstantValues::defaults();
    assert!(values.is_empty());
    let values = values.with_index(
        2,
        crate::library::FunctionConstantValue::new(metal::DataType::Bool, vec![1]),
    );
    assert!(!values.is_empty());
    let key = crate::library::FunctionCompileKey::new([1; 32]);
    assert_eq!(key.digest(), &[1; 32]);
    assert!(format!("{key:?}").starts_with("FunctionCompileKey(0101"));
}

#[cfg(feature = "serde")]
#[test]
fn descriptor_and_library_serde_round_trips() {
    use crate::library::{
        FunctionCompileKey, FunctionConstant, FunctionConstantValue, FunctionConstantValues,
    };

    fn round_trip<T>(value: &T)
    where
        T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
    {
        let json = serde_json::to_string(value).expect("serializes");
        let back: T = serde_json::from_str(&json).expect("deserializes");
        assert_eq!(&back, value, "{json}");
    }

    let mut attachment = ColorAttachmentDescriptor::new(metal::PixelFormat::RGBA8Unorm);
    attachment.blending_enabled = true;
    attachment.write_mask = metal::ColorWriteMask::RED;
    round_trip(&attachment);
    let mut vd = VertexDescriptor::new();
    vd.layouts
        .insert(30, VertexBufferLayoutDescriptor::new(None));
    vd.attributes.insert(
        0,
        VertexAttributeDescriptor::new(metal::VertexFormat::Float3, 0, 30),
    );
    round_trip(&vd);
    let mut si = StageInputDescriptor::new();
    si.layouts
        .insert(0, StageInputLayoutDescriptor::new(Some(16)));
    si.attributes.insert(
        1,
        StageInputAttributeDescriptor::new(metal::AttributeFormat::Float4, 0, 0),
    );
    round_trip(&si);
    round_trip(&TessellationDescriptor::default());
    round_trip(
        &FunctionConstantValues::defaults()
            .with_index(
                0,
                FunctionConstantValue::new(metal::DataType::Int, vec![7, 0, 0, 0]),
            )
            .with_name(
                "scale",
                FunctionConstantValue::new(metal::DataType::Float, vec![0; 4]),
            ),
    );
    round_trip(&FunctionConstant {
        name: "use_fog".into(),
        index: Some(3),
        ty: Some(metal::DataType::Bool),
        raw_flag: Some(1),
    });
    round_trip(&FunctionCompileKey::new([9; 32]));
}

#[cfg(feature = "serde")]
mod bind_serde_round_trips {
    use std::collections::{BTreeMap, BTreeSet};

    use super::*;
    use crate::bind::*;
    use crate::cache::SpecializationKey;
    use crate::metal;
    use crate::pipeline::sym::{PushField, SampledType};

    fn round_trip<T>(value: &T)
    where
        T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
    {
        let json = serde_json::to_string(value).expect("serializes");
        let back: T = serde_json::from_str(&json).expect("deserializes");
        assert_eq!(&back, value, "{json}");
    }

    #[test]
    fn bind_state_and_shape() {
        let mut table = StageTable::default();
        table.buffers.insert(
            0,
            BoundBuffer::Buffer {
                length: 4096,
                offset: 256,
                gpu_address: 0x1_0000_0000,
                device_address: Some(0xdead_0000),
                attribute_stride: None,
            },
        );
        table.buffers.insert(
            1,
            BoundBuffer::Bytes {
                len: 16,
                attribute_stride: None,
            },
        );
        table.textures.insert(
            3,
            BoundTexture::Texture {
                facts: TextureFacts::new(
                    metal::TextureType::Type2D,
                    metal::PixelFormat::RGBA8Unorm,
                ),
                resource_id: Some(9),
            },
        );
        table.samplers.insert(
            0,
            BoundSampler {
                desc: SamplerDescriptor {
                    min_filter: metal::SamplerMinMagFilter::Linear,
                    mag_filter: metal::SamplerMinMagFilter::Nearest,
                    mip_filter: metal::SamplerMipFilter::Linear,
                    s_address_mode: metal::SamplerAddressMode::ClampToEdge,
                    t_address_mode: metal::SamplerAddressMode::ClampToEdge,
                    r_address_mode: metal::SamplerAddressMode::ClampToEdge,
                    border_color: metal::SamplerBorderColor::TransparentBlack,
                    normalized_coordinates: false,
                    lod_min_clamp: 0.0,
                    lod_max_clamp: 1000.0,
                    lod_average: false,
                    max_anisotropy: 1,
                    compare_function: metal::CompareFunction::Never,
                    reduction_mode: metal::SamplerReductionMode::WeightedAverage,
                },
                resource_id: None,
            },
        );
        table.threadgroup_lengths.insert(0, 1024);
        table.at_bind_views = 0b101;
        let mut stages = BTreeMap::new();
        stages.insert(Stage::Kernel, table);
        round_trip(&BindState {
            stages,
            view_base_alignment: 64,
            used_resources: vec![UsedResource {
                class: ResourceClass::Heap,
                gpu_address: Some(0x2_0000_0000),
                length: 1 << 20,
                device_address: None,
                usage: metal::ResourceUsage::READ,
                stages: metal::RenderStages::empty(),
                token: 7,
                resource_id: None,
            }],
            vertex_amplification: VertexAmplification::new(2, vec![ViewMapping::new(0, 3)]),
            imageblock_size: ImageblockSize::new(16, 8),
            render_target_array_length: 6,
            stage_in_region: Some(StageInRegion::new([1, 2, 3], [4, 5, 6])),
            tessellation_factors: TessellationFactors::new(
                Some(TessellationFactorBuffer::new(4096, 16, 64)),
                0.5,
            ),
            render_area: [40, 24],
            argument_samplers: vec![ArgumentSampler::new(
                SamplerDescriptor {
                    normalized_coordinates: false,
                    ..SamplerDescriptor::new()
                },
                0x5a17_0001,
            )],
            depth_stencil: metal::state::DepthStencilDescriptor {
                front_face_stencil: metal::state::StencilDescriptor {
                    stencil_compare_function: metal::CompareFunction::LessEqual,
                    depth_stencil_pass_operation: metal::StencilOperation::IncrementWrap,
                    write_mask: 0x0F,
                    ..Default::default()
                },
                depth_write_enabled: true,
                ..Default::default()
            },
            heap_placement: HeapPlacement::ResourceId,
            viewport_count: 3,
            function_tables: vec![crate::bind::ArgumentFunctionTable::new(0x7ab1_0001, 9)],
            tile_threadgroup: tile_threadgroup(),
            // Clip mode's vertex clamp, unbounded above, which JSON holds as `null`.
            viewport_compensation: Some(
                crate::bind::CompensatedViewport::new(
                    ViewportCompensation::new(true).with_vertex_clamp(true),
                    [-1.0, 0.5, 1.5, 1.0],
                )
                .with_vertex_clamp([0.0, f32::INFINITY]),
            ),
            zeroed_colors: BTreeMap::from([(1, metal::PixelFormat::R32Uint)]),
            dropped_colors: BTreeSet::from([4]),
            kept_imageblock_bytes: Some(4),
            depth_clamp_range: Some([0.25, 0.5]),
            rotated_weights: true,
            mesh_last_first: true,
            resident_ranges: vec![crate::bind::ResidentRange::new(0x9_0000, 0x3_0000, 100)],
            interlocked_colors: BTreeSet::from([0, 3]),
            fragment_tests: crate::bind::FragmentTests::Depth { writes: true },
            extended_shadows: BTreeSet::from([0]),
            snapshot_textures: BTreeSet::from([2]),
            depth_range: [0.25, 0.75],
            depth_clip_mode_clamp: true,
            stencil_references: [0x5A, 0xC3],
            blend_color: [0.25, -0.5, 1.5, 0.75],
            alpha_test_reference: Some(0.375),
            primitive_type: Some(metal::PrimitiveType::TriangleStrip),
            depth_bias: [1.5 / 65535.0, -2.0, 0.125],
            depth16_pass: Some(true),
            depth16_biased: true,
            depth16_bias_raises: Some(true),
            primitive_id_indices: Some(metal::IndexType::UInt16),
            quad_primitive_ids: true,
            private_list: true,
        });
        let mut shape = StageShape::default();
        shape.buffers.insert(
            0,
            BufferShape::Buffer {
                attribute_stride: Some(24),
                device_address: false,
                length_class: 20,
                offset_alignment: 2,
            },
        );
        shape.samplers.insert(
            0,
            BoundSamplerShape {
                shape: SamplerShape {
                    reduction: metal::SamplerReductionMode::Minimum,
                    compare: true,
                    anisotropic: true,
                    mirror_clamp_to_edge: false,
                    valid_lod_clamps: true,
                    opaque_black_border: true,
                    linear: true,
                    cubic: true,
                },
                word: crate::metal::sampler::SamplerWord(0x0012_3456),
                min_level: 160,
                degree: 3,
            },
        );
        let mut stages = BTreeMap::new();
        stages.insert(Stage::Fragment, shape);
        round_trip(&BindShape {
            stages,
            used: BTreeMap::new(),
            vertex_amplification: VertexAmplification::default(),
            imageblock_size: ImageblockSize::new(32, 32),
            render_target_array_length: 1,
            tessellation_factor_buffer: true,
            render_area: [1920, 1080],
            argument_samplers: ArgumentSamplersShape::of(&[ArgumentSampler::new(
                SamplerDescriptor::new(),
                3,
            )]),
            function_tables: crate::bind::FunctionTablesShape::of(&[
                crate::bind::ArgumentFunctionTable::new(3, 0),
            ]),
            tile_threadgroup: tile_threadgroup(),
            viewport_compensation: Some(ViewportCompensation::new(false)),
            zeroed_colors: BTreeMap::from([(1, metal::PixelFormat::R32Uint)]),
            dropped_colors: BTreeSet::from([4]),
            kept_imageblock_bytes: Some(4),
            depth_clamp_range: true,
            rotated_weights: true,
            mesh_last_first: true,
            interlocked_colors: BTreeSet::from([0, 3]),
            fragment_tests: crate::bind::FragmentTests::Pass,
            extended_shadows: BTreeSet::from([0]),
            snapshot_textures: BTreeSet::from([2]),
            depth16_pass: Some(true),
            depth16_biased: true,
            depth16_bias_raises: Some(true),
            primitive_id_indices: Some(metal::IndexType::UInt32),
            quad_primitive_ids: true,
            private_list: true,
        });
    }

    /// A tile threadgroup memory of 96 bytes with two arguments.
    fn tile_threadgroup() -> crate::bind::TileThreadgroupMemory {
        let mut memory = crate::bind::TileThreadgroupMemory::new(96);
        for (index, length, offset) in [(0, 32, 0), (3, 64, 32)] {
            memory.arguments.insert(
                index,
                crate::bind::TileThreadgroupArgument::new(length, offset),
            );
        }
        memory
    }

    #[test]
    fn plans_and_commands() {
        let slot = slot(Stage::Kernel, SlotClass::Buffer, 2);
        let neutral = NeutralResource::Image {
            view_type: vk::ImageViewType::TYPE_2D_ARRAY,
            sampled_type: SampledType::Uint,
            depth_compare: false,
            samples: 1,
            storage_format: Some(vk::Format::R32_UINT),
        };
        round_trip(&BindingPlan {
            variant: VariantKey::new(),
            specialization: SpecializationKey::new(),
            descriptors: vec![
                DescriptorWrite {
                    set: 0,
                    binding: 0,
                    array_element: 0,
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    source: WriteSource::Buffer { slot },
                },
                DescriptorWrite {
                    set: 0,
                    binding: 1,
                    array_element: 0,
                    ty: vk::DescriptorType::STORAGE_IMAGE,
                    source: WriteSource::Neutral(neutral),
                },
                DescriptorWrite {
                    set: 1,
                    binding: 0,
                    array_element: 5,
                    ty: vk::DescriptorType::SAMPLER,
                    source: WriteSource::HeapEntry {
                        class: crate::pipeline::sym::HandleClass::Sampler {
                            compare: false,
                            integer: false,
                            cube_gradient: false,
                        },
                        element: 5,
                        source: HeapSource::Used(3),
                        view: ImageViewDesc {
                            view_type: vk::ImageViewType::TYPE_2D,
                            format: vk::Format::R8G8B8A8_UNORM,
                            aspect: vk::ImageAspectFlags::COLOR,
                            layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                            components: vk::ComponentMapping::IDENTITY,
                            level: None,
                            twin: false,
                            shader_swizzle: false,
                        },
                    },
                },
            ],
            vertex_buffers: vec![VertexBufferBind {
                binding: 0,
                buffer_index: 30,
                per_bind_stride: false,
            }],
            uses: vec![ResourceUse {
                resource: UseTarget::Slot(slot),
                access: AccessSet::WRITE,
                class: AccessClass::Storage,
                stages: vk::PipelineStageFlags2::COMPUTE_SHADER,
                range: Some(ByteRange { offset: 0, len: 64 }),
            }],
            push: PushTemplate {
                fields: vec![PushFieldPlacement {
                    stage: Stage::Kernel,
                    field: PushField::SlotBindOffset(slot),
                    location: FieldLocation::PushWindow { offset: 0 },
                    size: 4,
                }],
            },
            diagnostics: vec![PlanDiagnostic::IncompatibleBinding {
                slot: crate::pipeline::sym::SlotRef {
                    class: SlotClass::Texture,
                    ..slot
                },
                substituted: neutral,
            }],
            compute: None,
            graphics: None,
            views: ViewSetup {
                view_mask: 0b11,
                base_layer: 4,
            },
        });
        round_trip(&DispatchCommands {
            regions: vec![DispatchRegion {
                specialization: SpecializationKey::new(),
                group_count: [4, 1, 1],
                push_constants: vec![PushConstantWrite {
                    stages: vk::ShaderStageFlags::COMPUTE,
                    offset: 0,
                    bytes: vec![1, 2, 3, 4],
                }],
                indirect: None,
                stage_flags: vk::PipelineShaderStageCreateFlags::empty(),
                required_subgroup_size: Some(32),
            }],
            descriptor_bytes: vec![DescriptorBytes {
                set: 0,
                binding: 3,
                bytes: vec![0; 8],
            }],
            extents: vec![BufferExtent {
                slot,
                window: Window::Whole,
            }],
            descriptor_ranges: Vec::new(),
            push_holes: vec![PushHole::RootTableAddress {
                stages: vk::ShaderStageFlags::COMPUTE,
                offset: 8,
            }],
            root_table: None,
            indirect_clamp: IndirectClamp::new(IndirectRef::new(1, 16), [65535; 3], 64),
            uses: vec![ResourceUse {
                resource: UseTarget::Indirect,
                access: AccessSet::READ,
                class: AccessClass::Storage,
                stages: vk::PipelineStageFlags2::COMPUTE_SHADER,
                range: Some(ByteRange { offset: 0, len: 12 }),
            }],
        });
        round_trip(&Dispatch::ThreadgroupsIndirect {
            indirect: IndirectRef::new(2, 0),
            threads_per_group: [32, 1, 1],
        });
        round_trip(&DrawCall::Indirect {
            indirect: IndirectRef::new(4, 0),
            indexed: Some((vk::IndexType::UINT16, IndexBufferRef::new(5, 256), 64)),
        });
        round_trip(&DrawCall::Patches {
            control_points: 4,
            patch_start: 2,
            patch_count: 9,
            patch_index: None,
            instance_count: 3,
            base_instance: 1,
            control_point_index: Some((IndexBufferRef::new(7, 96), 2)),
        });
        round_trip(&DrawCall::PatchesIndirect {
            control_points: 3,
            patch_index: Some((IndexBufferRef::new(6, 64), 8)),
            indirect: IndirectRef::new(4, 16),
            control_point_index: None,
        });
        round_trip(&WriteSource::TessellationFactors);
        round_trip(&WriteSource::ShadowStencil);
        round_trip(&UseTarget::TessellationFactors);
        round_trip(&WriteSource::PatchIndices { min_bytes: 16 });
        round_trip(&WriteSource::TileMemory { bytes: 480 });
        round_trip(&UseTarget::PatchIndices);
        round_trip(&WriteSource::ControlPointIndices);
        round_trip(&WriteSource::TessellationWork);
        round_trip(&WriteSource::TessellationRecords);
        round_trip(&UseTarget::ControlPointIndices);
        round_trip(&DrawCommands {
            push_constants: Vec::new(),
            descriptor_bytes: Vec::new(),
            buffer_offsets: vec![(0, 256)],
            draw: vk::DrawCommand::DrawIndexed {
                index_type: vk::IndexType::UINT32,
                index_buffer_offset: 0,
                index_count: 36,
                instance_count: 1,
                first_index: 0,
                vertex_offset: -4,
                first_instance: 0,
            },
            indirect_needs_arguments: false,
            mesh_arguments: None,
            object_fold: None,
            discards_command_buffer: false,
            index_bounds_pass: false,
            extents: Vec::new(),
            push_holes: Vec::new(),
            descriptor_ranges: Vec::new(),
            root_tables: Vec::new(),
            vertex_strides: vec![(0, 32)],
            uses: vec![ResourceUse {
                resource: UseTarget::IndexBuffer,
                access: AccessSet::READ,
                class: AccessClass::Index,
                stages: vk::PipelineStageFlags2::INDEX_INPUT,
                range: Some(ByteRange {
                    offset: 0,
                    len: 144,
                }),
            }],
            index_overrun: Some(IndexOverrun::Clamped { past_end: 4 }),
            amplifications: vec![
                Vec::new(),
                vec![PushConstantWrite {
                    stages: vk::ShaderStageFlags::VERTEX,
                    offset: 16,
                    bytes: 1u32.to_le_bytes().to_vec(),
                }],
            ],
            specialization: None,
            // An emulated mesh draw's dispatch.
            mesh_dispatch: Some(crate::bind::MeshDispatch {
                work: vec![1, 0, 0, 2],
                work_bytes: 112,
                object_groups: Some([2, 1, 1]),
                object_arguments: Some(crate::bind::BufferRange {
                    offset: 0,
                    range: 12,
                }),
                mesh: crate::bind::MeshGroups::Scanned {
                    arguments: Some(crate::bind::BufferRange {
                        offset: 64,
                        range: 16,
                    }),
                },
                records: crate::bind::MeshRecordsRegion::Pool,
            }),
            // An emulated tessellation draw's scan.
            tessellation_dispatch: Some(crate::bind::TessellationDispatch {
                kernel: crate::pipeline::EmulatedTessellation {
                    domain: crate::pipeline::interface::PatchDomain::Triangle,
                    control_points: 3,
                    descriptor: crate::pipeline::TessellationDescriptor::default(),
                }
                .scan_kernel(),
                header: vec![2, 1, 0, 3],
                records: crate::bind::TessellationRecords::Exact { records: 2 },
                factors: Some(crate::bind::BufferRange {
                    offset: 0,
                    range: 32,
                }),
                arguments: None,
                chunk_fields: vec![(vk::ShaderStageFlags::VERTEX, 12)],
                patch_triangles: 64,
            }),
        });
        // A mesh draw's commands.
        round_trip(&vk::DrawCommand::DrawMeshTasks {
            group_count_x: 2,
            group_count_y: 1,
            group_count_z: 1,
        });
        round_trip(&vk::DrawCommand::DrawMeshTasksIndirect {
            offset: 12,
            draw_count: 1,
            stride: 12,
        });
        round_trip(&vk::DrawCommand::DrawMeshTasksIndirectCount {
            offset: 36,
            count_offset: 32,
            max_draw_count: 128,
            stride: 12,
        });
    }
}
