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
        own_pixel_textures: Vec::new(),
        level_reads: Vec::new(),
        private_bytes: 0,
        specialized_notes: Vec::new(),
        stats: ModuleStats::default(),
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
    let texel = SpecKey::BicubicTexels(texture);
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
            PrecisionNote::SrgbTexelDecode,
        ],
        strategies: StrategyCensus::default(),
        constant_blob: None,
        rounded_colors: Vec::new(),
        own_pixel_textures: Vec::new(),
        level_reads: Vec::new(),
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
                note: PrecisionNote::SrgbTexelDecode,
                when: vec![SpecClause {
                    all: vec![SpecTest::equal(texel, 0x109, 0)],
                }],
            },
        ],
        stats: ModuleStats::default(),
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
    use PrecisionNote::{GradFromBias as G, PixelSamplerScaling as P, SrgbTexelDecode as H};
    for (values, want) in [
        // Every default: the class 0 takes the sRGB note, the word 0 no scaling.
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
        assert_eq!(
            m.precision_notes_bound(&key(&values), &|_| None),
            sorted,
            "{values:?}"
        );
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
            emulated_simdgroups: true,
            srgb_textures: BTreeSet::from([slot(Stage::Kernel, SlotClass::Texture, 4)]),
            texel_buffer_parts: BTreeMap::from([(slot(Stage::Kernel, SlotClass::Texture, 11), 2)]),
            flat_grid: true,
            call_threadgroups: true,
            replay_bindings: true,
            merged_draws: true,
            amplification_draws: Some(AmplificationDraws::new(true, false)),
            folded_groups: true,
            partial_threadgroups: true,
            extended_range: BTreeMap::from([(
                slot(Stage::Kernel, SlotClass::Texture, 5),
                crate::metal::format::ExtendedRange::Bgra10 { srgb: true },
            )]),
            zeroed_colors: BTreeMap::from([(1, metal::PixelFormat::RGBA8Unorm)]),
            dropped_colors: BTreeSet::from([2]),
            kept_imageblock_bytes: Some(8),
            depth_clamp_range: true,
            rotated_weights: true,
            line_band: Some(crate::pipeline::BandPrimitive::Triangles),
            swizzled_storage: BTreeSet::from([slot(Stage::Kernel, SlotClass::Texture, 6)]),
            addressed_function_tables: true,
            address_maps: BTreeMap::from([(
                slot(Stage::Kernel, SlotClass::Texture, 7),
                metal::sampler::AddressModeMap::IDENTITY.with(
                    metal::SamplerAddressMode::Repeat,
                    metal::SamplerAddressMode::ClampToEdge,
                ),
            )]),
            cubic_samplers: BTreeMap::from([(slot(Stage::Fragment, SlotClass::Sampler, 0), 0x2d0)]),
            primitive_id_indices: Some(metal::IndexType::UInt16),
            quad_primitive_ids: true,
            private_list: true,
            cross_class: BTreeMap::from([(
                slot(Stage::Kernel, SlotClass::Texture, 8),
                SampledType::Uint,
            )]),
            anisotropic_samplers: BTreeSet::from([slot(Stage::Fragment, SlotClass::Sampler, 1)]),
            open_cl_lines: BTreeSet::from([slot(Stage::Kernel, SlotClass::Texture, 1)]),
            region_viewports: true,
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
        attributes.insert(0, VertexFetch::FixedFunction);
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
            signed_color_attachments: BTreeSet::from([0, 6]),
            tile_attachments: Some(BTreeMap::from([
                (0, metal::PixelFormat::RGBA16Float),
                (3, metal::PixelFormat::R32Uint),
            ])),
            multisampled: true,
            replay_bindings: true,
            merged_draws: true,
            srgb_textures: BTreeSet::from([slot(Stage::Fragment, SlotClass::Texture, 7)]),
            srgb_storage: BTreeSet::from([slot(Stage::Kernel, SlotClass::Texture, 3)]),
            texel_buffer_parts: BTreeMap::from([(
                slot(Stage::Fragment, SlotClass::Texture, 11),
                4,
            )]),
            viewport_index_varying: true,
            max_call_stack_depth: Some(8),
            open_cl_mode: true,
            flat_grid: true,
            call_threadgroups: true,
            render_target_array_index_varying: true,
            primitive_id_varying: true,
            primitive_id_indices: Some(crate::metal::IndexType::UInt16),
            quad_primitive_ids: true,
            private_list: true,
            indirect_range: true,
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
            folded_groups: true,
            partial_threadgroups: true,
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
            rotated_blends: BTreeSet::from([4, 6]),
            cross_class: BTreeMap::from([(
                slot(Stage::Fragment, SlotClass::Texture, 10),
                SampledType::Float,
            )]),
            texture_write_rounding: metal::TextureWriteRoundingMode::NearestEven,
            anisotropic_samplers: BTreeSet::from([slot(Stage::Fragment, SlotClass::Sampler, 3)]),
            open_cl_lines: BTreeSet::from([slot(Stage::Kernel, SlotClass::Texture, 2)]),
            uniform_varyings: BTreeSet::from([VaryingKey::user(b"user(tint)")]),
            instanced_attributes: BTreeSet::from([1]),
            object_payload: Some(64),
            region_viewports: true,
            push_base: 24,
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
                        immutable_samplers: Vec::new(),
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
                        immutable_samplers: Vec::new(),
                    },
                ],
                flags: vk::DescriptorSetLayoutCreateFlags::empty(),
            }],
            push_constants: vec![PushConstantRange {
                stages: vk::ShaderStageFlags::FRAGMENT,
                offset: 64,
                size: 16,
            }],
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
            vertex_writes_memory: true,
            writes_viewport_index: true,
            writes_point_size: true,
            writes_depth: true,
            reads_barycentric_coord: true,
            reads_primitive_id: true,
            fragment_writes_memory: true,
            color_outputs: 0b101,
            sample_mask: Some(0b1001),
            logic_op: Some(vk::LogicOp::EQUIVALENT),
            line_band: Some(crate::pipeline::LineBandState {
                push_offset: 112,
                primitive: crate::pipeline::BandPrimitive::Triangles,
            }),
        });
        let mut slots = BTreeMap::new();
        slots.insert(
            buffer(0),
            SlotInterface::Buffer(BufferSlot {
                access: AccessSet::READ_WRITE,
                window: Window::Bounded { start: -8, len: 24 },
                form: BufferForm::Descriptor,
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
                storage: true,
                declared_format: false,
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
                kinds: CommandWrites::RENDER_COMMANDS,
            },
            texture_writing_stages: vec![Stage::Vertex, Stage::Fragment],
            reads_unbound_resources: true,
            writes_unbound_resources: false,
            unbound_storage_textures: false,
            writes_memory: true,
            numbered_samplers: vec![
                (
                    Stage::Fragment,
                    vec![NumberedSampler::Constexpr([7, 0]), NumberedSampler::Read],
                ),
                (Stage::Vertex, vec![NumberedSampler::Write]),
            ],
            stage_in: vec![crate::pipeline::stage_io::StageInFetch {
                format: metal::AttributeFormat::Float2,
                buffer_index: 3,
                offset: 8,
                stride: None,
                step_function: metal::StepFunction::ThreadPositionInGridY,
                step_rate: 2,
                index: None,
            }],
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
                    fields: crate::metal::sampler::SamplerWord::PIXEL_COORDINATES,
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
mod bind_serde_round_trips {}
