//! Device-free tests of the engine render candidate: case mapping, companion refusals, the
//! oracle's vertex-buffer slots, output selection, and the owned and public graphics pipelines
//! compiled and planned for committed profiles. No device is opened.

use super::*;
use crate::case::{AttributeFormat, VertexObservation};
use crate::check::interface::tests::vertex_buffers_at;
use crate::executor_contract::depth_stencil_test;
use crate::source::public_sources;
use crate::store::CorpusStore;
use metal2vulkan::device::{Device, DeviceDescription, DevicePolicy};
use metal2vulkan::library::FunctionConstantValues;
use metal2vulkan::pipeline::PrecisionNote;
use metal2vulkan::vk;
use metal2vulkan_devtools::{fixtures_dir, llvm_assemble, AssembleError};
use std::path::Path;

fn device(stem: &str) -> Device {
    let path = crate::engine::profiles_dir().join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).unwrap();
    let description = DeviceDescription::from_khronos_json(&text, None).unwrap();
    Device::new(description, DevicePolicy::default()).unwrap()
}

/// The module text a case maps with when its module's stage metadata declares no render target
/// slice: the oracle draws it without layered rendering, at length 0.
pub(crate) const WITHOUT_SLICE: &str = "";

fn not_implemented(result: Result<RenderCase, ExecutorError>) -> String {
    match result {
        Err(ExecutorError::NotSupported(NotSupported::NotImplemented(what))) => what,
        other => panic!("expected NotImplemented, got {other:?}"),
    }
}

/// The tooling requirements a render case is refused for.
fn requirements(
    result: Result<RenderCase, ExecutorError>,
) -> std::collections::BTreeSet<crate::requirement::ToolingRequirement> {
    match result {
        Err(ExecutorError::NotSupported(NotSupported::Requirements(requirements))) => requirements,
        other => panic!("expected Requirements, got {other:?}"),
    }
}

/// The checker's companion of `case` over `bitcode`, and the AIR interface it maps with.
pub(crate) fn checked_companion(
    case: &AuthoredCase,
    bitcode: &[u8],
) -> (AirInterface, Option<Companion>) {
    let decoded = crate::check::loop_free::DecodedModule::from_bitcode("test", bitcode).unwrap();
    let abi = decoded.abi().unwrap();
    let interface =
        AirInterface::new(decoded.module(), abi, &case.entry, &BTreeMap::new()).unwrap();
    let companion = crate::companion::derive(case, decoded.module(), abi, &interface).unwrap();
    (interface, companion)
}

/// The product's declared fragment imageblock of `entry` in the module text `ll`: the reflection
/// the checker validates a case's planes against, whose master the render executor's cells
/// follow.
pub(crate) fn reflected_imageblock(
    ll: &str,
    entry: &str,
) -> Option<crate::air::reflect::FragmentImageblock> {
    crate::air::reflect::inspect(ll, crate::case::Stage::Fragment, entry, &[])
        .unwrap()
        .fragment_imageblock
}

/// A fragment case's authored depth and stencil aspects are its depth/stencil attachment (the
/// oracle's `Depth32Float` and `Stencil8`), its depth-stencil state is the one its outputs imply,
/// and each aspect is an observable output; a vertex case's attachment, which the oracle does not
/// attach, is refused.
#[test]
fn a_fragment_cases_depth_stencil_attachment_and_state_are_the_oracles() {
    use crate::literal::LiteralDepthStencil;
    let Some(bitcode) = assembled(&fixtures_dir().join("fragment_depth_stencil.ll")) else {
        return;
    };
    let mut authored = crate::companion::tests::case(
        Stage::Fragment,
        "fragment_depth_stencil",
        TextureFormat::Rgba32Float,
    );
    authored.render_targets.clear();
    authored.depth_stencil = Some(crate::case::DepthStencilResource {
        dimensions: [1, 1],
        initial_depth_b64: Some("AACAPw==".into()),
        initial_stencil_b64: Some("qw==".into()),
        state: None,
    });
    authored.output = OutputSelection::Stencil {
        origin: [0, 0],
        dimensions: [1, 1],
    };
    let (interface, companion) = checked_companion(&authored, &bitcode);
    let render = render_case(
        &authored,
        &interface,
        WITHOUT_SLICE,
        companion.as_ref(),
        None,
    )
    .unwrap();
    assert!(render.color_targets.is_empty());
    assert_eq!(
        render.depth_stencil,
        Some(LiteralDepthStencil {
            dimensions: [1, 1],
            samples: 1,
            depth: Some(1.0f32.to_le_bytes().to_vec()),
            stencil: Some(vec![0xab]),
        })
    );
    // The state is the one the Metal oracle sets, from the same derivation.
    assert_eq!(
        render.depth_stencil_state,
        DrawDepthStencil::derived(depth_stencil_test(&interface))
    );
    assert_eq!(
        render.output,
        RenderOutput::Stencil {
            origin: [0, 0],
            dimensions: [1, 1]
        }
    );
    let template = render_template(&render);
    assert_eq!(
        (template.depth_format, template.stencil_format),
        (
            Some(metal::PixelFormat::Depth32Float),
            Some(metal::PixelFormat::Stencil8)
        )
    );

    let finals = RenderFinals {
        depth_stencil: Some(LiteralDepthStencil {
            dimensions: [2, 1],
            samples: 1,
            depth: Some(
                [0.25f32, 0.5]
                    .iter()
                    .flat_map(|d| d.to_le_bytes())
                    .collect(),
            ),
            stencil: Some(vec![3, 7]),
        }),
        ..RenderFinals::default()
    };
    let second = |output| select_render_output(output, &finals).unwrap();
    assert_eq!(
        second(RenderOutput::Depth {
            origin: [1, 0],
            dimensions: [1, 1]
        }),
        0.5f32.to_le_bytes()
    );
    assert_eq!(
        second(RenderOutput::Stencil {
            origin: [1, 0],
            dimensions: [1, 1]
        }),
        [7]
    );

    let mut vertex = authored.clone();
    vertex.stage = Stage::Vertex;
    vertex.output = OutputSelection::None;
    assert!(
        requirements(render_case(&vertex, &interface, WITHOUT_SLICE, None, None))
            .contains(&crate::requirement::ToolingRequirement::VertexDepthStencilAttachment)
    );

    // Only the compute executor builds acceleration structures.
    let mut traced = authored.clone();
    traced.acceleration_structures = vec![crate::case::AccelerationStructureResource {
        binding: 5,
        kind: crate::case::AccelerationStructureKind::Instance,
        primitive_triangles_b64: None,
        child_references: vec![0],
    }];
    assert_eq!(
        requirements(render_case(
            &traced,
            &interface,
            WITHOUT_SLICE,
            companion.as_ref(),
            None
        )),
        [crate::requirement::ToolingRequirement::RenderAccelerationStructure].into()
    );
}

/// A fragment case's `[[stencil]]` output plans its shader-side stencil where the device has no
/// stencil export: on nvidia-rtx50-linux the pipeline keeps a shadow, the plan binds it once, and
/// the bind state carries the draw's descriptor, the one the Metal oracle sets. On lavapipe, which
/// has the extension, the pipeline keeps none. A state whose stencil write could depend on the
/// depth test is refused at the draw as needing the extension.
#[test]
fn a_stencil_output_case_plans_its_shadow_where_the_device_lacks_export() {
    use metal2vulkan::bind::WriteSource;
    use metal2vulkan::device::VkExtension;
    use metal2vulkan::error::{BindError, Capability, Feature, Needs};
    let Some(bitcode) = assembled(&fixtures_dir().join("fragment_depth_stencil.ll")) else {
        return;
    };
    let mut authored = crate::companion::tests::case(
        Stage::Fragment,
        "fragment_depth_stencil",
        TextureFormat::Rgba32Float,
    );
    authored.render_targets.clear();
    authored.depth_stencil = Some(crate::case::DepthStencilResource {
        dimensions: [1, 1],
        initial_depth_b64: Some("AACAPw==".into()),
        initial_stencil_b64: Some("qw==".into()),
        state: None,
    });
    authored.output = OutputSelection::Stencil {
        origin: [0, 0],
        dimensions: [1, 1],
    };
    let (interface, companion) = checked_companion(&authored, &bitcode);
    let render = render_case(
        &authored,
        &interface,
        WITHOUT_SLICE,
        companion.as_ref(),
        None,
    )
    .unwrap();
    let companion_bitcode = render.companion.as_ref().unwrap().opaque_bitcode().unwrap();
    let libraries = RenderLibraries {
        vertex: &companion_bitcode,
        fragment: Some(&bitcode),
    };
    let (pipeline, _, _) = planned_draw(&device("lavapipe"), &render, libraries).unwrap();
    assert!(pipeline.graphics.shadow_stencil.is_none());

    let nvidia = device("nvidia-rtx50-linux");
    let (pipeline, plan, _) = planned_draw(&nvidia, &render, libraries).unwrap();
    let shadow = pipeline.graphics.shadow_stencil.expect("the shadow");
    assert_eq!(shadow.format, vk::Format::R8_UINT);
    assert_eq!(
        plan.descriptors
            .iter()
            .filter(|write| matches!(write.source, WriteSource::ShadowStencil))
            .count(),
        1
    );
    let state = bind_state(&render, &BTreeMap::new(), &BTreeMap::new()).unwrap();
    assert_eq!(state.depth_stencil, render.depth_stencil_state.descriptor);

    // A depth test that can fail, under a face whose depth-failure operation (`Keep`) is not its
    // pass operation (`Replace`).
    let mut dependent = render.clone();
    dependent
        .depth_stencil_state
        .descriptor
        .depth_compare_function = metal2vulkan::metal::CompareFunction::Less;
    let state = bind_state(&dependent, &BTreeMap::new(), &BTreeMap::new()).unwrap();
    match plan.instantiate_draw(&state, &dependent.draw) {
        Err(BindError::Unsupported(u)) => assert_eq!(
            (u.feature, u.needs),
            (
                Feature::StencilExport,
                Needs::AnyOf(vec![vec![Capability::Extension(
                    VkExtension::ExtShaderStencilExport
                )]])
            )
        ),
        other => panic!("a depth-dependent stencil write: {:?}", other.map(|_| ())),
    }
}

/// A multisampled pass's `[[stencil]]` output plans a multisampled shadow on nvidia-rtx50-linux:
/// the pipeline keeps a shadow of the pass's 4 samples, requires `maintenance8` (its copies are
/// image to image) and multisampled storage images of 4 samples, and does not run the stencil-only
/// function per sample. The case's stencil, whose samples hold one value, is cleared to it by both
/// executors.
#[test]
fn a_multisampled_stencil_output_case_plans_a_shadow_of_its_samples() {
    use metal2vulkan::device::{VkFeature, VkProperty};
    let Some(bitcode) = assembled(&fixtures_dir().join("fragment_depth_stencil.ll")) else {
        return;
    };
    let mut authored = crate::companion::tests::case(
        Stage::Fragment,
        "fragment_depth_stencil",
        TextureFormat::Rgba32Float,
    );
    authored.render_targets.clear();
    authored.raster_sample_count = Some(4);
    authored.depth_stencil = Some(crate::case::DepthStencilResource {
        dimensions: [1, 1],
        initial_depth_b64: Some(b64(&[1.0f32.to_le_bytes(); 4].concat())),
        initial_stencil_b64: Some(b64(&[0xab; 4])),
        state: None,
    });
    authored.output = OutputSelection::Stencil {
        origin: [0, 0],
        dimensions: [1, 1],
    };
    let (interface, companion) = checked_companion(&authored, &bitcode);
    let render = render_case(
        &authored,
        &interface,
        WITHOUT_SLICE,
        companion.as_ref(),
        None,
    )
    .unwrap();
    assert_eq!(
        render.attachment_transfer(),
        AttachmentTransfer::PerSample { samples: 4 }
    );
    let companion_bitcode = render.companion.as_ref().unwrap().opaque_bitcode().unwrap();
    let libraries = RenderLibraries {
        vertex: &companion_bitcode,
        fragment: Some(&bitcode),
    };
    let (pipeline, _, _) = planned_draw(&device("nvidia-rtx50-linux"), &render, libraries).unwrap();
    let shadow = pipeline.graphics.shadow_stencil.expect("the shadow");
    assert_eq!((shadow.format, shadow.samples), (vk::Format::R8_UINT, 4));
    assert!(!pipeline.graphics.sample_shading);
    let requirements = &pipeline.requirements;
    assert!(requirements.features.contains(&VkFeature::Maintenance8));
    assert!(requirements
        .features
        .contains(&VkFeature::ShaderStorageImageMultisample));
    assert!(requirements
        .properties
        .contains(&VkProperty::StorageImageSampleCount(4)));
}

/// A fragment case that reads its color attachment maps like any fragment case (its render target
/// is the attachment it reads; no requirement stops the engine's candidate) and plans on
/// lavapipe, whose `rasterizationOrderColorAttachmentAccess` orders the read: the plan reads color
/// attachment 0 through an input-attachment write, and the pipeline states rasterization-order
/// access and requires the feature. On moltenvk-m3, which lacks the feature but has the pixel
/// interlock, it plans the interlocked storage form: one storage write of the attachment, which is
/// no color attachment of the subpass. Without the interlock too, it plans the input attachment
/// with no rasterization-order access, which the consumer orders between draws alone (its
/// barriers, `platform::graphics_pipeline::fetch_between_draws`), the fragment stage noting
/// `PrecisionNote::UnorderedFetchWithinDraw`, and requires neither ordered form.
#[test]
fn a_framebuffer_fetch_case_plans_its_ordered_input_attachment() {
    use metal2vulkan::bind::WriteSource;
    use metal2vulkan::device::VkFeature;
    let Some(bitcode) = assembled(&metal2vulkan_devtools::engine_dir().join("fragment_fetch.ll"))
    else {
        return;
    };
    let mut authored =
        crate::companion::tests::case(Stage::Fragment, "fetch_scalar", TextureFormat::R32Float);
    authored.render_targets[0].initial_bytes_b64 = b64(&3.0f32.to_le_bytes());
    authored.output = OutputSelection::RenderTarget {
        index: 0,
        origin: [0, 0],
        dimensions: [1, 1],
    };
    let (interface, companion) = checked_companion(&authored, &bitcode);
    let render = render_case(
        &authored,
        &interface,
        WITHOUT_SLICE,
        companion.as_ref(),
        None,
    )
    .unwrap();
    assert_eq!(render.color_targets[&0].bytes, 3.0f32.to_le_bytes());
    let companion_bitcode = render.companion.as_ref().unwrap().opaque_bitcode().unwrap();
    let libraries = RenderLibraries {
        vertex: &companion_bitcode,
        fragment: Some(&bitcode),
    };
    let (pipeline, plan, _) = planned_draw(&device("lavapipe"), &render, libraries).unwrap();
    assert_eq!(pipeline.graphics.input_attachments.len(), 1);
    assert!(pipeline.graphics.rasterization_order_attachment_access);
    assert!(pipeline
        .requirements
        .features
        .contains(&VkFeature::RasterizationOrderColorAttachmentAccess));
    assert_eq!(
        plan.descriptors
            .iter()
            .filter(|write| matches!(write.source, WriteSource::Attachment { color_index: 0 }))
            .count(),
        1
    );
    // Moltenvk-m3 orders the read through the pixel interlock, as a storage image of the attachment
    // the function reads and writes.
    let (pipeline, plan, _) = planned_draw(&device("moltenvk-m3"), &render, libraries).unwrap();
    assert!(pipeline.graphics.input_attachments.is_empty());
    assert_eq!(pipeline.graphics.color_attachments, [None]);
    assert_eq!(
        pipeline
            .graphics
            .storage_attachments
            .iter()
            .map(|a| (a.color_index, a.writes))
            .collect::<Vec<_>>(),
        [(0, true)]
    );
    assert!(pipeline
        .requirements
        .features
        .contains(&VkFeature::FragmentShaderPixelInterlock));
    assert_eq!(
        plan.descriptors
            .iter()
            .filter(|write| matches!(
                write.source,
                WriteSource::TileAttachment { color_index: 0, .. }
            ))
            .count(),
        1
    );
    // Without the interlock either, the input attachment ordered between draws alone, noted.
    let path = crate::engine::profiles_dir().join("moltenvk-m3.json");
    let mut description =
        DeviceDescription::from_khronos_json(&std::fs::read_to_string(path).unwrap(), None)
            .unwrap();
    description
        .features
        .remove(&VkFeature::FragmentShaderPixelInterlock);
    let unordered = Device::new(description, DevicePolicy::default()).unwrap();
    let (pipeline, plan, _) = planned_draw(&unordered, &render, libraries).unwrap();
    let read: Vec<u32> = (pipeline.graphics.input_attachments.iter())
        .map(|input| input.color_index)
        .collect();
    assert_eq!(read, [0]);
    assert!(!pipeline.graphics.rasterization_order_attachment_access);
    assert!(pipeline.graphics.storage_attachments.is_empty());
    assert!(pipeline.graphics.color_attachments[0].is_some());
    for ordered in [
        VkFeature::RasterizationOrderColorAttachmentAccess,
        VkFeature::FragmentShaderPixelInterlock,
    ] {
        assert!(
            !pipeline.requirements.features.contains(&ordered),
            "{ordered:?}"
        );
    }
    assert_eq!(
        plan.descriptors
            .iter()
            .filter(|write| matches!(write.source, WriteSource::Attachment { color_index: 0 }))
            .count(),
        1
    );
    assert!(unordered_within_draw(&pipeline));
}

/// Whether `pipeline`'s fragment stage records `PrecisionNote::UnorderedFetchWithinDraw`: its
/// fetch is ordered between draws alone.
fn unordered_within_draw(pipeline: &metal2vulkan::pipeline::RenderPipeline) -> bool {
    (pipeline.stages.iter())
        .filter(|stage| stage.stage == vk::ShaderStage::FRAGMENT)
        .any(|stage| {
            stage
                .precision_notes
                .contains(&PrecisionNote::UnorderedFetchWithinDraw)
        })
}

/// A fragment case draws through its passthrough vertex companion with its own resources on the
/// fragment stage, its render targets attached, and a render-target output; an observed vertex
/// case draws into its observer with its resources, attributes, and target on the vertex side.
#[test]
fn graphics_cases_map_onto_a_draw_with_their_companion() {
    use crate::companion::tests::case;
    let public = fixtures_dir();
    let (Some(fragment), Some(triangle)) = (
        assembled(&public.join("fragment_varying_color.ll")),
        assembled(&public.join("vertex_varying_triangle.ll")),
    ) else {
        return;
    };
    let mut authored = case(
        Stage::Fragment,
        "fragment_varying_color",
        TextureFormat::Rgba32Float,
    );
    authored.output = OutputSelection::RenderTarget {
        index: 0,
        origin: [0, 0],
        dimensions: [1, 1],
    };
    let (interface, companion) = checked_companion(&authored, &fragment);
    let render = render_case(
        &authored,
        &interface,
        WITHOUT_SLICE,
        companion.as_ref(),
        None,
    )
    .unwrap();
    assert_eq!(
        render.vertex.entry,
        crate::companion::PASSTHROUGH_VERTEX_ENTRY
    );
    assert_eq!(render.vertex.resources, StageResources::default());
    assert_eq!(
        render.fragment.as_ref().map(|f| f.entry.as_str()),
        Some("fragment_varying_color")
    );
    assert!(render.vertex_buffers.is_empty());
    assert_eq!(render.color_targets[&0].format, TextureFormat::Rgba32Float);
    assert_eq!(render.color_targets[&0].dimensions, [1, 1]);
    assert_eq!(
        render.output,
        RenderOutput::RenderTarget {
            index: 0,
            origin: [0, 0],
            dimensions: [1, 1]
        }
    );
    assert_eq!(render.companion, companion);

    // A buffer output of a fragment case is the fragment stage's.
    authored.buffers = vec![crate::case::BufferResource {
        binding: 3,
        role: crate::case::ResourceRole::Output,
        bytes_b64: None,
        initial_bytes_b64: Some("AAAAAA==".into()),
        rasterization_rate_map: None,
    }];
    authored.output = OutputSelection::Buffer {
        binding: 3,
        offset: 0,
        length: 4,
    };
    let render = render_case(
        &authored,
        &interface,
        WITHOUT_SLICE,
        companion.as_ref(),
        None,
    )
    .unwrap();
    assert_eq!(
        render.fragment.as_ref().unwrap().resources.buffers[&3],
        vec![0; 4]
    );
    assert_eq!(
        render.output,
        RenderOutput::Buffer {
            stage: ShaderStage::Fragment,
            binding: 3,
            offset: 0,
            length: 4
        }
    );

    let mut observed = case(
        Stage::Vertex,
        "vertex_varying_triangle",
        TextureFormat::Rgba32Float,
    );
    observed.vertex_observation = Some(VertexObservation::Varying { location: 0 });
    observed.output = OutputSelection::RenderTarget {
        index: 0,
        origin: [0, 0],
        dimensions: [1, 1],
    };
    let (interface, observer) = checked_companion(&observed, &triangle);
    let render = render_case(
        &observed,
        &interface,
        WITHOUT_SLICE,
        observer.as_ref(),
        None,
    )
    .unwrap();
    assert_eq!(render.vertex.entry, "vertex_varying_triangle");
    assert_eq!(
        render.fragment.as_ref().map(|f| f.entry.as_str()),
        Some(crate::companion::OBSERVER_FRAGMENT_ENTRY)
    );
    assert_eq!(render.color_targets[&0].format, TextureFormat::Rgba32Float);
    assert_eq!(render.companion, observer);

    // A companion that does not complete the case's stage is the executor's failure, not a gap.
    for (authored, companion) in [(&observed, None), (&authored, observer.as_ref())] {
        match render_case(authored, &interface, WITHOUT_SLICE, companion, None) {
            Err(ExecutorError::Failed(why)) => assert!(why.contains("has companion"), "{why}"),
            other => panic!("a mismatched companion mapped: {other:?}"),
        }
    }
    let mut kernel = crate::case::narrow_vertex_test_case(String::new(), "k".into());
    kernel.stage = Stage::Kernel;
    assert!(not_implemented(render_case(
        &kernel,
        &vertex_buffers_at(&[]),
        WITHOUT_SLICE,
        None,
        None
    ))
    .contains("not a render case"));
    // A rasterization-disabled vertex case's authored targets are not the oracle's sink.
    let mut vertex = crate::case::narrow_vertex_test_case(String::new(), "v".into());
    vertex.render_targets = observed.render_targets.clone();
    assert_eq!(
        requirements(render_case(
            &vertex,
            &vertex_buffers_at(&[0]),
            WITHOUT_SLICE,
            None,
            None
        )),
        [crate::requirement::ToolingRequirement::RasterizationDisabledRenderTarget].into()
    );
}

/// The narrow-attribute case maps onto the oracle's rasterization-disabled draw: attributes at
/// the highest free slots, the 1×1 sink, the buffer output of the vertex stage.
#[test]
fn a_rasterization_disabled_vertex_case_maps_onto_the_oracles_draw() {
    let case = crate::case::narrow_vertex_test_case(String::new(), "v".into());
    let render = render_case(&case, &vertex_buffers_at(&[0]), WITHOUT_SLICE, None, None).unwrap();
    assert_eq!(render.fragment, None);
    assert_eq!(render.companion, None);
    assert_eq!(
        render
            .vertex_buffers
            .iter()
            .map(|b| (b.location, b.buffer_index, b.stride, b.bytes.clone()))
            .collect::<Vec<_>>(),
        [(0, 30, 1, vec![2]), (1, 29, 4, vec![3, 0, 4, 0])]
    );
    assert_eq!(render.color_targets[&0].format, TextureFormat::R8Unorm);
    assert_eq!(
        render.draw,
        DrawCall::Primitives {
            vertex_start: 0,
            vertex_count: 1,
            instance_count: 1,
            base_instance: 0
        }
    );
    assert_eq!(
        render.output,
        RenderOutput::Buffer {
            stage: ShaderStage::Vertex,
            binding: 0,
            offset: 0,
            length: 4
        }
    );
    let addresses = BTreeMap::from([(
        (ShaderStage::Vertex, 0),
        synthetic_address(ShaderStage::Vertex, 0),
    )]);
    let state = bind_state(&render, &addresses, &BTreeMap::new()).unwrap();
    let vertex = &state.stages[&ShaderStage::Vertex];
    assert_eq!(
        vertex.buffers.keys().copied().collect::<Vec<_>>(),
        [0, 29, 30]
    );
    assert_ne!(
        synthetic_address(ShaderStage::Vertex, 3).guest,
        synthetic_address(ShaderStage::Fragment, 3).guest
    );
}

#[test]
fn render_target_outputs_select_rows_of_the_final_contents() {
    let target = ColorTarget {
        format: TextureFormat::R8Unorm,
        dimensions: [3, 2],
        samples: 1,
        bytes: (0..6).collect(),
    };
    assert_eq!(select_target(&target, [1, 0], [2, 2]), Ok(vec![1, 2, 4, 5]));
    assert!(select_target(&target, [2, 1], [2, 1]).is_err());
    let finals = RenderFinals {
        targets: BTreeMap::from([(1, target)]),
        buffers: BTreeMap::from([((ShaderStage::Fragment, 2), vec![7, 8, 9])]),
        ..Default::default()
    };
    let region = RenderOutput::RenderTarget {
        index: 1,
        origin: [0, 1],
        dimensions: [3, 1],
    };
    assert_eq!(select_render_output(region, &finals), Ok(vec![3, 4, 5]));
    let buffer = RenderOutput::Buffer {
        stage: ShaderStage::Fragment,
        binding: 2,
        offset: 1,
        length: 2,
    };
    assert_eq!(select_render_output(buffer, &finals), Ok(vec![8, 9]));
    let vertex_buffer = RenderOutput::Buffer {
        stage: ShaderStage::Vertex,
        binding: 2,
        offset: 0,
        length: 1,
    };
    assert!(select_render_output(vertex_buffer, &finals).is_err());
}

fn assembled(path: &Path) -> Option<Vec<u8>> {
    match llvm_assemble(path) {
        Ok(bitcode) => Some(bitcode),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {}: {why}", path.display()),
    }
}

/// The AIR bitcode each stage's function comes from: owned device tests draw a vertex and a
/// fragment function of two libraries through one pipeline.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RenderLibraries<'a> {
    pub vertex: &'a [u8],
    /// Present exactly when the case has a fragment function.
    pub fragment: Option<&'a [u8]>,
}

/// The request the tools send the compile worker for `case`'s pipeline with bind shape `shape`,
/// and its libraries in request order: a vertex function alone as `PipelineRequest::VertexDraw`, a
/// vertex and a fragment function of two libraries as `PipelineRequest::Draw`, both with the shape.
pub(crate) fn draw_request<'a>(
    case: &RenderCase,
    libraries: RenderLibraries<'a>,
    shape: &BindShape,
) -> (crate::acceptance::PipelineRequest, Vec<&'a [u8]>) {
    let function = |f: &RenderFunction| {
        crate::acceptance::RequestFunction::new(f.entry.as_str(), f.constants.clone())
    };
    match (&case.fragment, libraries.fragment) {
        (None, None) => (
            crate::acceptance::PipelineRequest::VertexDraw {
                function: function(&case.vertex),
                template: render_template(case),
                shape: Some(shape.clone()),
                form: crate::candidate::engine::variants::DrawForm::Bound,
            },
            vec![libraries.vertex],
        ),
        (Some(fragment), Some(bitcode)) => (
            crate::acceptance::PipelineRequest::Draw {
                vertex: function(&case.vertex),
                fragment: function(fragment),
                template: render_template(case),
                shape: Some(shape.clone()),
            },
            vec![libraries.vertex, bitcode],
        ),
        _ => panic!("a fragment function and its library come together"),
    }
}

/// `case`'s pipeline compiled in this process for bind shape `shape`, by [`draw_request`]: the
/// device-free plan tests compile here; the device tests compile the same request in the compile
/// worker.
pub(crate) fn compile_in_process(
    device: &Device,
    case: &RenderCase,
    libraries: RenderLibraries<'_>,
    shape: &BindShape,
) -> Result<metal2vulkan::pipeline::RenderPipeline, Error> {
    let (request, bitcode) = draw_request(case, libraries, shape);
    match crate::acceptance::compile_pipeline(&bitcode, &request, device, None).map(|c| c.pipeline)
    {
        Ok(crate::acceptance::AcceptancePipeline::Render(pipeline)) => Ok(*pipeline),
        Ok(other) => panic!("a draw compiled to {other:?}"),
        Err(crate::acceptance::PipelineCompileError::Product(error)) => Err(error),
        Err(error) => panic!("the test's libraries are not its request's: {error}"),
    }
}

/// A two-stage case over `targets` with no resources and a triangle-list draw.
pub(crate) fn two_stage_case(
    vertex: &str,
    fragment: &str,
    vertex_buffers: Vec<VertexBuffer>,
    targets: BTreeMap<u32, ColorTarget>,
    vertex_count: u32,
) -> RenderCase {
    let function = |entry: &str| RenderFunction {
        entry: entry.into(),
        constants: FunctionConstantValues::defaults(),
        resources: StageResources::default(),
    };
    RenderCase {
        vertex: function(vertex),
        fragment: Some(function(fragment)),
        vertex_buffers,
        color_targets: targets,
        depth_stencil: None,
        depth_stencil_state: DrawDepthStencil::default(),
        render_target_array_length: 0,
        assembly: Assembly::Primitives(Primitive::Triangle),
        cull_mode: metal::CullMode::None,
        draw: DrawCall::Primitives {
            vertex_start: 0,
            vertex_count,
            instance_count: 1,
            base_instance: 0,
        },
        index_buffer: None,
        out_of_bounds: Vec::new(),
        imageblock: None,
        output: RenderOutput::RenderTarget {
            index: 0,
            origin: [0, 0],
            dimensions: [1, 1],
        },
        companion: None,
        rate_map_pass: None,
        raster_sample_count: None,
    }
}

/// Compile and plan `case` on `device`: no variant but the dropped vertex `Layer` output of a
/// vertex function that chooses its slice in the one-layer pass the executor binds, no diagnostic,
/// and one direct draw; or the product's compile refusal.
fn plan_render(
    device: &Device,
    case: &RenderCase,
    libraries: RenderLibraries<'_>,
) -> Result<(), Error> {
    planned_draw(device, case, libraries).map(|_| ())
}

/// [`plan_render`]'s pipeline, plan, and instantiated draw.
fn planned_draw(
    device: &Device,
    case: &RenderCase,
    libraries: RenderLibraries<'_>,
) -> Result<
    (
        metal2vulkan::pipeline::RenderPipeline,
        metal2vulkan::bind::BindingPlan,
        metal2vulkan::bind::DrawCommands,
    ),
    Error,
> {
    // Every buffer the case binds, vertex buffers and argument-buffer buffers included, with a
    // device address equal to its guest base, as the render executor binds them on a device with
    // buffer device addresses: a slot the pipeline reaches through a physical pointer needs one
    // to plan.
    let shape = render_shape(case, true).unwrap();
    let pipeline = compile_in_process(device, case, libraries, &shape)?;
    assert_eq!(
        pipeline.graphics.rasterizer_discard,
        case.fragment.is_none()
    );
    let own = |guest| BufferAddress {
        guest,
        host: Some(guest),
    };
    let addresses = case
        .stages()
        .flat_map(|(stage, f)| {
            f.resources
                .buffers
                .keys()
                .chain(f.resources.function_tables.keys())
                .map(move |&index| (stage, index))
        })
        .chain(
            case.vertex_buffers
                .iter()
                .map(|buffer| (ShaderStage::Vertex, buffer.buffer_index)),
        )
        .map(|(stage, index)| ((stage, index), own(synthetic_address(stage, index).guest)))
        .collect();
    let embedded = synthetic_used_addresses(case)
        .unwrap()
        .into_iter()
        .map(|(key, address)| (key, own(address.guest)))
        .collect();
    let state = bind_state(case, &addresses, &embedded).unwrap();
    assert_eq!(state.shape(), shape);
    let plan = pipeline.plan(device, &shape).unwrap();
    let chooses_slice = pipeline.dependencies.iter().any(|dependency| {
        matches!(
            dependency.fact,
            metal2vulkan::pipeline::RuntimeFact::RenderTargetArrayLength { .. }
        )
    });
    assert_eq!(
        plan.variant.dropped_vertex_layer,
        chooses_slice && case.render_target_array_length <= 1,
        "{}",
        case.vertex.entry
    );
    let mut rest = plan.variant.clone();
    rest.dropped_vertex_layer = false;
    assert!(rest.is_empty(), "{}", case.vertex.entry);
    assert!(plan.diagnostics.is_empty(), "{:?}", plan.diagnostics);
    let commands = plan.instantiate_draw(&state, &case.draw).unwrap();
    // An indexed case's draw instantiates indexed, every other one direct.
    match (&case.index_buffer, commands.draw) {
        (None, vk::DrawCommand::Draw { .. }) | (Some(_), vk::DrawCommand::DrawIndexed { .. }) => {}
        (_, other) => panic!("{}: {other:?}", case.vertex.entry),
    }
    assert!(!commands.indirect_needs_arguments);
    assert_eq!(
        commands.buffer_offsets.len(),
        pipeline.vertex_input.bindings.len()
    );
    Ok((pipeline, plan, commands))
}

/// A tessellation case maps onto the Metal oracle's patch draw: its tessellation state and factor
/// bytes (`executor_contract::tessellation`), its control points per patch control point and its
/// patch attribute per patch at the oracle's slots, one patch draw of the function's control points
/// over every patch and instance through the observer, and the factor buffer in the bind state. It
/// compiles to a tessellation pipeline and plans on committed profiles: the control stage reads the
/// factors over exactly the draw's bytes, and the draw is one vertex per patch.
#[test]
fn a_tessellation_case_maps_onto_the_oracles_patch_draw() {
    use crate::case::{
        AttributeInput, CullMode, TessellationDraw, TessellationFactors as Factors,
        TessellationPartition, TessellationWinding,
    };
    use crate::executor_contract::tessellation as shared;
    let Some(bitcode) = assembled(&fixtures_dir().join("engine/tessellation.ll")) else {
        return;
    };
    let mut authored = crate::companion::tests::case(
        Stage::Vertex,
        "tess_triangle_points",
        TextureFormat::Rgba32Float,
    );
    authored.vertex_observation = Some(VertexObservation::Varying { location: 0 });
    authored.output = OutputSelection::RenderTarget {
        index: 0,
        origin: [0, 0],
        dimensions: [1, 1],
    };
    let input = |location, format, stride: u32, records: u32| AttributeInput {
        location,
        format,
        stride,
        dynamic_stride: false,
        bytes_b64: b64(&vec![0; (stride * records) as usize]),
    };
    let factors = vec![
        Factors {
            edge_f16: vec![0x3c00, 0x4000, 0x4400],
            inside_f16: vec![0x4200],
        },
        Factors {
            edge_f16: vec![0x0000, 0x3c00, 0x3c00],
            inside_f16: vec![0x3c00],
        },
    ];
    authored.draw = None;
    authored.tessellation = Some(TessellationDraw {
        factors: factors.clone(),
        instance_count: 3,
        amplification_count: 1,
        control_points: vec![
            input(0, AttributeFormat::Float4, 16, 6),
            input(1, AttributeFormat::Half2, 4, 6),
        ],
        patch_inputs: vec![input(5, AttributeFormat::Float4, 16, 2)],
        partition_mode: TessellationPartition::FractionalOdd,
        output_winding: TessellationWinding::Clockwise,
        cull_mode: CullMode::Back,
    });
    let draw = authored.tessellation.clone().unwrap();
    let (interface, observer) = checked_companion(&authored, &bitcode);
    let render = render_case(
        &authored,
        &interface,
        WITHOUT_SLICE,
        observer.as_ref(),
        None,
    )
    .unwrap();
    let bytes = shared::factor_bytes(&factors);
    assert_eq!(bytes.len(), 16);
    // The case's partition mode, winding, and cull mode.
    let state = shared::descriptor(&draw);
    assert_eq!(
        state.partition_mode,
        metal::TessellationPartitionMode::FractionalOdd
    );
    assert_eq!(state.output_winding_order, metal::Winding::Clockwise);
    assert_eq!(
        render.assembly,
        Assembly::Patches {
            tessellation: state,
            factors: bytes.clone(),
        }
    );
    assert_eq!(render.cull_mode, metal::CullMode::Back);
    assert_eq!(
        cull_mode_flags(render.cull_mode),
        Ok(metal2vulkan::vk::CullModeFlags::BACK)
    );
    assert_eq!(
        render.draw,
        DrawCall::Patches {
            control_points: 3,
            patch_start: 0,
            patch_count: 2,
            patch_index: None,
            instance_count: 3,
            base_instance: 0,
            control_point_index: None,
        }
    );
    assert_eq!(
        render.fragment.as_ref().map(|f| f.entry.as_str()),
        Some(crate::companion::OBSERVER_FRAGMENT_ENTRY)
    );
    // The oracle's slots: the highest free slot in location order, each stepped as its kind.
    let steps = render
        .vertex_buffers
        .iter()
        .map(|b| (b.location, b.buffer_index, b.step))
        .collect::<Vec<_>>();
    assert_eq!(
        steps,
        [
            (0, 30, metal::VertexStepFunction::PerPatchControlPoint),
            (1, 29, metal::VertexStepFunction::PerPatchControlPoint),
            (5, 28, metal::VertexStepFunction::PerPatch),
        ]
    );
    let template = render_template(&render);
    assert_eq!(template.tessellation, Some(state));
    assert_eq!(
        template.input_primitive_topology,
        metal::PrimitiveTopologyClass::Unspecified
    );
    let layouts = &template.vertex_descriptor.as_ref().unwrap().layouts;
    assert_eq!(
        layouts[&28].step_function,
        metal::VertexStepFunction::PerPatch
    );
    assert_eq!(
        assembly_input(&render.assembly),
        (vk::PrimitiveTopology::PATCH_LIST, false)
    );
    let shape = render_shape(&render, true).unwrap();
    assert!(shape.tessellation_factor_buffer);

    let companion = observer.as_ref().unwrap().opaque_bitcode().unwrap();
    let libraries = RenderLibraries {
        vertex: &bitcode,
        fragment: Some(&companion),
    };
    for stem in ["lavapipe", "moltenvk-m3"] {
        let (pipeline, plan, commands) = planned_draw(&device(stem), &render, libraries).unwrap();
        let tessellation = pipeline
            .graphics
            .tessellation
            .expect("a tessellation pipeline");
        assert_eq!(tessellation.control_points, 3, "{stem}");
        let factors = plan
            .descriptors
            .iter()
            .find(|write| write.source == metal2vulkan::bind::WriteSource::TessellationFactors)
            .unwrap_or_else(|| panic!("{stem}: no factor write"));
        let range = commands
            .descriptor_ranges
            .iter()
            .find(|r| (r.set, r.binding) == (factors.set, factors.binding))
            .unwrap_or_else(|| panic!("{stem}: no factor range"));
        assert_eq!(range.range.offset, 0, "{stem}");
        assert!(range.range.range >= bytes.len() as u64, "{stem}");
        assert_eq!(
            commands.draw,
            vk::DrawCommand::Draw {
                vertex_count: 2,
                instance_count: 3,
                first_vertex: 0,
                first_instance: 0,
            },
            "{stem}"
        );
        // The control stage's push range: no residual offset, no instance stride, scale 1, no
        // patch index buffer, and no control-point index buffer.
        let control = commands
            .push_constants
            .iter()
            .find(|w| w.stages == vk::ShaderStageFlags::TESSELLATION_CONTROL)
            .unwrap_or_else(|| panic!("{stem}: no control push"));
        let words = [0u32, 0, 1.0f32.to_bits(), 0, 0, 0]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(control.bytes, words, "{stem}");
    }
}

/// The owned vertex/fragment pairs the device tests draw compile into one pipeline and plan
/// cleanly on committed profiles.
#[test]
fn owned_render_pipelines_plan_without_variants_or_diagnostics() {
    let public = fixtures_dir();
    let (Some(triangle), Some(constant), Some(varying)) = (
        assembled(&public.join("vertex_varying_triangle.ll")),
        assembled(&public.join("fragment_constant_color.ll")),
        assembled(&public.join("fragment_varying_color.ll")),
    ) else {
        return;
    };
    let target = ColorTarget {
        format: TextureFormat::Rgba32Float,
        dimensions: [4, 4],
        samples: 1,
        bytes: vec![0; 256],
    };
    let vertices = vec![
        VertexBuffer {
            location: 0,
            buffer_index: 30,
            format: AttributeFormat::Float2,
            stride: 8,
            dynamic: false,
            step: metal::VertexStepFunction::PerVertex,
            bytes: vec![0; 24],
        },
        VertexBuffer {
            location: 1,
            buffer_index: 29,
            format: AttributeFormat::Float2,
            stride: 8,
            dynamic: false,
            step: metal::VertexStepFunction::PerVertex,
            bytes: vec![0; 24],
        },
    ];
    for stem in ["lavapipe", "moltenvk-m3"] {
        let device = device(stem);
        for (fragment, bitcode) in [
            ("fragment_constant_color", &constant),
            ("fragment_varying_color", &varying),
        ] {
            let case = two_stage_case(
                "vertex_varying_triangle",
                fragment,
                vertices.clone(),
                BTreeMap::from([(0, target.clone())]),
                3,
            );
            plan_render(
                &device,
                &case,
                RenderLibraries {
                    vertex: &triangle,
                    fragment: Some(bitcode),
                },
            )
            .unwrap_or_else(|error| panic!("{stem} {fragment}: {error}"));
        }
    }
}

/// A render case's argument-buffer buffers are its own stage's: its owner covers the pointer field,
/// each is a used buffer resident for reading and writing in the vertex and fragment stages (the
/// Metal oracle's `useResource:usage:stages:`), and its final range is an output. The public
/// raster-ordered argument-buffer fragment case maps so, compiles for the shape the executor binds,
/// and plans with its pointee in the root table, host and guest address (what a loaded pointer is
/// translated through). A texture array's elements are the stage's textures at their slots, and an
/// element output is that texture's region.
#[test]
fn render_cases_bind_argument_buffer_buffers_and_texture_arrays() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus");
    let store = CorpusStore::new(&root);
    let sources = public_sources().unwrap();
    let (case, row) = (0..64)
        .flat_map(|shard| store.read_case_shard(shard).unwrap())
        .filter(|case| {
            matches!(case.stage, Stage::Vertex | Stage::Fragment)
                && !case.argument_buffer_buffers.is_empty()
        })
        .find_map(|case| {
            let row = sources
                .iter()
                .find(|row| row.air_sha256 == case.air_sha256 && row.entry == case.entry)?;
            Some((case, row.clone()))
        })
        .expect("a public render case with an argument-buffer buffer");
    let checked = crate::check::check_case_against_source(&root, case.clone(), &row)
        .unwrap_or_else(|errors| panic!("{}: {errors:?}", case.name));
    let interface = &checked.primary.interface;
    let companion = checked.primary.companion.as_ref();
    let render = render_case(&case, interface, &row.air_ll, companion, None).unwrap();
    let own = render.fragment.as_ref().expect("a fragment case");
    let key = (
        case.argument_buffer_buffers[0].buffer_binding,
        case.argument_buffer_buffers[0].field_offset,
    );
    assert_eq!(
        own.resources
            .argument_buffer_buffers
            .keys()
            .copied()
            .collect::<Vec<_>>(),
        [key]
    );
    assert!(own.resources.buffers[&key.0].len() >= key.1 as usize + 8);
    assert!(render.vertex.resources.argument_buffer_buffers.is_empty());
    let pointee_length = own.resources.argument_buffer_buffers[&key].len();
    let OutputSelection::ArgumentBufferBuffer { offset, length, .. } = case.output else {
        panic!("{:?}", case.output);
    };
    assert_eq!(
        render.output,
        RenderOutput::ArgumentBufferBuffer {
            stage: ShaderStage::Fragment,
            owner: key.0,
            field_offset: key.1,
            offset,
            length,
        }
    );
    let finals = RenderFinals {
        argument_buffer_buffers: BTreeMap::from([(
            (ShaderStage::Fragment, key.0, key.1),
            (0..pointee_length).map(|byte| byte as u8).collect(),
        )]),
        ..RenderFinals::default()
    };
    let window = |offset: u64, length: u64| RenderOutput::ArgumentBufferBuffer {
        stage: ShaderStage::Fragment,
        owner: key.0,
        field_offset: key.1,
        offset,
        length,
    };
    assert_eq!(
        select_render_output(window(2, 3), &finals),
        Ok(vec![2, 3, 4])
    );
    assert!(select_render_output(window(pointee_length as u64, 1), &finals).is_err());
    let mut vertex_side = window(0, 1);
    if let RenderOutput::ArgumentBufferBuffer { stage, .. } = &mut vertex_side {
        *stage = ShaderStage::Vertex;
    }
    assert!(select_render_output(vertex_side, &finals).is_err());

    // The bind state: the owner at its address, the pointee a used buffer of both stages.
    let address = |guest: u64| BufferAddress {
        guest,
        host: Some(guest + 0x10_0000_0000),
    };
    let addresses = own
        .resources
        .buffers
        .keys()
        .map(|&index| {
            (
                (ShaderStage::Fragment, index),
                address(0x1000_0000 * u64::from(index + 1)),
            )
        })
        .collect();
    let pointee = address(0x7000_0000);
    let embedded = BTreeMap::from([((ShaderStage::Fragment, key.0, key.1), pointee)]);
    let state = bind_state(&render, &addresses, &embedded).unwrap();
    let [used] = state.used_resources.as_slice() else {
        panic!("{:?}", state.used_resources);
    };
    assert_eq!(used.token, 0);
    assert_eq!(
        (used.gpu_address, used.device_address, used.length),
        (Some(pointee.guest), pointee.host, pointee_length as u64)
    );
    assert_eq!(
        used.usage,
        metal::ResourceUsage::READ | metal::ResourceUsage::WRITE
    );
    assert_eq!(used.stages, RenderStages::VERTEX | RenderStages::FRAGMENT);

    // It compiles for the shape the executor binds, and plans with the pointee resident.
    let scratch = crate::ScratchDir::new("public-render-argument-buffer").unwrap();
    let ll = scratch.path().join("case.ll");
    std::fs::write(&ll, &row.air_ll).unwrap();
    let Some(bitcode) = assembled(&ll) else {
        return;
    };
    let generated = render
        .companion
        .as_ref()
        .expect("a fragment case draws with its companion")
        .opaque_bitcode()
        .unwrap();
    let libraries = RenderLibraries {
        vertex: &generated,
        fragment: Some(&bitcode),
    };
    for stem in ["lavapipe", "moltenvk-m3"] {
        let device = device(stem);
        let shape = render_shape(&render, true).unwrap();
        assert_eq!(state.shape(), shape, "{stem}");
        let pipeline = compile_in_process(&device, &render, libraries, &shape)
            .unwrap_or_else(|error| panic!("{stem}: compile: {error}"));
        let plan = pipeline.plan(&device, &shape).unwrap();
        assert!(
            plan.diagnostics.is_empty(),
            "{stem}: {:?}",
            plan.diagnostics
        );
        let commands = plan.instantiate_draw(&state, &render.draw).unwrap();
        // A root-table entry is 32 bytes: host, guest, size, and bind offset.
        let word =
            |entry: &[u8], at: usize| u64::from_le_bytes(entry[at..at + 8].try_into().unwrap());
        assert!(
            commands.root_tables.iter().any(|(_, table)| table
                .bytes
                .as_chunks::<32>()
                .0
                .iter()
                .any(|entry| [word(entry, 0), word(entry, 8)]
                    == [pointee.host.unwrap(), pointee.guest])),
            "{stem}: the pointee is not resident"
        );
    }

    // A texture array's elements are the fragment stage's textures at their slots.
    let mut arrays = case.clone();
    arrays
        .texture_arrays
        .push(crate::case::TextureArrayResource {
            binding: 3,
            role: crate::case::ResourceRole::Output,
            texture_type: crate::case::TextureType::D2,
            format: TextureFormat::R8Unorm,
            sample_count: 1,
            overrides_texture_at_base: false,
            elements: [[1u8], [2u8]]
                .iter()
                .map(|bytes| crate::case::TextureArrayElement {
                    dimensions: [1, 1, 1],
                    bytes_b64: None,
                    initial_bytes_b64: Some(b64(bytes)),
                })
                .collect(),
        });
    arrays.output = OutputSelection::TextureArrayElement {
        binding: 3,
        element: 1,
        origin: [0, 0, 0],
        dimensions: [1, 1, 1],
    };
    let render = render_case(&arrays, interface, &row.air_ll, companion, None).unwrap();
    let textures = &render.fragment.as_ref().unwrap().resources.textures;
    assert_eq!(
        textures
            .iter()
            .map(|(&slot, texture)| (slot, texture.bytes.clone()))
            .collect::<Vec<_>>(),
        [(3, vec![1]), (4, vec![2])]
    );
    assert!(render.vertex.resources.textures.is_empty());
    assert_eq!(
        render.output,
        RenderOutput::Texture {
            stage: ShaderStage::Fragment,
            binding: 4,
            origin: [0, 0, 0],
            dimensions: [1, 1, 1],
        }
    );

    // An argument-buffer texture of the fragment stage is a used texture of both stages whose token
    // follows the buffers' and whose resource id its owner holds at its field; a command buffer is
    // its record storage, an argument-buffer buffer (initialized under its descriptor) whose guest
    // address its owner holds, and a pipeline state is its token there; the texture's region is an
    // output. An argument-buffer sampler is an argument sampler of the bind state whose id its
    // owner holds at its field.
    use crate::candidate::engine::argument_buffers::{
        field_words, records_storage, resource_id, sampler_id,
    };
    let mut members = case.clone();
    let (texture_field, command_field, pipeline_field) = (key.1 + 64, key.1 + 72, key.1 + 80);
    let sampler_field = key.1 + 88;
    members
        .argument_buffer_samplers
        .push(crate::case::ArgumentBufferSamplerResource {
            buffer_binding: key.0,
            field_offset: sampler_field,
            address_mode: crate::case::SamplerAddressMode::Repeat,
            min_filter: crate::case::SamplerFilter::Nearest,
            mag_filter: crate::case::SamplerFilter::Linear,
            mip_filter: crate::case::SamplerMipFilter::NotMipmapped,
            normalized_coordinates: true,
            compare_function: None,
        });
    members
        .argument_buffer_textures
        .push(crate::case::ArgumentBufferTextureResource {
            buffer_binding: key.0,
            field_offset: texture_field,
            role: crate::case::ResourceRole::Output,
            texture_type: crate::case::TextureType::D2,
            format: TextureFormat::R8Unorm,
            dimensions: [2, 1, 1],
            sample_count: 1,
            bytes_b64: None,
            initial_bytes_b64: Some(b64(&[7, 8])),
        });
    members.command_buffers.push(
        serde_json::from_value(serde_json::json!({
            "buffer_binding": key.0,
            "field_offset": command_field,
            "capacity": 1,
            "descriptor": {"kind": "render", "max_fragment_buffer_bind_count": 0,
                           "max_vertex_buffer_bind_count": 0,
                           "support_dynamic_attribute_stride": false},
            "execution_range": [0, 1],
            "inherit_buffers": true,
            "inherit_pipeline_state": true
        }))
        .unwrap(),
    );
    members
        .pipeline_states
        .push(crate::case::indirect_commands::PipelineResource {
            buffer_binding: key.0,
            field_offset: pipeline_field,
            token: 3,
            air_sha256: "22".repeat(32),
            case_id: "33".repeat(32),
        });
    members.output = OutputSelection::ArgumentBufferTexture {
        buffer_binding: key.0,
        field_offset: texture_field,
        origin: [1, 0, 0],
        dimensions: [1, 1, 1],
    };
    let render = render_case(&members, interface, &row.air_ll, companion, None).unwrap();
    let fragment = &render.fragment.as_ref().unwrap().resources;
    assert!(fragment.buffers[&key.0].len() >= sampler_field as usize + 8);
    assert_eq!(
        fragment
            .argument_buffer_samplers
            .keys()
            .copied()
            .collect::<Vec<_>>(),
        [(key.0, sampler_field)]
    );
    assert_eq!(
        fragment
            .argument_buffer_textures
            .keys()
            .copied()
            .collect::<Vec<_>>(),
        [(key.0, texture_field)]
    );
    let records = &fragment.argument_buffer_buffers[&(key.0, command_field)];
    let command = &members.command_buffers[0];
    let mut initialized = vec![0; command.record_storage_bytes().unwrap()];
    command.initialize_records(&mut initialized).unwrap();
    assert_eq!(records, &initialized);
    assert_eq!(
        fragment.command_buffers,
        BTreeMap::from([((key.0, command_field), command.clone())])
    );
    assert_eq!(
        fragment.objects,
        BTreeMap::from([((key.0, pipeline_field), 3)])
    );
    let member = (ShaderStage::Fragment, key.0, texture_field);
    let tokens = texture_tokens(&render).unwrap();
    assert_eq!(tokens, BTreeMap::from([(member, 2)]));
    // The command buffer's ID is 1, its record storage at that ID's records address with a
    // device address of its own.
    let command_key = (ShaderStage::Fragment, key.0, command_field);
    let ids = records_ids(&render).unwrap();
    assert_eq!(ids, BTreeMap::from([(command_key, 1)]));
    assert_eq!(
        synthetic_used_addresses(&render).unwrap()[&command_key],
        records_storage(1, None)
    );
    let storage = records_storage(1, Some(0x7100_0000));
    let embedded = BTreeMap::from([
        ((ShaderStage::Fragment, key.0, key.1), pointee),
        (command_key, storage),
    ]);
    let state = bind_state(&render, &addresses, &embedded).unwrap();
    let [_, stored, used] = state.used_resources.as_slice() else {
        panic!("{:?}", state.used_resources);
    };
    assert_eq!(
        (
            stored.class,
            stored.token,
            stored.gpu_address,
            stored.device_address,
            stored.length
        ),
        (
            metal2vulkan::bind::ResourceClass::Buffer,
            1,
            Some(storage.guest),
            storage.host,
            records.len() as u64
        )
    );
    assert!(
        matches!(used.class, metal2vulkan::bind::ResourceClass::Texture(_)),
        "{used:?}"
    );
    assert_eq!(
        (used.token, used.resource_id, used.stages),
        (
            2,
            Some(resource_id(2)),
            RenderStages::VERTEX | RenderStages::FRAGMENT
        )
    );
    assert_eq!(render_shape(&render, true).unwrap(), state.shape());
    let sampler = (ShaderStage::Fragment, key.0, sampler_field);
    let samplers = sampler_ids(&render).unwrap();
    assert_eq!(
        samplers,
        BTreeMap::from([(sampler, sampler_id(0).unwrap())])
    );
    let [listed] = state.argument_samplers.as_slice() else {
        panic!("{:?}", state.argument_samplers);
    };
    assert_eq!(listed.resource_id, sampler_id(0).unwrap());
    assert_eq!(
        listed.desc,
        resources::argument_sampler_descriptor(&members.argument_buffer_samplers[0])
    );
    let words = field_words(&embedded, &ids, &tokens, &samplers, render.objects()).unwrap();
    assert_eq!(
        words,
        BTreeMap::from([
            ((ShaderStage::Fragment, key.0, key.1), pointee.guest),
            (member, resource_id(2)),
            (command_key, 1),
            ((ShaderStage::Fragment, key.0, pipeline_field), 3),
            (sampler, sampler_id(0).unwrap()),
        ])
    );
    // The run checks each command buffer's final records: as initialized they decode, and a set
    // fault flag (an encoding the descriptor's policy refused) fails the run.
    let mut finals = BTreeMap::from([(command_key, records.clone())]);
    assert_eq!(
        argument_buffers::validate_records(render.command_buffers(), &finals),
        Ok(())
    );
    finals.get_mut(&command_key).unwrap()[4] = 1;
    assert!(
        argument_buffers::validate_records(render.command_buffers(), &finals)
            .unwrap_err()
            .contains("encoding failed"),
    );
    let finals = RenderFinals {
        argument_buffer_textures: BTreeMap::from([(
            member,
            fragment.argument_buffer_textures[&(key.0, texture_field)].clone(),
        )]),
        ..RenderFinals::default()
    };
    assert_eq!(select_render_output(render.output, &finals), Ok(vec![8]));
}

/// The public case whose vertex function chooses its render target slice
/// (`vertex_amplification_id_layer`) is drawn in the one-layer pass the Metal oracle draws it in:
/// its render case and bind state carry the length the shared derivation gives, and the
/// pipeline compiled for that shape is the variant that does not write `Layer`.
#[test]
fn a_slice_choosing_case_binds_the_oracles_one_layer_pass() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus");
    let store = CorpusStore::new(&root);
    let sources = public_sources().unwrap();
    let (case, row) = (0..64)
        .flat_map(|shard| store.read_case_shard(shard).unwrap())
        .filter(|case| case.entry == "vertex_amplification_id_layer")
        .find_map(|case| {
            let row = sources
                .iter()
                .find(|row| row.air_sha256 == case.air_sha256 && row.entry == case.entry)?;
            Some((case, row.clone()))
        })
        .expect("the public slice-choosing vertex case");
    let checked = crate::check::check_case_against_source(&root, case.clone(), &row)
        .unwrap_or_else(|errors| panic!("{}: {errors:?}", case.name));
    let interface = &checked.primary.interface;
    let render = render_case(
        &case,
        interface,
        &row.air_ll,
        checked.primary.companion.as_ref(),
        checked.primary.reflection.fragment_imageblock.as_ref(),
    )
    .unwrap();
    assert_eq!(
        render.render_target_array_length,
        crate::executor_contract::render_target_array::render_target_array_length(
            case.stage,
            &row.air_ll
        )
    );
    assert_eq!(render.render_target_array_length, 1);
    let shape = render_shape(&render, true).unwrap();
    assert_eq!(shape.render_target_array_length, 1);

    let scratch = crate::ScratchDir::new("slice-choosing-case").unwrap();
    let ll = scratch.path().join("case.ll");
    std::fs::write(&ll, &row.air_ll).unwrap();
    let Some(bitcode) = assembled(&ll) else {
        return;
    };
    let observer = render
        .companion
        .as_ref()
        .expect("an observed vertex case")
        .opaque_bitcode()
        .unwrap();
    let pipeline = compile_in_process(
        &device("moltenvk-m3"),
        &render,
        RenderLibraries {
            vertex: &bitcode,
            fragment: Some(&observer),
        },
        &shape,
    )
    .unwrap();
    assert!(!pipeline.graphics.declares_layer);
    assert!(pipeline
        .dependencies
        .iter()
        .any(|dependency| dependency.fact
            == metal2vulkan::pipeline::RuntimeFact::RenderTargetArrayLength {
                layer_dropped: true
            }));
}

/// `bytes` as the base64 an authored case spells.
fn b64(bytes: &[u8]) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// Every committed public vertex or fragment case either maps onto a draw (with its generated
/// companion when it needs one) that compiles and plans cleanly on committed profiles, or is
/// refused typed.
#[test]
fn public_render_cases_plan_or_refuse_typed() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus");
    let store = CorpusStore::new(&root);
    let sources = public_sources().unwrap();
    let mut planned = 0;
    let mut with_companion = 0;
    for shard in 0..64 {
        for case in store.read_case_shard(shard).unwrap() {
            let Some(row) = sources
                .iter()
                .find(|row| row.air_sha256 == case.air_sha256 && row.entry == case.entry)
            else {
                continue;
            };
            // A kernel case is no draw, and an object or mesh case is a mesh draw, which
            // `engine::mesh` maps (`mesh_case`), never `render_case`.
            if !matches!(case.stage, Stage::Vertex | Stage::Fragment) {
                continue;
            }
            let scratch = crate::ScratchDir::new("public-render-case").unwrap();
            let ll = scratch.path().join("case.ll");
            std::fs::write(&ll, &row.air_ll).unwrap();
            let Some(bitcode) = assembled(&ll) else {
                return;
            };
            // The interface and companion the prepare path maps with: the checked entry's, under
            // the case's constants.
            let checked = crate::check::check_case_against_source(&root, case.clone(), row)
                .unwrap_or_else(|errors| panic!("{}: {errors:?}", case.name));
            let render = match render_case(
                &case,
                &checked.primary.interface,
                &row.air_ll,
                checked.primary.companion.as_ref(),
                checked.primary.reflection.fragment_imageblock.as_ref(),
            ) {
                Ok(render) => render,
                Err(ExecutorError::NotSupported(
                    NotSupported::NotImplemented(_) | NotSupported::Requirements(_),
                )) => continue,
                Err(other) => panic!("{}: {other:?}", case.name),
            };
            let companion = render
                .companion
                .as_ref()
                .map(|companion| companion.opaque_bitcode().unwrap());
            let libraries = match (&render.companion, &companion) {
                (None, _) => RenderLibraries {
                    vertex: &bitcode,
                    fragment: None,
                },
                (Some(Companion::PassthroughVertex(_)), Some(generated)) => RenderLibraries {
                    vertex: generated,
                    fragment: Some(&bitcode),
                },
                (Some(Companion::ObserverFragment(_)), Some(generated)) => RenderLibraries {
                    vertex: &bitcode,
                    fragment: Some(generated),
                },
                (Some(Companion::MeshStages(_)), Some(_)) => {
                    unreachable!("a render case's companion is no mesh render pipeline's stages")
                }
                (Some(_), None) => unreachable!("the companion's bitcode was assembled"),
            };
            with_companion += usize::from(render.companion.is_some());
            for stem in ["lavapipe", "moltenvk-m3"] {
                match plan_render(&device(stem), &render, libraries) {
                    Ok(()) => planned += 1,
                    // The product's typed refusal of the function, not the executor's.
                    Err(error) => assert!(
                        !crate::engine::Outcome::from(&error).is_defect(),
                        "{stem} {}: {error}",
                        case.name
                    ),
                }
            }
        }
    }
    assert!(planned >= 1, "{planned} public render cases planned");
    assert!(with_companion > 0, "no public case drew with a companion");
}

/// Every installed public tessellation case maps onto the oracle's patch draw in its authored state
/// (partition mode, output winding, cull mode) and plans on lavapipe and moltenvk-m3: a
/// tessellation pipeline of the function's control points, the factor write over the case's factor
/// bytes, and one vertex per patch. A case without a vertex observation draws its patches with
/// rasterization disabled and no observer, and at least one (the domain-point cases) does.
#[test]
fn public_tessellation_cases_plan_their_patch_draws() {
    use crate::executor_contract::tessellation as shared;
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus");
    let store = CorpusStore::new(&root);
    let sources = public_sources().unwrap();
    let mut planned = 0;
    let mut unrasterized = 0;
    for shard in 0..64 {
        for case in store.read_case_shard(shard).unwrap() {
            let Some(draw) = case.tessellation.as_ref() else {
                continue;
            };
            let Some(row) = sources
                .iter()
                .find(|row| row.air_sha256 == case.air_sha256 && row.entry == case.entry)
            else {
                continue;
            };
            let scratch = crate::ScratchDir::new("public-tessellation-case").unwrap();
            let ll = scratch.path().join("case.ll");
            std::fs::write(&ll, &row.air_ll).unwrap();
            let Some(bitcode) = assembled(&ll) else {
                return;
            };
            let checked = crate::check::check_case_against_source(&root, case.clone(), row)
                .unwrap_or_else(|errors| panic!("{}: {errors:?}", case.name));
            let render = render_case(
                &case,
                &checked.primary.interface,
                &row.air_ll,
                checked.primary.companion.as_ref(),
                checked.primary.reflection.fragment_imageblock.as_ref(),
            )
            .unwrap_or_else(|error| panic!("{}: {error:?}", case.name));
            let factors = shared::factor_bytes(&draw.factors);
            assert_eq!(
                render.assembly,
                Assembly::Patches {
                    tessellation: shared::descriptor(draw),
                    factors: factors.clone(),
                },
                "{}",
                case.name
            );
            assert_eq!(render.cull_mode, shared::cull_mode(draw), "{}", case.name);
            assert!(cull_mode_flags(render.cull_mode).is_ok(), "{}", case.name);
            let companion = match render.companion.as_ref() {
                Some(companion) => Some(companion.opaque_bitcode().unwrap()),
                None => {
                    assert!(
                        case.is_rasterization_disabled_vertex(),
                        "{}: no observer",
                        case.name
                    );
                    unrasterized += 1;
                    None
                }
            };
            let libraries = RenderLibraries {
                vertex: &bitcode,
                fragment: companion.as_deref(),
            };
            for stem in ["lavapipe", "moltenvk-m3"] {
                let (pipeline, plan, commands) = planned_draw(&device(stem), &render, libraries)
                    .unwrap_or_else(|error| panic!("{stem} {}: {error}", case.name));
                let control_points = shared::control_points(&checked.primary.interface).unwrap();
                assert_eq!(
                    pipeline.graphics.tessellation.map(|t| t.control_points),
                    Some(control_points),
                    "{stem} {}",
                    case.name
                );
                let write = plan
                    .descriptors
                    .iter()
                    .find(|write| {
                        write.source == metal2vulkan::bind::WriteSource::TessellationFactors
                    })
                    .unwrap_or_else(|| panic!("{stem} {}: no factor write", case.name));
                let range = commands
                    .descriptor_ranges
                    .iter()
                    .find(|r| (r.set, r.binding) == (write.set, write.binding))
                    .unwrap_or_else(|| panic!("{stem} {}: no factor range", case.name));
                assert!(
                    range.range.range >= factors.len() as u64,
                    "{stem} {}",
                    case.name
                );
                assert_eq!(
                    commands.draw,
                    vk::DrawCommand::Draw {
                        vertex_count: draw.factors.len() as u32,
                        instance_count: draw.instance_count,
                        first_vertex: 0,
                        first_instance: 0,
                    },
                    "{stem} {}",
                    case.name
                );
                planned += 1;
            }
        }
    }
    assert!(planned > 0, "no public tessellation case planned");
    assert!(
        unrasterized > 0,
        "no public patch draw with rasterization disabled"
    );
}

/// The committed profiles whose descriptions have `fragmentShaderPixelInterlock` but not
/// `rasterizationOrderColorAttachmentAccess`: the devices that order a fetch only through the
/// pixel interlock, read from what each states rather than from its name.
fn interlock_only() -> Vec<String> {
    use metal2vulkan::device::VkFeature;
    let stems = crate::engine::committed_profiles().unwrap();
    let stems = stems
        .into_iter()
        .filter(|stem| {
            let description = crate::engine::load_description(stem).unwrap();
            description.has(VkFeature::FragmentShaderPixelInterlock)
                && !description.has(VkFeature::RasterizationOrderColorAttachmentAccess)
        })
        .collect::<Vec<_>>();
    assert!(
        !stems.is_empty(),
        "a committed profile orders fetch only by the interlock"
    );
    stems
}

/// Every installed public framebuffer-fetch case maps onto its fullscreen triangle drawn as its
/// instances, with each read render target attached, and plans on lavapipe (one input-attachment
/// write per color input, rasterization-order access, the feature required). On each committed
/// device that orders fetch only through the pixel interlock ([`interlock_only`]), it plans one
/// storage write per color input (no input attachment), or, where its state blocks that form,
/// refuses its fetch, typed; at least one case plans the storage form. Every case carries no
/// requirement, at least one of them drawing more than one instance (overlapping fragments). The
/// lavapipe pipeline masks every write to a color attachment the function reads but has no output
/// for, which at least one case does. A case of a multisampled pass, which at least one is, reads
/// its colors at each invocation's own sample (sample-rate shading, the pipeline's sample count):
/// on lavapipe as input attachments, on each of those devices with multisampled storage images as
/// storage attachments of the pipeline's samples (`shaderStorageImageMultisample`, the count's
/// `storageImageSampleCounts` bit), and on each without them as planes, single-sample layers of one
/// sample each, which need neither; or it is refused, typed: the pixel's coverage read beside it
/// (`[[sample_mask]]` in, on no device), or a sample count the device's attachments lack. At least
/// one plans in each of the three forms.
#[test]
fn public_framebuffer_fetch_cases_plan_their_ordered_input_attachments() {
    use std::collections::BTreeSet;

    use crate::air::reflect::ResourceKind;
    use metal2vulkan::bind::WriteSource;
    use metal2vulkan::device::VkFeature;
    use metal2vulkan::pipeline::StorageAttachmentView;
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus");
    let store = CorpusStore::new(&root);
    let sources = public_sources().unwrap();
    let mut planned = 0;
    let mut overlapping = 0;
    let mut unwritten = 0;
    let mut stored_form = 0;
    let mut multisampled = 0;
    let mut per_sample: BTreeSet<&str> = BTreeSet::new();
    let interlocked = interlock_only();
    for shard in 0..64 {
        for case in store.read_case_shard(shard).unwrap() {
            let Some(row) = sources
                .iter()
                .find(|row| row.air_sha256 == case.air_sha256 && row.entry == case.entry)
            else {
                continue;
            };
            if case.stage != Stage::Fragment {
                continue;
            }
            let checked = crate::check::check_case_against_source(&root, case.clone(), row)
                .unwrap_or_else(|errors| panic!("{}: {errors:?}", case.name));
            let colors: Vec<u32> = checked
                .primary
                .reflection
                .bindings
                .iter()
                .filter(|binding| binding.kind == ResourceKind::ColorInput)
                .map(|binding| binding.metal_index)
                .collect();
            if colors.is_empty() {
                continue;
            }
            let draw = case.draw.as_ref().expect("a fragment case's draw");
            assert!(checked.requirements().is_empty(), "{}", case.name);
            overlapping += usize::from(draw.instance_count > 1);
            let scratch = crate::ScratchDir::new("public-fetch-case").unwrap();
            let ll = scratch.path().join("case.ll");
            std::fs::write(&ll, &row.air_ll).unwrap();
            let Some(bitcode) = assembled(&ll) else {
                return;
            };
            let render = render_case(
                &case,
                &checked.primary.interface,
                &row.air_ll,
                checked.primary.companion.as_ref(),
                checked.primary.reflection.fragment_imageblock.as_ref(),
            )
            .unwrap_or_else(|error| panic!("{}: {error:?}", case.name));
            for color in &colors {
                assert!(render.color_targets.contains_key(color), "{}", case.name);
            }
            let companion = render
                .companion
                .as_ref()
                .map(|companion| companion.opaque_bitcode().unwrap())
                .unwrap_or_else(|| panic!("{}: no passthrough vertex", case.name));
            let libraries = RenderLibraries {
                vertex: &companion,
                fragment: Some(&bitcode),
            };
            // Each invocation reads its own sample, in the form the device orders.
            if let Some(samples) = case.raster_sample_count {
                let mut expected = colors.clone();
                expected.sort_unstable();
                let interlocked = interlocked.iter().map(String::as_str);
                for stem in std::iter::once("lavapipe").chain(interlocked) {
                    let described = |feature| {
                        let path = crate::engine::profiles_dir().join(format!("{stem}.json"));
                        let text = std::fs::read_to_string(path).unwrap();
                        DeviceDescription::from_khronos_json(&text, None)
                            .unwrap()
                            .features
                            .contains(&feature)
                    };
                    match planned_draw(&device(stem), &render, libraries) {
                        Ok((pipeline, _, _)) => {
                            let g = &pipeline.graphics;
                            assert_eq!(g.rasterization_samples.as_raw(), samples, "{stem}");
                            let required = &pipeline.requirements;
                            assert!(required.features.contains(&VkFeature::SampleRateShading));
                            let sample_count =
                                metal2vulkan::device::VkProperty::StorageImageSampleCount(samples);
                            let mut fetched: Vec<u32> = if stem == "lavapipe" {
                                assert!(g.storage_attachments.is_empty(), "{}", case.name);
                                assert!(g.rasterization_order_attachment_access);
                                per_sample.insert("input attachments");
                                g.input_attachments.iter().map(|a| a.color_index).collect()
                            } else {
                                assert!(g.input_attachments.is_empty(), "{}", case.name);
                                let image = described(VkFeature::ShaderStorageImageMultisample);
                                let view = if image {
                                    StorageAttachmentView::Attachment
                                } else {
                                    StorageAttachmentView::SamplePlanes { layers: samples }
                                };
                                assert!(
                                    g.storage_attachments.iter().all(|a| a.view == view),
                                    "{stem} {}",
                                    case.name
                                );
                                assert_eq!(
                                    required
                                        .features
                                        .contains(&VkFeature::ShaderStorageImageMultisample),
                                    image,
                                    "{stem} {}",
                                    case.name
                                );
                                assert_eq!(
                                    required.properties.contains(&sample_count),
                                    image,
                                    "{stem} {}",
                                    case.name
                                );
                                per_sample.insert(if image {
                                    "multisampled storage"
                                } else {
                                    "planes"
                                });
                                g.storage_attachments
                                    .iter()
                                    .map(|a| a.color_index)
                                    .collect()
                            };
                            fetched.sort_unstable();
                            assert_eq!(fetched, expected, "{stem} {}", case.name);
                        }
                        Err(Error::Unsupported(refusal)) => {
                            use metal2vulkan::error::{Feature, Needs};
                            match (&refusal.feature, &refusal.needs) {
                                (Feature::PixelCoverageAtSampleRate, Needs::NotRepresentable)
                                | (Feature::RasterSampleCount, Needs::AnyOf(_)) => {}
                                _ => panic!("{stem} {}: {refusal:?}", case.name),
                            }
                        }
                        Err(other) => panic!("{stem} {}: {other:?}", case.name),
                    }
                }
                multisampled += 1;
                continue;
            }
            let (pipeline, plan, commands) = planned_draw(&device("lavapipe"), &render, libraries)
                .unwrap_or_else(|error| panic!("lavapipe {}: {error}", case.name));
            assert!(
                pipeline.graphics.rasterization_order_attachment_access,
                "{}",
                case.name
            );
            assert!(pipeline
                .requirements
                .features
                .contains(&VkFeature::RasterizationOrderColorAttachmentAccess));
            // A color attachment the fragment function reads but has no output for keeps its
            // contents, as Metal kept it in case f39cf462: the pipeline the executor builds from
            // the artifact masks every write to it, and keeps the descriptor's mask on the others.
            let written: BTreeSet<u32> = checked
                .primary
                .reflection
                .render_targets
                .iter()
                .map(|target| target.location)
                .collect();
            for (index, state) in pipeline.graphics.color_attachments.iter().enumerate() {
                let Some(state) = state else { continue };
                let index = index as u32;
                assert_eq!(
                    state.write_mask.is_empty(),
                    !written.contains(&index),
                    "{} attachment {index}",
                    case.name
                );
                unwritten += usize::from(!written.contains(&index));
            }
            let mut read: Vec<u32> = plan
                .descriptors
                .iter()
                .filter_map(|write| match write.source {
                    WriteSource::Attachment { color_index } => Some(color_index),
                    _ => None,
                })
                .collect();
            read.sort_unstable();
            let mut expected = colors.clone();
            expected.sort_unstable();
            assert_eq!(read, expected, "{}", case.name);
            assert_eq!(
                commands.draw,
                vk::DrawCommand::Draw {
                    vertex_count: 3,
                    instance_count: draw.instance_count,
                    first_vertex: 0,
                    first_instance: 0,
                },
                "{}",
                case.name
            );
            // A device that orders fetch only through the pixel interlock reads (and writes) each
            // color the case reads as a storage image of its own attachment inside the interlock,
            // where the case's state lets that be exact: no input attachment, no color attachment
            // state for it, and a storage write of it in the plan. Where the state blocks that
            // form, the default policy refuses the fetch, typed.
            for stem in interlocked.iter().map(String::as_str) {
                match planned_draw(&device(stem), &render, libraries) {
                    Ok((pipeline, plan, _)) => {
                        let mut stored: Vec<u32> = pipeline
                            .graphics
                            .storage_attachments
                            .iter()
                            .map(|a| a.color_index)
                            .collect();
                        stored.sort_unstable();
                        assert_eq!(stored, expected, "{stem} {}", case.name);
                        assert!(
                            pipeline.graphics.input_attachments.is_empty(),
                            "{stem} {}",
                            case.name
                        );
                        assert!(!pipeline.graphics.rasterization_order_attachment_access);
                        for &color in &expected {
                            assert!(
                                pipeline
                                    .graphics
                                    .color_attachments
                                    .get(color as usize)
                                    .is_none_or(Option::is_none),
                                "{stem} {} attachment {color}",
                                case.name
                            );
                        }
                        let mut written: Vec<u32> = plan
                            .descriptors
                            .iter()
                            .filter_map(|write| match write.source {
                                WriteSource::TileAttachment { color_index, .. } => {
                                    Some(color_index)
                                }
                                _ => None,
                            })
                            .collect();
                        written.sort_unstable();
                        assert_eq!(written, expected, "{stem} {}", case.name);
                        stored_form += 1;
                    }
                    Err(Error::Unsupported(refusal)) => assert_eq!(
                        refusal.feature,
                        metal2vulkan::error::Feature::FramebufferFetch,
                        "{stem} {}",
                        case.name
                    ),
                    Err(other) => panic!("{stem} {}: {other:?}", case.name),
                }
            }
            planned += 1;
        }
    }
    assert!(planned > 0, "no public framebuffer-fetch case planned");
    assert!(
        stored_form > 0,
        "no public framebuffer-fetch case planned in the storage form"
    );
    assert!(
        overlapping > 0,
        "no public framebuffer-fetch case overlaps its fragments"
    );
    assert!(
        unwritten > 0,
        "no public framebuffer-fetch case reads an attachment it does not write"
    );
    assert!(
        multisampled > 0,
        "no public framebuffer-fetch case draws a multisampled pass"
    );
    assert_eq!(
        per_sample,
        BTreeSet::from(["input attachments", "multisampled storage", "planes"]),
        "a multisampled case plans in each per-sample form"
    );
}

/// On `lunarg-desktop-baseline-2023` and `vk12-floor`, which offer no ordered form of framebuffer
/// fetch (neither rasterization-order attachment access nor the pixel interlock), every installed
/// public framebuffer-fetch case plans its color inputs as input attachments with no
/// rasterization-order access, which the consumer orders between draws alone, and its fragment
/// stage notes `PrecisionNote::UnorderedFetchWithinDraw`; or, a multisampled pass, is refused as on
/// every device (the coverage read beside it, or a sample count the attachments lack).
#[test]
fn public_framebuffer_fetch_cases_plan_input_attachments_without_an_ordered_form() {
    use crate::air::reflect::ResourceKind;
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus");
    let store = CorpusStore::new(&root);
    let sources = public_sources().unwrap();
    for stem in ["lunarg-desktop-baseline-2023", "vk12-floor"] {
        let unordered = device(stem);
        let mut fetches = 0;
        for shard in 0..64 {
            for case in store.read_case_shard(shard).unwrap() {
                let Some(row) = sources
                    .iter()
                    .find(|row| row.air_sha256 == case.air_sha256 && row.entry == case.entry)
                else {
                    continue;
                };
                if case.stage != Stage::Fragment {
                    continue;
                }
                let checked = crate::check::check_case_against_source(&root, case.clone(), row)
                    .unwrap_or_else(|errors| panic!("{}: {errors:?}", case.name));
                let colors: Vec<u32> = checked
                    .primary
                    .reflection
                    .bindings
                    .iter()
                    .filter(|binding| binding.kind == ResourceKind::ColorInput)
                    .map(|binding| binding.metal_index)
                    .collect();
                if colors.is_empty() {
                    continue;
                }
                fetches += 1;
                let scratch = crate::ScratchDir::new("public-fetch-unordered").unwrap();
                let ll = scratch.path().join("case.ll");
                std::fs::write(&ll, &row.air_ll).unwrap();
                let Some(bitcode) = assembled(&ll) else {
                    return;
                };
                let render = render_case(
                    &case,
                    &checked.primary.interface,
                    &row.air_ll,
                    checked.primary.companion.as_ref(),
                    checked.primary.reflection.fragment_imageblock.as_ref(),
                )
                .unwrap_or_else(|error| panic!("{}: {error:?}", case.name));
                let companion = render
                    .companion
                    .as_ref()
                    .map(|companion| companion.opaque_bitcode().unwrap())
                    .unwrap_or_else(|| panic!("{}: no passthrough vertex", case.name));
                let libraries = RenderLibraries {
                    vertex: &companion,
                    fragment: Some(&bitcode),
                };
                let pipeline = match planned_draw(&unordered, &render, libraries) {
                    Ok((pipeline, _, _)) => pipeline,
                    // A multisampled pass's refusals of every device's form (the lavapipe test's).
                    Err(Error::Unsupported(refusal)) if case.raster_sample_count.is_some() => {
                        use metal2vulkan::error::{Feature, Needs};
                        match (&refusal.feature, &refusal.needs) {
                            (Feature::PixelCoverageAtSampleRate, Needs::NotRepresentable)
                            | (Feature::RasterSampleCount, Needs::AnyOf(_)) => continue,
                            _ => panic!("{stem} {}: {refusal:?}", case.name),
                        }
                    }
                    Err(other) => panic!("{stem} {}: {other:?}", case.name),
                };
                let mut read: Vec<u32> = (pipeline.graphics.input_attachments.iter())
                    .map(|input| input.color_index)
                    .collect();
                read.sort_unstable();
                let mut colors = colors;
                colors.sort_unstable();
                assert_eq!(read, colors, "{stem} {}", case.name);
                assert!(
                    !pipeline.graphics.rasterization_order_attachment_access,
                    "{stem} {}",
                    case.name
                );
                assert!(unordered_within_draw(&pipeline), "{stem} {}", case.name);
            }
        }
        assert!(fetches > 0, "{stem}: no public framebuffer-fetch case");
    }
}

/// A draw's input assembly restarts exactly the strips, as Metal does, within the topology class
/// the oracle compiles the pipeline for.
#[test]
fn a_draw_restarts_exactly_its_strips_within_its_class() {
    for primitive in [
        Primitive::Point,
        Primitive::Line,
        Primitive::LineStrip,
        Primitive::Triangle,
        Primitive::TriangleStrip,
    ] {
        let (topology, restart) = input_assembly(primitive);
        let strip = matches!(primitive, Primitive::LineStrip | Primitive::TriangleStrip);
        assert_eq!(restart, strip, "{primitive:?}");
        let class = match topology {
            vk::PrimitiveTopology::POINT_LIST => metal::PrimitiveTopologyClass::Point,
            vk::PrimitiveTopology::LINE_LIST | vk::PrimitiveTopology::LINE_STRIP => {
                metal::PrimitiveTopologyClass::Line
            }
            _ => metal::PrimitiveTopologyClass::Triangle,
        };
        assert_eq!(class, topology_class(primitive), "{primitive:?}");
    }
}

/// The render executor's cells follow the checked reflection's master, the one the checker
/// validates the planes against and the Metal oracle's initializer writes and resolver reads, each
/// member at its reflected offset with its reflected size. `imageblock_two_masters` reads `a`
/// through an input view of `Master` (`a` at byte 0 of 16) and writes it through an output view of
/// `Swapped` (`a` at byte 2 of 4): the reflection, and so the cells, follow the larger master,
/// `Master`, whose cell both views reach, not the output view's. Every owned function's cell covers
/// its reflected data size.
#[test]
fn the_cells_follow_the_reflected_master() {
    let views = std::fs::read_to_string(
        metal2vulkan_devtools::engine_dir().join("fragment_imageblock_views.ll"),
    )
    .unwrap();
    let custom =
        std::fs::read_to_string(fixtures_dir().join("fragment_custom_imageblock.ll")).unwrap();
    let mut compared = 0;
    for (ll, entries) in [
        (&custom, &["fragment_custom_imageblock"][..]),
        (
            &views,
            &[
                "imageblock_transition",
                "imageblock_accumulate",
                "imageblock_clear_odd",
                "imageblock_resolve",
                "imageblock_masked",
                "imageblock_two_masters",
            ][..],
        ),
    ] {
        for &entry in entries {
            let reflected = reflected_imageblock(ll, entry)
                .unwrap_or_else(|| panic!("{entry} reflects an imageblock"));
            let layout = super::imageblock::layout(&reflected).unwrap();
            for member in &reflected.members {
                assert_eq!(
                    layout.members[&member.semantic],
                    super::imageblock::Member {
                        offset: member.offset,
                        size: member.size,
                    },
                    "{entry}: {}",
                    member.semantic
                );
            }
            assert_eq!(layout.members.len(), reflected.members.len(), "{entry}");
            assert!(
                layout.cell_bytes >= reflected.sample_size && layout.cell_bytes.is_multiple_of(4),
                "{entry}: a {}-byte cell of a {}-byte imageblock",
                layout.cell_bytes,
                reflected.sample_size
            );
            compared += 1;
        }
    }
    assert_eq!(compared, 7);
    // `Master` is the master: `a` at byte 0 and `b` at byte 2 of a 16-byte cell, beside `tag`,
    // `sum`, and `n`.
    let two =
        super::imageblock::layout(&reflected_imageblock(&views, "imageblock_two_masters").unwrap())
            .unwrap();
    let at = |offset, size| super::imageblock::Member { offset, size };
    assert_eq!(
        two.members,
        BTreeMap::from([
            ("user(a)".into(), at(0, 2)),
            ("user(b)".into(), at(2, 2)),
            ("user(tag)".into(), at(4, 4)),
            ("user(sum)".into(), at(8, 4)),
            ("user(n)".into(), at(12, 2)),
        ])
    );
    assert_eq!(two.cell_bytes, 16);
}

/// Every committed public fragment imageblock case maps onto a draw whose pipeline reaches the
/// pass's imageblock storage, and it plans on the committed profiles with pixel interlock: one
/// storage-image write of the imageblock, the pipeline's words within the case's cell, the
/// interlock required, and fragment stores required exactly when the case writes a member (a member
/// whose role is not `input`). A profile without interlock refuses it typed
/// (`Unsupported{Imageblocks}` naming the feature), the product's refusal, not the executor's.
#[test]
fn public_fragment_imageblock_cases_plan_their_storage_image() {
    use metal2vulkan::bind::WriteSource;
    use metal2vulkan::device::VkFeature;
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus");
    let sources = public_sources().unwrap();
    let mut planned = 0;
    let mut writers = 0;
    for case in CorpusStore::new(&root).read_all_cases().unwrap() {
        let Some(authored) = case.fragment_imageblock.as_ref() else {
            continue;
        };
        let Some(row) = sources
            .iter()
            .find(|row| row.air_sha256 == case.air_sha256 && row.entry == case.entry)
        else {
            continue;
        };
        let scratch = crate::ScratchDir::new("public-imageblock-case").unwrap();
        let ll = scratch.path().join("case.ll");
        std::fs::write(&ll, &row.air_ll).unwrap();
        let Some(bitcode) = assembled(&ll) else {
            return;
        };
        let name = &case.name;
        let checked = crate::check::check_case_against_source(&root, case.clone(), row)
            .unwrap_or_else(|errors| panic!("{name}: {errors:?}"));
        let render = render_case(
            &case,
            &checked.primary.interface,
            &row.air_ll,
            checked.primary.companion.as_ref(),
            checked.primary.reflection.fragment_imageblock.as_ref(),
        )
        .unwrap_or_else(|error| panic!("{name}: {error:?}"));
        let cells = render.imageblock.as_ref().expect("the case's cells");
        let companion = render.companion.as_ref().unwrap().opaque_bitcode().unwrap();
        let libraries = RenderLibraries {
            vertex: &companion,
            fragment: Some(&bitcode),
        };
        let writes_members = authored
            .members
            .iter()
            .any(|member| member.role != crate::case::ResourceRole::Input);
        writers += usize::from(writes_members);
        for stem in ["lavapipe", "moltenvk-m3"] {
            let (pipeline, plan, _) = planned_draw(&device(stem), &render, libraries)
                .unwrap_or_else(|error| panic!("{name} on {stem}: {error}"));
            let state = pipeline
                .graphics
                .fragment_imageblock
                .unwrap_or_else(|| panic!("{name} on {stem}: no imageblock state"));
            assert!(
                (1..=cells.words()).contains(&state.words),
                "{name} on {stem}: {} words",
                state.words
            );
            let features = &pipeline.requirements.features;
            assert!(
                features.contains(&VkFeature::FragmentShaderPixelInterlock),
                "{name} on {stem}"
            );
            assert_eq!(
                features.contains(&VkFeature::FragmentStoresAndAtomics),
                writes_members,
                "{name} on {stem}"
            );
            let writes: Vec<_> = plan
                .descriptors
                .iter()
                .filter(|write| write.source == WriteSource::Imageblock)
                .collect();
            assert_eq!(writes.len(), 1, "{name} on {stem}");
            assert_eq!(
                writes[0].ty,
                vk::DescriptorType::STORAGE_IMAGE,
                "{name} on {stem}"
            );
            planned += 1;
        }
        match plan_render(&device("swiftshader"), &render, libraries) {
            Err(Error::Unsupported(refusal)) => {
                assert_eq!(refusal.feature, metal2vulkan::error::Feature::Imageblocks)
            }
            other => panic!("{name}: swiftshader ran an imageblock without interlock: {other:?}"),
        }
    }
    // fragment-custom-imageblock-depth, the ten cases, nine of which write a member, the
    // imageblock_transition case (a float member read, three written), the two splat cases (two
    // uint members read and written), and the six cases on `rfs_imageblock_cover` (a masked writer
    // and views of two masters, each writing a member).
    assert_eq!((planned, writers), (40, 19));
}

/// A draw whose imageblock the product does not model reaches the render executor's plan as the
/// product's typed refusal, never as a draw: a layered pass (the binder, the per-layer imageblock).
/// A depth attachment behind a writer that does not declare early fragment tests links and plans:
/// the write runs after the depth test (the writer under early tests, or its draws testing first).
#[test]
fn imageblock_draws_the_product_does_not_model_are_refused_typed() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus");
    let case = CorpusStore::new(&root)
        .read_all_cases()
        .unwrap()
        .into_iter()
        .find(|case| case.name == "fragment-custom-imageblock-depth")
        .expect("the committed public case");
    let row = public_sources()
        .unwrap()
        .into_iter()
        .find(|row| row.air_sha256 == case.air_sha256 && row.entry == case.entry)
        .expect("its public source");
    let scratch = crate::ScratchDir::new("refused-imageblock-draw").unwrap();
    let ll = scratch.path().join("case.ll");
    std::fs::write(&ll, &row.air_ll).unwrap();
    let Some(bitcode) = assembled(&ll) else {
        return;
    };
    let checked = crate::check::check_case_against_source(&root, case.clone(), &row)
        .unwrap_or_else(|errors| panic!("{errors:?}"));
    let render = render_case(
        &case,
        &checked.primary.interface,
        &row.air_ll,
        checked.primary.companion.as_ref(),
        checked.primary.reflection.fragment_imageblock.as_ref(),
    )
    .unwrap();
    let companion = render.companion.as_ref().unwrap().opaque_bitcode().unwrap();
    let libraries = RenderLibraries {
        vertex: &companion,
        fragment: Some(&bitcode),
    };
    let lavapipe = device("lavapipe");
    plan_render(&lavapipe, &render, libraries).expect("the case itself plans");
    let imageblocks = |what: &str, refusal: Option<&metal2vulkan::error::Unsupported>| {
        let refusal = refusal.unwrap_or_else(|| panic!("{what}: no typed refusal"));
        assert_eq!(
            refusal.feature,
            metal2vulkan::error::Feature::Imageblocks,
            "{what}"
        )
    };
    // The binder: the pipeline links, and its plan for a pass of two layers is refused.
    let layered = RenderCase {
        render_target_array_length: 2,
        ..render.clone()
    };
    let shape = render_shape(&layered, true).unwrap();
    let pipeline = compile_in_process(&lavapipe, &layered, libraries, &shape)
        .unwrap_or_else(|error| panic!("a layered pass links: {error}"));
    let refused = pipeline.plan(&lavapipe, &shape).err();
    imageblocks(
        "a layered pass",
        match &refused {
            Some(metal2vulkan::error::BindError::Unsupported(refusal)) => Some(refusal),
            _ => None,
        },
    );
    // Link: a depth attachment behind the writer, which declares no early fragment tests.
    let late_depth = RenderCase {
        depth_stencil: Some(crate::literal::LiteralDepthStencil {
            dimensions: [1, 1],
            samples: 1,
            depth: Some(1.0f32.to_le_bytes().to_vec()),
            stencil: None,
        }),
        ..render.clone()
    };
    let shape = render_shape(&late_depth, true).unwrap();
    let pipeline = compile_in_process(&lavapipe, &late_depth, libraries, &shape)
        .unwrap_or_else(|error| panic!("a late depth test links: {error}"));
    pipeline
        .plan(&lavapipe, &shape)
        .unwrap_or_else(|error| panic!("a late depth test plans: {error}"));
}

/// The graphics state of the public fixture `fragment_rate_map_position` drawn by its generated
/// fullscreen vertex function, compiled for lavapipe; `None` without `llvm-as`.
fn rate_map_position_graphics() -> Option<metal2vulkan::pipeline::GraphicsState> {
    let ll =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/public/fragment_rate_map_position.ll");
    let bitcode = assembled(&ll)?;
    let mut authored = crate::companion::tests::case(
        Stage::Fragment,
        "fragment_rate_map_position",
        TextureFormat::Rgba32Float,
    );
    authored.buffers = vec![crate::case::BufferResource {
        binding: 0,
        role: crate::case::ResourceRole::Input,
        bytes_b64: Some(b64(&[0; 8])),
        initial_bytes_b64: None,
        rasterization_rate_map: None,
    }];
    let (interface, companion) = checked_companion(&authored, &bitcode);
    let render = render_case(
        &authored,
        &interface,
        WITHOUT_SLICE,
        companion.as_ref(),
        None,
    )
    .unwrap();
    let vertex = render
        .companion
        .as_ref()
        .expect("a fragment case's generated vertex function")
        .opaque_bitcode()
        .unwrap();
    let shape = render_shape(&render, true).unwrap();
    let pipeline = compile_in_process(
        &device("lavapipe"),
        &render,
        RenderLibraries {
            vertex: &vertex,
            fragment: Some(&bitcode),
        },
        &shape,
    )
    .unwrap();
    Some(pipeline.graphics)
}

/// A rate-mapped pass is planned region by region with the product's compile of its map, for the
/// map's screen, on every device: its census states the compile's physical size and the region
/// count; a plan the device's viewport limits cannot hold, and a vertex function that writes memory
/// in a plan of several regions, are the binder's refusals.
#[test]
fn a_rate_mapped_pass_is_planned_region_by_region() {
    use metal2vulkan::bind::{
        RasterizationRateMap, RateMapDescriptor, RateMapLayerDescriptor, RegionLimits,
    };
    let compile = |screen: [u32; 2], horizontal: Vec<f32>, vertical: Vec<f32>| {
        let mut descriptor = RateMapDescriptor::new(screen);
        descriptor.layers = vec![RateMapLayerDescriptor::new(horizontal, vertical)];
        RasterizationRateMap::new(&descriptor).unwrap()
    };
    let limits = RegionLimits::new([4096, 4096], [-8192.0, 8191.0]);
    // The graphics state of the position fixture drawn by its generated vertex function, which
    // writes no memory; and the same state as if its vertex function wrote memory.
    let Some(pure) = rate_map_position_graphics() else {
        return;
    };
    assert!(!pure.vertex_writes_memory);
    let graphics = |vertex_writes_memory: bool| {
        let mut state = pure.clone();
        state.vertex_writes_memory = vertex_writes_memory;
        state
    };
    // The crate compiles this map to 36x23, as Apple M3 does: 33 screen pixels to 32, then 4 to 4,
    // across.
    let skewed = compile([37, 23], vec![0.5, 1.0], vec![1.0, 0.5]);
    let plan = rate_map_regions(&skewed, &limits, &graphics(false)).unwrap();
    assert_eq!(plan.physical_size(), [36, 23]);
    assert_eq!(
        rate_map_pass_census(&skewed, &plan),
        RateMapPassRun {
            screen_size: [37, 23],
            compiled_physical_size: [36, 23],
            regions: 2,
            precision: [PrecisionNote::RasterizationRateMap].into(),
        }
    );
    let mut layered = graphics(false);
    layered.declares_layer = true;
    for refused in [
        rate_map_regions(&skewed, &limits, &graphics(true)),
        rate_map_regions(&skewed, &limits, &layered),
        rate_map_regions(
            &skewed,
            &RegionLimits::new([16, 4096], [-8192.0, 8191.0]),
            &graphics(false),
        ),
    ] {
        match refused {
            Err(ExecutorError::NotSupported(NotSupported::Bind(why))) => {
                assert!(why.starts_with("rate-mapped render pass: "), "{why}");
            }
            other => panic!("expected the binder's refusal, got {other:?}"),
        }
    }
    // 32x32 at half quality compiles to the identity, as Apple M3 compiles it: one region, which
    // admits a vertex function that writes memory.
    let identity = compile([32, 32], vec![0.5], vec![0.5]);
    assert!(identity.is_identity());
    let plan = rate_map_regions(&identity, &limits, &graphics(true)).unwrap();
    assert_eq!(plan.regions().len(), 1);
    // The identity is Metal's pass exactly; every other map's pass is the decoder's
    // near-equivalent.
    assert!(rate_map_pass_census(&identity, &plan).precision.is_empty());
    // 0.25 compiles 126 screen pixels to 32 and the last 2 to 1, across, and 32 rows to 16: a
    // pass of tiles more than twice as wide as their physical pixels is drawn by regions too.
    let quarter = compile([128, 32], vec![0.25], vec![0.25]);
    let plan = rate_map_regions(&quarter, &limits, &graphics(false)).unwrap();
    assert_eq!((plan.physical_size(), plan.regions().len()), ([33, 16], 2));
    // The other authored cases' maps, whose outputs are these physical regions: 80x40 at 0.5 (63:32
    // then 17:16 across, 40 rows to 32), and a graded 120x60 map (32:32, 53:32, then 32:32 and 3:3
    // on one line across; 41:32 and 19:19 down).
    let half = compile([80, 40], vec![0.5], vec![0.5]);
    let plan = rate_map_regions(&half, &limits, &graphics(false)).unwrap();
    assert_eq!((plan.physical_size(), plan.regions().len()), ([48, 32], 2));
    let graded = compile([120, 60], vec![1.0, 0.25, 1.0], vec![0.25, 1.0]);
    let plan = rate_map_regions(&graded, &limits, &graphics(false)).unwrap();
    assert_eq!((plan.physical_size(), plan.regions().len()), ([99, 51], 6));
}

/// A render case of one sample draws as before: its template names no sample count (so the pipeline
/// keeps Metal's default of one, and the template crosses the compile worker's exchange in the
/// bytes it always did), and its attachments are uploaded and read back by copies. A multisampled
/// case's template carries its count, and its attachments are filled and read sample by sample.
#[test]
fn a_render_case_of_one_sample_draws_as_it_always_did() {
    let target = |samples: u32| ColorTarget {
        format: TextureFormat::R8Unorm,
        dimensions: [1, 1],
        samples,
        bytes: vec![0; samples as usize],
    };
    let single = two_stage_case("v", "f", vec![], BTreeMap::from([(0, target(1))]), 3);
    assert_eq!(single.raster_sample_count, None);
    assert_eq!(single.attachment_transfer(), AttachmentTransfer::Copy);
    let template = render_template(&single);
    assert_eq!(template.raster_sample_count, None);
    let text = serde_json::to_string(&template).unwrap();
    assert!(!text.contains("raster_sample_count"), "{text}");
    assert_eq!(
        serde_json::from_str::<RenderTemplate>(&text).unwrap(),
        template
    );
    let multisampled = RenderCase {
        raster_sample_count: Some(4),
        color_targets: BTreeMap::from([(0, target(4))]),
        ..single
    };
    assert_eq!(
        multisampled.attachment_transfer(),
        AttachmentTransfer::PerSample { samples: 4 }
    );
    let template = render_template(&multisampled);
    assert_eq!(template.raster_sample_count, Some(4));
    let text = serde_json::to_string(&template).unwrap();
    assert!(text.contains("\"raster_sample_count\":4"), "{text}");
}

/// The runtime path's generated initializer and resolver of each committed public fragment
/// imageblock case (`runtime_case::imageblock::helpers_text`, from the checked reflection's master
/// and the case's cells), each drawn as the oracle draws its own helpers (the case's vertex
/// function, no vertex buffer, the case's attachments, three vertices), compile on lavapipe and
/// moltenvk-m3 to the case pipeline's fragment imageblock: its master's fingerprint, its data size
/// and its alignment. So the runtime draws them in one pass with the case, and their imageblock
/// sample lengths are the case pipeline's, as the oracle requires of its helpers.
#[test]
fn the_imageblock_helpers_view_each_public_cases_master() {
    use crate::candidate::runtime_case::imageblock::{
        helpers_text, INITIALIZE_ENTRY, RESOLVE_ENTRY,
    };
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus");
    let sources = public_sources().unwrap();
    let mut compared = 0;
    for case in CorpusStore::new(&root).read_all_cases().unwrap() {
        if case.fragment_imageblock.is_none() {
            continue;
        }
        let Some(row) = sources
            .iter()
            .find(|row| row.air_sha256 == case.air_sha256 && row.entry == case.entry)
        else {
            continue;
        };
        let scratch = crate::ScratchDir::new("public-imageblock-helpers").unwrap();
        let ll = scratch.path().join("case.ll");
        std::fs::write(&ll, &row.air_ll).unwrap();
        let Some(bitcode) = assembled(&ll) else {
            return;
        };
        let name = &case.name;
        let checked = crate::check::check_case_against_source(&root, case.clone(), row)
            .unwrap_or_else(|errors| panic!("{name}: {errors:?}"));
        let master = checked.primary.reflection.fragment_imageblock.as_ref();
        let render = render_case(
            &case,
            &checked.primary.interface,
            &row.air_ll,
            checked.primary.companion.as_ref(),
            master,
        )
        .unwrap_or_else(|error| panic!("{name}: {error:?}"));
        let cells = render.imageblock.as_ref().expect("the case's cells");
        let text = helpers_text(master.expect("the reflected master"), cells)
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let helpers = crate::check::loop_free::assemble(&text)
            .unwrap_or_else(|error| panic!("{name}: {error}\n{text}"));
        let companion = render.companion.as_ref().unwrap().opaque_bitcode().unwrap();
        let imageblock = |pipeline: &metal2vulkan::pipeline::RenderPipeline| {
            pipeline
                .graphics
                .fragment_imageblock
                .map(|state| (state.master, state.data_size, state.alignment))
        };
        for stem in ["lavapipe", "moltenvk-m3"] {
            let device = device(stem);
            let libraries = RenderLibraries {
                vertex: &companion,
                fragment: Some(&bitcode),
            };
            let shape = render_shape(&render, true).unwrap();
            let own = compile_in_process(&device, &render, libraries, &shape)
                .unwrap_or_else(|error| panic!("{name} on {stem}: {error}"));
            let expected = imageblock(&own);
            assert!(expected.is_some(), "{name} on {stem}: no imageblock state");
            for entry in [INITIALIZE_ENTRY, RESOLVE_ENTRY] {
                let mut helper = two_stage_case(
                    &render.vertex.entry,
                    entry,
                    Vec::new(),
                    render.color_targets.clone(),
                    3,
                );
                helper.depth_stencil = render.depth_stencil.clone();
                let libraries = RenderLibraries {
                    vertex: &companion,
                    fragment: Some(&helpers),
                };
                let shape = render_shape(&helper, true).unwrap();
                let pipeline = compile_in_process(&device, &helper, libraries, &shape)
                    .unwrap_or_else(|error| panic!("{name} {entry} on {stem}: {error}"));
                assert_eq!(imageblock(&pipeline), expected, "{name} {entry} on {stem}");
            }
            compared += 1;
        }
    }
    // As `public_fragment_imageblock_cases_plan_their_storage_image` plans them: twenty cases,
    // each on both profiles.
    assert_eq!(compared, 40);
}

/// A vertex buffer of a dynamic layout: its layout's stride is `MTLBufferLayoutStrideDynamic`, so
/// the fetch reads the stride the buffer is bound with; a fixed layout's is its stride.
#[test]
fn a_dynamic_vertex_layout_takes_the_bound_stride() {
    let buffer = |location, buffer_index, dynamic| VertexBuffer {
        location,
        buffer_index,
        format: AttributeFormat::Uint,
        stride: 8,
        dynamic,
        step: metal::VertexStepFunction::PerVertex,
        bytes: vec![0; 32],
    };
    let descriptor = vertex_descriptor(&[buffer(0, 30, true), buffer(1, 29, false)]).unwrap();
    assert_eq!(descriptor.layouts[&30].stride, None);
    assert_eq!(descriptor.layouts[&29].stride, Some(8));
}
