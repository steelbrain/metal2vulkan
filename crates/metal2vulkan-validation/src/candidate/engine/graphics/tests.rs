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
    Device::from_queries(description, DevicePolicy::RESIDENT_TABLES).unwrap()
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

/// A fragment case's authored aspect is its depth/stencil attachment (the oracle's `Stencil8` here,
/// as its `Depth32Float` for a depth), its depth-stencil state is the one its outputs imply, and
/// each aspect is an observable output; an attachment of both aspects, the oracle's two textures,
/// is the runtime's limit (`SeparateDepthStencil`); a vertex case's attachment, which the oracle
/// does not attach, is refused.
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
    assert_eq!(
        requirements(render_case(
            &authored,
            &interface,
            WITHOUT_SLICE,
            companion.as_ref()
        )),
        [crate::requirement::ToolingRequirement::SeparateDepthStencil].into()
    );
    authored.depth_stencil.as_mut().unwrap().initial_depth_b64 = None;
    let render = render_case(&authored, &interface, WITHOUT_SLICE, companion.as_ref()).unwrap();
    assert!(render.color_targets.is_empty());
    assert_eq!(
        render.depth_stencil,
        Some(LiteralDepthStencil {
            dimensions: [1, 1],
            samples: 1,
            depth: None,
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
        (None, Some(metal::PixelFormat::Stencil8))
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
        requirements(render_case(&vertex, &interface, WITHOUT_SLICE, None))
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
            companion.as_ref()
        )),
        [crate::requirement::ToolingRequirement::RenderAccelerationStructure].into()
    );
}

/// A fragment case's `[[stencil]]` output plans through stencil export: on lavapipe, which has
/// the extension, the pipeline requires it and the bind state carries the draw's descriptor, the
/// one the Metal oracle sets. On nvidia-rtx50-linux, which lacks it, the compile refuses naming
/// the extension.
#[test]
fn a_stencil_output_case_plans_through_stencil_export() {
    use metal2vulkan::device::VkExtension;
    use metal2vulkan::error::{Capability, Feature, Needs};
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
        initial_depth_b64: None,
        initial_stencil_b64: Some("qw==".into()),
        state: None,
    });
    authored.output = OutputSelection::Stencil {
        origin: [0, 0],
        dimensions: [1, 1],
    };
    let (interface, companion) = checked_companion(&authored, &bitcode);
    let mut render = render_case(&authored, &interface, WITHOUT_SLICE, companion.as_ref()).unwrap();
    // Both aspects are the runtime's limit (`SeparateDepthStencil`), which stops such a case
    // before the engine plans it; the plan below is the translator's for both, so the case maps
    // with its stencil and takes its depth after.
    render.depth_stencil.as_mut().unwrap().depth = Some(1.0f32.to_le_bytes().to_vec());
    let companion_bitcode = render.companion.as_ref().unwrap().opaque_bitcode().unwrap();
    let libraries = RenderLibraries {
        vertex: &companion_bitcode,
        fragment: Some(&bitcode),
    };
    let (pipeline, _, _) = planned_draw(&device("lavapipe"), &render, libraries).unwrap();
    assert!(pipeline
        .requirements
        .extensions
        .contains(&VkExtension::ExtShaderStencilExport));
    let state = bind_state(
        &render,
        &BTreeMap::new(),
        &BTreeMap::new(),
        BufferRanges::default(),
    )
    .unwrap();
    assert_eq!(state.depth_stencil, render.depth_stencil_state.descriptor);

    match planned_draw(&device("nvidia-rtx50-linux"), &render, libraries) {
        Err(Error::Unsupported(u)) => assert_eq!(
            (u.feature, u.needs),
            (
                Feature::StencilExport,
                Needs::AnyOf(vec![vec![Capability::Extension(
                    VkExtension::ExtShaderStencilExport
                )]])
            )
        ),
        other => panic!("a stencil output without export: {:?}", other.map(|_| ())),
    }
}

/// A fragment case that reads its color attachment maps like any fragment case (its render target
/// is the attachment it reads; no requirement stops the engine's candidate) and plans on
/// lavapipe, whose `rasterizationOrderColorAttachmentAccess` orders the read: the plan reads color
/// attachment 0 through an input-attachment write, and the pipeline states rasterization-order
/// access and requires the feature. On moltenvk-m3, which lacks the feature, it plans the input
/// attachment with no rasterization-order access, which the consumer orders between draws alone
/// (its barriers, `platform::graphics_pipeline::fetch_between_draws`), the fragment stage noting
/// `PrecisionNote::UnorderedFetchWithinDraw`, and requires no ordered form.
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
    let render = render_case(&authored, &interface, WITHOUT_SLICE, companion.as_ref()).unwrap();
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
    // Moltenvk-m3: the input attachment ordered between draws alone, noted.
    let (pipeline, plan, _) = planned_draw(&device("moltenvk-m3"), &render, libraries).unwrap();
    let read: Vec<u32> = (pipeline.graphics.input_attachments.iter())
        .map(|input| input.color_index)
        .collect();
    assert_eq!(read, [0]);
    assert!(!pipeline.graphics.rasterization_order_attachment_access);
    assert!(pipeline.graphics.color_attachments[0].is_some());
    assert!(!pipeline
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
    let render = render_case(&authored, &interface, WITHOUT_SLICE, companion.as_ref()).unwrap();
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
    let render = render_case(&authored, &interface, WITHOUT_SLICE, companion.as_ref()).unwrap();
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
    let render = render_case(&observed, &interface, WITHOUT_SLICE, observer.as_ref()).unwrap();
    assert_eq!(render.vertex.entry, "vertex_varying_triangle");
    assert_eq!(
        render.fragment.as_ref().map(|f| f.entry.as_str()),
        Some(crate::companion::OBSERVER_FRAGMENT_ENTRY)
    );
    assert_eq!(render.color_targets[&0].format, TextureFormat::Rgba32Float);
    assert_eq!(render.companion, observer);

    // A companion that does not complete the case's stage is the executor's failure, not a gap.
    for (authored, companion) in [(&observed, None), (&authored, observer.as_ref())] {
        match render_case(authored, &interface, WITHOUT_SLICE, companion) {
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
    let render = render_case(&case, &vertex_buffers_at(&[0]), WITHOUT_SLICE, None).unwrap();
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
    let state = bind_state(
        &render,
        &addresses,
        &BTreeMap::new(),
        BufferRanges::default(),
    )
    .unwrap();
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
    match crate::acceptance::compile_pipeline(&bitcode, &request, device, None) {
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
        output: RenderOutput::RenderTarget {
            index: 0,
            origin: [0, 0],
            dimensions: [1, 1],
        },
        companion: None,
        raster_sample_count: None,
    }
}

/// Compile and plan `case` on `device`: no variant, no diagnostic, and one direct draw; or the
/// product's compile refusal.
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
    let shape = render_shape(case, true, BufferRanges::default()).unwrap();
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
        .flat_map(|(stage, f)| f.resources.buffers.keys().map(move |&index| (stage, index)))
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
    let state = bind_state(case, &addresses, &embedded, BufferRanges::default()).unwrap();
    assert_eq!(state.shape(), shape);
    let plan = pipeline.plan(device, &shape).unwrap();
    assert!(plan.variant.is_empty(), "{}", case.vertex.entry);
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
    let render = render_case(&authored, &interface, WITHOUT_SLICE, observer.as_ref()).unwrap();
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
    let shape = render_shape(&render, true, BufferRanges::default()).unwrap();
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
            step: metal::VertexStepFunction::PerVertex,
            bytes: vec![0; 24],
        },
        VertexBuffer {
            location: 1,
            buffer_index: 29,
            format: AttributeFormat::Float2,
            stride: 8,
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
    let root = metal2vulkan_devtools::validation_dir().join("corpus");
    let store = CorpusStore::new(&root);
    let sources = public_sources().unwrap();
    let (case, row) = (0..64)
        .flat_map(|shard| store.read_case_shard(shard).unwrap())
        .filter(|case| {
            matches!(case.stage, Stage::Vertex | Stage::Fragment)
                && !case.argument_buffer_buffers.is_empty()
                && crate::executor_contract::case_requirements(case).is_empty()
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
    let render = render_case(&case, interface, &row.air_ll, companion).unwrap();
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
    let state = bind_state(&render, &addresses, &embedded, BufferRanges::default()).unwrap();
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
        let shape = render_shape(&render, true, BufferRanges::default()).unwrap();
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
    let render = render_case(&arrays, interface, &row.air_ll, companion).unwrap();
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
    // follows the buffers' and whose resource id its owner holds at its field; the texture's region
    // is an output. An argument-buffer sampler is an argument sampler of the bind state whose id
    // its owner holds at its field.
    use crate::candidate::engine::argument_buffers::{field_words, resource_id, sampler_id};
    let mut members = case.clone();
    let texture_field = key.1 + 64;
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
    members.output = OutputSelection::ArgumentBufferTexture {
        buffer_binding: key.0,
        field_offset: texture_field,
        origin: [1, 0, 0],
        dimensions: [1, 1, 1],
    };
    let render = render_case(&members, interface, &row.air_ll, companion).unwrap();
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
    let member = (ShaderStage::Fragment, key.0, texture_field);
    let tokens = texture_tokens(&render).unwrap();
    assert_eq!(tokens, BTreeMap::from([(member, 1)]));
    let embedded = BTreeMap::from([((ShaderStage::Fragment, key.0, key.1), pointee)]);
    let state = bind_state(&render, &addresses, &embedded, BufferRanges::default()).unwrap();
    let [_, used] = state.used_resources.as_slice() else {
        panic!("{:?}", state.used_resources);
    };
    assert!(
        matches!(used.class, metal2vulkan::bind::ResourceClass::Texture(_)),
        "{used:?}"
    );
    assert_eq!(
        (used.token, used.resource_id, used.stages),
        (
            1,
            Some(resource_id(1)),
            RenderStages::VERTEX | RenderStages::FRAGMENT
        )
    );
    assert_eq!(
        render_shape(&render, true, BufferRanges::default()).unwrap(),
        state.shape()
    );
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
    let words = field_words(
        &embedded,
        &BTreeMap::new(),
        &tokens,
        &samplers,
        std::iter::empty(),
    )
    .unwrap();
    assert_eq!(
        words,
        BTreeMap::from([
            ((ShaderStage::Fragment, key.0, key.1), pointee.guest),
            (member, resource_id(1)),
            (sampler, sampler_id(0).unwrap()),
        ])
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
/// pipeline compiled for that shape writes `Layer` (0 for every index in that pass).
#[test]
fn a_slice_choosing_case_binds_the_oracles_one_layer_pass() {
    let root = metal2vulkan_devtools::validation_dir().join("corpus");
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
    let shape = render_shape(&render, true, BufferRanges::default()).unwrap();
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
    assert!(pipeline.graphics.declares_layer);
    assert!(pipeline
        .dependencies
        .iter()
        .any(|dependency| dependency.fact
            == metal2vulkan::pipeline::RuntimeFact::RenderTargetArrayLength {
                layer_dropped: false
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
    let root = metal2vulkan_devtools::validation_dir().join("corpus");
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
    let root = metal2vulkan_devtools::validation_dir().join("corpus");
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

/// The committed profiles whose descriptions lack `rasterizationOrderColorAttachmentAccess`: the
/// devices that order a fetch between draws alone, read from what each states rather than from its
/// name.
fn unordered_profiles() -> Vec<String> {
    use metal2vulkan::device::VkFeature;
    let stems = crate::engine::committed_profiles().unwrap();
    let stems = stems
        .into_iter()
        .filter(|stem| {
            let description = crate::engine::load_description(stem).unwrap();
            !description.has(VkFeature::RasterizationOrderColorAttachmentAccess)
        })
        .collect::<Vec<_>>();
    assert!(
        !stems.is_empty(),
        "a committed profile orders fetch between draws alone"
    );
    stems
}

/// Every installed public framebuffer-fetch case maps onto its fullscreen triangle drawn as its
/// instances, with each read render target attached, and plans on lavapipe (one input-attachment
/// write per color input, rasterization-order access, the feature required). Every case carries
/// no requirement, at least one of them drawing more than one instance (overlapping fragments).
/// The lavapipe pipeline masks every write to a color attachment the function reads but has no
/// output for, which at least one case does. A case of a multisampled pass, which at least one
/// is, reads its colors as input attachments at each invocation's own sample (sample-rate
/// shading, the pipeline's sample count), at least one so; or it is refused, typed: the pixel's
/// coverage read beside it (`[[sample_mask]]` in, on no device), or a sample count the device's
/// attachments lack.
#[test]
fn public_framebuffer_fetch_cases_plan_their_ordered_input_attachments() {
    use std::collections::BTreeSet;

    use crate::air::reflect::ResourceKind;
    use metal2vulkan::bind::WriteSource;
    use metal2vulkan::device::VkFeature;
    let root = metal2vulkan_devtools::validation_dir().join("corpus");
    let store = CorpusStore::new(&root);
    let sources = public_sources().unwrap();
    let mut planned = 0;
    let mut overlapping = 0;
    let mut unwritten = 0;
    let mut multisampled = 0;
    let mut per_sample = 0;
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
            // Each invocation reads its own sample.
            if let Some(samples) = case.raster_sample_count {
                let mut expected = colors.clone();
                expected.sort_unstable();
                match planned_draw(&device("lavapipe"), &render, libraries) {
                    Ok((pipeline, _, _)) => {
                        let g = &pipeline.graphics;
                        assert_eq!(g.rasterization_samples.as_raw(), samples, "{}", case.name);
                        let required = &pipeline.requirements;
                        assert!(required.features.contains(&VkFeature::SampleRateShading));
                        assert!(g.rasterization_order_attachment_access);
                        let mut fetched: Vec<u32> =
                            g.input_attachments.iter().map(|a| a.color_index).collect();
                        fetched.sort_unstable();
                        assert_eq!(fetched, expected, "{}", case.name);
                        per_sample += 1;
                    }
                    Err(Error::Unsupported(refusal)) => {
                        use metal2vulkan::error::{Feature, Needs};
                        match (&refusal.feature, &refusal.needs) {
                            (Feature::PixelCoverageAtSampleRate, Needs::NotRepresentable)
                            | (Feature::RasterSampleCount, Needs::AnyOf(_)) => {}
                            _ => panic!("lavapipe {}: {refusal:?}", case.name),
                        }
                    }
                    Err(other) => panic!("lavapipe {}: {other:?}", case.name),
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
            planned += 1;
        }
    }
    assert!(planned > 0, "no public framebuffer-fetch case planned");
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
    assert!(
        per_sample > 0,
        "no multisampled public framebuffer-fetch case plans"
    );
}

/// On every committed profile without rasterization-order color attachment access
/// ([`unordered_profiles`]), every installed public framebuffer-fetch case plans its color inputs
/// as input attachments with no
/// rasterization-order access, which the consumer orders between draws alone, and its fragment
/// stage notes `PrecisionNote::UnorderedFetchWithinDraw`; or, a multisampled pass, is refused as on
/// every device (the coverage read beside it, or a sample count the attachments lack), or for the
/// sample-rate shading each invocation's read of its own sample needs, where the device lacks it.
#[test]
fn public_framebuffer_fetch_cases_plan_input_attachments_without_an_ordered_form() {
    use crate::air::reflect::ResourceKind;
    let root = metal2vulkan_devtools::validation_dir().join("corpus");
    let store = CorpusStore::new(&root);
    let sources = public_sources().unwrap();
    for stem in unordered_profiles() {
        let stem = stem.as_str();
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
                    // A multisampled pass's refusals of every device's form (the lavapipe
                    // test's), and its fetch on a device without sample-rate shading, which each
                    // invocation's read of its own sample needs.
                    Err(Error::Unsupported(refusal)) if case.raster_sample_count.is_some() => {
                        use metal2vulkan::device::VkFeature;
                        use metal2vulkan::error::{Capability, Feature, Needs};
                        let shading = [vec![Capability::Feature(VkFeature::SampleRateShading)]];
                        match (&refusal.feature, &refusal.needs) {
                            (Feature::PixelCoverageAtSampleRate, Needs::NotRepresentable)
                            | (Feature::RasterSampleCount, Needs::AnyOf(_)) => continue,
                            (Feature::FramebufferFetch, Needs::AnyOf(needs))
                                if needs[..] == shading
                                    && !crate::engine::load_description(stem)
                                        .unwrap()
                                        .has(VkFeature::SampleRateShading) =>
                            {
                                continue
                            }
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
