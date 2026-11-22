//! Device-free tests of the engine render candidate: case mapping, companion refusals, the
//! oracle's vertex-buffer slots, output selection, and the owned and public graphics pipelines
//! compiled for committed profiles. No device is opened.

use super::*;
use crate::case::{AttributeFormat, VertexObservation};
use crate::check::interface::tests::vertex_buffers_at;
use crate::executor_contract::depth_stencil_test;
use crate::source::public_sources;
use crate::store::CorpusStore;
use metal2vulkan::device::{Device, DeviceDescription, DevicePolicy};
use metal2vulkan::library::FunctionConstantValues;
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

/// The request the tools send the compile worker for `case`'s pipeline, and its libraries in
/// request order: a vertex function alone as `PipelineRequest::VertexDraw`, a vertex and a
/// fragment function of two libraries as `PipelineRequest::Draw`.
pub(crate) fn draw_request<'a>(
    case: &RenderCase,
    libraries: RenderLibraries<'a>,
) -> (crate::acceptance::PipelineRequest, Vec<&'a [u8]>) {
    let function = |f: &RenderFunction| {
        crate::acceptance::RequestFunction::new(f.entry.as_str(), f.constants.clone())
    };
    match (&case.fragment, libraries.fragment) {
        (None, None) => (
            crate::acceptance::PipelineRequest::VertexDraw {
                function: function(&case.vertex),
                template: render_template(case),
            },
            vec![libraries.vertex],
        ),
        (Some(fragment), Some(bitcode)) => (
            crate::acceptance::PipelineRequest::Draw {
                vertex: function(&case.vertex),
                fragment: function(fragment),
                template: render_template(case),
            },
            vec![libraries.vertex, bitcode],
        ),
        _ => panic!("a fragment function and its library come together"),
    }
}

/// `case`'s pipeline compiled in this process as the runtime compiles it, by [`draw_request`]: the
/// device-free plan tests compile here; the device tests compile the same request in the compile
/// worker.
pub(crate) fn compile_in_process(
    device: &Device,
    case: &RenderCase,
    libraries: RenderLibraries<'_>,
) -> Result<metal2vulkan::pipeline::RenderPipeline, Error> {
    let (request, bitcode) = draw_request(case, libraries);
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

/// A tessellation case maps onto the Metal oracle's patch draw: its tessellation state and factor
/// bytes (`executor_contract::tessellation`), its control points per patch control point and its
/// patch attribute per patch at the oracle's slots, and one patch draw of the function's control
/// points over every patch and instance through the observer.
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
}

/// A render case's argument-buffer buffers are its own stage's: its owner covers the pointer field
/// and its final range is an output. The public raster-ordered argument-buffer fragment case maps
/// so. A texture array's elements are the stage's textures at their slots, and an element output is
/// that texture's region. An argument-buffer texture and sampler are the stage's members at their
/// fields, and the texture's region is an output.
#[test]
fn render_cases_map_argument_buffer_buffers_and_texture_arrays() {
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

    // An argument-buffer texture and sampler of the fragment stage are its members at their fields;
    // the texture's region is an output.
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
/// its render case carries the length the shared derivation gives, and its pipeline writes `Layer`
/// (0 for every index in that pass).
#[test]
fn a_slice_choosing_case_maps_the_oracles_one_layer_pass() {
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

/// A render case of one sample draws as before: its template names no sample count (so the pipeline
/// keeps Metal's default of one, and the template crosses the compile worker's exchange in the
/// bytes it always did). A multisampled case's template carries its count.
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
    let template = render_template(&multisampled);
    assert_eq!(template.raster_sample_count, Some(4));
    let text = serde_json::to_string(&template).unwrap();
    assert!(text.contains("\"raster_sample_count\":4"), "{text}");
}
