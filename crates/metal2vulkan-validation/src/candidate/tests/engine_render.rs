//! The engine render candidate on a device: owned vertex and fragment fixtures and authored
//! graphics cases with their generated companions, drawn through the runtime under
//! `VK_LAYER_KHRONOS_validation`, against expectations derived from the MSL and Metal's conventions
//! (y-up clip space, pixel-center coverage). `G-linux`'s `m2v_cpu_icd` step runs these on lavapipe.
//! The looping fragment runs as compiled on both lanes.

use super::*;
use crate::candidate::engine::graphics::tests::{
    checked_companion, two_stage_case, RenderLibraries, WITHOUT_SLICE,
};
use crate::candidate::engine::graphics::{
    render_case, ColorTarget, RenderCase, RenderOutput, VertexBuffer,
};
use crate::candidate::engine::ComputeRun;
use crate::case::{
    AttributeFormat, AttributeInput, AuthoredCase, Draw, OutputSelection, Primitive, Stage,
    TextureFormat, VertexObservation,
};
use crate::check::admission::ExecutorError;
use base64::Engine as _;
use metal2vulkan_devtools::{fixtures_dir, llvm_assemble};
use std::collections::BTreeMap;

/// Draw `case` of the stages of `libraries` through the runtime ([`run_on_runtime`]): the case's
/// own function's library, and the other stage's, where the case has one.
fn run(case: &RenderCase, libraries: RenderLibraries<'_>) -> Result<ComputeRun, ExecutorError> {
    match (case.companion_stage(), libraries.fragment) {
        (Some(metal2vulkan::pipeline::Stage::Vertex), Some(fragment)) => {
            run_on_runtime(case, fragment, Some(libraries.vertex))
        }
        (_, fragment) => run_on_runtime(case, libraries.vertex, fragment),
    }
}

/// `case` drawn through the runtime (`platform::execute_runtime_render`): the case's own function's
/// AIR and its companion's as the engine hands them over, its own stage's buffers as the case
/// holds them, each texture, bound or in an argument buffer, made with its role's usage, and no
/// float reduction.
#[cfg(target_os = "linux")]
fn run_on_runtime(
    case: &RenderCase,
    own: &[u8],
    companion: Option<&[u8]>,
) -> Result<ComputeRun, ExecutorError> {
    let function = match (case.companion_stage(), &case.fragment) {
        (Some(metal2vulkan::pipeline::Stage::Vertex), Some(fragment)) => fragment,
        _ => &case.vertex,
    };
    platform::execute_runtime_render(
        crate::gpu::TEST_DEVICE,
        "engine-render-test",
        case,
        crate::candidate::engine::RuntimeDraw {
            bitcode: own,
            companion,
            linked: &[],
            buffers: &function.resources.buffers,
            texture_usages: &function
                .resources
                .textures
                .iter()
                .map(|(&slot, texture)| {
                    let usage = crate::check::interface::texture_shader_usage(texture.role, None);
                    (slot, usage)
                })
                .collect(),
            member_usages: &function
                .resources
                .argument_buffer_textures
                .iter()
                .map(|(&key, texture)| {
                    let usage = crate::check::interface::texture_shader_usage(texture.role, None);
                    (key, usage)
                })
                .collect(),
            reductions: &std::collections::BTreeSet::new(),
        },
    )
    .result
}

fn public(name: &str) -> Vec<u8> {
    llvm_assemble(&fixtures_dir().join(name)).expect("llvm-as on the device lane")
}

fn floats(values: &[f32]) -> Vec<u8> {
    values.iter().copied().flat_map(f32::to_le_bytes).collect()
}

/// `vertex_varying_triangle`'s two `float2` attributes, one buffer each at the oracle's slots.
fn triangle_inputs(positions: &[[f32; 2]], uv: [f32; 2]) -> Vec<VertexBuffer> {
    let count = positions.len();
    vec![
        VertexBuffer {
            location: 0,
            buffer_index: 30,
            format: AttributeFormat::Float2,
            stride: 8,
            step: metal2vulkan::metal::VertexStepFunction::PerVertex,
            bytes: floats(&positions.concat()),
        },
        VertexBuffer {
            location: 1,
            buffer_index: 29,
            format: AttributeFormat::Float2,
            stride: 8,
            step: metal2vulkan::metal::VertexStepFunction::PerVertex,
            bytes: floats(&uv.repeat(count)),
        },
    ]
}

/// A 4×4 `RGBA32Float` target cleared to -1, observed whole.
fn quad_target() -> BTreeMap<u32, ColorTarget> {
    BTreeMap::from([(
        0,
        ColorTarget {
            format: TextureFormat::Rgba32Float,
            dimensions: [4, 4],
            samples: 1,
            bytes: floats(&[-1.0; 64]),
        },
    )])
}

/// The 4×4 target after covering clip-space `[0, 1]²` with `color`: Metal's y-up clip space puts
/// that square in the top-right 2×2 pixels (rows 0–1, columns 2–3); its edges lie on pixel
/// boundaries, so coverage does not depend on fill rules.
fn top_right(color: [f32; 4]) -> Vec<u8> {
    let mut pixels = vec![[-1.0f32; 4]; 16];
    for (y, x) in [(0, 2), (0, 3), (1, 2), (1, 3)] {
        pixels[y * 4 + x] = color;
    }
    floats(&pixels.concat())
}

const QUADRANT: [[f32; 2]; 6] = [
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
];

fn whole_target(case: &mut RenderCase) {
    case.output = RenderOutput::RenderTarget {
        index: 0,
        origin: [0, 0],
        dimensions: [4, 4],
    };
}

/// A vertex function and a separately compiled fragment function drawn through one pipeline:
/// the negative-height viewport puts Metal's y-up quadrant in the top rows.
#[cfg(target_os = "linux")]
#[test]
fn a_constant_fragment_colors_the_quadrant_metal_draws() {
    let mut case = two_stage_case(
        "vertex_varying_triangle",
        "fragment_constant_color",
        triangle_inputs(&QUADRANT, [0.0, 0.0]),
        quad_target(),
        6,
    );
    whole_target(&mut case);
    let vertex = public("vertex_varying_triangle.ll");
    let fragment = public("fragment_constant_color.ll");
    let run = run(
        &case,
        RenderLibraries {
            vertex: &vertex,
            fragment: Some(&fragment),
        },
    )
    .unwrap();
    assert_eq!(run.output, top_right([0.25, 0.5, 0.75, 1.0]));
    assert_eq!(run.environment["executor"], "runtime-render");
}

/// `rt6_layered_vertex` (engine `rt6_layered`) writes index 3, from its buffer, in a pass of
/// Metal's default length 0, a pass of one layer: Metal renders the triangle into the pass's slice
/// whatever index the function writes, and `rt6_layered_fragment` reads the index as written, so
/// the slice holds instance 0's word `1 << 8 | 3` (the conformance case
/// `layered-render-target-array-length`, recorded on an Apple M3 Max under macOS 27.0.1). The
/// pipeline stores `Layer` as 0 at or past the length it is pushed (1), a value the Vulkan spec
/// defines in a framebuffer of one layer. The validation layer still reports the draw's `Layer`
/// write as undefined; the sink withholds exactly that message for this row, and the row's
/// environment counts it (`platform::device`'s `CLAMPED_LAYER`). When the layer stops reporting
/// it, this test fails, and the exception goes with it.
#[cfg(target_os = "linux")]
#[test]
fn an_index_past_a_one_layer_pass_renders_the_pass_s_slice() {
    let library = llvm_assemble(&metal2vulkan_devtools::engine_dir().join("rt6_layered.ll"))
        .expect("llvm-as on the device lane");
    let target = BTreeMap::from([(
        0,
        ColorTarget {
            format: TextureFormat::R32Uint,
            dimensions: [1, 1],
            samples: 1,
            bytes: 7u32.to_le_bytes().to_vec(),
        },
    )]);
    let mut case = two_stage_case(
        "rt6_layered_vertex",
        "rt6_layered_fragment",
        Vec::new(),
        target,
        3,
    );
    assert_eq!(case.render_target_array_length, 0);
    (case.vertex.resources.buffers).insert(0, 3u32.to_le_bytes().to_vec());
    let run = run(
        &case,
        RenderLibraries {
            vertex: &library,
            fragment: Some(&library),
        },
    )
    .unwrap();
    assert_eq!(run.output, (1u32 << 8 | 3).to_le_bytes());
    assert_eq!(
        run.environment["withheld_validation_messages"],
        serde_json::json!({ "Undefined-Value-Layer-Written": 1 })
    );
}

/// A varying written by the vertex stage reaches the fragment stage of another library by its
/// user key; a value equal at every vertex interpolates to itself.
#[cfg(target_os = "linux")]
#[test]
fn a_vertex_varying_reaches_the_fragment_stage() {
    let mut case = two_stage_case(
        "vertex_varying_triangle",
        "fragment_varying_color",
        triangle_inputs(&QUADRANT, [0.25, 0.75]),
        quad_target(),
        6,
    );
    whole_target(&mut case);
    let vertex = public("vertex_varying_triangle.ll");
    let fragment = public("fragment_varying_color.ll");
    let run = run(
        &case,
        RenderLibraries {
            vertex: &vertex,
            fragment: Some(&fragment),
        },
    )
    .unwrap();
    assert_eq!(run.output, top_right([0.25, 0.75, 0.5, 1.0]));
}

/// `case` over its function's `bitcode` mapped with the checker's generated companion, and the
/// libraries its two stages come from: the companion's opaque spelling assembled, in stage order.
fn with_companion(case: &AuthoredCase, bitcode: &[u8]) -> (RenderCase, Vec<u8>) {
    let (interface, companion) = checked_companion(case, bitcode);
    let render = render_case(case, &interface, WITHOUT_SLICE, companion.as_ref()).unwrap();
    let companion = companion.expect("a graphics case with a companion");
    (render, companion.opaque_bitcode().unwrap())
}

fn b64(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// A 4×4 `RGBA32Float` render target 0 of `case` cleared to -1, observed whole.
fn authored_quad_target(case: &mut AuthoredCase) {
    case.render_targets[0].dimensions = [4, 4];
    case.render_targets[0].initial_bytes_b64 = b64(&floats(&[-1.0; 64]));
    case.output = OutputSelection::RenderTarget {
        index: 0,
        origin: [0, 0],
        dimensions: [4, 4],
    };
}

/// A fragment case drawn through its generated passthrough vertex companion: the fullscreen
/// triangle's `uv` varying interpolates to `((x + 1) / 2, (y + 1) / 2)` at each pixel center of
/// Metal's y-up clip space, so row 0 (the top) reads `v = 0.875`, and the case's function writes
/// `(u, v, 0.5, 1)` there.
#[cfg(target_os = "linux")]
#[test]
fn a_fragment_case_draws_through_its_generated_passthrough_vertex() {
    let fragment = public("fragment_varying_color.ll");
    let mut authored = crate::companion::tests::case(
        Stage::Fragment,
        "fragment_varying_color",
        TextureFormat::Rgba32Float,
    );
    authored_quad_target(&mut authored);
    let (case, companion) = with_companion(&authored, &fragment);
    let run = run(
        &case,
        RenderLibraries {
            vertex: &companion,
            fragment: Some(&fragment),
        },
    )
    .unwrap();
    let centers = [0.125f32, 0.375, 0.625, 0.875];
    let expected: Vec<f32> = centers
        .iter()
        .rev()
        .flat_map(|&v| centers.iter().flat_map(move |&u| [u, v, 0.5, 1.0]))
        .collect();
    assert_eq!(run.output, floats(&expected));
}

/// A fragment function samples the texture an argument buffer holds through the sampler beside it
/// (`tail_argument_samplers`' `argument_sampler_fragment`, `m.t.sample(m.s, in.uv)`): at each of
/// the 2x2 target's pixel centers the passthrough companion's `uv` is `(0.25 or 0.75, 0.75)` on
/// row 0 and `(0.25 or 0.75, 0.25)` on row 1, so the nearest sampler reads the texture's row 1 into
/// the target's row 0 and its row 0 into row 1, through the runtime, whose owner holds the member texture's and sampler's `gpuResourceID`s.
#[cfg(target_os = "linux")]
#[test]
fn a_fragment_samples_the_texture_an_argument_buffer_holds_through_its_sampler() {
    use crate::case::{
        ArgumentBufferSamplerResource, ArgumentBufferTextureResource, BufferResource, ResourceRole,
        SamplerAddressMode, SamplerFilter, SamplerMipFilter, TextureType,
    };
    let fragment = public("engine/tail_argument_samplers.ll");
    let mut authored = crate::companion::tests::case(
        Stage::Fragment,
        "argument_sampler_fragment",
        TextureFormat::Rgba32Float,
    );
    authored.render_targets[0].dimensions = [2, 2];
    authored.render_targets[0].initial_bytes_b64 = b64(&floats(&[-1.0; 16]));
    authored.output = OutputSelection::RenderTarget {
        index: 0,
        origin: [0, 0],
        dimensions: [2, 2],
    };
    authored.buffers = vec![BufferResource {
        binding: 0,
        role: ResourceRole::Input,
        bytes_b64: Some(b64(&[0; 16])),
        initial_bytes_b64: None,
        rasterization_rate_map: None,
    }];
    let texels = [
        [1.0f32, 0.0, 0.0, 1.0],
        [2.0, 10.0, 0.0, 1.0],
        [3.0, 0.0, 100.0, 1.0],
        [4.0, 10.0, 100.0, 1.0],
    ];
    authored.argument_buffer_textures = vec![ArgumentBufferTextureResource {
        buffer_binding: 0,
        field_offset: 0,
        role: ResourceRole::Input,
        texture_type: TextureType::D2,
        format: TextureFormat::Rgba32Float,
        dimensions: [2, 2, 1],
        sample_count: 1,
        bytes_b64: Some(b64(&floats(&texels.concat()))),
        initial_bytes_b64: None,
    }];
    authored.argument_buffer_samplers = vec![ArgumentBufferSamplerResource {
        buffer_binding: 0,
        field_offset: 8,
        address_mode: SamplerAddressMode::ClampToEdge,
        min_filter: SamplerFilter::Nearest,
        mag_filter: SamplerFilter::Nearest,
        mip_filter: SamplerMipFilter::NotMipmapped,
        normalized_coordinates: true,
        compare_function: None,
    }];
    let (case, companion) = with_companion(&authored, &fragment);
    let expected = floats(&[texels[2], texels[3], texels[0], texels[1]].concat());
    let run = run(
        &case,
        RenderLibraries {
            vertex: &companion,
            fragment: Some(&fragment),
        },
    )
    .unwrap();
    assert_eq!(run.output, expected);
}

/// A fragment case with a depth/stencil attachment of one aspect draws it as the executor's own
/// context does, through the runtime, which gives it no texture index (shaders neither read nor
/// write it): depth under `sample_rate_depth` (0.125 at one sample, over the authored 1.0), stencil
/// under `fragment_no_writes` (the oracle's always-pass stencil test replaces the authored 0xab with
/// the state's reference). An attachment of both aspects is the oracle's two textures, which the
/// runtime renders into in no one pass, so `render_case` refuses it as the runtime's limit
/// (`SeparateDepthStencil`). A fragment function's stencil output, `fragment_depth_stencil`'s 7,
/// is the fragment's stencil reference on a device that lists `VK_EXT_shader_stencil_export`
/// (lavapipe lists it), where the runtime stores 7; on
/// one that does not, the runtime refuses the state at creation naming `StencilExport`.
#[cfg(target_os = "linux")]
#[test]
fn a_fragment_case_writes_its_depth_and_stencil_attachment() {
    let fragment = public("fragment_depth_stencil.ll");
    let mut authored = crate::companion::tests::case(
        Stage::Fragment,
        "fragment_depth_stencil",
        TextureFormat::Rgba32Float,
    );
    authored.render_targets.clear();
    let depth_stencil = |depth: bool, stencil: bool| {
        Some(crate::case::DepthStencilResource {
            dimensions: [1, 1],
            initial_depth_b64: depth.then(|| b64(&1.0f32.to_le_bytes())),
            initial_stencil_b64: stencil.then(|| b64(&[0xab])),
            state: None,
        })
    };
    let stencil = OutputSelection::Stencil {
        origin: [0, 0],
        dimensions: [1, 1],
    };
    authored.depth_stencil = depth_stencil(true, true);
    authored.output = stencil.clone();
    let (interface, companion) = checked_companion(&authored, &fragment);
    match render_case(&authored, &interface, WITHOUT_SLICE, companion.as_ref()) {
        Err(ExecutorError::NotSupported(crate::check::admission::NotSupported::Requirements(
            requirements,
        ))) => assert_eq!(
            requirements,
            [crate::requirement::ToolingRequirement::SeparateDepthStencil].into()
        ),
        other => panic!("both aspects: {:?}", other.map(|_| ())),
    }
    #[cfg(target_os = "linux")]
    {
        let attachmentless =
            metal2vulkan_devtools::llvm_assemble_text(crate::case::ATTACHMENTLESS_FRAGMENT_AIR)
                .expect("llvm-as on the device lane");
        let depth = OutputSelection::Depth {
            origin: [0, 0],
            dimensions: [1, 1],
        };
        for (own, entry, aspect, output) in [
            (
                public("engine/sample_rate.ll"),
                "sample_rate_depth",
                true,
                depth,
            ),
            (attachmentless, "fragment_no_writes", false, stencil.clone()),
        ] {
            let mut authored =
                crate::companion::tests::case(Stage::Fragment, entry, TextureFormat::Rgba32Float);
            authored.render_targets.clear();
            authored.depth_stencil = depth_stencil(aspect, !aspect);
            authored.output = output;
            let (case, companion) = with_companion(&authored, &own);
            assert!(crate::candidate::engine::runs_render_through_runtime(&case));
            let through = run_on_runtime(&case, &own, Some(&companion)).unwrap();
            // The depth the function writes; a stencil no function writes keeps its initial byte
            // under the encoder's default state (always pass, keep).
            let expected = match aspect {
                true => 0.125f32.to_le_bytes().to_vec(),
                false => vec![0xab],
            };
            assert_eq!(through.output, expected, "{entry}");
        }
        authored.depth_stencil = depth_stencil(false, true);
        let (case, companion) = with_companion(&authored, &fragment);
        assert!(crate::candidate::engine::runs_render_through_runtime(&case));
        match run_on_runtime(&case, &fragment, Some(&companion)) {
            Ok(through) => assert_eq!(through.output, [7], "fragment_depth_stencil"),
            Err(error @ ExecutorError::NotSupported(_)) => {
                assert!(error.to_string().contains("StencilExport"), "{error}")
            }
            Err(error) => panic!("{error}"),
        }
    }
}

/// A vertex case observed through a varying and through its position, each drawn into its
/// generated observer fragment: the quadrant's pixels hold the varying padded to `(v, 0, 1)`, or
/// the window position `(x + 0.5, y + 0.5, z, 1 / w)`; the rest keep their initial -1.
#[cfg(target_os = "linux")]
#[test]
fn an_observed_vertex_case_draws_into_its_generated_observer() {
    let vertex = public("vertex_varying_triangle.ll");
    let mut authored = crate::companion::tests::case(
        Stage::Vertex,
        "vertex_varying_triangle",
        TextureFormat::Rgba32Float,
    );
    authored_quad_target(&mut authored);
    authored.vertex_inputs = vec![
        AttributeInput {
            location: 0,
            format: AttributeFormat::Float2,
            stride: 8,
            dynamic_stride: false,
            bytes_b64: b64(&floats(&QUADRANT.concat())),
        },
        AttributeInput {
            location: 1,
            format: AttributeFormat::Float2,
            stride: 8,
            dynamic_stride: false,
            bytes_b64: b64(&floats(&[0.25, 0.75].repeat(QUADRANT.len()))),
        },
    ];
    authored.draw = Some(Draw {
        primitive: Primitive::Triangle,
        vertex_start: 0,
        vertex_count: QUADRANT.len() as u32,
        instance_count: 1,
        index_buffer: None,
        base_vertex: 0,
        base_instance: 0,
    });
    let mut position = vec![[-1.0f32; 4]; 16];
    for (y, x) in [(0, 2), (0, 3), (1, 2), (1, 3)] {
        position[y * 4 + x] = [x as f32 + 0.5, y as f32 + 0.5, 0.0, 1.0];
    }
    for (observation, expected) in [
        (
            VertexObservation::Varying { location: 0 },
            top_right([0.25, 0.75, 0.0, 1.0]),
        ),
        (VertexObservation::Position, floats(&position.concat())),
    ] {
        authored.vertex_observation = Some(observation);
        let (case, companion) = with_companion(&authored, &vertex);
        let run = run(
            &case,
            RenderLibraries {
                vertex: &vertex,
                fragment: Some(&companion),
            },
        )
        .unwrap();
        assert_eq!(run.output, expected, "{observation:?}");
    }
}

/// A pass of 4 samples whose one pixel's samples are authored apart, sample `s` holding
/// `(10 + s, 20 + s, 30 + s, 40 + s)`: `sample_rate_mask_out` writes `(1, 0, 1, 1)` under the
/// sample mask `0x5`, so samples 0 and 2 take its color and samples 1 and 3 keep their own
/// authored contents. Through the runtime the target is filled and read back sample by sample
/// (`platform::runtime_multisample`) and holds the same words.
#[cfg(target_os = "linux")]
#[test]
fn each_sample_of_a_multisampled_target_keeps_its_own_contents() {
    let fragment = public("engine/sample_rate.ll");
    let mut authored = crate::companion::tests::case(
        Stage::Fragment,
        "sample_rate_mask_out",
        TextureFormat::Rgba32Float,
    );
    authored.raster_sample_count = Some(4);
    let sample = |s: f32| [10.0 + s, 20.0 + s, 30.0 + s, 40.0 + s];
    let authored_samples: Vec<f32> = (0..4u8).flat_map(|s| sample(f32::from(s))).collect();
    authored.render_targets[0].initial_bytes_b64 = b64(&floats(&authored_samples));
    authored.output = OutputSelection::RenderTarget {
        index: 0,
        origin: [0, 0],
        dimensions: [1, 1],
    };
    let (case, companion) = with_companion(&authored, &fragment);
    let written = [1.0, 0.0, 1.0, 1.0];
    let expected = floats(&[written, sample(1.0), written, sample(3.0)].concat());
    let run = run(
        &case,
        RenderLibraries {
            vertex: &companion,
            fragment: Some(&fragment),
        },
    )
    .unwrap();
    assert_eq!(run.output, expected, "through the runtime");
}

/// A fragment case of no attachment draws into the oracle's 1x1 pass at its samples:
/// `nat_sample` ORs each run's coverage, marked by 0x100, into its sample's word and counts the
/// run 8 words on, so at 1 and at 4 samples each sample runs once and reads its own bit. Through
/// the runtime the pass is `renderTargetWidth` by `renderTargetHeight` at
/// `defaultRasterSampleCount`, and its words are the executor's own.
#[cfg(target_os = "linux")]
#[test]
fn a_fragment_pass_of_no_attachment_runs_each_sample_once() {
    let fragment = public("engine/coverage_natural.ll");
    for samples in [1u32, 4] {
        let mut authored = crate::companion::tests::case(
            Stage::Fragment,
            "nat_sample",
            TextureFormat::Rgba32Float,
        );
        authored.render_targets.clear();
        authored.raster_sample_count = (samples > 1).then_some(samples);
        authored.buffers.push(crate::case::BufferResource {
            binding: 0,
            role: crate::case::ResourceRole::InOut,
            bytes_b64: None,
            initial_bytes_b64: Some(b64(&[0; 64])),
            rasterization_rate_map: None,
        });
        authored.output = OutputSelection::Buffer {
            binding: 0,
            offset: 0,
            length: 64,
        };
        let (case, companion) = with_companion(&authored, &fragment);
        let drawn = run(
            &case,
            RenderLibraries {
                vertex: &companion,
                fragment: Some(&fragment),
            },
        )
        .unwrap();
        let words: Vec<u32> = drawn
            .output
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        for sample in 0..8 {
            let ran = sample < samples;
            let own = 0x100 | (1 << sample);
            let word = words[sample as usize];
            assert_eq!(word & own, if ran { own } else { 0 }, "{samples}: {sample}");
            assert_eq!(
                words[8 + sample as usize],
                u32::from(ran),
                "{samples}: {sample}"
            );
        }
    }
}

/// A tessellation case drawn through the render executor's patch path: the factor buffer bound
/// where the plan's factor write names it, the probe as the evaluation stage, and the case's cull
/// mode. `k7_tess_quad_probe` draws patch 0 over the target's left half and patch 1, its mirror
/// image, over the right half, each tagging its pixels with its patch index (the flat `tag`,
/// `(patch, 1, 2, 3)`). Without culling both halves are tagged; with back faces culled exactly one
/// patch survives, a different one for each output winding, and the culled half keeps the target's
/// fill. Through the runtime, each draw gives the executor's own bytes.
#[cfg(target_os = "linux")]
#[test]
fn a_back_culled_patch_and_its_mirror_leave_exactly_one_drawn() {
    use crate::case::{CullMode, TessellationDraw, TessellationFactors, TessellationWinding};
    const FILL: u8 = 0xab;
    let vertex = public("engine/k7_tess_quad_probe.ll");
    let mut authored = crate::companion::tests::case(
        Stage::Vertex,
        "k7_tess_quad_probe",
        TextureFormat::Rgba32Uint,
    );
    authored.render_targets[0].dimensions = [4, 4];
    authored.render_targets[0].initial_bytes_b64 = b64(&[FILL; 4 * 4 * 16]);
    authored.output = OutputSelection::RenderTarget {
        index: 0,
        origin: [0, 0],
        dimensions: [4, 4],
    };
    authored.vertex_observation = Some(VertexObservation::Varying { location: 1 });
    authored.draw = None;
    let two = TessellationFactors {
        edge_f16: vec![0x4000; 4],
        inside_f16: vec![0x4000; 2],
    };
    authored.tessellation = Some(TessellationDraw {
        factors: vec![two.clone(), two],
        instance_count: 1,
        amplification_count: 1,
        control_points: vec![],
        // Each patch's clip-space rectangle: origin, then signed extent.
        patch_inputs: vec![AttributeInput {
            location: 0,
            format: AttributeFormat::Float4,
            stride: 16,
            dynamic_stride: false,
            bytes_b64: b64(&floats(&[-1.0, -1.0, 1.0, 2.0, 1.0, -1.0, -1.0, 2.0])),
        }],
        partition_mode: Default::default(),
        output_winding: Default::default(),
        cull_mode: CullMode::None,
    });
    // The target with its left two columns tagged by `left` and its right two by `right`, a
    // `None` half keeping the fill.
    let tagged = |left: Option<u32>, right: Option<u32>| -> Vec<u8> {
        (0..16)
            .flat_map(|pixel| match if pixel % 4 < 2 { left } else { right } {
                Some(patch) => [patch, 1, 2, 3]
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect::<Vec<_>>(),
                None => vec![FILL; 16],
            })
            .collect()
    };
    let draw = |authored: &AuthoredCase| {
        let (case, companion) = with_companion(authored, &vertex);
        run(
            &case,
            RenderLibraries {
                vertex: &vertex,
                fragment: Some(&companion),
            },
        )
        .unwrap()
        .output
    };
    assert_eq!(draw(&authored), tagged(Some(0), Some(1)));
    let one_survivor = [tagged(Some(0), None), tagged(None, Some(1))];
    let mut survivors = Vec::new();
    for winding in [
        TessellationWinding::Clockwise,
        TessellationWinding::CounterClockwise,
    ] {
        let tessellation = authored.tessellation.as_mut().unwrap();
        tessellation.output_winding = winding;
        tessellation.cull_mode = CullMode::Back;
        let output = draw(&authored);
        let survivor = one_survivor
            .iter()
            .position(|expected| *expected == output)
            .unwrap_or_else(|| panic!("{winding:?}: not exactly one patch drawn: {output:?}"));
        survivors.push(survivor);
    }
    assert_ne!(
        survivors[0], survivors[1],
        "the output winding chooses which patch faces front"
    );
}

/// Framebuffer fetch: `fetch_scalar` writes half of the color it reads, its attachment loaded with
/// the authored contents, and the fullscreen triangle drawn as `n` instances covers every pixel `n`
/// times. Every device reads it as an input attachment, after a barrier between draws
/// (`platform::graphics_pipeline::fetch_between_draws`), and its fragment module notes
/// `PrecisionNote::UnorderedFetchWithinDraw`: one instance halves the attachment exactly, and
/// three, one draw, leave each texel some order's `2^-k` of its contents, `k` in 1 to 3.
#[cfg(target_os = "linux")]
#[test]
fn a_framebuffer_fetch_halves_its_attachment_once_per_overlapping_instance() {
    use metal2vulkan::pipeline::PrecisionNote;
    let fragment = llvm_assemble(&metal2vulkan_devtools::engine_dir().join("fragment_fetch.ll"))
        .expect("llvm-as on the device lane");
    let mut authored =
        crate::companion::tests::case(Stage::Fragment, "fetch_scalar", TextureFormat::R32Float);
    let initial = [1.0f32, -2.0, 0.75, 96.0];
    authored.render_targets[0].dimensions = [2, 2];
    authored.render_targets[0].initial_bytes_b64 = b64(&floats(&initial));
    authored.output = OutputSelection::RenderTarget {
        index: 0,
        origin: [0, 0],
        dimensions: [2, 2],
    };
    for instances in [1u32, 3] {
        authored
            .draw
            .as_mut()
            .expect("the fullscreen triangle")
            .instance_count = instances;
        let (case, companion) = with_companion(&authored, &fragment);
        match run(
            &case,
            RenderLibraries {
                vertex: &companion,
                fragment: Some(&fragment),
            },
        ) {
            Ok(run) => {
                let unordered = (run.census.modules.iter()).any(|m| {
                    m.precision
                        .contains(&PrecisionNote::UnorderedFetchWithinDraw)
                });
                let scale = |k: u32| 0.5f32.powi(k as i32);
                let scaled = |k| floats(&initial.map(|value| value * scale(k)));
                if unordered && instances > 1 {
                    let orders: Vec<Vec<u8>> = (1..=instances).map(scaled).collect();
                    for (texel, got) in run.output.as_chunks::<4>().0.iter().enumerate() {
                        assert!(
                            (orders.iter()).any(|order| order[texel * 4..][..4] == got[..]),
                            "{instances} instances, texel {texel}: {got:?}"
                        );
                    }
                } else {
                    assert_eq!(run.output, scaled(instances), "{instances} instance(s)");
                }
                eprintln!(
                    "ENGINE-FETCH {instances} instance(s): unordered within the draw: {unordered}"
                );
            }
            Err(other) => panic!("{instances} instance(s): {other}"),
        }
    }
}

/// The authored narrow-attribute case's mapping, drawn: a rasterization-disabled vertex function
/// reads a `uchar` and a `ushort2` attribute and stores their sum.
#[cfg(target_os = "linux")]
#[test]
fn a_rasterization_disabled_vertex_draw_stores_through_its_buffer() {
    let authored =
        crate::case::narrow_vertex_test_case(String::new(), "vertex_narrow_attributes".into());
    let case = render_case(
        &authored,
        &crate::check::interface::tests::vertex_buffers_at(&[0]),
        WITHOUT_SLICE,
        None,
    )
    .unwrap();
    let vertex = public("vertex_narrow_attributes.ll");
    let run = run(
        &case,
        RenderLibraries {
            vertex: &vertex,
            fragment: None,
        },
    )
    .unwrap();
    assert_eq!(run.output, (2u32 + 3 + 4).to_le_bytes());
}

/// A fragment function that loops `n` times (`n` from its constant buffer) draws once, as
/// compiled, with no execution bound and no headroom probe, on both lanes.
#[cfg(target_os = "linux")]
#[test]
fn a_looping_fragment_draws_as_compiled() {
    use crate::candidate::engine::tests::counting_kernel;
    let header = counting_kernel()
        .split("define")
        .next()
        .expect("the shared header")
        .to_owned();
    let text = format!(
        "{header}define <4 x float> @fragment_loop(ptr addrspace(2) noundef \"air-buffer-no-alias\" %0) {{\n\
entry:\n\
  %n = load i32, ptr addrspace(2) %0, align 4\n\
  br label %loop\n\
loop:\n\
  %i = phi i32 [ 0, %entry ], [ %next, %body ]\n\
  %acc = phi float [ 0.000000e+00, %entry ], [ %sum, %body ]\n\
  %again = icmp ult i32 %i, %n\n\
  br i1 %again, label %body, label %done\n\
body:\n\
  %sum = fadd float %acc, 1.000000e+00\n\
  %next = add i32 %i, 1\n\
  br label %loop\n\
done:\n\
  %color = insertelement <4 x float> <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 1.000000e+00>, float %acc, i64 0\n\
  ret <4 x float> %color\n\
}}\n\n\
!air.fragment = !{{!0}}\n!air.version = !{{!6}}\n!air.language_version = !{{!7}}\n\
!0 = !{{ptr @fragment_loop, !1, !2}}\n!1 = !{{!3}}\n!2 = !{{!4}}\n\
!3 = !{{!\"air.render_target\", i32 0, i32 0, !\"air.arg_type_name\", !\"float4\", !\"air.arg_name\", !\"color\"}}\n\
!4 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read\", !\"air.address_space\", i32 2, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"n\"}}\n\
!6 = !{{i32 2, i32 8, i32 0}}\n!7 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n"
    );
    let fragment =
        crate::candidate::engine::tests::assemble(&text).expect("llvm-as on the device lane");
    let vertex = public("vertex_varying_triangle.ll");
    let covering = [[-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]];
    let case = {
        let mut case = two_stage_case(
            "vertex_varying_triangle",
            "fragment_loop",
            triangle_inputs(&covering, [0.0, 0.0]),
            BTreeMap::from([(
                0,
                ColorTarget {
                    format: TextureFormat::Rgba32Float,
                    dimensions: [1, 1],
                    samples: 1,
                    bytes: floats(&[-1.0; 4]),
                },
            )]),
            3,
        );
        case.fragment
            .as_mut()
            .expect("a fragment stage")
            .resources
            .buffers
            .insert(0, 3u32.to_le_bytes().to_vec());
        case
    };
    let libraries = RenderLibraries {
        vertex: &vertex,
        fragment: Some(&fragment),
    };
    let ok = run(&case, libraries).unwrap();
    assert_eq!(ok.output, floats(&[3.0, 0.0, 0.0, 1.0]));
}
