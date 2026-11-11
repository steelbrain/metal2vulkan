//! The engine render candidate on a device: owned vertex and fragment fixtures drawn through
//! one `RenderPipeline` under `VK_LAYER_KHRONOS_validation`, and authored graphics cases drawn with
//! their generated companions, against expectations derived from the MSL and Metal's conventions
//! (y-up clip space, pixel-center coverage). `G-linux`'s `m2v_cpu_icd` step runs these on lavapipe.
//! The looping fragment runs as compiled on both lanes.

use super::engine_compute::BACKEND;
use super::*;
use crate::candidate::engine::graphics::tests::{
    checked_companion, draw_request, reflected_imageblock, two_stage_case, RenderLibraries,
    WITHOUT_SLICE,
};
use crate::candidate::engine::graphics::{
    render_case, ColorTarget, RenderCase, RenderOutput, VertexBuffer,
};
use crate::candidate::engine::{compile_render, ComputeRun};
use crate::case::{
    AttributeFormat, AttributeInput, AuthoredCase, Draw, OutputSelection, Primitive, Stage,
    TextureFormat, VertexObservation,
};
use crate::check::admission::ExecutorError;
use base64::Engine as _;
use metal2vulkan_devtools::{fixtures_dir, llvm_assemble};
use std::collections::BTreeMap;

/// Draw `case`, compiled in the device tests' compile worker ([`compile_worker`]) by the request
/// the tools send it. Owned draws link nothing, so they bind no visible-function tables.
fn run(case: &RenderCase, libraries: RenderLibraries<'_>) -> Result<ComputeRun, ExecutorError> {
    platform::execute_engine_render(
        crate::gpu::TEST_DEVICE,
        "engine-render-test",
        case,
        |device, shape| {
            let (request, bitcode) = draw_request(case, libraries, shape);
            let sources = bitcode
                .into_iter()
                .map(crate::acceptance::retained)
                .collect::<Vec<_>>();
            compile_render(compile_worker(), device, &request, &sources, case, &[])
        },
        |_, token, _, _| Err(format!("no template has pipeline token {token}").into()),
        BACKEND,
    )
    .result
}

/// [`run`] of a mapped authored case whose own function is in `bitcode`: its generated companion
/// from the companion's own bitcode, each library at its stage (the runtime lane's fresh candidate,
/// `runtime_lane::renders`).
pub(crate) fn run_mapped(case: &RenderCase, bitcode: &[u8]) -> Result<ComputeRun, ExecutorError> {
    let companion = case
        .companion
        .as_ref()
        .map(crate::companion::Companion::opaque_bitcode)
        .transpose()
        .map_err(ExecutorError::Failed)?;
    let libraries = match (case.companion_stage(), companion.as_deref()) {
        (Some(metal2vulkan::pipeline::Stage::Vertex), Some(generated)) => RenderLibraries {
            vertex: generated,
            fragment: Some(bitcode),
        },
        (Some(metal2vulkan::pipeline::Stage::Fragment), Some(generated)) => RenderLibraries {
            vertex: bitcode,
            fragment: Some(generated),
        },
        _ => RenderLibraries {
            vertex: bitcode,
            fragment: None,
        },
    };
    run(case, libraries)
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
            dynamic: false,
            step: metal2vulkan::metal::VertexStepFunction::PerVertex,
            bytes: floats(&positions.concat()),
        },
        VertexBuffer {
            location: 1,
            buffer_index: 29,
            format: AttributeFormat::Float2,
            stride: 8,
            dynamic: false,
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
#[cfg(any(target_os = "linux", target_os = "macos"))]
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
    assert_eq!(run.status, metal2vulkan::bind::StatusReport::default());
    assert_eq!(run.environment["executor"], "engine-render");
}

/// A varying written by the vertex stage reaches the fragment stage of another library by its
/// user key; a value equal at every vertex interpolates to itself.
#[cfg(any(target_os = "linux", target_os = "macos"))]
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
    let render = render_case(case, &interface, WITHOUT_SLICE, companion.as_ref(), None).unwrap();
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
#[cfg(any(target_os = "linux", target_os = "macos"))]
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
    assert_eq!(run.status, metal2vulkan::bind::StatusReport::default());
}

/// A fragment case with a depth/stencil attachment: `fragment_depth_stencil` writes depth 0.5 under
/// `depth(any)` and stencil 7, and the oracle's depth-stencil state (always-pass depth test with
/// writes, always-pass stencil test that replaces) stores both over the attachment's authored 1.0
/// and 0xab, as the oracle does for case 685a8d84.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn a_fragment_case_writes_its_depth_and_stencil_attachment() {
    let fragment = public("fragment_depth_stencil.ll");
    let mut authored = crate::companion::tests::case(
        Stage::Fragment,
        "fragment_depth_stencil",
        TextureFormat::Rgba32Float,
    );
    authored.render_targets.clear();
    authored.depth_stencil = Some(crate::case::DepthStencilResource {
        dimensions: [1, 1],
        initial_depth_b64: Some(b64(&1.0f32.to_le_bytes())),
        initial_stencil_b64: Some(b64(&[0xab])),
        state: None,
    });
    for (output, expected) in [
        (
            OutputSelection::Stencil {
                origin: [0, 0],
                dimensions: [1, 1],
            },
            vec![7],
        ),
        (
            OutputSelection::Depth {
                origin: [0, 0],
                dimensions: [1, 1],
            },
            0.5f32.to_le_bytes().to_vec(),
        ),
    ] {
        authored.output = output;
        let (case, companion) = with_companion(&authored, &fragment);
        let run = run(
            &case,
            RenderLibraries {
                vertex: &companion,
                fragment: Some(&fragment),
            },
        )
        .unwrap();
        assert_eq!(run.output, expected, "{:?}", authored.output);
        assert_eq!(run.status, metal2vulkan::bind::StatusReport::default());
    }
}

/// A vertex case observed through a varying and through its position, each drawn into its
/// generated observer fragment: the quadrant's pixels hold the varying padded to `(v, 0, 1)`, or
/// the window position `(x + 0.5, y + 0.5, z, 1 / w)`; the rest keep their initial -1.
#[cfg(any(target_os = "linux", target_os = "macos"))]
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

/// A tessellation case drawn through the render executor's patch path: the factor buffer bound
/// where the plan's factor write names it, the probe as the evaluation stage, and the case's cull
/// mode. `k7_tess_quad_probe` draws patch 0 over the target's left half and patch 1, its mirror
/// image, over the right half, each tagging its pixels with its patch index (the flat `tag`,
/// `(patch, 1, 2, 3)`). Without culling both halves are tagged; with back faces culled exactly one
/// patch survives, a different one for each output winding, and the culled half keeps the target's
/// fill.
#[cfg(any(target_os = "linux", target_os = "macos"))]
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
/// times in order, so under rasterization-order access each instance reads what the one before it
/// wrote and the attachment ends at `2^-n` of its contents. A device without that access but with
/// the pixel interlock (MoltenVK, RADV, ANV) reads and writes the attachment as a storage image
/// inside the interlock, in the same order, to the same result. A device with neither reads it as
/// an input attachment with no rasterization-order access, after a barrier between draws
/// (`platform::graphics_pipeline::fetch_between_draws`), and its fragment module notes
/// `PrecisionNote::UnorderedFetchWithinDraw`: one instance halves the attachment exactly, and
/// three, one draw, leave each texel some order's `2^-k` of its contents, `k` in 1 to 3.
#[cfg(any(target_os = "linux", target_os = "macos"))]
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
                assert_eq!(run.status, metal2vulkan::bind::StatusReport::default());
            }
            Err(other) => panic!("{instances} instance(s): {other}"),
        }
    }
}

/// `fragment_custom_imageblock` adds one to its imageblock's `user(depth)` member (a half at byte 2
/// of an 8-byte cell of four halves). Drawn as 1 and 3 overlapping instances of the fullscreen
/// triangle over a 2x2 pass whose cells hold four authored depths, each pixel's depth gains exactly
/// one per instance: every instance reads the cell after the previous one wrote it, inside the
/// pixel interlock, through the storage image the render executor binds where the plan's imageblock
/// write names it. A device without the interlock refuses typed, as the product refuses
/// (`Imageblocks` needs the feature).
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn an_imageblock_member_gains_one_per_overlapping_instance() {
    use crate::check::admission::NotSupported;
    use crate::engine::Outcome;
    let fragment = public("fragment_custom_imageblock.ll");
    let reflected = reflected_imageblock(
        &std::fs::read_to_string(fixtures_dir().join("fragment_custom_imageblock.ll")).unwrap(),
        "fragment_custom_imageblock",
    );
    let mut authored = crate::case::fragment_imageblock_test_case(
        String::new(),
        "fragment_custom_imageblock".into(),
    );
    let halves = |bits: [u16; 4]| bits.iter().flat_map(|bits| bits.to_le_bytes()).collect();
    // 1.0, -2.0, 0.5, 96.0, and each plus one and plus three.
    let depths: Vec<u8> = halves([0x3c00, 0xc000, 0x3800, 0x5600]);
    let plus = [
        (1u32, halves([0x4000, 0xbc00, 0x3e00, 0x5610])),
        (3, halves([0x4400, 0x3c00, 0x4300, 0x5630])),
    ];
    let imageblock = authored.fragment_imageblock.as_mut().unwrap();
    imageblock.dimensions = [2, 2];
    imageblock.members[0].initial_bytes_b64 = Some(b64(&depths));
    authored.render_targets[0].dimensions = [2, 2];
    authored.render_targets[0].initial_bytes_b64 = b64(&[0; 32]);
    authored.output = OutputSelection::FragmentImageblock {
        semantic: "user(depth)".into(),
        origin: [0, 0],
        dimensions: [2, 2],
    };
    for (instances, expected) in plus {
        authored
            .draw
            .as_mut()
            .expect("the fullscreen triangle")
            .instance_count = instances;
        // The checked reflection's master, which the cells follow.
        let (interface, companion) = checked_companion(&authored, &fragment);
        let case = render_case(
            &authored,
            &interface,
            WITHOUT_SLICE,
            companion.as_ref(),
            reflected.as_ref(),
        )
        .unwrap();
        let companion = companion
            .expect("a passthrough vertex")
            .opaque_bitcode()
            .unwrap();
        match run(
            &case,
            RenderLibraries {
                vertex: &companion,
                fragment: Some(&fragment),
            },
        ) {
            Ok(run) => {
                assert_eq!(run.output, expected, "{instances} instance(s)");
                assert_eq!(run.status, metal2vulkan::bind::StatusReport::default());
            }
            Err(ExecutorError::NotSupported(NotSupported::Product {
                outcome: Outcome::DeviceCapabilityMissing { feature, .. },
                ..
            })) => assert!(feature.contains("Imageblocks"), "{feature}"),
            Err(other) => panic!("{instances} instance(s): {other}"),
        }
    }
}

/// The authored narrow-attribute case's mapping, drawn: a rasterization-disabled vertex function
/// reads a `uchar` and a `ushort2` attribute and stores their sum.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn a_rasterization_disabled_vertex_draw_stores_through_its_buffer() {
    let authored =
        crate::case::narrow_vertex_test_case(String::new(), "vertex_narrow_attributes".into());
    let case = render_case(
        &authored,
        &crate::check::interface::tests::vertex_buffers_at(&[0]),
        WITHOUT_SLICE,
        None,
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

/// The narrow-attribute case with layouts of `MTLBufferLayoutStrideDynamic`, its buffers bound
/// at strides 2 and 8, drawn: vertex 1 reads each attribute one stride on, as Metal's fetch does,
/// where the pipeline's strides are dynamic state (`VERTEX_INPUT_BINDING_STRIDE`), which the
/// executor records as it binds the vertex buffers.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn dynamic_stride_layouts_read_at_the_strides_the_buffers_are_bound_with() {
    let mut authored =
        crate::case::narrow_vertex_test_case(String::new(), "vertex_narrow_attributes".into());
    authored.buffers[0].initial_bytes_b64 = Some(b64(&[0xab; 8]));
    let inputs: [(u32, &[u8]); 2] = [
        (2, &[2, 9, 5, 9]),
        (8, &[3, 0, 4, 0, 99, 0, 99, 0, 6, 0, 7, 0]),
    ];
    for (input, (stride, bytes)) in authored.vertex_inputs.iter_mut().zip(inputs) {
        input.stride = stride;
        input.dynamic_stride = true;
        input.bytes_b64 = b64(bytes);
    }
    authored.draw.as_mut().expect("a draw").vertex_count = 2;
    authored.output = crate::case::OutputSelection::Buffer {
        binding: 0,
        offset: 0,
        length: 8,
    };
    let case = render_case(
        &authored,
        &crate::check::interface::tests::vertex_buffers_at(&[0]),
        WITHOUT_SLICE,
        None,
        None,
    )
    .unwrap();
    assert!(case.vertex_buffers.iter().all(|buffer| buffer.dynamic));
    let vertex = public("vertex_narrow_attributes.ll");
    let ran = run(
        &case,
        RenderLibraries {
            vertex: &vertex,
            fragment: None,
        },
    );
    let expected: Vec<u8> = [2u32 + 3 + 4, 5 + 6 + 7]
        .iter()
        .flat_map(|sum| sum.to_le_bytes())
        .collect();
    match ran {
        Ok(run) => assert_eq!(run.output, expected),
        // A profile without dynamic strides keeps a per-bind stride static, a pipeline per
        // stride, which the executor does not create.
        Err(ExecutorError::NotSupported(
            crate::check::admission::NotSupported::NotImplemented(why),
        )) if why.contains("a pipeline per stride") => {
            eprintln!("dynamic stride layouts: {why}")
        }
        Err(other) => panic!("dynamic stride layouts: {other}"),
    }
}

/// A fragment function that loops `n` times (`n` from its constant buffer) draws once, as
/// compiled, with no execution bound and no headroom probe, on both lanes.
#[cfg(any(target_os = "linux", target_os = "macos"))]
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
