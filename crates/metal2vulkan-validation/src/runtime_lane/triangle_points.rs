//! A triangle in fill mode `Points` on the device the cfg selects, under the validation layer.
//!
//! `TriangleFillMode::Points` is `setTriangleFillMode:`'s raw value 2, which no SDK header
//! declares and Apple's GL renderer sends for `glPolygonMode(GL_POINT)`. A Metal probe on an Apple
//! M3 Max (macOS 27.0.1) drew one triangle, `(3.5, 3.5)`, `(12.5, 3.5)` and `(3.5, 12.5)` in a
//! 16x16 target, clockwise and so front-facing under Metal's default winding: in `Points` each
//! vertex's pixel alone ([`VERTICES`]), one pixel wide whether the vertex function wrote a
//! `[[point_size]]` of 4, of 1 or none; nothing under `setCullMode:` `Front`, the three pixels under
//! `Back`. The consumer's probe drew the three pixels on the M3 Max and an Intel UHD Graphics 630,
//! where a Radeon Pro 5300M filled the triangle. The runtime draws the mode as
//! `VK_POLYGON_MODE_POINT`, which culls the triangle first as Metal does.
//!
//! Here `rt6_restart`'s vertex function, which writes a `[[point_size]]` of 1, draws that triangle
//! with `rt6_render_state`'s white fragment function: `Points` stores Metal's pixels, culled and
//! not, and announces `triangle-point-size` ([`TRIANGLE_POINT_SIZE`]), its vertex function writing
//! a point size; `Fill` covers the triangle's inside and `Lines` does not. `rt6_small`, which writes
//! none, draws points of one pixel at its vertices through a viewport that puts them at pixel
//! centers, announcing nothing. A device without `fillModeNonSolid` refuses the mode, as it does
//! `Lines`, and so does a portability subset without `pointPolygons` (MoltenVK's), which Vulkan
//! forbids the mode.

use super::buffer_with;
use super::degrade_notices::{assert_made_once, made_render_kinds, shared_in_system_memory};
use super::line_widths::line_ties;
use super::render_state::{quiet, validated, validated_without, Probe, PASS, WHITE};
use crate::air_library::AirLibrary;
use metal2vulkan::device::VkFeature;
use metal2vulkan::error::{Capability, Feature, Unsupported};
use metal2vulkan::metal::{CullMode, PixelFormat, PrimitiveType, TriangleFillMode};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, Degradation, Error, PipelineOptions, RenderCommandEncoder, RenderPipelineState,
    Viewport,
};
use std::collections::BTreeSet;

/// The kinds `triangle_points` takes: a triangle's points sized by its vertex function.
pub(super) const TRIANGLE_POINT_SIZE: [Degradation; 1] = [Degradation::TrianglePointSize];

/// The probe's triangle, in pixels of the 16x16 target.
const TRIANGLE: [[f32; 2]; 3] = [[3.5, 3.5], [12.5, 3.5], [3.5, 12.5]];

/// The pixels `(x, y)` Metal stored of [`TRIANGLE`] in fill mode `Points` (the module docs).
const VERTICES: [(u32, u32); 3] = [(3, 3), (12, 3), (3, 12)];

/// A pixel inside [`TRIANGLE`], on none of its edges.
const INSIDE: (u32, u32) = (5, 5);

/// The descriptor of a pipeline drawing `rt6_restart`'s vertices, white, of Metal's default
/// input primitive topology class (none: Metal refuses a point size under the triangle class).
fn sized_descriptor(probe: &Probe) -> RenderPipelineDescriptor {
    let function = |stem: &str, name: &str| {
        let bitcode = llvm_assemble(&engine_dir().join(format!("{stem}.ll"))).unwrap();
        probe
            .device
            .new_library_with_air(&bitcode)
            .unwrap()
            .new_function_with_name(name)
            .unwrap()
            .function_descriptor()
    };
    let mut descriptor = RenderPipelineDescriptor::new(StageSource::Function(function(
        "rt6_restart",
        "rt6_restart_vertex",
    )));
    descriptor.fragment = Some(StageSource::Function(function(
        "rt6_render_state",
        "rt6_white",
    )));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    descriptor
}

/// The kinds a test forces where the device has not announced them, beside its pipeline states'
/// makings': the `Shared` buffers' system memory (its passes' `Shared` textures announce none). A
/// test that makes no buffer forces nothing.
fn shared(probe: &Probe) -> BTreeSet<Degradation> {
    BTreeSet::from_iter(
        shared_in_system_memory(&probe.device).then_some(Degradation::SharedInSystemMemory),
    )
}

/// The lit pixels `(x, y)` of a 16x16 pass `body` draws into, the buffers it holds kept until the
/// pass completed.
fn lit(
    probe: &Probe,
    body: impl FnOnce(&mut RenderCommandEncoder, &mut Vec<Buffer>),
) -> BTreeSet<(u32, u32)> {
    let stored = probe.run(PASS.size(16), |_, e, held| body(e, held));
    (stored.color.chunks(4).enumerate())
        .filter(|(_, texel)| texel[0] != 0)
        .map(|(i, _)| ((i % 16) as u32, (i / 16) as u32))
        .collect()
}

/// The refusal `setTriangleFillMode:` of `fill` returns in a pass of `probe`'s device, if any.
fn refusal(probe: &Probe, fill: TriangleFillMode) -> Option<Unsupported> {
    let mut refusal = None;
    lit(probe, |e, _| refusal = e.set_triangle_fill_mode(fill).err());
    refusal.map(|error| match error {
        Error::Unsupported(unsupported) => unsupported,
        other => panic!("{fill:?} refused otherwise: {other:?}"),
    })
}

/// Whether `probe`'s device refuses `Points` for its portability subset's withheld
/// `pointPolygons` (MoltenVK), where the call returns that refusal.
fn refuses_points(probe: &Probe) -> bool {
    let polygons = VkFeature::PointPolygons;
    let profile = probe.device.compile_device().profile();
    let refuses = profile.enabled().portability_withholds(polygons);
    if refuses {
        let needs = Unsupported::needs(Feature::TriangleFillPoints, Capability::Feature(polygons));
        assert_eq!(refusal(probe, TriangleFillMode::Points), Some(needs));
    }
    refuses
}

/// [`TRIANGLE`] stores Metal's pixels in `Points`, culled and not, and its inside in `Fill` alone,
/// and announces `triangle-point-size` once, the vertex function writing a point size (and the
/// width-1 lines' `device-line-ties` where the driver's ties are not measured). A device that
/// refuses `Points` ([`refuses_points`]) draws none of it.
#[test]
fn a_triangle_in_fill_mode_points_draws_its_vertices() {
    let (messages, _cache, probe) = validated();
    if refuses_points(&probe) {
        assert_made_once(&messages, &[], &BTreeSet::new());
        quiet(probe, &messages);
        return;
    }
    let descriptor = sized_descriptor(&probe);
    let pipeline: RenderPipelineState = probe
        .device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap();
    let positions: Vec<u8> = TRIANGLE
        .iter()
        .flat_map(|[x, y]| [x / 8.0 - 1.0, 1.0 - y / 8.0])
        .flat_map(f32::to_le_bytes)
        .collect();
    let positions = buffer_with(&probe.device, &positions).unwrap();
    let draw = |fill, cull| {
        lit(&probe, |e, _| {
            e.set_render_pipeline_state(&pipeline).unwrap();
            e.set_vertex_buffer(Some(&positions), 0, 0).unwrap();
            e.set_cull_mode(cull);
            e.set_triangle_fill_mode(fill).unwrap();
            e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
        })
    };
    let points = draw(TriangleFillMode::Points, CullMode::None);
    let front = draw(TriangleFillMode::Points, CullMode::Front);
    let back = draw(TriangleFillMode::Points, CullMode::Back);
    let fill = draw(TriangleFillMode::Fill, CullMode::None);
    let lines = draw(TriangleFillMode::Lines, CullMode::None);
    eprintln!(
        "TRIANGLE-POINTS {}: points {points:?}, culled front {front:?}, culled back {back:?}, \
         fill {} pixels, lines {}",
        probe.device.name(),
        fill.len(),
        lines.len()
    );
    let kinds = [&TRIANGLE_POINT_SIZE[..], line_ties(&probe.device)].concat();
    let mut first = made_render_kinds(&probe.device, &descriptor);
    first.extend(shared(&probe));
    assert_made_once(&messages, &kinds, &first);
    quiet(probe, &messages);
    let metal = BTreeSet::from(VERTICES);
    assert_eq!(points, metal);
    assert_eq!(front, BTreeSet::new());
    assert_eq!(back, metal);
    assert!(fill.contains(&INSIDE), "{fill:?}");
    assert!(
        !lines.contains(&INSIDE) && lines.len() > metal.len(),
        "{lines:?}"
    );
}

/// `rt6_small`, which writes no point size, draws in fill mode `Points` a point of one pixel at
/// each vertex: a 6x6 viewport puts its `(-0.5, -0.5)`, `(0.5, -0.5)` and `(-0.5, 0.5)` at the
/// pixel centers `(1.5, 4.5)`, `(4.5, 4.5)` and `(1.5, 1.5)`. Nothing is announced.
#[test]
fn a_function_without_a_point_size_draws_points_of_one_pixel() {
    let (messages, _cache, probe) = validated();
    if refuses_points(&probe) {
        assert_made_once(&messages, &[], &BTreeSet::new());
        quiet(probe, &messages);
        return;
    }
    let pipe = WHITE.vertex("rt6_small");
    let stored = lit(&probe, |e, held| {
        e.set_viewport(Viewport::new(0.0, 0.0, 6.0, 6.0, 0.0, 1.0))
            .unwrap();
        e.set_triangle_fill_mode(TriangleFillMode::Points).unwrap();
        probe.draw(e, held, pipe, 0.5, 0.0);
    });
    let mut first = probe.made(pipe);
    first.extend(shared(&probe));
    assert_made_once(&messages, &[], &first);
    quiet(probe, &messages);
    assert_eq!(stored, BTreeSet::from([(1, 4), (4, 4), (1, 1)]));
}

/// A device without `fillModeNonSolid` refuses `Points` at the call, naming the feature, as it
/// refuses `Lines`.
#[test]
fn points_need_fill_mode_non_solid() {
    let (messages, _cache, probe) = validated_without(&[VkFeature::FillModeNonSolid]);
    for (fill, what) in [
        (TriangleFillMode::Points, Feature::TriangleFillPoints),
        (TriangleFillMode::Lines, Feature::TriangleFillLines),
    ] {
        let needs = Unsupported::needs(what, Capability::Feature(VkFeature::FillModeNonSolid));
        assert_eq!(refusal(&probe, fill), Some(needs), "{fill:?}");
    }
    assert_made_once(&messages, &[], &BTreeSet::new());
    quiet(probe, &messages);
}
