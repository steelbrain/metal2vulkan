//! A render pipeline variant that draws its lines, or its triangles' edges, as bands
//! (`VariantKey::line_band`, `pipeline::link::band`), over the owned fixtures `engine/rt6_provoking` (a vertex function that
//! writes a smooth and a flat varying, and fragment functions that read them, the primitive ID and
//! the barycentric weights) and `engine/stage_io` (a vertex function that writes clip
//! distances, and a fragment function that reads the primitive ID):
//!
//! - On every committed profile with `tessellationShader` the variant of either primitive carries
//!   the band's synthesized control and evaluation stages right after the vertex stage, draws a
//!   `PATCH_LIST` that no draw changes, of the primitive's control points (two for a line, three
//!   for a triangle, as the control stage outputs), places the evaluation stage's push range where
//!   no other range lies, and every module passes L3 (`spirv-val` and requirements within the
//!   profile); the binder plans it for a shape that draws bands of its primitive. The pipeline
//!   without the variant has no tessellation stage.
//! - The band takes a varying the fragment stage reads flat, or an integer one, from the first
//!   control point, and a smooth one from each corner's endpoint.
//! - The control stage's per-vertex inputs and outputs, and the evaluation stage's inputs, are the
//!   vertex stage's outputs entire, its point size included where the device's tessellation
//!   stages take one.
//! - Without `tessellationShader` the variant is refused needing it; a fragment function that
//!   reads `[[barycentric_coord]]` is not implemented.
//!
//! `llvm-as` assembles the fixtures; a missing tool skips visibly. No GPU is used.

use crate::bind::BindState;
use crate::budget::Budget;
use crate::cache::VariantKey;
use crate::device::requirements::VkFeature as F;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{Capability, Error, Feature, Needs};
use crate::library::Library;
use crate::metal::format::tests::committed_devices;
use crate::metal::PixelFormat;
use crate::pipeline::descriptor::{
    ColorAttachmentDescriptor, FunctionDescriptor, RenderPipelineDescriptor, StageSource,
};
use crate::pipeline::link::band;
use crate::pipeline::testing::{compile, engine_dir, llvm_assemble, AssembleError};
use crate::pipeline::BandPrimitive;
use crate::pipeline::RenderPipeline;
use crate::spirv::testing::{check_words_valid, committed_profiles};
use crate::vk;

fn assemble(name: &str) -> Option<Vec<u8>> {
    let path = engine_dir().join(name);
    match llvm_assemble(&path) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {path:?}: {why}"),
    }
}

/// A render pipeline of `vertex` and `fragment` from `bc`, drawing into one color of `format`.
fn render(
    bc: &[u8],
    vertex: &str,
    fragment: &str,
    format: PixelFormat,
) -> RenderPipelineDescriptor {
    let lib = Library::from_air(bc).expect("library");
    let stage = |name: &str| {
        StageSource::Function(FunctionDescriptor::new(
            lib.function(name).expect("function"),
        ))
    };
    let mut d = RenderPipelineDescriptor::new(stage(vertex));
    d.fragment = Some(stage(fragment));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(format));
    d
}

/// Both primitives a band draws.
const PRIMITIVES: [BandPrimitive; 2] = [BandPrimitive::Lines, BandPrimitive::Triangles];

/// The variant key of a band of `primitive`.
fn band_key(primitive: BandPrimitive) -> VariantKey {
    let mut key = VariantKey::new();
    key.line_band = Some(primitive);
    key
}

/// The `OutputVertices` execution mode of module `words`.
fn output_vertices(words: &[u32]) -> Option<u32> {
    use crate::spirv_binary::{parse_bytes, Operand};
    use crate::spv::{ExecutionMode as M, Op as O};
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let parsed = parse_bytes(&bytes).expect("parse");
    (parsed.instructions.iter())
        .filter(|i| i.opcode == O::ExecutionMode)
        .find_map(|i| match i.operands.as_slice() {
            [_, Operand::Word(m), Operand::Word(n)] if *m == M::OutputVertices as u32 => Some(*n),
            _ => None,
        })
}

/// Each stage of `p` valid on `dev`; and `p`'s stages in order.
fn valid_stages(p: &RenderPipeline, dev: &Device, what: &str) -> Vec<vk::ShaderStage> {
    for m in &p.stages {
        check_words_valid(&m.spirv, &p.requirements, dev.profile())
            .unwrap_or_else(|e| panic!("{what} {:?}: L3: {e}", m.stage));
    }
    p.stages.iter().map(|m| m.stage).collect()
}

/// On every committed profile with `tessellationShader`, the band variant of either primitive
/// carries the two synthesized stages after the vertex stage, each valid, draws a patch list of
/// the primitive's control points and pushes its band to the evaluation stage's own range; without
/// the variant, no stage.
#[test]
fn the_band_variant_draws_patches_of_its_primitive_through_valid_stages() {
    let Some(provoking) = assemble("rt6_provoking.ll") else {
        return;
    };
    let Some(stage_io) = assemble("stage_io.ll") else {
        return;
    };
    let pipelines = [
        (
            "flat and smooth",
            render(&provoking, "pv_vertex", "pv_flat", PixelFormat::RGBA8Unorm),
        ),
        (
            "smooth",
            render(
                &provoking,
                "pv_vertex_smooth",
                "pv_smooth",
                PixelFormat::RGBA8Unorm,
            ),
        ),
        (
            "clip distances",
            render(
                &stage_io,
                "stage_io_vertex_clip",
                "stage_io_fragment_primitive",
                PixelFormat::RGBA8Unorm,
            ),
        ),
    ];
    let budget = Budget::standard();
    let mut drawn = [0; 2];
    for (stem, dev) in committed_devices() {
        let tessellates = (dev.profile().enabled().features).contains(&F::TessellationShader);
        for ((what, d), (k, primitive)) in pipelines
            .iter()
            .flat_map(|p| std::iter::repeat(p).zip(PRIMITIVES.into_iter().enumerate()))
        {
            let what = format!("{stem} {what} {primitive:?}");
            let plain = match dev.compile_render_pipeline(d, &budget) {
                Ok(p) => p,
                // A pipeline the profile refuses without a band has no band to test.
                Err(Error::Unsupported(_)) => continue,
                Err(e) => panic!("{what}: {e}"),
            };
            assert!(plain.graphics.line_band.is_none(), "{what}");
            assert!(
                !valid_stages(&plain, &dev, &what).contains(&vk::ShaderStage::TESSELLATION_CONTROL),
                "{what}"
            );
            let banded = match dev.compile_render_variant(d, &band_key(primitive), &budget) {
                Ok(p) => p,
                Err(Error::Unsupported(u)) if !tessellates => {
                    assert_eq!(u.feature, Feature::WideLines, "{what}");
                    assert_eq!(
                        u.needs,
                        Needs::AnyOf(vec![vec![Capability::Feature(F::TessellationShader)]]),
                        "{what}"
                    );
                    continue;
                }
                Err(e) => panic!("{what}: {e}"),
            };
            assert!(tessellates, "{what}: a band without tessellationShader");
            assert_eq!(
                valid_stages(&banded, &dev, &what),
                [
                    vk::ShaderStage::VERTEX,
                    vk::ShaderStage::TESSELLATION_CONTROL,
                    vk::ShaderStage::TESSELLATION_EVALUATION,
                    vk::ShaderStage::FRAGMENT,
                ],
                "{what}"
            );
            let g = &banded.graphics;
            assert_eq!(g.topology, vk::PrimitiveTopology::PATCH_LIST, "{what}");
            assert!(
                !g.dynamic_states.iter().any(|&s| matches!(
                    s,
                    vk::DynamicState::PRIMITIVE_TOPOLOGY
                        | vk::DynamicState::PRIMITIVE_RESTART_ENABLE
                )),
                "{what}"
            );
            let band = g.line_band.expect("the band's state");
            assert_eq!(band.primitive, primitive, "{what}");
            assert_eq!(
                output_vertices(&banded.stages[1].spirv),
                Some(primitive.patch_control_points()),
                "{what}"
            );
            let ranges = &banded.layout.push_constants;
            let own = ranges
                .iter()
                .find(|r| r.stages == vk::ShaderStageFlags::TESSELLATION_EVALUATION)
                .unwrap_or_else(|| panic!("{what}: the band's push range"));
            assert_eq!((own.offset, own.size), (band.push_offset, 16), "{what}");
            for r in ranges.iter().filter(|r| r.stages != own.stages) {
                assert!(
                    r.offset + r.size <= own.offset || own.offset + own.size <= r.offset,
                    "{what}: {r:?} overlaps the band's range"
                );
            }
            assert!(
                banded
                    .requirements
                    .features
                    .contains(&F::TessellationShader),
                "{what}"
            );
            assert_ne!(banded.key, plain.key, "{what}: the variant's own key");
            // The binder plans it for a shape that draws bands: its synthesized stages run no
            // Metal function.
            let state = BindState {
                line_band: Some((primitive, [8.0, -8.0, 1.0, 1.0])),
                ..BindState::default()
            };
            banded
                .plan(&dev, &state.shape())
                .unwrap_or_else(|e| panic!("{what}: the band's plan: {e:?}"));
            drawn[k] += 1;
        }
    }
    assert!(
        drawn.iter().all(|&n| n > 0),
        "a primitive no profile drew: {drawn:?}"
    );
}

/// A flat varying, or an integer one, is the first control point's at every corner; a smooth one
/// is its endpoint's.
#[test]
fn the_band_takes_flat_varyings_from_the_first_control_point() {
    let Some(bc) = assemble("rt6_provoking.ll") else {
        return;
    };
    let (_, profile) = committed_profiles()
        .expect("profiles")
        .into_iter()
        .find(|(stem, _)| stem == "moltenvk-m3")
        .expect("moltenvk-m3");
    let vertex = compile(&bc, "pv_vertex", &profile).expect("vertex");
    let fragment = compile(&bc, "pv_flat", &profile).expect("fragment");
    for primitive in PRIMITIVES {
        let plan = band::plan(
            &[&vertex.interface, &fragment.interface],
            (primitive, false),
            1,
            &profile,
        )
        .expect("plan");
        let flats: Vec<(u8, bool)> = plan
            .varyings
            .iter()
            .map(|v| (v.ty.components, v.flat))
            .collect();
        // The smooth `float` vertex ID, then the flat `uint` one.
        assert_eq!(flats, [(1, false), (1, true)], "{:?}", plan.varyings);
        assert_eq!((plan.primitive, plan.clip_distances), (primitive, 0));
    }
}

/// Without `tessellationShader` a band is refused needing it, and a fragment function that reads
/// `[[barycentric_coord]]` is not implemented.
#[test]
fn a_band_needs_tessellation_and_no_barycentric_weights() {
    let Some(bc) = assemble("rt6_provoking.ll") else {
        return;
    };
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/moltenvk-m3.json");
    let text = std::fs::read_to_string(path).expect("profile");
    let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    let budget = Budget::standard();
    let both = render(&bc, "pv_vertex", "pv_both", PixelFormat::RGBA8Unorm);
    let dev = Device::new(desc.clone(), DevicePolicy::default()).expect("device");
    for primitive in PRIMITIVES {
        match dev.compile_render_variant(&both, &band_key(primitive), &budget) {
            Err(Error::Unsupported(u)) => {
                assert_eq!(u.feature, Feature::WideLines);
                assert_eq!(u.needs, Needs::NotImplemented);
            }
            other => panic!("{primitive:?}: {:?}", other.map(|p| p.key)),
        }
    }
    desc.features.remove(&F::TessellationShader);
    let dev = Device::new(desc, DevicePolicy::default()).expect("device");
    let flat = render(&bc, "pv_vertex", "pv_flat", PixelFormat::RGBA8Unorm);
    for primitive in PRIMITIVES {
        match dev.compile_render_variant(&flat, &band_key(primitive), &budget) {
            Err(Error::Unsupported(u)) => {
                assert_eq!(u.feature, Feature::WideLines);
                assert_eq!(
                    u.needs,
                    Needs::AnyOf(vec![vec![Capability::Feature(F::TessellationShader)]])
                );
            }
            other => panic!("{primitive:?}: {:?}", other.map(|p| p.key)),
        }
    }
}

/// A per-vertex value a stage declares: its builtin or its location.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Slot {
    BuiltIn(u32),
    Location(u32),
}

/// The per-vertex inputs (`Input`) or outputs (`Output`) module `words` declares: every
/// location, and the builtins `Position`, `PointSize`, `ClipDistance` and `CullDistance`.
fn per_vertex(words: &[u32], class: crate::spv::StorageClass) -> Vec<Slot> {
    use crate::spirv_binary::{parse_bytes, Operand};
    use crate::spv::{BuiltIn as B, Decoration as D, Op as O};
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let parsed = parse_bytes(&bytes).expect("parse");
    let declared: Vec<u32> = (parsed.instructions.iter())
        .filter(|i| {
            i.opcode == O::Variable && i.operands.first() == Some(&Operand::Word(class as u32))
        })
        .filter_map(|i| i.result_id)
        .collect();
    let per_vertex =
        [B::Position, B::PointSize, B::ClipDistance, B::CullDistance].map(|b| b as u32);
    let mut slots: Vec<Slot> = (parsed.instructions.iter())
        .filter(|i| i.opcode == O::Decorate)
        .filter_map(|i| match i.operands.as_slice() {
            [Operand::IdRef(id), Operand::Word(d), Operand::Word(v), ..]
                if declared.contains(id) =>
            {
                match *d {
                    d if d == D::BuiltIn as u32 && per_vertex.contains(v) => {
                        Some(Slot::BuiltIn(*v))
                    }
                    d if d == D::Location as u32 => Some(Slot::Location(*v)),
                    _ => None,
                }
            }
            _ => None,
        })
        .collect();
    slots.sort();
    slots
}

/// The control stage's per-vertex inputs are the vertex stage's outputs entire, its point size
/// and every varying included whether or not the fragment stage reads them, and the evaluation
/// stage's inputs are the control stage's outputs: an implementation that lays each stage's
/// interface out as one record (MoltenVK, under tessellation) reads the second control point
/// from the wrong place otherwise.
#[test]
fn the_control_stage_takes_every_value_the_vertex_stage_writes() {
    use crate::spv::StorageClass as C;
    let Some(provoking) = assemble("rt6_provoking.ll") else {
        return;
    };
    let Some(front) = assemble("a2_front_facing.ll") else {
        return;
    };
    let lib = Library::from_air(&provoking).expect("library");
    let front = Library::from_air(&front).expect("library");
    let stage = |lib: &Library, name: &str| {
        StageSource::Function(FunctionDescriptor::new(
            lib.function(name).expect("function"),
        ))
    };
    let mut unread = RenderPipelineDescriptor::new(stage(&lib, "pv_vertex"));
    unread.fragment = Some(stage(&front, "a2_front_facing"));
    unread.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    let pipelines = [
        ("unread varyings", unread),
        (
            "read varyings",
            render(&provoking, "pv_vertex", "pv_flat", PixelFormat::RGBA8Unorm),
        ),
    ];
    let budget = Budget::standard();
    let mut checked = 0;
    let mut point_sizes = 0;
    for (stem, dev) in committed_devices() {
        if !(dev.profile().enabled().features).contains(&F::TessellationShader) {
            continue;
        }
        for ((what, d), primitive) in pipelines
            .iter()
            .flat_map(|p| std::iter::repeat(p).zip(PRIMITIVES))
        {
            let what = format!("{stem} {what} {primitive:?}");
            let p = dev
                .compile_render_variant(d, &band_key(primitive), &budget)
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            let [vertex, control, evaluation, _] = &p.stages[..] else {
                panic!(
                    "{what}: {:?}",
                    p.stages.iter().map(|m| m.stage).collect::<Vec<_>>()
                );
            };
            // A device without `shaderTessellationAndGeometryPointSize` takes no point size in
            // its tessellation stages.
            let point_size = Slot::BuiltIn(crate::spv::BuiltIn::PointSize as u32);
            let carries_size = (dev.profile().enabled().features)
                .contains(&F::ShaderTessellationAndGeometryPointSize);
            let mut written = per_vertex(&vertex.spirv, C::Output);
            written.retain(|&s| carries_size || s != point_size);
            assert!(written.len() > 2, "{what}: {written:?}");
            assert_eq!(per_vertex(&control.spirv, C::Input), written, "{what}");
            assert_eq!(per_vertex(&control.spirv, C::Output), written, "{what}");
            assert_eq!(per_vertex(&evaluation.spirv, C::Input), written, "{what}");
            point_sizes += usize::from(written.contains(&point_size));
            checked += 1;
        }
    }
    assert!(checked > 0 && point_sizes > 0, "{checked} {point_sizes}");
}
