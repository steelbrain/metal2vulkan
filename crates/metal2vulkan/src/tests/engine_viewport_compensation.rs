//! A vertex function compiled for a viewport Vulkan's cannot be
//! (`LoweringInputs::viewport_compensation`), over the owned fixtures `engine/rfb_viewports`
//! (Metal's fullscreen triangle), `engine/stage_io` (a bare position return, and a position
//! beside two clip distances of the function's own) and `engine/vertex_amplification` (a
//! post-tessellation vertex function):
//!
//! - Compensated, the function states `RuntimeFact::ViewportCompensation` and reads
//!   `PushField::ViewportCompensation` in its push window. Under the depth's compensation it
//!   writes two `ClipDistance` elements after its own: two where it has none, four beside its
//!   two. Under Metal's guard band for points it reads `PushField::PointGuard` too and writes four
//!   more after those.
//! - Every module passes L3 (`spirv-val` and requirements within the profile), and the words the
//!   compensation adds are bounded: measured, 73 to 91 for `x` alone and 179 to 211 with the
//!   depth (the push block's declaration among them where the function had none). The common
//!   path, compiled without it, is unchanged.
//! - The depth's compensation needs `shaderClipDistance` and two clip distances beyond the
//!   function's own within `maxClipDistances` and `maxCombinedClipAndCullDistances`, and the
//!   point guard band four, refused typed otherwise; the negation of `x` alone needs neither.
//! - A render plan whose shape asks for a compensation states its variant, which the pipeline
//!   compiled without it does not draw (`BindError::NeedsVariant`); compiled, the variant's
//!   modules pass L3, it states the fact, its plan settles on itself, and the draw's push data
//!   holds the state's four values at the vertex function's field, and the guard's two at its
//!   own.
//! - Under the mirrors of viewports in one draw the function reads `PushField::ViewportMirrors`
//!   (one `u32`) too, through the viewport index it stores (`engine/rt6_viewports`'
//!   `rt6_select`) or the first where it stores none, and the draw's push data holds the state's
//!   bits there.
//! - Under Metal's clamp of each vertex's depth a compensated depth reads
//!   `PushField::VertexDepthClamp` (two `f32`s) too, and the draw's push data holds the state's
//!   bounds there; without the depth's compensation nothing is clamped and nothing read.
//!
//! `llvm-as` assembles the fixtures; a missing tool skips visibly. No GPU is used.

use std::path::Path;

use crate::bind::{BindState, CompensatedViewport, DrawCall};
use crate::budget::Budget;
use crate::device::requirements::{VkFeature as F, VkLimit};
use crate::device::{Device, DeviceDescription, DevicePolicy, DeviceProfile};
use crate::error::{BindError, Capability, Error, Feature, Unsupported};
use crate::library::{FunctionArtifact, Library};
use crate::pipeline::descriptor::{FunctionDescriptor, RenderPipelineDescriptor, StageSource};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::link::{link, LinkInputs};
use crate::pipeline::lower_inputs::{LoweringInputs, ViewportCompensation};
use crate::pipeline::relocate::relocate;
use crate::pipeline::stage_io::StageBuiltin;
use crate::pipeline::sym::{PushField, Stage};
use crate::pipeline::testing::{compile_with, engine_dir, llvm_assemble, AssembleError};
use crate::spirv::testing::check_words_valid;

/// The bitcode of the owned fixture `stem` (`engine/<stem>.ll`), or `None` (printed) where
/// `llvm-as` is missing.
fn fixture(stem: &str) -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join(format!("{stem}.ll"))) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {stem}.ll: {why}"),
    }
}

fn description(stem: &str) -> DeviceDescription {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    DeviceDescription::from_khronos_json(&text, None).expect("description")
}

fn profile_of(desc: DeviceDescription) -> DeviceProfile {
    Device::from_queries(desc, DevicePolicy::default())
        .expect("device")
        .profile()
        .clone()
}

/// The lowering inputs of `compensation`.
fn inputs(compensation: Option<ViewportCompensation>) -> LoweringInputs {
    let mut inputs = LoweringInputs::new();
    inputs.viewport_compensation = compensation;
    inputs
}

/// Link `a` alone, relocate it, and run L3; its relocated words.
fn validate(a: &FunctionArtifact, p: &DeviceProfile) -> Vec<u32> {
    let linked = link(&[a], &LinkInputs::standalone(), p).unwrap_or_else(|e| panic!("link: {e}"));
    let words = relocate(a, &linked.table).expect("relocate");
    check_words_valid(&words, &a.requirements, p).unwrap_or_else(|e| panic!("L3: {e}"));
    words
}

/// The clip distances `a`'s stage interface declares.
fn clip_distances(a: &FunctionArtifact) -> u32 {
    a.interface
        .io
        .builtins
        .iter()
        .find_map(|b| match b {
            StageBuiltin::ClipDistance { count } => Some(*count),
            _ => None,
        })
        .unwrap_or(0)
}

/// `e` without the location an `Unsupported` carries.
fn unlocated(e: Error) -> Error {
    match e {
        Error::Unsupported(u) => Error::Unsupported(Unsupported {
            location: None,
            ..u
        }),
        other => other,
    }
}

/// Each entry compiled without the compensation, with `x`'s alone and with the depth's: the
/// fact, the push field in the vertex window, the clip distances, L3, and the words each adds
/// (printed; bounded).
#[test]
fn a_compensated_vertex_function_states_its_compensation() {
    let entries = [
        ("rfb_viewports", "rfb_full", 0),
        ("stage_io", "stage_io_vertex_position", 0),
        ("stage_io", "stage_io_vertex_clip", 2),
        ("vertex_amplification", "amp_post_tessellation", 0),
    ];
    let mut over = Vec::new();
    for stem in ["lavapipe", "moltenvk-m3", "intel-irisxe-rplp-anv"] {
        let p = profile_of(description(stem));
        for (fixture_stem, entry, own) in entries {
            let Some(bc) = fixture(fixture_stem) else {
                return;
            };
            let plain = compile_with(&bc, entry, &p, &inputs(None))
                .unwrap_or_else(|e| panic!("{stem} {entry}: {e}"));
            let plain_words = validate(&plain, &p).len();
            assert!(
                !plain
                    .facts
                    .iter()
                    .any(|f| matches!(f, RuntimeFact::ViewportCompensation { .. })),
                "{stem} {entry}"
            );
            assert_eq!(clip_distances(&plain), own, "{stem} {entry}");
            for (depth, point_guard, bound) in [
                (false, false, 120),
                (true, false, 240),
                (false, true, 240),
                (true, true, 360),
            ] {
                let compensation = ViewportCompensation::new(depth).with_point_guard(point_guard);
                let a = compile_with(&bc, entry, &p, &inputs(Some(compensation)))
                    .unwrap_or_else(|e| panic!("{stem} {entry} {depth} {point_guard}: {e}"));
                let words = validate(&a, &p).len();
                assert!(
                    a.facts.contains(&RuntimeFact::ViewportCompensation {
                        depth,
                        point_guard,
                        mirrors: false,
                        vertex_clamp: false
                    }),
                    "{stem} {entry} {depth}: {:?}",
                    a.facts
                );
                let guard = a
                    .interface
                    .push
                    .fields
                    .iter()
                    .find(|f| f.field == PushField::PointGuard);
                match point_guard {
                    true => assert!(
                        guard.is_some_and(|f| matches!(f.loc, PushLoc::Push { .. }) && f.size == 8),
                        "{stem} {entry} {depth}: {guard:?}"
                    ),
                    false => assert_eq!(guard, None, "{stem} {entry} {depth}"),
                }
                let field = a
                    .interface
                    .push
                    .fields
                    .iter()
                    .find(|f| f.field == PushField::ViewportCompensation)
                    .unwrap_or_else(|| panic!("{stem} {entry} {depth}: no compensation field"));
                assert!(
                    matches!(field.loc, PushLoc::Push { .. }) && field.size == 16,
                    "{stem} {entry} {depth}: {field:?}"
                );
                let expected = own + 2 * u32::from(depth) + 4 * u32::from(point_guard);
                assert_eq!(clip_distances(&a), expected, "{stem} {entry} {depth}");
                let added = words as i64 - plain_words as i64;
                println!(
                    "{stem} {entry}: depth {depth} point guard {point_guard} adds {added} words \
                     to {plain_words}"
                );
                if !(0..=bound).contains(&added) {
                    over.push(format!(
                        "{stem} {entry} {depth} {point_guard}: {added} words past {bound}"
                    ));
                }
            }
        }
    }
    assert!(over.is_empty(), "{over:#?}");
}

/// The depth's compensation needs `shaderClipDistance`, and its two clip distances beyond the
/// function's own within the device's limits, and the point guard band its four; the negation
/// of `x` alone needs neither.
#[test]
fn the_depth_compensation_needs_its_clip_distances() {
    let Some(stage_io) = fixture("stage_io") else {
        return;
    };
    let Some(viewports) = fixture("rfb_viewports") else {
        return;
    };
    let mut desc = description("moltenvk-m3");
    assert!(desc.features.remove(&F::ShaderClipDistance));
    let without = profile_of(desc);
    let x = inputs(Some(ViewportCompensation::new(false)));
    let depth = inputs(Some(ViewportCompensation::new(true)));
    let a = compile_with(&viewports, "rfb_full", &without, &x).expect("x alone");
    validate(&a, &without);
    assert_eq!(clip_distances(&a), 0);
    let refusal = compile_with(&viewports, "rfb_full", &without, &depth)
        .map(|_| ())
        .map_err(unlocated);
    assert_eq!(
        refusal,
        Err(Error::Unsupported(Unsupported::needs(
            Feature::ClipDistance,
            Capability::Feature(F::ShaderClipDistance)
        )))
    );
    // Three clip distances: the function's two and the compensation's two do not fit.
    let mut desc = description("moltenvk-m3");
    desc.limits.max_clip_distances = 3;
    let three = profile_of(desc);
    compile_with(&stage_io, "stage_io_vertex_clip", &three, &x).expect("x alone");
    let refusal = compile_with(&stage_io, "stage_io_vertex_clip", &three, &depth)
        .map(|_| ())
        .map_err(unlocated);
    assert_eq!(
        refusal,
        Err(Error::Unsupported(Unsupported::needs(
            Feature::ClipDistance,
            Capability::Limit {
                limit: VkLimit::MaxClipDistances,
                required: 4,
                available: 3,
            }
        )))
    );
    compile_with(&viewports, "rfb_full", &three, &depth).expect("two clip distances");
    let points = inputs(Some(
        ViewportCompensation::new(false).with_point_guard(true),
    ));
    let refusal = compile_with(&viewports, "rfb_full", &without, &points)
        .map(|_| ())
        .map_err(unlocated);
    assert_eq!(
        refusal,
        Err(Error::Unsupported(Unsupported::needs(
            Feature::ClipDistance,
            Capability::Feature(F::ShaderClipDistance)
        )))
    );
    let refusal = compile_with(&viewports, "rfb_full", &three, &points)
        .map(|_| ())
        .map_err(unlocated);
    assert_eq!(
        refusal,
        Err(Error::Unsupported(Unsupported::needs(
            Feature::ClipDistance,
            Capability::Limit {
                limit: VkLimit::MaxClipDistances,
                required: 4,
                available: 3,
            }
        )))
    );
}

/// A compensation of mirrored and unmirrored viewports in one draw: the function states it,
/// reads `PushField::ViewportMirrors` (one `u32` in its push window), passes L3, and adds a
/// bounded handful of words to the same compensation without it (printed): through the
/// viewport index it stores (`rt6_select`), and through the first where it stores none.
#[test]
fn a_compensation_of_mirrors_reads_the_selected_viewports_bit() {
    let entries = [
        ("rt6_viewports", "rt6_select"),
        ("rfb_viewports", "rfb_full"),
        ("vertex_amplification", "amp_post_tessellation"),
    ];
    for stem in ["lavapipe", "moltenvk-m3", "intel-irisxe-rplp-anv"] {
        let p = profile_of(description(stem));
        for (fixture_stem, entry) in entries {
            let Some(bc) = fixture(fixture_stem) else {
                return;
            };
            for depth in [false, true] {
                let what = format!("{stem} {entry} {depth}");
                let one = ViewportCompensation::new(depth);
                let without = compile_with(&bc, entry, &p, &inputs(Some(one)))
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                let without = validate(&without, &p).len();
                let a = compile_with(&bc, entry, &p, &inputs(Some(one.with_mirrors(true))))
                    .unwrap_or_else(|e| panic!("{what} mirrors: {e}"));
                let words = validate(&a, &p).len();
                assert!(
                    a.facts.contains(&RuntimeFact::ViewportCompensation {
                        depth,
                        point_guard: false,
                        mirrors: true,
                        vertex_clamp: false
                    }),
                    "{what}: {:?}",
                    a.facts
                );
                let field = a
                    .interface
                    .push
                    .fields
                    .iter()
                    .find(|f| f.field == PushField::ViewportMirrors)
                    .unwrap_or_else(|| panic!("{what}: no mirrors field"));
                assert!(
                    matches!(field.loc, PushLoc::Push { .. }) && field.size == 4,
                    "{what}: {field:?}"
                );
                let added = words as i64 - without as i64;
                println!("{what}: the mirrors add {added} words to {without}");
                assert!((0..=160).contains(&added), "{what}: {added} words");
            }
        }
    }
}

/// A compensation that clamps each vertex's depth as Metal does: under the depth's compensation
/// the function states it, reads `PushField::VertexDepthClamp` (two `f32`s' bits in its push
/// window), passes L3, and adds at most 120 words to the same compensation without it (printed;
/// measured 82 for `rfb_span` on lavapipe); without the depth's compensation there is no depth
/// to clamp, and no field.
#[test]
fn a_compensation_clamping_each_vertexs_depth_reads_its_bounds() {
    let entries = [
        ("rfb_viewports", "rfb_span"),
        ("rfb_viewports", "rfb_full"),
        ("stage_io", "stage_io_vertex_clip"),
        ("vertex_amplification", "amp_post_tessellation"),
    ];
    for stem in ["lavapipe", "moltenvk-m3", "intel-irisxe-rplp-anv"] {
        let p = profile_of(description(stem));
        for (fixture_stem, entry) in entries {
            let Some(bc) = fixture(fixture_stem) else {
                return;
            };
            for depth in [false, true] {
                let what = format!("{stem} {entry} {depth}");
                let one = ViewportCompensation::new(depth);
                let without = compile_with(&bc, entry, &p, &inputs(Some(one)))
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                let without = validate(&without, &p).len();
                let a = compile_with(&bc, entry, &p, &inputs(Some(one.with_vertex_clamp(true))))
                    .unwrap_or_else(|e| panic!("{what} vertex clamp: {e}"));
                let words = validate(&a, &p).len();
                assert!(
                    a.facts.contains(&RuntimeFact::ViewportCompensation {
                        depth,
                        point_guard: false,
                        mirrors: false,
                        vertex_clamp: true
                    }),
                    "{what}: {:?}",
                    a.facts
                );
                let field = a
                    .interface
                    .push
                    .fields
                    .iter()
                    .find(|f| f.field == PushField::VertexDepthClamp);
                let added = words as i64 - without as i64;
                println!("{what}: the vertex clamp adds {added} words to {without}");
                match depth {
                    true => {
                        let field = field.unwrap_or_else(|| panic!("{what}: no clamp field"));
                        assert!(
                            matches!(field.loc, PushLoc::Push { .. }) && field.size == 8,
                            "{what}: {field:?}"
                        );
                        assert!((0..=120).contains(&added), "{what}: {added} words");
                    }
                    false => {
                        assert!(field.is_none(), "{what}: {field:?}");
                        assert_eq!(added, 0, "{what}");
                    }
                }
            }
        }
    }
}

/// A plan of the bare position return drawn through a compensated viewport: the variant, which
/// the base pipeline does not draw; compiled, its modules, fact and settled plan, and the four
/// values in the draw's push data at the vertex function's field.
#[test]
fn a_compensated_draw_runs_its_variant_with_the_states_values() {
    let Some(bc) = fixture("stage_io") else {
        return;
    };
    let budget = Budget::standard();
    let lib = Library::from_air(&bc).expect("library");
    let d = RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(
        lib.function("stage_io_vertex_position").expect("function"),
    )));
    let triangle = DrawCall::Primitives {
        vertex_start: 0,
        vertex_count: 3,
        instance_count: 1,
        base_instance: 0,
    };
    for stem in ["moltenvk-m3", "lavapipe"] {
        let dev = Device::from_queries(description(stem), DevicePolicy::default()).expect("device");
        let base = dev
            .compile_render_pipeline(&d, &budget)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        for (depth, point_guard, mirrors, vertex_clamp) in [
            (false, false, false, false),
            (true, false, false, false),
            (true, true, false, false),
            (false, false, true, false),
            (true, true, true, false),
            (true, false, false, true),
            (true, false, true, true),
        ] {
            let what = format!("{stem} {depth} {point_guard} {mirrors} {vertex_clamp}");
            let compensation = ViewportCompensation::new(depth)
                .with_point_guard(point_guard)
                .with_mirrors(mirrors)
                .with_vertex_clamp(vertex_clamp);
            let values = [-1.0f32, 0.5, 1.5, 1.0];
            let guard = [0.25f32, 0.5];
            let bits = 0b101u32;
            let clamp = [-0.5f32, 1.25];
            let state = BindState {
                viewport_compensation: Some(
                    CompensatedViewport::new(compensation, values)
                        .with_guard(guard)
                        .with_mirrors(bits)
                        .with_vertex_clamp(clamp),
                ),
                ..Default::default()
            };
            let plan = base
                .plan(&dev, &state.shape())
                .unwrap_or_else(|e| panic!("{what}: {e:?}"));
            assert_eq!(
                plan.variant.viewport_compensation,
                Some(compensation),
                "{what}"
            );
            assert_eq!(
                plan.instantiate_draw(&state, &triangle),
                Err(BindError::NeedsVariant {
                    variant: Box::new(plan.variant.clone())
                }),
                "{what}"
            );
            let key = plan.variant.clone();
            let p = dev
                .compile_render_variant(&d, &key, &budget)
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            for m in &p.stages {
                check_words_valid(&m.spirv, &p.requirements, dev.profile())
                    .unwrap_or_else(|e| panic!("{what} {:?}: L3: {e}", m.stage));
            }
            assert!(
                p.dependencies.iter().any(|x| x.fact
                    == RuntimeFact::ViewportCompensation {
                        depth,
                        point_guard,
                        mirrors,
                        vertex_clamp
                    }),
                "{what}"
            );
            let plan = p
                .plan(&dev, &state.shape())
                .unwrap_or_else(|e| panic!("{what}: {e:?}"));
            assert_eq!(plan.variant, key, "{what}");
            let commands = plan
                .instantiate_draw(&state, &triangle)
                .unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let offset_of = |field: PushField| {
                p.interface
                    .push
                    .iter()
                    .filter(|pl| pl.stage == Stage::Vertex)
                    .flat_map(|pl| pl.fields.iter())
                    .find(|f| f.field == field)
                    .map(|f| match f.loc {
                        PushLoc::Push { offset } => offset as usize,
                        other => panic!("{what}: {other:?}"),
                    })
            };
            let mut pushed = vec![None; 256];
            for w in &commands.push_constants {
                for (i, b) in w.bytes.iter().enumerate() {
                    if let Some(slot) = pushed.get_mut(w.offset as usize + i) {
                        *slot = Some(*b);
                    }
                }
            }
            let bytes = |v: &[f32]| -> Vec<Option<u8>> {
                v.iter().flat_map(|v| v.to_le_bytes()).map(Some).collect()
            };
            let offset = offset_of(PushField::ViewportCompensation)
                .unwrap_or_else(|| panic!("{what}: no compensation field"));
            assert_eq!(
                pushed.get(offset..offset + 16),
                Some(&bytes(&values)[..]),
                "{what}"
            );
            match offset_of(PushField::PointGuard) {
                Some(offset) => {
                    assert!(point_guard, "{what}");
                    assert_eq!(
                        pushed.get(offset..offset + 8),
                        Some(&bytes(&guard)[..]),
                        "{what}"
                    );
                }
                None => assert!(!point_guard, "{what}"),
            }
            match offset_of(PushField::ViewportMirrors) {
                Some(offset) => {
                    assert!(mirrors, "{what}");
                    let expected: Vec<Option<u8>> =
                        bits.to_le_bytes().iter().copied().map(Some).collect();
                    assert_eq!(
                        pushed.get(offset..offset + 4),
                        Some(&expected[..]),
                        "{what}"
                    );
                }
                None => assert!(!mirrors, "{what}"),
            }
            match offset_of(PushField::VertexDepthClamp) {
                Some(offset) => {
                    assert!(vertex_clamp, "{what}");
                    assert_eq!(
                        pushed.get(offset..offset + 8),
                        Some(&bytes(&clamp)[..]),
                        "{what}"
                    );
                }
                None => assert!(!vertex_clamp, "{what}"),
            }
            // Without the compensation the variant asks for the base again.
            let plan = p
                .plan(&dev, &BindState::default().shape())
                .unwrap_or_else(|e| panic!("{what}: {e:?}"));
            assert_eq!(plan.variant.viewport_compensation, None, "{what}");
            assert!(
                matches!(
                    plan.instantiate_draw(&BindState::default(), &triangle),
                    Err(BindError::NeedsVariant { .. })
                ),
                "{what}"
            );
        }
    }
}
