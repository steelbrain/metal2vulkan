//! A fragment function's barycentric weights read back from a triangle drawn with Metal's last
//! vertex first (`LoweringInputs::rotated_weights`), over the owned fixtures
//! `engine/rt6_provoking` (`pv_both` reads a `float3` of the weights and a flat input, `pv_flat`
//! the flat input alone) and `engine/stage_io_tail` (`tail_fragment_barycentric_scalar` reads a
//! `float`):
//!
//! - Under the input a function that reads the weights states `RuntimeFact::RotatedWeights` and
//!   computes Metal's weights from the drawn triangle's first two lanes `(x, y)`: `y`, and
//!   `1 - (x + y)` once more than without it, each operation `NoContraction`; a `float` read
//!   reads lane 1 where it read lane 0.
//! - A function that reads no weights is the same module under it, and states nothing.
//! - A draw whose shape draws its triangles rotated (`BindState::rotated_weights`) runs the
//!   variant where its pipeline reads the weights (`BindError::NeedsVariant` from the base),
//!   whose plan states it; a pipeline that reads none draws as it is.
//!
//! What the weights hold is a Metal probe's (the M3 Max, macOS 27.0.1: the first weight bit for
//! bit, the others within `2^-15`). Every module passes L3. `llvm-as` assembles the fixtures; a
//! missing tool skips visibly. No GPU is used.

use std::path::Path;

use crate::device::{Device, DeviceDescription, DevicePolicy, DeviceProfile};
use crate::library::FunctionArtifact;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::link::{link, LinkInputs};
use crate::pipeline::lower_inputs::LoweringInputs;
use crate::pipeline::relocate::relocate;
use crate::pipeline::testing::{compile_with, engine_dir, llvm_assemble, AssembleError};
use crate::spirv::testing::check_words_valid;

/// The bitcode of `engine/<stem>.ll`, or `None` (printed) where `llvm-as` is missing.
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

/// The profiles the tests compile for: with `fragmentShaderBarycentric` (`BaryCoordKHR`).
const PROFILES: [&str; 3] = ["moltenvk-m3", "amd-780m-radv", "nvidia-rtx50-linux"];

/// Link `a` alone, relocate it, and run L3; its relocated words.
fn validate(a: &FunctionArtifact, p: &DeviceProfile) -> Vec<u32> {
    let linked = link(&[a], &LinkInputs::standalone(), p).unwrap_or_else(|e| panic!("link: {e}"));
    let words = relocate(a, &linked.table).expect("relocate");
    check_words_valid(&words, &a.requirements, p).unwrap_or_else(|e| panic!("L3: {e}"));
    words
}

/// The instructions of `words` (a SPIR-V module), each as its words.
fn instructions(words: &[u32]) -> Vec<&[u32]> {
    let mut out = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let count = (words[i] >> 16) as usize;
        assert!(count > 0, "a zero-length instruction at word {i}");
        out.push(&words[i..i + count]);
        i += count;
    }
    out
}

/// `OpFSub`.
const F_SUB: u32 = 131;
/// `OpCompositeExtract`.
const EXTRACT: u32 = 81;
/// `OpConstant`.
const CONSTANT: u32 = 43;
/// `OpDecorate`.
const DECORATE: u32 = 71;
/// The `NoContraction` decoration.
const NO_CONTRACTION: u32 = 42;

/// How many `1 - v` subtractions `words` holds that are decorated `NoContraction`: Metal's third
/// weight's, and the rotation's.
fn ones_less(words: &[u32]) -> usize {
    let insts = instructions(words);
    let ones: Vec<u32> = insts
        .iter()
        .filter(|i| i[0] & 0xffff == CONSTANT && i.len() == 4 && i[3] == 1.0f32.to_bits())
        .map(|i| i[2])
        .collect();
    let exact: Vec<u32> = insts
        .iter()
        .filter(|i| i[0] & 0xffff == DECORATE && i.get(2) == Some(&NO_CONTRACTION))
        .map(|i| i[1])
        .collect();
    insts
        .iter()
        .filter(|i| i[0] & 0xffff == F_SUB && ones.contains(&i[3]) && exact.contains(&i[2]))
        .count()
}

/// The lanes `words` extracts from a single-index `OpCompositeExtract`, in order.
fn extracted_lanes(words: &[u32]) -> Vec<u32> {
    instructions(words)
        .iter()
        .filter(|i| i[0] & 0xffff == EXTRACT && i.len() == 5)
        .map(|i| i[4])
        .collect()
}

/// Under `rotated_weights`, a function that reads the weights states the fact and computes the
/// rotation's `1 - (x + y)` besides Metal's third weight's; a `float` read takes lane 1; a
/// function that reads none is the same module.
#[test]
fn rotated_weights_take_metal_s_weights_from_the_drawn_triangle_s() {
    let (Some(provoking), Some(tail)) = (fixture("rt6_provoking"), fixture("stage_io_tail")) else {
        return;
    };
    let plain = LoweringInputs::new();
    let mut rotated = LoweringInputs::new();
    rotated.rotated_weights = true;
    for stem in PROFILES {
        let p = crate::pipeline::testing::profile(stem, DevicePolicy::RESIDENT_TABLES).expect(stem);
        let compile = |bc: &[u8], entry: &str, inputs: &LoweringInputs| {
            compile_with(bc, entry, &p, inputs).unwrap_or_else(|e| panic!("{stem} {entry}: {e}"))
        };
        // The flat input alone: nothing to rotate.
        let with = compile(&provoking, "pv_flat", &rotated);
        let without = compile(&provoking, "pv_flat", &plain);
        assert!(!with.facts.contains(&RuntimeFact::RotatedWeights), "{stem}");
        assert_eq!(
            validate(&with, &p),
            validate(&without, &p),
            "{stem} pv_flat"
        );
        // A `float3` read: one rotation per read of its lanes, besides the third weight.
        let with = compile(&provoking, "pv_both", &rotated);
        let without = compile(&provoking, "pv_both", &plain);
        assert!(
            with.facts.contains(&RuntimeFact::RotatedWeights),
            "{stem}: {:?}",
            with.facts
        );
        assert!(
            !without.facts.contains(&RuntimeFact::RotatedWeights),
            "{stem}"
        );
        let (a, b) = (validate(&with, &p), validate(&without, &p));
        assert!(ones_less(&b) > 0, "{stem}: the third weight");
        assert_eq!(ones_less(&a), 2 * ones_less(&b), "{stem}: pv_both");
        // A `float` read: lane 1 of the drawn weights, and the rotation's sum it does not read.
        let with = compile(&tail, "tail_fragment_barycentric_scalar", &rotated);
        let without = compile(&tail, "tail_fragment_barycentric_scalar", &plain);
        assert!(with.facts.contains(&RuntimeFact::RotatedWeights), "{stem}");
        let (a, b) = (validate(&with, &p), validate(&without, &p));
        assert_eq!(
            ones_less(&b),
            0,
            "{stem}: a float read computes no third weight"
        );
        assert!(
            extracted_lanes(&b).contains(&0),
            "{stem}: {:?}",
            extracted_lanes(&b)
        );
        assert!(
            !extracted_lanes(&b).contains(&1),
            "{stem}: {:?}",
            extracted_lanes(&b)
        );
        assert!(
            extracted_lanes(&a).contains(&1),
            "{stem}: {:?}",
            extracted_lanes(&a)
        );
    }
}

/// A draw that draws its triangles rotated runs the variant of a pipeline that reads the weights,
/// whose plan states it and draws; one that reads none draws as it is.
#[test]
fn a_rotated_draw_runs_the_rotated_weights_variant() {
    use crate::bind::{BindState, DrawCall};
    use crate::budget::Budget;
    use crate::error::BindError;
    use crate::library::Library;
    use crate::metal::PixelFormat;
    use crate::pipeline::descriptor::{
        ColorAttachmentDescriptor, FunctionDescriptor, RenderPipelineDescriptor, StageSource,
    };
    let Some(bc) = fixture("rt6_provoking") else {
        return;
    };
    let budget = Budget::standard();
    let lib = Library::from_air(&bc).expect("library");
    let function =
        |name| StageSource::Function(FunctionDescriptor::new(lib.function(name).expect(name)));
    let triangle = DrawCall::Primitives {
        vertex_start: 0,
        vertex_count: 3,
        instance_count: 1,
        base_instance: 0,
    };
    let state = BindState {
        rotated_weights: true,
        ..Default::default()
    };
    for stem in ["moltenvk-m3", "amd-780m-radv"] {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../metal2vulkan-validation/profiles")
            .join(format!("{stem}.json"));
        let text = std::fs::read_to_string(path).expect("profile");
        let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
        let dev = Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES).expect("device");
        for (fragment, reads) in [("pv_both", true), ("pv_flat", false)] {
            let what = format!("{stem} {fragment}");
            let mut d = RenderPipelineDescriptor::new(function("pv_vertex"));
            d.fragment = Some(function(fragment));
            for (i, format) in [
                PixelFormat::RGBA32Float,
                PixelFormat::RGBA32Float,
                PixelFormat::RGBA32Uint,
            ]
            .into_iter()
            .enumerate()
            {
                d.color_attachments[i] = Some(ColorAttachmentDescriptor::new(format));
            }
            let base = dev
                .compile_render_pipeline(&d, &budget)
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            assert_eq!(base.graphics.reads_barycentric_coord, reads, "{what}");
            let plan = base
                .plan(&dev, &state.shape())
                .unwrap_or_else(|e| panic!("{what}: {e:?}"));
            assert_eq!(plan.variant.rotated_weights, reads, "{what}");
            if !reads {
                plan.instantiate_draw(&state, &triangle)
                    .unwrap_or_else(|e| panic!("{what}: {e:?}"));
                continue;
            }
            assert_eq!(
                plan.instantiate_draw(&state, &triangle),
                Err(BindError::NeedsVariant {
                    variant: Box::new(plan.variant.clone())
                }),
                "{what}"
            );
            let key = plan.variant.clone();
            let v = dev
                .compile_render_variant(&d, &key, &budget)
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            for m in &v.stages {
                check_words_valid(&m.spirv, &v.requirements, dev.profile())
                    .unwrap_or_else(|e| panic!("{what} {:?}: L3: {e}", m.stage));
            }
            assert!(
                v.dependencies
                    .iter()
                    .any(|x| x.fact == RuntimeFact::RotatedWeights),
                "{what}"
            );
            let plan = v
                .plan(&dev, &state.shape())
                .unwrap_or_else(|e| panic!("{what}: {e:?}"));
            assert_eq!(plan.variant, key, "{what}");
            plan.instantiate_draw(&state, &triangle)
                .unwrap_or_else(|e| panic!("{what}: {e:?}"));
            // Drawn as it is, the variant asks for the base again.
            let plan = v
                .plan(&dev, &BindState::default().shape())
                .unwrap_or_else(|e| panic!("{what}: {e:?}"));
            assert!(!plan.variant.rotated_weights, "{what}");
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
