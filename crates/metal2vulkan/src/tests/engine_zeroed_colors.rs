//! A pass's color attachments its render pipeline lacks (`VariantKey::zeroed_colors`), over the
//! owned fixtures `engine/rt6_render_state` (Metal's fullscreen triangle and a white fragment
//! function writing color 0) and `engine/stage_io` (a fragment function writing colors 0 and
//! 2):
//!
//! - A render plan whose shape zeroes colors states the variant, which the pipeline compiled
//!   without it does not draw (`BindError::NeedsVariant`). Compiled, the variant's modules pass
//!   L3; it states `RuntimeFact::ZeroedColor` for each color, and its fragment function declares
//!   an output there of the format's numeric class (`float`, `uint`, `int`), four lanes; its
//!   pipeline has the attachment in that format, every channel written, unblended; and its plan
//!   settles on itself. Without the colors the variant asks for the base again.
//! - A function that writes a color it would zero stores the zero there in place of its own
//!   value, as plain Metal does (a Metal probe): its one output at that color is the zero's.
//! - A render plan whose shape drops colors (`VariantKey::dropped_colors`: the pipeline's colors
//!   the pass lacks) states the variant, which the base pipeline does not draw; compiled, it
//!   passes L3, states `RuntimeFact::DroppedColor` for each color, written or not, and its
//!   fragment function keeps no output there; its plan settles on itself.
//!
//! `llvm-as` assembles the fixtures; a missing tool skips visibly. No GPU is used.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use crate::bind::{BindState, DrawCall};
use crate::budget::Budget;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::BindError;
use crate::library::Library;
use crate::metal::PixelFormat;
use crate::pipeline::descriptor::{
    ColorAttachmentDescriptor, FunctionDescriptor, RenderPipelineDescriptor, StageSource,
};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::lower_inputs::LoweringInputs;
use crate::pipeline::stage_io::{IoNumeric, IoRole};
use crate::pipeline::sym::Stage;
use crate::pipeline::testing::{compile_with, engine_dir, llvm_assemble, AssembleError};
use crate::spirv::testing::check_words_valid;
use crate::vk;

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

/// A draw into a pass whose colors the pipeline (color 0, `RGBA8Unorm`) lacks: the variant, which
/// the base pipeline does not draw; compiled, its modules, facts, outputs, attachments and
/// settled plan; without the colors, the base again.
#[test]
fn a_draw_into_colors_the_pipeline_lacks_runs_the_variant_writing_zero() {
    let Some(bc) = fixture("rt6_render_state") else {
        return;
    };
    let budget = Budget::standard();
    let lib = Library::from_air(&bc).expect("library");
    let mut d = RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(
        lib.function("rt6_full").expect("vertex"),
    )));
    d.fragment = Some(StageSource::Function(FunctionDescriptor::new(
        lib.function("rt6_white").expect("fragment"),
    )));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    let triangle = DrawCall::Primitives {
        vertex_start: 0,
        vertex_count: 3,
        instance_count: 1,
        base_instance: 0,
    };
    let cases = [
        (
            BTreeMap::from([(1, PixelFormat::RGBA8Unorm)]),
            [(1, IoNumeric::Float)].as_slice(),
        ),
        (
            BTreeMap::from([(1, PixelFormat::R32Uint), (3, PixelFormat::RGBA32Sint)]),
            [(1, IoNumeric::Uint), (3, IoNumeric::Sint)].as_slice(),
        ),
    ];
    for stem in ["moltenvk-m3", "lavapipe"] {
        let dev = Device::new(description(stem), DevicePolicy::default()).expect("device");
        let base = dev
            .compile_render_pipeline(&d, &budget)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        for (zeroed, numerics) in &cases {
            let what = format!("{stem} {zeroed:?}");
            let state = BindState {
                zeroed_colors: zeroed.clone(),
                ..Default::default()
            };
            let plan = base
                .plan(&dev, &state.shape())
                .unwrap_or_else(|e| panic!("{what}: {e:?}"));
            assert_eq!(plan.variant.zeroed_colors, *zeroed, "{what}");
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
            for (&color, &format) in zeroed {
                assert!(
                    p.dependencies
                        .iter()
                        .any(|x| x.fact == RuntimeFact::ZeroedColor { color, format }),
                    "{what}: the fact of color {color}"
                );
                let state = p.graphics.color_attachments.get(color as usize);
                let state = state.and_then(Option::as_ref);
                let all = vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A;
                assert!(
                    state.is_some_and(|s| s.write_mask == all && s.blend.is_none()),
                    "{what}: the attachment of color {color}: {state:?}"
                );
            }
            // The fragment function compiled alone under the variant's inputs declares each
            // color's output of its format's numeric class.
            let fragment = compile_with(
                &bc,
                "rt6_white",
                dev.profile(),
                &key.lowering_inputs(Stage::Fragment),
            )
            .unwrap_or_else(|e| panic!("{what}: {e}"));
            for &(color, numeric) in numerics.iter() {
                let output = fragment.interface.io.outputs.iter().find(|o| {
                    o.role
                        == IoRole::Color {
                            attachment: color,
                            index: 0,
                        }
                });
                assert!(
                    output.is_some_and(|o| o.ty.numeric == numeric && o.ty.components == 4),
                    "{what}: the output of color {color}: {output:?}"
                );
            }
            let plan = p
                .plan(&dev, &state.shape())
                .unwrap_or_else(|e| panic!("{what}: {e:?}"));
            assert_eq!(plan.variant, key, "{what}");
            plan.instantiate_draw(&state, &triangle)
                .unwrap_or_else(|e| panic!("{what}: {e:?}"));
            // Without the colors the variant asks for the base again.
            let plan = p
                .plan(&dev, &BindState::default().shape())
                .unwrap_or_else(|e| panic!("{what}: {e:?}"));
            assert!(plan.variant.zeroed_colors.is_empty(), "{what}");
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

/// A fragment function writing a color it would zero (`stage_io_fragment` writes colors 0 and
/// 2, color 2 a `uint4`) stores the zero there in place of its own value, as plain Metal does: its
/// one output at color 2 is the zero's, of the zeroed format's numeric class (a `float` one too,
/// beside the `uint4` the function writes), and color 0 keeps its own output.
#[test]
fn a_function_writing_a_color_it_would_zero_stores_the_zero_there() {
    let Some(bc) = fixture("stage_io") else {
        return;
    };
    let dev = Device::new(description("lavapipe"), DevicePolicy::default()).expect("device");
    let p = dev.profile();
    for (format, numeric) in [
        (PixelFormat::RGBA32Uint, IoNumeric::Uint),
        (PixelFormat::RGBA8Unorm, IoNumeric::Float),
    ] {
        let mut inputs = LoweringInputs::new();
        inputs.zeroed_colors = BTreeMap::from([(2, format)]);
        let a = compile_with(&bc, "stage_io_fragment", p, &inputs)
            .unwrap_or_else(|e| panic!("{format:?}: {e}"));
        let at = |color| {
            a.interface
                .io
                .outputs
                .iter()
                .filter(|o| {
                    o.role
                        == IoRole::Color {
                            attachment: color,
                            index: 0,
                        }
                })
                .collect::<Vec<_>>()
        };
        let zero = at(2);
        assert!(
            matches!(zero.as_slice(), [o] if o.ty.numeric == numeric && o.ty.components == 4),
            "{format:?}: {zero:?}"
        );
        let own = at(0);
        assert!(
            matches!(own.as_slice(), [o] if o.ty.numeric == IoNumeric::Float),
            "{format:?}: {own:?}"
        );
        assert!(a.interface.io.unattached_colors.is_empty(), "{format:?}");
    }
}

/// A draw whose pass lacks colors the pipeline has: the variant that drops what the fragment
/// function writes there, which the base pipeline does not draw; compiled, its modules, facts and
/// outputs, and its settled plan. A function compiled alone drops a color it writes and states
/// one it does not.
#[test]
fn a_draw_without_colors_the_pipeline_has_runs_the_variant_dropping_them() {
    let Some(bc) = fixture("rt6_render_state") else {
        return;
    };
    let budget = Budget::standard();
    let lib = Library::from_air(&bc).expect("library");
    let mut d = RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(
        lib.function("rt6_full").expect("vertex"),
    )));
    d.fragment = Some(StageSource::Function(FunctionDescriptor::new(
        lib.function("rt6_white").expect("fragment"),
    )));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    d.color_attachments[1] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    let triangle = DrawCall::Primitives {
        vertex_start: 0,
        vertex_count: 3,
        instance_count: 1,
        base_instance: 0,
    };
    for stem in ["moltenvk-m3", "lavapipe"] {
        let dev = Device::new(description(stem), DevicePolicy::default()).expect("device");
        let base = dev
            .compile_render_pipeline(&d, &budget)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        // The fixture writes color 0 alone: dropping it leaves no color output; dropping color
        // 1 changes no output but is stated.
        for dropped in [BTreeSet::from([0]), BTreeSet::from([1])] {
            let what = format!("{stem} {dropped:?}");
            let state = BindState {
                dropped_colors: dropped.clone(),
                ..Default::default()
            };
            let plan = base
                .plan(&dev, &state.shape())
                .unwrap_or_else(|e| panic!("{what}: {e:?}"));
            assert_eq!(plan.variant.dropped_colors, dropped, "{what}");
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
            for &color in &dropped {
                assert!(
                    p.dependencies
                        .iter()
                        .any(|x| x.fact == RuntimeFact::DroppedColor { color }),
                    "{what}: the fact of color {color}"
                );
            }
            let fragment = compile_with(
                &bc,
                "rt6_white",
                dev.profile(),
                &key.lowering_inputs(Stage::Fragment),
            )
            .unwrap_or_else(|e| panic!("{what}: {e}"));
            let colors: BTreeSet<u32> = fragment
                .interface
                .io
                .outputs
                .iter()
                .filter_map(|o| match o.role {
                    IoRole::Color { attachment, .. } => Some(attachment),
                    _ => None,
                })
                .collect();
            let kept: BTreeSet<u32> = BTreeSet::from([0]).difference(&dropped).copied().collect();
            assert_eq!(colors, kept, "{what}: the color outputs");
            let plan = p
                .plan(&dev, &state.shape())
                .unwrap_or_else(|e| panic!("{what}: {e:?}"));
            assert_eq!(plan.variant, key, "{what}");
            plan.instantiate_draw(&state, &triangle)
                .unwrap_or_else(|e| panic!("{what}: {e:?}"));
        }
    }
}
