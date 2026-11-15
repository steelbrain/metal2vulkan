//! A render pipeline's fan-out variant (`VariantKey::region_fan_out`, `pipeline::link::fan_out`),
//! over the owned fixtures `engine/rfb_viewports` (Metal's fullscreen triangle) and
//! `engine/rt6_viewports` (a fragment function writing white):
//!
//! - A plan whose shape fans a draw's triangles out to a rate-mapped pass's regions states the
//!   variant, which the base pipeline does not draw (`BindError::NeedsVariant`); compiled, its
//!   modules pass L3, its graphics state states the fan-out, it states no synthesized-geometry
//!   fact, and its plan settles on itself. A draw binds the region table; a probe binds its flags
//!   too and pushes its flag's index to the geometry stage. A region-copy variant draws no geometry
//!   stage: its vertex function draws its region's copy, sixteen more `NoContraction`
//!   operations than the plain one's and its clip distances `Invariant` too, binds the region
//!   table and pushes its first region to the vertex stage.
//! - A merged draw replay's variant (`VariantKey::merged_draws`, `bind::replay::merged`) fans out
//!   in every mode as a draw's does: its modules pass L3, its vertex function pushes the header's
//!   address, and its plan settles on itself; its region-copy variant draws no geometry stage, its
//!   vertex function storing its region's viewport index and pushing its first region beside the
//!   header's address.
//! - Drawn amplified, the plan keeps the plain variant, and draws the regions one at a time.
//!
//! `llvm-as` assembles the fixtures; a missing tool skips visibly. No GPU is used.

use std::path::Path;

use crate::bind::{BindState, DrawCall, WriteSource};
use crate::budget::Budget;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::BindError;
use crate::library::Library;
use crate::metal::PixelFormat;
use crate::pipeline::descriptor::{
    ColorAttachmentDescriptor, FunctionDescriptor, RenderPipelineDescriptor, StageSource,
};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::testing::{engine_dir, llvm_assemble, AssembleError};
use crate::pipeline::RegionFanOut;
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

/// The fullscreen triangle's vertex function and the white fragment function, into one
/// `RGBA8Unorm` attachment.
fn descriptor() -> Option<RenderPipelineDescriptor> {
    let full = Library::from_air(&fixture("rfb_viewports")?).expect("library");
    let white = Library::from_air(&fixture("rt6_viewports")?).expect("library");
    let mut d = RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(
        full.function("rfb_full").expect("function"),
    )));
    d.fragment = Some(StageSource::Function(FunctionDescriptor::new(
        white.function("rt6_white").expect("function"),
    )));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    Some(d)
}

const TRIANGLE: DrawCall = DrawCall::Primitives {
    vertex_start: 0,
    vertex_count: 3,
    instance_count: 1,
    base_instance: 0,
};

/// A plan whose shape fans triangles out states the variant, which the base pipeline does not
/// draw; compiled, the variant's modules pass L3, its state states the fan-out, and its plan
/// settles on itself, binding the region table and, for a probe, the flags and the index.
#[test]
fn a_fanned_out_draw_runs_its_variant() {
    let Some(d) = descriptor() else {
        return;
    };
    let budget = Budget::standard();
    for stem in ["lavapipe", "intel-irisxe-rplp-anv", "nvidia-pascal-linux"] {
        let dev = Device::new(description(stem), DevicePolicy::default()).expect("device");
        let base = dev
            .compile_render_pipeline(&d, &budget)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let plain = (base.stages.iter())
            .find(|m| m.stage == vk::ShaderStage::VERTEX)
            .map(|m| decorated(&m.spirv))
            .expect("a vertex module");
        for mode in [
            RegionFanOut::Draw,
            RegionFanOut::Probe,
            RegionFanOut::DrawFew,
            RegionFanOut::RegionCopies,
        ] {
            let state = BindState {
                region_fan_out: Some((mode, 5)),
                region_viewports: true,
                ..Default::default()
            };
            let plan = base
                .plan(&dev, &state.shape())
                .unwrap_or_else(|e| panic!("{stem}: {e:?}"));
            assert_eq!(plan.variant.region_fan_out, Some(mode), "{stem}");
            // It draws no regions several at a time.
            assert!(!plan.variant.region_viewports, "{stem}");
            assert_eq!(
                plan.instantiate_draw(&state, &TRIANGLE),
                Err(BindError::NeedsVariant {
                    variant: Box::new(plan.variant.clone())
                }),
                "{stem}"
            );
            let key = plan.variant.clone();
            let p = dev
                .compile_render_variant(&d, &key, &budget)
                .unwrap_or_else(|e| panic!("{stem} {mode:?}: {e}"));
            for m in &p.stages {
                check_words_valid(&m.spirv, &p.requirements, dev.profile())
                    .unwrap_or_else(|e| panic!("{stem} {mode:?} {:?}: L3: {e}", m.stage));
            }
            let geometry = (p.stages.iter())
                .filter(|m| m.stage == vk::ShaderStage::GEOMETRY)
                .count();
            let stages = match mode {
                RegionFanOut::RegionCopies => 0,
                _ => 3,
            };
            assert_eq!(geometry, stages, "{stem} {mode:?}");
            let fan = p.graphics.region_fan_out.expect("the fan-out state");
            assert_eq!(fan.mode, mode, "{stem}");
            assert!(
                !(p.dependencies.iter())
                    .any(|x| matches!(x.fact, RuntimeFact::SynthesizedGeometry { .. })),
                "{stem} {mode:?}"
            );
            let plan = p
                .plan(&dev, &state.shape())
                .unwrap_or_else(|e| panic!("{stem}: {e:?}"));
            assert_eq!(plan.variant, key, "{stem}");
            let sources: Vec<&WriteSource> = plan.descriptors.iter().map(|w| &w.source).collect();
            assert!(sources.contains(&&WriteSource::RateMapRegions), "{stem}");
            assert_eq!(
                sources.contains(&&WriteSource::RateMapFlags),
                mode == RegionFanOut::Probe,
                "{stem}"
            );
            let commands = plan
                .instantiate_draw(&state, &TRIANGLE)
                .unwrap_or_else(|e| panic!("{stem} {mode:?}: {e:?}"));
            let pushed: Vec<_> = (commands.push_constants.iter())
                .filter(|w| w.stages == vk::ShaderStageFlags::GEOMETRY)
                .map(|w| (w.offset, w.bytes.clone()))
                .collect();
            match mode {
                RegionFanOut::Probe => {
                    assert_eq!(pushed, [(fan.push_offset, 5u32.to_le_bytes().to_vec())])
                }
                _ => assert!(pushed.is_empty(), "{stem}"),
            }
            if mode != RegionFanOut::RegionCopies {
                continue;
            }
            // Its box of regions, in the vertex stage's push range: its first region, a box of
            // one column.
            let at = fan.push_offset;
            let boxed = (commands.push_constants.iter())
                .filter(|w| w.stages.contains(vk::ShaderStageFlags::VERTEX))
                .find(|w| w.offset <= at && at + 8 <= w.offset + w.bytes.len() as u32)
                .map(|w| w.bytes[(at - w.offset) as usize..][..8].to_vec());
            let words = [5u32, 1].map(u32::to_le_bytes);
            assert_eq!(boxed, Some(words.as_flattened().to_vec()), "{stem}");
            let vertex = (p.stages.iter())
                .find(|m| m.stage == vk::ShaderStage::VERTEX)
                .map(|m| decorated(&m.spirv))
                .expect("a vertex module");
            // Of each axis's coordinate and its two clip distances, five products and three sums.
            assert_eq!(vertex[0], plain[0] + 2 * 8, "{stem}");
            assert_eq!(vertex[1], plain[1] + 1, "{stem}");
            // It stores a viewport index, the recording's one viewport's.
            assert_eq!(vertex[2], plain[2] + 1, "{stem}");
        }
    }
}

/// A merged draw replay's variant fans out in every mode: its modules pass L3, its graphics state
/// states the fan-out, its vertex function pushes the header's address (and a region-copy one its
/// first region too, storing its region's viewport index), and its plan settles on itself.
#[test]
fn a_merged_draw_s_variant_fans_out() {
    use crate::pipeline::sym::PushField;
    let Some(d) = descriptor() else {
        return;
    };
    let budget = Budget::standard();
    for stem in ["lavapipe", "intel-irisxe-rplp-anv", "nvidia-pascal-linux"] {
        let dev = Device::new(description(stem), DevicePolicy::default()).expect("device");
        let base = dev
            .compile_render_pipeline(&d, &budget)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let plain = (base.stages.iter())
            .find(|m| m.stage == vk::ShaderStage::VERTEX)
            .map(|m| decorated(&m.spirv))
            .expect("a vertex module");
        for mode in [
            RegionFanOut::Draw,
            RegionFanOut::Probe,
            RegionFanOut::DrawFew,
            RegionFanOut::RegionCopies,
        ] {
            let state = BindState {
                region_fan_out: Some((mode, 5)),
                region_viewports: true,
                ..Default::default()
            };
            let plan = base
                .plan(&dev, &state.shape())
                .unwrap_or_else(|e| panic!("{stem}: {e:?}"));
            // The runtime's merged form (`BufferForms::Merged`).
            let mut key = plan.variant.clone();
            key.replay_bindings = true;
            key.merged_draws = true;
            let p = dev
                .compile_render_variant(&d, &key, &budget)
                .unwrap_or_else(|e| panic!("{stem} {mode:?}: {e}"));
            for m in &p.stages {
                check_words_valid(&m.spirv, &p.requirements, dev.profile())
                    .unwrap_or_else(|e| panic!("{stem} {mode:?} {:?}: L3: {e}", m.stage));
            }
            let geometry = (p.stages.iter())
                .filter(|m| m.stage == vk::ShaderStage::GEOMETRY)
                .count();
            let stages = match mode {
                RegionFanOut::RegionCopies => 0,
                _ => 3,
            };
            assert_eq!(geometry, stages, "{stem} {mode:?}");
            let fan = p.graphics.region_fan_out.expect("the fan-out state");
            assert_eq!(fan.mode, mode, "{stem}");
            assert!(
                (p.dependencies.iter()).any(|x| x.fact == RuntimeFact::MergedDraws),
                "{stem} {mode:?}"
            );
            let fields: Vec<PushField> = (p.interface.push.iter())
                .flat_map(|p| p.fields.iter().map(|f| f.field))
                .collect();
            assert!(
                fields.contains(&PushField::MergedDrawAddress),
                "{stem} {mode:?}: {fields:?}"
            );
            assert_eq!(
                fields.contains(&PushField::RegionBox),
                mode == RegionFanOut::RegionCopies,
                "{stem} {mode:?}: {fields:?}"
            );
            let settled = p
                .plan(&dev, &state.shape())
                .unwrap_or_else(|e| panic!("{stem} {mode:?}: {e:?}"));
            assert_eq!(settled.variant, key, "{stem} {mode:?}");
            if mode != RegionFanOut::RegionCopies {
                continue;
            }
            let vertex = (p.stages.iter())
                .find(|m| m.stage == vk::ShaderStage::VERTEX)
                .map(|m| decorated(&m.spirv))
                .expect("a vertex module");
            // Its copy's arithmetic, as a draw's: five products and three sums of each axis's
            // coordinate and its two clip distances; and the recording's one viewport's index.
            assert_eq!(vertex[0], plain[0] + 2 * 8, "{stem}");
            assert_eq!(vertex[2], plain[2] + 1, "{stem}");
        }
    }
}

/// How many `NoContraction`, `Invariant` and `BuiltIn ViewportIndex` decorations module `words`
/// holds.
fn decorated(words: &[u32]) -> [usize; 3] {
    use crate::spv::{BuiltIn, Decoration};
    let mut n = [0; 3];
    let mut i = 5;
    while let Some(&w) = words.get(i) {
        let len = (w >> 16) as usize;
        if w & 0xFFFF == crate::spv::Op::Decorate as u32 {
            match words.get(i + 2).copied() {
                Some(d) if d == Decoration::NoContraction as u32 => n[0] += 1,
                Some(d) if d == Decoration::Invariant as u32 => n[1] += 1,
                Some(d)
                    if d == Decoration::BuiltIn as u32
                        && words.get(i + 3).copied() == Some(BuiltIn::ViewportIndex as u32) =>
                {
                    n[2] += 1
                }
                _ => {}
            }
        }
        if len == 0 {
            break;
        }
        i += len;
    }
    n
}

/// Drawn amplified, a plan whose shape fans triangles out keeps the plain variant.
#[test]
fn an_amplified_draw_keeps_the_plain_variant() {
    let Some(mut d) = descriptor() else {
        return;
    };
    d.max_vertex_amplification_count = 2;
    let budget = Budget::standard();
    let dev = Device::new(description("lavapipe"), DevicePolicy::default()).expect("device");
    let base = dev
        .compile_render_pipeline(&d, &budget)
        .unwrap_or_else(|e| panic!("{e}"));
    let state = BindState {
        region_fan_out: Some((RegionFanOut::Draw, 0)),
        ..Default::default()
    };
    let plan = base
        .plan(&dev, &state.shape())
        .unwrap_or_else(|e| panic!("{e:?}"));
    assert_eq!(plan.variant.region_fan_out, None);
}
