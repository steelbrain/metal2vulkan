//! A render pipeline compiled to draw a rate-mapped pass's regions several at a time
//! (`LoweringInputs::region_viewports`, `VariantKey::region_viewports`), over the owned fixtures
//! `engine/rfb_viewports` (Metal's fullscreen triangle), `engine/rt6_viewports` (a vertex function
//! that selects its viewport, and a fragment function that reads the index) and
//! `engine/vertex_amplification` (a post-tessellation vertex function):
//!
//! - Under it the vertex function states `RuntimeFact::RegionViewports`, reads `DrawIndex` and
//!   stores `ViewportIndex` (needing `shaderDrawParameters` and `shaderOutputViewportIndex`) while
//!   its contract states no viewport selection, and passes L3 (`spirv-val` and requirements
//!   within the profile), adding a bounded handful of words; compiled without it, it reads no
//!   `DrawIndex` and stores no `ViewportIndex`. The variant's pipeline selects none of the encoder's viewports
//!   (`GraphicsState::writes_viewport_index`), so a rate-mapped pass's plan admits it.
//! - A fragment function compiled under it that reads `[[viewport_array_index]]` reads 0 and
//!   declares no `ViewportIndex`.
//! - A vertex function that selects its own viewport, a post-tessellation one, and one under
//!   draws of one amplification each or a viewport compensation is
//!   `NotImplemented(RasterizationRateMap)`; a device without `shaderDrawParameters` names the
//!   three features the variant needs.
//! - A render plan whose shape asks for it states the variant, which the pipeline compiled
//!   without it does not draw (`BindError::NeedsVariant`); compiled, the variant states the fact
//!   and its plan settles on itself.
//!
//! `llvm-as` assembles the fixtures; a missing tool skips visibly. No GPU is used.

use std::path::Path;

use crate::bind::{BindState, DrawCall};
use crate::budget::Budget;
use crate::device::requirements::VkFeature as F;
use crate::device::{Device, DeviceDescription, DevicePolicy, DeviceProfile};
use crate::error::{BindError, Capability, Error, Feature, Unsupported};
use crate::library::{FunctionArtifact, Library};
use crate::metal::PixelFormat;
use crate::pipeline::descriptor::{
    ColorAttachmentDescriptor, FunctionDescriptor, RenderPipelineDescriptor, StageSource,
};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::link::{link, LinkInputs};
use crate::pipeline::lower_inputs::{AmplificationDraws, LoweringInputs, ViewportCompensation};
use crate::pipeline::relocate::relocate;
use crate::pipeline::stage_io::StageBuiltin;
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

/// The lowering inputs with `region_viewports` as given.
fn inputs(region_viewports: bool) -> LoweringInputs {
    let mut inputs = LoweringInputs::new();
    inputs.region_viewports = region_viewports;
    inputs
}

/// Link `a` alone, relocate it, and run L3; its relocated words.
fn validate(a: &FunctionArtifact, p: &DeviceProfile) -> Vec<u32> {
    let linked = link(&[a], &LinkInputs::standalone(), p).unwrap_or_else(|e| panic!("link: {e}"));
    let words = relocate(a, &linked.table).expect("relocate");
    check_words_valid(&words, &a.requirements, p).unwrap_or_else(|e| panic!("L3: {e}"));
    words
}

/// Whether `a`'s stage interface declares `builtin`.
fn declares(a: &FunctionArtifact, builtin: StageBuiltin) -> bool {
    a.interface.io.builtins.contains(&builtin)
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

/// The fullscreen triangle under the field and without it: the fact, `DrawIndex` and
/// `ViewportIndex` declared under it alone, L3, and the words it adds (printed; bounded). Its
/// fragment function reading the index reads 0 under it, declaring no `ViewportIndex`.
#[test]
fn a_region_vertex_function_stores_its_draw_index_as_its_viewport() {
    let Some(viewports) = fixture("rfb_viewports") else {
        return;
    };
    let Some(selects) = fixture("rt6_viewports") else {
        return;
    };
    for stem in ["lavapipe", "moltenvk-m3", "intel-irisxe-rplp-anv"] {
        let p = profile_of(description(stem));
        let plain = compile_with(&viewports, "rfb_full", &p, &inputs(false))
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let plain_words = validate(&plain, &p).len();
        assert!(
            !plain.facts.contains(&RuntimeFact::RegionViewports),
            "{stem}"
        );
        assert!(!declares(&plain, StageBuiltin::DrawIndex), "{stem}");
        assert!(!declares(&plain, StageBuiltin::ViewportIndex), "{stem}");
        let a = compile_with(&viewports, "rfb_full", &p, &inputs(true))
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let words = validate(&a, &p).len();
        assert!(a.facts.contains(&RuntimeFact::RegionViewports), "{stem}");
        assert!(declares(&a, StageBuiltin::DrawIndex), "{stem}");
        // Its `ViewportIndex` selects its region's viewport, none of the encoder's: the module
        // stores it, needing the output, and the contract states no selection.
        assert!(!declares(&a, StageBuiltin::ViewportIndex), "{stem}");
        let needs = &a.requirements.features;
        assert!(needs.contains(&F::ShaderOutputViewportIndex), "{stem}");
        assert!(needs.contains(&F::ShaderDrawParameters), "{stem}");
        assert!(!plain
            .requirements
            .features
            .contains(&F::ShaderOutputViewportIndex));
        let added = words as i64 - plain_words as i64;
        println!("{stem} rfb_full: region viewports add {added} words to {plain_words}");
        assert!((1..=40).contains(&added), "{stem}: {added} words");
        // The fragment function reads 0 under it, the builtin otherwise.
        let read = compile_with(&selects, "rt6_viewport_read", &p, &inputs(false))
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        validate(&read, &p);
        assert!(declares(&read, StageBuiltin::ViewportIndex), "{stem}");
        let zero = compile_with(&selects, "rt6_viewport_read", &p, &inputs(true))
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        validate(&zero, &p);
        assert!(!declares(&zero, StageBuiltin::ViewportIndex), "{stem}");
        assert!(zero.interface.io.zero_indices.is_empty(), "{stem}");
    }
}

/// What the runtime draws region by region is refused under the field, and a device without
/// `shaderDrawParameters` names the three features the variant needs.
#[test]
fn region_viewports_refuse_what_is_drawn_region_by_region() {
    let Some(viewports) = fixture("rfb_viewports") else {
        return;
    };
    let Some(selects) = fixture("rt6_viewports") else {
        return;
    };
    let Some(amplification) = fixture("vertex_amplification") else {
        return;
    };
    let p = profile_of(description("lavapipe"));
    let not_implemented = Err(Error::Unsupported(Unsupported::not_implemented(
        Feature::RasterizationRateMap,
    )));
    let refusal = |bc: &[u8], entry: &str, inputs: &LoweringInputs| {
        compile_with(bc, entry, &p, inputs)
            .map(|_| ())
            .map_err(unlocated)
    };
    assert_eq!(
        refusal(&selects, "rt6_select", &inputs(true)),
        not_implemented
    );
    assert_eq!(
        refusal(&amplification, "amp_post_tessellation", &inputs(true)),
        not_implemented
    );
    let mut amplified = inputs(true);
    amplified.amplification_draws = Some(AmplificationDraws {
        layered: false,
        viewports: true,
    });
    assert_eq!(refusal(&viewports, "rfb_full", &amplified), not_implemented);
    let mut compensated = inputs(true);
    compensated.viewport_compensation = Some(ViewportCompensation::new(true));
    assert_eq!(
        refusal(&viewports, "rfb_full", &compensated),
        not_implemented
    );
    let mut desc = description("lavapipe");
    assert!(desc.features.remove(&F::ShaderDrawParameters));
    let without = profile_of(desc);
    let refusal = compile_with(&viewports, "rfb_full", &without, &inputs(true))
        .map(|_| ())
        .map_err(unlocated);
    assert_eq!(
        refusal,
        Err(Error::Unsupported(Unsupported::needs_any_of(
            Feature::RasterizationRateMap,
            vec![vec![
                Capability::Feature(F::ShaderOutputViewportIndex),
                Capability::Feature(F::ShaderDrawParameters),
                Capability::Feature(F::MultiViewport),
            ]],
        )))
    );
    compile_with(&viewports, "rfb_full", &without, &inputs(false)).expect("without the field");
}

/// A plan whose shape draws regions several at a time states the variant, which the base
/// pipeline does not draw; compiled, the variant's modules pass L3, it states the fact, and its
/// plan settles on itself and draws.
#[test]
fn a_region_draw_runs_its_variant() {
    let Some(bc) = fixture("rt6_viewports") else {
        return;
    };
    let budget = Budget::standard();
    let lib = Library::from_air(&bc).expect("library");
    let Some(full) = fixture("rfb_viewports") else {
        return;
    };
    let full = Library::from_air(&full).expect("library");
    let mut d = RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(
        full.function("rfb_full").expect("function"),
    )));
    d.fragment = Some(StageSource::Function(FunctionDescriptor::new(
        lib.function("rt6_viewport_read").expect("function"),
    )));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    let triangle = DrawCall::Primitives {
        vertex_start: 0,
        vertex_count: 3,
        instance_count: 1,
        base_instance: 0,
    };
    for stem in ["lavapipe", "intel-irisxe-rplp-anv"] {
        let dev = Device::from_queries(description(stem), DevicePolicy::default()).expect("device");
        let base = dev
            .compile_render_pipeline(&d, &budget)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let state = BindState {
            region_viewports: true,
            ..Default::default()
        };
        let plan = base
            .plan(&dev, &state.shape())
            .unwrap_or_else(|e| panic!("{stem}: {e:?}"));
        assert!(plan.variant.region_viewports, "{stem}");
        assert_eq!(
            plan.instantiate_draw(&state, &triangle),
            Err(BindError::NeedsVariant {
                variant: Box::new(plan.variant.clone())
            }),
            "{stem}"
        );
        let key = plan.variant.clone();
        let p = dev
            .compile_render_variant(&d, &key, &budget)
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        for m in &p.stages {
            check_words_valid(&m.spirv, &p.requirements, dev.profile())
                .unwrap_or_else(|e| panic!("{stem} {:?}: L3: {e}", m.stage));
        }
        assert!(
            p.dependencies
                .iter()
                .any(|x| x.fact == RuntimeFact::RegionViewports),
            "{stem}"
        );
        // It selects none of the encoder's viewports: a rate-mapped pass's plan admits it.
        assert!(!p.graphics.writes_viewport_index, "{stem}");
        let plan = p
            .plan(&dev, &state.shape())
            .unwrap_or_else(|e| panic!("{stem}: {e:?}"));
        assert_eq!(plan.variant, key, "{stem}");
        plan.instantiate_draw(&state, &triangle)
            .unwrap_or_else(|e| panic!("{stem}: {e:?}"));
        // Drawn one region at a time, the base pipeline's plan is its own.
        let plain = base
            .plan(&dev, &BindState::default().shape())
            .unwrap_or_else(|e| panic!("{stem}: {e:?}"));
        assert!(!plain.variant.region_viewports, "{stem}");
    }
}
