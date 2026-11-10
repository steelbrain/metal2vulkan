//! Inputs a fragment function declares whose reads its constants make dead, over the owned
//! fixture `engine/dfk_dead_rog` (each read in the arm of a branch on the `bool` function constant
//! 0) and `engine/color_groups`' `cg_cover`, against plain Metal's answers
//! (`metal2vulkan-conformance/probes/dead_ordered_and_unattached_reads.m`, an M3 Max under
//! macOS 27.0.1):
//!
//! - **A raster order group on an input the body never reads** orders nothing. Metal makes and
//!   draws `dfr_color` (a color 0 input in group 0) and `dfr_buffer` (a buffer in group 0) over
//!   five colors, and `dfr_output` (a color 0 input of a color 0 output in group 0) over one, with
//!   the read dead as with it live. Specialized dead, each compiles on every profile here without
//!   an interlock, including a device without one, passing L3; specialized live, each takes the
//!   pixel interlock where the device has it, and is refused where it has neither interlock.
//! - **A color input without an attachment** is Metal's pipeline error whether or not the body
//!   reads it ("Shaders reads from a color attachment whose pixel format is
//!   MTLPixelFormatInvalid", for `dfr_lacks`'s color 1 input with its read dead and live, beside
//!   a pipeline of color 0 alone): `InvalidDescriptor(ColorAttachment)` at that index, dead or
//!   live. With color 1 in the pipeline, the dead read compiles and reads nothing.
//!
//! The misses are counted by kind before a test fails. `llvm-as` assembles the fixtures; a missing
//! tool skips visibly. No GPU is used.

use std::collections::BTreeMap;

use crate::budget::Budget;
use crate::cache::VariantKey;
use crate::device::requirements::VkFeature;
use crate::device::{Device, DevicePolicy};
use crate::error::{DescriptorError, Error, Feature, MetalAnswer};
use crate::library::{FunctionConstantValue, FunctionConstantValues, Library};
use crate::metal::{DataType, PixelFormat};
use crate::pipeline::artifact::RenderPipeline;
use crate::pipeline::compile::StageEntries;
use crate::pipeline::descriptor::{
    ColorAttachmentDescriptor, FunctionDescriptor, RenderPipelineDescriptor, StageSource,
};
use crate::pipeline::testing::{description, engine_dir, llvm_assemble, AssembleError};
use crate::spirv::testing::check_words_valid;
use crate::vk;

fn fixture(name: &str) -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join(format!("{name}.ll"))) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}.ll: {why}"),
    }
}

/// The committed profile `stem` as a device, without `features`.
fn device_without(stem: &str, features: &[VkFeature]) -> Device {
    let mut desc = description(stem).expect("profile");
    for f in features {
        assert!(desc.features.remove(f), "{stem} has {f:?}");
    }
    Device::new(desc, DevicePolicy::default()).expect("device")
}

/// The devices of the raster order tests, each with whether it has the pixel interlock.
fn devices() -> Vec<(&'static str, Device, bool)> {
    let neither = [
        VkFeature::FragmentShaderPixelInterlock,
        VkFeature::FragmentShaderSampleInterlock,
    ];
    vec![
        ("lavapipe", device_without("lavapipe", &[]), true),
        ("moltenvk-m3", device_without("moltenvk-m3", &[]), true),
        (
            "moltenvk-m3 without interlock",
            device_without("moltenvk-m3", &neither),
            false,
        ),
    ]
}

/// The pipeline of `cg_cover` and `fragment` of `fs` drawing `RGBA16Float` at `colors`, the
/// fragment function specialized with its `bool` function constant 0 `reads`.
fn specialized(
    vs: &[u8],
    fs: &[u8],
    fragment: &str,
    colors: &[usize],
    reads: bool,
) -> RenderPipelineDescriptor {
    let function = |bc: &[u8], name: &str| {
        FunctionDescriptor::new(
            Library::from_air(bc)
                .expect("library")
                .function(name)
                .expect("function"),
        )
    };
    let mut d = RenderPipelineDescriptor::new(StageSource::Function(function(vs, "cg_cover")));
    let mut f = function(fs, fragment);
    f.constants = FunctionConstantValues::defaults().with_index(
        0,
        FunctionConstantValue::new(DataType::Bool, [u8::from(reads)]),
    );
    f.specialized = true;
    d.fragment = Some(StageSource::Function(f));
    for &c in colors {
        d.color_attachments[c] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA16Float));
    }
    d
}

/// Whether `words` interlocks: an interlock execution mode (`PixelInterlockOrderedEXT` 5366 to
/// `ShadingRateInterlockUnorderedEXT` 5371) or an `OpBeginInvocationInterlockEXT` (5364).
fn interlocks(words: &[u32]) -> bool {
    const OP_EXECUTION_MODE: u32 = 16;
    const OP_BEGIN_INTERLOCK: u32 = 5364;
    let mut at = 5;
    while at < words.len() {
        let len = (words[at] >> 16) as usize;
        let op = words[at] & 0xffff;
        let mode = words.get(at + 2).copied().unwrap_or(0);
        if op == OP_BEGIN_INTERLOCK || (op == OP_EXECUTION_MODE && (5366..=5371).contains(&mode)) {
            return true;
        }
        at += len.max(1);
    }
    false
}

fn fragment(p: &RenderPipeline) -> &[u32] {
    let m = p
        .stages
        .iter()
        .find(|m| m.stage == vk::ShaderStage::FRAGMENT);
    &m.expect("a fragment stage").spirv
}

fn validate(p: &RenderPipeline, dev: &Device, what: &str) {
    for m in &p.stages {
        check_words_valid(&m.spirv, &p.requirements, dev.profile())
            .unwrap_or_else(|e| panic!("{what} {:?}: L3: {e}", m.stage));
    }
}

/// A raster order group on an input the specialized body never reads orders nothing (the module
/// docs): dead, `dfr_color`, `dfr_buffer` (five colors) and `dfr_output` (one) compile without an
/// interlock on `lavapipe`, `moltenvk-m3` and `moltenvk-m3` without either interlock, and
/// `dfr_output` on `vk12-floor` (no interlock); live, each interlocks where the device has the
/// pixel interlock, and is refused where it has neither (it lacks the interlock, or a fetch form).
#[test]
fn a_raster_order_group_on_an_input_dead_after_specialization_takes_no_interlock() {
    let (Some(vs), Some(fs)) = (fixture("color_groups"), fixture("dfk_dead_rog")) else {
        return;
    };
    let budget = Budget::standard();
    let mut cases: Vec<(String, Device, bool, &str, Vec<usize>)> = Vec::new();
    for (name, dev, interlock) in devices() {
        for (entry, colors) in [
            ("dfr_color", vec![0, 1, 2, 3, 4]),
            ("dfr_buffer", vec![0, 1, 2, 3, 4]),
        ] {
            cases.push((name.to_owned(), dev.clone(), interlock, entry, colors));
        }
        cases.push((name.to_owned(), dev, interlock, "dfr_output", vec![0]));
    }
    let floor = Device::new(
        description("vk12-floor").expect("profile"),
        DevicePolicy::default(),
    )
    .expect("device");
    cases.push(("vk12-floor".to_owned(), floor, false, "dfr_output", vec![0]));
    let mut misses = BTreeMap::<String, Vec<String>>::new();
    let mut checked = 0;
    for (name, dev, interlock, entry, colors) in &cases {
        let what = format!("{entry} dead on {name}");
        match dev.compile_render_pipeline(&specialized(&vs, &fs, entry, colors, false), &budget) {
            Ok(p) if interlocks(fragment(&p)) => {
                misses.entry("interlocked".into()).or_default().push(what);
            }
            Ok(p) => {
                validate(&p, dev, &what);
                checked += 1;
            }
            Err(Error::Unsupported(u)) if u.feature == Feature::RasterOrderGroups => {
                misses
                    .entry("refused RasterOrderGroups".into())
                    .or_default()
                    .push(what);
            }
            Err(e) => panic!("{what}: {e}"),
        }
        let what = format!("{entry} live on {name}");
        let live =
            dev.compile_render_pipeline(&specialized(&vs, &fs, entry, colors, true), &budget);
        match (interlock, live) {
            (true, Ok(p)) => {
                assert!(interlocks(fragment(&p)), "{what}: no interlock");
                validate(&p, dev, &what);
            }
            // What the device lacks: the interlock, or a fetch form for a color read.
            (false, Err(Error::Unsupported(_))) => {}
            (_, other) => panic!("{what}: {:?}", other.map(|_| ())),
        }
    }
    for (kind, whats) in &misses {
        eprintln!("{}: {kind} (first: {})", whats.len(), whats[0]);
    }
    assert!(misses.is_empty(), "{misses:#?}");
    assert_eq!(checked, cases.len());
}

/// A color input without an attachment is Metal's pipeline error whether or not the specialized
/// body reads it (the module docs): `dfr_lacks` beside a pipeline of color 0 alone is
/// `InvalidDescriptor(ColorAttachment)` at index 1, dead and live, on `lavapipe` and `moltenvk-m3`,
/// compiled and keyed (`Device::render_variant_key`, which a runtime's creation asks first). With
/// color 1 in the pipeline, the dead read compiles.
#[test]
fn a_color_input_without_an_attachment_is_refused_dead_or_live() {
    let (Some(vs), Some(fs)) = (fixture("color_groups"), fixture("dfk_dead_rog")) else {
        return;
    };
    let budget = Budget::standard();
    let mut misses = BTreeMap::<String, Vec<String>>::new();
    for stem in ["lavapipe", "moltenvk-m3"] {
        let dev = device_without(stem, &[]);
        for reads in [false, true] {
            let what = format!(
                "dfr_lacks {} on {stem}",
                if reads { "live" } else { "dead" }
            );
            let d = specialized(&vs, &fs, "dfr_lacks", &[0], reads);
            // The compile, and the keys a runtime's creation asks for first, which compiles each
            // stage under the inputs they name.
            let keys = dev
                .render_variant_key(&d, &VariantKey::new(), &budget, &StageEntries::new())
                .map(|_| ());
            let compiled = dev.compile_render_pipeline(&d, &budget).map(|_| ());
            for (how, made) in [("compiled", compiled), ("keyed", keys)] {
                let what = format!("{what}, {how}");
                match made {
                    Err(Error::InvalidDescriptor(DescriptorError::ColorAttachment {
                        index: 1,
                        metal: MetalAnswer::NilWithError,
                    })) => {}
                    Ok(()) => misses.entry("made".into()).or_default().push(what),
                    Err(e) => misses
                        .entry(format!("refused: {e}"))
                        .or_default()
                        .push(what),
                }
            }
        }
        let d = specialized(&vs, &fs, "dfr_lacks", &[0, 1], false);
        let p = dev
            .compile_render_pipeline(&d, &budget)
            .unwrap_or_else(|e| panic!("dfr_lacks dead with color 1 on {stem}: {e}"));
        assert!(p.graphics.input_attachments.is_empty(), "{stem}");
        assert!(p.graphics.storage_attachments.is_empty(), "{stem}");
        validate(&p, &dev, stem);
    }
    for (kind, whats) in &misses {
        eprintln!("{}: {kind} (first: {})", whats.len(), whats[0]);
    }
    assert!(misses.is_empty(), "{misses:#?}");
}
