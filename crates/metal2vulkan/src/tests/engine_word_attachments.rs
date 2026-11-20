//! Color attachments a device renders through the view of their word
//! (`metal::format::WordAttachment`, `LoweringInputs::word_colors`).
//!
//! - On the RTX 5080's profile, which samples `R5G5B5A1_UNORM_PACK16` and `R4G4B4A4_UNORM_PACK16`
//!   but renders neither, a fragment function compiled for a pipeline with an `A1BGR5Unorm` or
//!   `ABGR4Unorm` attachment (`rt6_fill_rgba32float`, `rt6_fill_rgba16float`) records the
//!   attachment (`StageIo::word_colors`), keeps its interface variable's Metal type, declares the
//!   output variable as one `uint`, bounds its emission by the plan's estimate, and links: the
//!   attachment takes `R16_UINT`, a write of its one component under a full mask and of none under
//!   an empty one, no blend, and both modules pass L3.
//! - Where the device renders the format (lavapipe), the attachment takes the format itself.
//! - A mask of some of the format's channels, or a blend, on the 5080, which orders fetch through
//!   the pixel interlock: the whole render pipeline, made on the profile's device, writes the
//!   attachment in the interlocked storage form (`color_writes::stored_word`), a written storage
//!   attachment of `R16_UINT` without a view alias, keeping the channels the mask leaves out, and
//!   no color attachment of the subpass; a blended one's fragment notes `DeviceColorConversion`;
//!   its stages pass L3. Compiled without the pipeline's fetch state, such a mask or blend is the
//!   device's lack of the format.
//! - The 8-bit sRGB formats the 5080 renders through their words (`R8Unorm_sRGB`,
//!   `RG8Unorm_sRGB`) write the word too, and merge a partial mask in the interlock; a blend of one
//!   is the device's lack of the format.
//! - On lavapipe, which renders neither `R8_SRGB` nor `R8G8_SRGB` and orders fetch through
//!   rasterization-order input attachments, an `RG8Unorm_sRGB` attachment written under a mask of
//!   some of its channels keeps its `R16_UINT` color attachment, written whole, and its fragment
//!   merges the word into its input attachment's (`lower::fragment`'s `word_input`), stating the
//!   channels it keeps (`RuntimeFact::WordInputMerge`); a fragment compiled under another mask, or
//!   under a full one, does not link there (`NeedsInput(FramebufferFetch)`).
//! - `RGB9E5Float`, which the 5080 samples and does not render, takes a `float` output's word
//!   through `R32_UINT`; a `half` output is `NotImplemented(Format)`, and a partial mask or a
//!   blend the device's lack of the format.
//! - A fragment compiled without the pipeline's attachments does not link
//!   (`NeedsInput(ColorWrites)`).
//!
//! The encode's and the blend's words are `lower::arith`'s tests'
//! (`word_attachments_store_metals_words`, `a_packed_word_attachment_blend_stores_metals_words`).

use std::collections::BTreeMap;

use std::path::Path;

use crate::budget::Budget;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{Error, Feature, Needs, RuntimeInput};
use crate::library::{FunctionConstantValues, Library};
use crate::lower::select::LowerInputs;
use crate::metal::format::WordAttachment;
use crate::metal::{self, ColorWriteMask, PixelFormat};
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::compile::{plan_bitcode, EntryRef};
use crate::pipeline::descriptor::{
    ColorAttachmentDescriptor, FunctionDescriptor, RenderPipelineDescriptor, StageSource,
};
use crate::pipeline::link::{link, LinkInputs, RenderLinkInputs};
use crate::pipeline::relocate::relocate;
use crate::pipeline::stage_io::{IoNumeric, IoRole};
use crate::pipeline::testing::{
    compile, compile_with, engine_dir, fragment_inputs, llvm_assemble, meter, profile,
    AssembleError,
};
use crate::spirv::testing::{check_words_valid, spirv_dis};
use crate::vk;

const FORMATS: [PixelFormat; 2] = [PixelFormat::A1BGR5Unorm, PixelFormat::ABGR4Unorm];

/// The 8-bit sRGB formats the 5080 renders through their words.
const SRGB: [PixelFormat; 2] = [PixelFormat::R8UnormSRGB, PixelFormat::RG8UnormSRGB];

const ENTRIES: [&str; 2] = ["rt6_fill_rgba32float", "rt6_fill_rgba16float"];

fn fixture() -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join("rt6_samples.ll")) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected rt6_samples.ll: {why}"),
    }
}

fn venue(stem: &str) -> crate::device::profile::DeviceProfile {
    profile(stem, DevicePolicy::default()).expect("committed profile")
}

/// A render pipeline of one color attachment, `pf`, written under `mask` and blended where
/// `blended` (adding the source to the destination).
fn pipeline_of(pf: PixelFormat, mask: ColorWriteMask, blended: bool) -> RenderLinkInputs {
    let mut a = ColorAttachmentDescriptor::new(pf);
    a.write_mask = mask;
    if blended {
        a.blending_enabled = true;
        a.destination_rgb_blend_factor = metal::BlendFactor::One;
        a.destination_alpha_blend_factor = metal::BlendFactor::One;
    }
    let mut r = RenderLinkInputs::new();
    r.color_attachments[0] = Some(a);
    r
}

fn render(r: &RenderLinkInputs) -> LinkInputs {
    LinkInputs::Render(Box::new(r.clone()))
}

/// The device's lack of a format as a color attachment (blended or not).
fn lacks_attachment(e: &Error) -> bool {
    matches!(e, Error::Unsupported(u) if u.feature == Feature::Format && u.needs != Needs::NotImplemented)
}

/// The fixture's `float4` and `half4` fills write each format's word through one `uint` on the
/// RTX 5080's profile and link with `R16_UINT` (the module docs).
#[test]
fn fragment_outputs_to_word_attachments_write_the_word() {
    let Some(bc) = fixture() else {
        return;
    };
    let consts = FunctionConstantValues::defaults();
    let p = venue("nvidia-rtx5080-laptop");
    let v = compile(&bc, "rt6_fill_vertex", &p).expect("vertex");
    for pf in FORMATS.into_iter().chain(SRGB) {
        let word = WordAttachment::of(pf, &p).expect("a word attachment on the 5080");
        let channels = [
            ColorWriteMask::RED,
            ColorWriteMask::GREEN,
            ColorWriteMask::BLUE,
            ColorWriteMask::ALPHA,
        ]
        .into_iter()
        .zip(word.fields())
        .filter(|(_, f)| f.is_some())
        .fold(ColorWriteMask::NONE, |m, (c, _)| m.union(c));
        for entry in ENTRIES {
            for (mask, written) in [
                (ColorWriteMask::ALL, true),
                (ColorWriteMask::NONE, false),
                (ColorWriteMask::RED, true),
            ] {
                let what = format!("{entry} {pf:?} {mask:?}");
                let r = pipeline_of(pf, mask, false);
                let inputs = fragment_inputs(&r, &p);
                let f =
                    compile_with(&bc, entry, &p, &inputs).unwrap_or_else(|e| panic!("{what}: {e}"));
                let linked = link(&[&v, &f], &render(&r), &p);
                // Without the pipeline's fetch state, a mask of some of the format's channels is
                // the device's lack of the format.
                let kept = mask.intersection(channels);
                if !kept.is_empty() && kept != channels {
                    assert!(inputs.word_colors.is_empty(), "{what}");
                    let e = linked.expect_err(&what);
                    assert!(lacks_attachment(&e), "{what}: {e}");
                    continue;
                }
                let l = linked.unwrap_or_else(|e| panic!("{what}: link: {e}"));
                assert_eq!(
                    f.interface.io.word_colors,
                    BTreeMap::from([(0, word)]),
                    "{what}"
                );
                assert!(f.interface.io.color_writes.is_empty(), "{what}");
                let color = f
                    .interface
                    .io
                    .outputs
                    .iter()
                    .find(|o| {
                        matches!(
                            o.role,
                            IoRole::Color {
                                attachment: 0,
                                index: 0
                            }
                        )
                    })
                    .expect("color output");
                assert_eq!(
                    (color.ty.numeric, color.ty.components),
                    (IoNumeric::Float, 4),
                    "{what}"
                );
                let plan = plan_bitcode(
                    &bc,
                    EntryRef::Name(entry),
                    &consts,
                    &p,
                    &inputs,
                    &mut meter(),
                )
                .unwrap_or_else(|e| panic!("{what}: {e}"));
                assert!(
                    plan.estimate.words >= f.module.words.len() as u64,
                    "{what}: estimate {} < emitted {}",
                    plan.estimate.words,
                    f.module.words.len()
                );
                let g = l.graphics.as_ref().expect("graphics");
                let a = g.color_attachments[0].expect("attachment");
                assert_eq!(a.format, word.word_format(), "{what}");
                assert_eq!(
                    a.write_mask,
                    match written {
                        true => vk::ColorComponentFlags::R,
                        false => vk::ColorComponentFlags::empty(),
                    },
                    "{what}"
                );
                assert!(a.blend.is_none(), "{what}");
                let words: Vec<Vec<u32>> = [&*v, &*f]
                    .iter()
                    .map(|s| {
                        let w = relocate(s, &l.table).expect("relocate");
                        check_words_valid(&w, &s.requirements, &p)
                            .unwrap_or_else(|e| panic!("{what}: L3: {e}"));
                        w
                    })
                    .collect();
                let text = spirv_dis(&words[1]);
                assert!(
                    text.contains("%_ptr_Output_uint = OpTypePointer Output %uint"),
                    "{what}\n{text}"
                );
            }
        }
    }
}

/// The device of committed profile `stem`.
fn device(stem: &str) -> Device {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    Device::from_queries(desc, DevicePolicy::default()).expect("device")
}

/// A word attachment written under a mask of some of its channels, or blended, takes the
/// interlocked storage form in the whole pipeline on the 5080's device, a blended one noting
/// `DeviceColorConversion` (the module docs).
#[test]
fn masked_and_blended_word_attachments_take_the_interlock() {
    let Some(bc) = fixture() else {
        return;
    };
    let dev = device("nvidia-rtx5080-laptop");
    let library = Library::from_air(&bc).expect("library");
    let function = |name: &str| {
        StageSource::Function(FunctionDescriptor::new(
            library.function(name).expect("function"),
        ))
    };
    let note = PrecisionNote::DeviceColorConversion;
    let mut checked = 0;
    for pf in FORMATS {
        for entry in ENTRIES {
            for mask in [
                ColorWriteMask::RED | ColorWriteMask::GREEN,
                ColorWriteMask::ALPHA,
                ColorWriteMask::RED | ColorWriteMask::GREEN | ColorWriteMask::BLUE,
                ColorWriteMask::ALL,
            ] {
                for blended in [false, true] {
                    if mask == ColorWriteMask::ALL && !blended {
                        continue;
                    }
                    let what = format!("{entry} {pf:?} {mask:?} blended {blended}");
                    let mut d = RenderPipelineDescriptor::new(function("rt6_fill_vertex"));
                    d.fragment = Some(function(entry));
                    d.color_attachments[0] = pipeline_of(pf, mask, blended).color_attachments[0];
                    let pipe = dev
                        .compile_render_pipeline(&d, &Budget::standard())
                        .unwrap_or_else(|e| panic!("{what}: {e}"));
                    let storage: Vec<_> = (pipe.graphics.storage_attachments.iter())
                        .map(|u| (u.color_index, u.format, u.writes, u.view_alias))
                        .collect();
                    assert_eq!(storage, [(0, vk::Format::R16_UINT, true, None)], "{what}");
                    let kept = vk::ColorComponentFlags::known_bits()
                        .difference(crate::metal::state::color_write_mask(mask).expect("mask"));
                    assert_eq!(pipe.graphics.storage_attachments[0].kept, kept, "{what}");
                    assert!(pipe.graphics.color_attachments[0].is_none(), "{what}");
                    let fragment = (pipe.stages.iter())
                        .find(|s| s.stage == vk::ShaderStage::FRAGMENT)
                        .expect("a fragment stage");
                    assert_eq!(
                        fragment.precision_notes.contains(&note),
                        blended,
                        "{what}: {:?}",
                        fragment.precision_notes
                    );
                    for s in &pipe.stages {
                        check_words_valid(&s.spirv, &pipe.requirements, dev.profile())
                            .unwrap_or_else(|e| panic!("{what}: L3: {e}"));
                    }
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(checked, 28);
}

/// An 8-bit sRGB word attachment written under a mask of some of its channels takes the
/// interlocked storage form in the whole pipeline on the 5080's device, without a note; blended,
/// it is the device's lack of the format (Metal's sRGB blend over the word is not built).
#[test]
fn srgb_word_attachments_merge_but_do_not_blend() {
    let Some(bc) = fixture() else {
        return;
    };
    let dev = device("nvidia-rtx5080-laptop");
    let library = Library::from_air(&bc).expect("library");
    let function = |name: &str| {
        StageSource::Function(FunctionDescriptor::new(
            library.function(name).expect("function"),
        ))
    };
    for entry in ENTRIES {
        for (pf, mask, blended) in [
            (PixelFormat::RG8UnormSRGB, ColorWriteMask::RED, false),
            (PixelFormat::RG8UnormSRGB, ColorWriteMask::GREEN, false),
            (PixelFormat::RG8UnormSRGB, ColorWriteMask::ALL, true),
            (PixelFormat::R8UnormSRGB, ColorWriteMask::ALL, true),
        ] {
            let what = format!("{entry} {pf:?} {mask:?} blended {blended}");
            let mut d = RenderPipelineDescriptor::new(function("rt6_fill_vertex"));
            d.fragment = Some(function(entry));
            d.color_attachments[0] = pipeline_of(pf, mask, blended).color_attachments[0];
            let made = dev.compile_render_pipeline(&d, &Budget::standard());
            if blended {
                let e = made.expect_err(&what);
                assert!(lacks_attachment(&e), "{what}: {e}");
                continue;
            }
            let pipe = made.unwrap_or_else(|e| panic!("{what}: {e}"));
            let storage: Vec<_> = (pipe.graphics.storage_attachments.iter())
                .map(|u| (u.color_index, u.format, u.writes, u.view_alias))
                .collect();
            assert_eq!(storage, [(0, vk::Format::R16_UINT, true, None)], "{what}");
            assert!(pipe.graphics.color_attachments[0].is_none(), "{what}");
            for s in &pipe.stages {
                assert!(
                    !s.precision_notes
                        .contains(&PrecisionNote::DeviceColorConversion),
                    "{what}: {:?}",
                    s.precision_notes
                );
                check_words_valid(&s.spirv, &pipe.requirements, dev.profile())
                    .unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            }
        }
    }
}

/// On lavapipe's device an `RG8Unorm_sRGB` attachment written under a mask of some of its channels
/// keeps its color attachment of `R16_UINT`, written whole and unblended, and reads its input
/// attachment as `uint`s; its fragment, compiled under the pipeline's fetch state, states the merge
/// of the channels the mask leaves out and passes L3. Under a full mask the pipeline reads no input
/// attachment. A fragment compiled under another partial mask, or under the full one, does not
/// link into a pipeline of a partial mask (the module docs).
#[test]
fn srgb_word_attachments_merge_over_input_attachments() {
    use crate::pipeline::facts::RuntimeFact;
    use crate::pipeline::lower_inputs::FramebufferFetchInputs;
    use crate::pipeline::sym::SampledType;
    let Some(bc) = fixture() else {
        return;
    };
    let dev = device("lavapipe");
    let pf = PixelFormat::RG8UnormSRGB;
    assert!(WordAttachment::of(pf, dev.profile()).is_some());
    let library = Library::from_air(&bc).expect("library");
    let function = |name: &str| {
        StageSource::Function(FunctionDescriptor::new(
            library.function(name).expect("function"),
        ))
    };
    let p = dev.profile();
    let v = compile(&bc, "rt6_fill_vertex", p).expect("vertex");
    // The fragment compiled for pipeline `r`, as the pipeline compile hands it the fetch state.
    let fragment = |entry: &str, r: &RenderLinkInputs| {
        let mut inputs = fragment_inputs(r, p);
        inputs.framebuffer_fetch = Some(FramebufferFetchInputs::of(
            [0],
            &r.color_attachments,
            false,
            false,
            1,
        ));
        compile_with(&bc, entry, p, &inputs).expect("fragment")
    };
    for entry in ENTRIES {
        for (mask, kept) in [
            (ColorWriteMask::RED, vk::ColorComponentFlags::G),
            (ColorWriteMask::GREEN, vk::ColorComponentFlags::R),
            (
                ColorWriteMask::GREEN | ColorWriteMask::ALPHA,
                vk::ColorComponentFlags::R,
            ),
            (ColorWriteMask::ALL, vk::ColorComponentFlags::empty()),
        ] {
            let what = format!("{entry} {mask:?}");
            let mut d = RenderPipelineDescriptor::new(function("rt6_fill_vertex"));
            d.fragment = Some(function(entry));
            d.color_attachments[0] = pipeline_of(pf, mask, false).color_attachments[0];
            let pipe = dev
                .compile_render_pipeline(&d, &Budget::standard())
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            let a = pipe.graphics.color_attachments[0].expect("a color attachment");
            assert_eq!(
                (a.format, a.write_mask, a.blend),
                (vk::Format::R16_UINT, vk::ColorComponentFlags::R, None),
                "{what}"
            );
            assert!(pipe.graphics.storage_attachments.is_empty(), "{what}");
            let inputs: Vec<_> = (pipe.graphics.input_attachments.iter())
                .map(|u| (u.color_index, u.input_attachment_index, u.format_class))
                .collect();
            let merged = !kept.is_empty();
            let want = match merged {
                true => vec![(0, 0, SampledType::Uint)],
                false => Vec::new(),
            };
            assert_eq!(inputs, want, "{what}");
            for s in &pipe.stages {
                check_words_valid(&s.spirv, &pipe.requirements, p)
                    .unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            }
            let r = pipeline_of(pf, mask, false);
            let f = fragment(entry, &r);
            let stated: Vec<_> = (f.facts.iter())
                .filter(|f| matches!(f, RuntimeFact::WordInputMerge { .. }))
                .copied()
                .collect();
            let want = match merged {
                true => vec![RuntimeFact::WordInputMerge { color: 0, kept }],
                false => Vec::new(),
            };
            assert_eq!(stated, want, "{what}");
            link(&[&v, &f], &render(&r), p).unwrap_or_else(|e| panic!("{what}: link: {e}"));
        }
        // Compiled under another mask, or the full one, for a pipeline of a partial mask.
        let red = pipeline_of(pf, ColorWriteMask::RED, false);
        for other in [ColorWriteMask::GREEN, ColorWriteMask::ALL] {
            let f = fragment(entry, &pipeline_of(pf, other, false));
            let e = link(&[&v, &f], &render(&red), p).expect_err(entry);
            assert!(
                matches!(e, Error::NeedsInput(RuntimeInput::FramebufferFetch)),
                "{entry} compiled under {other:?}: {e}"
            );
        }
    }
}

/// An `RGB9E5Float` attachment on the 5080's device renders a `float` output through an
/// `R32_UINT` view, no storage form and no conversion note; a `half` output is `NotImplemented(Format)`, and
/// a mask of some of its channels, or a blend, the device's lack of the format (the module docs).
#[test]
fn an_rgb9e5_attachment_takes_a_float_outputs_word() {
    let Some(bc) = fixture() else {
        return;
    };
    let dev = device("nvidia-rtx5080-laptop");
    let library = Library::from_air(&bc).expect("library");
    let function = |name: &str| {
        StageSource::Function(FunctionDescriptor::new(
            library.function(name).expect("function"),
        ))
    };
    let pf = PixelFormat::RGB9E5Float;
    for entry in ENTRIES {
        let half = entry == "rt6_fill_rgba16float";
        for (mask, blended) in [
            (ColorWriteMask::ALL, false),
            (
                ColorWriteMask::RED | ColorWriteMask::GREEN | ColorWriteMask::BLUE,
                false,
            ),
            (ColorWriteMask::RED, false),
            (ColorWriteMask::ALL, true),
        ] {
            let what = format!("{entry} {mask:?} blended {blended}");
            let mut d = RenderPipelineDescriptor::new(function("rt6_fill_vertex"));
            d.fragment = Some(function(entry));
            d.color_attachments[0] = pipeline_of(pf, mask, blended).color_attachments[0];
            let made = dev.compile_render_pipeline(&d, &Budget::standard());
            let whole =
                mask.contains(ColorWriteMask::RED | ColorWriteMask::GREEN | ColorWriteMask::BLUE);
            if blended || !whole {
                let e = made.expect_err(&what);
                assert!(lacks_attachment(&e), "{what}: {e}");
                continue;
            }
            if half {
                let e = made.expect_err(&what);
                assert!(
                    matches!(&e, Error::Unsupported(u) if u.feature == Feature::Format && u.needs == Needs::NotImplemented),
                    "{what}: {e}"
                );
                continue;
            }
            let pipe = made.unwrap_or_else(|e| panic!("{what}: {e}"));
            assert!(pipe.graphics.storage_attachments.is_empty(), "{what}");
            let a = pipe.graphics.color_attachments[0].expect("attachment");
            assert_eq!(a.format, vk::Format::R32_UINT, "{what}");
            assert_eq!(a.write_mask, vk::ColorComponentFlags::R, "{what}");
            assert!(a.blend.is_none(), "{what}");
            for s in &pipe.stages {
                assert!(
                    !s.precision_notes
                        .contains(&PrecisionNote::DeviceColorConversion),
                    "{what}: {:?}",
                    s.precision_notes
                );
                check_words_valid(&s.spirv, &pipe.requirements, dev.profile())
                    .unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            }
        }
    }
}

/// A blended word attachment compiled without the pipeline's fetch state is refused as the
/// device's lack of the format, and a fragment compiled without the pipeline's attachments does
/// not link; where the device renders the format, the attachment takes it (the module docs).
#[test]
fn word_attachments_refuse_blends_and_other_maps() {
    let Some(bc) = fixture() else {
        return;
    };
    let nvidia = venue("nvidia-rtx5080-laptop");
    let lavapipe = venue("lavapipe");
    for pf in FORMATS {
        for entry in ENTRIES {
            let what = format!("{entry} {pf:?}");
            // Blended: no word view, and the device has no attachment of the format.
            let v = compile(&bc, "rt6_fill_vertex", &nvidia).expect("vertex");
            let r = pipeline_of(pf, ColorWriteMask::ALL, true);
            let inputs = fragment_inputs(&r, &nvidia);
            assert!(inputs.word_colors.is_empty(), "{what}");
            let f = compile_with(&bc, entry, &nvidia, &inputs)
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            let e = link(&[&v, &f], &render(&r), &nvidia).expect_err(&what);
            assert!(lacks_attachment(&e), "{what}: blended: {e}");
            // Compiled without the pipeline's attachments.
            let r = pipeline_of(pf, ColorWriteMask::ALL, false);
            let f = compile_with(&bc, entry, &nvidia, &LowerInputs::new())
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            let e = link(&[&v, &f], &render(&r), &nvidia).expect_err(&what);
            assert!(
                matches!(e, Error::NeedsInput(RuntimeInput::ColorWrites)),
                "{what}: another map: {e}"
            );
            // A device that renders the format takes it.
            let v = compile(&bc, "rt6_fill_vertex", &lavapipe).expect("vertex");
            let inputs = fragment_inputs(&r, &lavapipe);
            assert!(inputs.word_colors.is_empty(), "{what}");
            let f = compile_with(&bc, entry, &lavapipe, &inputs)
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            let l = link(&[&v, &f], &render(&r), &lavapipe)
                .unwrap_or_else(|e| panic!("{what}: lavapipe: {e}"));
            let a = l.graphics.as_ref().expect("graphics").color_attachments[0].expect("color");
            assert_eq!(
                Some(a.format),
                crate::metal::format::direct_format(pf),
                "{what}"
            );
        }
    }
}
