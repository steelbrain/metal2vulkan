//! Color attachments a device renders through the view of their word
//! (`metal::format::WordAttachment`, `LoweringInputs::word_colors`).
//!
//! - On the RTX 5080's profile, which samples `R5G5B5A1_UNORM_PACK16` and `R4G4B4A4_UNORM_PACK16`
//!   but renders neither, a fragment function compiled for a pipeline with an `A1BGR5Unorm` or
//!   `ABGR4Unorm` attachment (`rt6_fill_rgba32float`, `rt6_fill_rgba16float`) records the
//!   attachment (`StageIo::word_colors`), keeps its interface variable's Metal type, declares the
//!   output variable as one `uint`, and links: the
//!   attachment takes `R16_UINT`, a write of its one component under a full mask and of none under
//!   an empty one, no blend, and both modules pass L3.
//! - Where the device renders the format (lavapipe), the attachment takes the format itself.
//! - A mask of some of the format's channels, or a blend, on the 5080: the whole render pipeline
//!   is the device's lack of the format.
//! - The 8-bit sRGB formats the 5080 renders through their words (`R8Unorm_sRGB`,
//!   `RG8Unorm_sRGB`) write the word too; a partial mask or a blend of one is the device's lack of
//!   the format there, and on lavapipe, which renders neither `R8_SRGB` nor `R8G8_SRGB`: the
//!   merge of the word into the texel over an ordered read of the attachment is not built.
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
use crate::library::Library;
use crate::lower::select::LowerInputs;
use crate::metal::format::WordAttachment;
use crate::metal::{self, ColorWriteMask, PixelFormat};
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::descriptor::{
    ColorAttachmentDescriptor, FunctionDescriptor, RenderPipelineDescriptor, StageSource,
};
use crate::pipeline::link::{link, LinkInputs, RenderLinkInputs};
use crate::pipeline::relocate::relocate;
use crate::pipeline::stage_io::{IoNumeric, IoRole};
use crate::pipeline::testing::{
    compile, compile_with, engine_dir, fragment_inputs, llvm_assemble, profile, AssembleError,
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
    profile(stem, DevicePolicy::RESIDENT_TABLES).expect("committed profile")
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
    Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES).expect("device")
}

/// A word attachment written under a mask of some of its channels, or blended, is the device's
/// lack of the format in the whole pipeline on the 5080's device, which does not order fetch (the
/// module docs).
#[test]
fn masked_and_blended_word_attachments_are_the_devices_lack_of_the_format() {
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
                    let e = dev
                        .compile_render_pipeline(&d, &Budget::standard())
                        .map(|_| ())
                        .expect_err(&what);
                    assert!(lacks_attachment(&e), "{what}: {e}");
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(checked, 28);
}

/// An 8-bit sRGB word attachment written under a mask of some of its channels, or blended, is the
/// device's lack of the format on the 5080's device and on lavapipe's, which render neither
/// format (the merge over the word and Metal's sRGB blend over it are not built).
#[test]
fn srgb_word_attachments_neither_merge_nor_blend() {
    let Some(bc) = fixture() else {
        return;
    };
    let library = Library::from_air(&bc).expect("library");
    let function = |name: &str| {
        StageSource::Function(FunctionDescriptor::new(
            library.function(name).expect("function"),
        ))
    };
    for (stem, entry) in ["nvidia-rtx5080-laptop", "lavapipe"]
        .into_iter()
        .flat_map(|s| ENTRIES.map(|e| (s, e)))
    {
        let dev = device(stem);
        assert!(WordAttachment::of(PixelFormat::RG8UnormSRGB, dev.profile()).is_some());
        for (pf, mask, blended) in [
            (PixelFormat::RG8UnormSRGB, ColorWriteMask::RED, false),
            (PixelFormat::RG8UnormSRGB, ColorWriteMask::GREEN, false),
            (
                PixelFormat::RG8UnormSRGB,
                ColorWriteMask::GREEN | ColorWriteMask::ALPHA,
                false,
            ),
            (PixelFormat::RG8UnormSRGB, ColorWriteMask::ALL, true),
            (PixelFormat::R8UnormSRGB, ColorWriteMask::ALL, true),
        ] {
            let what = format!("{stem} {entry} {pf:?} {mask:?} blended {blended}");
            let mut d = RenderPipelineDescriptor::new(function("rt6_fill_vertex"));
            d.fragment = Some(function(entry));
            d.color_attachments[0] = pipeline_of(pf, mask, blended).color_attachments[0];
            let e = dev
                .compile_render_pipeline(&d, &Budget::standard())
                .map(|_| ())
                .expect_err(&what);
            assert!(lacks_attachment(&e), "{what}: {e}");
        }
    }
}

/// An `RGB9E5Float` attachment on the 5080's device renders a `float` output through an
/// `R32_UINT` view and no conversion note; a `half` output is `NotImplemented(Format)`, and
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
