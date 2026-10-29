//! Extended-range render targets: a fragment function's `float` or `half` output to an attachment
//! of one of Apple's extended-range formats is written as the raw texel Metal stores, and the
//! attachment holds the format's raw texels.
//!
//! - On the CPU interpreter, the raw texel `ExtendedRange::clear_texel` gives a clear color (the
//!   runtime's clear value) is the texel the shader's encode stores for the same `half` values,
//!   over a grid of `half`s from below the format's range to past it, and NaN, the infinities and
//!   the largest finite `half`s.
//! - On every committed profile, a fragment function compiled for a pipeline with such an
//!   attachment (`rt6_fill_rgba32float`, `rt6_fill_rgba16float`) records the attachment
//!   (`StageIo::extended_range_colors`), keeps its interface variable's Metal type, declares the
//!   output variable as the raw format's unsigned components, bounds its emission by the plan's
//!   estimate, and links: the attachment takes the raw format, a write of every raw component
//!   under a full mask and of none under an empty one, no blend, and both modules pass L3.
//! - Refused, typed: a blended attachment (`NotImplemented(Format)`, by the lowering and, where
//!   the function does not write it, by link's format table; where the device orders fetch only
//!   through the pixel interlock, or through rasterization-order input attachments, the shader
//!   blends a `half4` or `float4` output into every form by the device's arithmetic, noted
//!   `ExtendedRangeBlendSource` or `ExtendedRangeFloatBlendSource`), a mask of some of the format's channels
//!   (`NotImplemented(Format)`; a padded format's are its raw words, so its mask is theirs), an
//!   integer or a `float` output (Metal's pipeline errors, `ColorAttachment`), a fragment compiled
//!   without the pipeline's attachments (`NeedsInput(ColorWrites)`).
//! - A framebuffer fetch of such an attachment through an input attachment (lavapipe) reads the
//!   raw format's `uint`s and decodes them: it links, and a fragment compiled without the
//!   pipeline's attachments does not.

use super::*;
use crate::error::{DescriptorError, Error, Needs, RuntimeInput};
use crate::metal::format::ClearRoute;
use crate::metal::ColorWriteMask;
use crate::pipeline::descriptor::ColorAttachmentDescriptor;
use crate::pipeline::link::{link, LinkInputs, RenderLinkInputs};
use crate::pipeline::lower_inputs::ExtendedRangeColor;
use crate::pipeline::relocate::relocate;
use crate::pipeline::stage_io::{IoNumeric, IoRole};
use crate::pipeline::testing::{compile, compile_with, engine_dir, fragment_inputs};
use crate::spirv::testing::{check_words_valid, spirv_dis};
use crate::vk;

/// The committed profiles of the venues: each desktop driver's and MoltenVK's.
const VENUES: [&str; 5] = [
    "amd-rdna3-radv",
    "intel-irisxe-rplp-anv",
    "lavapipe",
    "moltenvk-m3",
    "nvidia-rtx5080-laptop",
];

/// The six formats and their forms.
const FORMATS: [(PixelFormat, ExtendedRange); 6] = [
    (PixelFormat::BGR10XR, ExtendedRange::Bgr10 { srgb: false }),
    (
        PixelFormat::BGR10XRSRGB,
        ExtendedRange::Bgr10 { srgb: true },
    ),
    (PixelFormat::BGRA10XR, ExtendedRange::Bgra10 { srgb: false }),
    (
        PixelFormat::BGRA10XRSRGB,
        ExtendedRange::Bgra10 { srgb: true },
    ),
    (
        PixelFormat::RGB10A82PXR10,
        ExtendedRange::Planes { srgb: false },
    ),
    (
        PixelFormat::RGB10A82PXR10SRGB,
        ExtendedRange::Planes { srgb: true },
    ),
];

/// A kernel writing `float4` `src[gid.x + 32 gid.y]` (buffer 0) to `texture2d<float,
/// access::write>` slot 1 at `gid`.
const WRITE: &str = r#"define void @k(ptr addrspace(1) %src, ptr addrspace(1) %w, <2 x i32> %gid) {
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %row = shl i32 %y, 5
  %n = add i32 %x, %row
  %i = zext i32 %n to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %src, i64 %i
  %v = load <4 x float>, ptr addrspace(1) %p, align 16
  call void @air.write_texture_2d.v4f32(ptr addrspace(1) %w, <2 x i32> %gid, <4 x float> %v, i32 0, i32 2)
  ret void
}

declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"w"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}
"#;

/// The `half` bits of lane `l` of batch `b`: every fourth pattern from 0 to past 2.5, positive in
/// the even batches and negative in the odd ones, the first lanes NaN, the infinities, the
/// largest finite `half`s, zeros and one.
fn half_lane(b: u32, l: u32) -> u16 {
    const SPECIAL: [u16; 10] = [
        0x7e00, 0x7c00, 0xfc00, 0x7bff, 0xfbff, 0x0000, 0x8000, 0x3c00, 0x3e00, 0x4000,
    ];
    if b == 0 && (l as usize) < SPECIAL.len() {
        return SPECIAL[l as usize];
    }
    let bits = ((b / 2) * 4 * SIDE * SIDE + 4 * l + b % 4) as u16;
    match b % 2 {
        0 => bits,
        _ => bits | 0x8000,
    }
}

/// The `ExtendedRange::clear_texel` of `half` values is the texel the shader's encode stores
/// for them (the module docs): the runtime's clear value of a color is the kernel write's texel
/// of the color rounded to `half`, as Metal clears.
#[test]
fn clear_texels_are_the_kernel_writes_of_their_halves() {
    let Some(bc) = assemble("clear texels", WRITE) else {
        return;
    };
    let (stem, p) = profiles().into_iter().next().expect("profile");
    // An `_sRGB` form's write takes the device's `Pow`, which the interpreter does not model (and
    // its clear texel keeps Metal's table: the two may be a code apart).
    for form in FORMS.into_iter().filter(|f| !f.srgb()) {
        let mut inputs = LowerInputs::new();
        inputs.extended_range.insert(TARGET, form);
        inputs.storage_images.insert(
            TARGET,
            StorageImageChoice::new(form.raw_format(), StorageImageClass::Identity),
        );
        let what = format!("{form:?} on {stem}");
        let (plan, m) =
            typed_module_with(&bc, "k", &p, &inputs).unwrap_or_else(|e| panic!("{what}: {e}"));
        validate(&bc, &inputs, &p, &what);
        // Batches 0 to 3 reach every fourth pattern below 0x4400 (4.0) of either sign.
        for batch in 0..4 {
            let lanes = |n: u32| -> [u16; 4] { [0, 1, 2, 3].map(|c| half_lane(batch, 4 * n + c)) };
            let mut res = resources(&plan, 16, &[]);
            res.buffers[0].bytes = (0..SIDE * SIDE)
                .flat_map(&lanes)
                .flat_map(|h| crate::float16::f16_bits_to_f32(h).to_le_bytes())
                .collect();
            let images = [image(1, |_| [0; 4])];
            let (_, images) = run(&m, &res, &images);
            let target = texels_of(&images, 1);
            let layout = image(1, |_| [0; 4]);
            for n in 0..SIDE * SIDE {
                let color = lanes(n).map(|h| f64::from(crate::float16::f16_bits_to_f32(h)));
                let want = form.clear_texel(color, ClearRoute::ThroughHalf);
                let at = layout
                    .index(0, 0, 0, [n % SIDE, n / SIDE, 0])
                    .expect("texel");
                let raw = form.raw_components() as usize;
                assert_eq!(
                    target[at][..raw],
                    want[..raw],
                    "{what}: batch {batch} texel {n} of halves {:x?}",
                    lanes(n)
                );
            }
        }
    }
}

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

fn is_format_refusal(e: &Error) -> bool {
    matches!(e, Error::Unsupported(u) if u.feature == Feature::Format && u.needs == Needs::NotImplemented)
}

/// [`fragment_inputs`] with the framebuffer-fetch state the pipeline compile gives a function
/// that writes color 0 where it blends an extended-range attachment there, on a device that
/// orders fetch only through the interlock (`compile::fetch_inputs`) or through rasterization-order
/// input attachments.
fn blended_inputs(r: &RenderLinkInputs, p: &crate::device::profile::DeviceProfile) -> LowerInputs {
    let mut inputs = fragment_inputs(r, p);
    if blends_in_shader(p) {
        inputs.framebuffer_fetch = Some(crate::pipeline::FramebufferFetchInputs::of(
            [0],
            &r.color_attachments,
            false,
            false,
            1,
        ));
    }
    inputs
}

/// Whether `p` blends an extended-range attachment in the shader: in the storage form where it
/// orders fetch only through the interlock, over the input attachment where it orders fetch
/// through rasterization-order input attachments.
fn blends_in_shader(p: &crate::device::profile::DeviceProfile) -> bool {
    crate::lower::fragment::interlock_orders(p)
        || p.fields().fragment.rasterization_order_color_access
}

/// The fixture's `float4` and `half4` fills compile for each format's attachment on every venue,
/// write its raw texel through an unsigned output variable, and link with the raw format (the
/// module docs).
#[test]
fn fragment_outputs_to_extended_range_targets_write_raw_texels() {
    let Some(bc) = fixture() else {
        return;
    };
    let consts = FunctionConstantValues::defaults();
    for stem in VENUES {
        let p = venue(stem);
        let v = compile(&bc, "rt6_fill_vertex", &p).expect("vertex");
        for (pf, form) in FORMATS {
            for entry in ["rt6_fill_rgba32float", "rt6_fill_rgba16float"] {
                for (mask, written) in [
                    (ColorWriteMask::ALL, true),
                    (ColorWriteMask::NONE, false),
                    // `BGR10_XR` has no alpha: a mask of alpha alone keeps none of its channels.
                    (ColorWriteMask::ALPHA, false),
                ] {
                    let what = format!("{stem} {entry} {pf:?} {mask:?}");
                    let r = pipeline_of(pf, mask, false);
                    let inputs = fragment_inputs(&r, &p);
                    let f = compile_with(&bc, entry, &p, &inputs)
                        .unwrap_or_else(|e| panic!("{what}: {e}"));
                    let linked = link(&[&v, &f], &render(&r), &p);
                    // A mask of alpha alone keeps some of a four-channel format's channels.
                    if mask == ColorWriteMask::ALPHA && !matches!(form, ExtendedRange::Bgr10 { .. })
                    {
                        let e = linked.expect_err(&what);
                        assert!(is_format_refusal(&e), "{what}: {e}");
                        continue;
                    }
                    let l = linked.unwrap_or_else(|e| panic!("{what}: link: {e}"));
                    assert_eq!(
                        f.interface.io.extended_range_colors,
                        BTreeMap::from([(0, ExtendedRangeColor::new(form, false))]),
                        "{what}"
                    );
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
                    use vk::ColorComponentFlags as C;
                    let raw = match form.raw_components() {
                        1 => C::R,
                        2 => C::R | C::G,
                        _ => C::R | C::G | C::B | C::A,
                    };
                    assert_eq!(a.format, form.raw_format(), "{what}");
                    assert_eq!(
                        a.write_mask,
                        if written { raw } else { C::empty() },
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
                    let pointer = match form.raw_components() {
                        1 => "%_ptr_Output_uint = OpTypePointer Output %uint",
                        2 => "%_ptr_Output_v2uint = OpTypePointer Output %v2uint",
                        _ => "%_ptr_Output_v4uint = OpTypePointer Output %v4uint",
                    };
                    assert!(text.contains(pointer), "{what}\n{text}");
                }
            }
        }
    }
}

/// A bfloat attachment (`R16Bfloat`, `RG16Bfloat`, `RGBA16Bfloat`) takes the fixture's `float4`
/// and `half4` fills on every venue at either `textureWriteRoundingMode`: the function records
/// the attachment with the mode's rounding (`ExtendedRangeColor::nearest_even`), declares the raw
/// format's unsigned components, and links with the raw format, a write of the format's raw
/// components under a full mask and of none under an empty one, both modules passing L3. A
/// function compiled at the other mode does not link (`NeedsInput(ColorWrites)`), and neither
/// does a blend (`NotImplemented(Format)`), which Metal's validation refuses.
#[test]
fn fragment_outputs_to_bfloat_targets_write_raw_codes() {
    let Some(bc) = fixture() else {
        return;
    };
    use crate::metal::TextureWriteRoundingMode as Mode;
    let forms = [
        (PixelFormat::R16Bfloat, ExtendedRange::R16Bfloat),
        (PixelFormat::RG16Bfloat, ExtendedRange::Rg16Bfloat),
        (PixelFormat::RGBA16Bfloat, ExtendedRange::Rgba16Bfloat),
    ];
    for stem in VENUES {
        let p = venue(stem);
        let v = compile(&bc, "rt6_fill_vertex", &p).expect("vertex");
        for (pf, form) in forms {
            for entry in ["rt6_fill_rgba32float", "rt6_fill_rgba16float"] {
                for (mode, other) in [
                    (Mode::TowardZero, Mode::NearestEven),
                    (Mode::NearestEven, Mode::TowardZero),
                ] {
                    for (mask, written) in
                        [(ColorWriteMask::ALL, true), (ColorWriteMask::NONE, false)]
                    {
                        let what = format!("{stem} {entry} {pf:?} {mode:?} {mask:?}");
                        let mut r = pipeline_of(pf, mask, false);
                        r.texture_write_rounding_mode = mode;
                        let f = compile_with(&bc, entry, &p, &fragment_inputs(&r, &p))
                            .unwrap_or_else(|e| panic!("{what}: {e}"));
                        let write = f.interface.io.extended_range_colors.get(&0).copied();
                        assert_eq!(
                            write,
                            Some(ExtendedRangeColor::new(form, false).with_rounding(mode)),
                            "{what}"
                        );
                        assert_eq!(
                            write.map(|w| w.nearest_even),
                            Some(mode == Mode::NearestEven),
                            "{what}"
                        );
                        let l = link(&[&v, &f], &render(&r), &p)
                            .unwrap_or_else(|e| panic!("{what}: link: {e}"));
                        let g = l.graphics.as_ref().expect("graphics");
                        let a = g.color_attachments[0].expect("attachment");
                        use vk::ColorComponentFlags as C;
                        let raw = match form.raw_components() {
                            1 => C::R,
                            2 => C::R | C::G,
                            _ => C::R | C::G | C::B | C::A,
                        };
                        assert_eq!(a.format, form.raw_format(), "{what}");
                        assert_eq!(
                            a.write_mask,
                            if written { raw } else { C::empty() },
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
                        let pointer = match form.raw_components() {
                            1 => "%_ptr_Output_uint = OpTypePointer Output %uint",
                            2 => "%_ptr_Output_v2uint = OpTypePointer Output %v2uint",
                            _ => "%_ptr_Output_v4uint = OpTypePointer Output %v4uint",
                        };
                        assert!(text.contains(pointer), "{what}\n{text}");
                        // The function of the other mode's pipeline.
                        let mut o = r.clone();
                        o.texture_write_rounding_mode = other;
                        let e = link(&[&v, &f], &render(&o), &p).expect_err(&what);
                        assert!(
                            matches!(e, Error::NeedsInput(RuntimeInput::ColorWrites)),
                            "{what} at {other:?}: {e}"
                        );
                    }
                }
            }
            // Blended: refused by the lowering (by the storage form's plan where the device
            // orders fetch only through the interlock), and by link's format table where
            // unwritten.
            let what = format!("{stem} {pf:?} blended");
            let r = pipeline_of(pf, ColorWriteMask::ALL, true);
            for entry in ["rt6_fill_rgba32float", "rt6_fill_rgba16float"] {
                let e = compile_with(&bc, entry, &p, &blended_inputs(&r, &p)).expect_err(&what);
                assert!(is_format_refusal(&e), "{what} {entry}: {e}");
            }
            let e = link(&[&v], &render(&r), &p).expect_err(&what);
            assert!(is_format_refusal(&e), "{what}, unwritten: {e}");
        }
    }
}

/// A padded attachment (`R10Unorm_X6`, `RG10Unorm_X12`, and their `_sRGB` twins, the store's code
/// through the twin's table) links under every write mask, the fixture's `float4` fill writing the
/// raw words its mask keeps: a channel is a raw word, and one the mask leaves out keeps the word
/// the pass loaded. Each module passes L3.
#[test]
fn padded_targets_write_the_words_their_mask_keeps() {
    let Some(bc) = fixture() else {
        return;
    };
    use vk::ColorComponentFlags as C;
    for stem in VENUES {
        let p = venue(stem);
        let v = compile(&bc, "rt6_fill_vertex", &p).expect("vertex");
        for (pf, form) in [
            (PixelFormat::R10UnormX6, ExtendedRange::R10X6),
            (PixelFormat::RG10UnormX12, ExtendedRange::Rg10X12),
            (PixelFormat::R10UnormX6SRGB, ExtendedRange::R10X6Srgb),
            (PixelFormat::RG10UnormX12SRGB, ExtendedRange::Rg10X12Srgb),
        ] {
            let channels = match form.raw_components() {
                1 => C::R,
                _ => C::R | C::G,
            };
            for (mask, kept) in [
                (ColorWriteMask::ALL, C::R | C::G),
                (ColorWriteMask::NONE, C::empty()),
                (ColorWriteMask::RED, C::R),
                (ColorWriteMask::GREEN, C::G),
                (ColorWriteMask::RED.union(ColorWriteMask::BLUE), C::R),
                (ColorWriteMask::ALPHA, C::empty()),
            ] {
                let what = format!("{stem} {pf:?} {mask:?}");
                let r = pipeline_of(pf, mask, false);
                let f = compile_with(&bc, "rt6_fill_rgba32float", &p, &fragment_inputs(&r, &p))
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                let l = link(&[&v, &f], &render(&r), &p)
                    .unwrap_or_else(|e| panic!("{what}: link: {e}"));
                let g = l.graphics.as_ref().expect("graphics");
                let a = g.color_attachments[0].expect("attachment");
                assert_eq!(a.format, form.raw_format(), "{what}");
                assert_eq!(a.write_mask, kept & channels, "{what}");
                for s in [&*v, &*f] {
                    let w = relocate(s, &l.table).expect("relocate");
                    check_words_valid(&w, &s.requirements, &p)
                        .unwrap_or_else(|e| panic!("{what}: L3: {e}"));
                }
            }
        }
    }
}

/// What the module docs list as refused is refused, typed.
#[test]
fn unmodelled_extended_range_targets_are_refused() {
    let Some(bc) = fixture() else {
        return;
    };
    for stem in ["lavapipe", "moltenvk-m3"] {
        let p = venue(stem);
        let v = compile(&bc, "rt6_fill_vertex", &p).expect("vertex");
        for (pf, _) in FORMATS {
            let what = format!("{stem} {pf:?}");
            // Blended: where the device orders fetch only through the interlock, or through
            // rasterization-order input attachments, the shader blends the `float4` output by the
            // device's arithmetic, into every form; elsewhere the lowering refuses it.
            let r = pipeline_of(pf, ColorWriteMask::ALL, true);
            let blended = compile_with(&bc, "rt6_fill_rgba32float", &p, &blended_inputs(&r, &p));
            match blends_in_shader(&p) {
                true => {
                    let a = blended.unwrap_or_else(|e| panic!("{what} blended: {e}"));
                    assert!(
                        a.precision.contains(
                            &crate::pipeline::artifact::PrecisionNote::ExtendedRangeFloatBlendSource
                        ),
                        "{what} blended: {:?}",
                        a.precision
                    );
                }
                false => {
                    let e = blended.expect_err(&what);
                    assert!(is_format_refusal(&e), "{what} blended: {e}");
                }
            }
            // Blended but not written (no fragment function): link's format table refuses the
            // blend.
            let e = link(&[&v], &render(&r), &p).expect_err(&what);
            assert!(is_format_refusal(&e), "{what} blended, unwritten: {e}");
            // A mask of some channels.
            let r = pipeline_of(pf, ColorWriteMask::RED, false);
            let f = compile_with(&bc, "rt6_fill_rgba32float", &p, &fragment_inputs(&r, &p))
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            let e = link(&[&v, &f], &render(&r), &p).expect_err(&what);
            assert!(is_format_refusal(&e), "{what} red mask: {e}");
            // Metal's pipeline errors: an integer output, a `float` output to a format of
            // three or four channels.
            let r = pipeline_of(pf, ColorWriteMask::ALL, false);
            for entry in ["rt6_fill_rgba8uint", "rt6_fill_r32float"] {
                let f = compile_with(&bc, entry, &p, &fragment_inputs(&r, &p))
                    .unwrap_or_else(|e| panic!("{what} {entry}: {e}"));
                let e = link(&[&v, &f], &render(&r), &p).expect_err(&what);
                assert!(
                    matches!(
                        e,
                        Error::InvalidDescriptor(DescriptorError::ColorAttachment { index: 0 })
                    ),
                    "{what} {entry}: {e}"
                );
            }
            // Compiled without the pipeline's attachments.
            let f = compile(&bc, "rt6_fill_rgba32float", &p).expect("fragment");
            let e = link(&[&v, &f], &render(&r), &p).expect_err(&what);
            assert!(
                matches!(e, Error::NeedsInput(RuntimeInput::ColorWrites)),
                "{what} standalone: {e}"
            );
        }
    }
    // A framebuffer fetch of such an attachment (`exf_fetch_blend` reads `float4 [[color(0)]]`
    // and writes color 0 unblended), on lavapipe, which reads it through an input attachment:
    // of the raw format's `uint`s, decoded in the shader. The fragment records the attachment,
    // links, and passes L3; compiled without the pipeline's attachments, it does not link.
    let Ok(fetch) = llvm_assemble(&engine_dir().join("exf_fetch_blend.ll")) else {
        return;
    };
    let p = venue("lavapipe");
    let v = compile(&bc, "rt6_fill_vertex", &p).expect("vertex");
    for (pf, form) in FORMATS {
        let what = format!("{pf:?} fetch");
        let r = pipeline_of(pf, ColorWriteMask::ALL, false);
        let f = compile_with(&fetch, "exf_fetch_blend", &p, &fragment_inputs(&r, &p))
            .unwrap_or_else(|e| panic!("{what}: {e}"));
        assert_eq!(
            f.interface.io.extended_range_colors,
            BTreeMap::from([(0, ExtendedRangeColor::new(form, false))]),
            "{what}"
        );
        let l = link(&[&v, &f], &render(&r), &p).unwrap_or_else(|e| panic!("{what}: link: {e}"));
        let g = l.graphics.as_ref().expect("graphics");
        let inputs: Vec<(u32, crate::pipeline::sym::SampledType)> = g
            .input_attachments
            .iter()
            .map(|i| (i.color_index, i.format_class))
            .collect();
        assert_eq!(
            inputs,
            [(0, crate::pipeline::sym::SampledType::Uint)],
            "{what}"
        );
        assert_eq!(
            g.color_attachments[0].expect("attachment").format,
            form.raw_format()
        );
        for s in [&*v, &*f] {
            let w = relocate(s, &l.table).expect("relocate");
            check_words_valid(&w, &s.requirements, &p)
                .unwrap_or_else(|e| panic!("{what}: L3: {e}"));
        }
        let f = compile(&fetch, "exf_fetch_blend", &p).expect("fragment");
        let e = link(&[&v, &f], &render(&r), &p).expect_err(&what);
        assert!(
            matches!(e, Error::NeedsInput(RuntimeInput::ColorWrites)),
            "{what} standalone: {e}"
        );
    }
}
