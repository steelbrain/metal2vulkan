//! Bicubic samples of an extended-range texture are Metal's within the device's arithmetic.
//!
//! On the CPU interpreter, on the lavapipe and MoltenVK-on-M3 profiles, each sample of the
//! extended-range bicubic table (`metal2vulkan_devtools::xr_bicubic`: an Apple M3 Max on macOS
//! 26.7, through the owned fixture `xr_bicubic`) is Metal's lanes within the device's arithmetic
//! (`PrecisionNote::BicubicFilterArithmetic`), through the same kernel: constexpr bicubic samplers
//! under every address mode and both opaque borders, with pixel coordinates and with a constant
//! offset, and a runtime sampler whose filters are `Cubic` (`LoweringInputs::cubic_samplers`), with
//! `float` and `half` results. The footprint, the addressing, the weights' polynomial, the small
//! level's linear filter and the format's clamp are Metal's; the sum is the device's `f32`
//! multiply and add of the decoded taps under the weights computed from the fraction, where Metal
//! sums fixed-point weights exactly onto the lane's grid, and a `half` result is the device's
//! narrowing. Each compile is validated, and its plan's estimate bounds it. What no case measured
//! is refused, `NotImplemented(SamplerBicubicFilter)`: a padded or bfloat form's bicubic sample, a
//! swizzled texture's, and a gather through a bicubic sampler.

use metal2vulkan_devtools::xr_bicubic::{self, KERNELS, SIZES};

use super::samples::{form_of, run_samples, sample_kernel, sampled_state, words, Texels, Through};
use super::*;
use crate::bind::SamplerDescriptor;
use crate::error::Unsupported;
use crate::metal::sampler::SamplerWord;
use crate::metal::{SamplerAddressMode, SamplerMinMagFilter};
use crate::pipeline::testing::engine_dir;
use crate::pipeline::PrecisionNote;

/// The runtime sampler slot of the fixture's `xrb_runtime` kernels.
const SAMPLER: SlotRef = SlotRef {
    stage: Stage::Kernel,
    class: SlotClass::Sampler,
    index: 0,
};

/// Every sample of the table of the raw format `raw`, through each of the fixture's kernels on
/// each profile, is Metal's lanes, one step from them at a marked tie.
fn replay(raw: u32) {
    let bc = match llvm_assemble(&engine_dir().join("xr_bicubic.ll")) {
        Ok(bc) => bc,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected xr_bicubic: {why}"),
    };
    let form = form_of(&raw.to_string());
    let textures = xr_bicubic::textures();
    let table = xr_bicubic::table();
    // The probe's runtime sampler.
    let mut cubic = SamplerDescriptor::new();
    cubic.min_filter = SamplerMinMagFilter::Cubic;
    cubic.mag_filter = SamplerMinMagFilter::Cubic;
    cubic.s_address_mode = SamplerAddressMode::ClampToEdge;
    cubic.t_address_mode = SamplerAddressMode::ClampToEdge;
    let word = SamplerWord::of(&cubic).0;
    let (mut lanes, mut apart) = (0, 0);
    for (stem, p) in profiles() {
        for kernel in KERNELS {
            let half = kernel.ends_with("_half");
            let mut inputs = LowerInputs::new();
            inputs.extended_range.insert(SOURCE, form);
            if kernel.starts_with("xrb_runtime") {
                inputs
                    .cubic_samplers
                    .insert(SAMPLER, word & SamplerWord::CUBIC_STATE);
            }
            let what = format!("{stem} {raw} {kernel}");
            let a = validate_entry(&bc, kernel, &inputs, &p, &what);
            assert_eq!(
                a.strategies
                    .entries
                    .get("sampler.extended_range.bicubic")
                    .map(|u| u.uses),
                Some(1),
                "{what}"
            );
            assert!(
                a.precision
                    .contains(&PrecisionNote::BicubicFilterArithmetic),
                "{what}"
            );
            let (plan, m) = typed_module_with(&bc, kernel, &p, &inputs)
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            // A runtime sampler's word, where the variant still reads one.
            let bound = plan.families.sampler.slots.get(&0).map(|_| word);
            for size in SIZES {
                let samples = &table[&(raw, size, kernel.to_string())];
                let tex = Texels {
                    width: size.0,
                    codes: &textures[&(raw, size)],
                };
                let coords: Vec<[f32; 2]> = samples.iter().map(|s| s.at).collect();
                let what = format!("{what} {}x{}", size.0, size.1);
                let got = run_samples(&plan, &m, (form, &tex), (&coords, half, bound), None, &what);
                for (s, got) in samples.iter().zip(got) {
                    for (c, (&got, &want)) in got.iter().zip(&s.lanes).enumerate() {
                        let what = format!("{what} at {:?} lane {c}", s.at);
                        lanes += 1;
                        apart += usize::from(got != want);
                        assert!(
                            xr_bicubic::near(got, want, xr_bicubic::grid(raw, c), half),
                            "{what}: {got:#x}, Metal's {want:#x}"
                        );
                    }
                }
            }
        }
    }
    eprintln!("{raw}: {lanes} lanes, {apart} of them apart from Metal's, within its arithmetic");
}

#[test]
fn two_plane_samples_are_metals() {
    replay(550);
}

#[test]
fn bgra10_samples_are_metals() {
    replay(552);
}

#[test]
fn bgr10_samples_are_metals() {
    replay(554);
}

/// A padded or bfloat form's bicubic sample, a swizzled texture's, and a gather through a bicubic
/// sampler are refused, `NotImplemented(SamplerBicubicFilter)`, but a two-plane form's gather, which
/// is refused under any sampler, `NotImplemented(Format)`; each compiles without the slot's
/// variant.
#[test]
fn unmeasured_bicubic_samples_are_refused() {
    let fixture = match llvm_assemble(&engine_dir().join("xr_bicubic.ll")) {
        Ok(bc) => bc,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected xr_bicubic: {why}"),
    };
    // Bicubic minification and magnification, clamp to edge.
    let cubic_edge = words(false, 1, false, 0) | 2 << 9 | 2 << 11;
    let Some(gather) = assemble(
        "bicubic gather",
        &sample_kernel(Through::Constexpr(cubic_edge), false, (0, 0), true),
    ) else {
        return;
    };
    let unmeasured = [
        ExtendedRange::R10X6,
        ExtendedRange::R10X6Srgb,
        ExtendedRange::Rg10X12,
        ExtendedRange::Rg10X12Srgb,
        ExtendedRange::R16Bfloat,
        ExtendedRange::Rg16Bfloat,
        ExtendedRange::Rgba16Bfloat,
    ];
    let mut cases = Vec::new();
    for form in unmeasured {
        cases.push((format!("{form:?}"), &fixture, "xrb_edge", form, false));
    }
    for form in FORMS {
        cases.push((
            format!("{form:?} swizzled"),
            &fixture,
            "xrb_edge",
            form,
            true,
        ));
        cases.push((format!("{form:?} gather"), &gather, "k", form, false));
    }
    for (stem, p) in profiles() {
        for (what, bc, entry, form, swizzled) in &cases {
            let feature = match (*entry, form) {
                ("k", ExtendedRange::Planes { .. }) => Feature::Format,
                _ => Feature::SamplerBicubicFilter,
            };
            let want = Unsupported::not_implemented(feature);
            let mut inputs = LowerInputs::new();
            inputs.extended_range.insert(SOURCE, *form);
            if *swizzled {
                inputs.swizzled_storage.insert(SOURCE);
            }
            match typed_module_with(bc, entry, &p, &inputs) {
                Err(crate::error::Error::Unsupported(u)) => assert_eq!(
                    (&u.feature, &u.needs),
                    (&want.feature, &want.needs),
                    "{what} on {stem}"
                ),
                other => panic!("{what} on {stem}: {:?}", other.map(|_| ())),
            }
            typed_module_with(bc, entry, &p, &LowerInputs::new())
                .unwrap_or_else(|e| panic!("{what} on {stem} without the variant: {e}"));
        }
    }
}

/// A texture bound in an extended-range format joins its slot's variant through a constexpr
/// bicubic sampler, and through a runtime sampler whose filters are `Cubic`, whose slot joins the
/// cubic-sampler variant too, under the word the interpreter's replay compiles: the pipeline as
/// compiled plans that variant, whose pipeline plans the same key. On every committed profile.
#[test]
fn a_bicubic_sample_s_slots_join_the_variants() {
    let bc = match llvm_assemble(&engine_dir().join("xr_bicubic.ll")) {
        Ok(bc) => bc,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected xr_bicubic: {why}"),
    };
    let formats = [
        (PixelFormat::RGB10A82PXR10, 550),
        (PixelFormat::RGB10A82PXR10SRGB, 551),
        (PixelFormat::BGRA10XR, 552),
        (PixelFormat::BGRA10XRSRGB, 553),
        (PixelFormat::BGR10XR, 554),
        (PixelFormat::BGR10XRSRGB, 555),
    ];
    let mut cubic = SamplerDescriptor::new();
    cubic.min_filter = SamplerMinMagFilter::Cubic;
    cubic.mag_filter = SamplerMinMagFilter::Cubic;
    cubic.s_address_mode = SamplerAddressMode::ClampToEdge;
    cubic.t_address_mode = SamplerAddressMode::ClampToEdge;
    let word = SamplerWord::of(&cubic).0 & SamplerWord::CUBIC_STATE;
    let mut planned = 0;
    for (stem, dev) in devices() {
        for kernel in ["xrb_edge", "xrb_runtime"] {
            let plain =
                pipeline_entry(&dev, &bc, kernel, &LowerInputs::new()).expect("a sampling kernel");
            let cubic_samplers = match kernel {
                "xrb_runtime" => BTreeMap::from([(SAMPLER, word)]),
                _ => BTreeMap::new(),
            };
            for (pf, raw) in formats {
                let what = format!("{kernel} {pf:?} on {stem}");
                let shape = sampled_state(pf, cubic).shape();
                let plan = plain
                    .plan(&dev, &shape)
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                let form = form_of(&raw.to_string());
                assert_eq!(
                    plan.variant.extended_range,
                    BTreeMap::from([(SOURCE, form)]),
                    "{what}"
                );
                assert_eq!(plan.variant.cubic_samplers, cubic_samplers, "{what}");
                let key = plan.variant.lowering_inputs(Stage::Kernel);
                let variant =
                    pipeline_entry(&dev, &bc, kernel, &key).expect("the variant's inputs");
                let again = variant
                    .plan(&dev, &shape)
                    .unwrap_or_else(|e| panic!("{what}: the variant: {e}"));
                assert_eq!(again.variant, plan.variant, "{what}");
                planned += 1;
            }
        }
    }
    assert!(planned > 0);
}

/// A runtime sampler whose filters are `Cubic` and whose state Metal's bicubic filter is not
/// measured under whatever a function samples (a mip filter, anisotropy, a minimum reduction) is
/// refused by the plan of the pipeline as compiled, `NotImplemented(SamplerBicubicFilter)`, before
/// the plan names a variant: a dispatch of it compiles nothing, where the variant it would name
/// refuses the same word only once compiled, and the runtime keeps no compile that refused. At an
/// extended-range slot (which refused every `Cubic` word at planning before Metal's bicubic filter
/// of its texels was built) as at a slot of another format. On every committed profile, the
/// lowering refuses that variant by the same rule (`metal::bicubic::runtime_cubic_measured`), a
/// device without `samplerAnisotropy` (`vk12-floor`) an anisotropic one alike.
#[test]
fn an_unmeasured_cubic_sampler_is_refused_before_any_variant() {
    let bc = match llvm_assemble(&engine_dir().join("xr_bicubic.ll")) {
        Ok(bc) => bc,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected xr_bicubic: {why}"),
    };
    let mut cubic = SamplerDescriptor::new();
    cubic.min_filter = SamplerMinMagFilter::Cubic;
    cubic.mag_filter = SamplerMinMagFilter::Cubic;
    cubic.s_address_mode = SamplerAddressMode::ClampToEdge;
    cubic.t_address_mode = SamplerAddressMode::ClampToEdge;
    let mut mipmapped = cubic;
    mipmapped.mip_filter = crate::metal::SamplerMipFilter::Nearest;
    let mut anisotropic = cubic;
    anisotropic.max_anisotropy = 4;
    let mut minimum = cubic;
    minimum.reduction_mode = crate::metal::SamplerReductionMode::Minimum;
    let unmeasured = [
        ("a mip filter", mipmapped),
        ("anisotropy", anisotropic),
        ("a minimum reduction", minimum),
    ];
    let formats = [
        (PixelFormat::BGRA10XR, Some(form_of("552"))),
        (PixelFormat::RGB10A82PXR10, Some(form_of("550"))),
        (PixelFormat::RGBA8Unorm, None),
    ];
    let want = Unsupported::not_implemented(Feature::SamplerBicubicFilter);
    let mut refused = 0;
    for (stem, dev) in devices() {
        let plain = pipeline_entry(&dev, &bc, "xrb_runtime", &LowerInputs::new())
            .expect("a sampling kernel");
        // The measured word plans its variant (`a_bicubic_sample_s_slots_join_the_variants`).
        let measured = sampled_state(PixelFormat::BGRA10XR, cubic).shape();
        plain
            .plan(&dev, &measured)
            .unwrap_or_else(|e| panic!("the measured cubic word on {stem}: {e}"));
        for (state, sampler) in unmeasured {
            let word = SamplerWord::of(&sampler).0;
            assert!(word & SamplerWord::CUBIC != 0, "{state}");
            for (pf, form) in formats {
                let what = format!("a cubic sampler with {state}, {pf:?}, on {stem}");
                match plain.plan(&dev, &sampled_state(pf, sampler).shape()) {
                    Err(crate::error::BindError::Unsupported(u)) => {
                        assert_eq!(
                            (&u.feature, &u.needs),
                            (&want.feature, &want.needs),
                            "{what}"
                        )
                    }
                    other => panic!("{what}: planned {:?}", other.map(|p| p.variant)),
                }
                // The variant the word would name refuses it.
                let mut inputs = LowerInputs::new();
                inputs
                    .cubic_samplers
                    .insert(SAMPLER, word & SamplerWord::CUBIC_STATE);
                if let Some(form) = form {
                    inputs.extended_range.insert(SOURCE, form);
                }
                match typed_module_with(&bc, "xrb_runtime", dev.profile(), &inputs) {
                    Err(crate::error::Error::Unsupported(u)) => assert_eq!(
                        (&u.feature, &u.needs),
                        (&want.feature, &want.needs),
                        "{what}: the variant"
                    ),
                    other => panic!("{what}: the variant: {:?}", other.map(|_| ())),
                }
                refused += 1;
            }
        }
    }
    eprintln!("{refused} unmeasured cubic words refused");
    assert!(refused > 0, "no profile checked");
}
