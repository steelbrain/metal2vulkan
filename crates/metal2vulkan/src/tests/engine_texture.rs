//! Texture reads, writes, fences, queries, and null checks end to end.
//!
//! These are structural and L3 tests: every compiled module passes `spirv-val` under the
//! profile (`G-val`), and the tests check the declarations, strategies, facts, and interface the
//! lowering chose. Texel values are checked on the CPU interpreter's image model in
//! `src/tests/engine_texture_interp.rs`.
//!
//! - The storage-image format follows the slot's `StorageImageChoice`: without one, a formatless
//!   profile declares `Unknown` and every other profile needs the input.
//! - The bgra8/rgba8 storage-texture pair compiles to distinct keys and modules wherever bgra8 is
//!   not storage-capable, the bgra8 one swizzling (the golden hashes check the other half).
//! - Texel buffers are buffer images with their range fact; cube reads use the 2D-array view
//!   (`ImageViewShape`); `is_null_texture` reads a presence constant, and the census counts the
//!   code only a nil texture's arm runs apart.
//! - Writes are guarded exactly when `robustImageAccess` is missing; reads clamp exactly under
//!   containment without it.
//! - The image-operand legality matrix: the refusals of unmodelled forms.
//!
//! `llvm-as` assembles the fixtures; without it the tests skip visibly.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::budget::SPIRV_WORDS;
use crate::device::profile::DeviceProfile;
use crate::device::requirements::VkFeature;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{Error, Feature, LowerFamily, RuntimeInput, Unsupported};
use crate::library::{FunctionArtifact, FunctionConstantValues};
use crate::lower::select::LowerInputs;
use crate::lower::strategy::FamilyStrategy;
use crate::metal::format::storage_image_choice;
use crate::metal::{PixelFormat, TextureType};
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::compile::{compile_bitcode, EntryRef};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::interface::SlotInterface;
use crate::pipeline::lower_inputs::StorageImageClass;
use crate::pipeline::sym::{SlotClass, SlotRef, SpecKey, Stage};
use crate::pipeline::testing::{
    compile, compile_kept, engine_dir, exact_maths, link_and_validate, llvm_assemble, meter,
    profile, AssembleError,
};
use crate::spirv::testing::{committed_profiles, committed_profiles_with};

const DATALAYOUT: &str = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32";

fn fixture(name: &str) -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join(format!("{name}.ll"))) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}: {why}"),
    }
}

fn assemble(name: &str, body: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let text = format!(
        "target datalayout = \"{DATALAYOUT}\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n{body}"
    );
    let dir = std::env::temp_dir().join(format!(
        "m2v-texture-{}-{}-{name}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("k.ll");
    std::fs::write(&ll, &text).expect("write");
    let r = llvm_assemble(&ll);
    let _ = std::fs::remove_dir_all(&dir);
    match r {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}: {why}\n{text}"),
    }
}

fn texture_slot(index: u32) -> SlotRef {
    SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Texture,
        index,
    }
}

fn with_inputs(
    bc: &[u8],
    entry: &str,
    p: &DeviceProfile,
    inputs: &LowerInputs,
) -> Result<std::sync::Arc<FunctionArtifact>, Error> {
    compile_bitcode(
        bc,
        EntryRef::Name(entry),
        &FunctionConstantValues::defaults(),
        p,
        inputs,
        &mut meter(),
        SPIRV_WORDS,
    )
}

fn ok(
    a: Result<std::sync::Arc<FunctionArtifact>, Error>,
    p: &DeviceProfile,
    what: &str,
) -> std::sync::Arc<FunctionArtifact> {
    let a = a.unwrap_or_else(|e| panic!("{what}: compile: {e}"));
    link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
    a
}

/// The opcodes of a SPIR-V module, in order.
fn opcodes(words: &[u32]) -> Vec<u16> {
    let mut out = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let n = (words[i] >> 16) as usize;
        out.push((words[i] & 0xFFFF) as u16);
        i += n.max(1);
    }
    out
}

const OP_VECTOR_SHUFFLE: u16 = 79;
const OP_IMAGE_QUERY_SIZE: u16 = 104;
const OP_IMAGE_QUERY_LEVELS: u16 = 106;
const OP_IMAGE_FETCH: u16 = 95;
const OP_IMAGE_READ: u16 = 98;
const OP_IMAGE_WRITE: u16 = 99;

fn uses(a: &FunctionArtifact, name: &str) -> u32 {
    a.strategies.entries.get(name).map_or(0, |u| u.uses)
}

/// Without a choice, a formatless profile declares the storage image `Unknown`, and any other
/// profile needs the slot's format; with the planner's choice every profile compiles.
#[test]
fn storage_format_follows_the_choice_and_the_profile() {
    let Some(bc) = fixture("storage_texture_write") else {
        return;
    };
    let slot = texture_slot(0);
    let mut declared = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        let images = &p.fields().images;
        let r = compile(&bc, "storage_texture_write", &p);
        if images.storage_write_without_format {
            let a = ok(r, &p, &stem);
            assert!(
                a.facts
                    .contains(&RuntimeFact::StorageTextureFormat { slot }),
                "{stem}"
            );
        } else {
            assert_eq!(
                r.err(),
                Some(Error::NeedsInput(RuntimeInput::StorageTextureFormat {
                    slot
                })),
                "{stem}"
            );
        }
        // A profile whose format table lacks rgba8 storage has no rgba8 choice to compile.
        let Ok(choice) = storage_image_choice(PixelFormat::RGBA8Unorm, &p) else {
            continue;
        };
        if choice.vk_format != crate::vk::Format::UNDEFINED {
            declared += 1;
        }
        let mut inputs = LowerInputs::new();
        inputs.storage_images.insert(slot, choice);
        let a = ok(
            with_inputs(&bc, "storage_texture_write", &p, &inputs),
            &p,
            &format!("{stem} rgba8"),
        );
        assert_eq!(
            uses(&a, "texture.write") + uses(&a, "texture.write_guarded"),
            1
        );
        match a.interface.slots.get(&slot) {
            Some(SlotInterface::Texture(t)) => {
                assert_eq!(t.texture_type, TextureType::Type2D);
                assert!(t.access.write && !t.access.read);
            }
            other => panic!("{stem}: slot 0 interface {other:?}"),
        }
    }
    assert!(
        declared > 0,
        "no committed profile declares a storage format"
    );
}

/// The required pair: on every profile where bgra8 is not storage-capable, a bgra8 and an rgba8
/// storage texture give distinct compile keys and modules, and only the bgra8 one swizzles. The
/// sRGB class encodes in the shader by the device's conversion (the MSL transfer through its
/// `Pow`), declaring no Metal table and recording `DevicePackConversion`.
#[test]
fn bgra8_and_rgba8_storage_textures_compile_to_distinct_modules() {
    let Some(bc) = fixture("storage_texture_write") else {
        return;
    };
    let slot = texture_slot(0);
    let (mut swizzled, mut srgb, mut srgb_bgra) = (0, 0, 0);
    for (stem, p) in committed_profiles().expect("profiles") {
        let variant = |pf: PixelFormat| {
            let choice = storage_image_choice(pf, &p).ok()?;
            let mut inputs = LowerInputs::new();
            inputs.storage_images.insert(slot, choice);
            let a = ok(
                with_inputs(&bc, "storage_texture_write", &p, &inputs),
                &p,
                &format!("{stem} {pf:?}"),
            );
            Some((choice, a))
        };
        let Some((rgba_choice, rgba)) = variant(PixelFormat::RGBA8Unorm) else {
            continue;
        };
        assert_eq!(rgba_choice.class, StorageImageClass::Identity, "{stem}");
        let count = |a: &FunctionArtifact, op: u16| {
            opcodes(&a.module.words)
                .iter()
                .filter(|o| **o == op)
                .count()
        };
        if let Some((choice, bgra)) = variant(PixelFormat::BGRA8Unorm) {
            if choice.class == StorageImageClass::SwizzleBgra {
                swizzled += 1;
                assert_ne!(bgra.key, rgba.key, "{stem}");
                assert_ne!(bgra.module.words, rgba.module.words, "{stem}");
                assert!(
                    count(&bgra, OP_VECTOR_SHUFFLE) > count(&rgba, OP_VECTOR_SHUFFLE),
                    "{stem}"
                );
            }
        }
        let table =
            |a: &FunctionArtifact| super::engine_fragment::holds_srgb_table(&a.module.words);
        let srgb_rgba = variant(PixelFormat::RGBA8UnormSRGB);
        if let Some((choice, s)) = &srgb_rgba {
            if choice.class == StorageImageClass::SrgbEncode {
                srgb += 1;
                assert!(!s.precision.contains(&PrecisionNote::DeviceTranscendentals));
                assert!(
                    s.precision.contains(&PrecisionNote::DevicePackConversion),
                    "{stem}"
                );
                assert_ne!(s.key, rgba.key, "{stem}");
                // No Metal table: the encode is the device's.
                assert!(!table(s) && !table(&rgba), "{stem}");
            }
        }
        // bgra8 sRGB through rgba8: the device's encode, and red and blue exchanged after it.
        if let Some((choice, b)) = variant(PixelFormat::BGRA8UnormSRGB) {
            if choice.class == StorageImageClass::SrgbEncodeBgra {
                srgb_bgra += 1;
                assert!(!table(&b), "{stem}");
                assert_ne!(b.key, rgba.key, "{stem}");
                if let Some((c, s)) = &srgb_rgba {
                    if c.class == StorageImageClass::SrgbEncode {
                        assert_ne!(b.key, s.key, "{stem}");
                        assert!(
                            count(&b, OP_VECTOR_SHUFFLE) > count(s, OP_VECTOR_SHUFFLE),
                            "{stem}"
                        );
                    }
                }
            }
        }
    }
    assert!(swizzled > 0, "no committed profile exercises SwizzleBgra");
    assert!(srgb > 0, "no committed profile exercises SrgbEncode");
    assert!(
        srgb_bgra > 0,
        "no committed profile exercises SrgbEncodeBgra"
    );
}

/// A texture buffer is a buffer image; both slots carry the texel-buffer range fact and the storage
/// one its format fact.
#[test]
fn texel_buffers_are_buffer_images() {
    let Some(bc) = fixture("texture_ops") else {
        return;
    };
    let p = profile("moltenvk-m3", DevicePolicy::default()).expect("profile");
    let a = ok(compile(&bc, "texel_buffers", &p), &p, "texel_buffers");
    for i in [0, 1] {
        assert!(a.facts.contains(&RuntimeFact::TexelBufferRange {
            slot: texture_slot(i)
        }));
        match a.interface.slots.get(&texture_slot(i)) {
            Some(SlotInterface::Texture(t)) => {
                assert_eq!(t.texture_type, TextureType::TextureBuffer)
            }
            other => panic!("slot {i}: {other:?}"),
        }
    }
    assert!(a.facts.contains(&RuntimeFact::StorageTextureFormat {
        slot: texture_slot(1)
    }));
    assert!(!a.facts.contains(&RuntimeFact::StorageTextureFormat {
        slot: texture_slot(0)
    }));
    let ops = opcodes(&a.module.words);
    assert!(ops.contains(&OP_IMAGE_FETCH) && ops.contains(&OP_IMAGE_WRITE));
}

/// A `half` texel's NaN is the device's decode: Metal hands back the texel's own NaN, and Vulkan
/// states no encoding for a 16-bit float texel's decode, so a module that reads a `half` texel
/// records `DeviceNanEncoding`, and one that only reads `float` texels into memory, with no float
/// arithmetic, does not.
#[test]
fn a_half_texel_read_records_the_device_nan_encoding() {
    let Some(bc) = fixture("texture_ops") else {
        return;
    };
    let mut half = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        let f = ok(compile(&bc, "cube_faces", &p), &p, &stem);
        assert!(
            !f.precision.contains(&PrecisionNote::DeviceNanEncoding),
            "{stem}: {:?}",
            f.precision
        );
        if p.fields().images.storage_write_without_format {
            let h = ok(compile(&bc, "short_texels", &p), &p, &stem);
            assert!(
                h.precision.contains(&PrecisionNote::DeviceNanEncoding),
                "{stem}: {:?}",
                h.precision
            );
            half += 1;
        }
    }
    assert!(
        half > 0,
        "no committed profile compiles short_texels without inputs"
    );
}

/// A `half` read of a color texture slot records `DeviceNanEncoding` only where the format bound
/// there may hold a NaN: `short_texels`' read of texture 2, its one `half` source, which no
/// arithmetic follows, specializes the note on one clause, a test of the class of the format
/// bound at that slot (`SpecKey::BoundTexelFormat`) against `TexelFormat::NAN_FREE`, which no
/// pipeline's key carries. It holds of a half-float, 32-bit float, small-float or unknown format
/// (an Apple M4 hands back their NaNs), and not of a normalized or sRGB one, whose
/// texels decode to no NaN.
#[test]
fn a_half_texel_read_notes_the_device_nan_encoding_where_the_bound_format_may_hold_one() {
    use crate::metal::format::bound_texel_format;
    use crate::metal::texel_format::TexelFormat;
    let Some(bc) = fixture("texture_ops") else {
        return;
    };
    let mut half = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        if !p.fields().images.storage_write_without_format {
            continue;
        }
        let h = ok(compile(&bc, "short_texels", &p), &p, &stem);
        let entries: Vec<_> = h
            .specialized
            .iter()
            .filter(|s| s.note == PrecisionNote::DeviceNanEncoding)
            .collect();
        assert_eq!(entries.len(), 1, "{stem}: {:?}", h.specialized);
        let [clause] = entries[0].when.as_slice() else {
            panic!("{stem}: {:?}", entries[0].when);
        };
        assert!(clause.tests_bound_formats(), "{stem}: {clause:?}");
        let [test] = clause.all.as_slice() else {
            panic!("{stem}: {clause:?}");
        };
        assert_eq!(test.key, SpecKey::BoundTexelFormat(texture_slot(2)));
        assert_eq!(
            (test.mask, test.bits, test.equal),
            (TexelFormat::NAN_FREE, 0, true)
        );
        let holds = |pf: Option<PixelFormat>| {
            clause.holds_of_bound(&|slot| {
                assert_eq!(slot, texture_slot(2));
                pf.map(bound_texel_format)
            })
        };
        for pf in [
            PixelFormat::RGBA16Float,
            PixelFormat::R16Float,
            PixelFormat::RGBA32Float,
            PixelFormat::RG11B10Float,
        ] {
            assert!(holds(Some(pf)), "{stem}: {pf:?}");
        }
        assert!(holds(None), "{stem}: no format known");
        for pf in [
            PixelFormat::RGBA8Unorm,
            PixelFormat::R16Unorm,
            PixelFormat::RGBA8Snorm,
            PixelFormat::RGBA8UnormSRGB,
        ] {
            assert!(!holds(Some(pf)), "{stem}: {pf:?}");
        }
        half += 1;
    }
    assert!(
        half > 0,
        "no committed profile compiles short_texels without inputs"
    );
}

/// OpIsNan.
const OP_IS_NAN: u16 = 156;

/// A `half` written to a storage texture stores a NaN as the device converts the half widened,
/// noted where that need not be Metal's (the half's own bits in a 16-bit float channel,
/// `0x7fc00000` in a 32-bit one). `txt_level_write` writes halves no arithmetic makes. Bound in
/// `RGBA32Float` (where the choice declares it) its write tests no NaN and records
/// `DeviceNanEncoding`; bound in `RGBA16Float` it records the note on a profile without
/// `DriverDeviation::MetalHalfTexelWrites`, and requires `VkProperty::MetalHalfTexelWrites` on one
/// with it (a choice of a class rounding to `half` is a 16-bit float format's, declared or not;
/// such a device's own choice declares no format, so the test declares one there too);
/// bound in `RGBA8Unorm` it records none. Where the choice declares no format and no class, the
/// write records the note. A choice declaring `RGBA16Float` on every profile records the note or
/// requires the property as the profile's deviation says.
#[test]
fn a_half_texel_write_records_the_device_nan_encoding_where_it_may_not_be_metal_s() {
    use crate::device::requirements::VkProperty;
    use crate::error::DriverDeviation;
    let Some(bc) = fixture("txt_level_write") else {
        return;
    };
    let slot = texture_slot(0);
    let (mut float, mut formatless, mut measured, mut device) = (0, 0, 0, 0);
    for (stem, p) in committed_profiles().expect("profiles") {
        let keeps = p.takes_driver_deviation(DriverDeviation::MetalHalfTexelWrites);
        let compiled = |pf: PixelFormat| {
            let choice = storage_image_choice(pf, &p).ok()?;
            let mut inputs = LowerInputs::new();
            inputs.storage_images.insert(slot, choice);
            match with_inputs(&bc, "txt_level_write", &p, &inputs) {
                // A profile that cannot index the level views (or asks for more) has no module.
                Err(Error::NeedsInput(_) | Error::Unsupported(_)) => None,
                // A choice of no format and no class: one module for every float format.
                a => Some((
                    ok(a, &p, &format!("{stem} {pf:?}")),
                    choice.vk_format == crate::vk::Format::UNDEFINED
                        && choice.class == StorageImageClass::Identity,
                )),
            }
        };
        let noted = |a: &FunctionArtifact| a.precision.contains(&PrecisionNote::DeviceNanEncoding);
        let tests = |a: &FunctionArtifact| opcodes(&a.module.words).contains(&OP_IS_NAN);
        for pf in [
            PixelFormat::RGBA8Unorm,
            PixelFormat::RGBA16Float,
            PixelFormat::RGBX16Float,
            PixelFormat::RGBA32Float,
        ] {
            let Some((a, undeclared)) = compiled(pf) else {
                continue;
            };
            let what = format!("{stem}: {pf:?}: tests {}, noted {}", tests(&a), noted(&a));
            if undeclared {
                assert!(noted(&a), "{what}");
                formatless += 1;
                continue;
            }
            match pf {
                PixelFormat::RGBA16Float | PixelFormat::RGBX16Float => {
                    assert_eq!(noted(&a), !keeps, "{what}");
                    assert_eq!(
                        a.requirements
                            .properties
                            .contains(&VkProperty::MetalHalfTexelWrites),
                        keeps,
                        "{what}"
                    );
                    measured += usize::from(keeps);
                }
                PixelFormat::RGBA32Float => {
                    assert!(!tests(&a) && noted(&a), "{what}");
                    float += 1;
                }
                _ => assert!(!noted(&a), "{what}"),
            }
        }
        // A device of the deviation writes a half-float texture without a format, so its
        // committed choice declares none; one that declares `RGBA16Float` stores Metal's NaN there,
        // and the device's elsewhere.
        {
            let declared = crate::pipeline::lower_inputs::StorageImageChoice::new(
                crate::vk::Format::R16G16B16A16_SFLOAT,
                StorageImageClass::Identity,
            );
            let mut inputs = LowerInputs::new();
            inputs.storage_images.insert(slot, declared);
            let what = format!("{stem}: a declared RGBA16Float choice");
            match with_inputs(&bc, "txt_level_write", &p, &inputs) {
                Err(Error::NeedsInput(_) | Error::Unsupported(_)) => {}
                a => {
                    let a = ok(a, &p, &what);
                    assert_eq!(noted(&a), !keeps, "{what}");
                    assert_eq!(
                        a.requirements
                            .properties
                            .contains(&VkProperty::MetalHalfTexelWrites),
                        keeps,
                        "{what}"
                    );
                    measured += usize::from(keeps);
                    device += usize::from(!keeps);
                }
            }
        }
    }
    assert!(
        float > 0 && formatless > 0 && measured > 0 && device > 0,
        "the committed profiles cover a declared float format ({float}), a formatless choice \
         ({formatless}), the measured deviation ({measured}) and the device's NaN ({device})"
    );
}

/// Cube reads by face use the binder's 2D-array view (`ImageViewShape`), and a cube array's
/// array size is its layer count over six.
#[test]
fn cube_reads_use_the_array_view() {
    let Some(bc) = fixture("texture_ops") else {
        return;
    };
    for (stem, p) in committed_profiles().expect("profiles") {
        let a = ok(compile(&bc, "cube_faces", &p), &p, &stem);
        for i in [0, 1] {
            assert!(a.facts.contains(&RuntimeFact::ImageViewShape {
                slot: texture_slot(i)
            }));
        }
    }
}

/// `is_null_texture` reads the slot's presence constant, a `ResourcePresence` fact; a storage
/// slot only queried needs no format feature; a size query at a nonzero level answers from the
/// first level on every profile.
#[test]
fn queries_and_null_checks() {
    let Some(bc) = fixture("texture_ops") else {
        return;
    };
    for (stem, p) in committed_profiles().expect("profiles") {
        let a = ok(compile(&bc, "queries", &p), &p, &stem);
        let w = texture_slot(3);
        assert!(a.facts.contains(&RuntimeFact::ResourcePresence { slot: w }));
        assert!(a.interface.reads_key(SpecKey::Presence(w)));
        // The two queries at level 1 answer from the first level.
        assert_eq!(uses(&a, "texture.query_level"), 2, "{stem}");
        assert_eq!(uses(&a, "texture.query"), 6, "{stem}");
        assert_eq!(uses(&a, "texture.is_null"), 1, "{stem}");
        // The kernel stores whether the texture is nil and keeps no arm for it.
        assert_eq!(uses(&a, crate::lower::texture::ABSENT_ARM), 0, "{stem}");
        assert!(opcodes(&a.module.words).contains(&OP_IMAGE_QUERY_LEVELS));
    }
}

/// Profile `stem` without the robustness features `drop`.
fn without(stem: &str, drop: &[VkFeature]) -> DeviceProfile {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    for f in drop {
        desc.features.remove(f);
    }
    Device::new(desc, DevicePolicy::default())
        .expect("device")
        .profile()
        .clone()
}

/// A write is guarded by an image-size check exactly when `robustImageAccess` (either level)
/// does not discard it, and a read is checked in the shader (clamped, and Metal's out-of-bounds
/// texel selected) exactly under containment without `robustImageAccess2`, the one level whose
/// out-of-bounds read is Metal's; the guarded and checked forms pass L3 too.
#[test]
fn containment_follows_image_robustness() {
    let Some(bc) = fixture("texture_ops") else {
        return;
    };
    let mut profiles = committed_profiles().expect("profiles");
    profiles.push((
        "moltenvk-m3 without robustImageAccess2".to_owned(),
        without("moltenvk-m3", &[VkFeature::RobustImageAccess2]),
    ));
    profiles.push((
        "moltenvk-m3 without image robustness".to_owned(),
        without(
            "moltenvk-m3",
            &[VkFeature::RobustImageAccess, VkFeature::RobustImageAccess2],
        ),
    ));
    let (mut robust2, mut robust1, mut not) = (0, 0, 0);
    for (stem, p) in profiles {
        let f = p.fields();
        let robust_images = f.images.robust_image_access;
        match (f.images.robust_image_access2, robust_images) {
            (true, _) => robust2 += 1,
            (false, true) => robust1 += 1,
            (false, false) => not += 1,
        }
        let a = ok(compile(&bc, "depth_ms", &p), &p, &stem);
        let clamped = !f.images.robust_image_access2
            && matches!(
                f.containment,
                crate::device::profile::Containment::On { .. }
            );
        assert_eq!(uses(&a, "texture.read_clamped") == 2, clamped, "{stem}");
        assert_eq!(uses(&a, "texture.read") == 2, !clamped, "{stem}");
        if p.fields().images.storage_write_without_format {
            let a = ok(compile(&bc, "short_texels", &p), &p, &stem);
            assert_eq!(
                uses(&a, "texture.write_guarded") == 1,
                !robust_images,
                "{stem}"
            );
            assert_eq!(
                opcodes(&a.module.words).contains(&OP_IMAGE_QUERY_SIZE),
                !robust_images,
                "{stem}"
            );
        }
    }
    assert!(
        robust2 > 0 && robust1 > 0 && not > 0,
        "robust2 {robust2}, robust1 {robust1}, not {not}"
    );
}

/// The legality matrix of unmodelled forms: each is a typed refusal, never output.
#[test]
fn unmodelled_texture_forms_are_refused() {
    let Some(bc) = fixture("texture_ops") else {
        return;
    };
    let not_implemented = Unsupported::not_implemented(Feature::Lowering(LowerFamily::Texture));
    // A write at a dynamic level indexes the slot's level views by a non-uniform value: where the
    // profile cannot, the typed refusal names the feature (before any storage-format input, which
    // cannot change it).
    let non_uniform = Unsupported::needs(
        Feature::Lowering(LowerFamily::Texture),
        crate::error::Capability::Feature(
            crate::device::requirements::VkFeature::ShaderStorageImageArrayNonUniformIndexing,
        ),
    );
    for (stem, p) in committed_profiles().expect("profiles") {
        match compile(&bc, "write_lod", &p) {
            Ok(a) if p.fields().descriptors.non_uniform_storage_image => {
                ok(Ok(a), &p, &stem);
            }
            Err(Error::NeedsInput(_)) if p.fields().descriptors.non_uniform_storage_image => {}
            Err(Error::Unsupported(u)) if !p.fields().descriptors.non_uniform_storage_image => {
                assert_eq!(
                    (u.feature, u.needs),
                    (non_uniform.feature.clone(), non_uniform.needs.clone()),
                    "{stem}"
                )
            }
            other => panic!("{stem}: write_lod {:?}", other.map(|_| ())),
        }
    }
    let p = profile("moltenvk-m3", DevicePolicy::default()).expect("profile");
    // A read whose status word is read.
    let status = "%st = extractvalue { <4 x float>, i8 } %r, 1\n  %w = zext i8 %st to i32\n  store i32 %w, ptr addrspace(1) %out, align 4";
    let Some(bc) = assemble("status", &read_kernel(status)) else {
        return;
    };
    match compile(&bc, "k", &p) {
        Err(Error::Unsupported(u)) => assert_eq!(u.needs, not_implemented.needs),
        other => panic!("status read: {:?}", other.map(|_| ())),
    }
    // The same read of the texel only compiles, as a fetch.
    let texel = "%v = extractvalue { <4 x float>, i8 } %r, 0\n  %x = extractelement <4 x float> %v, i32 0\n  %w = bitcast float %x to i32\n  store i32 %w, ptr addrspace(1) %out, align 4";
    let Some(bc) = assemble("texel", &read_kernel(texel)) else {
        return;
    };
    let a = ok(compile(&bc, "k", &p), &p, "texel");
    let ops = opcodes(&a.module.words);
    assert!(ops.contains(&OP_IMAGE_FETCH) && !ops.contains(&OP_IMAGE_READ));
}

/// A kernel reading `texture2d<float, read>` slot 0 at `gid` into `%r`, then running `tail`
/// (which may store to `uint *out`, buffer 0).
fn read_kernel(tail: &str) -> String {
    format!(
        r#"define void @k(ptr addrspace(1) %t, ptr addrspace(1) %out, <2 x i32> %gid) {{
  %s = call ptr addrspace(2) @air.get_read_sampler()
  %r = call {{ <4 x float>, i8 }} @air.read_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x i32> %gid, <2 x i32> zeroinitializer, i32 0, i32 1)
  {tail}
  ret void
}}

declare ptr addrspace(2) @air.get_read_sampler()
declare {{ <4 x float>, i8 }} @air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)

!air.kernel = !{{!0}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}}
!5 = !{{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}}
"#
    )
}

const OP_TYPE_IMAGE: u16 = 25;

/// The `Image Format` operands of a module's `OpTypeImage`s.
fn image_formats(words: &[u32]) -> Vec<u32> {
    let mut out = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let n = (words[i] >> 16) as usize;
        if (words[i] & 0xFFFF) as u16 == OP_TYPE_IMAGE {
            out.extend(words.get(i + 8).copied());
        }
        i += n.max(1);
    }
    out
}

/// A kernel storing the width of `texture2d<uint, access::write>` slot 0, which it only queries.
const UINT_QUERY_ONLY: &str = r#"define void @k(ptr addrspace(1) %t, ptr addrspace(1) %out) {
  %w = call i32 @air.get_width_texture_2d(ptr addrspace(1) %t, i32 0)
  store i32 %w, ptr addrspace(1) %out, align 4
  ret void
}

declare i32 @air.get_width_texture_2d(ptr addrspace(1), i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<uint, write>", !"air.arg_name", !"t"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;

/// VUID-StandaloneSpirv-Image-04965: a storage slot only queried is declared `Unknown` whatever the
/// choice (the planner's own `R32Uint` choice, and choices whose numeric type or width disagree
/// with the declared `float`), so every committed profile validates; a read or written slot whose
/// choice disagrees with its texel class or width is `NotRepresentable`.
#[test]
fn storage_choices_must_match_the_texel_class() {
    let Some(query_only) = assemble("uint_query_only", UINT_QUERY_ONLY) else {
        return;
    };
    let Some(ops) = fixture("texture_ops") else {
        return;
    };
    let Some(write) = fixture("storage_texture_write") else {
        return;
    };
    use crate::pipeline::lower_inputs::StorageImageChoice;
    use crate::vk::Format as F;
    let slot = texture_slot(0);
    let identity = |f: F| {
        let mut inputs = LowerInputs::new();
        inputs.storage_images.insert(
            slot,
            StorageImageChoice::new(f, StorageImageClass::Identity),
        );
        inputs
    };
    let representable = Unsupported::not_representable(Feature::Format);
    let refused = |r: Result<std::sync::Arc<FunctionArtifact>, Error>, what: &str| match r {
        Err(Error::Unsupported(u)) => assert_eq!(
            (u.feature, u.needs),
            (representable.feature.clone(), representable.needs.clone()),
            "{what}"
        ),
        other => panic!("{what}: {:?}", other.map(|_| ())),
    };
    let mut planned = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        let mut choices = vec![F::R32_UINT, F::R8G8B8A8_UNORM, F::R32_SFLOAT, F::R64_UINT];
        if let Ok(c) = storage_image_choice(PixelFormat::R32Uint, &p) {
            planned += 1;
            choices.push(c.vk_format);
        }
        for f in choices {
            let what = format!("{stem} query-only {f:?}");
            let a = ok(with_inputs(&query_only, "k", &p, &identity(f)), &p, &what);
            assert!(
                a.facts
                    .contains(&RuntimeFact::StorageTextureFormat { slot }),
                "{what}"
            );
            assert_eq!(
                image_formats(&a.module.words),
                vec![0],
                "{what}: not Unknown"
            );
        }
        // `texture2d<uint, access::read_write>` with a float, unorm, or 64-bit choice.
        for f in [F::R8G8B8A8_UNORM, F::R32_SFLOAT, F::R64_UINT, F::R32_SINT] {
            refused(
                with_inputs(&ops, "read_write_fence", &p, &identity(f)),
                &format!("{stem} uint read_write {f:?}"),
            );
        }
        // `texture2d<float, access::write>` with an integer or 64-bit choice.
        for f in [F::R32_UINT, F::R8G8B8A8_SINT, F::R64_SINT] {
            refused(
                with_inputs(&write, "storage_texture_write", &p, &identity(f)),
                &format!("{stem} float write {f:?}"),
            );
        }
        // The matching choice compiles.
        ok(
            with_inputs(&ops, "read_write_fence", &p, &identity(F::R32_UINT)),
            &p,
            &format!("{stem} uint read_write R32_UINT"),
        );
    }
    assert!(
        planned > 0,
        "no committed profile plans an R32Uint storage image"
    );
}

/// The live block of the nil-arm kernels: read texture `t` at `gid` and store lane 0 to `out[0]`.
const NIL_LIVE: &str = "  %s = call ptr addrspace(2) @air.get_read_sampler()
  %r = call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x i32> %gid, <2 x i32> zeroinitializer, i32 0, i32 1)
  %v = extractvalue { <4 x float>, i8 } %r, 0
  %x = extractelement <4 x float> %v, i32 0
  %w = bitcast float %x to i32
  store i32 %w, ptr addrspace(1) %out, align 4";

/// The arm the nil-arm kernels keep for a nil texture: a sine of the thread's `x` to `out[1]`.
const NIL_ARM: &str = "  %ax = extractelement <2 x i32> %gid, i64 0
  %af = call fast float @air.convert.f.f32.u.i32(i32 %ax)
  %as = call fast float @air.fast_sin.f32(float %af)
  %am = fmul fast float %as, 3.000000e+00
  %ab = bitcast float %am to i32
  %ap = getelementptr inbounds i32, ptr addrspace(1) %out, i64 1
  store i32 %ab, ptr addrspace(1) %ap, align 4";

/// The nil-arm kernels: each computes a condition in `entry` (`(head, branch)`, the branch to
/// `%arm` or `%live`), then runs `%arm` ([`NIL_ARM`]) or `%live` ([`NIL_LIVE`]) and returns.
/// Textures `t` and `u` are `texture2d<float, read>` slots 0 and 1, `out` is `uint *` buffer 0.
const NIL_KERNELS: [(&str, &str, &str); 19] = [
    // A branch on `is_null`, and on its negation, its disjunctions, and its conjunction.
    (
        "direct",
        "%n = call i1 @air.is_null_texture_2d(ptr addrspace(1) %t)",
        "br i1 %n, label %arm, label %live",
    ),
    (
        "negated",
        "%n = call i1 @air.is_null_texture_2d(ptr addrspace(1) %t)\n  %p = xor i1 %n, true",
        "br i1 %p, label %live, label %arm",
    ),
    (
        "either_or",
        "%a = call i1 @air.is_null_texture_2d(ptr addrspace(1) %t)\n  %b = call i1 @air.is_null_texture_2d(ptr addrspace(1) %u)\n  %o = or i1 %a, %b",
        "br i1 %o, label %arm, label %live",
    ),
    (
        "either_select",
        "%a = call i1 @air.is_null_texture_2d(ptr addrspace(1) %t)\n  %b = call i1 @air.is_null_texture_2d(ptr addrspace(1) %u)\n  %o = select i1 %a, i1 true, i1 %b",
        "br i1 %o, label %arm, label %live",
    ),
    // A conjunction with a dynamic condition is decided by the `is_null` alone.
    (
        "and_dynamic",
        "%a = call i1 @air.is_null_texture_2d(ptr addrspace(1) %t)\n  %gx = extractelement <2 x i32> %gid, i64 0\n  %d = icmp eq i32 %gx, 0\n  %o = and i1 %d, %a",
        "br i1 %o, label %arm, label %live",
    ),
    (
        "switch",
        "%n = call i1 @air.is_null_texture_2d(ptr addrspace(1) %t)",
        "switch i1 %n, label %live [ i1 true, label %arm ]",
    ),
    // `icmp eq` with a constant, `icmp ne` of two slots' `is_null`s, a `freeze`, and a `select`
    // with a dynamic condition whose two operands are decided alike.
    (
        "icmp_eq",
        "%n = call i1 @air.is_null_texture_2d(ptr addrspace(1) %t)\n  %c = icmp eq i1 %n, false",
        "br i1 %c, label %live, label %arm",
    ),
    (
        "icmp_ne",
        "%a = call i1 @air.is_null_texture_2d(ptr addrspace(1) %t)\n  %b = call i1 @air.is_null_texture_2d(ptr addrspace(1) %u)\n  %c = icmp ne i1 %a, %b",
        "br i1 %c, label %arm, label %live",
    ),
    (
        "frozen",
        "%n = call i1 @air.is_null_texture_2d(ptr addrspace(1) %t)\n  %f = freeze i1 %n",
        "br i1 %f, label %arm, label %live",
    ),
    (
        "select_agree",
        "%a = call i1 @air.is_null_texture_2d(ptr addrspace(1) %t)\n  %b = call i1 @air.is_null_texture_2d(ptr addrspace(1) %u)\n  %gx = extractelement <2 x i32> %gid, i64 0\n  %d = icmp eq i32 %gx, 0\n  %o = select i1 %d, i1 %a, i1 %b",
        "br i1 %o, label %arm, label %live",
    ),
    // Not decided: a disjunction with a dynamic condition, a comparison with a dynamic condition,
    // a `select` with a dynamic condition and a dynamic operand or operands decided apart, a
    // condition through a phi, a
    // texture chosen between two slots (a switch-at-use whose null arm is a handle, not a
    // presence), and a dynamic branch beside an `is_null` the kernel stores.
    (
        "or_dynamic",
        "%a = call i1 @air.is_null_texture_2d(ptr addrspace(1) %t)\n  %gx = extractelement <2 x i32> %gid, i64 0\n  %d = icmp eq i32 %gx, 0\n  %o = or i1 %d, %a",
        "br i1 %o, label %arm, label %live",
    ),
    (
        "icmp_dynamic",
        "%a = call i1 @air.is_null_texture_2d(ptr addrspace(1) %t)\n  %gx = extractelement <2 x i32> %gid, i64 0\n  %d = icmp eq i32 %gx, 0\n  %c = icmp eq i1 %a, %d",
        "br i1 %c, label %arm, label %live",
    ),
    (
        "select_dynamic",
        "%a = call i1 @air.is_null_texture_2d(ptr addrspace(1) %t)\n  %gx = extractelement <2 x i32> %gid, i64 0\n  %d = icmp eq i32 %gx, 0\n  %e = icmp eq i32 %gx, 1\n  %o = select i1 %d, i1 %a, i1 %e",
        "br i1 %o, label %arm, label %live",
    ),
    (
        "select_disagree",
        "%a = call i1 @air.is_null_texture_2d(ptr addrspace(1) %t)\n  %b = call i1 @air.is_null_texture_2d(ptr addrspace(1) %u)\n  %nb = xor i1 %b, true\n  %gx = extractelement <2 x i32> %gid, i64 0\n  %d = icmp eq i32 %gx, 0\n  %o = select i1 %d, i1 %a, i1 %nb",
        "br i1 %o, label %arm, label %live",
    ),
    (
        "through_phi",
        "%a = call i1 @air.is_null_texture_2d(ptr addrspace(1) %t)\n  %gx = extractelement <2 x i32> %gid, i64 0\n  %d = icmp eq i32 %gx, 0\n  br i1 %d, label %one, label %two\none:\n  br label %join\ntwo:\n  br label %join\njoin:\n  %o = phi i1 [ %a, %one ], [ false, %two ]",
        "br i1 %o, label %arm, label %live",
    ),
    (
        "chosen",
        "%gx = extractelement <2 x i32> %gid, i64 0\n  %d = icmp eq i32 %gx, 0\n  %h = select i1 %d, ptr addrspace(1) %t, ptr addrspace(1) %u\n  %n = call i1 @air.is_null_texture_2d(ptr addrspace(1) %h)",
        "br i1 %n, label %arm, label %live",
    ),
    (
        "stored",
        "%n = call i1 @air.is_null_texture_2d(ptr addrspace(1) %t)\n  %z = zext i1 %n to i32\n  %zp = getelementptr inbounds i32, ptr addrspace(1) %out, i64 2\n  store i32 %z, ptr addrspace(1) %zp, align 4\n  %gx = extractelement <2 x i32> %gid, i64 0\n  %d = icmp eq i32 %gx, 0",
        "br i1 %d, label %arm, label %live",
    ),
    // The arm nested under another nil texture's arm, beside a read of that texture.
    (
        "nested",
        "%a = call i1 @air.is_null_texture_2d(ptr addrspace(1) %t)\n  br i1 %a, label %outer, label %live\nouter:\n  %b = call i1 @air.is_null_texture_2d(ptr addrspace(1) %u)\n  %gx = extractelement <2 x i32> %gid, i64 0\n  %d = icmp eq i32 %gx, 0\n  %o = or i1 %d, %b",
        "br i1 %o, label %arm, label %live",
    ),
    // `direct` with an empty arm.
    (
        "no_arm",
        "%n = call i1 @air.is_null_texture_2d(ptr addrspace(1) %t)",
        "br i1 %n, label %arm, label %live",
    ),
];

/// The module of [`NIL_KERNELS`].
fn nil_kernels() -> String {
    let mut text = String::new();
    let mut md = String::new();
    for (i, (name, head, branch)) in NIL_KERNELS.iter().enumerate() {
        // `no_arm` keeps no code in `%arm`.
        let arm_block = if *name == "no_arm" {
            "arm:\n  br label %done\n".to_string()
        } else {
            format!("arm:\n{NIL_ARM}\n  br label %done\n")
        };
        text.push_str(&format!(
            "define void @{name}(ptr addrspace(1) %t, ptr addrspace(1) %u, ptr addrspace(1) %out, <2 x i32> %gid) {{\nentry:\n  {head}\n  {branch}\n{arm_block}live:\n{NIL_LIVE}\n  br label %done\ndone:\n  ret void\n}}\n\n"
        ));
        md.push_str(&format!("!{} = !{{ptr @{name}, !100, !101}}\n", i));
    }
    let kernels: Vec<String> = (0..NIL_KERNELS.len()).map(|i| format!("!{i}")).collect();
    format!(
        r#"{text}declare ptr addrspace(2) @air.get_read_sampler()
declare {{ <4 x float>, i8 }} @air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)
declare i1 @air.is_null_texture_2d(ptr addrspace(1))
declare float @air.convert.f.f32.u.i32(i32)
declare float @air.fast_sin.f32(float)

!air.kernel = !{{{}}}
{md}!100 = !{{}}
!101 = !{{!102, !103, !104, !105}}
!102 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>", !"air.arg_name", !"t"}}
!103 = !{{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>", !"air.arg_name", !"u"}}
!104 = !{{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}}
!105 = !{{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}}
"#,
        kernels.join(", ")
    )
}

/// One kernel of [`NIL_KERNELS`] compiled and linked: its census by name (uses, words), the
/// number of its plan's instructions in the arm, its plan's strategies (count, words), and the
/// words of each shared routine of its arithmetic by census name.
struct NilCensus {
    census: BTreeMap<String, (u32, u64)>,
    in_arm: usize,
    strategies: (usize, u64),
    routines: BTreeMap<String, u64>,
}

/// Kernel `name` of [`NIL_KERNELS`] compiled for `p` and linked.
fn nil_census(bc: &[u8], name: &str, p: &DeviceProfile, stem: &str) -> NilCensus {
    let what = format!("{stem} {name}");
    let (a, plan, _) = compile_kept(bc, name, p).unwrap_or_else(|e| panic!("{what}: {e}"));
    link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
    let census: BTreeMap<String, (u32, u64)> = a
        .strategies
        .entries
        .iter()
        .map(|(n, u)| (n.clone(), (u.uses, u.words)))
        .collect();
    let plan_census: BTreeMap<String, (u32, u64)> = plan
        .census()
        .into_iter()
        .map(|(n, c)| (n.to_owned(), c))
        .collect();
    assert_eq!(
        census, plan_census,
        "{what}: the artifact's census is the plan's"
    );
    let all = plan.strategy.iter().flatten();
    let strategies = (
        all.clone().count(),
        all.map(|s| u64::from(s.words())).sum::<u64>(),
    );
    // A shared routine's words the census counts once under its own name, with no uses. So the
    // census words are the strategies' words and the shared routines', partitioned: the arm's
    // leave their names, and none is counted twice or dropped.
    let mut routines: BTreeMap<String, u64> = BTreeMap::new();
    for (n, w) in crate::lower::arith::routine_words(&plan.families.arith) {
        *routines.entry(n.to_owned()).or_default() += w;
    }
    // And the texture family's: a `float` read's conversion routine.
    for (n, w) in
        crate::lower::texture::routine_words(&plan.families.texture, &plan.families.access)
    {
        *routines.entry(n.to_owned()).or_default() += w;
    }
    let words: u64 = census.values().map(|(_, w)| w).sum();
    assert_eq!(
        words,
        strategies.1 + routines.values().sum::<u64>(),
        "{what}: the census partitions the plan's words"
    );
    NilCensus {
        census,
        in_arm: plan.families.texture.absent_arm.len(),
        strategies,
        routines,
    }
}

/// A nil texture's arm, the blocks the default presence specialization (every texture present)
/// never reaches, is counted under `texture.is_null.absent_arm`, exactly the words the arm adds,
/// while each of its strategies keeps its name and uses; a condition decided through `xor`, `or`,
/// `select` (with a decided condition, or a dynamic one choosing between operands decided alike),
/// `and`, `icmp eq`/`ne`, `freeze`, or an `i1` switch gives the same arm, and one not decided by a
/// presence (a dynamic disjunction or comparison, a dynamic `select` with a dynamic operand or with
/// operands decided apart, a phi, a texture chosen between slots, a dynamic branch) gives none.
#[test]
fn a_nil_textures_arm_is_counted_apart() {
    let arm = crate::lower::texture::ABSENT_ARM;
    // The name `validation/budgets/output-size-exceptions.json` excepts.
    assert_eq!(arm, "texture.is_null.absent_arm");
    let Some(bc) = assemble("nil_arms", &nil_kernels()) else {
        return;
    };
    for (stem, p) in committed_profiles_with(exact_maths()).expect("profiles") {
        let census = |name: &str| nil_census(&bc, name, &p, &stem);
        let direct = census("direct");
        let no_arm = census("no_arm");
        assert_eq!(no_arm.in_arm, 0, "{stem}: an empty arm has no instruction");
        assert!(!no_arm.census.contains_key(arm), "{stem}");
        // The arm's entry holds exactly the uses and words the arm adds to the kernel.
        let (uses, words) = direct.census[arm];
        assert_eq!(uses as usize, direct.in_arm, "{stem}");
        assert_eq!(
            (uses as usize, words),
            (
                direct.strategies.0 - no_arm.strategies.0,
                direct.strategies.1 - no_arm.strategies.1
            ),
            "{stem}: the arm's entry is what the arm adds"
        );
        assert!(words > 0, "{stem}");
        // Every other name keeps the live code's words, and a name only the arm uses stays
        // named, with its uses and no words. The words of a shared routine stay under its own
        // name whichever code calls it (the estimate does not follow them into the arm, so it only
        // ever under-counts the arm): they are set apart before the comparison.
        for (name, (u, w)) in &direct.census {
            if name == arm {
                continue;
            }
            let r = direct.routines.get(name).copied().unwrap_or(0);
            match no_arm.census.get(name) {
                Some((_, live)) => {
                    let live_r = no_arm.routines.get(name).copied().unwrap_or(0);
                    assert_eq!(w - r, live - live_r, "{stem}: {name}");
                }
                None => assert_eq!((*u > 0 || r > 0, w - r), (true, 0), "{stem}: {name}"),
            }
        }
        for name in [
            "negated",
            "either_or",
            "either_select",
            "and_dynamic",
            "switch",
            "icmp_eq",
            "icmp_ne",
            "frozen",
            "select_agree",
        ] {
            let c = census(name);
            assert_eq!(c.census.get(arm), Some(&(uses, words)), "{stem}: {name}");
        }
        // Nested under `t`'s arm, the condition on `u` is in the arm too.
        let nested = census("nested");
        assert!(
            nested
                .census
                .get(arm)
                .is_some_and(|(u, w)| *u > uses && *w > words),
            "{stem}: nested {:?}",
            nested.census.get(arm)
        );
        for name in [
            "or_dynamic",
            "icmp_dynamic",
            "select_dynamic",
            "select_disagree",
            "through_phi",
            "chosen",
            "stored",
        ] {
            let c = census(name);
            assert_eq!((c.census.get(arm), c.in_arm), (None, 0), "{stem}: {name}");
        }
    }
}
