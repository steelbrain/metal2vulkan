//! The sampling operations the decoder refused previously end to end: family B, over the owned
//! fixtures `unk_sample_grad`, `unk_gather`, and `unk_calculate_lod`, on every committed profile
//! that compiles them (and `moltenvk-m3` and `lavapipe` without robustness). Every module links and
//! passes L3 (`G-val`). The CPU interpreter does not model sampling, so the tests check the emitted
//! instructions:
//!
//! - gradient samples of 3D, cube, and cube array textures and of depth 2D, 2D array, cube, and
//!   cube array textures, with and without comparison, are explicit-LOD samples with the `Grad`
//!   operand (a comparison's with its reference); a depth cube array's sample and a depth cube and cube array's comparison at level 0 are
//!   explicit-LOD samples with the `Lod` operand;
//! - gathers of cube and cube array textures read their component, depth gathers component 0, and
//!   comparison gathers are `OpImageDrefGather`, none with an offset (a cube takes none);
//! - the clamped and unclamped LOD queries of 2D array, 3D, cube, and cube array textures and of
//!   depth 2D, 2D array, cube, and cube array textures are `OpImageQueryLod`;
//! - a cube's explicit gradients take Metal's footprint (`sampler.sample_cube_grad`, with
//!   `PrecisionNote::CubeGradientFootprint`), other gradients pass unchanged: an isotropic
//!   sampler's as Metal's length of the two tangent components, an anisotropic constexpr
//!   sampler's as Metal's two face components.
//!
//! Previously each kernel was refused `UnknownIntrinsic`, so every test fails without it. A profile
//! without cube arrays (`vk12-floor`) refuses the fixtures with the typed device gap, which the
//! tests accept and count.
//!
//! `llvm-as` assembles the fixtures; without it the tests skip visibly.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::device::profile::{CubeGradientRoute, DeviceProfile};
use crate::device::requirements::VkFeature;
use crate::device::DevicePolicy;
use crate::device::{Device, DeviceDescription};
use crate::error::{Error, Needs};
use crate::library::FunctionArtifact;
use crate::pipeline::testing::{
    compile, engine_dir, link_and_validate, llvm_assemble, AssembleError,
};
use crate::pipeline::PrecisionNote;
use crate::spirv::testing::committed_profiles;

const OP_EXT_INST: u16 = 12;
const OP_CONSTANT: u16 = 43;
/// `Length` in `GLSL.std.450`.
const GLSL_LENGTH: u32 = 66;
/// `FMax` in `GLSL.std.450`.
const GLSL_FMAX: u32 = 40;
const OP_IMAGE_SAMPLE_IMPLICIT_LOD: u16 = 87;
const OP_IMAGE_SAMPLE_EXPLICIT_LOD: u16 = 88;
const OP_IMAGE_SAMPLE_DREF_IMPLICIT_LOD: u16 = 89;
const OP_IMAGE_SAMPLE_DREF_EXPLICIT_LOD: u16 = 90;
const OP_IMAGE_GATHER: u16 = 96;
const OP_IMAGE_DREF_GATHER: u16 = 97;
const OP_IMAGE_QUERY_LOD: u16 = 105;
const OP_IMAGE_QUERY_LEVELS: u16 = 106;
const OP_LOOP_MERGE: u16 = 246;
const LOD: u32 = 0x2;
const GRAD: u32 = 0x4;
const CONST_OFFSET: u32 = 0x8;
const OFFSET: u32 = 0x10;

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

/// `text` (a whole module) assembled, through a scratch file removed afterwards.
fn assemble_text(name: &str, text: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-sampler-unk-{}-{}-{name}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("k.ll");
    std::fs::write(&ll, text).expect("write");
    let r = llvm_assemble(&ll);
    let _ = std::fs::remove_dir_all(&dir);
    match r {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}: {why}"),
    }
}

/// `stem` with the robustness features removed.
fn unrobust(stem: &str) -> DeviceProfile {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    for f in [
        VkFeature::RobustImageAccess,
        VkFeature::RobustImageAccess2,
        VkFeature::RobustBufferAccess,
        VkFeature::RobustBufferAccess2,
    ] {
        desc.features.remove(&f);
    }
    Device::from_queries(desc, DevicePolicy::default())
        .expect("device")
        .profile()
        .clone()
}

/// Every committed profile, and `moltenvk-m3` and `lavapipe` without robustness.
fn profiles() -> Vec<(String, DeviceProfile)> {
    let mut out = committed_profiles().expect("profiles");
    for stem in ["moltenvk-m3", "lavapipe"] {
        out.push((format!("{stem} unrobust"), unrobust(stem)));
    }
    out
}

/// The artifact of `entry` on `p`, linked and past L3; `None` when `p` lacks a capability it
/// needs (a typed device gap). Any other refusal fails the test.
fn artifact(
    bc: &[u8],
    entry: &str,
    stem: &str,
    p: &DeviceProfile,
) -> Option<std::sync::Arc<FunctionArtifact>> {
    match compile(bc, entry, p) {
        Ok(a) => {
            link_and_validate(&a, p).unwrap_or_else(|e| panic!("{stem}: {entry}: L3: {e}"));
            Some(a)
        }
        Err(Error::Unsupported(u)) if matches!(u.needs, Needs::AnyOf(_)) => None,
        Err(e) => panic!("{stem}: {entry}: {e}"),
    }
}

/// The instructions of a SPIR-V module: opcode and its words.
fn insts(words: &[u32]) -> Vec<(u16, &[u32])> {
    let mut out = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let n = ((words[i] >> 16) as usize).max(1);
        out.push((
            (words[i] & 0xFFFF) as u16,
            &words[i..(i + n).min(words.len())],
        ));
        i += n;
    }
    out
}

/// The image-operand mask of each instruction of opcode `op` whose mask is word `at` (0 when the
/// instruction has none).
fn masks(a: &FunctionArtifact, op: u16, at: usize) -> Vec<u32> {
    insts(&a.module.words)
        .into_iter()
        .filter(|(o, _)| *o == op)
        .map(|(_, w)| w.get(at).copied().unwrap_or(0))
        .collect()
}

fn count(a: &FunctionArtifact, op: u16) -> usize {
    masks(a, op, 0).len()
}

/// Each profile's artifact of `entry`, checked by `check`; returns how many compiled and how many
/// were refused for a device gap.
fn each(name: &str, entry: &str, check: impl Fn(&str, &FunctionArtifact)) -> Option<(u32, u32)> {
    let bc = fixture(name)?;
    let (mut compiled, mut gaps) = (0, 0);
    for (stem, p) in profiles() {
        match artifact(&bc, entry, &stem, &p) {
            Some(a) => {
                check(&stem, &a);
                compiled += 1;
            }
            None => gaps += 1,
        }
    }
    Some((compiled, gaps))
}

#[test]
fn gradient_and_level_samples_of_3d_cube_and_depth_textures() {
    let Some((compiled, _)) = each("unk_sample_grad", "unk_sample_grad", |stem, a| {
        // A kernel samples nothing implicitly.
        for op in [
            OP_IMAGE_SAMPLE_IMPLICIT_LOD,
            OP_IMAGE_SAMPLE_DREF_IMPLICIT_LOD,
        ] {
            assert_eq!(count(a, op), 0, "{stem}: {op}");
        }
        let lod = |m: &u32| m & (LOD | GRAD);
        // Three color and four depth gradient samples; the depth cube array's at level 0, twice.
        let mut plain: Vec<u32> = masks(a, OP_IMAGE_SAMPLE_EXPLICIT_LOD, 5)
            .iter()
            .map(lod)
            .collect();
        plain.sort();
        let mut expected = vec![LOD; 2];
        expected.extend(vec![GRAD; 7]);
        assert_eq!(plain, expected, "{stem}");
        // Four gradient comparisons and two at level 0.
        let mut dref: Vec<u32> = masks(a, OP_IMAGE_SAMPLE_DREF_EXPLICIT_LOD, 6)
            .iter()
            .map(lod)
            .collect();
        dref.sort();
        let mut expected = vec![LOD, LOD];
        expected.extend(vec![GRAD; 4]);
        assert_eq!(dref, expected, "{stem}");
    }) else {
        return;
    };
    assert!(compiled >= 4, "{compiled}");
}

#[test]
fn cube_depth_and_comparison_gathers() {
    let Some((compiled, _)) = each("unk_gather", "unk_gather", |stem, a| {
        let words = &a.module.words;
        // The component operand is a constant id: its value.
        let constant = |id: u32| {
            insts(words)
                .into_iter()
                .find(|(o, w)| *o == OP_CONSTANT && w.get(2) == Some(&id))
                .and_then(|(_, w)| w.get(3).copied())
                .unwrap_or_else(|| panic!("{stem}: component %{id} is no constant"))
        };
        let gathers: Vec<&[u32]> = insts(words)
            .into_iter()
            .filter(|(o, _)| *o == OP_IMAGE_GATHER)
            .map(|(_, w)| w)
            .collect();
        let mut components: Vec<u32> = gathers.iter().map(|w| constant(w[5])).collect();
        components.sort();
        // Cube x and w, cube array z, and the two depth gathers' component 0.
        let mut expected = vec![0; 3];
        expected.extend([2, 3]);
        assert_eq!(components, expected, "{stem}");
        assert_eq!(count(a, OP_IMAGE_DREF_GATHER), 4, "{stem}");
        for (op, at) in [(OP_IMAGE_GATHER, 6), (OP_IMAGE_DREF_GATHER, 6)] {
            for m in masks(a, op, at) {
                assert_eq!(m & (CONST_OFFSET | OFFSET), 0, "{stem}: an offset on {op}");
            }
        }
    }) else {
        return;
    };
    assert!(compiled >= 4, "{compiled}");
}

#[test]
fn lod_queries_of_arrays_3d_and_cubes() {
    let Some((compiled, _)) = each("unk_calculate_lod", "unk_calculate_lod", |stem, a| {
        assert_eq!(count(a, OP_IMAGE_QUERY_LOD), 16, "{stem}");
    }) else {
        return;
    };
    assert!(compiled >= 4, "{compiled}");
}

fn uses(a: &FunctionArtifact, name: &str) -> u32 {
    a.strategies.entries.get(name).map_or(0, |u| u.uses)
}

/// The `GLSL.std.450` instructions numbered `number` (an `OpExtInst`'s word 4).
fn glsl(a: &FunctionArtifact, number: u32) -> usize {
    insts(&a.module.words)
        .into_iter()
        .filter(|(o, w)| *o == OP_EXT_INST && w.get(4) == Some(&number))
        .count()
}

/// Each of the fixture's six explicit cube gradient samples (color, depth, and comparison; cube and
/// cube array) takes Metal's footprint: its gradients are Metal's length of the two tangent
/// components on one tangent (two `GLSL.std.450` `FMax` per gradient, no `Length`), its census
/// name is `sampler.sample_cube_grad`, and the module records `CubeGradientFootprint`. The 3D and
/// depth 2D and 2D array gradient samples pass their gradients unchanged (`sampler.sample_grad`),
/// and a module whose gradients are all 2D (the 2D fixture's, through a runtime sampler) records no
/// such note. On a driver that
/// hands a compute pipeline's cube gradients to Metal's `gradientcube`
/// (`CubeGradientRoute::Metal`), the kernel hands Metal's own gradients, exactly: no note.
#[test]
fn explicit_cube_gradients_take_metals_footprint() {
    let handed = handed_metal();
    let Some((compiled, _)) = each("unk_sample_grad", "unk_sample_grad", |stem, a| {
        assert_eq!(uses(a, "sampler.sample_cube_grad"), 6, "{stem}");
        assert_eq!(uses(a, "sampler.sample_grad"), 5, "{stem}");
        let metal = handed.iter().any(|m| m == stem);
        assert_eq!(glsl(a, GLSL_LENGTH), 0, "{stem}");
        if !metal {
            let samples = uses(a, "sampler.sample_cube_grad") as usize;
            assert!(glsl(a, GLSL_FMAX) >= 4 * samples, "{stem}");
        }
        assert_eq!(
            a.precision.contains(&PrecisionNote::CubeGradientFootprint),
            !metal,
            "{stem}: {:?}",
            a.precision
        );
    }) else {
        return;
    };
    assert!(compiled >= 4, "{compiled}");
    assert!(!handed.is_empty(), "no profile hands Metal's gradients");
    let Some((compiled, _)) = each("a4_sample_grad", "a4_sample_grad", |stem, a| {
        assert_eq!(uses(a, "sampler.sample_cube_grad"), 0, "{stem}");
        assert!(uses(a, "sampler.sample_grad") > 0, "{stem}");
        assert_eq!(glsl(a, GLSL_LENGTH), 0, "{stem}");
        assert!(
            !a.precision.contains(&PrecisionNote::CubeGradientFootprint),
            "{stem}"
        );
    }) else {
        return;
    };
    assert!(compiled >= 4, "{compiled}");
}

/// The textures `unk_cube_grad_lod` samples, a cube and a cube array, twice each: Metal's level and
/// taps of a cube's explicit gradients, emulated, is one sampling routine per texture
/// (`routine::Filter`), which each of its samples calls.
const CUBE_GRAD_TEXTURES: usize = 2;

/// An anisotropic constexpr sampler (`max_anisotropy(2)`, bits 20-23 of the state's first word) on
/// a cube's explicit gradients takes Metal's footprint on every profile that compiles the isotropic
/// fixture. Without `samplerAnisotropy`, whose Vulkan sampler is isotropic, the shader takes
/// Metal's level and taps: each texture's routine, which its samples call, queries the level count
/// and loops over its taps, `AnisotropicFilter` is recorded and `CubeGradientFootprint` is not;
/// with the feature, and through the isotropic sampler on every profile, the device's sample at
/// Metal's footprint: no loop, no level query, `CubeGradientFootprint` recorded. On a driver that
/// hands a compute pipeline's cube gradients to Metal's `gradientcube`, the kernel hands Metal's
/// own through either sampler: no note. Neither module has a `Length`.
#[test]
fn an_anisotropic_constexpr_sampler_takes_metals_cube_gradient_footprint() {
    let path = engine_dir().join("unk_cube_grad_lod.ll");
    let text = std::fs::read_to_string(&path).expect("fixture");
    let word: u64 = 34_901_797_601_019_977;
    let original = format!("[2 x i64] [i64 {word}, i64 0]");
    assert_eq!(
        text.matches(&original).count(),
        1,
        "the fixture's sampler state"
    );
    assert_eq!((word >> 20) & 0xf, 0, "the fixture's sampler is isotropic");
    let anisotropic = text.replace(
        &original,
        &format!("[2 x i64] [i64 {}, i64 0]", word | 1 << 20),
    );
    let (Some(iso), Some(aniso)) = (
        assemble_text("isotropic", &text),
        assemble_text("anisotropic", &anisotropic),
    ) else {
        return;
    };
    let mut compiled = 0;
    let handed = handed_metal();
    for (stem, p) in profiles() {
        let Some(i) = artifact(&iso, "unk_cube_grad_lod", &stem, &p) else {
            continue;
        };
        let metal = handed.contains(&stem);
        let shader = !p.fields().images.sampler_anisotropy;
        assert_eq!(glsl(&i, GLSL_LENGTH), 0, "{stem}");
        let samples = uses(&i, "sampler.sample_cube_grad") as usize;
        assert!(samples > 0, "{stem}");
        assert_eq!(count(&i, OP_IMAGE_QUERY_LEVELS), 0, "{stem}");
        assert_eq!(count(&i, OP_LOOP_MERGE), 0, "{stem}");
        assert_eq!(
            i.precision.contains(&PrecisionNote::CubeGradientFootprint),
            !metal,
            "{stem}"
        );
        assert!(
            !i.precision.contains(&PrecisionNote::AnisotropicFilter),
            "{stem}"
        );
        let a = artifact(&aniso, "unk_cube_grad_lod", &stem, &p)
            .unwrap_or_else(|| panic!("{stem}: the anisotropic sampler refused"));
        compiled += 1;
        assert_eq!(
            uses(&a, "sampler.sample_cube_grad") as usize,
            samples,
            "{stem}"
        );
        assert_eq!(glsl(&a, GLSL_LENGTH), 0, "{stem}");
        assert_eq!(
            count(&a, OP_LOOP_MERGE),
            if shader { CUBE_GRAD_TEXTURES } else { 0 },
            "{stem}"
        );
        assert_eq!(
            a.precision.contains(&PrecisionNote::AnisotropicFilter),
            shader,
            "{stem}: {:?}",
            a.precision
        );
        assert_eq!(
            a.precision.contains(&PrecisionNote::CubeGradientFootprint),
            !metal && !shader,
            "{stem}: {:?}",
            a.precision
        );
    }
    assert!(compiled >= 4, "{compiled}");
}

/// Vulkan's cube-gradient rule: `moltenvk-m3` takes the device's own reading of the form handed
/// to it: `unk_cube_grad_lod` through its isotropic sampler and through an anisotropic constexpr
/// one queries no level count, loops over no taps and records `CubeGradientFootprint`, not
/// `AnisotropicFilter`.
#[test]
fn a_cube_gradient_route_takes_the_devices_reading() {
    use crate::device::profile::CubeGradients;
    let path = engine_dir().join("unk_cube_grad_lod.ll");
    let text = std::fs::read_to_string(&path).expect("fixture");
    let word: u64 = 34_901_797_601_019_977;
    let original = format!("[2 x i64] [i64 {word}, i64 0]");
    let anisotropic = text.replace(
        &original,
        &format!("[2 x i64] [i64 {}, i64 0]", word | 1 << 20),
    );
    assert_ne!(anisotropic, text, "the fixture's sampler state");
    let (Some(iso), Some(aniso)) = (
        assemble_text("isotropic", &text),
        assemble_text("anisotropic", &anisotropic),
    ) else {
        return;
    };
    let profile_text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../metal2vulkan-validation/profiles/moltenvk-m3.json"),
    )
    .expect("profile");
    let desc = DeviceDescription::from_khronos_json(&profile_text, None).expect("description");
    let unmeasured = Device::from_queries(desc, DevicePolicy::default())
        .expect("device")
        .profile()
        .clone();
    assert_eq!(
        unmeasured.fields().cube_gradients,
        CubeGradients {
            graphics: CubeGradientRoute::Vulkan,
            compute: CubeGradientRoute::Vulkan,
        }
    );
    for (bc, taps) in [(&iso, false), (&aniso, true)] {
        let what = if taps { "anisotropic" } else { "isotropic" };
        let u = artifact(bc, "unk_cube_grad_lod", "unmeasured", &unmeasured).expect(what);
        let samples = uses(&u, "sampler.sample_cube_grad") as usize;
        assert!(samples > 0, "{what}");
        assert_eq!(count(&u, OP_IMAGE_QUERY_LEVELS), 0, "{what}");
        assert_eq!(count(&u, OP_LOOP_MERGE), 0, "{what}");
        assert!(
            !u.precision.contains(&PrecisionNote::AnisotropicFilter),
            "{what}: {:?}",
            u.precision
        );
        assert!(
            u.precision.contains(&PrecisionNote::CubeGradientFootprint),
            "{what}: {:?}",
            u.precision
        );
    }
}

/// A kernel sampling the cube at texture slot 0 with explicit gradients through the runtime
/// sampler at slot 0.
const RUNTIME_CUBE_GRAD: &str = r#"define void @runtime_cube_grad(ptr addrspace(1) %t, ptr addrspace(2) %s, ptr addrspace(1) %out, i32 %gid) {
  %v = call { <4 x float>, i8 } @air.sample_texture_cube_grad.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <3 x float> <float 1.0, float 0.25, float -0.5>, <3 x float> <float 0.0, float 0x3F847AE140000000, float 0.0>, <3 x float> <float 0.0, float 0.0, float 0x3F847AE140000000>, float 0.0, i32 0)
  %x = extractvalue { <4 x float>, i8 } %v, 0
  %o = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i32 %gid
  store <4 x float> %x, ptr addrspace(1) %o, align 16
  ret void
}
declare { <4 x float>, i8 } @air.sample_texture_cube_grad.v4f32(ptr addrspace(1), ptr addrspace(2), <3 x float>, <3 x float>, <3 x float>, float, i32)
!air.kernel = !{!0}
!air.version = !{!11}
!air.language_version = !{!12}
!0 = !{ptr @runtime_cube_grad, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texturecube<float, sample>", !"air.arg_name", !"t"}
!4 = !{i32 1, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
!6 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
!11 = !{i32 2, i32 8, i32 0}
!12 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// A cube's explicit gradients through a runtime sampler slot take their footprint by the slot's
/// word on every profile: the slot's `SamplerState`
/// specialization reads `SamplerWord::ANISOTROPIC`, the module computes the isotropic form
/// (Metal's length per gradient, two `FMax` each; macOS 26.7's fixup a `Length` instead) beside
/// the anisotropic one and records `CubeGradientFootprint`, so the plan takes an anisotropic
/// sampler there.
#[test]
fn a_runtime_samplers_cube_gradients_take_their_footprint_by_its_word() {
    use crate::metal::sampler::SamplerWord;
    use crate::pipeline::sym::SpecKey;
    let Some(bc) = assemble_text("runtime_cube_grad", RUNTIME_CUBE_GRAD) else {
        return;
    };
    let mut by_word = 0;
    for (stem, p) in profiles() {
        let Some(a) = artifact(&bc, "runtime_cube_grad", &stem, &p) else {
            continue;
        };
        let route = p.fields().cube_gradients.compute;
        let fields = a.interface.read_keys().into_iter().find_map(|k| match k {
            SpecKey::SamplerState { fields, .. } => Some(fields),
            _ => None,
        });
        let reads = fields.is_some_and(|f| f & SamplerWord::ANISOTROPIC != 0);
        let noted = a.precision.contains(&PrecisionNote::CubeGradientFootprint);
        let lengths = glsl(&a, GLSL_FMAX) >= 4 || glsl(&a, GLSL_LENGTH) >= 2;
        assert_eq!(route, CubeGradientRoute::Vulkan, "{stem}");
        assert!(reads, "{stem}: {fields:?}");
        assert!(lengths, "{stem}");
        assert!(noted, "{stem}");
        by_word += 1;
    }
    assert!(by_word >= 5, "{by_word}");
}

/// The test profiles whose driver hands a compute pipeline's cube gradients to Metal's
/// `gradientcube` as given (`CubeGradientRoute::Metal`).
fn handed_metal() -> Vec<String> {
    profiles()
        .into_iter()
        .filter(|(_, p)| p.fields().cube_gradients.compute == CubeGradientRoute::Metal)
        .map(|(stem, _)| stem)
        .collect()
}
