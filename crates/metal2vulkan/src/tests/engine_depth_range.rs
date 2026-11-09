//! Depth texels outside `[0, 1]`.
//!
//! Metal's `Depth32Float` texels hold any `float`; a Vulkan depth aspect keeps a value outside
//! `[0, 1]` only where `VK_EXT_depth_range_unrestricted` is enabled (RADV stores 1000.0 as 1.0
//! without it, measured). The product reads the extension from the device's description and
//! recommends it. A function whose depth comparisons the device makes, which read a depth image,
//! compiled for a device without it enabled records `PrecisionNote::RestrictedDepthRange` (narrowed
//! from every depth read to the device's comparisons), whatever its driver is measured to keep
//! (MoltenVK's included), and requires nothing in its place: a read, a sample and a gather read
//! the texels the consumer holds, which a colour image keeps, and record nothing, their use stated
//! by the interface's `TextureSlot::access`. A size query reads no texel, and a color texture's read
//! is not a depth texel.
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::device::profile::DeviceProfile;
use crate::device::requirements::VkExtension;
use crate::device::DevicePolicy;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::interface::SlotInterface;
use crate::pipeline::testing::{compile, link_and_validate, llvm_assemble, profile, AssembleError};

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// Nearest, clamp-to-edge, normalized coordinates (a public fixture's constexpr sampler).
const SAMPLER_WORDS: u64 = 34901797601017929;

fn assemble(name: &str, body: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-depth-range-{}-{}-{name}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("k.ll");
    std::fs::write(&ll, format!("{HEADER}{body}")).expect("write");
    let r = llvm_assemble(&ll);
    let _ = std::fs::remove_dir_all(&dir);
    match r {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}: {why}\n{body}"),
    }
}

/// A kernel that computes `%v` (a `float`) from texture `%t` of MSL type `tex_ty` by `body` and
/// stores it to buffer 0; `decl` declares the intrinsics `body` calls.
fn kernel(tex_ty: &str, body: &str, decl: &str) -> String {
    kernel_with(SAMPLER_WORDS, tex_ty, body, decl)
}

/// [`kernel`] with the constexpr sampler of `words`.
fn kernel_with(words: u64, tex_ty: &str, body: &str, decl: &str) -> String {
    format!(
        r#"@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {words}, i64 0], align 8

define void @k(ptr addrspace(1) %t, ptr addrspace(1) %out) {{
  {body}
  store float %v, ptr addrspace(1) %out, align 4
  ret void
}}

{decl}
declare ptr addrspace(2) @air.get_read_sampler()

!air.kernel = !{{!0}}
!air.sampler_states = !{{!5}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"{tex_ty}", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}}
!5 = !{{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}}
"#
    )
}

/// The forms of `depth2d<float>` access (and a color read), each a kernel, whether it reads the
/// texture's texels, and whether it compares a depth texel.
fn depth_kernels() -> Vec<(&'static str, String, bool, bool)> {
    let depth = "depth2d<float, sample>";
    vec![
        (
            "read",
            kernel(
                depth,
                "%rs = call ptr addrspace(2) @air.get_read_sampler()
  %r = call { float, i8 } @air.read_depth_2d.f32(ptr addrspace(1) %t, ptr addrspace(2) %rs, i32 1, <2 x i32> zeroinitializer, <2 x i32> zeroinitializer, i32 0, i32 0)
  %v = extractvalue { float, i8 } %r, 0",
                "declare { float, i8 } @air.read_depth_2d.f32(ptr addrspace(1), ptr addrspace(2), i32, <2 x i32>, <2 x i32>, i32, i32)",
            ),
            true,
            false,
        ),
        (
            "sample",
            kernel(
                depth,
                "%r = call { float, i8 } @air.sample_depth_2d.f32(ptr addrspace(1) %t, ptr addrspace(2) @__air_sampler_state, i32 1, <2 x float> <float 0.5, float 0.5>, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)
  %v = extractvalue { float, i8 } %r, 0",
                "declare { float, i8 } @air.sample_depth_2d.f32(ptr addrspace(1), ptr addrspace(2), i32, <2 x float>, i1, <2 x i32>, i1, float, float, i32)",
            ),
            true,
            false,
        ),
        (
            "compare",
            kernel(
                depth,
                "%r = call { float, i8 } @air.sample_compare_depth_2d.f32(ptr addrspace(1) %t, ptr addrspace(2) @__air_sampler_state, i32 1, <2 x float> <float 0.5, float 0.5>, float 0.5, i1 true, <2 x i32> zeroinitializer, i1 true, float 0.0, float 0.0, i32 0)
  %v = extractvalue { float, i8 } %r, 0",
                "declare { float, i8 } @air.sample_compare_depth_2d.f32(ptr addrspace(1), ptr addrspace(2), i32, <2 x float>, float, i1, <2 x i32>, i1, float, float, i32)",
            ),
            true,
            true,
        ),
        (
            "gather",
            kernel(
                depth,
                "%r = call { <4 x float>, i8 } @air.gather_depth_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) @__air_sampler_state, i32 1, <2 x float> <float 0.5, float 0.5>, i1 true, <2 x i32> zeroinitializer, i32 0)
  %g = extractvalue { <4 x float>, i8 } %r, 0
  %v = extractelement <4 x float> %g, i64 0",
                "declare { <4 x float>, i8 } @air.gather_depth_2d.v4f32(ptr addrspace(1), ptr addrspace(2), i32, <2 x float>, i1, <2 x i32>, i32)",
            ),
            true,
            false,
        ),
        (
            "width",
            kernel(
                depth,
                "%w = call i32 @air.get_width_depth_2d(ptr addrspace(1) %t, i32 0)
  %v = uitofp i32 %w to float",
                "declare i32 @air.get_width_depth_2d(ptr addrspace(1), i32)",
            ),
            false,
            false,
        ),
        (
            "color read",
            kernel(
                "texture2d<float, sample>",
                "%rs = call ptr addrspace(2) @air.get_read_sampler()
  %r = call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %rs, <2 x i32> zeroinitializer, <2 x i32> zeroinitializer, i32 0, i32 0)
  %g = extractvalue { <4 x float>, i8 } %r, 0
  %v = extractelement <4 x float> %g, i64 0",
                "declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)",
            ),
            true,
            false,
        ),
    ]
}

fn unrestricted(p: &DeviceProfile) -> bool {
    p.fields()
        .enabled
        .extensions
        .contains(&VkExtension::ExtDepthRangeUnrestricted)
}

/// The committed profiles the product reads the extension from: the RADV and lavapipe
/// profiles list it, MoltenVK's and the 1.2 floor's do not.
#[test]
fn the_extension_is_read_from_the_device_description() {
    for (stem, listed) in [
        ("amd-rdna3-radv", true),
        ("lavapipe", true),
        ("swiftshader", true),
        ("moltenvk-m3", false),
        ("intel-irisxe-anv", false),
        ("vk12-floor", false),
    ] {
        let p = profile(stem, DevicePolicy::default()).expect(stem);
        assert_eq!(unrestricted(&p), listed, "{stem}");
    }
}

/// A comparison of a depth texel the device makes records `RestrictedDepthRange` exactly where the
/// profile lacks the extension, MoltenVK's included, and no module requires the extension; a
/// read, a sample and a gather never do, their use stated by the
/// texture slot's access (read or sample) on every profile, as a color read's is; a size query
/// reads no texel. Every comparison is the device's, a constexpr sampler's nearest one included.
#[test]
fn depth_comparisons_record_the_restricted_range_without_the_extension() {
    let profiles = [
        (
            "radv",
            profile("amd-rdna3-radv", DevicePolicy::default()).expect("radv"),
        ),
        (
            "anv",
            profile("intel-irisxe-anv", DevicePolicy::default()).expect("anv"),
        ),
        (
            "m3",
            profile("moltenvk-m3", DevicePolicy::default()).expect("m3"),
        ),
    ];
    let [(_, radv), (_, anv), (_, m3)] = &profiles;
    assert!(unrestricted(radv) && !unrestricted(anv) && !unrestricted(m3));
    for (form, text, texels, compares) in depth_kernels() {
        let Some(bc) = assemble(form, &text) else {
            return;
        };
        for (device, p) in &profiles {
            let device_compares = compares;
            let expected = device_compares && !unrestricted(p);
            let a = compile(&bc, "k", p).unwrap_or_else(|e| panic!("{form}: compile: {e}"));
            link_and_validate(&a, p).unwrap_or_else(|e| panic!("{form}: L3: {e}"));
            assert_eq!(
                a.precision.contains(&PrecisionNote::RestrictedDepthRange),
                expected,
                "{form} on {device}: {:?}",
                a.precision
            );
            assert!(
                !a.requirements
                    .extensions
                    .contains(&VkExtension::ExtDepthRangeUnrestricted),
                "{form} on {device}: {:?}",
                a.requirements.extensions
            );
            let slot = a
                .interface
                .slots
                .values()
                .find_map(|slot| match slot {
                    SlotInterface::Texture(t) => Some(*t),
                    _ => None,
                })
                .unwrap_or_else(|| panic!("{form} on {device}: no texture slot"));
            assert_eq!(
                slot.access.read || slot.access.sample,
                texels,
                "{form} on {device}: {slot:?}"
            );
            assert_eq!(slot.depth_compare, device_compares, "{form} on {device}");
        }
    }
}
