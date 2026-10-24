//! A 16-bit float texel's own 16 bits through a read, under `HalfTexelBitsPolicy::Exact`.
//!
//! Metal's `texture2d<half>` never widens a 16-bit float texel, so the texel's NaN comes back as
//! its own code; a device whose decode of the texel into `f32` canonicalizes a NaN has destroyed
//! the code before the shader sees it, which is what the conformance family
//! `half-texel-nans-{r,rg,rgba}16f` records on MoltenVK at every one of its read sites. Under the
//! switch a colour texture slot the function declares `access::read` takes a second binding of the
//! texture's unsigned integer alias (`metal::format::half_bits_read_view`,
//! `RuntimeFact::HalfBitsTexture`) and each plain fetch branches on the bound format's class: the
//! alias's bits widened in integer arithmetic where the class names 16-bit float channels, the
//! device's texel elsewhere.
//!
//! - The declaration: under the default the slot has its own binding alone and records no fact;
//!   under the switch it has the alias binding too, declared as an unsigned integer sampled image,
//!   and records the fact.
//! - On the CPU interpreter, with the alias holding each `half`'s bits and the slot's own view
//!   holding what a canonicalizing decode gives (`0x7fc00000`): under Metal's answers
//!   (`DevicePolicy::exact_answers`) and a 16-bit float class every pattern comes back as its own
//!   bits, a signalling NaN's and a negative NaN's among them, and with this switch alone turned
//!   back off every NaN comes back `0x7e00`, which is what pins the fix to this switch rather than
//!   to the rounding ones. Under the default, and under a class that is not a 16-bit float
//!   format's, a NaN comes back `0x7e00` too. The switch on its own carries the bits on
//!   MoltenVK, whose `half` rounding is Metal's already; elsewhere the `half` the read hands back
//!   is rounded to 16 bits again when it is stored, which `half_rounding` governs and whose note
//!   the module records.
//!
//! `llvm-as` assembles the kernel; without it the tests skip visibly.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::device::profile::DeviceProfile;
use crate::device::{DevicePolicy, HalfTexelBitsPolicy};
use crate::lower::select::{LowerInputs, Plan};
use crate::metal::texel_format::texel_format;
use crate::metal::PixelFormat;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::sym::{Axis, ResourceKind, ResourceSlot, SlotClass, SlotRef, SpecKey, Stage};
use crate::pipeline::testing::{llvm_assemble, profile, typed_module_with, AssembleError};
use crate::spirv::interp::{run_with_images, Buffer, Config, Image, Order, Resources};
use crate::spirv::ir::{GlobalKind, Type as SType, TypedModule};

/// The texture's side: 16 x 16 texels, one `half` pattern a texel in each channel.
const SIDE: u32 = 16;

const TEXTURE: SlotRef = SlotRef {
    stage: Stage::Kernel,
    class: SlotClass::Texture,
    index: 0,
};

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// A kernel reading `texture2d<half, access::read>` slot 0 at its thread's texel of a 16 x 16
/// grid and storing the `half4`'s bits as four words to `out[4 n]` (buffer 0), so a NaN's payload
/// survives the store.
const KERNEL: &str = r#"define void @k(ptr addrspace(1) %t, ptr addrspace(1) %out, <2 x i32> %gid) {
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %row = shl i32 %y, 4
  %n = add i32 %x, %row
  %i = zext i32 %n to i64
  %rs = call ptr addrspace(2) @air.get_read_sampler()
  %r = call { <4 x half>, i8 } @air.read_texture_2d.v4f16(ptr addrspace(1) %t, ptr addrspace(2) %rs, <2 x i32> %gid, <2 x i32> zeroinitializer, i32 0, i32 1)
  %v = extractvalue { <4 x half>, i8 } %r, 0
  %b = bitcast <4 x half> %v to <4 x i16>
  %w = zext <4 x i16> %b to <4 x i32>
  %q = getelementptr inbounds <4 x i32>, ptr addrspace(1) %out, i64 %i
  store <4 x i32> %w, ptr addrspace(1) %q, align 16
  ret void
}

declare ptr addrspace(2) @air.get_read_sampler()
declare { <4 x half>, i8 } @air.read_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !6}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<half, access::read>", !"air.arg_name", !"t"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"uint4", !"air.arg_name", !"out"}
!6 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}
"#;

/// Bitcode of the kernel, or `None` without `llvm-as`.
fn assemble() -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-half-texel-bits-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("k.ll");
    std::fs::write(&ll, format!("{HEADER}{KERNEL}")).expect("write");
    let r = llvm_assemble(&ll);
    let _ = std::fs::remove_dir_all(&dir);
    match r {
        Ok(x) => Some(x),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("{why}"),
    }
}

/// Committed profile `stem` under the default policy, and under it with the switch on.
fn profiles(stem: &str) -> (DeviceProfile, DeviceProfile) {
    let device = profile(stem, DevicePolicy::UNTRUSTED).expect("profile");
    let exact = profile(
        stem,
        DevicePolicy {
            half_texel_bits: HalfTexelBitsPolicy::Exact,
            ..DevicePolicy::UNTRUSTED
        },
    )
    .expect("profile");
    (device, exact)
}

/// The module's resource at texture slot 0 of `kind`.
fn resource(m: &TypedModule, kind: ResourceKind) -> Option<ResourceSlot> {
    m.globals().iter().find_map(|g| match g.kind {
        GlobalKind::Resource(s) if s.index == 0 && s.kind == kind => Some(s),
        _ => None,
    })
}

/// The numeric type of the image the module's resource at texture slot 0 of `kind` is declared
/// as (`"uint"`, `"sint"` or `"float"`), `None` where it has no such resource.
fn numeric(m: &TypedModule, kind: ResourceKind) -> Option<&'static str> {
    let pointee = m.globals().iter().find_map(|g| match g.kind {
        GlobalKind::Resource(s) if s.index == 0 && s.kind == kind => Some(g.pointee),
        _ => None,
    })?;
    let image = match m.types().get(pointee).expect("a global's type") {
        SType::Image(i) => *i,
        _ => return Some("not an image"),
    };
    Some(
        match m.types().get(image.sampled_type).expect("a sampled type") {
            SType::Int { signed: false, .. } => "uint",
            SType::Int { signed: true, .. } => "sint",
            SType::Float { .. } => "float",
            _ => "other",
        },
    )
}

/// The patterns: every `half` NaN code of interest, the infinities, a denormal, the zeros and a
/// few finite values, one a texel. A texel's four channels take four consecutive patterns, so one
/// 16 x 16 texture covers 1,024 slots of the 16-bit space and every pattern below.
fn patterns() -> Vec<u16> {
    let mut v = vec![
        0x7e00, 0x7e01, 0xfe00, 0xfe01, 0x7c01, 0xfc01, 0x7dff, 0xfdff, 0x7c00, 0xfc00, 0x0001,
        0x8001, 0x0000, 0x8000, 0x3c00, 0xbc00, 0x3555, 0x7bff, 0xfbff, 0x0400,
    ];
    while v.len() < (SIDE * SIDE * 4) as usize {
        let n = v.len() as u16;
        v.push(n.wrapping_mul(2371) ^ 0x1234);
    }
    v.truncate((SIDE * SIDE * 4) as usize);
    v
}

/// The alias view's image: texel `(x, y)`'s channel `c` holds pattern `4 n + c`'s 16 bits as the
/// unsigned integer the alias reads.
fn alias_image(slot: ResourceSlot, p: &[u16]) -> Image {
    let mut img = Image::new(slot, [SIDE, SIDE, 1], 1, 1, 1);
    for n in 0..SIDE * SIDE {
        let at = img.index(0, 0, 0, [n % SIDE, n / SIDE, 0]).expect("texel");
        let i = 4 * n as usize;
        img.texels[at] = [0, 1, 2, 3].map(|c| u32::from(p[i + c]));
    }
    img
}

/// The slot's own view's image: what a device whose decode canonicalizes a NaN hands the shader,
/// `0x7fc00000` for every NaN pattern and the exact widening of every other.
fn own_image(slot: ResourceSlot, p: &[u16]) -> Image {
    let mut img = Image::new(slot, [SIDE, SIDE, 1], 1, 1, 1);
    for n in 0..SIDE * SIDE {
        let at = img.index(0, 0, 0, [n % SIDE, n / SIDE, 0]).expect("texel");
        let i = 4 * n as usize;
        img.texels[at] = [0, 1, 2, 3].map(|c| {
            let h = p[i + c];
            match h & 0x7fff > 0x7c00 {
                true => 0x7fc0_0000,
                false => half_of(h).to_bits(),
            }
        });
    }
    img
}

/// The `f32` of `half` bits `h`, which is not a NaN here.
fn half_of(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exponent = u32::from(h >> 10 & 0x1f);
    let mantissa = u32::from(h & 0x3ff);
    let bits = match (exponent, mantissa) {
        (0, 0) => sign,
        (0, _) => {
            let mut e = -1i32;
            let mut m = mantissa;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (((127 + e - 14) as u32) << 23) | ((m & 0x3ff) << 13)
        }
        (0x1f, _) => sign | 0x7f80_0000 | (mantissa << 13),
        _ => sign | ((exponent + 127 - 15) << 23) | (mantissa << 13),
    };
    f32::from_bits(bits)
}

/// Resources of a dispatch of one-thread threadgroups over the grid, with slot 0's `TexelFormat`
/// class set to `class`.
fn resources(plan: &Plan, class: u32) -> Resources {
    let mut spec = BTreeMap::new();
    for axis in [Axis::X, Axis::Y, Axis::Z] {
        spec.insert(SpecKey::LocalSize(axis), 1);
    }
    spec.insert(SpecKey::FlatLocalSize, 1);
    spec.insert(SpecKey::TexelFormat(TEXTURE), u64::from(class));
    let push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    Resources {
        buffers: vec![Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index: 0,
            }),
            address: None,
            bytes: vec![0u8; (SIDE * SIDE * 16) as usize],
        }],
        spec,
        push,
    }
}

/// The four words of texel `n` of the run's output.
fn texel(out: &[Vec<u8>], n: u32) -> [u32; 4] {
    let bytes = &out[0];
    let at = 16 * n as usize;
    [0, 1, 2, 3].map(|c| {
        let b = at + 4 * c;
        u32::from_le_bytes([bytes[b], bytes[b + 1], bytes[b + 2], bytes[b + 3]])
    })
}

/// Under the default a read-only slot has its own binding alone and records no fact; under the
/// switch it has the unsigned integer alias binding too, declared of the unsigned integer type,
/// and records `RuntimeFact::HalfBitsTexture`.
#[test]
fn the_switch_gives_a_read_only_slot_its_unsigned_integer_alias_binding() {
    let Some(bc) = assemble() else {
        return;
    };
    let mut seen = 0;
    for stem in ["moltenvk-m3", "lavapipe", "vk12-floor"] {
        let (device, exact) = profiles(stem);
        let inputs = LowerInputs::new();
        let (plan, m) = typed_module_with(&bc, "k", &device, &inputs)
            .unwrap_or_else(|e| panic!("{stem} default: {e}"));
        assert!(
            resource(&m, ResourceKind::RawTexture).is_none(),
            "{stem} default: an alias binding"
        );
        assert!(
            !plan
                .facts
                .contains(&RuntimeFact::HalfBitsTexture { slot: TEXTURE }),
            "{stem} default: the fact"
        );
        let (plan, m) = typed_module_with(&bc, "k", &exact, &inputs)
            .unwrap_or_else(|e| panic!("{stem} exact: {e}"));
        assert!(
            resource(&m, ResourceKind::RawTexture).is_some(),
            "{stem} exact: no alias binding"
        );
        assert_eq!(
            numeric(&m, ResourceKind::RawTexture),
            Some("uint"),
            "{stem} exact: the alias's numeric type"
        );
        assert_eq!(
            numeric(&m, ResourceKind::Texture),
            Some("float"),
            "{stem} exact: the slot's own numeric type"
        );
        assert!(
            plan.facts
                .contains(&RuntimeFact::HalfBitsTexture { slot: TEXTURE }),
            "{stem} exact: no fact"
        );
        seen += 1;
    }
    assert!(seen > 0);
}

/// On the CPU interpreter, under the switch and a 16-bit float class every pattern comes back as
/// its own bits; under a class that is not one, and under the default, a NaN comes back `0x7e00`
/// and every other pattern as the device's decode.
#[test]
fn the_switch_reads_a_16_bit_float_texel_s_own_bits() {
    let Some(bc) = assemble() else {
        return;
    };
    let p = patterns();
    let half = texel_format(PixelFormat::RGBA16Float).0;
    let other = texel_format(PixelFormat::RGBA32Float).0;
    for stem in ["moltenvk-m3", "lavapipe"] {
        let (device, exact) = profiles(stem);
        // Metal's answers everywhere, and Metal's answers with this switch alone off: the
        // isolation the leg is for, since the `half` a read hands back is rounded to 16 bits
        // again when it is stored, which the other switches govern.
        let answers = profile(stem, DevicePolicy::UNTRUSTED.exact_answers()).expect("profile");
        let without = profile(
            stem,
            DevicePolicy {
                half_texel_bits: HalfTexelBitsPolicy::Device,
                ..DevicePolicy::UNTRUSTED.exact_answers()
            },
        )
        .expect("profile");
        let inputs = LowerInputs::new();
        for (what, profile, class, bits) in [
            ("answers half", &answers, half, true),
            ("switch half", &exact, half, stem == "moltenvk-m3"),
            ("answers other", &answers, other, false),
            ("answers without the switch", &without, half, false),
            ("default half", &device, half, false),
        ] {
            let (plan, m) = typed_module_with(&bc, "k", profile, &inputs)
                .unwrap_or_else(|e| panic!("{stem} {what}: {e}"));
            let own = own_image(
                ResourceSlot {
                    stage: Stage::Kernel,
                    kind: ResourceKind::Texture,
                    index: 0,
                },
                &p,
            );
            let mut images = vec![own];
            if let Some(slot) = resource(&m, ResourceKind::RawTexture) {
                images.push(alias_image(slot, &p));
            }
            let cfg = Config {
                workgroups: [SIDE, SIDE, 1],
                order: Order::Forward,
                ..Config::default()
            };
            let (out, _) = run_with_images(&m, &cfg, resources(&plan, class), images)
                .unwrap_or_else(|e| panic!("{stem} {what}: interpreter: {e:?}"));
            for n in 0..SIDE * SIDE {
                let got = texel(&out.buffers, n);
                for c in 0..4usize {
                    let h = p[4 * n as usize + c];
                    let nan = h & 0x7fff > 0x7c00;
                    // Under the alias every pattern is its own bits. Under the device's decode
                    // only the NaN rule is this switch's: what the decode and the module's
                    // rounding make of a denormal is `half_rounding`'s row, whose note the
                    // module records, so a pattern that is not a NaN is left to it there.
                    let want = match (bits, nan) {
                        (true, _) => Some(h),
                        (false, true) => Some(0x7e00),
                        (false, false) => None,
                    };
                    let Some(want) = want else {
                        continue;
                    };
                    assert_eq!(
                        got[c],
                        u32::from(want),
                        "{stem} {what}: texel {n} channel {c} of pattern {h:#06x}"
                    );
                }
            }
        }
    }
}
