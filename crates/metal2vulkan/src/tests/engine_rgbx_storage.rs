//! RGBX textures at storage slots (`metal::format::x_channel`): under a slot's choice with
//! `x_channel`, the function reads every texel with alpha 1 and writes green into the fourth
//! channel, Metal's X field, as plain Metal reads and stores the RGBX formats (an M4 under macOS
//! 27.0.1).
//!
//! On the CPU interpreter, each kernel runs under the choice without the conversion and with it,
//! on the same image and values, for `float`, `half`, `uint` and `int` texels and the class a
//! held RGBA twin takes (`Identity`, a half-float twin's too): a `read_write` read
//! is the plain read with alpha 1, in bounds and past the texture (whose alpha Metal reads 1
//! too); a write, `access::write` or `read_write`, stores the plain write's texel with its fourth
//! lane the green lane's bits, after the class's conversion. An sRGB twin's `SrgbEncode`, and an
//! `_sRGB` format's at any storage slot, reads and writes its codes by the MSL transfer through
//! the device's `Pow`, with and without the RGBX conversion (the interpreter models no `Pow`, so
//! those modules are checked by their instructions and notes). An `RGB9E5Float` texture's `SharedExponentBits`
//! writes Metal's truncated shared-exponent word through its `R32_UINT` alias and reads it exactly.
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::device::profile::{Containment, DeviceProfile};
use crate::lower::select::{LowerInputs, Plan};
use crate::pipeline::interface::PushLoc;
use crate::pipeline::lower_inputs::{StorageImageChoice, StorageImageClass};
use crate::pipeline::sym::{Axis, PushField, ResourceKind, ResourceSlot, SlotClass, SlotRef};
use crate::pipeline::sym::{SpecKey, Stage};
use crate::pipeline::testing::{
    link_and_validate, llvm_assemble, typed_module_with, AssembleError,
};
use crate::spirv::interp::{run_with_images, Buffer, Config, Image, Order, Resources};
use crate::spirv::ir::TypedModule;
use crate::spirv::testing::committed_profiles;
use crate::vk;

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// The texture's width: a row of 4 texels, one thread each.
const WIDTH: u32 = 4;

const SLOT: SlotRef = SlotRef {
    stage: Stage::Kernel,
    class: SlotClass::Texture,
    index: 0,
};

/// A texel type of the kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Texel {
    Float,
    Half,
    Uint,
    Int,
}

impl Texel {
    /// The AIR vector, the intrinsic suffix, the Metal type, its vector's name and its size.
    fn air(self) -> (&'static str, &'static str, &'static str, &'static str, u32) {
        match self {
            Texel::Float => ("<4 x float>", "v4f32", "float", "float4", 16),
            Texel::Half => ("<4 x half>", "v4f16", "half", "half4", 8),
            Texel::Uint => ("<4 x i32>", "u.v4i32", "uint", "uint4", 16),
            Texel::Int => ("<4 x i32>", "s.v4i32", "int", "int4", 16),
        }
    }

    /// The bits of a stored lane's one: a `float`'s, a `half`'s, or the integer's.
    fn one(self) -> u32 {
        match self {
            Texel::Float => 1.0f32.to_bits(),
            Texel::Half => 0x3c00,
            Texel::Uint | Texel::Int => 1,
        }
    }

    /// The RGBA twins' choices a slot of this texel type takes: the image's format and class.
    fn choices(self) -> Vec<(vk::Format, StorageImageClass)> {
        match self {
            Texel::Float | Texel::Half => vec![
                (vk::Format::R8G8B8A8_UNORM, StorageImageClass::Identity),
                (vk::Format::R16G16B16A16_SFLOAT, StorageImageClass::Identity),
                (vk::Format::R32G32B32A32_SFLOAT, StorageImageClass::Identity),
            ],
            Texel::Uint => vec![(vk::Format::R32G32B32A32_UINT, StorageImageClass::Identity)],
            Texel::Int => vec![(vk::Format::R32G32B32A32_SINT, StorageImageClass::Identity)],
        }
    }

    /// Texel `n`'s lanes as the image holds them: `f32` bits of values a `half` holds exactly,
    /// or integers.
    fn texel(self, n: u32) -> [u32; 4] {
        match self {
            Texel::Uint | Texel::Int => [100 + n, 200 + n, 300 + n, 400 + n],
            _ => [0.25, 0.5, 0.75, 0.125].map(|x: f32| (x + 0.0625 * n as f32).to_bits()),
        }
    }

    /// Thread `n`'s value as the kernel loads it: a `half`'s bits, or a word, each lane.
    fn value_bytes(self, n: u32) -> Vec<u8> {
        let lanes = match self {
            Texel::Uint | Texel::Int => [7 + n, 9000 + n, 13 + n, 77 + n],
            _ => [0.375, 0.6875, 0.0625, 0.9375].map(|x: f32| (x - 0.03125 * n as f32).to_bits()),
        };
        lanes
            .iter()
            .flat_map(|&x| match self {
                Texel::Half => (half_bits(f32::from_bits(x)) as u16).to_le_bytes().to_vec(),
                _ => x.to_le_bytes().to_vec(),
            })
            .collect()
    }
}

/// Exact `half` bits of `x`, a normal `half` or zero.
fn half_bits(x: f32) -> u32 {
    if x == 0.0 {
        return 0;
    }
    let b = x.to_bits();
    let exp = (b >> 23 & 0xff) as i32 - 127 + 15;
    assert!(
        exp > 0 && exp < 31 && b & 0x1fff == 0,
        "{x} is no exact half"
    );
    (b >> 16 & 0x8000) | (exp as u32) << 10 | (b >> 13 & 0x3ff)
}

/// Bitcode of the module `body` (after the AIR header), or `None` without `llvm-as`.
fn assemble(name: &str, body: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-rgbx-storage-{}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed),
        name.replace(' ', "-")
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("k.ll");
    std::fs::write(&ll, format!("{HEADER}{body}")).expect("write");
    let r = llvm_assemble(&ll);
    let _ = std::fs::remove_dir_all(&dir);
    match r {
        Ok(x) => Some(x),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("{name}: {why}"),
    }
}

/// A kernel reading `read_write` texture slot 0 at its thread's texel moved `dx` texels right,
/// storing the texel to `out[gid.x]` (buffer 0).
fn read_kernel(texel: Texel, dx: u32) -> String {
    let (vec4, suffix, ty, out_ty, size) = texel.air();
    format!(
        r#"define void @k(ptr addrspace(1) %t, ptr addrspace(1) %out, <2 x i32> %gid) {{
  %x = extractelement <2 x i32> %gid, i64 0
  %i = zext i32 %x to i64
  %rx = add i32 %x, {dx}
  %at = insertelement <2 x i32> %gid, i32 %rx, i64 0
  %rs = call ptr addrspace(2) @air.get_read_sampler()
  %r = call {{ {vec4}, i8 }} @air.read_texture_2d.{suffix}(ptr addrspace(1) %t, ptr addrspace(2) %rs, <2 x i32> %at, <2 x i32> zeroinitializer, i32 0, i32 3)
  %v = extractvalue {{ {vec4}, i8 }} %r, 0
  %q = getelementptr inbounds {vec4}, ptr addrspace(1) %out, i64 %i
  store {vec4} %v, ptr addrspace(1) %q, align {size}
  ret void
}}

declare ptr addrspace(2) @air.get_read_sampler()
declare {{ {vec4}, i8 }} @air.read_texture_2d.{suffix}(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)

!air.kernel = !{{!0}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.arg_type_name", !"texture2d<{ty}, read_write>", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 {size}, !"air.arg_type_align_size", i32 {size}, !"air.arg_type_name", !"{out_ty}", !"air.arg_name", !"out"}}
!5 = !{{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}}
"#
    )
}

/// A kernel writing `value[gid.x]` (buffer 0) to texture slot 0 (`access` `write` or
/// `read_write`) at its thread's texel.
fn write_kernel(texel: Texel, access: &str) -> String {
    let (vec4, suffix, ty, out_ty, size) = texel.air();
    let code = if access == "write" { 2 } else { 3 };
    format!(
        r#"define void @k(ptr addrspace(1) %t, ptr addrspace(2) %value, <2 x i32> %gid) {{
  %x = extractelement <2 x i32> %gid, i64 0
  %i = zext i32 %x to i64
  %q = getelementptr inbounds {vec4}, ptr addrspace(2) %value, i64 %i
  %v = load {vec4}, ptr addrspace(2) %q, align {size}
  call void @air.write_texture_2d.{suffix}(ptr addrspace(1) %t, <2 x i32> %gid, {vec4} %v, i32 0, i32 {code})
  ret void
}}

declare void @air.write_texture_2d.{suffix}(ptr addrspace(1), <2 x i32>, {vec4}, i32, i32)

!air.kernel = !{{!0}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.{access}", !"air.arg_type_name", !"texture2d<{ty}, {access}>", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 {size}, !"air.arg_type_align_size", i32 {size}, !"air.arg_type_name", !"{out_ty}", !"air.arg_name", !"value"}}
!5 = !{{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}}
"#
    )
}

/// What the interpreter's image does outside itself on `p`: the device's zero texel where
/// `robustImageAccess2` covers the access; a poisoned texel where the shader checks its reads
/// under `robustImageAccess`; a trap where the shader contains every access.
fn robust(p: &DeviceProfile) -> Option<[u32; 4]> {
    let f = p.fields();
    let Containment::On { .. } = f.containment else {
        panic!("containment off")
    };
    if f.images.robust_image_access2 {
        Some([0; 4])
    } else if f.images.robust_image_access {
        Some([0xDEAD_BEEF; 4])
    } else {
        None
    }
}

/// Lowering inputs viewing slot 0 as `format` in `class`, with the RGBX conversion where `x`.
fn inputs((format, class): (vk::Format, StorageImageClass), x: bool) -> LowerInputs {
    let mut choice = StorageImageChoice::new(format, class);
    choice.x_channel = x;
    let mut i = LowerInputs::new();
    i.storage_images.insert(SLOT, choice);
    i
}

/// The image of slot 0: a row of [`WIDTH`] texels, `texels(n)` at texel `n`.
fn image(texels: impl Fn(u32) -> [u32; 4], robust: Option<[u32; 4]>) -> Image {
    image_of(WIDTH, texels, robust)
}

/// The image of slot 0: a row of `width` texels, `texels(n)` at texel `n`.
fn image_of(width: u32, texels: impl Fn(u32) -> [u32; 4], robust: Option<[u32; 4]>) -> Image {
    let slot = ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Texture,
        index: 0,
    };
    let mut img = Image::new(slot, [width, 1, 1], 1, 1, 1);
    for n in 0..width {
        let at = img.index(0, 0, 0, [n, 0, 0]).expect("texel");
        img.texels[at] = texels(n);
    }
    img.robust = robust;
    img
}

/// Resources of a dispatch of one-thread threadgroups, buffer 0 holding `buffer`, a read past
/// the texture reading alpha 0 where the format's own rule applies.
fn resources(plan: &Plan, buffer: Vec<u8>) -> Resources {
    let mut map = BTreeMap::new();
    for axis in [Axis::X, Axis::Y, Axis::Z] {
        map.insert(SpecKey::LocalSize(axis), 1);
    }
    map.insert(SpecKey::FlatLocalSize, 1);
    map.insert(SpecKey::OutOfBoundsAlpha(SLOT), 0);
    let push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    for f in &plan.push.fields {
        match (f.field, f.loc) {
            (PushField::SlotBindOffset(_) | PushField::DispatchRegion, PushLoc::Push { .. }) => {}
            // A bound value read as data, which the interpreter writes from `spec`.
            (PushField::Specialized(_), PushLoc::Push { .. }) => {}
            other => panic!("unexpected push field {other:?}"),
        }
    }
    let buffer = Buffer {
        slot: Some(ResourceSlot {
            stage: Stage::Kernel,
            kind: ResourceKind::Buffer,
            index: 0,
        }),
        address: None,
        bytes: buffer,
    };
    Resources {
        buffers: vec![buffer],
        push,
        spec: map,
    }
}

/// `m` over [`WIDTH`] threads with `img` bound, in both scheduling orders, which must agree:
/// buffer 0 and the image's texels.
fn run(m: &TypedModule, res: &Resources, img: &Image) -> (Vec<u8>, Vec<[u32; 4]>) {
    run_over(WIDTH, m, res, img)
}

/// [`run`] over `width` threads.
fn run_over(width: u32, m: &TypedModule, res: &Resources, img: &Image) -> (Vec<u8>, Vec<[u32; 4]>) {
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            workgroups: [width, 1, 1],
            order,
            ..Config::default()
        };
        let (o, imgs) = run_with_images(m, &cfg, res.clone(), vec![img.clone()])
            .unwrap_or_else(|e| panic!("interpreter: {e:?}"));
        let texels = imgs.into_iter().next().expect("image").texels;
        outs.push((o.buffers.into_iter().next().expect("buffer"), texels));
    }
    let second = outs.pop().expect("second");
    let first = outs.pop().expect("first");
    assert_eq!(first, second, "the scheduling orders disagree");
    first
}

/// Lane `c` of texel `n` of `out`, as the kernel stored it: a `half`'s bits, or a word.
fn lane(out: &[u8], texel: Texel, n: u32, c: usize) -> u32 {
    if texel == Texel::Half {
        let i = n as usize * 8 + 2 * c;
        u32::from(u16::from_le_bytes([out[i], out[i + 1]]))
    } else {
        let i = n as usize * 16 + 4 * c;
        u32::from_le_bytes([out[i], out[i + 1], out[i + 2], out[i + 3]])
    }
}

/// The module and plan of kernel `bc` under `inputs` on `p`, its compiled artifact linked and
/// validated.
fn compiled(bc: &[u8], p: &DeviceProfile, inputs: &LowerInputs, what: &str) -> (Plan, TypedModule) {
    let (plan, m) = typed_module_with(bc, "k", p, inputs).unwrap_or_else(|e| panic!("{what}: {e}"));
    let a = crate::pipeline::testing::compile_with(bc, "k", p, inputs)
        .unwrap_or_else(|e| panic!("{what}: {e}"));
    link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what}: {e}"));
    (plan, m)
}

/// A `read_write` read under the conversion is the read without it, its alpha 1: in bounds and
/// past the texture, for each texel type and twin choice.
#[test]
fn an_rgbx_storage_read_takes_alpha_one() {
    let (mut ran, mut outside) = (0, 0);
    for texel in [Texel::Float, Texel::Half, Texel::Uint, Texel::Int] {
        for dx in [0, WIDTH] {
            let what = format!("{texel:?} dx {dx}");
            let Some(bc) = assemble(&what, &read_kernel(texel, dx)) else {
                return;
            };
            for (stem, p) in committed_profiles().expect("profiles") {
                for choice in texel.choices() {
                    let what = format!("{what} {choice:?} on {stem}");
                    let (plain_plan, plain) = compiled(&bc, &p, &inputs(choice, false), &what);
                    let (x_plan, x) = compiled(&bc, &p, &inputs(choice, true), &what);
                    let img = image(|n| texel.texel(n), robust(&p));
                    let size = if texel == Texel::Half { 8 } else { 16 };
                    let fill = vec![0xab; size * WIDTH as usize];
                    let (want, _) = run(&plain, &resources(&plain_plan, fill.clone()), &img);
                    let (got, _) = run(&x, &resources(&x_plan, fill), &img);
                    for n in 0..WIDTH {
                        for c in 0..4 {
                            let w = match c {
                                3 => texel.one(),
                                _ => lane(&want, texel, n, c),
                            };
                            assert_eq!(lane(&got, texel, n, c), w, "{what}: texel {n} lane {c}");
                        }
                    }
                    ran += 1;
                    outside += usize::from(dx != 0);
                }
            }
        }
    }
    assert!(ran > 0 && outside > 0, "ran {ran}, outside {outside}");
}

/// A write under the conversion, `access::write` and `read_write`, stores the texel the write
/// without it stores, its fourth lane the green lane's bits, for each texel type and twin
/// choice.
#[test]
fn an_rgbx_storage_write_stores_green_as_x() {
    let mut ran = 0;
    for texel in [Texel::Float, Texel::Half, Texel::Uint, Texel::Int] {
        for access in ["write", "read_write"] {
            let what = format!("{texel:?} {access}");
            let Some(bc) = assemble(&what, &write_kernel(texel, access)) else {
                return;
            };
            let bytes: Vec<u8> = (0..WIDTH).flat_map(|n| texel.value_bytes(n)).collect();
            for (stem, p) in committed_profiles().expect("profiles") {
                for choice in texel.choices() {
                    let what = format!("{what} {choice:?} on {stem}");
                    let (plain_plan, plain) = compiled(&bc, &p, &inputs(choice, false), &what);
                    let (x_plan, x) = compiled(&bc, &p, &inputs(choice, true), &what);
                    let img = image(|_| [0x5a5a_5a5a; 4], robust(&p));
                    let (_, want) = run(&plain, &resources(&plain_plan, bytes.clone()), &img);
                    let (_, got) = run(&x, &resources(&x_plan, bytes.clone()), &img);
                    for n in 0..WIDTH {
                        let at = img.index(0, 0, 0, [n, 0, 0]).expect("texel");
                        let w = want[at];
                        assert_eq!(got[at], [w[0], w[1], w[2], w[1]], "{what}: texel {n}");
                    }
                    ran += 1;
                }
            }
        }
    }
    assert!(ran > 0);
}

/// The choice of an 8-bit sRGB format at a storage slot: its UNORM alias, converting by the
/// device's transfer (`SrgbEncode`).
const SRGB: (vk::Format, StorageImageClass) =
    (vk::Format::R8G8B8A8_UNORM, StorageImageClass::SrgbEncode);

/// A `read_write` read of an 8-bit sRGB format's codes through its UNORM alias (`SrgbEncode`)
/// decodes the codes the view reads by the MSL transfer through the device's `Pow`, declaring no
/// Metal table, and records `PrecisionNote::SrgbTexelDecode` (Metal decodes each code through its
/// twelve-bit table), `float` and `half`, with and without the RGBX conversion, on every committed
/// profile; each module validates.
#[test]
fn an_srgb_storage_read_takes_the_devices_transfer() {
    use crate::pipeline::PrecisionNote;
    let mut ran = 0;
    for texel in [Texel::Float, Texel::Half] {
        let what = format!("{texel:?}");
        let Some(bc) = assemble(&what, &read_kernel(texel, 0)) else {
            return;
        };
        for (stem, p) in committed_profiles().expect("profiles") {
            for x in [false, true] {
                let what = format!("{what} x {x} on {stem}");
                let a = crate::pipeline::testing::compile_with(&bc, "k", &p, &inputs(SRGB, x))
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
                let words = &a.module.words;
                assert!(super::engine_srgb_decode::takes_pow(words), "{what}");
                assert!(!super::engine_fragment::holds_srgb_table(words), "{what}");
                assert!(
                    a.precision.contains(&PrecisionNote::SrgbTexelDecode),
                    "{what}"
                );
                ran += 1;
            }
        }
    }
    assert!(ran > 0);
}

/// A write of an 8-bit sRGB format through its UNORM alias (`SrgbEncode`), `access::write` and
/// `read_write`, of `float` and of `half` texels, encodes each color channel by the MSL transfer
/// through the device's `Pow` and `PackUnorm4x8` (`pack::pack`'s `unorm4x8_srgb`), declaring no
/// Metal table, and records `PrecisionNote::DevicePackConversion` (Metal encodes through its
/// twelve-bit table), with and without the RGBX conversion, on every committed profile; each
/// module validates.
#[test]
fn an_srgb_storage_write_takes_the_devices_conversion() {
    use crate::pipeline::PrecisionNote;
    let mut ran = 0;
    for texel in [Texel::Float, Texel::Half] {
        for access in ["write", "read_write"] {
            let what = format!("{texel:?} {access}");
            let Some(bc) = assemble(&what, &write_kernel(texel, access)) else {
                return;
            };
            for (stem, p) in committed_profiles().expect("profiles") {
                for x in [false, true] {
                    let what = format!("{what} x {x} on {stem}");
                    let a = crate::pipeline::testing::compile_with(&bc, "k", &p, &inputs(SRGB, x))
                        .unwrap_or_else(|e| panic!("{what}: {e}"));
                    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
                    let words = &a.module.words;
                    assert!(super::engine_srgb_decode::takes_pow(words), "{what}");
                    assert!(!super::engine_fragment::holds_srgb_table(words), "{what}");
                    assert!(
                        a.precision.contains(&PrecisionNote::DevicePackConversion),
                        "{what}"
                    );
                    ran += 1;
                }
            }
        }
    }
    assert!(ran > 0);
}

/// The choice of an `RGB9E5Float` texture at a storage slot of a device that does not store the
/// format: its `R32_UINT` alias, holding Metal's word (`SharedExponentBits`).
const SHARED_EXPONENT: (vk::Format, StorageImageClass) =
    (vk::Format::R32_UINT, StorageImageClass::SharedExponentBits);

/// The `RGB9E5Float` word Metal stores for red, green and blue `rgb` (an Apple M3 Max under macOS
/// 27.0.1, our own probe: every texel of 65,536 `float` and 65,536 `half` kernel writes): each
/// channel clamped to `[0, 65408]` (a NaN, a negative value and `-0` 0), the exponent
/// `max(⌊log2 m⌋, -16) + 16` of the largest clamped channel `m`, and each mantissa the channel
/// times `2^(24 - e)` truncated.
fn shared_exponent_word(rgb: [f32; 3]) -> u32 {
    let c = rgb.map(|v| match v.is_nan() || v.is_sign_negative() {
        true => 0.0,
        false => v.min(65408.0),
    });
    let m = c[0].max(c[1]).max(c[2]);
    let field = (m.to_bits() >> 23) as i32;
    let e = (field.max(111) - 111) as u32;
    let scale = 2f64.powi(24 - e as i32);
    let mantissa = c.map(|v| ((f64::from(v) * scale).floor() as u32).min(511));
    mantissa[0] | mantissa[1] << 9 | mantissa[2] << 18 | e << 27
}

/// The values the shared-exponent tests write, three a texel: the specials, the values at and
/// around each exponent's mantissa steps and the clamp, and spread random values from `2^-30`
/// to past the clamp.
fn shared_exponent_values(half: bool) -> Vec<[f32; 3]> {
    use crate::float16::{f16_bits_to_f32, f32_to_f16_bits};
    let specials = [
        0.0,
        -0.0,
        -1.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        65408.0,
        65472.0,
        65504.0,
        65536.0,
        2f32.powi(-16),
        2f32.powi(-17),
        2f32.powi(-24),
        2f32.powi(-25),
        f32::from_bits(1),
        1.0,
        0.5,
    ];
    let mut out: Vec<[f32; 3]> = specials
        .iter()
        .flat_map(|&a| specials.iter().map(move |&b| [a, b, 0.25 * b]))
        .collect();
    let mut s: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    for _ in 0..384 {
        let e = (next() % 48) as i32 - 30;
        let big = 2f32.powi(e) * (1.0 + (next() % 1024) as f32 / 1024.0);
        let frac = |r: u64| (r % 65536) as f32 / 65536.0;
        out.push([big, big * frac(next()), big * frac(next()) * frac(next())]);
        // A mantissa step's edge: `k + 1/2` units of the exponent's step, and one ulp either side.
        let unit = 2f32.powi(e.clamp(-24, 7));
        let k = (next() % 511) as f32;
        let edge = (k + 0.5) * unit;
        out.push([
            edge,
            f32::from_bits(edge.to_bits() + 1),
            f32::from_bits(edge.to_bits() - 1),
        ]);
    }
    if half {
        for t in &mut out {
            *t = t.map(|v| f16_bits_to_f32(f32_to_f16_bits(v)));
        }
    }
    out
}

/// A write of an `RGB9E5Float` texture through its `R32_UINT` alias (`SharedExponentBits`),
/// `access::write` and `read_write`, of `float` and `half` texels, stores Metal's word.
#[test]
fn a_shared_exponent_storage_write_stores_metals_word() {
    let mut ran = 0;
    for texel in [Texel::Float, Texel::Half] {
        let values = shared_exponent_values(texel == Texel::Half);
        let width = u32::try_from(values.len()).expect("texels");
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|rgb| {
                let lanes = [rgb[0], rgb[1], rgb[2], 1.0];
                lanes.into_iter().flat_map(move |v| match texel {
                    Texel::Half => crate::float16::f32_to_f16_bits(v).to_le_bytes().to_vec(),
                    _ => v.to_bits().to_le_bytes().to_vec(),
                })
            })
            .collect();
        for access in ["write", "read_write"] {
            let what = format!("{texel:?} {access}");
            let Some(bc) = assemble(&what, &write_kernel(texel, access)) else {
                return;
            };
            for (stem, p) in committed_profiles().expect("profiles") {
                let what = format!("{what} on {stem}");
                let (plan, m) = compiled(&bc, &p, &inputs(SHARED_EXPONENT, false), &what);
                let img = image_of(width, |_| [0x5a5a_5a5a; 4], robust(&p));
                let (_, got) = run_over(width, &m, &resources(&plan, bytes.clone()), &img);
                for (n, rgb) in values.iter().enumerate() {
                    let at = img.index(0, 0, 0, [n as u32, 0, 0]).expect("texel");
                    let want = shared_exponent_word(*rgb);
                    assert_eq!(got[at][0], want, "{what}: values {rgb:?}");
                }
                ran += 1;
            }
        }
    }
    assert!(ran > 0);
}

/// A `read_write` read of an `RGB9E5Float` texture through its `R32_UINT` alias
/// (`SharedExponentBits`) is Metal's: each channel its mantissa times `2^(e - 24)`, exact in
/// `float` and in `half`, and alpha 1, for words of every exponent.
#[test]
fn a_shared_exponent_storage_read_decodes_exactly() {
    let mut ran = 0;
    let words: Vec<u32> = (0..256u32)
        .map(|n| (n % 32) << 27 | (n * 37 % 512) | (n * 101 % 512) << 9 | (511 - n % 512) << 18)
        .collect();
    let value = |w: u32, c: u32| ((w >> (9 * c)) & 0x1ff) as f32 * 2f32.powi((w >> 27) as i32 - 24);
    for texel in [Texel::Float, Texel::Half] {
        let what = format!("{texel:?}");
        let Some(bc) = assemble(&what, &read_kernel(texel, 0)) else {
            return;
        };
        for (stem, p) in committed_profiles().expect("profiles") {
            let what = format!("{what} on {stem}");
            let (plan, m) = compiled(&bc, &p, &inputs(SHARED_EXPONENT, false), &what);
            let img = image_of(256, |n| [words[n as usize], 0, 0, 1], robust(&p));
            let size = if texel == Texel::Half { 8 } else { 16 };
            let fill = vec![0xab; size * 256];
            let (got, _) = run_over(256, &m, &resources(&plan, fill), &img);
            for (n, &w) in words.iter().enumerate() {
                for c in 0..4u32 {
                    let v = if c == 3 { 1.0 } else { value(w, c) };
                    let want = match texel {
                        Texel::Half => u32::from(crate::float16::f32_to_f16_bits(v)),
                        _ => v.to_bits(),
                    };
                    let read = lane(&got, texel, n as u32, c as usize);
                    assert_eq!(read, want, "{what}: word {w:#x} lane {c}");
                }
            }
            ran += 1;
        }
    }
    assert!(ran > 0);
}
