//! Swizzled textures at storage slots: under a slot's `swizzled_storage` variant the function
//! applies the bound texture's swizzle to the texels it reads and undoes it on those it writes, as
//! plain Metal does through a storage binding of a swizzled view (an M4 under macOS 27.0.1; misuse,
//! which Metal's validation stops: the view's usage is stripped of `ShaderWrite`).
//!
//! - The reference rules reproduce the probes' answers ([`the_rules_are_the_probes_answers`]).
//! - On the CPU interpreter, a `read_write` read of each swizzle in [`swizzles`] is the converted
//!   texel swizzled, a read outside the texture Metal's zero texel (alpha the format's) swizzled,
//!   for `float`, `half` and `uint` results, through an identity storage view and through a
//!   `SwizzleBgra` one (whose channel swap the swizzle follows); a write, `access::write` or
//!   `read_write`, stores channel `c` from the first lane whose source is `c`, else lane `c`. An
//!   atomic ignores the swizzle.
//! - A `ulong` read under the variant is refused (`NotImplemented(TextureSwizzle)`).
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::device::profile::DeviceProfile;
use crate::device::requirements::VkFeature;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{Error, Feature};
use crate::lower::select::{LowerInputs, Plan};
use crate::metal::{TextureSwizzle as S, TextureSwizzleChannels as C};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::lower_inputs::{StorageImageChoice, StorageImageClass};
use crate::pipeline::sym::{Axis, PushField, ResourceKind, ResourceSlot, SlotClass, SlotRef};
use crate::pipeline::sym::{SpecKey, Stage};
use crate::pipeline::testing::{
    engine_dir, link_and_validate, llvm_assemble, typed_module_with, AssembleError,
};
use crate::spirv::interp::{run_with_images, Buffer, Config, Image, Order, Resources};
use crate::spirv::ir::TypedModule;
use crate::vk;

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// The texture's width: a row of 4 texels, one thread each.
const WIDTH: u32 = 4;

const SLOT: SlotRef = SlotRef {
    stage: Stage::Kernel,
    class: SlotClass::Texture,
    index: 0,
};

/// The swizzles: permutations, `Zero` and `One` sources, repeated sources and the identity, the
/// probes' among them.
fn swizzles() -> Vec<C> {
    let c = |s: &str| {
        let v: Vec<S> = s
            .chars()
            .map(|ch| match ch {
                'R' => S::Red,
                'G' => S::Green,
                'B' => S::Blue,
                'A' => S::Alpha,
                '0' => S::Zero,
                _ => S::One,
            })
            .collect();
        C::new(v[0], v[1], v[2], v[3])
    };
    [
        "RGBA", "GBAR", "BGRA", "ABGR", "GRBA", "RRRR", "R01A", "0000", "1111", "RGB1", "RGBR",
        "0GBA", "GR01", "0R01", "1RG0", "ARGB",
    ]
    .into_iter()
    .map(c)
    .collect()
}

// ---- The reference ----------------------------------------------------------------------------

/// The swizzle's four sources, red's first.
fn sources(s: C) -> [S; 4] {
    [s.red, s.green, s.blue, s.alpha]
}

/// Metal's read of texel `t` (its channels) through swizzle `s`: each lane the source its
/// channel names, `one` for `One`.
fn read_rule(s: C, t: [u32; 4], one: u32) -> [u32; 4] {
    sources(s).map(|source| match source {
        S::Zero => 0,
        S::One => one,
        S::Red => t[0],
        S::Green => t[1],
        S::Blue => t[2],
        _ => t[3],
    })
}

/// The channels Metal stores for a write of `v` through swizzle `s`: channel `c` takes the lane
/// of the first source that is `c`, else lane `c`.
fn write_rule(s: C, v: [u32; 4]) -> [u32; 4] {
    let channel = [S::Red, S::Green, S::Blue, S::Alpha];
    let from = sources(s);
    std::array::from_fn(|c| match from.iter().position(|&f| f == channel[c]) {
        Some(i) => v[i],
        None => v[c],
    })
}

/// The probes' answers (for reads and writes of `RGBA8Unorm`, `RG8Unorm`, `R32Float` and `R32Uint`
/// views, for reads outside the texture), which the rules reproduce: a write of `(10, 20, 30, 40)`
/// and a read of the texel `(10, 20, 30, 40)` in bytes; `RG8Unorm`'s two channels; `R32Uint`'s
/// texel `(1000, 0, 0, 1)` and its zero texel `(0, 0, 0, 1)`.
#[test]
fn the_rules_are_the_probes_answers() {
    let by = |name: &str| {
        swizzles()
            .into_iter()
            .zip([
                "RGBA", "GBAR", "BGRA", "ABGR", "GRBA", "RRRR", "R01A", "0000", "1111", "RGB1",
                "RGBR", "0GBA", "GR01", "0R01", "1RG0", "ARGB",
            ])
            .find_map(|(s, n)| (n == name).then_some(s))
            .expect("swizzle")
    };
    let t = [10, 20, 30, 40];
    for (name, read, written) in [
        ("GBAR", [20, 30, 40, 10], [40, 10, 20, 30]),
        ("BGRA", [30, 20, 10, 40], [30, 20, 10, 40]),
        ("ABGR", [40, 30, 20, 10], [40, 30, 20, 10]),
        ("GRBA", [20, 10, 30, 40], [20, 10, 30, 40]),
        ("RRRR", [10, 10, 10, 10], [10, 20, 30, 40]),
        ("R01A", [10, 0, 255, 40], [10, 20, 30, 40]),
        ("0000", [0, 0, 0, 0], [10, 20, 30, 40]),
        ("1111", [255, 255, 255, 255], [10, 20, 30, 40]),
        ("RGB1", [10, 20, 30, 255], [10, 20, 30, 40]),
        ("RGBR", [10, 20, 30, 10], [10, 20, 30, 40]),
        ("0GBA", [0, 20, 30, 40], [10, 20, 30, 40]),
    ] {
        assert_eq!(read_rule(by(name), t, 255), read, "{name} read");
        assert_eq!(write_rule(by(name), t), written, "{name} write");
    }
    // `RG8Unorm`: the texel `(10, 20, 0, 255)`; a write stores red and green.
    let rg = [10, 20, 0, 255];
    for (name, read, written) in [
        ("GR01", [20, 10, 0, 255], [20, 10]),
        ("RRRR", [10, 10, 10, 10], [10, 20]),
        ("0R01", [0, 10, 0, 255], [20, 20]),
    ] {
        assert_eq!(read_rule(by(name), rg, 255), read, "RG8 {name} read");
        assert_eq!(write_rule(by(name), t)[..2], written, "RG8 {name} write");
    }
    // `R32Uint`: probe 1's reads and writes of `(11, 22, 33, 44)`, probe 9's zero texel.
    let r = [1000, 0, 0, 1];
    assert_eq!(read_rule(by("GRBA"), r, 1), [0, 1000, 0, 1]);
    assert_eq!(read_rule(by("ARGB"), r, 1), [1, 1000, 0, 0]);
    assert_eq!(read_rule(by("1RG0"), r, 1), [1, 1000, 0, 0]);
    assert_eq!(read_rule(by("1RG0"), [0, 0, 0, 1], 1), [1, 0, 0, 0]);
    assert_eq!(read_rule(by("ARGB"), [0, 0, 0, 1], 1), [1, 0, 0, 0]);
    for (name, red) in [("RRRR", 11), ("GRBA", 22), ("ARGB", 22)] {
        assert_eq!(
            write_rule(by(name), [11, 22, 33, 44])[0],
            red,
            "R32Uint {name}"
        );
    }
}

// ---- The kernels ------------------------------------------------------------------------------

/// A texel type of the kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Texel {
    Float,
    Half,
    Uint,
}

impl Texel {
    /// The AIR vector, the intrinsic suffix, the Metal type, its vector's name and its size.
    fn air(self) -> (&'static str, &'static str, &'static str, &'static str, u32) {
        match self {
            Texel::Float => ("<4 x float>", "v4f32", "float", "float4", 16),
            Texel::Half => ("<4 x half>", "v4f16", "half", "half4", 8),
            Texel::Uint => ("<4 x i32>", "u.v4i32", "uint", "uint4", 16),
        }
    }

    /// The bits of a lane's one as the image holds it (`f32` bits for a float image).
    fn one(self) -> u32 {
        match self {
            Texel::Uint => 1,
            _ => 1.0f32.to_bits(),
        }
    }

    /// The image's format for a storage view of class `class`.
    fn format(self, class: StorageImageClass) -> vk::Format {
        match (self, class) {
            (Texel::Uint, _) => vk::Format::R32G32B32A32_UINT,
            (_, StorageImageClass::SwizzleBgra) => vk::Format::R8G8B8A8_UNORM,
            _ => vk::Format::R32G32B32A32_SFLOAT,
        }
    }
}

/// Bitcode of the module `body` (after the AIR header), or `None` without `llvm-as`.
fn assemble(name: &str, body: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-swizzled-storage-{}-{}-{}",
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

/// A kernel reading texture slot 0 (`access` `read` or `read_write`) at its thread's texel moved
/// `dx` texels right, storing the texel to `out[gid.x]` (buffer 0).
fn read_kernel(texel: Texel, access: &str, dx: u32) -> String {
    let (vec4, suffix, ty, out_ty, size) = texel.air();
    let code = if access == "read" { 1 } else { 3 };
    format!(
        r#"define void @k(ptr addrspace(1) %t, ptr addrspace(1) %out, <2 x i32> %gid) {{
  %x = extractelement <2 x i32> %gid, i64 0
  %i = zext i32 %x to i64
  %rx = add i32 %x, {dx}
  %at = insertelement <2 x i32> %gid, i32 %rx, i64 0
  %rs = call ptr addrspace(2) @air.get_read_sampler()
  %r = call {{ {vec4}, i8 }} @air.read_texture_2d.{suffix}(ptr addrspace(1) %t, ptr addrspace(2) %rs, <2 x i32> %at, <2 x i32> zeroinitializer, i32 0, i32 {code})
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
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.{access}", !"air.arg_type_name", !"texture2d<{ty}, {access}>", !"air.arg_name", !"t"}}
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

// ---- The interpreter --------------------------------------------------------------------------

/// Every committed profile, and `moltenvk-m3` and `lavapipe` without robustness, whose shader
/// checks its reads and guards its writes.
fn profiles() -> Vec<(String, DeviceProfile)> {
    let mut out = crate::spirv::testing::committed_profiles().expect("profiles");
    let all = [VkFeature::RobustImageAccess, VkFeature::RobustImageAccess2];
    for stem in ["moltenvk-m3", "lavapipe"] {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../metal2vulkan-validation/profiles")
            .join(format!("{stem}.json"));
        let text = std::fs::read_to_string(path).expect("profile");
        let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
        for f in &all {
            desc.features.remove(f);
        }
        let p = Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES)
            .expect("device")
            .profile()
            .clone();
        out.push((format!("{stem} unrobust"), p));
    }
    out
}

/// What the interpreter's image does outside itself on `p`: the device's texel, zero with the
/// format's alpha (`alpha`, a lane's bits), where `robustImageAccess2` covers the access; a
/// poisoned texel where the shader checks its reads under `robustImageAccess`; a trap where the
/// shader contains every access.
fn robust(p: &DeviceProfile, alpha: u32) -> Option<[u32; 4]> {
    let f = p.fields();
    if f.images.robust_image_access2 {
        Some([0, 0, 0, alpha])
    } else if f.images.robust_image_access {
        Some([0xDEAD_BEEF; 4])
    } else {
        None
    }
}

/// Lowering inputs viewing slot 0 in `class` (`texel`'s format), swizzled in the shader when
/// `swizzled`.
fn inputs(texel: Texel, class: StorageImageClass, swizzled: bool) -> LowerInputs {
    let mut i = LowerInputs::new();
    i.storage_images
        .insert(SLOT, StorageImageChoice::new(texel.format(class), class));
    if swizzled {
        i.swizzled_storage.insert(SLOT);
    }
    i
}

/// The image of slot 0: a row of [`WIDTH`] texels, `texels(n)` at texel `n`.
fn image(texels: impl Fn(u32) -> [u32; 4], robust: Option<[u32; 4]>) -> Image {
    let slot = ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Texture,
        index: 0,
    };
    let mut img = Image::new(slot, [WIDTH, 1, 1], 1, 1, 1);
    for n in 0..WIDTH {
        let at = img.index(0, 0, 0, [n, 0, 0]).expect("texel");
        img.texels[at] = texels(n);
    }
    img.robust = robust;
    img
}

/// Resources of a dispatch of one-thread threadgroups, buffer 0 holding `buffer`, with `spec`.
fn resources(plan: &Plan, buffer: Vec<u8>, spec: &[(SpecKey, u64)]) -> Resources {
    let mut map = BTreeMap::new();
    for axis in [Axis::X, Axis::Y, Axis::Z] {
        map.insert(SpecKey::LocalSize(axis), 1);
    }
    map.insert(SpecKey::FlatLocalSize, 1);
    map.extend(spec.iter().copied());
    let push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    for f in &plan.push.fields {
        match (f.field, f.loc) {
            (PushField::SlotBindOffset(_) | PushField::DispatchRegion, PushLoc::Push { .. }) => {}
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
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            workgroups: [WIDTH, 1, 1],
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

/// Texel `n`'s lanes as the image holds them: `f32` bits of values a `half` holds exactly, or
/// integers.
fn texel_of(texel: Texel, n: u32) -> [u32; 4] {
    match texel {
        Texel::Uint => [100 + n, 200 + n, 300 + n, 400 + n],
        _ => [0.25, 0.5, 0.75, 1.5].map(|x: f32| (x + n as f32).to_bits()),
    }
}

/// The Metal channels of image lanes `lanes` read through a view of `class`: `SwizzleBgra`'s
/// image holds blue in its first lane.
fn channels(class: StorageImageClass, lanes: [u32; 4]) -> [u32; 4] {
    match class {
        StorageImageClass::SwizzleBgra => [lanes[2], lanes[1], lanes[0], lanes[3]],
        _ => lanes,
    }
}

/// `x`, a lane the image holds, as the kernel stores it.
fn stored(texel: Texel, x: u32) -> u32 {
    match texel {
        Texel::Half => half_bits(f32::from_bits(x)),
        _ => x,
    }
}

/// A `read_write` read of each swizzle is Metal's: the converted texel swizzled, in bounds and
/// past the texture (Metal's zero texel, alpha the format's: 1 for one without alpha, 0 for one
/// with it), for `float`, `half` and `uint` results, through an identity view and a
/// `SwizzleBgra` one. The function states `TextureSwizzle` for the slot; without the variant it
/// states none and reads the texel unswizzled.
#[test]
fn a_storage_read_takes_the_bound_swizzle_out_of_bounds_too() {
    let (mut ran, mut outside) = (0, 0);
    for texel in [Texel::Float, Texel::Half, Texel::Uint] {
        let classes: &[StorageImageClass] = match texel {
            Texel::Uint => &[StorageImageClass::Identity],
            _ => &[StorageImageClass::Identity, StorageImageClass::SwizzleBgra],
        };
        for dx in [0, WIDTH] {
            let what = format!("{texel:?} dx {dx}");
            let Some(bc) = assemble(&what, &read_kernel(texel, "read_write", dx)) else {
                return;
            };
            for (stem, p) in profiles() {
                for &class in classes {
                    let what = format!("{what} {class:?} on {stem}");
                    let plain = inputs(texel, class, false);
                    let (plan, _) = typed_module_with(&bc, "k", &p, &plain)
                        .unwrap_or_else(|e| panic!("{what}: {e}"));
                    assert!(
                        !plan
                            .facts
                            .contains(&RuntimeFact::TextureSwizzle { slot: SLOT }),
                        "{what}"
                    );
                    let swizzled = inputs(texel, class, true);
                    let (plan, m) = typed_module_with(&bc, "k", &p, &swizzled)
                        .unwrap_or_else(|e| panic!("{what}: {e}"));
                    assert!(
                        plan.facts
                            .contains(&RuntimeFact::TextureSwizzle { slot: SLOT }),
                        "{what}"
                    );
                    let a = crate::pipeline::testing::compile_with(&bc, "k", &p, &swizzled)
                        .unwrap_or_else(|e| panic!("{what}: {e}"));
                    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
                    for alpha in [false, true] {
                        let alpha_bits = if alpha { texel.one() } else { 0 };
                        let img = image(|n| texel_of(texel, n), robust(&p, alpha_bits));
                        let size = if texel == Texel::Half { 8 } else { 16 };
                        for s in swizzles() {
                            let spec = [
                                (SpecKey::TextureSwizzle(SLOT), u64::from(s.word())),
                                (SpecKey::OutOfBoundsAlpha(SLOT), u64::from(alpha)),
                            ];
                            let res = resources(&plan, vec![0xab; size * WIDTH as usize], &spec);
                            let (out, _) = run(&m, &res, &img);
                            for n in 0..WIDTH {
                                let t = match dx {
                                    0 => channels(class, texel_of(texel, n)),
                                    _ => [0, 0, 0, alpha_bits],
                                };
                                let want = read_rule(s, t, texel.one()).map(|x| stored(texel, x));
                                let got: Vec<u32> =
                                    (0..4).map(|c| lane(&out, texel, n, c)).collect();
                                assert_eq!(got, want, "{what} {s:?} alpha {alpha}: texel {n}");
                            }
                        }
                        outside += usize::from(dx != 0);
                    }
                    ran += 1;
                }
            }
        }
    }
    assert!(ran > 0 && outside > 0, "ran {ran}, outside {outside}");
}

/// A write of each swizzle, `access::write` and `read_write`, stores Metal's channels: channel
/// `c` from the first lane whose source is `c`, else lane `c`, for `float`, `half` and `uint`
/// values, through an identity view and a `SwizzleBgra` one (which then swaps the stored lanes).
#[test]
fn a_storage_write_undoes_the_bound_swizzle() {
    let mut ran = 0;
    for texel in [Texel::Float, Texel::Half, Texel::Uint] {
        let classes: &[StorageImageClass] = match texel {
            Texel::Uint => &[StorageImageClass::Identity],
            _ => &[StorageImageClass::Identity, StorageImageClass::SwizzleBgra],
        };
        for access in ["write", "read_write"] {
            let what = format!("{texel:?} {access}");
            let Some(bc) = assemble(&what, &write_kernel(texel, access)) else {
                return;
            };
            for (stem, p) in profiles() {
                for &class in classes {
                    let what = format!("{what} {class:?} on {stem}");
                    let swizzled = inputs(texel, class, true);
                    let (plan, m) = typed_module_with(&bc, "k", &p, &swizzled)
                        .unwrap_or_else(|e| panic!("{what}: {e}"));
                    assert!(
                        plan.facts
                            .contains(&RuntimeFact::TextureSwizzle { slot: SLOT }),
                        "{what}"
                    );
                    let a = crate::pipeline::testing::compile_with(&bc, "k", &p, &swizzled)
                        .unwrap_or_else(|e| panic!("{what}: {e}"));
                    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
                    // Each thread's value as the kernel loads it, and as the image holds it.
                    let held = |n: u32| texel_of(texel, 4 * n);
                    let bytes: Vec<u8> = (0..WIDTH)
                        .flat_map(held)
                        .flat_map(|x| match texel {
                            Texel::Half => {
                                (half_bits(f32::from_bits(x)) as u16).to_le_bytes().to_vec()
                            }
                            _ => x.to_le_bytes().to_vec(),
                        })
                        .collect();
                    let img = image(|_| [0; 4], robust(&p, 0));
                    for s in swizzles() {
                        let spec = [(SpecKey::TextureSwizzle(SLOT), u64::from(s.word()))];
                        let res = resources(&plan, bytes.clone(), &spec);
                        let (_, texels) = run(&m, &res, &img);
                        for n in 0..WIDTH {
                            // A `SwizzleBgra` view's image holds Metal's blue in its first lane.
                            let want = channels(class, write_rule(s, held(n)));
                            let at = img.index(0, 0, 0, [n, 0, 0]).expect("texel");
                            assert_eq!(texels[at], want, "{what} {s:?}: texel {n}");
                        }
                    }
                    ran += 1;
                }
            }
        }
    }
    assert!(ran > 0);
}

/// An atomic ignores the swizzle (`atomic_fetch_add` through a swizzled
/// `R32Uint` view adds to red and returns the texel unswizzled): `rrp_sw_atomic_u` under the
/// variant states no `TextureSwizzle` and compiles to the same words as without it (the
/// interpreter does not model image atomics).
#[test]
fn an_atomic_ignores_the_bound_swizzle() {
    let bc = match llvm_assemble(&engine_dir().join("rrp_swizzled_storage.ll")) {
        Ok(b) => b,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("rrp_swizzled_storage: {why}"),
    };
    let atomic = StorageImageChoice::new(vk::Format::R32_UINT, StorageImageClass::Identity);
    let mut ran = 0;
    for (stem, p) in profiles() {
        let mut plain = LowerInputs::new();
        plain.storage_images.insert(SLOT, atomic);
        let mut swizzled = plain.clone();
        swizzled.swizzled_storage.insert(SLOT);
        let entry = "rrp_sw_atomic_u";
        let (Ok(a), Ok(b)) = (
            crate::pipeline::testing::compile_with(&bc, entry, &p, &plain),
            crate::pipeline::testing::compile_with(&bc, entry, &p, &swizzled),
        ) else {
            // A profile without the format's atomics refuses both alike.
            continue;
        };
        let (plan, _) =
            typed_module_with(&bc, entry, &p, &swizzled).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert!(
            !plan
                .facts
                .contains(&RuntimeFact::TextureSwizzle { slot: SLOT }),
            "{stem}"
        );
        assert_eq!(a.module.words, b.module.words, "{stem}");
        ran += 1;
    }
    assert!(ran > 0);
}

/// A `ulong` read of a slot under the variant is refused `NotImplemented(TextureSwizzle)`:
/// Metal's swizzled `ulong` texel is not measured.
#[test]
fn a_ulong_read_under_the_variant_is_refused() {
    let bc = match llvm_assemble(&engine_dir().join("unk_ulong_rw.ll")) {
        Ok(b) => b,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("unk_ulong_rw: {why}"),
    };
    let two_d = SlotRef { index: 2, ..SLOT };
    let mut refused = 0;
    for (stem, p) in profiles() {
        if typed_module_with(&bc, "unk_ulong_rw", &p, &LowerInputs::new()).is_err() {
            continue;
        }
        let mut i = LowerInputs::new();
        i.swizzled_storage.insert(two_d);
        match typed_module_with(&bc, "unk_ulong_rw", &p, &i) {
            Err(Error::Unsupported(u)) => {
                assert_eq!(u.feature, Feature::TextureSwizzle, "{stem}");
                refused += 1;
            }
            other => panic!("{stem}: {:?}", other.map(|_| ())),
        }
    }
    assert!(refused > 0);
}

// ---- The binder -------------------------------------------------------------------------------
