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
//! - The binder joins a storage slot bound to a swizzled texture to the variant, which plans the
//!   same key again, views the texture unswizzled (`ImageViewDesc::shader_swizzle`) and
//!   specializes the slot's swizzle word; a storage slot the function only queries views it
//!   unswizzled under no variant; a sampled slot keeps its swizzled view. A swizzled
//!   texture's shape states its level count.
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::bind::{BindState, BoundBuffer, BoundTexture, StageTable, TextureFacts, WriteSource};
use crate::budget::Budget;
use crate::device::profile::{Containment, DeviceProfile};
use crate::device::requirements::{VkFeature, VkProperty};
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{Error, Feature};
use crate::library::Library;
use crate::lower::select::{LowerInputs, Plan};
use crate::metal::{self, PixelFormat, TextureSwizzle as S, TextureSwizzleChannels as C};
use crate::metal::{TextureType, TextureUsage};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::lower_inputs::{StorageImageChoice, StorageImageClass};
use crate::pipeline::sym::{Axis, PushField, ResourceKind, ResourceSlot, SlotClass, SlotRef};
use crate::pipeline::sym::{SpecKey, Stage};
use crate::pipeline::testing::{
    engine_dir, link_and_validate, llvm_assemble, typed_module_with, AssembleError,
};
use crate::pipeline::StageSource;
use crate::pipeline::{ComputePipeline, ComputePipelineDescriptor, FunctionDescriptor};
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
    let all = [
        VkFeature::RobustImageAccess,
        VkFeature::RobustImageAccess2,
        VkFeature::RobustBufferAccess,
        VkFeature::RobustBufferAccess2,
    ];
    for stem in ["moltenvk-m3", "lavapipe"] {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../metal2vulkan-validation/profiles")
            .join(format!("{stem}.json"));
        let text = std::fs::read_to_string(path).expect("profile");
        let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
        for f in &all {
            desc.features.remove(f);
        }
        let p = Device::new(desc, DevicePolicy::default())
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
    let Containment::On { .. } = f.containment else {
        panic!("containment off")
    };
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

fn devices() -> Vec<(String, Device)> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles");
    let mut out = Vec::new();
    for e in std::fs::read_dir(&dir).expect("profiles") {
        let path = e.expect("entry").path();
        if path.extension().is_none_or(|x| x != "json") {
            continue;
        }
        let stem = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let text = std::fs::read_to_string(&path).expect("profile");
        let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
        out.push((
            stem,
            Device::new(desc, crate::device::DevicePolicy::default()).expect("device"),
        ));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// The kernel compiled into a compute pipeline under `inputs`, or `None` where the device needs a
/// storage slot's format as an input.
fn pipeline(dev: &Device, bc: &[u8], inputs: &LowerInputs) -> Option<ComputePipeline> {
    pipeline_of(dev, bc, "k", inputs)
}

/// [`pipeline`] of the kernel `entry`.
fn pipeline_of(
    dev: &Device,
    bc: &[u8],
    entry: &str,
    inputs: &LowerInputs,
) -> Option<ComputePipeline> {
    let lib = Library::from_air(bc).expect("library");
    let f = lib.function(entry).expect("function");
    let compiled =
        match dev.compile_function(&FunctionDescriptor::new(f), inputs, &Budget::standard()) {
            Ok(c) => c,
            Err(Error::NeedsInput(_)) => return None,
            Err(e) => panic!("compile: {e}"),
        };
    Some(
        dev.compile_compute_pipeline(
            &ComputePipelineDescriptor::new(StageSource::Compiled(compiled)),
            &Budget::standard(),
        )
        .expect("pipeline"),
    )
}

/// The facts of an `RGBA8Unorm` texture through `swizzle`: a view made with a swizzle, whose
/// usage Metal keeps only `ShaderRead` and `PixelFormatView` of, of 3 levels.
fn swizzled_facts(swizzle: C) -> TextureFacts {
    let mut facts = TextureFacts::new(TextureType::Type2D, PixelFormat::RGBA8Unorm);
    facts.usage = TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW;
    facts.swizzle = swizzle;
    facts.mipmap_level_count = 3;
    facts
}

/// A kernel's state: texture slot 0 holding `facts`, buffer 0.
fn state(facts: TextureFacts) -> BindState {
    let mut t = StageTable::default();
    t.textures.insert(
        0,
        BoundTexture::Texture {
            facts,
            resource_id: None,
        },
    );
    t.buffers.insert(
        0,
        BoundBuffer::Buffer {
            length: 65536,
            offset: 0,
            gpu_address: 0x1_0000,
            device_address: None,
            attribute_stride: None,
        },
    );
    let mut s = BindState::default();
    s.stages.insert(Stage::Kernel, t);
    s
}

/// A storage slot bound to a swizzled view joins the plan's variant; the variant's pipeline plans
/// the same key for the same shape, binds the view unswizzled for the function to swizzle
/// (`shader_swizzle`, the identity mapping) and specializes the slot's `TextureSwizzle` to the
/// swizzle's word. An unswizzled texture leaves the variant, and a sampled slot (`access::read`)
/// keeps its view's swizzle, which the consumer composes. On every committed profile.
#[test]
fn a_storage_slot_bound_to_a_swizzled_texture_joins_the_variant() {
    let gbar = C::new(S::Green, S::Blue, S::Alpha, S::Red);
    let (mut planned, mut sampled) = (0, 0);
    for access in ["read_write", "read"] {
        let Some(bc) = assemble(access, &read_kernel(Texel::Float, access, 0)) else {
            return;
        };
        for (stem, dev) in devices() {
            let Some(plain) = pipeline(&dev, &bc, &LowerInputs::new()) else {
                continue;
            };
            let what = format!("{access} on {stem}");
            let shape = state(swizzled_facts(gbar)).shape();
            let plan = plain
                .plan(&dev, &shape)
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            let view = |plan: &crate::bind::BindingPlan| {
                plan.descriptors
                    .iter()
                    .find_map(|d| match &d.source {
                        WriteSource::Texture { view, .. } => Some(*view),
                        _ => None,
                    })
                    .expect("the texture's view")
            };
            if access == "read" {
                assert!(plan.variant.swizzled_storage.is_empty(), "{what}");
                assert!(!view(&plan).shader_swizzle, "{what}");
                sampled += 1;
                continue;
            }
            assert_eq!(
                plan.variant.swizzled_storage,
                BTreeSet::from([SLOT]),
                "{what}"
            );
            let key = plan.variant.lowering_inputs(Stage::Kernel);
            assert_eq!(
                key.swizzled_storage, plan.variant.swizzled_storage,
                "{what}"
            );
            let variant = pipeline(&dev, &bc, &key).expect("the variant's inputs");
            let again = variant
                .plan(&dev, &shape)
                .unwrap_or_else(|e| panic!("{what}: variant {e}"));
            assert_eq!(again.variant, plan.variant, "{what}");
            let v = view(&again);
            assert!(v.shader_swizzle, "{what}");
            assert_eq!(v.components, vk::ComponentMapping::IDENTITY, "{what}");
            let id = variant
                .stage
                .specialization
                .constants
                .iter()
                .find_map(|c| (c.key == SpecKey::TextureSwizzle(SLOT)).then_some(c.id))
                .unwrap_or_else(|| panic!("{what}: no TextureSwizzle constant"));
            let e = again
                .specialization
                .entries
                .iter()
                .find(|e| e.constant_id == id)
                .expect("TextureSwizzle entry");
            let at = e.offset as usize;
            let word = u32::from_le_bytes(
                again.specialization.data[at..at + 4]
                    .try_into()
                    .expect("a u32"),
            );
            assert_eq!(word, gbar.word(), "{what}");
            let own = plain
                .plan(&dev, &state(swizzled_facts(C::DEFAULT)).shape())
                .unwrap_or_else(|e| panic!("{what}: unswizzled {e}"));
            assert!(own.variant.swizzled_storage.is_empty(), "{what}");
            planned += 1;
        }
    }
    assert!(
        planned > 0 && sampled > 0,
        "{planned} planned, {sampled} sampled"
    );
}

/// A kernel storing texture slot 0's width (`access` `write` or `read_write`), which it neither
/// reads nor writes, to `out[0]` (buffer 0).
fn query_kernel(access: &str) -> String {
    format!(
        r#"define void @k(ptr addrspace(1) %t, ptr addrspace(1) %out) {{
  %w = call i32 @air.get_width_texture_2d(ptr addrspace(1) %t, i32 0)
  store i32 %w, ptr addrspace(1) %out, align 4
  ret void
}}

declare i32 @air.get_width_texture_2d(ptr addrspace(1), i32)

!air.kernel = !{{!0}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.{access}", !"air.arg_type_name", !"texture2d<float, {access}>", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}}
"#
    )
}

/// A storage slot its function only queries, bound to a swizzled view, reads and writes no texel,
/// and no swizzle reaches its width (Metal answers the view's size: our probe
/// `opencl_swizzled_images`, job m3-max-16336): its plan joins no variant and binds the view with
/// no swizzle (`shader_swizzle`, the identity mapping), where a Vulkan storage view could take
/// none and the consumer refused it. On every committed profile.
#[test]
fn a_storage_slot_only_queried_binds_a_swizzled_texture_unswizzled() {
    let bgra = C::new(S::Blue, S::Green, S::Red, S::Alpha);
    let mut planned = 0;
    for access in ["write", "read_write"] {
        let Some(bc) = assemble(&format!("query {access}"), &query_kernel(access)) else {
            return;
        };
        for (stem, dev) in devices() {
            let Some(plain) = pipeline(&dev, &bc, &LowerInputs::new()) else {
                continue;
            };
            let what = format!("{access} on {stem}");
            let plan = plain
                .plan(&dev, &state(swizzled_facts(bgra)).shape())
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            assert!(plan.variant.swizzled_storage.is_empty(), "{what}");
            let view = plan
                .descriptors
                .iter()
                .find_map(|d| match &d.source {
                    WriteSource::Texture { view, .. } => Some(*view),
                    _ => None,
                })
                .unwrap_or_else(|| panic!("{what}: the texture's view"));
            assert!(view.shader_swizzle, "{what}");
            assert_eq!(view.components, vk::ComponentMapping::IDENTITY, "{what}");
            planned += 1;
        }
    }
    assert!(planned > 0);
}

/// The value `plan` specializes `pipeline`'s constant `key` to.
fn spec_word(pipeline: &ComputePipeline, plan: &crate::bind::BindingPlan, key: SpecKey) -> u32 {
    let id = pipeline
        .stage
        .specialization
        .constants
        .iter()
        .find_map(|c| (c.key == key).then_some(c.id))
        .unwrap_or_else(|| panic!("no {key:?} constant"));
    let e = plan
        .specialization
        .entries
        .iter()
        .find(|e| e.constant_id == id)
        .unwrap_or_else(|| panic!("no {key:?} entry"));
    let at = e.offset as usize;
    u32::from_le_bytes(
        plan.specialization.data[at..at + 4]
            .try_into()
            .expect("a u32"),
    )
}

/// A sampled slot (`access::read`) bound to a swizzled texture of a packed normalized format
/// joins the plan's variant on a device that does not sample through Metal's own sampler, whose
/// view's swizzle would move a channel's code onto another channel's lane before the function
/// converts it by its own code scale: the variant's pipeline plans the same key for the same
/// shape, binds the view unswizzled for the function to swizzle (`shader_swizzle`, the
/// unswizzled texture's mapping) and specializes the slot's `TextureSwizzle` to the swizzle's word.
/// A swizzle that reads every
/// texel as none does (`RGB1` of `B5G6R5Unorm`, whose alpha reads 1) leaves the variant, and a
/// device with Metal's own sampler keeps the view's swizzle. On every committed profile that
/// holds the format.
#[test]
fn a_sampled_slot_bound_to_a_swizzled_packed_texture_swizzles_in_its_function() {
    let gbar = C::new(S::Green, S::Blue, S::Alpha, S::Red);
    let rgb1 = C::new(S::Red, S::Green, S::Blue, S::One);
    let Some(bc) = assemble("packed read", &read_kernel(Texel::Float, "read", 0)) else {
        return;
    };
    let facts = |pf: PixelFormat, swizzle: C| {
        let mut facts = swizzled_facts(swizzle);
        facts.pixel_format = pf;
        facts
    };
    let view = |plan: &crate::bind::BindingPlan| {
        plan.descriptors
            .iter()
            .find_map(|d| match &d.source {
                WriteSource::Texture { view, .. } => Some(*view),
                _ => None,
            })
            .expect("the texture's view")
    };
    let (mut joined, mut kept) = (0, 0);
    for (stem, dev) in devices() {
        let Some(plain) = pipeline(&dev, &bc, &LowerInputs::new()) else {
            continue;
        };
        let metal = dev
            .profile()
            .fields()
            .enabled
            .properties
            .contains(&VkProperty::MetalSampler);
        for pf in [
            PixelFormat::B5G6R5Unorm,
            PixelFormat::A1BGR5Unorm,
            PixelFormat::BGR5A1Unorm,
            PixelFormat::ABGR4Unorm,
        ] {
            let what = format!("{pf:?} on {stem}");
            let shape = state(facts(pf, gbar)).shape();
            let plan = match plain.plan(&dev, &shape) {
                Ok(plan) => plan,
                // A device that holds no such texture refuses it unswizzled too.
                Err(e) => {
                    let own = plain.plan(&dev, &state(facts(pf, C::DEFAULT)).shape());
                    assert!(own.is_err(), "{what}: {e}, but unswizzled it plans");
                    continue;
                }
            };
            if metal {
                assert!(plan.variant.swizzled_storage.is_empty(), "{what}");
                assert!(!view(&plan).shader_swizzle, "{what}");
                kept += 1;
                continue;
            }
            assert_eq!(
                plan.variant.swizzled_storage,
                BTreeSet::from([SLOT]),
                "{what}"
            );
            let key = plan.variant.lowering_inputs(Stage::Kernel);
            let variant = pipeline(&dev, &bc, &key).expect("the variant's inputs");
            let again = variant
                .plan(&dev, &shape)
                .unwrap_or_else(|e| panic!("{what}: variant {e}"));
            assert_eq!(again.variant, plan.variant, "{what}");
            // The view's mapping is the unswizzled texture's: its format's own alone.
            let own = plain
                .plan(&dev, &state(facts(pf, C::DEFAULT)).shape())
                .unwrap_or_else(|e| panic!("{what}: unswizzled {e}"));
            let v = view(&again);
            assert!(v.shader_swizzle, "{what}");
            assert_eq!(v.components, view(&own).components, "{what}");
            let word = spec_word(&variant, &again, SpecKey::TextureSwizzle(SLOT));
            assert_eq!(word, gbar.word(), "{what}");
            if pf == PixelFormat::B5G6R5Unorm {
                let own = plain
                    .plan(&dev, &state(facts(pf, rgb1)).shape())
                    .unwrap_or_else(|e| panic!("{what}: RGB1 {e}"));
                assert!(own.variant.swizzled_storage.is_empty(), "{what}");
            }
            joined += 1;
        }
    }
    assert!(joined > 0 && kept > 0, "{joined} joined, {kept} kept");
}

/// A sample through a runtime sampler (`fpv_swizzled_reads`' `fpv_swz_sample`, a combined image
/// sampler) of a swizzled packed normalized texture joins the variant as a read does, on every
/// committed profile without Metal's own sampler: its combined image's view takes no swizzle
/// (`shader_swizzle`, the unswizzled texture's mapping), the variant plans the same key again and
/// specializes the slot's swizzle. With Metal's own sampler the view keeps the swizzle.
#[test]
fn a_combined_sampler_of_a_swizzled_packed_texture_swizzles_in_its_function() {
    use crate::bind::{BoundSampler, ImageSource, SamplerDescriptor};
    let gbar = C::new(S::Green, S::Blue, S::Alpha, S::Red);
    let Some(bc) = (match llvm_assemble(&engine_dir().join("fpv_swizzled_reads.ll")) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("fpv_swizzled_reads.ll: {why}"),
    }) else {
        return;
    };
    let shape = |pf: PixelFormat, swizzle: C| {
        let mut facts = swizzled_facts(swizzle);
        facts.pixel_format = pf;
        let mut s = state(facts);
        let t = s
            .stages
            .get_mut(&Stage::Kernel)
            .expect("the kernel's table");
        t.buffers.insert(
            1,
            BoundBuffer::Buffer {
                length: 65536,
                offset: 0,
                gpu_address: 0x2_0000,
                device_address: None,
                attribute_stride: None,
            },
        );
        t.samplers
            .insert(0, BoundSampler::new(SamplerDescriptor::new()));
        s.shape()
    };
    let view = |plan: &crate::bind::BindingPlan| {
        plan.descriptors
            .iter()
            .find_map(|d| match &d.source {
                WriteSource::CombinedImageSampler {
                    image: ImageSource::Texture { view, .. },
                    ..
                } => Some(*view),
                _ => None,
            })
            .expect("the combined image's view")
    };
    let (mut joined, mut kept) = (0, 0);
    for (stem, dev) in devices() {
        let entry = "fpv_swz_sample";
        let Some(plain) = pipeline_of(&dev, &bc, entry, &LowerInputs::new()) else {
            continue;
        };
        let metal = dev
            .profile()
            .fields()
            .enabled
            .properties
            .contains(&VkProperty::MetalSampler);
        for pf in [
            PixelFormat::B5G6R5Unorm,
            PixelFormat::A1BGR5Unorm,
            PixelFormat::BGR5A1Unorm,
            PixelFormat::ABGR4Unorm,
        ] {
            let what = format!("{pf:?} on {stem}");
            let swizzled = shape(pf, gbar);
            let Ok(plan) = plain.plan(&dev, &swizzled) else {
                let own = plain.plan(&dev, &shape(pf, C::DEFAULT));
                assert!(
                    own.is_err(),
                    "{what}: refused swizzled, but unswizzled it plans"
                );
                continue;
            };
            if metal {
                assert!(plan.variant.swizzled_storage.is_empty(), "{what}");
                assert!(!view(&plan).shader_swizzle, "{what}");
                kept += 1;
                continue;
            }
            assert_eq!(
                plan.variant.swizzled_storage,
                BTreeSet::from([SLOT]),
                "{what}"
            );
            let key = plan.variant.lowering_inputs(Stage::Kernel);
            let variant = pipeline_of(&dev, &bc, entry, &key).expect("the variant's inputs");
            let again = variant
                .plan(&dev, &swizzled)
                .unwrap_or_else(|e| panic!("{what}: variant {e}"));
            assert_eq!(again.variant, plan.variant, "{what}");
            let own = plain
                .plan(&dev, &shape(pf, C::DEFAULT))
                .unwrap_or_else(|e| panic!("{what}: unswizzled {e}"));
            let v = view(&again);
            assert!(v.shader_swizzle, "{what}");
            assert_eq!(v.components, view(&own).components, "{what}");
            let word = spec_word(&variant, &again, SpecKey::TextureSwizzle(SLOT));
            assert_eq!(word, gbar.word(), "{what}");
            joined += 1;
        }
    }
    assert!(joined > 0 && kept > 0, "{joined} joined, {kept} kept");
}

/// A swizzled texture's shape states its level count, which a storage slot reading its levels
/// needs, though Metal strips its usage of `ShaderWrite`; an unswizzled texture without
/// `ShaderWrite` states it too, as plain Metal writes one at every level.
#[test]
fn a_swizzled_texture_s_shape_states_its_levels() {
    let gbar = C::new(S::Green, S::Blue, S::Alpha, S::Red);
    let levels = |facts: TextureFacts| match facts.shape() {
        crate::bind::TextureShape::Texture {
            mipmap_level_count, ..
        } => mipmap_level_count,
        other => panic!("{other:?}"),
    };
    assert_eq!(levels(swizzled_facts(gbar)), 3);
    assert_eq!(levels(swizzled_facts(C::DEFAULT)), 3);
    let mut written = swizzled_facts(C::DEFAULT);
    written.usage = metal::TextureUsage::SHADER_READ | metal::TextureUsage::SHADER_WRITE;
    assert_eq!(levels(written), 3);
}
