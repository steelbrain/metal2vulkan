//! Metal's bicubic filter emulated by fetches, run on the CPU interpreter against an exact
//! reference of the emulated filter. It takes from Metal (Apple M3 Max, macOS 26.7: the authored
//! `bcu_*` cases):
//! - per plane axis, the base tap `floor(256 (x - 1/2) + 1/2) >> 8` and the fraction rounded half
//!   up to 1/256;
//! - each tap addressed on its own (edge, repeat, or the border color through the bound format's
//!   channels);
//! - the bound format's clamp or `half` rounding of the sum, chosen by the texture slot's
//!   `BicubicTexels` specialization;
//! - Metal's fixed-point weights (`metal::bicubic::weights`), or the bilinear inner two for a
//!   32-bit float format and for every 1D texture;
//! - the exact sum of an `EXACT` format's codes, rounded half up onto its grid.
//!
//! An `EXACT` format's lane must be the reference's exactly. The other formats' emulation sums in
//! `f32` against the reference's exact arithmetic, so a lane may differ by the `f32` rounding of
//! sixteen products (and one `half` ulp where a sum rounds to `half` near a tie); everything the
//! filter selects (taps, fractions, borders, clamps) is exact.
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::device::profile::{DeviceProfile, FloatFunctions};
use crate::device::DevicePolicy;
use crate::error::{Error, Feature, Needs, Unsupported};
use crate::lower::select::Plan;
use crate::metal::bicubic::{texels, BicubicTexels};
use crate::metal::PixelFormat;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{
    Axis, PushField, ResourceKind, ResourceSlot, SlotClass, SlotRef, SpecKey, Stage,
};
use crate::pipeline::testing::{
    compile, compile_with, engine_dir, link_and_validate, llvm_assemble, profile, typed_module,
    typed_module_with, AssembleError,
};
use crate::pipeline::PrecisionNote;
use crate::spirv::interp::{run_with_images, Buffer, Config, Image, Order, Resources};
use crate::spirv::ir::{GlobalKind, Inst, TypedModule};

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// Bitcode of the module `body` (after the AIR header), or `None` without `llvm-as`.
fn assemble(name: &str, body: &str) -> Option<Vec<u8>> {
    assemble_module(name, &format!("{HEADER}{body}"))
}

/// Bitcode of the whole module `text`, or `None` without `llvm-as`.
fn assemble_module(name: &str, text: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-bicubic-{}-{}-{name}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("k.ll");
    std::fs::write(&ll, text).expect("write");
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

fn m3() -> DeviceProfile {
    profile("moltenvk-m3", DevicePolicy::default()).expect("profile")
}

/// The words of `constexpr sampler(filter::bicubic, ...)` with every axis addressed by `address`
/// (AIR code: 0 zero or border, 1 edge, 2 repeat, 3 mirrored repeat), `pixel` coordinates, and
/// border color code `border` (0 transparent black, 2 opaque white), as Apple's compiler
/// writes them (the `bcu_address` fixture).
fn bicubic_words(address: u64, pixel: bool, border: u64) -> u64 {
    const BASE: u64 = 0x007b_ff00_0008_1400;
    BASE | address | address << 3 | address << 6 | u64::from(pixel) << 15 | border << 56
}

/// `words` with its `s` axis (`axis` 0) or its `t` axis (1) addressed by AIR code `address`.
fn with_axis(words: u64, axis: u32, address: u64) -> u64 {
    words & !(7 << (3 * axis)) | address << (3 * axis)
}

/// How a test kernel samples.
#[derive(Clone, Copy)]
enum Form {
    /// `texture2d<float>`: `t.sample(s, coord[tid], offset)` with constant `offset`.
    Plain { offset: (i32, i32) },
    /// `texture2d_array<float>`: layer `layer`.
    Array { layer: i32 },
    /// `texture1d<float>`: `coord[tid].x`.
    OneD,
    /// `texture2d<float>` with `min_lod_clamp(1)`.
    MinLodClamp,
    /// `texture2d<float>` with lane `tid`'s dynamic `int2` offset (buffer 2).
    DynamicOffset,
}

/// A kernel sampling texture slot 0 through the constexpr sampler `words` at `float2` buffer 0's
/// lane `tid`, storing the `float4` to buffer 1.
fn kernel(words: u64, form: Form) -> String {
    let (tex_ty, call, extra_param, extra_md) = match form {
        Form::Plain { offset: (ox, oy) } => (
            "texture2d<float, sample>",
            format!(
                "@air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) @__air_sampler_state, <2 x float> %uv, i1 true, <2 x i32> <i32 {ox}, i32 {oy}>, i1 false, float 0.0, float 0.0, i32 0)"
            ),
            "",
            "",
        ),
        Form::Array { layer } => (
            "texture2d_array<float, sample>",
            format!(
                "@air.sample_texture_2d_array.v4f32(ptr addrspace(1) %t, ptr addrspace(2) @__air_sampler_state, <2 x float> %uv, i32 {layer}, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)"
            ),
            "",
            "",
        ),
        Form::OneD => (
            "texture1d<float, sample>",
            "@air.sample_texture_1d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) @__air_sampler_state, float %u, i1 false, i32 0, i1 false, float 0.0, float 0.0, i32 0)".to_string(),
            "",
            "",
        ),
        Form::MinLodClamp => (
            "texture2d<float, sample>",
            "@air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) @__air_sampler_state, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 1.0, i32 0)".to_string(),
            "",
            "",
        ),
        Form::DynamicOffset => (
            "texture2d<float, sample>",
            "@air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) @__air_sampler_state, <2 x float> %uv, i1 true, <2 x i32> %o, i1 false, float 0.0, float 0.0, i32 0)".to_string(),
            ", ptr addrspace(1) %offs",
            ", !8",
        ),
    };
    let (decl_name, decl_args) = match form {
        Form::Array { .. } => (
            "air.sample_texture_2d_array.v4f32",
            "ptr addrspace(1), ptr addrspace(2), <2 x float>, i32, i1, <2 x i32>, i1, float, float, i32",
        ),
        Form::OneD => (
            "air.sample_texture_1d.v4f32",
            "ptr addrspace(1), ptr addrspace(2), float, i1, i32, i1, float, float, i32",
        ),
        _ => (
            "air.sample_texture_2d.v4f32",
            "ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32",
        ),
    };
    let load_offset = if matches!(form, Form::DynamicOffset) {
        "  %po = getelementptr inbounds <2 x i32>, ptr addrspace(1) %offs, i64 %i\n  %o = load <2 x i32>, ptr addrspace(1) %po, align 8\n"
    } else {
        ""
    };
    format!(
        r#"@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {words}, i64 0], align 8

define void @k(ptr addrspace(1) %t, ptr addrspace(1) %c, ptr addrspace(1) %out{extra_param}, i32 %tid) {{
  %i = zext i32 %tid to i64
  %p = getelementptr inbounds <2 x float>, ptr addrspace(1) %c, i64 %i
  %uv = load <2 x float>, ptr addrspace(1) %p, align 8
  %u = extractelement <2 x float> %uv, i64 0
{load_offset}  %r = call {{ <4 x float>, i8 }} {call}
  %v = extractvalue {{ <4 x float>, i8 }} %r, 0
  %q = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %i
  store <4 x float> %v, ptr addrspace(1) %q, align 16
  ret void
}}

declare {{ <4 x float>, i8 }} @{decl_name}({decl_args})

!air.kernel = !{{!0}}
!air.sampler_states = !{{!7}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5{extra_md}, !6}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"{tex_ty}", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"float2", !"air.arg_name", !"c"}}
!5 = !{{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}}
!6 = !{{i32 {tid_index}, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}}
!7 = !{{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}}
!8 = !{{i32 3, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"int2", !"air.arg_name", !"offs"}}
"#,
        tid_index = if matches!(form, Form::DynamicOffset) {
            4
        } else {
            3
        },
    )
}

/// A kernel whose texture slot 0 is a `tex_ty` (`texture2d<float, sample>`, `depth2d<float,
/// sample>`, ...) sampled through the constexpr sampler `words` by `body`, which may read the
/// lane's `float2` coordinate `%uv` and its `float3` extension `%uvw` and defines the stored
/// `%v` of type `value_ty`; `decl` declares the intrinsic `body` calls.
fn probe_kernel(words: u64, tex_ty: &str, body: &str, decl: &str, value_ty: &str) -> String {
    format!(
        r#"@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {words}, i64 0], align 8

define void @k(ptr addrspace(1) %t, ptr addrspace(1) %c, ptr addrspace(1) %out, i32 %tid) {{
  %i = zext i32 %tid to i64
  %p = getelementptr inbounds <2 x float>, ptr addrspace(1) %c, i64 %i
  %uv = load <2 x float>, ptr addrspace(1) %p, align 8
  %uvw = shufflevector <2 x float> %uv, <2 x float> zeroinitializer, <3 x i32> <i32 0, i32 1, i32 2>
  {body}
  %q = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %i
  store {value_ty} %v, ptr addrspace(1) %q, align 16
  ret void
}}

declare {decl}

!air.kernel = !{{!0}}
!air.sampler_states = !{{!7}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5, !6}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"{tex_ty}", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"float2", !"air.arg_name", !"c"}}
!5 = !{{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}}
!6 = !{{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}}
!7 = !{{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}}
"#
    )
}

/// A fragment function returning the LOD Metal's `calculate_clamped_lod` gives a
/// `texture2d<float>` sampled through the constexpr sampler `words` at the fragment's position
/// (an LOD query needs fragment derivatives).
fn lod_fragment(words: u64) -> String {
    format!(
        r#"@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {words}, i64 0], align 8

define <4 x float> @k(<4 x float> %pos, ptr addrspace(1) %t) {{
  %uv = shufflevector <4 x float> %pos, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %l = call float @air.calculate_clamped_lod_texture_2d(ptr addrspace(1) %t, ptr addrspace(2) @__air_sampler_state, <2 x float> %uv, i32 0)
  %v = insertelement <4 x float> zeroinitializer, float %l, i64 0
  ret <4 x float> %v
}}

declare float @air.calculate_clamped_lod_texture_2d(ptr addrspace(1), ptr addrspace(2), <2 x float>, i32)

!air.fragment = !{{!0}}
!air.sampler_states = !{{!6}}
!0 = !{{ptr @k, !1, !3}}
!1 = !{{!2}}
!2 = !{{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}}
!3 = !{{!4, !5}}
!4 = !{{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}}
!5 = !{{i32 1, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}}
!6 = !{{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}}
"#
    )
}

/// The combined image-sampler binding a module declares for texture slot 0.
fn combined_slot(m: &TypedModule) -> ResourceSlot {
    m.globals()
        .iter()
        .find_map(|g| match g.kind {
            GlobalKind::Resource(
                s @ ResourceSlot {
                    kind: ResourceKind::CombinedTextureSampler(_),
                    ..
                },
            ) => Some(s),
            _ => None,
        })
        .expect("a combined binding")
}

const TEXTURE: SlotRef = SlotRef {
    stage: Stage::Kernel,
    class: SlotClass::Texture,
    index: 0,
};

/// Resources of a dispatch of one-thread threadgroups: `[[buffer(i)]]` holding `init[i]`, and
/// the texture slot's `BicubicTexels` specialization `class`.
fn resources(plan: &Plan, init: &[Vec<u8>], class: BicubicTexels) -> Resources {
    let mut spec = BTreeMap::new();
    for axis in [Axis::X, Axis::Y, Axis::Z] {
        spec.insert(SpecKey::LocalSize(axis), 1);
    }
    spec.insert(SpecKey::FlatLocalSize, 1);
    spec.insert(SpecKey::BicubicTexels(TEXTURE), u64::from(class.0));
    let push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    for f in &plan.push.fields {
        match (f.field, f.loc) {
            (PushField::SlotBindOffset(_) | PushField::DispatchRegion, PushLoc::Push { .. }) => {}
            // A bound value read as data, which the interpreter writes from `spec`.
            (PushField::Specialized(_), PushLoc::Push { .. }) => {}
            other => panic!("unexpected push field {other:?}"),
        }
    }
    let buffers = init
        .iter()
        .enumerate()
        .map(|(i, b)| Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index: i as u32,
            }),
            address: None,
            bytes: b.clone(),
        })
        .collect();
    Resources {
        buffers,
        push,
        spec,
    }
}

/// `m` over `groups` one-thread threadgroups with `image` bound, in both scheduling orders, which
/// must agree: buffer 1's `float4`s.
fn run(m: &TypedModule, res: &Resources, image: &Image, groups: u32) -> Vec<[f32; 4]> {
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            workgroups: [groups, 1, 1],
            order,
            ..Config::default()
        };
        let (o, _) = run_with_images(m, &cfg, res.clone(), vec![image.clone()])
            .unwrap_or_else(|e| panic!("interpreter: {e:?}"));
        outs.push(o.buffers);
    }
    assert_eq!(outs[0], outs[1], "the scheduling orders disagree");
    outs[0][1]
        .as_chunks::<16>()
        .0
        .iter()
        .map(|c| {
            let f = |k: usize| f32::from_le_bytes([c[k], c[k + 1], c[k + 2], c[k + 3]]);
            [f(0), f(4), f(8), f(12)]
        })
        .collect()
}

/// A deterministic texel source: a 64-bit LCG.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A texture's texels as the bound format delivers them (one layer, `w` by `h`).
struct Texels {
    w: i64,
    h: i64,
    t: Vec<[f64; 4]>,
}

impl Texels {
    /// Random texels in the range of the format of `class` (`[0, 1]` normalized unsigned,
    /// `[-1, 1]` signed, `[-1, 2]` float, a `half` format's values rounded to `half`, an `EXACT`
    /// format's on its codes), with the format's missing channels as a fetch reads them (0 for
    /// red, green, blue; 1 for alpha).
    fn random(class: BicubicTexels, w: i64, h: i64, seed: u64) -> Texels {
        let mut rng = Lcg(seed);
        let (lo, hi) = if class.0 & BicubicTexels::UNORM != 0 {
            (0.0, 1.0)
        } else if class.0 & BicubicTexels::SNORM != 0 {
            (-1.0, 1.0)
        } else {
            (-1.0, 2.0)
        };
        let t = (0..w * h)
            .map(|_| {
                let mut x = [0.0, 0.0, 0.0, 1.0];
                for (c, v) in x.iter_mut().enumerate() {
                    let r = lo + (hi - lo) * rng.next();
                    if class.has_channel(c as u32) {
                        *v = if class.0 & BicubicTexels::HALF != 0 {
                            f64::from(half(r as f32))
                        } else if let Some(scale) = code_scale(class) {
                            f64::from(((r * scale).round() / scale) as f32)
                        } else {
                            f64::from(r as f32)
                        };
                    }
                }
                x
            })
            .collect();
        Texels { w, h, t }
    }

    /// The interpreter image of these texels in slot `slot`, as `layers` copies (layer `k`'s
    /// texels scaled by `2^k`, so a wrong layer shows, and a `half` format's texels stay the
    /// `half` values the format holds).
    fn image(&self, slot: ResourceSlot, dims: u32, layers: u32) -> Image {
        let h = if dims == 1 { 1 } else { self.h as u32 };
        let mut image = Image::new(slot, [self.w as u32, h, 1], layers, 1, 1);
        for l in 0..layers {
            for y in 0..h {
                for x in 0..self.w as u32 {
                    let at = image.index(0, l, 0, [x, y, 0]).expect("texel");
                    let v = self.t[(y as i64 * self.w + x as i64) as usize];
                    let k = f64::from(1u32 << l);
                    image.texels[at] = v.map(|c| ((c * k) as f32).to_bits());
                }
            }
        }
        image
    }
}

/// The spacing of `half` values at magnitude `a`: `2^(e - 10)` for a normal half of exponent
/// `e`, `2^-24` below the smallest normal.
fn half_ulp(a: f64) -> f64 {
    let a = a.abs() as f32;
    if a < 6.103_515_6e-5 {
        return 2f64.powi(-24);
    }
    let e = ((a.to_bits() >> 23) & 0xff) as i32 - 127;
    2f64.powi(e - 10)
}

/// `x` rounded to `half` and back (round to nearest even, overflow to infinity).
fn half(x: f32) -> f32 {
    if x.is_nan() || x.is_infinite() {
        return x;
    }
    if x.abs() >= 65520.0 {
        return f32::INFINITY.copysign(x);
    }
    let q = half_ulp(f64::from(x));
    ((f64::from(x) / q).round_ties_even() * q) as f32
}

/// How a tap outside the level is addressed.
#[derive(Clone, Copy, Debug)]
enum Address {
    Edge,
    Repeat,
    Border([f64; 4]),
    /// `mirrored_repeat` as the linear filter reflects a tap.
    Mirror,
}

/// Metal's split of axis coordinate `x` (texels): the base tap and the fraction, rounded half up
/// to 1/256. Where `q` does not fit an `i32` it saturates into `[-2^31, 2^31 - 128]`, the
/// translation's choice (Metal's value there is not measured).
fn split(x: f64) -> (i64, f64) {
    let q = ((x - 0.5) * 256.0 + 0.5).floor() as i64;
    let q = q.clamp(-(1 << 31), (1 << 31) - 128);
    (q >> 8, (q & 255) as f64 / 256.0)
}

/// The code scale `2^b - 1` (`2^(b-1) - 1` signed) of an 8- or 16-bit normalized format of
/// `class`, whose texels are codes.
fn code_scale(class: BicubicTexels) -> Option<f64> {
    if class.0 & BicubicTexels::EXACT == 0 {
        return None;
    }
    let wide = class.0 & BicubicTexels::WIDE16 != 0;
    let signed = class.0 & BicubicTexels::SNORM != 0;
    Some(match (wide, signed) {
        (false, false) => 255.0,
        (false, true) => 127.0,
        (true, false) => 65535.0,
        (true, true) => 32767.0,
    })
}

/// The Catmull-Rom weights (`a = -1/2`) at fraction `f`.
fn catmull_rom(f: f64) -> [f64; 4] {
    let (f2, f3) = (f * f, f * f * f);
    [
        -0.5 * f3 + f2 - 0.5 * f,
        1.5 * f3 - 2.5 * f2 + 1.0,
        -1.5 * f3 + 2.0 * f2 + 0.5 * f,
        0.5 * f3 - 0.5 * f2,
    ]
}

/// The border color `b` through the channels of `class`.
fn through_channels(class: BicubicTexels, b: [f64; 4]) -> [f64; 4] {
    let missing = [0.0, 0.0, 0.0, 1.0];
    std::array::from_fn(|c| {
        if class.has_channel(c as u32) {
            b[c]
        } else {
            missing[c]
        }
    })
}

/// The texel of tap `(i, j)` under `address`, layer factor `k`.
fn tap(tex: &Texels, class: BicubicTexels, address: Address, i: i64, j: i64, k: f64) -> [f64; 4] {
    let fix = |v: i64, n: i64| match address {
        Address::Edge | Address::Border(_) => v.clamp(0, n - 1),
        Address::Repeat => v.rem_euclid(n),
        Address::Mirror => {
            let r = (if v < 0 { -1 - v } else { v }).rem_euclid(2 * n);
            if r < n {
                r
            } else {
                2 * n - 1 - r
            }
        }
    };
    if let Address::Border(b) = address {
        if i < 0 || i >= tex.w || j < 0 || j >= tex.h {
            return through_channels(class, b);
        }
    }
    tex.t[(fix(j, tex.h) * tex.w + fix(i, tex.w)) as usize].map(|c| c * k)
}

/// Metal's bicubic sample of `tex` at texel-space `(x, y)` for the bound format's `class`, with
/// constant texel offset `offset`. The offset shifts the base taps after `split`, as the emulation
/// adds it: the same as shifting `(x, y)` wherever `q` fits an `i32`, but past that range it
/// moves the saturated footprint (the translation's choice).
fn reference(
    tex: &Texels,
    class: BicubicTexels,
    address: Address,
    (x, y): (f64, f64),
    offset: (i32, i32),
    k: f64,
) -> [f64; 4] {
    let (i, fx) = split(x);
    let (j, fy) = split(y);
    let (i, j) = (i + i64::from(offset.0), j + i64::from(offset.1));
    let linear = class.0 & BicubicTexels::LINEAR != 0;
    let (wx, wy) = if linear {
        ([0.0, 1.0 - fx, fx, 0.0], [0.0, 1.0 - fy, fy, 0.0])
    } else {
        (catmull_rom(fx), catmull_rom(fy))
    };
    let mut r = [0.0; 4];
    for (b, wyb) in wy.iter().enumerate() {
        for (a, wxa) in wx.iter().enumerate() {
            if *wxa == 0.0 || *wyb == 0.0 {
                continue;
            }
            let t = tap(tex, class, address, i - 1 + a as i64, j - 1 + b as i64, k);
            for c in 0..4 {
                r[c] += wxa * wyb * t[c];
            }
        }
    }
    r.map(|v| {
        if class.0 & BicubicTexels::UNORM != 0 {
            v.clamp(0.0, 1.0)
        } else if class.0 & BicubicTexels::SNORM != 0 {
            v.clamp(-1.0, 1.0)
        } else {
            v
        }
    })
}

/// `got` is `want` within the device's `f32` arithmetic of the sum (sixteen products and the
/// weights from the fraction): `2^-18` relative to the larger of 1 and `|want|`, and for a `half`
/// result also one `half` ulp of `want` (the device's narrowing).
fn close(got: f32, want: f64, half: bool) -> bool {
    let arith = want.abs().max(1.0) * 2f64.powi(-18);
    let tol = if half {
        arith.max(half_ulp(want))
    } else {
        arith
    };
    (f64::from(got) - want).abs() <= tol
}

/// Coordinates (texels) from outside one edge to outside the other on both axes, with fractions
/// on and off Metal's 1/256 grid (halfway between two steps included).
///
/// Some sit where `256 x - 127.5` does not round exactly in `f32`: just below and above a half
/// step near 0 (`(1/2 - 2^-25) / 256`, `-(1/2 + 2^-24) / 256`), and past `2^15` texels, where
/// every `x` is a multiple of 1/256 (40012.50390625 has an odd `256 x`, which a tie to even would
/// round up).
///
/// Others pin the range where `q` fits an `i32`, `[-2^23 + 1/2, 2^23]` texels, as a clamping
/// sampler reads them (a repeating one wraps each into one period first); each is listed with the
/// texel coordinate it becomes. Inside it: `2^22 + 1/2` and `5 10^6` (past a clamp at `±2^30`),
/// `2^23 - 1/2` (the clamp's top), `2^23` (above it, restored), and `-2^23 + 1/2` (`q = -2^31`).
/// Outside it: `2^23 + 1` and `-2^23`, which saturate.
fn coordinates(w: i64, h: i64) -> Vec<(f64, f64)> {
    let below_half = (0.5 - 2f64.powi(-25)) / 256.0;
    let above_half = -(0.5 + 2f64.powi(-24)) / 256.0;
    let xs = [
        -2.25,
        -0.75,
        above_half,
        0.0,
        below_half,
        0.5,
        1.0 + 1.0 / 512.0,
        2.0 + 3.0 / 512.0,
        3.3,
        w as f64 / 2.0 + 0.123,
        w as f64 - 0.75,
        w as f64 + 0.25,
        w as f64 + 2.25,
        32768.0 + 5.0 / 256.0,
        40_012.503_906_25,
        40_012.511_718_75,
        4_194_304.5,   // 2^22 + 1/2
        4_999_999.875, // 5 10^6
        8_388_607.5,   // 2^23 - 1/2
        8_388_608.25,  // 2^23
        8_388_609.0,   // 2^23 + 1
        -8_388_607.5,  // -2^23 + 1/2
        -8_388_608.25, // -2^23
    ];
    let ys = [
        -1.5,
        below_half,
        0.25,
        1.75,
        h as f64 / 2.0 + 0.6,
        h as f64 - 0.1,
        h as f64 + 1.5,
    ];
    ys.iter()
        .flat_map(|y| xs.iter().map(move |x| (*x, *y)))
        .collect()
}

/// The exact real `x` (an `f64` holding it exactly) rounded to `f32` with ties away from zero.
fn ties_away(x: f64) -> f32 {
    let r = x as f32;
    let d = x - f64::from(r);
    if d == 0.0 || !r.is_finite() || r == 0.0 {
        return r;
    }
    let other = if (d > 0.0) == (r > 0.0) {
        f32::from_bits(r.to_bits() + 1)
    } else {
        f32::from_bits(r.to_bits() - 1)
    };
    if (f64::from(other) - x).abs() == d.abs() && other.abs() > r.abs() {
        other
    } else {
        r
    }
}

/// The texel-space coordinate a kernel reads for `x` (texels) under `scale` (the extent, or 1
/// for pixel coordinates, `pixel`) on an axis addressed by `address`: Metal's position, the `f32`
/// coordinate it is given wrapped into one period first on a repeating axis (rounded up), then
/// times the extent rounded to `f32` with ties away from zero; a pixel coordinate as given.
fn seen(x: f64, scale: f64, address: Address, pixel: bool) -> f64 {
    let u = (x / scale) as f32;
    if pixel {
        return f64::from(u);
    }
    let u = match address {
        Address::Repeat => wrap_up(u, u.floor()),
        Address::Mirror => wrap_up(u, 2.0 * (u / 2.0).floor()),
        Address::Edge | Address::Border(_) => u,
    };
    f64::from(ties_away(f64::from(u) * scale))
}

/// `u - k` (`k` the integer start of `u`'s period) rounded toward positive infinity, as Metal's
/// wrap rounds. Exact in `f64` but where `|u| < 2^-29`, where it is just below `-k` and rounds up
/// to `-k` either way.
fn wrap_up(u: f32, k: f32) -> f32 {
    let exact = f64::from(u) - f64::from(k);
    let s = exact as f32;
    if f64::from(s) < exact {
        f32::from_bits(s.to_bits() + 1)
    } else {
        s
    }
}

/// The texel-space coordinate the emulation read previously for `x` (texels) under `scale`: the
/// `f32` coordinate times the extent, rounded to `f32` to nearest even, unwrapped.
fn product(x: f64, scale: f64) -> f64 {
    f64::from(((x / scale) as f32) * scale as f32)
}

fn coord_bytes(points: &[(f64, f64)], scale: (f64, f64)) -> Vec<u8> {
    points
        .iter()
        .flat_map(|(x, y)| {
            [
                ((x / scale.0) as f32).to_le_bytes(),
                ((y / scale.1) as f32).to_le_bytes(),
            ]
        })
        .flatten()
        .collect()
}

/// Each address mode, normalized and pixel coordinates, and each format family on a 12x10
/// texture: every lane is the reference filter's sample, on the interpreter.
#[test]
fn bicubic_samples_match_the_reference_filter() {
    const WHITE: [f64; 4] = [1.0, 1.0, 1.0, 1.0];
    let samplers = [
        ("edge", bicubic_words(1, false, 0), Address::Edge),
        (
            "zero",
            bicubic_words(0, false, 0),
            Address::Border([0.0; 4]),
        ),
        ("white", bicubic_words(0, false, 2), Address::Border(WHITE)),
        ("repeat", bicubic_words(2, false, 0), Address::Repeat),
        ("pixel edge", bicubic_words(1, true, 0), Address::Edge),
        (
            "pixel zero",
            bicubic_words(0, true, 0),
            Address::Border([0.0; 4]),
        ),
    ];
    let formats = [
        PixelFormat::RGBA16Float,
        PixelFormat::R8Unorm,
        PixelFormat::A8Unorm,
        PixelFormat::RGBA8Snorm,
        PixelFormat::RGBA32Float,
        PixelFormat::RG11B10Float,
    ];
    let (w, h) = (12i64, 10i64);
    let points = coordinates(w, h);
    let p = m3();
    for (name, words, address) in samplers {
        let Some(bc) = assemble(name, &kernel(words, Form::Plain { offset: (0, 0) })) else {
            return;
        };
        let a = compile(&bc, "k", &p).expect("compile");
        assert_eq!(
            a.strategies.entries.get("sampler.bicubic").map(|u| u.uses),
            Some(1),
            "{name}"
        );
        assert!(
            a.facts
                .contains(&crate::pipeline::RuntimeFact::BicubicTexels { slot: TEXTURE }),
            "{name}"
        );
        assert!(
            a.precision
                .contains(&PrecisionNote::BicubicFilterArithmetic),
            "{name}"
        );
        link_and_validate(&a, &p).expect("L3");
        let (plan, m) = typed_module(&bc, "k", &p).expect("typed module");
        let pixel = name.starts_with("pixel");
        let scale = if pixel {
            (1.0, 1.0)
        } else {
            (w as f64, h as f64)
        };
        for (seed, pf) in formats.iter().enumerate() {
            let class = texels(*pf).expect("measured");
            let tex = Texels::random(class, w, h, seed as u64 + 1);
            let image = tex.image(combined_slot(&m), 2, 1);
            let res = resources(
                &plan,
                &[coord_bytes(&points, scale), vec![0xab; 16 * points.len()]],
                class,
            );
            let out = run(&m, &res, &image, points.len() as u32);
            // The lanes whose answer the position previously (the unwrapped product) changes.
            let mut told = 0;
            for (lane, (x, y)) in points.iter().enumerate() {
                // The coordinate the kernel read, in texels.
                let fx = seen(*x, scale.0, address, pixel);
                let fy = seen(*y, scale.1, address, pixel);
                let want = reference(&tex, class, address, (fx, fy), (0, 0), 1.0);
                if !pixel {
                    let before = (product(*x, scale.0), product(*y, scale.1));
                    let then = reference(&tex, class, address, before, (0, 0), 1.0);
                    told += usize::from(then != want);
                }
                for c in 0..4 {
                    let want_c = if class.0 & BicubicTexels::HALF != 0 {
                        f64::from(half(want[c] as f32))
                    } else {
                        want[c]
                    };
                    assert!(
                        close(out[lane][c], want_c, class.0 & BicubicTexels::HALF != 0),
                        "{name} {pf:?} lane {lane} ({x}, {y}) channel {c}: {} vs {want_c}",
                        out[lane][c]
                    );
                }
            }
            // A repeating sampler's lanes tell Metal's position from the unwrapped product, so
            // the emulation previously fails them.
            assert!(
                pixel || !matches!(address, Address::Repeat) || told > 0,
                "{name} {pf:?}: no lane tells the wrapped position from the unwrapped product"
            );
        }
    }
}

/// A 1D texture is filtered bilinearly whatever its format; a 2D array's layer is clamped into
/// the array; a constant offset shifts the footprint's base taps, after a repeating axis wrapped
/// its coordinate (its position lies in one period, so its offset lanes wrap the shifted taps; a
/// clamping axis's lanes near `2^23`, where `q` saturates before the offset is added, read the edge
/// either way).
#[test]
fn a_1d_texture_an_array_layer_and_an_offset() {
    let p = m3();
    let (w, h) = (12i64, 10i64);
    let points = coordinates(w, h);
    let scale = (w as f64, h as f64);
    let class = texels(PixelFormat::RGBA16Float).expect("measured");
    let tex = Texels::random(class, w, h, 7);
    let (edge, repeat) = (bicubic_words(1, false, 0), bicubic_words(2, false, 0));
    let offset = Form::Plain { offset: (2, -1) };
    let cases: [(&str, u64, Address, Form, u32, f64); 5] = [
        ("1d", edge, Address::Edge, Form::OneD, 1, 1.0),
        (
            "layer 1",
            edge,
            Address::Edge,
            Form::Array { layer: 1 },
            3,
            2.0,
        ),
        (
            "layer 9",
            edge,
            Address::Edge,
            Form::Array { layer: 9 },
            3,
            4.0,
        ),
        ("offset", edge, Address::Edge, offset, 1, 1.0),
        ("repeat offset", repeat, Address::Repeat, offset, 1, 1.0),
    ];
    for (name, words, address, form, layers, k) in cases {
        let Some(bc) = assemble(name, &kernel(words, form)) else {
            return;
        };
        let a = compile(&bc, "k", &p).expect("compile");
        link_and_validate(&a, &p).expect("L3");
        let (plan, m) = typed_module(&bc, "k", &p).expect("typed module");
        let one_d = matches!(form, Form::OneD);
        let tex1d = Texels {
            w,
            h: 1,
            t: tex.t[..w as usize].to_vec(),
        };
        let used = if one_d { &tex1d } else { &tex };
        let image = used.image(combined_slot(&m), if one_d { 1 } else { 2 }, layers);
        let res = resources(
            &plan,
            &[coord_bytes(&points, scale), vec![0xab; 16 * points.len()]],
            class,
        );
        let out = run(&m, &res, &image, points.len() as u32);
        let (ox, oy) = match form {
            Form::Plain { offset } => offset,
            _ => (0, 0),
        };
        for (lane, (x, y)) in points.iter().enumerate() {
            let fx = seen(*x, scale.0, address, false);
            let fy = seen(*y, scale.1, address, false);
            let want: [f64; 4] = if one_d {
                // Bilinear over taps i and i + 1, no clamp, the half format's rounding.
                let (i, f) = split(fx);
                let i = i + i64::from(ox);
                let t0 = tap(used, class, address, i, 0, 1.0);
                let t1 = tap(used, class, address, i + 1, 0, 1.0);
                std::array::from_fn(|c| f64::from(half(((1.0 - f) * t0[c] + f * t1[c]) as f32)))
            } else {
                reference(used, class, address, (fx, fy), (ox, oy), k)
                    .map(|v| f64::from(half(v as f32)))
            };
            for c in 0..4 {
                assert!(
                    close(out[lane][c], want[c], class.0 & BicubicTexels::HALF != 0),
                    "{name} lane {lane} ({x}, {y}) channel {c}: {} vs {}",
                    out[lane][c],
                    want[c]
                );
            }
        }
    }
}

/// A constant offset on a repeating axis of 5, 6 or 7 texels is dropped, as Metal's other filters
/// drop it (the round-3 offset probe: the M3 Max's and an M4's bicubic samples with an offset equal
/// the ones without it on those axes, byte for byte), and added before the wrap on a 10- or
/// 12-texel axis: every lane is the reference with the offset so placed, and on each texture some
/// lane tells it from the offset added on every axis.
#[test]
fn an_offset_on_a_repeating_axis_takes_metals_placement() {
    let p = m3();
    let (ox, oy) = (2, -1);
    let form = Form::Plain { offset: (ox, oy) };
    let Some(bc) = assemble("repeat offset", &kernel(bicubic_words(2, false, 0), form)) else {
        return;
    };
    let a = compile(&bc, "k", &p).expect("compile");
    link_and_validate(&a, &p).expect("L3");
    let (plan, m) = typed_module(&bc, "k", &p).expect("typed module");
    let class = texels(PixelFormat::RGBA16Float).expect("measured");
    let drops = |n: i64| matches!(n, 3 | 5 | 6 | 7);
    for (w, h) in [(5i64, 10i64), (12, 6), (7, 5)] {
        let points = coordinates(w, h);
        let scale = (w as f64, h as f64);
        let tex = Texels::random(class, w, h, 11);
        let image = tex.image(combined_slot(&m), 2, 1);
        let res = resources(
            &plan,
            &[coord_bytes(&points, scale), vec![0xab; 16 * points.len()]],
            class,
        );
        let out = run(&m, &res, &image, points.len() as u32);
        let placed = (if drops(w) { 0 } else { ox }, if drops(h) { 0 } else { oy });
        let mut told = 0;
        for (lane, (x, y)) in points.iter().enumerate() {
            let fx = seen(*x, scale.0, Address::Repeat, false);
            let fy = seen(*y, scale.1, Address::Repeat, false);
            let at = |o: (i32, i32)| {
                reference(&tex, class, Address::Repeat, (fx, fy), o, 1.0)
                    .map(|v| f64::from(half(v as f32)))
            };
            let want = at(placed);
            told += usize::from(want != at((ox, oy)));
            for c in 0..4 {
                assert!(
                    close(out[lane][c], want[c], class.0 & BicubicTexels::HALF != 0),
                    "{w}x{h} lane {lane} ({x}, {y}) channel {c}: {} vs {}",
                    out[lane][c],
                    want[c]
                );
            }
        }
        assert!(
            told > 0,
            "{w}x{h}: no lane tells Metal's placement from the added offset"
        );
    }
}

/// Metal's bicubic filter on mirrored axes (`lower::sampler::bicubic`'s module docs; the probes on
/// the M3 Max, the primary):
/// - On a 12x10 level, a mirrored axis of a format Metal filters bicubically reads texel 1 at
///   fraction 0 whatever the coordinate (with both axes mirrored, texel `(1, 1)`), while the other
///   axis filters as it is addressed; a 32-bit float format's mirrored sample is the linear
///   filter's, its taps reflected about the mirrored position.
#[test]
fn mirrored_axes() {
    let (w, h) = (12i64, 10i64);
    let points = coordinates(w, h);
    let scale = (w as f64, h as f64);
    let edge = bicubic_words(1, false, 0);
    let mirrors = [
        ("mirror", bicubic_words(3, false, 0), [true, true]),
        ("mirror s", with_axis(edge, 0, 3), [true, false]),
        ("mirror t", with_axis(edge, 1, 3), [false, true]),
    ];
    let p = m3();
    for (name, words, pinned) in mirrors {
        let Some(bc) = assemble(name, &kernel(words, Form::Plain { offset: (0, 0) })) else {
            return;
        };
        let a = compile(&bc, "k", &p).expect("compile");
        link_and_validate(&a, &p).expect("L3");
        let (plan, m) = typed_module(&bc, "k", &p).expect("typed module");
        let formats = [
            PixelFormat::R8Unorm,
            PixelFormat::RGBA8Unorm,
            PixelFormat::RGBA16Float,
            PixelFormat::RGBA32Float,
        ];
        for (seed, pf) in formats.iter().enumerate() {
            let class = texels(*pf).expect("measured");
            let linear = class.0 & BicubicTexels::LINEAR != 0;
            // A 32-bit float format's both-axes case: the linear filter's mirror; its one-axis
            // cases mix two addressings the reference does not model.
            if linear && pinned != [true, true] {
                continue;
            }
            let tex = Texels::random(class, w, h, seed as u64 + 11);
            let image = tex.image(combined_slot(&m), 2, 1);
            let res = resources(
                &plan,
                &[coord_bytes(&points, scale), vec![0xab; 16 * points.len()]],
                class,
            );
            let out = run(&m, &res, &image, points.len() as u32);
            for (lane, (x, y)) in points.iter().enumerate() {
                let want = if linear {
                    let fx = seen(*x, scale.0, Address::Mirror, false);
                    let fy = seen(*y, scale.1, Address::Mirror, false);
                    reference(&tex, class, Address::Mirror, (fx, fy), (0, 0), 1.0)
                } else {
                    // The pinned axis at texel 1's centre, the other clamped to the edge.
                    let at = |pin: bool, c: f64, n: f64| {
                        if pin {
                            1.5
                        } else {
                            seen(c, n, Address::Edge, false)
                        }
                    };
                    let (fx, fy) = (at(pinned[0], *x, scale.0), at(pinned[1], *y, scale.1));
                    reference(&tex, class, Address::Edge, (fx, fy), (0, 0), 1.0)
                };
                for c in 0..4 {
                    let want_c = if class.0 & BicubicTexels::HALF != 0 {
                        f64::from(half(want[c] as f32))
                    } else {
                        want[c]
                    };
                    assert!(
                        close(out[lane][c], want_c, class.0 & BicubicTexels::HALF != 0),
                        "{name} {pf:?} lane {lane} ({x}, {y}) channel {c}: {} vs {want_c}",
                        out[lane][c]
                    );
                }
                if pinned == [true, true] && !linear {
                    assert_eq!(
                        out[lane], out[0],
                        "{name} {pf:?} lane {lane}: not texel (1, 1)"
                    );
                }
            }
        }
    }
}

fn refused(e: &Error) -> bool {
    matches!(
        e,
        Error::Unsupported(Unsupported {
            feature: Feature::SamplerBicubicFilter,
            needs: Needs::NotImplemented,
            ..
        })
    )
}

/// What no Metal reading establishes is refused `NotImplemented(SamplerBicubicFilter)`, never
/// lowered through the binding's nearest-filtered immutable sampler:
/// - `min_lod_clamp` (Apple's device compiler fails to build the pipeline) and a dynamic offset;
/// - `mirrored_repeat` on a 1D texture, with pixel coordinates, or beside a constant offset on
///   the mirrored axis (a 2D sample's mirrored axis is in `mirrored_axes`);
/// - a depth comparison (refused by `check`'s depth term: AIR compares only depth textures) and
///   an LOD query;
/// - a 3D, cube, depth, or integer texture;
/// - anisotropy (a min/max reduction reads the weighted average);
/// - a bicubic filter beside a linear one, or with a mip filter (Metal's front end admits
///   neither, so no reading exists).
///
/// The heap texture is in `a_heap_texture_through_a_bicubic_sampler_is_refused`. One axis
/// clamping to zero and another to a colored border is in `lower::sampler::bicubic`'s tests,
/// because AIR cannot spell it: every axis coded 0 takes the sampler's one border color. AIR
/// samples no buffer texture.
#[test]
fn bicubic_forms_without_a_metal_reference_are_refused() {
    let edge = bicubic_words(1, false, 0);
    let mut cases: Vec<(&str, String)> = vec![
        ("min_lod_clamp", kernel(edge, Form::MinLodClamp)),
        (
            "mirrored 1d",
            kernel(bicubic_words(3, false, 0), Form::OneD),
        ),
        (
            "mirrored pixel",
            kernel(bicubic_words(3, true, 0), Form::Plain { offset: (0, 0) }),
        ),
        (
            "mirrored offset",
            kernel(bicubic_words(3, false, 0), Form::Plain { offset: (0, 1) }),
        ),
        ("dynamic offset", kernel(edge, Form::DynamicOffset)),
        (
            "anisotropy 2",
            kernel(edge | 1 << 20, Form::Plain { offset: (0, 0) }),
        ),
        (
            "mag linear",
            kernel(edge & !(3 << 9) | 1 << 9, Form::Plain { offset: (0, 0) }),
        ),
        (
            "min linear",
            kernel(edge & !(3 << 11) | 1 << 11, Form::Plain { offset: (0, 0) }),
        ),
        (
            "mip nearest",
            kernel(edge | 1 << 13, Form::Plain { offset: (0, 0) }),
        ),
    ];
    // A comparison sampler (`compare_func::less`) on a depth texture.
    let compare = edge & !(0xf << 16) | 1 << 16;
    let probes: [(&str, u64, &str, &str, &str, &str); 5] = [
        (
            "compare",
            compare,
            "depth2d<float, sample>",
            "%r = call { float, i8 } @air.sample_compare_depth_2d.f32(ptr addrspace(1) %t, ptr addrspace(2) @__air_sampler_state, i32 1, <2 x float> %uv, float 0.5, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)\n  %v = extractvalue { float, i8 } %r, 0",
            "{ float, i8 } @air.sample_compare_depth_2d.f32(ptr addrspace(1), ptr addrspace(2), i32, <2 x float>, float, i1, <2 x i32>, i1, float, float, i32)",
            "float",
        ),
        (
            "3d",
            edge,
            "texture3d<float, sample>",
            "%r = call { <4 x float>, i8 } @air.sample_texture_3d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) @__air_sampler_state, <3 x float> %uvw, i1 true, <3 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)\n  %v = extractvalue { <4 x float>, i8 } %r, 0",
            "{ <4 x float>, i8 } @air.sample_texture_3d.v4f32(ptr addrspace(1), ptr addrspace(2), <3 x float>, i1, <3 x i32>, i1, float, float, i32)",
            "<4 x float>",
        ),
        (
            "cube",
            edge,
            "texturecube<float, sample>",
            "%r = call { <4 x float>, i8 } @air.sample_texture_cube.v4f32(ptr addrspace(1) %t, ptr addrspace(2) @__air_sampler_state, <3 x float> %uvw, i1 false, float 0.0, float 0.0, i32 0)\n  %v = extractvalue { <4 x float>, i8 } %r, 0",
            "{ <4 x float>, i8 } @air.sample_texture_cube.v4f32(ptr addrspace(1), ptr addrspace(2), <3 x float>, i1, float, float, i32)",
            "<4 x float>",
        ),
        (
            "depth",
            edge,
            "depth2d<float, sample>",
            "%r = call { float, i8 } @air.sample_depth_2d.f32(ptr addrspace(1) %t, ptr addrspace(2) @__air_sampler_state, i32 1, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)\n  %v = extractvalue { float, i8 } %r, 0",
            "{ float, i8 } @air.sample_depth_2d.f32(ptr addrspace(1), ptr addrspace(2), i32, <2 x float>, i1, <2 x i32>, i1, float, float, i32)",
            "float",
        ),
        (
            "integer",
            edge,
            "texture2d<int, sample>",
            "%r = call { <4 x i32>, i8 } @air.sample_texture_2d.s.v4i32(ptr addrspace(1) %t, ptr addrspace(2) @__air_sampler_state, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)\n  %v = extractvalue { <4 x i32>, i8 } %r, 0",
            "{ <4 x i32>, i8 } @air.sample_texture_2d.s.v4i32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)",
            "<4 x i32>",
        ),
    ];
    for (name, words, tex_ty, body, decl, value_ty) in probes {
        cases.push((name, probe_kernel(words, tex_ty, body, decl, value_ty)));
    }
    cases.push(("lod query", lod_fragment(edge)));
    // Every committed profile: a capability one device lacks must not be what refuses the form.
    let profiles = crate::spirv::testing::committed_profiles().expect("profiles");
    for (name, text) in &cases {
        let Some(bc) = assemble(name, text) else {
            return;
        };
        for (stem, p) in &profiles {
            match compile(&bc, "k", p) {
                Err(e) if refused(&e) => {}
                other => panic!(
                    "{name} on {stem}: {:?}",
                    other.map(|a| a.strategies.clone())
                ),
            }
        }
    }
}

/// A heap texture (an argument buffer's member) through a bicubic sampler is refused, by
/// `check`'s heap term before the operation is chosen. `handles`' `argument_buffer_sample`
/// samples a heap texture through a linear and a pixel nearest sampler, and compiles where the
/// device has the heap. Two variants are refused there:
/// - both samplers made bicubic (a sample, which the sample arm's slot guard also refuses: no
///   texture slot specializes the bound format);
/// - the linear sample turned into a gather and its sampler made bicubic. The heap term alone
///   refuses this one: a gather through a bicubic sampler is otherwise an ordinary gather, and no
///   reading covers it on a heap texture.
///
/// Each variant is compiled beside its control, the same module with the samplers' own words.
#[test]
fn a_heap_texture_through_a_bicubic_sampler_is_refused() {
    let text = std::fs::read_to_string(engine_dir().join("handles.ll")).expect("handles.ll");
    let linear = "@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601020489, i64 0]";
    let pixel = "@__air_sampler_state.1 = internal addrspace(2) constant [2 x i64] [i64 34901797601050697, i64 0]";
    let sample = "@air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %15, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <2 x float> %13, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)";
    assert!(
        text.contains(linear) && text.contains(pixel) && text.matches(sample).count() == 1,
        "handles.ll's samplers or argument_buffer_sample's linear sample moved"
    );
    let gather = format!(
        "{}\ndeclare {{ <4 x float>, i8 }} @air.gather_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i32, i32)\n",
        text.replace(
            sample,
            "@air.gather_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %15, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <2 x float> %13, i1 true, <2 x i32> zeroinitializer, i32 0, i32 0)",
        )
    );
    let state = |name: &str, words: u64| {
        format!("{name} = internal addrspace(2) constant [2 x i64] [i64 {words}, i64 0]")
    };
    let bicubic_sample = text
        .replace(
            linear,
            &state("@__air_sampler_state", bicubic_words(1, false, 0)),
        )
        .replace(
            pixel,
            &state("@__air_sampler_state.1", bicubic_words(1, true, 0)),
        );
    let bicubic_gather = gather.replace(
        linear,
        &state("@__air_sampler_state", bicubic_words(1, false, 0)),
    );
    let profiles = crate::spirv::testing::committed_profiles().expect("profiles");
    for (name, plain, bicubic) in [
        ("sample", &text, &bicubic_sample),
        ("gather", &gather, &bicubic_gather),
    ] {
        let (Some(plain), Some(bc)) = (
            assemble_module(&format!("heap {name} plain"), plain),
            assemble_module(&format!("heap {name} bicubic"), bicubic),
        ) else {
            return;
        };
        let mut refused_on = 0;
        for (stem, p) in &profiles {
            if compile(&plain, "argument_buffer_sample", p).is_err() {
                continue;
            }
            match compile(&bc, "argument_buffer_sample", p) {
                Err(e) if refused(&e) => refused_on += 1,
                other => panic!(
                    "{name} on {stem}: {:?}",
                    other.map(|a| a.strategies.clone())
                ),
            }
        }
        assert!(refused_on > 0, "{name}: no committed profile has the heap");
    }
}

/// The emulation is fetches, integer and float arithmetic, and specialization constants, which
/// every Vulkan 1.2 device runs: a 2D and a 1D bicubic sample compile and pass L3 on every
/// committed profile.
#[test]
fn the_emulation_validates_on_every_committed_profile() {
    let edge = bicubic_words(0, false, 2);
    let forms = [("2d", Form::Plain { offset: (1, -1) }), ("1d", Form::OneD)];
    for (name, form) in forms {
        let Some(bc) = assemble(name, &kernel(edge, form)) else {
            return;
        };
        for (stem, p) in crate::spirv::testing::committed_profiles().expect("profiles") {
            let a = compile(&bc, "k", &p).unwrap_or_else(|e| panic!("{name} on {stem}: {e}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{name} on {stem}: {e}"));
        }
    }
}

/// The ids the footprint's three `f32` operations produce in the entry point of a linked module
/// `words`, and whether each is rounded on its own. The operations are the normalized coordinate
/// times the extent (a product of an `OpConvertUToF` result), `y = 256 x` (a product by the
/// constant 256), and `r = y - floor(y)` (an `OpFSub` of a GLSL `Floor` result). A product
/// rounded on its own is `GLSL.std.450 Fma(a, b, -0.0)` where the profile's `float` functions are
/// Metal's (`metal`), and an `OpFMul` elsewhere; it and the subtraction are rounded on their own
/// when decorated `NoContraction`.
fn footprint_ops(words: &[u32], metal: bool) -> [Vec<(u32, bool)>; 3] {
    let insts: Vec<&[u32]> = {
        let (mut v, mut i) = (Vec::new(), 5);
        while i < words.len() {
            let len = (words[i] >> 16) as usize;
            assert!(len > 0 && i + len <= words.len(), "malformed module");
            v.push(&words[i..i + len]);
            i += len;
        }
        v
    };
    let op = |w: &[u32]| w[0] & 0xFFFF;
    // The entry point's own instructions: the footprint is computed there, before the calls of
    // the filter's routines, whose arithmetic is their own.
    let entry = insts
        .iter()
        .find(|w| op(w) == crate::spv::Op::EntryPoint as u32)
        .map(|w| w[2])
        .expect("an entry point");
    let body: Vec<&[u32]> = insts
        .iter()
        .skip_while(|w| !(op(w) == crate::spv::Op::Function as u32 && w[2] == entry))
        .take_while(|w| op(w) != crate::spv::Op::FunctionEnd as u32)
        .copied()
        .collect();
    let ids_of = |pred: &dyn Fn(&[u32]) -> bool| -> Vec<u32> {
        body.iter().filter(|w| pred(w)).map(|w| w[2]).collect()
    };
    let floats = insts
        .iter()
        .filter(|w| op(w) == crate::spv::Op::TypeFloat as u32 && w[2] == 32)
        .map(|w| w[1])
        .collect::<Vec<_>>();
    let k256 = insts
        .iter()
        .filter(|w| {
            op(w) == crate::spv::Op::Constant as u32
                && floats.contains(&w[1])
                && w[3] == 256f32.to_bits()
        })
        .map(|w| w[2])
        .collect::<Vec<_>>();
    let to_float = ids_of(&|w| op(w) == crate::spv::Op::ConvertUToF as u32);
    let glsl = insts
        .iter()
        .filter(|w| op(w) == crate::spv::Op::ExtInstImport as u32)
        .filter(|w| {
            let bytes: Vec<u8> = w[2..].iter().flat_map(|x| x.to_le_bytes()).collect();
            bytes.starts_with(b"GLSL.std.450\0")
        })
        .map(|w| w[1])
        .collect::<Vec<_>>();
    let floors = ids_of(&|w| {
        op(w) == crate::spv::Op::ExtInst as u32
            && glsl.contains(&w[3])
            && w[4] == crate::spv::GlslStd450Op::Floor as u32
    });
    let negative_zero = insts
        .iter()
        .filter(|w| {
            op(w) == crate::spv::Op::Constant as u32
                && floats.contains(&w[1])
                && w[3] == 0x8000_0000
        })
        .map(|w| w[2])
        .collect::<Vec<_>>();
    // The product of `a` and `b` rounded on its own, in the profile's form: `Fma(a, b, -0.0)`
    // where its `float` functions are Metal's, an `OpFMul` elsewhere.
    let product = |w: &[u32], of: &[u32]| {
        if metal {
            op(w) == crate::spv::Op::ExtInst as u32
                && glsl.contains(&w[3])
                && w[4] == crate::spv::GlslStd450Op::Fma as u32
                && negative_zero.contains(&w[7])
                && (of.contains(&w[5]) || of.contains(&w[6]))
        } else {
            op(w) == crate::spv::Op::FMul as u32 && (of.contains(&w[3]) || of.contains(&w[4]))
        }
    };
    let decorated: Vec<u32> = insts
        .iter()
        .filter(|w| {
            op(w) == crate::spv::Op::Decorate as u32
                && w[2] == crate::spv::Decoration::NoContraction as u32
        })
        .map(|w| w[1])
        .collect();
    let with = |ids: Vec<u32>| {
        ids.into_iter()
            .map(|id| (id, decorated.contains(&id)))
            .collect()
    };
    [
        with(ids_of(&|w| product(w, &to_float))),
        with(ids_of(&|w| product(w, &k256))),
        with(ids_of(&|w| {
            op(w) == crate::spv::Op::FSub as u32 && floors.contains(&w[4])
        })),
    ]
}

/// The footprint's `x = c n` (normalized coordinates only), `y = 256 x`, and `r = y - floor(y)` are
/// each rounded on its own on every committed profile, one of each per plane axis, so no device
/// fuses them into the arithmetic after them: the products in the profile's form (`Fma(a, b, -0.0)`
/// where its `float` functions are Metal's, an `OpFMul` elsewhere), each decorated `NoContraction`
/// as the subtraction is. A re-bless of the golden table would not notice the form dropped; this
/// test does.
#[test]
fn the_footprint_arithmetic_is_no_contraction() {
    let samplers = [
        ("normalized", bicubic_words(1, false, 0), 2),
        ("pixel", bicubic_words(1, true, 0), 0),
    ];
    for (name, words, products) in samplers {
        let Some(bc) = assemble(name, &kernel(words, Form::Plain { offset: (0, 0) })) else {
            return;
        };
        for (stem, p) in crate::spirv::testing::committed_profiles().expect("profiles") {
            let a = compile(&bc, "k", &p).unwrap_or_else(|e| panic!("{name} on {stem}: {e}"));
            let (_, spv) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{name} on {stem}: {e}"));
            let metal = p.fields().float_functions == FloatFunctions::Metal;
            let [cn, y, r] = footprint_ops(&spv, metal);
            for (what, ops, want) in [
                ("c n", cn, products),
                ("256 x", y, 2),
                ("y - floor(y)", r, 2),
            ] {
                assert_eq!(ops.len(), want, "{name} on {stem}: {what}: {ops:?}");
                assert!(
                    ops.iter().all(|(_, d)| *d),
                    "{name} on {stem}: {what} not rounded on its own: {ops:?}"
                );
            }
        }
    }
}

/// The plan's word estimate bounds the emitted module of every public bicubic fixture (its samples'
/// call sites, and the routines once) on every committed profile.
#[test]
fn the_estimate_bounds_the_bicubic_fixtures() {
    use crate::budget::SPIRV_WORDS;
    use crate::library::FunctionConstantValues;
    use crate::lower::select::LowerInputs;
    use crate::pipeline::compile::{compile_bitcode, plan_bitcode, EntryRef};
    use crate::pipeline::testing::meter;
    let fixtures = [
        "bcu_address",
        "bcu_edges",
        "bcu_formats",
        "bcu_forms",
        "bcu_fragment",
        "bcx_rules",
        "bcx_table",
    ];
    for name in fixtures {
        let Ok(bc) = llvm_assemble(&engine_dir().join(format!("{name}.ll"))) else {
            eprintln!("SKIPPED: llvm-as");
            return;
        };
        for (stem, p) in crate::spirv::testing::committed_profiles().expect("profiles") {
            let what = format!("{name} on {stem}");
            let (entry, consts) = (EntryRef::Name(name), FunctionConstantValues::defaults());
            let inputs = LowerInputs::new();
            let plan = plan_bitcode(&bc, entry, &consts, &p, &inputs, &mut meter())
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            let a = compile_bitcode(&bc, entry, &consts, &p, &inputs, &mut meter(), SPIRV_WORDS)
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            let words = a.module.words.len() as u64;
            eprintln!("{what}: estimate {} emitted {words}", plan.estimate.words);
            assert!(
                plan.estimate.words >= words,
                "{what}: estimate {} < emitted {words}",
                plan.estimate.words
            );
        }
    }
}

/// A 2D bicubic sample is one call of the plane's routine (its sixteen taps, four fractions, the
/// bilinear test and the format's bits: 22 parameters), whose sum is the device's arithmetic: no
/// other routine of the old fixed-point datapath (the small level's separate quad call, the weights'
/// table, the half accumulator) is called. `bcx_rules`'s twelve 2D samples make twelve calls, and
/// its 1D ones call the line's sum.
#[test]
fn a_bicubic_sample_is_one_call_of_the_planes_sum() {
    let bc = match llvm_assemble(&engine_dir().join("bcx_rules.ll")) {
        Ok(b) => b,
        Err(AssembleError::Missing(t)) => {
            eprintln!("SKIPPED: {t}");
            return;
        }
        Err(AssembleError::Rejected(e)) => panic!("llvm-as rejected bcx_rules: {e}"),
    };
    let p = profile("intel-irisxe-rplp-anv", DevicePolicy::default()).expect("profile");
    let (_, m) = typed_module(&bc, "bcx_rules", &p).expect("typed module");
    let functions = m.functions();
    let entry = functions.get(m.entry().func.index()).expect("entry");
    let calls: Vec<usize> = entry
        .blocks
        .iter()
        .filter_map(|b| m.blocks().get(b.index()))
        .flat_map(|b| b.insts.iter())
        .filter_map(|node| match &node.inst {
            Inst::Call { func, .. } => functions.get(func.index()).map(|f| f.params.len()),
            _ => None,
        })
        .collect();
    assert_eq!(
        calls.iter().filter(|n| **n == 22).count(),
        12,
        "each of bcx_rules's twelve 2D samples calls the plane's sum once: {calls:?}"
    );
    // Its 1D samples call the line's sum (two taps, the fraction and the format's bits).
    assert!(
        calls.iter().all(|n| *n == 22 || *n == 4),
        "no other routine is called: {calls:?}"
    );
}

/// A kernel sampling texture slot 0 through the runtime sampler at `[[sampler(0)]]` at `float2`
/// buffer 0's lane `tid`, storing the `float4` to buffer 1: [`kernel`]'s `Form::Plain` sample
/// without an offset, its sampler an argument.
fn runtime_kernel() -> String {
    r#"define void @k(ptr addrspace(1) %t, ptr addrspace(2) %s, ptr addrspace(1) %c, ptr addrspace(1) %out, i32 %tid) {
  %i = zext i32 %tid to i64
  %p = getelementptr inbounds <2 x float>, ptr addrspace(1) %c, i64 %i
  %uv = load <2 x float>, ptr addrspace(1) %p, align 8
  %r = call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)
  %v = extractvalue { <4 x float>, i8 } %r, 0
  %q = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i64 %i
  store <4 x float> %v, ptr addrspace(1) %q, align 16
  ret void
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !7, !4, !5, !6}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t"}
!7 = !{i32 1, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}
!4 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"float2", !"air.arg_name", !"c"}
!5 = !{i32 3, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
!6 = !{i32 4, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
"#
    .to_string()
}

/// The runtime sampler slot of [`runtime_kernel`].
const SAMPLER: SlotRef = SlotRef {
    stage: Stage::Kernel,
    class: SlotClass::Sampler,
    index: 0,
};

/// The runtime word (`SamplerWord`'s fields) of a cubic sampler addressing every axis by
/// `MTLSamplerAddressMode` value `mode`, with border color value `border`, at pixel coordinates
/// where `pixel`.
fn cubic_word(mode: u32, border: u32, pixel: bool) -> u32 {
    use crate::metal::sampler::SamplerWord as W;
    let mut w = border << W::BORDER_SHIFT;
    for shift in W::ADDRESS_SHIFT {
        w |= mode << shift;
    }
    if pixel {
        w |= W::PIXEL_COORDINATES;
    }
    w
}

/// The texture slot's image binding of `m`: its combined binding, or its sampled image.
fn image_slot(m: &TypedModule) -> ResourceSlot {
    m.globals()
        .iter()
        .find_map(|g| match g.kind {
            GlobalKind::Resource(
                s @ ResourceSlot {
                    kind: ResourceKind::CombinedTextureSampler(_) | ResourceKind::Texture,
                    index: 0,
                    ..
                },
            ) => Some(s),
            _ => None,
        })
        .expect("an image binding")
}

/// A sample through a runtime sampler whose min or mag filter is Metal's `Cubic` is the sample
/// through a constexpr bicubic sampler of the same state (bit for bit on Metal),
/// so under the cubic-sampler variant the function emulates it as that one's, and here every lane
/// of both kernels is the same bits, under each address mode and border color, at normalized and
/// pixel coordinates, for formats of each family: a runtime `clamp_to_zero` reads the sampler's
/// border color, so with an opaque-white border it is the constexpr white border's sample. Without
/// the variant the function records the slot's `RuntimeFact::CubicSampler` and filters as the word
/// says.
#[test]
fn a_runtime_cubic_sampler_samples_as_a_constexpr_bicubic_one() {
    use crate::metal::{SamplerAddressMode as A, SamplerBorderColor as B};
    let (edge, zero, border, repeat) = (
        A::ClampToEdge.as_raw(),
        A::ClampToZero.as_raw(),
        A::ClampToBorderColor.as_raw(),
        A::Repeat.as_raw(),
    );
    let (black, white) = (B::TransparentBlack.as_raw(), B::OpaqueWhite.as_raw());
    let cases = [
        (
            "edge",
            bicubic_words(1, false, 0),
            cubic_word(edge, black, false),
        ),
        (
            "zero",
            bicubic_words(0, false, 0),
            cubic_word(zero, black, false),
        ),
        (
            "white",
            bicubic_words(0, false, 2),
            cubic_word(border, white, false),
        ),
        (
            "zero white",
            bicubic_words(0, false, 2),
            cubic_word(zero, white, false),
        ),
        (
            "repeat",
            bicubic_words(2, false, 0),
            cubic_word(repeat, black, false),
        ),
        (
            "pixel edge",
            bicubic_words(1, true, 0),
            cubic_word(edge, black, true),
        ),
        (
            "pixel zero",
            bicubic_words(0, true, 0),
            cubic_word(zero, black, true),
        ),
    ];
    let formats = [
        PixelFormat::RGBA16Float,
        PixelFormat::R8Unorm,
        PixelFormat::RGBA8Snorm,
        PixelFormat::RGBA32Float,
        PixelFormat::RG11B10Float,
    ];
    let (w, h) = (12i64, 10i64);
    let points = coordinates(w, h);
    let p = m3();
    let Some(runtime) = assemble("runtime cubic", &runtime_kernel()) else {
        return;
    };
    let plain = compile(&runtime, "k", &p).expect("compile");
    assert!(
        plain
            .facts
            .contains(&crate::pipeline::RuntimeFact::CubicSampler { slot: SAMPLER }),
        "every sample through a runtime sampler slot states the slot's cubic fact"
    );
    assert_eq!(plain.strategies.entries.get("sampler.bicubic"), None);
    for (name, words, word) in cases {
        let Some(constexpr) = assemble(name, &kernel(words, Form::Plain { offset: (0, 0) })) else {
            return;
        };
        let mut inputs = crate::lower::select::LowerInputs::new();
        inputs.cubic_samplers.insert(SAMPLER, word);
        let a = compile_with(&runtime, "k", &p, &inputs).expect("compile");
        assert_eq!(
            a.strategies.entries.get("sampler.bicubic").map(|u| u.uses),
            Some(1),
            "{name}"
        );
        assert!(
            a.facts
                .contains(&crate::pipeline::RuntimeFact::BicubicTexels { slot: TEXTURE }),
            "{name}"
        );
        link_and_validate(&a, &p).expect("L3");
        let (cplan, cm) = typed_module(&constexpr, "k", &p).expect("typed module");
        let (rplan, rm) = typed_module_with(&runtime, "k", &p, &inputs).expect("typed module");
        let scale = if name.starts_with("pixel") {
            (1.0, 1.0)
        } else {
            (w as f64, h as f64)
        };
        for (seed, pf) in formats.iter().enumerate() {
            let class = texels(*pf).expect("measured");
            let tex = Texels::random(class, w, h, seed as u64 + 7);
            let init = [coord_bytes(&points, scale), vec![0xab; 16 * points.len()]];
            let want = run(
                &cm,
                &resources(&cplan, &init, class),
                &tex.image(combined_slot(&cm), 2, 1),
                points.len() as u32,
            );
            let got = run(
                &rm,
                &resources(&rplan, &init, class),
                &tex.image(image_slot(&rm), 2, 1),
                points.len() as u32,
            );
            for (lane, (g, c)) in got.iter().zip(&want).enumerate() {
                assert_eq!(
                    g.map(f32::to_bits),
                    c.map(f32::to_bits),
                    "{name} {pf:?} lane {lane}: the runtime cubic sampler's {g:?}, the constexpr \
                     bicubic one's {c:?}"
                );
            }
        }
    }
}

/// A runtime cubic sampler's state that no bicubic reading covers is refused
/// `NotImplemented(SamplerBicubicFilter)` under its variant: a mip filter (Metal samples through
/// it, but the level it reads is not modelled), `MirrorClampToEdge`, a min or max reduction, and
/// anisotropy.
#[test]
fn a_runtime_cubic_sampler_state_without_a_reading_is_refused() {
    use crate::metal::sampler::SamplerWord as W;
    use crate::metal::SamplerAddressMode as A;
    let p = m3();
    let Some(runtime) = assemble("runtime cubic", &runtime_kernel()) else {
        return;
    };
    let edge = cubic_word(A::ClampToEdge.as_raw(), 0, false);
    let mirror_clamp = edge & !W::ADDRESS[0] | A::MirrorClampToEdge.as_raw() << W::ADDRESS_SHIFT[0];
    for (name, word) in [
        ("nearest mip filter", edge | 1 << W::MIP_FILTER_SHIFT),
        ("linear mip filter", edge | 2 << W::MIP_FILTER_SHIFT),
        ("mirror clamp to edge", mirror_clamp),
        ("minimum reduction", edge | 1 << W::REDUCTION_SHIFT),
        ("anisotropy", edge | W::ANISOTROPIC),
    ] {
        let mut inputs = crate::lower::select::LowerInputs::new();
        inputs.cubic_samplers.insert(SAMPLER, word);
        match compile_with(&runtime, "k", &p, &inputs) {
            Err(e) if refused(&e) => {}
            other => panic!("{name}: {other:?}"),
        }
    }
}
