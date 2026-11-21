//! A switch-at-use whose every arm takes the same emulation emulates once (SEMANTICS 1012): a
//! kernel samples one element of an array of `n` textures, chosen per lane, through one constexpr
//! sampler whose sampling the translation emulates on lavapipe: Metal's bicubic filter.
//!
//! - **The words do not scale with arms × emulation.** Each arm adds the same words whatever the
//!   emulation's size (a repeating sampler's position against a pixel sampler's none), and less
//!   than a quarter of the emulation itself.
//! - **The values are each candidate's own.** On the interpreter, every lane of the `n`-candidate
//!   kernel equals the lane of the one-texture kernel bound to its candidate (with that
//!   candidate's size and format), and the null texture (the switch's null arm) reads zero.
//!   Candidates differ in size and format, so the emulation's facts must be the named one's.
//! - **The estimate bounds the emission** on every committed profile.
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::device::profile::DeviceProfile;
use crate::device::DevicePolicy;
use crate::lower::select::Plan;
use crate::metal::bicubic::BicubicTexels;
use crate::metal::PixelFormat;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{
    Axis, PushField, ResourceKind, ResourceSlot, SlotClass, SlotRef, SpecKey, Stage,
};
use crate::pipeline::testing::{
    compile, link_and_validate, llvm_assemble, profile, typed_module, AssembleError,
};
use crate::spirv::interp::{run_with_images, Buffer, Config, Image, Order, Resources};
use crate::spirv::ir::{GlobalKind, TypedModule};

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// Bitcode of the module `body` (after the AIR header), or `None` without `llvm-as`.
fn assemble(name: &str, body: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-shared-switch-{}-{}-{}",
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

fn lavapipe() -> DeviceProfile {
    profile("lavapipe", DevicePolicy::default()).expect("profile")
}

const EDGE: u64 = 1;
const REPEAT: u64 = 2;

/// The words of a constexpr sampler: minification and magnification `filter` (0 nearest, 1
/// linear), no mip filter, every axis addressed by `address` (0 zero, 1 edge, 2 repeat, 3
/// mirrored repeat), and `pixel` coordinates, with the fields Apple's compiler writes for the
/// rest (the `hlf_half_address` fixture's sampler).
fn sampler(filter: u64, address: u64, pixel: bool) -> u64 {
    const BASE: u64 = 0x007b_ff00_0008_0000;
    BASE | address
        | address << 3
        | address << 6
        | filter << 9
        | filter << 11
        | u64::from(pixel) << 15
}

/// The words of `constexpr sampler(filter::bicubic, ...)` with every axis addressed by `address`
/// and `pixel` coordinates (the `bcu_address` fixture's).
fn bicubic(address: u64, pixel: bool) -> u64 {
    const BASE: u64 = 0x007b_ff00_0008_1400;
    BASE | address | address << 3 | address << 6 | u64::from(pixel) << 15
}

/// A form of the tests: a `texture2d<float>::sample` through the constexpr sampler `words`.
#[derive(Clone, Copy, Debug)]
struct Form {
    name: &'static str,
    words: u64,
}

/// The emulation twice, as a pair whose emulations differ in size: through a repeating sampler,
/// which reads Metal's position, and through a pixel sampler, which reads none.
fn forms() -> [Form; 2] {
    [
        Form {
            name: "bicubic repeat",
            words: bicubic(REPEAT, false),
        },
        Form {
            name: "bicubic pixel edge",
            words: bicubic(EDGE, true),
        },
    ]
}

/// The same sample without an emulation: a nearest sample in pixel coordinates.
fn plain() -> Form {
    Form {
        name: "plain",
        words: sampler(0, EDGE, true),
    }
}

/// A kernel performing `form` at `float2` buffer 0's lane `tid`, storing the result to buffer 1:
/// on element `ks[tid]` (`uint` buffer 2) of an array of `n` textures (the null texture where
/// `ks[tid]` is past the array), or on one texture where `n` is `None`.
fn kernel(n: Option<u32>, form: Form) -> String {
    let (texel, vec4, suffix, size, vec_name) = ("float", "<4 x float>", "v4f32", 16, "float4");
    let call = format!("@air.sample_texture_2d.{suffix}(ptr addrspace(1) %h, ptr addrspace(2) @__air_sampler_state, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)");
    let decl = format!("@air.sample_texture_2d.{suffix}(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)");
    let (param, pick, count, tex_ty) = match n {
        // An index past the array names the null texture (the switch's null arm).
        Some(n) => (
            format!("ptr readonly byval([{n} x ptr addrspace(1)]) %t"),
            format!("  %past = icmp uge i32 %k, {n}\n  %kc = select i1 %past, i32 0, i32 %k\n  %kk = zext i32 %kc to i64\n  %e = getelementptr inbounds [{n} x %\"struct.metal::texture2d\"], ptr %t, i64 0, i64 %kk, i32 0\n  %elem = load ptr addrspace(1), ptr %e, align 8\n  %h = select i1 %past, ptr addrspace(1) null, ptr addrspace(1) %elem\n"),
            n,
            format!("array<texture2d<{texel}, sample>, {n}>"),
        ),
        None => (
            "ptr addrspace(1) %h".to_string(),
            String::new(),
            1,
            format!("texture2d<{texel}, sample>"),
        ),
    };
    let w = form.words;
    format!(
        r#"%"struct.metal::texture2d" = type {{ ptr addrspace(1) }}

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {w}, i64 0], align 8

define void @k({param}, ptr addrspace(1) %c, ptr addrspace(1) %out, ptr addrspace(1) %ks, i32 %tid) {{
  %i = zext i32 %tid to i64
  %p = getelementptr inbounds <2 x float>, ptr addrspace(1) %c, i64 %i
  %uv = load <2 x float>, ptr addrspace(1) %p, align 8
  %pk = getelementptr inbounds i32, ptr addrspace(1) %ks, i64 %i
  %k = load i32, ptr addrspace(1) %pk, align 4
{pick}  %r = call {{ {vec4}, i8 }} {call}
  %v = extractvalue {{ {vec4}, i8 }} %r, 0
  %q = getelementptr inbounds {vec4}, ptr addrspace(1) %out, i64 %i
  store {vec4} %v, ptr addrspace(1) %q, align {size}
  ret void
}}

declare {{ {vec4}, i8 }} {decl}

!air.kernel = !{{!0}}
!air.sampler_states = !{{!6}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5, !7, !8}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 {count}, !"air.sample", !"air.arg_type_name", !"{tex_ty}", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"float2", !"air.arg_name", !"c"}}
!5 = !{{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 {size}, !"air.arg_type_align_size", i32 {size}, !"air.arg_type_name", !"{vec_name}", !"air.arg_name", !"out"}}
!6 = !{{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}}
!7 = !{{i32 3, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"ks"}}
!8 = !{{i32 4, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}}
"#
    )
}

/// The words of `form`'s kernel over `n` candidates on lavapipe, which validates.
fn words(n: Option<u32>, form: Form) -> Option<usize> {
    let what = format!("{} n={n:?}", form.name);
    let bc = assemble(&what, &kernel(n, form))?;
    let p = lavapipe();
    let a = compile(&bc, "k", &p).unwrap_or_else(|e| panic!("{what}: {e}"));
    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
    Some(a.module.words.len())
}

/// An arm of a switch-at-use whose arms take one emulation costs its image's reads, not the
/// emulation. Between 16 and 32 candidates each arm adds the same words whether the emulation reads
/// Metal's position (a repeating sampler) or not (a pixel one), and an arm's words past the plain
/// sample's are less than a quarter of the emulation's own.
#[test]
fn the_switch_words_do_not_scale_with_arms_times_emulation() {
    let [big, small] = forms();
    let mut slopes = Vec::new();
    // The 16 arms from 16 to 32 candidates (and one more interned constant at most, the same
    // for both forms).
    for form in [big, small] {
        let (Some(w16), Some(w32)) = (words(Some(16), form), words(Some(32), form)) else {
            return;
        };
        slopes.push((w32 - w16) / 16);
    }
    let plain = plain();
    let (Some(p16), Some(p32)) = (words(Some(16), plain), words(Some(32), plain)) else {
        return;
    };
    let plain_slope = (p32 - p16) / 16;
    let (Some(one), Some(one_plain)) = (words(None, big), words(None, plain)) else {
        return;
    };
    let emulation = one - one_plain;
    eprintln!(
        "{}: per arm {} ({}: {}), plain {plain_slope}, emulation {emulation}",
        big.name, slopes[0], small.name, slopes[1]
    );
    assert_eq!(
        slopes[0], slopes[1],
        "{} and {}: an arm's words depend on the emulation's",
        big.name, small.name
    );
    assert!(
        4 * slopes[0].saturating_sub(plain_slope) < emulation,
        "{}: an arm adds {} words past the plain {plain_slope}, against an emulation of {emulation}",
        big.name,
        slopes[0]
    );
}

/// The combined image-sampler bindings of `m`, by texture slot.
fn combined_slots(m: &TypedModule) -> BTreeMap<u32, ResourceSlot> {
    m.globals()
        .iter()
        .filter_map(|g| match g.kind {
            GlobalKind::Resource(
                s @ ResourceSlot {
                    kind: ResourceKind::CombinedTextureSampler(_),
                    ..
                },
            ) => Some((s.index, s)),
            _ => None,
        })
        .collect()
}

fn texture(index: u32) -> SlotRef {
    SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Texture,
        index,
    }
}

/// Resources of a dispatch of one-thread threadgroups: `[[buffer(i)]]` holding `init[i]`, and
/// the specializations `spec`.
fn resources(plan: &Plan, init: &[Vec<u8>], spec: &[(SpecKey, u64)]) -> Resources {
    let mut s = BTreeMap::new();
    for axis in [Axis::X, Axis::Y, Axis::Z] {
        s.insert(SpecKey::LocalSize(axis), 1);
    }
    s.insert(SpecKey::FlatLocalSize, 1);
    s.extend(spec.iter().copied());
    let push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    for f in &plan.push.fields {
        match (f.field, f.loc) {
            (PushField::SlotBindOffset(_) | PushField::DispatchRegion, PushLoc::Push { .. }) => {}
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
        spec: s,
    }
}

/// `m` over `lanes` one-thread threadgroups with `images` bound, in both scheduling orders, which
/// must agree: buffer 1's bytes.
fn run(m: &TypedModule, res: &Resources, images: Vec<Image>, lanes: u32) -> Vec<u8> {
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            workgroups: [lanes, 1, 1],
            order,
            ..Config::default()
        };
        let (o, _) = run_with_images(m, &cfg, res.clone(), images.clone())
            .unwrap_or_else(|e| panic!("interpreter: {e:?}"));
        outs.push(o.buffers);
    }
    assert_eq!(outs[0], outs[1], "the scheduling orders disagree");
    outs[0][1].clone()
}

/// A deterministic source: a 64-bit LCG.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 11
    }

    /// A value in `[lo, hi)`.
    fn unit(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * (self.next() % 4096) as f32 / 4096.0
    }
}

/// A `half`'s value.
fn half_value(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let (e, m) = (i32::from(h >> 10 & 0x1F), f64::from(h & 0x3FF));
    let a = match e {
        0 => m * 2f64.powi(-24),
        _ => (1024.0 + m) * 2f64.powi(e - 25),
    };
    (sign * a) as f32
}

/// A texel channel as format class `class` stores it, fetched as a `float`.
fn channel(class: u32, rng: &mut Lcg) -> f32 {
    let has = |b: u32| class & b != 0;
    if has(BicubicTexels::LINEAR) {
        return rng.unit(-4.0, 4.0);
    }
    if has(BicubicTexels::HALF) {
        // A finite `half` below 8 in magnitude.
        let h = (rng.next() % 0x4800) as u16 | (rng.next() as u16 & 0x8000);
        return half_value(h);
    }
    let wide = has(BicubicTexels::WIDE16);
    let scale = if wide { 65535 } else { 255 };
    (rng.next() % (scale + 1)) as f32 / scale as f32
}

/// Candidate `k`'s image at `slot`: its own size, its format class's texels.
fn image(slot: ResourceSlot, k: u32, class: u32) -> Image {
    let (w, h) = (5 + k, 4 + k % 3);
    let mut img = Image::new(slot, [w, h, 1], 1, 1, 1);
    let mut rng = Lcg(u64::from(k) * 977 + u64::from(class));
    for y in 0..h {
        for x in 0..w {
            let at = img.index(0, 0, 0, [x, y, 0]).expect("texel");
            img.texels[at] = std::array::from_fn(|_| channel(class, &mut rng).to_bits());
        }
    }
    img
}

/// Every lane of the shared emulation over `n` candidates of different sizes and formats is its
/// candidate's own lane, and the null texture reads zero, for Metal's bicubic filter on the
/// interpreter.
#[test]
fn the_shared_emulation_answers_each_candidates_own_lane() {
    let p = lavapipe();
    let classes: Vec<u32> = [
        PixelFormat::RGBA8Unorm,
        PixelFormat::RGBA32Float,
        PixelFormat::R8Unorm,
    ]
    .iter()
    .map(|pf| {
        crate::metal::bicubic::texels(*pf)
            .expect("a bicubic class")
            .0
    })
    .collect();
    for form in forms() {
        for n in [3u32, 4] {
            let what = format!("{} n={n}", form.name);
            let Some(one) = assemble(&what, &kernel(None, form)) else {
                return;
            };
            let Some(many) = assemble(&what, &kernel(Some(n), form)) else {
                return;
            };
            let lanes = 40u32;
            let mut rng = Lcg(u64::from(n) + 5);
            let pixel = form.name.contains("pixel");
            let mut coords = Vec::new();
            for _ in 0..lanes {
                let (u, v) = if pixel {
                    (rng.unit(-1.0, 9.0), rng.unit(-1.0, 7.0))
                } else {
                    (rng.unit(-0.75, 1.75), rng.unit(-0.75, 1.75))
                };
                coords.extend(u.to_le_bytes());
                coords.extend(v.to_le_bytes());
            }
            let index: Vec<u32> = (0..lanes).map(|l| l % (n + 1)).collect();
            let ks: Vec<u8> = index.iter().flat_map(|k| k.to_le_bytes()).collect();
            let size = 16;
            let out = vec![0xabu8; size * lanes as usize];
            let key = |slot: u32| SpecKey::BicubicTexels(texture(slot));
            let class_of = |k: u32| classes[k as usize % classes.len()];
            // Each candidate alone.
            let (plan, m) = typed_module(&one, "k", &p).expect("typed module");
            let slot = *combined_slots(&m)
                .values()
                .next()
                .expect("a combined binding");
            let mut alone = Vec::new();
            for k in 0..n {
                let res = resources(
                    &plan,
                    &[coords.clone(), out.clone(), ks.clone()],
                    &[(key(0), u64::from(class_of(k)))],
                );
                alone.push(run(&m, &res, vec![image(slot, k, class_of(k))], lanes));
            }
            // All of them behind the switch.
            let (plan, m) = typed_module(&many, "k", &p).expect("typed module");
            let slots = combined_slots(&m);
            assert_eq!(
                slots.keys().copied().collect::<Vec<_>>(),
                (0..n).collect::<Vec<_>>(),
                "{what}"
            );
            let spec: Vec<(SpecKey, u64)> =
                (0..n).map(|k| (key(k), u64::from(class_of(k)))).collect();
            let images = (0..n).map(|k| image(slots[&k], k, class_of(k))).collect();
            let res = resources(&plan, &[coords.clone(), out.clone(), ks.clone()], &spec);
            let got = run(&m, &res, images, lanes);
            for (lane, k) in index.iter().enumerate() {
                let at = lane * size..(lane + 1) * size;
                let want = match alone.get(*k as usize) {
                    Some(a) => a[at.clone()].to_vec(),
                    None => vec![0; size],
                };
                assert_eq!(got[at], want[..], "{what} lane {lane} (candidate {k})");
            }
        }
    }
}

/// The plan's word estimate bounds the emitted module of each shared form over 2, 8 and 32
/// candidates on every committed profile that admits it.
#[test]
fn the_shared_estimate_bounds_its_emission() {
    use crate::budget::SPIRV_WORDS;
    use crate::library::FunctionConstantValues;
    use crate::lower::select::LowerInputs;
    use crate::pipeline::compile::{compile_bitcode, plan_bitcode, EntryRef};
    use crate::pipeline::testing::meter;
    let profiles = crate::spirv::testing::committed_profiles().expect("profiles");
    let nearest = Form {
        name: "nearest repeat",
        words: sampler(0, REPEAT, false),
    };
    let mut all = forms().to_vec();
    all.push(nearest);
    for form in all {
        for n in [2u32, 8, 32] {
            let name = format!("{} n={n}", form.name);
            let Some(bc) = assemble(&name, &kernel(Some(n), form)) else {
                return;
            };
            for (stem, p) in &profiles {
                let what = format!("{name} on {stem}");
                let (entry, consts) = (EntryRef::Name("k"), FunctionConstantValues::defaults());
                let inputs = LowerInputs::new();
                let plan = match plan_bitcode(&bc, entry, &consts, p, &inputs, &mut meter()) {
                    Ok(plan) => plan,
                    Err(e) => {
                        assert!(stem != "lavapipe", "{what}: {e}");
                        eprintln!("{what}: refused: {e}");
                        continue;
                    }
                };
                let a = compile_bitcode(&bc, entry, &consts, p, &inputs, &mut meter(), SPIRV_WORDS)
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                let emitted = a.module.words.len() as u64;
                eprintln!("{what}: estimate {} emitted {emitted}", plan.estimate.words);
                assert!(
                    plan.estimate.words >= emitted,
                    "{what}: estimate {} < emitted {emitted}",
                    plan.estimate.words
                );
            }
        }
    }
}
