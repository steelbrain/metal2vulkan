//! A two-plane YCbCr texture's planes read in place (`lower::sampler::in_place`), on the CPU
//! interpreter against Metal's texels and Metal's filter of them, bit for bit.
//!
//! The slot under its in-place variant (`LoweringInputs::ycbcr_in_place`) is a uniform texel
//! buffer of the planes' codes (`R8G8_UINT` elements of 8-bit codes, `R16G16_UINT` ones of 10-bit
//! codes, each 10-bit code the high bits of its word over junk low bits), and its push field
//! states the texture's size, its planes' places and rows, and its conversion
//! (`bind::YCbCrInPlace::words`). Each texel the samples read is Metal's (`ycbcr_convert`'s module
//! docs: the luma code, the chroma pairs' 9/3/3/1 filter, the range's and the matrix's
//! conversion), and a linear sample is Metal's filter of those steps on the grid `2048`
//! (`metal_step`) under `YCbCrInPlacePolicy::Exact`; a nearest sample is the texel at Metal's
//! position. Under `YCbCrInPlacePolicy::Model` (the default) a linear sample is Vulkan's Y′CbCr
//! model of the codes (`model`), within the `float` arithmetic's rounding. The texel buffer
//! traps every fetch outside it, so each tap's elements lie within the planes.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::bind::ycbcr_convert::{ycbcr_read, YCbCrCodes, YCbCrMatrix, YCbCrPlanes, YCbCrRange};
use crate::bind::YCbCrInPlace;
use crate::device::profile::DeviceProfile;
use crate::device::{DevicePolicy, YCbCrInPlacePolicy};
use crate::lower::select::{LowerInputs, Plan};
use crate::metal::format::ycbcr_format;
use crate::metal::sampler::SamplerWord;
use crate::metal::{PixelFormat, SamplerAddressMode};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{
    Axis, PushField, ResourceKind, ResourceSlot, SlotClass, SlotRef, SpecKey, Stage,
};
use crate::pipeline::testing::{
    compile_with, link_and_validate, llvm_assemble, profile, typed_module_with, AssembleError,
};
use crate::spirv::interp::{run_with_images, Buffer, Config, Image, Order, Resources};
use crate::spirv::ir::{GlobalKind, TypedModule};

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// Bitcode of the module `body` (after the AIR header), or `None` without `llvm-as`.
fn assemble(name: &str, body: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-linear-half-{}-{}-{}",
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

fn m3() -> DeviceProfile {
    profile("moltenvk-m3", DevicePolicy::default()).expect("profile")
}

/// Profile `stem` under `YCbCrInPlacePolicy::Exact`, the default policy's every other field.
fn exact(stem: &str) -> DeviceProfile {
    let policy = DevicePolicy {
        ycbcr_in_place_reads: YCbCrInPlacePolicy::Exact,
        ..DevicePolicy::default()
    };
    profile(stem, policy).expect("profile")
}

/// The words of `constexpr sampler(filter::linear, ...)`: minification and magnification
/// `filter` (AIR code: 0 nearest, 1 linear), mip filter `mip` (0 none, 2 linear), every axis
/// addressed by `address` (0 zero or border, 1 edge, 2 repeat, 3 mirrored repeat), `pixel`
/// coordinates, and border color code `border` (0 transparent black, 2 opaque white), with the
/// fields Apple's compiler writes for the rest (the `hlf_half_address` fixture's sampler).
fn sampler_words(filter: u64, mip: u64, address: u64, pixel: bool, border: u64) -> u64 {
    const BASE: u64 = 0x007b_ff00_0008_0000;
    BASE | address
        | address << 3
        | address << 6
        | filter << 9
        | filter << 11
        | mip << 13
        | u64::from(pixel) << 15
        | border << 56
}

fn linear(address: u64, pixel: bool, border: u64) -> u64 {
    sampler_words(1, 0, address, pixel, border)
}

/// How a test kernel samples its `half` texture.
#[derive(Clone, Copy, Debug)]
enum Form {
    /// `texture2d<half>`: `t.sample(s, coord[tid], offset)` with constant `offset`.
    Plain { offset: (i32, i32) },
    /// `texture2d<half>` with `min_lod_clamp(1)`.
    MinLodClamp,
    /// `texture2d<half>` with lane `tid`'s dynamic `int2` offset (buffer 2).
    DynamicOffset,
    /// `texture3d<half>` at `float4` buffer 0's lane `tid`'s `xyz`, with constant `offset`.
    ThreeD { offset: (i32, i32, i32) },
    /// `texture2d<float>`: a `float` result.
    Float,
    /// `texture2d<half>`: `t.gather(s, coord[tid], int2(0), component)`, `component` 0 to 3.
    Gather { component: u32 },
    /// `texture2d<half>`: a sample, then a gather of red, of one coordinate, written to elements
    /// `2 tid` and `2 tid + 1`: the shape of a probe that samples or gathers as a lane asks.
    SampleAndGather,
}

/// A kernel sampling texture slot 0 through sampler `sampler` (a constexpr sampler's words, or
/// `None` for the runtime `[[sampler(0)]]`) at `float2` buffer 0's lane `tid`, storing the
/// `half4` (`float4` for [`Form::Float`]) to buffer 1.
fn kernel(sampler: Option<u64>, form: Form) -> String {
    kernel_of(sampler, form, matches!(form, Form::Float))
}

/// [`kernel`] with a `float` result when `float`, whatever the form.
fn kernel_of(sampler: Option<u64>, form: Form, float: bool) -> String {
    let half = !float;
    let (texel, vec4, size) = if half {
        ("half", "<4 x half>", 8)
    } else {
        ("float", "<4 x float>", 16)
    };
    let suffix = if half { "v4f16" } else { "v4f32" };
    let s = if sampler.is_some() {
        "@__air_sampler_state"
    } else {
        "%s"
    };
    const SAMPLE_2D: &str =
        "ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32";
    const GATHER_2D: &str =
        "ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i32, i32";
    let plain = format!("texture2d<{texel}, sample>");
    let gather = |component: u32| {
        (
            "gather_texture_2d",
            GATHER_2D,
            format!("<2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i32 {component}, i32 0"),
        )
    };
    let (tex_ty, calls) = match form {
        Form::Plain { offset: (ox, oy) } => (
            plain,
            vec![(
                "sample_texture_2d",
                SAMPLE_2D,
                format!("<2 x float> %uv, i1 true, <2 x i32> <i32 {ox}, i32 {oy}>, i1 false, float 0.0, float 0.0, i32 0"),
            )],
        ),
        Form::Float => (
            plain,
            vec![(
                "sample_texture_2d",
                SAMPLE_2D,
                "<2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0".to_string(),
            )],
        ),
        Form::MinLodClamp => (
            plain,
            vec![(
                "sample_texture_2d",
                SAMPLE_2D,
                "<2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 1.0, i32 0".to_string(),
            )],
        ),
        Form::DynamicOffset => (
            plain,
            vec![(
                "sample_texture_2d",
                SAMPLE_2D,
                "<2 x float> %uv, i1 true, <2 x i32> %o, i1 false, float 0.0, float 0.0, i32 0".to_string(),
            )],
        ),
        Form::ThreeD { offset: (ox, oy, oz) } => (
            format!("texture3d<{texel}, sample>"),
            vec![(
                "sample_texture_3d",
                "ptr addrspace(1), ptr addrspace(2), <3 x float>, i1, <3 x i32>, i1, float, float, i32",
                format!("<3 x float> %uvw, i1 true, <3 x i32> <i32 {ox}, i32 {oy}, i32 {oz}>, i1 false, float 0.0, float 0.0, i32 0"),
            )],
        ),
        Form::Gather { component } => (plain, vec![gather(component)]),
        Form::SampleAndGather => (
            plain,
            vec![
                (
                    "sample_texture_2d",
                    SAMPLE_2D,
                    "<2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0".to_string(),
                ),
                gather(0),
            ],
        ),
    };
    // Each call's result to element `tid` of `out`, or of two calls `2 tid + k`.
    let mut body = String::new();
    let mut decls = String::new();
    if calls.len() > 1 {
        body.push_str("  %i2 = mul i64 %i, 2\n");
    }
    for (k, (intrinsic, args, call_args)) in calls.iter().enumerate() {
        let at = match calls.len() {
            1 => "%i".to_string(),
            _ => {
                body.push_str(&format!("  %i2_{k} = add i64 %i2, {k}\n"));
                format!("%i2_{k}")
            }
        };
        body.push_str(&format!(
            "  %r{k} = call {{ {vec4}, i8 }} @air.{intrinsic}.{suffix}(ptr addrspace(1) %t, ptr addrspace(2) {s}, {call_args})\n  %v{k} = extractvalue {{ {vec4}, i8 }} %r{k}, 0\n  %q{k} = getelementptr inbounds {vec4}, ptr addrspace(1) %out, i64 {at}\n  store {vec4} %v{k}, ptr addrspace(1) %q{k}, align {size}\n"
        ));
        decls.push_str(&format!(
            "declare {{ {vec4}, i8 }} @air.{intrinsic}.{suffix}({args})\n"
        ));
    }
    let dynamic = matches!(form, Form::DynamicOffset);
    let runtime = sampler.is_none();
    // Parameters after `out`: the offsets buffer, then the runtime sampler.
    let mut params = String::new();
    let mut mds = Vec::new();
    let mut next = 3;
    if dynamic {
        params.push_str(", ptr addrspace(1) %offs");
        mds.push(format!("!{{i32 {next}, !\"air.buffer\", !\"air.location_index\", i32 2, i32 1, !\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 8, !\"air.arg_type_align_size\", i32 8, !\"air.arg_type_name\", !\"int2\", !\"air.arg_name\", !\"offs\"}}"));
        next += 1;
    }
    if runtime {
        params.push_str(", ptr addrspace(2) %s");
        mds.push(format!("!{{i32 {next}, !\"air.sampler\", !\"air.location_index\", i32 0, i32 1, !\"air.arg_type_name\", !\"sampler\", !\"air.arg_name\", !\"s\"}}"));
        next += 1;
    }
    mds.push(format!("!{{i32 {next}, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"tid\"}}"));
    let first = 7;
    let arg_mds: Vec<String> = (0..mds.len()).map(|i| format!("!{}", first + i)).collect();
    let tail: String = mds
        .iter()
        .enumerate()
        .map(|(i, m)| format!("!{} = {m}\n", first + i))
        .collect();
    let load_offset = if dynamic {
        "  %po = getelementptr inbounds <2 x i32>, ptr addrspace(1) %offs, i64 %i\n  %o = load <2 x i32>, ptr addrspace(1) %po, align 8\n"
    } else {
        ""
    };
    let state = match sampler {
        Some(w) => format!(
            "@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {w}, i64 0], align 8\n\n"
        ),
        None => String::new(),
    };
    let (states, state_md) = if sampler.is_some() {
        (
            "!air.sampler_states = !{!6}\n",
            "!6 = !{!\"air.sampler_state\", ptr addrspace(2) @__air_sampler_state}\n",
        )
    } else {
        ("", "")
    };
    let half_name = if half { "half4" } else { "float4" };
    // A 3D sample's coordinates are a `float4`'s `xyz`, every other's a `float2`.
    let (load_coord, coord_size, coord_name) = if matches!(form, Form::ThreeD { .. }) {
        (
            "  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %c, i64 %i\n  %c4 = load <4 x float>, ptr addrspace(1) %p, align 16\n  %uvw = shufflevector <4 x float> %c4, <4 x float> zeroinitializer, <3 x i32> <i32 0, i32 1, i32 2>\n",
            16,
            "float4",
        )
    } else {
        (
            "  %p = getelementptr inbounds <2 x float>, ptr addrspace(1) %c, i64 %i\n  %uv = load <2 x float>, ptr addrspace(1) %p, align 8\n",
            8,
            "float2",
        )
    };
    format!(
        r#"{state}define void @k(ptr addrspace(1) %t, ptr addrspace(1) %c, ptr addrspace(1) %out{params}, i32 %tid) {{
  %i = zext i32 %tid to i64
{load_coord}{load_offset}{body}  ret void
}}

{decls}
!air.kernel = !{{!0}}
{states}!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5, {arg_mds}}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"{tex_ty}", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 {coord_size}, !"air.arg_type_align_size", i32 {coord_size}, !"air.arg_type_name", !"{coord_name}", !"air.arg_name", !"c"}}
!5 = !{{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 {size}, !"air.arg_type_align_size", i32 {size}, !"air.arg_type_name", !"{half_name}", !"air.arg_name", !"out"}}
{state_md}{tail}"#,
        arg_mds = arg_mds.join(", "),
    )
}

const TEXTURE: SlotRef = SlotRef {
    stage: Stage::Kernel,
    class: SlotClass::Texture,
    index: 0,
};

/// `res` with each `SamplerState` specialization artifact `a` declares valued by runtime sampler
/// word `word`, masked by its fields; unchanged without a word.
fn with_word(
    mut res: Resources,
    a: &crate::library::FunctionArtifact,
    word: Option<u32>,
) -> Resources {
    let Some(word) = word else {
        return res;
    };
    let mut declared = 0;
    for key in a.interface.read_keys() {
        if let SpecKey::SamplerState { fields, .. } = key {
            res.spec.insert(key, u64::from(word & fields));
            declared += 1;
        }
    }
    assert_eq!(
        declared, 1,
        "the runtime sampler's `SamplerState` specialization"
    );
    res
}

/// `m` over `groups` one-thread threadgroups with `image` bound, in both scheduling orders, which
/// must agree: buffer 1's `half4`s (`float4`s unless `half`), as bits.
fn run(m: &TypedModule, res: &Resources, image: &Image, groups: u32, half: bool) -> Vec<[u32; 4]> {
    run_images(m, res, std::slice::from_ref(image), groups, half)
}

/// [`run`] with `images` bound.
fn run_images(
    m: &TypedModule,
    res: &Resources,
    images: &[Image],
    groups: u32,
    half: bool,
) -> Vec<[u32; 4]> {
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            workgroups: [groups, 1, 1],
            order,
            ..Config::default()
        };
        let (o, _) = run_with_images(m, &cfg, res.clone(), images.to_vec())
            .unwrap_or_else(|e| panic!("interpreter: {e:?}"));
        outs.push(o.buffers);
    }
    assert_eq!(outs[0], outs[1], "the scheduling orders disagree");
    let size = if half { 2 } else { 4 };
    outs[0][1]
        .chunks_exact(4 * size)
        .map(|c| {
            std::array::from_fn(|k| {
                let mut b = [0u8; 4];
                b[..size].copy_from_slice(&c[size * k..size * (k + 1)]);
                u32::from_le_bytes(b)
            })
        })
        .collect()
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
}

/// Metal's step of a 2D linear sample of a YCbCr format: the taps' steps `k` (`(i, j)`, `(i + 1,
/// j)`, `(i, j + 1)`, `(i + 1, j + 1)`) under the fractions `a` and `b` in 1/256 steps, a merged
/// axis's 0; a signed term and sum floored.
fn metal_step(taps: [i64; 4], a: i64, b: i64) -> i64 {
    let w = [(256 - a) * (256 - b), a * (256 - b), (256 - a) * b, a * b];
    let t: i64 = taps.iter().zip(w).map(|(k, w)| (k * w) >> 7).sum();
    (t + 256) >> 9
}

/// The `half` bits of `k / d` (`d` a power of two, `1 <= |k| / d`'s binade above `2^-14`, or `k
/// = 0`), rounded with ties away from zero.
fn half_away(k: i64, d: i64) -> u16 {
    if k == 0 {
        return 0;
    }
    let sign = if k < 0 { 0x8000 } else { 0 };
    let m = k.unsigned_abs();
    let lead = i64::from(63 - m.leading_zeros());
    let e = lead - i64::from(d.trailing_zeros());
    assert!(e >= -14, "{k} / {d} is below the normal halves");
    // The significand's 11 bits: the magnitude rounded half up there, a carry raising `e`.
    let shift = lead - 10;
    let q = match shift > 0 {
        true => (m + (1 << (shift - 1))) >> shift,
        false => m << -shift,
    };
    let (q, e) = if q == 2048 { (1024, e + 1) } else { (q, e) };
    sign | ((e + 15) as u16) << 10 | (q as u16 - 1024)
}

/// The result lanes of steps `steps` on grid `d` (alpha 1): `k / d` as `f32` bits (exact), or as
/// `half` bits rounded with ties away from zero when `half`.
fn result_bits(steps: [i64; 3], half: bool, d: i64) -> [u32; 4] {
    let lanes = [steps[0], steps[1], steps[2], d];
    lanes.map(|k| match half {
        true => u32::from(half_away(k, d)),
        false => (k as f32 / d as f32).to_bits(),
    })
}

/// The texture's width and height: odd, so the chroma plane's last pair serves one column and one
/// row alone.
const W: u32 = 7;
const H: u32 = 5;

/// Lowering inputs with texture slot 0 under its in-place variant, or none.
fn inputs(in_place: bool) -> LowerInputs {
    let mut i = LowerInputs::new();
    if in_place {
        i.ycbcr_in_place.insert(TEXTURE);
    }
    i
}

/// One texture's planes: their codes by texel, where they lie, and the texel buffer's elements.
struct Planes {
    planes: YCbCrPlanes,
    words: YCbCrInPlace,
    /// Each element's two lanes, as the texel buffer's format reads them.
    elements: Vec<[u32; 2]>,
    luma: Vec<u16>,
    chroma: Vec<[u16; 2]>,
}

impl Planes {
    /// Random planes of `pf` (`YCBCR8_420_2P` or `YCBCR10_420_2P`) under `conversion`: the chroma
    /// plane first, three elements in, then the luma plane, each row padded past its codes.
    fn random(pf: PixelFormat, conversion: (YCbCrRange, YCbCrMatrix), rng: &mut Lcg) -> Planes {
        let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).expect("codes");
        let unit = codes.bytes;
        let element = 2 * unit;
        let [cw, ch] = [W.div_ceil(2), H.div_ceil(2)];
        let chroma = [3 * element, (cw + 2) * element];
        let luma_at = (chroma[0] + ch * chroma[1]).next_multiple_of(element) + element;
        let luma = [luma_at, (W.div_ceil(2) + 1) * element];
        let planes = YCbCrPlanes::new([W, H], luma, chroma, conversion, codes);
        let words = YCbCrInPlace::of(&planes).expect("in place");
        let max = (1u64 << codes.bits) - 1;
        let mut code = || (rng.next() % (max + 1)) as u16;
        let luma_codes: Vec<u16> = (0..W * H).map(|_| code()).collect();
        let chroma_codes: Vec<[u16; 2]> = (0..cw * ch).map(|_| [code(), code()]).collect();
        // Every lane a junk value, then each code's place written: a 10-bit code is the high bits
        // of its 16-bit word.
        let n = words.elements() as usize;
        let mut lanes: Vec<u32> = (0..2 * n)
            .map(|_| (rng.next() % (1 << (8 * unit))) as u32)
            .collect();
        let shift = 8 * unit - codes.bits;
        let mut put = |byte: u32, code: u16, junk: u64| {
            let lane = (byte / unit) as usize;
            lanes[lane] = (u32::from(code) << shift) | (junk as u32 & ((1 << shift) - 1));
        };
        for y in 0..H {
            for x in 0..W {
                let c = luma_codes[(y * W + x) as usize];
                put(luma[0] + y * luma[1] + x * unit, c, rng.next());
            }
        }
        for y in 0..ch {
            for x in 0..cw {
                let [cb, cr] = chroma_codes[(y * cw + x) as usize];
                let at = chroma[0] + y * chroma[1] + 2 * x * unit;
                put(at, cb, rng.next());
                put(at + unit, cr, rng.next());
            }
        }
        let elements = lanes.as_chunks::<2>().0.to_vec();
        Planes {
            planes,
            words,
            elements,
            luma: luma_codes,
            chroma: chroma_codes,
        }
    }

    /// Metal's texel `(x, y)`: each channel's step on the grid `2048`.
    fn texel(&self, x: i64, y: i64) -> [i64; 3] {
        let [cw, ch] = [W.div_ceil(2), H.div_ceil(2)].map(i64::from);
        // An axis's chroma texels weighted 3 and 1: its own, and the one after it for an odd
        // index, before it for an even one, clamped to the plane.
        let taps = |q: i64, n: i64| -> [(i64, u32); 2] {
            let own = q / 2;
            let other = if q % 2 == 1 { own + 1 } else { own - 1 };
            [(own, 3), (other.clamp(0, n - 1), 1)]
        };
        let mut sums = [0u32; 2];
        for (cy, wy) in taps(y, ch) {
            for (cx, wx) in taps(x, cw) {
                let pair = self.chroma[(cy * cw + cx) as usize];
                for (sum, c) in sums.iter_mut().zip(pair) {
                    *sum += wx * wy * u32::from(c);
                }
            }
        }
        let [cb, cr] = sums.map(|s| ((s + 8) >> 4) as u16);
        let luma = self.luma[(y * i64::from(W) + x) as usize];
        let p = &self.planes;
        ycbcr_read(p.codes, (p.range, p.matrix), [luma, cb, cr]).map(i64::from)
    }

    /// The texel buffer bound at `slot`.
    fn image(&self, slot: ResourceSlot) -> Image {
        let n = self.elements.len() as u32;
        let mut image = Image::new(slot, [n, 1, 1], 1, 1, 1);
        for (x, [a, b]) in (0u32..).zip(&self.elements) {
            let i = image.index(0, 0, 0, [x, 0, 0]).expect("element");
            image.texels[i] = [*a, *b, 0, 1];
        }
        image
    }
}

/// The texel buffer binding a module declares for texture slot 0.
fn texel_buffer(m: &TypedModule) -> ResourceSlot {
    m.globals()
        .iter()
        .find_map(|g| match g.kind {
            GlobalKind::Resource(
                s @ ResourceSlot {
                    kind: ResourceKind::Texture,
                    index: 0,
                    ..
                },
            ) => Some(s),
            _ => None,
        })
        .expect("the planes' binding")
}

/// Resources of a dispatch of one-thread threadgroups: `[[buffer(i)]]` holding `init[i]`, and
/// the slot's push field holding `words`.
fn resources(plan: &Plan, init: &[Vec<u8>], words: &YCbCrInPlace) -> Resources {
    let mut spec = BTreeMap::new();
    for axis in [Axis::X, Axis::Y, Axis::Z] {
        spec.insert(SpecKey::LocalSize(axis), 1);
    }
    spec.insert(SpecKey::FlatLocalSize, 1);
    let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    let mut written = false;
    for f in &plan.push.fields {
        match (f.field, f.loc) {
            (PushField::YCbCrInPlace(TEXTURE), PushLoc::Push { offset }) => {
                let at = offset as usize;
                for (i, w) in words.words().into_iter().enumerate() {
                    push[at + 4 * i..at + 4 * i + 4].copy_from_slice(&w.to_le_bytes());
                }
                written = true;
            }
            (PushField::SlotBindOffset(_) | PushField::DispatchRegion, PushLoc::Push { .. }) => {}
            (PushField::Specialized(_), _) => {}
            other => panic!("unexpected push field {other:?}"),
        }
    }
    assert!(written, "the planes' push field");
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

/// A sampler a test samples through: a constexpr sampler's words, or a runtime sampler's word.
#[derive(Clone, Copy, Debug)]
enum Through {
    Constexpr(u64),
    Runtime(u32),
}

/// The word of a runtime sampler of `filter` (linear, else nearest), every axis addressed by
/// `mode`, normalized coordinates, transparent black.
fn word(linear: bool, mode: SamplerAddressMode) -> u32 {
    let mut w = if linear { SamplerWord::LINEAR } else { 0 };
    for shift in SamplerWord::ADDRESS_SHIFT {
        w |= mode.as_raw() << shift;
    }
    w
}

/// Metal's sample of `planes` at footprint position `q` (`256 (x - 1/2)` per axis, the texel
/// coordinate `x`) offset by `offset` texels, clamped to the edge: linear, the rule's steps of
/// the four taps; nearest, the texel at `⌊x⌋`.
fn metal(planes: &Planes, linear: bool, (qx, qy): (i64, i64), offset: (i64, i64)) -> [i64; 3] {
    let edge = |i: i64, n: u32| i.clamp(0, i64::from(n) - 1);
    if !linear {
        let x = edge(((qx + 128) >> 8) + offset.0, W);
        let y = edge(((qy + 128) >> 8) + offset.1, H);
        return planes.texel(x, y);
    }
    let (i, j) = ((qx >> 8) + offset.0, (qy >> 8) + offset.1);
    let (i0, i1) = (edge(i, W), edge(i + 1, W));
    let (j0, j1) = (edge(j, H), edge(j + 1, H));
    let a = if i0 == i1 { 0 } else { qx & 255 };
    let b = if j0 == j1 { 0 } else { qy & 255 };
    let taps = [
        planes.texel(i0, j0),
        planes.texel(i1, j0),
        planes.texel(i0, j1),
        planes.texel(i1, j1),
    ];
    std::array::from_fn(|c| metal_step(taps.map(|t| t[c]), a, b))
}

/// Under `YCbCrInPlacePolicy::Exact` every sample of each format's random planes under each
/// conversion, through constexpr and runtime samplers (nearest and linear; a runtime sampler's
/// repeat sampled as clamp-to-edge, as a 4:2:0 texture's), with a constant offset or none, `float`
/// and `half` results, on lavapipe and on MoltenVK, is Metal's: a quarter of the positions past an
/// edge.
#[test]
fn in_place_samples_are_metals() {
    let conversions = [
        (YCbCrRange::Video, YCbCrMatrix::Bt601),
        (YCbCrRange::Video, YCbCrMatrix::Bt709),
        (YCbCrRange::Video, YCbCrMatrix::Bt2020),
        (YCbCrRange::Full, YCbCrMatrix::Bt601),
        (YCbCrRange::Full, YCbCrMatrix::Bt709),
        (YCbCrRange::Full, YCbCrMatrix::Bt2020),
    ];
    let mut rng = Lcg(0x1c_bc_7a_1e);
    let planes: Vec<Planes> = [PixelFormat::YCBCR84202P, PixelFormat::YCBCR104202P]
        .into_iter()
        .flat_map(|pf| conversions.map(|c| (pf, c)))
        .map(|(pf, c)| Planes::random(pf, c, &mut rng))
        .collect();
    // Footprint positions `q = 256 (x - 1/2)`, each axis within one and a half texels of the
    // texture; a nearest sample's off the texel edges, where `⌊x⌋` holds for any rounding.
    let positions: Vec<(i64, i64)> = (0..64)
        .map(|_| {
            let mut q = |n: u32| {
                let q = (rng.next() % (256 * u64::from(n) + 768)) as i64 - 384;
                match (q + 128) & 255 {
                    0..8 => q + 8,
                    248.. => q - 8,
                    _ => q,
                }
            };
            (q(W), q(H))
        })
        .collect();
    let out = |q: i64, n: u32| q < 0 || q >= 256 * (i64::from(n) - 1);
    let edges = positions
        .iter()
        .filter(|&&(qx, qy)| out(qx, W) || out(qy, H))
        .count();
    assert!(edges >= positions.len() / 4, "{edges} past an edge");
    let coords: Vec<u8> = positions
        .iter()
        .flat_map(|&(qx, qy)| {
            let n = |q: i64, n: u32| (q + 128) as f32 / (256 * i64::from(n)) as f32;
            [n(qx, W).to_le_bytes(), n(qy, H).to_le_bytes()]
        })
        .flatten()
        .collect();
    let edge = SamplerAddressMode::ClampToEdge;
    let cases = [
        (
            "constexpr linear",
            Through::Constexpr(linear(1, false, 0)),
            true,
            (0, 0),
        ),
        (
            "constexpr nearest",
            Through::Constexpr(sampler_words(0, 0, 1, false, 0)),
            false,
            (0, 0),
        ),
        (
            "constexpr offset",
            Through::Constexpr(linear(1, false, 0)),
            true,
            (2, -1),
        ),
        (
            "runtime linear",
            Through::Runtime(word(true, edge)),
            true,
            (0, 0),
        ),
        (
            "runtime nearest",
            Through::Runtime(word(false, edge)),
            false,
            (0, 0),
        ),
        (
            "runtime repeat",
            Through::Runtime(word(true, SamplerAddressMode::Repeat)),
            true,
            (0, 0),
        ),
    ];
    for (name, through, linear, offset) in cases {
        for half in [false, true] {
            let sampler = match through {
                Through::Constexpr(w) => Some(w),
                Through::Runtime(_) => None,
            };
            let form = Form::Plain {
                offset: (offset.0 as i32, offset.1 as i32),
            };
            let Some(bc) = assemble(name, &kernel_of(sampler, form, !half)) else {
                return;
            };
            for stem in ["lavapipe", "moltenvk-m3"] {
                let p = exact(stem);
                let what = format!("{name} half={half} on {stem}");
                let a = compile_with(&bc, "k", &p, &inputs(true))
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
                let forms: Vec<&String> = a.strategies.entries.keys().collect();
                assert!(
                    forms
                        .iter()
                        .any(|k| k.starts_with("sampler.ycbcr_in_place."))
                        && !forms
                            .iter()
                            .any(|k| k.starts_with("sampler.ycbcr_in_place.model")),
                    "{what}: {forms:?}"
                );
                let (plan, m) = typed_module_with(&bc, "k", &p, &inputs(true))
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                let slot = texel_buffer(&m);
                let n = positions.len();
                let size = if half { 8 } else { 16 };
                for planes in &planes {
                    let init = [coords.clone(), vec![0xab; size * n]];
                    let res = resources(&plan, &init, &planes.words);
                    let word = match through {
                        Through::Runtime(w) => Some(w),
                        Through::Constexpr(_) => None,
                    };
                    let res = with_word(res, &a, word);
                    let got = run(&m, &res, &planes.image(slot), n as u32, half);
                    for (q, g) in positions.iter().zip(got) {
                        let want = metal(planes, linear, *q, offset);
                        let p = &planes.planes;
                        assert_eq!(
                            g,
                            result_bits(want, half, 2048),
                            "{what} {:?} {}-bit q {q:?}: {g:x?}",
                            (p.range, p.matrix),
                            p.codes.bits,
                        );
                    }
                }
            }
        }
    }
}

/// The model's sample (`in_place`'s **The model**) of `planes` at texel coordinate `(x, y)`, in
/// `f64`: each channel's value. The luma codes around the point blended bilinearly, the chroma
/// pairs around half the point blended bilinearly, each index clamped to its plane, then Metal's
/// coefficients over `2^23`, each channel clamped into `[0, 1]`.
fn model(planes: &Planes, (x, y): (f64, f64)) -> [f64; 3] {
    let p = &planes.planes;
    let [cw, ch] = [W.div_ceil(2), H.div_ceil(2)];
    let blend = |(x, y): (f64, f64), [w, h]: [u32; 2], v: &dyn Fn(usize) -> f64| -> f64 {
        let (px, py) = (x - 0.5, y - 0.5);
        let (fx, fy) = (px.floor(), py.floor());
        let (kx, ky) = (px - fx, py - fy);
        let edge = |i: f64, n: u32| (i as i64).clamp(0, i64::from(n) - 1) as usize;
        let (x0, x1) = (edge(fx, w), edge(fx + 1.0, w));
        let (y0, y1) = (edge(fy, h), edge(fy + 1.0, h));
        let at = |x: usize, y: usize| v(y * w as usize + x);
        let low = at(x0, y0) * (1.0 - kx) + at(x1, y0) * kx;
        let high = at(x0, y1) * (1.0 - kx) + at(x1, y1) * kx;
        low * (1.0 - ky) + high * ky
    };
    let luma = blend((x, y), [W, H], &|i| f64::from(planes.luma[i]));
    let chroma = |c: usize| {
        blend((x / 2.0, y / 2.0), [cw, ch], &|i| {
            f64::from(planes.chroma[i][c])
        })
    };
    let bits = p.codes.bits;
    let scale = f64::from(1u32 << (10 - bits));
    let (l, cb, cr) = (luma * scale, chroma(0) * scale, chroma(1) * scale);
    let grid = f64::from(1u32 << 23);
    let term = match p.range {
        YCbCrRange::Video => 9576.0 * (l - 64.0) / grid,
        YCbCrRange::Full => l / (scale * f64::from((1u32 << bits) - 1)),
    };
    crate::bind::ycbcr_convert::coefficients(p.range, p.matrix).map(|(kb, kr)| {
        let c = f64::from(kb) * (cb - 512.0) + f64::from(kr) * (cr - 512.0);
        (term + c / grid).clamp(0.0, 1.0)
    })
}

/// The value of `half` bits `h` (a finite value).
fn half_value(h: u32) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let (e, m) = ((h >> 10) & 0x1f, f64::from(h & 0x3ff));
    sign * match e {
        0 => m * 2f64.powi(-24),
        _ => (1.0 + m / 1024.0) * 2f64.powi(e as i32 - 15),
    }
}

/// Under `YCbCrInPlacePolicy::Model` (the default) a linear sample of each format's random planes
/// under each conversion, through a constexpr sampler (normalized and pixel coordinates, with a
/// texel offset on edge axes) and a runtime one (clamp-to-edge, and repeat sampled as
/// clamp-to-edge), `float` and `half` results, on lavapipe and on MoltenVK, is the model's within
/// the `float` arithmetic's rounding (and a `half`'s), alpha 1, and its compile records the filter
/// note under the bound format (and the bound word's linear filter that addresses no border);
/// a runtime sampler's nearest word takes Metal's sample, bit for bit. A constexpr nearest or
/// border sampler takes Metal's sample under either value.
#[test]
fn in_place_linear_samples_are_the_model_s() {
    use crate::pipeline::artifact::PrecisionNote;
    let conversions = [
        (YCbCrRange::Video, YCbCrMatrix::Bt601),
        (YCbCrRange::Video, YCbCrMatrix::Bt709),
        (YCbCrRange::Video, YCbCrMatrix::Bt2020),
        (YCbCrRange::Full, YCbCrMatrix::Bt601),
        (YCbCrRange::Full, YCbCrMatrix::Bt709),
        (YCbCrRange::Full, YCbCrMatrix::Bt2020),
    ];
    let mut rng = Lcg(0x3d_e1_5a);
    let planes: Vec<Planes> = [PixelFormat::YCBCR84202P, PixelFormat::YCBCR104202P]
        .into_iter()
        .flat_map(|pf| conversions.map(|c| (pf, c)))
        .map(|(pf, c)| Planes::random(pf, c, &mut rng))
        .collect();
    // Footprint positions as `in_place_samples_are_metals` takes them.
    let positions: Vec<(i64, i64)> = (0..64)
        .map(|_| {
            let mut q = |n: u32| {
                let q = (rng.next() % (256 * u64::from(n) + 768)) as i64 - 384;
                match (q + 128) & 255 {
                    0..8 => q + 8,
                    248.. => q - 8,
                    _ => q,
                }
            };
            (q(W), q(H))
        })
        .collect();
    let coords = |pixel: bool| -> Vec<u8> {
        positions
            .iter()
            .flat_map(|&(qx, qy)| {
                let n = |q: i64, n: u32| match pixel {
                    true => (q + 128) as f32 / 256.0,
                    false => (q + 128) as f32 / (256 * i64::from(n)) as f32,
                };
                [n(qx, W).to_le_bytes(), n(qy, H).to_le_bytes()]
            })
            .flatten()
            .collect()
    };
    let edge = SamplerAddressMode::ClampToEdge;
    // Each case: its sampler, its coordinates in pixels, its offset, and whether the model takes
    // it (else Metal's nearest sample).
    let cases = [
        (
            "constexpr",
            Through::Constexpr(linear(1, false, 0)),
            false,
            (0, 0),
            true,
        ),
        (
            "constexpr pixel",
            Through::Constexpr(linear(1, true, 0)),
            true,
            (0, 0),
            true,
        ),
        (
            "constexpr offset",
            Through::Constexpr(linear(1, false, 0)),
            false,
            (2, -1),
            true,
        ),
        (
            "runtime",
            Through::Runtime(word(true, edge)),
            false,
            (0, 0),
            true,
        ),
        (
            "runtime repeat",
            Through::Runtime(word(true, SamplerAddressMode::Repeat)),
            false,
            (0, 0),
            true,
        ),
        (
            "runtime nearest",
            Through::Runtime(word(false, edge)),
            false,
            (0, 0),
            false,
        ),
    ];
    for (name, through, pixel, offset, takes) in cases {
        for half in [false, true] {
            let sampler = match through {
                Through::Constexpr(w) => Some(w),
                Through::Runtime(_) => None,
            };
            let form = Form::Plain {
                offset: (offset.0 as i32, offset.1 as i32),
            };
            let Some(bc) = assemble(name, &kernel_of(sampler, form, !half)) else {
                return;
            };
            let note = match half {
                true => PrecisionNote::HalfSampleFilter,
                false => PrecisionNote::FloatSampleFilter,
            };
            for (stem, p) in [("lavapipe", lavapipe()), ("moltenvk-m3", m3())] {
                let what = format!("{name} half={half} on {stem}");
                let a = compile_with(&bc, "k", &p, &inputs(true))
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
                let form = match through {
                    Through::Constexpr(_) => "sampler.ycbcr_in_place.model",
                    Through::Runtime(_) => "sampler.ycbcr_in_place.model.runtime",
                };
                assert!(
                    a.strategies.entries.contains_key(form),
                    "{what}: {:?}",
                    a.strategies.entries.keys().collect::<Vec<_>>()
                );
                let noted: Vec<_> = a
                    .specialized
                    .iter()
                    .filter(|s| s.note == note)
                    .flat_map(|s| &s.when)
                    .collect();
                let words = match through {
                    Through::Constexpr(_) => 0,
                    Through::Runtime(_) => 3,
                };
                assert!(
                    noted.len() == 1
                        && noted[0].all.len() == 1 + words
                        && noted[0]
                            .all
                            .iter()
                            .any(|t| t.key == SpecKey::BoundTexelFormat(TEXTURE)),
                    "{what}: {:?}",
                    a.specialized
                );
                let exact_form = compile_with(&bc, "k", &exact(stem), &inputs(true))
                    .unwrap_or_else(|e| panic!("{what}, exact: {e}"));
                assert!(
                    !exact_form.specialized.iter().any(|s| s.note == note)
                        && !exact_form.precision.contains(&note),
                    "{what}, exact: {:?}",
                    exact_form.specialized
                );
                let (plan, m) = typed_module_with(&bc, "k", &p, &inputs(true))
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                let slot = texel_buffer(&m);
                let n = positions.len();
                let size = if half { 8 } else { 16 };
                for planes in &planes {
                    let init = [coords(pixel), vec![0xab; size * n]];
                    let res = resources(&plan, &init, &planes.words);
                    let word = match through {
                        Through::Runtime(w) => Some(w),
                        Through::Constexpr(_) => None,
                    };
                    let res = with_word(res, &a, word);
                    let got = run(&m, &res, &planes.image(slot), n as u32, half);
                    let pl = &planes.planes;
                    let conversion = ((pl.range, pl.matrix), pl.codes.bits);
                    for (&(qx, qy), g) in positions.iter().zip(got) {
                        if !takes {
                            let want = metal(planes, false, (qx, qy), offset);
                            assert_eq!(
                                g,
                                result_bits(want, half, 2048),
                                "{what} {conversion:?} q {:?}",
                                (qx, qy)
                            );
                            continue;
                        }
                        let x = (qx + 128) as f64 / 256.0 + offset.0 as f64;
                        let y = (qy + 128) as f64 / 256.0 + offset.1 as f64;
                        let want = model(planes, (x, y));
                        let (value, within) = match half {
                            true => (half_value as fn(u32) -> f64, 3e-4),
                            false => ((|b| f64::from(f32::from_bits(b))) as fn(u32) -> f64, 2e-5),
                        };
                        for c in 0..3 {
                            let v = value(g[c]);
                            assert!(
                                (v - want[c]).abs() <= within,
                                "{what} {conversion:?} q {:?} channel {c}: {v} against {}",
                                (qx, qy),
                                want[c]
                            );
                        }
                        assert_eq!(value(g[3]), 1.0, "{what}: alpha");
                    }
                }
            }
        }
    }
    // A constexpr nearest sampler and a border one take Metal's sample under the model too.
    for (name, words, form) in [
        (
            "nearest",
            sampler_words(0, 0, 1, false, 0),
            "sampler.ycbcr_in_place.nearest",
        ),
        (
            "border",
            linear(0, false, 0),
            "sampler.ycbcr_in_place.linear",
        ),
    ] {
        let Some(bc) = assemble(name, &kernel(Some(words), Form::Float)) else {
            return;
        };
        let a = compile_with(&bc, "k", &lavapipe(), &inputs(true))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(
            a.strategies.entries.contains_key(form)
                && !a
                    .specialized
                    .iter()
                    .any(|s| s.note == PrecisionNote::FloatSampleFilter),
            "{name}: {:?} {:?}",
            a.strategies.entries.keys().collect::<Vec<_>>(),
            a.specialized
        );
    }
}

/// Metal's gather of component `c` of `planes` at footprint position `q` (`256 (x - 1/2)` per
/// axis, the texel coordinate `x`), clamped to the edge: component `c`'s steps of the taps
/// `(i, j + 1)`, `(i + 1, j + 1)`, `(i + 1, j)` and `(i, j)` (`i = ⌊q / 256⌋` per axis, each tap
/// clamped), in that order; alpha's step `2048` at every tap.
fn metal_gather(planes: &Planes, (qx, qy): (i64, i64), c: usize) -> [i64; 4] {
    let edge = |i: i64, n: u32| i.clamp(0, i64::from(n) - 1);
    let (i, j) = (qx >> 8, qy >> 8);
    let (i0, i1) = (edge(i, W), edge(i + 1, W));
    let (j0, j1) = (edge(j, H), edge(j + 1, H));
    let step = |x: i64, y: i64| match c {
        3 => 2048,
        _ => planes.texel(x, y)[c],
    };
    [step(i0, j1), step(i1, j1), step(i1, j0), step(i0, j0)]
}

/// Every gather of each component of each format's random planes under each conversion, through
/// constexpr and runtime samplers (either filter, which a gather does not follow; a runtime
/// sampler's repeat gathered as clamp-to-edge, as a 4:2:0 texture's), `float` and `half` results,
/// on lavapipe and on MoltenVK, is Metal's taps of the texels Metal reads: a quarter of the
/// positions past an edge.
#[test]
fn in_place_gathers_are_metals() {
    let conversions = [
        (YCbCrRange::Video, YCbCrMatrix::Bt601),
        (YCbCrRange::Full, YCbCrMatrix::Bt709),
        (YCbCrRange::Video, YCbCrMatrix::Bt2020),
    ];
    let mut rng = Lcg(0x6a_7e_42);
    let planes: Vec<Planes> = [PixelFormat::YCBCR84202P, PixelFormat::YCBCR104202P]
        .into_iter()
        .flat_map(|pf| conversions.map(|c| (pf, c)))
        .map(|(pf, c)| Planes::random(pf, c, &mut rng))
        .collect();
    // Footprint positions `q = 256 (x - 1/2)`, each axis within one and a half texels of the
    // texture, off the places where `⌊q / 256⌋` changes, which hold for any rounding of the
    // coordinate.
    let positions: Vec<(i64, i64)> = (0..64)
        .map(|_| {
            let mut q = |n: u32| {
                let q = (rng.next() % (256 * u64::from(n) + 768)) as i64 - 384;
                match q & 255 {
                    0..16 => q + 16,
                    240.. => q - 16,
                    _ => q,
                }
            };
            (q(W), q(H))
        })
        .collect();
    let out = |q: i64, n: u32| q < 0 || q >= 256 * (i64::from(n) - 1);
    let edges = positions
        .iter()
        .filter(|&&(qx, qy)| out(qx, W) || out(qy, H))
        .count();
    assert!(edges >= positions.len() / 4, "{edges} past an edge");
    let coords: Vec<u8> = positions
        .iter()
        .flat_map(|&(qx, qy)| {
            let n = |q: i64, n: u32| (q + 128) as f32 / (256 * i64::from(n)) as f32;
            [n(qx, W).to_le_bytes(), n(qy, H).to_le_bytes()]
        })
        .flatten()
        .collect();
    let edge = SamplerAddressMode::ClampToEdge;
    let cases = [
        (
            "constexpr linear gather",
            Through::Constexpr(linear(1, false, 0)),
            &[0, 1, 2, 3][..],
        ),
        (
            "constexpr nearest gather",
            Through::Constexpr(sampler_words(0, 0, 1, false, 0)),
            &[0, 2][..],
        ),
        (
            "runtime linear gather",
            Through::Runtime(word(true, edge)),
            &[0, 1, 2, 3][..],
        ),
        (
            "runtime repeat gather",
            Through::Runtime(word(false, SamplerAddressMode::Repeat)),
            &[0, 2][..],
        ),
    ];
    for (name, through, components) in cases {
        for &component in components {
            for half in [false, true] {
                let sampler = match through {
                    Through::Constexpr(w) => Some(w),
                    Through::Runtime(_) => None,
                };
                let body = kernel_of(sampler, Form::Gather { component }, !half);
                let Some(bc) = assemble(name, &body) else {
                    return;
                };
                for (stem, p) in [("lavapipe", lavapipe()), ("moltenvk-m3", m3())] {
                    let what = format!("{name} of {component} half={half} on {stem}");
                    let a = compile_with(&bc, "k", &p, &inputs(true))
                        .unwrap_or_else(|e| panic!("{what}: {e}"));
                    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
                    assert!(
                        a.strategies
                            .entries
                            .contains_key("sampler.ycbcr_in_place.gather"),
                        "{what}: {:?}",
                        a.strategies.entries.keys().collect::<Vec<_>>()
                    );
                    let (plan, m) = typed_module_with(&bc, "k", &p, &inputs(true))
                        .unwrap_or_else(|e| panic!("{what}: {e}"));
                    let slot = texel_buffer(&m);
                    let n = positions.len();
                    let size = if half { 8 } else { 16 };
                    for planes in &planes {
                        let init = [coords.clone(), vec![0xab; size * n]];
                        let res = resources(&plan, &init, &planes.words);
                        let word = match through {
                            Through::Runtime(w) => Some(w),
                            Through::Constexpr(_) => None,
                        };
                        let res = with_word(res, &a, word);
                        let got = run(&m, &res, &planes.image(slot), n as u32, half);
                        for (q, g) in positions.iter().zip(got) {
                            let want = metal_gather(planes, *q, component as usize);
                            let want = want.map(|k| match half {
                                true => u32::from(half_away(k, 2048)),
                                false => (k as f32 / 2048.0).to_bits(),
                            });
                            let p = &planes.planes;
                            assert_eq!(
                                g,
                                want,
                                "{what} {:?} {}-bit q {q:?}: {g:x?}",
                                (p.range, p.matrix),
                                p.codes.bits,
                            );
                        }
                    }
                }
            }
        }
    }
}

/// A function that samples a slot and gathers it through one runtime sampler, as a probe that
/// samples or gathers as each lane asks: its compile on MoltenVK on the M3 binds the texture's
/// image and notes the device's filter of the format there (`PrecisionNote::HalfSampleFilter` and
/// `FloatSampleFilter`, announced `ycbcr-filter-not-exact`), and states the in-place form with
/// the sampler, whose compile reads the planes for both and notes no pixel sampler's scaling:
/// under `YCbCrInPlacePolicy::Exact` Metal's filter and Metal's taps, noting no filter; under the
/// model (the default) the model's sample where the bound word filters linearly and addresses no
/// border, noting the filter there alone, and Metal's taps.
#[test]
fn a_slot_that_samples_and_gathers_reads_its_planes_in_place() {
    use crate::pipeline::artifact::PrecisionNote;
    for half in [false, true] {
        let what = format!("sample and gather half={half}");
        let Some(bc) = assemble(&what, &kernel_of(None, Form::SampleAndGather, !half)) else {
            return;
        };
        let filter = match half {
            true => PrecisionNote::HalfSampleFilter,
            false => PrecisionNote::FloatSampleFilter,
        };
        let p = m3();
        let notes = |a: &crate::library::FunctionArtifact| -> Vec<PrecisionNote> {
            let specialized = a.specialized.iter().map(|s| s.note);
            a.precision.iter().copied().chain(specialized).collect()
        };
        let image =
            compile_with(&bc, "k", &p, &inputs(false)).unwrap_or_else(|e| panic!("{what}: {e}"));
        assert!(
            notes(&image).contains(&filter),
            "{what}: {:?}",
            notes(&image)
        );
        assert!(
            image.facts.contains(&RuntimeFact::YCbCrInPlace {
                slot: TEXTURE,
                samplers: 1,
                device: false,
            }),
            "{what}: {:?}",
            image.facts
        );
        for (policy, p, sample) in [
            (
                "exact",
                exact("moltenvk-m3"),
                "sampler.ycbcr_in_place.runtime",
            ),
            ("model", m3(), "sampler.ycbcr_in_place.model.runtime"),
        ] {
            let what = format!("{what}, in place, {policy}");
            let a =
                compile_with(&bc, "k", &p, &inputs(true)).unwrap_or_else(|e| panic!("{what}: {e}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            for form in [sample, "sampler.ycbcr_in_place.gather"] {
                assert!(
                    a.strategies.entries.contains_key(form),
                    "{what}: {:?}",
                    a.strategies.entries.keys().collect::<Vec<_>>()
                );
            }
            assert!(
                !notes(&a).contains(&PrecisionNote::PixelSamplerScaling),
                "{what}"
            );
            let other = match half {
                true => PrecisionNote::FloatSampleFilter,
                false => PrecisionNote::HalfSampleFilter,
            };
            assert!(!notes(&a).contains(&other), "{what}: {:?}", notes(&a));
            // The model's note holds only under the bound format and word, never outright.
            assert!(!a.precision.contains(&filter), "{what}: {:?}", a.precision);
            let noted = a.specialized.iter().any(|s| s.note == filter);
            assert_eq!(noted, policy == "model", "{what}: {:?}", a.specialized);
        }
    }
}

/// A slot whose every access the form reads states `RuntimeFact::YCbCrInPlace` with the runtime
/// samplers it samples through, on a profile that takes the form (lavapipe, a CPU device; MoltenVK
/// on the M3, an integrated one) and on no other (the RTX 5080, a discrete one); a dynamic offset,
/// `min_lod_clamp` and a 3D texture keep it from the form, and a gather from the device's own
/// conversion alone (`device`). Its compile under the variant states the same fact, so its plan
/// names the same variant.
#[test]
fn a_slot_states_the_in_place_form_where_it_reads_only_it() {
    let fact = |samplers, device| RuntimeFact::YCbCrInPlace {
        slot: TEXTURE,
        samplers,
        device,
    };
    let reads = [
        (
            "linear",
            kernel(Some(linear(1, false, 0)), Form::Float),
            Some((0, true)),
        ),
        ("runtime", kernel(None, Form::Float), Some((1, true))),
        (
            "gather",
            kernel(Some(linear(1, false, 0)), Form::Gather { component: 2 }),
            Some((0, false)),
        ),
        (
            "runtime sample and gather",
            kernel(None, Form::SampleAndGather),
            Some((1, false)),
        ),
        (
            "dynamic offset",
            kernel(Some(linear(1, false, 0)), Form::DynamicOffset),
            None,
        ),
        (
            "min_lod_clamp",
            kernel(Some(linear(1, false, 0)), Form::MinLodClamp),
            None,
        ),
        (
            "3d",
            kernel(
                Some(linear(1, false, 0)),
                Form::ThreeD { offset: (0, 0, 0) },
            ),
            None,
        ),
    ];
    for (name, body, samplers) in reads {
        let Some(bc) = assemble(name, &body) else {
            return;
        };
        for (stem, takes) in [
            ("lavapipe", true),
            ("moltenvk-m3", true),
            ("nvidia-rtx5080-laptop", false),
        ] {
            let p = profile(stem, DevicePolicy::default()).expect("profile");
            assert_eq!(p.fields().ycbcr_in_place, takes, "{stem}");
            let what = format!("{name} on {stem}");
            let (plan, _) = typed_module_with(&bc, "k", &p, &inputs(false))
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            let stated = plan.facts.iter().find_map(|f| match f {
                RuntimeFact::YCbCrInPlace {
                    slot,
                    samplers,
                    device,
                } if *slot == TEXTURE => Some((*samplers, *device)),
                _ => None,
            });
            assert_eq!(stated, samplers.filter(|_| takes), "{what}");
            if let (Some((s, device)), true) = (samplers, takes) {
                let (variant, _) = typed_module_with(&bc, "k", &p, &inputs(true))
                    .unwrap_or_else(|e| panic!("{what}, in place: {e}"));
                assert!(variant.facts.contains(&fact(s, device)), "{what}, in place");
            }
        }
    }
}

/// Under the planes-image variant (`LoweringInputs::ycbcr_planes_sampled`) the slot is one
/// combined image sampler of the device's conversion (`ResourceKind::YCbCrPlanes`), every sample
/// is the device's `OpImageSampleExplicitLod` at level 0 with no offset operand
/// (`VUID-RuntimeSpirv-ConstOffset-10718`), and the module is valid, through a constexpr sampler
/// (nearest and linear, normalized and pixel coordinates, with a texel offset on edge axes) and a
/// runtime one. A sample through a border mode, or with an offset on an axis that repeats, states
/// the in-place fact without `device`, so no plan takes the variant for it.
#[test]
fn a_planes_image_slot_samples_through_the_device_s_conversion() {
    use crate::pipeline::interface::StageDescriptor;
    use crate::vk;
    let codes = ycbcr_format(PixelFormat::YCBCR84202P)
        .and_then(YCbCrCodes::of)
        .expect("codes");
    let planes = YCbCrPlanes::new(
        [8, 4],
        [0, 16],
        [64, 16],
        (YCbCrRange::Video, YCbCrMatrix::Bt709),
        codes,
    );
    let sampler = YCbCrInPlace::of(&planes).expect("in place").sampler();
    let device_inputs = || {
        let mut i = inputs(true);
        i.ycbcr_planes_sampled.insert(TEXTURE, sampler);
        i
    };
    let nearest = sampler_words(0, 0, 1, false, 0);
    let device_state = |plan: &Plan| {
        plan.facts.iter().find_map(|f| match f {
            RuntimeFact::YCbCrInPlace { slot, device, .. } if *slot == TEXTURE => Some(*device),
            _ => None,
        })
    };
    let taken = [
        ("device linear", Some(linear(1, false, 0)), (0, 0)),
        ("device nearest", Some(nearest), (0, 0)),
        ("device pixel", Some(linear(1, true, 0)), (0, 0)),
        ("device offset", Some(linear(1, false, 0)), (1, -2)),
        ("device runtime", None, (0, 0)),
    ];
    for (name, sampler_words, offset) in taken {
        for half in [false, true] {
            let form = Form::Plain { offset };
            let Some(bc) = assemble(name, &kernel_of(sampler_words, form, !half)) else {
                return;
            };
            for (stem, p) in [("lavapipe", lavapipe()), ("moltenvk-m3", m3())] {
                let what = format!("{name} half={half} on {stem}");
                let (plan, _) = typed_module_with(&bc, "k", &p, &inputs(true))
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                assert_eq!(device_state(&plan), Some(true), "{what}");
                let a = compile_with(&bc, "k", &p, &device_inputs())
                    .unwrap_or_else(|e| panic!("{what}, planes image: {e}"));
                let want = StageDescriptor::new(
                    ResourceSlot {
                        stage: Stage::Kernel,
                        kind: ResourceKind::YCbCrPlanes(sampler),
                        index: 0,
                    },
                    vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    1,
                );
                assert!(
                    a.interface.descriptors.contains(&want),
                    "{what}: {:?}",
                    a.interface.descriptors
                );
                assert!(
                    a.facts.iter().any(|f| matches!(
                        f,
                        RuntimeFact::YCbCrInPlace { slot, device: true, .. } if *slot == TEXTURE
                    )),
                    "{what}: the variant states the same fact"
                );
                let (_, words) =
                    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
                // Every sample: `OpImageSampleExplicitLod` (88) whose operands are `Lod` (0x2)
                // alone.
                let mut samples = 0;
                let mut at = 5;
                while let Some(&first) = words.get(at) {
                    let (count, op) = ((first >> 16) as usize, first & 0xffff);
                    assert!(count > 0, "{what}: a zero-length instruction");
                    if op == 88 {
                        samples += 1;
                        assert_eq!(words.get(at + 5), Some(&0x2), "{what}: operands");
                        assert_eq!(count, 7, "{what}: the Lod operand alone");
                    }
                    assert!(
                        !(87..=94).contains(&op) || op == 88,
                        "{what}: another sample opcode {op}"
                    );
                    at += count;
                }
                assert_eq!(samples, 1, "{what}");
            }
        }
    }
    // Through a border mode, or with an offset on an axis that repeats: no device sample.
    let kept = [
        ("device border", Some(linear(0, false, 0)), (0, 0)),
        ("device repeat offset", Some(linear(2, false, 0)), (1, 0)),
    ];
    for (name, sampler_words, offset) in kept {
        let Some(bc) = assemble(
            name,
            &kernel_of(sampler_words, Form::Plain { offset }, false),
        ) else {
            return;
        };
        let (plan, _) = typed_module_with(&bc, "k", &lavapipe(), &inputs(true))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(device_state(&plan), Some(false), "{name}");
    }
}

/// A sample of a slot through a runtime sampler slot, with a `half` and a `float` result, on
/// MoltenVK on the M3 notes the device's filter of a two-plane YCbCr format bound there
/// (`PrecisionNote::HalfSampleFilter`, `FloatSampleFilter`) only where the sampler bound at the
/// slot filters: each clause tests the texture slot's bound format
/// (`SpecKey::BoundTexelFormat`) and the sampler slot's bound word
/// (`SpecKey::BoundSamplerWord`, `lower::sampler::linear::filtering_word`), as the runtime reads
/// them at each draw or dispatch. One holds under each two-plane YCbCr format tried
/// (`YCBCR8_420_2P`, `YCBCR10_422_2P`, `YCBCR8_444_2P_sRGB`) for a linear minification,
/// magnification or mip filter and for anisotropy; none for a nearest sampler with no mip filter
/// or a nearest one, whose sample reads the one texel the view holds, Metal's read; and none under
/// another format (`RGBA16Float`, the code-reading `RGB8_420_2P`) whatever the word. lavapipe,
/// which samples natively, records neither note.
#[test]
fn a_runtime_sampler_s_filter_note_holds_only_where_the_bound_word_filters() {
    use crate::bind::SamplerDescriptor;
    use crate::metal::texel_format::texel_format;
    use crate::metal::{SamplerMinMagFilter as F, SamplerMipFilter as Mip};
    use crate::pipeline::artifact::{PrecisionNote, SpecClause};
    let sampler = SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Sampler,
        index: 0,
    };
    let desc = |min, mag, mip, anisotropy| {
        let mut d = SamplerDescriptor::new();
        (d.min_filter, d.mag_filter, d.mip_filter) = (min, mag, mip);
        d.max_anisotropy = anisotropy;
        d
    };
    // Each bound sampler, and whether a sample through it filters.
    let words = [
        (
            "nearest",
            desc(F::Nearest, F::Nearest, Mip::NotMipmapped, 1),
            false,
        ),
        (
            "nearest mip filter",
            desc(F::Nearest, F::Nearest, Mip::Nearest, 1),
            false,
        ),
        (
            "linear",
            desc(F::Linear, F::Linear, Mip::NotMipmapped, 1),
            true,
        ),
        (
            "linear min",
            desc(F::Linear, F::Nearest, Mip::NotMipmapped, 1),
            true,
        ),
        (
            "linear mag",
            desc(F::Nearest, F::Linear, Mip::NotMipmapped, 1),
            true,
        ),
        (
            "linear mip filter",
            desc(F::Nearest, F::Nearest, Mip::Linear, 1),
            true,
        ),
        (
            "anisotropic",
            desc(F::Nearest, F::Nearest, Mip::NotMipmapped, 16),
            true,
        ),
    ];
    let formats = [
        (PixelFormat::YCBCR84202P, true),
        (PixelFormat::YCBCR104222P, true),
        (PixelFormat::YCBCR84442PSRGB, true),
        (PixelFormat::RGBA16Float, false),
        (PixelFormat::RGB84202P, false),
    ];
    let holds = |clause: &SpecClause, value: &dyn Fn(SpecKey) -> Option<u32>| {
        clause.all.iter().all(|t| {
            let v = value(t.key).unwrap_or_else(|| panic!("an unvalued test {t:?}"));
            ((v & t.mask) == t.bits) == t.equal
        })
    };
    for float in [false, true] {
        let what = format!("a runtime sampler's sample, float {float}");
        let Some(bc) = assemble(
            &what,
            &kernel_of(None, Form::Plain { offset: (0, 0) }, float),
        ) else {
            return;
        };
        let note = match float {
            true => PrecisionNote::FloatSampleFilter,
            false => PrecisionNote::HalfSampleFilter,
        };
        let native = compile_with(&bc, "k", &lavapipe(), &inputs(false))
            .unwrap_or_else(|e| panic!("{what} on lavapipe: {e}"));
        assert!(
            !native.precision.contains(&note) && native.specialized.iter().all(|s| s.note != note),
            "{what} on lavapipe: {:?}",
            native.precision
        );
        let a =
            compile_with(&bc, "k", &m3(), &inputs(false)).unwrap_or_else(|e| panic!("{what}: {e}"));
        let noted: Vec<&SpecClause> = a
            .specialized
            .iter()
            .filter(|s| s.note == note)
            .flat_map(|s| &s.when)
            .collect();
        assert!(!noted.is_empty(), "{what}: {:?}", a.specialized);
        for clause in &noted {
            let keys: Vec<SpecKey> = clause.all.iter().map(|t| t.key).collect();
            assert!(
                keys.contains(&SpecKey::BoundTexelFormat(TEXTURE))
                    && keys.contains(&SpecKey::BoundSamplerWord(sampler)),
                "{what}: {clause:?}"
            );
        }
        for (name, d, filters) in &words {
            let word = SamplerWord::of(d).0;
            for (pf, ycbcr) in formats {
                let value = |key: SpecKey| match key {
                    SpecKey::BoundTexelFormat(t) if t == TEXTURE => Some(texel_format(pf).0),
                    SpecKey::BoundSamplerWord(s) if s == sampler => Some(word),
                    _ => None,
                };
                let held = noted.iter().any(|c| holds(c, &value));
                assert_eq!(held, *filters && ycbcr, "{what}: {name} over {pf:?}");
            }
        }
    }
}
