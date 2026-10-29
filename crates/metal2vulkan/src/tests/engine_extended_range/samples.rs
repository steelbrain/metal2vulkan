//! Samples of an extended-range texture: under the slot's variant the function fetches the raw
//! texels and filters them as Metal does, on every device.
//!
//! - On the CPU interpreter, 192 linear 2D samples Metal took, 16 per format and result type, are
//!   Metal's bytes, through a constexpr sampler and through a runtime one: 128 of the one-plane
//!   formats (an Apple M3 Max on macOS 26.7, byte-identical to the Apple M4's probe),
//!   whose device lane `runtime_lane::extended_range_samples` compares the same samples on each
//!   venue, and 64 of the two-plane formats (an Apple M4 on macOS 27.0.1: footprints inside the
//!   texture of its linear clamp-to-edge samples), whose device lane
//!   `runtime_lane::two_plane_samples` replays every sample of that probe.
//! - Under every address mode, each border color, a constant offset, and pixel coordinates,
//!   through constexpr and runtime samplers (a runtime sampler's `MirrorClampToEdge` too, each
//!   tap mirrored once and clamped), nearest samples are the decoded texel at Metal's position and
//!   linear samples the model's (the footprint of `lower::sampler::linear`, the format's exact sum:
//!   the module docs of `lower::sampler::extended`).
//! - What is not built is refused, `NotImplemented(Format)`: by the lowering (here a two-plane
//!   form's gather, a linear mip filter of a two-plane form, and mixed constexpr filters), and by
//!   the binder (a runtime sampler whose word states mixed filters).
//! - A swizzled texture's samples and gathers are the unswizzled texture's, swizzled, as plain
//!   Metal answers.
//! - The binder joins a sampled slot bound in such a format to the variant, whose plan views the
//!   slot's own image in the raw format and binds no sampler.

use super::*;
use crate::bind::{BoundSampler, SamplerDescriptor};
use crate::error::Unsupported;
use crate::metal::format::extended_srgb::SRGB_DECODE_STEPS;
use crate::metal::sampler::SamplerWord;
use crate::metal::{SamplerAddressMode, SamplerBorderColor, SamplerMinMagFilter};

/// Metal's linear samples, the files `xr_samples.tsv` and `xr_plane_samples.tsv` beside the owned
/// fixture `xr_samples.metal`, one per line: the raw format (550 `RGB10A8_2P_XR10`, 551 its `_sRGB`
/// twin, 552 `BGRA10_XR`, 553 its twin, 554 `BGR10_XR`, 555 its twin), the result (`f` `float`, `h`
/// `half`), the fractions `kx` and `ky` in 1/256 steps, the codes (red, green, blue, alpha; a
/// two-plane texel's alpha its byte) of the taps `(i, j)`, `(i + 1, j)`, `(i, j + 1)`, and `(i + 1,
/// j + 1)`, and the result's lanes as bits.
fn metal_linear() -> String {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/engine");
    ["xr_samples.tsv", "xr_plane_samples.tsv"]
        .into_iter()
        .map(|name| {
            let path = dir.join(name);
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        })
        .collect()
}

pub(super) use metal2vulkan_devtools::xr_bicubic::filtered_close;
/// Whether lane `got` (bits; `half` bits where `half`) is the device's answer for `want`: a NaN
/// for a NaN (its bits the device's), a `half` denormal or its signed zero (a device's quantize
/// may flush it), else `want` itself.
fn native_lane(got: u32, want: u32, half: bool) -> bool {
    match half {
        false if f32::from_bits(want).is_nan() => f32::from_bits(got).is_nan(),
        false => got == want,
        true if want & 0x7FFF > 0x7C00 => got & 0x7FFF > 0x7C00,
        true if want & 0x7C00 == 0 => got == want || got == want & 0x8000,
        true => got == want,
    }
}

/// Whether the lanes `got` (bits) lie within `DevicePackConversion`'s bound of Metal's `want`:
/// each at most two units in the last place apart (a decode's product one `f32` ulp from Metal's
/// quotient, or a `half` narrowed from it, carried through a filter's sum).
pub(super) fn near_metals(got: &[u32], want: &[u32]) -> bool {
    got.len() == want.len() && got.iter().zip(want).all(|(&g, &w)| g.abs_diff(w) <= 2)
}

/// The format of a raw format number of [`metal_linear`].
pub(super) fn form_of(raw: &str) -> ExtendedRange {
    match raw {
        "550" => ExtendedRange::Planes { srgb: false },
        "551" => ExtendedRange::Planes { srgb: true },
        "552" => ExtendedRange::Bgra10 { srgb: false },
        "553" => ExtendedRange::Bgra10 { srgb: true },
        "554" => ExtendedRange::Bgr10 { srgb: false },
        "555" => ExtendedRange::Bgr10 { srgb: true },
        _ => panic!("raw format {raw}"),
    }
}

/// The raw texel of codes `[r, g, b, a]` in `form`, as the view of its raw format reads it (a
/// two-plane texel's `a` its alpha byte).
pub(super) fn raw_of(form: ExtendedRange, [r, g, b, a]: [u32; 4]) -> [u32; 4] {
    match form {
        ExtendedRange::Bgr10 { .. } => [r << 20 | g << 10 | b, 0, 0, 1],
        ExtendedRange::Bgra10 { .. } => [b << 6, g << 6, r << 6, a << 6],
        ExtendedRange::Planes { .. } => [r << 20 | g << 10 | b, a, 0, 1],
        ExtendedRange::R10X6 | ExtendedRange::R10X6Srgb => [r << 6, 0, 0, 1],
        ExtendedRange::Rg10X12 | ExtendedRange::Rg10X12Srgb => [r << 6, g << 6, 0, 1],
        ExtendedRange::R16Bfloat => [r, 0, 0, 1],
        ExtendedRange::Rg16Bfloat => [r, g, 0, 1],
        ExtendedRange::Rgba16Bfloat => [r, g, b, a],
        ExtendedRange::Rgb10A2Srgb => [a << 30 | b << 20 | g << 10 | r, 0, 0, 1],
    }
}

/// How a kernel's sample reaches its sampler.
#[derive(Clone, Copy, Debug)]
pub(super) enum Through {
    /// A constexpr sampler of these AIR state words ([`words`]).
    Constexpr(u64),
    /// The runtime `[[sampler(0)]]`.
    Runtime,
}

/// The AIR state words of a constexpr sampler: linear minification and magnification where
/// `linear`, else nearest, no mip filter, address `code` on every axis (0 clamp to zero, or to
/// the border of another color than transparent black; 1 clamp to edge; 2 repeat; 3 mirrored
/// repeat), pixel coordinates where `pixel`, and border color `border` (0 transparent black, 1
/// opaque black, 2 opaque white).
pub(super) fn words(linear: bool, code: u64, pixel: bool, border: u64) -> u64 {
    const BASE: u64 = 0x007b_ff00_0008_0000;
    let f = u64::from(linear);
    BASE | code | code << 3 | code << 6 | f << 9 | f << 11 | u64::from(pixel) << 15 | border << 56
}

/// The word of a runtime sampler: linear minification and magnification where `linear`, else
/// nearest, no mip filter, address mode `mode` on every axis, pixel coordinates where `pixel`, and
/// border color `border`.
fn word(linear: bool, mode: SamplerAddressMode, pixel: bool, border: SamplerBorderColor) -> u32 {
    let mut w = border.as_raw() << SamplerWord::BORDER_SHIFT;
    if linear {
        w |= SamplerWord::LINEAR;
    }
    if pixel {
        w |= SamplerWord::PIXEL_COORDINATES;
    }
    for shift in SamplerWord::ADDRESS_SHIFT {
        w |= mode.as_raw() << shift;
    }
    w
}

/// A kernel sampling `texture2d<T, sample>` slot 0 (`half` texels, or `float`) through `through`
/// at `float2` buffer 0's lane `tid` with constant texel offset `offset`, storing the texel to
/// buffer 1; or with `gather`, gathering its red channel there.
pub(super) fn sample_kernel(
    through: Through,
    half: bool,
    offset: (i32, i32),
    gather: bool,
) -> String {
    let (texel, vec4, size, suffix, out_ty) = if half {
        ("half", "<4 x half>", 8, "v4f16", "half4")
    } else {
        ("float", "<4 x float>", 16, "v4f32", "float4")
    };
    let (state, states, state_md, s, params, sampler_md) = match through {
        Through::Constexpr(w) => (
            format!(
                "@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {w}, i64 0], align 8\n\n"
            ),
            "!air.sampler_states = !{!6}\n",
            "!6 = !{!\"air.sampler_state\", ptr addrspace(2) @__air_sampler_state}\n",
            "@__air_sampler_state",
            "",
            "",
        ),
        Through::Runtime => (
            String::new(),
            "",
            "",
            "%s",
            ", ptr addrspace(2) %s",
            "!7 = !{i32 3, !\"air.sampler\", !\"air.location_index\", i32 0, i32 1, !\"air.arg_type_name\", !\"sampler\", !\"air.arg_name\", !\"s\"}\n",
        ),
    };
    let (tid_index, args_md) = match through {
        Through::Constexpr(_) => (3, "!3, !4, !5, !8"),
        Through::Runtime => (4, "!3, !4, !5, !7, !8"),
    };
    let (ox, oy) = offset;
    let (call, decl) = match gather {
        false => (
            format!("@air.sample_texture_2d.{suffix}(ptr addrspace(1) %t, ptr addrspace(2) {s}, <2 x float> %uv, i1 true, <2 x i32> <i32 {ox}, i32 {oy}>, i1 false, float 0.0, float 0.0, i32 0)"),
            format!("@air.sample_texture_2d.{suffix}(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)"),
        ),
        true => (
            format!("@air.gather_texture_2d.{suffix}(ptr addrspace(1) %t, ptr addrspace(2) {s}, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i32 0, i32 0)"),
            format!("@air.gather_texture_2d.{suffix}(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i32, i32)"),
        ),
    };
    format!(
        r#"{state}define void @k(ptr addrspace(1) %t, ptr addrspace(1) %c, ptr addrspace(1) %out{params}, i32 %tid) {{
  %i = zext i32 %tid to i64
  %p = getelementptr inbounds <2 x float>, ptr addrspace(1) %c, i64 %i
  %uv = load <2 x float>, ptr addrspace(1) %p, align 8
  %r = call {{ {vec4}, i8 }} {call}
  %v = extractvalue {{ {vec4}, i8 }} %r, 0
  %q = getelementptr inbounds {vec4}, ptr addrspace(1) %out, i64 %i
  store {vec4} %v, ptr addrspace(1) %q, align {size}
  ret void
}}

declare {{ {vec4}, i8 }} {decl}

!air.kernel = !{{!0}}
{states}!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{{args_md}}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<{texel}, sample>", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"float2", !"air.arg_name", !"c"}}
!5 = !{{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 {size}, !"air.arg_type_align_size", i32 {size}, !"air.arg_type_name", !"{out_ty}", !"air.arg_name", !"out"}}
{state_md}{sampler_md}!8 = !{{i32 {tid_index}, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}}
"#
    )
}

/// A texture of `width` texels per row holding the codes `codes` (red, green, blue, alpha; row
/// by row).
pub(super) struct Texels<'a> {
    pub(super) width: u32,
    pub(super) codes: &'a [[u32; 4]],
}

/// The samples kernel `bc` takes on `p` of `tex` in `form` at `coords`, through a runtime
/// sampler of word `word` where it has one: each result's lanes as bits (`half` lanes where
/// `half`). The compile is validated, and the plan's estimate bounds it.
fn sample(
    bc: &[u8],
    p: &crate::device::profile::DeviceProfile,
    texture: (ExtendedRange, &Texels<'_>),
    at: (&[[f32; 2]], bool, Option<u32>),
    what: &str,
) -> Vec<[u32; 4]> {
    sample_swizzled(bc, p, texture, at, None, what)
}

/// [`sample`] of a texture swizzled by `swizzle` (its word) where it has one: the slot joins
/// `LoweringInputs::swizzled_storage` and its `TextureSwizzle` specialization is the word.
fn sample_swizzled(
    bc: &[u8],
    p: &crate::device::profile::DeviceProfile,
    (form, tex): (ExtendedRange, &Texels<'_>),
    (coords, half, word): (&[[f32; 2]], bool, Option<u32>),
    swizzle: Option<u32>,
    what: &str,
) -> Vec<[u32; 4]> {
    let mut inputs = LowerInputs::new();
    inputs.extended_range.insert(SOURCE, form);
    if swizzle.is_some() {
        inputs.swizzled_storage.insert(SOURCE);
    }
    validate(bc, &inputs, p, what);
    let (plan, m) =
        typed_module_with(bc, "k", p, &inputs).unwrap_or_else(|e| panic!("{what}: {e}"));
    run_samples(&plan, &m, (form, tex), (coords, half, word), swizzle, what)
}

/// The samples the module `m` of `plan` takes of `tex` in `form` at `coords` on the CPU
/// interpreter, its `float2` coordinates at buffer 0 and results at buffer 1, one thread each,
/// through a runtime sampler of word `word` where it has one and under the `TextureSwizzle` word
/// `swizzle` where it has one: each result's lanes as bits (`half` lanes where `half`). Both
/// scheduling orders must agree.
pub(super) fn run_samples(
    plan: &Plan,
    m: &TypedModule,
    (form, tex): (ExtendedRange, &Texels<'_>),
    (coords, half, word): (&[[f32; 2]], bool, Option<u32>),
    swizzle: Option<u32>,
    what: &str,
) -> Vec<[u32; 4]> {
    let mut spec = BTreeMap::new();
    for axis in [Axis::X, Axis::Y, Axis::Z] {
        spec.insert(SpecKey::LocalSize(axis), 1);
    }
    spec.insert(SpecKey::FlatLocalSize, 1);
    if let Some(w) = swizzle {
        spec.insert(SpecKey::TextureSwizzle(SOURCE), u64::from(w));
    }
    if let Some(word) = word {
        let s = plan
            .families
            .sampler
            .slots
            .get(&0)
            .expect("the runtime sampler");
        let fields = s.fields();
        spec.insert(
            SpecKey::SamplerState {
                slot: s.slot,
                fields,
            },
            u64::from(word & fields),
        );
    }
    let push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    for f in &plan.push.fields {
        match (f.field, f.loc) {
            (PushField::SlotBindOffset(_) | PushField::DispatchRegion, PushLoc::Push { .. }) => {}
            // A bound value the function reads as data, which the interpreter writes from the
            // specialization it is given.
            (PushField::Specialized(_), PushLoc::Push { .. }) => {}
            other => panic!("{what}: unexpected push field {other:?}"),
        }
    }
    let size = if half { 8 } else { 16 };
    let buffer = |index: u32, bytes: Vec<u8>| Buffer {
        slot: Some(ResourceSlot {
            stage: Stage::Kernel,
            kind: ResourceKind::Buffer,
            index,
        }),
        address: None,
        bytes,
    };
    let input = coords
        .iter()
        .flat_map(|c| c.iter().flat_map(|x| x.to_le_bytes()))
        .collect();
    let res = Resources {
        buffers: vec![buffer(0, input), buffer(1, vec![0xab; size * coords.len()])],
        push,
        spec,
    };
    let slot = ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Texture,
        index: 0,
    };
    let height = tex.codes.len() as u32 / tex.width;
    let mut img = Image::new(slot, [tex.width, height, 1], 1, 1, 1);
    for (n, c) in (0u32..).zip(tex.codes) {
        let at = img
            .index(0, 0, 0, [n % tex.width, n / tex.width, 0])
            .expect("texel");
        img.texels[at] = raw_of(form, *c);
    }
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            workgroups: [coords.len() as u32, 1, 1],
            order,
            ..Config::default()
        };
        let (o, _) = run_with_images(m, &cfg, res.clone(), vec![img.clone()])
            .unwrap_or_else(|e| panic!("{what}: interpreter: {e:?}"));
        outs.push(o.buffers.into_iter().nth(1).expect("out"));
    }
    assert_eq!(outs[0], outs[1], "{what}: the scheduling orders disagree");
    (0..coords.len() as u32)
        .map(|n| std::array::from_fn(|c| lane(&outs[0], half, n, c)))
        .collect()
}

/// Each of Metal's linear samples ([`metal_linear`]) is Metal's bytes, through a constexpr
/// linear clamp-to-edge sampler and through a runtime sampler of that state: row `k`'s taps are
/// texels `2k` and `2k + 1` of a 32 x 2 texture, sampled at the normalized coordinate whose
/// fractions are the row's.
#[test]
fn linear_samples_are_metals_bytes() {
    let edge = word(
        true,
        SamplerAddressMode::ClampToEdge,
        false,
        SamplerBorderColor::TransparentBlack,
    );
    let metal = metal_linear();
    for half in [false, true] {
        for (through, w) in [
            (Through::Constexpr(words(true, 1, false, 0)), None),
            (Through::Runtime, Some(edge)),
        ] {
            let Some(bc) = assemble("metal", &sample_kernel(through, half, (0, 0), false)) else {
                return;
            };
            // The `_sRGB` twins' (551, 553, 555) taps take the device's `Pow`, which the
            // interpreter does not model: their samples are the device lanes'.
            for raw in ["550", "552", "554"] {
                let form = form_of(raw);
                let kind = if half { "h" } else { "f" };
                let rows: Vec<Vec<&str>> = metal
                    .lines()
                    .map(|l| l.split(' ').collect::<Vec<_>>())
                    .filter(|r| r[0] == raw && r[1] == kind)
                    .collect();
                assert_eq!(rows.len(), 16);
                let parse = |s: &str| -> [u32; 4] {
                    let v: Vec<u32> = s.split(',').map(|x| x.parse().expect("code")).collect();
                    [v[0], v[1], v[2], v[3]]
                };
                let mut codes = vec![[0u32; 4]; 64];
                let mut coords = Vec::with_capacity(16);
                for (k, r) in rows.iter().enumerate() {
                    let (kx, ky): (u32, u32) =
                        (r[2].parse().expect("kx"), r[3].parse().expect("ky"));
                    for (t, (dx, dy)) in [(0, 0), (1, 0), (0, 1), (1, 1)].into_iter().enumerate() {
                        codes[dy * 32 + 2 * k + dx] = parse(r[4 + t]);
                    }
                    coords.push([
                        (512 * k as u32 + 128 + kx) as f32 / (256.0 * 32.0),
                        (128 + ky) as f32 / 512.0,
                    ]);
                }
                let tex = Texels {
                    width: 32,
                    codes: &codes,
                };
                for (stem, p) in profiles() {
                    let what = format!("{raw} {kind} {through:?} on {stem}");
                    let got = sample(&bc, &p, (form, &tex), (&coords, half, w), &what);
                    for (r, g) in rows.iter().zip(got) {
                        let want: Vec<u32> = r[8]
                            .split(',')
                            .map(|x| u32::from_str_radix(x, 16).expect("hex"))
                            .collect();
                        assert!(
                            filtered_close(&g, &want, half, 0.0),
                            "{what}: row {r:?}: {g:x?}"
                        );
                    }
                }
            }
        }
    }
}

// ---- The reference model ----------------------------------------------------------------------

/// How an axis addresses a tap index outside the level ([`address`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Edge,
    Repeat,
    Mirror,
    /// `MirrorClampToEdge`, which only a runtime sampler states.
    MirrorClamp,
    Border,
}

/// The texel of index `j` on an axis of `n` texels under `mode`, `None` outside a border axis.
fn address(mode: Mode, j: i64, n: i64) -> Option<i64> {
    match mode {
        Mode::Edge => Some(j.clamp(0, n - 1)),
        Mode::Repeat => Some(j.rem_euclid(n)),
        Mode::Mirror => {
            let m = if j < 0 { -1 - j } else { j } % (2 * n);
            Some(if m >= n { 2 * n - 1 - m } else { m })
        }
        // Mirrored once about the level's start, then clamped.
        Mode::MirrorClamp => Some(if j < 0 { -1 - j } else { j }.min(n - 1)),
        Mode::Border => (0..n).contains(&j).then_some(j),
    }
}

/// The grid code of code `c` of channel `ch` of `form` (red 0, alpha 3): `16 (c - 384)` for a
/// color lane (an `_sRGB` twin's `±` step), `8 (c - 384)` for alpha, its code clamped to `[384,
/// 894]` (`BGR10_XR`'s alpha 1, `8 · 510`), and `16 c` for a two-plane texel's alpha byte `c`
/// (on the same grid, `16 · 255`).
fn grid_code(form: ExtendedRange, ch: usize, c: u32) -> i64 {
    if ch == 3 {
        return match form {
            ExtendedRange::Bgr10 { .. } => 4080,
            ExtendedRange::Planes { .. } => 16 * i64::from(c),
            _ => 8 * (i64::from(c.clamp(384, 894)) - 384),
        };
    }
    let d = i64::from(c) - 384;
    match form.srgb() {
        true => d.signum() * i64::from(SRGB_DECODE_STEPS[d.unsigned_abs() as usize]),
        false => 16 * d,
    }
}

/// The grid of channel `ch` of `form`: the code of 1.
fn grid(form: ExtendedRange, ch: usize) -> i64 {
    match (ch, form.srgb()) {
        (3, _) => 4080,
        (_, true) => 8192,
        (_, false) => 8160,
    }
}

/// `x` (nonzero, a normal `half` once rounded) rounded to `half` with ties to even.
fn half_even(x: f64) -> u16 {
    if x == 0.0 {
        return 0;
    }
    let sign = if x < 0.0 { 0x8000 } else { 0 };
    let a = x.abs();
    let mut e = a.log2().floor() as i32;
    while 2f64.powi(e) > a {
        e -= 1;
    }
    while 2f64.powi(e + 1) <= a {
        e += 1;
    }
    let m = (a / 2f64.powi(e - 10)).round_ties_even() as u32;
    let (e, m) = if m == 2048 { (e + 1, 1024) } else { (e, m) };
    assert!((-14..=15).contains(&e), "{x} is no normal half");
    sign | ((e + 15) as u16) << 10 | (m - 1024) as u16
}

/// Metal's linear value of channel `ch` of `form` over the taps' grid codes `taps` (`(i, j)`,
/// `(i + 1, j)`, `(i, j + 1)`, `(i + 1, j + 1)`) at fractions `kx` and `ky`, as bits: each term
/// of the product weights truncated onto `2^-s` of a step (`s` 7 for a color lane, 4 for
/// alpha), the step rounded half up, the value `n / D` rounded to `f32`, or to `half` with ties
/// to even, an `_sRGB` color lane's from its `f32` with ties away from zero.
fn linear_bits(
    form: ExtendedRange,
    ch: usize,
    taps: [i64; 4],
    (kx, ky): (i64, i64),
    half: bool,
) -> u32 {
    let s = if ch == 3 { 4 } else { 7 };
    let w = [
        (256 - kx) * (256 - ky),
        kx * (256 - ky),
        (256 - kx) * ky,
        kx * ky,
    ];
    let total: i64 = taps.iter().zip(w).map(|(c, w)| (c * w) >> s).sum();
    let n = (total + (1 << (15 - s))) >> (16 - s);
    let x = n as f64 / grid(form, ch) as f64;
    match half {
        false => (x as f32).to_bits(),
        true if form.srgb() && ch < 3 => u32::from(half_bits(x as f32)),
        true => u32::from(half_even(x)),
    }
}

/// The border texel of `form` for border color `b` (0 transparent black, 1 opaque black, 2 opaque
/// white): the color, `BGR10_XR`'s alpha 1 (a two-plane format's the color's).
fn border_texel(form: ExtendedRange, b: u32) -> [bool; 4] {
    let alpha = b > 0 || matches!(form, ExtendedRange::Bgr10 { .. });
    [b == 2, b == 2, b == 2, alpha]
}

/// One sample the model answers: the axes' mode and border color, the texel offset, and the
/// texel-space position as `q = 256 x - 128` on each axis.
#[derive(Clone, Copy, Debug)]
struct Case {
    mode: Mode,
    border: u32,
    offset: (i64, i64),
    q: (i64, i64),
}

/// Metal's sample of `case` of an `n` x `n` texture of `form` holding `codes`, linear or
/// nearest, as bits.
fn model(
    form: ExtendedRange,
    (n, codes): (i64, &[[u32; 4]]),
    case: Case,
    linear: bool,
    half: bool,
) -> [u32; 4] {
    let bt = border_texel(form, case.border);
    if !linear {
        // The texel Metal's position `x = (q + 128) / 256` floors to.
        let x = address(
            case.mode,
            (case.q.0 + 128).div_euclid(256) + case.offset.0,
            n,
        );
        let y = address(
            case.mode,
            (case.q.1 + 128).div_euclid(256) + case.offset.1,
            n,
        );
        return std::array::from_fn(|ch| {
            let v = match (x, y) {
                (Some(x), Some(y)) => value(form, ch, codes[(y * n + x) as usize][ch]),
                _ => f32::from(u8::from(bt[ch])),
            };
            match half {
                true => u32::from(half_bits(v)),
                false => v.to_bits(),
            }
        });
    }
    // Each axis's taps and fraction, a merged pair's fraction 0.
    let axis = |q: i64, off: i64| {
        let i = q.div_euclid(256) + off;
        let t = [address(case.mode, i, n), address(case.mode, i + 1, n)];
        let k = if t[0] == t[1] { 0 } else { q.rem_euclid(256) };
        (k, t)
    };
    let (kx, [x0, x1]) = axis(case.q.0, case.offset.0);
    let (ky, [y0, y1]) = axis(case.q.1, case.offset.1);
    std::array::from_fn(|ch| {
        let code = |x: Option<i64>, y: Option<i64>| match (x, y) {
            (Some(x), Some(y)) => grid_code(form, ch, codes[(y * n + x) as usize][ch]),
            _ => i64::from(bt[ch]) * grid(form, ch),
        };
        let taps = [code(x0, y0), code(x1, y0), code(x0, y1), code(x1, y1)];
        linear_bits(form, ch, taps, (kx, ky), half)
    })
}

/// A deterministic source: a 64-bit LCG.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

/// Nearest and linear samples of a 4 x 4 texture of each format under each address mode and
/// border color, a constant offset, and pixel coordinates, through a constexpr sampler, with
/// `float` and `half` results, are the model's ([`model`]) at 96 positions each, around and past
/// the edges. (A runtime sampler's `MirrorClampToEdge`, which no constexpr sampler states, is
/// [`runtime_samples_are_the_models`]'s alone.)
#[test]
fn constexpr_samples_are_the_models() {
    models(false);
}

/// [`constexpr_samples_are_the_models`] through the runtime sampler of each state's word.
#[test]
fn runtime_samples_are_the_models() {
    models(true);
}

/// The samples of [`constexpr_samples_are_the_models`], through the runtime sampler where
/// `runtime`.
fn models(runtime: bool) {
    use SamplerAddressMode as A;
    use SamplerBorderColor as B;
    const N: i64 = 4;
    let mut rng = Lcg(0x5eed);
    let codes: Vec<[u32; 4]> = (0..N * N)
        .map(|_| std::array::from_fn(|_| rng.next() % 1024))
        .collect();
    let q: Vec<(i64, i64)> = (0..96)
        .map(|_| {
            let mut c = || i64::from(rng.next() % (256 * N as u32 + 256)) - 256;
            (c(), c())
        })
        .collect();
    // Each state: its model's mode, border color, and offset, the constexpr words' address code
    // and border (none: a runtime sampler's alone), the runtime word's mode and border, and pixel
    // coordinates.
    let states = [
        (
            Mode::Edge,
            0,
            (0, 0),
            Some((1, 0)),
            (A::ClampToEdge, B::TransparentBlack),
            false,
        ),
        (
            Mode::Edge,
            0,
            (1, -2),
            Some((1, 0)),
            (A::ClampToEdge, B::TransparentBlack),
            false,
        ),
        (
            Mode::Edge,
            0,
            (0, 0),
            Some((1, 0)),
            (A::ClampToEdge, B::TransparentBlack),
            true,
        ),
        (
            Mode::Repeat,
            0,
            (0, 0),
            Some((2, 0)),
            (A::Repeat, B::TransparentBlack),
            false,
        ),
        (
            Mode::Mirror,
            0,
            (0, 0),
            Some((3, 0)),
            (A::MirrorRepeat, B::TransparentBlack),
            false,
        ),
        (
            Mode::Border,
            0,
            (0, 0),
            Some((0, 0)),
            (A::ClampToZero, B::TransparentBlack),
            false,
        ),
        (
            Mode::Border,
            1,
            (2, 1),
            Some((0, 1)),
            (A::ClampToBorderColor, B::OpaqueBlack),
            false,
        ),
        (
            Mode::Border,
            2,
            (0, 0),
            Some((0, 2)),
            (A::ClampToZero, B::OpaqueWhite),
            false,
        ),
        (
            Mode::MirrorClamp,
            0,
            (0, 0),
            None,
            (A::MirrorClampToEdge, B::TransparentBlack),
            false,
        ),
        (
            Mode::MirrorClamp,
            0,
            (1, -2),
            None,
            (A::MirrorClampToEdge, B::TransparentBlack),
            false,
        ),
        (
            Mode::MirrorClamp,
            0,
            (0, 0),
            None,
            (A::MirrorClampToEdge, B::TransparentBlack),
            true,
        ),
    ];
    // Each form's texels: a two-plane texel's alpha code is a byte. An `_sRGB` form's decode takes
    // the device's `Pow`, which the interpreter does not model.
    let texels: Vec<(ExtendedRange, Vec<[u32; 4]>)> = FORMS
        .into_iter()
        .filter(|f| !f.srgb())
        .map(|form| {
            let codes = codes
                .iter()
                .map(|&[r, g, b, a]| match form {
                    ExtendedRange::Planes { .. } => [r, g, b, a % 256],
                    _ => [r, g, b, a],
                })
                .collect();
            (form, codes)
        })
        .collect();
    for (mode, border, offset, constexpr, (rm, rb), pixel) in states {
        for linear in [false, true] {
            for half in [false, true] {
                let (through, w) = match (runtime, constexpr) {
                    (false, Some((code, cb))) => {
                        (Through::Constexpr(words(linear, code, pixel, cb)), None)
                    }
                    (false, None) => continue,
                    (true, _) => (Through::Runtime, Some(word(linear, rm, pixel, rb))),
                };
                let kernel =
                    sample_kernel(through, half, (offset.0 as i32, offset.1 as i32), false);
                let Some(bc) = assemble("model", &kernel) else {
                    return;
                };
                let scale = if pixel { 256.0 } else { 256.0 * N as f32 };
                let coords: Vec<[f32; 2]> = q
                    .iter()
                    .map(|&(x, y)| [(x + 128) as f32 / scale, (y + 128) as f32 / scale])
                    .collect();
                for (form, codes) in &texels {
                    let (form, codes) = (*form, codes.as_slice());
                    let tex = Texels {
                        width: N as u32,
                        codes,
                    };
                    for (stem, p) in profiles() {
                        let what = format!(
                            "{form:?} {mode:?} border {border} offset {offset:?} pixel {pixel} linear {linear} half {half} {through:?} on {stem}"
                        );
                        let got = sample(&bc, &p, (form, &tex), (&coords, half, w), &what);
                        for (&q, g) in q.iter().zip(got) {
                            let case = Case {
                                mode,
                                border,
                                offset,
                                q,
                            };
                            let want = model(form, (N, codes), case, linear, half);
                            match linear {
                                true => assert!(
                                    filtered_close(&g, &want, half, 0.0),
                                    "{what}: {case:?}: {g:x?} against {want:x?}"
                                ),
                                false => assert_eq!(g, want, "{what}: {case:?}"),
                            }
                        }
                    }
                }
            }
        }
    }
}

/// The lowering refuses what it does not build under the variant, `NotImplemented(Format)`: a
/// two-plane form's gather (another form's gather is built, `float` or `half`), a constexpr
/// sampler with a linear mip filter of a two-plane form (a nearest one is built for every form),
/// and one whose minification and magnification filters differ. Each kernel compiles without the
/// variant.
#[test]
fn unbuilt_samples_are_refused() {
    let edge = words(false, 1, false, 0);
    let planes = |f: ExtendedRange| matches!(f, ExtendedRange::Planes { .. });
    let every = |_: ExtendedRange| true;
    let none = |_: ExtendedRange| false;
    let kernels: [(&str, String, &dyn Fn(ExtendedRange) -> bool); 5] = [
        (
            "gather",
            sample_kernel(Through::Constexpr(edge), false, (0, 0), true),
            &planes,
        ),
        (
            "half gather",
            sample_kernel(Through::Constexpr(edge), true, (0, 0), true),
            &planes,
        ),
        (
            "nearest mip",
            sample_kernel(Through::Constexpr(edge | 1 << 13), false, (0, 0), false),
            &none,
        ),
        (
            "linear mip",
            sample_kernel(Through::Constexpr(edge | 2 << 13), false, (0, 0), false),
            &planes,
        ),
        (
            "mixed",
            sample_kernel(Through::Constexpr(edge | 1 << 11), false, (0, 0), false),
            &every,
        ),
    ];
    for (name, kernel, refused) in kernels {
        let Some(bc) = assemble(name, &kernel) else {
            return;
        };
        for (stem, p) in profiles() {
            for form in FORMS {
                let mut i = LowerInputs::new();
                i.extended_range.insert(SOURCE, form);
                match (typed_module_with(&bc, "k", &p, &i), refused(form)) {
                    (Err(crate::error::Error::Unsupported(u)), true) => {
                        let want = Unsupported::not_implemented(Feature::Format);
                        let got = (u.feature, u.needs);
                        assert_eq!(got, (want.feature, want.needs), "{name} {form:?} on {stem}")
                    }
                    (Ok(_), false) => {}
                    (other, _) => panic!("{name} {form:?} on {stem}: {:?}", other.map(|_| ())),
                }
            }
            typed_module_with(&bc, "k", &p, &LowerInputs::new())
                .unwrap_or_else(|e| panic!("{name} on {stem} without the variant: {e}"));
        }
    }
}

/// Nearest and linear samples and red gathers of a 4 x 4 texture of each bfloat format, under
/// each address mode and border color, a constant offset, and pixel coordinates, through a
/// constexpr and a runtime sampler, with `float` and `half` results, at 96 positions each around
/// and past the edges: a sample is the nearest texel under either filter, its `float` the code
/// shifted up 16 (a NaN code a NaN as the device computes it, where Metal reads the quiet NaN; a
/// channel the format lacks 0 and alpha 1) and its `half` the device's narrowing of it (where
/// Metal rounds a sample's toward zero); a gather reads the red of the linear footprint's four
/// taps in Vulkan's order `(x0, y1)`, `(x1, y1)`, `(x1, y0)`, `(x0, y0)`, its `half` the device's
/// narrowing too; a tap outside a border axis reads the border color through
/// the format's channels.
#[test]
fn bfloat_samples_read_the_nearest_texel() {
    use SamplerAddressMode as A;
    use SamplerBorderColor as B;
    const N: i64 = 4;
    let mut rng = Lcg(0xbf16);
    let specials = [
        0x0000, 0x8000, 0x0001, 0x7f80, 0xff80, 0x7fc0, 0xffc1, 0x477f, 0x4780,
    ];
    let codes: Vec<[u32; 4]> = (0..N * N)
        .map(|_| {
            std::array::from_fn(|_| match rng.next() % 4 {
                0 => specials[(rng.next() % specials.len() as u32) as usize],
                _ => rng.next() & 0xFFFF,
            })
        })
        .collect();
    let q: Vec<(i64, i64)> = (0..96)
        .map(|_| {
            let mut c = || i64::from(rng.next() % (256 * N as u32 + 256)) - 256;
            (c(), c())
        })
        .collect();
    // Each state: its model's mode, border color and offset, the constexpr words' address code
    // and border (none: a runtime sampler's alone), the runtime word's mode and border, and
    // pixel coordinates.
    let states = [
        (
            Mode::Edge,
            0,
            (0, 0),
            Some((1, 0)),
            (A::ClampToEdge, B::TransparentBlack),
            false,
        ),
        (
            Mode::Edge,
            0,
            (1, -2),
            Some((1, 0)),
            (A::ClampToEdge, B::TransparentBlack),
            false,
        ),
        (
            Mode::Edge,
            0,
            (0, 0),
            Some((1, 0)),
            (A::ClampToEdge, B::TransparentBlack),
            true,
        ),
        (
            Mode::Repeat,
            0,
            (0, 0),
            Some((2, 0)),
            (A::Repeat, B::TransparentBlack),
            false,
        ),
        (
            Mode::Mirror,
            0,
            (0, 0),
            Some((3, 0)),
            (A::MirrorRepeat, B::TransparentBlack),
            false,
        ),
        (
            Mode::Border,
            0,
            (0, 0),
            Some((0, 0)),
            (A::ClampToZero, B::TransparentBlack),
            false,
        ),
        (
            Mode::Border,
            1,
            (2, 1),
            Some((0, 1)),
            (A::ClampToBorderColor, B::OpaqueBlack),
            false,
        ),
        (
            Mode::Border,
            2,
            (0, 0),
            Some((0, 2)),
            (A::ClampToZero, B::OpaqueWhite),
            false,
        ),
        (
            Mode::MirrorClamp,
            0,
            (0, 0),
            None,
            (A::MirrorClampToEdge, B::TransparentBlack),
            false,
        ),
    ];
    let forms = [
        (ExtendedRange::R16Bfloat, 1usize),
        (ExtendedRange::Rg16Bfloat, 2),
        (ExtendedRange::Rgba16Bfloat, 4),
    ];
    let read = |c: u32| c << 16;
    // Lane `c` of texel `(x, y)`, or of the border texel of color `b` where it is `None`.
    let texel = |ch: usize, at: Option<(i64, i64)>, b: u32, c: usize| -> u32 {
        match (at, c < ch) {
            (Some((x, y)), true) => read(codes[(y * N + x) as usize][c]),
            (None, true) => match b == 2 || (b == 1 && c == 3) {
                true => 1.0f32.to_bits(),
                false => 0,
            },
            (_, false) => match c == 3 {
                true => 1.0f32.to_bits(),
                false => 0,
            },
        }
    };
    // A gather's and a sample's `half` alike: the device's narrowing, to nearest even.
    let bits = |v: u32, half: bool, _gather: bool| match half {
        false => v,
        true => u32::from(crate::float16::f32_to_f16_bits(f32::from_bits(v))),
    };
    for runtime in [false, true] {
        for (mode, border, offset, constexpr, (rm, rb), pixel) in states {
            for (gather, linear) in [(false, false), (false, true), (true, false)] {
                // A gather of pixel coordinates or an offset is not asked.
                if gather && (pixel || offset != (0, 0)) {
                    continue;
                }
                for half in [false, true] {
                    let (through, w) = match (runtime, constexpr) {
                        (false, Some((code, cb))) => {
                            (Through::Constexpr(words(linear, code, pixel, cb)), None)
                        }
                        (false, None) => continue,
                        (true, _) => (Through::Runtime, Some(word(linear, rm, pixel, rb))),
                    };
                    let kernel =
                        sample_kernel(through, half, (offset.0 as i32, offset.1 as i32), gather);
                    let Some(bc) = assemble("bfloat", &kernel) else {
                        return;
                    };
                    let scale = if pixel { 256.0 } else { 256.0 * N as f32 };
                    let coords: Vec<[f32; 2]> = q
                        .iter()
                        .map(|&(x, y)| [(x + 128) as f32 / scale, (y + 128) as f32 / scale])
                        .collect();
                    for (form, ch) in forms {
                        let tex = Texels {
                            width: N as u32,
                            codes: &codes,
                        };
                        for (stem, p) in profiles() {
                            let what = format!(
                                "{form:?} {mode:?} border {border} offset {offset:?} pixel {pixel} linear {linear} gather {gather} half {half} {through:?} on {stem}"
                            );
                            let got = sample(&bc, &p, (form, &tex), (&coords, half, w), &what);
                            for (&(qx, qy), g) in q.iter().zip(got) {
                                let want: [u32; 4] = match gather {
                                    false => {
                                        let x =
                                            address(mode, (qx + 128).div_euclid(256) + offset.0, N);
                                        let y =
                                            address(mode, (qy + 128).div_euclid(256) + offset.1, N);
                                        let at = x.zip(y);
                                        std::array::from_fn(|c| {
                                            bits(texel(ch, at, border, c), half, false)
                                        })
                                    }
                                    true => {
                                        let (i, j) = (qx.div_euclid(256), qy.div_euclid(256));
                                        let (x0, x1) =
                                            (address(mode, i, N), address(mode, i + 1, N));
                                        let (y0, y1) =
                                            (address(mode, j, N), address(mode, j + 1, N));
                                        let taps = [(x0, y1), (x1, y1), (x1, y0), (x0, y0)];
                                        taps.map(|(x, y)| {
                                            bits(texel(ch, x.zip(y), border, 0), half, true)
                                        })
                                    }
                                };
                                let ok = g.iter().zip(want).all(|(&g, w)| native_lane(g, w, half));
                                assert!(ok, "{what}: q ({qx}, {qy}): {g:x?}, {want:x?}");
                            }
                        }
                    }
                }
            }
        }
    }
}

/// A swizzled texture's samples and gathers are the unswizzled texture's, swizzled, as plain
/// Metal answers: nearest and linear samples (`float` and `half`, through a
/// constexpr sampler and a runtime one) and `float` gathers of each component, of a 4 x 4
/// texture of each one-plane extended-range, padded and bfloat form (a two-plane form's samples
/// alone), clamped to an opaque-white border, at 32 positions around and past the edges, on one
/// profile, under
/// four swizzles: each lane the source its swizzle names, `Zero` 0 and `One` 1, a border texel
/// included; a gather of a component the gather of the component its swizzle names, or 0 or 1 at
/// every tap.
#[test]
fn swizzled_samples_and_gathers_are_the_unswizzled_swizzled() {
    use crate::metal::{TextureSwizzle as S, TextureSwizzleChannels as C};
    const N: i64 = 4;
    let swizzles = [
        C::new(S::Alpha, S::Red, S::One, S::Zero),
        C::new(S::Zero, S::Alpha, S::Green, S::One),
        C::new(S::Blue, S::Green, S::Red, S::Alpha),
        C::new(S::Green, S::Green, S::Green, S::Green),
    ];
    // The `_sRGB` forms' decode takes the device's `Pow`, which the interpreter does not model
    // (`RG10Unorm_X12_sRGB` reads its codes linearly).
    let forms = [
        ExtendedRange::Bgr10 { srgb: false },
        ExtendedRange::Bgra10 { srgb: false },
        ExtendedRange::Planes { srgb: false },
        ExtendedRange::R10X6,
        ExtendedRange::Rg10X12,
        ExtendedRange::Rg10X12Srgb,
        ExtendedRange::R16Bfloat,
        ExtendedRange::Rg16Bfloat,
        ExtendedRange::Rgba16Bfloat,
    ];
    let mut rng = Lcg(0x5a17);
    let coords: Vec<[f32; 2]> = (0..32)
        .map(|_| {
            let mut c = || (i64::from(rng.next() % (256 * N as u32 + 256)) - 128) as f32;
            [c() / (256.0 * N as f32), c() / (256.0 * N as f32)]
        })
        .collect();
    let white = word(
        true,
        SamplerAddressMode::ClampToBorderColor,
        false,
        SamplerBorderColor::OpaqueWhite,
    );
    // Each kernel: its name, its module, its result's `half`, its runtime sampler's word, and
    // the component it gathers.
    let mut kernels = Vec::new();
    for (linear, name) in [(false, "nearest"), (true, "linear")] {
        for half in [false, true] {
            let k = sample_kernel(
                Through::Constexpr(words(linear, 0, false, 2)),
                half,
                (0, 0),
                false,
            );
            kernels.push((format!("{name} half {half}"), k, half, None, None));
        }
    }
    kernels.push((
        "runtime linear".to_string(),
        sample_kernel(Through::Runtime, false, (0, 0), false),
        false,
        Some(white),
        None,
    ));
    for c in 0..4u32 {
        let k = sample_kernel(
            Through::Constexpr(words(false, 0, false, 2)),
            false,
            (0, 0),
            true,
        )
        .replace(
            "<2 x i32> zeroinitializer, i32 0, i32 0)",
            &format!("<2 x i32> zeroinitializer, i32 {c}, i32 0)"),
        );
        kernels.push((format!("gather {c}"), k, false, None, Some(c)));
    }
    let kernels: Vec<_> = kernels
        .into_iter()
        .map(|(name, k, half, w, gather)| {
            let bc = assemble("swizzled samples", &k);
            (name, bc, half, w, gather)
        })
        .collect();
    let mut checked = 0;
    for form in forms {
        // A two-plane texel's alpha is its byte.
        let bits = |c: usize| match form {
            _ if form.bfloat() => 0xFFFF,
            ExtendedRange::Planes { .. } if c == 3 => 0xFF,
            _ => 0x3FF,
        };
        let codes: Vec<[u32; 4]> = (0..N * N)
            .map(|_| std::array::from_fn(|c| rng.next() & bits(c)))
            .collect();
        let tex = Texels {
            width: N as u32,
            codes: &codes,
        };
        // The lowering takes no profile's choice here: one profile serves (the device lane
        // `runtime_lane::extended_swizzles` runs each venue).
        for (stem, p) in profiles().into_iter().take(1) {
            // Each kernel's unswizzled results.
            let mut base = BTreeMap::new();
            for (name, bc, half, w, gather) in &kernels {
                let Some(bc) = bc else {
                    return;
                };
                if gather.is_some() && matches!(form, ExtendedRange::Planes { .. }) {
                    continue;
                }
                let what = format!("{form:?} {name} on {stem}");
                let got = sample(bc, &p, (form, &tex), (&coords, *half, *w), &what);
                base.insert(name.clone(), got);
            }
            for swizzle in swizzles {
                for (name, bc, half, w, gather) in &kernels {
                    let Some(bc) = bc else {
                        return;
                    };
                    let Some(own) = base.get(name) else {
                        continue;
                    };
                    let what = format!("{form:?} {name} {swizzle:?} on {stem}");
                    let got = sample_swizzled(
                        bc,
                        &p,
                        (form, &tex),
                        (&coords, *half, *w),
                        Some(swizzle.word()),
                        &what,
                    );
                    let one = match half {
                        true => 0x3C00,
                        false => 1.0f32.to_bits(),
                    };
                    let sources = [swizzle.red, swizzle.green, swizzle.blue, swizzle.alpha];
                    // The unswizzled result of the channel source `s` names: a sample's lane, or
                    // the gather of that channel.
                    let pick = |s: S, n: usize, lane: usize| -> u32 {
                        match (s, gather) {
                            (S::Zero, _) => 0,
                            (S::One, _) => one,
                            (s, None) => own[n][s.as_raw() as usize - 2],
                            (s, Some(_)) => base[&format!("gather {}", s.as_raw() - 2)][n][lane],
                        }
                    };
                    for (n, g) in got.iter().enumerate() {
                        let want: [u32; 4] = match gather {
                            None => std::array::from_fn(|c| pick(sources[c], n, c)),
                            Some(c) => std::array::from_fn(|k| pick(sources[*c as usize], n, k)),
                        };
                        assert_eq!(*g, want, "{what}: at {:?}", coords[n]);
                        checked += 1;
                    }
                }
            }
        }
    }
    assert!(checked > 0);
}

/// The words of the 300 x 77 texture of `ch` channels (red then green per texel) Metal sampled
/// for `padded_samples.tsv`: the probe's generator, a 64-bit linear congruential state seeded
/// `0x5eed_0577 + 300 · 7919 + 77`, each word the low 16 bits of the state's bits 33 and up after
/// a step.
fn padded_words(ch: usize) -> Vec<u16> {
    let mut seed: u64 = 0x5eed_0577 + 300 * 7919 + 77;
    (0..300 * 77 * ch)
        .map(|_| {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (seed >> 33) as u16
        })
        .collect()
}

/// Each of Metal's samples in `padded_samples.tsv` (an Apple M4 under
/// macOS 27.0.1, byte-identical on an Apple M3 Max under macOS 26.7) is Metal's bytes, through a
/// runtime sampler of the row's state and, where a constexpr sampler can state it, through one:
/// `R10Unorm_X6` and `RG10Unorm_X12` under every address mode and border color, nearest and
/// linear, normalized and pixel coordinates, over the probe's 300 x 77 texture
/// ([`padded_words`]).
#[test]
fn padded_samples_are_metals_bytes() {
    replay_padded_samples("padded_samples.tsv");
}

/// Each of Metal's samples in `padded_srgb_samples.tsv` (an Apple M4 under macOS 27.0.1,
/// byte-identical on an Apple M3 Max under macOS 26.7) is within `DevicePackConversion`'s bound of
/// Metal's bytes, as [`padded_samples_are_metals_bytes`] replays them: `RG10Unorm_X12_sRGB`'s
/// (its reads and filter `RG10Unorm_X12`'s) under every address mode and border color, nearest
/// and linear, normalized and pixel coordinates. `R10Unorm_X6_sRGB`'s decode takes the device's
/// `Pow`, which the interpreter does not model.
#[test]
fn padded_srgb_samples_are_metals_bytes() {
    replay_padded_samples("padded_srgb_samples.tsv");
}

/// Replay the samples of fixture `name` (under `validation/fixtures/public/engine`), each
/// through a runtime sampler of the row's state and, where a constexpr sampler can state it,
/// through one, over the probes' 300 x 77 texture ([`padded_words`]).
fn replay_padded_samples(name: &str) {
    use SamplerAddressMode as M;
    use SamplerBorderColor as B;
    let words1 = padded_words(1);
    assert_eq!(words1[..4], [41917, 59041, 12598, 14278]);
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/engine")
        .join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    // (raw, filter, mode, coordinates) -> (coordinate, Metal's red and green bits) rows.
    let mut groups: BTreeMap<(String, String, String, String), Vec<([f32; 2], [u32; 2])>> =
        BTreeMap::new();
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let f: Vec<&str> = line.split(' ').collect();
        let hex = |s: &str| u32::from_str_radix(s, 16).expect("hex");
        let key = (f[0].into(), f[1].into(), f[2].into(), f[3].into());
        let uv = [f32::from_bits(hex(f[4])), f32::from_bits(hex(f[5]))];
        groups
            .entry(key)
            .or_default()
            .push((uv, [hex(f[6]), hex(f[7])]));
    }
    assert!(groups.len() >= 8, "{:?}", groups.keys());
    let mut checked = 0;
    for ((raw, filter, mode, space), rows) in &groups {
        // `R10Unorm_X6_sRGB`'s decode takes the device's `Pow`, which the interpreter does not
        // model: its samples are the device lanes'.
        if raw == "577" {
            continue;
        }
        let (form, ch) = match raw.as_str() {
            "576" => (ExtendedRange::R10X6, 1),
            "578" => (ExtendedRange::Rg10X12, 2),
            "577" => (ExtendedRange::R10X6Srgb, 1),
            "579" => (ExtendedRange::Rg10X12Srgb, 2),
            other => panic!("raw format {other}"),
        };
        let linear = filter == "linear";
        let pixel = space == "pixel";
        // The mode, its border color, and the constexpr words' address code and border.
        let (m, b, constexpr) = match mode.as_str() {
            "edge" => (M::ClampToEdge, B::TransparentBlack, Some((1, 0))),
            "repeat" => (M::Repeat, B::TransparentBlack, Some((2, 0))),
            "mirror" => (M::MirrorRepeat, B::TransparentBlack, Some((3, 0))),
            "mirrorclamp" => (M::MirrorClampToEdge, B::TransparentBlack, None),
            "zero" => (M::ClampToZero, B::TransparentBlack, Some((0, 0))),
            "clear" => (M::ClampToBorderColor, B::TransparentBlack, Some((0, 0))),
            "black" => (M::ClampToBorderColor, B::OpaqueBlack, Some((0, 1))),
            "white" => (M::ClampToBorderColor, B::OpaqueWhite, Some((0, 2))),
            other => panic!("mode {other}"),
        };
        let texel_words = padded_words(ch);
        let codes: Vec<[u32; 4]> = (0..300 * 77)
            .map(|i| {
                let c = |k: usize| u32::from(texel_words[i * ch + k]) >> 6;
                [c(0), if ch == 2 { c(1) } else { 0 }, 0, 0]
            })
            .collect();
        let tex = Texels {
            width: 300,
            codes: &codes,
        };
        let coords: Vec<[f32; 2]> = rows.iter().map(|(uv, _)| *uv).collect();
        // A pixel-coordinate sampler's repeat and mirrored-repeat axes are clamp-to-edge, as the
        // runtime takes them (`sampler::taken_descriptor`) and as Metal samples them (the rows of
        // pixel repeat); Apple's compiler makes no constexpr sampler of them ("address modes must
        // be either 'clamp_to_zero', 'clamp_to_edge', or 'clamp_to_border'").
        let wraps = matches!(m, M::Repeat | M::MirrorRepeat);
        let taken = if pixel && wraps { M::ClampToEdge } else { m };
        let mut throughs = vec![(Through::Runtime, Some(word(linear, taken, pixel, b)))];
        if let Some((code, border)) = constexpr.filter(|_| !(pixel && wraps)) {
            throughs.push((Through::Constexpr(words(linear, code, pixel, border)), None));
        }
        for (through, w) in throughs {
            let Some(bc) = assemble(
                "padded metal",
                &sample_kernel(through, false, (0, 0), false),
            ) else {
                return;
            };
            for (stem, p) in profiles() {
                let what = format!("{raw} {filter} {mode} {space} {through:?} on {stem}");
                let got = sample(&bc, &p, (form, &tex), (&coords, false, w), &what);
                for ((uv, [r, g]), got) in rows.iter().zip(got) {
                    let want = [*r, *g, 0, 1f32.to_bits()];
                    assert!(
                        filtered_close(&got, &want, false, 0.0),
                        "{what}: at {uv:?}: {got:x?} against {want:x?}"
                    );
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 0);
}

/// A kernel's state: texture slot 0 bound in `pf`, buffers 0 and 1, and sampler slot 0 bound to
/// `desc`.
pub(super) fn sampled_state(pf: PixelFormat, desc: SamplerDescriptor) -> BindState {
    sampled_state_of(TextureType::Type2D, pf, desc)
}

/// [`sampled_state`] with a texture of type `ty`.
pub(super) fn sampled_state_of(
    ty: TextureType,
    pf: PixelFormat,
    desc: SamplerDescriptor,
) -> BindState {
    let mut t = StageTable::default();
    t.textures.insert(
        0,
        BoundTexture::Texture {
            facts: TextureFacts::new(ty, pf),
            resource_id: None,
        },
    );
    for index in [0, 1] {
        t.buffers.insert(
            index,
            BoundBuffer::Buffer {
                length: 65536,
                offset: 0,
                gpu_address: 0x1_0000 * u64::from(index + 1),
                device_address: None,
                attribute_stride: None,
            },
        );
    }
    t.samplers.insert(0, BoundSampler::new(desc));
    let mut s = BindState::default();
    s.stages.insert(Stage::Kernel, t);
    s
}

/// A sampled slot bound in an extended-range format joins the plan's variant through a
/// constexpr and a runtime sampler: the variant's pipeline plans the same key, views the slot's
/// image in the raw format, and binds no sampler. The binder refuses a runtime sampler whose
/// word states mixed filters, `NotImplemented(Format)`, and plans one stating a nearest mip
/// filter, whose level the function reads (`levels`), or `MirrorClampToEdge`, whose taps the
/// fetches address. On every committed profile.
#[test]
fn a_sampled_slot_joins_the_variant() {
    use SamplerMinMagFilter as F;
    let formats = [
        (PixelFormat::BGR10XR, ExtendedRange::Bgr10 { srgb: false }),
        (
            PixelFormat::BGRA10XRSRGB,
            ExtendedRange::Bgra10 { srgb: true },
        ),
        (
            PixelFormat::RGB10A82PXR10,
            ExtendedRange::Planes { srgb: false },
        ),
        (PixelFormat::R10UnormX6, ExtendedRange::R10X6),
        (PixelFormat::RG10UnormX12, ExtendedRange::Rg10X12),
    ];
    let mut linear = SamplerDescriptor::new();
    linear.min_filter = F::Linear;
    linear.mag_filter = F::Linear;
    let mut planned = 0;
    for through in [
        Through::Constexpr(words(true, 1, false, 0)),
        Through::Runtime,
    ] {
        let Some(bc) = assemble("bind sample", &sample_kernel(through, false, (0, 0), false))
        else {
            return;
        };
        for (stem, dev) in devices() {
            let plain = pipeline(&dev, &bc, &LowerInputs::new()).expect("a sampling kernel");
            for (pf, form) in formats {
                let what = format!("{through:?} {pf:?} on {stem}");
                let shape = sampled_state(pf, linear).shape();
                let plan = plain
                    .plan(&dev, &shape)
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                assert_eq!(
                    plan.variant.extended_range,
                    BTreeMap::from([(SOURCE, form)]),
                    "{what}"
                );
                let key = plan.variant.lowering_inputs(Stage::Kernel);
                let variant = pipeline(&dev, &bc, &key).expect("the variant's inputs");
                let again = variant.plan(&dev, &shape).expect("variant plan");
                assert_eq!(again.variant, plan.variant, "{what}");
                let mut views = Vec::new();
                for d in &again.descriptors {
                    match &d.source {
                        WriteSource::Texture { view, .. } => views.push(view.format),
                        WriteSource::Sampler { .. } | WriteSource::CombinedImageSampler { .. } => {
                            panic!("{what}: the variant binds {:?}", d.source)
                        }
                        _ => {}
                    }
                }
                assert_eq!(views, [form.raw_format()], "{what}");
                planned += 1;
                // A swizzled texture joins `swizzled_storage` too, under a border sampler as
                // under this one, and the variant's pipeline plans the same key.
                let mut border = linear;
                border.s_address_mode = SamplerAddressMode::ClampToBorderColor;
                border.t_address_mode = SamplerAddressMode::ClampToBorderColor;
                border.border_color = SamplerBorderColor::OpaqueWhite;
                for desc in [linear, border] {
                    let mut state = sampled_state(pf, desc);
                    if let Some(BoundTexture::Texture { facts, .. }) = state
                        .stages
                        .get_mut(&Stage::Kernel)
                        .and_then(|t| t.textures.get_mut(&0))
                    {
                        use crate::metal::{TextureSwizzle as S, TextureSwizzleChannels as C};
                        facts.swizzle = C::new(S::Alpha, S::Red, S::One, S::Zero);
                    }
                    let shape = state.shape();
                    let what = format!("{what} swizzled {:?}", desc.s_address_mode);
                    let plan = plain
                        .plan(&dev, &shape)
                        .unwrap_or_else(|e| panic!("{what}: {e}"));
                    assert!(plan.variant.swizzled_storage.contains(&SOURCE), "{what}");
                    let key = plan.variant.lowering_inputs(Stage::Kernel);
                    let variant = pipeline(&dev, &bc, &key).expect("the variant's inputs");
                    let again = variant.plan(&dev, &shape).expect("variant plan");
                    assert_eq!(again.variant, plan.variant, "{what}");
                }
                if let Through::Runtime = through {
                    let mut mip = linear;
                    mip.mip_filter = crate::metal::SamplerMipFilter::Nearest;
                    let mut mixed = linear;
                    mixed.mag_filter = F::Nearest;
                    let mut clamp = linear;
                    clamp.t_address_mode = SamplerAddressMode::MirrorClampToEdge;
                    for (name, desc) in [("mirror clamp", clamp), ("mip", mip)] {
                        let shape = sampled_state(pf, desc).shape();
                        for p in [&plain, &variant] {
                            let named = p
                                .plan(&dev, &shape)
                                .unwrap_or_else(|e| panic!("{what} {name}: {e}"));
                            assert_eq!(named.variant, plan.variant, "{what} {name}");
                        }
                    }
                    for (name, desc) in [("mixed", mixed)] {
                        match plain.plan(&dev, &sampled_state(pf, desc).shape()) {
                            Err(crate::error::BindError::Unsupported(u)) => assert_eq!(
                                u,
                                Unsupported::not_implemented(Feature::Format),
                                "{what} {name}"
                            ),
                            other => panic!("{what} {name}: {:?}", other.map(|_| ())),
                        }
                    }
                }
            }
        }
    }
    assert!(planned > 0);
}

/// [`sample_kernel`]'s `float` sample through `through` taken `n` times, the `i`-th at the
/// coordinate plus `i / 8` on each axis, and where `offsets` at its own constant texel offset
/// (`(i + 1, -i)`), the sum stored.
fn samples_kernel(through: Through, n: usize, offsets: bool) -> String {
    let k = sample_kernel(through, false, (0, 0), false);
    let call = k
        .lines()
        .find(|l| l.starts_with("  %r = call"))
        .expect("the sample's call")
        .to_string();
    let extract = "  %v = extractvalue { <4 x float>, i8 } %r, 0";
    assert!(k.contains(extract), "the sample's texel");
    let mut body = String::new();
    for i in 0..n {
        let d = i as f32 / 8.0;
        body.push_str(&format!(
            "  %uv{i} = fadd <2 x float> %uv, <float {d:?}, float {d:?}>\n"
        ));
        let offset = match offsets {
            true => format!("<2 x i32> <i32 {}, i32 {}>", i + 1, -(i as i32)),
            false => "<2 x i32> <i32 0, i32 0>".to_string(),
        };
        body.push_str(
            &call
                .replacen("%r =", &format!("%r{i} ="), 1)
                .replacen("<2 x float> %uv,", &format!("<2 x float> %uv{i},"), 1)
                .replacen("<2 x i32> <i32 0, i32 0>", &offset, 1),
        );
        body.push_str(&format!(
            "\n  %v{i} = extractvalue {{ <4 x float>, i8 }} %r{i}, 0\n"
        ));
    }
    let mut sum = "%v0".to_string();
    for i in 1..n {
        body.push_str(&format!("  %s{i} = fadd <4 x float> {sum}, %v{i}\n"));
        sum = format!("%s{i}");
    }
    body.push_str(&format!("  %v = fadd <4 x float> {sum}, zeroinitializer"));
    k.replacen(&format!("{call}\n{extract}"), &body, 1)
}

/// The words a sample of an extended-range slot past the first may add beside its routine: the
/// call with its arguments, the status word and the sum's addition.
const SITE_WORDS: usize = 160;

/// The samples of one extended-range slot call one routine (`routine::Extended`) on a profile
/// whose driver keeps a `DontInline` call a call (`DriverWorkaround::DontInlineKeepsCalls`): a
/// kernel of 1, 2 and 4 samples through the runtime sampler, or through a constexpr linear
/// clamp-to-edge sampler each at its own constant texel offset (the call's value), holds one
/// routine, which its entry calls once per sample and which holds every fetch, and each sample
/// past the first adds no more than a call's words. On any other profile the entry holds every
/// fetch, its code written at each site (`extended::Sample::routine`).
#[test]
fn the_samples_of_one_slot_call_one_routine() {
    const OP_FUNCTION: u32 = 54;
    const OP_FUNCTION_END: u32 = 56;
    const OP_FUNCTION_CALL: u32 = 57;
    const OP_IMAGE_FETCH: u32 = 95;
    let clamped = Through::Constexpr(words(true, 1, false, 0));
    let cases = [(Through::Runtime, false), (clamped, true)];
    for ((stem, p), (through, offsets)) in profiles()
        .into_iter()
        .flat_map(|sp| cases.map(|c| (sp.clone(), c)))
    {
        let keeps = p
            .fields()
            .workarounds
            .contains(&crate::device::profile::DriverWorkaround::DontInlineKeepsCalls);
        let mut words = Vec::new();
        for n in [1usize, 2, 4] {
            let what = format!("{stem}, {n} samples, offsets {offsets}");
            let Some(bc) = assemble(&what, &samples_kernel(through, n, offsets)) else {
                return;
            };
            let mut inputs = LowerInputs::new();
            inputs
                .extended_range
                .insert(SOURCE, ExtendedRange::Planes { srgb: false });
            let a = validate_entry(&bc, "k", &inputs, &p, &what);
            // Each function's calls and fetches, by its result id.
            let w = &a.module.words;
            let (mut calls, mut fetches) = (BTreeMap::new(), BTreeMap::new());
            let mut current = None;
            let mut i = 5;
            while i < w.len() {
                let (len, op) = ((w[i] >> 16) as usize, w[i] & 0xffff);
                assert!(len > 0 && i + len <= w.len(), "{what}: malformed module");
                match op {
                    OP_FUNCTION => current = Some(w[i + 2]),
                    OP_FUNCTION_END => current = None,
                    OP_FUNCTION_CALL => {
                        let f = current.expect("a call outside a function");
                        *calls
                            .entry(f)
                            .or_insert_with(BTreeMap::new)
                            .entry(w[i + 3])
                            .or_insert(0usize) += 1;
                    }
                    OP_IMAGE_FETCH => *fetches.entry(current.expect("a fetch")).or_insert(0) += 1,
                    _ => {}
                }
                i += len;
            }
            let called: std::collections::BTreeSet<u32> =
                calls.values().flat_map(|c| c.keys().copied()).collect();
            if !keeps {
                let holders: Vec<u32> = fetches.keys().copied().collect();
                assert!(
                    matches!(holders[..], [f] if !called.contains(&f)),
                    "{what}: the entry holds every fetch, {holders:?}"
                );
                continue;
            }
            // The entry is the calling function no other calls.
            let entries: Vec<u32> = calls
                .keys()
                .copied()
                .filter(|f| !called.contains(f))
                .collect();
            let [entry] = entries[..] else {
                panic!("{what}: one entry, {entries:?}");
            };
            assert_eq!(
                fetches.get(&entry),
                None,
                "{what}: the entry fetches nothing"
            );
            let routines: Vec<u32> = calls
                .get(&entry)
                .into_iter()
                .flatten()
                .filter(|(f, k)| **k == n && fetches.contains_key(f))
                .map(|(f, _)| *f)
                .collect();
            assert_eq!(
                routines.len(),
                1,
                "{what}: the routine the entry calls per sample"
            );
            words.push((n, w.len()));
        }
        if !keeps {
            continue;
        }
        let (n0, words0) = words[0];
        for &(n, w) in &words[1..] {
            assert!(
                w - words0 <= (n - n0) * SITE_WORDS,
                "{stem}, offsets {offsets}: {n} samples grew {} words from {n0}",
                w - words0
            );
        }
    }
}
