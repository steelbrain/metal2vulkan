//! The codes of Apple's extended-range and padded texels, and the bound within which the device's
//! conversion of them answers Metal's: `PrecisionNote::DevicePackConversion`, which every module
//! converting such a texel records and the runtime announces (`docs/FAST-PATHS.md`, the
//! `air.pack` row). A write's code lies within one of Metal's; a read is within one `f32` unit in
//! the last place of Metal's quotient, an `_sRGB` twin's colour within half a `2^-13` step of
//! Metal's table (`R10Unorm_X6_sRGB`'s and `RGB10A2Unorm_sRGB`'s 1.64 steps) beyond the device
//! `Pow`'s bound; a `half` result within one `half` unit in the last place. The lanes that hold
//! such texels to Metal's compare them through these, the bits Metal does not read exactly, against
//! Metal's values: a model (`ExtendedRange::write_code` and `read_value`) the lane holds to Metal's
//! digests, or Metal's recording of the same case ([`recorded`]).

use super::degrade_notices::counts;
use super::extended_range_targets::{half_bits, half_value};
use super::Messages;
use metal2vulkan::metal::format::ExtendedRange;
use metal2vulkan::pipeline::PrecisionNote;
use metal2vulkan_runtime::Degradation;
use std::collections::BTreeMap;

/// The channel codes of one raw texel `bytes` of `form` as Metal's texture holds plane 0 (a
/// two-plane form's word alone, its alpha byte apart), in the field order of the bytes, and the
/// bits of it Metal does not read.
pub(super) fn codes(form: ExtendedRange, bytes: &[u8]) -> (Vec<u32>, u64) {
    let word = || u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    match form {
        ExtendedRange::Bgr10 { .. } | ExtendedRange::Planes { .. } => {
            let w = word();
            (
                vec![w >> 20 & 0x3ff, w >> 10 & 0x3ff, w & 0x3ff],
                u64::from(w >> 30),
            )
        }
        ExtendedRange::Rgb10A2Srgb => {
            let w = word();
            (
                vec![w & 0x3ff, w >> 10 & 0x3ff, w >> 20 & 0x3ff, w >> 30],
                0,
            )
        }
        // A code in each 16-bit field's top 10 bits, its low 6 bits unread.
        ExtendedRange::Bgra10 { .. }
        | ExtendedRange::R10X6
        | ExtendedRange::R10X6Srgb
        | ExtendedRange::Rg10X12
        | ExtendedRange::Rg10X12Srgb => {
            let fields: Vec<u16> = bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|f| u16::from_le_bytes(*f))
                .collect();
            let unread = fields
                .iter()
                .fold(0u64, |m, &f| m << 6 | u64::from(f & 0x3f));
            (fields.iter().map(|&f| u32::from(f >> 6)).collect(), unread)
        }
        _ => panic!("{form:?} holds no codes"),
    }
}

/// Whether the texels `got` and `want` of `form` (`got` the device's, `want` Metal's) hold the same
/// bits Metal does not read and each code within one of the other's: a write's bound under
/// `PrecisionNote::DevicePackConversion`.
pub(super) fn texel_within(form: ExtendedRange, got: &[u8], want: &[u8]) -> bool {
    let ((g, g_unread), (w, w_unread)) = (codes(form, got), codes(form, want));
    g_unread == w_unread && g.iter().zip(&w).all(|(a, b)| a.abs_diff(*b) <= 1)
}

/// How many codes apart the device's write of the value `v` to a channel of `form` (alpha where
/// `alpha`) may store from Metal's code under `PrecisionNote::DevicePackConversion`'s bound: one;
/// two for an `R10Unorm_X6_sRGB` or `RGB10A2Unorm_sRGB` colour below `0.0031308`, the transfer's
/// linear segment, where Metal rounds the value to `2^-13` before its table and a step spans 1.6
/// codes; any for a NaN, whose code is the device's.
pub(super) fn write_apart(form: ExtendedRange, v: f32, alpha: bool) -> u32 {
    let coarse = matches!(form, ExtendedRange::R10X6Srgb | ExtendedRange::Rgb10A2Srgb);
    match () {
        _ if v.is_nan() => u32::MAX,
        _ if coarse && !alpha && v < 0.003_130_8 => 2,
        _ => 1,
    }
}

/// [`texel_within`] of a write of `values` (in the order of [`codes`]' fields): each code within
/// [`write_apart`] of Metal's.
pub(super) fn written_within(form: ExtendedRange, got: &[u8], want: &[u8], values: &[f32]) -> bool {
    let ((g, g_unread), (w, w_unread)) = (codes(form, got), codes(form, want));
    let alpha = |c: usize| matches!(form, ExtendedRange::Rgb10A2Srgb) && c == 3;
    g_unread == w_unread
        && (0..g.len()).all(|c| g[c].abs_diff(w[c]) <= write_apart(form, values[c], alpha(c)))
}

/// The largest code of a channel of `form`, alpha where `alpha`: a two-plane form's alpha byte's
/// 255, else 1023.
fn largest_code(form: ExtendedRange, alpha: bool) -> u32 {
    match (form, alpha) {
        (ExtendedRange::Planes { .. }, true) => 255,
        _ => 1023,
    }
}

/// Metal's `float` read of `code` of a channel of `form`, alpha where `alpha`.
pub(super) fn metal_read(form: ExtendedRange, code: u32, alpha: bool) -> f32 {
    form.read_value(code, alpha)
        .unwrap_or_else(|| panic!("{form:?}: no Metal read of code {code}"))
}

/// The code of a channel of `form` (alpha where `alpha`) whose Metal read lies nearest `v`, the
/// value a read answered: Metal's reads rise with the code, so the code a device's read within
/// its bound came from.
pub(super) fn nearest_code(form: ExtendedRange, v: f32, alpha: bool) -> u32 {
    let top = largest_code(form, alpha);
    let read = |c: u32| f64::from(metal_read(form, c, alpha));
    let v = f64::from(v);
    // The first code whose read is at least `v` (`top + 1` where none is).
    let (mut low, mut high) = (0, top + 1);
    while low < high {
        let mid = (low + high) / 2;
        match read(mid) < v {
            true => low = mid + 1,
            false => high = mid,
        }
    }
    let above = low;
    [above.checked_sub(1), (above <= top).then_some(above)]
        .into_iter()
        .flatten()
        .min_by(|&a, &b| (read(a) - v).abs().total_cmp(&(read(b) - v).abs()))
        .unwrap_or(0)
}

/// The device `Pow`'s bound at the `_sRGB` transfer's operands, relative to its result: Vulkan's
/// `exp2(y · log2(x))` with `y` 2.4 and `x` in `[0.052, 1.24]` (codes up to 1023), `log2`'s three
/// units in the last place at most `3 · 2^-21` of `|log2 x|` below 4.3, times 2.4, `exp2`'s
/// `3 + 2 · |z|` units at `|z|` up to 10.3, and the `OpFDiv` before it, rounded up.
const POW_RELATIVE: f64 = 8e-6;

/// Whether `got`, the device's `float` read of the code `code` of a channel of `form` (alpha where
/// `alpha`), holds as Metal's read of it ([`value_within`]).
pub(super) fn read_within(form: ExtendedRange, got: f32, code: u32, alpha: bool) -> bool {
    value_within(form, got, metal_read(form, code, alpha), alpha)
}

/// The steps of `2^-13` an `_sRGB` colour channel of `form` may lie from Metal's table under
/// `PrecisionNote::DevicePackConversion`'s bound, beyond the `Pow`'s: half a step for the
/// extended-range twins, 1.64 for `R10Unorm_X6_sRGB`'s and `RGB10A2Unorm_sRGB`'s colours (which
/// convert as `R10Unorm_X6_sRGB`'s); none for a channel read linearly.
fn srgb_steps(form: ExtendedRange, alpha: bool) -> Option<f64> {
    match form {
        _ if alpha => None,
        ExtendedRange::R10X6Srgb | ExtendedRange::Rgb10A2Srgb => Some(1.64),
        _ if form.srgb() => Some(0.5),
        _ => None,
    }
}

/// Whether `got`, the device's `float` read of a channel of `form` (alpha where `alpha`), holds as
/// Metal's read `want` of the same code under `PrecisionNote::DevicePackConversion`'s bound: a
/// linear value or alpha within one `f32` unit in the last place, an `_sRGB` colour within its
/// steps of `2^-13` ([`srgb_steps`]) beyond the `Pow`'s bound ([`POW_RELATIVE`]).
pub(super) fn value_within(form: ExtendedRange, got: f32, want: f32, alpha: bool) -> bool {
    if got.to_bits() == want.to_bits() || (got == 0.0 && want == 0.0) {
        return true;
    }
    if !got.is_finite() || !want.is_finite() {
        return false;
    }
    if let Some(steps) = srgb_steps(form, alpha) {
        let off = (f64::from(got) - f64::from(want)).abs();
        return off <= steps * 2f64.powi(-13) + f64::from(want).abs() * POW_RELATIVE;
    }
    got.is_sign_negative() == want.is_sign_negative() && got.to_bits().abs_diff(want.to_bits()) <= 1
}

/// Whether `got`, the device's `half` result of a channel of `form` (alpha where `alpha`), holds
/// as Metal's `float` read `want` of the same code narrowed: the device's narrowing of a `float`
/// within [`value_within`] of `want`, so within that bound and one `half` unit in the last place at
/// `got` (`PrecisionNote::DevicePackConversion`'s `half` result within one unit).
pub(super) fn half_within(form: ExtendedRange, got: f32, want: f32, alpha: bool) -> bool {
    if !got.is_finite() || !want.is_finite() {
        return got.to_bits() == want.to_bits();
    }
    // A `half`'s unit in the last place at `got`: 2^-24 among the denormals.
    let ulp = match got.abs() {
        g if g < 2f32.powi(-14) => 2f64.powi(-24),
        g => 2f64.powi(g.log2().floor() as i32 - 10),
    };
    let reach = match srgb_steps(form, alpha) {
        Some(steps) => steps * 2f64.powi(-13) + f64::from(want).abs() * POW_RELATIVE,
        // One `f32` unit in the last place.
        None => f64::from(want.abs()) * 2f64.powi(-23),
    };
    (f64::from(got) - f64::from(want)).abs() <= reach + ulp
}

/// The `PrecisionNote::DevicePackConversion` announcement the lanes require where `moved` of their
/// values hold within its bound but not as Metal's: the runtime announced it (`what` names the
/// lane's values), among the notices `messages` holds, which it takes.
pub(super) fn assert_announced(messages: &Messages, moved: usize, what: &str) {
    assert_counted(&counts(&messages.take_notices()), moved, what);
}

/// [`assert_announced`] of the notices `announced` counts, for a lane that reads other kinds of
/// the same notices.
pub(super) fn assert_counted(announced: &BTreeMap<Degradation, usize>, moved: usize, what: &str) {
    let note = Degradation::from_precision_note(PrecisionNote::DevicePackConversion);
    assert!(
        moved == 0 || announced.contains_key(&note),
        "{what}: {moved} values within the device's conversion, unannounced: {announced:?}"
    );
}

/// The bytes Metal's recording of the conformance case `case` (the Apple M3 Max, macOS 27.0.1)
/// holds at the site `path` below the case's root (`["output", "codes"]`): Metal's own outputs,
/// whole, which the lanes compare value by value.
pub(super) fn recorded(case: &str, path: &[&str]) -> Vec<u8> {
    use base64::Engine as _;
    let file = format!(
        "{}/../metal2vulkan-conformance/recordings/{case}.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&file).unwrap_or_else(|e| panic!("{file}: {e}"));
    let recording: serde_json::Value = serde_json::from_str(&text).unwrap();
    let answer = recording["answers"]
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["key"]["chip"] == "apple-m3-max" && a["key"]["os_version"] == "27.0.1")
        .unwrap_or_else(|| panic!("{case}: no answer of the M3 Max under macOS 27.0.1"));
    let mut node = &answer["outcomes"][0]["transcript"]["root"];
    for name in path {
        node = node["children"]
            .as_array()
            .unwrap()
            .iter()
            .find(|n| n["name"] == *name)
            .unwrap_or_else(|| panic!("{case}: no site {path:?}"));
    }
    base64::engine::general_purpose::STANDARD
        .decode(node["observed"]["value"]["bytes"].as_str().unwrap())
        .unwrap()
}

/// Metal's `half` read of an extended-range code whose `float` read is `v`: `v` rounded to the
/// nearest `half`, ties away from zero (`PrecisionNote::DevicePackConversion`'s statement of
/// Metal's rounding; it gives Metal's digests of every `half` pattern and `float` grid written to
/// and read from a two-plane texture).
pub(super) fn metal_half(v: f32) -> u16 {
    let h = half_bits(v);
    let (sign, magnitude) = (h & 0x8000, h & 0x7fff);
    let a = f64::from(v.abs());
    let at = |m: u16| f64::from(half_value(m));
    // The `half` on the other side of `|v|` from the nearest one.
    let other = match at(magnitude) > a {
        true => magnitude.saturating_sub(1),
        false => magnitude + 1,
    };
    match a == (at(magnitude) + at(other)) / 2.0 {
        true => sign | magnitude.max(other),
        false => h,
    }
}

/// The `float4` reads `floats` and `half4` reads `halves` of a texture of `form`, one a grid
/// thread, with each lane whose thread reads a texel Metal holds the codes `metal` of (red, green,
/// blue and alpha; `None` for a channel the form lacks, or a thread past the texture) replaced by
/// Metal's reads of Metal's code where the device's lanes hold within
/// `PrecisionNote::DevicePackConversion`'s bound: the code the device's `float` read came from
/// ([`nearest_code`]) within one of Metal's (a write's bound), that read within its bound of
/// Metal's read of that code ([`read_within`]) and the `half` read within its bound of it
/// ([`half_within`]). What Metal's reads give where the device held the bound, so their digests
/// are Metal's; a lane past it stays as the device read it. Also how many texels the bound moved.
pub(super) fn as_metal_reads(
    form: ExtendedRange,
    (floats, halves): (&[u8], &[u8]),
    metal: &[[Option<u32>; 4]],
) -> (Vec<u8>, Vec<u8>, usize) {
    let (mut f, mut h) = (floats.to_vec(), halves.to_vec());
    let mut moved = 0;
    for (t, codes) in metal.iter().enumerate() {
        let mut changed = false;
        for (c, code) in codes.iter().enumerate() {
            let Some(code) = *code else { continue };
            let alpha = c == 3;
            let (fi, hi) = (16 * t + 4 * c, 8 * t + 2 * c);
            let v = f32::from_le_bytes(f[fi..fi + 4].try_into().unwrap());
            let hv = half_value(u16::from_le_bytes([h[hi], h[hi + 1]]));
            let read = nearest_code(form, v, alpha);
            let held = read.abs_diff(code) <= 1
                && read_within(form, v, read, alpha)
                && half_within(form, hv, metal_read(form, read, alpha), alpha);
            if !held {
                continue;
            }
            let want = metal_read(form, code, alpha);
            changed |= v.to_bits() != want.to_bits()
                || hv.to_bits() != half_value(metal_half(want)).to_bits();
            f[fi..fi + 4].copy_from_slice(&want.to_le_bytes());
            h[hi..hi + 2].copy_from_slice(&metal_half(want).to_le_bytes());
        }
        moved += usize::from(changed);
    }
    (f, h, moved)
}
