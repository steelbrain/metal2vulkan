//! What a kernel's `float` write stores in a half-float storage texture, over an exhaustive sweep
//! of every `half` bit pattern.
//!
//! Metal's conversion here is **not** IEEE's round-to-nearest, which is the whole reason the family
//! exists: `runtime_lane::half_storage_writes` measured that Metal rounds each channel **toward
//! zero**, stores a finite value at or past 65504 as 65504, keeps an infinity, and stores every NaN
//! as `0x7e00`. A device that rounds to nearest is wrong by one code on half the inputs, and a
//! device that quiets a NaN its own way is wrong in the payload — both are differences a comparison
//! in floats would call almost right, so every site here records unsigned words.
//!
//! **The sweep is exhaustive over a complete space**, which is why it records whole rather than as a
//! digest: all 65,536 `half` bit patterns, each widened to its exact `f32` and written back, so
//! every NaN payload, every denormal, both zeros, both infinities and every normal is a row. Ahead
//! of them sit the lane's own specials, which are the inputs no half code can be: the `f32`
//! denormals that have to flush, 65519 and 65520 either side of the half's last rounding boundary,
//! 1e10 past the format's range, and four `f32` NaN payloads. The tail of the last row holds one
//! signalling `f32` NaN, stated so every texel of the level is a value the case named.
//!
//! One case per format and per form, so a form or a format a venue honestly lacks is that venue's
//! gap ([`crate::verdict::Verdict::LacksFeature`]) at its own row.
//!
//! The owned fixture `hsw_storage_writes` holds the forms as separate kernels, each writing buffer
//! 0's `float4` at the thread's texel with the row's width in buffer 1:
//!
//! | Form | The kernel, and how it reaches the texture |
//! |---|---|
//! | `write` | `hsw_write`, a `texture2d<float, access::write>` at texture slot 0 |
//! | `read-write` | `hsw_read_write`, a `texture2d<float, access::read_write>` at slot 0 |
//! | `held` | `hsw_held`, the `texture2d<float, access::write>` an argument buffer at buffer 2 holds |
//!
//! The lane's fourth form, `hsw_buffer`'s `texture_buffer<float, access::write>`, is not ported: a
//! texture over a buffer's bytes has no call in the vocabulary a case is written against, so there
//! is nowhere to say it. The module docs of `runtime_lane::half_storage_writes` record that plain
//! Metal stored the same codes through all four.
//!
//! Every texel holds [`FILL`] before the dispatch, so a texel the kernel did not reach is a
//! recorded value rather than whatever the allocator left.
//!
//! Nothing here can fault a GPU: one compute pass, one texture, one thread a texel inside it, no
//! loop and no sampling.

use crate::allowance::Allowance;
use crate::case::{
    call, made, Abort, ArgumentSlot, Bindings, Case, Failure, Handle, Input, MetalSource, Side,
    TextureRole, TextureSpec,
};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::transcript::{
    Aspect, ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value,
};
use metal2vulkan::metal::PixelFormat;
use metal2vulkan_runtime::Degradation;

/// The fixture the three kernels live in.
const FIXTURE: &str = "hsw_storage_writes";

/// The texture's row of texels, which is the lane's own width and the `width` the kernels read
/// from buffer 1.
const WIDTH: u32 = 1024;

/// The rows the sweep takes: 65,536 codes and the specials fit in sixty-five.
const HEIGHT: u32 = 65;

/// The texels of one level.
const TEXELS: usize = (WIDTH * HEIGHT) as usize;

/// The byte every texel holds before the dispatch.
pub const FILL: u8 = 0xab;

/// The value the tail of the sweep's last row holds, as its bits: a signalling `f32` NaN, so every
/// texel of the level is a value the case stated and the NaN rule is asked once more.
const TAIL: u32 = 0x7f80_0001;

/// The lane's specials, as `f32` bits: the inputs no `half` code can be.
///
/// Zeros and ones of both signs, the two `f32` denormals that must flush, a tiny normal, both
/// infinities, 65504 and the two values either side of the half's last rounding boundary, a value
/// far past the range, and four `f32` NaN payloads. These are `runtime_lane::color_writes`'
/// `specials`, by their bits so a transcription cannot move one by a step.
const SPECIALS: [u32; 27] = [
    0x0000_0000, // 0.0
    0x8000_0000, // -0.0
    0x3f80_0000, // 1.0
    0xbf80_0000, // -1.0
    0x3f00_0000, // 0.5
    0xbf00_0000, // -0.5
    0x3fc0_0000, // 1.5
    0xbfc0_0000, // -1.5
    0x4000_0000, // 2.0
    0xc000_0000, // -2.0
    0x0000_0001, // 1e-45, the smallest `f32` denormal
    0x8000_0001, // -1e-45
    0x0080_0000, // the smallest `f32` normal
    0x8080_0000, // its negative
    0x2edb_e6ff, // 1e-10
    0xaedb_e6ff, // -1e-10
    0x7f80_0000, // infinity
    0xff80_0000, // -infinity
    0x477f_e000, // 65504, the largest half
    0x477f_fc00, // 65519, below the boundary
    0x477f_fe00, // 65520, at it
    0x5012_7500, // 1e10
    0xd012_7500, // -1e10
    0x7fc0_0000, // a quiet NaN
    0xffc0_0000, // a negative quiet NaN
    0x7f80_0001, // a signalling NaN
    0x7fff_ffff, // a NaN of every payload bit
];

/// The exact `f32` bits of half `bits`, denormals included: the widening that makes the sweep's
/// input the half code itself.
fn widened(bits: u16) -> u32 {
    let sign = u32::from(bits >> 15) << 31;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let mantissa = u32::from(bits & 0x3ff);
    match exponent {
        0x1f => sign | 0x7f80_0000 | mantissa << 13,
        0 => sign | (mantissa as f32 * 2f32.powi(-24)).to_bits(),
        raised => sign | (raised + 112) << 23 | mantissa << 13,
    }
}

/// Every value the sweep writes, in order: the specials, then every half code's exact widening,
/// then [`TAIL`] to the end of the level.
fn values() -> Vec<u32> {
    let mut out: Vec<u32> = SPECIALS.to_vec();
    out.extend((0..=u16::MAX).map(widened));
    out.resize(TEXELS, TAIL);
    out
}

/// The buffer the kernels read: one `float4` a texel, every channel the texel's own value, so a
/// store that treated one channel differently from another shows as a channel of a texel.
fn value_bytes() -> Vec<u8> {
    values()
        .into_iter()
        .flat_map(|word| [word; 4])
        .flat_map(u32::to_le_bytes)
        .collect()
}

/// Buffer 1's bytes: the row's width, as the kernels' `constant uint &width`.
fn width_bytes() -> Vec<u8> {
    WIDTH.to_le_bytes().to_vec()
}

/// Which kernel a case runs, and how it reaches the texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Form {
    /// `hsw_write`: a `texture2d<float, access::write>` at texture slot 0.
    Write,
    /// `hsw_read_write`: a `texture2d<float, access::read_write>` at texture slot 0.
    ReadWrite,
    /// `hsw_held`: the `texture2d<float, access::write>` an argument buffer at buffer 2 holds.
    Held,
}

impl Form {
    /// The kernel's entry point.
    fn entry(self) -> &'static str {
        match self {
            Form::Write => "hsw_write",
            Form::ReadWrite => "hsw_read_write",
            Form::Held => "hsw_held",
        }
    }

    /// The usage the texture needs: a `read_write` declaration needs both, the other two the write
    /// alone, and the role is part of what the case means.
    fn role(self) -> TextureRole {
        match self {
            Form::ReadWrite => TextureRole::ReadWrite,
            Form::Write | Form::Held => TextureRole::Write,
        }
    }
}

/// Every form the case runs, with the word its row's name uses.
const FORMS: [(Form, &str); 3] = [
    (Form::Write, "write"),
    (Form::ReadWrite, "read-write"),
    (Form::Held, "held"),
];

/// The case: one half-float format written through one form.
pub struct HalfStorageWrites {
    name: &'static str,
    format: PixelFormat,
    /// The channels the format's texel holds, which is how many of the `float4`'s lanes reach it.
    channels: u32,
    form: Form,
    /// The form's name, for the input fingerprint's label.
    form_name: &'static str,
}

/// Every pairing of the three half-float formats with the three forms.
pub fn all() -> Vec<HalfStorageWrites> {
    // Written out rather than built, so a case's recording file is named by something a reader can
    // grep for.
    const NAMES: [[&str; 3]; 3] = [
        [
            "half-storage-writes-r16f-write",
            "half-storage-writes-r16f-read-write",
            "half-storage-writes-r16f-held",
        ],
        [
            "half-storage-writes-rg16f-write",
            "half-storage-writes-rg16f-read-write",
            "half-storage-writes-rg16f-held",
        ],
        [
            "half-storage-writes-rgba16f-write",
            "half-storage-writes-rgba16f-read-write",
            "half-storage-writes-rgba16f-held",
        ],
    ];
    const FORMATS: [(PixelFormat, u32); 3] = [
        (PixelFormat::R16Float, 1),
        (PixelFormat::RG16Float, 2),
        (PixelFormat::RGBA16Float, 4),
    ];
    let mut cases = Vec::new();
    for ((format, channels), names) in FORMATS.into_iter().zip(NAMES) {
        for ((form, form_name), name) in FORMS.into_iter().zip(names) {
            cases.push(HalfStorageWrites {
                name,
                format,
                channels,
                form,
                form_name,
            });
        }
    }
    cases
}

impl HalfStorageWrites {
    /// The fixture under the case's own kernel.
    fn fixture(&self) -> Fixture {
        Fixture {
            name: FIXTURE,
            entry: self.form.entry(),
        }
    }

    /// The bytes one texel takes: a half a channel.
    fn texel_bytes(&self) -> u32 {
        2 * self.channels
    }

    /// The bytes one level takes, which is also the bytes the case fills it with.
    fn level_bytes(&self) -> usize {
        TEXELS * self.texel_bytes() as usize
    }

    /// The texture the kernel writes.
    fn spec(&self) -> TextureSpec {
        TextureSpec::new(
            self.format,
            [WIDTH, HEIGHT],
            WIDTH * self.texel_bytes(),
            self.form.role(),
        )
    }

    /// The layout the stored codes are compared under: one row a texel, a half a channel, so the
    /// element a diff names is the input's index and the channel inside it.
    fn layout(&self) -> ByteLayout {
        let pitch = self.texel_bytes();
        let length = pitch * TEXELS as u32;
        ByteLayout {
            region: Region::Texture {
                level: 0,
                slice: 0,
                aspect: Aspect::Color,
                sample: 0,
                origin: [0, 0, 0],
                size: [WIDTH, HEIGHT, 1],
            },
            element: Element::U16,
            row_offset: 0,
            row_elements: self.channels,
            row_pitch: pitch,
            rows: TEXELS as u32,
            image_pitch: length,
            images: 1,
            // A row is one texel's channels and the texels abut, so there is no byte outside one.
            gaps: Gaps::Ignored,
        }
    }
}

impl Case for HalfStorageWrites {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // A **notice declaration**, not a tolerance: it loosens no stored code. The sweep's grid is
        // 1,024 by 65 threads at 32 by 32 a threadgroup, so 65 is not a whole number of
        // threadgroups and the runtime runs the dispatch as more than one Vulkan dispatch and says
        // so. It is a performance note -- it states nothing about a value -- and it is chosen from
        // the case's own shape, the grid an exhaustive sweep of 65,536 codes needs, never from the
        // device. Metal announces nothing, so without the declaration every row of the family
        // would read `divergent` on its notice site alone and the codes, which are the question,
        // would be unreadable beside it.
        // Every row reads its answer back through a `Shared` buffer, so on a device without
        // unified memory the runtime announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice` holds the whole reason). It declares the
        // notices site alone, so every value of this family still compares against Metal's
        // exactly.
        //
        // Measured on the RTX 5080 Laptop (615.71.09): it fired for
        // all 9 rows of the family, and each was divergent on that field and
        // nothing else (replay-nvidia.log). On MoltenVK, where
        // the chip's GPU shares the CPU's memory, it cannot fire at all, which is why no
        // recording of this suite declares it.
        vec![
            crate::replay::shared_memory_notice(),
            crate::replay::notice_allowance(Degradation::SplitDispatch.code()),
        ]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = Vec::new();
        for extension in [RUNTIME_TEXT, METAL_TEXT] {
            shaders.push(ShaderRef {
                fixture: format!("{FIXTURE}{extension}"),
                entry: self.form.entry().to_owned(),
                source_sha256: crate::sha256_hex(self.fixture().text(extension)?.as_bytes()),
            });
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = self.fixture();
        Ok(vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            // The format and the form are the two questions the case asks, so a case re-pointed at
            // either is unrecorded rather than compared against the first's codes.
            Input::bytes("format", &self.format.as_raw().to_le_bytes()),
            Input::bytes("form", self.form_name.as_bytes()),
            Input::bytes("buffer:0", &value_bytes()),
            Input::bytes("buffer:1", &width_bytes()),
            Input::bytes("texture:fill", &vec![FILL; self.level_bytes()]),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = self.fixture().loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let pipeline = made(
            sites,
            "pipeline",
            side.compute_pipeline(library, loaded.entry),
        )?;
        let values = made(sites, "buffer/0", side.buffer(&value_bytes()))?;
        let width = made(sites, "buffer/1", side.buffer(&width_bytes()))?;
        let texture = made(
            sites,
            "texture",
            side.texture_2d_in_role(&self.spec(), &vec![FILL; self.level_bytes()]),
        )?;
        // The held form reaches the texture through an argument buffer at buffer 2; the other two
        // bind it at texture slot 0.
        let (textures, buffers): (Vec<Handle>, Vec<Handle>) = match self.form {
            Form::Held => {
                let held = made(
                    sites,
                    "argument-buffer",
                    side.argument_buffer(&[ArgumentSlot::Texture(texture)]),
                )?;
                (Vec::new(), vec![values, width, held])
            }
            Form::Write | Form::ReadWrite => (vec![texture], vec![values, width]),
        };
        let command_buffer = made(
            sites,
            "dispatch",
            side.dispatch_threads(
                pipeline,
                &Bindings::new(&textures, &buffers),
                &[],
                [WIDTH, HEIGHT, 1],
                [32, 32, 1],
            ),
        )?;
        let status = call(sites, "status", side.wait(command_buffer), Clone::clone)?;
        if status != Value::unit_variant("Completed") {
            return Err(Abort {
                at: "status".to_owned(),
                failure: Failure::named("CommandBufferStatus", crate::diff::print(&status)),
            });
        }
        let stored = call(
            sites,
            "read-back",
            side.read_texture(texture, [WIDTH, HEIGHT], WIDTH * self.texel_bytes()),
            |_| Value::Absent,
        )?;
        sites.push(Node::leaf("stored", Observed::bytes(stored, self.layout())));
        Ok(())
    }
}

/// The code Metal's conversion stores for `x`, as `runtime_lane::half_storage_writes` measured it:
/// a NaN is `0x7e00`, an infinity is kept, a finite value at or past 65504 is the largest half, and
/// everything else is rounded **toward zero**.
///
/// Public so the cross-check can name it and so a reader of the family can find the rule in one
/// place rather than in prose.
pub fn half_toward_zero(x: f32) -> u16 {
    if x.is_nan() {
        return 0x7e00;
    }
    let sign = if x.is_sign_negative() { 0x8000u16 } else { 0 };
    let size = x.abs();
    if size.is_infinite() {
        return sign | 0x7c00;
    }
    if size >= 65504.0 {
        return sign | 0x7bff;
    }
    if size < 2f32.powi(-14) {
        // Exact: a power-of-two scaling of a value below the smallest half normal.
        return sign | (size * 2f32.powi(24)).floor() as u16;
    }
    let bits = size.to_bits();
    let exponent = ((bits >> 23) & 0xff) as u16 - (127 - 15);
    sign | (exponent << 10) | ((bits >> 13) & 0x3ff) as u16
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// The sweep holds every half code exactly once, the specials ahead of them, and nothing past
    /// the level: an exhaustive sweep that lost a code would ask less than it claims.
    #[test]
    fn the_sweep_is_every_half_code_and_the_specials() {
        let taken = values();
        assert_eq!(taken.len(), TEXELS);
        assert_eq!(&taken[..SPECIALS.len()], &SPECIALS[..]);
        let codes = &taken[SPECIALS.len()..SPECIALS.len() + 65_536];
        let distinct: BTreeSet<u32> = codes.iter().copied().collect();
        // Every half code widens to its own `f32`, so the 65,536 widenings are 65,536 words.
        assert_eq!(distinct.len(), 65_536);
        for (code, &word) in codes.iter().enumerate() {
            assert_eq!(word, widened(code as u16), "code {code}");
        }
        assert!(taken[SPECIALS.len() + 65_536..]
            .iter()
            .all(|&word| word == TAIL));
        assert_eq!(value_bytes().len(), TEXELS * 16);
    }

    /// Every case states bytes for its whole level, and its layout covers exactly them: a layout
    /// reaching further would read past the level and one stopping short would leave codes
    /// uncompared.
    #[test]
    fn every_layout_covers_its_whole_level() {
        for case in all() {
            assert_eq!(case.layout().extent(), case.level_bytes(), "{}", case.name);
            assert_eq!(case.spec().row_pitch, WIDTH * case.texel_bytes());
        }
    }

    /// The cases name distinct rows, and every pairing of the three formats with the three forms
    /// is one of them.
    #[test]
    fn the_cases_name_every_format_and_form_once() {
        let named: BTreeSet<&str> = all().iter().map(|case| case.name).collect();
        assert_eq!(named.len(), 9);
        let pairs: BTreeSet<(u32, &str)> = all()
            .iter()
            .map(|case| (case.format.as_raw(), case.form_name))
            .collect();
        assert_eq!(pairs.len(), 9);
        // And each form names its own kernel.
        let entries: BTreeSet<&str> = FORMS.iter().map(|(form, _)| form.entry()).collect();
        assert_eq!(entries.len(), FORMS.len());
    }

    /// The rule rounds toward zero rather than to nearest, keeps an infinity, clamps a finite value
    /// past the range and answers `0x7e00` for every NaN.
    ///
    /// The inputs are the ones that separate the two roundings: the midpoint between two halves
    /// takes the lower code under Metal's rule and the even one under IEEE's.
    #[test]
    fn the_rule_rounds_toward_zero() {
        // The tie between 1.0 and the next half up: toward zero keeps 1.0's code.
        let tie = f32::from_bits(0x3f80_0800);
        assert_eq!(half_toward_zero(tie), 0x3c00);
        assert_eq!(half_toward_zero(-tie), 0xbc00);
        // Just below the tie between 2048 and 2050, and just above it.
        assert_eq!(half_toward_zero(2049.0), 0x6800);
        assert_eq!(half_toward_zero(2049.9), 0x6800);
        assert_eq!(half_toward_zero(f32::INFINITY), 0x7c00);
        assert_eq!(half_toward_zero(f32::NEG_INFINITY), 0xfc00);
        assert_eq!(half_toward_zero(65519.0), 0x7bff);
        assert_eq!(half_toward_zero(1e10), 0x7bff);
        assert_eq!(half_toward_zero(-1e10), 0xfbff);
        assert_eq!(half_toward_zero(f32::from_bits(0x7f80_0001)), 0x7e00);
        assert_eq!(half_toward_zero(f32::from_bits(0xffc0_0000)), 0x7e00);
        // An `f32` denormal flushes to the half zero of its sign.
        assert_eq!(half_toward_zero(f32::from_bits(1)), 0x0000);
        assert_eq!(half_toward_zero(f32::from_bits(0x8000_0001)), 0x8000);
        // Both signed zeros keep their sign.
        assert_eq!(half_toward_zero(0.0), 0x0000);
        assert_eq!(half_toward_zero(-0.0), 0x8000);
        // Every half code round-trips through its own exact widening.
        for code in 0..=u16::MAX {
            let stored = half_toward_zero(f32::from_bits(widened(code)));
            let want = match code & 0x7c00 == 0x7c00 && code & 0x3ff != 0 {
                true => 0x7e00,
                false => code,
            };
            assert_eq!(stored, want, "code {code:#06x}");
        }
    }

    /// Metal's recorded codes are the rule the lane measured, code for code, for every format,
    /// every form and every channel.
    ///
    /// This is the cross-check the port exists for: the lane holds the rule in a function and
    /// asserts a live device against it, which says nothing on a host with no Metal. Here the rule
    /// is held against Metal's own recorded answer, so a release that changes Metal's conversion
    /// shows as the input that moved rather than as a lane that stops passing on one Mac.
    #[test]
    fn metals_recorded_codes_are_the_rule_the_lane_measured() {
        use metal2vulkan_runtime::MetalLevel;
        let inputs = values();
        for case in all() {
            let Ok(recording) = crate::recording::read(case.name()) else {
                // A case recorded in a later commit of this batch.
                continue;
            };
            let answer = recording
                .answer(MetalLevel::NEWEST)
                .unwrap_or_else(|why| panic!("{}: {why}", case.name));
            let taken = answer
                .deterministic()
                .unwrap_or_else(|| panic!("{}: one outcome", case.name));
            let site = taken
                .root
                .child("stored")
                .unwrap_or_else(|| panic!("{}: no stored site", case.name));
            let Value::Bytes(bytes) = &site.observed.value else {
                panic!("{}: stored holds {}", case.name, site.observed.value.kind())
            };
            assert_eq!(bytes.len(), case.level_bytes(), "{}", case.name);
            let layout = case.layout();
            for (index, &word) in inputs.iter().enumerate() {
                let want = half_toward_zero(f32::from_bits(word));
                for channel in 0..case.channels {
                    let at = layout.at(0, index as u32, channel);
                    let got = u16::from_le_bytes(bytes[at..at + 2].try_into().unwrap());
                    assert_eq!(
                        got, want,
                        "{} input {index} ({word:#010x}) channel {channel}: Metal stored \
                         {got:#06x}, the lane's rule is {want:#06x}",
                        case.name
                    );
                }
            }
        }
    }
}
