//! `RGB9E5Float` written and read through a storage slot, ported from
//! `runtime_lane::shared_exponent_storage`.
//!
//! This is a format whose storage is nothing like its read type: one 32-bit word holds three
//! nine-bit mantissas and the five-bit exponent they share, and what a kernel writes as three
//! `float`s becomes that word. Metal's word, as the lane measured it on **this M3 Max under macOS
//! 27.0.1**: each channel clamped to `[0, 65408]` (a NaN, a negative value and `-0` giving 0), the
//! exponent `e = max(floor(log2 m), -16) + 16` of the largest clamped channel `m`, and each
//! mantissa the channel times `2^(24 - e)` **truncated**. A read gives each channel
//! `mantissa * 2^(e - 24)`, exact in `float` and in `half`, and alpha 1.
//!
//! The truncation is the whole point: every device writes and reads this format through its
//! `R32_UINT` alias, where the shader encodes Metal's word itself, because a device's own conversion
//! rounds each mantissa half up where Metal truncates it.
//!
//! The kernels are the owned fixture `fpv_srgb_storage`'s, which take any `float` storage texture:
//! `fpv_srgb_write` and `fpv_srgb_write_half` write buffer 0's value at the thread's texel through
//! `access::write`, `fpv_srgb_update` and `fpv_srgb_update_half` write it through
//! `access::read_write`, and `fpv_srgb_read_write` and `fpv_srgb_read_write_half` read the texel
//! into buffer 0. Buffer 1 holds the row's width.
//!
//! **One case per kernel**, five in all, because a `half` sweep is 65,536 texels and two of them in
//! one recording would pass the recording ceiling. The two `half` cases are exhaustive over a
//! complete small space — every one of the 65,536 bit patterns, each reaching three channels — which
//! the suite's record-shape rule keeps whole whatever its size.
//!
//! **The `float` sweep is a stated reduction of the lane's.** The lane takes the 196 pairs of its
//! fourteen specials and then 16,384 iterations of its generator, two texels each: 33,024 texels,
//! whose stored words alone would be 132 KiB a kernel. The case keeps the 196 special pairs whole
//! and [`ITERATIONS`] of the same generator, in the same order, so the values are the probe's own
//! prefix rather than a different set: [`FLOAT_TEXELS`] texels. Each iteration still contributes
//! both of the lane's texels — a spread magnitude with two smaller channels, and a mantissa-step
//! edge with its two neighbouring `f32`s — so both shapes the generator exists for are kept.
//!
//! Every texel holds `0xA5A5A5A5` before a write ([`BLANK_WORD`]), so a texel the kernel did not
//! reach is a recorded value rather than whatever the allocator left.
//!
//! Nothing here can fault a GPU: one compute pass a case, one thread a texel, no loop and no
//! sampling.

use crate::allowance::Allowance;
use crate::case::{call, made, Abort, Bindings, Case, Failure, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::texture::{TextureRequest, Upload};
use crate::transcript::{
    Aspect, ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};

/// The fixture every kernel lives in.
const FIXTURE: &str = "fpv_srgb_storage";

/// The texture's width; its height holds the values.
pub const WIDTH: u32 = 256;

/// The word every texel holds before a write.
pub const BLANK_WORD: u32 = 0xA5A5_A5A5;

/// The generator iterations the `float` sweep keeps of the lane's 16,384, each contributing two
/// texels.
pub const ITERATIONS: usize = 512;

/// The texels the `float` sweep writes: the 196 pairs of the specials, then two a generator
/// iteration, padded to a whole row.
pub const FLOAT_TEXELS: usize = 1280;

/// Metal's word for red, green and blue `rgb` (the module docs).
pub fn word(rgb: [f32; 3]) -> u32 {
    let clamped = rgb.map(|value| match value.is_nan() || value.is_sign_negative() {
        true => 0.0,
        false => value.min(65408.0),
    });
    let largest = clamped[0].max(clamped[1]).max(clamped[2]);
    let exponent = (((largest.to_bits() >> 23) as i32).max(111) - 111) as u32;
    let scale = 2f64.powi(24 - exponent as i32);
    let mantissa = clamped.map(|value| ((f64::from(value) * scale).floor() as u32).min(511));
    mantissa[0] | mantissa[1] << 9 | mantissa[2] << 18 | exponent << 27
}

/// The value of `half` bits `h`, exactly (a NaN for a NaN).
pub fn half_value(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((h >> 10) & 0x1f);
    let mantissa = f32::from(h & 0x3ff);
    sign * match exponent {
        0 => mantissa * 2f32.powi(-24),
        31 if mantissa == 0.0 => f32::INFINITY,
        31 => f32::NAN,
        _ => (1024.0 + mantissa) * 2f32.powi(exponent - 25),
    }
}

/// The specials the sweep pairs: zeros, a negative, a NaN, an infinity, the clamp and past it, and
/// values at the edges of the exponent range.
const SPECIALS: [f32; 14] = [
    0.0,
    -0.0,
    -1.0,
    f32::NAN,
    f32::INFINITY,
    65408.0,
    65472.0,
    65536.0,
    2e-5,
    1.0,
    0.5,
    f32::from_bits(1),
    // 2^-16 and 2^-24, the exponent floor and the smallest mantissa step at it.
    f32::from_bits(0x3780_0000),
    f32::from_bits(0x3380_0000),
];

/// The texels the `float` sweep writes, in the lane's order.
pub fn floats() -> Vec<[f32; 4]> {
    let mut out: Vec<[f32; 4]> = SPECIALS
        .iter()
        .flat_map(|&a| {
            SPECIALS
                .iter()
                .map(move |&b| [a, b, 0.5 * a + 0.25 * b, 1.0])
        })
        .collect();
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        state >> 33
    };
    for _ in 0..ITERATIONS {
        let exponent = (next() % 48) as i32 - 30;
        let big = 2f32.powi(exponent) * (1.0 + (next() % 4096) as f32 / 4096.0);
        let fraction = |random: u64| (random % 65536) as f32 / 65536.0;
        out.push([
            big,
            big * fraction(next()),
            big * fraction(next()) * fraction(next()),
            0.5,
        ]);
        let unit = 2f32.powi(exponent.clamp(-24, 7));
        let edge = ((next() % 511) as f32 + 0.5) * unit;
        out.push([
            edge,
            f32::from_bits(edge.to_bits() + 1),
            f32::from_bits(edge.to_bits() - 1),
            0.0,
        ]);
    }
    while !out.len().is_multiple_of(WIDTH as usize) {
        out.push([0.0; 4]);
    }
    out
}

/// The halves the `half` sweeps write: every bit pattern, each reaching the first three channels
/// through its own lane and its two multiples, with alpha 1.0.
pub fn halves() -> Vec<[u16; 4]> {
    (0..=u16::MAX)
        .map(|h| {
            [
                h,
                h.wrapping_mul(7).wrapping_add(3),
                h.wrapping_mul(13).wrapping_add(5),
                0x3c00,
            ]
        })
        .collect()
}

/// The words the read cases' texture holds: every exponent, with three mantissas of its own.
pub fn source_words() -> Vec<u32> {
    (0..WIDTH * 4)
        .map(|n| (n % 32) << 27 | (n * 37 % 512) | (n * 101 % 512) << 9 | (n * 7 % 512) << 18)
        .collect()
}

/// Which sweep a case runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sweep {
    /// A write of the `float` texels through the kernel named.
    WriteFloat,
    /// A write of every `half` through the kernel named.
    WriteHalf,
    /// A `read_write` read of [`source_words`], `float` and `half` in one case: 1,024 texels, so
    /// both fit well inside the ceiling and the two decodes are compared side by side.
    Read,
}

/// One kernel's sweep.
pub struct SharedExponentStorage {
    name: &'static str,
    kernel: &'static str,
    sweep: Sweep,
}

/// The `half` read kernel, which the read case runs beside the `float` one.
const READ_HALF: &str = "fpv_srgb_read_write_half";

/// Every case of the family.
pub fn all() -> Vec<SharedExponentStorage> {
    vec![
        SharedExponentStorage {
            name: "shared-exponent-write-float",
            kernel: "fpv_srgb_write",
            sweep: Sweep::WriteFloat,
        },
        SharedExponentStorage {
            name: "shared-exponent-update-float",
            kernel: "fpv_srgb_update",
            sweep: Sweep::WriteFloat,
        },
        SharedExponentStorage {
            name: "shared-exponent-write-half",
            kernel: "fpv_srgb_write_half",
            sweep: Sweep::WriteHalf,
        },
        SharedExponentStorage {
            name: "shared-exponent-update-half",
            kernel: "fpv_srgb_update_half",
            sweep: Sweep::WriteHalf,
        },
        SharedExponentStorage {
            name: "shared-exponent-reads",
            kernel: "fpv_srgb_read_write",
            sweep: Sweep::Read,
        },
    ]
}

/// Buffer 1's bytes: the row's width, as the kernels' `constant uint &width`.
fn width_bytes() -> Vec<u8> {
    WIDTH.to_le_bytes().to_vec()
}

impl SharedExponentStorage {
    /// The kernels the case runs, in order.
    fn kernels(&self) -> Vec<&'static str> {
        match self.sweep {
            Sweep::Read => vec![self.kernel, READ_HALF],
            _ => vec![self.kernel],
        }
    }

    /// The texels the case's sweep covers.
    fn texels(&self) -> usize {
        match self.sweep {
            Sweep::WriteFloat => floats().len(),
            Sweep::WriteHalf => halves().len(),
            Sweep::Read => source_words().len(),
        }
    }

    /// The texture's height: one row of [`WIDTH`] texels at a time.
    fn height(&self) -> u32 {
        (self.texels() / WIDTH as usize) as u32
    }

    /// Buffer 0's bytes: the values a write hands the kernel, or the zeros a read fills in.
    fn buffer_bytes(&self) -> Vec<u8> {
        match self.sweep {
            Sweep::WriteFloat => floats()
                .iter()
                .flatten()
                .flat_map(|value| value.to_bits().to_le_bytes())
                .collect(),
            Sweep::WriteHalf => halves()
                .iter()
                .flatten()
                .flat_map(|bits| bits.to_le_bytes())
                .collect(),
            // A read writes a `float4` or a `half4` a texel; the case reads back whichever the
            // kernel it runs wrote, so the buffer is the wider of the two.
            Sweep::Read => vec![0u8; self.texels() * 16],
        }
    }

    /// The bytes one lane of `kernel`'s output takes: four `float`s, or four `half`s.
    fn lane_bytes(kernel: &str) -> usize {
        match kernel.ends_with("_half") {
            true => 8,
            false => 16,
        }
    }

    /// The texture's words before the case runs: the blank a write starts from, or the source
    /// words a read decodes.
    fn level_words(&self) -> Vec<u32> {
        match self.sweep {
            Sweep::Read => source_words(),
            _ => vec![BLANK_WORD; self.texels()],
        }
    }

    /// Those words as the bytes the texture is loaded with.
    fn level_bytes(&self) -> Vec<u8> {
        self.level_words()
            .iter()
            .flat_map(|held| held.to_le_bytes())
            .collect()
    }

    /// The texture the case runs over.
    fn request<'a>(&self, uploads: &'a [Upload<'a>]) -> TextureRequest<'a> {
        TextureRequest::read_2d(PixelFormat::RGB9E5Float.as_raw(), WIDTH, self.height())
            .used(TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE)
            .loaded(uploads)
    }

    /// The layout the stored words are read under: one row a texel row, one unsigned word a texel,
    /// so a diff names the texel and prints the word Metal packed.
    fn stored_layout(&self) -> ByteLayout {
        let pitch = WIDTH * 4;
        ByteLayout {
            region: Region::Texture {
                level: 0,
                slice: 0,
                aspect: Aspect::Color,
                sample: 0,
                origin: [0, 0, 0],
                size: [WIDTH, self.height(), 1],
            },
            element: Element::U32,
            row_offset: 0,
            row_elements: WIDTH,
            row_pitch: pitch,
            rows: self.height(),
            image_pitch: pitch * self.height(),
            images: 1,
            // One word a texel covers every byte of a row, so there is no gap.
            gaps: Gaps::Ignored,
        }
    }

    /// The layout a read's answer is read under: one row a texel, four channels each, as the bits
    /// of the `float` or the `half` the kernel wrote.
    fn read_layout(&self, kernel: &str) -> ByteLayout {
        let lane = Self::lane_bytes(kernel);
        let length = self.texels() * lane;
        ByteLayout {
            region: Region::Buffer {
                offset: 0,
                length: length as u64,
            },
            element: match lane {
                8 => Element::U16,
                _ => Element::U32,
            },
            row_offset: 0,
            row_elements: 4,
            row_pitch: lane as u32,
            rows: self.texels() as u32,
            image_pitch: length as u32,
            images: 1,
            gaps: Gaps::Ignored,
        }
    }
}

impl Case for SharedExponentStorage {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // One declaration and no tolerance: the buffer each dispatch reads its words back through is `Shared`, which is
        // what [`crate::replay::shared_memory_notice`] is for. It loosens the notices site alone,
        // so every value this case reads still compares exactly.
        vec![crate::replay::shared_memory_notice()]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = Vec::new();
        for kernel in self.kernels() {
            let fixture = Fixture {
                name: FIXTURE,
                entry: kernel,
            };
            for extension in [RUNTIME_TEXT, METAL_TEXT] {
                shaders.push(ShaderRef {
                    fixture: format!("{FIXTURE}{extension}"),
                    entry: kernel.to_owned(),
                    source_sha256: crate::sha256_hex(fixture.text(extension)?.as_bytes()),
                });
            }
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = Fixture::new(FIXTURE);
        let level = self.level_bytes();
        let uploads = [Upload::level_0(
            [WIDTH, self.height()],
            WIDTH * 4,
            self.height(),
            &level,
        )];
        Ok(vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("buffer:0", &self.buffer_bytes()),
            Input::bytes("buffer:1", &width_bytes()),
            Input::bytes("texture", &self.request(&uploads).words()),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = Fixture::new(FIXTURE).loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let shape = made(sites, "buffer/width", side.buffer(&width_bytes()))?;
        let level = self.level_bytes();
        let height = self.height();
        for kernel in self.kernels() {
            let at = |site: &str| format!("{kernel}/{site}");
            let pipeline = made(
                sites,
                &at("pipeline"),
                side.compute_pipeline(library, kernel),
            )?;
            let uploads = [Upload::level_0([WIDTH, height], WIDTH * 4, height, &level)];
            let texture = made(sites, &at("texture"), side.texture(&self.request(&uploads)))?;
            let values = made(sites, &at("buffer/0"), side.buffer(&self.buffer_bytes()))?;
            let command_buffer = made(
                sites,
                &at("dispatch"),
                side.dispatch(
                    pipeline,
                    &Bindings::new(&[texture], &[values, shape]),
                    [WIDTH / 16, height, 1],
                    [16, 1, 1],
                ),
            )?;
            call(
                sites,
                &at("status"),
                side.wait(command_buffer),
                Clone::clone,
            )?;
            match self.sweep {
                Sweep::Read => {
                    let lane = Self::lane_bytes(kernel);
                    let got = call(
                        sites,
                        &at("read-back"),
                        side.read_buffer(values, 0, self.texels() * lane),
                        |_| Value::Absent,
                    )?;
                    sites.push(Node::leaf(
                        format!("{kernel}/decoded"),
                        Observed::bytes(got, self.read_layout(kernel)),
                    ));
                }
                _ => {
                    let stored = call(
                        sites,
                        &at("texels"),
                        side.read_texture(texture, [WIDTH, height], WIDTH * 4),
                        |_| Value::Absent,
                    )?;
                    sites.push(Node::leaf(
                        format!("{kernel}/stored"),
                        Observed::bytes(stored, self.stored_layout()),
                    ));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// Every outcome's bytes at `site`, over every answer of `case`'s recording.
    fn recorded(case: &str, site: &str) -> Option<Vec<Vec<u8>>> {
        let recording = crate::recording::read(case).ok()?;
        let mut found = Vec::new();
        for answer in &recording.answers {
            for outcome in &answer.outcomes {
                if let Some(node) = outcome.transcript.root.child(site) {
                    if let Value::Bytes(bytes) = &node.observed.value {
                        found.push(bytes.clone());
                    }
                }
            }
        }
        Some(found)
    }

    /// Every word Metal stored is the lane's rule's, over every write case.
    ///
    /// The lane measured this chip, so this is the one family of the batch whose rule was already
    /// the oracle's — and the recording is what stops that measurement evaporating the next time
    /// macOS moves. The truncation is what the rule turns on: a device's own conversion rounds each
    /// mantissa half up, so a leg that let the device encode would differ on about half the inputs.
    #[test]
    fn every_recorded_word_is_the_lanes_rule() {
        let mut unrecorded = Vec::new();
        let mut differ = Vec::new();
        let mut checked = 0usize;
        for case in all() {
            let wanted: Vec<u32> = match case.sweep {
                Sweep::WriteFloat => floats()
                    .iter()
                    .map(|texel| word([texel[0], texel[1], texel[2]]))
                    .collect(),
                Sweep::WriteHalf => halves()
                    .iter()
                    .map(|texel| word([0, 1, 2].map(|at| half_value(texel[at]))))
                    .collect(),
                Sweep::Read => continue,
            };
            let site = format!("{}/stored", case.kernel);
            let Some(outcomes) = recorded(case.name, &site) else {
                unrecorded.push(case.name);
                continue;
            };
            if outcomes.is_empty() {
                differ.push(format!("{}: no site {site}", case.name));
                continue;
            }
            for bytes in outcomes {
                for (texel, want) in wanted.iter().enumerate() {
                    let at = texel * 4;
                    let got = u32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"));
                    checked += 1;
                    if got != *want && differ.len() < 24 {
                        differ.push(format!(
                            "{}/{site} texel {texel}: Metal {got:#010x}, the rule's {want:#010x}",
                            case.name
                        ));
                    }
                }
            }
        }
        assert!(
            differ.is_empty(),
            "{checked} words checked; Metal's recorded words differ from the lane's rule:\n{}",
            differ.join("\n")
        );
        assert!(
            unrecorded.is_empty(),
            "no recording yet for {unrecorded:?}: run `conformance record` on a Mac"
        );
        assert!(checked > 0, "the recordings held no word to check");
    }

    /// Every read Metal recorded decodes its word exactly: each channel `mantissa * 2^(e - 24)`,
    /// and alpha 1, in `float` and in `half` alike.
    #[test]
    fn every_recorded_read_decodes_exactly() {
        let case = all()
            .into_iter()
            .find(|case| case.sweep == Sweep::Read)
            .expect("the read case");
        let words = source_words();
        let mut differ = Vec::new();
        let mut checked = 0usize;
        let mut unrecorded = false;
        for kernel in case.kernels() {
            let lane = SharedExponentStorage::lane_bytes(kernel);
            let site = format!("{kernel}/decoded");
            let Some(outcomes) = recorded(case.name, &site) else {
                unrecorded = true;
                continue;
            };
            if outcomes.is_empty() {
                differ.push(format!("{}: no site {site}", case.name));
                continue;
            }
            for bytes in outcomes {
                for (texel, held) in words.iter().enumerate() {
                    for channel in 0..4usize {
                        let want = match channel {
                            3 => 1.0f32,
                            _ => {
                                ((held >> (9 * channel)) & 0x1ff) as f32
                                    * 2f32.powi((held >> 27) as i32 - 24)
                            }
                        };
                        let at = texel * lane + channel * (lane / 4);
                        let got = match lane {
                            8 => half_value(u16::from_le_bytes(
                                bytes[at..at + 2].try_into().expect("two bytes"),
                            )),
                            _ => f32::from_bits(u32::from_le_bytes(
                                bytes[at..at + 4].try_into().expect("four bytes"),
                            )),
                        };
                        checked += 1;
                        if got.to_bits() != want.to_bits() && differ.len() < 24 {
                            differ.push(format!(
                                "{site} texel {texel} channel {channel}: Metal {got}, the rule's \
                                 {want}"
                            ));
                        }
                    }
                }
            }
        }
        assert!(
            differ.is_empty(),
            "{checked} channels checked; Metal's recorded reads are not exact:\n{}",
            differ.join("\n")
        );
        assert!(
            !unrecorded,
            "no recording yet for {}: run `conformance record` on a Mac",
            case.name
        );
        assert!(checked > 0, "the recording held no channel to check");
    }

    /// The cases name distinct recordings and distinct kernels, and each sweep fills whole rows.
    #[test]
    fn the_cases_are_distinct_and_fill_whole_rows() {
        let named: BTreeSet<&str> = all().iter().map(|case| case.name).collect();
        assert_eq!(named.len(), all().len());
        let kernels: BTreeSet<&str> = all().iter().map(|case| case.kernel).collect();
        assert_eq!(kernels.len(), all().len());
        for case in all() {
            assert_eq!(
                case.texels() % WIDTH as usize,
                0,
                "{}: {} texels is not whole rows of {WIDTH}",
                case.name,
                case.texels()
            );
            assert!(case.height() > 0, "{}", case.name);
            // The dispatch covers every texel: `WIDTH / 16` groups of 16 threads, one row each.
            assert_eq!(WIDTH % 16, 0);
            assert_eq!(
                case.stored_layout().extent(),
                case.texels() * 4,
                "{}",
                case.name
            );
        }
        assert_eq!(floats().len(), FLOAT_TEXELS);
        assert_eq!(halves().len(), 1 << 16, "every half, exhaustively");
    }

    /// The rule's hard cases are really in the sweep, so a run cannot pass without meeting them:
    /// a NaN, a negative value, a signed zero, an infinity, the clamp and a value past it, and a
    /// mantissa-step edge whose two neighbours straddle it.
    ///
    /// Without those the sweep would check the easy middle of the range, where truncating and
    /// rounding mostly agree.
    #[test]
    fn the_float_sweep_holds_the_rules_hard_cases() {
        let texels = floats();
        let channels = || texels.iter().flat_map(|texel| texel[..3].iter().copied());
        assert!(channels().any(f32::is_nan), "a NaN");
        assert!(channels().any(|value| value < 0.0), "a negative value");
        assert!(
            channels().any(|value| value == 0.0 && value.is_sign_negative()),
            "a negative zero"
        );
        assert!(channels().any(f32::is_infinite), "an infinity");
        assert!(channels().any(|value| value == 65408.0), "the clamp");
        assert!(channels().any(|value| value > 65408.0), "past the clamp");
        // A NaN, a negative value and a negative zero all give 0, and a value past the clamp gives
        // the clamp's own mantissa: the four corners of the clamping half of the rule.
        assert_eq!(word([f32::NAN, 0.0, 0.0]), word([0.0; 3]));
        assert_eq!(word([-1.0, 0.0, 0.0]), word([0.0; 3]));
        assert_eq!(word([-0.0, 0.0, 0.0]), word([0.0; 3]));
        assert_eq!(word([f32::INFINITY, 0.0, 0.0]), word([65408.0, 0.0, 0.0]));
        // And truncation really differs from rounding half up, which is what the shader's own
        // encode exists for: a channel one part in 2^24 below a mantissa step truncates down.
        let below = f32::from_bits(1.0f32.to_bits() - 1);
        assert_ne!(
            word([below, 0.0, 0.0]) & 0x1ff,
            word([1.0, 0.0, 0.0]) & 0x1ff,
            "a value just below a mantissa step truncates to the step below it"
        );
    }
}
