//! What a `float` result gets back from a float texel's NaNs, denormals and signed zeros, read and
//! sampled.
//!
//! This is the other half of [`crate::cases::half_write_nans`]: that family asks which bits a
//! `half` store leaves in a texel, this one asks which bits a texel hands a function. Both answer
//! in words, so every site here records unsigned words and compares them exactly — a NaN's payload,
//! a denormal's exact code and a signed zero are all lost the moment a site reads as a float.
//!
//! What `runtime_lane::half_read_nans` measured on plain Metal, and what these recordings hold the
//! implementation to: with a `float` result, **every** NaN of a 16-bit float texel — whatever its
//! sign, its payload and its signalling bit — is `0x7fc00000`, and every other half its exact
//! widening with denormals kept; a **32-bit** float texel's word is handed back as stored, a
//! signalling NaN's and a negative NaN's included. So `0x7c01` in an `R16Float` texel reads
//! `0x7fc00000` while `0x7f800001` in an `R32Float` texel reads `0x7f800001`, and the two formats
//! are the question this family exists to ask.
//!
//! One case per format, so a format a venue honestly lacks a readable or a filterable texture of is
//! that venue's gap ([`crate::verdict::Verdict::LacksFeature`]) at its own row rather than a hole
//! in the others.
//!
//! **Two accesses, two sites, because they are two questions.** A device may fetch a texel one way
//! and filter it another, which is exactly what the lane found elsewhere: ANV's nearest samples
//! lose a denormal its fetches keep. The owned fixture `format_scout` holds both as kernels of one
//! library — `scout_read_f` reads texel `g` of a `texture2d<float, access::read>` into
//! `out[16 g.y + g.x]`, and `scout_sample_f` takes a nearest sample at `(g + (0.25, 0.75)) / 16`
//! of a `texture2d<float>` through the **runtime sampler at slot 0**, which is the lane's own
//! `SampleSlot` access rather than a constexpr sampler folded into the shader. The texture is 16 by
//! 16, which the kernels' `16 g.y + g.x` and their division by 16 both fix.
//!
//! The lane's four other accesses — a `texture2d<float, access::read_write>`, a
//! `texture_buffer<float, read>`, the texture an argument buffer holds, and a gather — ran on the
//! lane's own generated AIR, and no committed fixture reads an arbitrary format through one of them
//! with a single texture bound. They are not ported here; the module docs of
//! `runtime_lane::half_read_nans` record that plain Metal answered the same words through all
//! seven.
//!
//! Every word of both output buffers holds [`BLANK`] before its dispatch, so a word the kernel did
//! not reach is a recorded value rather than whatever the allocator left.
//!
//! Nothing here can fault a GPU: one compute pass an access, a 16 by 16 grid over a 16 by 16
//! texture, one thread a texel, no loop and no coordinate outside the texture.

use crate::allowance::Allowance;
use crate::case::{
    call, made, Abort, Bindings, Case, Failure, Handle, Input, MetalSource, SamplerSpec, Side,
    TextureRole, TextureSpec,
};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::transcript::{ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value};
use metal2vulkan::metal::PixelFormat;
use metal2vulkan_runtime::Degradation;

/// The fixture both kernels live in.
const FIXTURE: &str = "format_scout";

/// The kernel that reads texel `g` through `access::read`.
const READ: &str = "scout_read_f";

/// The kernel that takes a nearest sample inside texel `g` through the sampler at slot 0.
const SAMPLE: &str = "scout_sample_f";

/// The accesses the case compares, by the site each writes and the kernel that writes it.
const ACCESSES: [(&str, &str); 2] = [("read", READ), ("nearest-sample", SAMPLE)];

/// The texture's side, which the kernels' `16 g.y + g.x` and their division by 16 both fix.
const SIDE: u32 = 16;

/// The texels of one level.
const TEXELS: usize = (SIDE * SIDE) as usize;

/// The byte every output word holds before a dispatch, so a word the kernel did not write is a
/// recorded value rather than whatever was there.
pub const BLANK: u8 = 0xab;

/// The halves a 16-bit float texel holds, in the lane's own order: quiet and signalling NaNs of
/// both signs and both payload edges, then the infinities, the denormals, the zeros and two
/// normals.
///
/// A signalling NaN is in here on purpose, and so is a NaN of each sign: they are the inputs that
/// tell a widening that keeps the texel's bits from one that answers Metal's `0x7fc00000`.
const V: [u16; 24] = [
    0x7e00, 0x7c01, 0x7dff, 0x7fff, 0x7e01, 0x7d55, // NaNs, positive
    0xfe00, 0xfc01, 0xfdff, 0xffff, 0xfe01, 0x7f00, // NaNs, negative, and one more positive
    0x7c00, 0xfc00, 0x0001, 0x8001, 0x03ff, 0x3c00, // infinities, denormals, a normal
    0x0000, 0x8000, 0x7bff, 0xfbff, 0x3555, 0xff00, // zeros, the largest halves, two more
];

/// The 32-bit words a 32-bit float texel holds, in the lane's own order: NaNs of every class, then
/// the infinities, the denormals, the zeros and the normals.
///
/// These are the words Metal hands back **as stored**, so each one is also its own expected value.
const F: [u32; 24] = [
    0x7fc00000, 0x7f800001, 0x7fbfffff, 0xffc00001, 0xff800001, 0x7fffffff, //
    0x7fc12345, 0xffffffff, 0x7f800000, 0x00000001, 0x80000001, 0x3f800000, //
    0x7fe00000, 0xffc00000, 0x7f802000, 0x00000000, 0x7fc02000, 0x7fa00000, //
    0xff800000, 0x80000000, 0x7f7fffff, 0x3eaaaaab, 0x7fd00000, 0xfff00000,
];

/// How far apart two channels of one texel are in [`V`] and [`F`], so each value lands at every
/// channel position across the texels and a fetch that treated one channel differently from
/// another shows as a texel rather than as a hash.
const STRIDE: usize = 5;

/// The case: one pixel format's read and sampled words.
pub struct HalfReadNans {
    name: &'static str,
    format: PixelFormat,
    /// The channels the format's texel holds.
    channels: usize,
    /// Whether a channel is a 16-bit float, which is the whole question: a half's NaN comes back as
    /// Metal's canonical `float` NaN, a 32-bit float's as the word that was stored.
    half: bool,
}

/// Every format the lane reads: the three 16-bit float widths and the three 32-bit float widths.
pub fn all() -> Vec<HalfReadNans> {
    let case = |name, format, channels, half| HalfReadNans {
        name,
        format,
        channels,
        half,
    };
    vec![
        case("half-read-nans-r16f", PixelFormat::R16Float, 1, true),
        case("half-read-nans-rg16f", PixelFormat::RG16Float, 2, true),
        case("half-read-nans-rgba16f", PixelFormat::RGBA16Float, 4, true),
        case("half-read-nans-r32f", PixelFormat::R32Float, 1, false),
        case("half-read-nans-rg32f", PixelFormat::RG32Float, 2, false),
        case("half-read-nans-rgba32f", PixelFormat::RGBA32Float, 4, false),
    ]
}

impl HalfReadNans {
    /// The fixture under the kernel `entry`.
    fn fixture(entry: &'static str) -> Fixture {
        Fixture {
            name: FIXTURE,
            entry,
        }
    }

    /// The bytes one texel takes.
    fn texel_bytes(&self) -> u32 {
        (self.channels * if self.half { 2 } else { 4 }) as u32
    }

    /// Channel `channel` of texel `index`: a half's bits or a 32-bit word's.
    fn word(&self, index: usize, channel: usize) -> u32 {
        let at = (index + STRIDE * channel) % V.len();
        match self.half {
            true => u32::from(V[at]),
            false => F[at],
        }
    }

    /// The texture's bytes: the format's channels of each texel, row major.
    fn texels(&self) -> Vec<u8> {
        (0..TEXELS)
            .flat_map(|index| (0..self.channels).map(move |channel| (index, channel)))
            .flat_map(|(index, channel)| {
                let word = self.word(index, channel);
                match self.half {
                    true => (word as u16).to_le_bytes().to_vec(),
                    false => word.to_le_bytes().to_vec(),
                }
            })
            .collect()
    }

    /// The texture one case reads: 16 by 16 of its format, shader-read and nothing else.
    ///
    /// Shader-read alone because both kernels only read: a texture of shader-write usage is a
    /// different descriptor, and the role is part of what the case means rather than something
    /// either side may pick.
    fn spec(&self) -> TextureSpec {
        TextureSpec::new(
            self.format,
            [SIDE, SIDE],
            SIDE * self.texel_bytes(),
            TextureRole::Read,
        )
    }

    /// Run one access and answer the `float4` words it wrote.
    fn sweep(
        &self,
        side: &mut dyn Side,
        sites: &mut Sites,
        access: &str,
        pipeline: Handle,
        texture: Handle,
        samplers: &[Handle],
    ) -> Result<Vec<u8>, Abort> {
        let at = |call: &str| format!("{access}/{call}");
        let output = made(sites, &at("buffer"), side.buffer(&blank_output()))?;
        let command_buffer = made(
            sites,
            &at("dispatch"),
            side.dispatch_threads(
                pipeline,
                &Bindings::new(&[texture], &[output]).sampled(samplers),
                &[],
                [SIDE, SIDE, 1],
                [SIDE, SIDE, 1],
            ),
        )?;
        let status = call(
            sites,
            &at("status"),
            side.wait(command_buffer),
            Clone::clone,
        )?;
        if status != Value::unit_variant("Completed") {
            return Err(Abort {
                at: at("status"),
                failure: Failure::named("CommandBufferStatus", crate::diff::print(&status)),
            });
        }
        call(
            sites,
            &at("read-back"),
            side.read_buffer(output, 0, TEXELS * 16),
            |_| Value::Absent,
        )
    }
}

/// An output buffer before its dispatch: one `float4` a texel, every byte [`BLANK`].
fn blank_output() -> Vec<u8> {
    vec![BLANK; TEXELS * 16]
}

/// The layout a read-back is compared under: one row a texel, four words a row, so the element a
/// diff names is the texel and the lane inside it.
fn layout() -> ByteLayout {
    let length = (TEXELS * 16) as u32;
    ByteLayout {
        region: Region::Buffer {
            offset: 0,
            length: u64::from(length),
        },
        element: Element::U32,
        row_offset: 0,
        row_elements: 4,
        row_pitch: 16,
        rows: TEXELS as u32,
        image_pitch: length,
        images: 1,
        // Each row is four words and the rows abut, so there is no byte outside an element.
        gaps: Gaps::Ignored,
    }
}

impl Case for HalfReadNans {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // `format_scout`'s AIR asks for fast math, and a device whose compiler rewrites the
        // arithmetic it allows announces that at the pipeline's making. Only the notice set is
        // loosened: every one of the 2,048 words still compares exactly, so a device that rewrites
        // nothing observable reads as announced with nothing differing -- which is what this M3
        // Max shows, every word of all six formats byte-identical to Metal's -- and one that
        // changes a texel shows the texel. The allowance fails the case if the notice stops
        // firing, so it cannot outlive the reason for it.
        //
        // Nothing else is loosened. The lane names `device-nan-encoding` for a device whose own
        // decode of a half NaN is not Metal's, and it is deliberately **not** named here: it fired
        // on neither venue of this round, and an allowance no run justified is a claim without
        // evidence. A venue that needs it adds it with the run that shows it.
        // Every row reads its answer back through a `Shared` buffer, so on a device without
        // unified memory the runtime announces `shared-in-system-memory`
        // (`crate::replay::shared_memory_notice` holds the whole reason). It declares the
        // notices site alone, so every value of this family still compares against Metal's
        // exactly.
        //
        // Measured on the RTX 5080 Laptop (615.71.09): it fired for
        // all 6 rows of the family, and each was divergent on that field and
        // nothing else (replay-nvidia.log). On MoltenVK, where
        // the chip's GPU shares the CPU's memory, it cannot fire at all, which is why no
        // recording of this suite declares it.
        vec![
            crate::replay::shared_memory_notice(),
            crate::replay::notice_allowance(Degradation::FastMathRewrites.code()),
        ]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let mut shaders = Vec::new();
        for (_, entry) in ACCESSES {
            for extension in [RUNTIME_TEXT, METAL_TEXT] {
                shaders.push(ShaderRef {
                    fixture: format!("{FIXTURE}{extension}"),
                    entry: entry.to_owned(),
                    source_sha256: crate::sha256_hex(
                        Self::fixture(entry).text(extension)?.as_bytes(),
                    ),
                });
            }
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = Self::fixture(READ);
        Ok(vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            // The format is the question the case asks, so a case re-pointed at another one is
            // unrecorded rather than compared against the first's words.
            Input::bytes("format", &self.format.as_raw().to_le_bytes()),
            Input::bytes("texture:texels", &self.texels()),
            Input::bytes("buffer:blank", &blank_output()),
            // The sampler the nearest-sample access reads through: a different filter or address
            // mode is a different question.
            Input::bytes("sampler", &SamplerSpec::new().words()),
        ])
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let stop = |at: &str, why: String| Abort {
            at: at.to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        };
        let loaded = Self::fixture(READ)
            .loaded()
            .map_err(|why| stop("library", why))?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let texture = made(
            sites,
            "texture",
            side.texture_2d_in_role(&self.spec(), &self.texels()),
        )?;
        // `MTLSamplerDescriptor`'s defaults, which are the nearest, clamp-to-edge, normalized
        // state the lane sampled its runtime sampler slot through.
        let sampler = made(sites, "sampler", side.sampler(&SamplerSpec::new()))?;

        let mut words = Node::branch("words", Vec::new());
        for (access, entry) in ACCESSES {
            let pipeline = made(
                sites,
                &format!("pipeline/{access}"),
                side.compute_pipeline(library, entry),
            )?;
            // The read kernel declares no sampler; the sampled one reads slot 0.
            let samplers: &[Handle] = if entry == SAMPLE { &[sampler] } else { &[] };
            let read = self.sweep(side, sites, access, pipeline, texture, samplers)?;
            words = words.with(Node::leaf(access, Observed::bytes(read, layout())));
        }
        sites.push(words);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// The exact `f32` bits of half `bits`, denormals included: the widening the lane's model
    /// takes, written here so the cross-check below has one.
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

    /// Metal's `float` lane for channel `channel` of texel `index`, as
    /// `runtime_lane::half_read_nans` measured it: a 16-bit float texel's NaN is the canonical
    /// `0x7fc00000` and every other half its exact widening; a 32-bit float texel's word is as
    /// stored; a missing green or blue reads 0 and a missing alpha 1.
    fn metal(case: &HalfReadNans, index: usize, channel: usize) -> u32 {
        if channel >= case.channels {
            return if channel == 3 { 0x3f80_0000 } else { 0 };
        }
        let word = case.word(index, channel);
        match case.half {
            true if f32::from_bits(widened(word as u16)).is_nan() => 0x7fc0_0000,
            true => widened(word as u16),
            false => word,
        }
    }

    /// Every case's texture holds exactly the bytes its level takes, and the blank output covers
    /// every word the kernels write: a texture short of its level would be refused and a short
    /// output would leave written words uncompared.
    #[test]
    fn every_case_states_bytes_for_its_whole_level() {
        for case in all() {
            assert_eq!(
                case.texels().len(),
                TEXELS * case.texel_bytes() as usize,
                "{}",
                case.name
            );
            assert_eq!(case.spec().row_pitch, SIDE * case.texel_bytes());
        }
        assert_eq!(layout().extent(), TEXELS * 16);
        assert_eq!(blank_output().len(), TEXELS * 16);
    }

    /// The cases name distinct rows and distinct formats, so no two share a recording.
    #[test]
    fn the_cases_name_distinct_formats() {
        let named: BTreeSet<&str> = all().iter().map(|case| case.name).collect();
        let formats: BTreeSet<u32> = all().iter().map(|case| case.format.as_raw()).collect();
        assert_eq!(named.len(), all().len());
        assert_eq!(formats.len(), all().len());
    }

    /// The inputs hold what the family turns on: a signalling NaN and a NaN of each sign among the
    /// halves, and a signalling and a negative NaN among the 32-bit words. A sweep that lost them
    /// would compare NaN handling without asking the one question that separates the two widths.
    #[test]
    fn the_values_hold_the_nans_the_family_turns_on() {
        let signalling = V
            .iter()
            .filter(|&&bits| bits & 0x7c00 == 0x7c00 && bits & 0x3ff != 0 && bits & 0x200 == 0)
            .count();
        assert!(signalling >= 2, "{signalling} signalling halves");
        assert!(V.iter().any(|bits| bits >> 15 == 1 && bits & 0x3ff != 0));
        assert!(V.contains(&0x0000) && V.contains(&0x8000), "both zeros");
        assert!(V.contains(&0x0001) && V.contains(&0x03ff), "denormals");
        // A signalling 32-bit NaN, whose payload Metal hands back, and a negative one.
        assert!(F.contains(&0x7f800001) && F.contains(&0xff800001));
        assert!(F.contains(&0x00000001), "a 32-bit denormal");
    }

    /// The layout names the texel and the lane inside it, so a diff over a read-back says which
    /// texel moved rather than which byte.
    #[test]
    fn the_layout_names_the_texel_and_the_lane() {
        let layout = layout();
        for (texel, lane) in [(0usize, 0usize), (0, 3), (1, 0), (37, 2), (TEXELS - 1, 3)] {
            assert_eq!(
                layout.at(0, texel as u32, lane as u32),
                texel * 16 + lane * 4,
                "texel {texel} lane {lane}"
            );
        }
    }

    /// Each half format's texels hold every one of the lane's values, and a half case's texel is
    /// never the 32-bit table's word: the two tables must not be mixed up, because the whole
    /// family turns on telling a half texel from a 32-bit one.
    #[test]
    fn each_case_reads_its_own_table() {
        for case in all() {
            let seen: BTreeSet<u32> = (0..TEXELS)
                .flat_map(|index| (0..case.channels).map(move |channel| (index, channel)))
                .map(|(index, channel)| case.word(index, channel))
                .collect();
            let table: BTreeSet<u32> = match case.half {
                true => V.iter().map(|bits| u32::from(*bits)).collect(),
                false => F.iter().copied().collect(),
            };
            assert_eq!(seen, table, "{}", case.name);
        }
    }

    /// Metal's recorded words are the rule the lane measured, word for word, for every format,
    /// every texel, every lane and both accesses.
    ///
    /// This is the cross-check the port exists for: the lane holds the rule in a function and
    /// asserts a live device against it, which cannot say anything on a host with no Metal. Here
    /// the rule is checked against Metal's own recorded answer, so a release that changes Metal's
    /// widening shows as the texel that moved rather than as a lane that stops passing on one Mac.
    #[test]
    fn metals_recorded_words_are_the_rule_the_lane_measured() {
        use metal2vulkan_runtime::MetalLevel;
        for case in all() {
            let recording = match crate::recording::read(case.name()) {
                Ok(recording) => recording,
                // A case recorded in a later commit of this batch: the test says what it checked
                // rather than failing on a file that is not there yet.
                Err(_) => continue,
            };
            let answer = recording
                .answer(MetalLevel::NEWEST)
                .unwrap_or_else(|why| panic!("{}: {why}", case.name));
            let taken = answer
                .deterministic()
                .unwrap_or_else(|| panic!("{}: one outcome", case.name));
            let words = taken
                .root
                .child("words")
                .unwrap_or_else(|| panic!("{}: no words site", case.name));
            for (access, _) in ACCESSES {
                let site = words
                    .child(access)
                    .unwrap_or_else(|| panic!("{}: no {access} site", case.name));
                let Value::Bytes(bytes) = &site.observed.value else {
                    panic!("{}: {access} holds no bytes", case.name)
                };
                assert_eq!(bytes.len(), TEXELS * 16, "{} {access}", case.name);
                let layout = layout();
                for texel in 0..TEXELS {
                    for lane in 0..4 {
                        let at = layout.at(0, texel as u32, lane as u32);
                        let got = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
                        let want = metal(&case, texel, lane);
                        assert_eq!(
                            got, want,
                            "{} {access} texel {texel} lane {lane}: Metal recorded {got:#010x}, \
                             the lane's rule is {want:#010x}",
                            case.name
                        );
                    }
                }
            }
        }
    }
}
